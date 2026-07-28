// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Command oracle drives the differential-testing harness: it renders the scenario corpus, writes golden images, and
// diffs golden directories under the threshold profiles.
//
//	oracle list                                 print the scenario corpus
//	oracle gen -out DIR [-gpu]                  render all scenarios into DIR (+ manifest); -gpu renders
//	                                            through the GL backend instead of raster
//	oracle diff -a DIR -b DIR [-profile P] [-artifacts DIR]
//	                                            compare two golden directories; artifacts get side-by-side +
//	                                            heatmap PNGs for failures; exit 1 on any failure
//	oracle soak -n N [-gpu|-dmsaa]              render the full corpus N times — each pass in a fresh session
//	                                            (new GL context for the GPU lanes) — and fail on any
//	                                            pass-to-pass divergence: the determinism proof behind
//	                                            self-captured goldens; run it twice and compare the printed
//	                                            pass-1 hash lines / corpus digests to also prove cross-process
//	                                            determinism. The raster lane compares strict hashes; the GPU
//	                                            lanes compare later passes per-pixel against the retained
//	                                            pass-1 buffers under the ±1 LSB envelope (see soak.go)
//	oracle bless -lane {raster|gpu|gpudmsaa} [-artifacts DIR]
//	                                            capture the self-captured golden set for a lane into
//	                                            goldens/<lane>/<GOOS_GOARCH>/ (path derived, never
//	                                            user-supplied); renders the corpus in two fresh sessions and
//	                                            refuses to write on any cross-pass divergence (strict hash on
//	                                            raster, the ±1 LSB envelope on the GPU lanes); prints an
//	                                            old-vs-new change summary
//
// soak and bless exit with code 3 (not 1) when the GPU backend cannot bring up a GL context, so callers (the
// capture-goldens workflow) can distinguish "this leg has no GL stack — skip the lane loudly" from a real failure.
//
// gen renders through the canvas library (internal/oracle/gorender). The checked-in goldens under ../../goldens are
// the library's own output, captured per platform by `bless`, so `gen` + `diff` against a raster set is a pure change
// detector: a failure means rendering changed on this platform, full stop. A frozen archive of the retired C Skia
// oracle's renders sits, non-gating, under ../../goldens-skia; `diff` with the cpu/text/gpu tolerance profiles exists
// for comparing against that archive.
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"sort"

	"github.com/richardwilkes/canvas/internal/oracle/golden"
	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "list":
		for _, s := range scenario.All() {
			fmt.Printf("%-32s %dx%d\n", s.Name, s.Width, s.Height)
		}
	case "gen":
		fs := flag.NewFlagSet("gen", flag.ExitOnError)
		out := fs.String("out", "", "output directory (required)")
		useGPU := fs.Bool("gpu", false, "render through the GL backend instead of raster")
		if err := fs.Parse(os.Args[2:]); err != nil {
			fatal(err)
		}
		if *out == "" {
			fs.Usage()
			os.Exit(2)
		}
		if err := gen(*out, *useGPU); err != nil {
			fatal(err)
		}
	case "diff":
		fs := flag.NewFlagSet("diff", flag.ExitOnError)
		a := fs.String("a", "", "first golden directory (required)")
		b := fs.String("b", "", "second golden directory (required)")
		profileName := fs.String("profile", "exact", "threshold profile: exact, exact1, cpu, text, gpu")
		artifacts := fs.String("artifacts", "", "directory for failure artifacts (side-by-side + heatmap PNGs)")
		if err := fs.Parse(os.Args[2:]); err != nil {
			fatal(err)
		}
		if *a == "" || *b == "" {
			fs.Usage()
			os.Exit(2)
		}
		profile, ok := imgdiff.ProfileByName(*profileName)
		if !ok {
			fatal(fmt.Errorf("unknown profile %q", *profileName))
		}
		failures, err := diff(*a, *b, profile, *artifacts)
		if err != nil {
			fatal(err)
		}
		if failures > 0 {
			os.Exit(1)
		}
	case "soak":
		fs := flag.NewFlagSet("soak", flag.ExitOnError)
		n := fs.Int("n", 3, "number of full-corpus renders to compare (>= 2)")
		useGPU := fs.Bool("gpu", false, "render through the GL backend instead of raster")
		useDMSAA := fs.Bool("dmsaa", false, "render through the GL backend with dynamic MSAA (the gpudmsaa lane)")
		if err := fs.Parse(os.Args[2:]); err != nil {
			fatal(err)
		}
		if *useGPU && *useDMSAA {
			fatal(errNoBothGPUFlags)
		}
		if *n < 2 {
			fatal(fmt.Errorf("soak needs -n >= 2, got %d", *n))
		}
		lane := laneName(*useGPU, *useDMSAA)
		cfg := soakConfig{
			out:        os.Stdout,
			n:          *n,
			lane:       lane,
			scenarios:  scenario.All(),
			newSession: func() (*laneSession, error) { return newLaneSession(lane) },
		}
		if err := soak(&cfg); err != nil {
			fatalMaybeNoGL(err)
		}
	case "bless":
		fs := flag.NewFlagSet("bless", flag.ExitOnError)
		lane := fs.String("lane", "", "golden lane: raster, gpu, or gpudmsaa (required)")
		artifacts := fs.String("artifacts", "",
			"directory for old-vs-new change artifacts (side-by-side + heatmap PNGs)")
		if err := fs.Parse(os.Args[2:]); err != nil {
			fatal(err)
		}
		if *lane != laneRaster && *lane != laneGPU && *lane != laneGPUDMSAA {
			fatal(fmt.Errorf("bless -lane must be %s, %s, or %s; got %q", laneRaster, laneGPU, laneGPUDMSAA, *lane))
		}
		// The output path is derived relative to the oracle module root (goldens/<lane>/<platform>), so require that
		// root as the working directory rather than silently creating a goldens tree somewhere else.
		if data, err := os.ReadFile("go.mod"); err != nil || !bytes.Contains(data, []byte("internal/oracle")) {
			fatal(errors.New(
				"bless must run from the internal/oracle module root (its output path is derived, not user-supplied)"))
		}
		cfg := blessConfig{
			out:        os.Stdout,
			dir:        blessDir(*lane, runtime.GOOS, runtime.GOARCH),
			lane:       *lane,
			platform:   runtime.GOOS + "_" + runtime.GOARCH,
			artifacts:  *artifacts,
			scenarios:  scenario.All(),
			newSession: func() (*laneSession, error) { return newLaneSession(*lane) },
		}
		if err := bless(&cfg); err != nil {
			fatalMaybeNoGL(err)
		}
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr,
		"usage: oracle {list | gen -out DIR [-gpu] | diff -a DIR -b DIR [-profile P] [-artifacts DIR] |"+
			" soak -n N [-gpu|-dmsaa] | bless -lane {raster|gpu|gpudmsaa} [-artifacts DIR]}")
	os.Exit(2)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "oracle:", err)
	os.Exit(1)
}

// fatalMaybeNoGL is fatal, except that a missing-GL-context failure exits with exitNoGLContext so the capture workflow
// can loudly skip the lane instead of failing the leg (see errNoGLContext).
func fatalMaybeNoGL(err error) {
	if errors.Is(err, errNoGLContext) {
		fmt.Fprintf(os.Stderr, "oracle: SKIP-WORTHY (exit %d): %v\n", exitNoGLContext, err)
		os.Exit(exitNoGLContext)
	}
	fatal(err)
}

func gen(dir string, useGPU bool) error {
	// The GL backend renders through a headless GL context, which is bound to one OS thread; the whole gen loop must
	// run on the goroutine that created it, so build it here and render inline (no goroutines). Honors
	// CANVAS_GLTEST_RENDERER=software the same way gpu/gl/gltest does.
	render := gorender.RenderScenarioRaster
	if useGPU {
		g, err := gorender.NewGPUContext()
		if err != nil {
			return fmt.Errorf("GPU backend unavailable: %w", err)
		}
		defer g.Dispose()
		render = func(s scenario.Scenario) []byte { return gorender.RenderScenarioGPU(g, s) }
	}
	return writeGoldens(dir, scenario.All(), render)
}

// writeGoldens renders scenarios through render and writes the golden PNGs + manifest to dir.
//
// The manifest is schema 2, the same schema `bless` writes: schema 1 means "a frozen Skia-era archive set" (see
// golden.Manifest), which both bless and capture.sh refuse to overwrite, so stamping gen's freshly rendered
// output with it would poison the directory against any later capture with a diagnosis naming an archive that was
// never there. The lane and GL-stack fields stay empty — gen's output is a scratch comparison set, not a blessed one.
func writeGoldens(dir string, scenarios []scenario.Scenario, render func(scenario.Scenario) []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	m := golden.Manifest{Schema: blessSchema, Platform: runtime.GOOS + "_" + runtime.GOARCH}
	for _, s := range scenarios {
		pixels := render(s)
		if err := golden.WritePNG(filepath.Join(dir, s.Name+".png"), pixels, s.Width, s.Height); err != nil {
			return err
		}
		m.Entries = append(m.Entries, golden.Entry{
			Name:   s.Name,
			Width:  s.Width,
			Height: s.Height,
			SHA256: golden.HashPixels(pixels),
		})
		fmt.Printf("rendered %s\n", s.Name)
	}
	sort.Slice(m.Entries, func(i, j int) bool { return m.Entries[i].Name < m.Entries[j].Name })
	return golden.WriteManifest(dir, &m)
}

// diff compares two golden directories under profile.
func diff(aDir, bDir string, profile imgdiff.Profile, artifacts string) (failures int, err error) {
	am, err := golden.ReadManifest(aDir)
	if err != nil {
		return 0, fmt.Errorf("reading manifest in %s: %w", aDir, err)
	}
	bm, err := golden.ReadManifest(bDir)
	if err != nil {
		return 0, fmt.Errorf("reading manifest in %s: %w", bDir, err)
	}
	bByName := make(map[string]golden.Entry, len(bm.Entries))
	for _, e := range bm.Entries {
		bByName[e.Name] = e
	}
	if artifacts != "" {
		if err = os.MkdirAll(artifacts, 0o755); err != nil {
			return 0, err
		}
	}
	for _, ae := range am.Entries {
		be, ok := bByName[ae.Name]
		if !ok {
			fmt.Printf("FAIL %-32s missing from %s\n", ae.Name, bDir)
			failures++
			continue
		}
		delete(bByName, ae.Name)
		if ae.Width != be.Width || ae.Height != be.Height {
			fmt.Printf("FAIL %-32s size %dx%d vs %dx%d\n", ae.Name, ae.Width, ae.Height, be.Width, be.Height)
			failures++
			continue
		}
		// Both PNGs are read even when the manifests agree. The manifest hashes describe the pixels a golden was
		// captured from, not the bytes now on disk, so short-circuiting on hash equality would leave every passing
		// golden PNG unopened — and the raster lane's only gate is `oracle gen` + `oracle diff`, which would then never
		// notice a corrupted or truncated checked-in golden. Reading them is what makes the files comparison data
		// first (see package golden).
		apx, aw, ah, pngErr := golden.ReadPNG(filepath.Join(aDir, ae.Name+".png"))
		if pngErr != nil {
			return failures, pngErr
		}
		bpx, bw, bh, pngErr := golden.ReadPNG(filepath.Join(bDir, be.Name+".png"))
		if pngErr != nil {
			return failures, pngErr
		}
		if aw != ae.Width || ah != ae.Height || bw != be.Width || bh != be.Height {
			fmt.Printf("FAIL %-32s png size disagrees with manifest\n", ae.Name)
			failures++
			continue
		}
		if ae.SHA256 == be.SHA256 {
			// Equal manifest hashes still have to be the hashes of these two files: a PNG whose pixels no longer hash
			// to its own entry is a damaged golden, however well the two manifests agree with each other.
			if aHash, bHash := golden.HashPixels(apx), golden.HashPixels(bpx); aHash != ae.SHA256 || bHash != be.SHA256 {
				fmt.Printf("FAIL %-32s png content disagrees with its manifest entry (%s: %s, %s: %s, manifest: %s)\n",
					ae.Name, aDir, aHash, bDir, bHash, ae.SHA256)
				failures++
				continue
			}
			fmt.Printf("ok   %-32s identical\n", ae.Name)
			continue
		}
		res, pngErr := imgdiff.Compare(apx, bpx, aw, ah, profile)
		if pngErr != nil {
			return failures, pngErr
		}
		if res.Pass() {
			fmt.Printf("ok   %-32s %s\n", ae.Name, res)
			continue
		}
		fmt.Printf("FAIL %-32s %s\n", ae.Name, res)
		failures++
		if artifacts != "" {
			img := imgdiff.SideBySide(apx, bpx, aw, ah)
			f, fileErr := os.Create(filepath.Join(artifacts, ae.Name+"-diff.png"))
			if fileErr != nil {
				return failures, fileErr
			}
			if fileErr = png.Encode(f, img); fileErr != nil {
				f.Close() //nolint:errcheck // the encode error is the one worth reporting
				return failures, fileErr
			}
			if fileErr = f.Close(); fileErr != nil {
				return failures, fileErr
			}
		}
	}
	for name := range bByName {
		fmt.Printf("FAIL %-32s missing from %s\n", name, aDir)
		failures++
	}
	if failures > 0 {
		fmt.Printf("%d failure(s) under profile %q\n", failures, profile.Name)
	} else {
		fmt.Printf("all %d scenarios pass under profile %q\n", len(am.Entries), profile.Name)
	}
	return failures, nil
}
