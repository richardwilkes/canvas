// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package main

import (
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/richardwilkes/canvas/internal/oracle/golden"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// blessSchema is the manifest schema version bless writes (see golden.Manifest).
const blessSchema = 2

// blessConfig carries one bless capture. The session factory and scenario list are injected so tests can drive bless
// with a synthetic corpus and a deliberately nondeterministic renderer.
type blessConfig struct {
	dir       string    // target golden directory (blessDir; derived, never user-supplied)
	lane      string    // laneRaster, laneGPU, or laneGPUDMSAA
	platform  string    // GOOS_GOARCH
	artifacts string    // optional directory for old-vs-new change artifacts
	out       io.Writer // progress + change-summary output
	// newSession opens a fresh rendering session for one full corpus pass (see laneSession for why passes get fresh
	// sessions rather than re-rendering on a warm context). bless opens exactly two: the capture pass and the verify
	// pass.
	newSession func() (*laneSession, error)
	scenarios  []scenario.Scenario // the corpus to capture, in gate order
}

// bless captures a lane's self-captured golden set: it renders the corpus once in a fresh session (the capture pass,
// mirroring how the gates render), renders it again in a second fresh session and refuses to write anything on any
// cross-pass divergence (the permanent capture-time determinism guard — this is the fresh-context corpus-pass
// invariant the gates rely on; see laneSession), stages the PNGs + schema-2 manifest beside the target directory,
// prints a per-scenario old-vs-new change summary when a prior set exists (the reviewable diff is generated at
// capture time), and only then swaps the staged set into place. On any error the target directory is untouched.
//
// The guard is per-lane, mirroring soak: raster is strictly bit-exact (any hash difference refuses), while the gpu
// and gpudmsaa lanes compare the verify pass per-pixel against the retained capture-pass buffers under the ±1 LSB
// envelope (imgdiff.Exact1) — software GL rasterizers wobble ±1 intermittently between GL sessions, proven
// driver-internal (see the soak doc comment for the evidence). Any delta > 1 still hard-refuses; within-envelope
// wobble is logged but accepted. What gets written is always the capture pass — the first context in a fresh process,
// the most reproducible render available (cold renders of separate processes agree within the same envelope).
//
// It refuses to replace a schema-1 manifest: those are the frozen Skia-era archive sets, which live only under
// goldens-skia/ and must never be silently overwritten — finding one under goldens/ means the working tree is in a
// state bless should not touch.
func bless(cfg *blessConfig) error {
	prior, hasPrior, err := priorManifest(cfg.dir)
	if err != nil {
		return err
	}
	staging := cfg.dir + ".staging"
	if err = os.RemoveAll(staging); err != nil {
		return err
	}
	if err = os.MkdirAll(staging, 0o755); err != nil {
		return err
	}
	committed := false
	defer func() {
		if !committed {
			os.RemoveAll(staging) //nolint:errcheck // best-effort cleanup; the primary error is already on its way out
		}
	}()

	// Capture pass: a fresh session renders the corpus in order, exactly as a gate will.
	capture, err := cfg.newSession()
	if err != nil {
		return err
	}
	m := golden.Manifest{
		Schema:     blessSchema,
		Platform:   cfg.platform,
		Lane:       cfg.lane,
		GLRenderer: capture.glRenderer,
		GLVersion:  capture.glVersion,
		CapturedAt: time.Now().UTC().Format("2006-01-02"),
	}
	envelope := cfg.lane != laneRaster
	var capturePixels [][]byte // envelope lanes only: retained capture-pass buffers (~30 MiB for the full corpus)
	if envelope {
		capturePixels = make([][]byte, len(cfg.scenarios))
	}
	for i, s := range cfg.scenarios {
		pixels := capture.render(s)
		if err = golden.WritePNG(filepath.Join(staging, s.Name+".png"), pixels, s.Width, s.Height); err != nil {
			capture.dispose()
			return err
		}
		if envelope {
			capturePixels[i] = pixels
		}
		m.Entries = append(m.Entries, golden.Entry{
			Name:   s.Name,
			Width:  s.Width,
			Height: s.Height,
			SHA256: golden.HashPixels(pixels),
		})
		fmt.Fprintf(cfg.out, "rendered %s\n", s.Name)
	}
	capture.dispose()

	// Verify pass: a second fresh session must reproduce the capture pass, every time, forever — bit-exactly on
	// raster, within the ±1 LSB envelope on the GPU lanes (the verify session is a warm 2nd context, which the Apple
	// software renderer wobbles ±1 on; see the bless doc comment). A divergence beyond the lane's tolerance means the
	// premise of self-captured goldens does not hold on this stack and nothing may be written.
	verify, err := cfg.newSession()
	if err != nil {
		return err
	}
	if verify.glRenderer != m.GLRenderer || verify.glVersion != m.GLVersion {
		verify.dispose()
		return fmt.Errorf(
			"bless: the GL stack changed between the capture and verify passes (%q %q -> %q %q) — refusing to write goldens",
			m.GLRenderer, m.GLVersion, verify.glRenderer, verify.glVersion)
	}
	wobbles := 0
	for i, s := range cfg.scenarios {
		pixels := verify.render(s)
		if !envelope {
			if rehash := golden.HashPixels(pixels); rehash != m.Entries[i].SHA256 {
				verify.dispose()
				return fmt.Errorf(
					"bless: %s rendered nondeterministically across fresh-session corpus passes (%s then %s) — refusing to write goldens",
					s.Name, m.Entries[i].SHA256, rehash)
			}
			continue
		}
		res, cmpErr := imgdiff.Compare(capturePixels[i], pixels, s.Width, s.Height, imgdiff.Exact1)
		if cmpErr != nil {
			verify.dispose()
			return cmpErr
		}
		switch {
		case res.DiffPixels > 0:
			verify.dispose()
			return fmt.Errorf(
				"bless: %s diverged beyond the ±1 envelope across fresh-session corpus passes (max channel delta %d, %d px beyond) — refusing to write goldens",
				s.Name, res.MaxDelta, res.DiffPixels)
		case res.AnyDiffPixels > 0:
			fmt.Fprintf(cfg.out, "wobble   %-32s verify pass: max channel delta %d on %d px (within the ±1 envelope; capture pass is canonical)\n",
				s.Name, res.MaxDelta, res.AnyDiffPixels)
			wobbles++
		}
	}
	verify.dispose()
	if envelope {
		fmt.Fprintf(cfg.out,
			"verify pass: all %d scenarios reproduced the capture pass within the ±1 LSB envelope (%d within-envelope wobble(s))\n",
			len(m.Entries), wobbles)
	} else {
		fmt.Fprintf(cfg.out, "verify pass: all %d scenarios reproduced their capture-pass hashes\n", len(m.Entries))
	}

	sort.Slice(m.Entries, func(i, j int) bool { return m.Entries[i].Name < m.Entries[j].Name })
	if err = golden.WriteManifest(staging, &m); err != nil {
		return err
	}

	if hasPrior {
		if err = blessSummary(cfg, &prior, &m, staging); err != nil {
			return err
		}
	} else {
		fmt.Fprintf(cfg.out, "no prior manifest in %s — capturing a new set (%d scenarios)\n", cfg.dir, len(m.Entries))
	}

	if err = os.RemoveAll(cfg.dir); err != nil {
		return err
	}
	if err = os.Rename(staging, cfg.dir); err != nil {
		return err
	}
	committed = true
	fmt.Fprintf(cfg.out, "blessed %d scenarios into %s (lane %s, platform %s, schema %d)\n",
		len(m.Entries), cfg.dir, cfg.lane, cfg.platform, blessSchema)
	return nil
}

// priorManifest loads the manifest already at dir, if any. It refuses (error) when the manifest is unreadable or is
// not schema 2 — in particular a schema-1 manifest marks a frozen Skia-era set that must never be overwritten in
// place.
func priorManifest(dir string) (m golden.Manifest, hasPrior bool, err error) {
	m, err = golden.ReadManifest(dir)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return golden.Manifest{}, false, nil
	case err != nil:
		return golden.Manifest{}, false, fmt.Errorf("bless: unreadable existing manifest in %s (fix or remove it): %w",
			dir, err)
	case m.Schema == 1:
		return golden.Manifest{}, false, fmt.Errorf(
			"bless: %s holds a schema-1 manifest — a frozen Skia-era golden set; move it aside to goldens-skia/ "+
				"before capturing (refusing to overwrite it)", dir)
	case m.Schema != blessSchema:
		return golden.Manifest{}, false, fmt.Errorf(
			"bless: %s holds an unexpected schema-%d manifest (refusing to overwrite it)", dir, m.Schema)
	default:
		return m, true, nil
	}
}

// blessSummary prints the per-scenario old-vs-new change summary: unchanged/changed/new/removed, with imgdiff stats
// for every changed image and optional side-by-side + heatmap artifacts. This runs before the staged set replaces the
// prior one, while the old PNGs are still in place.
func blessSummary(cfg *blessConfig, prior, next *golden.Manifest, staging string) error {
	priorByName := make(map[string]golden.Entry, len(prior.Entries))
	for _, e := range prior.Entries {
		priorByName[e.Name] = e
	}
	if cfg.artifacts != "" {
		if err := os.MkdirAll(cfg.artifacts, 0o755); err != nil {
			return err
		}
	}
	unchanged, changed, added := 0, 0, 0
	for _, ne := range next.Entries {
		pe, ok := priorByName[ne.Name]
		if !ok {
			fmt.Fprintf(cfg.out, "NEW       %-32s no prior golden\n", ne.Name)
			added++
			continue
		}
		delete(priorByName, ne.Name)
		if pe.SHA256 == ne.SHA256 {
			unchanged++
			continue
		}
		changed++
		if pe.Width != ne.Width || pe.Height != ne.Height {
			fmt.Fprintf(cfg.out, "CHANGED   %-32s size %dx%d -> %dx%d\n", ne.Name, pe.Width, pe.Height, ne.Width,
				ne.Height)
			continue
		}
		oldPix, _, _, err := golden.ReadPNG(filepath.Join(cfg.dir, ne.Name+".png"))
		if err != nil {
			return fmt.Errorf("bless: reading prior golden for change summary: %w", err)
		}
		newPix, _, _, err := golden.ReadPNG(filepath.Join(staging, ne.Name+".png"))
		if err != nil {
			return err
		}
		res, err := imgdiff.Compare(oldPix, newPix, ne.Width, ne.Height, imgdiff.Exact)
		if err != nil {
			return err
		}
		fmt.Fprintf(cfg.out, "CHANGED   %-32s %s\n", ne.Name, res)
		if cfg.artifacts != "" {
			if err = writeImagePNG(filepath.Join(cfg.artifacts, ne.Name+"-diff.png"),
				imgdiff.SideBySide(oldPix, newPix, ne.Width, ne.Height)); err != nil {
				return err
			}
			if err = writeImagePNG(filepath.Join(cfg.artifacts, ne.Name+"-heat.png"),
				imgdiff.Heatmap(oldPix, newPix, ne.Width, ne.Height)); err != nil {
				return err
			}
		}
	}
	removedNames := make([]string, 0, len(priorByName))
	for name := range priorByName {
		removedNames = append(removedNames, name)
	}
	sort.Strings(removedNames)
	for _, name := range removedNames {
		fmt.Fprintf(cfg.out, "REMOVED   %-32s no longer in the corpus\n", name)
	}
	fmt.Fprintf(cfg.out, "change summary vs prior set (captured %s): %d unchanged, %d changed, %d new, %d removed\n",
		priorCaptureLabel(prior), unchanged, changed, added, len(removedNames))
	return nil
}

// priorCaptureLabel describes when/where the prior set came from for the change-summary header.
func priorCaptureLabel(m *golden.Manifest) string {
	if m.CapturedAt == "" {
		return "date unknown"
	}
	return m.CapturedAt
}

// writeImagePNG encodes img to path.
func writeImagePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err = png.Encode(f, img); err != nil {
		f.Close() //nolint:errcheck // the encode error is the one worth reporting
		return err
	}
	return f.Close()
}
