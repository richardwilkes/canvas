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
	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// blessSchema is the manifest schema version bless writes (see golden.Manifest).
const blessSchema = 2

// renameDir is os.Rename, indirected so tests can drive the commit-swap failure paths (a rename that fails after the
// prior set has already been moved aside — an open handle on Windows, a cross-device staging directory, and so on).
var renameDir = os.Rename

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
// capture time), and only then swaps the staged set into place — renaming the prior set aside rather than removing it,
// so a failed swap puts it back. On any error the target directory is untouched.
//
// The guard is per-lane, mirroring soak: raster is strictly bit-exact (any hash difference refuses), while the gpu
// and gpudmsaa lanes compare the verify pass per-pixel against the retained capture-pass buffers under the ±1 LSB
// envelope (imgdiff.Exact1) — software GL rasterizers wobble ±1 intermittently between GL sessions, proven
// driver-internal (see the soak doc comment for the evidence). Any delta > 1 still hard-refuses, except for the
// scenarios gorender.DriverBimodal names for the live GL stack, where a beyond-envelope difference is the driver
// picking its other bit-exact flavor for the verify session; those are logged as `bimodal` and accepted for the same
// reason soak excuses them and the gates do not pixel-gate them. Within-envelope wobble is logged but accepted. What
// gets written is always the capture pass — the first context in a fresh process, the most reproducible render
// available (cold renders of separate processes agree within the same envelope).
//
// It refuses to replace a schema-1 manifest: those are the frozen Skia-era archive sets, which live only under
// goldens-skia/ and must never be silently overwritten — finding one under goldens/ means the working tree is in a
// state bless should not touch.
//
// It also repairs the disk state a previously interrupted commit swap can leave behind before doing anything else (see
// recoverInterruptedSwap), so re-running bless after that failure cannot destroy the preserved prior set.
func bless(cfg *blessConfig) error {
	if err := recoverInterruptedSwap(cfg); err != nil {
		return err
	}
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
	// The staging directory is scratch space until the commit swap moves it into place, so it is removed on every error
	// path. keepStaging turns that off for the two cases where staging is not scratch: the swap succeeded (staging no
	// longer exists), or the swap failed *and* the prior set could not be restored, leaving the staged capture as the
	// only copy of a verified render — deleting it there would throw away exactly what the operation produced.
	keepStaging := false
	defer func() {
		if !keepStaging {
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
	bimodals := 0
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
		case res.DiffPixels > 0 && gorender.DriverBimodal(verify.glRenderer, s.Name):
			// The verify session drew the driver's other flavor of this scenario. Not a refusal: the flip is
			// driver-internal (gorender.DriverBimodal) and the golden gates do not pixel-gate this scenario on this
			// stack, so the capture pass stays canonical exactly as it does for within-envelope wobble.
			fmt.Fprintf(cfg.out, "bimodal  %-32s verify pass: max channel delta %d on %d px (driver-internal flavor flip; capture pass is canonical)\n",
				s.Name, res.MaxDelta, res.AnyDiffPixels)
			bimodals++
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
			"verify pass: all %d scenarios reproduced the capture pass within the ±1 LSB envelope (%d within-envelope wobble(s), %d driver-bimodal flip(s))\n",
			len(m.Entries), wobbles, bimodals)
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

	// Commit swap. The prior set is renamed aside rather than removed, so a failed swap can put it back: there is no
	// window in which both it and the freshly captured set are gone from the disk. Removing it first would mean a
	// rename failure (an open handle on Windows, the target directory recreated concurrently) left the target gone and
	// the deferred cleanup deleting the only remaining copy of the verified capture.
	backup := cfg.dir + ".prior"
	hasBackup := false
	if _, err = os.Stat(cfg.dir); err == nil {
		// Any backup still on disk here is stale: recoverInterruptedSwap has already moved a preserved prior set back,
		// so a backup sitting beside an existing target can only be the residue of a successful swap whose backup
		// removal failed below. Clearing it is gated on the target existing precisely so that a preserved prior set —
		// which is only ever the last copy when the target is *missing* — can never be the thing removed here.
		if err = os.RemoveAll(backup); err != nil {
			return err
		}
		if err = renameDir(cfg.dir, backup); err != nil {
			return err
		}
		hasBackup = true
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if err = renameDir(staging, cfg.dir); err != nil {
		if hasBackup {
			if restoreErr := renameDir(backup, cfg.dir); restoreErr != nil {
				keepStaging = true
				return fmt.Errorf("bless: could not swap the staged set into %s (%w) and could not restore the prior "+
					"set from %s (%v) — the prior set is in %s and the captured set is in %s", cfg.dir, err, backup,
					restoreErr, backup, staging)
			}
		}
		return err
	}
	keepStaging = true // the swap consumed the staging directory
	if hasBackup {
		if rmErr := os.RemoveAll(backup); rmErr != nil {
			fmt.Fprintf(cfg.out, "note: the new set is in place, but the prior set could not be removed from %s: %v\n",
				backup, rmErr)
		}
	}
	fmt.Fprintf(cfg.out, "blessed %d scenarios into %s (lane %s, platform %s, schema %d)\n",
		len(m.Entries), cfg.dir, cfg.lane, cfg.platform, blessSchema)
	return nil
}

// recoverInterruptedSwap repairs the disk state left by the commit swap's worst case: the staged set could not be
// renamed into cfg.dir and the prior set could not be moved back either, so cfg.dir is gone and cfg.dir+".prior" holds
// the only copy of the prior set. Moving it back turns that into an ordinary re-bless — the prior set is read for the
// change summary and renamed aside again under the swap's normal protections — where leaving it would mean the next
// run captured with no prior set to compare against and, worse, with no target directory to move aside, so a second
// swap failure would find nothing to restore.
//
// A backup sitting beside a target directory that does exist is the harmless residue of a successful swap whose backup
// removal failed; it is left alone here and cleared by the commit swap.
func recoverInterruptedSwap(cfg *blessConfig) error {
	backup := cfg.dir + ".prior"
	if _, err := os.Stat(backup); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}
	if _, err := os.Stat(cfg.dir); err == nil {
		return nil
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if err := renameDir(backup, cfg.dir); err != nil {
		return fmt.Errorf("bless: %s is missing while a prior set is preserved in %s (an interrupted commit swap), but "+
			"it could not be moved back (%w) — restore it by hand before re-running", cfg.dir, backup, err)
	}
	fmt.Fprintf(cfg.out, "note: restored the prior set from %s into %s (an earlier bless left its commit swap "+
		"interrupted)\n", backup, cfg.dir)
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
