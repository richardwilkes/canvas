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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"

	"github.com/richardwilkes/canvas/internal/oracle/golden"
	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// soakConfig carries one soak run. The session factory and scenario list are injected so tests can drive soak with a
// synthetic corpus and a deliberately wobbling renderer.
type soakConfig struct {
	out        io.Writer
	newSession func() (*laneSession, error)
	lane       string // laneRaster, laneGPU, or laneGPUDMSAA — selects strict-hash vs ±1-envelope comparison
	scenarios  []scenario.Scenario
	n          int // number of full-corpus passes (>= 2)
}

// soak renders the full corpus n times and fails on any pass-to-pass divergence — the determinism proof behind
// comparing self-captured goldens. Each pass runs in a fresh session (a brand-new GL context for the GPU lanes; see
// laneSession for why a warm context is not re-rendered), so what soak proves is exactly the invariant the gates rely
// on: a fresh-context, in-order corpus pass reproduces pass 1 every time. Repetition in-process catches
// scheduling/parallelism nondeterminism (fillPathShadedParallel and friends) and session-recreation state leaks
// (sync.Pool reuse, map iteration order, process-global caches); the printed pass-1 per-scenario `hash` lines and
// corpus digest let two separate invocations be compared to catch cross-process nondeterminism (both are always
// pass-1-based, so they stay comparable across processes regardless of later-pass wobble — though for the GPU lanes
// cross-process agreement itself only holds within the envelope below, so digest equality there is expected, not
// guaranteed; raster digests must match exactly).
//
// What "reproduces" means is per-lane. The raster lane is strictly bit-exact: any pass-to-pass hash difference is a
// mismatch. The gpu and gpudmsaa lanes compare later passes per-pixel against the retained pass-1 buffers under the
// ±1 LSB envelope (imgdiff.Exact1): a pass is clean when every channel delta is ≤ 1 and zero pixels exceed it. The
// envelope exists because software GL rasterizers (measured on Apple's; 2026-07-17) wobble ±1 intermittently between
// GL sessions — mostly on 2nd+ sessions in one process, occasionally between cold first-context renders of separate
// processes. That wobble is driver-internal, not an app-level read: with every texture texel forced to a known value,
// every intermediate's full backing store and the complete GL command/data stream were byte-identical across runs
// whose outputs differed. Real rendering breaks measure ≥32 LSB, so the envelope costs no detection power.
// Within-envelope wobble is logged per scenario (max delta + differing-pixel count) but is informational, not a
// failure; any delta > 1 is a MISMATCH failure exactly as a raster hash difference is.
//
// The one exception is a scenario gorender.DriverBimodal names for the live GL stack: there the driver picks one of
// two bit-exact flavors per session, so a beyond-envelope difference is the stack's own behavior rather than
// app-level nondeterminism. Those are logged as `bimodal` and counted separately. The exception exists so capture and
// gating agree — the golden gates already report-not-gate exactly these scenario/renderer pairs, and without the
// matching rule here a lane's capture succeeded or failed by luck depending on whether the flip happened to land
// mid-soak (darwin_arm64 gpu, 2026-07-27). It is deliberately not a general tolerance: every other beyond-envelope
// difference still fails.
//
// The full corpus soaks in every lane, text scenarios included: the self-captured raster sets are per-platform, so
// platform-scaler-dependent output is in scope for them too.
//
// Rendered inline on the calling goroutine — for the GPU lanes the GL context is thread-bound (see laneSession).
func soak(cfg *soakConfig) error {
	envelope := cfg.lane != laneRaster
	scens := cfg.scenarios
	base := make([]string, len(scens))
	var basePixels [][]byte // envelope lanes only: retained pass-1 buffers (~30 MiB for the full corpus)
	if envelope {
		basePixels = make([][]byte, len(scens))
	}
	mismatches := 0
	wobbles := 0
	bimodals := 0
	for iter := 1; iter <= cfg.n; iter++ {
		session, err := cfg.newSession()
		if err != nil {
			if iter == 1 {
				return err
			}
			// A context that came up for pass 1 and not for pass n is its own kind of broken; report it as a plain
			// failure (err.Error() rather than %w strips errNoGLContext so this cannot masquerade as the loud-skip
			// case).
			return fmt.Errorf("soak: session for pass %d failed after pass 1 succeeded: %s", iter, err.Error())
		}
		iterMismatches := 0
		for i, s := range scens {
			pixels := session.render(s)
			if iter == 1 {
				base[i] = golden.HashPixels(pixels)
				if envelope {
					basePixels[i] = pixels
				}
				fmt.Fprintf(cfg.out, "hash %-32s %s\n", s.Name, base[i])
				continue
			}
			if !envelope {
				if h := golden.HashPixels(pixels); h != base[i] {
					fmt.Fprintf(cfg.out, "MISMATCH %-32s pass %d: %s, first pass was %s\n", s.Name, iter, h, base[i])
					iterMismatches++
				}
				continue
			}
			res, cmpErr := imgdiff.Compare(basePixels[i], pixels, s.Width, s.Height, imgdiff.Exact1)
			if cmpErr != nil {
				session.dispose()
				return cmpErr
			}
			switch {
			case res.DiffPixels > 0 && gorender.DriverBimodal(session.glRenderer, s.Name):
				// The driver picked its other internal precision mode for this session. Reported, not counted: it is
				// not app-level nondeterminism and the golden gates do not pixel-gate this scenario on this stack
				// either (gorender.DriverBimodal). Excusing it here is what keeps capture from succeeding or failing
				// by luck on the Apple software renderer.
				fmt.Fprintf(cfg.out, "bimodal  %-32s pass %d: max channel delta %d on %d px (driver-internal flavor flip; reported, not a mismatch)\n",
					s.Name, iter, res.MaxDelta, res.AnyDiffPixels)
				bimodals++
			case res.DiffPixels > 0:
				fmt.Fprintf(cfg.out, "MISMATCH %-32s pass %d: max channel delta %d, %d px beyond the ±1 envelope (%d px differ at all)\n",
					s.Name, iter, res.MaxDelta, res.DiffPixels, res.AnyDiffPixels)
				iterMismatches++
			case res.AnyDiffPixels > 0:
				fmt.Fprintf(cfg.out, "wobble   %-32s pass %d: max channel delta %d on %d px (within the ±1 envelope; informational)\n",
					s.Name, iter, res.MaxDelta, res.AnyDiffPixels)
				wobbles++
			}
		}
		session.dispose()
		if iter > 1 {
			fmt.Fprintf(cfg.out, "pass %d/%d: %d scenario(s) diverged from the first pass\n", iter, cfg.n, iterMismatches)
		}
		mismatches += iterMismatches
	}
	digest := sha256.New()
	for _, h := range base {
		io.WriteString(digest, h) //nolint:errcheck // hash.Hash writes never fail
	}
	fmt.Fprintf(cfg.out, "corpus digest (%d scenarios, %d passes): %s\n", len(scens), cfg.n,
		hex.EncodeToString(digest.Sum(nil)))
	if mismatches > 0 {
		return fmt.Errorf(
			"soak: %d pass-to-pass mismatch(es) across %d passes — the corpus does not render deterministically on this stack",
			mismatches, cfg.n)
	}
	if envelope {
		fmt.Fprintf(cfg.out,
			"soak: all %d scenarios stayed within the ±1 LSB envelope of pass 1 across %d fresh-session passes (%d within-envelope wobble(s), %d driver-bimodal flip(s))\n",
			len(scens), cfg.n, wobbles, bimodals)
	} else {
		fmt.Fprintf(cfg.out, "soak: all %d scenarios rendered bit-identically across %d fresh-session passes\n",
			len(scens), cfg.n)
	}
	return nil
}
