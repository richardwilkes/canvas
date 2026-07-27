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
	"bytes"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// bimodalScenarioName is a scenario gorender.DriverBimodal lists for the Apple software renderer.
// TestDriverBimodalTableCoversTestName asserts it really is listed, so these tests cannot silently degrade into
// testing the ordinary path if the table changes.
const bimodalScenarioName = "clip-persp"

// TestDriverBimodalTableCoversTestName guards the assumption the bimodal soak/bless tests rest on.
func TestDriverBimodalTableCoversTestName(t *testing.T) {
	if !gorender.DriverBimodal(gorender.AppleSoftwareRenderer, bimodalScenarioName) {
		t.Fatalf("gorender.DriverBimodal no longer lists %q for %q; update bimodalScenarioName to a listed scenario "+
			"or drop the bimodal tests if the quirk is gone", bimodalScenarioName, gorender.AppleSoftwareRenderer)
	}
}

// wobbleFactory returns a session factory whose first session renders solidRender(10) exactly and whose later
// sessions offset the first channel of the first pixel by delta — a synthetic warm-context wobble for exercising the
// per-lane pass-to-pass comparison in soak and bless.
func wobbleFactory(delta byte) func() (*laneSession, error) {
	sessions := 0
	base := solidRender(10)
	return func() (*laneSession, error) {
		sessions++
		warm := sessions > 1
		render := func(s scenario.Scenario) []byte {
			pixels := base(s)
			if warm {
				pixels[0] += delta
			}
			return pixels
		}
		return &laneSession{render: render, dispose: func() {}}, nil
	}
}

// bimodalScenarios is testScenarios plus the driver-bimodal scenario name, so the bimodal exception can be exercised
// alongside ordinary scenarios that must stay strictly guarded.
func bimodalScenarios() []scenario.Scenario {
	return append(testScenarios(), scenario.Scenario{Name: bimodalScenarioName, Width: 2, Height: 2})
}

// flipFactory returns a session factory whose later sessions offset the first channel of the named scenario by delta,
// reproducing the driver's per-session flavor flip. glRenderer is stamped on every session so DriverBimodal can key on
// it.
func flipFactory(name string, delta byte, glRenderer string) func() (*laneSession, error) {
	sessions := 0
	base := solidRender(10)
	return func() (*laneSession, error) {
		sessions++
		warm := sessions > 1
		render := func(s scenario.Scenario) []byte {
			pixels := base(s)
			if warm && s.Name == name {
				pixels[0] += delta
			}
			return pixels
		}
		return &laneSession{render: render, dispose: func() {}, glRenderer: glRenderer}, nil
	}
}

// TestSoakAcceptsDriverBimodalFlip covers the darwin_arm64 capture failure of 2026-07-27: on the Apple software
// renderer a listed scenario's beyond-envelope flavor flip is the driver's own behavior, so soak must report it
// without calling the corpus nondeterministic. Without this, capturing that lane succeeds or fails by luck.
func TestSoakAcceptsDriverBimodalFlip(t *testing.T) {
	var out bytes.Buffer
	cfg := soakConfig{
		out:        &out,
		n:          3,
		lane:       laneGPU,
		scenarios:  bimodalScenarios(),
		newSession: flipFactory(bimodalScenarioName, 209, gorender.AppleSoftwareRenderer),
	}
	if err := soak(&cfg); err != nil {
		t.Fatalf("soak with a driver-bimodal flip on the Apple software renderer: %v, want clean", err)
	}
	text := out.String()
	if !strings.Contains(text, "bimodal  "+bimodalScenarioName) {
		t.Fatalf("soak did not report the bimodal flip:\n%s", text)
	}
	if strings.Contains(text, "MISMATCH") {
		t.Fatalf("soak reported a MISMATCH for a driver-bimodal flip:\n%s", text)
	}
	if !strings.Contains(text, "driver-bimodal flip(s)") {
		t.Fatalf("soak summary did not count the bimodal flip:\n%s", text)
	}
}

// TestSoakBimodalExceptionIsNarrow verifies the exception is keyed to both the renderer and the scenario: the same
// beyond-envelope flip is still a hard mismatch on any other GL stack, and on the listed stack for any other
// scenario. This is what keeps it from becoming a general tolerance.
func TestSoakBimodalExceptionIsNarrow(t *testing.T) {
	for _, tc := range []struct {
		name        string
		scenario    string
		glRenderer  string
		description string
	}{
		{"other renderer", bimodalScenarioName, "llvmpipe (LLVM 15.0.7, 256 bits)", "listed scenario, unlisted stack"},
		{"other scenario", "alpha", gorender.AppleSoftwareRenderer, "unlisted scenario, listed stack"},
		{"no GL context", bimodalScenarioName, "", "raster-style empty renderer string"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cfg := soakConfig{
				out:        &out,
				n:          2,
				lane:       laneGPU,
				scenarios:  bimodalScenarios(),
				newSession: flipFactory(tc.scenario, 209, tc.glRenderer),
			}
			err := soak(&cfg)
			if err == nil || !strings.Contains(err.Error(), "mismatch") {
				t.Fatalf("%s: err = %v, want a mismatch failure", tc.description, err)
			}
			if !strings.Contains(out.String(), "MISMATCH "+tc.scenario) {
				t.Fatalf("%s: soak did not report the mismatch:\n%s", tc.description, out.String())
			}
		})
	}
}

// TestSoakGPUEnvelopeAcceptsDelta1 verifies the gpu lane's ±1 envelope: a later pass that differs from pass 1 by a
// single channel delta of 1 is clean — logged as informational wobble, not a mismatch.
func TestSoakGPUEnvelopeAcceptsDelta1(t *testing.T) {
	var out bytes.Buffer
	cfg := soakConfig{
		out:        &out,
		n:          3,
		lane:       laneGPU,
		scenarios:  testScenarios(),
		newSession: wobbleFactory(1),
	}
	if err := soak(&cfg); err != nil {
		t.Fatalf("soak with ±1 wobble on the gpu lane: %v, want clean", err)
	}
	text := out.String()
	if !strings.Contains(text, "wobble   alpha") || !strings.Contains(text, "within the ±1 envelope") {
		t.Fatalf("soak did not log the within-envelope wobble:\n%s", text)
	}
	if strings.Contains(text, "MISMATCH") {
		t.Fatalf("soak reported a MISMATCH for within-envelope wobble:\n%s", text)
	}
	if got := strings.Count(text, "hash "); got != len(testScenarios()) {
		t.Fatalf("soak printed %d pass-1 hash lines, want %d:\n%s", got, len(testScenarios()), text)
	}
	if !strings.Contains(text, "corpus digest") {
		t.Fatalf("soak did not print the corpus digest:\n%s", text)
	}
}

// TestSoakGPUEnvelopeRefusesDelta2 verifies any channel delta > 1 is still a hard mismatch on the gpu lanes.
func TestSoakGPUEnvelopeRefusesDelta2(t *testing.T) {
	var out bytes.Buffer
	cfg := soakConfig{
		out:        &out,
		n:          2,
		lane:       laneGPUDMSAA,
		scenarios:  testScenarios(),
		newSession: wobbleFactory(2),
	}
	err := soak(&cfg)
	if err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("soak with a delta-2 divergence on a GPU lane: err = %v, want a mismatch failure", err)
	}
	if !strings.Contains(out.String(), "beyond the ±1 envelope") {
		t.Fatalf("soak did not report the over-envelope delta:\n%s", out.String())
	}
}

// TestSoakRasterStaysStrict verifies the raster lane refuses even a ±1 delta — its guard is strict hash comparison,
// with no envelope.
func TestSoakRasterStaysStrict(t *testing.T) {
	var out bytes.Buffer
	cfg := soakConfig{
		out:        &out,
		n:          2,
		lane:       laneRaster,
		scenarios:  testScenarios(),
		newSession: wobbleFactory(1),
	}
	err := soak(&cfg)
	if err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("soak with a ±1 delta on the raster lane: err = %v, want a strict-hash mismatch failure", err)
	}
	if !strings.Contains(out.String(), "MISMATCH alpha") {
		t.Fatalf("soak did not report the raster hash mismatch:\n%s", out.String())
	}
}

// TestSoakCleanDeterministic verifies a fully deterministic renderer soaks clean on both a strict and an envelope
// lane.
func TestSoakCleanDeterministic(t *testing.T) {
	for _, lane := range []string{laneRaster, laneGPU} {
		var out bytes.Buffer
		cfg := soakConfig{
			out:        &out,
			n:          3,
			lane:       lane,
			scenarios:  testScenarios(),
			newSession: sessionFactory(solidRender(10), "", ""),
		}
		if err := soak(&cfg); err != nil {
			t.Fatalf("lane %s: soak with a deterministic renderer: %v", lane, err)
		}
		if strings.Contains(out.String(), "wobble   ") || strings.Contains(out.String(), "MISMATCH") {
			t.Fatalf("lane %s: deterministic soak logged wobble or mismatch:\n%s", lane, out.String())
		}
	}
}
