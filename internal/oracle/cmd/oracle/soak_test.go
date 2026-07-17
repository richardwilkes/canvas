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

	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

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
