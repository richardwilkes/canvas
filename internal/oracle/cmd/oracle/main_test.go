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
	"os"
	"path/filepath"
	"testing"

	"github.com/richardwilkes/canvas/internal/oracle/golden"
	"github.com/richardwilkes/canvas/internal/oracle/imgdiff"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// writeTestSet writes a golden set of the synthetic corpus into dir through render, exactly as `oracle gen` does.
func writeTestSet(t *testing.T, dir string, render func(scenario.Scenario) []byte) {
	t.Helper()
	if err := writeGoldens(dir, testScenarios(), render); err != nil {
		t.Fatalf("writeGoldens into %s: %v", dir, err)
	}
}

// TestGenWritesBlessableSchema pins gen's manifest schema. Schema 1 means "a frozen Skia-era archive set" — a
// designation both bless and capture.sh refuse to overwrite — so stamping it on freshly rendered output poisons
// the directory against every later capture, with a diagnosis naming an archive that was never there.
func TestGenWritesBlessableSchema(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gen")
	writeTestSet(t, dir, solidRender(10))
	m, err := golden.ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if m.Schema != blessSchema {
		t.Fatalf("gen wrote a schema-%d manifest, want schema %d", m.Schema, blessSchema)
	}
	prior, hasPrior, err := priorManifest(dir)
	if err != nil {
		t.Fatalf("bless cannot read a gen-written set as its prior: %v", err)
	}
	if !hasPrior || len(prior.Entries) != len(testScenarios()) {
		t.Fatalf("priorManifest over a gen-written set: hasPrior = %v, %d entries, want true and %d", hasPrior,
			len(prior.Entries), len(testScenarios()))
	}
}

// TestDiffReadsPassingGoldenPNGs is the regression test for diff's manifest-hash short circuit: the raster lane's only
// gate is `oracle gen` + `oracle diff`, so a corrupted or truncated checked-in golden PNG whose manifest entry is still
// correct went unread — and therefore undetected — on the passing path.
func TestDiffReadsPassingGoldenPNGs(t *testing.T) {
	root := t.TempDir()
	aDir := filepath.Join(root, "a")
	bDir := filepath.Join(root, "b")
	writeTestSet(t, aDir, solidRender(10))
	writeTestSet(t, bDir, solidRender(10))
	failures, err := diff(aDir, bDir, imgdiff.Exact, "")
	if err != nil {
		t.Fatalf("diff over two identical sets: %v", err)
	}
	if failures != 0 {
		t.Fatalf("diff over two identical sets: %d failure(s), want 0", failures)
	}

	// Rewrite one PNG's pixels, leaving both manifests untouched: the two manifest entries still agree with each
	// other, so only opening the file can catch this.
	alpha := testScenarios()[0]
	corrupt := bytes.Repeat([]byte{0x7F}, alpha.Width*alpha.Height*4)
	if err = golden.WritePNG(filepath.Join(aDir, alpha.Name+".png"), corrupt, alpha.Width, alpha.Height); err != nil {
		t.Fatal(err)
	}
	failures, err = diff(aDir, bDir, imgdiff.Exact, "")
	if err != nil {
		t.Fatalf("diff with a golden whose pixels no longer match its manifest: %v", err)
	}
	if failures != 1 {
		t.Fatalf("diff with a golden whose pixels no longer match its manifest: %d failure(s), want 1", failures)
	}

	// A file that cannot decode at all has to reach the caller as an error (exit 1), never a silent pass.
	beta := testScenarios()[1]
	if err = os.WriteFile(filepath.Join(aDir, beta.Name+".png"), []byte("\x89PNG\r\n\x1a\ntruncated"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err = diff(aDir, bDir, imgdiff.Exact, ""); err == nil {
		t.Fatal("diff over a truncated golden PNG returned no error")
	}
}
