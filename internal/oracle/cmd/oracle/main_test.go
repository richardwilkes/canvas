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
	"strings"
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

// TestGenWritesBlessableSchema pins gen's manifest schema to the one bless writes. bless refuses to overwrite a set
// carrying any other schema, so stamping one on freshly rendered output poisons the directory against every later
// capture.
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

// TestDiffCrossChecksPNGsWhenManifestsDiffer is the regression test for the manifest/PNG cross-check that used to run
// only inside the equal-hashes branch. A golden PNG replaced without updating manifest.json leaves that side's entry
// hash permanently stale, so the two manifest hashes never match, the pixel comparison is the only thing that ever
// runs, and the desynchronization reports ok forever — the entry's integrity hash written but never verified.
func TestDiffCrossChecksPNGsWhenManifestsDiffer(t *testing.T) {
	root := t.TempDir()
	aDir := filepath.Join(root, "a")
	bDir := filepath.Join(root, "b")
	writeTestSet(t, aDir, solidRender(10))
	writeTestSet(t, bDir, solidRender(20))
	want := len(testScenarios())
	failures, err := diff(aDir, bDir, imgdiff.Exact, "")
	if err != nil {
		t.Fatalf("diff over two differing sets: %v", err)
	}
	if failures != want {
		t.Fatalf("diff over two differing sets: %d failure(s), want %d", failures, want)
	}

	// Take b's image for alpha without taking b's manifest entry — a hand edit or a partial merge. The pixels now
	// match, so the comparison passes; only re-hashing against a's own entry can catch that a's manifest is stale.
	alpha := testScenarios()[0]
	bpx, w, h, err := golden.ReadPNG(filepath.Join(bDir, alpha.Name+".png"))
	if err != nil {
		t.Fatal(err)
	}
	if err = golden.WritePNG(filepath.Join(aDir, alpha.Name+".png"), bpx, w, h); err != nil {
		t.Fatal(err)
	}
	failures, err = diff(aDir, bDir, imgdiff.Exact, "")
	if err != nil {
		t.Fatalf("diff with a golden desynchronized from its manifest: %v", err)
	}
	if failures != want {
		t.Fatalf("diff with a golden whose pixels no longer hash to its own manifest entry: %d failure(s), want %d "+
			"— the desynchronized entry reported ok", failures, want)
	}
}

// TestDiffRejectsUnusableManifests covers diff's degenerate-input hole: golden.ReadManifest accepts any well-formed
// JSON, so `{}`, `null`, and a schema-2 manifest listing nothing all unmarshal without error into something both of
// diff's loops skip entirely — failures stays 0 and the run prints "all 0 scenarios pass". `oracle gen` + `oracle
// diff` is the raster lane's only gate, so a manifest that cannot describe a golden set has to be a hard error.
func TestDiffRejectsUnusableManifests(t *testing.T) {
	root := t.TempDir()
	aDir := filepath.Join(root, "a")
	bDir := filepath.Join(root, "b")
	dirs := []string{aDir, bDir}
	writeManifestBytes := func(data string) {
		t.Helper()
		for _, d := range dirs {
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(d, golden.ManifestName), []byte(data), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, data := range []string{"{}\n", "null\n"} {
		writeManifestBytes(data)
		_, err := diff(aDir, bDir, imgdiff.Exact, "")
		if err == nil {
			t.Fatalf("diff over two %q manifests reported success, want a refusal", strings.TrimSpace(data))
		}
		if !strings.Contains(err.Error(), "schema") {
			t.Fatalf("diff over two %q manifests: err = %v, want a missing-schema diagnosis",
				strings.TrimSpace(data), err)
		}
	}
	for _, d := range dirs {
		m := golden.Manifest{Schema: blessSchema, Platform: "test_platform"}
		if err := golden.WriteManifest(d, &m); err != nil {
			t.Fatal(err)
		}
	}
	_, err := diff(aDir, bDir, imgdiff.Exact, "")
	if err == nil || !strings.Contains(err.Error(), "no entries") {
		t.Fatalf("diff over two entry-less manifests: err = %v, want a nothing-to-compare refusal", err)
	}

	// A real set on both sides still compares normally — the guard must not reject usable input.
	writeTestSet(t, aDir, solidRender(10))
	writeTestSet(t, bDir, solidRender(10))
	failures, err := diff(aDir, bDir, imgdiff.Exact, "")
	if err != nil || failures != 0 {
		t.Fatalf("diff over two identical real sets: %d failure(s), err = %v; want 0 and no error", failures, err)
	}
}
