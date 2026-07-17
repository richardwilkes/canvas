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
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// testScenarios is a tiny synthetic corpus for driving bless without rendering anything real (the session factory is
// injected, so Draw is never called).
func testScenarios() []scenario.Scenario {
	return []scenario.Scenario{
		{Name: "alpha", Width: 2, Height: 2},
		{Name: "beta", Width: 2, Height: 2},
	}
}

// solidRender returns a deterministic renderer that fills each scenario with a per-scenario byte value offset by base,
// so two renderers with different bases produce different (but internally deterministic) images.
func solidRender(base byte) func(scenario.Scenario) []byte {
	return func(s scenario.Scenario) []byte {
		v := base + byte(len(s.Name))
		pixels := make([]byte, s.Width*s.Height*4)
		for i := range pixels {
			pixels[i] = v
		}
		return pixels
	}
}

// sessionFactory wraps a render func as a session factory whose sessions carry the given GL identity strings.
func sessionFactory(render func(scenario.Scenario) []byte, glRenderer, glVersion string) func() (*laneSession, error) {
	return func() (*laneSession, error) {
		return &laneSession{render: render, dispose: func() {}, glRenderer: glRenderer, glVersion: glVersion}, nil
	}
}

func TestBlessDirDerivation(t *testing.T) {
	got := blessDir("raster", "darwin", "arm64")
	want := filepath.Join("goldens", "raster", "darwin_arm64")
	if got != want {
		t.Fatalf("blessDir = %q, want %q", got, want)
	}
	got = blessDir("gpudmsaa", "windows", "amd64")
	want = filepath.Join("goldens", "gpudmsaa", "windows_amd64")
	if got != want {
		t.Fatalf("blessDir = %q, want %q", got, want)
	}
}

// TestBlessWritesSchema2 checks the happy path: a fresh capture writes the PNGs and a schema-2 manifest carrying the
// lane, platform, GL stack identity, and capture date.
func TestBlessWritesSchema2(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "gpu", "test_platform")
	const glRenderer = "llvmpipe (LLVM 17.0.6, 256 bits)"
	const glVersion = "4.5 (Core Profile) Mesa 26.1.3"
	var out bytes.Buffer
	cfg := blessConfig{
		out:        &out,
		dir:        dir,
		lane:       laneGPU,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: sessionFactory(solidRender(10), glRenderer, glVersion),
	}
	if err := bless(&cfg); err != nil {
		t.Fatalf("bless: %v", err)
	}
	m, err := golden.ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if m.Schema != 2 || m.Lane != laneGPU || m.Platform != "test_platform" ||
		m.GLRenderer != glRenderer || m.GLVersion != glVersion || m.CapturedAt == "" {
		t.Fatalf("manifest metadata wrong: %+v", m)
	}
	if len(m.Entries) != 2 || m.Entries[0].Name != "alpha" || m.Entries[1].Name != "beta" {
		t.Fatalf("manifest entries wrong: %+v", m.Entries)
	}
	for _, e := range m.Entries {
		pixels, w, h, pngErr := golden.ReadPNG(filepath.Join(dir, e.Name+".png"))
		if pngErr != nil {
			t.Fatalf("ReadPNG %s: %v", e.Name, pngErr)
		}
		if w != e.Width || h != e.Height {
			t.Fatalf("%s: PNG is %dx%d, manifest says %dx%d", e.Name, w, h, e.Width, e.Height)
		}
		if golden.HashPixels(pixels) != e.SHA256 {
			t.Fatalf("%s: PNG pixels do not hash to the manifest entry", e.Name)
		}
	}
	if _, err = os.Stat(dir + ".staging"); !os.IsNotExist(err) {
		t.Fatalf("staging directory left behind after a successful bless")
	}
}

// TestBlessRefusesNondeterminism injects a session factory whose second session renders differently from the first and
// verifies bless refuses and writes nothing — the permanent capture-time determinism guard over fresh-session corpus
// passes.
func TestBlessRefusesNondeterminism(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "raster", "test_platform")
	sessions := 0
	newSession := func() (*laneSession, error) {
		sessions++
		return &laneSession{render: solidRender(byte(10 * sessions)), dispose: func() {}}, nil
	}
	var out bytes.Buffer
	cfg := blessConfig{
		out:        &out,
		dir:        dir,
		lane:       laneRaster,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: newSession,
	}
	err := bless(&cfg)
	if err == nil || !strings.Contains(err.Error(), "nondeterministically") {
		t.Fatalf("bless with a nondeterministic renderer: err = %v, want a nondeterminism refusal", err)
	}
	if _, statErr := os.Stat(dir); !os.IsNotExist(statErr) {
		t.Fatalf("bless wrote the target directory despite refusing")
	}
	if _, statErr := os.Stat(dir + ".staging"); !os.IsNotExist(statErr) {
		t.Fatalf("bless left its staging directory behind after refusing")
	}
}

// TestBlessGPUEnvelopeAcceptsDelta1 verifies the gpu lane's cross-pass guard tolerates a verify pass wobbling within
// the ±1 envelope, logs the wobble, and writes the capture pass (the canonical cold render) rather than the verify
// pass.
func TestBlessGPUEnvelopeAcceptsDelta1(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "gpu", "test_platform")
	var out bytes.Buffer
	cfg := blessConfig{
		out:        &out,
		dir:        dir,
		lane:       laneGPU,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: wobbleFactory(1),
	}
	if err := bless(&cfg); err != nil {
		t.Fatalf("bless with ±1 verify-pass wobble on the gpu lane: %v, want success", err)
	}
	text := out.String()
	if !strings.Contains(text, "wobble   alpha") || !strings.Contains(text, "within the ±1 envelope") {
		t.Fatalf("bless did not log the within-envelope wobble:\n%s", text)
	}
	m, err := golden.ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	// The written set must be the capture pass: solidRender(10) exactly, without the verify session's +1.
	wantHash := golden.HashPixels(solidRender(10)(testScenarios()[0]))
	if m.Entries[0].SHA256 != wantHash {
		t.Fatalf("bless did not write the capture pass as canonical")
	}
	pixels, _, _, err := golden.ReadPNG(filepath.Join(dir, m.Entries[0].Name+".png"))
	if err != nil {
		t.Fatalf("ReadPNG: %v", err)
	}
	if golden.HashPixels(pixels) != wantHash {
		t.Fatalf("written PNG is not the capture-pass render")
	}
}

// TestBlessGPUEnvelopeRefusesDelta2 verifies the gpu-lane guard still hard-refuses any cross-pass channel delta > 1.
func TestBlessGPUEnvelopeRefusesDelta2(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "gpudmsaa", "test_platform")
	cfg := blessConfig{
		out:        &bytes.Buffer{},
		dir:        dir,
		lane:       laneGPUDMSAA,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: wobbleFactory(2),
	}
	err := bless(&cfg)
	if err == nil || !strings.Contains(err.Error(), "±1 envelope") {
		t.Fatalf("bless with a delta-2 verify divergence: err = %v, want an over-envelope refusal", err)
	}
	if _, statErr := os.Stat(dir); !os.IsNotExist(statErr) {
		t.Fatalf("bless wrote the target directory despite refusing")
	}
	if _, statErr := os.Stat(dir + ".staging"); !os.IsNotExist(statErr) {
		t.Fatalf("bless left its staging directory behind after refusing")
	}
}

// TestBlessRasterStaysStrict verifies the raster lane's guard has no envelope: even a ±1 cross-pass delta refuses.
func TestBlessRasterStaysStrict(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "raster", "test_platform")
	cfg := blessConfig{
		out:        &bytes.Buffer{},
		dir:        dir,
		lane:       laneRaster,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: wobbleFactory(1),
	}
	err := bless(&cfg)
	if err == nil || !strings.Contains(err.Error(), "nondeterministically") {
		t.Fatalf("bless with a ±1 delta on the raster lane: err = %v, want a strict-hash refusal", err)
	}
	if _, statErr := os.Stat(dir); !os.IsNotExist(statErr) {
		t.Fatalf("bless wrote the target directory despite refusing")
	}
}

// TestBlessRefusesGLStackChange verifies bless refuses when the verify pass comes up on a different GL stack than the
// capture pass.
func TestBlessRefusesGLStackChange(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "gpu", "test_platform")
	sessions := 0
	newSession := func() (*laneSession, error) {
		sessions++
		return &laneSession{
			render:     solidRender(10),
			dispose:    func() {},
			glRenderer: []string{"first stack", "second stack"}[sessions-1],
			glVersion:  "4.1",
		}, nil
	}
	cfg := blessConfig{
		out:        &bytes.Buffer{},
		dir:        dir,
		lane:       laneGPU,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: newSession,
	}
	err := bless(&cfg)
	if err == nil || !strings.Contains(err.Error(), "GL stack changed") {
		t.Fatalf("bless across GL stacks: err = %v, want a GL-stack-changed refusal", err)
	}
	if _, statErr := os.Stat(dir); !os.IsNotExist(statErr) {
		t.Fatalf("bless wrote the target directory despite refusing")
	}
}

// TestBlessRefusesSchema1 verifies bless will not overwrite a schema-1 manifest — the frozen Skia-era sets must be
// moved aside to goldens-skia/, never silently replaced.
func TestBlessRefusesSchema1(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "goldens", "gpu", "test_platform")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	skia := golden.Manifest{Schema: 1, Platform: "test_platform"}
	if err := golden.WriteManifest(dir, &skia); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(filepath.Join(dir, golden.ManifestName))
	if err != nil {
		t.Fatal(err)
	}
	cfg := blessConfig{
		out:        &bytes.Buffer{},
		dir:        dir,
		lane:       laneGPU,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: sessionFactory(solidRender(10), "", ""),
	}
	blessErr := bless(&cfg)
	if blessErr == nil || !strings.Contains(blessErr.Error(), "goldens-skia") {
		t.Fatalf("bless over a schema-1 manifest: err = %v, want a refusal pointing at goldens-skia/", blessErr)
	}
	after, err := os.ReadFile(filepath.Join(dir, golden.ManifestName))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatalf("bless modified the schema-1 manifest it refused to overwrite")
	}
}

// TestBlessChangeSummary re-blesses over a prior schema-2 set with different pixels and checks the old-vs-new summary:
// changed scenarios get imgdiff stats and artifacts, and the target directory ends up holding the new set.
func TestBlessChangeSummary(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "goldens", "raster", "test_platform")
	artifacts := filepath.Join(root, "artifacts")
	first := blessConfig{
		out:        &bytes.Buffer{},
		dir:        dir,
		lane:       laneRaster,
		platform:   "test_platform",
		scenarios:  testScenarios(),
		newSession: sessionFactory(solidRender(10), "", ""),
	}
	if err := bless(&first); err != nil {
		t.Fatalf("first bless: %v", err)
	}
	var out bytes.Buffer
	second := blessConfig{
		out:        &out,
		dir:        dir,
		lane:       laneRaster,
		platform:   "test_platform",
		artifacts:  artifacts,
		scenarios:  testScenarios(),
		newSession: sessionFactory(solidRender(20), "", ""), // every image changes
	}
	if err := bless(&second); err != nil {
		t.Fatalf("second bless: %v", err)
	}
	text := out.String()
	if !strings.Contains(text, "CHANGED   alpha") || !strings.Contains(text, "CHANGED   beta") {
		t.Fatalf("change summary missing CHANGED lines:\n%s", text)
	}
	if !strings.Contains(text, "2 changed") {
		t.Fatalf("change summary missing totals line:\n%s", text)
	}
	for _, name := range []string{"alpha-diff.png", "alpha-heat.png", "beta-diff.png", "beta-heat.png"} {
		if _, err := os.Stat(filepath.Join(artifacts, name)); err != nil {
			t.Fatalf("missing change artifact %s: %v", name, err)
		}
	}
	m, err := golden.ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest after re-bless: %v", err)
	}
	wantHash := golden.HashPixels(solidRender(20)(testScenarios()[0]))
	if m.Entries[0].SHA256 != wantHash {
		t.Fatalf("re-bless did not swap the new set into place")
	}
}
