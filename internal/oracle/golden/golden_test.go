// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package golden

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// Round-trip both a translucent and a fully-opaque buffer. The opaque case matters: png.Encode writes fully-opaque
// images as truecolor without an alpha channel, which png.Decode returns as *image.RGBA rather than *image.NRGBA, and
// ReadPNG must accept both.
func TestWriteReadPNGRoundTrip(t *testing.T) {
	const w, h = 3, 2
	dir := t.TempDir()
	for name, alpha := range map[string]byte{"opaque": 255, "translucent": 128} {
		t.Run(name, func(t *testing.T) {
			pixels := make([]byte, w*h*4)
			for i := 0; i < len(pixels); i += 4 {
				pixels[i] = byte(i)
				pixels[i+1] = byte(i + 1)
				pixels[i+2] = byte(i + 2)
				pixels[i+3] = alpha
			}
			path := filepath.Join(dir, name+".png")
			if err := WritePNG(path, pixels, w, h); err != nil {
				t.Fatalf("WritePNG: %v", err)
			}
			got, gw, gh, err := ReadPNG(path)
			if err != nil {
				t.Fatalf("ReadPNG: %v", err)
			}
			if gw != w || gh != h {
				t.Fatalf("ReadPNG size = %dx%d, want %dx%d", gw, gh, w, h)
			}
			if !bytes.Equal(got, pixels) {
				t.Fatalf("ReadPNG pixels differ from what WritePNG was given")
			}
		})
	}
}

// TestManifestSchema2RoundTrip writes a schema-2 manifest (the self-captured-goldens form: lane, GL stack identity,
// capture date) and reads it back unchanged.
func TestManifestSchema2RoundTrip(t *testing.T) {
	dir := t.TempDir()
	want := Manifest{
		Schema:     2,
		Platform:   "darwin_arm64",
		Lane:       "gpu",
		GLRenderer: "Apple Software Renderer",
		GLVersion:  "4.1 ATI-6.5.5",
		CapturedAt: "2026-07-17",
		Entries: []Entry{
			{Name: "arcs", Width: 256, Height: 256, SHA256: "aa"},
			{Name: "hairlines", Width: 256, Height: 256, SHA256: "bb"},
		},
	}
	if err := WriteManifest(dir, &want); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	got, err := ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("schema-2 round trip mismatch:\ngot  %+v\nwant %+v", got, want)
	}
}

// TestManifestOptionalFieldsAbsent reads a manifest carrying only the always-written keys and verifies the optional
// ones come back zero-valued rather than erroring — a raster set records no GL stack, so this is the shape half the
// committed manifests have on disk.
func TestManifestOptionalFieldsAbsent(t *testing.T) {
	dir := t.TempDir()
	raw := `{
	"platform": "darwin_arm64",
	"entries": [
		{
			"name": "arcs",
			"sha256": "cc",
			"width": 256,
			"height": 256
		}
	],
	"schema": 2
}
`
	if err := os.WriteFile(filepath.Join(dir, ManifestName), []byte(raw), 0o644); err != nil {
		t.Fatalf("writing manifest: %v", err)
	}
	got, err := ReadManifest(dir)
	if err != nil {
		t.Fatalf("ReadManifest: %v", err)
	}
	want := Manifest{
		Schema:   2,
		Platform: "darwin_arm64",
		Entries:  []Entry{{Name: "arcs", Width: 256, Height: 256, SHA256: "cc"}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("manifest read mismatch (absent optional fields must be zero-valued):\ngot  %+v\nwant %+v", got, want)
	}
}
