// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package golden reads and writes golden images and their manifest. Goldens are PNGs whose RGBA samples are the
// surface's RGBA8888-*premultiplied* bytes stored verbatim (PNG itself is alpha-agnostic and lossless for 8-bit RGBA,
// so the round trip is byte-exact). Viewers will show slightly dark fringes where alpha < 255; that is expected — the
// files are comparison data first, human-viewable second.
package golden

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
)

// ManifestName is the manifest file name within a golden directory.
const ManifestName = "manifest.json"

// Entry describes one golden image.
type Entry struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"` // of the raw premul RGBA bytes, not the PNG file
	Width  int    `json:"width"`
	Height int    `json:"height"`
}

// Manifest indexes a golden directory.
//
// Schema history:
//   - Schema 1: Platform + Entries only — the Skia-era sets (now frozen archives; see the goldens READMEs). A schema-1
//     manifest reads fine through this struct: the schema-2 fields are simply zero-valued.
//   - Schema 2: adds Lane, GLRenderer/GLVersion, and CapturedAt — written by `oracle bless` for the self-captured
//     per-platform sets. GLRenderer/GLVersion pin the GL stack a GPU-lane set was captured on so a gate can diagnose
//     "the GL stack moved" as a one-line renderer-string mismatch instead of a wall of pixel diffs; they are empty for
//     the raster lane, which has no environment exposure (pure Go).
type Manifest struct {
	Platform   string  `json:"platform"`              // GOOS_GOARCH that generated the goldens
	Lane       string  `json:"lane,omitempty"`        // "raster", "gpu", or "gpudmsaa" (schema 2)
	GLRenderer string  `json:"gl_renderer,omitempty"` // GL_RENDERER of the capturing context; "" for raster (schema 2)
	GLVersion  string  `json:"gl_version,omitempty"`  // GL_VERSION of the capturing context; "" for raster (schema 2)
	CapturedAt string  `json:"captured_at,omitempty"` // UTC date (YYYY-MM-DD) the set was captured (schema 2)
	Entries    []Entry `json:"entries"`               // sorted by name
	Schema     int     `json:"schema"`
}

// HashPixels returns the manifest hash for a pixel buffer.
func HashPixels(pixels []byte) string {
	h := sha256.Sum256(pixels)
	return hex.EncodeToString(h[:])
}

// WritePNG stores raw premul RGBA8888 pixels as a PNG at path.
func WritePNG(path string, pixels []byte, width, height int) error {
	if len(pixels) != width*height*4 {
		return fmt.Errorf("golden: pixel buffer %d bytes, want %d", len(pixels), width*height*4)
	}
	img := &image.NRGBA{Pix: pixels, Stride: width * 4, Rect: image.Rect(0, 0, width, height)}
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

// ReadPNG loads a golden PNG back into raw premul RGBA8888 bytes.
func ReadPNG(path string) (pixels []byte, width, height int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close() //nolint:errcheck // read-only file; close errors are irrelevant
	img, err := png.Decode(f)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("golden: decode %s: %w", path, err)
	}
	var pix []byte
	var stride int
	switch m := img.(type) {
	case *image.NRGBA:
		pix, stride = m.Pix, m.Stride
	case *image.RGBA:
		// png.Encode writes a fully-opaque image as truecolor without an alpha channel, which decodes back as
		// *image.RGBA. Every alpha byte is then 255, so Pix is byte-identical to what WritePNG was given.
		pix, stride = m.Pix, m.Stride
	default:
		return nil, 0, 0, fmt.Errorf("golden: %s decoded to %T, want *image.NRGBA or *image.RGBA (was it written by this tool?)", path, img)
	}
	b := img.Bounds()
	width, height = b.Dx(), b.Dy()
	if stride == width*4 {
		return pix[:width*height*4], width, height, nil
	}
	pixels = make([]byte, width*height*4)
	for y := range height {
		copy(pixels[y*width*4:(y+1)*width*4], pix[y*stride:y*stride+width*4])
	}
	return pixels, width, height, nil
}

// WriteManifest writes the manifest into dir.
func WriteManifest(dir string, m *Manifest) error {
	data, err := json.MarshalIndent(m, "", "\t")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, ManifestName), append(data, '\n'), 0o644)
}

// ReadManifest loads the manifest from dir.
func ReadManifest(dir string) (Manifest, error) {
	var m Manifest
	data, err := os.ReadFile(filepath.Join(dir, ManifestName))
	if err != nil {
		return m, err
	}
	err = json.Unmarshal(data, &m)
	return m, err
}
