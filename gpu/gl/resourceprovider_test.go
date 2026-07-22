// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// newTightenCapsResourceProvider builds a caps-only ResourceProvider whose Caps report no WritePixelsRowBytesSupport,
// forcing prepareLevels down its tight-copy lane.
func newTightenCapsResourceProvider(t *testing.T) *ResourceProvider {
	t.Helper()
	ctxInfo := appleM4MaxDriver().newContextInfo(t, nil)
	ctxInfo.Caps.WritePixelsRowBytesSupport = false
	return &ResourceProvider{gpu: &Gpu{ctx: ctxInfo}}
}

// TestPrepareLevelsShortPixelsRejected verifies that a MipLevel whose Pixels buffer is shorter than its declared
// RowBytes stride requires is treated as an unsupported write instead of panicking with an out-of-range slice in the
// tight-copy loop.
func TestPrepareLevelsShortPixelsRejected(t *testing.T) {
	rp := newTightenCapsResourceProvider(t)
	const (
		colorType = gpu.ColorTypeBGRA8888
		format    = FormatRGBA8
	)
	baseSize := geom.ISize{Width: 4, Height: 4}
	minRB := int(baseSize.Width) * colorType.BytesPerPixel() // 16
	actualRB := minRB + 16                                   // padded stride of 32
	// The last row starts at actualRB*(height-1) and spans minRB bytes, so a fully valid buffer needs
	// actualRB*(height-1)+minRB bytes. Provide one byte short of that.
	required := actualRB*(int(baseSize.Height)-1) + minRB
	texels := []gpu.MipLevel{{Pixels: make([]byte, required-1), RowBytes: actualRB}}

	// Must not panic; must report the write as unsupported.
	_, out, ok := rp.prepareLevels(format, colorType, baseSize, texels, 1)
	if ok {
		t.Fatalf("prepareLevels accepted a too-short Pixels buffer (len=%d, required=%d)", required-1, required)
	}
	if out != nil {
		t.Errorf("prepareLevels returned levels on failure: %v", out)
	}
}

// TestPrepareLevelsTightCopy verifies the tight-copy lane still produces correctly tightened level data when the Pixels
// buffer fully covers its declared stride.
func TestPrepareLevelsTightCopy(t *testing.T) {
	rp := newTightenCapsResourceProvider(t)
	const (
		colorType = gpu.ColorTypeBGRA8888
		format    = FormatRGBA8
	)
	baseSize := geom.ISize{Width: 4, Height: 4}
	minRB := int(baseSize.Width) * colorType.BytesPerPixel() // 16
	actualRB := minRB + 16                                   // padded stride of 32
	required := actualRB*(int(baseSize.Height)-1) + minRB
	src := make([]byte, required)
	// Fill each row's live region with a recognizable per-row byte; leave the padding zeroed.
	for y := 0; y < int(baseSize.Height); y++ {
		for x := 0; x < minRB; x++ {
			src[y*actualRB+x] = byte(y*minRB + x + 1)
		}
	}
	texels := []gpu.MipLevel{{Pixels: src, RowBytes: actualRB}}

	_, out, ok := rp.prepareLevels(format, colorType, baseSize, texels, 1)
	if !ok {
		t.Fatal("prepareLevels rejected a valid padded buffer")
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 level, got %d", len(out))
	}
	if out[0].RowBytes != minRB {
		t.Errorf("tightened RowBytes = %d, want %d", out[0].RowBytes, minRB)
	}
	if len(out[0].Pixels) != minRB*int(baseSize.Height) {
		t.Fatalf("tightened Pixels len = %d, want %d", len(out[0].Pixels), minRB*int(baseSize.Height))
	}
	for y := 0; y < int(baseSize.Height); y++ {
		for x := 0; x < minRB; x++ {
			want := byte(y*minRB + x + 1)
			if got := out[0].Pixels[y*minRB+x]; got != want {
				t.Fatalf("tight[%d][%d] = %d, want %d", y, x, got, want)
			}
		}
	}
}
