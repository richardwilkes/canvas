// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
)

// constSpanShader shades every lane a fixed premultiplied color — enough to drive a ShaderBlitter.
type constSpanShader colorcore.PMColor4f

func (c constSpanShader) ShadeSpan(_, _ int32, dst []colorcore.PMColor4f) {
	for i := range dst {
		dst[i] = colorcore.PMColor4f(c)
	}
}

// TestShaderBlitterPoolReuse locks that a ShaderBlitter drawn from the pool after a recycle produces byte-identical
// output to a fresh one (its retained shade buffer must not carry stale data across the re-init) and that
// RecycleShaderBlitter drops the external device/shader references.
func TestShaderBlitterPoolReuse(t *testing.T) {
	shade := constSpanShader{R: 0.25, G: 0.5, B: 0.75, A: 1}

	// Fresh blitter over a wide device.
	wide := NewPixmap(40, 4)
	fresh := NewShaderBlitter(wide, shade, BlendSrcOver)
	fresh.BlitRect(0, 0, 40, 4)
	RecycleShaderBlitter(fresh)
	if fresh.dev != nil || fresh.shade != nil {
		t.Fatal("RecycleShaderBlitter did not drop device/shader references")
	}

	// A reused blitter (likely the same object, now with an oversized stale buffer) over a narrower device must produce
	// the same pixels a fresh one would.
	narrow := NewPixmap(12, 4)
	reused := NewShaderBlitter(narrow, shade, BlendSrcOver)
	reused.BlitRect(0, 0, 12, 4)

	ref := NewPixmap(12, 4)
	NewShaderBlitter(ref, shade, BlendSrcOver).BlitRect(0, 0, 12, 4)

	for i := range narrow.Pix {
		if narrow.Pix[i] != ref.Pix[i] {
			t.Fatalf("reused shader blitter pixel %d: got %#08x, want %#08x", i, narrow.Pix[i], ref.Pix[i])
		}
	}
}

// constSpanShader32 shades every lane a fixed premultiplied N32 pixel — enough to drive a LegacyShaderBlitter through
// the SpanShader32 interface.
type constSpanShader32 uint32

func (c constSpanShader32) ShadeSpan32(_, _ int32, dst []uint32) {
	for i := range dst {
		dst[i] = uint32(c)
	}
}

// TestLegacyShaderBlitterPoolReuse locks the same reset/reuse invariant for the legacy image-shader blitter, whose
// per-row buffer is resliced to the device width on every borrow.
func TestLegacyShaderBlitterPoolReuse(t *testing.T) {
	shade := constSpanShader32(0xFF203040)

	wide := NewPixmap(40, 3)
	fresh := NewLegacyShaderBlitter(wide, shade, true)
	fresh.BlitRect(0, 0, 40, 3)
	RecycleLegacyShaderBlitter(fresh)
	if fresh.dst != nil || fresh.shade != nil {
		t.Fatal("RecycleLegacyShaderBlitter did not drop device/shader references")
	}

	narrow := NewPixmap(9, 3)
	reused := NewLegacyShaderBlitter(narrow, shade, true)
	if len(reused.buffer) != 9 {
		t.Fatalf("reused legacy blitter buffer len %d, want 9 (device width)", len(reused.buffer))
	}
	reused.BlitH(0, 0, 9)

	ref := NewPixmap(9, 3)
	NewLegacyShaderBlitter(ref, shade, true).BlitH(0, 0, 9)
	for i := 0; i < 9; i++ {
		if narrow.Pix[i] != ref.Pix[i] {
			t.Fatalf("reused legacy blitter pixel %d: got %#08x, want %#08x", i, narrow.Pix[i], ref.Pix[i])
		}
	}
}
