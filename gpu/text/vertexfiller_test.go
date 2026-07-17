// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package text

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

func TestCanUseDirect(t *testing.T) {
	identity := geom.IdentityMatrix()

	// Integer translation of the same 2x2 is direct-compatible.
	var intTranslate geom.Matrix
	intTranslate.SetTranslate(3, -7)
	if ok, tr := canUseDirect(&identity, &intTranslate); !ok || tr != geom.Pt(3, -7) {
		t.Fatalf("integer translate: ok=%v tr=%v", ok, tr)
	}

	// Fractional translation is not.
	var fracTranslate geom.Matrix
	fracTranslate.SetTranslate(0.5, 0)
	if ok, _ := canUseDirect(&identity, &fracTranslate); ok {
		t.Fatal("fractional translate must not be direct-compatible")
	}

	// A 2x2 change is not.
	var scaled geom.Matrix
	scaled.SetScale(2, 2)
	if ok, _ := canUseDirect(&identity, &scaled); ok {
		t.Fatal("scale change must not be direct-compatible")
	}

	// Same scale with integer translation difference is.
	var a, b geom.Matrix
	a.SetScaleTranslate(2, 2, 1, 2)
	b.SetScaleTranslate(2, 2, 5, -4)
	if ok, tr := canUseDirect(&a, &b); !ok || tr != geom.Pt(4, -6) {
		t.Fatalf("scaled integer translate: ok=%v tr=%v", ok, tr)
	}

	// Perspective is never direct.
	p := identity
	p.Set(geom.MPersp0, 0.001)
	if ok, _ := canUseDirect(&identity, &p); ok {
		t.Fatal("perspective must not be direct-compatible")
	}
}

func TestVertexFillerStrides(t *testing.T) {
	identity := geom.IdentityMatrix()
	persp := identity
	persp.Set(geom.MPersp0, 0.001)

	a8 := MakeVertexFiller(gpu.MaskFormatA8, &identity, geom.Rect{}, nil, IsDirect)
	argb := MakeVertexFiller(gpu.MaskFormatARGB, &identity, geom.Rect{}, nil, IsDirect)

	if got := a8.VertexStride(&identity); got != 16 {
		t.Errorf("A8 2D stride = %d, want 16", got)
	}
	if got := a8.VertexStride(&persp); got != 20 {
		t.Errorf("A8 3D stride = %d, want 20", got)
	}
	if got := argb.VertexStride(&identity); got != 12 {
		t.Errorf("ARGB 2D stride = %d, want 12", got)
	}
	if got := argb.VertexStride(&persp); got != 16 {
		t.Errorf("ARGB 3D stride = %d, want 16", got)
	}
}

// testGlyphWithUVs builds a Glyph whose atlas locator covers the given texel rect.
func testGlyphWithUVs(l, t, r, b int16) *Glyph {
	g := NewGlyph(font.PackGlyphID(1))
	g.AtlasLocator.UpdateRect(gpu.IRect16MakeXYWH(l, t, r-l, b-t))
	return g
}

func TestDeviceRectAndCheckTransform(t *testing.T) {
	identity := geom.IdentityMatrix()
	bounds := geom.RectLTRB(10, 20, 30, 40)
	filler := MakeVertexFiller(gpu.MaskFormatA8, &identity, bounds,
		[]geom.Point{geom.Pt(10, 20)}, IsDirect)

	var translated geom.Matrix
	translated.SetTranslate(5, 6)
	direct, rect := filler.DeviceRectAndCheckTransform(&translated)
	if !direct || rect != geom.RectLTRB(15, 26, 35, 46) {
		t.Fatalf("integer translate: direct=%v rect=%v", direct, rect)
	}

	var scaled geom.Matrix
	scaled.SetScale(2, 2)
	direct, rect = filler.DeviceRectAndCheckTransform(&scaled)
	if direct || rect != geom.RectLTRB(20, 40, 60, 80) {
		t.Fatalf("scale: direct=%v rect=%v", direct, rect)
	}
}

func readF32(buf []byte, off int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(buf[off:]))
}

func readU16(buf []byte, off int) uint16 { return binary.LittleEndian.Uint16(buf[off:]) }

func TestFillVertexDataDirectA8(t *testing.T) {
	identity := geom.IdentityMatrix()
	glyph := testGlyphWithUVs(100, 200, 110, 216) // 10x16 texels
	filler := MakeVertexFiller(gpu.MaskFormatA8, &identity, geom.RectLTRB(5, 7, 15, 23),
		[]geom.Point{geom.Pt(5, 7)}, IsDirect)

	var positionMatrix geom.Matrix
	positionMatrix.SetTranslate(2, 3)

	buf := make([]byte, 4*16)
	white := colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}
	filler.FillVertexData(buf, 0, 1, []*Glyph{glyph}, white, &positionMatrix, geom.IRect{})

	// Vertex order: LT, LB, RT, RB. Layout: {x, y f32; packed color u32; u, v u16}.
	wantPos := [4][2]float32{{7, 10}, {7, 26}, {17, 10}, {17, 26}}
	wantUV := [4][2]uint16{{100, 200}, {100, 216}, {110, 200}, {110, 216}}
	for v := range 4 {
		off := v * 16
		if x, y := readF32(buf, off), readF32(buf, off+4); x != wantPos[v][0] || y != wantPos[v][1] {
			t.Errorf("vertex %d pos = (%g,%g), want %v", v, x, y, wantPos[v])
		}
		if c := binary.LittleEndian.Uint32(buf[off+8:]); c != 0xFFFFFFFF {
			t.Errorf("vertex %d color = %08x", v, c)
		}
		if u, vv := readU16(buf, off+12), readU16(buf, off+14); u != wantUV[v][0] || vv != wantUV[v][1] {
			t.Errorf("vertex %d uv = (%d,%d), want %v", v, u, vv, wantUV[v])
		}
	}
}

func TestFillVertexDataDirectClipped(t *testing.T) {
	identity := geom.IdentityMatrix()
	glyph := testGlyphWithUVs(0, 0, 10, 10)
	filler := MakeVertexFiller(gpu.MaskFormatA8, &identity, geom.RectLTRB(0, 0, 10, 10),
		[]geom.Point{geom.Pt(0, 0)}, IsDirect)

	buf := make([]byte, 4*16)
	white := colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}
	clip := geom.IRectLTRB(2, 3, 8, 9)
	filler.FillVertexData(buf, 0, 1, []*Glyph{glyph}, white, &identity, clip)

	// The device rect and UVs are both clipped by the same amounts.
	if x, y := readF32(buf, 0), readF32(buf, 4); x != 2 || y != 3 {
		t.Errorf("clipped LT = (%g,%g), want (2,3)", x, y)
	}
	if x, y := readF32(buf, 3*16), readF32(buf, 3*16+4); x != 8 || y != 9 {
		t.Errorf("clipped RB = (%g,%g), want (8,9)", x, y)
	}
	if u, v := readU16(buf, 12), readU16(buf, 14); u != 2 || v != 3 {
		t.Errorf("clipped LT uv = (%d,%d), want (2,3)", u, v)
	}
	if u, v := readU16(buf, 3*16+12), readU16(buf, 3*16+14); u != 8 || v != 9 {
		t.Errorf("clipped RB uv = (%d,%d), want (8,9)", u, v)
	}
}

func TestFillVertexDataTransformed(t *testing.T) {
	identity := geom.IdentityMatrix()
	glyph := testGlyphWithUVs(4, 6, 14, 16) // 10x10 texels
	filler := MakeVertexFiller(gpu.MaskFormatARGB, &identity, geom.RectLTRB(0, 0, 10, 10),
		[]geom.Point{geom.Pt(0, 0)}, IsTransformed)

	var scale geom.Matrix
	scale.SetScale(2, 2)
	buf := make([]byte, 4*12)
	white := colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}
	filler.FillVertexData(buf, 0, 1, []*Glyph{glyph}, white, &scale, geom.IRect{})

	// ARGB 2D layout: {x, y f32; u, v u16}; corners mapped through the view difference.
	wantPos := [4][2]float32{{0, 0}, {0, 20}, {20, 0}, {20, 20}}
	wantUV := [4][2]uint16{{4, 6}, {4, 16}, {14, 6}, {14, 16}}
	for v := range 4 {
		off := v * 12
		if x, y := readF32(buf, off), readF32(buf, off+4); x != wantPos[v][0] || y != wantPos[v][1] {
			t.Errorf("vertex %d pos = (%g,%g), want %v", v, x, y, wantPos[v])
		}
		if u, vv := readU16(buf, off+8), readU16(buf, off+10); u != wantUV[v][0] || vv != wantUV[v][1] {
			t.Errorf("vertex %d uv = (%d,%d), want %v", v, u, vv, wantUV[v])
		}
	}
}

func TestGrColorPacking(t *testing.T) {
	// toBytes_RGBA packs r in the low byte with x*255+0.5 truncation (Sk4f_toL32).
	c := packColor(colorcore.PMColor4f{R: 0.3, G: 0, B: 1, A: 0.5})
	// 0.3*255+0.5 = 77.0 in float (0.3f*255 is exactly 76.5); trunc = 77.
	if r := c & 0xFF; r != 77 {
		t.Errorf("r = %d, want 77", r)
	}
	if g := (c >> 8) & 0xFF; g != 0 {
		t.Errorf("g = %d, want 0", g)
	}
	if b := (c >> 16) & 0xFF; b != 255 {
		t.Errorf("b = %d, want 255", b)
	}
	if a := (c >> 24) & 0xFF; a != 128 {
		t.Errorf("a = %d, want 128", a)
	}
}
