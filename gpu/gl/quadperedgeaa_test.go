// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for per-edge-AA quad rendering: VertexSpec sizing/coverage-mode selection rules, color-type minimization,
// index-buffer option selection, and the tessellator's vertex output for the non-AA and coverage-AA lanes (byte-exact
// against hand-computed values).

package gl

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

func TestCalcIndexBufferOption(t *testing.T) {
	if got := CalcIndexBufferOption(gpu.AATypeCoverage, 1); got != IndexBufferOptionPictureFramed {
		t.Fatalf("coverage: got %v", got)
	}
	if got := CalcIndexBufferOption(gpu.AATypeNone, 4); got != IndexBufferOptionIndexedRects {
		t.Fatalf("none/multi: got %v", got)
	}
	if got := CalcIndexBufferOption(gpu.AATypeMSAA, 1); got != IndexBufferOptionTriStrips {
		t.Fatalf("msaa/single: got %v", got)
	}
}

func TestMinQuadColorType(t *testing.T) {
	if got := MinQuadColorType(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}); got !=
		QuadColorTypeNone {
		t.Fatalf("white: got %v", got)
	}
	if got := MinQuadColorType(colorcore.PMColor4f{R: 0.5, G: 0.25, B: 0, A: 1}); got !=
		QuadColorTypeByte {
		t.Fatalf("normalized: got %v", got)
	}
	if got := MinQuadColorType(colorcore.PMColor4f{R: 1.5, G: 0.25, B: 0, A: 1}); got !=
		QuadColorTypeFloat {
		t.Fatalf("out of range: got %v", got)
	}
}

func TestVertexSpecTable(t *testing.T) {
	cases := []struct {
		name         string
		vertexSize   int
		vertsPerQuad int
		spec         VertexSpec
		coverageMode QuadCoverageMode
		primitive    gpu.PrimitiveType
	}{
		{
			name: "non-AA byte color no locals tristrip",
			spec: NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeByte, QuadTypeAxisAligned,
				false, QuadSubsetNo, gpu.AATypeNone, true, IndexBufferOptionTriStrips),
			coverageMode: QuadCoverageModeNone,
			vertexSize:   8 + 4,
			vertsPerQuad: 4,
			primitive:    gpu.PrimitiveTypeTriangleStrip,
		},
		{
			name: "coverage AA byte color compatible-as-alpha",
			spec: NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeByte, QuadTypeAxisAligned,
				false, QuadSubsetNo, gpu.AATypeCoverage, true, IndexBufferOptionPictureFramed),
			coverageMode: QuadCoverageModeWithColor,
			vertexSize:   8 + 4,
			vertsPerQuad: 8,
			primitive:    gpu.PrimitiveTypeTriangles,
		},
		{
			name: "coverage AA byte color incompatible",
			spec: NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeByte, QuadTypeAxisAligned,
				false, QuadSubsetNo, gpu.AATypeCoverage, false, IndexBufferOptionPictureFramed),
			coverageMode: QuadCoverageModeWithPosition,
			vertexSize:   8 + 4 + 4,
			vertsPerQuad: 8,
			primitive:    gpu.PrimitiveTypeTriangles,
		},
		{
			name: "coverage AA general quad forces geometry subset",
			spec: NewVertexSpec(QuadTypeGeneral, QuadColorTypeByte, QuadTypeAxisAligned,
				true, QuadSubsetNo, gpu.AATypeCoverage, true, IndexBufferOptionPictureFramed),
			// The geometry subset acts as a second coverage source, so coverage cannot fold into the color.
			coverageMode: QuadCoverageModeWithPosition,
			vertexSize:   8 + 4 + 16 + 8 + 4,
			vertsPerQuad: 8,
			primitive:    gpu.PrimitiveTypeTriangles,
		},
		{
			name: "perspective device with coverage",
			spec: NewVertexSpec(QuadTypePerspective, QuadColorTypeFloat, QuadTypePerspective,
				true, QuadSubsetNo, gpu.AATypeCoverage, false, IndexBufferOptionPictureFramed),
			coverageMode: QuadCoverageModeWithPosition,
			vertexSize:   16 + 16 + 12 + 16,
			vertsPerQuad: 8,
			primitive:    gpu.PrimitiveTypeTriangles,
		},
		{
			name: "non-AA indexed rects with locals",
			spec: NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeNone, QuadTypeAxisAligned,
				true, QuadSubsetNo, gpu.AATypeNone, true, IndexBufferOptionIndexedRects),
			coverageMode: QuadCoverageModeNone,
			vertexSize:   8 + 8,
			vertsPerQuad: 4,
			primitive:    gpu.PrimitiveTypeTriangles,
		},
	}
	for _, c := range cases {
		if got := c.spec.CoverageMode(); got != c.coverageMode {
			t.Errorf("%s: coverage mode = %v, want %v", c.name, got, c.coverageMode)
		}
		if got := c.spec.VertexSize(); got != c.vertexSize {
			t.Errorf("%s: vertex size = %d, want %d", c.name, got, c.vertexSize)
		}
		if got := c.spec.VerticesPerQuad(); got != c.vertsPerQuad {
			t.Errorf("%s: verts/quad = %d, want %d", c.name, got, c.vertsPerQuad)
		}
		if got := c.spec.PrimitiveType(); got != c.primitive {
			t.Errorf("%s: primitive = %v, want %v", c.name, got, c.primitive)
		}
		// The GP's attribute layout must agree with the vertex size.
		gp := MakeQuadPerEdgeAAProcessor(&c.spec)
		if got := gp.gpBase().VertexStride(); got != c.spec.VertexSize() {
			t.Errorf("%s: GP stride = %d, want %d", c.name, got, c.spec.VertexSize())
		}
	}
}

func f32At(buf []byte, off int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(buf[off:]))
}

func TestQuadTessellatorNonAA(t *testing.T) {
	spec := NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeByte, QuadTypeAxisAligned, false,
		QuadSubsetNo, gpu.AATypeNone, true, IndexBufferOptionTriStrips)
	if spec.VertexSize() != 12 {
		t.Fatalf("vertex size = %d, want 12", spec.VertexSize())
	}
	buf := make([]byte, 4*12)
	tess := NewQuadTessellator(spec, buf)

	identity := geom.IdentityMatrix()
	device := MakeQuadFromRect(geom.Rect{Left: 1, Top: 2, Right: 3, Bottom: 4}, &identity)
	color := colorcore.PMColor4f{R: 1, G: 0.5, B: 0.25, A: 1}
	tess.Append(&device, nil, color, geom.Rect{}, gpu.QuadAAFlagsNone)

	wantPos := [4][2]float32{{1, 2}, {1, 4}, {3, 2}, {3, 4}} // TL, BL, TR, BR
	wantColor := [4]byte{255, 128, 64, 255}                  // trunc(v*255 + 0.5)
	for i := range 4 {
		base := i * 12
		if f32At(buf, base) != wantPos[i][0] || f32At(buf, base+4) != wantPos[i][1] {
			t.Fatalf("vertex %d position = (%v,%v), want %v", i, f32At(buf, base),
				f32At(buf, base+4), wantPos[i])
		}
		got := [4]byte{buf[base+8], buf[base+9], buf[base+10], buf[base+11]}
		if got != wantColor {
			t.Fatalf("vertex %d color = %v, want %v", i, got, wantColor)
		}
	}
}

func TestQuadTessellatorCoverageAA(t *testing.T) {
	// Coverage AA with coverage-as-alpha: 8 vertices, inner ring at the 0.5 inset carrying the full color, outer ring
	// at the 0.5 outset carrying color * 0.
	spec := NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeByte, QuadTypeAxisAligned, false,
		QuadSubsetNo, gpu.AATypeCoverage, true, IndexBufferOptionPictureFramed)
	if spec.CoverageMode() != QuadCoverageModeWithColor {
		t.Fatal("expected coverage-with-color")
	}
	buf := make([]byte, 8*12)
	tess := NewQuadTessellator(spec, buf)

	identity := geom.IdentityMatrix()
	device := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 30, Bottom: 20}, &identity)
	color := colorcore.PMColor4f{R: 0, G: 1, B: 0, A: 1}
	tess.Append(&device, nil, color, geom.Rect{}, gpu.QuadAAFlagsAll)

	inner := [4][2]float32{{10.5, 10.5}, {10.5, 19.5}, {29.5, 10.5}, {29.5, 19.5}}
	outer := [4][2]float32{{9.5, 9.5}, {9.5, 20.5}, {30.5, 9.5}, {30.5, 20.5}}
	near := func(a, b float32) bool { d := a - b; return d < 1e-4 && d > -1e-4 }
	for i := range 4 {
		base := i * 12
		if !near(f32At(buf, base), inner[i][0]) || !near(f32At(buf, base+4), inner[i][1]) {
			t.Fatalf("inner %d = (%v,%v), want %v", i, f32At(buf, base), f32At(buf, base+4),
				inner[i])
		}
		if got := [4]byte{buf[base+8], buf[base+9], buf[base+10], buf[base+11]}; got !=
			[4]byte{0, 255, 0, 255} {
			t.Fatalf("inner %d color = %v, want full green", i, got)
		}
	}
	for i := range 4 {
		base := (4 + i) * 12
		if !near(f32At(buf, base), outer[i][0]) || !near(f32At(buf, base+4), outer[i][1]) {
			t.Fatalf("outer %d = (%v,%v), want %v", i, f32At(buf, base), f32At(buf, base+4),
				outer[i])
		}
		if got := [4]byte{buf[base+8], buf[base+9], buf[base+10], buf[base+11]}; got !=
			[4]byte{0, 0, 0, 0} {
			t.Fatalf("outer %d color = %v, want transparent", i, got)
		}
	}

	// A non-AA quad batched into a coverage-AA op writes the same corners twice so the AA ramp has zero area.
	buf2 := make([]byte, 8*12)
	tess2 := NewQuadTessellator(spec, buf2)
	device2 := MakeQuadFromRect(geom.Rect{Left: 5, Top: 5, Right: 7, Bottom: 7}, &identity)
	tess2.Append(&device2, nil, color, geom.Rect{}, gpu.QuadAAFlagsNone)
	for i := range 4 {
		if f32At(buf2, i*12) != f32At(buf2, (4+i)*12) ||
			f32At(buf2, i*12+4) != f32At(buf2, (4+i)*12+4) {
			t.Fatalf("non-AA batched quad: inner/outer vertex %d differ", i)
		}
	}
}

func TestQuadTessellatorHairline(t *testing.T) {
	// A subpixel-wide AA quad: the inner ring collapses with reduced coverage folded into the color, and the outer ring
	// is stretched so its narrow axis reaches 2px.
	spec := NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeByte, QuadTypeAxisAligned, false,
		QuadSubsetNo, gpu.AATypeCoverage, true, IndexBufferOptionPictureFramed)
	buf := make([]byte, 8*12)
	tess := NewQuadTessellator(spec, buf)

	identity := geom.IdentityMatrix()
	device := MakeQuadFromRect(geom.Rect{Left: 10, Top: 10, Right: 10.4, Bottom: 20}, &identity)
	color := colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}
	tess.Append(&device, nil, color, geom.Rect{}, gpu.QuadAAFlagsAll)

	// Inner ring: collapsed to x = 10.2 with coverage ~0.4 modulated into the color bytes.
	for i := range 4 {
		base := i * 12
		if d := f32At(buf, base) - 10.2; d > 1e-3 || d < -1e-3 {
			t.Fatalf("inner x[%d] = %v, want ~10.2", i, f32At(buf, base))
		}
		a := buf[base+11]
		if a < 95 || a > 110 { // ~0.4 * 255
			t.Fatalf("inner alpha[%d] = %d, want ~102", i, a)
		}
	}
	// Outer ring: the narrow axis is widened to 2px (10.2 ± 1).
	for i := range 4 {
		base := (4 + i) * 12
		x := f32At(buf, base)
		if d := float32(math.Abs(float64(x - 10.2))); d < 0.95 || d > 1.05 {
			t.Fatalf("outer x[%d] = %v, want 10.2 +/- 1", i, x)
		}
	}
}
