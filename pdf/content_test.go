// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stream"
)

// capture runs f against a fresh memory stream and returns the bytes it wrote.
func capture(f func(s stream.WStream)) string {
	s := stream.NewMemoryWStream()
	f(s)
	return string(s.Bytes())
}

func TestAppendRectangle(t *testing.T) {
	got := capture(func(s stream.WStream) {
		appendRectangle(geom.RectLTRB(10, 20, 110, 70), s)
	})
	if got != "10 20 100 50 re\n" {
		t.Errorf("appendRectangle = %q", got)
	}
	// PDF's origin is bottom-left, so the smaller of top/bottom is emitted as the y; the width/height are the raw
	// signed differences (appendRectangle assumes callers pass sorted rects).
	got = capture(func(s stream.WStream) {
		appendRectangle(geom.Rect{Left: 0, Top: 70, Right: 40, Bottom: 20}, s)
	})
	if got != "0 20 40 -50 re\n" {
		t.Errorf("appendRectangle (unsorted) = %q", got)
	}
}

func TestEmitPathRectFastPath(t *testing.T) {
	p := (&path.Path{}).AddRect(geom.RectLTRB(5, 6, 25, 36), geom.DirectionCW)
	got := capture(func(s stream.WStream) {
		emitPath(p, styleFill, true, s, 0.25)
	})
	if got != "5 6 20 30 re\n" {
		t.Errorf("emitPath rect fast path = %q", got)
	}
}

func TestEmitPathLines(t *testing.T) {
	p := (&path.Path{}).MoveTo(1, 2).LineTo(3, 4).LineTo(5, 6)
	got := capture(func(s stream.WStream) {
		emitPath(p, styleStroke, false, s, 0.25)
	})
	want := "1 2 m\n3 4 l\n5 6 l\n"
	if got != want {
		t.Errorf("emitPath lines = %q, want %q", got, want)
	}
}

func TestPaintPath(t *testing.T) {
	cases := []struct {
		want  string
		style paintStyle
		fill  path.FillType
	}{
		{style: styleFill, fill: path.FillWinding, want: "f\n"},
		{style: styleFill, fill: path.FillEvenOdd, want: "f*\n"},
		{style: styleStroke, fill: path.FillWinding, want: "S\n"},
		{style: styleStrokeAndFill, fill: path.FillWinding, want: "B\n"},
		{style: styleStrokeAndFill, fill: path.FillEvenOdd, want: "B*\n"},
	}
	for _, c := range cases {
		got := capture(func(s stream.WStream) { paintPath(c.style, c.fill, s) })
		if got != c.want {
			t.Errorf("paintPath(%d,%d) = %q, want %q", c.style, c.fill, got, c.want)
		}
	}
}

func TestAppendTransform(t *testing.T) {
	m := geom.IdentityMatrix()
	m.SetScaleTranslate(2, -2, 10, 20)
	got := capture(func(s stream.WStream) { appendTransform(&m, s) })
	if got != "2 0 0 -2 10 20 cm\n" {
		t.Errorf("appendTransform scale-translate = %q", got)
	}
}

func TestConvertQuadToCubic(t *testing.T) {
	// A degenerate quad where all points coincide stays coincident.
	q := [3]geom.Point{{X: 5, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}
	c := convertQuadToCubic(q)
	for _, p := range c {
		if p != (geom.Point{X: 5, Y: 5}) {
			t.Fatalf("degenerate quad→cubic point %v", p)
		}
	}
	// Control point at 1/3 and 2/3 for a straight quad.
	q = [3]geom.Point{{X: 0, Y: 0}, {X: 3, Y: 0}, {X: 6, Y: 0}}
	c = convertQuadToCubic(q)
	if c[0] != (geom.Point{X: 0, Y: 0}) || c[3] != (geom.Point{X: 6, Y: 0}) {
		t.Errorf("endpoints not preserved: %v", c)
	}
	if c[1].X != 2 || c[2].X != 4 {
		t.Errorf("cubic controls = %v, want x=2 and x=4", c)
	}
}

func TestBlendModeName(t *testing.T) {
	cases := map[raster.BlendMode]string{
		raster.BlendSrcOver:    "Normal",
		raster.BlendXor:        "Normal",
		raster.BlendPlus:       "Normal",
		raster.BlendMultiply:   "Multiply",
		raster.BlendScreen:     "Screen",
		raster.BlendDarken:     "Darken",
		raster.BlendLuminosity: "Luminosity",
		raster.BlendClear:      "", // unsupported as a blend mode
		raster.BlendSrc:        "",
	}
	for mode, want := range cases {
		if got := blendModeName(mode); got != want {
			t.Errorf("blendModeName(%d) = %q, want %q", mode, got, want)
		}
	}
}

func TestPDFBlendModeReduction(t *testing.T) {
	// Modes PDF cannot express collapse to SrcOver; expressible ones pass through.
	if pdfBlendMode(raster.BlendClear) != uint8(raster.BlendSrcOver) {
		t.Error("Clear should reduce to SrcOver for the ExtGState")
	}
	if pdfBlendMode(raster.BlendMultiply) != uint8(raster.BlendMultiply) {
		t.Error("Multiply should pass through")
	}
}

func TestMatrixAsAffine(t *testing.T) {
	m := geom.IdentityMatrix()
	m.SetAll(2, 0.5, 10, 0.25, 3, 20, 0, 0, 1)
	a, ok := m.AsAffine()
	if !ok {
		t.Fatal("a non-perspective matrix should be affine")
	}
	// order is [scaleX, skewY, skewX, scaleY, transX, transY]
	want := [6]float32{2, 0.25, 0.5, 3, 10, 20}
	if a != want {
		t.Errorf("AsAffine = %v, want %v", a, want)
	}
	m.SetAll(1, 0, 0, 0, 1, 0, 0.001, 0, 1)
	if _, ok = m.AsAffine(); ok {
		t.Error("a perspective matrix must not be affine")
	}
}
