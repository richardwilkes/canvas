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
	"bytes"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// renderGradient draws a full-page rect filled with the given shader and returns the PDF bytes.
func renderGradient(t *testing.T, g shaders.Shader) []byte {
	t.Helper()
	p := canvas.NewPaint()
	p.Shader = g
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 100, 100), p)
	})
	validatePDF(t, data)
	return data
}

// allObjectDicts returns the dictionary portion of every indirect object, including the dict header of stream objects
// (the bytes before " stream\n"), so patterns/shadings/functions that carry a stream are covered too.
func allObjectDicts(data []byte) []string {
	var out []string
	rest := data
	for {
		oi := bytes.Index(rest, []byte(" obj\n"))
		if oi < 0 {
			break
		}
		start := oi + len(" obj\n")
		ei := bytes.Index(rest[start:], []byte("\nendobj"))
		if ei < 0 {
			break
		}
		body := rest[start : start+ei]
		if si := bytes.Index(body, []byte(" stream\n")); si >= 0 {
			body = body[:si]
		}
		out = append(out, string(body))
		rest = rest[start+ei+len("\nendobj"):]
	}
	return out
}

// anyObjectContains reports whether any indirect object's dictionary contains every one of the given substrings.
func anyObjectContains(data []byte, subs ...string) bool {
	for _, s := range allObjectDicts(data) {
		all := true
		for _, sub := range subs {
			if !strings.Contains(s, sub) {
				all = false
				break
			}
		}
		if all {
			return true
		}
	}
	return false
}

// anyStreamContains reports whether any (inflated) stream contains the substring.
func anyStreamContains(data []byte, sub string) bool {
	for _, s := range allStreamContents(data) {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

var twoStops = []colorcore.Color{colorcore.ARGB(255, 255, 0, 0), colorcore.ARGB(255, 0, 0, 255)}

func TestGradientLinearClampAxialShading(t *testing.T) {
	g := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 100}, twoStops, nil, shaders.TileClamp, nil)
	data := renderGradient(t, g)
	// The pattern is applied in the content stream and listed in /Resources.
	mustContain(t, pageContent(t, data), "/Pattern CS/Pattern cs")
	mustContain(t, pageContent(t, data), " SCN")
	mustContain(t, pageContent(t, data), " scn\n")
	if !anyObjectContains(data, "/Pattern <<") {
		t.Error("page /Resources missing a /Pattern subdictionary")
	}
	// A clamp linear gradient emits a native axial shading (type 2) with Coords and Extend.
	if !anyObjectContains(data, "/PatternType 2", "/ShadingType 2", "/Coords [0 0 100 0]",
		"/Extend [true true]", "/ColorSpace /DeviceRGB") {
		t.Errorf("no axial shading pattern; objects=%v", dictObjects(data))
	}
	// Two opaque stops collapse to a single Type 2 interpolation function inline in the shading.
	if !anyObjectContains(data, "/FunctionType 2", "/C0 [1 0 0]", "/C1 [0 0 1]") {
		t.Errorf("no Type 2 interpolation function; objects=%v", dictObjects(data))
	}
}

func TestGradientRadialClampRadialShading(t *testing.T) {
	g := shaders.NewRadialGradient(geom.Point{X: 50, Y: 50}, 40, twoStops, nil, shaders.TileClamp, nil)
	data := renderGradient(t, g)
	if !anyObjectContains(data, "/PatternType 2", "/ShadingType 3", "/Coords [50 50 0 50 50 40]") {
		t.Errorf("no radial shading pattern; objects=%v", dictObjects(data))
	}
}

func TestGradientConicalClampRadialShading(t *testing.T) {
	// Two distinct centers and radii, clamp → a native radial shading with both circles.
	g := shaders.NewTwoPointConicalGradient(geom.Point{X: 20, Y: 20}, 5, geom.Point{X: 60, Y: 60}, 30,
		twoStops, nil, shaders.TileClamp, nil)
	data := renderGradient(t, g)
	if !anyObjectContains(data, "/PatternType 2", "/ShadingType 3", "/Coords [20 20 5 60 60 30]") {
		t.Errorf("no conical radial shading; objects=%v", dictObjects(data))
	}
}

func TestGradientLinearRepeatFunctionShading(t *testing.T) {
	g := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 50}, twoStops, nil, shaders.TileRepeat, nil)
	data := renderGradient(t, g)
	// Repeat has no PDF-native equivalent, so it emits a Function-based (type 1) shading whose Type 4 PostScript
	// function references the tile-mode "truncate" code. No Extend for a non-clamp mode.
	if !anyObjectContains(data, "/PatternType 2", "/ShadingType 1", "/Function ") {
		t.Errorf("no function-based shading; objects=%v", dictObjects(data))
	}
	if anyObjectContains(data, "/Extend") {
		t.Error("a repeat gradient must not emit /Extend")
	}
	if !anyStreamContains(data, "dup truncate sub") {
		t.Error("repeat tile-mode PostScript ('dup truncate sub') not emitted")
	}
	if !anyObjectContains(data, "/FunctionType 4") {
		t.Errorf("no Type 4 PostScript function object; objects=%v", dictObjects(data))
	}
}

func TestGradientSweepFunctionShading(t *testing.T) {
	g := shaders.NewSweepGradient(geom.Point{X: 50, Y: 50}, twoStops, nil, shaders.TileClamp, 0, 360, nil)
	data := renderGradient(t, g)
	// Sweep is always a Function-based shading (no native PDF sweep); the PS function uses atan.
	if !anyObjectContains(data, "/PatternType 2", "/ShadingType 1") {
		t.Errorf("no function-based shading; objects=%v", dictObjects(data))
	}
	if !anyStreamContains(data, "atan 360 div") {
		t.Error("sweep PostScript ('atan 360 div') not emitted")
	}
}

func TestGradientMirrorFunctionShading(t *testing.T) {
	g := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 50}, twoStops, nil, shaders.TileMirror, nil)
	data := renderGradient(t, g)
	if !anyObjectContains(data, "/ShadingType 1") {
		t.Errorf("mirror gradient should be a function shading; objects=%v", dictObjects(data))
	}
	// The mirror tile-mode work-around uses "2 mod" and "cvi".
	if !anyStreamContains(data, "2 mod") || !anyStreamContains(data, "cvi") {
		t.Error("mirror tile-mode PostScript not emitted")
	}
}

func TestGradientThreeStopStitchFunction(t *testing.T) {
	colors := []colorcore.Color{
		colorcore.ARGB(255, 255, 0, 0),
		colorcore.ARGB(255, 0, 255, 0),
		colorcore.ARGB(255, 0, 0, 255),
	}
	g := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, shaders.TileClamp, nil)
	data := renderGradient(t, g)
	// Three stops → a Type 3 stitching function over two Type 2 interpolations with a Bounds at 0.5.
	if !anyObjectContains(data, "/FunctionType 3", "/Bounds [.5]", "/Functions [") {
		t.Errorf("no Type 3 stitch function; objects=%v", dictObjects(data))
	}
}

func TestGradientAlphaLuminositySMask(t *testing.T) {
	colors := []colorcore.Color{colorcore.ARGB(128, 255, 0, 0), colorcore.ARGB(255, 0, 0, 255)}
	g := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 100}, colors, nil, shaders.TileClamp, nil)
	data := renderGradient(t, g)
	// An alpha gradient wraps the color shading in a tiling pattern (PatternType 1) painted under a luminosity soft
	// mask that carries the alpha as a grayscale shading.
	if !anyObjectContains(data, "/PatternType 1", "/TilingType 1") {
		t.Errorf("no tiling pattern for the alpha gradient; objects=%v", dictObjects(data))
	}
	if !anyObjectContains(data, "/SMask <<", "/S /Luminosity") {
		t.Errorf("no luminosity SMask ExtGState; objects=%v", dictObjects(data))
	}
	// The color shading is opaque (alpha stripped) and the luminosity shading is grayscale from the alpha.
	if !anyObjectContains(data, "/C0 [1 0 0]", "/C1 [0 0 1]") {
		t.Errorf("no opaque color shading; objects=%v", dictObjects(data))
	}
	if !anyObjectContains(data, "/C0 [.502 .502 .502]", "/C1 [1 1 1]") {
		t.Errorf("no grayscale luminosity shading; objects=%v", dictObjects(data))
	}
	// A transparency-group form XObject carries the mask.
	if !anyObjectContains(data, "/Subtype /Form", "/Group", "/S /Transparency") {
		t.Errorf("no transparency-group form XObject; objects=%v", dictObjects(data))
	}
}

func TestGradientPatternDedup(t *testing.T) {
	// The same gradient drawn twice must produce exactly one pattern object.
	p := canvas.NewPaint()
	p.Shader = shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 100}, twoStops, nil, shaders.TileClamp, nil)
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 40, 40), p)
		c.DrawRect(geom.RectLTRB(50, 50, 90, 90), p)
	})
	validatePDF(t, data)
	n := 0
	for _, s := range dictObjects(data) {
		if strings.Contains(s, "/PatternType 2") {
			n++
		}
	}
	if n != 1 {
		t.Errorf("expected 1 shared pattern object, got %d\nobjects=%v", n, dictObjects(data))
	}
	// The pattern is selected once and both fills reuse it (the graphic-state machine avoids re-emitting an unchanged
	// pattern), and it appears once in /Resources.
	if c := strings.Count(pageContent(t, data), "/Pattern cs"); c != 1 {
		t.Errorf("expected the shared pattern to be selected once, got %d\n%s", c, pageContent(t, data))
	}
	if c := strings.Count(pageContent(t, data), "f\n"); c != 2 {
		t.Errorf("expected two fills reusing the pattern, got %d\n%s", c, pageContent(t, data))
	}
}

func TestGradientLocalMatrixInPatternMatrix(t *testing.T) {
	// A local matrix on the shader folds into the pattern's placement matrix (the shaderTransform).
	lm := geom.TranslateMatrix(10, 20)
	g := shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 100}, twoStops, nil, shaders.TileClamp, &lm)
	data := renderGradient(t, g)
	// finalMatrix = canvasTransform(=initialTransform [1 0 0 -1 0 100]) preConcat translate(10,20): the y-flip turns
	// the +20 y-translate into +80 (100 - 20) and keeps the +10 x-translate.
	if !anyObjectContains(data, "/Matrix [1 0 0 -1 10 80]") {
		t.Errorf("local matrix not folded into the pattern matrix; objects=%v", dictObjects(data))
	}
}

func TestColorShaderStillFoldsToColor(t *testing.T) {
	// A pure color shader is folded to an rg/RG color, not a pattern (regression for the shader branch).
	p := canvas.NewPaint()
	p.Shader = shaders.NewColor(colorcore.ARGB(255, 40, 60, 80))
	data := renderPDF(t, 100, 100, func(c *canvas.Canvas) {
		c.DrawRect(geom.RectLTRB(0, 0, 100, 100), p)
	})
	validatePDF(t, data)
	if strings.Contains(pageContent(t, data), "/Pattern cs") {
		t.Error("a color shader must not emit a pattern")
	}
	// 40/255, 60/255, 80/255 as the fill/stroke color.
	mustContain(t, pageContent(t, data), " rg\n")
}

// ---- unit tests for the standalone helpers ----------------------------------------------------------

func TestUnitToPointsMatrix(t *testing.T) {
	// Map the unit segment (0,0)-(1,0) onto (10,10)-(10,20): a 90-degree rotation, scale 10, translate.
	m := unitToPointsMatrix([2]geom.Point{{X: 10, Y: 10}, {X: 10, Y: 20}})
	got := m.MapPoint(geom.Point{X: 0, Y: 0})
	if !nearlyPt(got, geom.Point{X: 10, Y: 10}) {
		t.Errorf("unit origin mapped to %v, want (10,10)", got)
	}
	got = m.MapPoint(geom.Point{X: 1, Y: 0})
	if !nearlyPt(got, geom.Point{X: 10, Y: 20}) {
		t.Errorf("unit end mapped to %v, want (10,20)", got)
	}
}

func TestFixUpRadius(t *testing.T) {
	// Internally tangent circles (distance == |r1 - r2|) get the larger radius nudged out.
	r1, r2 := fixUpRadius(geom.Point{}, 10, geom.Point{X: 5, Y: 0}, 5)
	if r1 != 10.002 {
		t.Errorf("fixUpRadius r1 = %v, want 10.002", r1)
	}
	if r2 != 5 {
		t.Errorf("fixUpRadius r2 = %v, want 5 (unchanged)", r2)
	}
	// Well-separated circles are left alone.
	r1, r2 = fixUpRadius(geom.Point{}, 10, geom.Point{X: 100, Y: 0}, 5)
	if r1 != 10 || r2 != 5 {
		t.Errorf("fixUpRadius nudged non-tangent circles: r1=%v r2=%v", r1, r2)
	}
}

func TestSplitPerspective(t *testing.T) {
	var in geom.Matrix
	in.SetAll(2, 0, 3, 0, 2, 4, 0.01, 0.02, 1)
	affine, perspInv, ok := splitPerspective(in)
	if !ok {
		t.Fatal("splitPerspective failed on a valid perspective matrix")
	}
	if affine.HasPerspective() {
		t.Error("the affine part must have no perspective")
	}
	// The inverse-perspective part is 1/p2 at persp2, with -p0/p2 and -p1/p2 in the persp row.
	if !geom.ScalarNearlyEqual(perspInv.Get(geom.MPersp0), -0.01) ||
		!geom.ScalarNearlyEqual(perspInv.Get(geom.MPersp1), -0.02) {
		t.Errorf("inverse perspective row wrong: %v %v", perspInv.Get(geom.MPersp0), perspInv.Get(geom.MPersp1))
	}
}

func nearlyPt(a, b geom.Point) bool {
	return geom.ScalarNearlyEqual(a.X, b.X) && geom.ScalarNearlyEqual(a.Y, b.Y)
}
