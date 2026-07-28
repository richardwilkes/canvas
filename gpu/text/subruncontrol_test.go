// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests pinning the SubRunControl decision functions: the isSDFT/isDirect size gates (incl. the
// useSDFTForSmallText MinSDFTRange rule), the paint-style and mask-filter gates, the perspective rules, the getSDFFont
// bucket sizes/matrix ranges (with the extra-large bucket on darwin), and SDFTMatrixRange reuse.

package text

import (
	"runtime"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
)

func TestSubRunControlIsSDFTBoundaries(t *testing.T) {
	c := NewSubRunControl(true, false, true, 18, 324, false)
	identity := geom.IdentityMatrix()
	p := canvas.NewPaint()

	// useSDFTForSmallText=false raises the min to kLargeDFFontLimit (162).
	cases := []struct {
		size float32
		want bool
	}{
		{size: 0, want: false},
		{size: 17, want: false},
		{size: 18, want: false}, // below the raised min
		{size: 161.9, want: false},
		{size: 162, want: true}, // fMinDistanceFieldFontSize <= size
		{size: 200, want: true},
		{size: 324, want: true}, // size <= fMaxDistanceFieldFontSize
		{size: 324.1, want: false},
	}
	for _, tc := range cases {
		if got := c.IsSDFT(tc.size, p, &identity); got != tc.want {
			t.Errorf("isSDFT(%v) = %v, want %v", tc.size, got, tc.want)
		}
	}

	// useSDFTForSmallText=true keeps the configured min (18).
	small := NewSubRunControl(true, true, true, 18, 324, false)
	if !small.IsSDFT(18, p, &identity) || small.IsSDFT(17.9, p, &identity) {
		t.Error("small-text SDFT must gate at the configured min (18)")
	}

	// ableToUseSDFT=false disables the lane entirely.
	off := NewSubRunControl(false, false, true, 18, 324, false)
	if off.IsSDFT(200, p, &identity) {
		t.Error("isSDFT must be false when the caps cannot use SDFT")
	}
}

func TestSubRunControlPerOSMaxSizeBoundary(t *testing.T) {
	// The per-OS fGlyphsAsPathsFontSize defaults (mac 256, linux/windows 324) pinned explicitly so every GOOS checks
	// every variant: 300px device text is path-lane (neither SDFT nor direct) under the mac threshold and SDFT under
	// the others.
	identity := geom.IdentityMatrix()
	p := canvas.NewPaint()
	for _, tc := range []struct {
		maxSize  float32
		wantSDFT bool
	}{
		{maxSize: 256, wantSDFT: false},
		{maxSize: 324, wantSDFT: true},
	} {
		c := NewSubRunControl(true, false, true, 18, tc.maxSize, false)
		if got := c.IsSDFT(300, p, &identity); got != tc.wantSDFT {
			t.Errorf("max %v: isSDFT(300) = %v, want %v", tc.maxSize, got, tc.wantSDFT)
		}
		if c.IsDirect(300, p, &identity) {
			t.Errorf("max %v: isDirect(300) must be false (>= kSkSideTooBigForAtlas)", tc.maxSize)
		}
	}
}

func TestSubRunControlIsSDFTPaintGates(t *testing.T) {
	c := NewSubRunControl(true, false, true, 18, 324, false)
	identity := geom.IdentityMatrix()

	// A mask filter rejects.
	p := canvas.NewPaint()
	p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
	if c.IsSDFT(200, p, &identity) {
		t.Error("mask filter must reject SDFT")
	}

	// Hairline stroke rejects; a wide stroke is allowed; stroke-and-fill rejects.
	p = canvas.NewPaint()
	p.Style = canvas.StyleStroke
	p.StrokeWidth = 0
	if c.IsSDFT(200, p, &identity) {
		t.Error("hairline stroke must reject SDFT")
	}
	p.StrokeWidth = 2
	if !c.IsSDFT(200, p, &identity) {
		t.Error("wide stroke must allow SDFT")
	}
	p.Style = canvas.StyleStrokeAndFill
	if c.IsSDFT(200, p, &identity) {
		t.Error("stroke-and-fill must reject SDFT")
	}
}

func TestSubRunControlIsSDFTPerspective(t *testing.T) {
	p := canvas.NewPaint()
	persp := geom.IdentityMatrix()
	persp.Set(geom.MPersp0, 0.001)

	// With perspective SDFT allowed, perspective ignores the min-size gate (any positive size).
	c := NewSubRunControl(true, false, true, 18, 324, false)
	if !c.IsSDFT(30, p, &persp) {
		t.Error("perspective must bypass the min-size gate")
	}
	// The disable-perspective workaround rejects.
	noPersp := NewSubRunControl(true, false, false, 18, 324, false)
	if noPersp.IsSDFT(200, p, &persp) {
		t.Error("perspective SDFT must be rejected when disabled by the caps")
	}
	// isDirect never accepts perspective.
	if c.IsDirect(30, p, &persp) {
		t.Error("isDirect must reject perspective")
	}
}

func TestSubRunControlIsDirect(t *testing.T) {
	c := NewSubRunControl(true, false, true, 18, 324, false)
	identity := geom.IdentityMatrix()
	p := canvas.NewPaint()

	cases := []struct {
		size float32
		want bool
	}{
		{size: 0, want: false},
		{size: 16, want: true},
		{size: 161.9, want: true},  // below the SDFT min: direct
		{size: 162, want: false},   // SDFT window starts
		{size: 255.9, want: false}, // still SDFT
		{size: 256, want: false},   // kSkSideTooBigForAtlas
		{size: 400, want: false},
	}
	for _, tc := range cases {
		if got := c.IsDirect(tc.size, p, &identity); got != tc.want {
			t.Errorf("isDirect(%v) = %v, want %v", tc.size, got, tc.want)
		}
	}

	// With SDFT unavailable, direct extends to the atlas gate.
	off := NewSubRunControl(false, false, true, 18, 324, false)
	if !off.IsDirect(200, p, &identity) || off.IsDirect(256, p, &identity) {
		t.Error("without SDFT, direct must cover (0, 256)")
	}
}

func TestGetSDFFontBucketsAndMatrixRange(t *testing.T) {
	c := NewSubRunControl(true, false, true, 18, 324, false)
	f := loadTestFont(t, "Roboto-Regular.ttf", 20)
	origin := geom.Pt(0, 0)

	check := func(scale, wantDFSize, wantFloor, wantCeil float32) {
		t.Helper()
		var m geom.Matrix
		m.SetScale(scale, scale)
		dfFont, strikeToSourceScale, matrixRange := c.GetSDFFont(f, &m, origin)
		if dfFont.Size() != wantDFSize {
			t.Errorf("scale %v: dfFont size = %v, want %v", scale, dfFont.Size(), wantDFSize)
		}
		if want := f.Size() / wantDFSize; strikeToSourceScale != want {
			t.Errorf("scale %v: strikeToSourceScale = %v, want %v", scale,
				strikeToSourceScale, want)
		}
		wantRange := SDFTMatrixRange{matrixMin: wantFloor / f.Size(), matrixMax: wantCeil / f.Size()}
		if matrixRange != wantRange {
			t.Errorf("scale %v: matrix range = %+v, want %+v", scale, matrixRange, wantRange)
		}
		// The derived font is the SDF strike shape: AA edging, no subpixel.
		if dfFont.Edging() != 1 /* font.EdgingAntiAlias */ || dfFont.Subpixel() {
			t.Errorf("scale %v: dfFont edging/subpixel not normalized", scale)
		}
	}

	// scaledTextSize = 20*scale. Buckets: <=32 small, <=72 medium, then per-OS.
	check(1, 32, 162, 32) // 20px: small bucket; floor = fMinDistanceFieldFontSize (162 here)
	check(3, 72, 32, 72)  // 60px: medium bucket
	if runtime.GOOS == "darwin" {
		check(6, 162, 72, 162)   // 120px: mac large bucket
		check(10, 256, 162, 324) // 200px: mac extra-large bucket, ceil = maxSize
	} else {
		check(6, 162, 72, 324)  // 120px: large bucket, ceil = maxSize
		check(10, 162, 72, 324) // 200px: same bucket
	}
}

func TestSDFTMatrixRangeInRange(t *testing.T) {
	r := SDFTMatrixRange{matrixMin: 1.6, matrixMax: 3.6}
	mk := func(s float32) geom.Matrix {
		var m geom.Matrix
		m.SetScale(s, s)
		return m
	}
	for _, tc := range []struct {
		scale float32
		want  bool
	}{
		{scale: 1.6, want: false}, // exclusive floor
		{scale: 1.61, want: true},
		{scale: 3.6, want: true}, // inclusive ceiling
		{scale: 3.61, want: false},
	} {
		m := mk(tc.scale)
		if got := r.MatrixInRange(&m); got != tc.want {
			t.Errorf("MatrixInRange(scale %v) = %v, want %v", tc.scale, got, tc.want)
		}
	}
}
