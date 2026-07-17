// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for distance-field text: the device text lowering routes large text through the SDFT subrun onto
// AtlasTextOp's distance-field mask types and GPs, the generated GLSL carries the DF pipeline (multiplier/threshold,
// derivative-based afwidth, smoothstep — and the per-channel offset lookups plus DistanceAdjust uniform for the LCD
// variant), the atlas uploads SDF glyphs through the padding-0/inset-2 lane, and the small-text regression gate:
// axis-aligned small text keeps the bitmap-text GP.

package gl

import (
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// drawSDFTestText draws one blob of text at the given size and rotation through a fresh device, returning the recorded
// AtlasTextOp.
func drawSDFTestText(t *testing.T, sdc *SurfaceDrawContext, size, rotateDeg float32, edging font.Edging, msg string) *AtlasTextOp {
	t.Helper()
	dev := NewDevice(sdc)
	c := canvas.New(dev)

	f := loadAtlasTestFont(t, "Roboto-Regular.ttf", size)
	f.SetEdging(edging)
	blob := makeTestBlob(t, f, msg)
	paint := canvas.NewPaint()
	paint.Color = colorcore.Color(0xFF102030)
	paint.AntiAlias = true

	if rotateDeg != 0 {
		var m geom.Matrix
		m.SetRotate(rotateDeg)
		c.Concat(&m)
	}
	chainsBefore := sdc.GetOpsTask().NumOpChains()
	c.DrawTextBlob(blob, 8, 200, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 1 {
		t.Fatalf("op chains = %d, want 1", got)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(chainsBefore).(*AtlasTextOp)
	if !ok {
		t.Fatalf("chain head is %T, want *AtlasTextOp", sdc.GetOpsTask().GetChainHead(chainsBefore))
	}
	return op
}

// flushAndFragmentSources flushes and returns the captured fragment-shader sources.
func flushAndFragmentSources(dc *DirectContext) []string {
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	var frags []string
	for _, src := range capturedShaderSource {
		if strings.Contains(src, "gl_FragCoord") || !strings.Contains(src, "gl_Position") {
			frags = append(frags, src)
		}
	}
	return frags
}

func TestSDFTextGrayscaleDF(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 512, 512)
	defer sdc.Release()

	// 200px rotated text sits in the default SDFT window (162 <= size <= glyphsAsPathsFontSize on every platform).
	op := drawSDFTestText(t, sdc, 200, 30, font.EdgingAntiAlias, "SD")
	if op.maskType != maskTypeGrayscaleDistanceField {
		t.Fatalf("mask type = %d, want grayscale distance field", op.maskType)
	}
	if !op.needsGlyphTransform {
		t.Error("SDFT op must set needsGlyphTransform")
	}
	if op.dfgpFlags&similarityDistanceFieldEffectFlag == 0 {
		t.Error("rotation is a similarity transform: kSimilarity flag expected")
	}
	if op.dfgpFlags&scaleOnlyDistanceFieldEffectFlag != 0 {
		t.Error("rotation is not scale-translate: kScaleOnly flag unexpected")
	}
	if op.dfgpFlags&(aliasedDistanceFieldEffectFlag|useLCDDistanceFieldEffectFlag|
		gammaCorrectDistanceFieldEffectFlag) != 0 {
		t.Errorf("unexpected DFGP flags %#x", op.dfgpFlags)
	}

	frags := flushAndFragmentSources(dc)
	if uploads := counts("glTexSubImage2D"); uploads == 0 {
		t.Fatal("expected SDF atlas uploads during flush")
	}
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("draw calls = %d, want 1", draws)
	}
	// The GP must be the A8 distance-field processor.
	if _, ok := op.gp.(*distanceFieldA8TextGeoProc); !ok {
		t.Fatalf("gp is %T, want *distanceFieldA8TextGeoProc", op.gp)
	}
	// Structural GLSL check: the DF constants, similarity afwidth from derivatives, and the smoothstep coverage; no LCD
	// DistanceAdjust uniform in the A8 variant.
	var df string
	for _, src := range frags {
		if strings.Contains(src, "7.96875") {
			df = src
			break
		}
	}
	if df == "" {
		t.Fatalf("no fragment shader carries the DF multiplier; sources:\n%s",
			strings.Join(frags, "\n----\n"))
	}
	for _, want := range []string{"0.50196078431", "st_grad_len", "dFdx", "smoothstep(-afwidth, afwidth, distance)"} {
		if !strings.Contains(df, want) {
			t.Errorf("DF fragment shader missing %q:\n%s", want, df)
		}
	}
	if strings.Contains(df, "DistanceAdjust") {
		t.Error("A8 DF shader must not carry the LCD DistanceAdjust uniform")
	}
}

func TestSDFTextAliasedDF(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 512, 512)
	defer sdc.Release()

	op := drawSDFTestText(t, sdc, 200, 0, font.EdgingAlias, "A")
	if op.maskType != maskTypeAliasedDistanceField {
		t.Fatalf("mask type = %d, want aliased distance field", op.maskType)
	}
	if op.dfgpFlags&aliasedDistanceFieldEffectFlag == 0 {
		t.Error("kAliased flag expected")
	}
	if op.dfgpFlags&scaleOnlyDistanceFieldEffectFlag == 0 ||
		op.dfgpFlags&similarityDistanceFieldEffectFlag == 0 {
		t.Error("identity matrix is scale-translate and similarity")
	}

	frags := flushAndFragmentSources(dc)
	found := false
	for _, src := range frags {
		if strings.Contains(src, "distance > 0.0 ? 1.0 : 0.0") {
			found = true
		}
	}
	if !found {
		t.Error("aliased DF shader must use the monochrome step")
	}
}

func TestSDFTextLCDDF(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := lcdTestSDC(t, dc, 512, 512)
	defer sdc.Release()

	op := drawSDFTestText(t, sdc, 200, 30, font.EdgingSubpixelAntiAlias, "L")
	if op.maskType != maskTypeLCDDistanceField {
		t.Fatalf("mask type = %d, want LCD distance field", op.maskType)
	}
	if op.dfgpFlags&useLCDDistanceFieldEffectFlag == 0 {
		t.Error("kUseLCD flag expected")
	}
	if op.dfgpFlags&(bgrDistanceFieldEffectFlag|portraitDistanceFieldEffectFlag) != 0 {
		t.Error("RGB_H geometry must not set BGR/Portrait")
	}
	// The mask data is still the A8 (SDF) atlas, not A565.
	if got := op.maskType.maskFormat(); got != gpu.MaskFormatA8 {
		t.Errorf("mask format = %v, want A8", got)
	}

	frags := flushAndFragmentSources(dc)
	if _, ok := op.gp.(*distanceFieldLCDTextGeoProc); !ok {
		t.Fatalf("gp is %T, want *distanceFieldLCDTextGeoProc", op.gp)
	}
	var df string
	for _, src := range frags {
		if strings.Contains(src, "DistanceAdjust") {
			df = src
			break
		}
	}
	if df == "" {
		t.Fatal("no fragment shader carries the LCD DistanceAdjust uniform")
	}
	for _, want := range []string{
		"uv_adjusted", "distance.y", "distance.x", "distance.z",
		"st_grad",
	} {
		if !strings.Contains(df, want) {
			t.Errorf("LCD DF fragment shader missing %q:\n%s", want, df)
		}
	}
	// The DistanceAdjust uniform is uploaded (Set3f) during the draw.
	if counts("glUniform3f") == 0 {
		t.Error("expected the DistanceAdjust glUniform3f upload")
	}
}

func TestSDFTextSmallTextKeepsBitmapGP(t *testing.T) {
	// The default-output regression gate: 16px axis-aligned text stays on the direct-mask bitmap-text lane even though
	// the context supports SDFT.
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 512, 512)
	defer sdc.Release()

	op := drawSDFTestText(t, sdc, 16, 0, font.EdgingAntiAlias, "hi")
	if op.maskType != maskTypeGrayscaleCoverage {
		t.Fatalf("mask type = %d, want grayscale coverage", op.maskType)
	}
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if _, ok := op.gp.(*bitmapTextGeoProc); !ok {
		t.Fatalf("gp is %T, want *bitmapTextGeoProc", op.gp)
	}
}

// TestGLDeviceTextLaneBoundariesPerOS pins the per-OS fGlyphsAsPathsFontSize boundary through the device path with each
// OS's option value injected explicitly, so every GOOS runs every variant (the local simulation of the non-darwin CI
// legs: 300px text is path-lane under the mac 256 threshold but rides the SDFT atlas under the linux/windows 324
// threshold).
func TestGLDeviceTextLaneBoundariesPerOS(t *testing.T) {
	const size float32 = 300
	for _, tc := range []struct {
		name         string
		maxSize      float32
		wantAtlasSDF bool
	}{
		{name: "mac-256", maxSize: 256, wantAtlasSDF: false},
		{name: "linux-windows-324", maxSize: 324, wantAtlasSDF: true},
	} {
		options := gpu.DefaultContextOptions()
		options.GlyphsAsPathsFontSize = tc.maxSize
		dc := newShaderRecordingContextWithOptions(t, options)
		sdc := newDrawTestSDC(t, dc, 512, 512)

		dev := NewDevice(sdc)
		c := canvas.New(dev)
		f := loadAtlasTestFont(t, "Roboto-Regular.ttf", size)
		paint := canvas.NewPaint()
		paint.AntiAlias = true

		chainsBefore := sdc.GetOpsTask().NumOpChains()
		c.DrawTextBlob(makeTestBlob(t, f, "Q"), 0, 300, paint)
		if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got == 0 {
			t.Fatalf("%s: text recorded no draw ops", tc.name)
		}
		sawAtlasSDF := false
		for i := chainsBefore; i < sdc.GetOpsTask().NumOpChains(); i++ {
			if op, isAtlas := sdc.GetOpsTask().GetChainHead(i).(*AtlasTextOp); isAtlas {
				if !op.maskType.usesDistanceFields() {
					t.Errorf("%s: atlas op mask type = %d, want a distance-field type",
						tc.name, op.maskType)
				}
				sawAtlasSDF = true
			}
		}
		if sawAtlasSDF != tc.wantAtlasSDF {
			t.Errorf("%s: 300px text rode AtlasTextOp(SDF)=%v, want %v",
				tc.name, sawAtlasSDF, tc.wantAtlasSDF)
		}

		// Beyond this variant's max size (the derivation TestGLDeviceTextPathLane uses on the host defaults), text is
		// path-lane under every threshold.
		fOver := loadAtlasTestFont(t, "Roboto-Regular.ttf", tc.maxSize+44)
		chainsOver := sdc.GetOpsTask().NumOpChains()
		c.DrawTextBlob(makeTestBlob(t, fOver, "Q"), 0, 460, paint)
		if got := sdc.GetOpsTask().NumOpChains() - chainsOver; got == 0 {
			t.Fatalf("%s: oversized text recorded no draw ops", tc.name)
		}
		for i := chainsOver; i < sdc.GetOpsTask().NumOpChains(); i++ {
			if _, isAtlas := sdc.GetOpsTask().GetChainHead(i).(*AtlasTextOp); isAtlas {
				t.Errorf("%s: oversized (%v px) text must not ride AtlasTextOp",
					tc.name, tc.maxSize+44)
			}
		}
		dc.FlushAndSubmit(false)
		sdc.Release()
	}
}

func TestSDFTextDFOpsDoNotMergeWithCoverage(t *testing.T) {
	// A DF op and a coverage op with the same paint cannot combine (mask types differ), and two DF ops with different
	// luminance colors cannot combine either.
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 512, 512)
	defer sdc.Release()

	dev := NewDevice(sdc)
	c := canvas.New(dev)

	fBig := loadAtlasTestFont(t, "Roboto-Regular.ttf", 200)
	fSmall := loadAtlasTestFont(t, "Roboto-Regular.ttf", 16)
	paint := canvas.NewPaint()
	paint.Color = colorcore.Color(0xFF102030)
	paint.AntiAlias = true

	chainsBefore := sdc.GetOpsTask().NumOpChains()
	c.DrawTextBlob(makeTestBlob(t, fBig, "D"), 8, 200, paint)
	c.DrawTextBlob(makeTestBlob(t, fSmall, "d"), 8, 40, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 2 {
		t.Fatalf("op chains = %d, want 2 (DF and coverage cannot merge)", got)
	}
	dfOp, ok := sdc.GetOpsTask().GetChainHead(chainsBefore).(*AtlasTextOp)
	if !ok || !dfOp.maskType.usesDistanceFields() {
		t.Fatalf("first chain is not the DF op")
	}

	// Two more DF blobs with the same luminance color merge into the existing DF chain.
	c.DrawTextBlob(makeTestBlob(t, fBig, "E"), 8, 400, paint)
	c.DrawTextBlob(makeTestBlob(t, fBig, "F"), 200, 400, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 2 {
		t.Fatalf("op chains after merge = %d, want 2 (same luminance merges)", got)
	}
	if dfOp.numGlyphs != 3 {
		t.Fatalf("merged DF glyph count = %d, want 3", dfOp.numGlyphs)
	}

	// A different luminance color blocks the DF merge (a new chain appears).
	paint2 := canvas.NewPaint()
	paint2.Color = colorcore.Color(0xFFFFFFFF)
	paint2.AntiAlias = true
	c.DrawTextBlob(makeTestBlob(t, fBig, "G"), 8, 460, paint2)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 3 {
		t.Fatalf("op chains = %d, want 3 (luminance mismatch cannot merge)", got)
	}
	if dfOp.numGlyphs != 3 {
		t.Fatalf("DF glyph count after luminance mismatch = %d, want 3", dfOp.numGlyphs)
	}
	dc.FlushAndSubmit(false)
}
