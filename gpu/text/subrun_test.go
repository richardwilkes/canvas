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
	"os"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/stroke"
	"github.com/richardwilkes/canvas/textblob"
)

func loadTestFont(t *testing.T, name string, size float32) *font.Font {
	t.Helper()
	data, err := os.ReadFile("../../font/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return font.NewFont(tf, size, 1, 0)
}

// runListForText builds a blob-backed glyph-run list for text drawn at origin.
func runListForText(t *testing.T, f *font.Font, text string, origin geom.Point) *textblob.GlyphRunList {
	t.Helper()
	blob := textblob.MakeFromText([]byte(text), f, font.TextEncodingUTF8)
	if blob == nil {
		t.Fatal("blob creation failed")
	}
	builder := textblob.NewGlyphRunBuilder()
	list := builder.BlobToGlyphRunList(blob, origin)
	if list.Empty() {
		t.Fatal("empty glyph run list")
	}
	return list
}

func fillPaint() *canvas.Paint { return canvas.NewPaint() }

// noSDFTControl builds the SubRunControl equivalent to the pre-E.1 trimmed configuration: SDFT off (as on a device
// without derivatives support), maxMaskSize 256.
func noSDFTControl() *SubRunControl {
	c := NewSubRunControl(false, false, true, 18, 256, false)
	return &c
}

// sdftControl builds an SDFT-enabled SubRunControl at the default option thresholds (min 18, paths above 256; no
// small-text SDFT), mirroring getSubRunControl on a derivatives-capable device with kUseDeviceIndependentFonts unset.
func sdftControl() *SubRunControl {
	c := NewSubRunControl(true, false, true, 18, 256, false)
	return &c
}

func TestMakeSubRunsDirectMask(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "Hello", geom.Pt(10, 20))
	positionMatrix := geom.IdentityMatrix()
	positionMatrix.PreTranslate(10, 20)

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, noSDFTControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	direct, ok := container.SubRuns()[0].(*DirectMaskSubRun)
	if !ok {
		t.Fatalf("subrun type = %T, want DirectMaskSubRun", container.SubRuns()[0])
	}
	if direct.MaskFormat() != gpu.MaskFormatA8 {
		t.Errorf("mask format = %v, want A8", direct.MaskFormat())
	}
	if direct.GlyphCount() != 5 {
		t.Errorf("glyph count = %d, want 5", direct.GlyphCount())
	}
	if direct.GlyphSrcPadding() != 0 {
		t.Errorf("direct padding = %d, want 0", direct.GlyphSrcPadding())
	}

	// Integer translation reuses; fractional translation does not.
	intTranslate := positionMatrix
	intTranslate.PostTranslate(5, -3)
	if !container.CanReuse(&intTranslate) {
		t.Error("integer translate must be reusable")
	}
	fracTranslate := positionMatrix
	fracTranslate.PostTranslate(0.5, 0)
	if container.CanReuse(&fracTranslate) {
		t.Error("fractional translate must not be reusable")
	}
	var scaled geom.Matrix
	scaled.SetScale(3, 3)
	if container.CanReuse(&scaled) {
		t.Error("scale change must not be reusable")
	}

	// The direct subrun's device rect under the creation matrix must be non-empty and needs no transform.
	needsTransform, rect := direct.DeviceRectAndNeedsTransform(&positionMatrix)
	if needsTransform || rect.IsEmpty() {
		t.Errorf("deviceRect: needsTransform=%v rect=%v", needsTransform, rect)
	}
}

func TestMakeSubRunsPathLane(t *testing.T) {
	// 300px text exceeds the 256 direct-mask gate, so outline glyphs ride the path lane.
	f := loadTestFont(t, "Roboto-Regular.ttf", 300)
	list := runListForText(t, f, "A", geom.Pt(0, 300))
	positionMatrix := geom.IdentityMatrix()

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, noSDFTControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	if _, ok := container.SubRuns()[0].(*PathSubRun); !ok {
		t.Fatalf("subrun type = %T, want PathSubRun", container.SubRuns()[0])
	}
	// Path subruns reuse under any matrix.
	var scaled geom.Matrix
	scaled.SetScale(4, 4)
	if !container.CanReuse(&scaled) {
		t.Error("path subruns must reuse under any matrix")
	}
}

func TestMakeSubRunsHairlineSkipsDirect(t *testing.T) {
	// Hairline strokes bypass the atlas lanes entirely.
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "x", geom.Pt(0, 0))
	positionMatrix := geom.IdentityMatrix()

	paint := canvas.NewPaint()
	paint.Style = canvas.StyleStroke
	paint.StrokeWidth = 0
	container := MakeSubRuns(list, &positionMatrix, paint, nil, noSDFTControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	if _, ok := container.SubRuns()[0].(*PathSubRun); !ok {
		t.Fatalf("subrun type = %T, want PathSubRun", container.SubRuns()[0])
	}
}

func TestMakeSubRunsColorGlyphs(t *testing.T) {
	// Small color emoji ride the direct-mask lane as ARGB.
	f := loadTestFont(t, "colr.ttf", 16)
	blob := textblob.MakeFromText([]byte("\U0001F600"), f, font.TextEncodingUTF8)
	if blob == nil {
		t.Fatal("blob creation failed")
	}
	builder := textblob.NewGlyphRunBuilder()
	list := builder.BlobToGlyphRunList(blob, geom.Pt(2, 12))
	positionMatrix := geom.IdentityMatrix()

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, noSDFTControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	direct, ok := container.SubRuns()[0].(*DirectMaskSubRun)
	if !ok {
		t.Fatalf("subrun type = %T, want DirectMaskSubRun", container.SubRuns()[0])
	}
	if direct.MaskFormat() != gpu.MaskFormatARGB {
		t.Errorf("mask format = %v, want ARGB", direct.MaskFormat())
	}

	// Large color emoji cannot ride the path lane (no outlines), so they land in the transformed-mask drawing of last
	// resort with the bilerp padding.
	fBig := loadTestFont(t, "colr.ttf", 400)
	blobBig := textblob.MakeFromText([]byte("\U0001F600"), fBig, font.TextEncodingUTF8)
	builderBig := textblob.NewGlyphRunBuilder()
	listBig := builderBig.BlobToGlyphRunList(blobBig, geom.Pt(0, 0))
	containerBig := MakeSubRuns(listBig, &positionMatrix, fillPaint(), nil, noSDFTControl())
	if len(containerBig.SubRuns()) != 1 {
		t.Fatalf("big subrun count = %d, want 1", len(containerBig.SubRuns()))
	}
	transformed, ok := containerBig.SubRuns()[0].(*TransformedMaskSubRun)
	if !ok {
		t.Fatalf("big subrun type = %T, want TransformedMaskSubRun", containerBig.SubRuns()[0])
	}
	if transformed.MaskFormat() != gpu.MaskFormatARGB {
		t.Errorf("big mask format = %v, want ARGB", transformed.MaskFormat())
	}
	if transformed.GlyphSrcPadding() != 1 {
		t.Errorf("transformed padding = %d, want 1", transformed.GlyphSrcPadding())
	}
	needsTransform, rect := transformed.DeviceRectAndNeedsTransform(&positionMatrix)
	if !needsTransform || rect.IsEmpty() {
		t.Errorf("transformed deviceRect: needsTransform=%v rect=%v", needsTransform, rect)
	}
}

func TestMakeSubRunsPerspective(t *testing.T) {
	// Perspective skips the direct lane; outline glyphs go to paths.
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "p", geom.Pt(0, 0))
	positionMatrix := geom.IdentityMatrix()
	positionMatrix.Set(geom.MPersp0, 0.001)

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, noSDFTControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	if _, ok := container.SubRuns()[0].(*PathSubRun); !ok {
		t.Fatalf("subrun type = %T, want PathSubRun", container.SubRuns()[0])
	}
}

func TestMakeSubRunsSDFTLane(t *testing.T) {
	// 200px rotated text with SDFT enabled rides the distance-field lane.
	f := loadTestFont(t, "Roboto-Regular.ttf", 200)
	list := runListForText(t, f, "SDF", geom.Pt(0, 200))
	var positionMatrix geom.Matrix
	positionMatrix.SetRotate(30)

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, sdftControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	sdft, ok := container.SubRuns()[0].(*SDFTSubRun)
	if !ok {
		t.Fatalf("subrun type = %T, want SDFTSubRun", container.SubRuns()[0])
	}
	if sdft.MaskFormat() != gpu.MaskFormatA8 {
		t.Errorf("mask format = %v, want A8", sdft.MaskFormat())
	}
	if sdft.GlyphSrcPadding() != font.DistanceFieldInset {
		t.Errorf("padding = %d, want %d", sdft.GlyphSrcPadding(), font.DistanceFieldInset)
	}
	gp := sdft.GlyphParams()
	if !gp.IsSDF || gp.IsLCD || !gp.IsAA {
		t.Errorf("glyph params = %+v, want {IsSDF:true IsLCD:false IsAA:true}", gp)
	}
	needsTransform, rect := sdft.DeviceRectAndNeedsTransform(&positionMatrix)
	if !needsTransform || rect.IsEmpty() {
		t.Errorf("deviceRect: needsTransform=%v rect=%v", needsTransform, rect)
	}

	// Reuse follows the SDFT matrix range: any rotation at the same scale reuses; scales inside the strike's range
	// reuse; scales beyond it do not (200px text at 2x leaves any bucket).
	var rotated geom.Matrix
	rotated.SetRotate(-45)
	if !container.CanReuse(&rotated) {
		t.Error("rotation at the same scale must reuse the SDFT subrun")
	}
	var doubled geom.Matrix
	doubled.SetScale(2, 2)
	if container.CanReuse(&doubled) {
		t.Error("scaling to 2x (400px) must leave the matrix range")
	}

	// The accepted positions are the *inset* glyph bounds' left-top in creation space: strictly inside the raw padded
	// bounds.
	if got := sdft.GlyphCount(); got != 3 {
		t.Errorf("glyph count = %d, want 3", got)
	}
}

func TestMakeSubRunsSDFTSmallTextStaysDirect(t *testing.T) {
	// The default/unrotated regression gate: with SDFT enabled but useSDFTForSmallText off, axis-aligned small text
	// keeps taking the direct-mask path (isSDFT's min is 162).
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "Hi", geom.Pt(0, 20))
	positionMatrix := geom.IdentityMatrix()

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, sdftControl())
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	if _, ok := container.SubRuns()[0].(*DirectMaskSubRun); !ok {
		t.Fatalf("subrun type = %T, want DirectMaskSubRun", container.SubRuns()[0])
	}

	// A rotated small run also stays direct (rotation alone does not trigger SDFT; the strike bakes the rotation).
	var rot geom.Matrix
	rot.SetRotate(30)
	rotated := MakeSubRuns(list, &rot, fillPaint(), nil, sdftControl())
	if _, ok := rotated.SubRuns()[0].(*DirectMaskSubRun); !ok {
		t.Fatalf("rotated subrun type = %T, want DirectMaskSubRun", rotated.SubRuns()[0])
	}
}

func TestMakeSubRunsSDFTSmallTextWithDIF(t *testing.T) {
	// With useSDFTForSmallText (the kUseDeviceIndependentFonts surface flag), 30px text rides SDFT — the animated-text
	// demand case.
	f := loadTestFont(t, "Roboto-Regular.ttf", 30)
	list := runListForText(t, f, "anim", geom.Pt(0, 30))
	positionMatrix := geom.IdentityMatrix()

	difControl := NewSubRunControl(true, true, true, 18, 256, false)
	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, &difControl)
	if len(container.SubRuns()) != 1 {
		t.Fatalf("subrun count = %d, want 1", len(container.SubRuns()))
	}
	sdft, ok := container.SubRuns()[0].(*SDFTSubRun)
	if !ok {
		t.Fatalf("subrun type = %T, want SDFTSubRun", container.SubRuns()[0])
	}
	// 30px in the small bucket (32): the strike range covers modest scaling, so a 1.05x matrix (31.5px, still <= 32)
	// reuses while 2x does not.
	var slight geom.Matrix
	slight.SetScale(1.05, 1.05)
	if !sdft.CanReuse(&slight) {
		t.Error("slight scale within the bucket must reuse")
	}
	var doubled geom.Matrix
	doubled.SetScale(2, 2)
	if sdft.CanReuse(&doubled) {
		t.Error("2x scale must not reuse the small-bucket strike")
	}
}

func TestMakeSubRunsSDFTLCDGlyphParams(t *testing.T) {
	// An LCD-edged run keeps its useLCDText bit on the SDFT subrun (the LCD DF GP selector); the mask data itself stays
	// A8.
	f := loadTestFont(t, "Roboto-Regular.ttf", 200)
	f.SetEdging(font.EdgingSubpixelAntiAlias)
	list := runListForText(t, f, "L", geom.Pt(0, 200))
	positionMatrix := geom.IdentityMatrix()

	container := MakeSubRuns(list, &positionMatrix, fillPaint(), nil, sdftControl())
	sdft, ok := container.SubRuns()[0].(*SDFTSubRun)
	if !ok {
		t.Fatalf("subrun type = %T, want SDFTSubRun", container.SubRuns()[0])
	}
	gp := sdft.GlyphParams()
	if !gp.IsSDF || !gp.IsLCD || !gp.IsAA {
		t.Errorf("glyph params = %+v, want {IsSDF:true IsLCD:true IsAA:true}", gp)
	}
	if sdft.MaskFormat() != gpu.MaskFormatA8 {
		t.Errorf("mask format = %v, want A8", sdft.MaskFormat())
	}
}

func TestAddMultiMaskFormatSplits(t *testing.T) {
	accepted := []acceptedGlyph{
		{format: font.MaskA8},
		{format: font.MaskA8},
		{format: font.MaskARGB32},
		{format: font.MaskA8},
	}
	var got []struct {
		format gpu.MaskFormat
		count  int
	}
	addMultiMaskFormat(accepted, func(run []acceptedGlyph, format gpu.MaskFormat) {
		got = append(got, struct {
			format gpu.MaskFormat
			count  int
		}{format: format, count: len(run)})
	})
	want := []struct {
		format gpu.MaskFormat
		count  int
	}{{format: gpu.MaskFormatA8, count: 2}, {format: gpu.MaskFormatARGB, count: 1}, {format: gpu.MaskFormatA8, count: 1}}
	if len(got) != len(want) {
		t.Fatalf("split count = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("split %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestStrokeRecFromPaintHairline(t *testing.T) {
	paint := canvas.NewPaint()
	paint.Style = canvas.StyleStroke
	rec := strokeRecFromPaint(paint)
	if rec.Style() != stroke.StyleHairline {
		t.Errorf("style = %v, want hairline", rec.Style())
	}
}

func TestGpuStrikeCache(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	identity := geom.IdentityMatrix()
	spec := font.MakeMaskSpec(f, &font.ScalerPaint{}, &identity, nil)

	cache := NewStrikeCache()
	s1 := cache.FindOrCreateStrike(&spec)
	s2 := cache.FindOrCreateStrike(&spec)
	if s1 != s2 {
		t.Fatal("same spec must return the same strike")
	}
	g1 := s1.GetGlyph(font.PackGlyphID(40))
	g2 := s1.GetGlyph(font.PackGlyphID(40))
	if g1 != g2 {
		t.Fatal("same packed ID must return the same glyph")
	}
	if cache.StrikeCount() != 1 {
		t.Fatalf("strike count = %d", cache.StrikeCount())
	}
	before := cache.TotalMemoryUsed()
	s1.GetGlyph(font.PackGlyphID(41))
	if cache.TotalMemoryUsed() <= before {
		t.Fatal("glyph creation must grow the accounting")
	}
	cache.FreeAll()
	if cache.StrikeCount() != 0 || cache.TotalMemoryUsed() != 0 {
		t.Fatalf("freeAll left count=%d bytes=%d", cache.StrikeCount(), cache.TotalMemoryUsed())
	}
}

func TestGlyphVectorResolution(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	identity := geom.IdentityMatrix()
	spec := font.MakeMaskSpec(f, &font.ScalerPaint{}, &identity, nil)
	strike := spec.FindOrCreateStrike()

	gv := MakeGlyphVector(strike, &spec, []font.PackedGlyphID{
		font.PackGlyphID(40), font.PackGlyphID(41),
	})
	if gv.GlyphCount() != 2 {
		t.Fatalf("glyph count = %d", gv.GlyphCount())
	}
	if gv.AtlasGeneration() != gpu.InvalidAtlasGeneration {
		t.Fatal("fresh vector must have the invalid generation")
	}

	cache := NewStrikeCache()
	gv.PackedGlyphIDToGlyph(cache)
	glyphs := gv.Glyphs()
	if glyphs[0] == nil || glyphs[1] == nil || glyphs[0] == glyphs[1] {
		t.Fatalf("resolution produced %v", glyphs)
	}
	if glyphs[0].PackedID != font.PackGlyphID(40) {
		t.Errorf("glyph 0 packed ID = %v", glyphs[0].PackedID)
	}
	// Resolution is idempotent and stable.
	gv.PackedGlyphIDToGlyph(cache)
	if gv.Glyphs()[0] != glyphs[0] {
		t.Fatal("second resolution changed the glyphs")
	}
	if gv.Strike() == nil {
		t.Fatal("text strike must be resolved")
	}
}
