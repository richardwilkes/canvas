// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the GPU device's text lowering through BlobRedrawCoordinator → SubRunContainer →
// AtlasTextOp, verifying subrun staging, op batching, atlas uploads, and the draw/flush-token balance across a real
// flush.

package gl

import (
	"os"
	"slices"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/surface"
	"github.com/richardwilkes/canvas/textblob"
)

func loadAtlasTestFont(t *testing.T, name string, size float32) *font.Font {
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

func makeTestBlob(t *testing.T, f *font.Font, text string) *textblob.Blob {
	t.Helper()
	blob := textblob.MakeFromText([]byte(text), f, font.TextEncodingUTF8)
	if blob == nil {
		t.Fatal("blob creation failed")
	}
	return blob
}

func TestAtlasTextOpRecordAndFlush(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()

	dev := NewDevice(sdc)
	c := canvas.New(dev)

	f := loadAtlasTestFont(t, "Roboto-Regular.ttf", 16)
	blob := makeTestBlob(t, f, "Atlas")
	paint := canvas.NewPaint()
	paint.Color = colorcore.Color(0xFF102030)
	paint.AntiAlias = true

	chainsBefore := sdc.GetOpsTask().NumOpChains()
	c.DrawTextBlob(blob, 8, 100, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 1 {
		t.Fatalf("op chains after one blob = %d, want 1", got)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(chainsBefore).(*AtlasTextOp)
	if !ok {
		t.Fatalf("chain head is %T, want *AtlasTextOp", sdc.GetOpsTask().GetChainHead(chainsBefore))
	}
	if op.maskType != maskTypeGrayscaleCoverage {
		t.Errorf("mask type = %d, want grayscale coverage", op.maskType)
	}
	if op.numGlyphs != 5 {
		t.Errorf("glyph count = %d, want 5", op.numGlyphs)
	}

	// A second blob with the same paint merges into the same op at record time.
	blob2 := makeTestBlob(t, f, "merge")
	c.DrawTextBlob(blob2, 8, 60, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 1 {
		t.Fatalf("op chains after two blobs = %d, want 1 (merged)", got)
	}
	if op.numGlyphs != 10 {
		t.Errorf("merged glyph count = %d, want 10", op.numGlyphs)
	}
	if len(op.geometries) != 2 {
		t.Errorf("merged geometry count = %d, want 2", len(op.geometries))
	}

	// Flush: the atlas uploads the ten glyph masks (ASAP lane) and the merged op issues one patterned draw; the token
	// stream must balance (executeRenderTasks panics otherwise).
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if uploads := counts("glTexSubImage2D"); uploads == 0 {
		t.Fatal("expected atlas glyph uploads during flush")
	}
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("draw calls = %d, want 1 for the merged op", draws)
	}
	if len(op.recordedDraws) != 1 {
		t.Fatalf("recorded draws = %d, want 1", len(op.recordedDraws))
	}

	// A redraw of the first blob reuses the cached Blob (no new subrun conversion) and, with every glyph already in
	// the atlas, uploads nothing new.
	c.DrawTextBlob(blob, 8, 100, paint)
	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if uploads := counts("glTexSubImage2D"); uploads != 0 {
		t.Fatalf("re-draw uploaded %d subimages, want 0 (atlas hit)", uploads)
	}
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("re-draw draw calls = %d, want 1", draws)
	}
}

func TestAtlasTextOpColorEmoji(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 64, 64)
	defer sdc.Release()

	dev := NewDevice(sdc)
	c := canvas.New(dev)

	f := loadAtlasTestFont(t, "colr.ttf", 24)
	blob := makeTestBlob(t, f, "\U0001F600")
	paint := canvas.NewPaint()

	chainsBefore := sdc.GetOpsTask().NumOpChains()
	c.DrawTextBlob(blob, 4, 40, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 1 {
		t.Fatalf("op chains = %d, want 1", got)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(chainsBefore).(*AtlasTextOp)
	if !ok {
		t.Fatalf("chain head is %T, want *AtlasTextOp", sdc.GetOpsTask().GetChainHead(chainsBefore))
	}
	if op.maskType != maskTypeColorBitmap {
		t.Errorf("mask type = %d, want color bitmap", op.maskType)
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if uploads := counts("glTexSubImage2D"); uploads == 0 {
		t.Fatal("expected an ARGB atlas upload during flush")
	}
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("draw calls = %d, want 1", draws)
	}
}

func TestAtlasTextMergeGates(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()

	dev := NewDevice(sdc)
	c := canvas.New(dev)

	f := loadAtlasTestFont(t, "Roboto-Regular.ttf", 16)
	emoji := loadAtlasTestFont(t, "colr.ttf", 16)

	paint := canvas.NewPaint()
	chainsBefore := sdc.GetOpsTask().NumOpChains()
	// An A8 coverage op and an ARGB color op must not merge (mask types differ).
	c.DrawTextBlob(makeTestBlob(t, f, "a"), 4, 30, paint)
	c.DrawTextBlob(makeTestBlob(t, emoji, "\U0001F600"), 4, 60, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 2 {
		t.Fatalf("op chains = %d, want 2 (A8 and ARGB must not merge)", got)
	}
	dc.FlushAndSubmit(false)
}

func TestGLDeviceTextPathLane(t *testing.T) {
	// Text beyond the SubRunControl's max glyphs-as-paths font size (per-OS: mac 256, linux/windows 324) exceeds every
	// atlas gate: the PathSubRun re-enters the canvas as path draws, which ride the SDC shape funnel instead of
	// AtlasTextOp. The size is derived from the device's own control rather than hard-coded so the test holds on every
	// GOOS (a fixed 300px was path-lane on darwin but inside the SDFT window elsewhere); the per-OS boundary itself is
	// pinned GOOS-independently by TestGLDeviceTextLaneBoundariesPerOS.
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()

	dev := NewDevice(sdc)
	c := canvas.New(dev)

	control := dc.GetSubRunControl(false)
	oversized := control.MaxSize() + 44
	paint := canvas.NewPaint()
	paint.AntiAlias = true
	identity := geom.IdentityMatrix()
	// Precondition: with the identity CTM the approximate device text size equals the font size, and this size must be
	// beyond both atlas decisions on this configuration.
	if control.IsSDFT(oversized, paint, &identity) || control.IsDirect(oversized, paint, &identity) {
		t.Fatalf("size %v is not beyond the atlas gates (max %v)", oversized, control.MaxSize())
	}

	f := loadAtlasTestFont(t, "Roboto-Regular.ttf", oversized)
	chainsBefore := sdc.GetOpsTask().NumOpChains()
	c.DrawTextBlob(makeTestBlob(t, f, "Q"), 0, 120, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got == 0 {
		t.Fatal("path-lane text must record draw ops")
	}
	for i := chainsBefore; i < sdc.GetOpsTask().NumOpChains(); i++ {
		if _, isAtlas := sdc.GetOpsTask().GetChainHead(i).(*AtlasTextOp); isAtlas {
			t.Fatal("oversized text must not ride AtlasTextOp")
		}
	}
	dc.FlushAndSubmit(false)
}

func TestBitmapTextGeoProcKeysAndStrides(t *testing.T) {
	dc := newFakeDirectContext(t)
	caps := dc.Caps().ShaderCaps
	identity := geom.IdentityMatrix()

	a8 := newBitmapTextGeoProc(caps, colorcore.PMColor4f{A: 1}, nil, 0,
		bitmapTextFilterFromNeedsTransform(false), gpu.MaskFormatA8, &identity, false)
	// Vertex layout: float2 position + packed 4-byte color + ushort2 texture coords.
	if got := a8.VertexStride(); got != 16 {
		t.Errorf("A8 GP stride = %d, want 16", got)
	}
	argb := newBitmapTextGeoProc(caps, colorcore.PMColor4f{A: 1}, nil, 0,
		bitmapTextFilterFromNeedsTransform(false), gpu.MaskFormatARGB, &identity, false)
	if got := argb.VertexStride(); got != 12 {
		t.Errorf("ARGB GP stride = %d, want 12", got)
	}
	a8persp := newBitmapTextGeoProc(caps, colorcore.PMColor4f{A: 1}, nil, 0,
		bitmapTextFilterFromNeedsTransform(true), gpu.MaskFormatA8, &identity, true)
	if got := a8persp.VertexStride(); got != 20 {
		t.Errorf("A8 perspective GP stride = %d, want 20", got)
	}
	argbPersp := newBitmapTextGeoProc(caps, colorcore.PMColor4f{A: 1}, nil, 0,
		bitmapTextFilterFromNeedsTransform(true), gpu.MaskFormatARGB, &identity, true)
	if got := argbPersp.VertexStride(); got != 16 {
		t.Errorf("ARGB perspective GP stride = %d, want 16", got)
	}

	// Two GPs with different mask formats must produce different program keys so the program cache keeps them apart
	// (keys are compared as their raw packed words, per KeyBuilder).
	var wordsA8, wordsARGB []uint32
	bA8 := gpu.NewKeyBuilder(&wordsA8)
	a8.AddToKey(caps, bA8)
	bA8.Flush()
	bARGB := gpu.NewKeyBuilder(&wordsARGB)
	argb.AddToKey(caps, bARGB)
	bARGB.Flush()
	if slices.Equal(wordsA8, wordsARGB) {
		t.Error("A8 and ARGB GP keys must differ")
	}
}

// lcdTestSDC builds an SDC whose surface props carry RGB_H pixel geometry (the E.4 LCD enablement).
func lcdTestSDC(t *testing.T, dc *DirectContext, w, h int32) *SurfaceDrawContext {
	t.Helper()
	props := surface.Props{PixelGeometry: surface.PixelGeometryRGBH}
	sdc := MakeSurfaceDrawContextWithProps(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: w, Height: h}, gpu.BackingFitExact, 1, gpu.MipmappedNo,
		gpu.OriginTopLeft, gpu.BudgetedYes, "lcd-text-test", &props)
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContextWithProps failed")
	}
	return sdc
}

// TestAtlasTextOpLCD drives LCD16 glyphs through the full GPU lane: the subrun stages A565, the op takes the LCD
// coverage mask type, the atlas page is an RGB565 texture, and the flush uploads and draws under the token protocol.
func TestAtlasTextOpLCD(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := lcdTestSDC(t, dc, 128, 128)
	defer sdc.Release()

	dev := NewDevice(sdc)
	c := canvas.New(dev)

	f := loadAtlasTestFont(t, "Roboto-Regular.ttf", 16)
	f.SetEdging(font.EdgingSubpixelAntiAlias)
	blob := makeTestBlob(t, f, "Lcd")
	paint := canvas.NewPaint()
	paint.Color = colorcore.Color(0xFF102030)
	paint.AntiAlias = true

	chainsBefore := sdc.GetOpsTask().NumOpChains()
	c.DrawTextBlob(blob, 8, 100, paint)
	if got := sdc.GetOpsTask().NumOpChains() - chainsBefore; got != 1 {
		t.Fatalf("op chains = %d, want 1", got)
	}
	op, ok := sdc.GetOpsTask().GetChainHead(chainsBefore).(*AtlasTextOp)
	if !ok {
		t.Fatalf("chain head is %T, want *AtlasTextOp", sdc.GetOpsTask().GetChainHead(chainsBefore))
	}
	if op.maskType != maskTypeLCDCoverage {
		t.Fatalf("mask type = %d, want LCD coverage", op.maskType)
	}

	recCounts = map[string]int{}
	dc.FlushAndSubmit(false)
	if uploads := counts("glTexSubImage2D"); uploads == 0 {
		t.Fatal("expected A565 atlas uploads during flush")
	}
	if draws := counts("glDrawRangeElements") + counts("glDrawElements") +
		counts("glDrawArrays"); draws != 1 {
		t.Fatalf("draw calls = %d, want 1", draws)
	}
	// The atlas page backing the op is the 565 format (MaskFormatA565 → ColorTypeBGR565 → RGB565).
	if op.views == nil || op.numActiveViews == 0 {
		t.Fatal("op has no atlas views after flush")
	}
	if got := op.views[0].Proxy().Format(); got != FormatRGB565 {
		t.Fatalf("atlas page format = %v, want RGB565", got)
	}

	// The same text with kAntiAlias edging stays on the grayscale lane even on the LCD surface props path when drawn
	// into a layer-like unknown-geometry SDC.
	plainSDC := newDrawTestSDC(t, dc, 128, 128)
	defer plainSDC.Release()
	c2 := canvas.New(NewDevice(plainSDC))
	c2.DrawTextBlob(blob, 8, 100, paint)
	op2, ok := plainSDC.GetOpsTask().GetChainHead(0).(*AtlasTextOp)
	if !ok {
		t.Fatal("plain SDC chain head is not an AtlasTextOp")
	}
	if op2.maskType != maskTypeGrayscaleCoverage {
		t.Fatalf("unknown-geometry mask type = %d, want grayscale", op2.maskType)
	}
	dc.FlushAndSubmit(false)
}

// TestLCDXferProcessorLanes pins the blend configuration AtlasTextOp's LCD coverage produces under
// dual-source-available and fallback caps (PorterDuffXPFactory's LCD lanes).
func TestLCDXferProcessorLanes(t *testing.T) {
	dc := newFakeDirectContext(t)
	caps := dc.Caps()
	if !caps.ShaderCaps.DualSourceBlendingSupport {
		t.Fatal("fake driver should support dual-source blending")
	}
	factory := PorterDuffXPFactory(raster.BlendSrcOver)
	constant := AnalysisColorConstant(colorcore.PMColor4f{R: 0.1, G: 0.2, B: 0.3, A: 1})

	// Dual source available: the LCD blend formula with a secondary output (coverage as the second source), no dst
	// read.
	xp := XPFactoryMakeXferProcessor(factory, constant, AnalysisCoverageLCD, caps, gpu.ClampTypeAuto)
	info := xpGetBlendInfo(xp)
	want := gpu.GetLCDBlendFormula(raster.BlendSrcOver)
	if info.SrcBlend != want.SrcCoeff() || info.DstBlend != want.DstCoeff() ||
		info.Equation != want.Equation() {
		t.Fatalf("dual-source LCD blend = %+v, want formula %+v", info, want)
	}
	if xp.xpBase().WillReadDstColor() {
		t.Fatal("dual-source LCD XP must not read the dst")
	}

	// Dual source unavailable with a constant color: the PDLCD constant-color trick (the color folded into the blend
	// constant with (ConstC, ISC)), still no dst read. The dst coefficient is one-minus-*src*-color, since the shader's
	// src output carries the per-channel coverage times alpha and the dst must be attenuated by that coverage;
	// attenuating by the blend constant (IConstC) instead would scale the destination by the text color.
	noDual := *caps
	shaderCapsCopy := *caps.ShaderCaps
	shaderCapsCopy.DualSourceBlendingSupport = false
	shaderCapsCopy.DstReadInShaderSupport = false
	noDual.ShaderCaps = &shaderCapsCopy
	xp = XPFactoryMakeXferProcessor(factory, constant, AnalysisCoverageLCD, &noDual,
		gpu.ClampTypeAuto)
	if _, isPDLCD := xp.(*pdLCDXP); !isPDLCD {
		t.Fatalf("fallback XP is %T, want *pdLCDXP", xp)
	}
	info = xpGetBlendInfo(xp)
	if info.SrcBlend != gpu.BlendCoeffConstC || info.DstBlend != gpu.BlendCoeffISC {
		t.Fatalf("PDLCD blend = %+v, want (ConstC, ISC)", info)
	}
	if info.BlendConstant != [4]float32{0.1, 0.2, 0.3, 1} {
		t.Fatalf("PDLCD blend constant = %v, want the unpremultiplied color with opaque alpha",
			info.BlendConstant)
	}

	// Dual source unavailable with a non-constant color: the dst-read fallback.
	xp = XPFactoryMakeXferProcessor(factory, AnalysisColorUnknown(), AnalysisCoverageLCD, &noDual,
		gpu.ClampTypeAuto)
	if !xp.xpBase().WillReadDstColor() {
		t.Fatal("non-constant LCD color without dual source must read the dst")
	}
}

// TestGetPackedGlyphImageLCD16 pins the A565 atlas packing lanes: the row copy for LCD16 glyphs into a 565 page, and
// the 565→8888 expansion for drivers without a 565 format (bit-replicated channels, opaque alpha, and RGBA word order —
// N32 is RGBA8888 on every leg, so the native-BGRA flag is constant false).
func TestGetPackedGlyphImageLCD16(t *testing.T) {
	g := &font.Glyph{Width: 2, Height: 2, Format: font.MaskLCD16}
	g.Image16 = []uint16{0xF800, 0x07E0, 0x001F, 0x1234}

	// Row-copy lane into a padded destination (dstRB wider than the tight row).
	dst := make([]byte, 3*2*2) // dstRB = 6 bytes for 2 pixels of 2 bytes
	getPackedGlyphImage(g, 6, gpu.MaskFormatA565, dst)
	for i, want := range []uint16{0xF800, 0x07E0} {
		got := uint16(dst[2*i]) | uint16(dst[2*i+1])<<8
		if got != want {
			t.Errorf("row 0 texel %d = %#04x, want %#04x", i, got, want)
		}
	}
	for i, want := range []uint16{0x001F, 0x1234} {
		got := uint16(dst[6+2*i]) | uint16(dst[6+2*i+1])<<8
		if got != want {
			t.Errorf("row 1 texel %d = %#04x, want %#04x", i, got, want)
		}
	}

	// Expansion lane: 565 → RGBA8888 words with bit replication and opaque alpha.
	dst = make([]byte, 2*2*4)
	getPackedGlyphImage(g, 8, gpu.MaskFormatARGB, dst)
	wantWords := [][4]byte{
		{0xFF, 0x00, 0x00, 0xFF}, // pure red
		{0x00, 0xFF, 0x00, 0xFF}, // pure green
		{0x00, 0x00, 0xFF, 0xFF}, // pure blue
		{0x10, 0x45, 0xA5, 0xFF}, // 0x1234: r=2→0x10, g=0x11→0x45, b=0x14→0xA5
	}
	for i, want := range wantWords {
		got := [4]byte{dst[4*i], dst[4*i+1], dst[4*i+2], dst[4*i+3]}
		if got != want {
			t.Errorf("expanded texel %d = %v, want %v", i, got, want)
		}
	}
}
