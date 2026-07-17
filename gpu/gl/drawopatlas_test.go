// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the atlas stack: multitexture page activation/compaction over a testing upload target, the
// atlas config's dimension table, the inline-upload/try-again protocol, eviction bookkeeping, the atlas manager's
// initialization/format resolution/glyph lanes, and a full DrawingManager.Flush integration run where a test op adds
// subimages at prepare (ASAP and inline lanes) under the deferred-upload token protocol, verified against the fake
// driver's glTexSubImage2D counters.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/text"
)

// testingUploadTarget is a DeferredUploadTarget with a writeable token tracker that records uploads without executing
// them.
type testingUploadTarget struct {
	t             *testing.T
	asapUploads   []DeferredTextureUploadFn
	inlineUploads []DeferredTextureUploadFn
	tracker       gpu.TokenTracker
	allowInline   bool
}

func (u *testingUploadTarget) TokenTracker() *gpu.TokenTracker { return &u.tracker }

func (u *testingUploadTarget) AddInlineUpload(upload DeferredTextureUploadFn) gpu.Token {
	if !u.allowInline {
		u.t.Fatal("unexpected inline upload")
	}
	u.inlineUploads = append(u.inlineUploads, upload)
	return u.tracker.NextDrawToken()
}

func (u *testingUploadTarget) AddASAPUpload(upload DeferredTextureUploadFn) gpu.Token {
	u.asapUploads = append(u.asapUploads, upload)
	return u.tracker.NextFlushToken()
}

// assertOnEvict is a PlotEvictionCallback that fails the test if any plot is ever evicted.
type assertOnEvict struct{ t *testing.T }

func (a *assertOnEvict) Evict(gpu.PlotLocator) { a.t.Error("unexpected plot eviction") }

// countingEvictor counts evictions and remembers the last locator.
type countingEvictor struct {
	count int
	last  gpu.PlotLocator
}

func (c *countingEvictor) Evict(loc gpu.PlotLocator) {
	c.count++
	c.last = loc
}

const (
	testNumPlots  = 2
	testPlotSize  = 32
	testAtlasSize = testNumPlots * testPlotSize
)

func checkAtlas(t *testing.T, atlas *DrawOpAtlas, expectedActive uint32, expectedAlloced int) {
	t.Helper()
	if atlas.NumActivePages() != expectedActive {
		t.Fatalf("numActivePages = %d, want %d", atlas.NumActivePages(), expectedActive)
	}
	if got := atlas.NumAllocatedForTest(); got != expectedAlloced {
		t.Fatalf("numAllocated = %d, want %d", got, expectedAlloced)
	}
	if atlas.MaxPages() != gpu.MaxMultitexturePages {
		t.Fatalf("maxPages = %d, want %d", atlas.MaxPages(), gpu.MaxMultitexturePages)
	}
}

func fillPlot(atlas *DrawOpAtlas, rp *ResourceProvider, target DeferredUploadTarget, locator *gpu.AtlasLocator, alpha byte) bool {
	data := make([]byte, testPlotSize*testPlotSize)
	for i := range data {
		data[i] = alpha
	}
	code := atlas.AddToAtlas(rp, target, testPlotSize, testPlotSize, data, locator)
	return code == AtlasErrorCodeSucceeded
}

func makeA8TestAtlas(t *testing.T, dc *DirectContext, counter *gpu.AtlasGenerationCounter, multitexture AllowMultitexturing, evictor gpu.PlotEvictionCallback) *DrawOpAtlas {
	t.Helper()
	format := dc.GLCaps().GetDefaultBackendFormat(gpu.ColorTypeAlpha8, false)
	if format == FormatUnknown {
		t.Fatal("no default A8 format")
	}
	atlas := MakeDrawOpAtlas(dc.ProxyProvider(), format, gpu.ColorTypeAlpha8, 1, testAtlasSize,
		testAtlasSize, testPlotSize, testPlotSize, counter, multitexture, evictor,
		"DrawOpAtlasTest")
	if atlas == nil {
		t.Fatal("MakeDrawOpAtlas failed")
	}
	return atlas
}

// TestBasicDrawOpAtlas verifies that a multitexture DrawOpAtlas correctly adds a page when the current one fills up and
// compacts unused pages away, simulating flush-time behavior.
func TestBasicDrawOpAtlas(t *testing.T) {
	dc := newFakeDirectContext(t)
	var counter gpu.AtlasGenerationCounter
	atlas := makeA8TestAtlas(t, dc, &counter, AllowMultitexturingYes, &assertOnEvict{t: t})
	checkAtlas(t, atlas, 0, 0)

	uploadTarget := &testingUploadTarget{t: t}
	rp := dc.ResourceProvider()

	// Fill up the first page.
	var atlasLocators [testNumPlots * testNumPlots]gpu.AtlasLocator
	for i := range testNumPlots * testNumPlots {
		if !fillPlot(atlas, rp, uploadTarget, &atlasLocators[i], byte(i*32)) {
			t.Fatalf("fill_plot %d failed", i)
		}
		checkAtlas(t, atlas, 1, 1)
	}

	atlas.Instantiate()
	checkAtlas(t, atlas, 1, 1)

	// Force allocation of a second page.
	var atlasLocator gpu.AtlasLocator
	if !fillPlot(atlas, rp, uploadTarget, &atlasLocator, 4*32) {
		t.Fatal("second-page fill_plot failed")
	}
	checkAtlas(t, atlas, 2, 2)

	// Simulate a lot of draws using only the first plot. The last texture should be compacted.
	for range 512 {
		atlas.SetLastUseToken(&atlasLocators[0], uploadTarget.tracker.NextDrawToken())
		uploadTarget.tracker.IssueDrawToken()
		uploadTarget.tracker.IssueFlushToken()
		atlas.Compact(uploadTarget.tracker.NextFlushToken())
	}
	checkAtlas(t, atlas, 1, 1)
}

// TestDrawOpAtlasConfigBasic exercises the atlas config's dimension table.
func TestDrawOpAtlasConfigBasic(t *testing.T) {
	testConfig := func(maxTextureSize int, maxBytes uint64, format gpu.MaskFormat,
		expectedDims, expectedPlotDims geom.ISize,
	) {
		t.Helper()
		config := MakeDrawOpAtlasConfig(maxTextureSize, maxBytes)
		if got := config.AtlasDimensions(format); got != expectedDims {
			t.Fatalf("atlasDimensions(maxBytes=%d, fmt=%d) = %v, want %v", maxBytes, format, got,
				expectedDims)
		}
		if got := config.PlotDimensions(format); got != expectedPlotDims {
			t.Fatalf("plotDimensions(maxBytes=%d, fmt=%d) = %v, want %v", maxBytes, format, got,
				expectedPlotDims)
		}
	}
	sz := func(w, h int32) geom.ISize { return geom.ISize{Width: w, Height: h} }
	const argb, a8 = gpu.MaskFormatARGB, gpu.MaskFormatA8
	// 1/4 MB.
	testConfig(65536, 256*1024, argb, sz(256, 256), sz(256, 256))
	testConfig(65536, 256*1024, a8, sz(512, 512), sz(256, 256))
	// 1/2 MB.
	testConfig(65536, 512*1024, argb, sz(512, 256), sz(256, 256))
	testConfig(65536, 512*1024, a8, sz(1024, 512), sz(256, 256))
	// 1 MB.
	testConfig(65536, 1024*1024, argb, sz(512, 512), sz(256, 256))
	testConfig(65536, 1024*1024, a8, sz(1024, 1024), sz(256, 256))
	// 2 MB.
	testConfig(65536, 2*1024*1024, argb, sz(1024, 512), sz(256, 256))
	testConfig(65536, 2*1024*1024, a8, sz(2048, 1024), sz(512, 256))
	// 4 MB.
	testConfig(65536, 4*1024*1024, argb, sz(1024, 1024), sz(256, 256))
	testConfig(65536, 4*1024*1024, a8, sz(2048, 2048), sz(512, 512))
	// 8 MB.
	testConfig(65536, 8*1024*1024, argb, sz(2048, 1024), sz(256, 256))
	testConfig(65536, 8*1024*1024, a8, sz(2048, 2048), sz(512, 512))
	// 16 MB (should be same as 8 MB).
	testConfig(65536, 16*1024*1024, argb, sz(2048, 1024), sz(256, 256))
	testConfig(65536, 16*1024*1024, a8, sz(2048, 2048), sz(512, 512))
	// 4 MB, restricted texture size.
	testConfig(1024, 8*1024*1024, argb, sz(1024, 1024), sz(256, 256))
	testConfig(1024, 8*1024*1024, a8, sz(1024, 1024), sz(256, 256))
	// 3 MB (should be same as 2 MB).
	testConfig(65536, 3*1024*1024, argb, sz(1024, 512), sz(256, 256))
	testConfig(65536, 3*1024*1024, a8, sz(2048, 1024), sz(512, 256))
	// Minimum size.
	testConfig(65536, 0, argb, sz(256, 256), sz(256, 256))
	testConfig(65536, 0, a8, sz(512, 512), sz(256, 256))
}

// TestDrawOpAtlasInlineUpload exercises the kTryAgain → inline-upload protocol: when every plot is used by the draw
// being prepared, addToAtlas fails with kTryAgain; once the draw token advances, the LRU plot is evicted, cloned, and
// its upload scheduled inline.
func TestDrawOpAtlasInlineUpload(t *testing.T) {
	dc := newFakeDirectContext(t)
	var counter gpu.AtlasGenerationCounter
	evictor := &countingEvictor{}
	atlas := makeA8TestAtlas(t, dc, &counter, AllowMultitexturingNo, evictor)
	if atlas.MaxPages() != 1 {
		t.Fatalf("maxPages = %d, want 1", atlas.MaxPages())
	}
	uploadTarget := &testingUploadTarget{t: t, allowInline: true}
	rp := dc.ResourceProvider()
	startGen := atlas.AtlasGeneration()

	// Fill all four plots and mark them read by the draw being prepared.
	var locators [testNumPlots * testNumPlots]gpu.AtlasLocator
	for i := range locators {
		if !fillPlot(atlas, rp, uploadTarget, &locators[i], byte(i)) {
			t.Fatalf("fill_plot %d failed", i)
		}
		atlas.SetLastUseToken(&locators[i], uploadTarget.tracker.NextDrawToken())
	}
	if len(uploadTarget.asapUploads) != 4 || len(uploadTarget.inlineUploads) != 0 {
		t.Fatalf("uploads = %d asap / %d inline, want 4/0", len(uploadTarget.asapUploads),
			len(uploadTarget.inlineUploads))
	}

	// Full atlas, every plot used by the pending draw: kTryAgain.
	var loc gpu.AtlasLocator
	if code := atlas.AddToAtlas(rp, uploadTarget, testPlotSize, testPlotSize,
		make([]byte, testPlotSize*testPlotSize), &loc); code != AtlasErrorCodeTryAgain {
		t.Fatalf("AddToAtlas = %d, want kTryAgain", code)
	}
	if evictor.count != 0 {
		t.Fatal("kTryAgain must not evict")
	}

	// The op enqueues its draw; the retry now performs an inline upload into the LRU plot.
	uploadTarget.tracker.IssueDrawToken()
	if code := atlas.AddToAtlas(rp, uploadTarget, testPlotSize, testPlotSize,
		make([]byte, testPlotSize*testPlotSize), &loc); code != AtlasErrorCodeSucceeded {
		t.Fatalf("AddToAtlas after draw = %d, want kSucceeded", code)
	}
	if len(uploadTarget.inlineUploads) != 1 {
		t.Fatalf("inline uploads = %d, want 1", len(uploadTarget.inlineUploads))
	}
	if evictor.count != 1 {
		t.Fatalf("evictions = %d, want 1", evictor.count)
	}
	// The LRU plot was the first filled one; its old locator is stale, the others are not, and the atlas generation
	// advanced.
	if evictor.last != locators[0].PlotLocator() {
		t.Fatal("evicted locator mismatch")
	}
	if atlas.HasID(locators[0].PlotLocator()) {
		t.Fatal("evicted plot should have a stale ID")
	}
	if !atlas.HasID(locators[1].PlotLocator()) || !atlas.HasID(loc.PlotLocator()) {
		t.Fatal("live plots should keep their IDs")
	}
	if atlas.AtlasGeneration() == startGen {
		t.Fatal("atlas generation should advance on eviction")
	}
}

// TestAtlasManagerInit covers the manager's lazy atlas creation, the config plumbing, the A565 resolution rule, and the
// free/retain semantics.
func TestAtlasManagerInit(t *testing.T) {
	dc := newFakeDirectContext(t)
	am := dc.AtlasManager()
	if am == nil {
		t.Fatal("context should create the atlas manager")
	}
	am.SetAtlasDimensionsToMinimumForTest()

	views, numActive := am.GetViews(gpu.MaskFormatA8)
	if views == nil {
		t.Fatal("GetViews failed for A8")
	}
	if numActive != 0 {
		t.Fatalf("numActive = %d, want 0 (proxies available, none active)", numActive)
	}
	// Minimum config: ARGB 256x256 single plot, A8 512x512 with 256x256 plots.
	a8 := am.AtlasForTest(gpu.MaskFormatA8)
	if got := a8.Views()[0].Proxy().Dimensions(); got != (geom.ISize{Width: 512, Height: 512}) {
		t.Fatalf("A8 atlas dims = %v", got)
	}
	argb := am.AtlasForTest(gpu.MaskFormatARGB)
	if got := argb.Views()[0].Proxy().Dimensions(); got != (geom.ISize{Width: 256, Height: 256}) {
		t.Fatalf("ARGB atlas dims = %v", got)
	}

	// A565 resolves per the caps: with a valid 565 format it gets its own atlas, otherwise ARGB.
	got565 := am.AtlasForTest(gpu.MaskFormatA565)
	if dc.GLCaps().GetDefaultBackendFormat(gpu.ColorTypeBGR565, false) != FormatUnknown {
		if got565 == argb {
			t.Fatal("565 should have its own atlas when supported")
		}
	} else if got565 != argb {
		t.Fatal("565 should resolve to the ARGB atlas when unsupported")
	}

	// The atlas manager survives FreeGpuResources (retainOnFreeGpuResources) but drops its atlases.
	dc.FreeGpuResources()
	if am.AtlasForTest(gpu.MaskFormatA8) == a8 {
		t.Fatal("FreeGpuResources should drop the atlases")
	}
	found := false
	for _, cb := range dc.DrawingManager().onFlushCBObjects {
		if cb == OnFlushCallbackObject(am) {
			found = true
		}
	}
	if !found {
		t.Fatal("atlas manager should be retained in the on-flush callback list")
	}
}

// TestAtlasManagerAddGlyph covers AddGlyphToAtlas's A8 and ARGB lanes, the transformed-mask padding, and the locator
// inset.
func TestAtlasManagerAddGlyph(t *testing.T) {
	dc := newFakeDirectContext(t)
	am := dc.AtlasManager()
	am.SetAtlasDimensionsToMinimumForTest()
	uploadTarget := &testingUploadTarget{t: t}
	rp := dc.ResourceProvider()

	// Per the getViews contract, it must be called before any atlas-using function.
	if views, _ := am.GetViews(gpu.MaskFormatA8); views == nil {
		t.Fatal("GetViews(A8) failed")
	}
	if views, _ := am.GetViews(gpu.MaskFormatARGB); views == nil {
		t.Fatal("GetViews(ARGB) failed")
	}

	// A8 direct mask (srcPadding 0): stored at its exact size.
	skGlyph := &font.Glyph{
		Width:  3,
		Height: 2,
		Format: font.MaskA8,
		Image:  []uint8{1, 2, 3, 4, 5, 6},
	}
	glyph := text.NewGlyph(0)
	if code := am.AddGlyphToAtlas(skGlyph, glyph, 0, rp, uploadTarget); code != AtlasErrorCodeSucceeded {
		t.Fatalf("AddGlyphToAtlas(A8) = %d", code)
	}
	if glyph.AtlasLocator.Width() != 3 || glyph.AtlasLocator.Height() != 2 {
		t.Fatalf("A8 locator size = %dx%d, want 3x2", glyph.AtlasLocator.Width(),
			glyph.AtlasLocator.Height())
	}
	if !am.HasGlyph(gpu.MaskFormatA8, glyph) {
		t.Fatal("glyph should be present after add")
	}
	// The mask bytes landed in the plot's backing store at the locator.
	a8 := am.AtlasForTest(gpu.MaskFormatA8)
	plot := a8.pages[glyph.AtlasLocator.PageIndex()].plotArray[glyph.AtlasLocator.PlotIndex()]
	data := plot.DataAt(&glyph.AtlasLocator)
	plotRB := 256 // min-config A8 plots are 256x256 at 1 bpp
	for y := range 2 {
		for x := range 3 {
			if data[y*plotRB+x] != skGlyph.Image[y*3+x] {
				t.Fatalf("A8 texel (%d,%d) = %d", x, y, data[y*plotRB+x])
			}
		}
	}

	// Transformed mask (srcPadding 1): stored padded, locator inset back to the glyph bounds.
	glyphPadded := text.NewGlyph(0)
	if code := am.AddGlyphToAtlas(skGlyph, glyphPadded, 1, rp, uploadTarget); code != AtlasErrorCodeSucceeded {
		t.Fatalf("AddGlyphToAtlas(padded) = %d", code)
	}
	if glyphPadded.AtlasLocator.Width() != 3 || glyphPadded.AtlasLocator.Height() != 2 {
		t.Fatalf("padded locator size = %dx%d, want 3x2 after inset",
			glyphPadded.AtlasLocator.Width(), glyphPadded.AtlasLocator.Height())
	}
	dataPadded := plot.DataAt(&glyphPadded.AtlasLocator)
	if dataPadded[0] != 1 {
		t.Fatal("inset locator should address the glyph origin, not the pad")
	}
	// The padding ring around the glyph is zero: address the full padded rect (one texel out).
	tl := glyphPadded.AtlasLocator.TopLeft()
	var padLoc gpu.AtlasLocator
	padLoc.UpdateRect(gpu.IRect16MakeXYWH(int16(tl.X-1), int16(tl.Y-1), 5, 4))
	padData := plot.DataAt(&padLoc)
	if padData[0] != 0 || padData[1] != 0 || padData[plotRB] != 0 {
		t.Fatal("padding ring should be zero")
	}
	if padData[plotRB+1] != 1 {
		t.Fatal("glyph origin should sit inside the padding ring")
	}

	// ARGB color glyph: device words land byte-for-byte.
	colorGlyph := &font.Glyph{
		Width:   2,
		Height:  1,
		Format:  font.MaskARGB32,
		Image32: []uint32{0x11223344, 0x55667788},
	}
	gpuGlyph := text.NewGlyph(0)
	if code := am.AddGlyphToAtlas(colorGlyph, gpuGlyph, 0, rp, uploadTarget); code != AtlasErrorCodeSucceeded {
		t.Fatalf("AddGlyphToAtlas(ARGB) = %d", code)
	}
	argb := am.AtlasForTest(gpu.MaskFormatARGB)
	cplot := argb.pages[gpuGlyph.AtlasLocator.PageIndex()].plotArray[gpuGlyph.AtlasLocator.PlotIndex()]
	cdata := cplot.DataAt(&gpuGlyph.AtlasLocator)
	if cdata[0] != 0x44 || cdata[1] != 0x33 || cdata[2] != 0x22 || cdata[3] != 0x11 ||
		cdata[4] != 0x88 {
		t.Fatalf("ARGB texels = % x", cdata[:8])
	}

	// A glyph with no image fails.
	if code := am.AddGlyphToAtlas(&font.Glyph{Width: 1, Height: 1, Format: font.MaskA8},
		text.NewGlyph(0), 0, rp, uploadTarget); code != AtlasErrorCodeError {
		t.Fatalf("imageless glyph = %d, want kError", code)
	}
	// Unknown padding fails.
	if code := am.AddGlyphToAtlas(skGlyph, text.NewGlyph(0), 3, rp, uploadTarget); code != AtlasErrorCodeError {
		t.Fatalf("bad padding = %d, want kError", code)
	}

	// AddGlyphToBulkAndSetUseToken dedups through the updater.
	var updater gpu.BulkUsePlotUpdater
	am.AddGlyphToBulkAndSetUseToken(&updater, gpu.MaskFormatA8, glyph,
		uploadTarget.tracker.NextDrawToken())
	am.AddGlyphToBulkAndSetUseToken(&updater, gpu.MaskFormatA8, glyphPadded,
		uploadTarget.tracker.NextDrawToken())
	if updater.Count() != 1 {
		t.Fatalf("bulk updater count = %d, want 1 (same plot)", updater.Count())
	}
	am.SetUseTokenBulk(&updater, uploadTarget.tracker.NextDrawToken(), gpu.MaskFormatA8)
}

// TestAtlasFlushIntegration drives the atlas through a real DrawingManager.Flush on the fake driver: flush 1 adds one
// subimage (ASAP lane, page activation); flush 2 adds two subimages to the now-full single-plot atlas (eviction/ASAP
// lane + inline lane), checking the exact glTexSubImage2D counts attributable to atlas uploads and the postFlush
// compaction bookkeeping.
func TestAtlasFlushIntegration(t *testing.T) {
	dc := newFakeDirectContext(t)
	am := dc.AtlasManager()
	am.SetAtlasDimensionsToMinimumForTest()
	// The minimum ARGB atlas is a single 256x256 plot; capped to one page (as atlas manager test-only page cap) it is
	// the easiest way to force the inline lane.
	const format = gpu.MaskFormatARGB
	const size = 256
	if am.AtlasForTest(format) == nil {
		t.Fatal("atlas init failed")
	}
	am.SetMaxPagesForTest(1)

	sdc := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 64, Height: 64},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"atlas-flush-test")
	if sdc == nil {
		t.Fatal("MakeSurfaceDrawContext failed")
	}
	defer sdc.Release()

	// Flush 1: a single add rides the ASAP lane while activating page 0.
	sdc.Clear([4]float32{0, 0, 0, 1})
	op1 := NewAtlasUploadTestOp(dc, format, size, 1)
	sdc.GetOpsTask().AddOp(dc.DrawingManager(), op1, dc.GLCaps())
	dc.Flush(FlushInfo{})
	if len(op1.Codes) != 1 || op1.Codes[0] != AtlasErrorCodeSucceeded {
		t.Fatalf("flush-1 codes = %v", op1.Codes)
	}
	if op1.Executed != 1 {
		t.Fatalf("flush-1 executed = %d, want 1", op1.Executed)
	}
	atlas := am.AtlasForTest(format)
	if atlas.NumActivePages() != 1 {
		t.Fatalf("active pages = %d, want 1", atlas.NumActivePages())
	}
	// postFlush ran: the atlas's compaction watermark advanced to the next flush token.
	if atlas.prevFlushToken != dc.DrawingManager().TokenTracker().NextFlushToken() {
		t.Fatal("postFlush should have compacted the atlas")
	}

	// Flush 2: the plot is full and was used last flush. The first add evicts and re-uses it (ASAP); the second add
	// inside the same prepare hits the pending-draw hazard and goes inline. Exactly two glTexSubImage2D calls are
	// attributable to this flush's uploads (the page was instantiated in flush 1, so no allocation noise remains).
	texSubBefore := recCounts["glTexSubImage2D"]
	sdc.Clear([4]float32{0, 0, 0, 1})
	op2 := NewAtlasUploadTestOp(dc, format, size, 2)
	sdc.GetOpsTask().AddOp(dc.DrawingManager(), op2, dc.GLCaps())
	dc.Flush(FlushInfo{})
	if len(op2.Codes) != 2 || op2.Codes[0] != AtlasErrorCodeSucceeded ||
		op2.Codes[1] != AtlasErrorCodeSucceeded {
		t.Fatalf("flush-2 codes = %v", op2.Codes)
	}
	if op2.Executed != 2 {
		t.Fatalf("flush-2 executed = %d, want 2", op2.Executed)
	}
	if delta := recCounts["glTexSubImage2D"] - texSubBefore; delta != 2 {
		t.Fatalf("glTexSubImage2D delta = %d, want 2 (one ASAP + one inline)", delta)
	}
	// The single page never grew, so the second upload can only have gone inline: its plot was cloned, staling the
	// first locator.
	if atlas.NumActivePages() != 1 {
		t.Fatalf("active pages = %d, want 1", atlas.NumActivePages())
	}
	if atlas.HasID(op2.Locators[0].PlotLocator()) {
		t.Fatal("the inline-displaced plot should have a stale ID")
	}
	if !atlas.HasID(op2.Locators[1].PlotLocator()) {
		t.Fatal("the inline-uploaded plot should have a live ID")
	}
	// The token streams balanced (Flush would have panicked otherwise); the drawn tokens advanced by exactly the three
	// recorded draws.
	tracker := dc.DrawingManager().TokenTracker()
	if tracker.NextDrawToken() != tracker.NextFlushToken() {
		t.Fatal("token streams should balance after flush")
	}
	if tracker.CurrentFlushToken().Value() != 3 {
		t.Fatalf("flush token = %d, want 3", tracker.CurrentFlushToken().Value())
	}
}
