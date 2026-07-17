// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for the atlas support types: the skyline rectanizer (packing invariants and the lowest-then-narrowest
// placement rule), the PlotLocator/AtlasLocator UV encoding (page bits in bits 13-14 of the U coordinates, insetSrc,
// width/height as UV differences), BulkUsePlotUpdater dedup, Plot behavior (dirty-rect join, subimage copy,
// prepareForUpload's 4-byte clamp and atlas-space offset, resetRects generation bumps, clone), the PlotList LRU
// operations, and the Token/TokenTracker sequencing rules.

package gpu

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestRectanizerSkylineBasic(t *testing.T) {
	r := MakeRectanizerSkyline(64, 64)
	if r.PercentFull() != 0 {
		t.Fatal("new rectanizer should be empty")
	}

	// First rect lands at the origin.
	x, y, ok := r.AddRect(32, 16)
	if !ok || x != 0 || y != 0 {
		t.Fatalf("first rect at (%d,%d) ok=%v, want (0,0) true", x, y, ok)
	}
	// The second rect prefers the lowest y: beside the first, not above it.
	x, y, ok = r.AddRect(32, 16)
	if !ok || x != 32 || y != 0 {
		t.Fatalf("second rect at (%d,%d) ok=%v, want (32,0) true", x, y, ok)
	}
	// Now the full width at y=0 is used; next goes to y=16.
	x, y, ok = r.AddRect(64, 16)
	if !ok || x != 0 || y != 16 {
		t.Fatalf("third rect at (%d,%d) ok=%v, want (0,16) true", x, y, ok)
	}
	wantFull := float32(32*16+32*16+64*16) / float32(64*64)
	if r.PercentFull() != wantFull {
		t.Fatalf("percentFull = %v, want %v", r.PercentFull(), wantFull)
	}

	// Too-large rects fail without changing state.
	if _, _, ok = r.AddRect(65, 1); ok {
		t.Fatal("wider than the rectanizer should fail")
	}
	if _, _, ok = r.AddRect(1, 65); ok {
		t.Fatal("taller than the rectanizer should fail")
	}

	// Fill the rest: two more 64x16 rows fit, then nothing 16-tall does.
	if _, y, ok = r.AddRect(64, 16); !ok || y != 32 {
		t.Fatalf("fourth row misplaced: y=%d ok=%v", y, ok)
	}
	if _, y, ok = r.AddRect(64, 16); !ok || y != 48 {
		t.Fatalf("fifth row misplaced: y=%d ok=%v", y, ok)
	}
	if _, _, ok = r.AddRect(1, 16); ok {
		t.Fatal("full rectanizer accepted a rect")
	}
	if r.PercentFull() != 1 {
		t.Fatalf("percentFull = %v, want 1", r.PercentFull())
	}

	r.Reset()
	if r.PercentFull() != 0 {
		t.Fatal("reset should empty the rectanizer")
	}
	if x, y, ok = r.AddRect(64, 64); !ok || x != 0 || y != 0 {
		t.Fatal("full-size rect should fit after reset")
	}
}

// TestRectanizerSkylineNarrowest checks the tie-break: among equal-y candidates the narrowest skyline segment wins.
func TestRectanizerSkylineNarrowest(t *testing.T) {
	r := MakeRectanizerSkyline(64, 64)
	// Build a skyline with segments: [0,16) h=32, [16,48) h=16, [48,64) h=32.
	if _, _, ok := r.AddRect(64, 16); !ok {
		t.Fatal("base row failed")
	}
	if x, y, ok := r.AddRect(16, 16); !ok || x != 0 || y != 16 {
		t.Fatalf("left block at (%d,%d)", x, y)
	}
	if x, y, ok := r.AddRect(16, 16); !ok || x != 16 || y != 16 {
		t.Fatalf("mid block at (%d,%d)", x, y)
	}
	// Consume the left block up to 32 so the left segment is height 32.
	if x, y, ok := r.AddRect(16, 16); !ok || x != 32 || y != 16 {
		t.Fatalf("right-mid block at (%d,%d)", x, y)
	}
	// Skyline is now [0,48) h=32, [48,64) h=16. A 8x8 rect fits at y=32 on both; the placement picks the lowest y
	// first: 32 at x=48 belongs to the h=16 segment (y=16).
	if x, y, ok := r.AddRect(8, 8); !ok || x != 48 || y != 16 {
		t.Fatalf("tie-break rect at (%d,%d), want (48,16)", x, y)
	}
}

func TestPlotLocator(t *testing.T) {
	var invalid PlotLocator
	if invalid.IsValid() {
		t.Fatal("zero locator should be invalid")
	}
	loc := MakePlotLocator(2, 7, 12345)
	if !loc.IsValid() || loc.PageIndex() != 2 || loc.PlotIndex() != 7 || loc.GenID() != 12345 {
		t.Fatalf("locator fields wrong: %+v", loc)
	}
	loc.MakeInvalid()
	if loc.IsValid() {
		t.Fatal("makeInvalid should invalidate")
	}
	// Generation 0 with nonzero indices is still valid (validity ORs the three fields).
	if !(PlotLocator{plotIndex: 1}).IsValid() {
		t.Fatal("nonzero plot index should be valid")
	}
}

func TestAtlasLocatorEncoding(t *testing.T) {
	var al AtlasLocator
	al.UpdateRect(IRect16MakeXYWH(100, 200, 30, 40))
	if got := al.TopLeft(); got.X != 100 || got.Y != 200 {
		t.Fatalf("topLeft = %v", got)
	}
	if al.Width() != 30 || al.Height() != 40 {
		t.Fatalf("size = %dx%d", al.Width(), al.Height())
	}

	// The page index rides bits 13-14 of the U coordinates and must not disturb the positions.
	al.UpdatePlotLocator(MakePlotLocator(3, 5, 99))
	uvs := al.UVs()
	if uvs[0]>>13 != 3 || uvs[2]>>13 != 3 {
		t.Fatalf("page bits missing from UVs: %v", uvs)
	}
	if got := al.TopLeft(); got.X != 100 || got.Y != 200 {
		t.Fatalf("topLeft after page encode = %v", got)
	}
	// Width still subtracts cleanly (the page bits cancel).
	if al.Width() != 30 || al.Height() != 40 {
		t.Fatalf("size after page encode = %dx%d", al.Width(), al.Height())
	}
	if al.PageIndex() != 3 || al.PlotIndex() != 5 || al.GenID() != 99 {
		t.Fatalf("plot locator fields wrong")
	}

	// insetSrc shrinks all four edges.
	al.InsetSrc(1)
	if got := al.TopLeft(); got.X != 101 || got.Y != 201 {
		t.Fatalf("topLeft after inset = %v", got)
	}
	if al.Width() != 28 || al.Height() != 38 {
		t.Fatalf("size after inset = %dx%d", al.Width(), al.Height())
	}

	// Updating the rect preserves the page bits.
	al.UpdateRect(IRect16MakeXYWH(5, 6, 7, 8))
	uvs = al.UVs()
	if uvs[0] != 5|3<<13 || uvs[1] != 6 || uvs[2] != 12|3<<13 || uvs[3] != 14 {
		t.Fatalf("UVs after re-rect = %v", uvs)
	}
}

func TestBulkUsePlotUpdater(t *testing.T) {
	var updater BulkUsePlotUpdater
	var al1, al2 AtlasLocator
	al1.UpdatePlotLocator(MakePlotLocator(0, 3, 1))
	al2.UpdatePlotLocator(MakePlotLocator(1, 3, 2))

	if !updater.Add(&al1) {
		t.Fatal("first add should succeed")
	}
	if updater.Add(&al1) {
		t.Fatal("duplicate add should fail")
	}
	if !updater.Add(&al2) {
		t.Fatal("same plot on another page should succeed")
	}
	if updater.Count() != 2 {
		t.Fatalf("count = %d", updater.Count())
	}
	if pd := updater.PlotData(0); pd.PageIndex != 0 || pd.PlotIndex != 3 {
		t.Fatalf("plotData(0) = %+v", pd)
	}
	updater.Reset()
	if updater.Count() != 0 || !updater.Add(&al1) {
		t.Fatal("reset should clear the dedup state")
	}
}

func TestPlotAddAndUpload(t *testing.T) {
	var counter AtlasGenerationCounter
	// Page 1, plot 2, at plot-grid position (1, 1) in a 32x32-plot atlas.
	plot := NewPlot(1, 2, &counter, 1, 1, 32, 32, ColorTypeAlpha8, 1)
	if plot.GenID() != 1 {
		t.Fatalf("first genID = %d", plot.GenID())
	}
	if plot.NeedsUpload() || !plot.IsEmpty() || plot.HasAllocation() {
		t.Fatal("fresh plot state wrong")
	}

	img := make([]byte, 6*4)
	for i := range img {
		img[i] = byte(i + 1)
	}
	var al AtlasLocator
	if !plot.AddSubImage(6, 4, img, &al) {
		t.Fatal("addSubImage failed")
	}
	// The locator is in atlas space: offset by (32, 32) for plot-grid (1,1).
	if got := al.TopLeft(); got.X != 32 || got.Y != 32 {
		t.Fatalf("locator topLeft = %v, want (32,32)", got)
	}
	if !plot.NeedsUpload() || plot.IsEmpty() || !plot.HasAllocation() {
		t.Fatal("post-add plot state wrong")
	}

	// The data landed at the plot-local origin with plot-stride rows.
	data := plot.DataAt(&al)
	for y := range 4 {
		for x := range 6 {
			if data[y*32+x] != img[y*6+x] {
				t.Fatalf("pixel (%d,%d) = %d, want %d", x, y, data[y*32+x], img[y*6+x])
			}
		}
	}

	// prepareForUpload: dirty rect [0,6)x[0,4) clamps the right edge to a 4-byte boundary (8).
	upload, rect := plot.PrepareForUpload()
	if rect != geom.IRectLTRB(32, 32, 40, 36) {
		t.Fatalf("upload rect = %v, want atlas-space [32,32,40,36]", rect)
	}
	if upload[0] != 1 {
		t.Fatal("upload data should start at the dirty origin")
	}
	if plot.NeedsUpload() {
		t.Fatal("prepareForUpload should clear the dirty rect")
	}

	// A second add dirties only the new area: the 4x4 lands at plot-local (6,0), and the 4-byte clamp widens [6,10) to
	// [4,12).
	if !plot.AddSubImage(4, 4, img[:16], &al) {
		t.Fatal("second addSubImage failed")
	}
	_, rect = plot.PrepareForUpload()
	if rect != geom.IRectLTRB(36, 32, 44, 36) {
		t.Fatalf("second upload rect = %v, want [36,32,44,36]", rect)
	}

	// markFullIfUsed only latches when dirty.
	plot.MarkFullIfUsed()
	if !plot.AddRect(2, 2, &al) {
		t.Fatal("clean plot should not be marked full")
	}
	plot.MarkFullIfUsed()
	var al2 AtlasLocator
	if plot.AddSubImage(2, 2, img[:4], &al2) {
		t.Fatal("full plot should reject addSubImage")
	}

	// resetRects bumps the generation and clears everything but keeps the allocation.
	plot.SetLastUseToken(Token{sequenceNumber: 7})
	plot.ResetRects(false)
	if plot.GenID() == 1 || !plot.IsEmpty() || plot.NeedsUpload() {
		t.Fatal("resetRects state wrong")
	}
	if plot.LastUseToken() != InvalidToken() || plot.LastUploadToken() != InvalidToken() {
		t.Fatal("resetRects should clear tokens")
	}
	if !plot.HasAllocation() {
		t.Fatal("resetRects(false) should keep the allocation")
	}
	if plot.DataAt(&al)[0] != 0 {
		t.Fatal("resetRects should zero the data")
	}
	plot.ResetRects(true)
	if plot.HasAllocation() {
		t.Fatal("resetRects(true) should free the data")
	}

	// Clone matches identity but takes a fresh generation.
	clone := plot.Clone()
	if clone.PageIndex() != plot.PageIndex() || clone.PlotIndex() != plot.PlotIndex() {
		t.Fatal("clone identity wrong")
	}
	if clone.GenID() == plot.GenID() {
		t.Fatal("clone should advance the generation")
	}
}

func TestPlotList(t *testing.T) {
	var counter AtlasGenerationCounter
	var list PlotList
	p0 := NewPlot(0, 0, &counter, 0, 0, 16, 16, ColorTypeAlpha8, 1)
	p1 := NewPlot(0, 1, &counter, 1, 0, 16, 16, ColorTypeAlpha8, 1)
	p2 := NewPlot(0, 2, &counter, 0, 1, 16, 16, ColorTypeAlpha8, 1)

	list.AddToHead(p0)
	list.AddToHead(p1)
	list.AddToHead(p2)
	if list.Head() != p2 || list.Tail() != p0 {
		t.Fatal("head/tail wrong after adds")
	}

	// MRU move: remove + addToHead.
	list.Remove(p0)
	list.AddToHead(p0)
	if list.Head() != p0 || list.Tail() != p1 {
		t.Fatal("head/tail wrong after MRU move")
	}

	// Iteration order is MRU→LRU.
	var order []uint32
	for p := list.Head(); p != nil; p = p.Next() {
		order = append(order, p.PlotIndex())
	}
	if len(order) != 3 || order[0] != 0 || order[1] != 2 || order[2] != 1 {
		t.Fatalf("iteration order = %v", order)
	}

	list.Reset()
	if list.Head() != nil || list.Tail() != nil {
		t.Fatal("reset should empty the list")
	}
	// Plots are re-addable after reset.
	list.AddToHead(p1)
	if list.Head() != p1 || list.Tail() != p1 {
		t.Fatal("re-add after reset failed")
	}
}

func TestTokenTracker(t *testing.T) {
	var tracker TokenTracker
	if tracker.NextDrawToken() != tracker.NextFlushToken() {
		t.Fatal("fresh tracker should be balanced")
	}
	if tracker.CurrentFlushToken() != InvalidToken() {
		t.Fatal("fresh tracker's current flush token should be invalid")
	}

	d1 := tracker.IssueDrawToken()
	if d1 != (Token{sequenceNumber: 1}) || tracker.NextDrawToken() != (Token{sequenceNumber: 2}) {
		t.Fatal("draw token sequencing wrong")
	}
	// Flush tokens run independently.
	if tracker.NextFlushToken() != (Token{sequenceNumber: 1}) {
		t.Fatal("flush token should not advance with draws")
	}
	f1 := tracker.IssueFlushToken()
	if f1 != d1 || tracker.CurrentFlushToken() != f1 {
		t.Fatal("flush token sequencing wrong")
	}

	// Interval semantics are inclusive.
	if !d1.InInterval(InvalidToken(), f1) || d1.InInterval(f1.Next(), f1.Next().Next()) {
		t.Fatal("inInterval wrong")
	}
	if !d1.LessThan(d1.Next()) || d1.LessThan(d1) || !d1.LessThanOrEqual(d1) {
		t.Fatal("comparison wrong")
	}
}
