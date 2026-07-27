// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package textblob

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

func loadFont(t *testing.T, size float32) *font.Font {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return font.NewFont(tf, size, 1, 0)
}

func glyphsFor(f *font.Font, text string) []uint16 {
	glyphs := make([]uint16, len(text))
	n := f.TextToGlyphs([]byte(text), font.TextEncodingUTF8, glyphs)
	return glyphs[:n]
}

func TestBuilderEmpty(t *testing.T) {
	b := NewBuilder()
	if blob := b.Make(); blob != nil {
		t.Fatal("empty builder must make nil")
	}
	if buf := b.AllocRun(loadFont(t, 12), 0, 0, 0, nil); buf.Glyphs != nil {
		t.Fatal("zero count alloc must reject")
	}
}

func TestBuilderDefaultRunBounds(t *testing.T) {
	f := loadFont(t, 24)
	glyphs := glyphsFor(f, "Hello")

	b := NewBuilder()
	buf := b.AllocRun(f, len(glyphs), 10, 20, nil)
	copy(buf.Glyphs, glyphs)
	blob := b.Make()
	if blob == nil {
		t.Fatal("blob should exist")
	}

	// Default positioning uses TightRunBounds: measureText's bounds offset by the run offset.
	var want geom.Rect
	f.MeasureText(glyphBytes(glyphs), font.TextEncodingGlyphID, &want, nil)
	want = want.Offset(10, 20)
	if blob.Bounds() != want {
		t.Errorf("bounds %v, want %v", blob.Bounds(), want)
	}
}

func TestBuilderPosRunsConservativeBounds(t *testing.T) {
	f := loadFont(t, 24)
	glyphs := glyphsFor(f, "abc")

	b := NewBuilder()
	buf := b.AllocRunPos(f, len(glyphs), nil)
	copy(buf.Glyphs, glyphs)
	pos := []float32{5, 50, 25, 60, 45, 40}
	copy(buf.Pos, pos)
	blob := b.Make()

	fontBounds := font.GetFontBounds(f)
	want := geom.RectLTRB(5+fontBounds.Left, 40+fontBounds.Top, 45+fontBounds.Right, 60+fontBounds.Bottom)
	if blob.Bounds() != want {
		t.Errorf("bounds %v, want %v", blob.Bounds(), want)
	}

	// Horizontal positioning: x extremes plus font bounds, at the common y.
	b2 := NewBuilder()
	buf2 := b2.AllocRunPosH(f, len(glyphs), 30, nil)
	copy(buf2.Glyphs, glyphs)
	copy(buf2.Pos, []float32{12, 2, 40})
	blob2 := b2.Make()
	want2 := geom.RectLTRB(2+fontBounds.Left, 30+fontBounds.Top, 40+fontBounds.Right, 30+fontBounds.Bottom)
	if blob2.Bounds() != want2 {
		t.Errorf("posH bounds %v, want %v", blob2.Bounds(), want2)
	}
}

func TestBuilderExplicitBounds(t *testing.T) {
	f := loadFont(t, 24)
	glyphs := glyphsFor(f, "x")
	explicit := geom.RectLTRB(1, 2, 3, 4)
	b := NewBuilder()
	buf := b.AllocRun(f, 1, 0, 0, &explicit)
	copy(buf.Glyphs, glyphs)
	blob := b.Make()
	if blob.Bounds() != explicit {
		t.Errorf("explicit bounds %v, want %v", blob.Bounds(), explicit)
	}
}

func TestBuilderMergeRuns(t *testing.T) {
	f := loadFont(t, 24)
	glyphs := glyphsFor(f, "ab")

	// Two full-positioned runs with the same font merge into one.
	b := NewBuilder()
	buf := b.AllocRunPos(f, 2, nil)
	copy(buf.Glyphs, glyphs)
	copy(buf.Pos, []float32{0, 0, 10, 0})
	buf2 := b.AllocRunPos(f, 2, nil)
	copy(buf2.Glyphs, glyphs)
	copy(buf2.Pos, []float32{20, 0, 30, 0})
	blob := b.Make()
	if len(blob.runs) != 1 {
		t.Errorf("full-pos runs should merge: %d runs", len(blob.runs))
	}
	if len(blob.runs[0].glyphs) != 4 {
		t.Errorf("merged run has %d glyphs", len(blob.runs[0].glyphs))
	}

	// Default positioning never merges.
	b2 := NewBuilder()
	r1 := b2.AllocRun(f, 2, 0, 0, nil)
	copy(r1.Glyphs, glyphs)
	r2 := b2.AllocRun(f, 2, 0, 30, nil)
	copy(r2.Glyphs, glyphs)
	blob2 := b2.Make()
	if len(blob2.runs) != 2 {
		t.Errorf("default runs should not merge: %d runs", len(blob2.runs))
	}

	// Horizontal runs merge only at the same y.
	b3 := NewBuilder()
	h1 := b3.AllocRunPosH(f, 2, 10, nil)
	copy(h1.Glyphs, glyphs)
	copy(h1.Pos, []float32{0, 10})
	h2 := b3.AllocRunPosH(f, 2, 10, nil)
	copy(h2.Glyphs, glyphs)
	copy(h2.Pos, []float32{20, 30})
	h3 := b3.AllocRunPosH(f, 2, 40, nil)
	copy(h3.Glyphs, glyphs)
	copy(h3.Pos, []float32{0, 10})
	blob3 := b3.Make()
	if len(blob3.runs) != 2 {
		t.Errorf("posH runs: %d, want 2 (same-y merged, different-y separate)", len(blob3.runs))
	}

	// A different font blocks merging.
	f2 := loadFont(t, 30)
	b4 := NewBuilder()
	m1 := b4.AllocRunPos(f, 2, nil)
	copy(m1.Glyphs, glyphs)
	m2 := b4.AllocRunPos(f2, 2, nil)
	copy(m2.Glyphs, glyphs)
	blob4 := b4.Make()
	if len(blob4.runs) != 2 {
		t.Errorf("different fonts must not merge: %d runs", len(blob4.runs))
	}
}

func TestMakeFromText(t *testing.T) {
	f := loadFont(t, 24)
	blob := MakeFromText([]byte("Hi there"), f, font.TextEncodingUTF8)
	if blob == nil {
		t.Fatal("blob should exist")
	}
	if len(blob.runs) != 1 || blob.runs[0].positioning != FullPositioning {
		t.Fatalf("MakeFromText should produce one fully positioned run")
	}
	run := &blob.runs[0]
	// Positions advance monotonically from 0 by the glyph advances.
	if run.pos[0] != 0 || run.pos[1] != 0 {
		t.Errorf("first glyph at (%v,%v), want origin", run.pos[0], run.pos[1])
	}
	widths := make([]float32, len(run.glyphs))
	f.GlyphWidths(run.glyphs, widths)
	x := float32(0)
	for i := range run.glyphs {
		if run.pos[2*i] != x {
			t.Errorf("glyph %d at x=%v, want %v", i, run.pos[2*i], x)
		}
		x += widths[i]
	}

	if MakeFromText(nil, f, font.TextEncodingUTF8) != nil {
		t.Error("empty text must make nil")
	}
	if MakeFromText([]byte{0xff, 0xfe}, f, font.TextEncodingUTF8) != nil {
		t.Error("invalid UTF-8 must make nil")
	}
}

func TestGetIntercepts(t *testing.T) {
	f := loadFont(t, 50)
	// 'p' has a descender: an underline band below the baseline must be carved out around its stem.
	blob := MakeFromText([]byte("p"), f, font.TextEncodingUTF8)
	if blob == nil {
		t.Fatal("blob should exist")
	}
	nilSlice, count := blob.GetIntercepts([2]float32{4, 8}, nil, nil)
	if count != 2 {
		t.Fatalf("intercept count = %d, want 2", count)
	}
	if nilSlice != nil {
		t.Errorf("count-only pass must not allocate, got %v", nilSlice)
	}
	intervals, got := blob.GetIntercepts([2]float32{4, 8}, make([]float32, 0, count), nil)
	if got != 2 {
		t.Fatalf("second pass count = %d", got)
	}
	if len(intervals) != 2 {
		t.Fatalf("second pass returned %d scalars, want 2", len(intervals))
	}

	// A slice that must grow (no spare capacity, or too little) still reaches the caller via the returned slice.
	for _, in := range [][]float32{make([]float32, 0), make([]float32, 0, 1), make([]float32, 0, 2)} {
		out, n := blob.GetIntercepts([2]float32{4, 8}, in, nil)
		if n != 2 || len(out) != 2 {
			t.Errorf("grown slice: count = %d, len = %d, want 2 and 2", n, len(out))
			continue
		}
		if out[0] != intervals[0] || out[1] != intervals[1] {
			t.Errorf("grown slice: got %v, want %v", out, intervals)
		}
	}

	// A slice with a non-zero length is appended to, leaving the caller's existing contents alone.
	prefix := []float32{-1, -2, -3}
	appended, appendedCount := blob.GetIntercepts([2]float32{4, 8}, prefix, nil)
	if appendedCount != 2 || len(appended) != 5 {
		t.Fatalf("append to len-3 slice: count = %d, len = %d, want 2 and 5", appendedCount, len(appended))
	}
	if appended[0] != -1 || appended[1] != -2 || appended[2] != -3 {
		t.Errorf("append clobbered the prefix: %v", appended)
	}
	if appended[3] != intervals[0] || appended[4] != intervals[1] {
		t.Errorf("appended %v, want %v", appended[3:], intervals)
	}

	// The interval must lie inside the glyph's horizontal extent and be non-empty.
	vals := interceptsValues(t, blob, [2]float32{4, 8}, nil)
	if len(vals) != 2 || vals[0] >= vals[1] {
		t.Fatalf("bad interval: %v", vals)
	}
	var bounds geom.Rect
	f.MeasureText([]byte("p"), font.TextEncodingUTF8, &bounds, nil)
	if vals[0] < bounds.Left-1 || vals[1] > bounds.Right+1 {
		t.Errorf("interval %v outside glyph extent %v", vals, bounds)
	}

	// A band above the glyph has no intercepts.
	if _, n := blob.GetIntercepts([2]float32{-100, -90}, nil, nil); n != 0 {
		t.Error("band above the glyph should be clear")
	}

	// Multiple glyphs accumulate pairs.
	blob2 := MakeFromText([]byte("pq"), f, font.TextEncodingUTF8)
	if _, n := blob2.GetIntercepts([2]float32{4, 8}, nil, nil); n != 4 {
		t.Errorf("two descenders: count = %d, want 4", n)
	}
	if pairs, n := blob2.GetIntercepts([2]float32{4, 8}, make([]float32, 0), nil); n != 4 || len(pairs) != 4 {
		t.Errorf("two descenders into a grown slice: count = %d, len = %d, want 4 and 4", n, len(pairs))
	}

	// A stroked paint still reports fill-outline intercepts (intercepts always force fill style).
	paint := &stroke.PaintSpec{Style: stroke.PaintStyleStroke, Width: 3}
	if _, n := blob2.GetIntercepts([2]float32{4, 8}, nil, paint); n != 4 {
		t.Errorf("stroked paint: count = %d, want 4", n)
	}
}

// interceptsValues fetches intercepts into a fresh slice.
func interceptsValues(t *testing.T, blob *Blob, bounds [2]float32, paint *stroke.PaintSpec) []float32 {
	t.Helper()
	_, n := blob.GetIntercepts(bounds, nil, paint)
	if n == 0 {
		return nil
	}
	out, got := blob.GetIntercepts(bounds, make([]float32, 0, n), paint)
	if got != n {
		t.Fatalf("count-only pass reported %d scalars, second pass reported %d", n, got)
	}
	return out
}

func TestGlyphRunBuilderText(t *testing.T) {
	f := loadFont(t, 24)
	b := NewGlyphRunBuilder()
	list := b.TextToGlyphRunList(f, nil, []byte("Hi"), font.TextEncodingUTF8, geom.Pt(5, 9))
	if list.Empty() || len(list.Runs) != 1 {
		t.Fatal("expected one run")
	}
	run := &list.Runs[0]
	if len(run.Glyphs) != 2 || run.Positions[0] != geom.Pt(0, 0) {
		t.Fatalf("run shape wrong: %v %v", run.Glyphs, run.Positions)
	}
	if list.Origin != geom.Pt(5, 9) {
		t.Errorf("origin %v", list.Origin)
	}
	// Source bounds are conservative (font bounds at each position).
	fb := font.GetFontBounds(f)
	if list.SourceBounds.IsEmpty() || list.SourceBounds.Left != fb.Left {
		t.Errorf("source bounds %v (font bounds %v)", list.SourceBounds, fb)
	}
	wb := list.SourceBoundsWithOrigin()
	if wb.Left != list.SourceBounds.Left+5 || wb.Top != list.SourceBounds.Top+9 {
		t.Errorf("bounds with origin %v", wb)
	}

	// Empty/invalid text yields an empty list.
	if !b.TextToGlyphRunList(f, nil, nil, font.TextEncodingUTF8, geom.Pt(0, 0)).Empty() {
		t.Error("empty text should produce an empty list")
	}
}

func TestBlobToGlyphRunList(t *testing.T) {
	f := loadFont(t, 24)
	glyphs := glyphsFor(f, "ab")

	b := NewBuilder()
	// Default run at (3, 4).
	r1 := b.AllocRun(f, 2, 3, 4, nil)
	copy(r1.Glyphs, glyphs)
	// Horizontal run at y=40.
	r2 := b.AllocRunPosH(f, 2, 40, nil)
	copy(r2.Glyphs, glyphs)
	copy(r2.Pos, []float32{7, 21})
	// Full run.
	r3 := b.AllocRunPos(f, 2, nil)
	copy(r3.Glyphs, glyphs)
	copy(r3.Pos, []float32{1, 2, 3, 4})
	blob := b.Make()

	builder := NewGlyphRunBuilder()
	list := builder.BlobToGlyphRunList(blob, geom.Pt(100, 200))
	if len(list.Runs) != 3 {
		t.Fatalf("expected 3 runs, got %d", len(list.Runs))
	}
	// Default positioning: advances from the run offset.
	widths := make([]float32, 2)
	f.GlyphWidths(glyphs, widths)
	if list.Runs[0].Positions[0] != geom.Pt(3, 4) {
		t.Errorf("default run position[0] = %v", list.Runs[0].Positions[0])
	}
	if list.Runs[0].Positions[1] != geom.Pt(3+widths[0], 4) {
		t.Errorf("default run position[1] = %v", list.Runs[0].Positions[1])
	}
	// Horizontal positioning.
	if list.Runs[1].Positions[0] != geom.Pt(7, 40) || list.Runs[1].Positions[1] != geom.Pt(21, 40) {
		t.Errorf("posH positions = %v", list.Runs[1].Positions)
	}
	// Full positioning.
	if list.Runs[2].Positions[0] != geom.Pt(1, 2) || list.Runs[2].Positions[1] != geom.Pt(3, 4) {
		t.Errorf("full positions = %v", list.Runs[2].Positions)
	}
	if list.Blob != blob || list.SourceBounds != blob.Bounds() {
		t.Error("list must carry the blob and its bounds")
	}
}

// mixedBlob builds a blob with one default, one horizontal, and one full-positioned run of the given glyphs (positions
// derived from base so distinct blobs are distinguishable).
func mixedBlob(f *font.Font, glyphs []uint16, base float32) *Blob {
	b := NewBuilder()
	r1 := b.AllocRun(f, len(glyphs), base, base+1, nil)
	copy(r1.Glyphs, glyphs)
	r2 := b.AllocRunPosH(f, len(glyphs), base+2, nil)
	copy(r2.Glyphs, glyphs)
	for i := range glyphs {
		r2.Pos[i] = base + float32(i)
	}
	r3 := b.AllocRunPos(f, len(glyphs), nil)
	copy(r3.Glyphs, glyphs)
	for i := range glyphs {
		r3.Pos[2*i] = base + float32(i)
		r3.Pos[2*i+1] = base - float32(i)
	}
	return b.Make()
}

func copyPositions(list *GlyphRunList) [][]geom.Point {
	out := make([][]geom.Point, len(list.Runs))
	for i := range list.Runs {
		out[i] = append([]geom.Point(nil), list.Runs[i].Positions...)
	}
	return out
}

// TestGlyphRunBuilderPoolReuse locks the invariants the S91 pooling relies on: the returned list is the builder's own
// retained header, a builder reused for a smaller/different blob (its position/run buffers left "poisoned" with a prior
// larger blob's data) produces byte-identical output to a fresh builder, full-positioned runs correctly share the
// reused position buffer, and Release drops the source-blob reference.
func TestGlyphRunBuilderPoolReuse(t *testing.T) {
	f := loadFont(t, 24)
	big := glyphsFor(f, "abcdef")
	small := glyphsFor(f, "xy")
	blobBig := mixedBlob(f, big, 50)
	blobSmall := mixedBlob(f, small, 7)

	// Reference output from a fresh builder for the small blob.
	want := copyPositions(NewGlyphRunBuilder().BlobToGlyphRunList(blobSmall, geom.Pt(3, 9)))

	// The returned list must be the builder's own field (so it does not allocate a fresh header), and reusing the
	// builder for a different blob must reuse that same header.
	reused := AcquireGlyphRunBuilder()
	l1 := reused.BlobToGlyphRunList(blobBig, geom.Pt(1, 1))
	if l1 != &reused.list {
		t.Fatal("list must be the builder's retained header")
	}
	// Poison the buffers with the big blob, then process the small blob through the same builder.
	got := copyPositions(reused.BlobToGlyphRunList(blobSmall, geom.Pt(3, 9)))
	l2 := reused.BlobToGlyphRunList(blobSmall, geom.Pt(3, 9))
	if l2 != &reused.list || l1 != l2 {
		t.Fatal("reused builder must return the same retained header")
	}

	if len(got) != len(want) {
		t.Fatalf("run count: reused %d, fresh %d", len(got), len(want))
	}
	for r := range want {
		if len(got[r]) != len(want[r]) {
			t.Fatalf("run %d length: reused %d, fresh %d", r, len(got[r]), len(want[r]))
		}
		for i := range want[r] {
			if got[r][i] != want[r][i] {
				t.Fatalf("run %d position %d: reused %v, fresh %v", r, i, got[r][i], want[r][i])
			}
		}
	}

	// Release must drop the source-blob reference so a pooled builder does not pin it while idle.
	ReleaseGlyphRunBuilder(reused)
	if reused.list.Blob != nil {
		t.Error("Release must clear the retained list's blob reference")
	}
}

// TestReleaseGlyphRunBuilderClearsRuns verifies Release drops every external reference an idle pooled builder would
// otherwise pin: not just the retained list's blob, but the run scratch's per-run font and glyph-slice pointers,
// including the elements past the current length that an earlier, longer draw left stranded. The scratch capacity must
// survive so reuse stays allocation-free.
func TestReleaseGlyphRunBuilderClearsRuns(t *testing.T) {
	f := loadFont(t, 24)
	blobBig := mixedBlob(f, glyphsFor(f, "abcdef"), 50)

	b := AcquireGlyphRunBuilder()
	if n := len(b.BlobToGlyphRunList(blobBig, geom.Pt(1, 1)).Runs); n != 3 {
		t.Fatalf("expected 3 runs from the big blob, got %d", n)
	}
	// A shorter draw strands the big blob's trailing runs past the slice length.
	b.TextToGlyphRunList(f, nil, []byte("Hi"), font.TextEncodingUTF8, geom.Pt(0, 0))
	capBefore := cap(b.runs)
	if capBefore < 3 {
		t.Fatalf("run scratch capacity %d, want at least 3", capBefore)
	}
	if stranded := b.runs[:capBefore][len(b.runs):]; len(stranded) == 0 || stranded[0].Font == nil {
		t.Fatal("expected the big blob's runs to remain past the slice length")
	}

	ReleaseGlyphRunBuilder(b)

	if b.list.Blob != nil {
		t.Error("Release must clear the retained list's blob reference")
	}
	if cap(b.runs) != capBefore {
		t.Errorf("run scratch capacity %d after Release, want %d (backing array must be kept)", cap(b.runs), capBefore)
	}
	if len(b.runs) != 0 {
		t.Errorf("run scratch length %d after Release, want 0", len(b.runs))
	}
	for i, run := range b.runs[:cap(b.runs)] {
		if run.Font != nil || run.Glyphs != nil || run.Positions != nil {
			t.Errorf("run %d still holds references after Release: %+v", i, run)
		}
	}

	// A released builder must still be usable, and reuse must not reallocate the kept scratch.
	if n := len(b.BlobToGlyphRunList(blobBig, geom.Pt(1, 1)).Runs); n != 3 {
		t.Fatalf("expected 3 runs after reuse, got %d", n)
	}
	if cap(b.runs) != capBefore {
		t.Errorf("run scratch reallocated on reuse: capacity %d, want %d", cap(b.runs), capBefore)
	}
}
