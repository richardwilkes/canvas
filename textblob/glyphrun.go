// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Glyph run types (GlyphRun/GlyphRunList/GlyphRunBuilder) for the reachable run shapes (no RSXform, no cluster text).
// The canvas text entry points build run lists here and hand them to the device.

package textblob

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

// GlyphRun is parallel glyph/position spans sharing one font.
type GlyphRun struct {
	Font      *font.Font
	Glyphs    []uint16
	Positions []geom.Point
}

// GlyphRunList is an ordered list of glyph runs with the source bounds and draw origin needed to render them.
type GlyphRunList struct {
	Blob         *Blob // nil unless built from a text blob
	Runs         []GlyphRun
	SourceBounds geom.Rect
	Origin       geom.Point
}

// Empty reports whether the list has no runs.
func (l *GlyphRunList) Empty() bool { return len(l.Runs) == 0 }

// MaxGlyphRunSize returns the glyph count of the largest run in the list.
func (l *GlyphRunList) MaxGlyphRunSize() int {
	most := 0
	for i := range l.Runs {
		most = max(most, len(l.Runs[i].Glyphs))
	}
	return most
}

// SourceBoundsWithOrigin returns the source bounds translated by the list's draw origin.
func (l *GlyphRunList) SourceBoundsWithOrigin() geom.Rect {
	return l.SourceBounds.Offset(l.Origin.X, l.Origin.Y)
}

// GlyphRunBuilder assembles a GlyphRunList from text or a text blob. The scratch buffers (runs/positions/glyphIDs) and
// the returned list header are retained on the builder so a pooled builder reuses their backing storage across draws.
type GlyphRunBuilder struct {
	runs      []GlyphRun
	positions []geom.Point
	glyphIDs  []uint16
	list      GlyphRunList
}

// NewGlyphRunBuilder returns an empty builder.
func NewGlyphRunBuilder() *GlyphRunBuilder { return &GlyphRunBuilder{} }

var glyphRunBuilderPool = sync.Pool{New: func() any { return &GlyphRunBuilder{} }}

// AcquireGlyphRunBuilder borrows a builder from a process-wide pool. The caller must ReleaseGlyphRunBuilder it once the
// returned GlyphRunList (and everything reachable through it) has been fully consumed — the list and its position/run
// backing arrays live on the builder and are overwritten by the next borrower.
func AcquireGlyphRunBuilder() *GlyphRunBuilder { return glyphRunBuilderPool.Get().(*GlyphRunBuilder) }

// ReleaseGlyphRunBuilder returns a builder to the pool, dropping the external references its retained list and run
// scratch hold (the source blob, and each run's font and glyph slice) so a pooled builder does not pin them while idle;
// the scratch backing arrays are kept for reuse.
func ReleaseGlyphRunBuilder(b *GlyphRunBuilder) {
	b.list = GlyphRunList{}
	// Zero every run element, including those past the current length that an earlier, longer draw left behind, since
	// each still points at a font and a glyph slice owned by whatever was last drawn.
	runs := b.runs[:cap(b.runs)]
	clear(runs)
	b.runs = runs[:0]
	glyphRunBuilderPool.Put(b)
}

// glyphRunSourceBounds computes the source bounds of a glyph run (no RSXform case). paint may be nil.
func glyphRunSourceBounds(f *font.Font, paint *stroke.PaintSpec, glyphs []uint16, positions []geom.Point) geom.Rect {
	fontBounds := font.GetFontBounds(f)
	if fontBounds.IsEmpty() {
		// Empty font bounds are likely a font bug. TightBounds has a better chance of producing useful results in this
		// case: per-glyph bounds (paint applied) offset by each position.
		var bounds geom.Rect
		glyphBounds := make([]geom.Rect, len(glyphs))
		font.GlyphBoundsWithPaint(f, glyphs, paint, glyphBounds)
		for i, r := range glyphBounds {
			if !r.IsEmpty() {
				bounds.Join(r.Offset(positions[i].X, positions[i].Y))
			}
		}
		return bounds
	}

	// Use conservative bounds. All glyphs have a box of fontBounds size.
	bounds := boundsOrEmptyPoints(positions)
	bounds.Left += fontBounds.Left
	bounds.Top += fontBounds.Top
	bounds.Right += fontBounds.Right
	bounds.Bottom += fontBounds.Bottom
	return bounds
}

// boundsOrEmptyPoints computes the bounding box over pts: empty if any coordinate is non-finite.
func boundsOrEmptyPoints(pts []geom.Point) geom.Rect {
	if len(pts) == 0 {
		return geom.Rect{}
	}
	l, t := pts[0].X, pts[0].Y
	r, b := l, t
	accum := float32(0)
	for _, p := range pts {
		accum *= p.X
		accum *= p.Y
		l = min(l, p.X)
		t = min(t, p.Y)
		r = max(r, p.X)
		b = max(b, p.Y)
	}
	if accum != 0 { // accum is NaN if any coordinate was non-finite
		return geom.Rect{}
	}
	return geom.RectLTRB(l, t, r, b)
}

// TextToGlyphRunList builds one run positioned by the font's advances from (0,0), with the draw origin carried on the
// list. paint contributes only to the tight-bounds fallback.
func (b *GlyphRunBuilder) TextToGlyphRunList(f *font.Font, paint *stroke.PaintSpec, text []byte, encoding font.TextEncoding, origin geom.Point) *GlyphRunList {
	glyphs := b.textToGlyphIDs(f, text, encoding)
	var bounds geom.Rect
	b.runs = b.runs[:0]
	if len(glyphs) > 0 {
		b.positions = resize(b.positions, len(glyphs))
		font.DrawTextPositions(f, glyphs, geom.Pt(0, 0), b.positions)
		b.runs = append(b.runs, GlyphRun{Font: f, Glyphs: glyphs, Positions: b.positions[:len(glyphs)]})
		bounds = glyphRunSourceBounds(f, paint, glyphs, b.positions[:len(glyphs)])
	}
	b.list = GlyphRunList{Runs: b.runs, SourceBounds: bounds, Origin: origin}
	return &b.list
}

// BlobToGlyphRunList converts blob's runs into a GlyphRunList, resolving each run's positions.
func (b *GlyphRunBuilder) BlobToGlyphRunList(blob *Blob, origin geom.Point) *GlyphRunList {
	// Pre-size the position buffer (covering every run, including full-positioned ones) so the per-run sub-slices
	// handed out below never move as the loop fills them, then carve one sub-slice per run from it. This lets
	// full-positioned runs share the reused buffer too rather than allocating a fresh slice each.
	positionCount := 0
	for i := range blob.runs {
		positionCount += len(blob.runs[i].glyphs)
	}
	b.positions = resize(b.positions, positionCount)
	b.runs = b.runs[:0]

	cursor := 0
	for i := range blob.runs {
		run := &blob.runs[i]
		if len(run.glyphs) == 0 || !fontIsFinite(&run.font) {
			// If no glyphs or the font is not finite, don't add the run.
			continue
		}
		positions := b.positions[cursor : cursor+len(run.glyphs)]
		cursor += len(run.glyphs)
		switch run.positioning {
		case DefaultPositioning:
			font.DrawTextPositions(&run.font, run.glyphs, run.offset, positions)
		case HorizontalPositioning:
			for j, x := range run.pos {
				positions[j] = geom.Pt(x, run.offset.Y)
			}
		case FullPositioning:
			for j := range run.glyphs {
				positions[j] = geom.Pt(run.pos[2*j], run.pos[2*j+1])
			}
		}
		b.runs = append(b.runs, GlyphRun{Font: &run.font, Glyphs: run.glyphs, Positions: positions})
	}

	b.list = GlyphRunList{Runs: b.runs, SourceBounds: blob.bounds, Origin: origin, Blob: blob}
	return &b.list
}

// textToGlyphIDs converts text to glyph IDs, reusing the builder's scratch buffer when possible.
func (b *GlyphRunBuilder) textToGlyphIDs(f *font.Font, text []byte, encoding font.TextEncoding) []uint16 {
	if encoding != font.TextEncodingGlyphID {
		count := font.CountTextElements(text, encoding)
		if count <= 0 {
			return nil
		}
		b.glyphIDs = resize(b.glyphIDs, count)
		f.TextToGlyphs(text, encoding, b.glyphIDs)
		return b.glyphIDs[:count]
	}
	glyphs := make([]uint16, len(text)/2)
	for i := range glyphs {
		glyphs[i] = uint16(text[2*i]) | uint16(text[2*i+1])<<8
	}
	return glyphs
}

// fontIsFinite reports whether the font's size and skew parameters are all finite.
func fontIsFinite(f *font.Font) bool {
	return finite32(f.Size()) && finite32(f.ScaleX()) && finite32(f.SkewX())
}

func finite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func resize[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}
