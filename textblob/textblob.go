// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The text blob type and its builder, for the run kinds that are publicly reachable: default positioning (advances from
// a run origin), horizontal positioning (per-glyph x at a common y), and full positioning (per-glyph x,y). RSXform runs
// and extended (text/cluster) runs have no public entry points and are omitted. Runs are stored as per-run slices
// rather than one packed allocation; the observable surface (bounds, intercepts, drawing) is unaffected.

package textblob

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
)

// Positioning selects how many position scalars are stored per glyph in a run.
type Positioning uint8

// Positioning values.
const (
	DefaultPositioning    Positioning = 0 // zero scalars per glyph
	HorizontalPositioning Positioning = 1 // one scalar per glyph
	FullPositioning       Positioning = 2 // two scalars per glyph
)

// scalarsPerGlyph returns the number of position scalars per glyph for pos.
func scalarsPerGlyph(pos Positioning) int {
	switch pos {
	case HorizontalPositioning:
		return 1
	case FullPositioning:
		return 2
	default:
		return 0
	}
}

// runRecord holds one glyph run: the font state captured at alloc time, the run offset, and the glyph/position buffers.
type runRecord struct {
	font        font.Font // captured by value
	glyphs      []uint16
	pos         []float32
	offset      geom.Point
	positioning Positioning
}

// Blob is an immutable set of glyph runs with conservative bounds.
type Blob struct {
	runs     []runRecord
	bounds   geom.Rect
	uniqueID uint32
}

// nextBlobUniqueID backs UniqueID (0 is reserved as an invalid/unassigned ID).
var nextBlobUniqueID atomic.Uint32

// Bounds returns the blob's conservative bounds.
func (b *Blob) Bounds() geom.Rect { return b.bounds }

// UniqueID returns a non-zero value unique among all text blobs.
func (b *Blob) UniqueID() uint32 { return b.uniqueID }

// RunBuffer is returned by the Builder's Alloc* methods: the caller fills Glyphs (and Pos for positioned runs) before
// calling Make. A zero RunBuffer (nil slices) means the allocation was rejected.
type RunBuffer struct {
	Glyphs []uint16
	Pos    []float32
}

// Builder incrementally assembles a Blob from one or more glyph runs.
type Builder struct {
	runs           []runRecord
	bounds         geom.Rect
	deferredBounds bool
}

// NewBuilder returns an empty Builder.
func NewBuilder() *Builder { return &Builder{} }

// AllocRun allocates a default-positioned run of count glyphs at (x, y).
func (b *Builder) AllocRun(f *font.Font, count int, x, y float32, bounds *geom.Rect) RunBuffer {
	return b.allocInternal(f, DefaultPositioning, count, geom.Pt(x, y), bounds)
}

// AllocRunPosH allocates a run with per-glyph x positions at a common y.
func (b *Builder) AllocRunPosH(f *font.Font, count int, y float32, bounds *geom.Rect) RunBuffer {
	return b.allocInternal(f, HorizontalPositioning, count, geom.Pt(0, y), bounds)
}

// AllocRunPos allocates a run with per-glyph (x, y) positions.
func (b *Builder) AllocRunPos(f *font.Font, count int, bounds *geom.Rect) RunBuffer {
	return b.allocInternal(f, FullPositioning, count, geom.Pt(0, 0), bounds)
}

// allocInternal allocates (or extends, via mergeRun) a run of count glyphs.
func (b *Builder) allocInternal(f *font.Font, positioning Positioning, count int, offset geom.Point, bounds *geom.Rect) RunBuffer {
	if count <= 0 || f == nil {
		return RunBuffer{}
	}
	if !b.mergeRun(f, positioning, count, offset) {
		b.updateDeferredBounds()
		b.runs = append(b.runs, runRecord{
			font:        *f,
			offset:      offset,
			positioning: positioning,
			glyphs:      make([]uint16, count),
			pos:         make([]float32, count*scalarsPerGlyph(positioning)),
		})
	}
	run := &b.runs[len(b.runs)-1]
	buffer := RunBuffer{
		Glyphs: run.glyphs[len(run.glyphs)-count:],
	}
	if n := count * scalarsPerGlyph(positioning); n > 0 {
		buffer.Pos = run.pos[len(run.pos)-n:]
	}
	if !b.deferredBounds {
		if bounds != nil {
			b.bounds.Join(*bounds)
		} else {
			b.deferredBounds = true
		}
	}
	return buffer
}

// mergeRun grows the last run when the font/positioning allow it.
func (b *Builder) mergeRun(f *font.Font, positioning Positioning, count int, offset geom.Point) bool {
	if len(b.runs) == 0 {
		return false
	}
	run := &b.runs[len(b.runs)-1]
	if run.positioning != positioning || run.font != *f {
		return false
	}
	// We can only merge fully positioned runs, or horizontally positioned runs with the same y-offset.
	if positioning != FullPositioning &&
		(positioning != HorizontalPositioning || run.offset.Y != offset.Y) {
		return false
	}
	run.glyphs = append(run.glyphs, make([]uint16, count)...)
	run.pos = append(run.pos, make([]float32, count*scalarsPerGlyph(positioning))...)
	return true
}

// updateDeferredBounds computes and joins in the bounds of the most recently allocated run, if its bounds computation
// was deferred.
func (b *Builder) updateDeferredBounds() {
	if !b.deferredBounds {
		return
	}
	run := &b.runs[len(b.runs)-1]
	var runBounds geom.Rect
	// TODO: this should probably also use conservative bounds for DefaultPositioning.
	if run.positioning == DefaultPositioning {
		runBounds = tightRunBounds(run)
	} else {
		runBounds = conservativeRunBounds(run)
	}
	b.bounds.Join(runBounds)
	b.deferredBounds = false
}

// Make returns the assembled Blob, nil for an empty builder; the builder resets either way.
func (b *Builder) Make() *Blob {
	if len(b.runs) == 0 {
		return nil
	}
	b.updateDeferredBounds()
	blob := &Blob{runs: b.runs, bounds: b.bounds, uniqueID: nextBlobUniqueID.Add(1)}
	b.runs = nil
	b.bounds = geom.Rect{}
	b.deferredBounds = false
	return blob
}

// tightRunBounds computes the exact bounds of run by measuring each glyph's outline.
func tightRunBounds(run *runRecord) geom.Rect {
	f := &run.font
	if run.positioning == DefaultPositioning {
		var bounds geom.Rect
		f.MeasureText(glyphBytes(run.glyphs), font.TextEncodingGlyphID, &bounds, nil)
		return bounds.Offset(run.offset.X, run.offset.Y)
	}

	glyphBounds := make([]geom.Rect, len(run.glyphs))
	f.GlyphBounds(run.glyphs, glyphBounds)

	var bounds geom.Rect
	scalars := scalarsPerGlyph(run.positioning)
	for i, r := range glyphBounds {
		var gx, gy float32
		if run.positioning == FullPositioning {
			gx = run.pos[i*scalars]
			gy = run.pos[i*scalars+1]
		} else {
			gx = run.pos[i]
		}
		bounds.Join(r.Offset(gx, gy))
	}
	return bounds.Offset(run.offset.X, run.offset.Y)
}

// conservativeRunBounds computes a fast (possibly loose) bound for run using the font's overall glyph bounds rather
// than measuring each glyph.
func conservativeRunBounds(run *runRecord) geom.Rect {
	fontBounds := font.GetFontBounds(&run.font)
	if fontBounds.IsEmpty() {
		// Empty font bounds are likely a font bug. TightBounds has a better chance of producing useful results in this
		// case.
		return tightRunBounds(run)
	}

	// Compute the glyph position bbox.
	var bounds geom.Rect
	switch run.positioning {
	case HorizontalPositioning:
		minX := run.pos[0]
		maxX := run.pos[0]
		for _, x := range run.pos[1:] {
			minX = min(minX, x)
			maxX = max(maxX, x)
		}
		bounds = geom.RectLTRB(minX, 0, maxX, 0)
	case FullPositioning:
		bounds = boundsOrEmptyPairs(run.pos)
	default:
		return tightRunBounds(run)
	}

	// Expand by typeface glyph bounds.
	bounds.Left += fontBounds.Left
	bounds.Top += fontBounds.Top
	bounds.Right += fontBounds.Right
	bounds.Bottom += fontBounds.Bottom

	// Offset by run position.
	return bounds.Offset(run.offset.X, run.offset.Y)
}

// boundsOrEmptyPairs computes the bounding box over (x, y) scalar pairs: empty if any coordinate is non-finite.
func boundsOrEmptyPairs(pos []float32) geom.Rect {
	if len(pos) < 2 {
		return geom.Rect{}
	}
	l, t := pos[0], pos[1]
	r, b := l, t
	accum := float32(0)
	for i := 0; i+1 < len(pos); i += 2 {
		x, y := pos[i], pos[i+1]
		accum *= x
		accum *= y
		l = min(l, x)
		t = min(t, y)
		r = max(r, x)
		b = max(b, y)
	}
	if accum != 0 { // accum is NaN if any coordinate was non-finite
		return geom.Rect{}
	}
	return geom.RectLTRB(l, t, r, b)
}

// glyphBytes encodes glyph IDs as the little-endian byte stream font.TextEncodingGlyphID reads.
func glyphBytes(glyphs []uint16) []byte {
	out := make([]byte, len(glyphs)*2)
	for i, g := range glyphs {
		out[2*i] = byte(g)
		out[2*i+1] = byte(g >> 8)
	}
	return out
}

// MakeFromText builds a blob from text shaped with f, promoted to a fully positioned blob since the bounds cost is paid
// either way.
func MakeFromText(text []byte, f *font.Font, encoding font.TextEncoding) *Blob {
	count := font.CountTextElements(text, encoding)
	if count <= 0 {
		return nil
	}
	builder := NewBuilder()
	buffer := builder.AllocRunPos(f, count, nil)
	f.TextToGlyphs(text, encoding, buffer.Glyphs)
	pos := make([]geom.Point, count)
	f.GetPos(buffer.Glyphs, pos, geom.Pt(0, 0))
	for i, p := range pos {
		buffer.Pos[2*i] = p.X
		buffer.Pos[2*i+1] = p.Y
	}
	return builder.Make()
}
