// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Computes the x-intervals a blob's glyph outlines carve out of a horizontal band (underline carve-outs), through the
// canonical-size path strike with the paint's stroke applied.

package textblob

import (
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

// GetIntercepts computes the intercepts of the blob's glyph outlines against a horizontal band: bounds is the [top,
// bottom] band relative to the blob origin; intervals (may be nil to count only) receives (entry, exit) pairs; paint
// may be nil. Returns the number of scalars that were (or would be) written.
func (b *Blob) GetIntercepts(bounds [2]float32, intervals []float32, paint *stroke.PaintSpec) int {
	builder := NewGlyphRunBuilder()
	glyphRunList := builder.BlobToGlyphRunList(b, geom.Pt(0, 0))

	intervalCount := 0
	for i := range glyphRunList.Runs {
		intervals = getGlyphRunIntercepts(&glyphRunList.Runs[i], paint, bounds, intervals, &intervalCount)
	}
	return intervalCount
}

// getGlyphRunIntercepts computes the intercepts for one glyph run.
func getGlyphRunIntercepts(glyphRun *GlyphRun, paint *stroke.PaintSpec, bounds [2]float32, intervals []float32, intervalCount *int) []float32 {
	scale := float32(1)
	var interceptPaint stroke.PaintSpec
	if paint != nil {
		interceptPaint = *paint
	}
	interceptFont := *glyphRun.Font

	// stroke.PaintSpec carries no mask filter, so there is nothing to strip here.

	// Can't use our canonical size if we need to apply path effects.
	if interceptPaint.PathEffect == nil {
		// If the wrong size is going to be used, don't hint anything.
		interceptFont.SetHinting(font.HintingNone)
		interceptFont.SetSubpixel(true)
		scale = interceptFont.Size() / canonicalTextSizeForPaths
		interceptFont.SetSize(canonicalTextSizeForPaths)
		// Note: scale can be zero here (even if it wasn't before the divide). IEEE divide semantics carry through;
		// downstream checks for non-finite coordinates handle it.
		if interceptPaint.Width > 0 && interceptPaint.Style != stroke.PaintStyleFill {
			interceptPaint.Width /= scale
		}
	}

	interceptPaint.Style = stroke.PaintStyleFill
	interceptPaint.PathEffect = nil

	scalerPaint := font.ScalerPaint{
		Style:      interceptPaint.Style,
		Width:      interceptPaint.Width,
		MiterLimit: interceptPaint.MiterLimit,
		Cap:        interceptPaint.Cap,
		Join:       interceptPaint.Join,
	}
	spec := font.MakeWithNoDeviceSpec(&interceptFont, &scalerPaint)
	strike := spec.FindOrCreateStrike()

	results := make([]*font.Glyph, 0, len(glyphRun.Glyphs))
	results = strike.PreparePaths(glyphRun.Glyphs, results)
	for i, glyph := range results {
		pos := glyphRun.Positions[i]
		if glyph.Path() != nil {
			// The typeface is scaled, so un-scale the bounds to be in the space of the typeface. Also ensure the bounds
			// are properly offset by the vertical positioning of the glyph.
			scaledBounds := [2]float32{
				(bounds[0] - pos.Y) / scale,
				(bounds[1] - pos.Y) / scale,
			}
			intervals = strike.FindIntercepts(scaledBounds, scale, pos.X, glyph, intervals, intervalCount)
		}
	}
	return intervals
}

// canonicalTextSizeForPaths is the font size used when computing intercepts via path outlines, so results can be
// rescaled rather than recomputed per requested size.
const canonicalTextSizeForPaths = 64
