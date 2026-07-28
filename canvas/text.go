// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The canvas text stack: the simple-text/text-blob entry points, the glyph-run painter for the bitmap device, and the
// mask-blitting stage. Glyphs draw as paths (big text, hairline strokes, perspective) through the canvas — so mask
// filters and layers apply — as A8 strike masks blitted directly, as ARGB32 color-glyph sprites (emoji), or through the
// maxScale-rescaled drawBitmap stage (perspective/oversized color glyphs).
//
// Trims: drawable glyphs are not produced by any reachable scaler. LCD16 masks flow through the direct-mask stage when
// the device's pixel geometry is known and the run font's edging asks for subpixel AA; the srcOver-only fallback-props
// rule lives in drawGlyphRunListForBitmapDevice.

package canvas

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/textblob"
)

// glyphDrawScratch holds the per-draw accepted/rejected working slices that drawForBitmapDevice fills while
// partitioning a run's glyphs into the direct-mask, path, and rescaled-bitmap stages. They are pooled (retaining
// backing-array capacity across draws) rather than allocated fresh per draw.
type glyphDrawScratch struct {
	acceptedGlyphs    []*font.Glyph
	acceptedPositions []geom.Point
	rejectedGlyphIDs  []uint16
	rejectedPositions []geom.Point
	// mask is the reused per-glyph mask header paintMasks blits through; its address escapes into the BlitMask
	// interface call, so homing it on the pooled scratch (heap-resident) keeps that escape from allocating a fresh
	// header on every glyph-run draw.
	mask raster.Mask
}

var glyphDrawScratchPool = sync.Pool{New: func() any { return &glyphDrawScratch{} }}

// ensureCap grows each working slice to capacity n (the max run size) if smaller, so the subsequent appends — bounded
// by a single run's glyph count — never reallocate and the backing arrays stay stable for the life of the draw (no
// write-back needed before returning the scratch to the pool).
func (s *glyphDrawScratch) ensureCap(n int) {
	if cap(s.acceptedGlyphs) < n {
		s.acceptedGlyphs = make([]*font.Glyph, 0, n)
	}
	if cap(s.acceptedPositions) < n {
		s.acceptedPositions = make([]geom.Point, 0, n)
	}
	if cap(s.rejectedGlyphIDs) < n {
		s.rejectedGlyphIDs = make([]uint16, 0, n)
	}
	if cap(s.rejectedPositions) < n {
		s.rejectedPositions = make([]geom.Point, 0, n)
	}
}

// ScalerPaint bundles the paint fields the scaler consumes (path/mask effects, stroke geometry, the foreground color
// for COLR glyphs, and the luminance color for the LCD pre-blend).
func (p *Paint) ScalerPaint() font.ScalerPaint {
	spec := p.strokeSpec()
	return font.ScalerPaint{
		PathEffect: spec.PathEffect,
		MaskFilter: p.MaskFilter,
		Style:      spec.Style,
		Width:      spec.Width,
		MiterLimit: spec.MiterLimit,
		Cap:        spec.Cap,
		Join:       spec.Join,
		Color:      p.Color,
		LumColor:   p.LuminanceColor(),
	}
}

// LuminanceColor returns the color the LCD pre-blend keys on: the paint's single color when it has one (run through the
// color filter), else 50% gray. Shaders don't provide a luminance hint, so every shader paint takes the 50%-gray
// fallback (a pre-blend LUT-choice refinement only).
func (p *Paint) LuminanceColor() colorcore.Color {
	if p.Shader != nil {
		return colorcore.Color4f{R: 0.5, G: 0.5, B: 0.5, A: 1}.ToColor()
	}
	c4 := colorcore.Color4fFromColor(p.Color)
	if p.ColorFilter != nil {
		if out, ok := shaders.FilterColor4f(p.ColorFilter, c4.Premul()); ok {
			c4 = out.Unpremul()
		}
	}
	return c4.ToColor()
}

// DrawSimpleText draws text in the given encoding with the font's advances starting at (x, y).
func (c *Canvas) DrawSimpleText(text []byte, encoding font.TextEncoding, x, y float32, f *font.Font, paint *Paint) {
	if len(text) == 0 || f == nil || paint == nil {
		return
	}
	spec := paint.strokeSpec()
	builder := textblob.AcquireGlyphRunBuilder()
	defer textblob.ReleaseGlyphRunBuilder(builder)
	glyphRunList := builder.TextToGlyphRunList(f, &spec, text, encoding, geom.Pt(x, y))
	if !glyphRunList.Empty() {
		c.onDrawGlyphRunList(glyphRunList, paint)
	}
}

// DrawTextBlob draws the text blob with its origin at (x, y).
func (c *Canvas) DrawTextBlob(blob *textblob.Blob, x, y float32, paint *Paint) {
	if blob == nil || paint == nil {
		return
	}
	offsetBounds := blob.Bounds().Offset(x, y)
	if !offsetBounds.IsFinite() {
		return
	}
	builder := textblob.AcquireGlyphRunBuilder()
	defer textblob.ReleaseGlyphRunBuilder(builder)
	glyphRunList := builder.BlobToGlyphRunList(blob, geom.Pt(x, y))
	c.onDrawGlyphRunList(glyphRunList, paint)
}

// onDrawGlyphRunList draws a glyph-run list: quick reject, the about-to-draw layer (the paint's mask filter is applied
// inside the strike, so no mask-filter auto layer), then the top device.
func (c *Canvas) onDrawGlyphRunList(glyphRunList *textblob.GlyphRunList, paint *Paint) {
	bounds := glyphRunList.SourceBoundsWithOrigin()
	if c.internalQuickReject(bounds, paint) {
		return
	}
	working, restore, ok := c.aboutToDraw(paint, &bounds)
	if !ok {
		return
	}
	c.topDevice().DrawGlyphRunList(c, glyphRunList, working)
	if restore != nil {
		restore()
	}
}

// drawGlyphRunListForBitmapDevice partitions each run's glyphs into the path, direct-mask, and rescaled-bitmap stages
// and draws them (no drawables — no reachable scaler produces them). deviceProps carries the device's pixel geometry
// for the LCD16 lane; the bitmap blitters can only draw LCD text in srcOver, so a non-srcOver paint falls back to the
// unknown-geometry props.
//
// drawPaths is false for every tile but the first when the caller is tiling: the path stage draws through the canvas
// (which re-tiles over the whole device on its own), so issuing it per tile would draw those glyphs once per tile. The
// partition still runs when it is false — the path lane's rejects are what feed the mask stage.
func drawGlyphRunListForBitmapDevice(canvas *Canvas, dr *draw, glyphRunList *textblob.GlyphRunList, paint *Paint, drawMatrix *geom.Matrix, deviceProps *font.DeviceProps, drawPaths bool) {
	var props *font.DeviceProps
	if paint.BlendMode == raster.BlendSrcOver {
		props = deviceProps
	}
	maxRun := glyphRunList.MaxGlyphRunSize()
	scratch := glyphDrawScratchPool.Get().(*glyphDrawScratch)
	scratch.ensureCap(maxRun)
	defer glyphDrawScratchPool.Put(scratch)
	acceptedGlyphs := scratch.acceptedGlyphs[:0]
	acceptedPositions := scratch.acceptedPositions[:0]
	rejectedGlyphIDs := scratch.rejectedGlyphIDs[:0]
	rejectedPositions := scratch.rejectedPositions[:0]

	drawOrigin := glyphRunList.Origin
	positionMatrix := *drawMatrix
	positionMatrix.PreTranslate(drawOrigin.X, drawOrigin.Y)

	scalerPaint := paint.ScalerPaint()

	for runIdx := range glyphRunList.Runs {
		glyphRun := &glyphRunList.Runs[runIdx]
		runFont := glyphRun.Font

		sourceGlyphs := glyphRun.Glyphs
		sourcePositions := glyphRun.Positions

		if font.ShouldDrawAsPathMatrix(&scalerPaint, runFont, &positionMatrix) {
			spec, strikeToSourceScale := font.MakePathSpec(runFont, &scalerPaint)
			strike := spec.FindOrCreateStrike()

			acceptedGlyphs = acceptedGlyphs[:0]
			acceptedPositions = acceptedPositions[:0]
			rejectedGlyphIDs = rejectedGlyphIDs[:0]
			rejectedPositions = rejectedPositions[:0]
			for i, gid := range sourceGlyphs {
				pos := sourcePositions[i]
				if !pos.IsFinite() {
					continue
				}
				g, action := strike.DigestFor(font.ActionPath, font.PackGlyphID(gid))
				switch action {
				case font.GlyphActionAccept:
					acceptedGlyphs = append(acceptedGlyphs, g)
					acceptedPositions = append(acceptedPositions, pos)
				case font.GlyphActionReject:
					rejectedGlyphIDs = append(rejectedGlyphIDs, gid)
					rejectedPositions = append(rejectedPositions, pos)
				default:
				}
			}
			sourceGlyphs = append([]uint16(nil), rejectedGlyphIDs...)
			sourcePositions = append([]geom.Point(nil), rejectedPositions...)

			if drawPaths {
				// The paint we draw paths with must have the same anti-aliasing state as the run font, allowing the
				// paths to have the same edging as the glyph masks.
				pathPaint := *paint
				pathPaint.AntiAlias = runFont.HasSomeAntiAliasing()

				stroking := pathPaint.Style != StyleFill
				hairline := pathPaint.StrokeWidth == 0
				needsExactCTM := pathPaint.Shader != nil || pathPaint.PathEffect != nil ||
					pathPaint.MaskFilter != nil || (stroking && !hairline)

				if !needsExactCTM {
					for i, g := range acceptedGlyphs {
						glyphPath := g.Path()
						translate := drawOrigin.Add(acceptedPositions[i])
						var m geom.Matrix
						m.SetScaleTranslate(strikeToSourceScale, strikeToSourceScale, translate.X, translate.Y)
						canvas.Save()
						canvas.Concat(&m)
						canvas.DrawPath(glyphPath, &pathPaint)
						canvas.Restore()
					}
				} else {
					for i, g := range acceptedGlyphs {
						glyphPath := g.Path()
						translate := drawOrigin.Add(acceptedPositions[i])
						var m geom.Matrix
						m.SetScaleTranslate(strikeToSourceScale, strikeToSourceScale, translate.X, translate.Y)
						deviceOutline := path.New()
						glyphPath.TransformTo(&m, deviceOutline)
						canvas.DrawPath(deviceOutline, &pathPaint)
					}
				}
			}
			// (Drawable glyphs would be processed here; no reachable scaler produces them.)
		}
		if len(sourceGlyphs) > 0 && !positionMatrix.HasPerspective() {
			spec := font.MakeMaskSpec(runFont, &scalerPaint, &positionMatrix, props)
			strike := spec.FindOrCreateStrike()

			// Build up the mapping from source space to device space. Add the rounding constant halfSampleFreq, so we
			// just need to floor to get the device result.
			rounding := strike.RoundingSpec()
			positionMatrixWithRounding := positionMatrix
			half := rounding.HalfAxisSampleFreq
			positionMatrixWithRounding.PostTranslate(half.X, half.Y)
			fieldMask := rounding.IgnorePositionFieldMask

			acceptedGlyphs = acceptedGlyphs[:0]
			acceptedPositions = acceptedPositions[:0]
			rejectedGlyphIDs = rejectedGlyphIDs[:0]
			rejectedPositions = rejectedPositions[:0]
			for i, gid := range sourceGlyphs {
				pos := sourcePositions[i]
				if !pos.IsFinite() {
					continue
				}
				mappedPos := positionMatrixWithRounding.MapPoint(pos)
				packedID := font.PackGlyphIDPoint(gid, mappedPos, fieldMask)
				g, action := strike.DigestFor(font.ActionDirectMaskCPU, packedID)
				switch action {
				case font.GlyphActionAccept:
					acceptedGlyphs = append(acceptedGlyphs, g)
					acceptedPositions = append(acceptedPositions,
						geom.Pt(floorScalar(mappedPos.X), floorScalar(mappedPos.Y)))
				case font.GlyphActionReject:
					rejectedGlyphIDs = append(rejectedGlyphIDs, gid)
					rejectedPositions = append(rejectedPositions, pos)
				default:
				}
			}
			sourceGlyphs = append([]uint16(nil), rejectedGlyphIDs...)
			sourcePositions = append([]geom.Point(nil), rejectedPositions...)
			dr.paintMasks(acceptedGlyphs, acceptedPositions, paint, &scratch.mask)
		}
		if len(sourceGlyphs) > 0 {
			// The rescaled-bitmap stage: perspective draws and glyphs too big for the direct mask land here. Create a
			// strike in source space to calculate scale information, pick maxScale so the longest edge is 1:1 with its
			// side in the cache, then draw each ARGB32 mask through drawBitmap with the inverse scale (A8/BW rejects
			// are dropped).
			drawRescaledBitmaps(dr, runFont, &scalerPaint, sourceGlyphs, sourcePositions, drawOrigin,
				&positionMatrix, paint, props)
		}
	}
}

// drawRescaledBitmaps is the rescaled-bitmap stage: draw oversized/perspective color glyphs through drawBitmap at the
// inverse of the cache scale.
func drawRescaledBitmaps(dr *draw, runFont *font.Font, scalerPaint *font.ScalerPaint, sourceGlyphs []uint16, sourcePositions []geom.Point, drawOrigin geom.Point, positionMatrix *geom.Matrix, paint *Paint, props *font.DeviceProps) {
	// Create a strike in source space to calculate scale information.
	identity := geom.IdentityMatrix()
	scaleSpec := font.MakeMaskSpec(runFont, scalerPaint, &identity, props)
	scaleStrike := scaleSpec.FindOrCreateStrike()
	metricGlyphs := scaleStrike.Metrics(sourceGlyphs, make([]*font.Glyph, 0, len(sourceGlyphs)))

	// Calculate the scale that makes the longest edge 1:1 with its side in the cache.
	maxScale := float32(math.Inf(-1))
	for _, g := range metricGlyphs {
		if g.IsEmpty() {
			continue
		}
		rect := g.Rect()
		corners := [4]geom.Point{
			positionMatrix.MapXY(rect.Left, rect.Top),
			positionMatrix.MapXY(rect.Right, rect.Top),
			positionMatrix.MapXY(rect.Right, rect.Bottom),
			positionMatrix.MapXY(rect.Left, rect.Bottom),
		}
		length := func(a, b geom.Point) float32 {
			return float32(math.Hypot(float64(a.X-b.X), float64(a.Y-b.Y)))
		}
		maxScale = max32(maxScale, length(corners[1], corners[0])/rect.Width())
		maxScale = max32(maxScale, length(corners[2], corners[1])/rect.Height())
		maxScale = max32(maxScale, length(corners[3], corners[2])/rect.Width())
		maxScale = max32(maxScale, length(corners[0], corners[3])/rect.Height())
	}
	if maxScale <= 0 {
		return // to the next run.
	}
	if maxScale*runFont.Size() > 256 {
		maxScale = 256.0 / runFont.Size()
	}

	var cacheScale geom.Matrix
	cacheScale.SetScale(maxScale, maxScale)
	spec := font.MakeMaskSpec(runFont, scalerPaint, &cacheScale, props)
	strike := spec.FindOrCreateStrike()

	rounding := strike.RoundingSpec()
	creationMatrixWithRounding := cacheScale
	half := rounding.HalfAxisSampleFreq
	creationMatrixWithRounding.PostTranslate(half.X, half.Y)
	fieldMask := rounding.IgnorePositionFieldMask

	invMaxScale := 1.0 / maxScale
	for i, gid := range sourceGlyphs {
		srcPos := sourcePositions[i]
		if !srcPos.IsFinite() {
			continue
		}
		mappedPos := creationMatrixWithRounding.MapPoint(srcPos)
		packedID := font.PackGlyphIDPoint(gid, mappedPos, fieldMask)
		g, action := strike.DigestFor(font.ActionDirectMaskCPU, packedID)
		if action != font.GlyphActionAccept || g.Format != font.MaskARGB32 {
			// Only ARGB32 masks draw in this stage; other formats are dropped.
			continue
		}
		pm := g.Pixmap()
		px := imagecore.PixelsFromPixmap(pm)

		// Since the glyph in the cache is scaled by maxScale, its top left vector is too long. Reduce it to find proper
		// positions on the device.
		pos := drawOrigin.Add(srcPos).
			Add(geom.Pt(float32(g.Left)*invMaxScale, float32(g.Top)*invMaxScale))

		// The preConcat matrix for drawBitmap maps the rectangle from the (maxScale-sized) glyph cache to land in the
		// right place.
		var translate geom.Matrix
		translate.SetTranslate(pos.X, pos.Y)
		translate.PreScale(invMaxScale, invMaxScale)

		dr.drawBitmap(px, &translate, nil,
			shaders.SamplingOptions{Filter: shaders.FilterLinear}, paint)
	}
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func floorScalar(v float32) float32 { return float32(math.Floor(float64(v))) }

// checkGlyphPosition prevents glyphs from being drawn outside of or straddling the edge of device space. Comparisons
// written so NaN coordinates are treated safely.
func checkGlyphPosition(pos geom.Point) bool {
	const maxLimit = float32(math.MaxInt32 - (math.MaxInt16 + math.MaxUint16))
	const minLimit = float32(math.MinInt32 - math.MinInt16)
	gt := func(a, b float32) bool { return !(a <= b) }
	lt := func(a, b float32) bool { return !(a >= b) }
	return !(gt(pos.X, maxLimit) || lt(pos.X, minLimit) || gt(pos.Y, maxLimit) || lt(pos.Y, minLimit))
}

// paintMasks chooses the paint's blitter once, then blits each glyph mask clipped to the raster clip.
func (d *draw) paintMasks(glyphs []*font.Glyph, positions []geom.Point, paint *Paint, mask *raster.Mask) {
	blitter := chooseBlitter(d.dst, paint, d.ctm)
	// Recycle the pooled blitter chooseBlitter built (not the AA-clip wrapper below): capture it before the wrap and
	// recycle on return once every glyph mask has been blitted through it.
	defer recycleBlitter(blitter)

	// The AA-clip wrapper is a no-op for a BW clip (it returns the blitter unchanged), so wrap only for an AA clip. The
	// wrapper holds a self-referential pointer (&w.aaBlitter is returned through the blitter interface), so it must
	// escape to the heap when used — skipping it on the common BW path avoids that per-draw allocation.
	if !d.rc.IsBW() {
		var wrapper raster.AAClipBlitterWrapper
		wrapper.Init(d.rc, blitter)
		blitter = wrapper.Blitter()
	}

	useRegion := d.rc.IsBW() && !d.rc.IsRect()

	// mask is a caller-provided header (homed on the pooled glyphDrawScratch), reused across glyphs: its address
	// escapes into BlitMask (an interface call), so a fresh per-glyph `mask :=` inside the loop would heap-allocate
	// once per glyph and a function-local `var mask` once per draw; the heap-resident pooled header escapes nothing and
	// is fully overwritten each iteration (observationally identical).
	var clipper raster.RegionCliperator

	if useRegion {
		for i, glyph := range glyphs {
			pos := positions[i]
			if !checkGlyphPosition(pos) {
				continue
			}
			*mask = glyph.Mask(pos)
			clipper.Init(d.rc.BWRgn(), mask.Bounds)
			if clipper.Done() {
				continue
			}
			if glyph.Format == font.MaskARGB32 {
				// Color glyphs draw as N32-premul sprites (drawSprite clips internally, so once).
				d.paintColorGlyph(glyph, mask.Bounds, paint)
				continue
			}
			for !clipper.Done() {
				blitter.BlitMask(mask, clipper.Rect())
				clipper.Next()
			}
		}
	} else {
		var clipBounds geom.IRect
		if d.rc.IsBW() {
			clipBounds = d.rc.BWRgn().Bounds()
		} else {
			clipBounds = d.rc.AARgn().Bounds()
		}
		for i, glyph := range glyphs {
			pos := positions[i]
			if !checkGlyphPosition(pos) {
				continue
			}
			*mask = glyph.Mask(pos)
			bounds := mask.Bounds
			// This extra test is worth it, assuming that most of the time it succeeds.
			if !clipBounds.ContainsRect(mask.Bounds) {
				if !bounds.Intersect(clipBounds) {
					continue
				}
			}
			if glyph.Format == font.MaskARGB32 {
				d.paintColorGlyph(glyph, mask.Bounds, paint)
				continue
			}
			blitter.BlitMask(mask, bounds)
		}
	}
}

// paintColorGlyph draws an ARGB32 glyph mask as an N32-premul sprite at its mask bounds.
func (d *draw) paintColorGlyph(glyph *font.Glyph, maskBounds geom.IRect, paint *Paint) {
	pm := glyph.Pixmap()
	d.drawSprite(&pm, maskBounds.Left, maskBounds.Top, paint)
}
