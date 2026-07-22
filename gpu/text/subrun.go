// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SubRunContainer and its SubRun classes: the blob -> subrun conversion that stages each glyph run through the
// supported drawing lanes — direct atlas masks (the 99.99% case), glyph-outline paths, and transformed atlas masks (the
// drawing of last resort for large/perspective color glyphs), plus distance-field (SDFT) atlas masks. The drawable lane
// has no supported scaler, serialization only serves a remote glyph cache/text "slugs" and is not needed here,
// allocation uses ordinary Go allocation rather than a dedicated arena, and the 3D/emboss mask-filter auto-layer check
// is unreachable (no 3D mask format in this codebase).

package text

import (
	"math"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/patheffect"
	"github.com/richardwilkes/canvas/stroke"
	"github.com/richardwilkes/canvas/textblob"
)

// RegenerateAtlasDelegate is provided by the GPU text op at flush time; returns success and the number of glyphs placed
// in the atlas.
type RegenerateAtlasDelegate func(glyphs *GlyphVector, begin, end int,
	maskFormat gpu.MaskFormat, padding int) (ok bool, glyphsPlaced int)

// AtlasDrawDelegate is called by a subrun's Draw to hand off to the atlas text drawing op; only the subrun, origin, and
// paint are needed (subrun keep-alive is the garbage collector's job, and per-renderer data travels on the subrun
// itself via AtlasSubRun.GlyphParams).
type AtlasDrawDelegate func(subRun AtlasSubRun, drawOrigin geom.Point, paint *canvas.Paint)

// GlyphParams describes how the atlas drawing op should interpret a subrun's atlas data (mask type/geometry-processor
// selection, including the SDFT lane).
type GlyphParams struct {
	IsSDF bool
	IsLCD bool
	IsAA  bool
}

// SubRun is the ability to draw and to be reuse-checked.
type SubRun interface {
	// Draw renders the subrun.
	Draw(c *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, atlas AtlasDrawDelegate)
	// CanReuse reports whether an already-cached subrun can handle this position matrix.
	CanReuse(positionMatrix *geom.Matrix) bool
}

// AtlasSubRun is the API the atlas drawing op uses to generate vertex data.
type AtlasSubRun interface {
	SubRun
	// GlyphCount returns the number of glyphs in the subrun.
	GlyphCount() int
	// Glyphs returns the resolved glyphs (valid during flush, after regeneration resolved them).
	Glyphs() []*Glyph
	// MaskFormat returns the subrun's atlas mask format.
	MaskFormat() gpu.MaskFormat
	// GlyphSrcPadding returns the padding baked into each atlas entry for this subrun kind.
	GlyphSrcPadding() int
	// DeviceRectAndNeedsTransform returns the device bounding rect and whether it still needs further transformation to
	// draw.
	DeviceRectAndNeedsTransform(positionMatrix *geom.Matrix) (needsTransform bool, rect geom.Rect)
	// VertexStride returns the per-vertex byte size for the given draw matrix.
	VertexStride(drawMatrix *geom.Matrix) int
	// FillVertexData writes this subrun's vertex data for glyphs [offset, offset+count).
	FillVertexData(dst []byte, offset, count int, color colorcore.PMColor4f,
		drawMatrix *geom.Matrix, drawOrigin geom.Point, clip geom.IRect)
	// RegenerateAtlas ensures this subrun's glyphs are resolved and placed in the atlas. Not thread safe; call only
	// from the single-threaded flush.
	RegenerateAtlas(begin, end int, regenerate RegenerateAtlasDelegate) (ok bool, placed int)
	// GlyphParams returns how the atlas drawing op should interpret this subrun's atlas data.
	GlyphParams() GlyphParams
}

//////////////////////////////////////////////////////////////////////////////
// Atlas subruns.

// atlasSubRunBase carries the fields and methods shared by every AtlasSubRun implementation.
type atlasSubRunBase struct {
	glyphs       GlyphVector
	vertexFiller VertexFiller
}

func (a *atlasSubRunBase) GlyphCount() int { return a.glyphs.GlyphCount() }

func (a *atlasSubRunBase) Glyphs() []*Glyph { return a.glyphs.Glyphs() }

func (a *atlasSubRunBase) MaskFormat() gpu.MaskFormat { return a.vertexFiller.MaskType() }

func (a *atlasSubRunBase) VertexStride(drawMatrix *geom.Matrix) int {
	return a.vertexFiller.VertexStride(drawMatrix)
}

func (a *atlasSubRunBase) FillVertexData(dst []byte, offset, count int, color colorcore.PMColor4f, drawMatrix *geom.Matrix, drawOrigin geom.Point, clip geom.IRect) {
	positionMatrix := *drawMatrix
	positionMatrix.PreTranslate(drawOrigin.X, drawOrigin.Y)
	a.vertexFiller.FillVertexData(dst, offset, count, a.glyphs.Glyphs(), color, &positionMatrix,
		clip)
}

// DirectMaskSubRun holds mask pixels in 1:1 correspondence with device pixels; destination rectangles are in device
// space. Handles color glyphs too.
type DirectMaskSubRun struct {
	atlasSubRunBase
}

// makeDirectMaskSubRun returns a DirectMaskSubRun for the accepted glyphs.
func makeDirectMaskSubRun(creationBounds geom.Rect, accepted []acceptedGlyph, creationMatrix *geom.Matrix, strike *font.Strike, spec *font.StrikeSpec, maskType gpu.MaskFormat) *DirectMaskSubRun {
	positions, packedIDs := acceptedPositionsAndIDs(accepted)
	return &DirectMaskSubRun{atlasSubRunBase: atlasSubRunBase{
		vertexFiller: MakeVertexFiller(maskType, creationMatrix, creationBounds, positions, IsDirect),
		glyphs:       MakeGlyphVector(strike, spec, packedIDs),
	}}
}

// Draw hands the subrun off to the atlas drawing delegate.
func (s *DirectMaskSubRun) Draw(_ *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, atlas AtlasDrawDelegate) {
	atlas(s, drawOrigin, paint)
}

// GlyphSrcPadding returns the atlas padding for direct masks: none.
func (s *DirectMaskSubRun) GlyphSrcPadding() int { return 0 }

// DeviceRectAndNeedsTransform returns the device rect and whether the position matrix requires more than an integer
// translation of the creation matrix.
func (s *DirectMaskSubRun) DeviceRectAndNeedsTransform(positionMatrix *geom.Matrix) (bool, geom.Rect) {
	integerTranslate, deviceRect := s.vertexFiller.DeviceRectAndCheckTransform(positionMatrix)
	return !integerTranslate, deviceRect
}

// RegenerateAtlas ensures this subrun's glyphs are resolved and placed in the atlas.
func (s *DirectMaskSubRun) RegenerateAtlas(begin, end int, regenerate RegenerateAtlasDelegate) (ok bool, glyphsPlaced int) {
	return regenerate(&s.glyphs, begin, end, s.vertexFiller.MaskType(), s.GlyphSrcPadding())
}

// CanReuse reports whether positionMatrix is still an integer translation of the creation matrix.
func (s *DirectMaskSubRun) CanReuse(positionMatrix *geom.Matrix) bool {
	reuse, _ := s.vertexFiller.DeviceRectAndCheckTransform(positionMatrix)
	return reuse
}

// GlyphParams returns this subrun's atlas interpretation. Since this is non-SDF, isAA will be ignored so we just pass
// true.
func (s *DirectMaskSubRun) GlyphParams() GlyphParams {
	return GlyphParams{IsSDF: false, IsLCD: s.vertexFiller.IsLCD(), IsAA: true}
}

// TransformedMaskSubRun holds glyphs whose atlas image is transformed to the screen — large color glyphs and
// perspective draws; destination rectangles are in source space.
type TransformedMaskSubRun struct {
	atlasSubRunBase
	// isBigEnough: if the initial position matrix is not scaling the cache entry to be larger, then a cache with
	// smaller glyphs may be better.
	isBigEnough bool
}

// makeTransformedMaskSubRun returns a TransformedMaskSubRun for the accepted glyphs.
func makeTransformedMaskSubRun(accepted []acceptedGlyph, initialPositionMatrix *geom.Matrix, strike *font.Strike, spec *font.StrikeSpec, creationMatrix *geom.Matrix, creationBounds geom.Rect, maskType gpu.MaskFormat) *TransformedMaskSubRun {
	positions, packedIDs := acceptedPositionsAndIDs(accepted)
	return &TransformedMaskSubRun{
		atlasSubRunBase: atlasSubRunBase{
			vertexFiller: MakeVertexFiller(maskType, creationMatrix, creationBounds, positions,
				IsTransformed),
			glyphs: MakeGlyphVector(strike, spec, packedIDs),
		},
		isBigEnough: initialPositionMatrix.MaxScale() >= 1,
	}
}

// Draw hands the subrun off to the atlas drawing delegate.
func (s *TransformedMaskSubRun) Draw(_ *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, atlas AtlasDrawDelegate) {
	atlas(s, drawOrigin, paint)
}

// GlyphSrcPadding returns the atlas padding for transformed masks: one texel.
func (s *TransformedMaskSubRun) GlyphSrcPadding() int { return 1 }

// DeviceRectAndNeedsTransform returns the device rect; transformed masks always need a transform.
func (s *TransformedMaskSubRun) DeviceRectAndNeedsTransform(positionMatrix *geom.Matrix) (bool, geom.Rect) {
	_, deviceRect := s.vertexFiller.DeviceRectAndCheckTransform(positionMatrix)
	return true, deviceRect
}

// RegenerateAtlas ensures this subrun's glyphs are resolved and placed in the atlas.
func (s *TransformedMaskSubRun) RegenerateAtlas(begin, end int, regenerate RegenerateAtlasDelegate) (ok bool, glyphsPlaced int) {
	return regenerate(&s.glyphs, begin, end, s.vertexFiller.MaskType(), s.GlyphSrcPadding())
}

// CanReuse reports whether the cache entry is still large enough for the current draw.
func (s *TransformedMaskSubRun) CanReuse(*geom.Matrix) bool {
	return s.isBigEnough
}

// GlyphParams returns this subrun's atlas interpretation. Since this is non-SDF, isAA will be ignored so we just pass
// true.
func (s *TransformedMaskSubRun) GlyphParams() GlyphParams {
	return GlyphParams{IsSDF: false, IsLCD: s.vertexFiller.IsLCD(), IsAA: true}
}

// SDFTSubRun holds glyphs drawn from 8-bit distance-field atlas entries, transformed to the screen; reusable while the
// position matrix stays inside the strike's matrix range.
type SDFTSubRun struct {
	atlasSubRunBase
	useLCDText  bool
	antiAliased bool
	matrixRange SDFTMatrixRange
}

// makeSDFTSubRun returns an SDFTSubRun for the accepted glyphs.
func makeSDFTSubRun(runFont *font.Font, accepted []acceptedGlyph, creationMatrix *geom.Matrix, creationBounds geom.Rect, matrixRange SDFTMatrixRange, strike *font.Strike, spec *font.StrikeSpec) *SDFTSubRun {
	positions, packedIDs := acceptedPositionsAndIDs(accepted)
	return &SDFTSubRun{
		atlasSubRunBase: atlasSubRunBase{
			vertexFiller: MakeVertexFiller(gpu.MaskFormatA8, creationMatrix, creationBounds,
				positions, IsTransformed),
			glyphs: MakeGlyphVector(strike, spec, packedIDs),
		},
		useLCDText:  runFont.Edging() == font.EdgingSubpixelAntiAlias,
		antiAliased: runFont.HasSomeAntiAliasing(),
		matrixRange: matrixRange,
	}
}

// Draw hands the subrun off to the atlas drawing delegate.
func (s *SDFTSubRun) Draw(_ *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, atlas AtlasDrawDelegate) {
	atlas(s, drawOrigin, paint)
}

// GlyphSrcPadding returns the distance-field inset padding baked into the atlas entry.
func (s *SDFTSubRun) GlyphSrcPadding() int { return font.DistanceFieldInset }

// DeviceRectAndNeedsTransform returns the device rect; SDFT masks always need a transform.
func (s *SDFTSubRun) DeviceRectAndNeedsTransform(positionMatrix *geom.Matrix) (bool, geom.Rect) {
	_, deviceRect := s.vertexFiller.DeviceRectAndCheckTransform(positionMatrix)
	return true, deviceRect
}

// RegenerateAtlas ensures this subrun's glyphs are resolved and placed in the atlas.
func (s *SDFTSubRun) RegenerateAtlas(begin, end int, regenerate RegenerateAtlasDelegate) (ok bool, glyphsPlaced int) {
	return regenerate(&s.glyphs, begin, end, gpu.MaskFormatA8, s.GlyphSrcPadding())
}

// CanReuse reports whether positionMatrix still falls within the strike's reusable matrix range.
func (s *SDFTSubRun) CanReuse(positionMatrix *geom.Matrix) bool {
	return s.matrixRange.MatrixInRange(positionMatrix)
}

// GlyphParams returns this subrun's atlas interpretation.
func (s *SDFTSubRun) GlyphParams() GlyphParams {
	return GlyphParams{IsSDF: true, IsLCD: s.useLCDText, IsAA: s.antiAliased}
}

//////////////////////////////////////////////////////////////////////////////
// Path subrun.

// idOrPath holds a glyph ID until submitDraws converts it to a path.
type idOrPath struct {
	path    *path.Path // nil until converted (or when the glyph has no outline)
	glyphID uint16
}

// pathOpSubmitter holds glyph IDs until drawing, when they are converted to paths through the strike.
type pathOpSubmitter struct {
	strike              *font.Strike // strong ref keeping the strike alive; dropped after conversion
	idsOrPaths          []idOrPath
	positions           []geom.Point
	strikeToSourceScale float32
	isAntiAliased       bool
	pathsAreCreated     bool
}

// makePathOpSubmitter returns a pathOpSubmitter for the accepted glyphs.
func makePathOpSubmitter(accepted []acceptedGlyph, isAntiAliased bool, strikeToSourceScale float32, strike *font.Strike) pathOpSubmitter {
	idsOrPaths := make([]idOrPath, len(accepted))
	positions := make([]geom.Point, len(accepted))
	for i := range accepted {
		idsOrPaths[i].glyphID = accepted[i].packedID.GlyphID()
		positions[i] = accepted[i].position
	}
	return pathOpSubmitter{
		strike:              strike,
		idsOrPaths:          idsOrPaths,
		positions:           positions,
		strikeToSourceScale: strikeToSourceScale,
		isAntiAliased:       isAntiAliased,
	}
}

// blurQuery is the exported surface of a blur mask filter.
type blurQuery interface {
	Sigma() float32
	Style() maskfilter.BlurStyle
	RespectCTM() bool
}

// asABlur reports whether mf is a blur mask filter: only respect-CTM blurs report their spec.
func asABlur(mf maskfilter.MaskFilter) (blurQuery, bool) {
	b, ok := mf.(blurQuery)
	if !ok || !b.RespectCTM() {
		return nil, false
	}
	return b, true
}

// submitDraws draws each glyph as an outline path, converting glyph IDs to paths on first use.
func (p *pathOpSubmitter) submitDraws(c *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint) {
	// Convert the glyph IDs to paths if it hasn't been done yet.
	if !p.pathsAreCreated {
		glyphIDs := make([]uint16, len(p.idsOrPaths))
		for i := range p.idsOrPaths {
			glyphIDs[i] = p.idsOrPaths[i].glyphID
		}
		glyphs := p.strike.PreparePaths(glyphIDs, make([]*font.Glyph, 0, len(glyphIDs)))
		for i, g := range glyphs {
			p.idsOrPaths[i].path = g.Path()
		}
		// Drop the ref to the strike so that it can be purged from the cache if needed.
		p.strike = nil
		p.pathsAreCreated = true
	}

	runPaint := *paint
	runPaint.AntiAlias = p.isAntiAliased

	// If there are shaders, non-blur mask filters or styles, the path must be scaled into source space independently of
	// the CTM. This allows the CTM to be correct for the different effects.
	strikeToSource := geom.Matrix{}
	strikeToSource.SetScale(p.strikeToSourceScale, p.strikeToSourceScale)
	strikeToSource.PostTranslate(drawOrigin.X, drawOrigin.Y)

	rec := strokeRecFromPaint(&runPaint)
	var blur blurQuery
	if runPaint.MaskFilter != nil {
		blur, _ = asABlur(runPaint.MaskFilter)
	}
	needsExactCTM := runPaint.Shader != nil || runPaint.PathEffect != nil ||
		(!rec.IsFillStyle() && !rec.IsHairlineStyle()) ||
		(runPaint.MaskFilter != nil && blur == nil)

	if !needsExactCTM {
		// If there is a blur mask filter, then sigma needs to be adjusted to account for the scaling of
		// strikeToSourceScale.
		if blur != nil {
			runPaint.MaskFilter = maskfilter.NewBlur(blur.Style(),
				blur.Sigma()/p.strikeToSourceScale, true)
		}
		for i := range p.idsOrPaths {
			if p.idsOrPaths[i].path == nil {
				continue
			}
			pathMatrix := strikeToSource
			pathMatrix.PostTranslate(p.positions[i].X, p.positions[i].Y)
			c.Save()
			c.Concat(&pathMatrix)
			c.DrawPath(p.idsOrPaths[i].path, &runPaint)
			c.Restore()
		}
	} else {
		// Transform the path to source space, leaving the CTM unchanged for the effects.
		for i := range p.idsOrPaths {
			if p.idsOrPaths[i].path == nil {
				continue
			}
			pathMatrix := strikeToSource
			pathMatrix.PostTranslate(p.positions[i].X, p.positions[i].Y)
			deviceOutline := &path.Path{}
			p.idsOrPaths[i].path.TransformTo(&pathMatrix, deviceOutline)
			c.DrawPath(deviceOutline, &runPaint)
		}
	}
}

// PathSubRun draws glyphs as individual outline paths.
type PathSubRun struct {
	pathDrawing pathOpSubmitter
}

// Draw draws the subrun's glyph paths.
func (s *PathSubRun) Draw(c *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, _ AtlasDrawDelegate) {
	s.pathDrawing.submitDraws(c, drawOrigin, paint)
}

// CanReuse always returns true: path subruns have no position-dependent state.
func (s *PathSubRun) CanReuse(*geom.Matrix) bool { return true }

// strokeRecFromPaint builds the stroke spec a paint implies for path outlining.
func strokeRecFromPaint(paint *canvas.Paint) stroke.Rec {
	spec := stroke.PaintSpec{
		Style:      stroke.PaintStyle(paint.Style),
		Width:      paint.StrokeWidth,
		MiterLimit: paint.MiterLimit,
		Cap:        stroke.Cap(paint.Cap),
		Join:       stroke.Join(paint.Join),
	}
	return stroke.NewStrokeRecFromPaint(&spec, 1)
}

//////////////////////////////////////////////////////////////////////////////
// SubRunContainer.

// SubRunContainer holds all the subruns produced from one glyph run list.
type SubRunContainer struct {
	subRuns               []SubRun
	initialPositionMatrix geom.Matrix
}

// InitialPosition returns the position matrix the container was built with.
func (c *SubRunContainer) InitialPosition() *geom.Matrix { return &c.initialPositionMatrix }

// IsEmpty reports whether the container has no subruns.
func (c *SubRunContainer) IsEmpty() bool { return len(c.subRuns) == 0 }

// SubRuns exposes the subrun list (tests).
func (c *SubRunContainer) SubRuns() []SubRun { return c.subRuns }

// Draw draws every subrun in the container.
func (c *SubRunContainer) Draw(cv *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, atlas AtlasDrawDelegate) {
	for _, subRun := range c.subRuns {
		subRun.Draw(cv, drawOrigin, paint, atlas)
	}
}

// CanReuse reports whether every subrun in the container can handle this position matrix.
func (c *SubRunContainer) CanReuse(positionMatrix *geom.Matrix) bool {
	for _, subRun := range c.subRuns {
		if !subRun.CanReuse(positionMatrix) {
			return false
		}
	}
	return true
}

// acceptedGlyph is one accepted (packed ID, position, mask format) tuple from the prepare stages.
type acceptedGlyph struct {
	position geom.Point
	packedID font.PackedGlyphID
	format   font.MaskFormat
}

// rejectedGlyph is one rejected (glyph ID, position) pair passed to the next stage.
type rejectedGlyph struct {
	position geom.Point
	glyphID  uint16
}

func acceptedPositionsAndIDs(accepted []acceptedGlyph) ([]geom.Point, []font.PackedGlyphID) {
	positions := make([]geom.Point, len(accepted))
	packedIDs := make([]font.PackedGlyphID, len(accepted))
	for i := range accepted {
		positions[i] = accepted[i].position
		packedIDs[i] = accepted[i].packedID
	}
	return positions, packedIDs
}

// glyphBoundsAccumulator unions glyph bounding rects. Accepted glyphs are never empty, so a simple init-flagged union
// suffices.
type glyphBoundsAccumulator struct {
	rect  geom.Rect
	valid bool
}

func (a *glyphBoundsAccumulator) join(r geom.Rect) {
	if !a.valid {
		a.rect = r
		a.valid = true
		return
	}
	a.rect.Join(r)
}

// prepareForSDFTDrawing digests glyphIDs against the strike for distance-field-atlas drawing, splitting them into
// accepted and rejected and computing the accepted glyphs' bounding rect.
func prepareForSDFTDrawing(strike *font.Strike, creationMatrix *geom.Matrix, glyphIDs []uint16, positions []geom.Point, accepted []acceptedGlyph, rejected []rejectedGlyph) ([]acceptedGlyph, []rejectedGlyph, geom.Rect) {
	var boundingRect glyphBoundsAccumulator
	for i, gid := range glyphIDs {
		pos := positions[i]
		if !pos.IsFinite() {
			continue
		}
		packedID := font.PackGlyphID(gid)
		g, action := strike.DigestFor(font.ActionSDFT, packedID)
		switch action {
		case font.GlyphActionAccept:
			mappedPos := creationMatrix.MapPoint(pos)
			// The SDFT glyphs have 2-pixel wide padding that should not be used in calculating the source rectangle.
			glyphBounds := g.Rect().Inset(font.DistanceFieldInset, font.DistanceFieldInset).
				Offset(mappedPos.X, mappedPos.Y)
			boundingRect.join(glyphBounds)
			accepted = append(accepted, acceptedGlyph{
				position: geom.Pt(glyphBounds.Left, glyphBounds.Top),
				packedID: packedID,
				format:   g.Format,
			})
		case font.GlyphActionReject:
			rejected = append(rejected, rejectedGlyph{position: pos, glyphID: gid})
		default:
		}
	}
	return accepted, rejected, boundingRect.rect
}

// dashQuery is the exported surface of a dash path effect.
type dashQuery interface {
	DashIntervals() []float32
	DashPhase() float32
}

// makeSDFTStrikeSpec builds the SDF strike spec from the run font and paint. The SDF lane is selected by the strike
// spec format set in font.MakeSDFTMaskSpec (see there).
func makeSDFTStrikeSpec(runFont *font.Font, scalerPaint *font.ScalerPaint, deviceMatrix *geom.Matrix, textLocation geom.Point, control *SubRunControl) (font.StrikeSpec, float32, SDFTMatrixRange) {
	dfPaint := *scalerPaint

	dfFont, strikeToSourceScale, matrixRange := control.GetSDFFont(runFont, deviceMatrix,
		textLocation)

	// Adjust the stroke width by the scale factor for drawing the SDFT.
	dfPaint.Width = scalerPaint.Width / strikeToSourceScale

	// Check for dashing and adjust the intervals.
	if dash, ok := dfPaint.PathEffect.(dashQuery); ok {
		src := dash.DashIntervals()
		scaledIntervals := make([]float32, len(src))
		for i, v := range src {
			scaledIntervals[i] = v / strikeToSourceScale
		}
		dfPaint.PathEffect = patheffect.MakeDash(scaledIntervals,
			dash.DashPhase()/strikeToSourceScale)
	}

	// Fake-gamma and subpixel antialiasing are applied in the shader; the A8 lane is always gamma-free.
	return font.MakeSDFTMaskSpec(&dfFont, &dfPaint), strikeToSourceScale, matrixRange
}

// prepareForDirectMaskDrawing digests glyphIDs against the strike for direct-mask drawing, splitting them into accepted
// and rejected and computing the accepted glyphs' bounding rect.
func prepareForDirectMaskDrawing(strike *font.Strike, positionMatrix *geom.Matrix, glyphIDs []uint16, positions []geom.Point, accepted []acceptedGlyph, rejected []rejectedGlyph) ([]acceptedGlyph, []rejectedGlyph, geom.Rect) {
	rounding := strike.RoundingSpec()
	// Build up the mapping from source space to device space. Add the rounding constant halfSampleFreq, so we just need
	// to floor to get the device result.
	positionMatrixWithRounding := *positionMatrix
	half := rounding.HalfAxisSampleFreq
	positionMatrixWithRounding.PostTranslate(half.X, half.Y)
	fieldMask := rounding.IgnorePositionFieldMask

	var boundingRect glyphBoundsAccumulator
	for i, gid := range glyphIDs {
		pos := positions[i]
		if !pos.IsFinite() {
			continue
		}
		mappedPos := positionMatrixWithRounding.MapPoint(pos)
		packedID := font.PackGlyphIDPoint(gid, mappedPos, fieldMask)
		g, action := strike.DigestFor(font.ActionDirectMask, packedID)
		switch action {
		case font.GlyphActionAccept:
			roundedPos := geom.Pt(floorScalar(mappedPos.X), floorScalar(mappedPos.Y))
			glyphBounds := g.Rect().Offset(roundedPos.X, roundedPos.Y)
			boundingRect.join(glyphBounds)
			accepted = append(accepted, acceptedGlyph{
				position: geom.Pt(glyphBounds.Left, glyphBounds.Top),
				packedID: packedID,
				format:   g.Format,
			})
		case font.GlyphActionReject:
			rejected = append(rejected, rejectedGlyph{position: pos, glyphID: gid})
		default:
		}
	}
	return accepted, rejected, boundingRect.rect
}

// prepareForMaskDrawing digests glyphIDs against the strike for the transformed-mask lane, splitting them into accepted
// and rejected and computing the accepted glyphs' bounding rect.
func prepareForMaskDrawing(strike *font.Strike, creationMatrix *geom.Matrix, glyphIDs []uint16, positions []geom.Point, accepted []acceptedGlyph, rejected []rejectedGlyph) ([]acceptedGlyph, []rejectedGlyph, geom.Rect) {
	var boundingRect glyphBoundsAccumulator
	for i, gid := range glyphIDs {
		pos := positions[i]
		if !pos.IsFinite() {
			continue
		}
		packedID := font.PackGlyphID(gid)
		g, action := strike.DigestFor(font.ActionMask, packedID)
		switch action {
		case font.GlyphActionAccept:
			mappedPos := creationMatrix.MapPoint(pos)
			glyphBounds := g.Rect().Offset(mappedPos.X, mappedPos.Y)
			boundingRect.join(glyphBounds)
			accepted = append(accepted, acceptedGlyph{
				position: geom.Pt(glyphBounds.Left, glyphBounds.Top),
				packedID: packedID,
				format:   g.Format,
			})
		case font.GlyphActionReject:
			rejected = append(rejected, rejectedGlyph{position: pos, glyphID: gid})
		default:
		}
	}
	return accepted, rejected, boundingRect.rect
}

// prepareForPathDrawing digests glyphIDs against the strike for path drawing, splitting them into accepted and
// rejected.
func prepareForPathDrawing(strike *font.Strike, glyphIDs []uint16, positions []geom.Point, accepted []acceptedGlyph, rejected []rejectedGlyph) ([]acceptedGlyph, []rejectedGlyph) {
	for i, gid := range glyphIDs {
		pos := positions[i]
		if !pos.IsFinite() {
			continue
		}
		_, action := strike.DigestFor(font.ActionPath, font.PackGlyphID(gid))
		switch action {
		case font.GlyphActionAccept:
			accepted = append(accepted, acceptedGlyph{
				position: pos,
				packedID: font.PackGlyphID(gid),
			})
		case font.GlyphActionReject:
			rejected = append(rejected, rejectedGlyph{position: pos, glyphID: gid})
		default:
		}
	}
	return accepted, rejected
}

// findMaximumGlyphDimension returns the largest glyph dimension across glyphIDs, in the strike's space.
func findMaximumGlyphDimension(strike *font.Strike, glyphIDs []uint16) float32 {
	maxDimension := int32(0)
	for _, g := range strike.Metrics(glyphIDs, make([]*font.Glyph, 0, len(glyphIDs))) {
		maxDimension = max(maxDimension, g.MaxDimension())
	}
	return float32(maxDimension)
}

// absF32 is a float32 abs without the float64 round trip.
func absF32(v float32) float32 { return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)) }

// differentialAreaScale returns the absolute value of the determinant of the Jacobian of the projected map at p,
// computed in float64 for precision.
func differentialAreaScale(m *geom.Matrix, p geom.Point) float32 {
	x := m.Get(geom.MScaleX)*p.X + m.Get(geom.MSkewX)*p.Y + m.Get(geom.MTransX)
	y := m.Get(geom.MSkewY)*p.X + m.Get(geom.MScaleY)*p.Y + m.Get(geom.MTransY)
	w := m.Get(geom.MPersp0)*p.X + m.Get(geom.MPersp1)*p.Y + m.Get(geom.MPersp2)
	if w < geom.ScalarNearlyZeroTol {
		// Reaching the discontinuity of xy/w and where the point would clip to w >= 0.
		return float32(math.Inf(1))
	}
	a00, a01, a02 := float64(x), float64(y), float64(w)
	a10, a11, a12 := float64(m.Get(geom.MScaleX)), float64(m.Get(geom.MSkewY)), float64(m.Get(geom.MPersp0))
	a20, a21, a22 := float64(m.Get(geom.MSkewX)), float64(m.Get(geom.MScaleY)), float64(m.Get(geom.MPersp1))
	det := a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20)
	denom := 1.0 / float64(w) // 1/w
	denom = denom * denom * denom
	return absF32(float32(det * denom))
}

// approximateTransformedTextSize estimates the device-space text size of f under matrix at textLocation.
func approximateTransformedTextSize(f *font.Font, matrix *geom.Matrix, textLocation geom.Point) float32 {
	if !matrix.HasPerspective() {
		return f.Size() * matrix.MaxScale()
	}
	// Approximate the scale since we can't get it directly from the matrix.
	maxScaleSq := differentialAreaScale(matrix, textLocation)
	if !math.IsInf(float64(maxScaleSq), 0) && !math.IsNaN(float64(maxScaleSq)) &&
		!geom.ScalarNearlyZero(maxScaleSq) {
		return f.Size() * float32(math.Sqrt(float64(maxScaleSq)))
	}
	return -f.Size()
}

func floorScalar(v float32) float32 { return float32(math.Floor(float64(v))) }

// addMultiMaskFormat splits the accepted glyphs into runs of a single atlas mask format.
func addMultiMaskFormat(accepted []acceptedGlyph, addSameFormat func(accepted []acceptedGlyph, format gpu.MaskFormat)) {
	if len(accepted) == 0 {
		return
	}
	format := FormatFromGlyph(accepted[0].format)
	startIndex := 0
	for i := 1; i < len(accepted); i++ {
		nextFormat := FormatFromGlyph(accepted[i].format)
		if format != nextFormat {
			addSameFormat(accepted[startIndex:i], format)
			format = nextFormat
			startIndex = i
		}
	}
	addSameFormat(accepted[startIndex:], format)
}

// MakeSubRuns converts glyphRunList into a SubRunContainer, staging each run through the direct mask, SDFT, path, and
// transformed-mask lanes as appropriate. The drawable lane is unreachable. control carries the SDFT decision and must
// be non-nil.
func MakeSubRuns(glyphRunList *textblob.GlyphRunList, positionMatrix *geom.Matrix, runPaint *canvas.Paint, deviceProps *font.DeviceProps, control *SubRunControl) *SubRunContainer {
	container := &SubRunContainer{initialPositionMatrix: *positionMatrix}

	maxMaskSize := control.MaxSize()

	maxGlyphRunSize := glyphRunList.MaxGlyphRunSize()
	accepted := make([]acceptedGlyph, 0, maxGlyphRunSize)
	rejected := make([]rejectedGlyph, 0, maxGlyphRunSize)
	sourceGlyphs := make([]uint16, 0, maxGlyphRunSize)
	sourcePositions := make([]geom.Point, 0, maxGlyphRunSize)

	glyphRunListLocation := glyphRunList.SourceBounds.Center()
	scalerPaint := runPaint.ScalerPaint()

	// takeRejects moves the rejects into the source buffers for the next stage.
	takeRejects := func(rejects []rejectedGlyph) {
		sourceGlyphs = sourceGlyphs[:0]
		sourcePositions = sourcePositions[:0]
		for _, r := range rejects {
			sourceGlyphs = append(sourceGlyphs, r.glyphID)
			sourcePositions = append(sourcePositions, r.position)
		}
	}

	for runIdx := range glyphRunList.Runs {
		glyphRun := &glyphRunList.Runs[runIdx]
		runFont := glyphRun.Font

		sourceGlyphs = append(sourceGlyphs[:0], glyphRun.Glyphs...)
		sourcePositions = append(sourcePositions[:0], glyphRun.Positions...)

		approximateDeviceTextSize := approximateTransformedTextSize(runFont, positionMatrix,
			glyphRunListLocation)

		// Atlas mask cases — SDFT and direct mask. Only consider using direct or SDFT drawing if not drawing hairlines
		// and not too big.
		if (runPaint.Style != canvas.StyleStroke || runPaint.StrokeWidth != 0) &&
			approximateDeviceTextSize < maxMaskSize {
			// SDFT case (E.1) — this should be the .009% case.
			if control.IsSDFT(approximateDeviceTextSize, runPaint, positionMatrix) {
				spec, strikeToSourceScale, matrixRange := makeSDFTStrikeSpec(runFont,
					&scalerPaint, positionMatrix, glyphRunListLocation, control)

				if !geom.ScalarNearlyZero(strikeToSourceScale) {
					strike := spec.FindOrCreateStrike()

					// The creationMatrix needs to scale the strike data when inverted and multiplied by the
					// positionMatrix. The final CTM should be [positionMatrix][scale by strikeToSourceScale], which
					// equals [positionMatrix][creationMatrix]^-1 through the transform during the vertex calculation,
					// so the creation matrix is [scale by 1/strikeToSourceScale].
					var creationMatrix geom.Matrix
					creationMatrix.SetScale(1/strikeToSourceScale, 1/strikeToSourceScale)

					accepted = accepted[:0]
					rejected = rejected[:0]
					var creationBounds geom.Rect
					accepted, rejected, creationBounds = prepareForSDFTDrawing(strike,
						&creationMatrix, sourceGlyphs, sourcePositions, accepted, rejected)
					takeRejects(rejected)

					if len(accepted) > 0 {
						cm := creationMatrix
						container.subRuns = append(container.subRuns, makeSDFTSubRun(runFont,
							accepted, &cm, creationBounds, matrixRange, strike, &spec))
					}
				}
			}

			// Direct Mask case. (The 3D/emboss mask-filter auto-layer check is unreachable: this codebase has no 3D
			// mask format.)
			if len(sourceGlyphs) > 0 && !positionMatrix.HasPerspective() {
				// Process masks including ARGB — this should be the 99.99% case.
				spec := font.MakeMaskSpec(runFont, &scalerPaint, positionMatrix, deviceProps)
				strike := spec.FindOrCreateStrike()

				accepted = accepted[:0]
				rejected = rejected[:0]
				var creationBounds geom.Rect
				accepted, rejected, creationBounds = prepareForDirectMaskDrawing(strike,
					positionMatrix, sourceGlyphs, sourcePositions, accepted, rejected)
				takeRejects(rejected)

				addMultiMaskFormat(accepted,
					func(sameFormat []acceptedGlyph, format gpu.MaskFormat) {
						container.subRuns = append(container.subRuns, makeDirectMaskSubRun(
							creationBounds, sameFormat, container.InitialPosition(), strike,
							&spec, format,
						))
					})
			}
		}

		// (Drawable case trimmed — no reachable scaler produces drawable glyphs.)

		// Path case: mainly large or large-perspective glyphs with no color.
		if len(sourceGlyphs) > 0 {
			spec, strikeToSourceScale := font.MakePathSpec(runFont, &scalerPaint)
			if !geom.ScalarNearlyZero(strikeToSourceScale) {
				strike := spec.FindOrCreateStrike()

				accepted = accepted[:0]
				rejected = rejected[:0]
				accepted, rejected = prepareForPathDrawing(strike, sourceGlyphs,
					sourcePositions, accepted, rejected)
				takeRejects(rejected)

				if len(accepted) > 0 {
					isAntiAliased := control.ForcePathAA() || runFont.HasSomeAntiAliasing()
					container.subRuns = append(container.subRuns, &PathSubRun{
						pathDrawing: makePathOpSubmitter(accepted, isAntiAliased,
							strikeToSourceScale, strike),
					})
				}
			}
		}

		// Drawing of last resort: draw the rest of the rejects by scaling out of the atlas onto the screen — quality
		// will suffer. Mainly large color or perspective color glyphs.
		if len(sourceGlyphs) > 0 && !geom.ScalarNearlyZero(approximateDeviceTextSize) {
			// The creation matrix is conditioned so that there is no perspective (the strikes can't handle perspective
			// masks) and the maximum glyph dimension fits the atlas.
			creationMatrix := *positionMatrix

			if creationMatrix.HasPerspective() {
				// Find a scale factor that reduces pixelation caused by keystoning.
				center := glyphRunList.SourceBounds.Center()
				maxAreaScale := differentialAreaScale(&creationMatrix, center)
				perspectiveFactor := float32(1)
				if !math.IsInf(float64(maxAreaScale), 0) && !math.IsNaN(float64(maxAreaScale)) &&
					!geom.ScalarNearlyZero(maxAreaScale) {
					perspectiveFactor = float32(math.Sqrt(float64(maxAreaScale)))
				}
				creationMatrix = geom.Matrix{}
				creationMatrix.SetScale(perspectiveFactor, perspectiveFactor)
			}

			// Reduce to make a one pixel border for the bilerp padding.
			const maxBilerpAtlasDimension = font.SideTooBigForAtlas - 2

			maxGlyphDimension := func(m *geom.Matrix) float32 {
				gaugingSpec := font.MakeTransformMaskSpec(runFont, &scalerPaint, m, deviceProps)
				gaugingStrike := gaugingSpec.FindOrCreateStrike()
				return findMaximumGlyphDimension(gaugingStrike, sourceGlyphs)
			}

			// Condition the creationMatrix so that glyphs fit in the atlas.
			for maxDimension := maxGlyphDimension(&creationMatrix); maxBilerpAtlasDimension < maxDimension; maxDimension = maxGlyphDimension(&creationMatrix) {
				reductionFactor := maxBilerpAtlasDimension / maxDimension
				creationMatrix.PostScale(reductionFactor, reductionFactor)
			}

			// Draw using the creationMatrix.
			spec := font.MakeTransformMaskSpec(runFont, &scalerPaint, &creationMatrix, deviceProps)
			strike := spec.FindOrCreateStrike()

			accepted = accepted[:0]
			rejected = rejected[:0]
			var creationBounds geom.Rect
			accepted, rejected, creationBounds = prepareForMaskDrawing(strike, &creationMatrix,
				sourceGlyphs, sourcePositions, accepted, rejected)
			takeRejects(rejected)

			cm := creationMatrix
			addMultiMaskFormat(accepted, func(sameFormat []acceptedGlyph, format gpu.MaskFormat) {
				container.subRuns = append(container.subRuns, makeTransformedMaskSubRun(
					sameFormat, container.InitialPosition(), strike, &spec, &cm,
					creationBounds, format,
				))
			})
		}
	}

	return container
}
