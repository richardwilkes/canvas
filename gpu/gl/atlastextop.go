// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The batched draw op that renders atlas subruns as indexed quads through the bitmap text geometry processor — or, for
// SDFT subruns, the distance-field GPs — regenerating the glyph atlas as it prepares and interleaving inline uploads
// with its draws under the deferred-upload token protocol (see OpFlushState.IssueDrawToken). Trims: the color-space
// xform for color emoji is identity, since color emoji and the destination are both sRGB; glyph allocation is ordinary
// Go allocation; useGammaCorrectDistanceTable is constant false — sRGB destinations only. The LCD coverage type (A565
// atlas) is live.

package gl

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/text"
	"github.com/richardwilkes/canvas/surface"
)

// Vertex/index counts per glyph quad.
const (
	verticesPerGlyph = 4
	indicesPerGlyph  = 6
)

// atlasTextMaskType selects which atlas format and shading lane a glyph draw uses (all six members live as of E.1).
type atlasTextMaskType uint8

const (
	maskTypeGrayscaleCoverage atlasTextMaskType = iota
	maskTypeLCDCoverage
	maskTypeColorBitmap
	maskTypeAliasedDistanceField
	maskTypeGrayscaleDistanceField
	maskTypeLCDDistanceField
)

// opMaskType maps a glyph's atlas mask format to the corresponding non-SDF mask type.
func opMaskType(maskFormat gpu.MaskFormat) atlasTextMaskType {
	switch maskFormat {
	case gpu.MaskFormatA8:
		return maskTypeGrayscaleCoverage
	case gpu.MaskFormatA565:
		return maskTypeLCDCoverage
	case gpu.MaskFormatARGB:
		return maskTypeColorBitmap
	}
	panic("unreachable mask format")
}

// maskFormat is the inverse mapping: the distance-field types all ride the A8 atlas.
func (t atlasTextMaskType) maskFormat() gpu.MaskFormat {
	switch t {
	case maskTypeLCDCoverage:
		return gpu.MaskFormatA565
	case maskTypeColorBitmap:
		return gpu.MaskFormatARGB
	default:
		return gpu.MaskFormatA8
	}
}

// usesDistanceFields reports whether this mask type is one of the SDF lanes.
func (t atlasTextMaskType) usesDistanceFields() bool {
	return t == maskTypeAliasedDistanceField || t == maskTypeGrayscaleDistanceField ||
		t == maskTypeLCDDistanceField
}

// isLCD reports whether this mask type renders through the LCD coverage lane.
func (t atlasTextMaskType) isLCD() bool {
	return t == maskTypeLCDCoverage || t == maskTypeLCDDistanceField
}

// atlasTextGeometry is one subrun's per-draw state within a batched AtlasTextOp.
type atlasTextGeometry struct {
	subRun     text.AtlasSubRun
	drawMatrix geom.Matrix
	drawOrigin geom.Point
	// clipRect is only used in the DirectMaskSubRun case to clip geometrically; the transformed case expects an empty
	// rect.
	clipRect geom.IRect
	// color is updated after processor analysis if the shader resolves to a constant.
	color colorcore.PMColor4f
}

// fillVertexData writes count glyphs' worth of vertex data for this geometry, starting at offset.
func (g *atlasTextGeometry) fillVertexData(dst []byte, offset, count int) {
	g.subRun.FillVertexData(dst, offset, count, g.color, &g.drawMatrix, g.drawOrigin, g.clipRect)
}

// atlasTextDraw is one recorded draw's mesh.
type atlasTextDraw struct {
	mesh simpleMesh
}

// atlasTextGeoProcessor is the surface AtlasTextOp needs from its geometry processor, shared by the bitmap text
// geometry processor and the distance-field GPs.
type atlasTextGeoProcessor interface {
	GeometryProcessor
	VertexStride() int
	NumTextureSamplers() int
	AddNewViews(views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState)
}

// AtlasTextOp draws one or more atlas-backed glyph subruns, batched together when possible.
type AtlasTextOp struct {
	drawOpNoClipToShape
	// Flush-time state (prepare fills, execute consumes).
	gp            atlasTextGeoProcessor
	processors    *ProcessorSet
	views         *[gpu.MaxMultitexturePages]SurfaceProxyView
	geometries    []*atlasTextGeometry
	recordedDraws []atlasTextDraw
	OpBase
	numGlyphs int // Sum of glyphs in each geometry's subrun.
	// luminanceColor is the paint's luminance color (distance-field ops only).
	luminanceColor colorcore.Color
	numActiveViews uint32
	// dfgpFlags holds the distance-field GP flags (distance-field ops only).
	dfgpFlags           uint32
	needsGlyphTransform bool
	hasPerspective      bool // True if perspective affects the draw.
	// useGammaCorrectDistanceTable is constant false (sRGB destinations are never linearly blended), carried for the
	// merge check's shape.
	useGammaCorrectDistanceTable bool
	usesLocalCoords              bool // Filled in post processor analysis.
	maskType                     atlasTextMaskType
}

var atlasTextOpClassID = GenOpClassID()

// atlasTextFlushInfo tracks the vertex/index buffers and glyph count for the draw currently being assembled (the
// geometry-processor and proxy state live on the op itself; see the token-protocol note in OnPrepare).
type atlasTextFlushInfo struct {
	vertexBuffer  AnyBuffer
	indexBuffer   *Buffer
	glyphsToFlush int
	vertexOffset  int
}

// clipMethod is how a glyph draw's clip should be applied.
type clipMethod int32

const (
	kClippedOut clipMethod = iota
	kUnclipped
	kGPUClipped
	kGeometryClipped
)

// calculateClip decides how (and whether) the clip needs to be applied to a glyph draw whose device bounds and
// untransformed glyph bounds are given.
func calculateClip(clip Clip, deviceBounds, glyphBounds geom.Rect) (clipMethod, geom.IRect) {
	if clip == nil {
		if !deviceBounds.Intersects(glyphBounds) {
			return kClippedOut, geom.IRect{}
		}
		return kGPUClipped, geom.IRect{}
	}
	result := clip.PreApply(glyphBounds, gpu.AANo)
	switch result.Effect {
	case ClipEffectClippedOut:
		return kClippedOut, geom.IRect{}
	case ClipEffectUnclipped:
		return kUnclipped, geom.IRect{}
	case ClipEffectClipped:
		if result.IsRRect && result.RRect.Type == geom.RRectRect {
			r := result.RRect.Rect
			if result.AA == gpu.AANo || IsPixelAligned(r) {
				// Clip geometrically during onPrepare using clipRect.
				clipRect := r.Round()
				if clipRect.ContainsRect(glyphBounds.RoundOut()) {
					// If fully within the clip, signal no clipping using the empty rect.
					return kUnclipped, geom.IRect{}
				}
				// Use the clipRect to clip the geometry.
				return kGeometryClipped, clipRect
			}
			// Partial pixel clipped at this point. Have the GPU handle it.
		}
	}
	return kGPUClipped, geom.IRect{}
}

// calculateColors resolves the paint into a device-ready Paint and drawing color for this mask format; color-bitmap
// glyphs ignore the paint's shader and only use its alpha.
func calculateColors(sdc *SurfaceDrawContext, paint *canvas.Paint, matrix *geom.Matrix, maskFormat gpu.MaskFormat) (*Paint, colorcore.PMColor4f, bool) {
	pp := PaintParams{
		Color:          colorcore.Color4fFromColor(paint.Color),
		BlendMode:      paint.BlendMode,
		Shader:         paint.Shader,
		ColorFilter:    paint.ColorFilter,
		Dither:         paint.Dither,
		HasMaskFilter:  paint.MaskFilter != nil,
		HasImageFilter: paint.ImageFilter != nil,
	}
	if maskFormat == gpu.MaskFormatARGB {
		// Ignore the paint's shader and use its color; the texture supplies the actual color.
		gpuPaint, ok := makePaintImpl(sdc, &pp, *matrix, nil, true, false, 0)
		if !ok {
			return nil, colorcore.PMColor4f{}, false
		}
		a := gpuPaint.Color4f().A
		return gpuPaint, colorcore.PMColor4f{R: a, G: a, B: a, A: a}, true
	}
	gpuPaint, ok := MakePaint(sdc, &pp, *matrix)
	if !ok {
		return nil, colorcore.PMColor4f{}, false
	}
	color := gpuPaint.Color4f()
	return gpuPaint, color, true
}

// calculateSDFParameters computes the mask type and DF GP flags for an SDFT subrun draw. useGammaCorrectDistanceTable
// is constant false — sRGB destinations only — so it is folded away here.
func calculateSDFParameters(sdc *SurfaceDrawContext, drawMatrix *geom.Matrix, useLCDText, isAntiAliased bool) (maskType atlasTextMaskType, dfgpFlags uint32) {
	props := sdc.SurfaceProps()
	isLCD := useLCDText && props.PixelGeometry != surface.PixelGeometryUnknown
	switch {
	case !isAntiAliased:
		maskType = maskTypeAliasedDistanceField
	case isLCD:
		maskType = maskTypeLCDDistanceField
	default:
		maskType = maskTypeGrayscaleDistanceField
	}

	if drawMatrix.IsSimilarity() {
		dfgpFlags |= similarityDistanceFieldEffectFlag
	}
	if drawMatrix.IsScaleTranslate() {
		dfgpFlags |= scaleOnlyDistanceFieldEffectFlag
	}
	if maskType == maskTypeAliasedDistanceField {
		dfgpFlags |= aliasedDistanceFieldEffectFlag
	}
	if drawMatrix.HasPerspective() {
		dfgpFlags |= perspectiveDistanceFieldEffectFlag
	}

	if isLCD {
		dfgpFlags |= useLCDDistanceFieldEffectFlag
		if props.PixelGeometry == surface.PixelGeometryBGRH ||
			props.PixelGeometry == surface.PixelGeometryBGRV {
			dfgpFlags |= bgrDistanceFieldEffectFlag
		}
		if props.PixelGeometry == surface.PixelGeometryRGBV ||
			props.PixelGeometry == surface.PixelGeometryBGRV {
			dfgpFlags |= portraitDistanceFieldEffectFlag
		}
	}
	return maskType, dfgpFlags
}

// positionMatrixFor returns drawMatrix pre-translated by drawOrigin.
func positionMatrixFor(drawMatrix *geom.Matrix, drawOrigin geom.Point) geom.Matrix {
	positionMatrix := *drawMatrix
	positionMatrix.PreTranslate(drawOrigin.X, drawOrigin.Y)
	return positionMatrix
}

// MakeAtlasTextOp builds an op to draw subRun. It returns the clip to draw with (nil when the geometric clip absorbed
// it) and the op (nil means skip the draw entirely).
func MakeAtlasTextOp(sdc *SurfaceDrawContext, subRun text.AtlasSubRun, clip Clip, viewMatrix *geom.Matrix, drawOrigin geom.Point, paint *canvas.Paint) (Clip, DrawOp) {
	if subRun.GlyphCount() == 0 {
		panic("empty subrun")
	}
	positionMatrix := positionMatrixFor(viewMatrix, drawOrigin)

	needsTransform, subRunDeviceBounds := subRun.DeviceRectAndNeedsTransform(&positionMatrix)
	if subRunDeviceBounds.IsEmpty() {
		return nil, nil
	}

	geometricClipRect := geom.IRect{}
	if !needsTransform {
		// We can clip geometrically using clipRect and ignore clip when an axis-aligned rectangular non-AA clip is
		// used. If clipRect is empty, and clip is nil, then there is no clipping needed.
		dims := sdc.Dimensions()
		deviceBounds := geom.RectWH(float32(dims.Width), float32(dims.Height))
		method, clipRect := calculateClip(clip, deviceBounds, subRunDeviceBounds)
		switch method {
		case kClippedOut:
			// Returning a nil op means skip this op.
			return nil, nil
		case kUnclipped, kGeometryClipped:
			// GPU clip is not needed.
			clip = nil
		case kGPUClipped:
			// Use the GPU clip; clipRect is ignored.
		}
		geometricClipRect = clipRect
	}

	gpuPaint, drawingColor, ok := calculateColors(sdc, paint, viewMatrix, subRun.MaskFormat())
	if !ok {
		return nil, nil
	}

	geometry := &atlasTextGeometry{
		subRun:     subRun,
		drawMatrix: *viewMatrix,
		drawOrigin: drawOrigin,
		clipRect:   geometricClipRect,
		color:      drawingColor,
	}

	var op *AtlasTextOp
	if glyphParams := subRun.GlyphParams(); glyphParams.IsSDF {
		maskType, dfgpFlags := calculateSDFParameters(sdc, viewMatrix, glyphParams.IsLCD,
			glyphParams.IsAA)
		op = &AtlasTextOp{
			processors:          NewProcessorSetFromPaint(gpuPaint),
			geometries:          []*atlasTextGeometry{geometry},
			numGlyphs:           subRun.GlyphCount(),
			dfgpFlags:           dfgpFlags,
			maskType:            maskType,
			needsGlyphTransform: true,
			hasPerspective:      geometry.drawMatrix.HasPerspective(),
			luminanceColor:      paint.LuminanceColor(),
		}
	} else {
		op = &AtlasTextOp{
			processors:          NewProcessorSetFromPaint(gpuPaint),
			geometries:          []*atlasTextGeometry{geometry},
			numGlyphs:           subRun.GlyphCount(),
			maskType:            opMaskType(subRun.MaskFormat()),
			needsGlyphTransform: needsTransform,
			hasPerspective:      needsTransform && geometry.drawMatrix.HasPerspective(),
		}
	}
	op.InitOp(atlasTextOpClassID, op)
	// We don't have tight bounds on the glyph paths in device space. For the purposes of bounds we treat this as a set
	// of non-AA rects rendered with a texture.
	op.SetBounds(subRunDeviceBounds, HasAABloat(false), IsHairline(false))
	return clip, op
}

// Name implements Op.
func (o *AtlasTextOp) Name() string { return "AtlasTextOp" }

// VisitProxies implements Op. The atlas proxies are added to the sampled-proxy array during prepare instead.
func (o *AtlasTextOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	o.processors.VisitProxies(fn)
}

// UsesMSAA implements DrawOp (fixedFunctionFlags returns kNone).
func (o *AtlasTextOp) UsesMSAA() bool { return false }

// UsesStencil implements DrawOp.
func (o *AtlasTextOp) UsesStencil() bool { return false }

// Finalize implements DrawOp.
func (o *AtlasTextOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	var color ProcessorAnalysisColor
	var coverage AnalysisCoverage
	if o.maskType == maskTypeColorBitmap {
		color = AnalysisColorUnknown()
	} else {
		// finalize() is called before any merging is done, so at this point there's at most one geometry with a color.
		color = AnalysisColorConstant(o.geometries[0].color)
	}
	switch o.maskType {
	case maskTypeGrayscaleCoverage, maskTypeAliasedDistanceField,
		maskTypeGrayscaleDistanceField:
		coverage = AnalysisCoverageSingleChannel
	case maskTypeLCDCoverage, maskTypeLCDDistanceField:
		coverage = AnalysisCoverageLCD
	case maskTypeColorBitmap:
		coverage = AnalysisCoverageNone
	}
	analysis, overrideColor := o.processors.Finalize(color, coverage, clip,
		UnusedStencilSettings(), caps, clampType)
	if analysis.InputColorIsOverridden() {
		o.geometries[0].color = overrideColor
	}
	o.usesLocalCoords = analysis.UsesLocalCoords()
	return analysis
}

// OnCombineIfPossible implements Op.
func (o *AtlasTextOp) OnCombineIfPossible(t Op) CombineResult {
	that, isSame := t.(*AtlasTextOp)
	if !isSame {
		return CombineResultCannotCombine
	}
	if o.dfgpFlags != that.dfgpFlags ||
		o.maskType != that.maskType ||
		o.usesLocalCoords != that.usesLocalCoords ||
		o.needsGlyphTransform != that.needsGlyphTransform ||
		o.hasPerspective != that.hasPerspective ||
		o.useGammaCorrectDistanceTable != that.useGammaCorrectDistanceTable {
		// All flags must match for an op to be combined.
		return CombineResultCannotCombine
	}
	if !o.processors.Equal(that.processors) {
		return CombineResultCannotCombine
	}
	if o.usesLocalCoords {
		// If the fragment processors use local coordinates, the GPs compute them using the inverse of the view matrix
		// stored in a uniform, so all geometries must have the same matrix.
		if !o.geometries[0].drawMatrix.CheapEqual(&that.geometries[0].drawMatrix) {
			return CombineResultCannotCombine
		}
	}
	if o.maskType.usesDistanceFields() {
		if o.luminanceColor != that.luminanceColor {
			return CombineResultCannotCombine
		}
	} else if o.maskType == maskTypeColorBitmap &&
		o.geometries[0].color != that.geometries[0].color {
		// This ensures all merged bitmap color text ops have a constant color.
		return CombineResultCannotCombine
	}
	o.numGlyphs += that.numGlyphs
	o.geometries = append(o.geometries, that.geometries...)
	that.geometries = nil
	return CombineResultMerged
}

// regenerateAtlasForGL resolves the GPU glyphs, then places [begin, end) in the atlas, updating use tokens as it goes.
func regenerateAtlasForGL(gv *text.GlyphVector, begin, end int, maskFormat gpu.MaskFormat, srcPadding int, state *OpFlushState) (ok bool, glyphsPlaced int) {
	atlasManager := state.AtlasManager()
	currentAtlasGen := atlasManager.AtlasGeneration(maskFormat)

	gv.PackedGlyphIDToGlyph(state.TextStrikeCache())

	if gv.AtlasGeneration() != currentAtlasGen {
		// Calculate the texture coordinates for the vertexes during first use (the generation is invalid) or after the
		// atlas changed in subsequent calls.
		gv.BulkUseUpdater().Reset()

		// Re-resolve the CPU strike from the spec to generate mask images on demand.
		strike := gv.Strike().Spec().FindOrCreateStrike()

		tokenTracker := state.TokenTracker()
		variants := gv.Variants()[begin:end]
		glyphsPlacedInAtlas := 0
		success := true
		for i := range variants {
			gpuGlyph := variants[i].Glyph
			if gpuGlyph == nil {
				panic("unresolved gpu glyph")
			}
			if !atlasManager.HasGlyph(maskFormat, gpuGlyph) {
				skGlyph := strike.PrepareImage(gpuGlyph.PackedID)
				code := atlasManager.AddGlyphToAtlas(skGlyph, gpuGlyph, srcPadding,
					state.ResourceProvider(), state)
				if code != AtlasErrorCodeSucceeded {
					success = code != AtlasErrorCodeError
					break
				}
			}
			atlasManager.AddGlyphToBulkAndSetUseToken(gv.BulkUseUpdater(), maskFormat, gpuGlyph,
				tokenTracker.NextDrawToken())
			glyphsPlacedInAtlas++
		}

		// Update the atlas generation if there are no more glyphs to put in the atlas.
		if success && begin+glyphsPlacedInAtlas == gv.GlyphCount() {
			// Need to get the freshest value of the atlas' generation because updateTextureCoordinates may have changed
			// it.
			gv.SetAtlasGeneration(atlasManager.AtlasGeneration(maskFormat))
		}
		return success, glyphsPlacedInAtlas
	}

	// The atlas hasn't changed, so our texture coordinates are still valid.
	if end == gv.GlyphCount() {
		// The atlas hasn't changed and the texture coordinates are all still valid. Update all the plots used to the
		// new use token.
		atlasManager.SetUseTokenBulk(gv.BulkUseUpdater(),
			state.TokenTracker().NextDrawToken(), maskFormat)
	}
	return true, end - begin
}

// OnPrepare implements Op. Unlike the C++, which records mesh entries into the flush state for central replay, this
// records each generated mesh on the op and issues one draw token per mesh; OnExecute replays them under the same
// deferred-upload token protocol (see OpFlushState.IssueDrawToken).
func (o *AtlasTextOp) OnPrepare(state *OpFlushState) {
	// If we need local coordinates, compute an inverse view matrix. If this is a solid color, the processor analysis
	// will not require local coords and the GP will skip local coords when the matrix is identity. When the shaders
	// require local coords, combineIfPossible requires all geometries to have the same draw matrix.
	localMatrix := geom.IdentityMatrix()
	if o.usesLocalCoords {
		inverted, ok := o.geometries[0].drawMatrix.Invert()
		if !ok {
			return
		}
		localMatrix = inverted
	}

	atlasManager := state.AtlasManager()
	maskFormat := o.maskType.maskFormat()

	views, numActiveViews := atlasManager.GetViews(maskFormat)
	if views == nil {
		// Could not allocate the backing texture for the atlas.
		return
	}
	o.views = views
	o.numActiveViews = numActiveViews

	// This op does not know its atlas proxies when it is added to an OpsTask, so the proxies don't get added during the
	// visitProxies call. Thus we add them here.
	for i := uint32(0); i < numActiveViews; i++ {
		*state.SampledProxyArray() = append(*state.SampledProxyArray(), views[i].Proxy())
	}

	flushInfo := atlasTextFlushInfo{
		indexBuffer: state.ResourceProvider().RefNonAAQuadIndexBuffer(),
	}
	if flushInfo.indexBuffer == nil {
		// The context was abandoned or the shared quad index buffer could not be created; skip the draw rather than
		// crashing when createDrawForGeneratedGlyphs sizes the pattern from it.
		return
	}

	if o.maskType.usesDistanceFields() {
		o.gp = o.setupDfProcessor(state.Caps().ShaderCaps, &localMatrix, views[:],
			int(numActiveViews))
	} else {
		filter := bitmapTextFilterFromNeedsTransform(o.needsGlyphTransform)
		// Bitmap text uses a single color; combineIfPossible ensures all geometries have the same color, so we can use
		// the first's without worry.
		o.gp = newBitmapTextGeoProc(state.Caps().ShaderCaps, o.geometries[0].color,
			views[:], int(numActiveViews), filter, maskFormat, &localMatrix, o.hasPerspective)
	}

	vertexStride := o.gp.VertexStride()

	// Ensure we don't request an insanely large contiguous vertex allocation.
	const maxVertexBytes = DefaultBufferSize
	quadSize := vertexStride * verticesPerGlyph
	maxQuadsPerBuffer := maxVertexBytes / quadSize

	allGlyphsCursor := 0
	allGlyphsEnd := o.numGlyphs
	var quadCursor, quadEnd int
	var vertices []byte

	resetVertexBuffer := func() bool {
		quadCursor = 0
		quadEnd = min(maxQuadsPerBuffer, allGlyphsEnd-allGlyphsCursor)
		var vertexBuffer AnyBuffer
		var startVertex int
		vertices, vertexBuffer, startVertex = state.MakeVertexSpace(uint64(vertexStride),
			verticesPerGlyph*quadEnd)
		flushInfo.vertexBuffer = vertexBuffer
		flushInfo.vertexOffset = startVertex
		return vertices != nil && vertexBuffer != nil
	}

	if !resetVertexBuffer() {
		return
	}

	for _, geo := range o.geometries {
		subRun := geo.subRun
		if subRun.VertexStride(&geo.drawMatrix) != vertexStride {
			panic("subrun vertex stride mismatch")
		}
		subRunEnd := subRun.GlyphCount()
		regenerateDelegate := func(glyphs *text.GlyphVector, begin, end int,
			mf gpu.MaskFormat, padding int,
		) (bool, int) {
			return regenerateAtlasForGL(glyphs, begin, end, mf, padding, state)
		}
		for subRunCursor := 0; subRunCursor < subRunEnd; {
			// Regenerate the atlas for the remainder of the glyphs in the run, or the remainder of the glyphs that fit
			// the vertex buffer.
			regenEnd := subRunCursor + min(subRunEnd-subRunCursor, quadEnd-quadCursor)
			ok, glyphsRegenerated := subRun.RegenerateAtlas(subRunCursor, regenEnd,
				regenerateDelegate)
			// There was a problem allocating the glyph in the atlas. Bail.
			if !ok {
				return
			}

			geo.fillVertexData(vertices[quadCursor*quadSize:], subRunCursor, glyphsRegenerated)

			subRunCursor += glyphsRegenerated
			quadCursor += glyphsRegenerated
			allGlyphsCursor += glyphsRegenerated
			flushInfo.glyphsToFlush += glyphsRegenerated

			if quadCursor == quadEnd || subRunCursor < subRunEnd {
				// Flush if not all the glyphs are drawn because either the quad buffer is full or the atlas is out of
				// space.
				o.createDrawForGeneratedGlyphs(state, &flushInfo)
				if quadCursor == quadEnd && allGlyphsCursor < allGlyphsEnd {
					// If the vertex buffer is full and there are still glyphs to draw then get a new buffer.
					if !resetVertexBuffer() {
						return
					}
				}
			}
		}
	}
}

// createDrawForGeneratedGlyphs records one indexed-patterned draw for the glyphs generated so far (recorded on the op,
// with one draw token issued now and its flush token issued when OnExecute replays it).
func (o *AtlasTextOp) createDrawForGeneratedGlyphs(state *OpFlushState, flushInfo *atlasTextFlushInfo) {
	if flushInfo.glyphsToFlush == 0 {
		return
	}

	atlasManager := state.AtlasManager()
	maskFormat := o.maskType.maskFormat()
	views, numActiveViews := atlasManager.GetViews(maskFormat)
	if views == nil || numActiveViews == 0 {
		// Something has gone terribly wrong; bail.
		return
	}
	if o.gp.NumTextureSamplers() != int(numActiveViews) {
		// During preparation the number of atlas pages has increased. Update the proxies used in the GP to match.
		for i := o.gp.NumTextureSamplers(); i < int(numActiveViews); i++ {
			*state.SampledProxyArray() = append(*state.SampledProxyArray(), views[i].Proxy())
		}
		var filter gpu.SamplerState
		if o.maskType.usesDistanceFields() {
			filter = gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp,
				gpu.FilterModeLinear, gpu.MipmapModeNone)
		} else {
			filter = bitmapTextFilterFromNeedsTransform(o.needsGlyphTransform)
		}
		o.gp.AddNewViews(views[:], int(numActiveViews), filter)
	}
	o.views = views
	o.numActiveViews = numActiveViews

	maxGlyphsPerDraw := int(flushInfo.indexBuffer.Size()) / 2 / indicesPerGlyph
	var draw atlasTextDraw
	draw.mesh.setIndexedPatterned(flushInfo.indexBuffer, indicesPerGlyph, verticesPerGlyph,
		flushInfo.glyphsToFlush, maxGlyphsPerDraw, flushInfo.vertexBuffer,
		flushInfo.vertexOffset)
	state.IssueDrawToken()
	o.recordedDraws = append(o.recordedDraws, draw)

	flushInfo.vertexOffset += verticesPerGlyph * flushInfo.glyphsToFlush
	flushInfo.glyphsToFlush = 0
}

// setupDfProcessor builds the distance-field GP for this op's mask type. The grayscale/aliased lanes take the
// flags-only constructor form; only the LCD lane consults the distance-field adjustment table, and always its
// non-gamma-correct table (sRGB destinations are never linearly blended — see calculateSDFParameters).
func (o *AtlasTextOp) setupDfProcessor(caps *gpu.ShaderCaps, localMatrix *geom.Matrix, views []SurfaceProxyView, numActiveViews int) atlasTextGeoProcessor {
	linear := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeLinear,
		gpu.MipmapModeNone)
	if o.maskType.isLCD() {
		dfAdjustTable := text.GetDistanceFieldAdjustTable()
		widthAdjust := DistanceAdjust{
			R: dfAdjustTable.GetAdjustment(int(o.luminanceColor.R())),
			G: dfAdjustTable.GetAdjustment(int(o.luminanceColor.G())),
			B: dfAdjustTable.GetAdjustment(int(o.luminanceColor.B())),
		}
		return newDistanceFieldLCDTextGeoProc(caps, views, numActiveViews, linear, widthAdjust,
			o.dfgpFlags, localMatrix)
	}
	return newDistanceFieldA8TextGeoProc(caps, views, numActiveViews, linear, o.dfgpFlags,
		localMatrix)
}

// OnExecute implements Op under the deferred-upload token protocol: it processes the pending inline uploads before each
// recorded draw and issues its flush token after, even when the pipeline fails to bind (the token stream must stay
// balanced).
func (o *AtlasTextOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	if len(o.recordedDraws) == 0 {
		return
	}
	args := state.OpArgs()
	pipeline := NewPipeline(&PipelineInitArgs{
		InputFlags:   PipelineInputFlagNone,
		Caps:         state.Caps(),
		DstProxyView: *args.DstProxyView(),
		WriteSwizzle: args.WriteView().Swizzle(),
	}, o.processors, args.AppliedClip())
	o.processors = nil
	programInfo := NewProgramInfo(state.Caps(), args.WriteView(), args.UsesMSAASurface(),
		pipeline, UnusedStencilSettings(), o.gp, gpu.PrimitiveTypeTriangles,
		args.RenderPassBarriers(), args.ColorLoadOp())

	proxies := make([]*SurfaceProxy, o.numActiveViews)
	for i := range proxies {
		proxies[i] = o.views[i].Proxy()
	}

	renderPass := state.OpsRenderPass()
	for i := range o.recordedDraws {
		state.ProcessPendingInlineUploads()
		if renderPass.BindPipeline(programInfo, chainBounds) {
			if programInfo.Pipeline().IsScissorTestEnabled() {
				renderPass.SetScissorRect(args.AppliedClip().ScissorState().Rect())
			}
			renderPass.BindTexturesMulti(programInfo.GeomProc(), proxies,
				programInfo.Pipeline())
			o.recordedDraws[i].mesh.draw(renderPass)
		}
		state.IssueFlushToken()
	}
}

var _ DrawOp = (*AtlasTextOp)(nil)
