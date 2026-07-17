// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// FilterResult: the lazy image anchored in layer space that filter nodes hand each other. A FilterResult is a
// SpecialImage plus deferred sampling, tile mode, transform, color filter, and layer-bounds crop;
// ApplyCrop/ApplyTransform/ApplyColorFilter combine analytically when the deferred-operation order allows it and
// resolve to a new offscreen image otherwise. Color-space handling is dropped, since everything is sRGB.

package filtercore

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// DefaultSampling is bilinear, because it can be downgraded to nearest-neighbor when the final transform is pixel
// aligned.
var DefaultSampling = shaders.SamplingOptions{Filter: shaders.FilterLinear}

// nearestSampling is the nearest-neighbor sampling constant used in analyzeBounds.
var nearestSampling = shaders.SamplingOptions{}

// pixelBoundary records what's known about the pixels bordering a FilterResult's image subset.
type pixelBoundary int32

const (
	// boundaryUnknown: pixels outside the image subset are of unknown value.
	boundaryUnknown pixelBoundary = iota
	// boundaryTransparent: pixels bordering the image subset are transparent black.
	boundaryTransparent
	// boundaryInitialized: pixels bordering the image are known to be initialized.
	boundaryInitialized
)

// boundsAnalysis records which special-case handling a FilterResult draw/shader needs.
type boundsAnalysis int32

const (
	boundsSimple                 boundsAnalysis = 0
	boundsDstNotCovered          boundsAnalysis = 1 << 0
	boundsHasLayerFillingEffect  boundsAnalysis = 1 << 1
	boundsRequiresLayerCrop      boundsAnalysis = 1 << 2
	boundsRequiresShaderTiling   boundsAnalysis = 1 << 3
	boundsRequiresDecalInLayerSp boundsAnalysis = 1 << 4
)

// boundsScope selects how thorough analyzeBounds should be, trading precision for speed depending on how the result
// will be consumed.
type boundsScope int32

const (
	scopeDeferred boundsScope = iota
	scopeCanDrawDirectly
	scopeShaderOnly
	scopeRescale
)

// ShaderFlags hints asShader about how the consuming shader will sample its input.
type ShaderFlags int32

// ShaderFlags values.
const (
	ShaderFlagNone ShaderFlags = 0
	// ShaderFlagSampledRepeatedly hints the input will be sampled repeatedly per pixel, so deferred color filters are
	// worth resolving once.
	ShaderFlagSampledRepeatedly ShaderFlags = 1 << 0
	// ShaderFlagNonTrivialSampling declares the consuming shader samples at non-trivially-mapped coordinates, so
	// pixel-aligned inputs cannot downgrade to nearest-neighbor.
	ShaderFlagNonTrivialSampling ShaderFlags = 1 << 1
)

// FilterResult is the lazy image type filter nodes hand each other. The zero value is the empty (transparent black)
// result.
type FilterResult struct {
	colorFilter shaders.ColorFilter
	image       *SpecialImage
	transform   geom.Matrix // layer-space transform (typically an integer translation)
	sampling    shaders.SamplingOptions
	layerBounds geom.IRect
	boundary    pixelBoundary
	tileMode    shaders.TileMode
}

// NewFilterResult builds a FilterResult wrapping image, anchored at origin in layer space.
func NewFilterResult(image *SpecialImage, origin geom.IPoint) FilterResult {
	return newFilterResult(image, origin, boundaryUnknown)
}

func newFilterResult(image *SpecialImage, origin geom.IPoint, boundary pixelBoundary) FilterResult {
	var transform geom.Matrix
	transform.SetTranslate(float32(origin.X), float32(origin.Y))
	var dims geom.ISize
	if image != nil {
		dims = image.Dimensions()
	}
	return FilterResult{
		image:       image,
		boundary:    boundary,
		sampling:    DefaultSampling,
		tileMode:    shaders.TileDecal,
		transform:   transform,
		layerBounds: mapIRect(&transform, geom.IRectSize(dims)),
	}
}

// IsValid reports whether the result holds a non-transparent image.
func (f *FilterResult) IsValid() bool { return f.image != nil }

// Image returns the backing SpecialImage (nil for the empty result).
func (f *FilterResult) Image() *SpecialImage { return f.image }

// LayerBounds returns the layer-space bounds the result is cropped to.
func (f *FilterResult) LayerBounds() geom.IRect { return f.layerBounds }

// TileMode returns the tile mode applied outside LayerBounds.
func (f *FilterResult) TileMode() shaders.TileMode { return f.tileMode }

// Sampling returns the deferred sampling to apply when drawing the image.
func (f *FilterResult) Sampling() shaders.SamplingOptions { return f.sampling }

// ColorFilter returns the deferred color filter, if any.
func (f *FilterResult) ColorFilter() shaders.ColorFilter { return f.colorFilter }

// ImageAndOffset resolves against the context's desired output and returns the flat image plus its layer-space origin.
func (f *FilterResult) ImageAndOffset(ctx *Context) (*SpecialImage, geom.IPoint) {
	resolved := f.resolve(ctx, ctx.DesiredOutput(), false)
	return resolved.image, geom.IPoint{X: resolved.layerBounds.Left, Y: resolved.layerBounds.Top}
}

// InsetForSaveLayer adjusts metadata for the 1px transparent padding the canvas adds around filtered layers.
func (f *FilterResult) InsetForSaveLayer() FilterResult {
	if f.image == nil {
		return FilterResult{}
	}
	// pixelBoundary tracking assumes the subset does not include the padding, so inset by one pixel and then trust that
	// the canvas kept the padding transparent.
	inset := f.insetByPixel()
	inset.boundary = boundaryTransparent
	return inset
}

// insetByPixel returns a view of the result with the subset inset by one pixel on each side.
func (f *FilterResult) insetByPixel() FilterResult {
	insetBounds := f.layerBounds.Inset(1, 1)
	origin := geom.IPoint{X: f.layerBounds.Left, Y: f.layerBounds.Top}
	return f.subset(origin, insetBounds, false)
}

// analyzeBounds determines which special-case handling (layer crop, shader tiling, layer-space decal, etc.) is needed
// to draw/shade this result correctly over dstBounds, mapped through xtraTransform.
func (f *FilterResult) analyzeBounds(xtraTransform *geom.Matrix, dstBounds geom.IRect, scope boundsScope) boundsAnalysis {
	const halfPixel = 0.5
	const cubicRadius = 1.5

	analysis := boundsSimple
	fillsLayerBounds := f.tileMode != shaders.TileDecal ||
		(f.colorFilter != nil && ColorFilterAffectsTransparentBlack(f.colorFilter))

	// 1. Is the layer geometry visible in dstBounds (ignoring whether shading effects highlight it)?
	pixelCenterBounds := dstBounds.ToRect()
	if !quadContainsIRect(xtraTransform, f.layerBounds, dstBounds, roundEpsilon) {
		// 1a. If an effect doesn't fill out to the layer bounds, is the image content itself clipped by the layer
		//     bounds?
		requireLayerCrop := fillsLayerBounds
		if !fillsLayerBounds {
			imageBounds := mapIRect(&f.transform, geom.IRectSize(f.image.Dimensions()))
			requireLayerCrop = !f.layerBounds.ContainsRect(imageBounds)
		}
		if requireLayerCrop {
			analysis |= boundsRequiresLayerCrop
			// The layer crop applies externally, so restrict the sample bounds to the intersection of dstBounds and
			// layerBounds. A failed intersection leaves pixelCenterBounds as the original dstBounds (complex filter
			// graphs can produce a layer crop that doesn't overlap the sampled region at all).
			layerBoundsInDst := mapIRect(xtraTransform, f.layerBounds)
			_ = pixelCenterBounds.Intersect(layerBoundsInDst.ToRect())
		}
	}

	// 2. Are the tiling and deferred color filter effects visible in the sampled bounds?
	imageBounds := geom.RectWH(float32(f.image.Width()), float32(f.image.Height()))
	netTransform := f.transform
	netTransform.PostConcat(xtraTransform)

	xAxisAligned, yAxisAligned := areAxesNearlyIntegerAligned(&netTransform, nil)
	isPixelAligned := xAxisAligned && yAxisAligned
	// When decal sampling, use inset image bounds for checking dst coverage; an image exactly filling dst could still
	// sample transparent black depending on the scale factor.
	decalLeaks := scope != scopeRescale &&
		f.tileMode == shaders.TileDecal &&
		f.sampling != nearestSampling &&
		!isPixelAligned

	sampleRadius := float32(halfPixel)
	if f.sampling.UseCubic {
		sampleRadius = cubicRadius
	}
	safeImageBounds := imageBounds.Inset(sampleRadius, sampleRadius)
	if f.sampling == DefaultSampling && !isPixelAligned {
		// Default sampling with integer translations downgrades to nearest neighbor, so the 1/2px inset suffices;
		// sticking with linear filtering requires an epsilon inside the 1/2px.
		xInset, yInset := float32(roundEpsilon), float32(roundEpsilon)
		if xAxisAligned {
			xInset = 0
		}
		if yAxisAligned {
			yInset = 0
		}
		safeImageBounds = safeImageBounds.Inset(xInset, yInset)
	}
	hasPixelPadding := f.boundary != boundaryUnknown

	coverageBounds := imageBounds
	if decalLeaks {
		coverageBounds = safeImageBounds
	}
	if !quadContainsRect(&netTransform, coverageBounds, pixelCenterBounds, roundEpsilon) {
		analysis |= boundsDstNotCovered
		if fillsLayerBounds {
			analysis |= boundsHasLayerFillingEffect
		}
		if decalLeaks {
			// Decal tiling will be visible; check the relative texel-to-dst scale, and if it's not close to 1 the decal
			// must be applied in layer space to stay visually consistent.
			minScale, maxScale, ok := minMaxScales(&netTransform)
			if !(ok && nearlyEqual(minScale, 1, 0.2) && nearlyEqual(maxScale, 1, 0.2)) {
				analysis |= boundsRequiresDecalInLayerSp
				if f.boundary == boundaryTransparent {
					// Don't treat the transparent padding as safe, to prevent it multiplying with the layer-space decal
					// effect.
					hasPixelPadding = false
				}
			}
		}
	}

	switch {
	case scope == scopeDeferred:
		return analysis // skip sampling analysis
	case scope == scopeCanDrawDirectly && analysis&boundsHasLayerFillingEffect == 0:
		// When drawing the image directly, the geometry is limited to the image. If the texels are pixel aligned, it's
		// safe to skip shader-based tiling.
		nnOrBilerp := f.sampling == DefaultSampling || f.sampling == nearestSampling
		if nnOrBilerp && (hasPixelPadding || isPixelAligned) {
			return analysis
		}
	}

	// 3. Would image pixels outside its subset be sampled if shader-clamping is skipped?
	if hasPixelPadding {
		safeImageBounds = safeImageBounds.Outset(1, 1)
	}
	pixelCenterBounds = pixelCenterBounds.Inset(halfPixel, halfPixel)

	edgeMask := quadContainsRectMask(&netTransform, safeImageBounds, pixelCenterBounds, roundEpsilon)
	if !edgeMask[0] || !edgeMask[1] || !edgeMask[2] || !edgeMask[3] {
		// Sampling outside the image subset occurs, but exceeded edges that are backing-store edges avoid shader-based
		// tiling (ordered T, R, B, L).
		subset := f.image.Subset()
		store := f.image.BackingStoreDimensions()
		hwEdge := [4]bool{
			subset.Top == 0,
			subset.Right == store.Width,
			subset.Bottom == store.Height,
			subset.Left == 0,
		}
		if f.tileMode == shaders.TileRepeat || f.tileMode == shaders.TileMirror {
			// Periodic tile modes require both edges on an axis to be backing-store edges.
			hwEdge = [4]bool{
				hwEdge[0] && hwEdge[2], hwEdge[1] && hwEdge[3],
				hwEdge[2] && hwEdge[0], hwEdge[3] && hwEdge[1],
			}
		}
		for i := range 4 {
			if !edgeMask[i] && !hwEdge[i] {
				analysis |= boundsRequiresShaderTiling
				break
			}
		}
	}
	return analysis
}

func (f *FilterResult) analyzeBoundsLayer(dstBounds geom.IRect, scope boundsScope) boundsAnalysis {
	identity := geom.IdentityMatrix()
	return f.analyzeBounds(&identity, dstBounds, scope)
}

// canClampToTransparentBoundary reports whether f's decal tiling can be replaced by clamp sampling of a 1px-outset
// image, given that the known transparent boundary makes them equivalent.
func (f *FilterResult) canClampToTransparentBoundary(analysis boundsAnalysis) bool {
	return f.tileMode == shaders.TileDecal &&
		f.boundary == boundaryTransparent &&
		analysis&boundsRequiresDecalInLayerSp == 0
}

// updateTileMode sets the result's tile mode, expanding the layer bounds to the desired output for any non-decal mode.
func (f *FilterResult) updateTileMode(ctx *Context, tileMode shaders.TileMode) {
	if f.image != nil {
		f.tileMode = tileMode
		if tileMode != shaders.TileDecal {
			f.layerBounds = ctx.DesiredOutput()
		}
	}
}

// ApplyCrop returns a new FilterResult cropped to 'crop' (with tiling), reusing the underlying image when possible.
func (f *FilterResult) ApplyCrop(ctx *Context, crop geom.IRect, tileMode shaders.TileMode) FilterResult {
	if crop.IsEmpty() || ctx.DesiredOutput().IsEmpty() {
		// An empty crop cannot be anything other than fully transparent.
		return FilterResult{}
	}

	// First, intersect the layer bounds with the crop: the portion of 'crop' that could have non-transparent content.
	cropContent := crop
	if f.image == nil || !cropContent.Intersect(f.layerBounds) {
		// The pixels within 'crop' would be fully transparent, and tiling won't change that.
		return FilterResult{}
	}

	// Second, the subset of 'crop' relevant to the desired output.
	fittedCrop := RelevantSubset(crop, ctx.DesiredOutput(), tileMode)

	// Third, check overlap between the non-transparent cropped content and what tiles the desired output. This modifies
	// 'cropContent' (not 'fittedCrop') so transparent padding remains for repeat/mirror tiling of the original
	// geometry.
	if !cropContent.Intersect(fittedCrop) {
		return FilterResult{}
	}

	// Fourth, a periodic tiling covering the output with a single instance simplifies to a transform.
	if periodicTransform, ok := periodicAxisTransform(tileMode, fittedCrop, ctx.DesiredOutput()); ok {
		return f.ApplyTransform(ctx, &periodicTransform, DefaultSampling)
	}

	preserveTransparencyInCrop := false
	switch {
	case tileMode == shaders.TileDecal:
		// We can reduce the crop dimensions to what's non-transparent.
		fittedCrop = cropContent
	case fittedCrop.ContainsRect(ctx.DesiredOutput()):
		tileMode = shaders.TileDecal
		fittedCrop = ctx.DesiredOutput()
	case !cropContent.ContainsRect(fittedCrop):
		// There is transparency in fittedCrop that must be resolved to maintain the new tiling geometry.
		preserveTransparencyInCrop = true
		if f.tileMode == shaders.TileDecal && tileMode == shaders.TileClamp {
			// Include a 1px buffer for transparency from the original decal tiling.
			cropContent = cropContent.Inset(-1, -1)
			if !fittedCrop.Intersect(cropContent) {
				panic("fittedCrop must intersect outset cropContent")
			}
		}
	} // else cropContent == fittedCrop

	// Fifth, integer translations can often resolve prior and new tiling analytically without a new image; moving the
	// crop into the image dimensions lets future transforms/color filters compose without an intervening crop.
	doubleClamp := f.tileMode == shaders.TileClamp && tileMode == shaders.TileClamp
	var origin geom.IPoint
	if !preserveTransparencyInCrop &&
		isNearlyIntegerTranslation(&f.transform, &origin) &&
		(doubleClamp || f.analyzeBoundsLayer(fittedCrop, scopeDeferred)&boundsHasLayerFillingEffect == 0) {
		restrictedOutput := f.subset(origin, fittedCrop, doubleClamp)
		restrictedOutput.updateTileMode(ctx, tileMode)
		if restrictedOutput.boundary == boundaryInitialized || tileMode != shaders.TileDecal {
			// Discard boundaryInitialized since a crop is a strict constraint on sampling outside of it, but preserve
			// (boundaryTransparent + decal) for a no-op crop.
			restrictedOutput.boundary = boundaryUnknown
		}
		return restrictedOutput
	}
	if tileMode == shaders.TileDecal {
		// A decal crop can always be applied as the final operation by adjusting layer bounds.
		restrictedOutput := *f
		restrictedOutput.layerBounds = fittedCrop
		return restrictedOutput
	}
	// A non-trivial transform applies to the image data before the non-decal tile mode is meant to be applied to the
	// axis-aligned 'crop'.
	tiled := f.resolve(ctx, fittedCrop, true)
	tiled.updateTileMode(ctx, tileMode)
	return tiled
}

// ApplyColorFilter returns a new FilterResult with cf applied, composing with any deferred color filter already
// present.
func (f *FilterResult) ApplyColorFilter(ctx *Context, cf shaders.ColorFilter) FilterResult {
	if ctx.DesiredOutput().IsEmpty() {
		return FilterResult{}
	}

	// Color filters apply after the transform/sampling but before the layer-bounds crop, so they compose with prior
	// color filters as long as they respect the current crop's effect.
	newLayerBounds := f.layerBounds
	if ColorFilterAffectsTransparentBlack(cf) {
		if f.image == nil || !newLayerBounds.Intersect(ctx.DesiredOutput()) {
			// The image's intersection with the desired output is fully transparent, but the new color filter turns
			// that into a color that floods the desired output; render a 1x1 clamp-tiled surface holding that color.
			surface := newAutoSurface(ctx,
				geom.IRectXYWH(ctx.DesiredOutput().Left, ctx.DesiredOutput().Top, 1, 1),
				boundaryInitialized, false)
			if surface.valid {
				surface.drawPaint(&PaintParams{
					Color:       colorcore.Color4f{},
					ColorFilter: cf,
					Blend:       raster.BlendSrc,
				})
			}
			solidColor := surface.snap()
			solidColor.updateTileMode(ctx, shaders.TileClamp)
			return solidColor
		}

		if f.analyzeBoundsLayer(ctx.DesiredOutput(), scopeDeferred)&boundsRequiresLayerCrop != 0 {
			// The new result's layer bounds must be the desired output, but the current image is cropped: resolve to
			// avoid losing the current layer bounds' effect.
			newLayerBounds = newLayerBounds.Inset(-1, -1)
			if !newLayerBounds.Intersect(ctx.DesiredOutput()) {
				panic("outset layer bounds must intersect desired output")
			}
			filtered := f.resolve(ctx, newLayerBounds, true)
			filtered.colorFilter = cf
			filtered.updateTileMode(ctx, shaders.TileClamp)
			return filtered
		}

		// Otherwise fill out to the desired output without losing the crop.
		newLayerBounds = ctx.DesiredOutput()
	} else if f.image == nil || !newLayerBounds.Intersects(ctx.DesiredOutput()) {
		// The color filter does not modify transparent black, so the result stays transparent.
		return FilterResult{}
	}

	filtered := *f
	filtered.layerBounds = newLayerBounds
	filtered.colorFilter = colorfilter.NewCompose(cf, f.colorFilter)
	return filtered
}

// compatibleSampling reports whether two sampling operations can collapse into one. May update nextSampling.
func compatibleSampling(currentSampling shaders.SamplingOptions, currentXformWontAffectNearest bool, nextSampling *shaders.SamplingOptions, nextXformWontAffectNearest bool) bool {
	currentAniso := currentSampling.MaxAniso != 0
	nextAniso := nextSampling.MaxAniso != 0
	switch {
	case currentAniso && nextAniso:
		// Assume one sampling at the highest anisotropy level works.
		nextSampling.MaxAniso = max(currentSampling.MaxAniso, nextSampling.MaxAniso)
		return true
	case currentAniso && !nextSampling.UseCubic && nextSampling.Filter == shaders.FilterLinear:
		// Assume the current anisotropic filter suffices since the next is linear.
		*nextSampling = currentSampling
		return true
	case nextAniso && !currentSampling.UseCubic && currentSampling.Filter == shaders.FilterLinear:
		return true
	case currentSampling.UseCubic && ((!nextSampling.UseCubic && nextSampling.Filter == shaders.FilterLinear) ||
		(nextSampling.UseCubic && currentSampling.Cubic == nextSampling.Cubic)):
		// The current bicubic filter covers an equal cubic or an upgradeable bilerp.
		*nextSampling = currentSampling
		return true
	case nextSampling.UseCubic && !currentSampling.UseCubic && currentSampling.Filter == shaders.FilterLinear:
		return true
	case !currentSampling.UseCubic && !nextSampling.UseCubic && !currentAniso && !nextAniso &&
		currentSampling.Filter == shaders.FilterLinear && nextSampling.Filter == shaders.FilterLinear:
		// A single bilerp instead of two.
		return true
	case !nextSampling.UseCubic && nextSampling.Filter == shaders.FilterNearest &&
		currentXformWontAffectNearest:
		// The next nearest-neighbor filtering isn't impacted by the current transform.
		return true
	case !currentSampling.UseCubic && currentSampling.Filter == shaders.FilterNearest &&
		nextXformWontAffectNearest:
		*nextSampling = currentSampling
		return true
	default:
		// Nearest-neighbor sampling producing visible texels oriented with the current transform is assumed to be a
		// desired effect: preserve it.
		return false
	}
}

// ApplyTransform returns a new FilterResult with transform applied, using sampling for any new resampling this
// introduces.
func (f *FilterResult) ApplyTransform(ctx *Context, transform *geom.Matrix, sampling shaders.SamplingOptions) FilterResult {
	if f.image == nil || ctx.DesiredOutput().IsEmpty() {
		// Transformed transparent black remains transparent black.
		return FilterResult{}
	}
	if _, invertible := transform.Invert(); !invertible {
		return FilterResult{}
	}

	// Pick the sampling that matters based on the current and next transforms; the new sampling is bilerp (default)
	// whenever the new transform doesn't matter, which combines maximally.
	currentXformIsInteger := isNearlyIntegerTranslation(&f.transform, nil)
	nextXformIsInteger := isNearlyIntegerTranslation(transform, nil)
	nextSampling := sampling
	if nextXformIsInteger {
		nextSampling = DefaultSampling
	}

	// If the image is visibly cropped by the layer bounds, this transform can't merge with previous transforms (unless
	// it's an integer translation, whose visible edges stay aligned with the desired output).
	isCropped := !nextXformIsInteger &&
		f.analyzeBounds(transform, ctx.DesiredOutput(), scopeDeferred)&boundsRequiresLayerCrop != 0

	var transformed FilterResult
	if !isCropped && compatibleSampling(f.sampling, currentXformIsInteger,
		&nextSampling, nextXformIsInteger) {
		// Concat transforms; nextSampling is f.sampling, sampling, or a merged combination.
		transformed = *f
	} else {
		// Resolve before 'transform' and 'sampling' can be evaluated; nextSampling is 'sampling'.
		if tightBounds, ok := inverseMapIRect(transform, ctx.DesiredOutput()); ok {
			transformed = f.resolve(ctx, tightBounds, false)
		}
		if transformed.image == nil {
			return FilterResult{}
		}
	}

	transformed.sampling = nextSampling
	transformed.transform.PostConcat(transform)
	// Map the prior layer bounds (not the image bounds) by the new transform to preserve accumulated soft crops, then
	// restrict to the desired output.
	transformed.layerBounds = mapIRect(transform, transformed.layerBounds)
	if !transformed.layerBounds.Intersects(ctx.DesiredOutput()) {
		// The transformed output doesn't touch the desired output: transparent black.
		return FilterResult{}
	}
	return transformed
}

// resolve renders this FilterResult into a new, visually equivalent image filling dstBounds with default sampling, no
// color filter, and an integer-translation transform.
func (f *FilterResult) resolve(ctx *Context, dstBounds geom.IRect, preserveDstBounds bool) FilterResult {
	// The layer bounds are the final clip, so they can always restrict dstBounds.
	if f.image == nil || (!preserveDstBounds && !dstBounds.Intersect(f.layerBounds)) {
		return FilterResult{}
	}

	// With no extra effects, an integer translation can extract a subset instead of drawing.
	subsetCompatible := f.colorFilter == nil &&
		f.tileMode == shaders.TileDecal &&
		!preserveDstBounds
	var origin geom.IPoint
	if subsetCompatible && isNearlyIntegerTranslation(&f.transform, &origin) {
		return f.subset(origin, dstBounds, false)
	}

	boundary := boundaryTransparent
	if preserveDstBounds {
		boundary = boundaryUnknown
	}
	surface := newAutoSurface(ctx, dstBounds, boundary, false)
	if surface.valid {
		f.draw(ctx, surface.device, false, nil)
	}
	return surface.snap()
}

// subset returns a decal-tiled subset view, requiring that the transform is the integer translation knownOrigin.
// clampSrcIfDisjoint extracts the closest edge pixels when the image doesn't overlap subsetBounds (clamp semantics).
func (f *FilterResult) subset(knownOrigin geom.IPoint, subsetBounds geom.IRect, clampSrcIfDisjoint bool) FilterResult {
	imageBounds := geom.IRectXYWH(knownOrigin.X, knownOrigin.Y, f.image.Width(), f.image.Height())
	tile := shaders.TileDecal
	if clampSrcIfDisjoint {
		tile = shaders.TileClamp
	}
	imageBounds = RelevantSubset(imageBounds, subsetBounds, tile)
	if imageBounds.IsEmpty() {
		return FilterResult{}
	}

	// Offset the image subset directly to avoid issues negating the origin (e.g. INT_MIN).
	subsetRect := geom.IRect{
		Left:   imageBounds.Left - knownOrigin.X,
		Top:    imageBounds.Top - knownOrigin.Y,
		Right:  imageBounds.Right - knownOrigin.X,
		Bottom: imageBounds.Bottom - knownOrigin.Y,
	}
	result := NewFilterResult(f.image.MakeSubset(subsetRect),
		geom.IPoint{X: imageBounds.Left, Y: imageBounds.Top})
	result.colorFilter = f.colorFilter

	// Update pixelBoundary knowledge based on how the subset aligns.
	if f.image.Subset() == result.image.Subset() {
		result.boundary = f.boundary
	} else {
		// If the new pixel bounds are bordered by valid data, upgrade to boundaryInitialized.
		safeSubset := f.image.Subset()
		if f.boundary == boundaryUnknown {
			safeSubset = safeSubset.Inset(1, 1)
		}
		if safeSubset.ContainsRect(result.image.Subset()) {
			result.boundary = boundaryInitialized
		}
	}
	return result
}

// Draw applies the remaining layer-to-device transform and composites this result onto target. A nil blender means
// src-over.
func (f *FilterResult) Draw(ctx *Context, target Device, blender *raster.BlendMode) {
	// Install layerToDevice for the draw, restoring the prior transform afterwards.
	prior := target.LocalToDevice()
	target.SetLocalToDevice(ctx.Mapping().LayerToDevice())
	f.draw(ctx, target, true, blender)
	target.SetLocalToDevice(prior)
}

// blendModeAffectsTransparentBlack reports, for blend-mode blenders (the only reachable kind), whether a
// transparent-black source modifies the dst: true unless the dst coefficient evaluates to 1 (One, ISA, ISC); advanced
// modes never do.
func blendModeAffectsTransparentBlack(mode raster.BlendMode) bool {
	src, dst, ok := blendModeAsCoeff(mode)
	_ = src
	if !ok {
		return false // advanced blend modes do not affect transparent black
	}
	return dst != coeffOne && dst != coeffISA && dst != coeffISC
}

// draw is the internal draw entry point shared by Draw and resolve.
func (f *FilterResult) draw(ctx *Context, device Device, preserveDeviceState bool, blender *raster.BlendMode) {
	blendAffectsTransparentBlack := blender != nil && blendModeAffectsTransparentBlack(*blender)
	if f.image == nil {
		// The image is transparent black: a no-op unless the blend affects transparent black.
		if blendAffectsTransparentBlack {
			device.DrawPaint(&PaintParams{Color: colorcore.Color4f{}, Blend: *blender})
		}
		return
	}

	scope := scopeCanDrawDirectly
	if blendAffectsTransparentBlack {
		scope = scopeShaderOnly
	}
	localToDevice := device.LocalToDevice()
	analysis := f.analyzeBounds(&localToDevice, device.DevClipBounds(), scope)

	if analysis&boundsRequiresLayerCrop != 0 {
		if blendAffectsTransparentBlack {
			// Like applyColorFilter's resolve path: the blend must apply after the layer bounds clip. Mapping
			// devClipBounds by the local-to-device matrix works for both the final device and intermediate layers.
			dstBounds, ok := inverseMapIRect(&localToDevice, device.DevClipBounds())
			if !ok {
				return
			}
			clipped := f.resolve(ctx, dstBounds, false)
			clipped.draw(ctx, device, preserveDeviceState, blender)
			return
		}
		// Otherwise apply the layer bounds as a clip to avoid an intermediate render pass.
		if preserveDeviceState {
			device.PushClipStack()
		}
		device.ClipRect(f.layerBounds.ToRect(), true)
	}

	// An integer translate's default bilinear should reduce to nearest-neighbor: image draws detect this automatically,
	// image shaders do not, so detect and reduce explicitly.
	pixelAligned := isNearlyIntegerTranslation(&f.transform, nil) &&
		isNearlyIntegerTranslation(&localToDevice, nil)
	sampling := f.sampling
	if sampling == DefaultSampling && pixelAligned {
		sampling = nearestSampling
	}

	if analysis&boundsHasLayerFillingEffect != 0 ||
		(blendAffectsTransparentBlack && analysis&boundsDstNotCovered != 0) {
		// Fill the clip with a shader so pixels beyond the image are covered too (resolving tiling, color-filtering
		// transparent black, blending against dst, or any combination).
		pp := PaintParams{Shader: f.getAnalyzedShaderView(ctx, sampling, analysis)}
		switch {
		case !preserveDeviceState && blender == nil:
			pp.Blend = raster.BlendSrc
		case blender != nil:
			pp.Blend = *blender
		default:
			pp.Blend = raster.BlendSrcOver
		}
		device.DrawPaint(&pp)
	} else {
		// The default paint color here is opaque black, whose alpha must not zero out the image draw.
		pp := PaintParams{
			ColorFilter: f.colorFilter,
			Color:       colorcore.Color4f{A: 1},
			Blend:       raster.BlendSrcOver,
		}
		if blender != nil {
			pp.Blend = *blender
		}

		// The image's origin is embedded in f.transform; DrawSpecial does not apply the device's local-to-device
		// automatically.
		netTransform := f.transform
		netTransform.PostConcat(&localToDevice)

		// Check f.sampling (not the possibly-reduced 'sampling') for linear filtering.
		if f.canClampToTransparentBoundary(analysis) && f.sampling == DefaultSampling {
			// Draw non-AA with a 1px outset image so the transparent boundary filtering doesn't multiply with AA
			// coverage.
			if !preserveDeviceState && blender == nil {
				pp.Blend = raster.BlendSrc
			}
			var t geom.Matrix
			t.SetTranslate(-1, -1)
			netTransform.PreConcat(&t)
			device.DrawSpecial(f.image.MakePixelOutset(), netTransform, sampling, &pp)
		} else {
			pp.AntiAlias = true
			if analysis&boundsRequiresShaderTiling != 0 {
				ctx.markShaderBasedTilingRequired(shaders.TileClamp)
			}
			// The strict/fast constraint distinction collapses on the raster device (both sample the subset view), so
			// DrawSpecial has no constraint parameter.
			device.DrawSpecial(f.image, netTransform, sampling, &pp)
		}
	}

	if preserveDeviceState && analysis&boundsRequiresLayerCrop != 0 {
		device.PopClipStack()
	}
}

// asShader returns this FilterResult as a shader, resolving to an intermediate image only when needed.
func (f *FilterResult) asShader(ctx *Context, xtraSampling shaders.SamplingOptions, flags ShaderFlags, sampleBounds geom.IRect) shaders.Shader {
	if f.image == nil {
		return nil
	}
	currentXformIsInteger := isNearlyIntegerTranslation(&f.transform, nil)
	nextXformIsInteger := flags&ShaderFlagNonTrivialSampling == 0

	analysis := f.analyzeBoundsLayer(sampleBounds, scopeShaderOnly)

	// Deferred calculations on the input would repeat per sample; simple (blend-mode, coefficient class) color filters
	// are cheap enough to skip resolving.
	expensiveColorFilter := false
	if f.colorFilter != nil {
		if _, mode, ok := colorfilter.AsBlend(f.colorFilter); !ok || mode > raster.BlendScreen {
			expensiveColorFilter = true
		}
	}
	sampling := xtraSampling
	needsResolve := (flags&ShaderFlagSampledRepeatedly != 0 && expensiveColorFilter) ||
		!compatibleSampling(f.sampling, currentXformIsInteger, &sampling, nextXformIsInteger) ||
		analysis&boundsRequiresLayerCrop != 0

	// Downgrade to nearest-neighbor if the sequence of sampling does nothing.
	if sampling == DefaultSampling && nextXformIsInteger && (needsResolve || currentXformIsInteger) {
		sampling = nearestSampling
	}

	if needsResolve {
		// The resolve takes care of the transform (sans origin), tile mode, color filter, and layer bounds; redo the
		// analysis because HW edge tiling is hard to predict.
		resolved := f.resolve(ctx, sampleBounds, false)
		if resolved.image == nil {
			return nil
		}
		analysis = resolved.analyzeBoundsLayer(sampleBounds, scopeShaderOnly)
		return resolved.getAnalyzedShaderView(ctx, sampling, analysis)
	}
	return f.getAnalyzedShaderView(ctx, sampling, analysis)
}

// getAnalyzedShaderView builds the final shader for this result, given a pre-computed boundsAnalysis (from
// analyzeBounds or analyzeBoundsLayer).
func (f *FilterResult) getAnalyzedShaderView(ctx *Context, finalSampling shaders.SamplingOptions, analysis boundsAnalysis) shaders.Shader {
	localMatrix := f.transform
	imageBounds := geom.RectWH(float32(f.image.Width()), float32(f.image.Height()))
	// The decal must be applied at layer-space resolution. If the transform preserves rectangles, the image bounds can
	// map by the transform and the decal apply pre-shader; otherwise decompose into a non-scaling post-decal transform
	// and a scaling pre-decal transform.
	var postDecal, preDecal geom.Matrix
	if localMatrix.RectStaysRect() || analysis&boundsRequiresDecalInLayerSp == 0 {
		postDecal = geom.IdentityMatrix()
		preDecal = localMatrix
	} else {
		decomposeTransform(&localMatrix, imageBounds.Center(), &postDecal, &preDecal)
	}

	// If the image covers the dst bounds its tiling won't be visible, so switch to the faster clamp; when applying the
	// decal in layer space the extra shader implements the tiling, so the image shader itself can also clamp.
	effectiveTileMode := f.tileMode
	decalClampToTransparent := f.canClampToTransparentBoundary(analysis)
	strict := analysis&boundsRequiresShaderTiling != 0

	var imageShader shaders.Shader
	if strict && decalClampToTransparent {
		// Sample the 1px outset so the strict subset includes the transparent pixels.
		var t geom.Matrix
		t.SetTranslate(-1, -1)
		preDecal.PreConcat(&t)
		imageShader = f.image.MakePixelOutset().AsShader(shaders.TileClamp, finalSampling, &preDecal, true)
		effectiveTileMode = shaders.TileClamp
	} else {
		if analysis&boundsDstNotCovered == 0 || analysis&boundsRequiresDecalInLayerSp != 0 {
			effectiveTileMode = shaders.TileClamp
		}
		imageShader = f.image.AsShader(effectiveTileMode, finalSampling, &preDecal, strict)
	}
	if strict {
		ctx.markShaderBasedTilingRequired(effectiveTileMode)
	}

	if analysis&boundsRequiresDecalInLayerSp != 0 {
		// The known runtime decal effect: weight the image by the layer-space decal ramp.
		imageShader = shaders.NewFilterDecal(imageShader, mapRect(&preDecal, imageBounds))
	}
	if imageShader != nil && analysis&boundsRequiresDecalInLayerSp != 0 {
		imageShader = shaders.NewWithLocalMatrix(imageShader, postDecal)
	}
	if imageShader != nil && f.colorFilter != nil {
		imageShader = shaders.NewWithColorFilter(imageShader, f.colorFilter)
	}
	// The shader now includes image, sampling, tile mode, transform and color filter; the layer bounds crop must be
	// handled externally (resolve or device clip).
	return imageShader
}
