// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// FilterResult.rescale and its border helpers: the progressive 1/2x downscale loop that lets the blur engine cap its
// sigma, preserving deferred tile modes by rendering explicit border pixels at every step.

package filtercore

import (
	"math"
	"math/bits"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// nextLog2 returns ceil(log2(v)) for v >= 1.
func nextLog2(v int32) int {
	if v <= 1 {
		return 0
	}
	return bits.Len32(uint32(v - 1))
}

// downscaleStepCount returns the number of 1/2x downscale passes needed to reach netScaleFactor.
func downscaleStepCount(netScaleFactor float32) int {
	inv := 1 / netScaleFactor
	ceilInv := int32(math.Ceil(float64(inv)))
	steps := nextLog2(ceilInv)
	// There are (steps-1) 1/2x steps and one step between 1/2-1x. If the final step is practically the identity scale,
	// save a render pass by reducing the step count.
	if steps > 0 {
		const multiPassLimit = 0.9
		const nearIdentityLimit = 1 - roundEpsilon // 1px error in a 1000px image

		finalStepScale := netScaleFactor * float32(int32(1)<<(steps-1))
		limit := float32(multiPassLimit)
		if steps == 1 {
			limit = nearIdentityLimit
		}
		if finalStepScale >= limit {
			steps--
		}
	}
	return steps
}

// scaleAboutCenter downscales src relative to its own center to distribute sampling errors across the image.
func scaleAboutCenter(src geom.Rect, sx, sy float32) geom.Rect {
	var cx, cy float32
	if sx != 1 {
		cx = 0.5*src.Left + 0.5*src.Right
	}
	if sy != 1 {
		cy = 0.5*src.Top + 0.5*src.Bottom
	}
	return geom.Rect{
		Left:   (src.Left - cx) * sx,
		Top:    (src.Top - cy) * sy,
		Right:  (src.Right - cx) * sx,
		Bottom: (src.Bottom - cy) * sy,
	}
}

// drawColorFilteredBorder fills the 1px border with the transparency-affecting color filter's flood color.
func drawColorFilteredBorder(surface *autoSurface, border geom.IRect, cf shaders.ColorFilter) {
	cfOnly := PaintParams{Color: colorcore.Color4f{}, ColorFilter: cf, Blend: raster.BlendSrc}
	// Top and bottom rows (with corners), then left and right columns (no corners).
	surface.drawRect(geom.IRectLTRB(border.Left, border.Top, border.Right, border.Top+1).ToRect(), &cfOnly)
	surface.drawRect(geom.IRectLTRB(border.Left, border.Bottom-1, border.Right, border.Bottom).ToRect(), &cfOnly)
	surface.drawRect(geom.IRectLTRB(border.Left, border.Top+1, border.Left+1, border.Bottom-1).ToRect(), &cfOnly)
	surface.drawRect(geom.IRectLTRB(border.Right-1, border.Top+1, border.Right, border.Bottom-1).ToRect(), &cfOnly)
}

// drawTiledBorder samples the border pixels directly, scaling one axis at a time for edges and not at all for corners,
// respecting the tile mode.
func drawTiledBorder(surface *autoSurface, tileMode shaders.TileMode, pp *PaintParams, srcToDst *geom.Matrix, srcBorder, dstBorder geom.Rect) {
	drawEdge := func(src, dst geom.Rect) {
		m := rectToRectOrIdentity(src, dst)
		surface.drawRectWithCTM(src, &m, pp)
	}
	drawCorner := func(src, dst geom.Point) {
		drawEdge(geom.RectXYWH(src.X, src.Y, 1, 1), geom.RectXYWH(dst.X, dst.Y, 1, 1))
	}

	// dstBorder includes the 1px padding being filled; inset to reconstruct the sampled dst.
	dstSampleBounds := dstBorder.Inset(1, 1)
	// Reconstruct the original source coordinate bounds.
	srcSampleBounds, ok := inverseMapRect(srcToDst, dstSampleBounds)
	if !ok {
		panic("srcToDst must be invertible")
	}

	if tileMode == shaders.TileMirror || tileMode == shaders.TileRepeat {
		// Adjust srcBorder to the 1px rect centered over srcSampleBounds so the eventual sample coordinates average the
		// outermost two rows/columns of src pixels.
		adjusted, ok2 := inverseMapRect(srcToDst, dstSampleBounds.Inset(0.5, 0.5))
		if !ok2 {
			panic("srcToDst must be invertible")
		}
		srcBorder = adjusted.Outset(0.5, 0.5)
	}

	// Invert the dst coordinates for repeat so the left edge maps to the right edge of the output.
	if tileMode == shaders.TileRepeat {
		dstBorder = geom.Rect{
			Left:   dstBorder.Right - 1,
			Top:    dstBorder.Bottom - 1,
			Right:  dstBorder.Left + 1,
			Bottom: dstBorder.Top + 1,
		}
	}

	// Edges (excluding corners).
	drawEdge(geom.RectLTRB(srcBorder.Left, srcSampleBounds.Top, srcBorder.Left+1, srcSampleBounds.Bottom),
		geom.RectLTRB(dstBorder.Left, dstSampleBounds.Top, dstBorder.Left+1, dstSampleBounds.Bottom)) // Left
	drawEdge(geom.RectLTRB(srcBorder.Right-1, srcSampleBounds.Top, srcBorder.Right, srcSampleBounds.Bottom),
		geom.RectLTRB(dstBorder.Right-1, dstSampleBounds.Top, dstBorder.Right, dstSampleBounds.Bottom)) // Right
	drawEdge(geom.RectLTRB(srcSampleBounds.Left, srcBorder.Top, srcSampleBounds.Right, srcBorder.Top+1),
		geom.RectLTRB(dstSampleBounds.Left, dstBorder.Top, dstSampleBounds.Right, dstBorder.Top+1)) // Top
	drawEdge(geom.RectLTRB(srcSampleBounds.Left, srcBorder.Bottom-1, srcSampleBounds.Right, srcBorder.Bottom),
		geom.RectLTRB(dstSampleBounds.Left, dstBorder.Bottom-1, dstSampleBounds.Right, dstBorder.Bottom)) // Bottom

	// Corners, sampled directly to preserve their value (they can dominate a clamped blur).
	drawCorner(geom.Point{X: srcBorder.Left, Y: srcBorder.Top},
		geom.Point{X: dstBorder.Left, Y: dstBorder.Top}) // TL
	drawCorner(geom.Point{X: srcBorder.Right - 1, Y: srcBorder.Top},
		geom.Point{X: dstBorder.Right - 1, Y: dstBorder.Top}) // TR
	drawCorner(geom.Point{X: srcBorder.Right - 1, Y: srcBorder.Bottom - 1},
		geom.Point{X: dstBorder.Right - 1, Y: dstBorder.Bottom - 1}) // BR
	drawCorner(geom.Point{X: srcBorder.Left, Y: srcBorder.Bottom - 1},
		geom.Point{X: dstBorder.Left, Y: dstBorder.Bottom - 1}) // BL
}

// rescale returns an equivalent FilterResult whose backing image is reduced by the scale factors (in [0,1]), with a
// deferred upscale transform. When enforceDecal is true the result is decal-sampled with tiling already applied. All
// deferred effects except possibly the tile mode are applied. (Color-type/color-space conversion is a no-op here.)
func (f *FilterResult) rescale(ctx *Context, scale geom.Size, enforceDecal, allowOverscaling bool) FilterResult {
	visibleLayerBounds := f.layerBounds
	if f.image == nil || !visibleLayerBounds.Intersect(ctx.DesiredOutput()) ||
		scale.Width <= 0 || scale.Height <= 0 {
		return FilterResult{}
	}

	var origin geom.IPoint
	pixelAligned := isNearlyIntegerTranslation(&f.transform, &origin)
	analysis := f.analyzeBoundsLayer(ctx.DesiredOutput(), scopeRescale)

	// With no actual scaling and no effects that blur() requires resolved, just extract the subset.
	canDeferTiling := pixelAligned &&
		analysis&boundsRequiresLayerCrop == 0 &&
		(!enforceDecal || analysis&boundsHasLayerFillingEffect == 0)

	// A non-N32 image counts as an effect to apply even when nothing else does: the pipeline runs on N32 premul only
	// (specialimage.go's header), so the image must be re-rendered into an N32 surface here rather than handed to the
	// blur engine, which reads raster pixel storage as 32-bit words. (There is no color-space term: everything is sRGB.)
	hasEffectsToApply := !canDeferTiling || f.colorFilter != nil ||
		f.image.ColorType() != imagecore.ColorTypeN32

	xSteps := downscaleStepCount(scale.Width)
	ySteps := downscaleStepCount(scale.Height)
	if xSteps == 0 && ySteps == 0 && !hasEffectsToApply {
		if analysis&boundsHasLayerFillingEffect != 0 {
			// Only a non-decal tile mode could be visible: adjust layer bounds to the output.
			noop := *f
			noop.layerBounds = visibleLayerBounds
			return noop
		}
		// The visible layer bounds are tighter than the image itself.
		return f.subset(origin, visibleLayerBounds, false)
	}

	var srcRect geom.IRect
	var tileMode shaders.TileMode
	cfBorder := false
	deferPeriodicTiling := false
	if canDeferTiling && analysis&boundsHasLayerFillingEffect != 0 {
		// Deferring visible tiling means rescaling the original image uses smaller buffers.
		srcRect = geom.IRectXYWH(origin.X, origin.Y, f.image.Width(), f.image.Height())
		if f.tileMode == shaders.TileDecal {
			// Like applyColorFilter: evaluate the transparent color-filtered border and clamp.
			tileMode = shaders.TileClamp
			cfBorder = true
		} else {
			tileMode = f.tileMode
			deferPeriodicTiling = tileMode == shaders.TileRepeat || tileMode == shaders.TileMirror
		}
	} else {
		// Either the layer-bounds-sized image must be rescaled (!canDeferTiling) or the tiling isn't visible so the
		// layer bounds are the effective image.
		srcRect = visibleLayerBounds
		tileMode = shaders.TileDecal
	}

	srcRect = RelevantSubset(srcRect, ctx.DesiredOutput(), tileMode)
	// Track the logical image size in floats through the whole process to avoid accumulating rounding error; integers
	// only produce the conservative pixel buffers.
	stepBoundsF := srcRect.ToRect()
	if stepBoundsF.IsEmpty() {
		return FilterResult{}
	}
	// stepPixelBounds includes any padded outsetting rendered by the previous step.
	stepPixelBounds := srcRect.ToRect()

	// At least one iteration is required, even when xSteps and ySteps are 0.
	image := *f
	if !pixelAligned && (xSteps > 0 || ySteps > 0) {
		// A deferred downscaling transform shouldn't compose with the first rescale step (missing pixels in the
		// bilinear filtering create animation artifacts): resolve when the source's net scale requires more steps than
		// requested.
		netScale := mapSize(&image.transform, scale)
		nextXSteps := math.MaxInt
		if geom.IsFinite(netScale.Width) {
			nextXSteps = downscaleStepCount(netScale.Width)
		}
		nextYSteps := math.MaxInt
		if geom.IsFinite(netScale.Height) {
			nextYSteps = downscaleStepCount(netScale.Height)
		}
		if (xSteps > 0 && nextXSteps > xSteps) || (ySteps > 0 && nextYSteps > ySteps) {
			image = image.resolve(ctx, srcRect, false)
			if image.image == nil {
				return FilterResult{}
			}
			if !cfBorder {
				// Match either decal or the deferred tile mode.
				image.tileMode = tileMode
			} // else leave as decal while cfBorder is true
		}
	}

	if deferPeriodicTiling {
		// The periodic tiling is manually rendered into the low-res image so clamp tiling works at each decimation.
		image.tileMode = shaders.TileClamp
	} else {
		// Not overscaling behaves better for animated sigmas/scales.
		allowOverscaling = false
	}

	// Deferred periodic tiling needs pixel-aligned low-res bounds so the low-res period aligns with the original
	// period.
	finalScaleX := float32(1)
	if xSteps > 0 {
		if allowOverscaling {
			finalScaleX = 1 / float32(int32(1)<<xSteps)
		} else {
			finalScaleX = scale.Width
		}
	}
	finalScaleY := float32(1)
	if ySteps > 0 {
		if allowOverscaling {
			finalScaleY = 1 / float32(int32(1)<<ySteps)
		} else {
			finalScaleY = scale.Height
		}
	}

	for {
		sx := float32(1)
		if xSteps > 0 {
			if xSteps > 1 {
				sx = 0.5
			} else {
				sx = float32(srcRect.Width()) * finalScaleX / stepBoundsF.Width()
			}
			xSteps--
		}
		sy := float32(1)
		if ySteps > 0 {
			if ySteps > 1 {
				sy = 0.5
			} else {
				sy = float32(srcRect.Height()) * finalScaleY / stepBoundsF.Height()
			}
			ySteps--
		}

		dstBoundsF := scaleAboutCenter(stepBoundsF, sx, sy)
		finalXStep := xSteps == 0 && sx != 1
		finalYStep := ySteps == 0 && sy != 1
		if deferPeriodicTiling && (finalXStep || finalYStep) {
			dstPixels := RoundOut(dstBoundsF)
			if finalXStep {
				dstBoundsF.Left = float32(dstPixels.Left)
				dstBoundsF.Right = float32(dstPixels.Right)
			}
			if finalYStep {
				dstBoundsF.Top = float32(dstPixels.Top)
				dstBoundsF.Bottom = float32(dstPixels.Bottom)
			}
		}

		dstPixelBounds := RoundOut(dstBoundsF)

		boundary := boundaryUnknown
		sampleBounds := dstPixelBounds
		if tileMode == shaders.TileDecal {
			boundary = boundaryTransparent
		} else {
			// Roughly equivalent to boundary = boundaryInitialized but keeps the later logic simpler.
			dstPixelBounds = dstPixelBounds.Inset(-1, -1)
		}

		surface := newAutoSurface(ctx, dstPixelBounds, boundary, false)
		if !surface.valid {
			// Rescaling can't complete; no sense downscaling non-existent data.
			return FilterResult{}
		}
		scaleXform := rectToRectOrIdentity(stepBoundsF, dstBoundsF)

		// Redo analysis with the actual scale transform and padded low-res bounds.
		analysis = image.analyzeBounds(&scaleXform, sampleBounds, scopeRescale)

		// Primary fill covering all of sampleBounds.
		pp := PaintParams{
			Shader: image.getAnalyzedShaderView(ctx, image.sampling, analysis),
			Blend:  raster.BlendSrc,
		}

		srcSampled, ok := inverseMapRect(&scaleXform, sampleBounds.ToRect())
		if !ok {
			panic("scaleXform must be invertible")
		}
		surface.drawRectWithCTM(srcSampled, &scaleXform, &pp)

		switch {
		case cfBorder:
			// Fill the border with the transparency-affecting color filter — what the image shader's tile mode would
			// have produced, without shader-based tiling.
			drawColorFilteredBorder(&surface, dstPixelBounds, f.colorFilter)
			// Clamping preserves these values on subsequent rescale steps.
			cfBorder = false
		case tileMode != shaders.TileDecal:
			// Draw the shader's edges into the padded border, respecting the tile mode.
			drawTiledBorder(&surface, tileMode, &pp, &scaleXform, stepPixelBounds,
				dstPixelBounds.ToRect())
		}

		image = surface.snap()
		// With deferred periodic tiling, clamp on subsequent steps preserves the border pixels; the original tile mode
		// is restored at the end.
		if deferPeriodicTiling {
			image.tileMode = shaders.TileClamp
		} else {
			image.tileMode = tileMode
		}

		stepBoundsF = dstBoundsF
		stepPixelBounds = dstPixelBounds.ToRect()

		if xSteps <= 0 && ySteps <= 0 {
			break
		}
	}

	// Rebuild the downscaled image: restore the transform back to the original layer-space resolution, the layer bounds
	// it should fill, and the tile mode.
	if deferPeriodicTiling {
		// Undo the manually added border so the result gets the boundaryInitialized state.
		image = image.insetByPixel()
	}
	// Otherwise leave the image as-is: decal preserves the known transparent boundary; clamp keeps the carefully
	// maintained boundary pixels.
	image.tileMode = tileMode
	upscale := rectToRectOrIdentity(stepBoundsF, srcRect.ToRect())
	image.transform.PostConcat(&upscale)
	image.layerBounds = visibleLayerBounds
	return image
}
