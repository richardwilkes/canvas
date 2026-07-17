// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// MakeFromImage is the image-source filter's entry into the FilterResult world — a direct SpecialImage wrap for integer
// source rects, or a strict drawImageRect render for fractional ones.

package filtercore

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// MakeFromImage builds a FilterResult drawing image from srcRect to dstRect with the given sampling. srcRect is
// relative to the image's contents; dstRect is in parameter space. The caller guarantees a non-nil image. The image is
// polymorphic (S118): a texture-backed image on the GPU backend enters the FilterResult world without a CPU readback.
func MakeFromImage(ctx *Context, image imagecore.DrawableImage, srcRect, dstRect geom.Rect, sampling shaders.SamplingOptions) FilterResult {
	imageBounds := geom.RectWH(float32(image.Width()), float32(image.Height()))
	if !imageBounds.ContainsRect(srcRect) {
		srcToDst := rectToRectOrIdentity(srcRect, dstRect)
		if !srcRect.Intersect(imageBounds) {
			return FilterResult{} // no overlap, so return an empty/transparent image
		}
		// Adjust dstRect to match the updated srcRect.
		dstRect = mapRect(&srcToDst, srcRect)
	}

	if dstRect.IsEmpty() {
		return FilterResult{} // output collapses to empty
	}

	srcSubset := RoundOut(srcRect)
	if srcSubset.ToRect() == srcRect {
		// Construct a SpecialImage from the subset directly instead of drawing. The srcRect's top left is treated as
		// layer space since the src->dst and param->layer transforms fold into a single step. The pixelBoundary stays
		// boundaryUnknown even for contained subsets, because the client could be doing their own external
		// approximate-fit texturing.
		subset := NewFilterResult(ctx.Backend().MakeImage(srcSubset, image),
			geom.IPoint{X: srcSubset.Left, Y: srcSubset.Top})
		transform := rectToRectOrIdentity(srcRect, dstRect)
		layer := ctx.Mapping().LayerMatrix()
		transform.PostConcat(&layer)
		return subset.ApplyTransform(ctx, &transform, sampling)
	}

	// For now, draw the src->dst subset of the image into a new image.
	dstBounds := RoundOut(ctx.Mapping().ParamToLayerRect(dstRect))
	if !dstBounds.Intersect(ctx.DesiredOutput()) {
		return FilterResult{}
	}
	surface := newAutoSurface(ctx, dstBounds, boundaryTransparent, true)
	if surface.valid {
		// Anti-aliased paint; the implicit opaque-black color keeps the image draw unmodulated.
		surface.device.DrawImageRect(image, srcRect, dstRect, sampling, &PaintParams{
			Color:     colorcore.Color4f{A: 1},
			Blend:     raster.BlendSrcOver,
			AntiAlias: true,
		})
	}
	return surface.snap()
}
