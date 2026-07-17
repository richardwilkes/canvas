// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The image-source leaf image filter node: draws a source image (with source/destination rects and sampling) as the
// start of a filter DAG.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

type imageSourceFilter struct {
	// image is the polymorphic source: a CPU imagecore.Image or a texture-backed image, carried in its native form. The
	// backend resolves it as needed at evaluation (the raster backend to CPU pixels, the GPU backend not at all).
	image imagecore.DrawableImage
	filterDefaults
	sampling shaders.SamplingOptions
	// srcRect is relative to the image's contents (not parameter space, unlike dstRect).
	srcRect geom.Rect
	dstRect geom.Rect // parameter space
}

// Image builds an image-source filter drawing image from srcRect to dstRect with the given sampling. A texture-backed
// image is resolved to CPU pixels lazily at evaluation rather than eagerly at construction: the CPU filter DAG still
// needs pixels, so a drawn filter takes the readback once (cpuImage), but a filter built and never drawn costs no
// readback, and the source is carried in its native form so a future GPU-native filter backend can sample it directly.
func Image(image imagecore.DrawableImage, srcRect, dstRect geom.Rect, sampling shaders.SamplingOptions) filtercore.Filter {
	if srcRect.IsEmpty() || dstRect.IsEmpty() || isNilImage(image) {
		// There is no content to draw, so the filter produces transparent black.
		return Empty()
	}
	imageBounds := geom.RectWH(float32(image.Width()), float32(image.Height()))
	if !imageBounds.ContainsRect(srcRect) {
		srcToDst := rectToRectOrIdentityLocal(srcRect, dstRect)
		if !imageBounds.Intersect(srcRect) {
			return Empty() // no overlap, so draw empty
		}
		// Adjust dstRect to match the updated src (stored in imageBounds).
		mappedBounds, _ := srcToDst.MapRect(imageBounds)
		if mappedBounds.IsEmpty() {
			return Empty()
		}
		srcRect = imageBounds
		dstRect = mappedBounds
	}
	f := &imageSourceFilter{image: image, srcRect: srcRect, dstRect: dstRect, sampling: sampling}
	f.base = filtercore.NewFilterBase()
	return f
}

// ImageDefault builds an image-source filter with src = dst = the image bounds.
func ImageDefault(image imagecore.DrawableImage, sampling shaders.SamplingOptions) filtercore.Filter {
	var bounds geom.Rect
	if !isNilImage(image) {
		bounds = geom.RectWH(float32(image.Width()), float32(image.Height()))
	}
	return Image(image, bounds, bounds, sampling)
}

// isNilImage reports whether image is a nil interface or a nil *imagecore.Image boxed into it, so a nil source
// correctly produces Empty() regardless of which form it arrives in.
func isNilImage(image imagecore.DrawableImage) bool {
	if image == nil {
		return true
	}
	im, ok := image.(*imagecore.Image)
	return ok && im == nil
}

// rectToRectOrIdentityLocal returns the matrix mapping from onto to, or the identity matrix if either rect is empty.
func rectToRectOrIdentityLocal(from, to geom.Rect) geom.Matrix {
	if from.IsEmpty() || to.IsEmpty() {
		return geom.IdentityMatrix()
	}
	return rectToRect(from, to)
}

// OnCTMCapability reports that drawing the source image is independent of the layer matrix.
func (f *imageSourceFilter) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityComplex
}

// OnFilterImage draws the source image. The image passes to MakeFromImage in its native form: the raster backend's
// MakeImage resolves a texture backing to CPU pixels (the fallback's single readback, deferred off the construction
// path); the GPU backend wraps it directly.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *imageSourceFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	return filtercore.MakeFromImage(&ctx, f.image, f.srcRect, f.dstRect, f.sampling)
}

// GPUEvaluable implements filtercore.GPUEvaluable: the leaf draws via the image shader / DrawImageRect lanes, both
// texture-native.
func (f *imageSourceFilter) GPUEvaluable() bool { return true }

// OnInputLayerBounds returns an empty rect: a leaf filter requires no input.
func (f *imageSourceFilter) OnInputLayerBounds(_ *filtercore.Mapping, _ geom.IRect, _ *geom.IRect) geom.IRect {
	return geom.IRect{}
}

// OnOutputLayerBounds returns the layer-space bounds of dstRect.
func (f *imageSourceFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, _ *geom.IRect) *geom.IRect {
	out := filtercore.RoundOut(mapping.ParamToLayerRect(f.dstRect))
	return &out
}

// OnComputeFastBounds returns dstRect.
func (f *imageSourceFilter) OnComputeFastBounds(_ geom.Rect) geom.Rect { return f.dstRect }
