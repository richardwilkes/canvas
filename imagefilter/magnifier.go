// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The magnifier image filter node: zooms into a lens region of the input, with an optional inset blend band at the lens
// edge.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

type magnifierFilter struct {
	filterDefaults
	lensBounds geom.Rect // parameter space
	zoomAmount float32   // relative, so belongs to no coordinate space
	inset      float32
	sampling   shaders.SamplingOptions
}

// Magnifier builds a filter that zooms lensBounds by zoomAmount with an inset blend band, sampled with the given
// options.
func Magnifier(lensBounds geom.Rect, zoomAmount, inset float32, sampling shaders.SamplingOptions, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if lensBounds.IsEmpty() || !lensBounds.IsFinite() || zoomAmount <= 0 || inset < 0 ||
		!geom.IsFinite(zoomAmount, inset) {
		return nil // invalid
	}
	// The magnifier automatically restricts its output based on the size of the image it receives as input, so
	// 'cropRect' only applies to its input.
	if cropRect != nil {
		input = Crop(*cropRect, shaders.TileDecal, input)
	}
	if zoomAmount <= 1 {
		// Zooming with a value less than 1 is a downscaling with unintuitive non-linear distortion, and at 1 this
		// filter is an expensive identity, so treat zoomAmount <= 1 as a no-op.
		return input
	}
	f := &magnifierFilter{
		lensBounds: lensBounds, zoomAmount: zoomAmount, inset: inset,
		sampling: sampling,
	}
	f.base = filtercore.NewFilterBase(input)
	return f
}

// GPUEvaluable implements filtercore.GPUEvaluable: the magnifier kernel converts to an FP (gpu/gl/filterkernelfp.go).
func (f *magnifierFilter) GPUEvaluable() bool { return true }

// rectToRect returns the scale+translate matrix mapping from onto to, stretching to fill exactly; callers guarantee
// non-empty rects.
func rectToRect(from, to geom.Rect) geom.Matrix {
	var m geom.Matrix
	sx := to.Width() / from.Width()
	sy := to.Height() / from.Height()
	m.SetScaleTranslate(sx, sy, to.Left-from.Left*sx, to.Top-from.Top*sy)
	return m
}

// OnFilterImage computes the source rect to zoom from and evaluates the magnifier shader (or a plain transform+crop
// when the inset is zero).
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *magnifierFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	// The full lens bounds and the ideal zoom center if everything is visible.
	lensBounds := ctx.Mapping().ParamToLayerRect(f.lensBounds)
	zoomCenter := lensBounds.Center()

	// When magnifying near the edge of the screen part of the lens bounds can be offscreen (and the input filter can't
	// provide the full required input); auto-size to the visible portion.
	visibleLensBounds := lensBounds
	if !visibleLensBounds.Intersect(ctx.DesiredOutput().ToRect()) {
		return filtercore.FilterResult{}
	}

	// Pre-emptively fit the zoomed-in src rect to what the child input filter should produce.
	expectedChildOutput := lensBounds
	srcBounds := ctx.Source().LayerBounds()
	if output := f.base.OutputLayerBounds(0, ctx.Mapping(), &srcBounds); output != nil {
		expectedChildOutput = output.ToRect()
	}

	// Clamp the zoom center to be within the childOutput image.
	zoomCenter = clampPointToRect(zoomCenter, expectedChildOutput)

	// The layer-space zoom equals fZoomAmount because this filter only supports scale+translate matrices; clamp the max
	// zoom to scale half of a layer pixel to the entire lens.
	maxLensSize := max(float32(1), max(lensBounds.Width(), lensBounds.Height()))
	invZoom := 1 / min(f.zoomAmount, 2*maxLensSize)

	// srcRect bounds the source pixels that fill the magnified region (the inset of lensBounds). When lensBounds has
	// been cropped this shifts the source rect so the upscaled area stays within the input image.
	srcRect := geom.Rect{
		Left:   lensBounds.Left*invZoom + zoomCenter.X*(1-invZoom),
		Top:    lensBounds.Top*invZoom + zoomCenter.Y*(1-invZoom),
		Right:  lensBounds.Right*invZoom + zoomCenter.X*(1-invZoom),
		Bottom: lensBounds.Bottom*invZoom + zoomCenter.Y*(1-invZoom),
	}

	// When combined with backdrop offsets, more fitting is needed to pin the visible src rect to what's available.
	zoomXform := rectToRect(lensBounds, srcRect)
	if !expectedChildOutput.ContainsRect(visibleLensBounds) {
		// Pick a new srcRect contained within expectedChildOutput that fills visibleLens while maintaining the original
		// srcRect -> lensBounds aspect ratio.
		srcRect = filtercore.MapLayerRect(&zoomXform, visibleLensBounds)
		if expectedChildOutput.Width() >= srcRect.Width() &&
			expectedChildOutput.Height() >= srcRect.Height() {
			var left, top float32
			if srcRect.Left < expectedChildOutput.Left {
				left = expectedChildOutput.Left
			} else {
				left = min(srcRect.Right, expectedChildOutput.Right) - srcRect.Width()
			}
			if srcRect.Top < expectedChildOutput.Top {
				top = expectedChildOutput.Top
			} else {
				top = min(srcRect.Bottom, expectedChildOutput.Bottom) - srcRect.Height()
			}
			srcRect = geom.RectXYWH(left, top, srcRect.Width(), srcRect.Height())
			zoomXform = rectToRect(visibleLensBounds, srcRect)
		} // else not enough of the target is available to cover, so don't try adjusting
	}

	// With a 0 inset the magnifier is equivalent to a rect->rect transform and crop.
	inset := ctx.Mapping().ParamToLayerSize(geom.Size{Width: f.inset, Height: f.inset})
	if inset.Width <= 0 || inset.Height <= 0 {
		invZoomXform, ok := zoomXform.Invert()
		if !ok {
			return filtercore.FilterResult{} // pathological input
		}
		childCtx := ctx.WithNewDesiredOutput(filtercore.RoundOut(srcRect))
		childOutput := f.base.ChildOutput(0, &childCtx)
		transformed := childOutput.ApplyTransform(&ctx, &invZoomXform, f.sampling)
		return transformed.ApplyCrop(&ctx, filtercore.RoundOut(lensBounds), shaders.TileDecal)
	}

	builder := filtercore.NewBuilder(&ctx)
	visibleCtx := ctx.WithNewDesiredOutput(filtercore.RoundOut(visibleLensBounds))
	visibleOutput := f.base.ChildOutput(0, &visibleCtx)
	builder.AddSampled(&visibleOutput, nil, filtercore.ShaderFlagNonTrivialSampling, f.sampling)
	lensOutput := filtercore.RoundOut(lensBounds)
	return builder.Eval(func(inputs []shaders.Shader) shaders.Shader {
		// If the input resolved to a null shader, the magnified output is transparent too.
		if inputs[0] == nil {
			return nil
		}
		// Pack the magnifier shader's uniforms: zoomXform = (Tx, Ty, Sx, Sy).
		return shaders.NewMagnifier(inputs[0], lensBounds,
			[4]float32{
				zoomXform.Get(geom.MTransX), zoomXform.Get(geom.MTransY),
				zoomXform.Get(geom.MScaleX), zoomXform.Get(geom.MScaleY),
			},
			[2]float32{1 / inset.Width, 1 / inset.Height})
	}, &lensOutput)
}

// clampPointToRect clamps p to lie within r.
func clampPointToRect(p geom.Point, r geom.Rect) geom.Point {
	return geom.Point{
		X: min(max(p.X, r.Left), r.Right),
		Y: min(max(p.Y, r.Top), r.Bottom),
	}
}

// OnInputLayerBounds always requires the lens bounds, regardless of the desired output.
func (f *magnifierFilter) OnInputLayerBounds(mapping *filtercore.Mapping, _ geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := filtercore.RoundOut(mapping.ParamToLayerRect(f.lensBounds))
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds returns the lens bounds, or an empty rect if they don't intersect the child's output.
func (f *magnifierFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	output := f.base.OutputLayerBounds(0, mapping, contentBounds)
	lensBounds := filtercore.RoundOut(mapping.ParamToLayerRect(f.lensBounds))
	if output == nil || lensBounds.Intersect(*output) {
		return &lensBounds
	}
	empty := geom.IRect{} // nothing to magnify
	return &empty
}

// OnComputeFastBounds intersects the input's fast bounds with the lens bounds.
func (f *magnifierFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	bounds := src
	if in := f.base.Input(0); in != nil {
		bounds = in.OnComputeFastBounds(src)
	}
	if bounds.Intersect(f.lensBounds) {
		return bounds
	}
	return geom.Rect{}
}
