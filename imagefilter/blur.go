// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The Gaussian blur image filter node, built on filtercore.Builder.Blur, including the legacy tiling lane taken when a
// tile mode arrives without a crop rect.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// maxBlurSigma caps the box-blur kernel at 1000 pixels to match WebKit/Firefox and keep GPU/raster behavior consistent.
const maxBlurSigma = 532

type blurFilter struct {
	filterDefaults
	sigma geom.Size // parameter space
	// legacyTileMode: shaders.TileDecal means no legacy tiling (crops handle it); anything else emulates the tiling
	// applied when no crop rect was provided.
	legacyTileMode shaders.TileMode
}

// Blur builds a Gaussian blur filter with the given sigmas (in parameter space) and edge tile mode, optionally cropped
// to cropRect. Negative or non-finite sigmas are rejected by returning nil.
func Blur(sigmaX, sigmaY float32, tileMode shaders.TileMode, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if !geom.IsFinite(sigmaX, sigmaY) || sigmaX < 0 || sigmaY < 0 {
		// Negative or non-finite sigmas are errors; zero sigmas support 1D blurs and are detected in OnFilterImage.
		return nil
	}

	// Temporarily allow tiling with no crop rect (legacy behavior).
	if tileMode != shaders.TileDecal && cropRect == nil {
		f := &blurFilter{sigma: geom.Size{Width: sigmaX, Height: sigmaY}, legacyTileMode: tileMode}
		f.base = filtercore.NewFilterBase(input)
		return f
	}

	filter := input
	if tileMode != shaders.TileDecal && cropRect != nil {
		// Historically the input image was restricted to the crop rect when tiling was not decal, so the kernel
		// evaluates the tiled edge conditions; a decal crop only affects the output.
		filter = Crop(*cropRect, tileMode, filter)
	}
	blur := &blurFilter{
		sigma:          geom.Size{Width: sigmaX, Height: sigmaY},
		legacyTileMode: shaders.TileDecal,
	}
	blur.base = filtercore.NewFilterBase(filter)
	filter = blur
	if cropRect != nil {
		// Regardless of the tile mode, the output is always decal cropped.
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// mapSigma converts the filter's parameter-space sigma into layer space via mapping, clamping to maxBlurSigma and
// zeroing axes that are non-finite or effectively an identity blur.
func (f *blurFilter) mapSigma(mapping *filtercore.Mapping) geom.Size {
	sigma := mapping.ParamToLayerSize(f.sigma)
	// Clamp to the maximum sigma.
	sigma = geom.Size{Width: min(sigma.Width, maxBlurSigma), Height: min(sigma.Height, maxBlurSigma)}
	// Disable blurring on axes that are not finite or are effectively the identity.
	if !geom.IsFinite(sigma.Width) || filtercore.IsEffectivelyIdentity(sigma.Width) {
		sigma.Width = 0
	}
	if !geom.IsFinite(sigma.Height) || filtercore.IsEffectivelyIdentity(sigma.Height) {
		sigma.Height = 0
	}
	return sigma
}

// kernelBounds outsets bounds by the layer-space blur radius (3 sigma per axis), rounded up via filtercore.SizeCeil,
// which uses a tolerance for near-integer radii instead of a plain ceiling.
func (f *blurFilter) kernelBounds(mapping *filtercore.Mapping, bounds geom.IRect) geom.IRect {
	sigma := f.mapSigma(mapping)
	radii := filtercore.SizeCeil(geom.Size{Width: 3 * sigma.Width, Height: 3 * sigma.Height})
	return bounds.Inset(-radii.Width, -radii.Height)
}

// GPUEvaluable implements filtercore.GPUEvaluable: the blur engine and its image-shader edge lanes are all
// texture-native and run entirely on the GPU.
func (f *blurFilter) GPUEvaluable() bool { return true }

// OnFilterImage evaluates the child over the kernel-expanded input bounds, then blurs it; a legacy (non-decal) tile
// mode with no crop rect first crops the child output to its own layer bounds so the blur samples the tiled edge
// condition.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *blurFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	inputCtx := ctx.WithNewDesiredOutput(f.kernelBounds(ctx.Mapping(), ctx.DesiredOutput()))

	childOutput := f.base.ChildOutput(0, &inputCtx)
	sigma := f.mapSigma(ctx.Mapping())
	if sigma.Width == 0 && sigma.Height == 0 {
		// No actual blur, so return the input unmodified.
		return childOutput
	}

	// By default Builder.Blur calculates a more optimal output automatically; legacy tiling output also depends on the
	// original child output bounds ignoring the tile mode's effect.
	maxOutput := ctx.DesiredOutput()
	if f.legacyTileMode != shaders.TileDecal {
		maxOutput = f.kernelBounds(ctx.Mapping(), childOutput.LayerBounds())
		if !maxOutput.Intersect(ctx.DesiredOutput()) {
			return filtercore.FilterResult{}
		}
		// Legacy tiling applied to the input image when there was no explicit crop rect: use the child's layer bounds
		// as the crop to adjust the edge tile mode without restricting it.
		childOutput = childOutput.ApplyCrop(&inputCtx, childOutput.LayerBounds(), f.legacyTileMode)
	}

	croppedOutput := ctx.WithNewDesiredOutput(maxOutput)
	builder := filtercore.NewBuilder(&croppedOutput)
	builder.Add(&childOutput)
	return builder.Blur(sigma)
}

// OnInputLayerBounds requires the kernel-expanded desired output from the input.
func (f *blurFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := f.kernelBounds(mapping, desiredOutput)
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds expands the child's output bounds by the kernel radius.
func (f *blurFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	childOutput := f.base.OutputLayerBounds(0, mapping, contentBounds)
	if childOutput == nil {
		return nil
	}
	out := f.kernelBounds(mapping, *childOutput)
	return &out
}

// OnComputeFastBounds outsets the input's fast bounds by 3 sigma per axis.
func (f *blurFilter) OnComputeFastBounds(src geom.Rect) geom.Rect {
	bounds := src
	if in := f.base.Input(0); in != nil {
		bounds = in.OnComputeFastBounds(src)
	}
	return bounds.Outset(f.sigma.Width*3, f.sigma.Height*3)
}
