// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The matrix convolution image filter node. Kernels at or above the uniform limit are quantized to unorm8 (as if
// encoded into an A8 texture and reconstructed by a shader), so large-kernel outputs match the quantized results
// expected from a texture-backed implementation.

package imagefilter

import (
	"math"

	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// Kernel size tiers.
const (
	// convLargeKernelSize is the upper limit on kernel taps.
	convLargeKernelSize = 256
	// convMaxUniformKernelSize is the tap count below which kernels stay unquantized.
	convMaxUniformKernelSize = 28
)

type matrixConvolutionFilter struct {
	kernel []float32 // final per-tap coefficients (possibly quantization-reconstructed)
	filterDefaults
	kernelSize    geom.ISize
	kernelOffset  geom.IPoint
	gain          float32
	bias          float32 // [0-255] for historical reasons
	convolveAlpha bool
}

// quantizeConvKernel reproduces an unorm8 texture encoding followed by its shader-side reconstruction: k =
// (v/255)*innerGain + innerBias with v = round(255*(k-min)/innerGain).
func quantizeConvKernel(kernel []float32) []float32 {
	lo := kernel[0]
	hi := kernel[0]
	for _, k := range kernel[1:] {
		if k < lo {
			lo = k
		}
		if k > hi {
			hi = k
		}
	}
	innerGain := hi - lo
	innerBias := lo
	// Treat a near-0 gain (i.e. box blur) as 1 and let innerBias move everything to final value.
	if geom.ScalarNearlyZero(innerGain) {
		innerGain = 1
	}
	out := make([]float32, len(kernel))
	for i, k := range kernel {
		v := float32(math.Floor(float64(255*(k-lo)/innerGain + 0.5))) // round to nearest int
		out[i] = v*(1.0/255.0)*innerGain + innerBias
	}
	return out
}

// MatrixConvolution builds a filter that convolves the input with the given kernel, gain, bias, and offset, applying
// tileMode at the edges and optionally cropping to cropRect.
func MatrixConvolution(kernelSize geom.ISize, kernel []float32, gain, bias float32, kernelOffset geom.IPoint, tileMode shaders.TileMode, convolveAlpha bool, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if kernelSize.Width < 1 || kernelSize.Height < 1 {
		return nil
	}
	length := int64(kernelSize.Width) * int64(kernelSize.Height)
	if length > convLargeKernelSize {
		return nil
	}
	if kernel == nil || int64(len(kernel)) < length {
		return nil
	}
	if kernelOffset.X < 0 || kernelOffset.X >= kernelSize.Width ||
		kernelOffset.Y < 0 || kernelOffset.Y >= kernelSize.Height {
		return nil
	}

	values := append([]float32(nil), kernel[:length]...)
	if length >= convMaxUniformKernelSize {
		values = quantizeConvKernel(values)
	}

	// The tile mode behavior is not well-defined without a crop, so it only applies with one.
	filter := input
	if cropRect != nil && tileMode != shaders.TileDecal {
		// Historically the input image was restricted to the cropRect when tiling was not decal so the kernel evaluates
		// the tiled edge conditions; a decal crop only affects the output.
		filter = Crop(*cropRect, tileMode, filter)
	}
	f := &matrixConvolutionFilter{
		kernel:        values,
		kernelSize:    kernelSize,
		kernelOffset:  kernelOffset,
		gain:          gain,
		bias:          bias,
		convolveAlpha: convolveAlpha,
	}
	f.base = filtercore.NewFilterBase(filter)
	filter = f
	if cropRect != nil {
		// Regardless of the tileMode, the output is decal cropped.
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// OnAffectsTransparentBlack always reports true: the kernel applies in layer space with no transform available to
// compute fast bounds precisely, so this conflates with that limitation; the crop wrappers reset the property when
// possible.
func (f *matrixConvolutionFilter) OnAffectsTransparentBlack() bool { return true }

// GPUEvaluable implements filtercore.GPUEvaluable: the matrix-convolution kernel converts to an FP
// (gpu/gl/filterkernelfp.go).
func (f *matrixConvolutionFilter) GPUEvaluable() bool { return true }

// boundsSampledByKernel outsets bounds to the region of input pixels the kernel reads when producing that output
// region.
func (f *matrixConvolutionFilter) boundsSampledByKernel(bounds geom.IRect) geom.IRect {
	return geom.IRect{
		Left:   bounds.Left - f.kernelOffset.X,
		Top:    bounds.Top - f.kernelOffset.Y,
		Right:  bounds.Right + f.kernelSize.Width - f.kernelOffset.X - 1,
		Bottom: bounds.Bottom + f.kernelSize.Height - f.kernelOffset.Y - 1,
	}
}

// boundsAffectedByKernel outsets bounds to the region of output pixels the kernel touches when reading that input
// region (the inverse of boundsSampledByKernel).
func (f *matrixConvolutionFilter) boundsAffectedByKernel(bounds geom.IRect) geom.IRect {
	return geom.IRect{
		Left:   bounds.Left + f.kernelOffset.X - f.kernelSize.Width + 1,
		Top:    bounds.Top + f.kernelOffset.Y - f.kernelSize.Height + 1,
		Right:  bounds.Right + f.kernelOffset.X,
		Bottom: bounds.Bottom + f.kernelOffset.Y,
	}
}

// OnFilterImage evaluates the child over the kernel-sampled input region, then convolves it.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *matrixConvolutionFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	requiredInput := f.boundsSampledByKernel(ctx.DesiredOutput())
	inputCtx := ctx.WithNewDesiredOutput(requiredInput)
	childOutput := f.base.ChildOutput(0, &inputCtx)

	var outputBounds geom.IRect
	if f.convolveAlpha && f.bias != 0 {
		// The convolution will produce a non-trivial value for every pixel, so fill desired output.
		outputBounds = ctx.DesiredOutput()
	} else {
		outputBounds = f.boundsAffectedByKernel(childOutput.LayerBounds())
		if !outputBounds.Intersect(ctx.DesiredOutput()) {
			return filtercore.FilterResult{}
		}
	}

	sampleBounds := f.boundsSampledByKernel(outputBounds)
	builder := filtercore.NewBuilder(&ctx)
	builder.AddSampled(&childOutput, &sampleBounds, filtercore.ShaderFlagSampledRepeatedly,
		filtercore.DefaultSampling)
	return builder.Eval(func(inputs []shaders.Shader) shaders.Shader {
		// Scale the user-provided bias by 1/255 to match the [0,1] color channel range.
		return shaders.NewMatrixConv(inputs[0], f.kernel, f.kernelSize, f.kernelOffset,
			f.gain, f.bias/255, f.convolveAlpha)
	}, &outputBounds)
}

// OnInputLayerBounds requires the region the kernel samples to produce desiredOutput.
func (f *matrixConvolutionFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := f.boundsSampledByKernel(desiredOutput)
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds expands the child's output by the kernel's affected region, or reports unbounded when alpha
// convolution with a non-zero bias can flood transparent pixels.
func (f *matrixConvolutionFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	if f.convolveAlpha && f.bias != 0 {
		// Convolving the alpha channel with a non-zero bias can make transparent black pixels outside any input image
		// become non-transparent.
		return nil
	}
	outputBounds := f.base.OutputLayerBounds(0, mapping, contentBounds)
	if outputBounds == nil {
		return nil
	}
	out := f.boundsAffectedByKernel(*outputBounds)
	return &out
}

// OnComputeFastBounds returns unbounded: without the local-to-device transform the sampled pixel extent is unknown, so
// this is treated like a filter that affects transparent black.
func (f *matrixConvolutionFilter) OnComputeFastBounds(_ geom.Rect) geom.Rect {
	return filtercore.MakeILarge().ToRect()
}
