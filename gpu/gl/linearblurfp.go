// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hand-written GLSL Gaussian blur fragment processors, driven by GaussianBlur. Each is a FragmentProcessor whose single
// child is the source texture effect, sampled explicitly at kernel offsets and accumulated with per-tap Gaussian
// weights. The uniform layouts (offsetsAndKernel/dir for 1D; kernel/offsets for 2D) are fixed, and the loop bound is
// baked to the batched kernel width (gpu.BlurBatchedKernelWidth) so the program key is stable across draws with the
// same effective kernel size. Both share a single class ID and are separated in the key by a runtimeFPKind tag.

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

//////////////////////////////////////////////////////////////////////////////
// 1D linear-sampling Gaussian blur.

type linearBlur1DFP struct {
	FPBase
	offsetsAndKernel [gpu.MaxBlurSamples * 2]float32
	dir              [2]float32
	radius           int32
}

// NewLinearBlur1DFP returns a single-direction linear-filtered Gaussian convolution over child. offsetsAndKernel is
// Compute1DBlurLinearKernel's output and dir is the unit step (1,0) or (0,1).
func NewLinearBlur1DFP(radius int32, offsetsAndKernel *[gpu.MaxBlurSamples * 2]float32, dir geom.Point, child FragmentProcessor) FragmentProcessor {
	fp := &linearBlur1DFP{
		radius: radius, offsetsAndKernel: *offsetsAndKernel,
		dir: [2]float32{dir.X, dir.Y},
	}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, child, ExplicitSampleUsage())
	return fp
}

func (f *linearBlur1DFP) Name() string { return "GaussianBlur1D" }

func (f *linearBlur1DFP) Clone() FragmentProcessor {
	return NewLinearBlur1DFP(f.radius, &f.offsetsAndKernel, geom.Point{X: f.dir[0], Y: f.dir[1]},
		f.ChildProcessor(0).Clone())
}

func (f *linearBlur1DFP) loopLimit() int32 {
	return gpu.BlurBatchedKernelWidth(gpu.BlurLinearKernelWidth(f.radius)) / 2
}

func (f *linearBlur1DFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPBlur1D), "runtimeFPKind")
	b.Add32(uint32(f.loopLimit()), "blurLoopLimit")
}

func (f *linearBlur1DFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*linearBlur1DFP)
	return ok && f.radius == o.radius && f.dir == o.dir && f.offsetsAndKernel == o.offsetsAndKernel
}

func (f *linearBlur1DFP) onMakeProgramImpl() FPProgramImpl { return &linearBlur1DImpl{} }

type linearBlur1DImpl struct {
	FPImplBase
	offsetsAndKernelUni UniformHandle
	dirUni              UniformHandle
}

func (i *linearBlur1DImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*linearBlur1DFP)
	fb := args.FragBuilder
	var oakName, dirName string
	i.offsetsAndKernelUni, oakName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "offsetsAndKernel", gpu.MaxBlurSamples/2)
	i.dirUni, dirName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf2,
		"dir")

	coord := args.SampleCoord
	fb.CodeAppend("vec4 sum = vec4(0.0);")
	fb.CodeAppendf("for (int i = 0; i < %d; ++i) {", fp.loopLimit())
	fb.CodeAppendf("vec4 s = %s[i];", oakName)
	fb.CodeAppendf("sum += s.y * %s;",
		i.InvokeChildWithCoords(0, "", args, fmt.Sprintf("%s + s.x * %s", coord, dirName)))
	fb.CodeAppendf("sum += s.w * %s;",
		i.InvokeChildWithCoords(0, "", args, fmt.Sprintf("%s + s.z * %s", coord, dirName)))
	fb.CodeAppend("}")
	fb.CodeAppend("return sum;")
}

func (i *linearBlur1DImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*linearBlur1DFP)
	pdman.Set4fv(i.offsetsAndKernelUni, gpu.MaxBlurSamples/2, f.offsetsAndKernel[:])
	pdman.Set2f(i.dirUni, f.dir[0], f.dir[1])
}

//////////////////////////////////////////////////////////////////////////////
// 2D (non-separable) Gaussian blur for small radii.

type blur2DFP struct {
	FPBase
	kernel           [gpu.MaxBlurSamples]float32
	offsets          [gpu.MaxBlurSamples * 2]float32
	radiusX, radiusY int32
}

// NewBlur2DFP returns a single-pass non-separable Gaussian convolution. kernel is Compute2DBlurKernel's output, offsets
// is Compute2DBlurOffsets'.
func NewBlur2DFP(radiusX, radiusY int32, kernel *[gpu.MaxBlurSamples]float32, offsets *[gpu.MaxBlurSamples * 2]float32, child FragmentProcessor) FragmentProcessor {
	fp := &blur2DFP{kernel: *kernel, offsets: *offsets, radiusX: radiusX, radiusY: radiusY}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, child, ExplicitSampleUsage())
	return fp
}

func (f *blur2DFP) Name() string { return "GaussianBlur2D" }

func (f *blur2DFP) Clone() FragmentProcessor {
	return NewBlur2DFP(f.radiusX, f.radiusY, &f.kernel, &f.offsets, f.ChildProcessor(0).Clone())
}

func (f *blur2DFP) loopLimit() int32 {
	kernelArea := gpu.BlurKernelWidth(f.radiusX) * gpu.BlurKernelWidth(f.radiusY)
	return gpu.BlurBatchedKernelWidth(kernelArea) / 4
}

func (f *blur2DFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPBlur2D), "runtimeFPKind")
	b.Add32(uint32(f.loopLimit()), "blurLoopLimit")
}

func (f *blur2DFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*blur2DFP)
	return ok && f.radiusX == o.radiusX && f.radiusY == o.radiusY && f.kernel == o.kernel &&
		f.offsets == o.offsets
}

func (f *blur2DFP) onMakeProgramImpl() FPProgramImpl { return &blur2DImpl{} }

type blur2DImpl struct {
	FPImplBase
	kernelUni  UniformHandle
	offsetsUni UniformHandle
}

func (i *blur2DImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*blur2DFP)
	fb := args.FragBuilder
	var kernelName, offsetsName string
	i.kernelUni, kernelName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "kernel", gpu.MaxBlurSamples/4)
	i.offsetsUni, offsetsName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "offsets", gpu.MaxBlurSamples/2)

	coord := args.SampleCoord
	child := func(coordExpr string) string { return i.InvokeChildWithCoords(0, "", args, coordExpr) }
	fb.CodeAppend("vec4 sum = vec4(0.0);")
	fb.CodeAppendf("for (int i = 0; i < %d; ++i) {", fp.loopLimit())
	fb.CodeAppendf("vec4 k = %s[i];", kernelName)
	fb.CodeAppendf("vec4 o = %s[2*i];", offsetsName)
	fb.CodeAppendf("sum += k.x * %s;", child(fmt.Sprintf("%s + o.xy", coord)))
	fb.CodeAppendf("sum += k.y * %s;", child(fmt.Sprintf("%s + o.zw", coord)))
	fb.CodeAppendf("o = %s[2*i + 1];", offsetsName)
	fb.CodeAppendf("sum += k.z * %s;", child(fmt.Sprintf("%s + o.xy", coord)))
	fb.CodeAppendf("sum += k.w * %s;", child(fmt.Sprintf("%s + o.zw", coord)))
	fb.CodeAppend("}")
	fb.CodeAppend("return sum;")
}

func (i *blur2DImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*blur2DFP)
	pdman.Set4fv(i.kernelUni, gpu.MaxBlurSamples/4, f.kernel[:])
	pdman.Set4fv(i.offsetsUni, gpu.MaxBlurSamples/2, f.offsets[:])
}
