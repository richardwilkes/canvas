// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Blends the outputs of two child FPs (src, dst) with a raster.BlendMode. Blend logic is either shared via the
// uniform-driven blend helper functions or hardwired to the mode's dedicated function.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// doesCPUBlendImplMatchGPU reports whether the CPU implementation of mode differ from the GPU enough that
// constantOutputForConstantInput can't use them.
func doesCPUBlendImplMatchGPU(mode raster.BlendMode) bool {
	return mode <= raster.BlendMultiply && mode != raster.BlendSoftLight &&
		mode != raster.BlendColorBurn
}

type blendFP struct {
	FPBase
	mode            raster.BlendMode
	shareBlendLogic bool
}

// BlendFP builds a fragment processor that blends src's and dst's outputs with the given mode. Either child may be nil,
// in which case the input color is used.
func BlendFP(src, dst FragmentProcessor, mode raster.BlendMode) FragmentProcessor {
	// Clear, Src, and Dst simplify dramatically in the shader, but only if we bypass the shared logic.
	shareBlendLogic := mode != raster.BlendClear && mode != raster.BlendSrc && mode != raster.BlendDst
	return newBlendFP(src, dst, mode, shareBlendLogic)
}

func newBlendFP(src, dst FragmentProcessor, mode raster.BlendMode, shareBlendLogic bool) FragmentProcessor {
	fp := &blendFP{mode: mode, shareBlendLogic: shareBlendLogic}
	fp.initFP(BlendFragmentProcessorClassID, blendFPOptFlags(src, dst, mode))
	fp.setIsBlendFunction()
	registerChildOf(fp, src, PassThroughSampleUsage())
	registerChildOf(fp, dst, PassThroughSampleUsage())
	return fp
}

// blendFPOptFlags computes the optimization flags for a blend of src and dst under mode.
func blendFPOptFlags(src, dst FragmentProcessor, mode raster.BlendMode) uint32 {
	var flags uint32
	switch mode {
	case raster.BlendClear:
		flags = 0

	// Just propagates src, and its flags:
	case raster.BlendSrc:
		flags = fpProcessorOptimizationFlags(src) &^ FPConstantOutputForConstantInput

	// Just propagates dst, and its flags:
	case raster.BlendDst:
		flags = fpProcessorOptimizationFlags(dst) &^ FPConstantOutputForConstantInput

	// Produces opaque if both src and dst are opaque. These also will modulate the child's output by either the input
	// color or alpha; if the child is not compatible with coverage as alpha then it may produce a color that is not
	// valid premul.
	case raster.BlendSrcIn, raster.BlendDstIn, raster.BlendModulate:
		switch {
		case src != nil && dst != nil:
			flags = fpProcessorOptimizationFlags(src) & fpProcessorOptimizationFlags(dst) &
				FPPreservesOpaqueInput
		case src != nil:
			flags = fpProcessorOptimizationFlags(src) &^ FPConstantOutputForConstantInput
		case dst != nil:
			flags = fpProcessorOptimizationFlags(dst) &^ FPConstantOutputForConstantInput
		default:
			flags = 0
		}

	// Produces zero when both are opaque, indeterminate if one is opaque.
	case raster.BlendSrcOut, raster.BlendDstOut, raster.BlendXor:
		flags = 0

	// Is opaque if the dst is opaque.
	case raster.BlendSrcATop:
		flags = fpProcessorOptimizationFlags(dst) & FPPreservesOpaqueInput

	// DstATop is the converse of SrcATop. Screen is also opaque if the src is opaque.
	case raster.BlendDstATop, raster.BlendScreen:
		flags = fpProcessorOptimizationFlags(src) & FPPreservesOpaqueInput

	// These modes are all opaque if either src or dst is opaque. All the advanced modes compute alpha as src-over.
	default:
		flags = (fpProcessorOptimizationFlags(src) | fpProcessorOptimizationFlags(dst)) &
			FPPreservesOpaqueInput
	}
	if doesCPUBlendImplMatchGPU(mode) &&
		(src == nil || src.fpBase().HasConstantOutputForConstantInput()) &&
		(dst == nil || dst.fpBase().HasConstantOutputForConstantInput()) {
		flags |= FPConstantOutputForConstantInput
	}
	return flags
}

func (f *blendFP) Name() string { return "Blend" }

func (f *blendFP) Clone() FragmentProcessor {
	clone := &blendFP{mode: f.mode, shareBlendLogic: f.shareBlendLogic}
	clone.initFP(BlendFragmentProcessorClassID, f.optimizationFlags())
	clone.setIsBlendFunction()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *blendFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	if f.shareBlendLogic {
		b.Add32(uint32(GLSLBlendKey(f.mode)), "blendKey")
	} else {
		b.Add32(uint32(f.mode), "blendMode")
	}
}

func (f *blendFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*blendFP)
	return ok && f.mode == o.mode
}

func (f *blendFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	srcColor := fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
	dstColor := fpConstantOutputForConstantInput(f.ChildProcessor(1), input)
	return raster.BlendHighp4f(f.mode, srcColor, dstColor)
}

func (f *blendFP) onMakeProgramImpl() FPProgramImpl { return &blendFPImpl{} }

type blendFPImpl struct {
	FPImplBase
	blendUni UniformHandle
}

func (i *blendFPImpl) EmitCode(args *FPEmitArgs) {
	bfp := args.FP.(*blendFP)

	// Invoke src/dst with our input color (or the input color directly if no child FP).
	srcColor := i.InvokeChild(0, args)
	dstColor := i.InvokeChild(1, args)

	if bfp.shareBlendLogic {
		// Use a blend expression that may rely on uniforms.
		blendExpr := GLSLBlendExpression(args.FP, args.UniformHandler, args.FragBuilder,
			&i.blendUni, srcColor, dstColor, bfp.mode)
		args.FragBuilder.CodeAppendf("return %s;", blendExpr)
	} else {
		// Blend src and dst colors together using a hardwired blend function.
		fn := args.FragBuilder.ensureBlendFunction(bfp.mode)
		args.FragBuilder.CodeAppendf("return %s(%s, %s);", fn, srcColor, dstColor)
	}
}

func (i *blendFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	bfp := fp.(*blendFP)
	if bfp.shareBlendLogic {
		GLSLSetBlendModeUniformData(pdman, i.blendUni, bfp.mode)
	}
}
