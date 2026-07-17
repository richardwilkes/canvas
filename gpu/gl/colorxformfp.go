// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The color-space transform fragment processor and its GLSL emission, reduced to what an sRGB-only pipeline can reach:
// the step set is {unpremul, linearize via the sRGB transfer function, encode via its inverse, premul}, there is never
// a gamut matrix or OOTF, and the transfer functions are always sRGBish. The gradient interpolated-to-dst premul step
// and the high-contrast working-format sandwich are the two producers.

package gl

import (
	"fmt"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/shaders"
)

// Color-xform steps (the reachable subset of possible color-space transform steps).
const (
	// XformStepUnpremul divides RGB by alpha on entry.
	XformStepUnpremul uint32 = 1 << 0
	// XformStepLinearize applies the sRGB transfer function (EOTF) per channel.
	XformStepLinearize uint32 = 1 << 1
	// XformStepEncode applies the inverse sRGB transfer function (OETF) per channel.
	XformStepEncode uint32 = 1 << 2
	// XformStepPremul multiplies RGB by alpha on exit.
	XformStepPremul uint32 = 1 << 3
)

type colorXformFP struct {
	FPBase
	steps uint32
}

// ColorXformFP wraps child (which may be nil, meaning the input color) in the given color-space steps. Returns child
// unchanged when there are no steps.
func ColorXformFP(child FragmentProcessor, steps uint32) FragmentProcessor {
	if steps == 0 {
		return child
	}
	fp := &colorXformFP{steps: steps}
	// Optimization flags: coverage-as-alpha, preserves-opaque, and constant-output, intersected with the child's flags
	// (a nil child contributes all flags).
	declared := FPCompatibleWithCoverageAsAlpha | FPPreservesOpaqueInput |
		FPConstantOutputForConstantInput
	fp.initFP(ColorSpaceXformEffectClassID, declared)
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(child) &
		(FPCompatibleWithCoverageAsAlpha | FPPreservesOpaqueInput |
			FPConstantOutputForConstantInput))
	registerChildOf(fp, child, PassThroughSampleUsage())
	return fp
}

func (f *colorXformFP) Name() string { return "ColorSpaceXform" }

func (f *colorXformFP) Clone() FragmentProcessor {
	clone := &colorXformFP{steps: f.steps}
	clone.initFP(ColorSpaceXformEffectClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *colorXformFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	// The steps flags (transfer-function types are always sRGBish here; the coefficients are uniforms).
	b.Add32(f.steps, "xformKey")
}

func (f *colorXformFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*colorXformFP)
	return ok && f.steps == o.steps
}

func (f *colorXformFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	c := fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
	return applyColorXformSteps(f.steps, c)
}

// applyColorXformSteps is the CPU-side equivalent of the emitted shader (using the scalar transfer-function math).
func applyColorXformSteps(steps uint32, c colorcore.PMColor4f) colorcore.PMColor4f {
	if steps&XformStepUnpremul != 0 {
		if c.A != 0 {
			inv := 1 / c.A
			c.R *= inv
			c.G *= inv
			c.B *= inv
		} else {
			c.R, c.G, c.B = 0, 0, 0
		}
	}
	if steps&XformStepLinearize != 0 {
		tf := colorfilter.SRGBTF()
		c.R = colorfilter.EvalTransferFn(&tf, c.R)
		c.G = colorfilter.EvalTransferFn(&tf, c.G)
		c.B = colorfilter.EvalTransferFn(&tf, c.B)
	}
	if steps&XformStepEncode != 0 {
		tf := colorfilter.SRGBInvTF()
		c.R = colorfilter.EvalTransferFn(&tf, c.R)
		c.G = colorfilter.EvalTransferFn(&tf, c.G)
		c.B = colorfilter.EvalTransferFn(&tf, c.B)
	}
	if steps&XformStepPremul != 0 {
		c.R *= c.A
		c.G *= c.A
		c.B *= c.A
	}
	return c
}

func (f *colorXformFP) onMakeProgramImpl() FPProgramImpl { return &colorXformFPImpl{} }

type colorXformFPImpl struct {
	FPImplBase
	srcTFUni UniformHandle
	dstTFUni UniformHandle
}

// emitTFFunc emits a parametric transfer-function evaluator (the sRGBish branch).
func (i *colorXformFPImpl) emitTFFunc(args *FPEmitArgs, name string) (handle UniformHandle, fnName string) {
	uni, coeffs := args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment, GLSLTypeFloat,
		name, 7)
	body := fmt.Sprintf("float G = %[1]s[0];\n"+
		"float A = %[1]s[1];\n"+
		"float B = %[1]s[2];\n"+
		"float C = %[1]s[3];\n"+
		"float D = %[1]s[4];\n"+
		"float E = %[1]s[5];\n"+
		"float F = %[1]s[6];\n"+
		"float s = sign(x);\n"+
		"x = abs(x);\n"+
		"x = (x < D) ? (C * x) + F : pow(A * x + B, G) + E;\n"+
		"return s * x;", coeffs)
	fnName = args.FragBuilder.GetMangledFunctionName(name)
	args.FragBuilder.EmitFunction(GLSLTypeFloat, fnName,
		[]ShaderVar{NewShaderVar("x", GLSLTypeFloat)}, body)
	return uni, fnName
}

func (i *colorXformFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*colorXformFP)
	fb := args.FragBuilder

	srcTFFunc := ""
	if fp.steps&XformStepLinearize != 0 {
		i.srcTFUni, srcTFFunc = i.emitTFFunc(args, "src_tf")
	}
	dstTFFunc := ""
	if fp.steps&XformStepEncode != 0 {
		i.dstTFUni, dstTFFunc = i.emitTFFunc(args, "dst_tf")
	}

	// The wrapper function that applies all the intermediate steps.
	var body string
	if fp.steps&XformStepUnpremul != 0 {
		body += "color = vec4(color.rgb / max(color.a, 0.0001), color.a);\n"
	}
	if srcTFFunc != "" {
		body += fmt.Sprintf("color.r = %[1]s(color.r);\ncolor.g = %[1]s(color.g);\n"+
			"color.b = %[1]s(color.b);\n", srcTFFunc)
	}
	if dstTFFunc != "" {
		body += fmt.Sprintf("color.r = %[1]s(color.r);\ncolor.g = %[1]s(color.g);\n"+
			"color.b = %[1]s(color.b);\n", dstTFFunc)
	}
	if fp.steps&XformStepPremul != 0 {
		body += "color.rgb *= color.a;\n"
	}
	body += "return color;"
	xformFunc := fb.GetMangledFunctionName("color_xform")
	fb.EmitFunction(GLSLTypeHalf4, xformFunc,
		[]ShaderVar{NewShaderVar("color", GLSLTypeHalf4)}, body)

	fb.CodeAppendf("return %s(%s);", xformFunc, i.InvokeChild(0, args))
}

func (i *colorXformFPImpl) onSetData(pdman *ProgramDataManager, _ FragmentProcessor) {
	if i.srcTFUni.IsValid() {
		tf := colorfilter.SRGBTF()
		pdman.Set1fv(i.srcTFUni, 7, transferFnCoeffs(&tf))
	}
	if i.dstTFUni.IsValid() {
		tf := colorfilter.SRGBInvTF()
		pdman.Set1fv(i.dstTFUni, 7, transferFnCoeffs(&tf))
	}
}

// transferFnCoeffs packs a parametric transfer function in the shader's {G,A,B,C,D,E,F} order.
func transferFnCoeffs(tf *shaders.TransferFn) []float32 {
	return []float32{tf.G, tf.A, tf.B, tf.C, tf.D, tf.E, tf.F}
}
