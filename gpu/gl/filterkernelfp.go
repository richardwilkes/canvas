// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hand-written GLSL forms of the shader-evaluating image filters' runtime effects: morphology, displacement, lighting's
// normal+lighting pair, magnifier, matrix convolution, and the arithmetic blend. These are the GPU twins of
// shaders/filterkernels.go, which documents the same formulas for the CPU pipeline. With these, every shader form the
// last CPU-only filters emit converts to an FP, so their DAGs evaluate on the GPU backend instead of falling back to a
// readback-CPU-upload path. Each FP starts with no optimization flags, derives usesSampleCoords from the effect, and
// computes each child's sample usage (explicit for kernel-offset taps, pass-through for same-coord evals). The kernel
// uniforms (radius, size/offset, light and material state, convolve-alpha) stay uniforms with runtime branches, so each
// effect compiles to one program; the one structural divergence is matrix convolution, where large kernels (28+ taps)
// keep their already quantization-reconstructed coefficients (see imagefilter/convolution.go) in a uniform array with a
// loop bound baked per kernel shape, well within desktop GL uniform limits at the 256-tap cap.

package gl

import (
	"fmt"
	"slices"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/shaders"
)

// maxLinearMorphologyRadius is the fixed loop bound for the linear morphology GLSL form; keep it in sync with the
// matching constants in the shaders and imagefilter morphology kernels.
const maxLinearMorphologyRadius = 14

// convertKernelChild converts a filter kernel's child shader against a fresh identity MatrixRec: the kernels sample
// their children at explicitly computed layer-space coordinates, and the pending matrix applies around the kernel FP
// itself instead (the same pattern used for the decal-tiled child conversion in MakeShaderFP). A nil child evaluates as
// transparent black.
func convertKernelChild(child shaders.Shader, args *FPArgs) (FragmentProcessor, bool) {
	if child == nil {
		return MakeColorFP(colorcore.PMColor4f{}), true
	}
	mRec := shaders.NewMatrixRec(geom.IdentityMatrix())
	fp := MakeShaderFP(child, args, &mRec)
	return fp, fp != nil
}

// wrapKernelFP applies the pending local-to-device matrix around a kernel FP. Returns nil when the total matrix is not
// invertible.
func wrapKernelFP(mRec *shaders.MatrixRec, fp FragmentProcessor) FragmentProcessor {
	identity := geom.IdentityMatrix()
	total, ok := mRec.ApplyForFragmentProcessor(&identity)
	if !ok {
		return nil
	}
	return MatrixEffectFP(total, fp)
}

//////////////////////////////////////////////////////////////////////////////
// Morphology (linear and sparse forms)

type morphologyFP struct {
	FPBase
	offset [2]float32
	flip   float32
	radius int32 // meaningful only for the linear form
	linear bool  // linear (radius-unrolled) form; false = sparse (single ±offset tap pair)
}

// makeMorphologyFP converts a MorphologyShader: the linear form is a radius-uniform loop over the fixed
// kMaxLinearRadius bound, the sparse form a single ±offset tap pair.
func makeMorphologyFP(s *shaders.MorphologyShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	child, ok := convertKernelChild(s.Child(), args)
	if !ok {
		return nil
	}
	off := s.Offset()
	fp := &morphologyFP{offset: [2]float32{off.X, off.Y}, flip: s.Flip(), radius: s.Radius(), linear: s.IsLinear()}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, child, ExplicitSampleUsage())
	return wrapKernelFP(mRec, fp)
}

func (f *morphologyFP) Name() string { return "Morphology" }

func (f *morphologyFP) Clone() FragmentProcessor {
	clone := &morphologyFP{offset: f.offset, flip: f.flip, radius: f.radius, linear: f.linear}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *morphologyFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	if f.linear {
		b.Add32(uint32(runtimeFPLinearMorphology), "runtimeFPKind")
	} else {
		b.Add32(uint32(runtimeFPSparseMorphology), "runtimeFPKind")
	}
}

func (f *morphologyFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*morphologyFP)
	return ok && f.offset == o.offset && f.flip == o.flip && f.radius == o.radius && f.linear == o.linear
}

func (f *morphologyFP) onMakeProgramImpl() FPProgramImpl { return &morphologyFPImpl{} }

type morphologyFPImpl struct {
	FPImplBase
	offsetUni UniformHandle
	flipUni   UniformHandle
	radiusUni UniformHandle
}

func (i *morphologyFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*morphologyFP)
	fb := args.FragBuilder
	var offName, flipName string
	i.offsetUni, offName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf2, "offset")
	i.flipUni, flipName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "flip")
	coord := args.SampleCoord
	child := func(coordExpr string) string { return i.InvokeChildWithCoords(0, "", args, coordExpr) }
	if fp.linear {
		// The linear morphology loop bound is the fixed maxLinearMorphologyRadius, with the actual radius kept as a
		// uniform, so every linear radius value shares one compiled program.
		var radiusName string
		i.radiusUni, radiusName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeInt, "radius")
		fb.CodeAppendf("vec4 aggregate = %s * %s;", flipName, child(coord))
		fb.CodeAppendf("vec2 delta = %s;", offName)
		fb.CodeAppendf("for (int i = 1; i <= %d; ++i) {", maxLinearMorphologyRadius)
		fb.CodeAppendf("if (i > %s) break;", radiusName)
		fb.CodeAppendf("aggregate = max(aggregate, max(%s * %s, %s * %s));",
			flipName, child(coord+" + delta"), flipName, child(coord+" - delta"))
		fb.CodeAppendf("delta += %s;", offName)
		fb.CodeAppend("}")
		fb.CodeAppendf("return %s * aggregate;", flipName)
	} else {
		// Sparse morphology form: a single +/-offset tap pair.
		fb.CodeAppendf("vec4 aggregate = max(%s * %s, %s * %s);",
			flipName, child(fmt.Sprintf("%s + %s", coord, offName)),
			flipName, child(fmt.Sprintf("%s - %s", coord, offName)))
		fb.CodeAppendf("return %s * aggregate;", flipName)
	}
}

func (i *morphologyFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*morphologyFP)
	pdman.Set2f(i.offsetUni, f.offset[0], f.offset[1])
	pdman.Set1f(i.flipUni, f.flip)
	if f.linear {
		pdman.Set1i(i.radiusUni, f.radius)
	}
}

//////////////////////////////////////////////////////////////////////////////
// Displacement

type displacementFP struct {
	FPBase
	scale      [2]float32
	xSel, ySel int32
}

// makeDisplacementFP converts a DisplacementShader: the displacement map is sampled pass-through, the color map
// explicitly at the displaced coordinates.
func makeDisplacementFP(s *shaders.DisplacementShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	displMap, ok := convertKernelChild(s.DisplMap(), args)
	if !ok {
		return nil
	}
	colorMap, ok := convertKernelChild(s.ColorMap(), args)
	if !ok {
		return nil
	}
	scale := s.Scale()
	xSel, ySel := s.ChannelSelectors()
	fp := &displacementFP{scale: [2]float32{scale.X, scale.Y}, xSel: xSel, ySel: ySel}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, displMap, PassThroughSampleUsage())
	registerChildOf(fp, colorMap, ExplicitSampleUsage())
	return wrapKernelFP(mRec, fp)
}

func (f *displacementFP) Name() string { return "Displacement" }

func (f *displacementFP) Clone() FragmentProcessor {
	clone := &displacementFP{scale: f.scale, xSel: f.xSel, ySel: f.ySel}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *displacementFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPDisplacement), "runtimeFPKind")
}

func (f *displacementFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*displacementFP)
	return ok && f.scale == o.scale && f.xSel == o.xSel && f.ySel == o.ySel
}

func (f *displacementFP) onMakeProgramImpl() FPProgramImpl { return &displacementFPImpl{} }

type displacementFPImpl struct {
	FPImplBase
	scaleUni   UniformHandle
	xSelectUni UniformHandle
	ySelectUni UniformHandle
}

func (i *displacementFPImpl) EmitCode(args *FPEmitArgs) {
	fb := args.FragBuilder
	var scaleName, xSelName, ySelName string
	i.scaleUni, scaleName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf2, "scale")
	i.xSelectUni, xSelName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "xSelect")
	i.ySelectUni, ySelName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "ySelect")

	// displ = scale * (unpremul(displMap.eval(coord)).sel - 0.5), then the color map samples at coord + displ.
	fb.CodeAppendf("vec4 displColor = %s;", i.InvokeChild(0 /* displMap */, args))
	fb.CodeAppend("displColor = vec4(displColor.rgb / max(displColor.a, 0.0001), displColor.a);")
	fb.CodeAppendf("vec2 displ = vec2(dot(displColor, %s), dot(displColor, %s));", xSelName, ySelName)
	fb.CodeAppendf("displ = %s * (displ - 0.5);", scaleName)
	fb.CodeAppendf("vec2 displCoord = %s + displ;", args.SampleCoord)
	fb.CodeAppendf("return %s;", i.InvokeChildWithCoords(1 /* colorMap */, "", args, "displCoord"))
}

func (i *displacementFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*displacementFP)
	pdman.Set2f(i.scaleUni, f.scale[0], f.scale[1])
	var xSel, ySel [4]float32
	xSel[f.xSel] = 1
	ySel[f.ySel] = 1
	pdman.Set4f(i.xSelectUni, xSel[0], xSel[1], xSel[2], xSel[3])
	pdman.Set4f(i.ySelectUni, ySel[0], ySel[1], ySel[2], ySel[3])
}

//////////////////////////////////////////////////////////////////////////////
// Normal map

type normalFP struct {
	FPBase
	edgeBounds      geom.Rect
	negSurfaceDepth float32
}

// makeNormalFP converts a NormalShader: a Sobel filter over nine clamped alpha taps.
func makeNormalFP(s *shaders.NormalShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	alphaMap, ok := convertKernelChild(s.AlphaMap(), args)
	if !ok {
		return nil
	}
	fp := &normalFP{edgeBounds: s.EdgeBounds(), negSurfaceDepth: s.NegSurfaceDepth()}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, alphaMap, ExplicitSampleUsage())
	return wrapKernelFP(mRec, fp)
}

func (f *normalFP) Name() string { return "NormalMap" }

func (f *normalFP) Clone() FragmentProcessor {
	clone := &normalFP{edgeBounds: f.edgeBounds, negSurfaceDepth: f.negSurfaceDepth}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *normalFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPNormal), "runtimeFPKind")
}

func (f *normalFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*normalFP)
	return ok && f.edgeBounds == o.edgeBounds && f.negSurfaceDepth == o.negSurfaceDepth
}

func (f *normalFP) onMakeProgramImpl() FPProgramImpl { return &normalFPImpl{} }

type normalFPImpl struct {
	FPImplBase
	edgeBoundsUni UniformHandle
	negDepthUni   UniformHandle
}

func (i *normalFPImpl) EmitCode(args *FPEmitArgs) {
	fb := args.FragBuilder
	var ebName, depthName string
	i.edgeBoundsUni, ebName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "edgeBounds")
	i.negDepthUni, depthName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "negSurfaceDepth")

	coord := args.SampleCoord
	tap := func(dx, dy string) string {
		return i.InvokeChildWithCoords(0, "", args,
			fmt.Sprintf("clamp(%s + vec2(%s, %s), %s.xy, %s.zw)", coord, dx, dy, ebName, ebName))
	}
	// Nine taps in column-major order: columns x = -1, 0, 1 with y = -1, 0, 1 within each.
	fb.CodeAppendf("vec3 alphaC0 = vec3(%s.a, %s.a, %s.a);",
		tap("-1.0", "-1.0"), tap("-1.0", "0.0"), tap("-1.0", "1.0"))
	fb.CodeAppendf("vec3 alphaC1 = vec3(%s.a, %s.a, %s.a);",
		tap("0.0", "-1.0"), tap("0.0", "0.0"), tap("0.0", "1.0"))
	fb.CodeAppendf("vec3 alphaC2 = vec3(%s.a, %s.a, %s.a);",
		tap("1.0", "-1.0"), tap("1.0", "0.0"), tap("1.0", "1.0"))
	// $normal_filter with kSobel = 0.25 * (1, 2, 1) folded to (0.25, 0.5, 0.25).
	fb.CodeAppend("const vec3 kSobel = vec3(0.25, 0.5, 0.25);")
	fb.CodeAppend("vec3 alphaR0 = vec3(alphaC0.x, alphaC1.x, alphaC2.x);")
	fb.CodeAppend("vec3 alphaR2 = vec3(alphaC0.z, alphaC1.z, alphaC2.z);")
	fb.CodeAppend("float nx = dot(kSobel, alphaC2) - dot(kSobel, alphaC0);")
	fb.CodeAppend("float ny = dot(kSobel, alphaR2) - dot(kSobel, alphaR0);")
	fb.CodeAppendf("vec3 n = normalize(vec3(%s * vec2(nx, ny), 1.0));", depthName)
	fb.CodeAppend("return vec4(n, alphaC1.y);")
}

func (i *normalFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*normalFP)
	pdman.Set4f(i.edgeBoundsUni, f.edgeBounds.Left, f.edgeBounds.Top, f.edgeBounds.Right,
		f.edgeBounds.Bottom)
	pdman.Set1f(i.negDepthUni, f.negSurfaceDepth)
}

//////////////////////////////////////////////////////////////////////////////
// Lighting

type lightingFP struct {
	FPBase
	materialAndLightType   [4]float32 // (depth, shininess, materialType, lightType)
	lightPosAndSpotFalloff [4]float32
	lightDirAndSpotCutoff  [4]float32
	lightColor             [3]float32
}

// makeLightingFP converts a LightingShader: the Phong lighting equation over the normal-map child (sampled
// pass-through). All light and material state stays uniforms with runtime branches, so the six distant/point/spot ×
// diffuse/specular constructors share one program.
func makeLightingFP(s *shaders.LightingShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	normalMap, ok := convertKernelChild(s.NormalMap(), args)
	if !ok {
		return nil
	}
	mlt, lpf, ldc, lightColor := s.PackedUniforms()
	fp := &lightingFP{
		materialAndLightType: mlt, lightPosAndSpotFalloff: lpf,
		lightDirAndSpotCutoff: ldc, lightColor: lightColor,
	}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, normalMap, PassThroughSampleUsage())
	return wrapKernelFP(mRec, fp)
}

func (f *lightingFP) Name() string { return "Lighting" }

func (f *lightingFP) Clone() FragmentProcessor {
	clone := &lightingFP{
		materialAndLightType:   f.materialAndLightType,
		lightPosAndSpotFalloff: f.lightPosAndSpotFalloff,
		lightDirAndSpotCutoff:  f.lightDirAndSpotCutoff, lightColor: f.lightColor,
	}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *lightingFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPLighting), "runtimeFPKind")
}

func (f *lightingFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*lightingFP)
	return ok && f.materialAndLightType == o.materialAndLightType &&
		f.lightPosAndSpotFalloff == o.lightPosAndSpotFalloff &&
		f.lightDirAndSpotCutoff == o.lightDirAndSpotCutoff && f.lightColor == o.lightColor
}

func (f *lightingFP) onMakeProgramImpl() FPProgramImpl { return &lightingFPImpl{} }

type lightingFPImpl struct {
	FPImplBase
	materialUni   UniformHandle
	lightPosUni   UniformHandle
	lightDirUni   UniformHandle
	lightColorUni UniformHandle
}

func (i *lightingFPImpl) EmitCode(args *FPEmitArgs) {
	fb := args.FragBuilder
	var mltName, lpfName, ldcName, colName string
	i.materialUni, mltName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "materialAndLightType")
	i.lightPosUni, lpfName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "lightPosAndSpotFalloff")
	i.lightDirUni, ldcName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "lightDirAndSpotCutoff")
	i.lightColorUni, colName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf3, "lightColor")

	fb.CodeAppendf("vec4 normalAndA = %s;", i.InvokeChild(0 /* normalMap */, args))
	// $surface_to_light: spot (> 0) and point (== 0) share normalize(lightPos - coord); distant lights use the
	// pre-normalized uniform direction.
	fb.CodeAppend("vec3 surfaceToLight;")
	fb.CodeAppendf("if (%s.w >= 0.0) {", mltName)
	fb.CodeAppendf("surfaceToLight = normalize(%s.xyz - vec3(%s, %s.x * normalAndA.a));",
		lpfName, args.SampleCoord, mltName)
	fb.CodeAppendf("} else { surfaceToLight = %s.xyz; }", ldcName)
	fb.CodeAppendf("vec3 color = %s;", colName)
	// $spotlight_scale, with kConeAAThreshold = 0.016 and kConeScale = 1/kConeAAThreshold.
	fb.CodeAppendf("if (%s.w > 0.0) {", mltName)
	fb.CodeAppendf("float cosAngle = -dot(surfaceToLight, %s.xyz);", ldcName)
	fb.CodeAppend("float scale = 0.0;")
	fb.CodeAppendf("if (cosAngle >= %s.w) {", ldcName)
	fb.CodeAppendf("scale = pow(cosAngle, %s.w);", lpfName)
	fb.CodeAppendf("if (cosAngle < %s.w + 0.016) { scale *= (cosAngle - %s.w) * (1.0 / 0.016); }",
		ldcName, ldcName)
	fb.CodeAppend("}")
	fb.CodeAppend("color *= scale;")
	fb.CodeAppend("}")
	// $compute_lighting's diffuse/specular lanes; the emboss material is not publicly reachable and is trimmed like the
	// CPU kernel.
	fb.CodeAppendf("if (%s.z == 0.0) {", mltName)
	fb.CodeAppend("float coeff = dot(normalAndA.xyz, surfaceToLight);")
	fb.CodeAppend("color = clamp(coeff * color, 0.0, 1.0);")
	fb.CodeAppend("return vec4(color, 1.0);")
	fb.CodeAppend("} else {")
	fb.CodeAppend("vec3 halfDir = normalize(surfaceToLight + vec3(0.0, 0.0, 1.0));")
	fb.CodeAppendf("float coeff = pow(dot(normalAndA.xyz, halfDir), %s.y);", mltName)
	fb.CodeAppend("color = clamp(coeff * color, 0.0, 1.0);")
	fb.CodeAppend("return vec4(color, max(max(color.r, color.g), color.b));")
	fb.CodeAppend("}")
}

func (i *lightingFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*lightingFP)
	pdman.Set4f(i.materialUni, f.materialAndLightType[0], f.materialAndLightType[1],
		f.materialAndLightType[2], f.materialAndLightType[3])
	pdman.Set4f(i.lightPosUni, f.lightPosAndSpotFalloff[0], f.lightPosAndSpotFalloff[1],
		f.lightPosAndSpotFalloff[2], f.lightPosAndSpotFalloff[3])
	pdman.Set4f(i.lightDirUni, f.lightDirAndSpotCutoff[0], f.lightDirAndSpotCutoff[1],
		f.lightDirAndSpotCutoff[2], f.lightDirAndSpotCutoff[3])
	pdman.Set3f(i.lightColorUni, f.lightColor[0], f.lightColor[1], f.lightColor[2])
}

//////////////////////////////////////////////////////////////////////////////
// Magnifier

type magnifierFP struct {
	FPBase
	lensBounds geom.Rect
	zoomXform  [4]float32
	invInset   [2]float32
}

// makeMagnifierFP converts a MagnifierShader: coordinates distort toward the zoomed source rect with the rounded-inset
// weighting, then the source samples explicitly.
func makeMagnifierFP(s *shaders.MagnifierShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	src, ok := convertKernelChild(s.Src(), args)
	if !ok {
		return nil
	}
	fp := &magnifierFP{lensBounds: s.LensBounds(), zoomXform: s.ZoomXform(), invInset: s.InvInset()}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, src, ExplicitSampleUsage())
	return wrapKernelFP(mRec, fp)
}

func (f *magnifierFP) Name() string { return "Magnifier" }

func (f *magnifierFP) Clone() FragmentProcessor {
	clone := &magnifierFP{lensBounds: f.lensBounds, zoomXform: f.zoomXform, invInset: f.invInset}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *magnifierFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPMagnifier), "runtimeFPKind")
}

func (f *magnifierFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*magnifierFP)
	return ok && f.lensBounds == o.lensBounds && f.zoomXform == o.zoomXform &&
		f.invInset == o.invInset
}

func (f *magnifierFP) onMakeProgramImpl() FPProgramImpl { return &magnifierFPImpl{} }

type magnifierFPImpl struct {
	FPImplBase
	lensBoundsUni UniformHandle
	zoomXformUni  UniformHandle
	invInsetUni   UniformHandle
}

func (i *magnifierFPImpl) EmitCode(args *FPEmitArgs) {
	fb := args.FragBuilder
	var lensName, zoomName, insetName string
	i.lensBoundsUni, lensName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "lensBounds")
	i.zoomXformUni, zoomName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "zoomXform")
	i.invInsetUni, insetName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat2, "invInset")

	coord := args.SampleCoord
	fb.CodeAppendf("vec2 zoomCoord = %s.xy + %s.zw * %s;", zoomName, zoomName, coord)
	fb.CodeAppendf("vec2 edgeInset = min(%s - %s.xy, %s.zw - %s) * %s;", coord, lensName, lensName,
		coord, insetName)
	// Weighting rationale: 0 along lensBounds, a circular corner falloff within 2 insets of a corner, linear
	// compression elsewhere.
	fb.CodeAppend("float weight = all(lessThan(edgeInset, vec2(2.0)))" +
		" ? (2.0 - length(2.0 - edgeInset)) : min(edgeInset.x, edgeInset.y);")
	fb.CodeAppend("weight = clamp(weight, 0.0, 1.0);")
	fb.CodeAppendf("vec2 zoomedCoord = mix(%s, zoomCoord, weight * weight);", coord)
	fb.CodeAppendf("return %s;", i.InvokeChildWithCoords(0, "", args, "zoomedCoord"))
}

func (i *magnifierFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*magnifierFP)
	pdman.Set4f(i.lensBoundsUni, f.lensBounds.Left, f.lensBounds.Top, f.lensBounds.Right,
		f.lensBounds.Bottom)
	pdman.Set4f(i.zoomXformUni, f.zoomXform[0], f.zoomXform[1], f.zoomXform[2], f.zoomXform[3])
	pdman.Set2f(i.invInsetUni, f.invInset[0], f.invInset[1])
}

//////////////////////////////////////////////////////////////////////////////
// Matrix convolution (see the file comment for the large-kernel storage tradeoff)

type matrixConvFP struct {
	kernel []float32
	FPBase
	size          geom.ISize
	offset        geom.IPoint
	gain          float32
	bias          float32
	convolveAlpha bool
}

// makeMatrixConvFP converts a MatrixConvShader. The kernel coefficients arrive already quantization-reconstructed for
// 28+ taps, so the uniform lowering reproduces the same values regardless of kernel size.
func makeMatrixConvFP(s *shaders.MatrixConvShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	child, ok := convertKernelChild(s.Child(), args)
	if !ok {
		return nil
	}
	gain, bias := s.GainAndBias()
	fp := &matrixConvFP{
		kernel: s.Kernel(), size: s.Size(), offset: s.Offset(), gain: gain,
		bias: bias, convolveAlpha: s.ConvolveAlpha(),
	}
	fp.initFP(GLSLFPClassID, 0)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, child, ExplicitSampleUsage())
	return wrapKernelFP(mRec, fp)
}

func (f *matrixConvFP) Name() string { return "MatrixConvolution" }

func (f *matrixConvFP) Clone() FragmentProcessor {
	clone := &matrixConvFP{
		kernel: f.kernel, size: f.size, offset: f.offset, gain: f.gain,
		bias: f.bias, convolveAlpha: f.convolveAlpha,
	}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *matrixConvFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPMatrixConv), "runtimeFPKind")
	// The loop structure bakes the kernel shape rather than sharing one program across a fixed max uniform array with a
	// runtime size + break check.
	b.Add32(uint32(f.size.Width), "convKernelWidth")
	b.Add32(uint32(f.size.Height), "convKernelHeight")
}

func (f *matrixConvFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*matrixConvFP)
	return ok && slices.Equal(f.kernel, o.kernel) && f.size == o.size && f.offset == o.offset &&
		f.gain == o.gain && f.bias == o.bias && f.convolveAlpha == o.convolveAlpha
}

func (f *matrixConvFP) onMakeProgramImpl() FPProgramImpl { return &matrixConvFPImpl{} }

type matrixConvFPImpl struct {
	FPImplBase
	kernelUni        UniformHandle
	offsetUni        UniformHandle
	gainAndBiasUni   UniformHandle
	convolveAlphaUni UniformHandle
}

// convKernelVec4s is the uniform array length for a kernel: taps packed four to a vec4.
func convKernelVec4s(kernelLen int) int { return (kernelLen + 3) / 4 }

func (i *matrixConvFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*matrixConvFP)
	fb := args.FragBuilder
	var kernelName, offName, gbName, caName string
	i.kernelUni, kernelName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "kernel", convKernelVec4s(len(fp.kernel)))
	i.offsetUni, offName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf2, "offset")
	i.gainAndBiasUni, gbName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf2, "gainAndBias")
	i.convolveAlphaUni, caName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeInt, "convolveAlpha")

	coord := args.SampleCoord
	area := int(fp.size.Width) * int(fp.size.Height)
	fb.CodeAppend("vec4 sum = vec4(0.0);")
	fb.CodeAppend("float origAlpha = 0.0;")
	fb.CodeAppendf("for (int i = 0; i < %d; ++i) {", area)
	fb.CodeAppendf("vec2 kernelPos = vec2(float(i %% %d), float(i / %d));", fp.size.Width,
		fp.size.Width)
	fb.CodeAppendf("float k = %s[i / 4][i - (i / 4) * 4];", kernelName)
	fb.CodeAppendf("vec4 c = %s;", i.InvokeChildWithCoords(0, "", args,
		fmt.Sprintf("(%s + kernelPos) - %s", coord, offName)))
	fb.CodeAppendf("if (%s == 0) {", caName)
	// When not convolving alpha, remember the original alpha at the actual sample coord and accumulate on unpremul
	// colors.
	fb.CodeAppendf("if (kernelPos == %s) { origAlpha = c.a; }", offName)
	fb.CodeAppend("c = vec4(c.rgb / max(c.a, 0.0001), c.a);")
	fb.CodeAppend("}")
	fb.CodeAppend("sum += c * k;")
	fb.CodeAppend("}")
	fb.CodeAppendf("vec4 color = sum * %s.x + %s.y;", gbName, gbName)
	fb.CodeAppendf("if (%s == 0) {", caName)
	// Reset the alpha to the original and convert to premul RGB.
	fb.CodeAppend("color = vec4(color.rgb * origAlpha, origAlpha);")
	fb.CodeAppend("} else {")
	// Ensure convolved alpha is within [0, 1].
	fb.CodeAppend("color.a = clamp(color.a, 0.0, 1.0);")
	fb.CodeAppend("}")
	// Make RGB valid premul w/ respect to the alpha (either original or convolved).
	fb.CodeAppend("color.rgb = clamp(color.rgb, vec3(0.0), vec3(color.a));")
	fb.CodeAppend("return color;")
}

func (i *matrixConvFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*matrixConvFP)
	n := convKernelVec4s(len(f.kernel))
	padded := make([]float32, n*4)
	copy(padded, f.kernel)
	pdman.Set4fv(i.kernelUni, int32(n), padded)
	pdman.Set2f(i.offsetUni, float32(f.offset.X), float32(f.offset.Y))
	pdman.Set2f(i.gainAndBiasUni, f.gain, f.bias)
	ca := int32(0)
	if f.convolveAlpha {
		ca = 1
	}
	pdman.Set1i(i.convolveAlphaUni, ca)
}

//////////////////////////////////////////////////////////////////////////////
// Arithmetic blend (combines two child shaders' outputs via the arithmetic blend equation)

type arithmeticFP struct {
	FPBase
	k       [4]float32
	pmClamp float32
}

// makeArithmeticFP converts an ArithmeticShader: like MakeShaderFP's BlendShader case, both children convert against
// the incoming MatrixRec and are sampled pass-through; the arithmetic equation combines their outputs, collapsed into a
// single FP.
func makeArithmeticFP(s *shaders.ArithmeticShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	dstFP := MakeShaderFP(s.Dst(), args, mRec)
	srcFP := MakeShaderFP(s.Src(), args, mRec)
	if dstFP == nil || srcFP == nil {
		return nil
	}
	pmClamp := float32(1)
	if s.EnforcePremul() {
		pmClamp = 0
	}
	fp := &arithmeticFP{k: s.K(), pmClamp: pmClamp}
	fp.initFP(GLSLFPClassID, 0)
	registerChildOf(fp, srcFP, PassThroughSampleUsage())
	registerChildOf(fp, dstFP, PassThroughSampleUsage())
	return fp
}

func (f *arithmeticFP) Name() string { return "ArithmeticBlend" }

func (f *arithmeticFP) Clone() FragmentProcessor {
	clone := &arithmeticFP{k: f.k, pmClamp: f.pmClamp}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *arithmeticFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPArithmetic), "runtimeFPKind")
}

func (f *arithmeticFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*arithmeticFP)
	return ok && f.k == o.k && f.pmClamp == o.pmClamp
}

func (f *arithmeticFP) onMakeProgramImpl() FPProgramImpl { return &arithmeticFPImpl{} }

type arithmeticFPImpl struct {
	FPImplBase
	kUni       UniformHandle
	pmClampUni UniformHandle
}

func (i *arithmeticFPImpl) EmitCode(args *FPEmitArgs) {
	fb := args.FragBuilder
	var kName, pmName string
	i.kUni, kName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf4, "k")
	i.pmClampUni, pmName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "pmClamp")

	fb.CodeAppendf("vec4 srcColor = %s;", i.InvokeChild(0, args))
	fb.CodeAppendf("vec4 dstColor = %s;", i.InvokeChild(1, args))
	fb.CodeAppendf("vec4 color = clamp(%s.x * srcColor * dstColor + %s.y * srcColor + "+
		"%s.z * dstColor + %s.w, 0.0, 1.0);", kName, kName, kName, kName)
	fb.CodeAppendf("color.rgb = min(color.rgb, max(color.a, %s));", pmName)
	fb.CodeAppend("return color;")
}

func (i *arithmeticFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*arithmeticFP)
	pdman.Set4f(i.kUni, f.k[0], f.k[1], f.k[2], f.k[3])
	pdman.Set1f(i.pmClampUni, f.pmClamp)
}
