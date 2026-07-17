// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GLSL shader implementations of the analytic rect/circle blur runtime effects that MakeRectBlur and MakeCircleBlur
// build. Each is a FragmentProcessor whose single child is a profile/integral texture effect sampled explicitly, and
// whose own math reads the fragment coordinate (device space) — so an axis-aligned rect or a circle is blurred by one
// textured draw instead of a mask render + separable convolution. Both share the runtime FP's ClassID and are separated
// in the key by a runtimeFPKind tag; the rect effect's isFast flag is a compile-time specialization baked into both the
// emitted source and the key.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

//////////////////////////////////////////////////////////////////////////////
// Circle blur.

type circleBlurFP struct {
	FPBase
	// circleData is (center.x, center.y, solidRadius, 1/textureRadius).
	circleData [4]float32
}

// NewCircleBlurFP builds the "CircleBlur" fragment processor: blurProfile is the 1-D circular-blur profile texture
// effect (sampled explicitly at (dist, 0.5)).
func NewCircleBlurFP(circleData [4]float32, blurProfile FragmentProcessor) FragmentProcessor {
	fp := &circleBlurFP{circleData: circleData}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	registerChildOf(fp, blurProfile, ExplicitSampleUsage())
	return fp
}

func (f *circleBlurFP) Name() string { return "CircleBlur" }

func (f *circleBlurFP) Clone() FragmentProcessor {
	return NewCircleBlurFP(f.circleData, f.ChildProcessor(0).Clone())
}

func (f *circleBlurFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPCircleBlur), "runtimeFPKind")
}

func (f *circleBlurFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*circleBlurFP)
	return ok && f.circleData == o.circleData
}

func (f *circleBlurFP) onMakeProgramImpl() FPProgramImpl { return &circleBlurFPImpl{} }

type circleBlurFPImpl struct {
	FPImplBase
	circleDataUni UniformHandle
}

func (i *circleBlurFPImpl) EmitCode(args *FPEmitArgs) {
	var circleDataName string
	i.circleDataUni, circleDataName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf4, "circleData")
	pos := args.FragBuilder.FragmentPosition()
	fb := args.FragBuilder
	// Compute "(length(vec) - circleData.z + 0.5) * circleData.w", rearranged to avoid passing large values to length()
	// that would overflow.
	fb.CodeAppendf("vec2 circleVec = (%s - %s.xy) * %s.w;", pos, circleDataName, circleDataName)
	fb.CodeAppendf("float circleDist = length(circleVec) + (0.5 - %s.z) * %s.w;",
		circleDataName, circleDataName)
	fb.CodeAppendf("vec4 profile = %s;",
		i.InvokeChildWithCoords(0, "", args, "vec2(circleDist, 0.5)"))
	fb.CodeAppend("return profile.aaaa;")
}

func (i *circleBlurFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*circleBlurFP)
	d := f.circleData
	pdman.Set4f(i.circleDataUni, d[0], d[1], d[2], d[3])
}

//////////////////////////////////////////////////////////////////////////////
// Rect blur.

type rectBlurFP struct {
	FPBase
	insetRect geom.Rect
	// isFast selects the nearest-edge (>= 6 sigma wide) lane; !isFast considers both edges.
	isFast bool
}

// NewRectBlurFP builds the "RectBlur" fragment processor: integral is the normal-distribution integral texture effect
// (linear-sampled, explicit coords). insetRect is the blurred rect inset by 3*sigma so its edge corresponds to t=0 in
// the integral texture. Unlike the circle effect, the rect shader evaluates in its own sample coordinate (MakeRectBlur
// wraps it in an optional matrix effect + a DeviceSpace FP), so callers feed it device space that way rather than
// reading the fragment coordinate here.
func NewRectBlurFP(insetRect geom.Rect, isFast bool, integral FragmentProcessor) FragmentProcessor {
	fp := &rectBlurFP{insetRect: insetRect, isFast: isFast}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, integral, ExplicitSampleUsage())
	return fp
}

func (f *rectBlurFP) Name() string { return "RectBlur" }

func (f *rectBlurFP) Clone() FragmentProcessor {
	return NewRectBlurFP(f.insetRect, f.isFast, f.ChildProcessor(0).Clone())
}

func (f *rectBlurFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPRectBlur), "runtimeFPKind")
	b.AddBool(f.isFast, "isFast")
}

func (f *rectBlurFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*rectBlurFP)
	return ok && f.isFast == o.isFast && f.insetRect == o.insetRect
}

func (f *rectBlurFP) onMakeProgramImpl() FPProgramImpl { return &rectBlurFPImpl{} }

type rectBlurFPImpl struct {
	FPImplBase
	rectUni UniformHandle
}

func (i *rectBlurFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*rectBlurFP)
	var rectName string
	i.rectUni, rectName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "rect")
	pos := args.SampleCoord
	fb := args.FragBuilder
	// The integral texture goes "backwards" (from 3*sigma to -3*sigma) and 'rect' was pre-inset by 3*sigma, both
	// accounted for below. The sample-child helper reads the integral's alpha at t.
	xEval := func(t string) string {
		return i.InvokeChildWithCoords(0, "", args, "vec2("+t+", 0.5)")
	}
	if fp.isFast {
		// Nearest horizontal/vertical edge lookups (rect is at least 6 sigma wide in both axes).
		fb.CodeAppendf("vec2 rectXY = max(%s.xy - %s, %s - %s.zw);", rectName, pos, pos, rectName)
		fb.CodeAppendf("vec4 xInt = %s;", xEval("rectXY.x"))
		fb.CodeAppendf("vec4 yInt = %s;", xEval("rectXY.y"))
		fb.CodeAppend("float xCoverage = xInt.a;")
		fb.CodeAppend("float yCoverage = yInt.a;")
	} else {
		// Consider both edges: C = 1 - <integral -inf..L> - <integral -inf..-R> per axis.
		fb.CodeAppendf("vec4 r4 = vec4(%s.xy - %s, %s - %s.zw);", rectName, pos, pos, rectName)
		fb.CodeAppendf("vec4 xL = %s;", xEval("r4.x"))
		fb.CodeAppendf("vec4 xR = %s;", xEval("r4.z"))
		fb.CodeAppendf("vec4 yT = %s;", xEval("r4.y"))
		fb.CodeAppendf("vec4 yB = %s;", xEval("r4.w"))
		fb.CodeAppend("float xCoverage = 1.0 - xL.a - xR.a;")
		fb.CodeAppend("float yCoverage = 1.0 - yT.a - yB.a;")
	}
	fb.CodeAppend("return vec4(xCoverage * yCoverage);")
}

func (i *rectBlurFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*rectBlurFP)
	r := f.insetRect
	pdman.Set4f(i.rectUni, r.Left, r.Top, r.Right, r.Bottom)
}

//////////////////////////////////////////////////////////////////////////////
// RRect blur.

type rrectBlurFP struct {
	FPBase
	cornerRadius float32
	proxyRect    geom.Rect
	blurRadius   float32
}

// NewRRectBlurFP builds the "RRectBlur" fragment processor: ninePatch is the blurred-rrect nine-patch mask texture
// effect (sampled explicitly at the warped coords). Like the circle effect it reads the fragment coordinate (device
// space) directly; its math warps the fragment into the nine-patch mask by snipping out the rrect's (uniform) middle
// section so a corner-radius-sized border tiles across the whole rrect.
func NewRRectBlurFP(cornerRadius float32, proxyRect geom.Rect, blurRadius float32, ninePatch FragmentProcessor) FragmentProcessor {
	fp := &rrectBlurFP{cornerRadius: cornerRadius, proxyRect: proxyRect, blurRadius: blurRadius}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	registerChildOf(fp, ninePatch, ExplicitSampleUsage())
	return fp
}

func (f *rrectBlurFP) Name() string { return "RRectBlur" }

func (f *rrectBlurFP) Clone() FragmentProcessor {
	return NewRRectBlurFP(f.cornerRadius, f.proxyRect, f.blurRadius, f.ChildProcessor(0).Clone())
}

func (f *rrectBlurFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPRRectBlur), "runtimeFPKind")
}

func (f *rrectBlurFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*rrectBlurFP)
	return ok && f.cornerRadius == o.cornerRadius && f.proxyRect == o.proxyRect &&
		f.blurRadius == o.blurRadius
}

func (f *rrectBlurFP) onMakeProgramImpl() FPProgramImpl { return &rrectBlurFPImpl{} }

type rrectBlurFPImpl struct {
	FPImplBase
	cornerRadiusUni UniformHandle
	proxyRectUni    UniformHandle
	blurRadiusUni   UniformHandle
}

func (i *rrectBlurFPImpl) EmitCode(args *FPEmitArgs) {
	var cornerRadiusName, proxyRectName, blurRadiusName string
	i.cornerRadiusUni, cornerRadiusName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "cornerRadius")
	i.proxyRectUni, proxyRectName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "proxyRect")
	i.blurRadiusUni, blurRadiusName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "blurRadius")
	pos := args.FragBuilder.FragmentPosition()
	fb := args.FragBuilder
	// Warp the fragment position to the appropriate part of the nine-patch mask by snipping out the middle section of
	// the proxy rect (the rrect's uniform edges/interior).
	fb.CodeAppendf("vec2 translatedFragPos = %s.xy - %s.xy;", pos, proxyRectName)
	fb.CodeAppendf("vec2 proxyCenter = (%s.zw - %s.xy) * 0.5;", proxyRectName, proxyRectName)
	fb.CodeAppendf("float edgeSize = 2.0 * %s + %s + 0.5;", blurRadiusName, cornerRadiusName)
	// Reposition so (0, 0) is the proxy-rect center; strip the sign so x/y increase away from center.
	fb.CodeAppend("translatedFragPos -= proxyCenter;")
	fb.CodeAppend("vec2 fragDirection = sign(translatedFragPos);")
	fb.CodeAppend("translatedFragPos = abs(translatedFragPos);")
	// Subtract the middle section and clamp it away, then restore the sign and re-center on the mask's upper-left, so
	// the fragment lands in the [0, 2*edgeSize] nine-patch border strip.
	fb.CodeAppend("vec2 warped = translatedFragPos - (proxyCenter - vec2(edgeSize));")
	fb.CodeAppend("warped = max(warped, vec2(0.0));")
	fb.CodeAppend("warped *= fragDirection;")
	fb.CodeAppend("warped += vec2(edgeSize);")
	fb.CodeAppend("vec2 proxyDims = vec2(2.0 * edgeSize);")
	fb.CodeAppend("vec2 texCoord = warped / proxyDims;")
	fb.CodeAppendf("vec4 profile = %s;", i.InvokeChildWithCoords(0, "", args, "texCoord"))
	fb.CodeAppend("return profile.aaaa;")
}

func (i *rrectBlurFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*rrectBlurFP)
	pdman.Set1f(i.cornerRadiusUni, f.cornerRadius)
	r := f.proxyRect
	pdman.Set4f(i.proxyRectUni, r.Left, r.Top, r.Right, r.Bottom)
	pdman.Set1f(i.blurRadiusUni, f.blurRadius)
}
