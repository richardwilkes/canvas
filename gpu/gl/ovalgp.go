// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The geometry processors for drawing ovals: circleGeometryProcessor (with the arc clip planes and round caps),
// ellipseGeometryProcessor, and diEllipseGeometryProcessor, emitting GLSL directly (saturate() lowers to clamp, the
// half types widen to float — the desktop targets have no mediump). There is no dashed-circle geometry processor: this
// library's style carries no path effects (they devolve to paths at the device), so that lane is unreachable.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

//////////////////////////////////////////////////////////////////////////////
// CircleGeometryProcessor

// circleGeometryProcessor produces a modulation of the input color and coverage for a circle, operating in a space
// normalized by the circle radius (outer radius for strokes) with origin at the circle center. Additional clip planes
// support arcs, and round caps for stroking are supported through cap-center circles.
type circleGeometryProcessor struct {
	// attrs: inPosition, inColor, inCircleEdge, then the optional inClipPlane, inIsectPlane, inUnionPlane,
	// inRoundCapCenters.
	attrs [7]Attribute
	GPBase
	localMatrix geom.Matrix
	stroke      bool
}

func makeCircleGeometryProcessor(stroke, clipPlane, isectPlane, unionPlane, roundCaps, wideColor bool, localMatrix *geom.Matrix) GeometryProcessor {
	// Borrow the shell from the free list (gppool.go); the struct-literal assignment zeroes the inline attrs array,
	// re-populated by the conditional sets below.
	gp := circleGPPool.borrow()
	*gp = circleGeometryProcessor{localMatrix: *localMatrix, stroke: stroke}
	gp.initGP(CircleGeometryProcessorClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.attrs[1] = MakeColorAttribute("inColor", wideColor)
	gp.attrs[2] = MakeAttribute("inCircleEdge", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	if clipPlane {
		gp.attrs[3] = MakeAttribute("inClipPlane", gpu.VertexAttribTypeFloat3, GLSLTypeHalf3)
	}
	if isectPlane {
		gp.attrs[4] = MakeAttribute("inIsectPlane", gpu.VertexAttribTypeFloat3, GLSLTypeHalf3)
	}
	if unionPlane {
		gp.attrs[5] = MakeAttribute("inUnionPlane", gpu.VertexAttribTypeFloat3, GLSLTypeHalf3)
	}
	if roundCaps {
		if !clipPlane {
			panic("round caps require the clip plane")
		}
		gp.attrs[6] = MakeAttribute("inRoundCapCenters", gpu.VertexAttribTypeFloat4,
			GLSLTypeFloat4)
	}
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

func (g *circleGeometryProcessor) Name() string { return "CircleGeometryProcessor" }

func (g *circleGeometryProcessor) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBool(g.stroke, "stroked")
	b.AddBool(g.attrs[3].IsInitialized(), "clipPlane")
	b.AddBool(g.attrs[4].IsInitialized(), "isectPlane")
	b.AddBool(g.attrs[5].IsInitialized(), "unionPlane")
	b.AddBool(g.attrs[6].IsInitialized(), "roundCapCenters")
	b.AddBits(matrixKeyBits, ComputeMatrixKey(caps, &g.localMatrix), "localMatrixType")
}

func (g *circleGeometryProcessor) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &circleGPImpl{}
}

// circleGPImpl is the program implementation for circleGeometryProcessor.
type circleGPImpl struct {
	GPImplBase
	localMatrixPrev    geom.Matrix
	localMatrixPrevSet bool
	localMatrixUniform UniformHandle
}

func (i *circleGPImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	cgp := geomProc.(*circleGeometryProcessor)
	SetTransform(pdman, caps, i.localMatrixUniform, &cgp.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
}

func (i *circleGPImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	cgp := args.GeomProc.(*circleGeometryProcessor)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler
	fragBuilder := args.FragBuilder

	// Emit attributes.
	varyingHandler.EmitAttributes(cgp)
	fragBuilder.CodeAppend("vec4 circleEdge;")
	varyingHandler.AddPassThroughAttribute(cgp.attrs[2].AsShaderVar(), "circleEdge",
		InterpolationInterpolated)
	if cgp.attrs[3].IsInitialized() {
		fragBuilder.CodeAppend("vec3 clipPlane;")
		varyingHandler.AddPassThroughAttribute(cgp.attrs[3].AsShaderVar(), "clipPlane",
			InterpolationInterpolated)
	}
	if cgp.attrs[4].IsInitialized() {
		fragBuilder.CodeAppend("vec3 isectPlane;")
		varyingHandler.AddPassThroughAttribute(cgp.attrs[4].AsShaderVar(), "isectPlane",
			InterpolationInterpolated)
	}
	if cgp.attrs[5].IsInitialized() {
		if !cgp.attrs[3].IsInitialized() {
			panic("union plane requires the clip plane")
		}
		fragBuilder.CodeAppend("vec3 unionPlane;")
		varyingHandler.AddPassThroughAttribute(cgp.attrs[5].AsShaderVar(), "unionPlane",
			InterpolationInterpolated)
	}
	capRadius := NewVarying(GLSLTypeFloat)
	if cgp.attrs[6].IsInitialized() {
		fragBuilder.CodeAppend("vec4 roundCapCenters;")
		varyingHandler.AddPassThroughAttribute(cgp.attrs[6].AsShaderVar(), "roundCapCenters",
			InterpolationInterpolated)
		varyingHandler.AddVaryingWithInterpolation("capRadius", &capRadius,
			InterpolationCanBeFlat)
		// This is the cap radius in normalized space where the outer radius is 1 and circleEdge.w is the normalized
		// inner radius.
		vertBuilder.CodeAppendf("%s = (1.0 - %s.w) / 2.0;", capRadius.VsOut(),
			cgp.attrs[2].Name())
	}

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(cgp.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationCanBeFlat)

	// Set up position.
	WriteOutputPosition(vertBuilder, gpArgs, cgp.attrs[0].Name())
	WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		cgp.attrs[0].AsShaderVar(), &cgp.localMatrix, &i.localMatrixUniform)

	fragBuilder.CodeAppend("float d = length(circleEdge.xy);")
	fragBuilder.CodeAppend("float distanceToOuterEdge = circleEdge.z * (1.0 - d);")
	fragBuilder.CodeAppend("float edgeAlpha = clamp(distanceToOuterEdge, 0.0, 1.0);")
	if cgp.stroke {
		fragBuilder.CodeAppend(
			"float distanceToInnerEdge = circleEdge.z * (d - circleEdge.w);",
		)
		fragBuilder.CodeAppend("float innerAlpha = clamp(distanceToInnerEdge, 0.0, 1.0);")
		fragBuilder.CodeAppend("edgeAlpha *= innerAlpha;")
	}

	if cgp.attrs[3].IsInitialized() {
		fragBuilder.CodeAppend(
			"float clip = clamp(circleEdge.z * dot(circleEdge.xy, " +
				"clipPlane.xy) + clipPlane.z, 0.0, 1.0);",
		)
		if cgp.attrs[4].IsInitialized() {
			fragBuilder.CodeAppend(
				"clip *= clamp(circleEdge.z * dot(circleEdge.xy, " +
					"isectPlane.xy) + isectPlane.z, 0.0, 1.0);",
			)
		}
		if cgp.attrs[5].IsInitialized() {
			fragBuilder.CodeAppend(
				"clip = clamp(clip + clamp(circleEdge.z * dot(circleEdge.xy," +
					" unionPlane.xy) + unionPlane.z, 0.0, 1.0), 0.0, 1.0);",
			)
		}
		fragBuilder.CodeAppend("edgeAlpha *= clip;")
		if cgp.attrs[6].IsInitialized() {
			// We compute coverage of the round caps as circles at the butt caps produced by the clip planes. The
			// inverse of the clip planes is applied so that there is no double counting.
			fragBuilder.CodeAppendf(
				"float dcap1 = circleEdge.z * (%s - length(circleEdge.xy - "+
					"roundCapCenters.xy));"+
					"float dcap2 = circleEdge.z * (%s - length(circleEdge.xy - "+
					"roundCapCenters.zw));"+
					"float capAlpha = (1.0 - clip) * (max(dcap1, 0.0) + max(dcap2, 0.0));"+
					"edgeAlpha = min(edgeAlpha + capAlpha, 1.0);",
				capRadius.FsIn(), capRadius.FsIn(),
			)
		}
	}
	fragBuilder.CodeAppendf("vec4 %s = vec4(edgeAlpha);", args.OutputCoverage)
}

//////////////////////////////////////////////////////////////////////////////
// EllipseGeometryProcessor

// ellipseGeometryProcessor produces a modulation of the input color and coverage for an ellipse, specified as a 2D
// offset from center, an outer vector radii pair, and an inner pair for strokes.
type ellipseGeometryProcessor struct {
	// attrs: inPosition, inColor, inEllipseOffset, inEllipseRadii.
	attrs [4]Attribute
	GPBase
	localMatrix geom.Matrix
	stroke      bool
	useScale    bool
}

func makeEllipseGeometryProcessor(stroke, wideColor, useScale bool, localMatrix *geom.Matrix) GeometryProcessor {
	// Borrow the shell from the free list (gppool.go); the struct-literal assignment zeroes the inline attrs array,
	// re-populated by the sets below.
	gp := ellipseGPPool.borrow()
	*gp = ellipseGeometryProcessor{localMatrix: *localMatrix, stroke: stroke, useScale: useScale}
	gp.initGP(EllipseGeometryProcessorClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.attrs[1] = MakeColorAttribute("inColor", wideColor)
	if useScale {
		gp.attrs[2] = MakeAttribute("inEllipseOffset", gpu.VertexAttribTypeFloat3,
			GLSLTypeFloat3)
	} else {
		gp.attrs[2] = MakeAttribute("inEllipseOffset", gpu.VertexAttribTypeFloat2,
			GLSLTypeFloat2)
	}
	gp.attrs[3] = MakeAttribute("inEllipseRadii", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

func (g *ellipseGeometryProcessor) Name() string { return "EllipseGeometryProcessor" }

func (g *ellipseGeometryProcessor) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBool(g.stroke, "stroked")
	b.AddBits(matrixKeyBits, ComputeMatrixKey(caps, &g.localMatrix), "localMatrixType")
}

func (g *ellipseGeometryProcessor) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &ellipseGPImpl{}
}

// ellipseGPImpl is the program implementation for ellipseGeometryProcessor.
type ellipseGPImpl struct {
	GPImplBase
	localMatrixPrev    geom.Matrix
	localMatrixPrevSet bool
	localMatrixUniform UniformHandle
}

func (i *ellipseGPImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	egp := geomProc.(*ellipseGeometryProcessor)
	SetTransform(pdman, caps, i.localMatrixUniform, &egp.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
}

func (i *ellipseGPImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	egp := args.GeomProc.(*ellipseGeometryProcessor)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler

	// Emit attributes.
	varyingHandler.EmitAttributes(egp)

	offsetType := GLSLTypeFloat2
	if egp.useScale {
		offsetType = GLSLTypeFloat3
	}
	ellipseOffsets := NewVarying(offsetType)
	varyingHandler.AddVarying("EllipseOffsets", &ellipseOffsets)
	vertBuilder.CodeAppendf("%s = %s;", ellipseOffsets.VsOut(), egp.attrs[2].Name())

	ellipseRadii := NewVarying(GLSLTypeFloat4)
	varyingHandler.AddVarying("EllipseRadii", &ellipseRadii)
	vertBuilder.CodeAppendf("%s = %s;", ellipseRadii.VsOut(), egp.attrs[3].Name())

	fragBuilder := args.FragBuilder
	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(egp.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationCanBeFlat)

	// Set up position.
	WriteOutputPosition(vertBuilder, gpArgs, egp.attrs[0].Name())
	WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		egp.attrs[0].AsShaderVar(), &egp.localMatrix, &i.localMatrixUniform)

	// For stroked ellipses, we use the full ellipse equation (x^2/a^2 + y^2/b^2 = 1) to compute both the edges because
	// we need two separate test equations for the single offset. For filled ellipses we can use a unit circle equation
	// (x^2 + y^2 = 1), and warp the distance by the gradient, non-uniformly scaled by the inverse of the ellipse size.

	// On medium precision devices, we scale the denominator of the distance equation before taking the inverse square
	// root to minimize the chance that we're dividing by zero, then we scale the result back.

	// For the outer curve.
	fragBuilder.CodeAppendf("vec2 offset = %s.xy;", ellipseOffsets.FsIn())
	if egp.stroke {
		fragBuilder.CodeAppendf("offset *= %s.xy;", ellipseRadii.FsIn())
	}
	fragBuilder.CodeAppend("float test = dot(offset, offset) - 1.0;")
	if egp.useScale {
		fragBuilder.CodeAppendf("vec2 grad = 2.0*offset*(%s.z*%s.xy);",
			ellipseOffsets.FsIn(), ellipseRadii.FsIn())
	} else {
		fragBuilder.CodeAppendf("vec2 grad = 2.0*offset*%s.xy;", ellipseRadii.FsIn())
	}
	fragBuilder.CodeAppend("float grad_dot = dot(grad, grad);")

	// Avoid calling inversesqrt on zero.
	if args.ShaderCaps.FloatIs32Bits {
		fragBuilder.CodeAppend("grad_dot = max(grad_dot, 1.1755e-38);")
	} else {
		fragBuilder.CodeAppend("grad_dot = max(grad_dot, 6.1036e-5);")
	}
	if egp.useScale {
		fragBuilder.CodeAppendf("float invlen = %s.z*inversesqrt(grad_dot);",
			ellipseOffsets.FsIn())
	} else {
		fragBuilder.CodeAppend("float invlen = inversesqrt(grad_dot);")
	}
	fragBuilder.CodeAppend("float edgeAlpha = clamp(0.5-test*invlen, 0.0, 1.0);")

	// For the inner curve.
	if egp.stroke {
		fragBuilder.CodeAppendf("offset = %s.xy*%s.zw;", ellipseOffsets.FsIn(),
			ellipseRadii.FsIn())
		fragBuilder.CodeAppend("test = dot(offset, offset) - 1.0;")
		if egp.useScale {
			fragBuilder.CodeAppendf("grad = 2.0*offset*(%s.z*%s.zw);",
				ellipseOffsets.FsIn(), ellipseRadii.FsIn())
		} else {
			fragBuilder.CodeAppendf("grad = 2.0*offset*%s.zw;", ellipseRadii.FsIn())
		}
		fragBuilder.CodeAppend("grad_dot = dot(grad, grad);")
		if !args.ShaderCaps.FloatIs32Bits {
			fragBuilder.CodeAppend("grad_dot = max(grad_dot, 6.1036e-5);")
		}
		if egp.useScale {
			fragBuilder.CodeAppendf("invlen = %s.z*inversesqrt(grad_dot);",
				ellipseOffsets.FsIn())
		} else {
			fragBuilder.CodeAppend("invlen = inversesqrt(grad_dot);")
		}
		fragBuilder.CodeAppend("edgeAlpha *= clamp(0.5+test*invlen, 0.0, 1.0);")
	}

	fragBuilder.CodeAppendf("vec4 %s = vec4(edgeAlpha);", args.OutputCoverage)
}

//////////////////////////////////////////////////////////////////////////////
// DIEllipseGeometryProcessor

// diEllipseStyle selects the fill/stroke/hairline variant of the device-independent ellipse geometry processor.
type diEllipseStyle int32

const (
	diEllipseStyleStroke diEllipseStyle = iota
	diEllipseStyleHairline
	diEllipseStyleFill
)

// diEllipseGeometryProcessor draws a device-independent ellipse with the edge corrected by differentials (usable under
// any affine matrix).
type diEllipseGeometryProcessor struct {
	// attrs: inPosition, inColor, inEllipseOffsets0, inEllipseOffsets1.
	attrs [4]Attribute
	GPBase
	viewMatrix geom.Matrix
	style      diEllipseStyle
	useScale   bool
}

func makeDIEllipseGeometryProcessor(wideColor, useScale bool, viewMatrix *geom.Matrix, style diEllipseStyle) GeometryProcessor {
	// Borrow the shell from the free list (gppool.go); the struct-literal assignment zeroes the inline attrs array,
	// re-populated by the sets below.
	gp := diEllipseGPPool.borrow()
	*gp = diEllipseGeometryProcessor{viewMatrix: *viewMatrix, useScale: useScale, style: style}
	gp.initGP(DIEllipseGeometryProcessorClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.attrs[1] = MakeColorAttribute("inColor", wideColor)
	if useScale {
		gp.attrs[2] = MakeAttribute("inEllipseOffsets0", gpu.VertexAttribTypeFloat3,
			GLSLTypeFloat3)
	} else {
		gp.attrs[2] = MakeAttribute("inEllipseOffsets0", gpu.VertexAttribTypeFloat2,
			GLSLTypeFloat2)
	}
	gp.attrs[3] = MakeAttribute("inEllipseOffsets1", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

func (g *diEllipseGeometryProcessor) Name() string { return "DIEllipseGeometryProcessor" }

func (g *diEllipseGeometryProcessor) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBits(2, uint32(g.style), "style")
	b.AddBits(matrixKeyBits, ComputeMatrixKey(caps, &g.viewMatrix), "viewMatrixType")
}

func (g *diEllipseGeometryProcessor) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &diEllipseGPImpl{}
}

// diEllipseGPImpl is the program implementation for diEllipseGeometryProcessor.
type diEllipseGPImpl struct {
	GPImplBase
	viewMatrixPrev    geom.Matrix
	viewMatrixPrevSet bool
	viewMatrixUniform UniformHandle
}

func (i *diEllipseGPImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	diegp := geomProc.(*diEllipseGeometryProcessor)
	SetTransform(pdman, caps, i.viewMatrixUniform, &diegp.viewMatrix, &i.viewMatrixPrev,
		&i.viewMatrixPrevSet)
}

func (i *diEllipseGPImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	diegp := args.GeomProc.(*diEllipseGeometryProcessor)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler

	// Emit attributes.
	varyingHandler.EmitAttributes(diegp)

	offsetType := GLSLTypeFloat2
	if diegp.useScale {
		offsetType = GLSLTypeFloat3
	}
	offsets0 := NewVarying(offsetType)
	varyingHandler.AddVarying("EllipseOffsets0", &offsets0)
	vertBuilder.CodeAppendf("%s = %s;", offsets0.VsOut(), diegp.attrs[2].Name())

	offsets1 := NewVarying(GLSLTypeFloat2)
	varyingHandler.AddVarying("EllipseOffsets1", &offsets1)
	vertBuilder.CodeAppendf("%s = %s;", offsets1.VsOut(), diegp.attrs[3].Name())

	fragBuilder := args.FragBuilder
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(diegp.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationCanBeFlat)

	// Set up position.
	WriteOutputPositionWithMatrix(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		diegp.attrs[0].Name(), &diegp.viewMatrix, &i.viewMatrixUniform)
	gpArgs.LocalCoordVar = diegp.attrs[0].AsShaderVar()

	// For the outer curve.
	fragBuilder.CodeAppendf("vec2 scaledOffset = %s.xy;", offsets0.FsIn())
	fragBuilder.CodeAppend("float test = dot(scaledOffset, scaledOffset) - 1.0;")
	fragBuilder.CodeAppendf("vec2 duvdx = dFdx(%s.xy);", offsets0.FsIn())
	fragBuilder.CodeAppendf("vec2 duvdy = dFdy(%s.xy);", offsets0.FsIn())
	fragBuilder.CodeAppendf(
		"vec2 grad = vec2(%s.x*duvdx.x + %s.y*duvdx.y,"+
			"                     %s.x*duvdy.x + %s.y*duvdy.y);",
		offsets0.FsIn(), offsets0.FsIn(), offsets0.FsIn(), offsets0.FsIn(),
	)
	if diegp.useScale {
		fragBuilder.CodeAppendf("grad *= %s.z;", offsets0.FsIn())
	}

	fragBuilder.CodeAppend("float grad_dot = 4.0*dot(grad, grad);")
	// Avoid calling inversesqrt on zero.
	if args.ShaderCaps.FloatIs32Bits {
		fragBuilder.CodeAppend("grad_dot = max(grad_dot, 1.1755e-38);")
	} else {
		fragBuilder.CodeAppend("grad_dot = max(grad_dot, 6.1036e-5);")
	}
	fragBuilder.CodeAppend("float invlen = inversesqrt(grad_dot);")
	if diegp.useScale {
		fragBuilder.CodeAppendf("invlen *= %s.z;", offsets0.FsIn())
	}
	if diegp.style == diEllipseStyleHairline {
		// Can probably do this with one step.
		fragBuilder.CodeAppend("float edgeAlpha = clamp(1.0-test*invlen, 0.0, 1.0);")
		fragBuilder.CodeAppend("edgeAlpha *= clamp(1.0+test*invlen, 0.0, 1.0);")
	} else {
		fragBuilder.CodeAppend("float edgeAlpha = clamp(0.5-test*invlen, 0.0, 1.0);")
	}

	// For the inner curve.
	if diegp.style == diEllipseStyleStroke {
		fragBuilder.CodeAppendf("scaledOffset = %s.xy;", offsets1.FsIn())
		fragBuilder.CodeAppend("test = dot(scaledOffset, scaledOffset) - 1.0;")
		fragBuilder.CodeAppendf("duvdx = vec2(dFdx(%s));", offsets1.FsIn())
		fragBuilder.CodeAppendf("duvdy = vec2(dFdy(%s));", offsets1.FsIn())
		fragBuilder.CodeAppendf(
			"grad = vec2(%s.x*duvdx.x + %s.y*duvdx.y,"+
				"              %s.x*duvdy.x + %s.y*duvdy.y);",
			offsets1.FsIn(), offsets1.FsIn(), offsets1.FsIn(), offsets1.FsIn(),
		)
		if diegp.useScale {
			fragBuilder.CodeAppendf("grad *= %s.z;", offsets0.FsIn())
		}
		fragBuilder.CodeAppend("grad_dot = 4.0*dot(grad, grad);")
		if !args.ShaderCaps.FloatIs32Bits {
			fragBuilder.CodeAppend("grad_dot = max(grad_dot, 6.1036e-5);")
		}
		fragBuilder.CodeAppend("invlen = inversesqrt(grad_dot);")
		if diegp.useScale {
			fragBuilder.CodeAppendf("invlen *= %s.z;", offsets0.FsIn())
		}
		fragBuilder.CodeAppend("edgeAlpha *= clamp(0.5+test*invlen, 0.0, 1.0);")
	}

	fragBuilder.CodeAppendf("vec4 %s = vec4(edgeAlpha);", args.OutputCoverage)
}
