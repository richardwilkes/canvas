// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The conic and quad hairline geometry processors used by AAHairLinePathRenderer, emitting GLSL directly (half types
// widen to float — the desktop targets have no mediump). Both compute coverage as max(0, 1 - distance) using a
// first-order Taylor approximation of the distance to the implicit curve (Loop-Blinn): conics use K^2 - LM over the KLM
// functionals in the vertex attribute; quads use the canonical-space parabola u^2 - v. Note: makeConicEffect
// initializes its localMatrix field from the *view* matrix argument (not the local matrix parameter) — this quirk is
// preserved faithfully.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

//////////////////////////////////////////////////////////////////////////////
// Conic hairline geometry processor

// conicEffect is a hairline geometry processor for conics, computing coverage from the KLM implicit-form functionals.
type conicEffect struct {
	attrs [2]Attribute // inPosition, inConicCoeffs
	GPBase
	viewMatrix      geom.Matrix
	localMatrix     geom.Matrix
	color           colorcore.PMColor4f
	usesLocalCoords bool
	coverageScale   uint8
}

// makeConicEffect builds a conicEffect geometry processor, or nil without shader derivative support.
func makeConicEffect(color colorcore.PMColor4f, viewMatrix *geom.Matrix, caps *Caps, localMatrix *geom.Matrix, usesLocalCoords bool, coverage uint8) GeometryProcessor {
	if !caps.ShaderCaps.ShaderDerivativeSupport {
		return nil
	}
	gp := &conicEffect{
		color:      color,
		viewMatrix: *viewMatrix,
		// Initializes localMatrix with viewMatrix (not localMatrix); the quirk is preserved. localMatrix is accepted to
		// keep the call signature uniform.
		localMatrix:     *viewMatrix,
		usesLocalCoords: usesLocalCoords,
		coverageScale:   coverage,
	}
	_ = localMatrix
	gp.initGP(ConicEffectClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.attrs[1] = MakeAttribute("inConicCoeffs", gpu.VertexAttribTypeFloat4, GLSLTypeHalf4)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

func (g *conicEffect) Name() string { return "Conic" }

// AddToKey adds this geometry processor's shader-affecting state to the program key.
func (g *conicEffect) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	var key uint32
	if g.coverageScale == 0xff {
		key |= 0x8
	}
	if g.usesLocalCoords {
		key |= 0x10
	}
	localMatrix := geom.IdentityMatrix()
	if g.usesLocalCoords {
		localMatrix = g.localMatrix
	}
	key = AddMatrixKeys(caps, key, &g.viewMatrix, &localMatrix)
	b.Add32(key, "conicEffectKey")
}

func (g *conicEffect) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &conicEffectImpl{}
}

// conicEffectImpl is the shader implementation for conicEffect.
type conicEffectImpl struct {
	GPImplBase
	viewMatrixPrev       geom.Matrix
	localMatrixPrev      geom.Matrix
	color                colorcore.PMColor4f
	colorUniform         UniformHandle
	coverageScaleUniform UniformHandle
	viewMatrixUniform    UniformHandle
	localMatrixUniform   UniformHandle
	localMatrixPrevSet   bool
	colorSet             bool
	coverageScale        uint8
	coverageScaleSet     bool
	viewMatrixPrevSet    bool
}

func (i *conicEffectImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	ce := geomProc.(*conicEffect)

	SetTransform(pdman, caps, i.viewMatrixUniform, &ce.viewMatrix, &i.viewMatrixPrev,
		&i.viewMatrixPrevSet)
	SetTransform(pdman, caps, i.localMatrixUniform, &ce.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)

	if !i.colorSet || i.color != ce.color {
		pdman.Set4fv(i.colorUniform, 1, []float32{
			ce.color.R, ce.color.G, ce.color.B,
			ce.color.A,
		})
		i.color = ce.color
		i.colorSet = true
	}

	if ce.coverageScale != 0xff && (!i.coverageScaleSet || ce.coverageScale != i.coverageScale) {
		pdman.Set1f(i.coverageScaleUniform, float32(ce.coverageScale)*(1.0/255))
		i.coverageScale = ce.coverageScale
		i.coverageScaleSet = true
	}
}

func (i *conicEffectImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	gp := args.GeomProc.(*conicEffect)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler
	fragBuilder := args.FragBuilder

	// Emit attributes.
	varyingHandler.EmitAttributes(gp)

	v := NewVarying(GLSLTypeFloat4)
	varyingHandler.AddVarying("ConicCoeffs", &v)
	vertBuilder.CodeAppendf("%s = %s;", v.VsOut(), gp.attrs[1].Name())

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	i.setupUniformColor(fragBuilder, uniformHandler, args.OutputColor, &i.colorUniform)

	// Set up position.
	WriteOutputPositionWithMatrix(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		gp.attrs[0].Name(), &gp.viewMatrix, &i.viewMatrixUniform)
	if gp.usesLocalCoords {
		WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
			gp.attrs[0].AsShaderVar(), &gp.localMatrix, &i.localMatrixUniform)
	}

	fragBuilder.CodeAppend("float edgeAlpha;")
	fragBuilder.CodeAppend("vec3 dklmdx;")
	fragBuilder.CodeAppend("vec3 dklmdy;")
	fragBuilder.CodeAppend("float dfdx_;")
	fragBuilder.CodeAppend("float dfdy_;")
	fragBuilder.CodeAppend("vec2 gF;")
	fragBuilder.CodeAppend("float gFM;")
	fragBuilder.CodeAppend("float func_;")

	fragBuilder.CodeAppendf("dklmdx = dFdx(%s.xyz);", v.FsIn())
	fragBuilder.CodeAppendf("dklmdy = dFdy(%s.xyz);", v.FsIn())
	fragBuilder.CodeAppendf("dfdx_ = 2.0 * %s.x * dklmdx.x - %s.y * dklmdx.z - %s.z * dklmdx.y;",
		v.FsIn(), v.FsIn(), v.FsIn())
	fragBuilder.CodeAppendf("dfdy_ = 2.0 * %s.x * dklmdy.x - %s.y * dklmdy.z - %s.z * dklmdy.y;",
		v.FsIn(), v.FsIn(), v.FsIn())
	fragBuilder.CodeAppend("gF = vec2(dfdx_, dfdy_);")
	fragBuilder.CodeAppend("gFM = sqrt(dot(gF, gF));")
	fragBuilder.CodeAppendf("func_ = %s.x*%s.x - %s.y*%s.z;", v.FsIn(), v.FsIn(), v.FsIn(),
		v.FsIn())
	fragBuilder.CodeAppend("func_ = abs(func_);")
	fragBuilder.CodeAppend("edgeAlpha = func_ / gFM;")
	fragBuilder.CodeAppend("edgeAlpha = max(1.0 - edgeAlpha, 0.0);")
	// Add line below for smooth cubic ramp: edgeAlpha = edgeAlpha*edgeAlpha*(3.0-2.0*edgeAlpha);

	if gp.coverageScale != 0xff {
		var coverageScale string
		i.coverageScaleUniform, coverageScale = uniformHandler.AddUniform(nil,
			ShaderFlagFragment, GLSLTypeFloat, "Coverage")
		fragBuilder.CodeAppendf("vec4 %s = vec4(%s * edgeAlpha);", args.OutputCoverage,
			coverageScale)
	} else {
		fragBuilder.CodeAppendf("vec4 %s = vec4(edgeAlpha);", args.OutputCoverage)
	}
}

//////////////////////////////////////////////////////////////////////////////
// Quad hairline geometry processor

// quadEffect is a hairline geometry processor for quadratics specified by the 0 = u^2 - v canonical coords in the first
// two components of the vertex attribute (the control points have (u, v) of (0,0), (1/2, 0), and (1, 1)).
type quadEffect struct {
	attrs [2]Attribute // inPosition, inHairQuadEdge
	GPBase
	viewMatrix      geom.Matrix
	localMatrix     geom.Matrix
	color           colorcore.PMColor4f
	usesLocalCoords bool
	coverageScale   uint8
}

// makeQuadEffect builds a quadEffect geometry processor, or nil without shader derivative support.
func makeQuadEffect(color colorcore.PMColor4f, viewMatrix *geom.Matrix, caps *Caps, localMatrix *geom.Matrix, usesLocalCoords bool, coverage uint8) GeometryProcessor {
	if !caps.ShaderCaps.ShaderDerivativeSupport {
		return nil
	}
	gp := &quadEffect{
		color:           color,
		viewMatrix:      *viewMatrix,
		localMatrix:     *localMatrix,
		usesLocalCoords: usesLocalCoords,
		coverageScale:   coverage,
	}
	gp.initGP(QuadEffectClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	gp.attrs[1] = MakeAttribute("inHairQuadEdge", gpu.VertexAttribTypeFloat4, GLSLTypeHalf4)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

func (g *quadEffect) Name() string { return "Quad" }

// AddToKey adds this geometry processor's shader-affecting state to the program key (note the coverage-scale key bit
// tests != 0xff here, where conicEffect tests == 0xff — this asymmetry with conicEffect's key is intentional, matching
// the original design).
func (g *quadEffect) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	var key uint32
	if g.coverageScale != 0xff {
		key |= 0x8
	}
	if g.usesLocalCoords {
		key |= 0x10
	}
	localMatrix := geom.IdentityMatrix()
	if g.usesLocalCoords {
		localMatrix = g.localMatrix
	}
	key = AddMatrixKeys(caps, key, &g.viewMatrix, &localMatrix)
	b.Add32(key, "quadEffectKey")
}

func (g *quadEffect) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &quadEffectImpl{}
}

// quadEffectImpl is the shader implementation for quadEffect.
type quadEffectImpl struct {
	GPImplBase
	viewMatrixPrev       geom.Matrix
	localMatrixPrev      geom.Matrix
	color                colorcore.PMColor4f
	colorUniform         UniformHandle
	coverageScaleUniform UniformHandle
	viewMatrixUniform    UniformHandle
	localMatrixUniform   UniformHandle
	localMatrixPrevSet   bool
	colorSet             bool
	coverageScale        uint8
	coverageScaleSet     bool
	viewMatrixPrevSet    bool
}

func (i *quadEffectImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	qe := geomProc.(*quadEffect)

	SetTransform(pdman, caps, i.viewMatrixUniform, &qe.viewMatrix, &i.viewMatrixPrev,
		&i.viewMatrixPrevSet)
	SetTransform(pdman, caps, i.localMatrixUniform, &qe.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)

	if !i.colorSet || i.color != qe.color {
		pdman.Set4fv(i.colorUniform, 1, []float32{
			qe.color.R, qe.color.G, qe.color.B,
			qe.color.A,
		})
		i.color = qe.color
		i.colorSet = true
	}

	if qe.coverageScale != 0xff && (!i.coverageScaleSet || qe.coverageScale != i.coverageScale) {
		pdman.Set1f(i.coverageScaleUniform, float32(qe.coverageScale)*(1.0/255))
		i.coverageScale = qe.coverageScale
		i.coverageScaleSet = true
	}
}

func (i *quadEffectImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	gp := args.GeomProc.(*quadEffect)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler
	fragBuilder := args.FragBuilder

	// Emit attributes.
	varyingHandler.EmitAttributes(gp)

	v := NewVarying(GLSLTypeHalf4)
	varyingHandler.AddVarying("HairQuadEdge", &v)
	vertBuilder.CodeAppendf("%s = %s;", v.VsOut(), gp.attrs[1].Name())

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	i.setupUniformColor(fragBuilder, uniformHandler, args.OutputColor, &i.colorUniform)

	// Set up position.
	WriteOutputPositionWithMatrix(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		gp.attrs[0].Name(), &gp.viewMatrix, &i.viewMatrixUniform)
	if gp.usesLocalCoords {
		WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
			gp.attrs[0].AsShaderVar(), &gp.localMatrix, &i.localMatrixUniform)
	}

	fragBuilder.CodeAppend("float edgeAlpha;")

	fragBuilder.CodeAppendf("vec2 duvdx = dFdx(%s.xy);", v.FsIn())
	fragBuilder.CodeAppendf("vec2 duvdy = dFdy(%s.xy);", v.FsIn())
	fragBuilder.CodeAppendf("vec2 gF = vec2(2.0 * %s.x * duvdx.x - duvdx.y,"+
		"               2.0 * %s.x * duvdy.x - duvdy.y);", v.FsIn(), v.FsIn())
	fragBuilder.CodeAppendf("edgeAlpha = %s.x * %s.x - %s.y;", v.FsIn(), v.FsIn(), v.FsIn())
	fragBuilder.CodeAppend("edgeAlpha = sqrt(edgeAlpha * edgeAlpha / dot(gF, gF));")
	fragBuilder.CodeAppend("edgeAlpha = max(1.0 - edgeAlpha, 0.0);")
	// Add line below for smooth cubic ramp: edgeAlpha = edgeAlpha*edgeAlpha*(3.0-2.0*edgeAlpha);

	if gp.coverageScale != 0xff {
		var coverageScale string
		i.coverageScaleUniform, coverageScale = uniformHandler.AddUniform(nil,
			ShaderFlagFragment, GLSLTypeHalf, "Coverage")
		fragBuilder.CodeAppendf("vec4 %s = vec4(%s * edgeAlpha);", args.OutputCoverage,
			coverageScale)
	} else {
		fragBuilder.CodeAppendf("vec4 %s = vec4(edgeAlpha);", args.OutputCoverage)
	}
}
