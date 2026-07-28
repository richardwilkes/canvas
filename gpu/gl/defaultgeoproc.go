// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The default geometry processor — position times a uniform view matrix, with optional per-vertex color, local coords,
// and coverage.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// DefaultGeoProc GP flags.
const (
	gpFlagColorAttribute uint32 = 1 << iota
	gpFlagColorAttributeIsWide
	gpFlagLocalCoordAttribute
	gpFlagCoverageAttribute
	gpFlagCoverageAttributeTweak
	gpFlagCoverageAttributeUnclamped
)

// DefaultGeoProcColorType selects how color reaches the default geometry processor.
type DefaultGeoProcColorType int32

// DefaultGeoProcColorType values.
const (
	ColorTypePremulUniform DefaultGeoProcColorType = iota
	ColorTypePremulAttribute
	ColorTypePremulWideAttribute
)

// DefaultGeoProcCoverageType selects how coverage reaches the default geometry processor.
type DefaultGeoProcCoverageType int32

// DefaultGeoProcCoverageType values.
const (
	CoverageTypeSolid DefaultGeoProcCoverageType = iota
	CoverageTypeUniform
	CoverageTypeAttribute
	CoverageTypeAttributeTweakAlpha
	CoverageTypeAttributeUnclamped
)

// DefaultGeoProcLocalCoordsType selects how local coords reach the default geometry processor.
type DefaultGeoProcLocalCoordsType int32

// DefaultGeoProcLocalCoordsType values.
const (
	LocalCoordsTypeUnused DefaultGeoProcLocalCoordsType = iota
	LocalCoordsTypeUsePosition
	LocalCoordsTypeHasExplicit
)

// defaultGeoProc is the default geometry processor: position times a uniform view matrix, with optional per-vertex
// color, local coords, and coverage.
type defaultGeoProc struct {
	attrs [4]Attribute // inPosition, inColor, inLocalCoord, inCoverage
	GPBase
	viewMatrix            geom.Matrix
	localMatrix           geom.Matrix
	color                 colorcore.PMColor4f
	flags                 uint32
	coverage              uint8
	localCoordsWillBeRead bool
}

// MakeDefaultGeoProc builds a defaultGeoProc (with the Color/Coverage/LocalCoords options flattened into arguments; the
// arena parameter is dropped, since ops are ordinary allocations rather than arena-allocated).
func MakeDefaultGeoProc(colorType DefaultGeoProcColorType, color colorcore.PMColor4f, coverageType DefaultGeoProcCoverageType, coverage uint8, localCoordsType DefaultGeoProcLocalCoordsType, localMatrix, viewMatrix *geom.Matrix) GeometryProcessor {
	var flags uint32
	switch colorType {
	case ColorTypePremulAttribute:
		flags |= gpFlagColorAttribute
	case ColorTypePremulWideAttribute:
		flags |= gpFlagColorAttribute | gpFlagColorAttributeIsWide
	case ColorTypePremulUniform:
	}
	switch coverageType {
	case CoverageTypeAttribute:
		flags |= gpFlagCoverageAttribute
	case CoverageTypeAttributeTweakAlpha:
		flags |= gpFlagCoverageAttribute | gpFlagCoverageAttributeTweak
	case CoverageTypeAttributeUnclamped:
		flags |= gpFlagCoverageAttribute | gpFlagCoverageAttributeUnclamped
	case CoverageTypeSolid, CoverageTypeUniform:
	}
	if localCoordsType == LocalCoordsTypeHasExplicit {
		flags |= gpFlagLocalCoordAttribute
	}
	localCoordsWillBeRead := localCoordsType != LocalCoordsTypeUnused

	lm := geom.IdentityMatrix()
	if localMatrix != nil {
		lm = *localMatrix
	}
	// Borrow the shell from the free list (gppool.go) and overwrite it — the struct-literal assignment zeroes the
	// inline attrs array, which the conditional sets below then re-populate, so a reused shell is byte-identical to a
	// fresh &defaultGeoProc{}.
	gp := defaultGeoProcPool.borrow()
	*gp = defaultGeoProc{
		color:                 color,
		viewMatrix:            *viewMatrix,
		localMatrix:           lm,
		coverage:              coverage,
		flags:                 flags,
		localCoordsWillBeRead: localCoordsWillBeRead,
	}
	gp.initGP(DefaultGeoProcClassID)
	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	if flags&gpFlagColorAttribute != 0 {
		gp.attrs[1] = MakeColorAttribute("inColor", flags&gpFlagColorAttributeIsWide != 0)
	}
	if flags&gpFlagLocalCoordAttribute != 0 {
		gp.attrs[2] = MakeAttribute("inLocalCoord", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	}
	if flags&gpFlagCoverageAttribute != 0 {
		gp.attrs[3] = MakeAttribute("inCoverage", gpu.VertexAttribTypeFloat, GLSLTypeHalf)
	}
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])
	return gp
}

// MakeDefaultGeoProcForDeviceSpace builds a defaultGeoProc whose local coords map device-space positions back to local
// space via the inverse view matrix.
func MakeDefaultGeoProcForDeviceSpace(colorType DefaultGeoProcColorType, color colorcore.PMColor4f, coverageType DefaultGeoProcCoverageType, coverage uint8, localCoordsType DefaultGeoProcLocalCoordsType, localMatrix, viewMatrix *geom.Matrix) GeometryProcessor {
	inverted := geom.IdentityMatrix()
	if localCoordsType != LocalCoordsTypeUnused {
		if localCoordsType != LocalCoordsTypeUsePosition {
			panic("device-space GP requires use-position local coords")
		}
		inverse, ok := viewMatrix.Invert()
		if !ok {
			return nil
		}
		inverted = inverse
		if localMatrix != nil {
			inverted.PostConcat(localMatrix)
		}
	}
	identity := geom.IdentityMatrix()
	return MakeDefaultGeoProc(colorType, color, coverageType, coverage,
		LocalCoordsTypeUsePosition, &inverted, &identity)
}

func (g *defaultGeoProc) Name() string { return "DefaultGeometryProcessor" }

func (g *defaultGeoProc) hasVertexColor() bool { return g.attrs[1].IsInitialized() }

func (g *defaultGeoProc) hasVertexCoverage() bool { return g.attrs[3].IsInitialized() }

func (g *defaultGeoProc) tweaksAlphaForCoverage() bool {
	return g.flags&gpFlagCoverageAttributeTweak != 0
}

// usesCoverageUniform reports whether the fragment shader declares and reads the Coverage uniform, which is the emitter's
// fallback case: any coverage other than fully opaque that is not already being passed through from the vertex coverage
// attribute. Tweak-alpha folds the vertex coverage into the color instead of into the coverage output, so it takes the
// uniform lane too — SetData must agree with the emitter here or the declared uniform is never written, leaving the draw
// with zero coverage.
func (g *defaultGeoProc) usesCoverageUniform() bool {
	return g.coverage != 0xff && (!g.hasVertexCoverage() || g.tweaksAlphaForCoverage())
}

func (g *defaultGeoProc) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	key := g.flags
	if g.coverage == 0xff {
		key |= 0x80
	}
	if g.localCoordsWillBeRead {
		key |= 0x100
	}

	usesLocalMatrix := g.localCoordsWillBeRead && !g.attrs[2].IsInitialized()
	identity := geom.IdentityMatrix()
	localForKey := &identity
	if usesLocalMatrix {
		localForKey = &g.localMatrix
	}
	key = AddMatrixKeys(caps, key, &g.viewMatrix, localForKey)
	b.Add32(key, "defaultGeoProcKey")
}

func (g *defaultGeoProc) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &defaultGeoProcImpl{coverage: 0xff}
}

// defaultGeoProcImpl is the shader implementation for defaultGeoProc.
type defaultGeoProcImpl struct {
	GPImplBase
	viewMatrixPrev     geom.Matrix
	localMatrixPrev    geom.Matrix
	color              colorcore.PMColor4f
	coverage           uint8
	viewMatrixPrevSet  bool
	localMatrixPrevSet bool
	colorSet           bool
	viewMatrixUniform  UniformHandle
	localMatrixUniform UniformHandle
	colorUniform       UniformHandle
	coverageUniform    UniformHandle
}

func (i *defaultGeoProcImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	dgp := geomProc.(*defaultGeoProc)

	SetTransform(pdman, caps, i.viewMatrixUniform, &dgp.viewMatrix, &i.viewMatrixPrev,
		&i.viewMatrixPrevSet)
	SetTransform(pdman, caps, i.localMatrixUniform, &dgp.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)

	if !dgp.hasVertexColor() && (!i.colorSet || dgp.color != i.color) {
		pdman.Set4f(i.colorUniform, dgp.color.R, dgp.color.G, dgp.color.B, dgp.color.A)
		i.color = dgp.color
		i.colorSet = true
	}

	if dgp.coverage != i.coverage && dgp.usesCoverageUniform() {
		pdman.Set1f(i.coverageUniform, float32(dgp.coverage)/255)
		i.coverage = dgp.coverage
	}
}

func (i *defaultGeoProcImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	gp := args.GeomProc.(*defaultGeoProc)
	vertBuilder := args.VertBuilder
	fragBuilder := args.FragBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler

	// Emit attributes.
	varyingHandler.EmitAttributes(gp)

	tweakAlpha := gp.tweaksAlphaForCoverage()
	coverageNeedsSaturate := gp.flags&gpFlagCoverageAttributeUnclamped != 0
	if tweakAlpha && !gp.hasVertexCoverage() {
		panic("tweak-alpha requires vertex coverage")
	}
	if tweakAlpha && coverageNeedsSaturate {
		panic("tweak-alpha and unclamped coverage are exclusive")
	}

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	if gp.hasVertexColor() || tweakAlpha {
		varying := NewVarying(GLSLTypeHalf4)
		varyingHandler.AddVarying("color", &varying)

		// Start with the attribute or with uniform color.
		if gp.hasVertexColor() {
			vertBuilder.CodeAppendf("vec4 color = %s;", gp.attrs[1].Name())
		} else {
			var colorUniformName string
			i.colorUniform, colorUniformName = uniformHandler.AddUniform(nil,
				ShaderFlagVertex, GLSLTypeHalf4, "Color")
			vertBuilder.CodeAppendf("vec4 color = %s;", colorUniformName)
		}

		// Optionally fold coverage into alpha (color).
		if tweakAlpha {
			vertBuilder.CodeAppendf("color = color * %s;", gp.attrs[3].Name())
		}
		vertBuilder.CodeAppendf("%s = color;\n", varying.VsOut())
		fragBuilder.CodeAppendf("%s = %s;", args.OutputColor, varying.FsIn())
	} else {
		i.setupUniformColor(fragBuilder, uniformHandler, args.OutputColor, &i.colorUniform)
	}

	// Set up position.
	WriteOutputPositionWithMatrix(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		gp.attrs[0].Name(), &gp.viewMatrix, &i.viewMatrixUniform)

	// Emit transforms using either explicit local coords or positions.
	if gp.attrs[2].IsInitialized() {
		if !gp.localMatrix.IsIdentity() {
			panic("explicit local coords require an identity local matrix")
		}
		gpArgs.LocalCoordVar = gp.attrs[2].AsShaderVar()
	} else if gp.localCoordsWillBeRead {
		WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
			gp.attrs[0].AsShaderVar(), &gp.localMatrix, &i.localMatrixUniform)
	}

	// Set up coverage as pass through.
	switch {
	case gp.hasVertexCoverage() && !tweakAlpha:
		fragBuilder.CodeAppendf("float alpha = 1.0;")
		varyingHandler.AddPassThroughAttribute(gp.attrs[3].AsShaderVar(), "alpha",
			InterpolationInterpolated)
		if coverageNeedsSaturate {
			fragBuilder.CodeAppendf("vec4 %s = vec4(clamp(alpha, 0.0, 1.0));",
				args.OutputCoverage)
		} else {
			fragBuilder.CodeAppendf("vec4 %s = vec4(alpha);", args.OutputCoverage)
		}
	case !gp.usesCoverageUniform():
		fragBuilder.CodeAppendf("const vec4 %s = vec4(1);", args.OutputCoverage)
	default:
		// The uniform lane SetData uploads (see usesCoverageUniform).
		var fragCoverage string
		i.coverageUniform, fragCoverage = uniformHandler.AddUniform(nil, ShaderFlagFragment,
			GLSLTypeHalf, "Coverage")
		fragBuilder.CodeAppendf("vec4 %s = vec4(%s);", args.OutputCoverage, fragCoverage)
	}
}
