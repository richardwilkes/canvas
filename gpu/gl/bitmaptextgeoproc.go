// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The geometry processor for atlas text — device-space positions (2D, or 3D under perspective), packed atlas texel
// coordinates carrying the 2-bit page index in bits 13/14 of x, and a per-vertex color for the coverage formats. The
// GLSL is emitted directly. Trims: the color-space xform is identity (color emoji are sRGB, as is the destination) and
// wide color is never requested by AtlasTextOp (which also hard-codes wideColor=false there).

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// bitmapTextMaxTextures is the maximum number of atlas texture samplers a bitmapTextGeoProc binds.
const bitmapTextMaxTextures = 4

// bitmapTextGeoProc is the geometry processor for atlas-backed bitmap (non-SDF) text.
type bitmapTextGeoProc struct {
	attrs [3]Attribute // inPosition, inColor, inTextureCoords
	GPBase
	localMatrix     geom.Matrix
	color           colorcore.PMColor4f
	atlasDimensions geom.ISize
	maskFormat      gpu.MaskFormat
	usesW           bool
}

// newBitmapTextGeoProc builds a bitmapTextGeoProc (the color-space xform is dropped, since it is always the identity;
// wideColor is always false at the only call site).
func newBitmapTextGeoProc(caps *gpu.ShaderCaps, color colorcore.PMColor4f, views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState, maskFormat gpu.MaskFormat, localMatrix *geom.Matrix, usesW bool) *bitmapTextGeoProc {
	if numActiveViews > bitmapTextMaxTextures {
		panic("too many atlas views")
	}
	gp := &bitmapTextGeoProc{
		color:       color,
		localMatrix: *localMatrix,
		maskFormat:  maskFormat,
		usesW:       usesW,
	}
	gp.initGP(BitmapTextGeoProcClassID)

	if usesW {
		gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat3, GLSLTypeFloat3)
	} else {
		gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	}
	if maskFormat == gpu.MaskFormatA8 || maskFormat == gpu.MaskFormatA565 {
		gp.attrs[1] = MakeColorAttribute("inColor", false)
	}
	texCoordsType := GLSLTypeFloat2
	if caps.IntegerSupport {
		texCoordsType = GLSLTypeUShort2
	}
	gp.attrs[2] = MakeAttribute("inTextureCoords", gpu.VertexAttribTypeUShort2, texCoordsType)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])

	samplers := make([]TextureSampler, 0, numActiveViews)
	if numActiveViews > 0 {
		gp.atlasDimensions = views[0].Proxy().Dimensions()
	}
	for i := 0; i < numActiveViews; i++ {
		proxy := views[i].Proxy()
		if proxy.Dimensions() != gp.atlasDimensions {
			panic("atlas pages must share dimensions")
		}
		samplers = append(samplers,
			MakeGPTextureSampler(params, proxy.TextureType(), views[i].Swizzle()))
	}
	gp.setTextureSamplers(samplers)
	return gp
}

// Name implements Processor.
func (g *bitmapTextGeoProc) Name() string { return "BitmapText" }

func (g *bitmapTextGeoProc) hasVertexColor() bool { return g.attrs[1].IsInitialized() }

// AddNewViews updates the sampler set to match the atlas's current page count.
func (g *bitmapTextGeoProc) AddNewViews(views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState) {
	// Just to make sure we don't try to add too many proxies.
	numActiveViews = min(numActiveViews, bitmapTextMaxTextures)
	if len(g.textureSamplers) == 0 {
		g.atlasDimensions = views[0].Proxy().Dimensions()
	}
	samplers := make([]TextureSampler, numActiveViews)
	copy(samplers, g.textureSamplers)
	for i := len(g.textureSamplers); i < numActiveViews; i++ {
		proxy := views[i].Proxy()
		if proxy.Dimensions() != g.atlasDimensions {
			panic("atlas pages must share dimensions")
		}
		samplers[i] = MakeGPTextureSampler(params, proxy.TextureType(), views[i].Swizzle())
	}
	g.setTextureSamplers(samplers)
}

// AddToKey implements GeometryProcessor (the color-space-xform key is constant, since the xform is always the
// identity).
func (g *bitmapTextGeoProc) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBool(g.usesW, "usesW")
	b.AddBits(2, uint32(g.maskFormat), "maskFormat")
	b.AddBits(matrixKeyBits, ComputeMatrixKey(caps, &g.localMatrix), "localMatrixType")
	b.Add32(uint32(g.NumTextureSamplers()), "numTextures")
}

// MakeProgramImpl implements GeometryProcessor.
func (g *bitmapTextGeoProc) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &bitmapTextGeoProcImpl{}
}

// bitmapTextGeoProcImpl is the shader implementation for bitmapTextGeoProc.
type bitmapTextGeoProcImpl struct {
	GPImplBase
	localMatrixPrev        geom.Matrix
	color                  colorcore.PMColor4f
	atlasDimensions        geom.ISize
	colorUniform           UniformHandle
	atlasDimensionsInvUnif UniformHandle
	localMatrixUniform     UniformHandle
	colorSet               bool
	atlasDimensionsSet     bool
	localMatrixPrevSet     bool
}

// SetData implements GPProgramImpl.
func (i *bitmapTextGeoProcImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	btgp := geomProc.(*bitmapTextGeoProc)
	if (!i.colorSet || btgp.color != i.color) && !btgp.hasVertexColor() {
		pdman.Set4f(i.colorUniform, btgp.color.R, btgp.color.G, btgp.color.B, btgp.color.A)
		i.color = btgp.color
		i.colorSet = true
	}
	if !i.atlasDimensionsSet || i.atlasDimensions != btgp.atlasDimensions {
		pdman.Set2f(i.atlasDimensionsInvUnif,
			1/float32(btgp.atlasDimensions.Width), 1/float32(btgp.atlasDimensions.Height))
		i.atlasDimensions = btgp.atlasDimensions
		i.atlasDimensionsSet = true
	}
	SetTransform(pdman, caps, i.localMatrixUniform, &btgp.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
}

// appendIndexUVVaryings extracts the texture page index and texel coordinates from the packed inTextureCoords attribute
// (the 2-bit page lives in bits 13/14 of x) and hands them to the fragment shader as varyings. st (may be nil) is the
// unnormalized-coords varying only the SDFT processors consume (E.1).
func appendIndexUVVaryings(args *GPEmitArgs, numTextureSamplers int, inTexCoordsName, atlasDimensionsInvName string, uv, texIdx, st *Varying) {
	vb := args.VertBuilder
	if args.ShaderCaps.IntegerSupport {
		if numTextureSamplers <= 1 {
			vb.CodeAppendf("int texIdx = 0;"+
				"vec2 unormTexCoords = vec2(%s.x, %s.y);", inTexCoordsName, inTexCoordsName)
		} else {
			vb.CodeAppendf("ivec2 coords = ivec2(%s.x, %s.y);"+
				"int texIdx = coords.x >> 13;"+
				"vec2 unormTexCoords = vec2(coords.x & 0x1FFF, coords.y);",
				inTexCoordsName, inTexCoordsName)
		}
	} else {
		if numTextureSamplers <= 1 {
			vb.CodeAppendf("float texIdx = 0;"+
				"vec2 unormTexCoords = vec2(%s.x, %s.y);", inTexCoordsName, inTexCoordsName)
		} else {
			vb.CodeAppendf("vec2 coord = vec2(%s.x, %s.y);"+
				"float texIdx = floor(coord.x * exp2(-13.0));"+
				"vec2 unormTexCoords = vec2(coord.x - texIdx * exp2(13.0), coord.y);",
				inTexCoordsName, inTexCoordsName)
		}
	}

	// Multiply by 1/atlasDimensions to get normalized texture coordinates.
	*uv = NewVarying(GLSLTypeFloat2)
	args.VaryingHandler.AddVarying("TextureCoords", uv)
	vb.CodeAppendf("%s = unormTexCoords * %s;", uv.VsOut(), atlasDimensionsInvName)

	// A float varying even where int is available: int varyings are slow on some GL stacks and never worse as float.
	*texIdx = NewVarying(GLSLTypeFloat)
	cast := ""
	if args.ShaderCaps.IntegerSupport {
		cast = "float"
	}
	args.VaryingHandler.AddVaryingWithInterpolation("TexIndex", texIdx, InterpolationCanBeFlat)
	vb.CodeAppendf("%s = %s(texIdx);", texIdx.VsOut(), cast)

	if st != nil {
		*st = NewVarying(GLSLTypeFloat2)
		args.VaryingHandler.AddVarying("IntTextureCoords", st)
		vb.CodeAppendf("%s = unormTexCoords;", st.VsOut())
	}
}

// appendMultitextureLookup emits a conditional load from the indexed texture sampler.
func appendMultitextureLookup(args *GPEmitArgs, numTextureSamplers int, texIdx *Varying, coordName, colorName string) {
	fb := args.FragBuilder
	if numTextureSamplers <= 0 {
		// This shouldn't happen, but will avoid a crash if it does.
		fb.CodeAppendf("%s = vec4(1);", colorName)
		return
	}
	for i := 0; i < numTextureSamplers-1; i++ {
		fb.CodeAppendf("if (%s == %d.0) { %s = %s; } else ", texIdx.FsIn(), i, colorName,
			fb.AppendTextureLookup(args.UniformHandler, args.TexSamplers[i], coordName))
	}
	fb.CodeAppendf("{ %s = %s; }", colorName,
		fb.AppendTextureLookup(args.UniformHandler, args.TexSamplers[numTextureSamplers-1],
			coordName))
}

// onEmitCode implements GPProgramImpl.
func (i *bitmapTextGeoProcImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	btgp := args.GeomProc.(*bitmapTextGeoProc)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler
	fragBuilder := args.FragBuilder

	// Emit attributes.
	varyingHandler.EmitAttributes(btgp)

	var atlasDimensionsInvName string
	i.atlasDimensionsInvUnif, atlasDimensionsInvName = uniformHandler.AddUniform(nil,
		ShaderFlagVertex, GLSLTypeFloat2, "AtlasSizeInv")

	var uv, texIdx Varying
	appendIndexUVVaryings(args, btgp.NumTextureSamplers(), btgp.attrs[2].Name(),
		atlasDimensionsInvName, &uv, &texIdx, nil)

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	if btgp.hasVertexColor() {
		varyingHandler.AddPassThroughAttribute(btgp.attrs[1].AsShaderVar(), args.OutputColor,
			InterpolationInterpolated)
	} else {
		i.setupUniformColor(fragBuilder, uniformHandler, args.OutputColor, &i.colorUniform)
	}

	// Set up position.
	gpArgs.PositionVar = btgp.attrs[0].AsShaderVar()
	WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		btgp.attrs[0].AsShaderVar(), &btgp.localMatrix, &i.localMatrixUniform)

	fragBuilder.CodeAppend("vec4 texColor;")
	appendMultitextureLookup(args, btgp.NumTextureSamplers(), &texIdx, uv.FsIn(), "texColor")

	if btgp.maskFormat == gpu.MaskFormatARGB {
		// Modulate by color.
		fragBuilder.CodeAppendf("%s = %s * texColor;", args.OutputColor, args.OutputColor)
		fragBuilder.CodeAppendf("vec4 %s = vec4(1);", args.OutputCoverage)
	} else {
		fragBuilder.CodeAppendf("vec4 %s = texColor;", args.OutputCoverage)
	}
}

// Compile-time check.
var _ GeometryProcessor = (*bitmapTextGeoProc)(nil)

// bitmapTextFilterFromNeedsTransform maps the op's needsGlyphTransform flag to the atlas sampler filter
// AtlasTextOp.OnPrepare chooses.
func bitmapTextFilterFromNeedsTransform(needsTransform bool) gpu.SamplerState {
	filter := gpu.FilterModeNearest
	if needsTransform {
		filter = gpu.FilterModeLinear
	}
	return gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, filter, gpu.MipmapModeNone)
}
