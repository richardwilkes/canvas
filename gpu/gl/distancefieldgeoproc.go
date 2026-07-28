// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The three distance-field geometry processors — A8 text, path, and LCD text — plus the LCD atlas lookup helper,
// emitted as direct GLSL (half types lower to float, saturate to clamp). Trims: the gamma-apply-to-A8 lane is
// unimplemented, so the A8 GP's distance-adjust uniform lane is compiled out; the gamma-correct flag branch is kept in
// the emitters for fidelity but never set (this codebase's destinations are never linearly blended). The path GP's
// small-path-renderer consumer remains unimplemented, so it is exercised by the fake-driver GLSL tests until that
// renderer lands.

package gl

import (
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// distanceFieldMaxTextures is the maximum atlas texture samplers shared by all three DF GPs (kept equal to
// bitmapTextMaxTextures for AtlasTextOp).
const distanceFieldMaxTextures = bitmapTextMaxTextures

// Distance-field effect flags.
const (
	similarityDistanceFieldEffectFlag   = 0x001 // ctm is similarity matrix
	scaleOnlyDistanceFieldEffectFlag    = 0x002 // ctm has only scale and translate
	perspectiveDistanceFieldEffectFlag  = 0x004 // ctm has perspective (and positions are x,y,w)
	useLCDDistanceFieldEffectFlag       = 0x008 // use lcd text
	bgrDistanceFieldEffectFlag          = 0x010 // lcd display has bgr order
	portraitDistanceFieldEffectFlag     = 0x020 // lcd display is in portrait mode
	gammaCorrectDistanceFieldEffectFlag = 0x040 // assume gamma-correct output (linear blending)
	aliasedDistanceFieldEffectFlag      = 0x080 // monochrome output
	wideColorDistanceFieldEffectFlag    = 0x100 // use wide color (only for path)

	uniformScaleDistanceFieldEffectMask = similarityDistanceFieldEffectFlag |
		scaleOnlyDistanceFieldEffectFlag
	// The subset of the flags relevant to distanceFieldA8TextGeoProc.
	nonLCDDistanceFieldEffectMask = similarityDistanceFieldEffectFlag |
		scaleOnlyDistanceFieldEffectFlag |
		perspectiveDistanceFieldEffectFlag |
		gammaCorrectDistanceFieldEffectFlag |
		aliasedDistanceFieldEffectFlag
	// The subset of the flags relevant to distanceFieldPathGeoProc.
	pathDistanceFieldEffectMask = similarityDistanceFieldEffectFlag |
		scaleOnlyDistanceFieldEffectFlag |
		perspectiveDistanceFieldEffectFlag |
		gammaCorrectDistanceFieldEffectFlag |
		aliasedDistanceFieldEffectFlag |
		wideColorDistanceFieldEffectFlag
	// The subset of the flags relevant to distanceFieldLCDTextGeoProc.
	lcdDistanceFieldEffectMask = similarityDistanceFieldEffectFlag |
		scaleOnlyDistanceFieldEffectFlag |
		perspectiveDistanceFieldEffectFlag |
		useLCDDistanceFieldEffectFlag |
		bgrDistanceFieldEffectFlag |
		portraitDistanceFieldEffectFlag |
		gammaCorrectDistanceFieldEffectFlag
)

// distanceFieldAAFactorStr is the AA factor: a radius of a little less than the diagonal of the fragment.
const distanceFieldAAFactorStr = "0.65"

// dfTexCoordsAttribute builds the shared inTextureCoords attribute (packed atlas texel coords).
func dfTexCoordsAttribute(caps *gpu.ShaderCaps) Attribute {
	texCoordsType := GLSLTypeFloat2
	if caps.IntegerSupport {
		texCoordsType = GLSLTypeUShort2
	}
	return MakeAttribute("inTextureCoords", gpu.VertexAttribTypeUShort2, texCoordsType)
}

// dfInitSamplers fills the sampler set and atlas dimensions from the active views (the shared tail of all three
// constructors).
func dfInitSamplers(gp *GPBase, atlasDimensions *geom.ISize, views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState) {
	samplers := make([]TextureSampler, 0, numActiveViews)
	if numActiveViews > 0 {
		*atlasDimensions = views[0].Proxy().Dimensions()
	}
	for i := 0; i < numActiveViews; i++ {
		proxy := views[i].Proxy()
		if proxy.Dimensions() != *atlasDimensions {
			panic("atlas pages must share dimensions")
		}
		samplers = append(samplers,
			MakeGPTextureSampler(params, proxy.TextureType(), views[i].Swizzle()))
	}
	gp.setTextureSamplers(samplers)
}

// dfAddNewViews updates the sampler set shared by the DF GPs to match the atlas's current page count.
func dfAddNewViews(gp *GPBase, atlasDimensions *geom.ISize, views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState) {
	numActiveViews = min(numActiveViews, distanceFieldMaxTextures)
	if len(gp.textureSamplers) == 0 {
		*atlasDimensions = views[0].Proxy().Dimensions()
	}
	samplers := make([]TextureSampler, numActiveViews)
	copy(samplers, gp.textureSamplers)
	for i := len(gp.textureSamplers); i < numActiveViews; i++ {
		proxy := views[i].Proxy()
		if proxy.Dimensions() != *atlasDimensions {
			panic("atlas pages must share dimensions")
		}
		samplers[i] = MakeGPTextureSampler(params, proxy.TextureType(), views[i].Swizzle())
	}
	gp.setTextureSamplers(samplers)
}

//////////////////////////////////////////////////////////////////////////////
// A8 distance-field text geometry processor

// distanceFieldA8TextGeoProc is the geometry processor for A8 (grayscale) SDF text: the output color is a modulation of
// the input color and a sample from a distance-field texture, using a smoothed step function near 0.5.
type distanceFieldA8TextGeoProc struct {
	attrs [3]Attribute // inPosition, inColor, inTextureCoords
	GPBase
	localMatrix     geom.Matrix
	atlasDimensions geom.ISize
	flags           uint32
}

// newDistanceFieldA8TextGeoProc builds a distanceFieldA8TextGeoProc (the non-gamma-apply-to-A8 form). The local matrix
// should be identity if local coords are not required by the pipeline.
func newDistanceFieldA8TextGeoProc(caps *gpu.ShaderCaps, views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState, flags uint32, localMatrix *geom.Matrix) *distanceFieldA8TextGeoProc {
	if numActiveViews > distanceFieldMaxTextures {
		panic("too many atlas views")
	}
	if flags&^nonLCDDistanceFieldEffectMask != 0 {
		panic("invalid A8 DF flags")
	}
	gp := &distanceFieldA8TextGeoProc{
		localMatrix: *localMatrix,
		flags:       flags & nonLCDDistanceFieldEffectMask,
	}
	gp.initGP(DistanceFieldA8TextGeoProcClassID)

	if flags&perspectiveDistanceFieldEffectFlag != 0 {
		gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat3, GLSLTypeFloat3)
	} else {
		gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	}
	gp.attrs[1] = MakeAttribute("inColor", gpu.VertexAttribTypeUByte4Norm, GLSLTypeHalf4)
	gp.attrs[2] = dfTexCoordsAttribute(caps)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])

	dfInitSamplers(&gp.GPBase, &gp.atlasDimensions, views, numActiveViews, params)
	return gp
}

// Name implements Processor.
func (g *distanceFieldA8TextGeoProc) Name() string { return "DistanceFieldA8Text" }

// AddNewViews updates the sampler set to match the atlas's current page count.
func (g *distanceFieldA8TextGeoProc) AddNewViews(views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState) {
	dfAddNewViews(&g.GPBase, &g.atlasDimensions, views, numActiveViews, params)
}

// AddToKey implements GeometryProcessor.
func (g *distanceFieldA8TextGeoProc) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	key := g.flags
	key |= ComputeMatrixKey(caps, &g.localMatrix) << 16
	b.Add32(key, "dfA8Flags")
	b.Add32(uint32(g.NumTextureSamplers()), "numTextures")
}

// MakeProgramImpl implements GeometryProcessor.
func (g *distanceFieldA8TextGeoProc) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &distanceFieldA8TextGeoProcImpl{}
}

// distanceFieldA8TextGeoProcImpl is the shader implementation for distanceFieldA8TextGeoProc.
type distanceFieldA8TextGeoProcImpl struct {
	GPImplBase
	atlasDimensions        geom.ISize
	atlasDimensionsSet     bool
	localMatrixPrev        geom.Matrix
	localMatrixPrevSet     bool
	atlasDimensionsInvUnif UniformHandle
	localMatrixUniform     UniformHandle
}

// SetData implements GPProgramImpl.
func (i *distanceFieldA8TextGeoProcImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	dfa8gp := geomProc.(*distanceFieldA8TextGeoProc)
	if !i.atlasDimensionsSet || i.atlasDimensions != dfa8gp.atlasDimensions {
		pdman.Set2f(i.atlasDimensionsInvUnif,
			1/float32(dfa8gp.atlasDimensions.Width), 1/float32(dfa8gp.atlasDimensions.Height))
		i.atlasDimensions = dfa8gp.atlasDimensions
		i.atlasDimensionsSet = true
	}
	SetTransform(pdman, caps, i.localMatrixUniform, &dfa8gp.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
}

// emitDFAFWidth emits the "half afwidth;" computation shared by the A8 and path DF fragment shaders: the amount the
// transform scales one texel, from the st (unnormalized coords) varying's screen-space gradient.
func emitDFAFWidth(args *GPEmitArgs, flags uint32, st *Varying) {
	fb := args.FragBuilder
	isUniformScale := flags&uniformScaleDistanceFieldEffectMask ==
		uniformScaleDistanceFieldEffectMask
	isSimilarity := flags&similarityDistanceFieldEffectFlag != 0

	fb.CodeAppend("float afwidth;")
	switch {
	case isUniformScale:
		// For uniform scale, we adjust for the effect of the transformation on the distance by using the length of the
		// gradient of the t coordinate in the y direction. We use st coordinates to ensure we're mapping 1:1 from texel
		// space to pixel space. This gives us a smooth step across approximately one fragment.
		if args.ShaderCaps.AvoidDfDxForGradientsWhenPossible {
			fb.CodeAppend("afwidth = abs(" + distanceFieldAAFactorStr + "*dFdy(" + st.FsIn() +
				".y));")
		} else {
			fb.CodeAppend("afwidth = abs(" + distanceFieldAAFactorStr + "*dFdx(" + st.FsIn() +
				".x));")
		}
	case isSimilarity:
		// For similarity transforms, we adjust the effect of the transformation on the distance by using the length of
		// the gradient of the texture coordinates (the y gradient because there is a bug in the Mali 400 in the x
		// direction).
		if args.ShaderCaps.AvoidDfDxForGradientsWhenPossible {
			fb.CodeAppendf("float st_grad_len = length(dFdy(%s));", st.FsIn())
		} else {
			fb.CodeAppendf("float st_grad_len = length(dFdx(%s));", st.FsIn())
		}
		fb.CodeAppend("afwidth = abs(" + distanceFieldAAFactorStr + "*st_grad_len);")
	default:
		// For general transforms, to determine the amount of correction we multiply a unit vector pointing along the
		// SDF gradient direction by the Jacobian of the st coords (which is the inverse transform for this fragment)
		// and take the length of the result.
		fb.CodeAppend("vec2 dist_grad = vec2(dFdx(distance), dFdy(distance));")
		// The length of the gradient may be 0, so we need to check for this; this also compensates for the Adreno,
		// which likes to drop tiles on division by 0.
		fb.CodeAppend("float dg_len2 = dot(dist_grad, dist_grad);" +
			"if (dg_len2 < 0.0001) {" +
			"dist_grad = vec2(0.7071, 0.7071);" +
			"} else {" +
			"dist_grad = dist_grad*inversesqrt(dg_len2);" +
			"}")
		fb.CodeAppendf("mat2 jacobian = mat2(dFdx(%s), dFdy(%s));", st.FsIn(), st.FsIn())
		fb.CodeAppend("vec2 grad = jacobian * dist_grad;")
		// This gives us a smooth step across approximately one fragment.
		fb.CodeAppend("afwidth = " + distanceFieldAAFactorStr + "*length(grad);")
	}
}

// emitDFCoverage emits the distance→coverage step shared by the A8 and path DF fragment shaders.
func emitDFCoverage(args *GPEmitArgs, flags uint32) {
	fb := args.FragBuilder
	switch {
	case flags&aliasedDistanceFieldEffectFlag != 0:
		fb.CodeAppend("float val = distance > 0.0 ? 1.0 : 0.0;")
	case flags&gammaCorrectDistanceFieldEffectFlag != 0:
		// The smoothstep falloff compensates for the non-linear sRGB response curve. If we are doing gamma-correct
		// rendering (to an sRGB or F16 buffer), then we actually want distance mapped linearly to coverage, so use a
		// linear step. (Never set in the library, since destinations are always sRGB — kept for fidelity.)
		fb.CodeAppend("float val = clamp((distance + afwidth) / (2.0 * afwidth), 0.0, 1.0);")
	default:
		fb.CodeAppend("float val = smoothstep(-afwidth, afwidth, distance);")
	}
	fb.CodeAppendf("vec4 %s = vec4(val);", args.OutputCoverage)
}

// onEmitCode implements GPProgramImpl.
func (i *distanceFieldA8TextGeoProcImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	dfTexEffect := args.GeomProc.(*distanceFieldA8TextGeoProc)
	fragBuilder := args.FragBuilder
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler

	// Emit attributes.
	varyingHandler.EmitAttributes(dfTexEffect)

	var atlasDimensionsInvName string
	i.atlasDimensionsInvUnif, atlasDimensionsInvName = uniformHandler.AddUniform(nil,
		ShaderFlagVertex, GLSLTypeFloat2, "AtlasDimensionsInv")

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(dfTexEffect.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationInterpolated)

	// Set up position.
	gpArgs.PositionVar = dfTexEffect.attrs[0].AsShaderVar()
	WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		dfTexEffect.attrs[0].AsShaderVar(), &dfTexEffect.localMatrix, &i.localMatrixUniform)

	// Add varyings.
	var uv, texIdx, st Varying
	appendIndexUVVaryings(args, dfTexEffect.NumTextureSamplers(), dfTexEffect.attrs[2].Name(),
		atlasDimensionsInvName, &uv, &texIdx, &st)

	// Use highp to work around aliasing issues.
	fragBuilder.CodeAppendf("vec2 uv = %s;", uv.FsIn())
	fragBuilder.CodeAppend("vec4 texColor;")
	appendMultitextureLookup(args, dfTexEffect.NumTextureSamplers(), &texIdx, "uv", "texColor")

	fragBuilder.CodeAppend("float distance = " + font.DistanceFieldMultiplier +
		"*(texColor.r - " + font.DistanceFieldThreshold + ");")

	emitDFAFWidth(args, dfTexEffect.flags, &st)
	emitDFCoverage(args, dfTexEffect.flags)
}

var _ GeometryProcessor = (*distanceFieldA8TextGeoProc)(nil)

//////////////////////////////////////////////////////////////////////////////
// Path distance-field geometry processor

// distanceFieldPathGeoProc is the geometry processor for SDF path masks: as the A8 text form, but for path masks —
// positions always carry w, and the color attribute may be wide. No gamma-correct blending is applied.
type distanceFieldPathGeoProc struct {
	attrs [3]Attribute // inPosition, inColor, inTextureCoords
	GPBase
	localMatrix     geom.Matrix
	atlasDimensions geom.ISize
	flags           uint32
}

// NewDistanceFieldPathGeoProc builds a distance-field path geometry processor. It has no callers yet (the
// distance-field path renderer is not wired up), so it stays exported for that future use.
func NewDistanceFieldPathGeoProc(caps *gpu.ShaderCaps, views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState, localMatrix *geom.Matrix, flags uint32) GeometryProcessor {
	if numActiveViews > distanceFieldMaxTextures {
		panic("too many atlas views")
	}
	if flags&^pathDistanceFieldEffectMask != 0 {
		panic("invalid path DF flags")
	}
	gp := &distanceFieldPathGeoProc{
		localMatrix: *localMatrix,
		flags:       flags & pathDistanceFieldEffectMask,
	}
	gp.initGP(DistanceFieldPathGeoProcClassID)

	gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat3, GLSLTypeFloat3)
	gp.attrs[1] = MakeColorAttribute("inColor", flags&wideColorDistanceFieldEffectFlag != 0)
	gp.attrs[2] = dfTexCoordsAttribute(caps)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])

	dfInitSamplers(&gp.GPBase, &gp.atlasDimensions, views, numActiveViews, params)
	return gp
}

// Name implements Processor.
func (g *distanceFieldPathGeoProc) Name() string { return "DistanceFieldPath" }

// AddNewViews updates the sampler set to match the atlas's current page count.
func (g *distanceFieldPathGeoProc) AddNewViews(views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState) {
	dfAddNewViews(&g.GPBase, &g.atlasDimensions, views, numActiveViews, params)
}

// AddToKey implements GeometryProcessor.
func (g *distanceFieldPathGeoProc) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	key := g.flags
	key |= ComputeMatrixKey(caps, &g.localMatrix) << 16
	if g.localMatrix.HasPerspective() {
		key |= 1 << (16 + matrixKeyBits)
	}
	b.Add32(key, "dfPathFlags")
	b.Add32(uint32(g.NumTextureSamplers()), "numTextures")
}

// MakeProgramImpl implements GeometryProcessor.
func (g *distanceFieldPathGeoProc) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &distanceFieldPathGeoProcImpl{}
}

// distanceFieldPathGeoProcImpl is the shader implementation for distanceFieldPathGeoProc.
type distanceFieldPathGeoProcImpl struct {
	GPImplBase
	atlasDimensions        geom.ISize
	atlasDimensionsSet     bool
	localMatrixPrev        geom.Matrix
	localMatrixPrevSet     bool
	atlasDimensionsInvUnif UniformHandle
	localMatrixUniform     UniformHandle
}

// SetData implements GPProgramImpl. We always set the matrix uniform; it is used to transform from device to local for
// the local coord variable.
func (i *distanceFieldPathGeoProcImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	dfpgp := geomProc.(*distanceFieldPathGeoProc)
	SetTransform(pdman, caps, i.localMatrixUniform, &dfpgp.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
	if !i.atlasDimensionsSet || i.atlasDimensions != dfpgp.atlasDimensions {
		pdman.Set2f(i.atlasDimensionsInvUnif,
			1/float32(dfpgp.atlasDimensions.Width), 1/float32(dfpgp.atlasDimensions.Height))
		i.atlasDimensions = dfpgp.atlasDimensions
		i.atlasDimensionsSet = true
	}
}

// onEmitCode implements GPProgramImpl.
func (i *distanceFieldPathGeoProcImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	dfPathEffect := args.GeomProc.(*distanceFieldPathGeoProc)
	fragBuilder := args.FragBuilder
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler

	// Emit attributes.
	varyingHandler.EmitAttributes(dfPathEffect)

	var atlasDimensionsInvName string
	i.atlasDimensionsInvUnif, atlasDimensionsInvName = uniformHandler.AddUniform(nil,
		ShaderFlagVertex, GLSLTypeFloat2, "AtlasDimensionsInv")

	var uv, texIdx, st Varying
	appendIndexUVVaryings(args, dfPathEffect.NumTextureSamplers(), dfPathEffect.attrs[2].Name(),
		atlasDimensionsInvName, &uv, &texIdx, &st)

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(dfPathEffect.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationInterpolated)

	// Set up position (output position is pass-through, local coords are transformed).
	gpArgs.PositionVar = dfPathEffect.attrs[0].AsShaderVar()
	WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		dfPathEffect.attrs[0].AsShaderVar(), &dfPathEffect.localMatrix, &i.localMatrixUniform)

	// Use highp to work around aliasing issues.
	fragBuilder.CodeAppendf("vec2 uv = %s;", uv.FsIn())
	fragBuilder.CodeAppend("vec4 texColor;")
	appendMultitextureLookup(args, dfPathEffect.NumTextureSamplers(), &texIdx, "uv", "texColor")

	fragBuilder.CodeAppend("float distance = " + font.DistanceFieldMultiplier +
		"*(texColor.r - " + font.DistanceFieldThreshold + ");")

	// The path variant computes afwidth before the (non-aliased) coverage step, exactly as the A8 form; its flag mask
	// simply never carries the aliased flag.
	emitDFAFWidth(args, dfPathEffect.flags, &st)
	emitDFCoverage(args, dfPathEffect.flags&^aliasedDistanceFieldEffectFlag)
}

var _ GeometryProcessor = (*distanceFieldPathGeoProc)(nil)

//////////////////////////////////////////////////////////////////////////////
// LCD distance-field text geometry processor

// DistanceAdjust holds the per-channel distance-field adjustment for LCD text rendering.
type DistanceAdjust struct {
	R, G, B float32
}

// distanceFieldLCDTextGeoProc is the geometry processor for LCD-subpixel SDF text: samples the distance field three
// times, offset per subpixel, with per-channel distance adjustments from the distance-field adjustment table.
type distanceFieldLCDTextGeoProc struct {
	attrs [3]Attribute // inPosition, inColor, inTextureCoords
	GPBase
	localMatrix     geom.Matrix
	distanceAdjust  DistanceAdjust
	atlasDimensions geom.ISize
	flags           uint32
}

// newDistanceFieldLCDTextGeoProc builds a distanceFieldLCDTextGeoProc.
func newDistanceFieldLCDTextGeoProc(caps *gpu.ShaderCaps, views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState, distanceAdjust DistanceAdjust, flags uint32, localMatrix *geom.Matrix) *distanceFieldLCDTextGeoProc {
	if numActiveViews > distanceFieldMaxTextures {
		panic("too many atlas views")
	}
	if flags&^lcdDistanceFieldEffectMask != 0 || flags&useLCDDistanceFieldEffectFlag == 0 {
		panic("invalid LCD DF flags")
	}
	gp := &distanceFieldLCDTextGeoProc{
		localMatrix:    *localMatrix,
		distanceAdjust: distanceAdjust,
		flags:          flags & lcdDistanceFieldEffectMask,
	}
	gp.initGP(DistanceFieldLCDTextGeoProcClassID)

	if gp.flags&perspectiveDistanceFieldEffectFlag != 0 {
		gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat3, GLSLTypeFloat3)
	} else {
		gp.attrs[0] = MakeAttribute("inPosition", gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
	}
	gp.attrs[1] = MakeAttribute("inColor", gpu.VertexAttribTypeUByte4Norm, GLSLTypeHalf4)
	gp.attrs[2] = dfTexCoordsAttribute(caps)
	gp.setVertexAttributesWithImplicitOffsets(gp.attrs[:])

	dfInitSamplers(&gp.GPBase, &gp.atlasDimensions, views, numActiveViews, params)
	return gp
}

// Name implements Processor.
func (g *distanceFieldLCDTextGeoProc) Name() string { return "DistanceFieldLCDText" }

// AddNewViews updates the sampler set to match the atlas's current page count.
func (g *distanceFieldLCDTextGeoProc) AddNewViews(views []SurfaceProxyView, numActiveViews int, params gpu.SamplerState) {
	dfAddNewViews(&g.GPBase, &g.atlasDimensions, views, numActiveViews, params)
}

// AddToKey implements GeometryProcessor.
func (g *distanceFieldLCDTextGeoProc) AddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	key := ComputeMatrixKey(caps, &g.localMatrix)
	key |= g.flags << 16
	b.Add32(key, "dfLCDFlags")
	b.Add32(uint32(g.NumTextureSamplers()), "numTextures")
}

// MakeProgramImpl implements GeometryProcessor.
func (g *distanceFieldLCDTextGeoProc) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &distanceFieldLCDTextGeoProcImpl{
		distanceAdjust: DistanceAdjust{R: 1, G: 1, B: 1},
	}
}

// distanceFieldLCDTextGeoProcImpl is the shader implementation for distanceFieldLCDTextGeoProc.
type distanceFieldLCDTextGeoProcImpl struct {
	GPImplBase
	distanceAdjust         DistanceAdjust
	atlasDimensions        geom.ISize
	atlasDimensionsSet     bool
	localMatrixPrev        geom.Matrix
	localMatrixPrevSet     bool
	distanceAdjustUni      UniformHandle
	atlasDimensionsInvUnif UniformHandle
	localMatrixUniform     UniformHandle
}

// SetData implements GPProgramImpl.
func (i *distanceFieldLCDTextGeoProcImpl) SetData(pdman *ProgramDataManager, caps *gpu.ShaderCaps, geomProc GeometryProcessor) {
	dflcd := geomProc.(*distanceFieldLCDTextGeoProc)
	if dflcd.distanceAdjust != i.distanceAdjust {
		pdman.Set3f(i.distanceAdjustUni, dflcd.distanceAdjust.R, dflcd.distanceAdjust.G,
			dflcd.distanceAdjust.B)
		i.distanceAdjust = dflcd.distanceAdjust
	}
	if !i.atlasDimensionsSet || i.atlasDimensions != dflcd.atlasDimensions {
		pdman.Set2f(i.atlasDimensionsInvUnif,
			1/float32(dflcd.atlasDimensions.Width), 1/float32(dflcd.atlasDimensions.Height))
		i.atlasDimensions = dflcd.atlasDimensions
		i.atlasDimensionsSet = true
	}
	SetTransform(pdman, caps, i.localMatrixUniform, &dflcd.localMatrix, &i.localMatrixPrev,
		&i.localMatrixPrevSet)
}

// appendMultitextureLookupLCD emits the special lookup for SDF LCD, sampling the distance at uv (green), uv-offset
// (red), and uv+offset (blue) from the indexed texture sampler, avoiding duplicating the conditional logic three times.
func appendMultitextureLookupLCD(args *GPEmitArgs, numTextureSamplers int, texIdx *Varying, coordName, offsetName, distanceName string) {
	fb := args.FragBuilder
	if numTextureSamplers <= 0 {
		// This shouldn't happen, but will avoid a crash if it does.
		fb.CodeAppendf("%s = vec3(1);", distanceName)
		return
	}
	for i := 0; i < numTextureSamplers; i++ {
		if i < numTextureSamplers-1 {
			fb.CodeAppendf("if (%s == %d.0) {", texIdx.FsIn(), i)
		} else {
			// The last page is the unconditional else: texIdx is an interpolated float, so an equality test here would
			// leave the (uninitialized) distance untouched on any interpolation imprecision.
			fb.CodeAppend("{")
		}

		// Green is the distance to the uv center.
		fb.CodeAppendf("%s.y = %s.r;", distanceName,
			fb.AppendTextureLookup(args.UniformHandler, args.TexSamplers[i], coordName))

		// Red is the distance to the left offset.
		fb.CodeAppendf("vec2 uv_adjusted = %s - %s;", coordName, offsetName)
		fb.CodeAppendf("%s.x = %s.r;", distanceName,
			fb.AppendTextureLookup(args.UniformHandler, args.TexSamplers[i], "uv_adjusted"))

		// Blue is the distance to the right offset.
		fb.CodeAppendf("uv_adjusted = %s + %s;", coordName, offsetName)
		fb.CodeAppendf("%s.z = %s.r;", distanceName,
			fb.AppendTextureLookup(args.UniformHandler, args.TexSamplers[i], "uv_adjusted"))

		if i < numTextureSamplers-1 {
			fb.CodeAppend("} else ")
		} else {
			fb.CodeAppend("}")
		}
	}
}

// onEmitCode implements GPProgramImpl.
func (i *distanceFieldLCDTextGeoProcImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	dfTexEffect := args.GeomProc.(*distanceFieldLCDTextGeoProc)
	vertBuilder := args.VertBuilder
	varyingHandler := args.VaryingHandler
	uniformHandler := args.UniformHandler
	fragBuilder := args.FragBuilder

	// Emit attributes.
	varyingHandler.EmitAttributes(dfTexEffect)

	var atlasDimensionsInvName string
	i.atlasDimensionsInvUnif, atlasDimensionsInvName = uniformHandler.AddUniform(nil,
		ShaderFlagVertex, GLSLTypeFloat2, "AtlasDimensionsInv")

	// Set up pass-through color.
	fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
	varyingHandler.AddPassThroughAttribute(dfTexEffect.attrs[1].AsShaderVar(), args.OutputColor,
		InterpolationInterpolated)

	// Set up position.
	gpArgs.PositionVar = dfTexEffect.attrs[0].AsShaderVar()
	WriteLocalCoord(vertBuilder, uniformHandler, args.ShaderCaps, gpArgs,
		dfTexEffect.attrs[0].AsShaderVar(), &dfTexEffect.localMatrix, &i.localMatrixUniform)

	// Set up varyings.
	var uv, texIdx, st Varying
	appendIndexUVVaryings(args, dfTexEffect.NumTextureSamplers(), dfTexEffect.attrs[2].Name(),
		atlasDimensionsInvName, &uv, &texIdx, &st)

	delta := NewVarying(GLSLTypeFloat)
	varyingHandler.AddVarying("Delta", &delta)
	if dfTexEffect.flags&portraitDistanceFieldEffectFlag != 0 {
		if dfTexEffect.flags&bgrDistanceFieldEffectFlag != 0 {
			vertBuilder.CodeAppendf("%s = -%s.y/3.0;", delta.VsOut(), atlasDimensionsInvName)
		} else {
			vertBuilder.CodeAppendf("%s = %s.y/3.0;", delta.VsOut(), atlasDimensionsInvName)
		}
	} else {
		if dfTexEffect.flags&bgrDistanceFieldEffectFlag != 0 {
			vertBuilder.CodeAppendf("%s = -%s.x/3.0;", delta.VsOut(), atlasDimensionsInvName)
		} else {
			vertBuilder.CodeAppendf("%s = %s.x/3.0;", delta.VsOut(), atlasDimensionsInvName)
		}
	}

	isUniformScale := dfTexEffect.flags&uniformScaleDistanceFieldEffectMask ==
		uniformScaleDistanceFieldEffectMask
	isSimilarity := dfTexEffect.flags&similarityDistanceFieldEffectFlag != 0
	isGammaCorrect := dfTexEffect.flags&gammaCorrectDistanceFieldEffectFlag != 0

	// Create the LCD offset adjusted by the inverse of the transform. Use highp to work around aliasing issues.
	fragBuilder.CodeAppendf("vec2 uv = %s;", uv.FsIn())

	switch {
	case isUniformScale:
		if args.ShaderCaps.AvoidDfDxForGradientsWhenPossible {
			fragBuilder.CodeAppendf("float st_grad_len = abs(dFdy(%s.y));", st.FsIn())
		} else {
			fragBuilder.CodeAppendf("float st_grad_len = abs(dFdx(%s.x));", st.FsIn())
		}
		if dfTexEffect.flags&portraitDistanceFieldEffectFlag != 0 {
			fragBuilder.CodeAppendf("vec2 offset = vec2(0.0, st_grad_len*%s);", delta.FsIn())
		} else {
			fragBuilder.CodeAppendf("vec2 offset = vec2(st_grad_len*%s, 0.0);", delta.FsIn())
		}
	case isSimilarity:
		// For a similarity matrix with rotation, the gradient will not be aligned with the texel coordinate axes, so we
		// need to calculate it.
		if args.ShaderCaps.AvoidDfDxForGradientsWhenPossible {
			// We use dFdy instead and rotate -90 degrees to get the gradient in the x direction.
			fragBuilder.CodeAppendf("vec2 st_grad = dFdy(%s);", st.FsIn())
			if dfTexEffect.flags&portraitDistanceFieldEffectFlag != 0 {
				fragBuilder.CodeAppendf("vec2 offset = %s*st_grad;", delta.FsIn())
			} else {
				fragBuilder.CodeAppendf("vec2 offset = %s*vec2(st_grad.y, -st_grad.x);",
					delta.FsIn())
			}
		} else {
			fragBuilder.CodeAppendf("vec2 st_grad = dFdx(%s);", st.FsIn())
			if dfTexEffect.flags&portraitDistanceFieldEffectFlag != 0 {
				fragBuilder.CodeAppendf("vec2 offset = %s*vec2(-st_grad.y, st_grad.x);",
					delta.FsIn())
			} else {
				fragBuilder.CodeAppendf("vec2 offset = %s*st_grad;", delta.FsIn())
			}
		}
		fragBuilder.CodeAppend("float st_grad_len = length(st_grad);")
	default:
		fragBuilder.CodeAppendf("vec2 st = %s;", st.FsIn())
		fragBuilder.CodeAppend("mat2 jacobian = mat2(dFdx(st), dFdy(st));")
		if dfTexEffect.flags&portraitDistanceFieldEffectFlag != 0 {
			fragBuilder.CodeAppendf("vec2 offset = jacobian * vec2(0, %s);", delta.FsIn())
		} else {
			fragBuilder.CodeAppendf("vec2 offset = jacobian * vec2(%s, 0);", delta.FsIn())
		}
	}

	// Sample the texture by index.
	fragBuilder.CodeAppend("vec3 distance;")
	appendMultitextureLookupLCD(args, dfTexEffect.NumTextureSamplers(), &texIdx, "uv", "offset",
		"distance")
	fragBuilder.CodeAppend("distance = vec3(" + font.DistanceFieldMultiplier +
		")*(distance - vec3(" + font.DistanceFieldThreshold + "));")

	// Adjust width based on gamma.
	var distanceAdjustUniName string
	i.distanceAdjustUni, distanceAdjustUniName = uniformHandler.AddUniform(nil,
		ShaderFlagFragment, GLSLTypeHalf3, "DistanceAdjust")
	fragBuilder.CodeAppendf("distance -= %s;", distanceAdjustUniName)

	// To be strictly correct, we should compute the anti-aliasing factor separately for each color component. However,
	// this is only important when using perspective transformations, and even then using a single factor seems like a
	// reasonable trade-off between quality and speed.
	fragBuilder.CodeAppend("float afwidth;")
	if isSimilarity {
		// For similarity transforms (uniform scale-only is a subset), we adjust for the effect of the transformation on
		// the distance by using the length of the gradient of the texture coordinates — a smooth step across
		// approximately one fragment.
		fragBuilder.CodeAppend("afwidth = " + distanceFieldAAFactorStr + "*st_grad_len;")
	} else {
		// For general transforms, multiply a unit vector pointing along the SDF gradient direction by the Jacobian of
		// the st coords and take the length of the result.
		fragBuilder.CodeAppend("vec2 dist_grad = vec2(dFdx(distance.r), dFdy(distance.r));")
		// The length of the gradient may be 0, so we need to check for this; this also compensates for the Adreno,
		// which likes to drop tiles on division by 0.
		fragBuilder.CodeAppend("float dg_len2 = dot(dist_grad, dist_grad);" +
			"if (dg_len2 < 0.0001) {" +
			"dist_grad = vec2(0.7071, 0.7071);" +
			"} else {" +
			"dist_grad = dist_grad*inversesqrt(dg_len2);" +
			"}" +
			"vec2 grad = jacobian * dist_grad;")
		// This gives us a smooth step across approximately one fragment.
		fragBuilder.CodeAppend("afwidth = " + distanceFieldAAFactorStr + "*length(grad);")
	}

	// The smoothstep falloff compensates for the non-linear sRGB response curve; for gamma-correct rendering distance
	// maps linearly to coverage (never set here).
	if isGammaCorrect {
		fragBuilder.CodeAppendf("vec4 %s = vec4(clamp((distance + vec3(afwidth)) / "+
			"vec3(2.0 * afwidth), 0.0, 1.0), 1.0);", args.OutputCoverage)
	} else {
		fragBuilder.CodeAppendf("vec4 %s = vec4(smoothstep(vec3(-afwidth), vec3(afwidth), "+
			"distance), 1.0);", args.OutputCoverage)
	}
}

var _ GeometryProcessor = (*distanceFieldLCDTextGeoProc)(nil)
