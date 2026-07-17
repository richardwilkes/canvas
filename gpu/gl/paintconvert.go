// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Paint conversion (turning a canvas paint into a GPU Paint plus fragment processor tree) and the
// shader-to-fragment-processor dispatcher over the shader descriptors. The primitive-color blender and replace-shader
// lanes serve the ops that use them; there are no runtime blenders or non-blend-mode blenders, so the paint blend is
// always a blend mode (Porter-Duff XP or a custom transfer mode). Image shaders convert via makeImageShaderFP (over the
// polymorphic image, so a same-context texture-backed image shader draws from its own view) and perlin-noise shaders
// via makePerlinNoiseFP (perlinnoisefp.go); the remaining tail is the image-filter source lane.

package gl

import (
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// FPArgs carries the context, caps, and destination color type needed to build fragment processors for a draw, with
// color info reduced to just the destination color type.
type FPArgs struct {
	Ctx          *DirectContext
	Caps         *Caps
	DstColorType gpu.ColorType
}

// MakeFPArgs builds the FPArgs for a surface draw context.
func MakeFPArgs(sdc *SurfaceDrawContext) *FPArgs {
	return &FPArgs{Ctx: sdc.Context(), Caps: sdc.Caps(), DstColorType: sdc.ColorType()}
}

// MakeShaderFP converts a shader descriptor into a fragment processor. Returns nil when the shader cannot be
// represented (the draw is then skipped).
func MakeShaderFP(shader shaders.Shader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	if shader == nil {
		return nil
	}
	switch s := shader.(type) {
	case *shaders.ColorShader:
		// map_color: sRGB → dst is the identity; premultiply.
		c := s.Color()
		return MakeColorFP(colorcore.PMColor4f{R: c.R * c.A, G: c.G * c.A, B: c.B * c.A, A: c.A})
	case *shaders.BlendShader:
		fpA := MakeShaderFP(s.Dst(), args, mRec)
		fpB := MakeShaderFP(s.Src(), args, mRec)
		if fpA == nil || fpB == nil {
			// Unexpected: both src and dst shaders should be valid. Just fail.
			return nil
		}
		return BlendFP(fpB, fpA, s.Mode())
	case *shaders.LocalMatrixShader:
		concatenated := mRec.Concat(s.LocalMatrix())
		return MakeShaderFP(s.Wrapped(), args, &concatenated)
	case *shaders.ColorFilterShader:
		shaderFP := MakeShaderFP(s.Shader(), args, mRec)
		if shaderFP == nil {
			return nil
		}
		if s.Alpha() != 1 {
			panic("ColorFilterShader alpha != 1 is unreachable via the public surface")
		}
		// If the filter FP could not be created we still want to return the shader FP, so the success flag is
		// deliberately ignored.
		fp, _ := MakeColorFilterFP(args, s.Filter(), shaderFP)
		return fp
	case *shaders.LinearGradient:
		return makeLinearGradientFP(s, args, mRec)
	case *shaders.RadialGradient:
		return makeRadialGradientFP(s, args, mRec)
	case *shaders.SweepGradient:
		return makeSweepGradientFP(s, args, mRec)
	case *shaders.ConicalGradient:
		return makeConicalGradientFP(s, args, mRec)
	case *shaders.ImageShader:
		return makeImageShaderFP(s, args, mRec)
	case *shaders.PerlinNoiseShader:
		return makePerlinNoiseFP(s, args, mRec)
	case *shaders.FilterDecalShader:
		// The kDecal runtime effect (filtercore's layer-space decal wrap): the child samples at the decal's own coords
		// (explicit sampling), so it converts against a fresh MatrixRec, and the pending matrix applies around the
		// decal FP itself — the CPU leg's identity-seeded apply.
		childRec := shaders.NewMatrixRec(geom.IdentityMatrix())
		childFP := MakeShaderFP(s.Image(), args, &childRec)
		if childFP == nil {
			return nil
		}
		identity := geom.IdentityMatrix()
		total, ok := mRec.ApplyForFragmentProcessor(&identity)
		if !ok {
			return nil
		}
		return MatrixEffectFP(total, NewFilterDecalFP(s.DecalBounds(), childFP))
	case *shaders.MorphologyShader:
		return makeMorphologyFP(s, args, mRec)
	case *shaders.DisplacementShader:
		return makeDisplacementFP(s, args, mRec)
	case *shaders.NormalShader:
		return makeNormalFP(s, args, mRec)
	case *shaders.LightingShader:
		return makeLightingFP(s, args, mRec)
	case *shaders.MagnifierShader:
		return makeMagnifierFP(s, args, mRec)
	case *shaders.MatrixConvShader:
		return makeMatrixConvFP(s, args, mRec)
	case *shaders.ArithmeticShader:
		return makeArithmeticFP(s, args, mRec)
	default:
		if shaders.IsEmpty(shader) {
			return nil
		}
		// Any remaining unrepresentable shader fails FP conversion; the draw is skipped.
		return nil
	}
}

// makeImageShaderFP converts an image shader into a fragment processor: it resolves the shader's image to a texture
// view (drawableAsView — a same-context texture image contributes its own view with no readback, the GPU-native lane; a
// CPU image uploads once through the shared upload-by-image-ID cache the image-draw lane uses) and samples it with the
// shader's per-axis tile modes and sampling, then wraps the texture effect in the local-matrix effect that maps device
// coords back to image (local) space. Image shaders always cover the whole image — no entry point exposes a subset (see
// shaders/imageshader.go) — so the subset lane is never taken. Cubic/mipmap/aniso sampling degrades to the base level's
// filter, exactly as the DrawImageRect general lane does; the color-space transform is the identity. Returns nil if the
// upload/readback fails or the total local matrix is singular.
func makeImageShaderFP(s *shaders.ImageShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	img := s.DrawableImage()
	view, _ := drawableAsView(args.Ctx, img)
	if !view.IsValid() {
		return nil
	}
	// The texture holds premul (or opaque) pixels — a same-context texture image is stored premul, and a CPU upload
	// premultiplies — so the texture's alpha type is premul unless the image is tagged opaque, the same reduction
	// DrawImageRect makes.
	alphaType := gpu.AlphaTypePremul
	if img.AlphaType() == imagecore.AlphaTypeOpaque {
		alphaType = gpu.AlphaTypeOpaque
	}
	tmx, tmy := s.TileModes()
	sampler := gpu.MakeSamplerState(tileModeToWrapMode(tmx), tileModeToWrapMode(tmy),
		downgradeToFilter(s.Sampling()), gpu.MipmapModeNone)
	// The texture effect gets the identity matrix; the local-to-device mapping is carried by the outer matrix-effect
	// wrapper below.
	fp := MakeTextureEffect(view, alphaType, geom.IdentityMatrix(), sampler, args.Caps, [4]float32{})
	identity := geom.IdentityMatrix()
	total, ok := mRec.ApplyForFragmentProcessor(&identity)
	if !ok {
		return nil
	}
	fp = MatrixEffectFP(total, fp)
	// Alpha-only image shaders are tinted by the input (paint) color, matching how the CPU raster path tints alpha-only
	// images from the paint color.
	if img.IsAlphaOnly() {
		fp = BlendFP(fp, nil, raster.BlendDstIn)
	}
	return fp
}

//////////////////////////////////////////////////////////////////////////////
// Dither: the dither fragment processor and its runtime effect.

var (
	ditherLUTOnce  sync.Once
	ditherLUTBytes []byte
	ditherLUTKey   gpu.UniqueKey
)

// makeDitherEffectFP wraps inputFP so each fragment's RGB is offset by the 8x8 table value scaled by range, clamped to
// [0, alpha].
func makeDitherEffectFP(ctx *DirectContext, inputFP FragmentProcessor, rangeVal float32, caps *Caps) FragmentProcessor {
	if rangeVal == 0 || inputFP == nil {
		return inputFP
	}

	ditherLUTOnce.Do(func() {
		lut := gpu.MakeDitherLUT()
		ditherLUTBytes = lut[:]
		builder := gpu.UniqueKeyBuilder(&ditherLUTKey, gpu.GenerateUniqueKeyDomain(), 1,
			"DitherLUT")
		builder.Slice()[0] = 1
		builder.Finish()
	})
	view := FindOrCreatePixelsProxyView(ctx, &ditherLUTKey,
		geom.ISize{Width: gpu.DitherLUTSize, Height: gpu.DitherLUTSize}, gpu.ColorTypeAlpha8,
		ditherLUTBytes, gpu.DitherLUTSize, "MakeDitherEffect")
	if !view.IsValid() {
		return inputFP
	}
	sampler := gpu.MakeSamplerState(gpu.WrapModeRepeat, gpu.WrapModeRepeat, gpu.FilterModeNearest,
		gpu.MipmapModeNone)
	te := MakeTextureEffect(view, gpu.AlphaTypePremul, geom.IdentityMatrix(), sampler, caps,
		[4]float32{})
	return makeDitherFP(inputFP, te, rangeVal)
}

type ditherFP struct {
	FPBase
	rangeVal float32
}

func makeDitherFP(inputFP, table FragmentProcessor, rangeVal float32) FragmentProcessor {
	fp := &ditherFP{rangeVal: rangeVal}
	fp.initFP(GLSLFPClassID, FPPreservesOpaqueInput)
	// "inputFP" is a regular child (flags merged); "table" is IgnoreOptFlags.
	fp.mergeOptimizationFlags(fpProcessorOptimizationFlags(inputFP))
	registerChildOf(fp, inputFP, PassThroughSampleUsage())
	registerChildOf(fp, table, FragCoordSampleUsage())
	return fp
}

func (f *ditherFP) Name() string { return "Dither" }

func (f *ditherFP) Clone() FragmentProcessor {
	clone := &ditherFP{rangeVal: f.rangeVal}
	clone.initFP(GLSLFPClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *ditherFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*ditherFP)
	return ok && f.rangeVal == o.rangeVal
}

func (f *ditherFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPDither), "runtimeFPKind")
}

func (f *ditherFP) onMakeProgramImpl() FPProgramImpl { return &ditherFPImpl{} }

type ditherFPImpl struct {
	FPImplBase
	rangeUni UniformHandle
}

func (i *ditherFPImpl) EmitCode(args *FPEmitArgs) {
	var rangeName string
	i.rangeUni, rangeName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf, "range")
	fb := args.FragBuilder
	fb.CodeAppendf("vec4 color = %s;", i.InvokeChild(0 /* inputFP */, args))
	// Undo the bias in the table.
	fb.CodeAppendf("float value = %s.a - 0.5;", i.InvokeChildWithCoords(1 /* table */, "", args, fragCoordName))
	// For each color channel, add the random offset to the channel value and then clamp between 0 and alpha to keep the
	// color premultiplied.
	fb.CodeAppendf("return vec4(clamp(color.rgb + value * %s, 0.0, color.a), color.a);",
		rangeName)
}

func (i *ditherFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	pdman.Set1f(i.rangeUni, fp.(*ditherFP).rangeVal)
}

//////////////////////////////////////////////////////////////////////////////
// Paint conversion.

// PaintParams carries the paint state the conversion consumes (the device hands these over from canvas.Paint).
type PaintParams struct {
	Shader         shaders.Shader      // may be nil
	ColorFilter    shaders.ColorFilter // may be nil
	Color          colorcore.Color4f   // unpremul sRGB paint color
	BlendMode      raster.BlendMode
	Dither         bool
	HasMaskFilter  bool // participates in ShouldDither only
	HasImageFilter bool // participates in ShouldDither only
}

// shouldDitherPaint reports whether the paint's draw should apply dithering for the destination color type.
func shouldDitherPaint(pp *PaintParams, dstCT gpu.ColorType) bool {
	// The paint dither flag can veto.
	if !pp.Dither || dstCT == gpu.ColorTypeUnknown {
		return false
	}
	// We always dither 565 or 4444 when requested.
	if dstCT == gpu.ColorTypeBGR565 || dstCT == gpu.ColorTypeRGB565 ||
		dstCT == gpu.ColorTypeABGR4444 {
		return true
	}
	// Otherwise, dither is only needed for non-const paints.
	return pp.HasImageFilter || pp.HasMaskFilter ||
		(pp.Shader != nil && !pp.Shader.IsConstant())
}

// MakePaint converts pp into a GPU Paint, without the primitive-color-blender lane. Returns (nil, false) when the paint
// cannot be represented and the draw must be skipped.
func MakePaint(sdc *SurfaceDrawContext, pp *PaintParams, ctm geom.Matrix) (*Paint, bool) {
	return makePaintImpl(sdc, pp, ctm, nil, false, false, 0)
}

// MakePaintReplaceShader converts the paint with a pre-built fragment processor standing in for the shader (the
// image-draw lanes use this with their texture FPs). shaderFP must be non-nil.
func MakePaintReplaceShader(sdc *SurfaceDrawContext, pp *PaintParams, ctm geom.Matrix, shaderFP FragmentProcessor) (*Paint, bool) {
	if shaderFP == nil {
		return nil, false
	}
	return makePaintImpl(sdc, pp, ctm, shaderFP, true, false, 0)
}

// MakePaintWithBlend is the paint conversion for draws whose geometry processor supplies a primitive color
// (DrawAtlasOp's per-sprite colors), blended with the paint's shader/color through primColorMode. There are no runtime
// blenders, so the primitive-color blender is always a blend mode (blenderRequiresShader == mode != kDst).
func MakePaintWithBlend(sdc *SurfaceDrawContext, pp *PaintParams, ctm geom.Matrix, primColorMode raster.BlendMode) (*Paint, bool) {
	return makePaintImpl(sdc, pp, ctm, nil, false, true, primColorMode)
}

// makePaintImpl is the shared implementation behind the MakePaint* entry points. havePrimColorBlender selects the
// primitive-color-blender lanes (MakePaintWithBlend), with the blender reduced to primColorMode since only blend-mode
// blenders exist.
func makePaintImpl(sdc *SurfaceDrawContext, pp *PaintParams, ctm geom.Matrix, shaderFPOverride FragmentProcessor, haveShaderFPOverride, havePrimColorBlender bool, primColorMode raster.BlendMode) (*Paint, bool) {
	args := MakeFPArgs(sdc)
	gpuPaint := borrowPaint()

	// Destination color-space prep is the identity.
	origColor := pp.Color

	// blender_requires_shader: a blend-mode blender needs the shader lane unless it is kDst (which discards the
	// shader/paint side entirely).
	blenderRequiresShader := havePrimColorBlender && primColorMode != raster.BlendDst

	// Set up the initial color considering the shader, the paint color, and the presence or not of per-vertex colors.
	var paintFP FragmentProcessor
	if !havePrimColorBlender || blenderRequiresShader {
		switch {
		case haveShaderFPOverride:
			paintFP = shaderFPOverride
		case pp.Shader != nil:
			mRec := shaders.NewMatrixRec(ctm)
			paintFP = MakeShaderFP(pp.Shader, args, &mRec)
			if paintFP == nil {
				recyclePaint(gpuPaint)
				return nil, false
			}
		}
	}

	// Set if the shader/paint-color/paint-alpha output is a known constant, in which case the color filter is applied
	// to the paint color instead of converting it to an FP.
	applyColorFilterToPaintColor := false
	switch {
	case paintFP != nil && havePrimColorBlender:
		// There is a blend between the primitive color and the shader color. The shader sees the opaque paint color.
		// The shader's output is blended using the provided mode by the primitive color (the blend's dst is the FP
		// input color, which the geometry processor seeds with the per-vertex color — the Paint's own color is
		// ignored). The blended color is then modulated by the paint's alpha.
		shaderInput := colorcore.PMColor4f{R: origColor.R, G: origColor.G, B: origColor.B, A: 1}
		paintFP = OverrideInputFP(paintFP, shaderInput)
		paintFP = BlendFP(paintFP, nil, primColorMode)
		if paintFP == nil {
			recyclePaint(gpuPaint)
			return nil, false
		}
		if origColor.A != 1 {
			a := origColor.A
			paintFP = ModulateRGBAFP(paintFP, colorcore.PMColor4f{R: a, G: a, B: a, A: a})
		}
	case paintFP != nil:
		if origColor.A != 1 {
			// Invoke the shader's FP tree with an opaque version of the paint color, then multiply the final result by
			// the incoming (paint) alpha. The *unpremul* paint color goes on the Paint: the shader sees the original
			// opaque RGB, and ApplyPaintAlphaFP then creates a valid premul color by applying the paint alpha.
			paintFP = ApplyPaintAlphaFP(paintFP)
			gpuPaint.SetColor4f(colorcore.PMColor4f(origColor))
		} else {
			// paintFP will ignore its input color, so coverage-as-alpha must be disabled.
			paintFP = DisableCoverageAsAlphaFP(paintFP)
			gpuPaint.SetColor4f(premulColor4f(origColor))
		}
	case havePrimColorBlender:
		// The primitive itself has color (the GP's per-vertex color output); the Paint color won't be used. kDst keeps
		// the primitive color as-is; every other mode blends the opaque paint color with it, and the paint's alpha
		// applies after the blend.
		gpuPaint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}) // won't be used
		if blenderRequiresShader {
			paintFP = MakeColorFP(colorcore.PMColor4f{
				R: origColor.R, G: origColor.G, B: origColor.B, A: 1,
			})
			paintFP = BlendFP(paintFP, nil, primColorMode)
			if paintFP == nil {
				recyclePaint(gpuPaint)
				return nil, false
			}
		}
		if origColor.A != 1 {
			a := origColor.A
			paintFP = ModulateRGBAFP(paintFP, colorcore.PMColor4f{R: a, G: a, B: a, A: a})
		}
	default:
		// No shader, no primitive color.
		gpuPaint.SetColor4f(premulColor4f(origColor))
		applyColorFilterToPaintColor = true
	}

	if pp.ColorFilter != nil {
		if applyColorFilterToPaintColor {
			// FilterColor4f premuls, runs the filter's stages, and unpremuls; the round trip back to premul below
			// matches its exact rounding.
			filtered, ok := shaders.FilterColor4f(pp.ColorFilter, premulColor4f(origColor))
			if !ok {
				recyclePaint(gpuPaint)
				return nil, false
			}
			gpuPaint.SetColor4f(pmRoundTripUnpremul(filtered))
		} else {
			fp, ok := MakeColorFilterFP(args, pp.ColorFilter, paintFP)
			if !ok {
				recyclePaint(gpuPaint)
				return nil, false
			}
			paintFP = fp
		}
	}

	if paintFP != nil && shouldDitherPaint(pp, args.DstColorType) {
		ditherRange := gpu.DitherRangeForColorType(args.DstColorType)
		paintFP = makeDitherEffectFP(args.Ctx, paintFP, ditherRange, args.Caps)
	}

	// The final blend is always a blend mode, preferring the transfer processor over any FP-based blend: coefficient
	// modes blend directly without readback; advanced modes go through a custom transfer mode.
	if pp.BlendMode != raster.BlendSrcOver {
		gpuPaint.SetXPFactory(XPFactoryFromBlendMode(pp.BlendMode))
	}

	// The clamp type is always "auto" for every color type in the matrix, so the manual clamp lane (pinned paint color)
	// is unreachable.

	if paintFP != nil {
		gpuPaint.SetColorFragmentProcessor(paintFP)
	}
	return gpuPaint, true
}

// pmRoundTripUnpremul re-premultiplies a color that was unpremultiplied by a color filter, matching the
// unpremul-then-premul round trip exactly except at alpha == 0, where both produce transparent black.
func pmRoundTripUnpremul(c colorcore.PMColor4f) colorcore.PMColor4f {
	if c.A == 0 {
		return colorcore.PMColor4f{}
	}
	inv := 1 / c.A
	return colorcore.PMColor4f{R: c.R * inv * c.A, G: c.G * inv * c.A, B: c.B * inv * c.A, A: c.A}
}
