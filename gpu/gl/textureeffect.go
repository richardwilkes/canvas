// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// TextureEffect is the fragment processor that samples a texture, implementing whatever part of the requested
// wrap/filter behavior the hardware sampler cannot: shader-based clamp/repeat/mirror/ border, subset restriction, and
// the extra reads that emulate filtered repeat. The emitted code is GLSL. Only 2D and rectangle texture types are
// supported (no external/array textures).

package gl

import (
	"fmt"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// TextureEffectShaderMode identifies the possible implementations of a wrap mode in shader code. Some modes are
// specialized by filter.
type TextureEffectShaderMode uint16

// TextureEffectShaderMode values (8-bit key fields).
const (
	ShaderModeNone TextureEffectShaderMode = iota // using HW mode
	ShaderModeClamp
	ShaderModeRepeatNearestNone
	ShaderModeRepeatLinearNone
	ShaderModeRepeatLinearMipmap
	ShaderModeRepeatNearestMipmap
	ShaderModeMirrorRepeat
	ShaderModeClampToBorderNearest
	ShaderModeClampToBorderFilter
)

// textureEffectInsetEpsilon is an extra inset applied when clamping shader-sampled texture coordinates. At zero, linear
// filtering clamps coordinates to exactly 1/2px inside the defined data boundary; for the bottom/right edges of an
// approx-fit texture the 2x2 bilinear footprint then touches the first row/column of the undefined padding with an
// ideal weight of 0 — but devices with lower-precision texture coordinates (software rasterizers in particular) do not
// always evaluate that weight as exactly 0, producing nondeterministic ±1 LSB output from uninitialized texels. The
// same inset keeps the domain-is-safe promise honest: a domain that touches the 1/2px boundary exactly no longer
// qualifies for dropping the shader clamp. This mirrors upstream GrTextureEffect::kInsetEpsilon under
// SK_USE_SAFE_INSET_FOR_TEXTURE_SAMPLING, which exists for exactly this failure (caught there by Swiftshader+MSAN); the
// quad-based texture-op lane has always carried the equivalent guard (its linearInset is 0.5 + 1/1024).
const textureEffectInsetEpsilon = 0.0001

// TextureEffectLinearInset is the inset applied to a subset rect's edges for linear filtering.
const TextureEffectLinearInset = 0.5 + textureEffectInsetEpsilon

// getTextureEffectShaderMode picks the shader mode needed to implement wrap/filter/mipmap combinations that the
// hardware sampler cannot express directly.
func getTextureEffectShaderMode(wrap gpu.WrapMode, filter gpu.FilterMode, mm gpu.MipmapMode) TextureEffectShaderMode {
	switch wrap {
	case gpu.WrapModeMirrorRepeat:
		return ShaderModeMirrorRepeat
	case gpu.WrapModeClamp:
		return ShaderModeClamp
	case gpu.WrapModeRepeat:
		if mm == gpu.MipmapModeNone {
			if filter == gpu.FilterModeNearest {
				return ShaderModeRepeatNearestNone
			}
			return ShaderModeRepeatLinearNone
		}
		if filter == gpu.FilterModeNearest {
			return ShaderModeRepeatNearestMipmap
		}
		return ShaderModeRepeatLinearMipmap
	case gpu.WrapModeClampToBorder:
		if filter == gpu.FilterModeNearest {
			return ShaderModeClampToBorderNearest
		}
		return ShaderModeClampToBorderFilter
	}
	panic("unknown wrap mode")
}

// shaderModeIsClampToBorder reports whether m implements clamp-to-border in the shader.
func shaderModeIsClampToBorder(m TextureEffectShaderMode) bool {
	return m == ShaderModeClampToBorderNearest || m == ShaderModeClampToBorderFilter
}

// shaderModeRequiresUnormCoord reports whether m needs unnormalized texture coordinates: when filtering logic runs in
// the shader, coordinates are operated on unnormalized and normalized after the mode is handled (via the idims uniform)
// if the texture is not a rectangle texture.
func shaderModeRequiresUnormCoord(m TextureEffectShaderMode) bool {
	switch m {
	case ShaderModeRepeatLinearNone, ShaderModeRepeatNearestMipmap, ShaderModeRepeatLinearMipmap,
		ShaderModeClampToBorderNearest, ShaderModeClampToBorderFilter:
		return true
	default:
		return false
	}
}

// textureEffectSpan is a 1D interval, used to compute the shader subset/clamp rects for one axis at a time.
type textureEffectSpan struct {
	a, b float32
}

func (s textureEffectSpan) makeInset(o float32) textureEffectSpan {
	r := textureEffectSpan{a: s.a + o, b: s.b - o}
	if r.a > r.b {
		r.a = (r.a + r.b) / 2
		r.b = r.a
	}
	return r
}

func (s textureEffectSpan) contains(r textureEffectSpan) bool { return s.a <= r.a && s.b >= r.b }

// textureEffectSampling holds the resolved sampler state and shader-side parameters (modes, subset, clamp, border)
// needed to sample a texture with the requested wrap/filter behavior.
type textureEffectSampling struct {
	hwSampler    gpu.SamplerState
	shaderModes  [2]TextureEffectShaderMode
	shaderSubset geom.Rect
	shaderClamp  geom.Rect
	border       [4]float32
}

func (s *textureEffectSampling) hasBorderAlpha() bool {
	if s.hwSampler.WrapModeX == gpu.WrapModeClampToBorder ||
		s.hwSampler.WrapModeY == gpu.WrapModeClampToBorder {
		return true
	}
	if shaderModeIsClampToBorder(s.shaderModes[0]) || shaderModeIsClampToBorder(s.shaderModes[1]) {
		return s.border[3] < 1
	}
	return false
}

func isPow2(v int32) bool { return v > 0 && v&(v-1) == 0 }

// makeTextureEffectSampling resolves the sampler state and shader parameters needed to sample proxy with the given
// wrap/filter sampler, subset restriction, optional known-safe domain, and border color.
func makeTextureEffectSampling(proxy *SurfaceProxy, sampler gpu.SamplerState, subset geom.Rect, domain *geom.Rect, border [4]float32, alwaysUseShaderTileMode bool, caps *gpu.Caps, linearFilterInset geom.Point) textureEffectSampling {
	var out textureEffectSampling

	textureType := proxy.TextureType()
	filter := sampler.Filter
	mm := sampler.MipmapMode

	canDoWrapInHW := func(size int32, wrap gpu.WrapMode) bool {
		if alwaysUseShaderTileMode {
			return false
		}
		// TODO: use HW border color when available.
		if wrap == gpu.WrapModeClampToBorder &&
			(!caps.ClampToBorderSupport || border[0] != 0 || border[1] != 0 || border[2] != 0 ||
				border[3] != 0) {
			return false
		}
		if wrap != gpu.WrapModeClamp && !caps.NPOTTextureTileSupport && !isPow2(size) {
			return false
		}
		if textureType != gpu.TextureType2D &&
			(wrap != gpu.WrapModeClamp && wrap != gpu.WrapModeClampToBorder) {
			return false
		}
		return true
	}

	dim := geom.ISize{Width: -1, Height: -1}
	if !proxy.IsFullyLazy() {
		dim = proxy.BackingStoreDimensions()
	}

	// If we use shader-based subsetting for any reason we just completely drop aniso.
	aniso := sampler.IsAniso()
	if aniso && !caps.AnisoSupport {
		panic("aniso sampler without support")
	}
	if aniso {
		backing := geom.Rect{
			Right:  float32(dim.Width),
			Bottom: float32(dim.Height),
		}
		anisoSubset := !rectContains(subset, backing) && (domain == nil || !rectContains(subset, *domain))
		needsShaderWrap := !canDoWrapInHW(dim.Width, sampler.WrapModeX) ||
			!canDoWrapInHW(dim.Height, sampler.WrapModeY)
		if needsShaderWrap || anisoSubset {
			newMM := gpu.MipmapModeNone
			if proxy.AsTextureProxy().Mipmapped() == gpu.MipmappedYes {
				newMM = gpu.MipmapModeLinear
			}
			sampler = gpu.MakeSamplerState(sampler.WrapModeX, sampler.WrapModeY,
				gpu.FilterModeLinear, newMM)
			filter = sampler.Filter
			mm = sampler.MipmapMode
			aniso = false
		}
	}

	type result1D struct {
		shaderSubset textureEffectSpan
		shaderClamp  textureEffectSpan
		shaderMode   TextureEffectShaderMode
		hwWrap       gpu.WrapMode
	}

	resolve := func(size int32, wrap gpu.WrapMode, subsetSpan, domainSpan textureEffectSpan,
		linearInset float32,
	) result1D {
		var r result1D
		canDoModeInHW := canDoWrapInHW(size, wrap)
		if canDoModeInHW && size > 0 && subsetSpan.a <= 0 && subsetSpan.b >= float32(size) {
			r.shaderMode = ShaderModeNone
			r.hwWrap = wrap
			return r
		}

		r.shaderSubset = subsetSpan
		domainIsSafe := false
		if filter == gpu.FilterModeNearest {
			isubset := textureEffectSpan{
				a: float32(math.Floor(float64(subsetSpan.a))),
				b: float32(math.Ceil(float64(subsetSpan.b))),
			}
			if domainSpan.a > isubset.a && domainSpan.b < isubset.b {
				domainIsSafe = true
			}
			// This inset prevents sampling neighboring texels that could occur when texture coords fall exactly at
			// texel boundaries (depending on precision and GPU-specific snapping at the boundary).
			r.shaderClamp = isubset.makeInset(0.5 + textureEffectInsetEpsilon)
		} else {
			r.shaderClamp = subsetSpan.makeInset(linearInset + textureEffectInsetEpsilon)
			if r.shaderClamp.contains(domainSpan) {
				domainIsSafe = true
			}
		}
		if !alwaysUseShaderTileMode && domainIsSafe {
			// The domain of coords that will be used won't access texels outside of the subset, so the wrap mode
			// effectively doesn't matter. Use kClamp since it is always supported.
			r.shaderMode = ShaderModeNone
			r.hwWrap = gpu.WrapModeClamp
			r.shaderSubset = textureEffectSpan{}
			r.shaderClamp = textureEffectSpan{}
			return r
		}
		r.shaderMode = getTextureEffectShaderMode(wrap, filter, mm)
		r.hwWrap = gpu.WrapModeClamp
		return r
	}

	var x, y result1D
	if !aniso {
		infinite := textureEffectSpan{
			a: float32(math.Inf(-1)),
			b: float32(math.Inf(1)),
		}
		domainX, domainY := infinite, infinite
		if domain != nil {
			domainX = textureEffectSpan{a: domain.Left, b: domain.Right}
			domainY = textureEffectSpan{a: domain.Top, b: domain.Bottom}
		}
		x = resolve(dim.Width, sampler.WrapModeX, textureEffectSpan{a: subset.Left, b: subset.Right},
			domainX, linearFilterInset.X)
		y = resolve(dim.Height, sampler.WrapModeY, textureEffectSpan{a: subset.Top, b: subset.Bottom},
			domainY, linearFilterInset.Y)
	} else {
		x.hwWrap = sampler.WrapModeX
		y.hwWrap = sampler.WrapModeY
	}

	if aniso {
		out.hwSampler = gpu.SamplerStateAniso(x.hwWrap, y.hwWrap, sampler.MaxAniso,
			proxy.AsTextureProxy().Mipmapped())
	} else {
		out.hwSampler = gpu.MakeSamplerState(x.hwWrap, y.hwWrap, filter, mm)
	}
	out.shaderModes[0] = x.shaderMode
	out.shaderModes[1] = y.shaderMode
	out.shaderSubset = geom.Rect{
		Left: x.shaderSubset.a, Top: y.shaderSubset.a,
		Right: x.shaderSubset.b, Bottom: y.shaderSubset.b,
	}
	out.shaderClamp = geom.Rect{
		Left: x.shaderClamp.a, Top: y.shaderClamp.a,
		Right: x.shaderClamp.b, Bottom: y.shaderClamp.b,
	}
	out.border = border
	return out
}

func rectContains(outer, inner geom.Rect) bool {
	return outer.Left <= inner.Left && outer.Top <= inner.Top && outer.Right >= inner.Right &&
		outer.Bottom >= inner.Bottom
}

//////////////////////////////////////////////////////////////////////////////

// TextureEffect is the fragment processor that samples a texture with a given wrap/filter sampler, subset, and border.
type TextureEffect struct {
	view SurfaceProxyView
	FPBase
	border       [4]float32
	subset       geom.Rect
	clamp        geom.Rect
	samplerState gpu.SamplerState
	shaderModes  [2]TextureEffectShaderMode
}

// modulateForSamplerOptFlags returns the optimization flags a texture-sampling fragment processor can advertise given
// its alpha type and whether sampling can read outside the texture's bounds.
func modulateForSamplerOptFlags(alphaType gpu.AlphaType, samplingDecal bool) uint32 {
	if samplingDecal {
		return FPCompatibleWithCoverageAsAlpha
	}
	if alphaType == gpu.AlphaTypeOpaque {
		return FPCompatibleWithCoverageAsAlpha | FPPreservesOpaqueInput
	}
	return FPCompatibleWithCoverageAsAlpha
}

func newTextureEffect(view SurfaceProxyView, alphaType gpu.AlphaType, sampling *textureEffectSampling) *TextureEffect {
	te := &TextureEffect{
		view:         view,
		samplerState: sampling.hwSampler,
		border:       sampling.border,
		subset:       sampling.shaderSubset,
		clamp:        sampling.shaderClamp,
		shaderModes:  sampling.shaderModes,
	}
	te.initFP(TextureEffectClassID,
		modulateForSamplerOptFlags(alphaType, sampling.hasBorderAlpha()))
	// We always compare the range even when it isn't used, so assert canonical don't-care values.
	if te.shaderModes[0] == ShaderModeNone && (te.subset.Left != 0 || te.subset.Right != 0) {
		panic("non-canonical subset for ShaderModeNone")
	}
	if te.shaderModes[1] == ShaderModeNone && (te.subset.Top != 0 || te.subset.Bottom != 0) {
		panic("non-canonical subset for ShaderModeNone")
	}
	te.setUsesSampleCoordsDirectly()
	return te
}

// MakeTextureEffectSimple builds a TextureEffect that samples view with the given filter and mipmap mode, configured
// with clamp wrapping.
func MakeTextureEffectSimple(view SurfaceProxyView, alphaType gpu.AlphaType, matrix geom.Matrix, filter gpu.FilterMode, mm gpu.MipmapMode) FragmentProcessor {
	sampling := textureEffectSampling{hwSampler: gpu.SamplerStateFilter(filter, mm)}
	return MatrixEffectFP(matrix, newTextureEffect(view, alphaType, &sampling))
}

// MakeTextureEffect builds a TextureEffect that samples view with the given sampler state. caps determines support for
// hardware clamp-to-border, which is emulated in the shader when absent.
func MakeTextureEffect(view SurfaceProxyView, alphaType gpu.AlphaType, matrix geom.Matrix, sampler gpu.SamplerState, caps *Caps, border [4]float32) FragmentProcessor {
	sampling := makeTextureEffectSampling(view.Proxy(), sampler,
		geom.Rect{
			Right:  float32(view.Proxy().Dimensions().Width),
			Bottom: float32(view.Proxy().Dimensions().Height),
		},
		nil, border, false, &caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	return MatrixEffectFP(matrix, newTextureEffect(view, alphaType, &sampling))
}

// MakeTextureEffectSubset builds a TextureEffect that samples a subset of the texture, applying the wrap modes to the
// subset in the shader.
func MakeTextureEffectSubset(view SurfaceProxyView, alphaType gpu.AlphaType, matrix geom.Matrix, sampler gpu.SamplerState, subset geom.Rect, caps *Caps, border [4]float32, alwaysUseShaderTileMode bool) FragmentProcessor {
	sampling := makeTextureEffectSampling(view.Proxy(), sampler, subset, nil, border,
		alwaysUseShaderTileMode, &caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	return MatrixEffectFP(matrix, newTextureEffect(view, alphaType, &sampling))
}

// MakeTextureEffectSubsetWithDomain builds a TextureEffect like MakeTextureEffectSubset, but the caller promises that
// the pixel coords sampled fall within domain, allowing better wrap-mode implementations.
func MakeTextureEffectSubsetWithDomain(view SurfaceProxyView, alphaType gpu.AlphaType, matrix geom.Matrix, sampler gpu.SamplerState, subset, domain geom.Rect, caps *Caps, border [4]float32) FragmentProcessor {
	sampling := makeTextureEffectSampling(view.Proxy(), sampler, subset, &domain, border,
		false, &caps.Caps, geom.Point{X: 0.5, Y: 0.5})
	return MatrixEffectFP(matrix, newTextureEffect(view, alphaType, &sampling))
}

// SamplerState returns the hardware sampler state used by this effect.
func (t *TextureEffect) SamplerState() gpu.SamplerState { return t.samplerState }

// View returns the surface proxy view this effect samples from.
func (t *TextureEffect) View() *SurfaceProxyView { return &t.view }

// Texture returns the texture this effect samples from.
func (t *TextureEffect) Texture() *Texture { return t.view.Proxy().PeekTexture() }

// coordAdjustmentMatrix returns the matrix concatenated by the wrapping matrix effect to handle y-flip and coord
// normalization. It is computed at setData time because the texture size is not always known when the effect is made
// (fully-lazy proxies).
func (t *TextureEffect) coordAdjustmentMatrix() geom.Matrix {
	m := geom.IdentityMatrix()
	d := t.Texture().Surface().Dimensions()
	if t.matrixEffectShouldNormalize() {
		if t.view.Origin() == gpu.OriginBottomLeft {
			m.SetScaleTranslate(1/float32(d.Width), -1/float32(d.Height), 0, 1)
		} else {
			m.SetScale(1/float32(d.Width), 1/float32(d.Height))
		}
	} else if t.view.Origin() == gpu.OriginBottomLeft {
		m.SetScaleTranslate(1, -1, 0, float32(d.Height))
	}
	return m
}

// matrixEffectShouldNormalize reports whether the wrapping matrix effect should pre-normalize sample coordinates before
// this effect runs.
func (t *TextureEffect) matrixEffectShouldNormalize() bool {
	return t.view.Proxy().TextureType() != gpu.TextureTypeRectangle &&
		!shaderModeRequiresUnormCoord(t.shaderModes[0]) &&
		!shaderModeRequiresUnormCoord(t.shaderModes[1])
}

func (t *TextureEffect) hasClampToBorderShaderMode() bool {
	return shaderModeIsClampToBorder(t.shaderModes[0]) || shaderModeIsClampToBorder(t.shaderModes[1])
}

// Name implements Processor.
func (t *TextureEffect) Name() string { return "TextureEffect" }

// Clone implements FragmentProcessor.
func (t *TextureEffect) Clone() FragmentProcessor {
	clone := &TextureEffect{
		view:         t.view,
		samplerState: t.samplerState,
		border:       t.border,
		subset:       t.subset,
		clamp:        t.clamp,
		shaderModes:  t.shaderModes,
	}
	clone.initFP(TextureEffectClassID, t.optimizationFlags())
	clone.setUsesSampleCoordsDirectly()
	return clone
}

func (t *TextureEffect) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBits(8, uint32(t.shaderModes[0]), "shaderMode0")
	b.AddBits(8, uint32(t.shaderModes[1]), "shaderMode1")
}

func (t *TextureEffect) onIsEqual(other FragmentProcessor) bool {
	that := other.(*TextureEffect)
	if !t.view.Equal(&that.view) {
		return false
	}
	if t.samplerState != that.samplerState {
		return false
	}
	if t.shaderModes != that.shaderModes {
		return false
	}
	if t.subset != that.subset {
		return false
	}
	if t.hasClampToBorderShaderMode() && t.border != that.border {
		return false
	}
	return true
}

func (t *TextureEffect) constantOutputForConstantInput(colorcore.PMColor4f) colorcore.PMColor4f {
	panic("TextureEffect does not advertise constant output")
}

func (t *TextureEffect) onMakeProgramImpl() FPProgramImpl { return &textureEffectImpl{} }

//////////////////////////////////////////////////////////////////////////////

// textureEffectImpl is the program implementation (shader emission) for TextureEffect.
type textureEffectImpl struct {
	FPImplBase
	subsetUni     UniformHandle
	clampUni      UniformHandle
	idimsUni      UniformHandle
	borderUni     UniformHandle
	samplerHandle SamplerHandle
}

// setSamplerHandle is called by the program builder when it emits the sampler uniforms for the pipeline's texture
// effects.
func (i *textureEffectImpl) setSamplerHandle(handle SamplerHandle) { i.samplerHandle = handle }

func (i *textureEffectImpl) EmitCode(args *FPEmitArgs) {
	te := args.FP.(*TextureEffect)
	fb := args.FragBuilder

	if te.shaderModes[0] == ShaderModeNone && te.shaderModes[1] == ShaderModeNone {
		fb.CodeAppendf("return %s;",
			fb.AppendTextureLookup(args.UniformHandler, i.samplerHandle, args.SampleCoord))
		return
	}
	// The basic flow of the various ShaderModes is implemented in a series of steps below; not all the steps apply to
	// all the modes, and only the ones necessary for the given x/y modes are emitted.

	// Convert possible projective texture coordinates into non-homogeneous vec2.
	fb.CodeAppendf("vec2 inCoord = %s;", args.SampleCoord)

	m := te.shaderModes

	borderName := ""
	if te.hasClampToBorderShaderMode() {
		i.borderUni, borderName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeHalf4, "border")
	}
	modeUsesSubset := func(mode TextureEffectShaderMode) bool {
		return mode != ShaderModeNone && mode != ShaderModeClamp
	}
	modeUsesClamp := func(mode TextureEffectShaderMode) bool {
		return mode != ShaderModeNone && mode != ShaderModeClampToBorderNearest
	}

	useSubset := [2]bool{modeUsesSubset(m[0]), modeUsesSubset(m[1])}
	useClamp := [2]bool{modeUsesClamp(m[0]), modeUsesClamp(m[1])}

	subsetName := ""
	if useSubset[0] || useSubset[1] {
		i.subsetUni, subsetName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeFloat4, "subset")
	}

	clampName := ""
	if useClamp[0] || useClamp[1] {
		i.clampUni, clampName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeFloat4, "clamp")
	}

	unormCoordsRequired := shaderModeRequiresUnormCoord(m[0]) || shaderModeRequiresUnormCoord(m[1])
	// We should not pre-normalize the input coords with the matrix effect if we're going to operate on unnormalized
	// coords and then normalize after the shader mode.
	if unormCoordsRequired && te.matrixEffectShouldNormalize() {
		panic("unnormalized-coord shader mode with normalizing matrix effect")
	}
	sampleCoordsMustBeNormalized := te.view.Proxy().TextureType() != gpu.TextureTypeRectangle

	idims := ""
	if unormCoordsRequired && sampleCoordsMustBeNormalized {
		i.idimsUni, idims = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeFloat2, "idims")
	}

	// Generates a string to read at a coordinate, normalizing coords if necessary.
	read := func(coord string) string {
		normCoord := coord
		if idims != "" {
			normCoord = fmt.Sprintf("(%s) * %s", coord, idims)
		}
		return fb.AppendTextureLookup(args.UniformHandler, i.samplerHandle, normCoord)
	}

	// Implements coord wrapping for kRepeat and kMirrorRepeat.
	subsetCoord := func(mode TextureEffectShaderMode, coordSwizzle, subsetStartSwizzle,
		subsetStopSwizzle, extraCoord, coordWeight string,
	) {
		switch mode {
		case ShaderModeNone, ShaderModeClampToBorderNearest, ShaderModeClampToBorderFilter,
			ShaderModeClamp:
			// These modes either don't use the subset rect or don't need to map the coords to be within the subset.
			fb.CodeAppendf("subsetCoord.%s = inCoord.%s;", coordSwizzle, coordSwizzle)
		case ShaderModeRepeatNearestNone, ShaderModeRepeatLinearNone:
			fb.CodeAppendf("subsetCoord.%s = mod(inCoord.%s - %s.%s, %s.%s - %s.%s) + %s.%s;",
				coordSwizzle, coordSwizzle, subsetName, subsetStartSwizzle, subsetName,
				subsetStopSwizzle, subsetName, subsetStartSwizzle, subsetName, subsetStartSwizzle)
		case ShaderModeRepeatNearestMipmap, ShaderModeRepeatLinearMipmap:
			// Generate two sets of texture coords that are both "moving" at the same speed (if not direction) as
			// inCoords, by using two out-of-phase mirror repeat coords. Sample with both coords and select the
			// upward-sloping one's read via a weight that transitions near the reflection point (a saw-tooth,
			// phase-shifted, vertically translated, clamped to 0..1).
			fb.CodeAppend("{")
			fb.CodeAppendf("float w = %s.%s - %s.%s;", subsetName, subsetStopSwizzle, subsetName,
				subsetStartSwizzle)
			fb.CodeAppendf("float w2 = 2.0 * w;")
			fb.CodeAppendf("float d = inCoord.%s - %s.%s;", coordSwizzle, subsetName,
				subsetStartSwizzle)
			fb.CodeAppend("float m = mod(d, w2);")
			fb.CodeAppend("float o = mix(m, w2 - m, step(w, m));")
			fb.CodeAppendf("subsetCoord.%s = o + %s.%s;", coordSwizzle, subsetName,
				subsetStartSwizzle)
			fb.CodeAppendf("%s = w - o + %s.%s;", extraCoord, subsetName, subsetStartSwizzle)
			// coordWeight is used as the third param of mix() to blend between a sample taken using subsetCoord and a
			// sample at extraCoord.
			fb.CodeAppend("float hw = w/2.0;")
			fb.CodeAppend("float n = mod(d - hw, w2);")
			fb.CodeAppendf("%s = clamp(mix(n, w2 - n, step(w, n)) - hw + 0.5, 0.0, 1.0);",
				coordWeight)
			fb.CodeAppend("}")
		case ShaderModeMirrorRepeat:
			fb.CodeAppend("{")
			fb.CodeAppendf("float w = %s.%s - %s.%s;", subsetName, subsetStopSwizzle, subsetName,
				subsetStartSwizzle)
			fb.CodeAppendf("float w2 = 2.0 * w;")
			fb.CodeAppendf("float m = mod(inCoord.%s - %s.%s, w2);", coordSwizzle, subsetName,
				subsetStartSwizzle)
			fb.CodeAppendf("subsetCoord.%s = mix(m, w2 - m, step(w, m)) + %s.%s;", coordSwizzle,
				subsetName, subsetStartSwizzle)
			fb.CodeAppend("}")
		}
	}

	clampCoord := func(clamp bool, coordSwizzle, clampStartSwizzle, clampStopSwizzle string) {
		if clamp {
			fb.CodeAppendf("clampedCoord%s = clamp(subsetCoord%s, %s%s, %s%s);", coordSwizzle,
				coordSwizzle, clampName, clampStartSwizzle, clampName, clampStopSwizzle)
		} else {
			fb.CodeAppendf("clampedCoord%s = subsetCoord%s;", coordSwizzle, coordSwizzle)
		}
	}

	// Insert vars for extra coords and blending weights for repeat + mip map.
	extraRepeatCoordX, repeatCoordWeightX := "", ""
	extraRepeatCoordY, repeatCoordWeightY := "", ""

	mipmapRepeatX := m[0] == ShaderModeRepeatNearestMipmap || m[0] == ShaderModeRepeatLinearMipmap
	mipmapRepeatY := m[1] == ShaderModeRepeatNearestMipmap || m[1] == ShaderModeRepeatLinearMipmap

	if mipmapRepeatX || mipmapRepeatY {
		fb.CodeAppend("vec2 extraRepeatCoord;")
	}
	if mipmapRepeatX {
		fb.CodeAppend("float repeatCoordWeightX;")
		extraRepeatCoordX = "extraRepeatCoord.x"
		repeatCoordWeightX = "repeatCoordWeightX"
	}
	if mipmapRepeatY {
		fb.CodeAppend("float repeatCoordWeightY;")
		extraRepeatCoordY = "extraRepeatCoord.y"
		repeatCoordWeightY = "repeatCoordWeightY"
	}

	// Apply subset rect and clamp rect to coords.
	fb.CodeAppend("vec2 subsetCoord;")
	subsetCoord(m[0], "x", "x", "z", extraRepeatCoordX, repeatCoordWeightX)
	subsetCoord(m[1], "y", "y", "w", extraRepeatCoordY, repeatCoordWeightY)
	fb.CodeAppend("vec2 clampedCoord;")
	if useClamp[0] == useClamp[1] {
		clampCoord(useClamp[0], "", ".xy", ".zw")
	} else {
		clampCoord(useClamp[0], ".x", ".x", ".z")
		clampCoord(useClamp[1], ".y", ".y", ".w")
	}
	// Additional clamping for the extra coords for kRepeat with mip maps.
	switch {
	case mipmapRepeatX && mipmapRepeatY:
		fb.CodeAppendf("extraRepeatCoord = clamp(extraRepeatCoord, %s.xy, %s.zw);", clampName,
			clampName)
	case mipmapRepeatX:
		fb.CodeAppendf("extraRepeatCoord.x = clamp(extraRepeatCoord.x, %s.x, %s.z);", clampName,
			clampName)
	case mipmapRepeatY:
		fb.CodeAppendf("extraRepeatCoord.y = clamp(extraRepeatCoord.y, %s.y, %s.w);", clampName,
			clampName)
	}

	// Do the 2 or 4 texture reads for kRepeatMipMap and then apply the weight(s) to blend between them. If neither
	// direction is repeat or not using mip maps do a single read at clampedCoord.
	switch {
	case mipmapRepeatX && mipmapRepeatY:
		fb.CodeAppendf(
			"vec4 textureColor ="+
				"   mix(mix(%s, %s, repeatCoordWeightX),"+
				"       mix(%s, %s, repeatCoordWeightX),"+
				"       repeatCoordWeightY);",
			read("clampedCoord"),
			read("vec2(extraRepeatCoord.x, clampedCoord.y)"),
			read("vec2(clampedCoord.x, extraRepeatCoord.y)"),
			read("vec2(extraRepeatCoord.x, extraRepeatCoord.y)"),
		)
	case mipmapRepeatX:
		fb.CodeAppendf("vec4 textureColor = mix(%s, %s, repeatCoordWeightX);",
			read("clampedCoord"), read("vec2(extraRepeatCoord.x, clampedCoord.y)"))
	case mipmapRepeatY:
		fb.CodeAppendf("vec4 textureColor = mix(%s, %s, repeatCoordWeightY);",
			read("clampedCoord"), read("vec2(clampedCoord.x, extraRepeatCoord.y)"))
	default:
		fb.CodeAppendf("vec4 textureColor = %s;", read("clampedCoord"))
	}

	// Strings for extra texture reads used only in kRepeatLinear.
	repeatLinearReadX := ""
	repeatLinearReadY := ""

	// Calculate the amount the coord moved for clamping. This is used to implement shader-based filtering for
	// kClampToBorder and kRepeat.
	repeatLinearFilterX := m[0] == ShaderModeRepeatLinearNone || m[0] == ShaderModeRepeatLinearMipmap
	repeatLinearFilterY := m[1] == ShaderModeRepeatLinearNone || m[1] == ShaderModeRepeatLinearMipmap
	if repeatLinearFilterX || m[0] == ShaderModeClampToBorderFilter {
		fb.CodeAppend("float errX = subsetCoord.x - clampedCoord.x;")
		if repeatLinearFilterX {
			fb.CodeAppendf("float repeatCoordX = errX > 0.0 ? %s.x : %s.z;", clampName, clampName)
			repeatLinearReadX = read("vec2(repeatCoordX, clampedCoord.y)")
		}
	}
	if repeatLinearFilterY || m[1] == ShaderModeClampToBorderFilter {
		fb.CodeAppend("float errY = subsetCoord.y - clampedCoord.y;")
		if repeatLinearFilterY {
			fb.CodeAppendf("float repeatCoordY = errY > 0.0 ? %s.y : %s.w;", clampName, clampName)
			repeatLinearReadY = read("vec2(clampedCoord.x, repeatCoordY)")
		}
	}

	// Add logic for kRepeat + linear filter: 1 or 3 more texture reads depending on whether both modes are kRepeat and
	// whether we're near a single subset edge or a corner, then blend the reads using the err values calculated above.
	ifStr := "if"
	if repeatLinearFilterX && repeatLinearFilterY {
		repeatLinearReadXY := read("vec2(repeatCoordX, repeatCoordY)")
		fb.CodeAppendf(
			"if (errX != 0.0 && errY != 0.0) {"+
				"    errX = abs(errX);"+
				"    textureColor = mix(mix(textureColor, %s, errX),"+
				"                       mix(%s, %s, errX),"+
				"                       abs(errY));"+
				"}",
			repeatLinearReadX, repeatLinearReadY, repeatLinearReadXY,
		)
		ifStr = "else if"
	}
	if repeatLinearFilterX {
		fb.CodeAppendf(
			"%s (errX != 0.0) {"+
				"    textureColor = mix(textureColor, %s, abs(errX));"+
				"}",
			ifStr, repeatLinearReadX,
		)
	}
	if repeatLinearFilterY {
		fb.CodeAppendf(
			"%s (errY != 0.0) {"+
				"    textureColor = mix(textureColor, %s, abs(errY));"+
				"}",
			ifStr, repeatLinearReadY,
		)
	}

	// Do soft edge shader filtering against border color for kClampToBorderFilter using the err values calculated
	// above.
	if m[0] == ShaderModeClampToBorderFilter {
		fb.CodeAppendf("textureColor = mix(textureColor, %s, min(abs(errX), 1.0));", borderName)
	}
	if m[1] == ShaderModeClampToBorderFilter {
		fb.CodeAppendf("textureColor = mix(textureColor, %s, min(abs(errY), 1.0));", borderName)
	}

	// Do hard-edge shader transition to border color for kClampToBorderNearest at the subset boundaries. Snap the input
	// coordinates to nearest neighbor (with an epsilon) before comparing to the subset rect, to avoid GPU interpolation
	// errors.
	if m[0] == ShaderModeClampToBorderNearest {
		fb.CodeAppendf(
			"float snappedX = floor(inCoord.x + 0.001) + 0.5;"+
				"if (snappedX < %s.x || snappedX > %s.z) {"+
				"    textureColor = %s;"+
				"}",
			subsetName, subsetName, borderName,
		)
	}
	if m[1] == ShaderModeClampToBorderNearest {
		fb.CodeAppendf(
			"float snappedY = floor(inCoord.y + 0.001) + 0.5;"+
				"if (snappedY < %s.y || snappedY > %s.w) {"+
				"    textureColor = %s;"+
				"}",
			subsetName, subsetName, borderName,
		)
	}
	fb.CodeAppendf("return textureColor;")
}

func (i *textureEffectImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	te := fp.(*TextureEffect)

	dims := te.Texture().Surface().Dimensions()
	w := float32(dims.Width)
	h := float32(dims.Height)
	textureType := te.Texture().TextureType()

	idims := [2]float32{1 / w, 1 / h}

	if i.idimsUni.IsValid() {
		pdman.Set2fv(i.idimsUni, 1, idims[:])
		if textureType == gpu.TextureTypeRectangle {
			panic("idims uniform with a rectangle texture")
		}
	}

	pushRect := func(rect [4]float32, uni UniformHandle) {
		if te.view.Origin() == gpu.OriginBottomLeft {
			rect[1] = h - rect[1]
			rect[3] = h - rect[3]
			rect[1], rect[3] = rect[3], rect[1]
		}
		if !i.idimsUni.IsValid() && textureType != gpu.TextureTypeRectangle {
			rect[0] *= idims[0]
			rect[2] *= idims[0]
			rect[1] *= idims[1]
			rect[3] *= idims[1]
		}
		pdman.Set4fv(uni, 1, rect[:])
	}

	if i.subsetUni.IsValid() {
		pushRect([4]float32{te.subset.Left, te.subset.Top, te.subset.Right, te.subset.Bottom},
			i.subsetUni)
	}
	if i.clampUni.IsValid() {
		pushRect([4]float32{te.clamp.Left, te.clamp.Top, te.clamp.Right, te.clamp.Bottom},
			i.clampUni)
	}
	if i.borderUni.IsValid() {
		pdman.Set4fv(i.borderUni, 1, te.border[:])
	}
}
