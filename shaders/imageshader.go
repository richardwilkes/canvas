// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ImageShader: the CPU raster-pipeline image shader. Trims: no mipmapped sampling — the CPU leg always samples the base
// level, degrading mipmap requests to the requested filter (mipmaps would be built on demand — recorded as a
// divergence); subset and raw shader variants are unreachable through this package's public surface (subset shaders
// exist only for special images, raw ones have no entry point), so raw is always false.

package shaders

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
)

// ImageShader samples an image, applying tiling and filtering. Its image is the polymorphic imagecore.DrawableImage: a
// CPU-backed imagecore.Image or a texture-backed gpu/gl.TextureImage. The GPU fragment-processor lane
// (makeImageShaderFP) samples a same-context texture image from its own view with no readback; the CPU raster legs
// resolve pixels through cpuImage (MakeNonTextureImage — the identity for a raster image, a readback for a texture
// image).
type ImageShader struct {
	img               imagecore.DrawableImage
	sampling          SamplingOptions
	tileX, tileY      TileMode
	raw               bool
	clampAsIfUnpremul bool
}

// optimizeTile reports that mirror/repeat on a 1px axis is equivalent to clamping, while decal still transitions to
// transparent black.
func optimizeTile(tm TileMode, dimension int32) TileMode {
	if tm != TileDecal && dimension == 1 {
		return TileClamp
	}
	return tm
}

// NewImage builds an image shader (optionally wrapped in a local matrix) over a CPU-backed image. Returns nil for a nil
// image or invalid cubic parameters. The internal callers (filtercore, pdf, font) build shaders over raster images and
// use this concrete-typed entry so its nil check is on the pointer, not an interface; the general entry point goes
// through NewImageDrawable so a texture-backed image can draw GPU-natively.
func NewImage(img *imagecore.Image, tmx, tmy TileMode, sampling SamplingOptions, localMatrix *geom.Matrix) Shader {
	if img == nil {
		return nil
	}
	return newImageShader(img, tmx, tmy, sampling, localMatrix)
}

// NewImageDrawable builds an image shader over the polymorphic DrawableImage: img may be CPU-backed (imagecore.Image)
// or texture-backed (gpu/gl.TextureImage). A same-context texture image draws from its own view with no readback (the
// GPU fragment-processor lane); the CPU raster leg resolves pixels through cpuImage. Returns nil for a nil image (nil
// interface) or invalid cubic parameters.
func NewImageDrawable(img imagecore.DrawableImage, tmx, tmy TileMode, sampling SamplingOptions, localMatrix *geom.Matrix) Shader {
	if img == nil {
		return nil
	}
	return newImageShader(img, tmx, tmy, sampling, localMatrix)
}

// newImageShader builds the ImageShader (optionally wrapped in a local matrix) shared by NewImage and NewImageDrawable.
// img is non-nil.
func newImageShader(img imagecore.DrawableImage, tmx, tmy TileMode, sampling SamplingOptions, localMatrix *geom.Matrix) Shader {
	if sampling.UseCubic {
		isUnit := func(v float32) bool { return v >= 0 && v <= 1 }
		if !isUnit(sampling.Cubic.B) || !isUnit(sampling.Cubic.C) {
			return nil
		}
	}
	var s Shader = &ImageShader{
		img:      img,
		sampling: sampling,
		tileX:    optimizeTile(tmx, img.Width()),
		tileY:    optimizeTile(tmy, img.Height()),
	}
	if localMatrix != nil && !localMatrix.IsIdentity() {
		s = NewWithLocalMatrix(s, *localMatrix)
	}
	return s
}

// SetImage re-initializes s to sample img with the given tile modes and sampling, returning false for the invalid-cubic
// case where NewImage returns nil. It lets a caller reuse one ImageShader value across draws (the CPU image-draw
// scratch) instead of allocating a fresh *ImageShader per draw; the caller applies the local-matrix wrap
// (NewWithLocalMatrix) around the reused base shader exactly as NewImage does. Only for transient scratch shaders
// consumed synchronously within a draw.
func SetImage(s *ImageShader, img *imagecore.Image, tmx, tmy TileMode, sampling SamplingOptions) bool {
	if sampling.UseCubic {
		isUnit := func(v float32) bool { return v >= 0 && v <= 1 }
		if !isUnit(sampling.Cubic.B) || !isUnit(sampling.Cubic.C) {
			return false
		}
	}
	s.img = img
	s.sampling = sampling
	s.tileX = optimizeTile(tmx, img.Width())
	s.tileY = optimizeTile(tmy, img.Height())
	s.raw = false
	s.clampAsIfUnpremul = false
	return true
}

// DrawableImage returns the shader's polymorphic image (what the GPU fragment-processor conversion samples). The GPU
// image-shader FP lane (gpu/gl.makeImageShaderFP) hands it to drawableAsView so a same-context texture image is sampled
// from its own view without a readback.
func (s *ImageShader) DrawableImage() imagecore.DrawableImage { return s.img }

// cpuImage resolves the shader's image to CPU-backed pixels — the identity for a raster image (free), a GPU→CPU
// readback for a texture-backed one. The CPU raster legs (appendStages, the legacy sampler, the PDF AsImage
// introspection) call it; the GPU fragment-processor lane uses DrawableImage instead so a same-context texture never
// reads back. Returns nil on a failed readback.
func (s *ImageShader) cpuImage() *imagecore.Image { return s.img.MakeNonTextureImage() }

// TileModes returns the (optimized) tile modes.
func (s *ImageShader) TileModes() (tmx, tmy TileMode) { return s.tileX, s.tileY }

// AsImage reports whether s (possibly behind a local-matrix wrapper) is an image shader. On success it returns the
// image, its tile modes, and the shader's local matrix: an image shader's own texture matrix is the identity, so a
// wrapping local-matrix shader's matrix passes through unchanged. The PDF backend consumes this to build a tiled image
// pattern. Kept beside AsGradient as a small introspection surface separate from the raster-pipeline evaluation stages.
func AsImage(s Shader) (img *imagecore.Image, tmx, tmy TileMode, localMatrix geom.Matrix, ok bool) {
	localMatrix = geom.IdentityMatrix()
	if lm, isLM := s.(*LocalMatrixShader); isLM {
		localMatrix = lm.LocalMatrix()
		s = lm.Wrapped()
	}
	if is, isImg := s.(*ImageShader); isImg {
		// PDF (the only AsImage consumer) needs CPU pixels for the tiled image pattern; a texture-backed image reads
		// back here (rare). A failed readback reports not-an-image rather than a nil image.
		cpu := is.cpuImage()
		if cpu == nil {
			return nil, 0, 0, geom.IdentityMatrix(), false
		}
		tmx, tmy = is.TileModes()
		return cpu, tmx, tmy, localMatrix, true
	}
	return nil, 0, 0, geom.IdentityMatrix(), false
}

// Sampling returns the sampling options.
func (s *ImageShader) Sampling() SamplingOptions { return s.sampling }

// IsOpaque implements Shader. The image's opacity is its alpha type, read off the polymorphic image without a readback.
func (s *ImageShader) IsOpaque() bool {
	return s.img.AlphaType() == imagecore.AlphaTypeOpaque && s.tileX != TileDecal && s.tileY != TileDecal
}

// IsConstant implements Shader.
func (s *ImageShader) IsConstant() bool { return false }

// tweakSampling reports that, with an integer-translate total matrix, bilerp is equivalent to nearest.
func tweakSampling(s SamplingOptions, m *geom.Matrix) SamplingOptions {
	filter := s.Filter
	if filter == FilterLinear && m.Type() <= geom.TypeTranslate {
		tx := m.Get(geom.MTransX)
		ty := m.Get(geom.MTransY)
		if tx == float32(int32(tx)) && ty == float32(int32(ty)) {
			filter = FilterNearest
		}
	}
	return SamplingOptions{Filter: filter, Mipmap: s.Mipmap}
}

// appendStages appends the sampling stages for the base mip level.
func (s *ImageShader) appendStages(p *Pipeline, mRec MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	sampling := s.sampling
	if sampling.isAniso() {
		sampling = anisoFallback(false) // CPU images carry no mips in this port
	}

	// If the total matrix isn't valid (a runtime-effect kernel samples this shader at computed coordinates) the base
	// MIP level is always accessed and sampling cannot be tweaked.
	baseInv := geom.IdentityMatrix()
	if mRec.totalMatrixValid() {
		var ok bool
		baseInv, ok = mRec.totalInverse()
		if !ok {
			return false
		}
	}

	// Decode/peek the base level. Mipmap modes degrade to the base level; the level matrix is identity. A
	// texture-backed image reads back here (the CPU raster leg always needs pixels); a raster image resolves to itself
	// for free.
	cpu := s.cpuImage()
	if cpu == nil {
		return false
	}
	px := cpu.PeekPixels(imagecore.CachingAllow)
	if px == nil {
		return false
	}

	if !sampling.UseCubic && mRec.totalMatrixValid() {
		sampling = tweakSampling(sampling, &baseInv)
	}

	identity := geom.IdentityMatrix()
	mRec, applyOK := mRec.apply(p, &identity)
	if !applyOK {
		return false
	}
	_ = mRec

	// The gather/sampler/decal contexts come from the pipeline's pooled free lists (retained across recycled compiles;
	// the gatherCtx's px is dropped on recycle so an idle pooled pipeline does not pin the source pixels — see
	// RecyclePipeline) rather than a fresh allocation per compile.
	g := p.nextGatherCtx()
	g.setPixels(px)
	g.width = float32(px.Info.Width)
	g.height = float32(px.Info.Height)
	if sampling.UseCubic {
		g.weights = cubicResamplerWeights(sampling.Cubic.B, sampling.Cubic.C)
	}
	if !sampling.UseCubic && sampling.Filter == FilterNearest {
		g.roundDownAtInteger = true
	}
	g.setTileLimits()
	var decal *decalTileCtx
	if s.tileX == TileDecal || s.tileY == TileDecal {
		decal = p.nextDecalCtx()
		decal.limitX = g.width
		decal.limitY = g.height
		if g.roundDownAtInteger {
			decal.inclusiveEdgeX = g.width
			decal.inclusiveEdgeY = g.height
		}
	}

	appendMisc := func() bool {
		at := px.Info.AlphaType

		// Color for alpha-only images comes from the paint (already in dst space).
		if px.Info.ColorType == imagecore.ColorTypeAlpha8 && !s.raw {
			pc := p.paintColor
			p.appendSetRGB(g, pc.R, pc.G, pc.B)
			at = imagecore.AlphaTypeUnpremul
		}

		// Bicubic filtering naturally produces out of range values on both sides of [0,1].
		if sampling.UseCubic {
			if at == imagecore.AlphaTypeUnpremul || s.clampAsIfUnpremul {
				p.AppendClamp01()
			} else {
				p.AppendClampGamut()
			}
		}

		// The source and destination color spaces share a transfer function, so the color-space transform reduces to
		// just its alpha ops: unpremul sources premultiply, premul/ opaque sources pass through.
		if !s.raw && at == imagecore.AlphaTypeUnpremul {
			p.AppendPremul()
		}
		return true
	}

	// Fused fast paths (base level).
	ct := px.Info.ColorType
	is8888 := ct == imagecore.ColorTypeRGBA8888 || ct == imagecore.ColorTypeBGRA8888
	if is8888 && !sampling.UseCubic && sampling.Filter == FilterLinear &&
		sampling.Mipmap != MipmapLinear && s.tileX == TileClamp && s.tileY == TileClamp {
		p.appendBilerpClamp8888(g)
		return appendMisc()
	}
	if is8888 && sampling.UseCubic && s.tileX == TileClamp && s.tileY == TileClamp {
		p.appendBicubicClamp8888(g)
		return appendMisc()
	}

	sampler := p.nextSamplerCtx()
	sample := func(scaleX, scaleY float32, cubic bool) {
		if cubic {
			p.appendBicubicAxis(sampler, true, scaleX)
			p.appendBicubicAxis(sampler, false, scaleY)
		} else {
			p.appendBilinearAxis(sampler, true, scaleX)
			p.appendBilinearAxis(sampler, false, scaleY)
		}
		p.appendGather(g, s.tileX, s.tileY, decal)
		p.appendAccumulate(sampler)
	}

	switch {
	case sampling.UseCubic:
		sampler.weights = cubicResamplerWeights(sampling.Cubic.B, sampling.Cubic.C)
		p.appendSaveXYInit(sampler, true)
		for _, sy := range [4]float32{-3, -1, +1, +3} {
			for _, sx := range [4]float32{-3, -1, +1, +3} {
				sample(sx, sy, true)
			}
		}
		p.appendMoveDstSrc()
	case sampling.Filter == FilterLinear:
		p.appendSaveXYInit(sampler, false)
		sample(-1, -1, false)
		sample(+1, -1, false)
		sample(-1, +1, false)
		sample(+1, +1, false)
		p.appendMoveDstSrc()
	default:
		p.appendGather(g, s.tileX, s.tileY, decal)
	}
	return appendMisc()
}
