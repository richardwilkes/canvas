// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package colorfilter provides the color filters that are publicly reachable: the 4x5 matrix filter, the blend-mode
// filter, compose, lighting, luma, and high contrast (which runs in a linear-unpremul working format). The former
// runtime-effect filters are implemented as their shader math directly (no shader compiler); the working-format
// sandwich reduces to unpremul/linearize … encode/premul for sRGB-only content. Constructors return shaders.ColorFilter
// descriptors that append raster-pipeline stages.

package colorfilter

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

///////////////////////////////////////////////////////////////////////////////
// matrix (RGBA domain with clamping — the only publicly reachable form)

// matrixFilter is the 4x5 color-matrix filter for the RGBA domain with clamping.
type matrixFilter struct {
	mat            [20]float32
	alphaUnchanged bool
}

// matrixIsAlphaUnchanged reports whether the matrix's alpha row is (nearly) the identity.
func matrixIsAlphaUnchanged(m *[20]float32) bool {
	srcA := m[15:]
	return geom.ScalarNearlyZero(srcA[0]) && geom.ScalarNearlyZero(srcA[1]) &&
		geom.ScalarNearlyZero(srcA[2]) && geom.ScalarNearlyEqual(srcA[3], 1) &&
		geom.ScalarNearlyZero(srcA[4])
}

// NewMatrix returns a color filter applying a 4x5 row-major matrix to unpremultiplied float RGBA, clamping the result.
// Returns nil when the matrix is not finite.
func NewMatrix(array *[20]float32) shaders.ColorFilter {
	for _, v := range array {
		if !geom.IsFinite(v) {
			return nil
		}
	}
	return &matrixFilter{mat: *array, alphaUnchanged: matrixIsAlphaUnchanged(array)}
}

// Matrix returns the 4x5 matrix — the descriptor the GPU side consumes.
func (f *matrixFilter) Matrix() [20]float32 { return f.mat }

// IsAlphaUnchanged implements shaders.ColorFilter.
func (f *matrixFilter) IsAlphaUnchanged() bool { return f.alphaUnchanged }

// AppendStages implements shaders.ColorFilter: unpremul, apply the matrix, clamp, re-premul.
func (f *matrixFilter) AppendStages(p *shaders.Pipeline, shaderIsOpaque bool) bool {
	willStayOpaque := shaderIsOpaque && f.alphaUnchanged
	if !shaderIsOpaque {
		p.AppendUnpremul()
	}
	p.AppendMatrix4x5(&f.mat)
	p.AppendClamp01()
	if !willStayOpaque {
		p.AppendPremul()
	}
	return true
}

///////////////////////////////////////////////////////////////////////////////
// blend mode

// blendFilter blends a fixed color over the filtered color; the color is stored unpremultiplied in sRGB.
type blendFilter struct {
	color colorcore.Color4f
	mode  raster.BlendMode
}

// NewBlend returns the blend-mode color filter: it collapses modes against the color's alpha, and returns nil for the
// combinations that are no-ops.
func NewBlend(c colorcore.Color, mode raster.BlendMode) shaders.ColorFilter {
	if mode > raster.BlendLuminosity {
		return nil
	}
	srgb := colorcore.Color4fFromColor(c) // 8-bit colors are already alpha-pinned
	alpha := srgb.A

	// Next collapse some modes if possible
	switch mode {
	case raster.BlendClear:
		srgb = colorcore.Color4f{}
		mode = raster.BlendSrc
	case raster.BlendSrcOver:
		switch alpha {
		case 0:
			mode = raster.BlendDst
		case 1:
			mode = raster.BlendSrc
		}
		// else just stay srcover
	}

	// Finally weed out combinations that are noops, and just return null
	if mode == raster.BlendDst ||
		(alpha == 0 && (mode == raster.BlendSrcOver ||
			mode == raster.BlendDstOver ||
			mode == raster.BlendDstOut ||
			mode == raster.BlendSrcATop ||
			mode == raster.BlendXor ||
			mode == raster.BlendDarken)) ||
		(alpha == 1 && mode == raster.BlendDstIn) {
		return nil
	}
	return &blendFilter{color: srgb, mode: mode}
}

// Color returns the stored unpremul sRGB color; Mode returns the blend mode — the descriptor the GPU side consumes.
func (f *blendFilter) Color() colorcore.Color4f { return f.color }

// Mode returns the blend mode.
func (f *blendFilter) Mode() raster.BlendMode { return f.mode }

// IsAlphaUnchanged implements shaders.ColorFilter: only Dst and SrcATop preserve alpha exactly.
func (f *blendFilter) IsAlphaUnchanged() bool {
	switch f.mode {
	case raster.BlendDst, raster.BlendSrcATop:
		return true
	default:
		return false
	}
}

// AppendStages implements shaders.ColorFilter: the incoming color moves to dst, the filter color becomes src (converted
// to dst premul — a bare premultiply, since both are sRGB), and the blend stages run.
func (f *blendFilter) AppendStages(p *shaders.Pipeline, _ bool) bool {
	p.AppendMoveSrcDst()
	p.AppendConstantColor(f.color.Premul())
	p.AppendBlend(f.mode)
	return true
}

///////////////////////////////////////////////////////////////////////////////
// compose

// composeFilter chains two filters: result = outer(inner(color)).
type composeFilter struct {
	outer, inner shaders.ColorFilter
}

// NewCompose returns the composition outer(inner(color)); a nil outer yields inner and vice versa.
func NewCompose(outer, inner shaders.ColorFilter) shaders.ColorFilter {
	if outer == nil {
		return inner
	}
	if inner == nil {
		return outer
	}
	return &composeFilter{outer: outer, inner: inner}
}

// Outer and Inner return the children — the descriptor the GPU side consumes.
func (f *composeFilter) Outer() shaders.ColorFilter { return f.outer }

// Inner returns the inner child.
func (f *composeFilter) Inner() shaders.ColorFilter { return f.inner }

// IsAlphaUnchanged implements shaders.ColorFilter: true only when both children preserve alpha.
func (f *composeFilter) IsAlphaUnchanged() bool {
	return f.outer.IsAlphaUnchanged() && f.inner.IsAlphaUnchanged()
}

// AppendStages implements shaders.ColorFilter: inner's stages, then outer's.
func (f *composeFilter) AppendStages(p *shaders.Pipeline, shaderIsOpaque bool) bool {
	innerIsOpaque := shaderIsOpaque
	if !f.inner.IsAlphaUnchanged() {
		innerIsOpaque = false
	}
	return f.inner.AppendStages(p, shaderIsOpaque) && f.outer.AppendStages(p, innerIsOpaque)
}

///////////////////////////////////////////////////////////////////////////////
// lighting

// byteToUnitFloat converts a byte to a unit float, mapping 0xFF to exactly 1.
func byteToUnitFloat(b uint8) float32 {
	if b == 0xFF {
		// want to get this exact
		return 1
	}
	return float32(b) * 0.00392156862745
}

// NewLighting returns the lighting filter: a legacy multiply/add on RGB. A black add collapses to a modulate blend
// filter; otherwise it builds a 4x5 matrix (scale by mul, then translate by add).
func NewLighting(mul, add colorcore.Color) shaders.ColorFilter {
	const opaqueAlphaMask = colorcore.Color(0xFF000000) // opaque black
	// omit the alpha and compare only the RGB values
	if add&^opaqueAlphaMask == 0 {
		return NewBlend(mul|opaqueAlphaMask, raster.BlendModulate)
	}
	var m [20]float32
	// scale by (mulR, mulG, mulB, 1)
	m[0] = byteToUnitFloat(mul.R())
	m[6] = byteToUnitFloat(mul.G())
	m[12] = byteToUnitFloat(mul.B())
	m[18] = 1
	// post-translate by (addR, addG, addB, 0)
	m[4] = byteToUnitFloat(add.R())
	m[9] = byteToUnitFloat(add.G())
	m[14] = byteToUnitFloat(add.B())
	return NewMatrix(&m)
}
