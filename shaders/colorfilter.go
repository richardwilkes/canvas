// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The color-filter contract the pipeline consumes, and ColorFilterShader — the wrapper that folds a paint's color
// filter into a shader when one is present. Concrete filters live in the colorfilter package; this file holds only what
// the shader model needs, exposing color filters as an interface.

package shaders

import "github.com/richardwilkes/canvas/colorcore"

// ColorFilter is the surface a color filter exposes to the raster pipeline.
type ColorFilter interface {
	// IsAlphaUnchanged reports whether the filter leaves alpha untouched.
	IsAlphaUnchanged() bool

	// AppendStages appends the filter's raster-pipeline stages. It returns false if the filter cannot run on the raster
	// pipeline.
	AppendStages(p *Pipeline, shaderIsOpaque bool) bool
}

// ColorFilterShader wraps a shader with a color filter applied to its output.
type ColorFilterShader struct {
	shader Shader
	filter ColorFilter
	alpha  float32
}

// NewColorFilterShader builds a shader that runs s and then applies cf: nil shader yields nil, nil filter yields the
// shader unchanged. The paint-alpha-removal path passes the paint's alpha; a plain with-color-filter wrap passes 1.
func NewColorFilterShader(s Shader, alpha float32, cf ColorFilter) Shader {
	switch {
	case s == nil:
		return nil
	case cf == nil:
		return s
	default:
		return &ColorFilterShader{shader: s, filter: cf, alpha: alpha}
	}
}

// NewWithColorFilter wraps s with cf applied to its output.
func NewWithColorFilter(s Shader, cf ColorFilter) Shader {
	return NewColorFilterShader(s, 1, cf)
}

// Shader returns the wrapped shader; Filter and Alpha return the other descriptor fields — the GPU side consumes these.
func (s *ColorFilterShader) Shader() Shader { return s.shader }

// Filter returns the wrapped color filter.
func (s *ColorFilterShader) Filter() ColorFilter { return s.filter }

// Alpha returns the folded paint alpha.
func (s *ColorFilterShader) Alpha() float32 { return s.alpha }

// IsOpaque implements Shader.
func (s *ColorFilterShader) IsOpaque() bool {
	return s.shader.IsOpaque() && s.alpha == 1 && s.filter.IsAlphaUnchanged()
}

// IsConstant implements Shader: a filtered constant shader never takes the blitter's constant collapse.
func (s *ColorFilterShader) IsConstant() bool { return false }

// appendStages runs the wrapped shader, applies the folded alpha (if any), then appends the color filter's stages.
func (s *ColorFilterShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	if !s.shader.appendStages(p, m) {
		return false
	}
	if s.alpha != 1 {
		p.appendScale1Float(s.alpha)
	}
	return s.filter.AppendStages(p, s.alpha == 1 && s.shader.IsOpaque())
}

// FilterColor4f runs the filter's stages over the single premultiplied color and returns the raw result (no clamping;
// the caller is responsible for pinning alpha and unpremultiplying if needed). ok is false when the filter cannot run
// on the raster pipeline.
func FilterColor4f(cf ColorFilter, c colorcore.PMColor4f) (result colorcore.PMColor4f, ok bool) {
	p := &Pipeline{}
	p.AppendConstantColor(c)
	if !cf.AppendStages(p, c.A == 1) {
		return colorcore.PMColor4f{}, false
	}
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(0, 0, out[:])
	return out[0], true
}
