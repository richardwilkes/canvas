// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The shader mask filter: coverage becomes mask * shader-alpha. Conceptually this is the shader drawn over a copy of
// the mask through an A8 canvas with a src-in paint; per pixel, that reduces to: evaluate the shader, clamp alpha to
// [0, 1], load the existing A8 coverage (normalized), multiply (a *= da), and store back to A8 — which is exactly the
// per-pixel math applied here.

package maskfilter

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// shaderMaskFilter is the shader mask filter implementation.
type shaderMaskFilter struct {
	shader shaders.Shader
}

// NewShader creates a shader mask filter; nil shader yields nil.
func NewShader(s shaders.Shader) MaskFilter {
	if s == nil {
		return nil
	}
	return &shaderMaskFilter{shader: s}
}

// Shader returns the shader — the descriptor the GPU side consumes.
func (f *shaderMaskFilter) Shader() shaders.Shader { return f.shader }

// FilterMask implements MaskFilter: multiplies src's coverage by the shader's alpha.
func (f *shaderMaskFilter) FilterMask(src *raster.Mask, ctm *geom.Matrix) (*raster.Mask, geom.IPoint, bool) {
	dst := &raster.Mask{Bounds: src.Bounds, RowBytes: src.Bounds.Width()}
	if src.Image == nil {
		return dst, geom.IPoint{}, true
	}
	size := computeImageSize(dst.Bounds, dst.RowBytes)
	if size == 0 {
		return nil, geom.IPoint{}, false // too big to allocate, abort
	}

	// Allocate and initialize dst image with a copy of the src image.
	dst.Image = make([]uint8, size)
	w := int(src.Bounds.Width())
	for y := range int(src.Bounds.Height()) {
		copy(dst.Image[y*int(dst.RowBytes):y*int(dst.RowBytes)+w], src.Image[y*int(src.RowBytes):])
	}

	// The canvas translates to mask-local coordinates then concats the CTM, so the shader sees total = translate(-L,
	// -T) * ctm; the default paint is opaque black (no alpha scale).
	var total geom.Matrix
	total.SetTranslate(float32(-dst.Bounds.Left), float32(-dst.Bounds.Top))
	total.PreConcat(ctm)
	pipeline := shaders.Compile(f.shader, total, colorcore.Color(0xFF000000))
	if pipeline == nil {
		// The shader cannot draw: leave dst as the unmodified copy of src.
		return dst, geom.IPoint{}, true
	}

	buf := make([]colorcore.PMColor4f, w)
	for y := range int(dst.Bounds.Height()) {
		pipeline.ShadeSpan(0, int32(y), buf)
		row := dst.Image[y*int(dst.RowBytes) : y*int(dst.RowBytes)+w]
		for i, c := range buf {
			// clamp_01, load_a8 (da = byte/255), srcin (a *= da), store_a8 (to_unorm rounds to nearest even like the
			// hardware conversion).
			sa := c.A
			if sa < 0 {
				sa = 0
			} else if sa > 1 {
				sa = 1
			}
			da := float32(row[i]) * (1.0 / 255.0)
			v := sa * da * 255
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			row[i] = uint8(math.RoundToEven(float64(v)))
		}
	}
	return dst, geom.IPoint{}, true
}

// ComputeFastBounds implements MaskFilter: coverage modulation does not move bounds.
func (f *shaderMaskFilter) ComputeFastBounds(src geom.Rect) geom.Rect {
	return src
}
