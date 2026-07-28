// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// FilterDecalShader: a decal-tiling weight ramp applied to an image shader at layer-space resolution, with no
// shader-language support — the ramp math is implemented directly per lane:
//
//	d = saturate((decalBounds - coord.xyxy) * (-1, -1, 1, 1) + 0.5);
//	return (d.x * d.y * d.z * d.w) * image.eval(coord);

package shaders

import "github.com/richardwilkes/canvas/geom"

// FilterDecalShader weights a child shader by a layer-space decal ramp.
type FilterDecalShader struct {
	image       Shader
	decalBounds geom.Rect
}

// NewFilterDecal wraps image with the decal ramp over decalBounds. A nil image yields nil.
func NewFilterDecal(image Shader, decalBounds geom.Rect) Shader {
	if image == nil {
		return nil
	}
	return &FilterDecalShader{image: image, decalBounds: decalBounds}
}

// Image returns the wrapped shader (for the GPU side).
func (s *FilterDecalShader) Image() Shader { return s.image }

// DecalBounds returns the layer-space decal bounds.
func (s *FilterDecalShader) DecalBounds() geom.Rect { return s.decalBounds }

// IsOpaque implements Shader: the ramp introduces transparency.
func (s *FilterDecalShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *FilterDecalShader) IsConstant() bool { return false }

// filterDecalCtx holds the decal kernel's saved coordinates plus the ramp bounds (copied from the shader) in reused
// pipeline storage, so the weight stage reads them through z.ctx instead of a per-compile new([2][stride]float32) plus
// two capturing closures. coords is written in full by appendStoreCoords before the weight stage reads it within each
// ShadeSpan chunk, so pooled reuse needs no zeroing.
type filterDecalCtx struct {
	coords     [2][stride]float32
	l, t, r, b float32
}

func (p *Pipeline) nextFilterDecalCtx() *filterDecalCtx {
	if p.filterDecalCtxN == len(p.filterDecalCtxs) {
		p.filterDecalCtxs = append(p.filterDecalCtxs, &filterDecalCtx{})
	}
	c := p.filterDecalCtxs[p.filterDecalCtxN]
	p.filterDecalCtxN++
	return c
}

// appendStages evaluates main(coord): resolve the local coords, remember them, run the child at the same coords, then
// scale by the decal weights.
func (s *FilterDecalShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	c := p.nextFilterDecalCtx()
	c.l, c.t, c.r, c.b = s.decalBounds.Left, s.decalBounds.Top, s.decalBounds.Right, s.decalBounds.Bottom
	p.appendStoreCoords(&c.coords)
	if !s.image.appendStages(p, seeded) {
		return false
	}
	p.appendCtx(filterDecalWeightStage, c)
	return true
}

// filterDecalWeightStage scales the child's output by the decal ramp weight at the saved coordinates.
func filterDecalWeightStage(z *lanes) {
	c := z.ctx.(*filterDecalCtx)
	l, t, r, b := c.l, c.t, c.r, c.b
	for i := range z.n {
		x := c.coords[0][i]
		y := c.coords[1][i]
		// d = saturate((bounds - coord.xyxy) * (-1,-1,1,1) + 0.5)
		dx0 := clamp01(x - l + 0.5)
		dy0 := clamp01(y - t + 0.5)
		dx1 := clamp01(r - x + 0.5)
		dy1 := clamp01(b - y + 0.5)
		w := dx0 * dy0 * dx1 * dy1
		z.r[i] *= w
		z.g[i] *= w
		z.b[i] *= w
		z.a[i] *= w
	}
}
