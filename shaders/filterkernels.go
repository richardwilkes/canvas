// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The runtime-effect shader kernels consumed by the shader-evaluating image filters (morphology, displacement,
// lighting, magnifier, matrix convolution, arithmetic), implemented directly, with no shader-language support. Each
// shader evaluates its kernel by writing the sample coordinates into the (r,g) registers and appending the child's
// stages with the already-applied matrix state; a null child evaluates as transparent black, and the per-lane math
// follows the reference lowering exactly — dot/mix/length fuse through mad, everything else is plain float32 mul/add,
// and pow is an approximate power function. Loops whose trip counts are uniforms (morphology radius, convolution kernel
// size) are unrolled at build time, which executes the identical per-lane math as branching would.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// storeCoordsStage saves the current (r,g) local coordinates into the caller-provided buffer (the context), so a filter
// kernel that re-derives child sample coordinates can restore the originals. The buffer is written in full before it is
// read within each ShadeSpan chunk, so pooled reuse needs no zeroing.
func storeCoordsStage(z *lanes) {
	c := z.ctx.(*[2][stride]float32)
	c[0] = z.r
	c[1] = z.g
}

// appendStoreCoords appends a stage saving the current (r,g) local coordinates into dst (a field on the kernel's pooled
// context), replacing a per-compile new([2][stride]float32) + capturing closure with a static stage reading dst through
// z.ctx.
func (p *Pipeline) appendStoreCoords(dst *[2][stride]float32) {
	p.appendCtx(storeCoordsStage, dst)
}

// appendChildOrTransparent evaluates a child shader with the coordinates currently in (r,g), or splats transparent
// black for a null child.
func (p *Pipeline) appendChildOrTransparent(child Shader, m *MatrixRec) bool {
	if child == nil {
		p.AppendConstantColor(colorcore.PMColor4f{})
		return true
	}
	return child.appendStages(p, *m)
}

// dot3 computes the 3-component dot product via mad(a0,b0, mad(a1,b1, a2*b2)).
func dot3(a0, a1, a2, b0, b1, b2 float32) float32 {
	return madf(a0, b0, madf(a1, b1, a2*b2))
}

///////////////////////////////////////////////////////////////////////////////
// Morphology (linear and sparse forms)

// maxLinearMorphologyRadius is the largest radius the linear (unrolled) morphology form handles (keep in sync with
// imagefilter's morphology passes).
const maxLinearMorphologyRadius = 14

// MorphologyShader is the linear (radius-unrolled) or sparse (two-tap) morphology kernel over one child. flip is +1 for
// dilate (max) and -1 for erode (the max() calls become min()).
type MorphologyShader struct {
	child  Shader
	offset geom.Point
	flip   float32
	radius int32 // 0 = sparse form (single ±offset tap pair)
}

// NewLinearMorphology builds a morphology shader with (2*radius+1) taps along offset's axis.
func NewLinearMorphology(child Shader, offset geom.Point, dilate bool, radius int32) Shader {
	flip := float32(-1)
	if dilate {
		flip = 1
	}
	return &MorphologyShader{child: child, offset: offset, flip: flip, radius: radius}
}

// NewSparseMorphology builds a morphology shader with two taps at ±offset.
func NewSparseMorphology(child Shader, offset geom.Point, dilate bool) Shader {
	flip := float32(-1)
	if dilate {
		flip = 1
	}
	return &MorphologyShader{child: child, offset: offset, flip: flip}
}

// Child returns the sampled child shader (for the GPU FP conversion).
func (s *MorphologyShader) Child() Shader { return s.child }

// Offset returns the per-step tap offset (for the GPU FP conversion).
func (s *MorphologyShader) Offset() geom.Point { return s.offset }

// Flip returns the aggregate sign: +1 for dilate (max), -1 for erode (for the GPU FP conversion).
func (s *MorphologyShader) Flip() float32 { return s.flip }

// Radius returns the linear radius, or 0 for the sparse form (for the GPU FP conversion).
func (s *MorphologyShader) Radius() int32 { return s.radius }

// IsOpaque implements Shader.
func (s *MorphologyShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *MorphologyShader) IsConstant() bool { return false }

// morphologyCtx holds the linear/sparse morphology kernel's register-file scratch (coords saved by appendStoreCoords,
// the running aggregate, and — for the linear form — the +delta tap output held across the -delta child eval), plus the
// flip sign, in reused pipeline storage so the stages read it through z.ctx instead of capturing it. Every buffer is
// written before it is read within a ShadeSpan chunk (agg by the seed stage, plus by the take-plus stage, coords by
// store-coords), so pooled reuse needs no zeroing. steps carries one morphStep per linear delta; nextStep hands out a
// distinct one so a shared step would not clobber a prior delta held in the appended stage list.
type morphologyCtx struct {
	steps  []*morphStep
	stepN  int
	agg    [4][stride]float32
	plus   [4][stride]float32
	coords [2][stride]float32
	flip   float32
}

// morphStep carries one linear/sparse tap's ±delta and a back-pointer to its owning morphologyCtx (both the coord-setup
// and take-plus stages need the parent's coords/plus buffers alongside the per-step delta). The back-pointer is
// self-referential within the retained pool, so it pins nothing external.
type morphStep struct {
	c      *morphologyCtx
	dx, dy float32
}

// nextStep hands the delta at (dx, dy) a distinct morphStep, retained on the ctx (a []*morphStep so a regrow leaves
// prior step pointers — already stored in the appended stage list — valid).
func (c *morphologyCtx) nextStep(dx, dy float32) *morphStep {
	if c.stepN == len(c.steps) {
		c.steps = append(c.steps, &morphStep{})
	}
	st := c.steps[c.stepN]
	st.c = c
	st.dx = dx
	st.dy = dy
	c.stepN++
	return st
}

func (p *Pipeline) nextMorphologyCtx() *morphologyCtx {
	if p.morphCtxN == len(p.morphCtxs) {
		p.morphCtxs = append(p.morphCtxs, &morphologyCtx{})
	}
	c := p.morphCtxs[p.morphCtxN]
	p.morphCtxN++
	c.stepN = 0
	return c
}

func (s *MorphologyShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	// The child's SampleUsage is explicit (coord ± delta), so its total matrix is invalid.
	seeded = seeded.markTotalMatrixInvalid()
	c := p.nextMorphologyCtx()
	c.flip = s.flip
	p.appendStoreCoords(&c.coords)

	if s.radius > 0 {
		// Linear form: aggregate = flip * child.eval(coord), then for each delta up to radius: aggregate =
		// max(aggregate, max(flip*eval(coord+delta), flip*eval(coord-delta))).
		if !p.appendChildOrTransparent(s.child, &seeded) {
			return false
		}
		p.appendCtx(morphAggInitStage, c)
		delta := s.offset
		for step := int32(1); step <= maxLinearMorphologyRadius; step++ {
			if step > s.radius {
				break
			}
			st := c.nextStep(delta.X, delta.Y)
			p.appendCtx(morphPlusCoordsStage, st)
			if !p.appendChildOrTransparent(s.child, &seeded) {
				return false
			}
			p.appendCtx(morphTakePlusStage, st)
			if !p.appendChildOrTransparent(s.child, &seeded) {
				return false
			}
			p.appendCtx(morphMaxAggStage, c)
			delta.X += s.offset.X
			delta.Y += s.offset.Y
		}
	} else {
		// Sparse form: aggregate = max(flip*eval(coord+offset), flip*eval(coord-offset)).
		st := c.nextStep(s.offset.X, s.offset.Y)
		p.appendCtx(morphPlusCoordsStage, st)
		if !p.appendChildOrTransparent(s.child, &seeded) {
			return false
		}
		p.appendCtx(morphSparseAggMinusStage, st)
		if !p.appendChildOrTransparent(s.child, &seeded) {
			return false
		}
		p.appendCtx(morphSparseMaxAggStage, c)
	}

	// return flip * aggregate
	p.appendCtx(morphReturnStage, c)
	return true
}

// morphAggInitStage seeds the linear aggregate: agg = flip * child.eval(coord).
func morphAggInitStage(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := c.flip
	for i := range z.n {
		c.agg[0][i] = flip * z.r[i]
		c.agg[1][i] = flip * z.g[i]
		c.agg[2][i] = flip * z.b[i]
		c.agg[3][i] = flip * z.a[i]
	}
}

// morphPlusCoordsStage sets the child's sample coordinates to coord + delta (the +delta tap).
func morphPlusCoordsStage(z *lanes) {
	st := z.ctx.(*morphStep)
	c := st.c
	for i := range z.n {
		z.r[i] = c.coords[0][i] + st.dx
		z.g[i] = c.coords[1][i] + st.dy
	}
}

// morphTakePlusStage records the +delta tap output (plus = flip*child) and sets up the -delta tap (coord - delta),
// mirroring the linear kernel's two per-delta child evals.
func morphTakePlusStage(z *lanes) {
	st := z.ctx.(*morphStep)
	c := st.c
	flip := c.flip
	for i := range z.n {
		c.plus[0][i] = flip * z.r[i]
		c.plus[1][i] = flip * z.g[i]
		c.plus[2][i] = flip * z.b[i]
		c.plus[3][i] = flip * z.a[i]
	}
	for i := range z.n {
		z.r[i] = c.coords[0][i] - st.dx
		z.g[i] = c.coords[1][i] - st.dy
	}
}

// morphMaxAggStage folds the delta into the linear aggregate: agg = max(agg, max(plus, flip*minus)).
func morphMaxAggStage(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := c.flip
	minus := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	for ch := range 4 {
		for i := range z.n {
			c.agg[ch][i] = maxf(c.agg[ch][i], maxf(c.plus[ch][i], flip*minus[ch][i]))
		}
	}
}

// morphSparseAggMinusStage records the sparse +offset tap (agg = flip*child) and sets up the -offset tap (coord -
// offset).
func morphSparseAggMinusStage(z *lanes) {
	st := z.ctx.(*morphStep)
	c := st.c
	flip := c.flip
	for i := range z.n {
		c.agg[0][i] = flip * z.r[i]
		c.agg[1][i] = flip * z.g[i]
		c.agg[2][i] = flip * z.b[i]
		c.agg[3][i] = flip * z.a[i]
	}
	for i := range z.n {
		z.r[i] = c.coords[0][i] - st.dx
		z.g[i] = c.coords[1][i] - st.dy
	}
}

// morphSparseMaxAggStage folds the sparse -offset tap: agg = max(flip*minus, agg) (argument order preserved from the C
// kernel).
func morphSparseMaxAggStage(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := c.flip
	minus := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	for ch := range 4 {
		for i := range z.n {
			c.agg[ch][i] = maxf(flip*minus[ch][i], c.agg[ch][i])
		}
	}
}

// morphReturnStage yields flip * aggregate.
func morphReturnStage(z *lanes) {
	c := z.ctx.(*morphologyCtx)
	flip := c.flip
	for i := range z.n {
		z.r[i] = flip * c.agg[0][i]
		z.g[i] = flip * c.agg[1][i]
		z.b[i] = flip * c.agg[2][i]
		z.a[i] = flip * c.agg[3][i]
	}
}

///////////////////////////////////////////////////////////////////////////////
// Displacement

// DisplacementShader displaces the colorMap's sample coordinates by the displMap's selected channels: displ = scale *
// (unpremul(displMap.eval(coord)).sel - 0.5).
type DisplacementShader struct {
	displMap   Shader
	colorMap   Shader
	scale      geom.Point // layer-space vector (negations preserved)
	xSel, ySel int32      // channel indices 0=R 1=G 2=B 3=A (one-hot channel selection)
}

// NewDisplacement builds a displacement-map shader. colorMap must be non-nil (the filter returns nil before building
// the shader when the color input is transparent); a nil displMap is evaluated as transparent black.
func NewDisplacement(displMap, colorMap Shader, scale geom.Point, xSel, ySel int32) Shader {
	return &DisplacementShader{displMap: displMap, colorMap: colorMap, scale: scale, xSel: xSel, ySel: ySel}
}

// DisplMap returns the displacement-map child; may be nil (transparent black). For the GPU FP conversion.
func (s *DisplacementShader) DisplMap() Shader { return s.displMap }

// ColorMap returns the displaced color child (for the GPU FP conversion).
func (s *DisplacementShader) ColorMap() Shader { return s.colorMap }

// Scale returns the layer-space displacement scale vector (for the GPU FP conversion).
func (s *DisplacementShader) Scale() geom.Point { return s.scale }

// ChannelSelectors returns the X/Y displacement channel indices (0=R 1=G 2=B 3=A), for the GPU FP conversion.
func (s *DisplacementShader) ChannelSelectors() (xSel, ySel int32) { return s.xSel, s.ySel }

// IsOpaque implements Shader.
func (s *DisplacementShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *DisplacementShader) IsConstant() bool { return false }

// displacementCtx holds the displacement kernel's saved coordinates and the scale/channel-select scalars (copied from
// the shader) in reused pipeline storage so displacementStage reads them through z.ctx instead of capturing them.
type displacementCtx struct {
	coords     [2][stride]float32
	sx, sy     float32
	xSel, ySel int32
}

func (p *Pipeline) nextDisplacementCtx() *displacementCtx {
	if p.dispCtxN == len(p.dispCtxs) {
		p.dispCtxs = append(p.dispCtxs, &displacementCtx{})
	}
	c := p.dispCtxs[p.dispCtxN]
	p.dispCtxN++
	return c
}

func (s *DisplacementShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	c := p.nextDisplacementCtx()
	c.sx = s.scale.X
	c.sy = s.scale.Y
	c.xSel = s.xSel
	c.ySel = s.ySel
	p.appendStoreCoords(&c.coords)
	// displMap.eval(coord) is a passthrough sample; colorMap.eval(coord + displ) is explicit and invalidates the total
	// matrix.
	if !p.appendChildOrTransparent(s.displMap, &seeded) {
		return false
	}
	p.appendCtx(displacementStage, c)
	return s.colorMap.appendStages(p, seeded.markTotalMatrixInvalid())
}

func displacementStage(z *lanes) {
	c := z.ctx.(*displacementCtx)
	ch := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	sx, sy := c.sx, c.sy
	xSel, ySel := c.xSel, c.ySel
	for i := range z.n {
		// unpremul: rgb / max(a, 0.0001); the one-hot dot selects a single channel exactly (the zero terms of
		// dot_4_floats contribute exact zeros).
		inv := maxf(z.a[i], 0.0001)
		sel := func(sc int32) float32 {
			if sc == 3 {
				return z.a[i]
			}
			return ch[sc][i] / inv
		}
		dx := sx * (sel(xSel) - 0.5)
		dy := sy * (sel(ySel) - 0.5)
		z.r[i] = c.coords[0][i] + dx
		z.g[i] = c.coords[1][i] + dy
	}
}

///////////////////////////////////////////////////////////////////////////////
// Normal map

// NormalShader runs the SVG Sobel filter on the alpha channel of its child, producing surface normals in RGB and the
// child's alpha in A.
type NormalShader struct {
	alphaMap        Shader
	edgeBounds      geom.Rect // already inset by 0.5 (make_normal_shader)
	negSurfaceDepth float32
}

// NewNormal builds a Sobel-normal shader (uniform prep included by the caller).
func NewNormal(alphaMap Shader, edgeBounds geom.Rect, negSurfaceDepth float32) Shader {
	return &NormalShader{alphaMap: alphaMap, edgeBounds: edgeBounds, negSurfaceDepth: negSurfaceDepth}
}

// AlphaMap returns the Sobel-sampled alpha child; may be nil (transparent black). For the GPU FP conversion.
func (s *NormalShader) AlphaMap() Shader { return s.alphaMap }

// EdgeBounds returns the half-pixel-inset clamp bounds (for the GPU FP conversion).
func (s *NormalShader) EdgeBounds() geom.Rect { return s.edgeBounds }

// NegSurfaceDepth returns the negated surface depth (for the GPU FP conversion).
func (s *NormalShader) NegSurfaceDepth() float32 { return s.negSurfaceDepth }

// IsOpaque implements Shader.
func (s *NormalShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *NormalShader) IsConstant() bool { return false }

// normalCtx holds the Sobel normal kernel's saved coordinates, the nine tap alphas, and the edge/depth scalars (copied
// from the shader) in reused pipeline storage. taps carries the nine constant tap offsets with a back-pointer so the
// coord-setup stage reads its ±offset alongside the parent's coords; each alpha slot is written by its save-alpha stage
// before normalFilterStage reads it, so pooled reuse needs no zeroing.
type normalCtx struct {
	taps            [9]normalTapCtx
	alpha           [9][stride]float32
	coords          [2][stride]float32
	l               float32
	t               float32
	r               float32
	b               float32
	negSurfaceDepth float32
}

// normalTapCtx is one of the nine fixed Sobel taps: its ±offset and a back-pointer to the owning normalCtx (both
// self-referential within the retained pool).
type normalTapCtx struct {
	c        *normalCtx
	ddx, ddy float32
}

func (p *Pipeline) nextNormalCtx() *normalCtx {
	if p.normalCtxN == len(p.normalCtxs) {
		p.normalCtxs = append(p.normalCtxs, &normalCtx{})
	}
	c := p.normalCtxs[p.normalCtxN]
	p.normalCtxN++
	return c
}

func (s *NormalShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	// The alpha map's SampleUsage is explicit (clamped offsets), invalidating the total matrix.
	seeded = seeded.markTotalMatrixInvalid()
	c := p.nextNormalCtx()
	c.l, c.t, c.r, c.b = s.edgeBounds.Left, s.edgeBounds.Top, s.edgeBounds.Right, s.edgeBounds.Bottom
	c.negSurfaceDepth = s.negSurfaceDepth
	p.appendStoreCoords(&c.coords)

	// Nine taps in column-major order: columns x=-1,0,1 with y=-1,0,1 within each. clamp(v, lo, hi) lowers to
	// min(max(v, lo), hi).
	tap := 0
	for dx := float32(-1); dx <= 1; dx++ {
		for dy := float32(-1); dy <= 1; dy++ {
			tp := &c.taps[tap]
			tp.c = c
			tp.ddx = dx
			tp.ddy = dy
			p.appendCtx(normalSetCoordsStage, tp)
			if !p.appendChildOrTransparent(s.alphaMap, &seeded) {
				return false
			}
			p.appendCtx(saveAlphaStage, &c.alpha[tap])
			tap++
		}
	}

	p.appendCtx(normalFilterStage, c)
	return true
}

// normalSetCoordsStage clamps coord + tap-offset into the edge bounds for one Sobel tap.
func normalSetCoordsStage(z *lanes) {
	tp := z.ctx.(*normalTapCtx)
	c := tp.c
	for i := range z.n {
		z.r[i] = minf(maxf(c.coords[0][i]+tp.ddx, c.l), c.r)
		z.g[i] = minf(maxf(c.coords[1][i]+tp.ddy, c.t), c.b)
	}
}

// saveAlphaStage copies the child's alpha channel into the caller-provided slot (a normalCtx tap alpha). The slot is
// fully written before it is read within a ShadeSpan chunk, so pooled reuse needs no zeroing.
func saveAlphaStage(z *lanes) {
	a := z.ctx.(*[stride]float32)
	*a = z.a
}

// normalFilterStage runs $normal_filter over the nine saved tap alphas.
func normalFilterStage(z *lanes) {
	c := z.ctx.(*normalCtx)
	negDepth := c.negSurfaceDepth
	for i := range z.n {
		// $normal_filter with kSobel = 0.25*(1,2,1) constant-folded to (0.25, 0.5, 0.25).
		c0 := [3]float32{c.alpha[0][i], c.alpha[1][i], c.alpha[2][i]} // x = -1
		c1 := [3]float32{c.alpha[3][i], c.alpha[4][i], c.alpha[5][i]} // x = 0
		c2 := [3]float32{c.alpha[6][i], c.alpha[7][i], c.alpha[8][i]} // x = +1
		nx := dot3(0.25, 0.5, 0.25, c2[0], c2[1], c2[2]) - dot3(0.25, 0.5, 0.25, c0[0], c0[1], c0[2])
		ny := dot3(0.25, 0.5, 0.25, c0[2], c1[2], c2[2]) - dot3(0.25, 0.5, 0.25, c0[0], c1[0], c2[0])
		vx := negDepth * nx
		vy := negDepth * ny
		// normalize(v) lowers to v / vec(sqrt(dot(v, v))).
		invLen := sqrtf(dot3(vx, vy, 1, vx, vy, 1))
		z.r[i] = vx / invLen
		z.g[i] = vy / invLen
		z.b[i] = 1 / invLen
		z.a[i] = c1[1]
	}
}

///////////////////////////////////////////////////////////////////////////////
// Lighting

// LightingShader lighting/material types (values match the packed uniform layout).
const (
	// LightDistant maps to lightType < 0.
	LightDistant float32 = -1
	// LightPoint maps to lightType == 0.
	LightPoint float32 = 0
	// LightSpot maps to lightType > 0.
	LightSpot float32 = 1
)

// LightingShader evaluates the Phong lighting equation over a normal-map child.
type LightingShader struct {
	normalMap      Shader
	depth          float32 // surface depth in layer space
	shininess      float32
	specular       bool // materialType: 0 = diffuse, 1 = specular (emboss not publicly reachable)
	lightType      float32
	lightPos       [3]float32
	spotFalloff    float32
	lightDir       [3]float32 // pre-normalized by the caller
	cosCutoffAngle float32
	lightColor     [3]float32 // material k already multiplied in
}

// NewLighting builds a Phong lighting shader (the caller performs the normalization and color scaling).
func NewLighting(normalMap Shader, depth, shininess float32, specular bool, lightType float32, lightPos [3]float32, spotFalloff float32, lightDir [3]float32, cosCutoffAngle float32, lightColor [3]float32) Shader {
	return &LightingShader{
		normalMap:      normalMap,
		depth:          depth,
		shininess:      shininess,
		specular:       specular,
		lightType:      lightType,
		lightPos:       lightPos,
		spotFalloff:    spotFalloff,
		lightDir:       lightDir,
		cosCutoffAngle: cosCutoffAngle,
		lightColor:     lightColor,
	}
}

// NormalMap returns the normal-map child; may be nil (transparent black). For the GPU FP conversion.
func (s *LightingShader) NormalMap() Shader { return s.normalMap }

// PackedUniforms returns the four uniform vectors in the shader's packed layout — materialAndLightType (depth,
// shininess, materialType, lightType), lightPosAndSpotFalloff, lightDirAndSpotCutoff, and lightColor — for the GPU FP
// conversion.
func (s *LightingShader) PackedUniforms() (materialAndLightType, lightPosAndSpotFalloff, lightDirAndSpotCutoff [4]float32, lightColor [3]float32) {
	materialType := float32(0)
	if s.specular {
		materialType = 1
	}
	materialAndLightType = [4]float32{s.depth, s.shininess, materialType, s.lightType}
	lightPosAndSpotFalloff = [4]float32{s.lightPos[0], s.lightPos[1], s.lightPos[2], s.spotFalloff}
	lightDirAndSpotCutoff = [4]float32{s.lightDir[0], s.lightDir[1], s.lightDir[2], s.cosCutoffAngle}
	return materialAndLightType, lightPosAndSpotFalloff, lightDirAndSpotCutoff, s.lightColor
}

// IsOpaque implements Shader.
func (s *LightingShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *LightingShader) IsConstant() bool { return false }

// lightingCtx holds the Phong lighting kernel's saved coordinates plus the light/material uniforms (copied from the
// shader — the same fields make_lighting_shader packs) in reused pipeline storage so lightingStage reads them through
// z.ctx instead of a whole-shader closure capture.
type lightingCtx struct {
	coords         [2][stride]float32
	depth          float32
	shininess      float32
	specular       bool
	lightType      float32
	lightPos       [3]float32
	spotFalloff    float32
	lightDir       [3]float32
	cosCutoffAngle float32
	lightColor     [3]float32
}

func (p *Pipeline) nextLightingCtx() *lightingCtx {
	if p.lightingCtxN == len(p.lightingCtxs) {
		p.lightingCtxs = append(p.lightingCtxs, &lightingCtx{})
	}
	c := p.lightingCtxs[p.lightingCtxN]
	p.lightingCtxN++
	return c
}

func (s *LightingShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	c := p.nextLightingCtx()
	c.depth = s.depth
	c.shininess = s.shininess
	c.specular = s.specular
	c.lightType = s.lightType
	c.lightPos = s.lightPos
	c.spotFalloff = s.spotFalloff
	c.lightDir = s.lightDir
	c.cosCutoffAngle = s.cosCutoffAngle
	c.lightColor = s.lightColor
	p.appendStoreCoords(&c.coords)
	if !p.appendChildOrTransparent(s.normalMap, &seeded) {
		return false
	}
	p.appendCtx(lightingStage, c)
	return true
}

func lightingStage(z *lanes) {
	sh := z.ctx.(*lightingCtx)
	const coneAAThreshold = 0.016
	const coneScale = 1.0 / coneAAThreshold // precomputed reciprocal
	for i := range z.n {
		nx, ny, nz := z.r[i], z.g[i], z.b[i]
		na := z.a[i]

		// $surface_to_light: spot (>0) and point (==0) share the normalize(lightPos - coord) equation; distant lights
		// use the uniform direction.
		var slx, sly, slz float32
		if sh.lightType >= 0 {
			vx := sh.lightPos[0] - sh.coords[0][i]
			vy := sh.lightPos[1] - sh.coords[1][i]
			vz := sh.lightPos[2] - sh.depth*na
			length := sqrtf(dot3(vx, vy, vz, vx, vy, vz))
			slx = vx / length
			sly = vy / length
			slz = vz / length
		} else {
			slx = sh.lightDir[0]
			sly = sh.lightDir[1]
			slz = sh.lightDir[2]
		}

		cr := sh.lightColor[0]
		cg := sh.lightColor[1]
		cb := sh.lightColor[2]
		if sh.lightType > 0 {
			// $spotlight_scale
			cosAngle := -dot3(slx, sly, slz, sh.lightDir[0], sh.lightDir[1], sh.lightDir[2])
			var scale float32
			if cosAngle >= sh.cosCutoffAngle {
				scale = colorcore.ApproxPowf(cosAngle, sh.spotFalloff)
				if cosAngle < sh.cosCutoffAngle+coneAAThreshold {
					scale = scale * (cosAngle - sh.cosCutoffAngle) * coneScale
				}
			}
			cr *= scale
			cg *= scale
			cb *= scale
		}

		if !sh.specular { // diffuse
			coeff := dot3(nx, ny, nz, slx, sly, slz)
			z.r[i] = clamp01(coeff * cr)
			z.g[i] = clamp01(coeff * cg)
			z.b[i] = clamp01(coeff * cb)
			z.a[i] = 1
		} else {
			hx := slx
			hy := sly
			hz := slz + 1
			length := sqrtf(dot3(hx, hy, hz, hx, hy, hz))
			coeff := colorcore.ApproxPowf(dot3(nx, ny, nz, hx/length, hy/length, hz/length), sh.shininess)
			r := clamp01(coeff * cr)
			g := clamp01(coeff * cg)
			b := clamp01(coeff * cb)
			z.r[i] = r
			z.g[i] = g
			z.b[i] = b
			z.a[i] = maxf(maxf(r, g), b)
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Magnifier

// MagnifierShader distorts sample coordinates toward the zoomed source rect with rounded-inset weighting.
type MagnifierShader struct {
	src        Shader
	lensBounds geom.Rect
	zoomXform  [4]float32 // (Tx, Ty, Sx, Sy)
	invInset   [2]float32
}

// NewMagnifier builds a magnifier distortion shader.
func NewMagnifier(src Shader, lensBounds geom.Rect, zoomXform [4]float32, invInset [2]float32) Shader {
	return &MagnifierShader{src: src, lensBounds: lensBounds, zoomXform: zoomXform, invInset: invInset}
}

// Src returns the zoom-sampled source child (for the GPU FP conversion).
func (s *MagnifierShader) Src() Shader { return s.src }

// LensBounds returns the layer-space lens bounds (for the GPU FP conversion).
func (s *MagnifierShader) LensBounds() geom.Rect { return s.lensBounds }

// ZoomXform returns the (Tx, Ty, Sx, Sy) zoom transform (for the GPU FP conversion).
func (s *MagnifierShader) ZoomXform() [4]float32 { return s.zoomXform }

// InvInset returns the reciprocal inset distances (for the GPU FP conversion).
func (s *MagnifierShader) InvInset() [2]float32 { return s.invInset }

// IsOpaque implements Shader.
func (s *MagnifierShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *MagnifierShader) IsConstant() bool { return false }

func (s *MagnifierShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	// The kernel reads only the immutable shader uniforms (no per-eval scratch), so the shader itself is the stage
	// context — a pointer in z.ctx allocates nothing, and RecyclePipeline's clear(stages) drops the reference.
	p.appendCtx(magnifierStage, s)
	// The src's SampleUsage is explicit (mixed coordinates), invalidating the total matrix.
	return s.src.appendStages(p, seeded.markTotalMatrixInvalid())
}

func magnifierStage(z *lanes) {
	sh := z.ctx.(*MagnifierShader)
	l, t, r, b := sh.lensBounds.Left, sh.lensBounds.Top, sh.lensBounds.Right, sh.lensBounds.Bottom
	for i := range z.n {
		x := z.r[i]
		y := z.g[i]
		zoomX := sh.zoomXform[0] + sh.zoomXform[2]*x
		zoomY := sh.zoomXform[1] + sh.zoomXform[3]*y
		// edgeInset = min(coord - lensBounds.xy, lensBounds.zw - coord) * invInset
		eiX := minf(x-l, r-x) * sh.invInset[0]
		eiY := minf(y-t, b-y) * sh.invInset[1]
		var weight float32
		if eiX < 2 && eiY < 2 {
			// Circular distortion weighted by distance to the inset corner: 2 - length(2 - edgeInset), with length(v2)
			// = sqrt(mad(x,x, y*y)).
			dx := 2 - eiX
			dy := 2 - eiY
			weight = 2 - sqrtf(madf(dx, dx, dy*dy))
		} else {
			weight = minf(eiX, eiY)
		}
		weight = clamp01(weight)
		w2 := weight * weight
		// mix(coord, zoomCoord, w2) lowers to lerp: mad(to - from, t, from).
		z.r[i] = madf(zoomX-x, w2, x)
		z.g[i] = madf(zoomY-y, w2, y)
	}
}

///////////////////////////////////////////////////////////////////////////////
// Matrix convolution

// MatrixConvShader applies an NxM convolution kernel to its child. kernel holds the per-tap coefficients exactly as the
// runtime effect reconstructs them (raw for the uniform variant, unorm8-quantized then gain/bias-reconstructed for the
// texture variants).
type MatrixConvShader struct {
	child         Shader
	kernel        []float32
	size          geom.ISize
	offset        geom.IPoint
	gain          float32
	bias          float32 // already scaled by 1/255
	convolveAlpha bool
}

// NewMatrixConv builds a matrix-convolution shader.
func NewMatrixConv(child Shader, kernel []float32, size geom.ISize, offset geom.IPoint, gain, bias float32, convolveAlpha bool) Shader {
	return &MatrixConvShader{
		child:         child,
		kernel:        kernel,
		size:          size,
		offset:        offset,
		gain:          gain,
		bias:          bias,
		convolveAlpha: convolveAlpha,
	}
}

// Child returns the convolved child shader; may be nil (transparent black). For the GPU FP conversion.
func (s *MatrixConvShader) Child() Shader { return s.child }

// Kernel returns the per-tap coefficients (for the GPU FP conversion).
func (s *MatrixConvShader) Kernel() []float32 { return s.kernel }

// Size returns the kernel dimensions (for the GPU FP conversion).
func (s *MatrixConvShader) Size() geom.ISize { return s.size }

// Offset returns the kernel offset (for the GPU FP conversion).
func (s *MatrixConvShader) Offset() geom.IPoint { return s.offset }

// GainAndBias returns the gain and the pre-scaled bias (for the GPU FP conversion).
func (s *MatrixConvShader) GainAndBias() (gain, bias float32) { return s.gain, s.bias }

// ConvolveAlpha reports whether the alpha channel convolves too (for the GPU FP conversion).
func (s *MatrixConvShader) ConvolveAlpha() bool { return s.convolveAlpha }

// IsOpaque implements Shader.
func (s *MatrixConvShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *MatrixConvShader) IsConstant() bool { return false }

// matrixConvCtx holds the convolution kernel's saved coordinates, the per-lane accumulator, the origin-tap alpha, and
// the offset/gain/bias/convolve-alpha scalars (copied from the shader) in reused pipeline storage. The init stage
// zeroes sum/origAlpha at the head of each ShadeSpan chunk, so pooled reuse needs no zeroing on handout. taps carries
// one matrixConvTap per kernel coefficient.
type matrixConvCtx struct {
	taps          []*matrixConvTap
	tapN          int
	sum           [4][stride]float32
	coords        [2][stride]float32
	origAlpha     [stride]float32
	ox            float32
	oy            float32
	gain          float32
	bias          float32
	convolveAlpha bool
}

// matrixConvTap is one kernel coefficient's data: its position (fkx, fky), coefficient k, whether it is the origin tap,
// and a back-pointer to the owning matrixConvCtx (self-referential within the pool).
type matrixConvTap struct {
	c        *matrixConvCtx
	fkx, fky float32
	k        float32
	isOrigin bool
}

// nextTap hands one kernel coefficient a distinct matrixConvTap, retained on the ctx (a []*matrixConvTap so a regrow
// leaves prior tap pointers — already in the appended stage list — valid).
func (c *matrixConvCtx) nextTap() *matrixConvTap {
	if c.tapN == len(c.taps) {
		c.taps = append(c.taps, &matrixConvTap{})
	}
	t := c.taps[c.tapN]
	t.c = c
	c.tapN++
	return t
}

func (p *Pipeline) nextMatrixConvCtx() *matrixConvCtx {
	if p.matConvCtxN == len(p.matConvCtxs) {
		p.matConvCtxs = append(p.matConvCtxs, &matrixConvCtx{})
	}
	c := p.matConvCtxs[p.matConvCtxN]
	p.matConvCtxN++
	c.tapN = 0
	return c
}

func (s *MatrixConvShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	seeded, ok := m.apply(p, &identity)
	if !ok {
		return false
	}
	// The child's SampleUsage is explicit (kernel offsets), invalidating the total matrix.
	seeded = seeded.markTotalMatrixInvalid()
	c := p.nextMatrixConvCtx()
	c.ox = float32(s.offset.X)
	c.oy = float32(s.offset.Y)
	c.gain = s.gain
	c.bias = s.bias
	c.convolveAlpha = s.convolveAlpha
	p.appendStoreCoords(&c.coords)
	p.appendCtx(matrixConvInitStage, c)

	tapIdx := 0
	for ky := int32(0); ky < s.size.Height; ky++ {
		for kx := int32(0); kx < s.size.Width; kx++ {
			t := c.nextTap()
			t.fkx, t.fky = float32(kx), float32(ky)
			t.k = s.kernel[tapIdx]
			t.isOrigin = kx == s.offset.X && ky == s.offset.Y
			tapIdx++
			p.appendCtx(matrixConvCoordsStage, t)
			if !p.appendChildOrTransparent(s.child, &seeded) {
				return false
			}
			p.appendCtx(matrixConvAccumStage, t)
		}
	}

	p.appendCtx(matrixConvFinalizeStage, c)
	return true
}

// matrixConvInitStage zeroes the accumulator and origin-alpha at the head of each ShadeSpan chunk.
func matrixConvInitStage(z *lanes) {
	c := z.ctx.(*matrixConvCtx)
	c.sum = [4][stride]float32{}
	c.origAlpha = [stride]float32{}
}

// matrixConvCoordsStage sets the child's sample coordinates for one kernel tap: coord + half2(kernelPos) -
// half2(offset) (add then subtract, preserving order).
func matrixConvCoordsStage(z *lanes) {
	t := z.ctx.(*matrixConvTap)
	c := t.c
	for i := range z.n {
		z.r[i] = (c.coords[0][i] + t.fkx) - c.ox
		z.g[i] = (c.coords[1][i] + t.fky) - c.oy
	}
}

// matrixConvAccumStage folds one tap into the accumulator (unpremultiplying first when not convolving alpha, and
// recording the origin tap's alpha).
func matrixConvAccumStage(z *lanes) {
	t := z.ctx.(*matrixConvTap)
	c := t.c
	k := t.k
	for i := range z.n {
		cr, cg, cb, ca := z.r[i], z.g[i], z.b[i], z.a[i]
		if !c.convolveAlpha {
			if t.isOrigin {
				c.origAlpha[i] = ca
			}
			inv := maxf(ca, 0.0001)
			cr /= inv
			cg /= inv
			cb /= inv
		}
		c.sum[0][i] += cr * k
		c.sum[1][i] += cg * k
		c.sum[2][i] += cb * k
		c.sum[3][i] += ca * k
	}
}

// matrixConvFinalizeStage applies gain/bias and reconstructs a valid premultiplied color.
func matrixConvFinalizeStage(z *lanes) {
	c := z.ctx.(*matrixConvCtx)
	gain, bias := c.gain, c.bias
	for i := range z.n {
		cr := c.sum[0][i]*gain + bias
		cg := c.sum[1][i]*gain + bias
		cb := c.sum[2][i]*gain + bias
		ca := c.sum[3][i]*gain + bias
		if !c.convolveAlpha {
			// Reset the alpha to the original and convert to premul RGB.
			ca = c.origAlpha[i]
			cr *= ca
			cg *= ca
			cb *= ca
		} else {
			ca = clamp01(ca)
		}
		// Make RGB valid premul w/ respect to the alpha: clamp(rgb, 0, a).
		z.r[i] = minf(maxf(cr, 0), ca)
		z.g[i] = minf(maxf(cg, 0), ca)
		z.b[i] = minf(maxf(cb, 0), ca)
		z.a[i] = ca
	}
}

///////////////////////////////////////////////////////////////////////////////
// Arithmetic blend

// ArithmeticShader blends the outputs of two child shaders with the arithmetic equation k1*src*dst + k2*src + k3*dst +
// k4.
type ArithmeticShader struct {
	dst, src      Shader
	k             [4]float32
	enforcePremul bool
}

// NewArithmeticBlend builds an arithmetic-blend shader. Children must be non-nil (the blend image filter substitutes
// transparent color shaders).
func NewArithmeticBlend(k [4]float32, enforcePremul bool, dst, src Shader) Shader {
	return &ArithmeticShader{dst: dst, src: src, k: k, enforcePremul: enforcePremul}
}

// Dst returns the background (dst) child (for the GPU FP conversion).
func (s *ArithmeticShader) Dst() Shader { return s.dst }

// Src returns the foreground (src) child (for the GPU FP conversion).
func (s *ArithmeticShader) Src() Shader { return s.src }

// K returns the arithmetic coefficients (for the GPU FP conversion).
func (s *ArithmeticShader) K() [4]float32 { return s.k }

// EnforcePremul reports whether the result RGB clamps to alpha (for the GPU FP conversion).
func (s *ArithmeticShader) EnforcePremul() bool { return s.enforcePremul }

// IsOpaque implements Shader.
func (s *ArithmeticShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *ArithmeticShader) IsConstant() bool { return false }

// arithmeticCtx holds the two-child blend's store/load scratch (coords across the dst child, res0 = dst's output across
// the src child) plus the arithmetic coefficients and premul clamp (copied from the shader), in reused pipeline storage
// so the four stages need not capture it. All scratch is written before it is read within each ShadeSpan chunk, so
// pooled reuse needs no zeroing.
type arithmeticCtx struct {
	coords  [2][stride]float32
	res0    [4][stride]float32
	k       [4]float32
	pmClamp float32
}

func (p *Pipeline) nextArithmeticCtx() *arithmeticCtx {
	if p.arithCtxN == len(p.arithCtxs) {
		p.arithCtxs = append(p.arithCtxs, &arithmeticCtx{})
	}
	c := p.arithCtxs[p.arithCtxN]
	p.arithCtxN++
	return c
}

// appendStages runs dst, then src, saving/restoring scratch between them as BlendShader.appendStages does, then applies
// the arithmetic blend equation.
func (s *ArithmeticShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	c := p.nextArithmeticCtx()
	c.k = s.k
	c.pmClamp = 1
	if s.enforcePremul {
		c.pmClamp = 0
	}
	if m.coordsSeeded() {
		p.appendCtx(arithStoreCoordsStage, c)
	}
	if !s.dst.appendStages(p, m) {
		return false
	}
	p.appendCtx(arithStoreRes0Stage, c)
	if m.coordsSeeded() {
		p.appendCtx(arithLoadCoordsStage, c)
	}
	if !s.src.appendStages(p, m) {
		return false
	}
	p.appendCtx(arithBlendStage, c)
	return true
}

// arithStoreCoordsStage saves the seeded coordinates before the dst child overwrites them.
func arithStoreCoordsStage(z *lanes) {
	c := z.ctx.(*arithmeticCtx)
	c.coords[0] = z.r
	c.coords[1] = z.g
}

// arithLoadCoordsStage restores the seeded coordinates for the src child.
func arithLoadCoordsStage(z *lanes) {
	c := z.ctx.(*arithmeticCtx)
	z.r = c.coords[0]
	z.g = c.coords[1]
}

// arithStoreRes0Stage saves the dst child's output across the src child.
func arithStoreRes0Stage(z *lanes) {
	c := z.ctx.(*arithmeticCtx)
	c.res0[0] = z.r
	c.res0[1] = z.g
	c.res0[2] = z.b
	c.res0[3] = z.a
}

// arithBlendStage applies k1*src*dst + k2*src + k3*dst + k4 and re-establishes a valid premul color.
func arithBlendStage(z *lanes) {
	c := z.ctx.(*arithmeticCtx)
	k := c.k
	pmClamp := c.pmClamp
	srcCh := [4]*[stride]float32{&z.r, &z.g, &z.b, &z.a}
	var out [4]float32
	for i := range z.n {
		for ch := range 4 {
			sv := srcCh[ch][i]
			dv := c.res0[ch][i]
			// saturate(k.x*src*dst + k.y*src + k.z*dst + k.w), left-associative plain ops.
			out[ch] = clamp01(k[0]*sv*dv + k[1]*sv + k[2]*dv + k[3])
		}
		// color.rgb = min(color.rgb, max(color.a, pmClamp))
		lim := maxf(out[3], pmClamp)
		z.r[i] = minf(out[0], lim)
		z.g[i] = minf(out[1], lim)
		z.b[i] = minf(out[2], lim)
		z.a[i] = out[3]
	}
}
