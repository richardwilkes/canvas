// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// gradientBase: the shared gradient model — stop preprocessing, the tile stages, and the piecewise-linear color fill
// stages. Gradients are built with byte colors, so interpolation is always the default: unpremul lerp in sRGB,
// premultiplied afterwards (the interpolated-to-dst conversion reduces to a premul stage for sRGB-only content).

package shaders

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// gradDegenerateThreshold is the tolerance below which a gradient's geometry is treated as degenerate.
const gradDegenerateThreshold = float32(1.0) / (1 << 15)

// validGradient reports whether the given colors, positions, and tile mode form a valid gradient (the interpolation
// enums are fixed at their defaults). An explicit position slice, when supplied, must carry exactly one offset per color;
// a mismatched length is rejected here so the downstream stop preprocessing never indexes past the end of positions.
func validGradient(colors []colorcore.Color, pos []float32, tileMode TileMode) bool {
	return len(colors) >= 1 && tileMode < tileModeCount && (pos == nil || len(pos) == len(colors))
}

// colors4f converts byte colors to unpremultiplied float colors.
func colors4f(colors []colorcore.Color) []colorcore.Color4f {
	out := make([]colorcore.Color4f, len(colors))
	for i, c := range colors {
		out[i] = colorcore.Color4fFromColor(c)
	}
	return out
}

// gradientBase holds the state the CPU and GPU gradient evaluators consume.
type gradientBase struct {
	colors          []colorcore.Color4f // stop-fixed, bracketed by [0, 1]
	positions       []float32           // nil when the stops are uniform
	ptsToUnit       geom.Matrix
	tileMode        TileMode
	colorsAreOpaque bool
}

// Colors returns the stop-fixed colors; Positions returns the stop positions (nil when uniform); TileMode returns the
// tile mode; PtsToUnit returns the gradient's unit-space matrix — the descriptors the GPU side consumes.
func (g *gradientBase) Colors() []colorcore.Color4f { return g.colors }

// Positions returns the stop positions, nil when the stops are uniform.
func (g *gradientBase) Positions() []float32 { return g.positions }

// TileMode returns the gradient's tile mode.
func (g *gradientBase) TileMode() TileMode { return g.tileMode }

// PtsToUnit returns the matrix mapping the gradient's geometry to unit space.
func (g *gradientBase) PtsToUnit() geom.Matrix { return g.ptsToUnit }

// IsConstant implements Shader.
func (g *gradientBase) IsConstant() bool { return false }

// initGradientBase initializes the gradient's stop data: insert implicit stops at 0 and 1, force positions monotonic in
// [0,1], detect uniform stops, and dedupe runs of more than two equal stops.
func (g *gradientBase) initGradientBase(colors []colorcore.Color4f, positions []float32, tileMode TileMode, ptsToUnit geom.Matrix) {
	g.ptsToUnit = ptsToUnit
	g.tileMode = tileMode
	g.colorsAreOpaque = true

	descColorCount := len(colors)
	firstStopIsImplicit := false
	lastStopIsImplicit := false
	colorCount := descColorCount
	if positions != nil {
		firstStopIsImplicit = positions[0] > 0
		lastStopIsImplicit = positions[descColorCount-1] != 1
		if firstStopIsImplicit {
			colorCount++
		}
		if lastStopIsImplicit {
			colorCount++
		}
	}

	g.colors = make([]colorcore.Color4f, colorCount)
	dst := g.colors
	if firstStopIsImplicit {
		dst[0] = colors[0]
		dst = dst[1:]
	}
	for i, c := range colors {
		dst[i] = c
		if c.A != 1 {
			g.colorsAreOpaque = false
		}
	}
	if lastStopIsImplicit {
		g.colors[colorCount-1] = colors[descColorCount-1]
	}

	if positions == nil {
		g.positions = nil
		return
	}

	pos := make([]float32, colorCount)
	prev := float32(0)
	pos[0] = prev // force the first pos to 0
	pi := 1

	startIndex := 1
	if firstStopIsImplicit {
		startIndex = 0
	}
	count := descColorCount
	if lastStopIsImplicit {
		count++
	}

	uniformStops := true
	uniformStep := positions[startIndex] - prev
	for i := startIndex; i < count; i++ {
		// Pin the last value to 1.0, and make sure pos is monotonic.
		curr := float32(1)
		if i != descColorCount {
			curr = pinFloat(positions[i], prev, 1)
			// (Upstream also clears lastStopIsImplicit here when a value pins to 1.0 before the last stop, because it
			// serializes that member. Nothing below reads it — colorCount and count were both fixed before this loop —
			// and there is no serialization here, so the write is omitted rather than left dead.)
		}
		if !geom.ScalarNearlyEqual(uniformStep, curr-prev) {
			uniformStops = false
		}
		pos[pi] = curr
		pi++
		prev = curr
	}

	if uniformStops {
		// If the stops are uniform, treat them as implicit.
		g.positions = nil
		return
	}

	// Remove duplicate stops with more than two of the same stop, keeping the leftmost and rightmost stop colors.
	g.positions = pos
	i := 0
	dedupedColorCount := 0
	for j := 1; j <= colorCount; j++ {
		if j != colorCount && g.positions[i] == g.positions[j] {
			continue
		}
		dupStop := j-i > 1
		// Ignore the leftmost stop (i) if it is a non-clamp tilemode with a duplicate stop on t = 0.
		ignoreLeftmost := dupStop && g.tileMode != TileClamp && g.positions[i] == 0
		if !ignoreLeftmost {
			g.positions[dedupedColorCount] = g.positions[i]
			g.colors[dedupedColorCount] = g.colors[i]
			dedupedColorCount++
		}
		// Include the rightmost stop (j-1) only if the stop has a duplicate, ignoring the rightmost stop if it is a
		// non-clamp tilemode with t = 1.
		ignoreRightmost := g.tileMode != TileClamp && g.positions[j-1] == 1
		if dupStop && !ignoreRightmost {
			g.positions[dedupedColorCount] = g.positions[j-1]
			g.colors[dedupedColorCount] = g.colors[j-1]
			dedupedColorCount++
		}
		i = j
	}
	g.colors = g.colors[:dedupedColorCount]
	g.positions = g.positions[:dedupedColorCount]
}

// pinFloat clamps x to the inclusive range [lo, hi], pinning NaN to lo. It spells out max(lo, min(x, hi)) with the
// C++ comparison order it replaces — std::min/std::max return their first argument only when the compare is true, so an
// unordered compare against NaN falls through to x for the min and to lo for the max. The package's own minf/maxf
// propagate the other operand and would pin NaN to hi instead, which a NaN stop position would then smear across every
// following stop (each one pins to the running prev).
func pinFloat(x, lo, hi float32) float32 {
	v := x
	if hi < v { // std::min(x, hi): false for NaN, leaving v = NaN
		v = hi
	}
	if !(lo < v) { // std::max(lo, v): false for NaN, so NaN pins to lo
		v = lo
	}
	return v
}

// isOpaqueBase implements the shared IsOpaque logic (ConicalGradient overrides it).
func (g *gradientBase) isOpaqueBase() bool {
	return g.colorsAreOpaque && g.tileMode != TileDecal
}

// appendBaseStages applies the matrix record with ptsToUnit, lets the concrete gradient append its coordinate stages
// (gradStages may push post-pipeline stages, e.g. apply_vector_mask), then the tile stage, the color fill stages, the
// interpolated-to-dst conversion (a bare premul), and the decal mask check.
func (g *gradientBase) appendBaseStages(p *Pipeline, m *MatrixRec, gradStages func(p *Pipeline)) bool {
	if _, ok := m.apply(p, &g.ptsToUnit); !ok {
		return false
	}

	// The concrete gradient may record deferred tail stages (conical's apply_vector_mask) onto p.postStages; the fill
	// below runs first, then they are spliced on and the field is rewound.
	postStart := len(p.postStages)
	gradStages(p)

	var decalMask *laneMask
	switch g.tileMode {
	case TileMirror:
		p.appendMirrorX1()
	case TileRepeat:
		p.appendRepeatX1()
	case TileDecal, TileClamp:
		if g.tileMode == TileDecal {
			decalMask = p.nextLaneMask()
			p.appendDecalX(decalMask)
		}
		if g.positions == nil {
			// We clamp only when the stops are evenly spaced. If not, there may be hard stops, and clamping ruins hard
			// stops at 0 and/or 1. In that case, we must make sure we're using the general "gradient" stage, which is
			// the only stage that will correctly handle unclamped t.
			p.appendClampX1()
		}
	}

	g.appendFillStages(p)
	// Converting from sRGB-unpremul to sRGB-premul is just a premul stage, skipped entirely when the colors are opaque.
	if !g.colorsAreOpaque {
		p.AppendPremul()
	}

	if decalMask != nil {
		p.appendApplyVectorMask(decalMask)
	}
	p.stages = append(p.stages, p.postStages[postStart:]...)
	p.postStages = p.postStages[:postStart]
	return true
}

// gradStop holds one stop's per-channel factor/bias pair contiguously, so gradientLookup reads its eight floats through
// a single bounds-checked index into one 32-byte record. A SIMD implementation might instead keep eight parallel arrays
// to gather per channel; this scalar lookup is load-latency-bound instead, and the array-of-structs layout halves its
// address math and bounds checks.
type gradStop struct {
	fr, fg, fb, fa float32 // factor per channel
	br, bg, bb, ba float32 // bias per channel
}

// gradientCtx holds per-channel factor/bias lanes so that color = factor*t + bias within each stop's span. The three
// fill stages read their scalar parameters (the 2-stop fast lane's cl/factor, the evenly-spaced lane's gapCount, the
// searched lane's stop count n) from here rather than capturing them, so the fill stage function stays static.
type gradientCtx struct {
	stops    []gradStop
	ts       []float32
	cl       colorcore.Color4f // 2-stop fast lane: left color
	factor   colorcore.Color4f // 2-stop fast lane: right-minus-left
	gapCount float32           // evenly-spaced lane
	n        int               // searched lane: stop count
}

// nextGradCtx hands appendFillStages a distinct gradientCtx for one gradient fill stage, retained on the pipeline so
// its factor/bias/ts storage is reused across pooled compiles. Distinct (not shared) because a BlendShader inlines two
// gradient children into one pipeline, and each fill stage's closure captures its own ctx — a shared one would let the
// second compile clobber the first.
func (p *Pipeline) nextGradCtx() *gradientCtx {
	if p.gradCtxN == len(p.gradCtxs) {
		p.gradCtxs = append(p.gradCtxs, &gradientCtx{})
	}
	c := p.gradCtxs[p.gradCtxN]
	p.gradCtxN++
	return c
}

// ensureF32 returns s resized to length n with the first n elements zeroed, reusing s's backing array when it is large
// enough. It matches make([]float32, n) observably (a zeroed slice of length n) while avoiding the allocation on reuse.
func ensureF32(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n)
	}
	s = s[:n]
	clear(s)
	return s
}

// ensureStops is ensureF32 for the gradient stop records.
func ensureStops(s []gradStop, n int) []gradStop {
	if cap(s) < n {
		return make([]gradStop, n)
	}
	s = s[:n]
	clear(s)
	return s
}

func (c *gradientCtx) addStopColor(stop int, factor, bias colorcore.Color4f) {
	c.stops[stop] = gradStop{
		fr: factor.R, fg: factor.G, fb: factor.B, fa: factor.A,
		br: bias.R, bg: bias.G, bb: bias.B, ba: bias.A,
	}
}

func (c *gradientCtx) addConstColor(stop int, color colorcore.Color4f) {
	c.addStopColor(stop, colorcore.Color4f{}, color)
}

// initStopEvenly computes the factor/bias for one evenly-spaced stop.
func (c *gradientCtx) initStopEvenly(gapCount float32, stop int, left, right colorcore.Color4f) {
	var factor, bias colorcore.Color4f
	t := float32(stop) / gapCount
	factor.R = (right.R - left.R) * gapCount
	factor.G = (right.G - left.G) * gapCount
	factor.B = (right.B - left.B) * gapCount
	factor.A = (right.A - left.A) * gapCount
	bias.R = left.R - factor.R*t
	bias.G = left.G - factor.G*t
	bias.B = left.B - factor.B*t
	bias.A = left.A - factor.A*t
	c.addStopColor(stop, factor, bias)
}

// initStopPos computes the factor/bias for one explicitly-positioned stop.
func (c *gradientCtx) initStopPos(stop int, tL, gapReciprocal float32, left, right colorcore.Color4f) {
	var factor, bias colorcore.Color4f
	factor.R = (right.R - left.R) * gapReciprocal
	factor.G = (right.G - left.G) * gapReciprocal
	factor.B = (right.B - left.B) * gapReciprocal
	factor.A = (right.A - left.A) * gapReciprocal
	bias.R = left.R - factor.R*tL
	bias.G = left.G - factor.G*tL
	bias.B = left.B - factor.B*tL
	bias.A = left.A - factor.A*tL
	c.ts[stop] = tL
	c.addStopColor(stop, factor, bias)
}

// appendFillStages appends the stage(s) that interpolate between stop colors for parameter t. The colors lerp in
// unpremultiplied float (the always-default interpolation mode).
func (g *gradientBase) appendFillStages(p *Pipeline) {
	colors := g.colors
	count := len(colors)

	// The two-stop case with stops at 0 and 1.
	if count == 2 && g.positions == nil {
		cl, cr := colors[0], colors[1]
		ctx := p.nextGradCtx()
		ctx.cl = cl
		ctx.factor = colorcore.Color4f{R: cr.R - cl.R, G: cr.G - cl.G, B: cr.B - cl.B, A: cr.A - cl.A}
		p.appendCtx(gradient2StopStageFn, ctx)
		return
	}

	ctx := p.nextGradCtx()
	ctx.stops = ensureStops(ctx.stops, count+1)

	if g.positions == nil {
		// Handle evenly distributed stops.
		stopCount := count
		gapCount := float32(stopCount - 1)
		cl := colors[0]
		for i := 0; i < stopCount-1; i++ {
			cr := colors[i+1]
			ctx.initStopEvenly(gapCount, i, cl, cr)
			cl = cr
		}
		ctx.addConstColor(stopCount-1, cl)

		// evenly_spaced_gradient
		ctx.gapCount = gapCount
		p.appendCtx(gradientEvenlyStageFn, ctx)
		return
	}

	// Handle arbitrary stops.
	ctx.ts = ensureF32(ctx.ts, count+1)

	// Remove the default stops inserted by the constructor because they are naturally handled by the search method.
	var firstStop, lastStop int
	if count > 2 {
		firstStop = 0
		if colors[0] == colors[1] {
			firstStop = 1
		}
		lastStop = count - 1
		if colors[count-2] == colors[count-1] {
			lastStop = count - 2
		}
	} else {
		firstStop = 0
		lastStop = 1
	}

	stopCount := 0
	tL := g.positions[firstStop]
	cl := colors[firstStop]
	ctx.addConstColor(stopCount, cl)
	stopCount++

	for i := firstStop; i < lastStop; i++ {
		tR := g.positions[i+1]
		cr := colors[i+1]
		if tL < tR {
			cScale := 1 / (tR - tL)
			if !math.IsInf(float64(cScale), 0) && !math.IsNaN(float64(cScale)) {
				ctx.initStopPos(stopCount, tL, cScale, cl, cr)
				stopCount++
			}
		}
		tL = tR
		cl = cr
	}

	ctx.ts[stopCount] = tL
	ctx.addConstColor(stopCount, cl)
	stopCount++

	// gradient (searched)
	ctx.n = stopCount
	p.appendCtx(gradientSearchedStage, ctx)
}

// gradient2StopStage computes the fused two-stop lerp: color = factor*t + cl.
func gradient2StopStage(z *lanes) {
	c := z.ctx.(*gradientCtx)
	cl, factor := c.cl, c.factor
	for i := range z.n {
		t := z.r[i]
		z.r[i] = madf(t, factor.R, cl.R)
		z.g[i] = madf(t, factor.G, cl.G)
		z.b[i] = madf(t, factor.B, cl.B)
		z.a[i] = madf(t, factor.A, cl.A)
	}
}

// gradientEvenlyStage computes the stop index as t scaled by the gap count, for evenly-spaced stops.
func gradientEvenlyStage(z *lanes) {
	c := z.ctx.(*gradientCtx)
	gapCount := c.gapCount
	for i := range z.n {
		t := z.r[i]
		idx := int32(t * gapCount) // trunc_
		gradientLookup(c, idx, t, z, i)
	}
}

// gradientSearchedStage is the general stage for arbitrarily-positioned stops: count the stops at or below t.
func gradientSearchedStage(z *lanes) {
	c := z.ctx.(*gradientCtx)
	n := c.n
	for i := range z.n {
		t := z.r[i]
		idx := int32(0)
		// N.B. The loop starts at 1 because idx 0 is the color to use before the first stop.
		for s := 1; s < n; s++ {
			if t >= c.ts[s] {
				idx++
			}
		}
		gradientLookup(c, idx, t, z, i)
	}
}

// gradientLookup evaluates one stop's factor/bias for one lane.
func gradientLookup(c *gradientCtx, idx int32, t float32, z *lanes, i int) {
	s := &c.stops[idx]
	z.r[i] = madf(t, s.fr, s.br)
	z.g[i] = madf(t, s.fg, s.bg)
	z.b[i] = madf(t, s.fb, s.bb)
	z.a[i] = madf(t, s.fa, s.ba)
}

///////////////////////////////////////////////////////////////////////////////
// degenerate fallbacks

// averageGradientColor returns the area-weighted average of the gradient's stop colors, for the degenerate-gradient
// fallback.
func averageGradientColor(colors []colorcore.Color4f, pos []float32) colorcore.Color4f {
	var blend [4]float32
	colorCount := len(colors)
	add := func(c colorcore.Color4f, w float32) {
		blend[0] += w * c.R
		blend[1] += w * c.G
		blend[2] += w * c.B
		blend[3] += w * c.A
	}
	for i := 0; i < colorCount-1; i++ {
		var w float32
		if pos != nil {
			p0 := pinFloat(pos[i], 0, 1)
			p1 := pinFloat(pos[i+1], p0, 1)
			w = p1 - p0
			if i == 0 && p0 > 0 {
				add(colors[0], p0)
			}
			if i == colorCount-2 && p1 < 1 {
				add(colors[colorCount-1], 1-p1)
			}
		} else {
			w = 1 / float32(colorCount-1)
		}
		c0, c1 := colors[i], colors[i+1]
		add(colorcore.Color4f{R: c0.R + c1.R, G: c0.G + c1.G, B: c0.B + c1.B, A: c0.A + c1.A}, 0.5*w)
	}
	return colorcore.Color4f{R: blend[0], G: blend[1], B: blend[2], A: blend[3]}
}

// makeDegenerateGradient builds the fallback shader for a gradient whose geometry is degenerate.
func makeDegenerateGradient(colors []colorcore.Color4f, pos []float32, mode TileMode) Shader {
	switch mode {
	case TileDecal:
		return Empty()
	case TileRepeat, TileMirror:
		return NewColor4f(averageGradientColor(colors, pos))
	case TileClamp:
		return NewColor4f(colors[len(colors)-1])
	}
	return nil
}
