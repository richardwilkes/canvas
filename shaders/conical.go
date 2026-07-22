// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ConicalGradient: the two-point conical gradient, including the focal-point formulation.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// conicalType classifies a conical gradient's geometry.
type conicalType uint8

const (
	conicalRadial conicalType = iota
	conicalStrip
	conicalFocal
)

// focalData holds the normalized focal-point parameters used by the conical-focal shading path.
type focalData struct {
	r1        float32 // r1 after mapping the focal point to (0, 0)
	focalX    float32 // f
	isSwapped bool
}

func (f *focalData) isFocalOnCircle() bool { return geom.ScalarNearlyZero(1 - f.r1) }
func (f *focalData) isWellBehaved() bool   { return !f.isFocalOnCircle() && f.r1 > 1 }
func (f *focalData) isSwappedFlag() bool   { return f.isSwapped }
func (f *focalData) isNativelyFocal() bool { return geom.ScalarNearlyZero(f.focalX) }

// set normalizes the radii/matrix so the focal point maps to the origin and the end circle crosses (1, 0). Returns
// false when the focal transform is not constructible.
func (f *focalData) set(r0, r1 float32, matrix *geom.Matrix) bool {
	f.isSwapped = false
	f.focalX = r0 / (r0 - r1)
	if geom.ScalarNearlyZero(f.focalX - 1) {
		// Swap r0 and r1. Only r1 is read below, so the other half of the swap is elided: the swapped-in r0 (the old
		// r1) is 0 here, which is why focalX becomes 0.
		matrix.PostTranslate(-1, 0)
		matrix.PostScale(-1, 1)
		r1 = r0
		f.focalX = 0
		f.isSwapped = true
	}

	// Map {focal point, (1, 0)} to {(0, 0), (1, 0)}
	focalMatrix, ok := polyToPoly2(
		geom.Point{X: f.focalX, Y: 0}, geom.Point{X: 1, Y: 0},
		geom.Point{X: 0, Y: 0}, geom.Point{X: 1, Y: 0},
	)
	if !ok {
		return false
	}
	matrix.PostConcat(&focalMatrix)
	f.r1 = r1 / geom.ScalarAbs(1-f.focalX) // focalMatrix has a scale of 1/(1-f)

	// The following transformations are just to accelerate the shader computation by saving some arithmetic operations.
	if f.isFocalOnCircle() {
		matrix.PostScale(0.5, 0.5)
	} else {
		matrix.PostScale(f.r1/(f.r1*f.r1-1), 1/sqrtf(geom.ScalarAbs(f.r1*f.r1-1)))
	}
	matrix.PostScale(geom.ScalarAbs(1-f.focalX), geom.ScalarAbs(1-f.focalX)) // scale |1 - f|
	return true
}

// poly2 returns the matrix mapping (0,0)-(1,0) to (p0, p1).
func poly2(p0, p1 geom.Point) geom.Matrix {
	var m geom.Matrix
	m.SetAll(
		p1.Y-p0.Y, p1.X-p0.X, p0.X,
		p0.X-p1.X, p1.Y-p0.Y, p0.Y,
		0, 0, 1,
	)
	return m
}

// polyToPoly2 returns the matrix mapping (src0, src1) to (dst0, dst1).
func polyToPoly2(src0, src1, dst0, dst1 geom.Point) (geom.Matrix, bool) {
	srcM := poly2(src0, src1)
	inv, ok := srcM.Invert()
	if !ok {
		return geom.Matrix{}, false
	}
	dstM := poly2(dst0, dst1)
	var out geom.Matrix
	out.SetConcat(&dstM, &inv)
	return out, true
}

// ConicalGradient is a two-point conical gradient shader.
type ConicalGradient struct {
	gradientBase
	center1, center2 geom.Point
	radius1, radius2 float32
	kind             conicalType
	focal            focalData
}

// mapToUnitX returns the matrix mapping startCenter to (0,0) and endCenter to (1,0).
func mapToUnitX(startCenter, endCenter geom.Point) (geom.Matrix, bool) {
	return polyToPoly2(startCenter, endCenter, geom.Point{X: 0, Y: 0}, geom.Point{X: 1, Y: 0})
}

// newConical builds a ConicalGradient (or a radial-gradient equivalent for the concentric-circles special case),
// classifying its geometry as radial, strip, or focal.
func newConical(c0 geom.Point, r0 float32, c1 geom.Point, r1 float32, colors []colorcore.Color4f, pos []float32, tileMode TileMode, localMatrix *geom.Matrix) Shader {
	var gradientMatrix geom.Matrix
	var gradientType conicalType

	d := geom.Point{X: c0.X - c1.X, Y: c0.Y - c1.Y}
	if geom.ScalarNearlyZero(d.Length()) {
		if geom.ScalarNearlyZero(maxf(r0, r1)) || geom.ScalarNearlyEqual(r0, r1) {
			// Degenerate case; avoid dividing by zero. Should have been caught by caller but just in case, recheck
			// here.
			return nil
		}
		// Concentric case: we can pretend we're radial (with a tiny twist).
		scale := 1 / maxf(r0, r1)
		gradientMatrix.SetTranslate(-c1.X, -c1.Y)
		gradientMatrix.PostScale(scale, scale)
		gradientType = conicalRadial
	} else {
		mx, ok := mapToUnitX(c0, c1)
		if !ok {
			// Degenerate case.
			return nil
		}
		gradientMatrix = mx
		if geom.ScalarNearlyZero(r1 - r0) {
			gradientType = conicalStrip
		} else {
			gradientType = conicalFocal
		}
	}

	var focal focalData
	if gradientType == conicalFocal {
		dCenter := d.Length()
		if !focal.set(r0/dCenter, r1/dCenter, &gradientMatrix) {
			return nil
		}
	}

	g := &ConicalGradient{
		center1: c0,
		center2: c1,
		radius1: r0,
		radius2: r1,
		kind:    gradientType,
		focal:   focal,
	}
	g.initGradientBase(colors, pos, tileMode, gradientMatrix)
	return withLocalMatrixPtr(g, localMatrix)
}

// NewTwoPointConicalGradient builds a two-point conical gradient shader from byte colors; localMatrix may be nil.
func NewTwoPointConicalGradient(start geom.Point, startRadius float32, end geom.Point, endRadius float32, colors []colorcore.Color, pos []float32, tileMode TileMode, localMatrix *geom.Matrix) Shader {
	if startRadius < 0 || endRadius < 0 {
		return nil
	}
	if !validGradient(colors, pos, tileMode) {
		return nil
	}
	c4f := colors4f(colors)
	d := geom.Point{X: start.X - end.X, Y: start.Y - end.Y}
	if geom.ScalarNearlyZeroTolerance(d.Length(), gradDegenerateThreshold) {
		// The centers are the same: a radial variant, an actual radial gradient (startRadius == 0), or fully degenerate
		// (startRadius == endRadius).
		switch {
		case geom.ScalarNearlyEqualTolerance(startRadius, endRadius, gradDegenerateThreshold):
			// The interpolation region area approaches zero: default degenerate behavior, except when mode = clamp and
			// the radii > 0.
			if tileMode == TileClamp && endRadius > gradDegenerateThreshold {
				// The interpolation region becomes an infinitely thin ring at the radius: first color repeated from p=0
				// to 1, then a hard stop to the last color at p=1.
				circleColors := []colorcore.Color{colors[0], colors[0], colors[len(colors)-1]}
				return NewRadialGradient(start, endRadius, circleColors, []float32{0, 1, 1},
					tileMode, localMatrix)
			}
			return withLocalMatrixPtr(makeDegenerateGradient(c4f, pos, tileMode), localMatrix)
		case geom.ScalarNearlyZeroTolerance(startRadius, gradDegenerateThreshold):
			// We can treat this gradient as radial, which is faster.
			return NewRadialGradient(start, endRadius, colors, pos, tileMode, localMatrix)
		}
		// Else it's the 2pt conical radial variant with no degenerate radii; fall through.
	}

	if localMatrix != nil {
		if _, ok := localMatrix.Invert(); !ok {
			return nil
		}
	}
	// A single color needs a second stop to interpolate against.
	if len(c4f) == 1 {
		c4f = []colorcore.Color4f{c4f[0], c4f[0]}
		pos = nil
	}
	return newConical(start, startRadius, end, endRadius, c4f, pos, tileMode, localMatrix)
}

// Center1 returns the start center; Center2 the end center; Radius1/Radius2 the radii.
func (g *ConicalGradient) Center1() geom.Point { return g.center1 }

// Center2 returns the end center.
func (g *ConicalGradient) Center2() geom.Point { return g.center2 }

// Radius1 returns the start radius.
func (g *ConicalGradient) Radius1() float32 { return g.radius1 }

// Radius2 returns the end radius.
func (g *ConicalGradient) Radius2() float32 { return g.radius2 }

// IsOpaque implements Shader: areas outside the cone are left untouched, so the shader is never treated as opaque.
func (g *ConicalGradient) IsOpaque() bool { return false }

// ConicalKind is the exported form of conicalType, for the GPU fragment-processor conversion.
type ConicalKind uint8

// ConicalKind values (matching the internal conicalType order).
const (
	ConicalKindRadial ConicalKind = ConicalKind(conicalRadial)
	ConicalKindStrip  ConicalKind = ConicalKind(conicalStrip)
	ConicalKindFocal  ConicalKind = ConicalKind(conicalFocal)
)

// Kind returns the gradient's geometry classification.
func (g *ConicalGradient) Kind() ConicalKind { return ConicalKind(g.kind) }

// CenterX1 returns the distance between the two centers.
func (g *ConicalGradient) CenterX1() float32 { return g.centerX1() }

// DiffRadius returns radius2 - radius1.
func (g *ConicalGradient) DiffRadius() float32 { return g.radius2 - g.radius1 }

// FocalR1 returns r1 after mapping the focal point to the origin.
func (g *ConicalGradient) FocalR1() float32 { return g.focal.r1 }

// FocalX returns the normalized focal-point x coordinate.
func (g *ConicalGradient) FocalX() float32 { return g.focal.focalX }

// FocalIsSwapped reports whether the focal transform swapped the two radii.
func (g *ConicalGradient) FocalIsSwapped() bool { return g.focal.isSwapped }

// FocalOnCircle reports whether the focal point lies on the end circle.
func (g *ConicalGradient) FocalOnCircle() bool { return g.focal.isFocalOnCircle() }

// FocalWellBehaved reports whether the focal-point math is numerically well-behaved.
func (g *ConicalGradient) FocalWellBehaved() bool { return g.focal.isWellBehaved() }

// FocalNativelyFocal reports whether the focal point is already at the origin.
func (g *ConicalGradient) FocalNativelyFocal() bool { return g.focal.isNativelyFocal() }

// centerX1 returns the distance between the two centers.
func (g *ConicalGradient) centerX1() float32 {
	d := geom.Point{X: g.center2.X - g.center1.X, Y: g.center2.Y - g.center1.Y}
	return d.Length()
}

// appendStages runs the base gradient stages, then appends the conical-specific coordinate stages. The coordinate
// stages read their scalars (the strip lane's r0^2, the focal lanes' 1/r1 and focal x) from the immutable gradient via
// z.ctx rather than capturing them, so each stage function stays static; recomputing those scalars once per ShadeSpan
// chunk from the gradient's fields is byte-identical to computing them once at compile time.
func (g *ConicalGradient) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	return g.appendBaseStages(p, &m, func(p *Pipeline) {
		dRadius := g.radius2 - g.radius1

		if g.kind == conicalRadial {
			p.appendXYToRadius()
			// Tiny twist: radial computes a t for [0, r2], but we want a t for [r1, r2].
			scale := maxf(g.radius1, g.radius2) / dRadius
			bias := -g.radius1 / dRadius
			remap := geom.TranslateMatrix(bias, 0)
			var sc geom.Matrix
			sc.SetScale(scale, 1)
			remap.PreConcat(&sc)
			p.appendMatrix(&remap)
			return
		}

		if g.kind == conicalStrip {
			mask := p.nextLaneMask()
			p.appendCtx(conicalStripStage, g)
			p.appendMask2PtConicalNaN(mask)
			p.appendPostCtx(maskApplyStage, mask)
			return
		}

		f := g.focal
		switch {
		case f.isFocalOnCircle():
			p.append(conicalFocalOnCircleStage)
		case f.isWellBehaved():
			p.appendCtx(conicalWellBehavedStage, g)
		case f.isSwappedFlag() || 1-f.focalX < 0:
			p.appendCtx(conicalSmallerStage, g)
		default:
			p.appendCtx(conicalGreaterStage, g)
		}

		var mask *laneMask
		if !f.isWellBehaved() {
			mask = p.nextLaneMask()
			p.appendMask2PtConicalDegenerates(mask)
		}
		if 1-f.focalX < 0 {
			p.append(conicalNegateXStage)
		}
		if !f.isNativelyFocal() {
			p.appendCtx(conicalCompensateFocalStage, g)
		}
		if f.isSwappedFlag() {
			p.append(conicalUnswapStage)
		}
		if !f.isWellBehaved() {
			p.appendPostCtx(maskApplyStage, mask)
		}
	})
}

// conicalStripStage computes t = x + sqrt(r0^2 - y^2) for the conical-strip lane.
func conicalStripStage(z *lanes) {
	g := z.ctx.(*ConicalGradient)
	scaledR0 := g.radius1 / g.centerX1()
	p0 := scaledR0 * scaledR0
	for i := range z.n {
		z.r[i] += sqrtf(p0 - z.g[i]*z.g[i])
	}
}

// conicalFocalOnCircleStage computes t = x + y^2/x for the focal-on-circle lane.
func conicalFocalOnCircleStage(z *lanes) {
	for i := range z.n {
		z.r[i] += z.g[i] * z.g[i] / z.r[i]
	}
}

// conicalWellBehavedStage computes t = sqrt(x^2 + y^2) - x/r1 for the well-behaved focal lane.
func conicalWellBehavedStage(z *lanes) {
	fp0 := 1 / z.ctx.(*ConicalGradient).focal.r1
	for i := range z.n {
		x := z.r[i]
		z.r[i] = sqrtf(x*x+z.g[i]*z.g[i]) - x*fp0
	}
}

// conicalSmallerStage computes t = -sqrt(x^2 - y^2) - x/r1 for the smaller-root focal lane.
func conicalSmallerStage(z *lanes) {
	fp0 := 1 / z.ctx.(*ConicalGradient).focal.r1
	for i := range z.n {
		x := z.r[i]
		z.r[i] = -sqrtf(x*x-z.g[i]*z.g[i]) - x*fp0
	}
}

// conicalGreaterStage computes t = sqrt(x^2 - y^2) - x/r1 for the greater-root focal lane.
func conicalGreaterStage(z *lanes) {
	fp0 := 1 / z.ctx.(*ConicalGradient).focal.r1
	for i := range z.n {
		x := z.r[i]
		z.r[i] = sqrtf(x*x-z.g[i]*z.g[i]) - x*fp0
	}
}

// conicalNegateXStage negates the computed t.
func conicalNegateXStage(z *lanes) {
	for i := range z.n {
		z.r[i] = -z.r[i]
	}
}

// conicalCompensateFocalStage computes t += f to compensate for a non-origin focal point.
func conicalCompensateFocalStage(z *lanes) {
	fp1 := z.ctx.(*ConicalGradient).focal.focalX
	for i := range z.n {
		z.r[i] += fp1
	}
}

// conicalUnswapStage computes t = 1 - t to undo the radii swap.
func conicalUnswapStage(z *lanes) {
	for i := range z.n {
		z.r[i] = 1 - z.r[i]
	}
}
