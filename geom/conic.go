// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "math"

// MaxConicToQuadPOW2 is the limit on the power-of-two *exponent* passed to ChopIntoQuadsPOW2, not on the quad count
// itself: at this limit the conic becomes 1<<5 = 32 quads occupying 2*32+1 = 65 points. Size a pts buffer from
// MaxConicToQuadPointCount, never from this constant.
const MaxConicToQuadPOW2 = 5

// MaxConicToQuadPointCount is the number of points ChopIntoQuadsPOW2 can write, i.e. the size a pts buffer must have
// to accept any pow2 up to MaxConicToQuadPOW2.
const MaxConicToQuadPointCount = 2*(1<<MaxConicToQuadPOW2) + 1

// MaxConicsForArc is the maximum number of conics BuildUnitArc can emit.
const MaxConicsForArc = 5

// PathDirection is the winding direction of a path's contour.
type PathDirection uint8

const (
	// DirectionCW is clockwise in this package's y-down coordinate system.
	DirectionCW PathDirection = iota
	// DirectionCCW is counter-clockwise.
	DirectionCCW
)

// RotationDirection is the sweep direction used when building an arc.
type RotationDirection uint8

const (
	// RotationCW rotates clockwise.
	RotationCW RotationDirection = iota
	// RotationCCW rotates counter-clockwise.
	RotationCCW
)

// Conic is a rational quadratic Bezier with weight W. W in (0, 1) describes an elliptical arc of less than 180 degrees;
// W == 1 is a parabola (plain quad); W > 1 is hyperbolic.
type Conic struct {
	Pts [3]Point
	W   float32
}

// MakeConic builds a Conic from three points and a weight.
func MakeConic(p0, p1, p2 Point, w float32) Conic {
	return Conic{Pts: [3]Point{p0, p1, p2}, W: w}
}

// conicCoeff holds the numerator and denominator quad coefficients for rational evaluation of a conic.
type conicCoeff struct {
	numer quadCoeff
	denom quadCoeff
}

func makeConicCoeff(c *Conic) conicCoeff {
	var cc conicCoeff
	p1wx := c.Pts[1].X * c.W
	p1wy := c.Pts[1].Y * c.W
	cc.numer.cx = c.Pts[0].X
	cc.numer.cy = c.Pts[0].Y
	cc.numer.ax = c.Pts[2].X - 2*p1wx + c.Pts[0].X
	cc.numer.ay = c.Pts[2].Y - 2*p1wy + c.Pts[0].Y
	cc.numer.bx = 2 * (p1wx - c.Pts[0].X)
	cc.numer.by = 2 * (p1wy - c.Pts[0].Y)
	// Denominator: C = 1, B = 2*(w - 1), A = -B.
	cc.denom.cx = 1
	cc.denom.cy = 1
	cc.denom.bx = 2 * (c.W - 1)
	cc.denom.by = cc.denom.bx
	cc.denom.ax = -cc.denom.bx
	cc.denom.ay = cc.denom.ax
	return cc
}

// EvalAt evaluates the conic at parameter t.
func (c *Conic) EvalAt(t float32) Point {
	cc := makeConicCoeff(c)
	n := cc.numer.eval(t)
	d := cc.denom.eval(t)
	return Point{X: n.X / d.X, Y: n.Y / d.Y}
}

// EvalTangentAt returns the tangent vector of the conic at parameter t.
func (c *Conic) EvalTangentAt(t float32) Point {
	// The derivative is the zero vector when t is 0 or 1 and the control point equals that end point; use the conic
	// endpoints in that case.
	if (t == 0 && c.Pts[0] == c.Pts[1]) || (t == 1 && c.Pts[1] == c.Pts[2]) {
		return c.Pts[2].Sub(c.Pts[0])
	}
	p20x := c.Pts[2].X - c.Pts[0].X
	p20y := c.Pts[2].Y - c.Pts[0].Y
	p10x := c.Pts[1].X - c.Pts[0].X
	p10y := c.Pts[1].Y - c.Pts[0].Y
	cx := c.W * p10x
	cy := c.W * p10y
	ax := c.W*p20x - p20x
	ay := c.W*p20y - p20y
	bx := p20x - cx - cx
	by := p20y - cy - cy
	q := quadCoeff{ax: ax, ay: ay, bx: bx, by: by, cx: cx, cy: cy}
	return q.eval(t)
}

// subdivideWValue returns the weight of each half after a midpoint subdivision.
func subdivideWValue(w float32) float32 {
	return ScalarSqrt(0.5 + w*0.5)
}

// Chop splits the conic at its parametric midpoint into two conics, using an overflow-safe computation.
func (c *Conic) Chop(dst *[2]Conic) {
	scale := ScalarInvert(1 + c.W)
	ws := c.W * scale
	t0x := c.Pts[0].X * scale
	t0y := c.Pts[0].Y * scale
	t1x := c.Pts[1].X * ws
	t1y := c.Pts[1].Y * ws
	t2x := c.Pts[2].X * scale
	t2y := c.Pts[2].Y * scale
	p1 := Point{X: t0x + t1x, Y: t0y + t1y}
	p3 := Point{X: t1x + t2x, Y: t1y + t2y}
	// p2 = (t0 + 2*t1 + t2) / 2, with the halving distributed before the sum to avoid overflow.
	p2 := Point{X: 0.5*t0x + t1x + 0.5*t2x, Y: 0.5*t0y + t1y + 0.5*t2y}
	dst[0].Pts[0] = c.Pts[0]
	dst[0].Pts[1] = p1
	dst[0].Pts[2] = p2
	dst[1].Pts[0] = p2
	dst[1].Pts[1] = p3
	dst[1].Pts[2] = c.Pts[2]
	w := subdivideWValue(c.W)
	dst[0].W = w
	dst[1].W = w
}

// p3dInterp interpolates one coordinate lane (x, y or z) of three homogeneous points.
func p3dInterp(src, dst *[9]float32, offset int, t float32) {
	ab := ScalarInterp(src[offset], src[offset+3], t)
	bc := ScalarInterp(src[offset+3], src[offset+6], t)
	dst[offset] = ab
	dst[offset+3] = ScalarInterp(ab, bc, t)
	dst[offset+6] = bc
}

// ChopAt splits the conic at an arbitrary t using homogeneous interpolation. Returns false if any produced value is
// non-finite.
func (c *Conic) ChopAt(t float32, dst *[2]Conic) bool {
	// tmp holds three homogeneous points as x,y,z triples.
	var tmp, tmp2 [9]float32
	tmp = [9]float32{
		c.Pts[0].X, c.Pts[0].Y, 1,
		c.Pts[1].X * c.W, c.Pts[1].Y * c.W, c.W,
		c.Pts[2].X, c.Pts[2].Y, 1,
	}
	p3dInterp(&tmp, &tmp2, 0, t) // x lane
	p3dInterp(&tmp, &tmp2, 1, t) // y lane
	p3dInterp(&tmp, &tmp2, 2, t) // z lane
	dst[0].Pts[0] = c.Pts[0]
	dst[0].Pts[1] = Point{X: tmp2[0] / tmp2[2], Y: tmp2[1] / tmp2[2]}
	dst[0].Pts[2] = Point{X: tmp2[3] / tmp2[5], Y: tmp2[4] / tmp2[5]}
	dst[1].Pts[0] = dst[0].Pts[2]
	dst[1].Pts[1] = Point{X: tmp2[6] / tmp2[8], Y: tmp2[7] / tmp2[8]}
	dst[1].Pts[2] = c.Pts[2]
	// Renormalize to standard form (w0 == w2 == 1): w1 /= sqrt(w0*w2), knowing dst[0].w0 == 1 and dst[1].w2 == 1.
	root := ScalarSqrt(tmp2[5])
	dst[0].W = tmp2[2] / root
	dst[1].W = tmp2[8] / root
	return IsFinite(
		dst[0].Pts[0].X, dst[0].Pts[0].Y, dst[0].Pts[1].X, dst[0].Pts[1].Y, dst[0].Pts[2].X, dst[0].Pts[2].Y,
		dst[0].W,
		dst[1].Pts[0].X, dst[1].Pts[0].Y, dst[1].Pts[1].X, dst[1].Pts[1].Y, dst[1].Pts[2].X, dst[1].Pts[2].Y,
		dst[1].W,
	)
}

// ChopAtRange extracts the sub-conic over [t1, t2].
func (c *Conic) ChopAtRange(dst *Conic, t1, t2 float32) {
	if t1 == 0 || t2 == 1 {
		if t1 == 0 && t2 == 1 {
			*dst = *c
			return
		}
		var pair [2]Conic
		t := t1
		if t1 == 0 {
			t = t2
		}
		if c.ChopAt(t, &pair) {
			idx := 0
			if t1 != 0 {
				idx = 1
			}
			*dst = pair[idx]
			return
		}
	}
	cc := makeConicCoeff(c)
	aXY := cc.numer.eval(t1)
	aZZ := cc.denom.eval(t1)
	midT := (t1 + t2) / 2
	dXY := cc.numer.eval(midT)
	dZZ := cc.denom.eval(midT)
	cXY := cc.numer.eval(t2)
	cZZ := cc.denom.eval(t2)
	bXY := Point{X: 2*dXY.X - (aXY.X+cXY.X)*0.5, Y: 2*dXY.Y - (aXY.Y+cXY.Y)*0.5}
	bZZx := 2*dZZ.X - (aZZ.X+cZZ.X)*0.5
	dst.Pts[0] = Point{X: aXY.X / aZZ.X, Y: aXY.Y / aZZ.Y}
	dst.Pts[1] = Point{X: bXY.X / bZZx, Y: bXY.Y / bZZx}
	dst.Pts[2] = Point{X: cXY.X / cZZ.X, Y: cXY.Y / cZZ.Y}
	dst.W = bZZx / ScalarSqrt(aZZ.X*cZZ.X)
}

func badConicW(w float32) bool {
	return w < 0 || !IsFinite(w)
}

// ComputeQuadPOW2 returns how many power-of-two subdivisions are needed to approximate the conic with quads within tol.
func (c *Conic) ComputeQuadPOW2(tol float32) int {
	if tol < 0 || !IsFinite(tol) ||
		!IsFinite(c.Pts[0].X, c.Pts[0].Y, c.Pts[1].X, c.Pts[1].Y, c.Pts[2].X, c.Pts[2].Y) ||
		badConicW(c.W) {
		return 0
	}
	a := c.W - 1
	k := a / (4 * (2 + a))
	x := k * (c.Pts[0].X - 2*c.Pts[1].X + c.Pts[2].X)
	y := k * (c.Pts[0].Y - 2*c.Pts[1].Y + c.Pts[2].Y)
	err := ScalarSqrt(x*x + y*y)
	pow2 := 0
	for ; pow2 < MaxConicToQuadPOW2; pow2++ {
		if err <= tol {
			break
		}
		err *= 0.25
	}
	return pow2
}

// between reports whether b lies between a and c: (a <= b <= c) || (a >= b >= c).
func between(a, b, c float32) bool {
	return (a-b)*(c-b) <= 0
}

// subdivideConic recursively chops into quads, writing 2 points per level-0 leaf, with the y-order fixups that keep
// monotonic input monotonic in the output.
func subdivideConic(src *Conic, pts []Point, level int) []Point {
	if level == 0 {
		pts[0] = src.Pts[1]
		pts[1] = src.Pts[2]
		return pts[2:]
	}
	var dst [2]Conic
	src.Chop(&dst)
	startY := src.Pts[0].Y
	endY := src.Pts[2].Y
	if between(startY, src.Pts[1].Y, endY) {
		// If the input is monotonic and the output is not, the scan converter hangs; ensure the chopped conics maintain
		// their y-order.
		midY := dst[0].Pts[2].Y
		if !between(startY, midY, endY) {
			// The computed midpoint is outside the ends; move it to the closer one.
			closerY := endY
			if ScalarAbs(midY-startY) < ScalarAbs(midY-endY) {
				closerY = startY
			}
			dst[0].Pts[2].Y = closerY
			dst[1].Pts[0].Y = closerY
		}
		if !between(startY, dst[0].Pts[1].Y, dst[0].Pts[2].Y) {
			// The first control is not between the start and end; put it at the start, reducing to a line.
			dst[0].Pts[1].Y = startY
		}
		if !between(dst[1].Pts[0].Y, dst[1].Pts[1].Y, endY) {
			// The second control is not between the start and end; put it at the end, reducing to a line.
			dst[1].Pts[1].Y = endY
		}
	}
	level--
	pts = subdivideConic(&dst[0], pts, level)
	return subdivideConic(&dst[1], pts, level)
}

// ChopIntoQuadsPOW2 writes 1 + 2*(1<<pow2) points describing a chain of 1<<pow2 quads into pts, returning the quad
// count.
func (c *Conic) ChopIntoQuadsPOW2(pts []Point, pow2 int) int {
	if badConicW(c.W) {
		pow2 = 0
	}
	pts[0] = c.Pts[0]
	if pow2 == MaxConicToQuadPOW2 {
		// An extreme weight can generate many quads; check whether the first chop produces a pair of lines and emit
		// just those instead.
		var dst [2]Conic
		c.Chop(&dst)
		if dst[0].Pts[1].EqualsWithinTolerance(dst[0].Pts[2]) &&
			dst[1].Pts[0].EqualsWithinTolerance(dst[1].Pts[1]) {
			pts[1] = dst[0].Pts[1]
			pts[2] = dst[0].Pts[1]
			pts[3] = dst[0].Pts[1]
			pts[4] = dst[1].Pts[2]
			pow2 = 1
			return c.finishQuads(pts, pow2)
		}
	}
	subdivideConic(c, pts[1:], pow2)
	return c.finishQuads(pts, pow2)
}

func (c *Conic) finishQuads(pts []Point, pow2 int) int {
	quadCount := 1 << pow2
	ptCount := 2*quadCount + 1
	allFinite := true
	for i := 0; i < ptCount; i++ {
		if !IsFinite(pts[i].X, pts[i].Y) {
			allFinite = false
			break
		}
	}
	if !allFinite {
		// A non-finite point was generated; pin the interior points to the middle of the hull, since the first and last
		// are already on the hull ends.
		for i := 1; i < ptCount-1; i++ {
			pts[i] = c.Pts[1]
		}
	}
	return quadCount
}

// conicDerivCoeff computes the quadratic coefficients of the conic's derivative.
func conicDerivCoeff(src0, src1, src2, w float32) (a, b, c float32) {
	p20 := src2 - src0
	p10 := src1 - src0
	wP10 := w * p10
	a = w*p20 - p20
	b = p20 - 2*wP10
	c = wP10
	return a, b, c
}

func conicFindExtrema(src0, src1, src2, w float32) (float32, bool) {
	a, b, c := conicDerivCoeff(src0, src1, src2, w)
	var roots [2]float32
	n := FindUnitQuadRoots(a, b, c, &roots)
	if n == 1 {
		return roots[0], true
	}
	return 0, false
}

// FindXExtrema returns the parameter t of the conic's local x extremum, if it has one.
func (c *Conic) FindXExtrema() (float32, bool) {
	return conicFindExtrema(c.Pts[0].X, c.Pts[1].X, c.Pts[2].X, c.W)
}

// FindYExtrema returns the parameter t of the conic's local y extremum, if it has one.
func (c *Conic) FindYExtrema() (float32, bool) {
	return conicFindExtrema(c.Pts[0].Y, c.Pts[1].Y, c.Pts[2].Y, c.W)
}

// ChopAtXExtrema splits the conic at its local x extremum, flattening the middle to the exact extremum on success.
func (c *Conic) ChopAtXExtrema(dst *[2]Conic) bool {
	t, ok := c.FindXExtrema()
	if !ok {
		return false
	}
	if !c.ChopAt(t, dst) {
		// If the chop cannot produce finite values, do not chop.
		return false
	}
	value := dst[0].Pts[2].X
	dst[0].Pts[1].X = value
	dst[1].Pts[0].X = value
	dst[1].Pts[1].X = value
	return true
}

// ChopAtYExtrema splits the conic at its local y extremum, flattening the middle to the exact extremum on success.
func (c *Conic) ChopAtYExtrema(dst *[2]Conic) bool {
	t, ok := c.FindYExtrema()
	if !ok {
		return false
	}
	if !c.ChopAt(t, dst) {
		return false
	}
	value := dst[0].Pts[2].Y
	dst[0].Pts[1].Y = value
	dst[1].Pts[0].Y = value
	dst[1].Pts[1].Y = value
	return true
}

// ComputeTightBounds returns the conic's exact bounds, including any interior x/y extrema.
func (c *Conic) ComputeTightBounds() Rect {
	var pts [4]Point
	pts[0] = c.Pts[0]
	pts[1] = c.Pts[2]
	count := 2
	if t, ok := c.FindXExtrema(); ok {
		pts[count] = c.EvalAt(t)
		count++
	}
	if t, ok := c.FindYExtrema(); ok {
		pts[count] = c.EvalAt(t)
		count++
	}
	return BoundsOrEmpty(pts[:count])
}

// ComputeFastBounds returns the conic's control-point bounds, a cheap superset of its tight bounds.
func (c *Conic) ComputeFastBounds() Rect {
	return BoundsOrEmpty(c.Pts[:])
}

// BuildUnitArc appends up to MaxConicsForArc conics sweeping from uStart to uStop (unit vectors) in the given
// direction, optionally transformed by userMatrix. Returns the conics used, which may be zero for degenerate sweeps.
func BuildUnitArc(uStart, uStop Point, dir RotationDirection, userMatrix *Matrix, dst *[MaxConicsForArc]Conic) int {
	// Rotate by x,y so that uStart is (1, 0).
	x := uStart.Dot(uStop)
	y := uStart.Cross(uStop)
	absY := ScalarAbs(y)
	// Check for (effectively) coincident vectors: angle nearly 0 or nearly 180 (y == 0), using the dot product (x > 0)
	// to distinguish between the two.
	if absY <= ScalarNearlyZeroTol && x > 0 && ((y >= 0 && dir == RotationCW) || (y <= 0 && dir == RotationCCW)) {
		return 0
	}
	if dir == RotationCCW {
		y = -y
	}
	// Determine the quadrant [x, y] lies in: one conic per quadrant of the circle.
	quadrant := 0
	switch {
	case y == 0:
		quadrant = 2 // 180 degrees
	case x == 0:
		if y > 0 {
			quadrant = 1 // 90 degrees
		} else {
			quadrant = 3 // 270 degrees
		}
	default:
		if y < 0 {
			quadrant += 2
		}
		if (x < 0) != (y < 0) {
			quadrant++
		}
	}
	quadrantPts := [8]Point{
		{X: 1, Y: 0},
		{X: 1, Y: 1},
		{X: 0, Y: 1},
		{X: -1, Y: 1},
		{X: -1, Y: 0},
		{X: -1, Y: -1},
		{X: 0, Y: -1},
		{X: 1, Y: -1},
	}
	const quadrantWeight = ScalarRoot2Over2
	conicCount := quadrant
	for i := 0; i < conicCount; i++ {
		dst[i] = MakeConic(quadrantPts[i*2], quadrantPts[i*2+1], quadrantPts[(i*2+2)%8], quadrantWeight)
	}
	// Now compute any remaining (sub-90-degree) arc for the last conic.
	finalP := Point{X: x, Y: y}
	lastQ := quadrantPts[quadrant*2] // already a unit vector
	dot := lastQ.Dot(finalP)
	if ScalarIsNaN(dot) {
		return 0
	}
	if dot < 1 {
		offCurve := Point{X: lastQ.X + x, Y: lastQ.Y + y}
		// The bisector vector rescaled to the off-curve point: length = 1/cos(theta/2) via the half-angle identity, and
		// cos(theta/2) is also the conic weight.
		cosThetaOver2 := ScalarSqrt((1 + dot) / 2)
		offCurve.SetLength(ScalarInvert(cosThetaOver2))
		if !lastQ.EqualsWithinTolerance(offCurve) {
			dst[conicCount] = MakeConic(lastQ, offCurve, finalP, cosThetaOver2)
			conicCount++
		}
	}
	// Handle counter-clockwise and the initial unitStart rotation.
	var matrix Matrix
	matrix.SetSinCos(uStart.Y, uStart.X)
	if dir == RotationCCW {
		matrix.PreScale(1, -1)
	}
	if userMatrix != nil {
		matrix.PostConcat(userMatrix)
	}
	for i := 0; i < conicCount; i++ {
		matrix.MapPointsInPlace(dst[i].Pts[:])
	}
	return conicCount
}

// TransformW returns the weight a conic should have after its points are mapped through matrix. For non-perspective
// matrices the weight is unchanged; under perspective the conic is lifted to homogeneous 3-space (control point scaled
// by w), the matrix's perspective row is applied, and w' = sqrt(w1*w1 / (w0*w2)), computed in doubles to handle small
// numerators/denominators.
func TransformW(pts []Point, w float32, matrix *Matrix) float32 {
	if !matrix.HasPerspective() {
		return w
	}
	// Only the mapped Z components are needed: z' = persp0*x + persp1*y + persp2*z.
	p0 := matrix.mat[MPersp0]
	p1 := matrix.mat[MPersp1]
	p2 := matrix.mat[MPersp2]
	w0 := float64(p0*pts[0].X + p1*pts[0].Y + p2)
	w1 := float64(p0*(pts[1].X*w) + p1*(pts[1].Y*w) + p2*w)
	w2 := float64(p0*pts[2].X + p1*pts[2].Y + p2)
	// Go float64 division already has the IEEE semantics this needs (0/0 = NaN, x/0 = Inf).
	return DoubleToScalar(math.Sqrt(w1 * w1 / (w0 * w2)))
}
