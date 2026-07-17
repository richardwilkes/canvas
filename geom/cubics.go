// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Quadratic and cubic root-finding, evaluation, and subdivision helpers that the mono-cubic choppers and the edge
// clipper depend on. All of this machinery computes in float64 for numerical stability, so results are bit-exact
// regardless of the input's float32 precision.

package geom

import "math"

// doubleNearlyZero reports whether a is within one float32 (not float64) epsilon of zero.
func doubleNearlyZero(a float64) bool {
	return a == 0 || math.Abs(a) < 1.1920928955078125e-07 // FLT_EPSILON
}

// doubleMagnitude returns the positive magnitude of a float64 — 2^e for normalized values, 0 for subnormals, +Inf for
// NaN/infinity.
func doubleMagnitude(a float64) float64 {
	const extractMagnitude = 0x7FF0000000000000
	return math.Float64frombits(math.Float64bits(a) & extractMagnitude)
}

// doublesNearlyEqualULPs reports whether a and b are within 16 ULPs of each other.
func doublesNearlyEqualULPs(a, b float64) bool {
	const maxUlpsDiff = 16
	// The proper magnitude for subnormal numbers is math.SmallestNonzeroFloat64's normal cousin (2^-1022), so if a and
	// b are subnormal (having a magnitude of 0) use that. If a or b are infinity or nan, then maxMagnitude will be
	// +infinity, making the tolerance infinite too, but then b - a is NaN or infinity, so it doesn't matter.
	const minMagnitude = 2.2250738585072014e-308 // DBL_MIN
	maxMagnitude := math.Max(math.Max(doubleMagnitude(a), minMagnitude), doubleMagnitude(b))
	const ulpFactor = 2.220446049250313e-16 // DBL_EPSILON
	tolerance := maxMagnitude * (ulpFactor * (maxUlpsDiff + 1))
	return a == b || math.Abs(b-a) < tolerance
}

///////////////////////////////////////////////////////////////////////////////
// Quadratic roots

// solveLinear solves 0 = M*x + B, returning the root count.
func solveLinear(m, b float64, solution *[2]float64) int {
	if doubleNearlyZero(m) {
		solution[0] = 0
		if doubleNearlyZero(b) {
			return 1
		}
		return 0
	}
	solution[0] = -b / m
	if math.IsInf(solution[0], 0) || math.IsNaN(solution[0]) {
		return 0
	}
	return 1
}

// closeToLinear reports whether the quadratic A*x^2 + B*x is close enough to linear that solving it as a line is more
// accurate than the quadratic formula.
func closeToLinear(a, b float64) bool {
	if a != 0 {
		// Return if B is much bigger than A.
		return math.Abs(b/a) >= 1.0e+16
	}
	// Otherwise A is zero, and the quadratic is linear.
	return true
}

// QuadDiscriminant computes b*b - a*c to within 2 bits of error using fused multiply-add to recover the rounding error
// when b*b and a*c nearly cancel.
func QuadDiscriminant(a, b, c float64) float64 {
	b2 := b * b
	ac := a * c
	roughDiscriminant := b2 - ac
	// If 3*|B2 - AC| >= AC + B2, the rough discriminant has 2 bits of rounding error or less.
	if 3*math.Abs(roughDiscriminant) >= b2+ac {
		return roughDiscriminant
	}
	// Use the extra internal precision afforded by fma to calculate the rounding error for b^2 and ac.
	b2RoundingError := math.FMA(b, b, -b2)
	acRoundingError := math.FMA(a, c, -ac)
	// Add the total rounding error back into the discriminant guess.
	return (b2 - ac) + (b2RoundingError - acRoundingError)
}

// quadRoots returns the roots of A*x^2 - 2*B*x + C (note the -2 convention), along with the discriminant.
func quadRoots(a, b, c float64) (discriminant, root0, root1 float64) {
	discriminant = QuadDiscriminant(a, b, c)
	if a == 0 {
		var root float64
		switch b {
		case 0:
			if c == 0 {
				root = math.Inf(1)
			} else {
				root = math.NaN()
			}
		default:
			// Solve -2*B*x + C == 0; x = c/(2*b).
			root = c / (2 * b)
		}
		return discriminant, root, root
	}
	if discriminant == 0 {
		return discriminant, b / a, b / a
	}
	if discriminant > 0 {
		d := math.Sqrt(discriminant)
		var r float64
		if b > 0 {
			r = b + d
		} else {
			r = b - d
		}
		return discriminant, r / a, c / r
	}
	// The discriminant is negative or is not finite.
	return discriminant, math.NaN(), math.NaN()
}

func zeroIfTiny(x float64) float64 {
	if doubleNearlyZero(x) {
		return 0
	}
	return x
}

// QuadRootsReal returns the real roots of A*x^2 + B*x + C, deduplicated.
func QuadRootsReal(a, b, c float64, solution *[2]float64) int {
	if closeToLinear(a, b) {
		return solveLinear(b, c, solution)
	}
	discriminant, root0, root1 := quadRoots(a, -0.5*b, c)
	// Handle invariants to mesh with existing code from here on.
	if math.IsNaN(discriminant) || math.IsInf(discriminant, 0) || discriminant < 0 {
		return 0
	}
	roots := 0
	if r0 := zeroIfTiny(root0); !math.IsInf(r0, 0) && !math.IsNaN(r0) {
		solution[roots] = r0
		roots++
	}
	if r1 := zeroIfTiny(root1); !math.IsInf(r1, 0) && !math.IsNaN(r1) {
		solution[roots] = r1
		roots++
	}
	if roots == 2 && doublesNearlyEqualULPs(solution[0], solution[1]) {
		roots = 1
	}
	return roots
}

///////////////////////////////////////////////////////////////////////////////
// Cubic roots

// CubicEvalAt evaluates A*t^3 + B*t^2 + C*t + D using fused multiply-add to reduce round-off.
func CubicEvalAt(a, b, c, d, t float64) float64 {
	return math.FMA(t, math.FMA(t, math.FMA(t, a, b), c), d)
}

// nearlyEqualDouble reports whether x and y are close enough to be treated as the same root.
func nearlyEqualDouble(x, y float64) bool {
	if doubleNearlyZero(x) {
		return doubleNearlyZero(y)
	}
	return doublesNearlyEqualULPs(x, y)
}

// closeToAQuadratic reports whether the cubic's leading coefficient a is negligible relative to b, so the cubic is
// effectively a quadratic.
func closeToAQuadratic(a, b float64) bool {
	if doubleNearlyZero(b) {
		return doubleNearlyZero(a)
	}
	return math.Abs(a/b) < 1.0e-7
}

// CubicRootsReal returns the real roots of A*t^3 + B*t^2 + C*t + D, deduplicated.
func CubicRootsReal(a, b, c, d float64, solution *[3]float64) int {
	if closeToAQuadratic(a, b) {
		var quadSolution [2]float64
		n := QuadRootsReal(b, c, d, &quadSolution)
		for i := 0; i < n; i++ {
			solution[i] = quadSolution[i]
		}
		return n
	}
	if doubleNearlyZero(d) { // 0 is one root
		var quadSolution [2]float64
		num := QuadRootsReal(a, b, c, &quadSolution)
		for i := 0; i < num; i++ {
			solution[i] = quadSolution[i]
			if doubleNearlyZero(quadSolution[i]) {
				return num
			}
		}
		solution[num] = 0
		return num + 1
	}
	if doubleNearlyZero(a + b + c + d) { // 1 is one root
		var quadSolution [2]float64
		num := QuadRootsReal(a, a+b, -d, &quadSolution)
		for i := 0; i < num; i++ {
			solution[i] = quadSolution[i]
			if doublesNearlyEqualULPs(quadSolution[i], 1) {
				return num
			}
		}
		solution[num] = 1
		return num + 1
	}
	// If A is zero (e.g. B was NaN and thus closeToAQuadratic was false), we will temporarily have infinities rolling
	// about, but will catch that when checking r2MinusQ3. Go's / already has plain IEEE-754 division semantics, so no
	// special-casing is needed here.
	invA := 1 / a
	pa := b * invA
	pb := c * invA
	pc := d * invA
	a2 := pa * pa
	q := (a2 - pb*3) / 9
	r := (2*a2*pa - 9*pa*pb + 27*pc) / 54
	r2 := r * r
	q3 := q * q * q
	r2MinusQ3 := r2 - q3
	// If one of R2 Q3 is infinite or nan, subtracting them will also be infinite/nan. If both are infinite or nan, the
	// subtraction will be nan. In either case, we have no finite roots.
	if math.IsInf(r2MinusQ3, 0) || math.IsNaN(r2MinusQ3) {
		return 0
	}
	adiv3 := pa / 3
	count := 0
	if r2MinusQ3 < 0 { // we have 3 real roots
		// the divide/root can, due to finite precisions, be slightly outside of -1...1
		theta := math.Acos(math.Max(-1, math.Min(1, r/math.Sqrt(q3))))
		neg2RootQ := -2 * math.Sqrt(q)

		rr := neg2RootQ*math.Cos(theta/3) - adiv3
		solution[count] = rr
		count++

		rr = neg2RootQ*math.Cos((theta+2*math.Pi)/3) - adiv3
		if !nearlyEqualDouble(solution[0], rr) {
			solution[count] = rr
			count++
		}
		rr = neg2RootQ*math.Cos((theta-2*math.Pi)/3) - adiv3
		if !nearlyEqualDouble(solution[0], rr) && (count == 1 || !nearlyEqualDouble(solution[1], rr)) {
			solution[count] = rr
			count++
		}
	} else { // we have 1 real root
		sqrtR2MinusQ3 := math.Sqrt(r2MinusQ3)
		aa := math.Abs(r) + sqrtR2MinusQ3
		aa = math.Cbrt(aa)
		if r > 0 {
			aa = -aa
		}
		if !doubleNearlyZero(aa) {
			aa += q / aa
		}
		solution[count] = aa - adiv3
		count++
		if !doubleNearlyZero(r2) && doublesNearlyEqualULPs(r2, q3) {
			rr := -aa/2 - adiv3
			if !nearlyEqualDouble(solution[0], rr) {
				solution[count] = rr
				count++
			}
		}
	}
	return count
}

// CubicRootsValidT returns the real roots pinned/deduplicated into [0, 1].
func CubicRootsValidT(a, b, c, d float64, solution *[3]float64) int {
	var allRoots [3]float64
	realRoots := CubicRootsReal(a, b, c, d, &allRoots)
	foundRoots := 0
	for index := 0; index < realRoots; index++ {
		tValue := allRoots[index]
		switch {
		case tValue <= 1.00005 && (tValue >= 1.0 || doublesNearlyEqualULPs(tValue, 1.0)):
			// Make sure we do not already have 1 (or something very close) in the list of roots.
			if (foundRoots < 1 || !doublesNearlyEqualULPs(solution[0], 1)) &&
				(foundRoots < 2 || !doublesNearlyEqualULPs(solution[1], 1)) {
				solution[foundRoots] = 1
				foundRoots++
			}
		case tValue >= -0.00005 && (tValue <= 0.0 || doubleNearlyZero(tValue)):
			// Make sure we do not already have 0 (or something very close) in the list of roots.
			if (foundRoots < 1 || !doubleNearlyZero(solution[0])) &&
				(foundRoots < 2 || !doubleNearlyZero(solution[1])) {
				solution[foundRoots] = 0
				foundRoots++
			}
		case tValue > 0.0 && tValue < 1.0:
			solution[foundRoots] = tValue
			foundRoots++
		}
	}
	return foundRoots
}

// approximatelyZeroCubic is the cutoff used by the binary-search root fallback.
func approximatelyZeroCubic(x float64) bool {
	return math.Abs(x) < 0.00000001
}

// findExtremaValidT finds the cubic's local min/max parameters within [0, 1].
func findExtremaValidT(a, b, c float64, t *[2]float64) int {
	// To find the local min and max of a cubic, we take the derivative and solve when that is equal to 0. d/dt (A*t^3 +
	// B*t^2 + C*t + D) = 3A*t^2 + 2B*t + C
	var roots [2]float64
	numRoots := QuadRootsReal(3*a, 2*b, c, &roots)
	validRoots := 0
	for i := 0; i < numRoots; i++ {
		if roots[i] >= 0 && roots[i] <= 1.0 {
			t[validRoots] = roots[i]
			validRoots++
		}
	}
	return validRoots
}

// binarySearchCubicRoot bisects [start, stop] for a sign change in the cubic, returning the root found or -1 if none
// was found.
func binarySearchCubicRoot(a, b, c, d, start, stop float64) float64 {
	left := CubicEvalAt(a, b, c, d, start)
	if approximatelyZeroCubic(left) {
		return start
	}
	right := CubicEvalAt(a, b, c, d, stop)
	if math.IsInf(left, 0) || math.IsNaN(left) || math.IsInf(right, 0) || math.IsNaN(right) {
		return -1 // Not going to deal with one or more endpoints being non-finite.
	}
	if (left > 0 && right > 0) || (left < 0 && right < 0) {
		return -1 // We can only have a root if one is above 0 and the other is below 0.
	}
	const maxIterations = 1000 // prevent infinite loop
	for i := 0; i < maxIterations; i++ {
		step := (start + stop) / 2
		curr := CubicEvalAt(a, b, c, d, step)
		if approximatelyZeroCubic(curr) {
			return step
		}
		if (curr < 0 && left < 0) || (curr > 0 && left > 0) {
			start = step // go right
		} else {
			stop = step // go left
		}
	}
	return -1
}

// CubicBinarySearchRootsValidT is a slow but accurate fallback used when the analytic roots fail to evaluate close
// enough to zero.
func CubicBinarySearchRootsValidT(a, b, c, d float64, solution *[3]float64) int {
	if math.IsInf(a, 0) || math.IsNaN(a) || math.IsInf(b, 0) || math.IsNaN(b) ||
		math.IsInf(c, 0) || math.IsNaN(c) || math.IsInf(d, 0) || math.IsNaN(d) {
		return 0
	}
	regions := [4]float64{0, 0, 0, 1}
	// Find local minima and maxima
	var minMax [2]float64
	extremaCount := findExtremaValidT(a, b, c, &minMax)
	startIndex := 2 - extremaCount
	if extremaCount == 1 {
		regions[startIndex+1] = minMax[0]
	}
	if extremaCount == 2 {
		// While the roots will be in the range 0 to 1 inclusive, they might not be sorted.
		regions[startIndex+1] = math.Min(minMax[0], minMax[1])
		regions[startIndex+2] = math.Max(minMax[0], minMax[1])
	}
	// Starting at regions[startIndex] and going up through regions[3], we have an ascending list of numbers in the
	// range 0 to 1.0, between which are the possible locations of a root.
	foundRoots := 0
	for ; startIndex < 3; startIndex++ {
		root := binarySearchCubicRoot(a, b, c, d, regions[startIndex], regions[startIndex+1])
		if root >= 0 {
			// Check for duplicates
			if (foundRoots < 1 || !approximatelyZeroCubic(solution[0]-root)) &&
				(foundRoots < 2 || !approximatelyZeroCubic(solution[1]-root)) {
				solution[foundRoots] = root
				foundRoots++
			}
		}
	}
	return foundRoots
}

///////////////////////////////////////////////////////////////////////////////
// Double-precision cubic Bezier helpers

// bezierCubicConvertToPolynomial converts one axis of a cubic Bezier to polynomial form: p0..p3 are the 4 control
// points' coordinates for that axis.
func bezierCubicConvertToPolynomial(p0, p1, p2, p3 float64) (a, b, c, d float64) {
	a = -p0 + 3*p1 - 3*p2 + p3
	b = 3*p0 - 6*p1 + 3*p2
	c = -3*p0 + 3*p1
	d = p0
	return a, b, c, d
}

func interpolateDouble(a, b, t float64) float64 {
	return a + (b-a)*t
}

// dPoint is a double-precision point used by the cubic Bezier subdivision.
type dPoint struct {
	x, y float64
}

// bezierCubicSubdivide performs De Casteljau subdivision at t, producing the two halves as 7 shared double-precision
// points.
func bezierCubicSubdivide(curve *[4]dPoint, t float64, twoCurves *[7]dPoint) {
	twoCurves[0] = curve[0]
	twoCurves[6] = curve[3]

	x01 := interpolateDouble(curve[0].x, curve[1].x, t)
	y01 := interpolateDouble(curve[0].y, curve[1].y, t)
	x12 := interpolateDouble(curve[1].x, curve[2].x, t)
	y12 := interpolateDouble(curve[1].y, curve[2].y, t)
	x23 := interpolateDouble(curve[2].x, curve[3].x, t)
	y23 := interpolateDouble(curve[2].y, curve[3].y, t)

	twoCurves[1] = dPoint{x: x01, y: y01}
	twoCurves[5] = dPoint{x: x23, y: y23}

	twoCurves[2] = dPoint{x: interpolateDouble(x01, x12, t), y: interpolateDouble(y01, y12, t)}
	twoCurves[4] = dPoint{x: interpolateDouble(x12, x23, t), y: interpolateDouble(y12, y23, t)}

	twoCurves[3] = dPoint{
		x: interpolateDouble(twoCurves[2].x, twoCurves[4].x, t),
		y: interpolateDouble(twoCurves[2].y, twoCurves[4].y, t),
	}
}

///////////////////////////////////////////////////////////////////////////////
// Mono-cubic chops

// closeEnoughToZero is the tolerance used to validate a candidate axis-intersection root.
func closeEnoughToZero(x float64) bool {
	return math.Abs(x) < 0.00001
}

// firstAxisIntersection finds the parameter t at which curve crosses axisIntercept along the x or y axis.
func firstAxisIntersection(curve *[4]dPoint, yDirection bool, axisIntercept float64) (float64, bool) {
	var a, b, c, d float64
	if yDirection {
		a, b, c, d = bezierCubicConvertToPolynomial(curve[0].y, curve[1].y, curve[2].y, curve[3].y)
	} else {
		a, b, c, d = bezierCubicConvertToPolynomial(curve[0].x, curve[1].x, curve[2].x, curve[3].x)
	}
	d -= axisIntercept
	var roots [3]float64
	count := CubicRootsValidT(a, b, c, d, &roots)
	if count == 0 {
		return 0, false
	}
	// Verify that at least one of the roots is accurate.
	for i := 0; i < count; i++ {
		if closeEnoughToZero(CubicEvalAt(a, b, c, d, roots[i])) {
			return roots[i], true
		}
	}
	// None of the roots returned by our normal cubic solver were correct enough, so we need to fall back to a more
	// accurate solution.
	count = CubicBinarySearchRootsValidT(a, b, c, d, &roots)
	if count == 0 {
		return 0, false
	}
	for i := 0; i < count; i++ {
		if closeEnoughToZero(CubicEvalAt(a, b, c, d, roots[i])) {
			return roots[i], true
		}
	}
	return 0, false
}

// chopMonoCubicAt implements the shared body of ChopMonoCubicAtX/ChopMonoCubicAtY. src must hold 4 points and dst 7.
func chopMonoCubicAt(src, dst []Point, yDirection bool, axisIntercept float32) bool {
	curve := [4]dPoint{
		{x: float64(src[0].X), y: float64(src[0].Y)},
		{x: float64(src[1].X), y: float64(src[1].Y)},
		{x: float64(src[2].X), y: float64(src[2].Y)},
		{x: float64(src[3].X), y: float64(src[3].Y)},
	}
	solution, ok := firstAxisIntersection(&curve, yDirection, float64(axisIntercept))
	if !ok {
		return false
	}
	var cubicPair [7]dPoint
	bezierCubicSubdivide(&curve, solution, &cubicPair)
	for i := 0; i < 7; i++ {
		dst[i] = Point{X: DoubleToScalar(cubicPair[i].x), Y: DoubleToScalar(cubicPair[i].y)}
	}
	return true
}

// ChopMonoCubicAtY chops a Y-monotonic cubic at the y intercept, computing in double precision. Returns false if no
// accurate-enough root was found. src must hold 4 points, dst 7.
func ChopMonoCubicAtY(src, dst []Point, y float32) bool {
	return chopMonoCubicAt(src, dst, true, y)
}

// ChopMonoCubicAtX chops an X-monotonic cubic at the x intercept, computing in double precision. Returns false if no
// accurate-enough root was found. src must hold 4 points, dst 7.
func ChopMonoCubicAtX(src, dst []Point, x float32) bool {
	return chopMonoCubicAt(src, dst, false, x)
}

///////////////////////////////////////////////////////////////////////////////
// X-extrema chops

// ChopQuadAtXExtrema is the X-axis analog of ChopQuadAtYExtrema. dst must hold 5 points.
func ChopQuadAtXExtrema(src, dst []Point) int {
	a := src[0].X
	b := src[1].X
	c := src[2].X
	if isNotMonotonic(a, b, c) {
		if t, ok := ValidUnitDivide(a-b, a-b-b+c); ok {
			ChopQuadAt(src, dst, t)
			// Force the chopped point's neighbors to share its X.
			dst[1].X = dst[2].X
			dst[3].X = dst[2].X
			return 1
		}
		// if we get here, we need to force dst to be monotonic, even though we couldn't compute a unit_divide value
		// (probably underflow).
		if ScalarAbs(a-b) < ScalarAbs(b-c) {
			b = a
		} else {
			b = c
		}
	}
	dst[0] = Point{X: a, Y: src[0].Y}
	dst[1] = Point{X: b, Y: src[1].Y}
	dst[2] = Point{X: c, Y: src[2].Y}
	return 0
}

// ChopCubicAtXExtrema is the X-axis analog of ChopCubicAtYExtrema. dst must hold 10 points.
func ChopCubicAtXExtrema(src, dst []Point) int {
	var tValues [2]float32
	roots := FindCubicExtrema(src[0].X, src[1].X, src[2].X, src[3].X, &tValues)
	ChopCubicAtList(src, dst, tValues[:roots])
	if roots > 0 {
		// Force the chopped point's neighbors to share its X.
		dst[2].X = dst[3].X
		dst[4].X = dst[3].X
		if roots == 2 {
			dst[5].X = dst[6].X
			dst[7].X = dst[6].X
		}
	}
	return roots
}
