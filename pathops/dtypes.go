// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The numeric core of the pathops engine: the ULPs-based "almost equal" family (via the 2s-complement float-bit trick)
// and the epsilon-tolerance predicates that the double-precision pathops geometry relies on. This is the foundation of
// the engine that backs Op/Simplify/Builder (op.go/simplify.go/common.go). Pathops computes in double throughout for
// precision; the ULPs helpers deliberately narrow to float32 first, since they compare values at the granularity the
// geometry is ultimately stored and rendered at.

package pathops

import "math"

// Epsilon constants. fltEpsilon is the float32 machine epsilon (2^-23), promoted to double for these expressions;
// dblEpsilon is the float64 machine epsilon (2^-52).
//
// The multipliers on the machine epsilons are empirical: they were chosen to make the suite pass, not derived from an
// error analysis, and there is no metric that says a given value is right. dblEpsilonErr in particular sets the slack
// of six of the seven precisely* predicates below (preciselyZeroWhenComparedTo is the exception; it scales the plain
// dblEpsilon by its comparand), so changing it shifts pathops output across the board.
const (
	fltEpsilon             = 1.1920928955078125e-07 // float32 machine epsilon == 2^-23
	dblEpsilon             = 2.2204460492503131e-16 // float64 machine epsilon == 2^-52
	dblEpsilonErr          = dblEpsilon * 4         // a few bits of slack; see note above
	fltEpsilonHalf         = fltEpsilon / 2
	fltEpsilonDouble       = fltEpsilon * 2
	fltEpsilonInverse      = 1 / fltEpsilon
	fltEpsilonOrderableErr = fltEpsilon * 16 // tolerance used by the *Orderable predicates below
	roughEpsilon           = fltEpsilon * 64
	moreRoughEpsilon       = fltEpsilon * 256
)

// absF32 returns the absolute value of x.
func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// f32IsFinite reports whether x is neither infinite nor NaN.
func f32IsFinite(x float32) bool {
	d := float64(x)
	return !math.IsInf(d, 0) && !math.IsNaN(d)
}

// dIsFinite reports whether a and b are both finite.
func dIsFinite(a, b float64) bool {
	return !math.IsInf(a, 0) && !math.IsNaN(a) && !math.IsInf(b, 0) && !math.IsNaN(b)
}

// ieeeDoubleDivide is an ordinary IEEE divide that yields inf/NaN rather than trapping on divide-by-zero. Go's float64
// division already has these semantics; this wrapper documents the intent at each call site.
func ieeeDoubleDivide(numer, denom float64) float64 {
	return numer / denom
}

// floatAs2sCompliment reinterprets the float's bits as a signed int, then maps sign-magnitude to 2s-complement so
// ordinary integer compares order floats monotonically. Converts -0 to 0.
func floatAs2sCompliment(x float32) int32 {
	bits := int32(math.Float32bits(x))
	if bits < 0 {
		bits &= 0x7FFFFFFF
		bits = -bits
	}
	return bits
}

// argumentsDenormalized reports whether both a and b are small enough (relative to epsilon) that they should be treated
// as denormalized noise rather than compared for real equality.
func argumentsDenormalized(a, b float32, epsilon int) bool {
	denormalizedCheck := float32(fltEpsilon) * float32(epsilon) / 2
	return absF32(a) <= denormalizedCheck && absF32(b) <= denormalizedCheck
}

func equalUlps(a, b float32, epsilon, depsilon int) bool {
	if argumentsDenormalized(a, b, depsilon) {
		return true
	}
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits < bBits+int32(epsilon) && bBits < aBits+int32(epsilon)
}

func equalUlpsNoNormalCheck(a, b float32, epsilon int) bool {
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits < bBits+int32(epsilon) && bBits < aBits+int32(epsilon)
}

func equalUlpsPin(a, b float32, epsilon, depsilon int) bool {
	if !f32IsFinite(a) || !f32IsFinite(b) {
		return false
	}
	if argumentsDenormalized(a, b, depsilon) {
		return true
	}
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits < bBits+int32(epsilon) && bBits < aBits+int32(epsilon)
}

func dEqualUlps(a, b float32, epsilon int) bool {
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits < bBits+int32(epsilon) && bBits < aBits+int32(epsilon)
}

func notEqualUlpsPin(a, b float32, epsilon int) bool {
	if !f32IsFinite(a) || !f32IsFinite(b) {
		return false
	}
	if argumentsDenormalized(a, b, epsilon) {
		return false
	}
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits >= bBits+int32(epsilon) || bBits >= aBits+int32(epsilon)
}

func dNotEqualUlps(a, b float32, epsilon int) bool {
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits >= bBits+int32(epsilon) || bBits >= aBits+int32(epsilon)
}

func notEqualUlps(a, b float32, epsilon int) bool {
	if argumentsDenormalized(a, b, epsilon) {
		return false
	}
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits >= bBits+int32(epsilon) || bBits >= aBits+int32(epsilon)
}

func lessOrEqualUlps(a, b float32, epsilon int) bool {
	if argumentsDenormalized(a, b, epsilon) {
		return a < b+float32(fltEpsilon)*float32(epsilon)
	}
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits < bBits+int32(epsilon)
}

func lessUlps(a, b float32, epsilon int) bool {
	if argumentsDenormalized(a, b, epsilon) {
		return a <= b-float32(fltEpsilon)*float32(epsilon)
	}
	aBits := floatAs2sCompliment(a)
	bBits := floatAs2sCompliment(b)
	return aBits <= bBits-int32(epsilon)
}

// The public ULPs comparisons below take doubles because all pathops geometry is double; each narrows its inputs to
// float32 first, since ULPs comparisons are only meaningful at a fixed bit width.

func almostEqualUlps(a, b float64) bool { return equalUlps(float32(a), float32(b), 16, 16) }

func almostEqualUlpsNoNormalCheck(a, b float64) bool {
	return equalUlpsNoNormalCheck(float32(a), float32(b), 16)
}

func almostEqualUlpsPin(a, b float64) bool { return equalUlpsPin(float32(a), float32(b), 16, 16) }

func almostBequalUlps(a, b float64) bool { return equalUlps(float32(a), float32(b), 2, 2) }

func almostLessUlps(a, b float64) bool { return lessUlps(float32(a), float32(b), 16) }

func almostLessOrEqualUlps(a, b float64) bool { return lessOrEqualUlps(float32(a), float32(b), 16) }

func almostPequalUlps(a, b float64) bool { return equalUlps(float32(a), float32(b), 8, 8) }

func roughlyEqualUlps(a, b float64) bool { return equalUlps(float32(a), float32(b), 256, 1024) }

func notAlmostEqualUlpsPin(a, b float64) bool { return notEqualUlpsPin(float32(a), float32(b), 16) }

func notAlmostDequalUlps(a, b float64) bool { return dNotEqualUlps(float32(a), float32(b), 16) }

// notAlmostEqualUlps is the negation of almostEqualUlps.
func notAlmostEqualUlps(a, b float64) bool { return notEqualUlps(float32(a), float32(b), 16) }

// almostDequalUlps reports approximate equality of a and b at 16 ULPs: for in-range values it narrows to float32 and
// uses the 16-ULP compare; for values at or above the float32 max it falls back to a relative magnitude test (which
// correctly yields false for a finite-vs-NaN pair via the IEEE divide).
func almostDequalUlps(a, b float64) bool {
	if math.Abs(a) < math.MaxFloat32 && math.Abs(b) < math.MaxFloat32 {
		return dEqualUlps(float32(a), float32(b), 16)
	}
	return ieeeDoubleDivide(math.Abs(a-b), math.Max(math.Abs(a), math.Abs(b))) < fltEpsilon*16
}

func almostBetweenUlps(a, b, c float64) bool {
	return almostBetweenUlpsF(float32(a), float32(b), float32(c))
}

func almostBetweenUlpsF(a, b, c float32) bool {
	const ulpsEpsilon = 2
	if a <= c {
		return lessOrEqualUlps(a, b, ulpsEpsilon) && lessOrEqualUlps(b, c, ulpsEpsilon)
	}
	return lessOrEqualUlps(b, a, ulpsEpsilon) && lessOrEqualUlps(c, b, ulpsEpsilon)
}

// The epsilon-tolerance predicates below are used for comparing T values and small quantities; coordinate comparisons
// use the ULPs family above.

func zeroOrOne(x float64) bool { return x == 0 || x == 1 }

func approximatelyZero(x float64) bool { return math.Abs(x) < fltEpsilon }

func preciselyZero(x float64) bool { return math.Abs(x) < dblEpsilonErr }

func approximatelyEqual(x, y float64) bool { return approximatelyZero(x - y) }

func approximatelyZeroHalf(x float64) bool { return math.Abs(x) < fltEpsilonHalf }

func approximatelyEqualHalf(x, y float64) bool { return approximatelyZeroHalf(x - y) }

func preciselyEqual(x, y float64) bool { return preciselyZero(x - y) }

func roughlyEqual(x, y float64) bool { return math.Abs(x-y) < roughEpsilon }

func roughlyNegative(x float64) bool { return x < roughEpsilon }

// roughlyBetween reports whether b lies (roughly) between a and c in either order.
func roughlyBetween(a, b, c float64) bool {
	if a <= c {
		return roughlyNegative(a-b) && roughlyNegative(b-c)
	}
	return roughlyNegative(b-a) && roughlyNegative(c-b)
}

// approximatelyBetween reports whether b lies (approximately) between a and c in either order, using
// approximatelyLessThanZero (x < fltEpsilon) as the per-side negativity test.
func approximatelyBetween(a, b, c float64) bool {
	if a <= c {
		return approximatelyLessThanZero(a-b) && approximatelyLessThanZero(b-c)
	}
	return approximatelyLessThanZero(b-a) && approximatelyLessThanZero(c-b)
}

func moreRoughlyEqual(x, y float64) bool { return math.Abs(x-y) < moreRoughEpsilon }

func approximatelyGreaterThanOne(x float64) bool { return x > 1-fltEpsilon }

func approximatelyLessThanZero(x float64) bool { return x < fltEpsilon }

func approximatelyOneOrLess(x float64) bool { return x < 1+fltEpsilon }

func approximatelyOneOrLessDouble(x float64) bool { return x < 1+fltEpsilonDouble }

func approximatelyZeroOrMore(x float64) bool { return x > -fltEpsilon }

func approximatelyZeroOrMoreDouble(x float64) bool { return x > -fltEpsilonDouble }

// approximatelyZeroInverse reports whether x is large enough that its reciprocal underflows toward zero.
func approximatelyZeroInverse(x float64) bool { return math.Abs(x) > fltEpsilonInverse }

// approximatelyZeroWhenComparedTo reports whether x is negligible relative to the magnitude of y.
func approximatelyZeroWhenComparedTo(x, y float64) bool {
	return x == 0 || math.Abs(x) < math.Abs(y*fltEpsilon)
}

// preciselyZeroWhenComparedTo is like approximatelyZeroWhenComparedTo but uses the tighter double-precision epsilon.
func preciselyZeroWhenComparedTo(x, y float64) bool {
	return x == 0 || math.Abs(x) < math.Abs(y*dblEpsilon)
}

// roughlyZeroWhenComparedTo reports whether x is negligible relative to y at the rough epsilon.
func roughlyZeroWhenComparedTo(x, y float64) bool {
	return x == 0 || math.Abs(x) < math.Abs(y*roughEpsilon)
}

// approximatelyNegativeOrderable reports whether x is negative at the orderable tolerance (fltEpsilonOrderableErr),
// used where a coarser epsilon than approximatelyLessThanZero is needed to keep a comparison transitive (orderable).
func approximatelyNegativeOrderable(x float64) bool { return x < fltEpsilonOrderableErr }

// approximatelyZeroOrderable reports whether x is zero at the orderable tolerance.
func approximatelyZeroOrderable(x float64) bool { return math.Abs(x) < fltEpsilonOrderableErr }

// approximatelyEqualOrderable reports whether x and y are equal at the orderable tolerance.
func approximatelyEqualOrderable(x, y float64) bool { return approximatelyZeroOrderable(x - y) }

// approximatelyBetweenOrderable reports whether b lies (approximately, at the orderable tolerance) between a and c in
// either order.
func approximatelyBetweenOrderable(a, b, c float64) bool {
	if a <= c {
		return approximatelyNegativeOrderable(a-b) && approximatelyNegativeOrderable(b-c)
	}
	return approximatelyNegativeOrderable(b-a) && approximatelyNegativeOrderable(c-b)
}

func preciselyNegative(x float64) bool { return x < dblEpsilonErr }

func preciselyLessThanZero(x float64) bool { return x < dblEpsilonErr }

func preciselyGreaterThanOne(x float64) bool { return x > 1-dblEpsilonErr }

// between reports whether (a <= b <= c) || (a >= b >= c), computed with a single product: the two differences (a-b) and
// (c-b) have the same sign (or one is zero) exactly when b lies between a and c.
func between(a, b, c float64) bool {
	return (a-b)*(c-b) <= 0
}

// preciselyBetween reports whether b lies (precisely, at the tight double-precision epsilon) between a and c in either
// order.
func preciselyBetween(a, b, c float64) bool {
	if a <= c {
		return preciselyNegative(a-b) && preciselyNegative(b-c)
	}
	return preciselyNegative(b-a) && preciselyNegative(c-b)
}

// pinT clamps a T value to [0,1] using the precise epsilon.
func pinT(t float64) float64 {
	if preciselyLessThanZero(t) {
		return 0
	}
	if preciselyGreaterThanOne(t) {
		return 1
	}
	return t
}

// b2f converts a bool to 0.0/1.0, used throughout the intersection code where a boolean condition needs to participate
// directly in a T-value expression.
func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// dInterp linearly interpolates between a and b at parameter t: a + (b-a)*t.
func dInterp(a, b, t float64) float64 { return a + (b-a)*t }
