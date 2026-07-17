// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package colorcore

import "math"

// TransferFn is a parametric transfer curve in the sRGB-style piecewise form (the only form reachable in this sRGB-only
// library):
//
//	f(x) = c*x + f          x < d
//	f(x) = (a*x + b)^g + e  x ≥ d
type TransferFn struct {
	G, A, B, C, D, E, F float32
}

// EvalParametric evaluates the transfer function at v exactly as the raster pipeline's parametric stage does per lane:
// mirror-image handling of negative inputs (strip the sign, evaluate, reapply) with fused multiply-adds and the
// ApproxPowf approximation for the power segment.
func (t *TransferFn) EvalParametric(v float32) float32 {
	// strip the sign, evaluate the magnitude, reapply the sign at the end
	bits := math.Float32bits(v)
	sign := bits & 0x80000000
	v = math.Float32frombits(bits ^ sign)
	var r float32
	if v <= t.D {
		r = tfMad(t.C, v, t.F)
	} else {
		r = ApproxPowf(tfMad(t.A, v, t.B), t.G) + t.E
	}
	return math.Float32frombits(sign | math.Float32bits(r))
}

// tfMad returns f*m + a computed with a single rounding (fused multiply-add).
func tfMad(f, m, a float32) float32 {
	return float32(math.FMA(float64(f), float64(m), float64(a)))
}

// tfNmad returns a - f*m, fused like tfMad.
func tfNmad(f, m, a float32) float32 {
	return tfMad(-f, m, a)
}

// ApproxLog2 is a fast log2 approximation built from the float's exponent bits plus a mantissa refinement.
func ApproxLog2(x float32) float32 {
	// e - 127 is a fair approximation of log2(x) in its own right... The bit pattern must convert through *signed*
	// int32: a negative x has the sign bit set, lands at a hugely negative e, and ApproxPowf(negative, y) then
	// underflows to exactly 0 rather than blowing up to +inf — the lighting shader kernel feeds pow() unclamped dot
	// products at silhouette edges and relies on this. Positive floats sit below 1<<31, where signed and unsigned
	// conversions agree bit-exactly.
	e := float32(int32(math.Float32bits(x))) * (1.0 / (1 << 23))
	// ... but using the mantissa to refine its error is _much_ better.
	m := math.Float32frombits(math.Float32bits(x)&0x007fffff | 0x3f000000)
	return tfNmad(m, 1.498030302, e-124.225514990) - 1.725879990/(0.3520887068+m)
}

// ApproxPow2 is a fast exp2 approximation; the final float→bits conversion rounds to nearest, ties to even.
func ApproxPow2(x float32) float32 {
	const infinityBits = float32(0x7f800000)
	f := x - float32(math.Floor(float64(x))) // fract
	approx := tfNmad(f, 1.490129070, x+121.274057500)
	approx += 27.728023300 / (4.84252568 - f)
	approx *= 1 << 23
	// guard against underflow/overflow with explicit comparisons (not Go's NaN-propagating min/max) so that a NaN
	// approx clamps to 0
	if !(approx > 0) { //nolint:staticcheck // NaN-aware clamp to 0
		approx = 0
	}
	if approx > infinityBits {
		approx = infinityBits
	}
	return math.Float32frombits(uint32(math.RoundToEven(float64(approx))))
}

// ApproxPowf computes x^y via ApproxLog2/ApproxPow2; 0 and 1 pass through exactly.
func ApproxPowf(x, y float32) float32 {
	if x == 0 || x == 1 {
		return x
	}
	return ApproxPow2(ApproxLog2(x) * y)
}
