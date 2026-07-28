// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The sRGB transfer function and its inverse. The inverse constants are derived at init time with the same scalar powf
// approximation used to derive the reference values (log2/exp2 with truncating float→int conversion — subtly different
// from ApproxPowf, which rounds), so the derived constants are stable bit for bit. They live here, rather than in
// colorfilter, so imagecore's pixel-conversion steps can share them without an import cycle.

package colorcore

import "math"

// SRGBTF is the parametric sRGB transfer function.
var SRGBTF = TransferFn{
	G: 2.4,
	A: float32(1 / 1.055),
	B: float32(0.055 / 1.055),
	C: float32(1 / 12.92),
	D: 0.04045,
	E: 0,
	F: 0,
}

// SRGBInvTF is the inverse of SRGBTF, computed once at init.
var SRGBInvTF = InvertSRGBishTF(&SRGBTF)

// skcmsLog2 is a fast scalar log2 approximation built from the float's exponent bits plus a mantissa refinement.
func skcmsLog2(x float32) float32 {
	bits := int32(math.Float32bits(x))
	e := float32(bits) * (1.0 / (1 << 23))
	m := math.Float32frombits(uint32(bits)&0x007fffff | 0x3f000000)
	return e - 124.225514990 - 1.498030302*m - 1.725879990/(0.3520887068+m)
}

// skcmsExp2 is the matching fast scalar exp2 approximation (note the truncating cast, where ApproxPow2 rounds).
func skcmsExp2(x float32) float32 {
	if x > 128.0 {
		return math.Float32frombits(0x7f800000) // +inf
	} else if x < -127.0 {
		return 0.0
	}
	fract := x - float32(math.Floor(float64(x)))
	fbits := (1.0 * (1 << 23)) * (x + 121.274057500 - 1.490129070*fract + 27.728023300/(4.84252568-fract))
	if fbits >= float32(math.MaxInt32) {
		return math.Float32frombits(0x7f800000)
	} else if fbits < 0 {
		return 0
	}
	return math.Float32frombits(uint32(int32(fbits)))
}

// skcmsPow computes x^y via skcmsLog2/skcmsExp2; non-positive x returns 0 and x == 1 returns 1 exactly.
func skcmsPow(x, y float32) float32 {
	if x <= 0 {
		return 0
	}
	if x == 1 {
		return 1
	}
	return skcmsExp2(skcmsLog2(x) * y)
}

// EvalSkcmsTF evaluates the parametric transfer function at x using the scalar approximations, with mirror-image
// handling of negative inputs.
func EvalSkcmsTF(tf *TransferFn, x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign = -1
		x = -x
	}
	if x < tf.D {
		return sign * (tf.C*x + tf.F)
	}
	return sign * (skcmsPow(tf.A*x+tf.B, tf.G) + tf.E)
}

// InvertSRGBishTF computes the parametric inverse of an sRGB-style transfer function, adjusting the final constants so
// that inv(src(1.0)) == 1.0 exactly.
func InvertSRGBishTF(src *TransferFn) TransferFn {
	var inv TransferFn

	// We'll start by finding the new threshold inv.d.
	dl := src.C*src.D + src.F
	dr := skcmsPow(src.A*src.D+src.B, src.G) + src.E
	if d := dl - dr; d > 1.0/512.0 || d < -1.0/512.0 {
		panic("colorcore: transfer function is discontinuous")
	}
	inv.D = dl

	// When d=0, the linear section collapses to a point. We leave c,d,f all zero in that case.
	if inv.D > 0 {
		inv.C = 1.0 / src.C
		inv.F = -src.F / src.C
	}

	// Inverting the nonlinear section: x = (1/a)(y - e)^(1/g) - b/a, with k = (1/a)^g moved inside the exponentiation.
	k := skcmsPow(src.A, -src.G) // (1/a)^g == a^-g
	inv.G = 1.0 / src.G
	inv.A = k
	inv.B = -k * src.E
	inv.E = -src.B / src.A

	if inv.A < 0 {
		panic("colorcore: transfer function is not invertible")
	}
	if inv.A*inv.D+inv.B < 0 {
		inv.B = -inv.A * inv.D
	}

	// Tweak e or f so that inv(src(1.0f)) == 1.0f, preserving that valuable invariant.
	s := EvalSkcmsTF(src, 1.0)
	sign := float32(1)
	if s < 0 {
		sign = -1
		s = -s
	}
	if s < inv.D {
		inv.F = 1.0 - sign*inv.C*s
	} else {
		inv.E = 1.0 - sign*skcmsPow(inv.A*s+inv.B, inv.G)
	}
	return inv
}
