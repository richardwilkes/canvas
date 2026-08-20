// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && arm64

package raster

import "simd/archsimd"

// simdBlitSupported reports whether the CPU can run the simd blit rows. Everything they compile to on arm64 (the
// 128-bit loads/stores, the integer lane arithmetic, UXTL, ZIP1, USHL/SSHL, UMIN/UMAX, CMHI/CMEQ, BSL) is baseline
// NEON, present on every arm64 CPU Go supports.
func simdBlitSupported() bool { return true }

// Per-kernel dispatch preference: whether the simd blit row is at least as fast as this build's default lane. Unlike
// the span kernels, none of these rows has a hand-written NEON lane to lose to — the alternative is the portable
// scalar/SWAR form in the blitter files — so every one of them is preferred on arm64 as well. (simdKernelsPreferred,
// which the span kernels gate on, answers a different question: whether archsimd beats span_arm64.s.)
const (
	preferSIMDFillWords              = true
	preferSIMDFillBytes              = true
	preferSIMDColor32Row             = true
	preferSIMDBlitMaskTranslucentRow = true
	preferSIMDInterp256Row           = true
	preferSIMDPremulRow              = true
	preferSIMDPMBlendRow             = true
	preferSIMDBlitRowLCD16           = true
	preferSIMDBlitRowLCD16Opaque     = true
	preferSIMDBlendRowLCD16Opaque    = true
)

// shiftI16 is the signed counterpart of shift16 (see span_simd_arm64.go for why the count is hoisted): a negative count
// makes VSSHL an arithmetic right shift, which is what blend32's int32 >>5 performs.
func shiftI16(x archsimd.Int16x8, by shiftCount16) archsimd.Int16x8 { return x.Shift(by) }
