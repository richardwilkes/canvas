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

// simdKernelsSupported reports whether the CPU can run the simd span kernels. Everything they compile to on arm64 (the
// 128-bit loads/stores, the integer and float lane arithmetic, FRINTN/FCVTZS, UQADD, ZIP1/ZIP2, UXTL, USHL) is baseline
// NEON, present on every arm64 CPU Go supports. This mirrors shaders.simdKernelsSupported; the two packages cannot
// share an unexported helper, so each carries its own copy of the policy.
func simdKernelsSupported() bool { return true }

// simdKernelsPreferred reports whether the simd span kernels are the fastest lane on this hardware, which is what the
// init-time dispatch gates on — supported is what the equivalence tests gate on, so the arm64 test runs still lock the
// kernels bit-for-bit even though dispatch declines them. On arm64 the answer today is no: the hand-written NEON
// kernels in span_arm64.s beat the archsimd forms by 1.1-1.6x on these four (measured M4 Max, 256-px rows — the
// archsimd loop pays per-iteration bounds checks and assembles some idioms, like FCVTN2 and the byte gathers, from
// several instructions), so the default arm64 dispatch stays. Unlike the shader stages, where archsimd beats the
// assembly because it inlines into the dispatch call, these kernels do a full row per call and the call overhead the
// stages save is noise here. Revisit if the codegen improves; the shaders package needs no such split because its simd
// kernels won everywhere.
func simdKernelsPreferred() bool { return false }

// The lane shifts of the two integer kernels go through a hoisted count because arm64 has no shift-by-immediate in this
// API: ShiftAllLeft/ShiftAllRight lower to a VDUP of the (negated) count plus VUSHL, and the compiler re-materializes
// that VDUP on every iteration instead of lifting it out of the loop — three instructions per shift in a body that has
// half a dozen of them. Broadcasting the count once and issuing the bare VUSHL is worth roughly 2-3x on the integer
// kernels here. It is also exactly the instruction ShiftAllRight would have emitted, so the results are unchanged: a
// negative count is a logical right shift for an unsigned vector, and every count these kernels use (8, 16, 24) is
// inside the lane width.
type (
	shiftCount16 = archsimd.Int16x8
	shiftCount32 = archsimd.Int32x4
)

func rightShift16(n uint8) shiftCount16 { return archsimd.BroadcastInt16x8(-int16(n)) }
func leftShift16(n uint8) shiftCount16  { return archsimd.BroadcastInt16x8(int16(n)) }
func rightShift32(n uint8) shiftCount32 { return archsimd.BroadcastInt32x4(-int32(n)) }
func leftShift32(n uint8) shiftCount32  { return archsimd.BroadcastInt32x4(int32(n)) }

func shift16(x archsimd.Uint16x8, by shiftCount16) archsimd.Uint16x8 { return x.Shift(by) }
func shift32(x archsimd.Uint32x4, by shiftCount32) archsimd.Uint32x4 { return x.Shift(by) }
