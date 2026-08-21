// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd

package vp8enc

import "simd/archsimd"

// simdKernelsSupported reports whether the CPU can run the simd DSP kernels. Everything they compile to on arm64 is
// baseline NEON, present on every arm64 CPU Go supports — the compiled kernels use exactly VABS, VADD, VAND, VBIF,
// VCMEQ, VCMGT, VDUP, VEXT, VMOV/VMOVI, VMUL, VNEG, VNOT, VORR, VSMAX/VSMIN, VSSHL/VUSHL (with hoisted counts), VSUB,
// VSXTL, VTBL, VUMAX/VUMIN, VUMAXV, VUMULL/VUMULL2, VUQSUB, VUXTL, VUZP1 and VZIP1/VZIP2, plus the 128-bit loads and
// stores. This mirrors the other packages' per-arch guards; packages cannot share an unexported helper, so each
// carries its own copy of the policy.
func simdKernelsSupported() bool { return true }

// Per-kernel dispatch preference: whether the simd kernel is at least as fast as this build's default lane. There is
// no hand-written NEON lane in this package, so the alternative is always the portable form in dsp.go, and every
// kernel here beats it: fTransform -41%, iTransformOne -50%, getSSE -55% (4x4) to -84% (16x16), quantizeBlock -75%,
// geomean -69%; and -36% geomean on the whole-encode benchmarks (-42% on a 512x384 photo). Measured on an M4 Max,
// darwin/arm64, benchstat n=10, 2026-08-20.
const (
	preferSIMDFTransform    = true
	preferSIMDITransformOne = true
	preferSIMDGetSSE        = true
	preferSIMDQuantizeBlock = true
)

// shuffleIdx is the byte-gather index vector this arch's byteShuffle takes. arm64's gather is TBL, whose index vector
// is an ordinary Uint8x16 and which zeroes any lane whose index is out of range — 0xFF in the shared tables, which is
// also what amd64's VPSHUFB reads as "zero this byte" (-1), so one table serves both arches.
type shuffleIdx = archsimd.Uint8x16

// loadShuffle builds a byte-gather index vector. It must not be a package-level variable — a package-level archsimd
// value would run vector code during package init, on CPUs that failed simdKernelsSupported and must never execute any
// of it.
func loadShuffle(v *[16]uint8) shuffleIdx { return archsimd.LoadUint8x16Array(v) }

// byteShuffle gathers bytes of x named by idx, zeroing the lanes whose index is 0xFF. Lane order probed before use.
func byteShuffle(x archsimd.Uint8x16, idx shuffleIdx) archsimd.Uint8x16 { return x.LookupOrZero(idx) }

// mulWiden16 returns the eight exact 32-bit products of a and b: lanes 0..3 of a and b in lo, lanes 4..7 in hi. arm64
// has the widening multiply directly (VUMULL over the low half of a register), so the high half is rotated down first.
// Probed against the scalar products before use.
func mulWiden16(a, b archsimd.Uint16x8) (lo, hi archsimd.Uint32x4) {
	return a.MulWidenLo(b), a.HiToLo().MulWidenLo(b.HiToLo())
}

// narrow32Pair packs the low 16 bits of each of eight 32-bit lanes into one 16-bit vector, lo's four lanes first.
// Every caller guarantees that each lane, read as an int32, is inside [-32768, 32767], which is what lets amd64 use
// its saturating pack for this; arm64 truncates instead (VUZP1 keeps the even 16-bit halves, which little-endian are
// exactly the low halves), and inside that range the two agree exactly. Probed on both arches.
func narrow32Pair(lo, hi archsimd.Uint32x4) archsimd.Uint16x8 {
	return lo.ReshapeToUint16s().ConcatEven(hi.ReshapeToUint16s())
}

// anyNonZero16 reports whether any 16-bit lane of x is non-zero.
func anyNonZero16(x archsimd.Uint16x8) bool { return x.ReduceMax() != 0 }

// sumSquares16 adds the squares of d's eight lanes into the four 32-bit accumulator lanes. Which accumulator lane a
// given square lands in differs between the arches (arm64 keeps the lane, amd64's VPMADDWD folds lane pairs), but the
// callers only ever read the sum of all four lanes, and integer addition is exact and associative, so the totals are
// identical. Every d here is a byte difference magnitude, so d*d <= 65025 and no lane can overflow.
func sumSquares16(d archsimd.Uint16x8, acc archsimd.Uint32x4) archsimd.Uint32x4 {
	dh := d.HiToLo()
	return acc.Add(d.MulWidenLo(d)).Add(dh.MulWidenLo(dh))
}

// The lane shifts of the transform kernels go through a hoisted count because arm64 has no shift-by-immediate in this
// API: ShiftAllLeft/ShiftAllRight lower to a VDUP of the (negated) count plus VSSHL, and the compiler re-materializes
// that VDUP at every use instead of lifting it out (Go's SSA has no loop-invariant code motion). Broadcasting the
// count once and issuing the bare VSSHL is exactly the instruction ShiftAllRight would have emitted, so the values are
// unchanged: a negative count is an arithmetic right shift for a signed vector, and every count these kernels use (3,
// 4, 9, 16, 17) is inside the lane width. This mirrors raster's span_simd_arm64.go.
type shiftCount32 = archsimd.Int32x4

func rightShift32(n uint8) shiftCount32 { return archsimd.BroadcastInt32x4(-int32(n)) }
func leftShift32(n uint8) shiftCount32  { return archsimd.BroadcastInt32x4(int32(n)) }

func shiftI32(x archsimd.Int32x4, by shiftCount32) archsimd.Int32x4 { return x.Shift(by) }
