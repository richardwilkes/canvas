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

// simdKernelsSupported reports whether the CPU can run the simd DSP kernels. The gate is AVX2 only: these kernels are
// all-integer and execute no FMA anywhere, so the rule the other packages follow applies here too — "gate on what the
// kernels actually execute". Requiring FMA would be inaccurate and would disable the path (and skip its equivalence
// tests) under Rosetta 2, which offers AVX2 without FMA. Everything they compile to is SSE4.1 or below — the compiled
// kernels use exactly VPABSW, VPACKSSDW, VPADDD/VPADDW, VPALIGNR, VPAND/VPOR/VPXOR, VPBLENDVB, VPCMPEQD/VPCMPEQW/
// VPCMPGTW, VPEXTRD/VPINSRD/VMOVD, VPMADDWD, VPMAXSD/VPMINSD, VPMAXUW, VPMINUD, VPMOVSXWD, VPMOVZXBW, VPMULHUW,
// VPMULLD, VPMULLW, VPSHUFB, VPSLLD/VPSRAD/VPSRLD (immediate counts), VPSUBD/VPSUBW, VPSUBUSB/VPSUBUSW, VPTEST and
// the VPUNPCK* family, plus the 128-bit loads and stores — except the two broadcasts (VPBROADCASTD/VPBROADCASTW),
// which are AVX2. Unqualified CPUs keep the portable dispatch.
//
// Note what is deliberately *not* used here: Uint32x4.TruncToUint16/SaturateToUint16, Uint16x8.TruncToUint8/
// SaturateToUint8 and Mask16x8.ToBits are all AVX-512 in archsimd (probed: Mask16x8.ToBits SIGILLs under Rosetta 2),
// so the 32->16 narrowing goes through the two-operand VPACKSSDW pack (see narrow32Pair), the 16->8 gathers go through
// VPSHUFB (see byteShuffle) and the lane counting stays in the vector domain.
func simdKernelsSupported() bool { return archsimd.X86.AVX2() }

// Per-kernel dispatch preference: whether the simd kernel is at least as fast as this build's default lane. On amd64
// the only alternative is the portable form in dsp.go, which every one of these kernels beats: getSSE -37% (4x4) to
// -77% (16x16), quantizeBlock -39%, iTransformOne -11%, fTransform -8%, geomean -46%; and -18% geomean on the
// whole-encode benchmarks (-28% on a 512x384 photo). Those numbers come from darwin/amd64 under Rosetta 2 on an M4
// Max (benchstat n=8, 2026-08-20), the only amd64 lane available here, so re-measure on native silicon before
// treating them as amd64's.
//
// Two of them deserve a second look when that happens. The transforms' margins are thin here because their butterflies
// multiply in 32-bit lanes (VPMULLD, which is a slow multi-uop instruction on Intel and is very likely worse under
// Rosetta's translation), where libwebp's SSE2 transforms instead fold each "a*2217 + b*5352" pair into one VPMADDWD
// on 16-bit lanes — exact here too, since both a-values fit in an int16 and the pair sum stays under 1.3e8. That would
// mean a per-arch transform body (arm64 has no pairwise dot product), so it was left alone rather than guessed at from
// emulated timings; if native amd64 shows fTransform and iTransformOne as marginal, that is the change to make.
const (
	preferSIMDFTransform    = true
	preferSIMDITransformOne = true
	preferSIMDGetSSE        = true
	preferSIMDQuantizeBlock = true
)

// shuffleIdx is the byte-gather index vector this arch's byteShuffle takes. amd64's gather is VPSHUFB, whose index
// vector is signed and which zeroes any lane whose index byte is negative — 0xFF in the shared tables, which is also
// what arm64's TBL reads as out of range, so one table serves both arches.
type shuffleIdx = archsimd.Int8x16

// loadShuffle builds a byte-gather index vector. It must not be a package-level variable — a package-level archsimd
// value would run AVX code during package init, on CPUs that failed simdKernelsSupported and must never execute any of
// it.
func loadShuffle(v *[16]uint8) shuffleIdx { return archsimd.LoadUint8x16Array(v).BitsToInt8() }

// byteShuffle gathers bytes of x named by idx, zeroing the lanes whose index is 0xFF. Lane order probed before use.
func byteShuffle(x archsimd.Uint8x16, idx shuffleIdx) archsimd.Uint8x16 { return x.PermuteOrZero(idx) }

// mulWiden16 returns the eight exact 32-bit products of a and b: lanes 0..3 of a and b in lo, lanes 4..7 in hi. amd64
// has no widening 16-bit multiply, so the products are taken as their low and high halves (VPMULLW and VPMULHUW) and
// the interleaves stitch each pair back into the 32-bit product it came from — little-endian, the low half is the low
// 16 bits. Probed against the scalar products before use.
func mulWiden16(a, b archsimd.Uint16x8) (lo, hi archsimd.Uint32x4) {
	l, h := a.Mul(b), a.MulHigh(b)
	return l.InterleaveLo(h).ReshapeToUint32s(), l.InterleaveHi(h).ReshapeToUint32s()
}

// narrow32Pair packs the low 16 bits of each of eight 32-bit lanes into one 16-bit vector, lo's four lanes first.
// Every caller guarantees that each lane, read as an int32, is inside [-32768, 32767], so VPACKSSDW's signed
// saturation never fires and the pack is the plain truncation arm64 performs. Probed on both arches.
func narrow32Pair(lo, hi archsimd.Uint32x4) archsimd.Uint16x8 {
	return lo.BitsToInt32().SaturateToInt16Concat(hi.BitsToInt32()).ToBits()
}

// anyNonZero16 reports whether any 16-bit lane of x is non-zero. VPTEST is one instruction and, unlike the mask-to-
// bits move, is plain AVX.
func anyNonZero16(x archsimd.Uint16x8) bool { return !x.IsZero() }

// sumSquares16 adds the squares of d's eight lanes into the four 32-bit accumulator lanes. VPMADDWD squares and folds
// each lane pair in one instruction; arm64 keeps each lane's square in its own accumulator lane instead. The callers
// only ever read the sum of all four lanes, and integer addition is exact and associative, so the totals are
// identical. Every d here is a byte difference magnitude, so the pair sums stay far inside a 32-bit lane and the
// signed multiply VPMADDWD performs is the unsigned one (both factors are at most 255).
func sumSquares16(d archsimd.Uint16x8, acc archsimd.Uint32x4) archsimd.Uint32x4 {
	s := d.BitsToInt16()
	return acc.Add(s.DotProductPairs(s).ToBits())
}

// The lane shifts of the transform kernels go through the same hoisted-count helpers arm64 needs (see
// dsp_simd_arm64.go for why it needs them). On amd64 there is nothing to hoist — VPSRAD/VPSLLD take the count as an
// immediate — so the count is just the signed amount and the direction test folds away wherever the caller's count is
// the compile-time constant these kernels always pass.
type shiftCount32 = int

func rightShift32(n uint8) shiftCount32 { return -int(n) }
func leftShift32(n uint8) shiftCount32  { return int(n) }

func shiftI32(x archsimd.Int32x4, by shiftCount32) archsimd.Int32x4 {
	if by < 0 {
		return x.ShiftAllRight(uint64(-by))
	}
	return x.ShiftAllLeft(uint64(by))
}
