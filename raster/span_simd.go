// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package raster

import (
	"simd/archsimd"
	"unsafe"

	"github.com/richardwilkes/canvas/colorcore"
)

// The goexperiment.simd span kernels, written against archsimd's 128-bit types — the widest shape arm64 and amd64
// share, and the shape the NEON kernels in span_arm64.s already use, so the two vector lanes compute the same
// expressions in the same order. Every operation below was picked for bit-exactness with the portable form, not for
// convenience:
//
//   - clamp01f is minf(maxf(v, 0), 1), and minf/maxf are comparison-based ("comparison false -> second operand"), so
//     NaN clamps to +0 and -0 clamps to +0. The vector Min/Max methods propagate NaN (FMIN/FMAX semantics), so the
//     clamps here are built from compare + IfElse instead. x.IfElse(mask, y) keeps x where mask is true.
//   - toUnorm is round-to-nearest-even of v*255 clamped to [0, 255], with the "!(f > 0) -> 0" guard applied BEFORE any
//     float-to-int conversion, because that guard is what removes NaN — and an unclamped NaN converts to 0 on arm64
//     but to 0x80000000 on amd64. After the clamp every lane is a finite value in [0, 255], where both arches agree.
//   - archsimd's ConvertToInt32 is the *truncating* convert (VFCVTZS / VCVTTPS2DQ), not FCVTNS, so the rounding is
//     done first by Round (VFRINTN / VROUNDPS with the round-to-nearest-even mode), which is exactly
//     math.RoundToEven's rule; the convert then only has to move an already-integral in-range value across.
//   - the two integer kernels keep every per-byte product in its own 16-bit lane (the vector spelling of the portable
//     SWAR forms in blitter_sprite.go and blitter_solid.go) and the per-lane bounds below show nothing ever carries
//     into a neighbor, so the packed results are identical to the scalar words.
//
// All four are locked against the portable forms by TestSpanSIMDMatchesScalar, whose integer subtests are exhaustive
// over the full (value, scale) domains.
//
// The lane shifts go through the shift16/shift32 helpers with counts hoisted out of the loop (rightShift16 and
// friends), which is a codegen accommodation only — see span_simd_arm64.go. The values they compute are the ones the
// straight ShiftAllLeft/ShiftAllRight calls compute.

// The kernels view a PMColor4f as a 16-byte {R,G,B,A} float quad, the same layout the NEON kernels require.
var (
	_ [16 - unsafe.Sizeof(colorcore.PMColor4f{})]byte
	_ [unsafe.Sizeof(colorcore.PMColor4f{}) - 16]byte
)

// init swaps the dispatch variables to the simd kernels where they are the fastest lane: on amd64 with AVX2. On arm64
// simdKernelsPreferred declines — the NEON kernels in span_arm64.s are faster (see span_simd_arm64.go) — so the
// default dispatch stands there, while the equivalence tests still run these kernels (they gate on
// simdKernelsSupported, not preference).
func init() {
	if !simdKernelsPreferred() {
		return
	}
	clampSpan01Fn = clampSpan01SIMD
	storeSpanSrcFn = storeSpanSrcSIMD
	pmSrcOverRowFn = pmSrcOverRowSIMD
	blitMaskOpaqueRowFn = blitMaskOpaqueRowSIMD
}

// spanFloats views a premultiplied color span as the flat float stream its AoS layout already is: 4 consecutive floats
// per pixel, 16-byte pixels (asserted above), so a Float32x4 is exactly one pixel and the whole span is a contiguous
// 4*len(buf) float run.
func spanFloats(buf []colorcore.PMColor4f) []float32 {
	if len(buf) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&buf[0].R)), len(buf)*4)
}

// maxf4 is the lanewise maxf: "a > b ? a : b". The comparison is false for NaN and for a == -0 against b == +0, so both
// select b, exactly as the scalar comparison does — which a hardware Max would not, since it propagates NaN.
func maxf4(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.IfElse(a.Greater(b), b) }

// minf4 is the lanewise minf: "a < b ? a : b", built from the same compare+select for the same reason as maxf4.
func minf4(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.IfElse(a.Less(b), b) }

// clamp014 is the lanewise clamp01f: minf(maxf(v, 0), 1), in that nesting order — a NaN lane becomes +0 at the maxf and
// then survives the minf, exactly as the scalar clamp01f does.
func clamp014(v, zero, one archsimd.Float32x4) archsimd.Float32x4 {
	return minf4(maxf4(v, zero), one)
}

// toUnorm4 is the lanewise toUnorm, leaving each result as a uint32 in [0, 255]. The scalar guards read "!(f > 0) -> 0"
// and "f > 255 -> 255", which are precisely maxf(f, 0) and minf(f, 255); running them before the convert is what makes
// the arches agree, since only after them is every lane a finite in-range value. See the file header for why Round is
// separate from ConvertToInt32.
func toUnorm4(v, zero, c255 archsimd.Float32x4) archsimd.Uint32x4 {
	return minf4(maxf4(v.Mul(c255), zero), c255).Round().ConvertToInt32().ToBits()
}

// transpose4 transposes the 4x4 matrix whose rows are r0..r3, returning its columns — here, four AoS pixels in, four
// channel quads out. It is the archsimd spelling of the ZIP1/ZIP2 pyramid storeSpanQuads uses: the 32-bit interleaves
// pair rows 0/1 and 2/3, then the 64-bit interleaves stitch those halves into columns. Only lane shuffling is involved,
// so no value is inspected or rounded; the 32-bit and 64-bit interleave primitives produce the same lane order on arm64
// (ZIP1/ZIP2) and amd64 (PUNPCKLDQ/PUNPCKHDQ, PUNPCKLQDQ/PUNPCKHQDQ), verified by probe.
func transpose4(r0, r1, r2, r3 archsimd.Float32x4) (c0, c1, c2, c3 archsimd.Float32x4) {
	b0, b1, b2, b3 := r0.ToBits(), r1.ToBits(), r2.ToBits(), r3.ToBits()
	lo01 := b0.InterleaveLo(b1).ReshapeToUint64s()
	hi01 := b0.InterleaveHi(b1).ReshapeToUint64s()
	lo23 := b2.InterleaveLo(b3).ReshapeToUint64s()
	hi23 := b2.InterleaveHi(b3).ReshapeToUint64s()
	return lo01.InterleaveLo(lo23).ReshapeToUint32s().BitsToFloat32(),
		lo01.InterleaveHi(lo23).ReshapeToUint32s().BitsToFloat32(),
		hi01.InterleaveLo(hi23).ReshapeToUint32s().BitsToFloat32(),
		hi01.InterleaveHi(hi23).ReshapeToUint32s().BitsToFloat32()
}

// clampSpan01SIMD clamps every channel of buf to [0, 1]. One pixel per vector, so the span divides evenly and there is
// no tail — the same property the NEON kernel relies on.
func clampSpan01SIMD(buf []colorcore.PMColor4f) {
	f := spanFloats(buf)
	if len(f) == 0 {
		return
	}
	zero := archsimd.BroadcastFloat32x4(0)
	one := archsimd.BroadcastFloat32x4(1)
	for i := 0; i < len(f); i += 4 {
		clamp014(archsimd.LoadFloat32x4(f[i:]), zero, one).Store(f[i:])
	}
}

// storeSpanSrcSIMD stores shaded premultiplied floats as 8888 words. Four AoS pixels are transposed into channel quads
// (r0..r3, g0..g3, ...) so one toUnorm4 covers four pixels' worth of a channel, then the quads are shifted into place
// and OR-ed, reproducing storeWord's r | g<<8 | b<<16 | a<<24 four pixels at a time. The sub-quad tail runs the
// portable form.
func storeSpanSrcSIMD(buf []colorcore.PMColor4f, span []uint32) {
	n := len(span)
	f := spanFloats(buf)
	zero := archsimd.BroadcastFloat32x4(0)
	c255 := archsimd.BroadcastFloat32x4(255)
	l8, l16, l24 := leftShift32(8), leftShift32(16), leftShift32(24)
	i := 0
	for ; i+4 <= n; i += 4 {
		r, g, b, a := transpose4(
			archsimd.LoadFloat32x4(f[i*4:]),
			archsimd.LoadFloat32x4(f[i*4+4:]),
			archsimd.LoadFloat32x4(f[i*4+8:]),
			archsimd.LoadFloat32x4(f[i*4+12:]),
		)
		toUnorm4(r, zero, c255).
			Or(shift32(toUnorm4(g, zero, c255), l8)).
			Or(shift32(toUnorm4(b, zero, c255), l16)).
			Or(shift32(toUnorm4(a, zero, c255), l24)).
			Store(span[i:])
	}
	if i < n {
		storeSpanSrcGeneric(buf[i:], span[i:])
	}
}

// pmSrcOverRowSIMD blends a row of premultiplied src pixels over dst: per channel satAdd8(src, mulDiv255Round(dst,
// 255-srcA)). Four pixels per iteration, the sub-quad tail on the portable form.
//
// The per-byte arithmetic runs in 16-bit lanes, two lanes per pixel: the even bytes (channels 0 and 2) come from
// d16 & 0x00FF and the odd bytes (channels 1 and 3) from d16 >> 8, so each byte gets its own lane with 8 bits of
// headroom. That is the vector twin of pmSrcOverRowGeneric's spread64 layout, and the same bound applies: with
// prod = d*nalpha + 128 <= 255*255+128 = 0xFE81, prod>>8 <= 0xFE, and prod + (prod>>8) <= 0xFF7F, every value stays
// inside its 16-bit lane, so (prod + (prod>>8)) >> 8 is exactly mulDiv255Round's result and nothing carries into a
// neighboring channel. Recombining with lo | hi<<8 is exact because both halves are <= 0xFF. The final add is
// AddSaturated on bytes, which is satAdd8's clamp at 255 per channel.
func pmSrcOverRowSIMD(dst, src []uint32) {
	n := len(src)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	half := archsimd.BroadcastUint16x8(128)
	c255 := archsimd.BroadcastUint32x4(255)
	spread := archsimd.BroadcastUint32x4(0x00010001)
	r8, l8, r24 := rightShift16(8), leftShift16(8), rightShift32(24)
	i := 0
	for ; i+4 <= n; i += 4 {
		s32 := archsimd.LoadUint32x4(src[i:])
		d16 := archsimd.LoadUint32x4(dst[i:]).ReshapeToUint16s()
		// nalpha = 255 - srcA, replicated into both 16-bit lanes of its pixel. The multiply by 0x00010001 cannot
		// overflow a 32-bit lane (nalpha <= 255) and is the cheapest way to reach both halves at once.
		na := c255.Sub(shift32(s32, r24)).Mul(spread).ReshapeToUint16s()
		lo := d16.And(byteMask).Mul(na).Add(half)
		lo = shift16(lo.Add(shift16(lo, r8)), r8)
		hi := shift16(d16, r8).Mul(na).Add(half)
		hi = shift16(hi.Add(shift16(hi, r8)), r8)
		scaled := lo.Or(shift16(hi, l8)).ReshapeToUint8s()
		s32.ReshapeToUint8s().AddSaturated(scaled).ReshapeToUint32s().Store(dst[i:])
	}
	if i < n {
		pmSrcOverRowGeneric(dst[i:], src[i:n])
	}
}

// blitMaskOpaqueRowSIMD blends an opaque solid color through an A8 coverage row: per byte
// (pm_b*(m+1))>>8 + (d_b*(256-m))>>8, which is blitMaskOpaqueRowGeneric's alphaMulQ(pm, m+1) + alphaMulQ(dev, 256-m)
// spelled per channel. Four pixels per iteration, the sub-quad tail on the portable form.
//
// The same two-lanes-per-pixel split as pmSrcOverRowSIMD carries the byte arithmetic, and the products are rewritten as
// pm_b*m + pm_b and d_b*(255-m) + d_b — exact integer identities that keep both factors inside a byte, so each 16-bit
// lane holds at most 255*255 + 255 = 0xFF00. The two shifted terms sum to at most 255 (they are a 257/256-weighted
// convex pair of pm_b and d_b, and only m = 255 reaches weight 257, where the second term is 0), which is what lets the
// halves recombine with lo | hi<<8; the scalar form's uint32 addition would carry between channels if that bound ever
// failed, and TestSpanSIMDMatchesScalar checks it exhaustively over every (pm_b, dev_b, m) triple.
func blitMaskOpaqueRowSIMD(dev []uint32, aa []uint8, pm uint32) {
	n := len(aa)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	c255 := archsimd.BroadcastUint16x8(255)
	r8, l8 := rightShift16(8), leftShift16(8)
	pmw := archsimd.BroadcastUint32x4(pm).ReshapeToUint16s()
	pmLo := pmw.And(byteMask)
	pmHi := shift16(pmw, r8)
	i := 0
	for ; i+4 <= n; i += 4 {
		// Gather the quad's four coverage bytes through a scalar word rather than a partial vector load
		// (LoadUint8x16Part is an out-of-line call, far too expensive for a four-pixel loop body; the byte loads
		// below fold into a single word load). The word packs aa[i+j] at bit 8*j, and reinterpreting a broadcast of
		// it as bytes puts aa[i+j] in byte lane j of every 32-bit lane on the little-endian architectures archsimd
		// supports; the extend then lifts those four bytes to 16-bit lanes and the self-interleave duplicates each
		// into the two lanes of its pixel.
		m32 := uint32(aa[i]) | uint32(aa[i+1])<<8 | uint32(aa[i+2])<<16 | uint32(aa[i+3])<<24
		mb := archsimd.BroadcastUint32x4(m32).ReshapeToUint8s().ExtendLo8ToUint16()
		m := mb.InterleaveLo(mb)
		nm := c255.Sub(m)
		d16 := archsimd.LoadUint32x4(dev[i:]).ReshapeToUint16s()
		dLo := d16.And(byteMask)
		dHi := shift16(d16, r8)
		lo := shift16(pmLo.Mul(m).Add(pmLo), r8).Add(shift16(dLo.Mul(nm).Add(dLo), r8))
		hi := shift16(pmHi.Mul(m).Add(pmHi), r8).Add(shift16(dHi.Mul(nm).Add(dHi), r8))
		lo.Or(shift16(hi, l8)).ReshapeToUint32s().Store(dev[i:])
	}
	if i < n {
		blitMaskOpaqueRowGeneric(dev[i:n], aa[i:n], pm)
	}
}
