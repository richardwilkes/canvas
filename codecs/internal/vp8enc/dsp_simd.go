// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package vp8enc

import "simd/archsimd"

// The goexperiment.simd encoder DSP kernels, written against archsimd's 128-bit types — the widest shape arm64 and
// amd64 share, so both vector lanes compute the same expressions in the same order. The encoder's output must be
// bit-identical in both build modes (a VP8 bitstream is not "close enough": the reconstruction loop feeds every
// rounding decision back into the next macroblock), so every operation below was picked for exactness with the
// portable form in dsp.go, not for convenience:
//
//   - All the arithmetic is integer, and every intermediate is proved to fit its lane width below. Where a bound
//     cannot be proved for the whole input domain — only iTransformOne, whose input is an arbitrary int16 array —
//     the kernel checks the domain up front and hands the block to the portable form instead.
//   - The 32-bit lane shifts are arithmetic (Int32x4.ShiftAllRight is VSSHL / VPSRAD) and so are Go's, so the
//     rounding shifts of negative values match. The 32-bit lane multiplies keep the low 32 bits, which is exactly
//     what Go's int32-ranged arithmetic would produce, and never wrap here.
//   - The 16-bit narrowing goes through narrow32Pair, whose two arch spellings agree over the range every caller
//     stays inside; see the note there.
//   - "level * q" and the sign flip run in 16-bit lanes, whose low-16-bit product is precisely the int16 conversion
//     the portable quantizer applies (two's complement multiplication is sign-agnostic in the low half).
//
// Every kernel here is locked against its portable twin by TestDSPSIMDMatchesScalar, and the whole encoder is locked
// against the portable lane, bitstream byte for byte, by TestDSPSIMDEncodeMatchesScalar.
//
// The lane shifts go through the shiftI32 helper with counts hoisted out of the expression, which is a codegen
// accommodation only — see dsp_simd_arm64.go. The byte gathers and the 16-bit narrowing are per-arch for the reason
// maskfilter's and imagecore's are (every pack-and-narrow method archsimd offers for the 16-bit case is AVX-512 on
// amd64); see byteShuffle and narrow32Pair.

// The DSP entry points for builds with the simd kernels available. Each consults the dispatch bool its init set; see
// the dispatch note at the top of dsp.go for why these are static calls rather than function variables.

func fTransform(src, ref []uint8, out []int16) {
	if useSIMDFTransform {
		fTransformSIMD(src, ref, out)
		return
	}
	fTransformGeneric(src, ref, out)
}

func iTransformOne(ref []uint8, in []int16, dst []uint8) {
	if useSIMDITransformOne {
		iTransformOneSIMD(ref, in, dst)
		return
	}
	iTransformOneGeneric(ref, in, dst)
}

func getSSE(a, b []uint8, w, h int) int64 {
	if useSIMDGetSSE {
		return getSSESIMD(a, b, w, h)
	}
	return getSSEGeneric(a, b, w, h)
}

func quantizeBlock(in, out *[16]int16, m *matrix) int {
	if useSIMDQuantizeBlock {
		return quantizeBlockSIMD(in, out, m)
	}
	return quantizeBlockGeneric(in, out, m)
}

// The dispatch state, set once by init from the CPU check and this arch's preference constants. They are variables
// rather than constants only so the equivalence tests can run a whole encode down each lane; nothing else writes them.
var (
	useSIMDFTransform    bool
	useSIMDITransformOne bool
	useSIMDGetSSE        bool
	useSIMDQuantizeBlock bool
)

func init() {
	if !simdKernelsSupported() {
		return
	}
	useSIMDFTransform = preferSIMDFTransform
	useSIMDITransformOne = preferSIMDITransformOne
	useSIMDGetSSE = preferSIMDGetSSE
	useSIMDQuantizeBlock = preferSIMDQuantizeBlock
}

// transpose4x4Idx is the byte gather that transposes a 4x4 byte block: the row-major sixteen bytes of loadBlock4x4
// come out column-major, so widening the halves yields the block's columns rather than its rows. Both transform
// kernels need their input in column order (the first pass of each combines the four samples of a column), and one
// gather beats the four-vector interleave pyramid a 32-bit transpose would cost.
var transpose4x4Idx = [16]uint8{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}

// rowGatherIdx[j] gathers the low byte of each of four 32-bit lanes — one clamped sample per row of column j — into
// the byte positions column j occupies in a row-major 4x4 block, zeroing every other byte. OR-ing the four gathered
// vectors therefore rebuilds the block in row-major order, which is the order iTransformOne stores it in. 0xFF means
// "zero this byte" on both arches; see byteShuffle.
var rowGatherIdx = [4][16]uint8{
	{0, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 12, 0xFF, 0xFF, 0xFF},
	{0xFF, 0, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 12, 0xFF, 0xFF},
	{0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 12, 0xFF},
	{0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 12},
}

// zigzagGatherIdx are the four byte gathers that reorder the sixteen quantized levels from natural order into the
// zigzag order quantizeBlock writes to out. Entry 0 gathers out[0..7] from the low eight levels and entry 1 gathers
// the one lane of out[0..7] that comes from the high eight (zigzag[3] == 8); entries 2 and 3 do the same for
// out[8..15], whose one straggler is zigzag[12] == 7. Each 16-bit level is two bytes, so a lane pair (2j, 2j+1) moves
// to (2n, 2n+1). TestDSPSIMDZigzagGather recomputes all four from the zigzag table itself.
var zigzagGatherIdx = [4][16]uint8{
	{0, 1, 2, 3, 8, 9, 0xFF, 0xFF, 10, 11, 4, 5, 6, 7, 12, 13},
	{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
	{2, 3, 8, 9, 10, 11, 4, 5, 0xFF, 0xFF, 6, 7, 12, 13, 14, 15},
	{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 14, 15, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
}

// iTransformSafeBound is the largest input magnitude iTransformOneSIMD accepts. Its horizontal pass multiplies the
// intermediate column values by 35468 in 32-bit lanes, and those intermediates grow to at most 3.85x the input
// magnitude (|a| <= 2x from the outer butterfly plus |d| <= 1.8478x + 2 from mul1 + mul2), so an arbitrary int16 input
// could push that product past 2^31, where the portable form — which computes in Go's int — would not wrap. At this
// bound the worst case is |c| <= 31529 and |c| * 35468 <= 1.12e9, comfortably inside a signed 32-bit lane, and every
// later sum stays under 1.3e5. Real encoder input never comes close: the coefficients this reconstructs are the
// dequantized forward-transform output, bounded by the transform's 12-bit range plus one quantizer step, and an
// instrumented run over five images at four qualities — 841628 calls — saw a largest magnitude of 1657, a fifth of the
// bound, with the fallback never taken. It is a proof obligation rather than a live path, but the tests drive both
// sides of it: TestITransformSIMDBoundProof checks the arithmetic claim directly (an output comparison cannot, since
// the clamp swallows a wrapped intermediate) and the boundary subtest walks the guard itself.
const iTransformSafeBound = 8192

// hiToLoBytes moves the high eight bytes of x into its low eight. Both arches spell this as the byte-shift of the
// concatenation of x with itself (VEXT / VPALIGNR), which is a rotate, so the upper half of the result is the input's
// lower half — no caller reads it. Probed on both arches.
func hiToLoBytes(x archsimd.Uint8x16) archsimd.Uint8x16 { return x.ConcatShiftBytesRight(x, 8) }

// hiToLo16 moves the high four 16-bit lanes of x into its low four, through the byte view (see hiToLoBytes).
func hiToLo16(x archsimd.Int16x8) archsimd.Int16x8 {
	return hiToLoBytes(x.ToBits().ReshapeToUint8s()).ReshapeToUint16s().BitsToInt16()
}

// transpose4i32 transposes the 4x4 matrix whose rows are r0..r3, returning its columns: result k, lane i is input i,
// lane k. It is the integer twin of raster's float transpose4 — the 32-bit interleaves pair rows 0/1 and 2/3, then the
// 64-bit interleaves stitch those halves into columns. Only lane shuffling is involved, so no value is inspected, and
// the two interleave primitives produce the same lane order on arm64 (ZIP1/ZIP2) and amd64 (VPUNPCKLDQ/VPUNPCKHDQ,
// VPUNPCKLQDQ/VPUNPCKHQDQ), verified by probe.
func transpose4i32(r0, r1, r2, r3 archsimd.Int32x4) (c0, c1, c2, c3 archsimd.Int32x4) {
	b0, b1, b2, b3 := r0.ToBits(), r1.ToBits(), r2.ToBits(), r3.ToBits()
	lo01 := b0.InterleaveLo(b1).ReshapeToUint64s()
	hi01 := b0.InterleaveHi(b1).ReshapeToUint64s()
	lo23 := b2.InterleaveLo(b3).ReshapeToUint64s()
	hi23 := b2.InterleaveHi(b3).ReshapeToUint64s()
	return lo01.InterleaveLo(lo23).ReshapeToUint32s().BitsToInt32(),
		lo01.InterleaveHi(lo23).ReshapeToUint32s().BitsToInt32(),
		hi01.InterleaveLo(hi23).ReshapeToUint32s().BitsToInt32(),
		hi01.InterleaveHi(hi23).ReshapeToUint32s().BitsToInt32()
}

// loadBlock4x4 reads the 4x4 byte block at the start of buf (four rows of four bytes, stride bps) into one vector in
// row-major order. The four-byte rows are gathered through scalar words rather than partial vector loads:
// LoadUint8x16Part is an out-of-line call, far too expensive for a kernel this small, while each byte quad below folds
// into a single 32-bit load. The words then go straight into lanes rather than through a stack array, which measured
// -23% on arm64 and -70% on amd64 against LoadUint32x4Array — the array form has to store four words and reload them
// as a vector, and the store-to-load forwarding is the whole cost of this helper. Row i's bytes land in bytes
// 4i..4i+3 on the little-endian architectures this file builds for (probed).
func loadBlock4x4(buf []uint8) archsimd.Uint8x16 {
	return archsimd.BroadcastUint32x4(rowWord(buf, 0)).
		SetElem(1, rowWord(buf, 1)).
		SetElem(2, rowWord(buf, 2)).
		SetElem(3, rowWord(buf, 3)).
		ReshapeToUint8s()
}

// rowWord reads row i's four bytes as one little-endian word.
func rowWord(buf []uint8, i int) uint32 {
	r := buf[i*bps : i*bps+4]
	return uint32(r[0]) | uint32(r[1])<<8 | uint32(r[2])<<16 | uint32(r[3])<<24
}

// columns4x4 splits the sixteen bytes of a column-major 4x4 block (loadBlock4x4 followed by the transpose gather)
// into its four columns, each widened to 32-bit lanes.
func columns4x4(t archsimd.Uint8x16) (c0, c1, c2, c3 archsimd.Int32x4) {
	lo := t.ExtendLo8ToUint16().BitsToInt16()
	hi := hiToLoBytes(t).ExtendLo8ToUint16().BitsToInt16()
	return lo.ExtendLo4ToInt32(), hiToLo16(lo).ExtendLo4ToInt32(),
		hi.ExtendLo4ToInt32(), hiToLo16(hi).ExtendLo4ToInt32()
}

// fTransformSIMD is the vector form of fTransformGeneric: the forward DCT of the 4x4 difference block src-ref.
//
// The two passes are the portable loops with their loop variable turned into the lane index. The first pass wants a
// column of differences per vector, which the byte transpose above delivers; the second pass wants a row of the
// intermediate per vector, which is what transpose4i32 turns the first pass's output into (pass one's result vector j
// holds T[row][j] across its lanes, so transposing gives vector k = T[k][*], and the portable pass two indexes
// tmp[0+i], tmp[4+i], tmp[8+i], tmp[12+i] — exactly T[0..3][i]).
//
// Lane bounds, from |src - ref| <= 255: the a-values of pass one reach 1020, so a2*2217 + a3*5352 <= 7.7e6 and the
// tmp values stay inside 14 bits (8160). Pass two's a-values then reach 16320, so a2*2217 + a3*5352 + 51000 <= 1.24e8
// — three orders of magnitude inside a signed 32-bit lane — and every output is inside 12 bits, which is what lets
// narrow32Pair pack the results.
func fTransformSIMD(src, ref []uint8, out []int16) {
	tIdx := loadShuffle(&transpose4x4Idx)
	s := byteShuffle(loadBlock4x4(src), tIdx)
	r := byteShuffle(loadBlock4x4(ref), tIdx)
	// The differences, still packed two columns to a vector: subtracting in 16-bit lanes and reading the result as
	// signed is the two's complement difference, exact for byte inputs.
	dLo := s.ExtendLo8ToUint16().Sub(r.ExtendLo8ToUint16()).BitsToInt16()
	dHi := hiToLoBytes(s).ExtendLo8ToUint16().Sub(hiToLoBytes(r).ExtendLo8ToUint16()).BitsToInt16()
	d0 := dLo.ExtendLo4ToInt32()
	d1 := hiToLo16(dLo).ExtendLo4ToInt32()
	d2 := dHi.ExtendLo4ToInt32()
	d3 := hiToLo16(dHi).ExtendLo4ToInt32()

	k2217 := archsimd.BroadcastInt32x4(2217)
	k5352 := archsimd.BroadcastInt32x4(5352)
	left3, right9 := leftShift32(3), rightShift32(9)
	a0 := d0.Add(d3)
	a1 := d1.Add(d2)
	a2 := d1.Sub(d2)
	a3 := d0.Sub(d3)
	t0 := shiftI32(a0.Add(a1), left3)
	t1 := shiftI32(a2.Mul(k2217).Add(a3.Mul(k5352)).Add(archsimd.BroadcastInt32x4(1812)), right9)
	t2 := shiftI32(a0.Sub(a1), left3)
	t3 := shiftI32(a3.Mul(k2217).Sub(a2.Mul(k5352)).Add(archsimd.BroadcastInt32x4(937)), right9)

	q0, q1, q2, q3 := transpose4i32(t0, t1, t2, t3)
	right4, right16 := rightShift32(4), rightShift32(16)
	seven := archsimd.BroadcastInt32x4(7)
	b0 := q0.Add(q3)
	b1 := q1.Add(q2)
	b2 := q1.Sub(q2)
	b3 := q0.Sub(q3)
	o0 := shiftI32(b0.Add(b1).Add(seven), right4)
	// The portable form adds one when a3 is non-zero; here that is a lane predicate over b3.
	o1 := shiftI32(b2.Mul(k2217).Add(b3.Mul(k5352)).Add(archsimd.BroadcastInt32x4(12000)), right16).
		Add(archsimd.BroadcastInt32x4(1).Masked(b3.NotEqual(archsimd.BroadcastInt32x4(0))))
	o2 := shiftI32(b0.Sub(b1).Add(seven), right4)
	o3 := shiftI32(b3.Mul(k2217).Sub(b2.Mul(k5352)).Add(archsimd.BroadcastInt32x4(51000)), right16)
	narrow32Pair(o0.ToBits(), o1.ToBits()).BitsToInt16().Store(out[0:])
	narrow32Pair(o2.ToBits(), o3.ToBits()).BitsToInt16().Store(out[8:])
}

// mul1v and mul2v are the portable mul1 and mul2 in 32-bit lanes. Both shifts are arithmetic, as Go's are.
func mul1v(a, k archsimd.Int32x4, right16 shiftCount32) archsimd.Int32x4 {
	return shiftI32(a.Mul(k), right16).Add(a)
}

func mul2v(a, k archsimd.Int32x4, right16 shiftCount32) archsimd.Int32x4 {
	return shiftI32(a.Mul(k), right16)
}

// iTransformOneSIMD is the vector form of iTransformOneGeneric: dst = clip(ref + idct(in)).
//
// The vertical pass reads in row by row, so the four row vectors are just the widened halves of the input; its output
// vector j holds c[4*lane+j], which transpose4i32 turns into the four vectors the horizontal pass indexes (c[i],
// c[i+4], c[i+8], c[i+12] with i the lane). The horizontal pass's output vector j is then column j of the block, which
// is why the reference samples are loaded column-major too and why the store goes through the rowGatherIdx gathers.
//
// Unlike the other kernels this one cannot prove its lane bounds for every input it could be handed — see
// iTransformSafeBound — so out-of-range blocks go to the portable form.
func iTransformOneSIMD(ref []uint8, in []int16, dst []uint8) {
	v01 := archsimd.LoadInt16x8(in[0:])
	v23 := archsimd.LoadInt16x8(in[8:])
	// The magnitudes are compared unsigned, which is what makes the int16 minimum test correctly: its absolute value
	// is 32768, a value no signed 16-bit lane can hold (Abs leaves the bit pattern 0x8000, which a signed compare
	// would read as negative and wave through). The saturating difference is non-zero exactly where the magnitude
	// exceeds the bound, the same trick the quantizer uses for its zero threshold.
	mag := v01.Abs().ToBits().Max(v23.Abs().ToBits())
	if anyNonZero16(mag.SubSaturated(archsimd.BroadcastUint16x8(iTransformSafeBound))) {
		iTransformOneGeneric(ref, in, dst)
		return
	}
	k1 := archsimd.BroadcastInt32x4(ac3Mul1Const)
	k2 := archsimd.BroadcastInt32x4(ac3Mul2Const)
	right16, right3 := rightShift32(16), rightShift32(3)
	r0 := v01.ExtendLo4ToInt32()
	r1 := hiToLo16(v01).ExtendLo4ToInt32()
	r2 := v23.ExtendLo4ToInt32()
	r3 := hiToLo16(v23).ExtendLo4ToInt32()
	a := r0.Add(r2)
	b := r0.Sub(r2)
	c := mul2v(r1, k2, right16).Sub(mul1v(r3, k1, right16))
	d := mul1v(r1, k1, right16).Add(mul2v(r3, k2, right16))
	w0, w1, w2, w3 := transpose4i32(a.Add(d), b.Add(c), b.Sub(c), a.Sub(d))

	dc := w0.Add(archsimd.BroadcastInt32x4(4))
	a2 := dc.Add(w2)
	b2 := dc.Sub(w2)
	c2 := mul2v(w1, k2, right16).Sub(mul1v(w3, k1, right16))
	d2 := mul1v(w1, k1, right16).Add(mul2v(w3, k2, right16))

	// The reference samples ride along in column order so that each output column can be clipped where it stands.
	p0, p1, p2, p3 := columns4x4(byteShuffle(loadBlock4x4(ref), loadShuffle(&transpose4x4Idx)))
	zero := archsimd.BroadcastInt32x4(0)
	c255 := archsimd.BroadcastInt32x4(255)
	o0 := shiftI32(a2.Add(d2), right3).Add(p0).Max(zero).Min(c255)
	o1 := shiftI32(b2.Add(c2), right3).Add(p1).Max(zero).Min(c255)
	o2 := shiftI32(b2.Sub(c2), right3).Add(p2).Max(zero).Min(c255)
	o3 := shiftI32(a2.Sub(d2), right3).Add(p3).Max(zero).Min(c255)

	// Each clipped column contributes its four bytes to the row-major block, then one 32-bit lane per row is written
	// out as the row's four destination bytes.
	rows := byteShuffle(o0.ToBits().ReshapeToUint8s(), loadShuffle(&rowGatherIdx[0])).
		Or(byteShuffle(o1.ToBits().ReshapeToUint8s(), loadShuffle(&rowGatherIdx[1]))).
		Or(byteShuffle(o2.ToBits().ReshapeToUint8s(), loadShuffle(&rowGatherIdx[2]))).
		Or(byteShuffle(o3.ToBits().ReshapeToUint8s(), loadShuffle(&rowGatherIdx[3]))).
		ReshapeToUint32s()
	storeRow4(dst[0*bps:], rows.GetElem(0))
	storeRow4(dst[1*bps:], rows.GetElem(1))
	storeRow4(dst[2*bps:], rows.GetElem(2))
	storeRow4(dst[3*bps:], rows.GetElem(3))
}

// storeRow4 writes the four bytes of w, low byte first, which is the order loadBlock4x4 reads them in.
func storeRow4(dst []uint8, w uint32) {
	_ = dst[3]
	dst[0] = uint8(w)
	dst[1] = uint8(w >> 8)
	dst[2] = uint8(w >> 16)
	dst[3] = uint8(w >> 24)
}

// getSSESIMD is the vector form of getSSEGeneric. The distortion of a 16-wide block is accumulated a row at a time; a
// 4x4 block is gathered into a single vector, since sse4x4 is called ten times per intra4 sub-block and its per-call
// overhead matters more than its loop.
//
// The per-pixel term is the *magnitude* of the difference, which the pair of saturating byte subtractions computes
// exactly (one of them is zero), so the squares can be taken in unsigned 16-bit lanes. A byte difference squares to at
// most 65025 and a 16x16 block has 256 of them, so the 32-bit accumulator lanes cannot overflow (16.6e6 at worst) and
// the int64 total is exact.
func getSSESIMD(a, b []uint8, w, h int) int64 {
	var acc archsimd.Uint32x4
	switch {
	case w == 16:
		for y := range h {
			av := archsimd.LoadUint8x16(a[y*bps:])
			bv := archsimd.LoadUint8x16(b[y*bps:])
			d := av.SubSaturated(bv).Or(bv.SubSaturated(av))
			acc = sumSquares16(d.ExtendLo8ToUint16(), acc)
			acc = sumSquares16(hiToLoBytes(d).ExtendLo8ToUint16(), acc)
		}
	case w == 4 && h == 4:
		av := loadBlock4x4(a)
		bv := loadBlock4x4(b)
		d := av.SubSaturated(bv).Or(bv.SubSaturated(av))
		acc = sumSquares16(d.ExtendLo8ToUint16(), acc)
		acc = sumSquares16(hiToLoBytes(d).ExtendLo8ToUint16(), acc)
	default:
		return getSSEGeneric(a, b, w, h)
	}
	var lanes [4]uint32
	acc.StoreArray(&lanes)
	return int64(lanes[0]) + int64(lanes[1]) + int64(lanes[2]) + int64(lanes[3])
}

// quantizeBlockSIMD is the vector form of quantizeBlockGeneric, eight coefficients at a time and branch-free: the
// "coeff > zthresh" test becomes a lane predicate, and the levels are permuted into zigzag order on the way out rather
// than gathered one at a time on the way in (which would need the same shuffle plus its inverse for the write-back).
// The portable form's loop-carried "last" index needs no vector equivalent, because all the caller learns from it is
// whether any level was non-zero.
//
// Lane bounds, over the whole input domain (in is an arbitrary int16 array): the magnitude reaches 32768, which is why
// it is carried unsigned; sharpen tops out at 12 across every matrix the encoder can build, so coeff stays inside 16
// bits. The reciprocal iq is (1 << 17) / q with q at least 4 (the smallest entry of any quantizer table), so iq is at
// most 32768 and coeff * iq + bias reaches 1073800704 — a quarter of an unsigned 32-bit lane. zthresh tops out at 255,
// so it fits the 16-bit compare. TestQuantMatrixSIMDDomain enumerates every quantizer index of every matrix type to
// keep those three facts true.
func quantizeBlockSIMD(in, out *[16]int16, m *matrix) int {
	maxLvl := archsimd.BroadcastUint32x4(maxLevel)
	zeroI16 := archsimd.BroadcastInt16x8(0)
	zeroU16 := archsimd.BroadcastUint16x8(0)
	var levels [2]archsimd.Int16x8
	var nz archsimd.Uint16x8
	for h := range 2 {
		k := h * 8
		v := archsimd.LoadInt16x8(in[k:])
		// The magnitude of the most negative int16 is 32768, which stays exact in an unsigned lane — the same value
		// the portable form's uint32(-int(in[j])) produces.
		coeff := v.Abs().ToBits().Add(archsimd.LoadUint16x8(m.sharpen[k:]))
		// "coeff > zthresh" is the saturating difference being non-zero, which needs no unsigned compare.
		keep := coeff.SubSaturated(archsimd.LoadUint16x8(m.zthresh16[k:])).NotEqual(zeroU16)
		lo, hi := mulWiden16(coeff, archsimd.LoadUint16x8(m.iq16[k:]))
		lo = lo.Add(archsimd.LoadUint32x4(m.bias[k:])).ShiftAllRight(qFix).Min(maxLvl)
		hi = hi.Add(archsimd.LoadUint32x4(m.bias[k+4:])).ShiftAllRight(qFix).Min(maxLvl)
		lvl := narrow32Pair(lo, hi).Masked(keep)
		nz = nz.Or(lvl)
		signed := lvl.BitsToInt16()
		signed = signed.Neg().IfElse(v.Less(zeroI16), signed)
		levels[h] = signed
		// The dequantized coefficient is the int16 truncation of level * q, which is the low half of the 16-bit
		// product regardless of sign.
		signed.Mul(archsimd.LoadUint16x8(m.q[k:]).BitsToInt16()).Store(in[k:])
	}
	b0 := levels[0].ToBits().ReshapeToUint8s()
	b1 := levels[1].ToBits().ReshapeToUint8s()
	byteShuffle(b0, loadShuffle(&zigzagGatherIdx[0])).
		Or(byteShuffle(b1, loadShuffle(&zigzagGatherIdx[1]))).
		ReshapeToUint16s().BitsToInt16().Store(out[0:])
	byteShuffle(b1, loadShuffle(&zigzagGatherIdx[2])).
		Or(byteShuffle(b0, loadShuffle(&zigzagGatherIdx[3]))).
		ReshapeToUint16s().BitsToInt16().Store(out[8:])
	if anyNonZero16(nz) {
		return 1
	}
	return 0
}
