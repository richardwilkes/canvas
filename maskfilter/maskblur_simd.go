// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package maskfilter

import (
	"encoding/binary"
	"simd/archsimd"
)

// The goexperiment.simd kernels for the small-sigma direct convolution. fp88 is a literal 8-lane 8.8 fixed-point
// vector, so it maps one-to-one onto archsimd's Uint16x8 and the port is a transcription rather than a redesign:
//
//   - fp88Add is a wrapping 16-bit lane add, which is what Uint16x8.Add does (AddSaturated would not be — the
//     accumulators are allowed to wrap, and the probe confirms both forms wrap identically).
//   - mulhi is the unsigned 16x16 high half, spelled per arch by mulhiV (VPMULHUW on amd64; widened VUMULL products
//     with the odd halves gathered on arm64).
//   - loadA8/store are a widen-with-<<8 and a >>8-narrow. The narrow needs no saturation reasoning: v[i] is a uint16,
//     so v[i]>>8 can never exceed 255 and "take the high byte of each lane" is exact — which is all narrowHiBytes
//     does, with a byte gather rather than a pack.
//   - addShifted is a cross-lane shift across the (d0, d8) register pair, which is the EXT/VPALIGNR family. archsimd
//     spells it x.ConcatShiftBytesRight(y, s), and the probe pinned the lane order: the result is bytes s..s+15 of the
//     32-byte concatenation y ++ x, i.e. the *receiver* supplies the high half. So shifting v up by "offset" lanes is
//     v.ConcatShiftBytesRight(zero, 16-2*offset), and the lanes that spill past lane 7 into d8 are the complementary
//     zero.ConcatShiftBytesRight(v, 16-2*offset).
//
// Every one of those is exact integer arithmetic, so the kernels are bit-identical to the fp88 forms, not merely close
// — locked by TestMaskBlurSIMDMatchesScalar, which fuzzes whole rects (scalar drivers vs. simd drivers) and requires
// byte-identical output buffers.
//
// Two structural rules keep them honest. First, the shift immediates must be compile-time constants for the hardware
// instruction to be used at all, so both passes are unrolled per radius — the four cases of blurRowSIMD's and
// blurColumnSIMD's switches — instead of looping over a dynamic offset; the radius is fixed for a whole rect, so the
// switch costs one predictable branch per row or per column strip. Second, only whole 8-lane groups run in vector
// registers: every partial group, every unexpected radius, and every edge phase is handed back to the scalar helpers
// (blurRowTail, blurColumn, blurRow) rather than re-implemented here.

// init swaps the dispatch variables to the simd kernels. On arm64 everything they need is baseline NEON; on amd64
// simdKernelsSupported gates on AVX2, falling back to the portable fp88 dispatch when the hardware does not qualify.
func init() {
	if !simdKernelsSupported() {
		return
	}
	directBlurXFn = directBlurXSIMD
	directBlurYFn = directBlurYSIMD
}

///////////////////////////////////////////////////////////////////////////////
// shared vector helpers

// loadA8V is loadA8 for a full 8-lane group: eight A8 bytes widened into 8.8 fixed point. Every call site is inside a
// main loop that has already proven eight readable bytes remain (the row loops stop at srcW-8, and both callers' rows
// are at least srcW bytes long), and the u64 read bounds-checks that claim anyway. Reading the group as one u64 keeps
// this to a scalar load plus a lane insert; a 16-byte vector load would over-read the final group of the final row.
// The little-endian byte order the u64 assumes is guaranteed by this file's build tag.
func loadA8V(from []uint8) archsimd.Uint16x8 {
	return widenA8(archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(from)).ReshapeToUint8s())
}

// storeA8 is store for a full 8-lane group: the high byte of each 8.8 lane written as eight A8 bytes. Like loadA8V it
// moves the group as one u64 rather than as a vector store, so it can never touch the byte after the eighth.
func storeA8(to []uint8, v archsimd.Uint16x8, n a8Narrow) {
	binary.LittleEndian.PutUint64(to, narrowHiBytes(v, n).ReshapeToUint64s().GetElem(0))
}

// The addShiftedNV family is addShifted with the offset fixed at N: v's lanes move up N places into d0 and the N lanes
// that fall off the top land in d8's bottom N lanes. Both halves come from the same byte shift, 16-2*N, with the roles
// of v and zero swapped (see this file's header for the lane order that pins which operand goes where). Offset 0 is
// not in the family because it needs no shift at all — the callers just add v straight into d0.

func addShifted1V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 14).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 14).ReshapeToUint16s())
}

func addShifted2V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 12).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 12).ReshapeToUint16s())
}

func addShifted3V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 10).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 10).ReshapeToUint16s())
}

func addShifted4V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 8).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 8).ReshapeToUint16s())
}

func addShifted5V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 6).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 6).ReshapeToUint16s())
}

func addShifted6V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 4).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 4).ReshapeToUint16s())
}

func addShifted7V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 2).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 2).ReshapeToUint16s())
}

// addShifted8V is the degenerate end of the family: the whole vector clears d0 and lands in d8. The general formula
// still produces it (byte shift 0 makes the first result the all-zero operand and the second the whole of v), so it is
// spelled the same way as its siblings rather than special-cased.
func addShifted8V(d0, d8, v archsimd.Uint16x8, zero archsimd.Uint8x16) (nd0, nd8 archsimd.Uint16x8) {
	b := v.ReshapeToUint8s()
	return d0.Add(b.ConcatShiftBytesRight(zero, 0).ReshapeToUint16s()),
		d8.Add(zero.ConcatShiftBytesRight(b, 0).ReshapeToUint16s())
}

///////////////////////////////////////////////////////////////////////////////
// the horizontal pass

// blurRowSIMD is blurRow with the 8-wide main loop in vector registers: the accumulator pair lives in two registers
// across the whole row, and only the leftover source bytes and the trailing destination bytes go back to the scalar
// blurRowTail — which needs the pair as fp88s, so the two registers are spilled to it once per row. A radius outside
// the 1..4 the small-sigma path produces (sigma below the no-blur cutoff would give 0) has no unrolled form and runs
// the scalar row whole.
//
// Each case is blurXRadius for one radius, unrolled inline: the products v[k] = mulhi(s, g[k]) accumulated into the
// (d0, d8) pair at offsets j = 0..2*radius, with the factor for offset j taken from index |radius-j| — the same
// symmetric walk the scalar loop performs, with every shift immediate now a constant. The chain is written out here
// rather than called through a per-radius helper because a call cannot be inlined at these sizes, and passing the
// accumulator pair and the factors through the ABI costs about a fifth of the loop's time. The adds are 16-bit lane
// adds that may wrap, and wrapping addition is associative and commutative, so this order is bit-identical to the
// scalar's ascending-j order however far the accumulators wrap.
func blurRowSIMD(radius int, g *[5]fp88, gv *[5]archsimd.Uint16x8, src []uint8, srcW int, dst []uint8, dstW int, n a8Narrow) {
	if radius < 1 || radius > 4 {
		blurRow(radius, g, src, srcW, dst, dstW)
		return
	}
	var zero archsimd.Uint8x16
	half := archsimd.BroadcastUint16x8(fpHalf)
	d0, d8 := half, half
	x, si, di := 0, 0, 0
	switch radius {
	case 1:
		g0, g1 := gv[0], gv[1]
		for ; x <= srcW-8; x += 8 {
			s := loadA8V(src[si:])
			v0 := mulhiV(s, g0)
			v1 := mulhiV(s, g1)
			d0 = d0.Add(v1)
			d0, d8 = addShifted1V(d0, d8, v0, zero)
			d0, d8 = addShifted2V(d0, d8, v1, zero)
			storeA8(dst[di:], d0, n)
			d0, d8 = d8, half
			si += 8
			di += 8
		}
	case 2:
		g0, g1, g2 := gv[0], gv[1], gv[2]
		for ; x <= srcW-8; x += 8 {
			s := loadA8V(src[si:])
			v0 := mulhiV(s, g0)
			v1 := mulhiV(s, g1)
			v2 := mulhiV(s, g2)
			d0 = d0.Add(v2)
			d0, d8 = addShifted1V(d0, d8, v1, zero)
			d0, d8 = addShifted2V(d0, d8, v0, zero)
			d0, d8 = addShifted3V(d0, d8, v1, zero)
			d0, d8 = addShifted4V(d0, d8, v2, zero)
			storeA8(dst[di:], d0, n)
			d0, d8 = d8, half
			si += 8
			di += 8
		}
	case 3:
		g0, g1, g2, g3 := gv[0], gv[1], gv[2], gv[3]
		for ; x <= srcW-8; x += 8 {
			s := loadA8V(src[si:])
			v0 := mulhiV(s, g0)
			v1 := mulhiV(s, g1)
			v2 := mulhiV(s, g2)
			v3 := mulhiV(s, g3)
			d0 = d0.Add(v3)
			d0, d8 = addShifted1V(d0, d8, v2, zero)
			d0, d8 = addShifted2V(d0, d8, v1, zero)
			d0, d8 = addShifted3V(d0, d8, v0, zero)
			d0, d8 = addShifted4V(d0, d8, v1, zero)
			d0, d8 = addShifted5V(d0, d8, v2, zero)
			d0, d8 = addShifted6V(d0, d8, v3, zero)
			storeA8(dst[di:], d0, n)
			d0, d8 = d8, half
			si += 8
			di += 8
		}
	default:
		g0, g1, g2, g3, g4 := gv[0], gv[1], gv[2], gv[3], gv[4]
		for ; x <= srcW-8; x += 8 {
			s := loadA8V(src[si:])
			v0 := mulhiV(s, g0)
			v1 := mulhiV(s, g1)
			v2 := mulhiV(s, g2)
			v3 := mulhiV(s, g3)
			v4 := mulhiV(s, g4)
			d0 = d0.Add(v4)
			d0, d8 = addShifted1V(d0, d8, v3, zero)
			d0, d8 = addShifted2V(d0, d8, v2, zero)
			d0, d8 = addShifted3V(d0, d8, v1, zero)
			d0, d8 = addShifted4V(d0, d8, v0, zero)
			d0, d8 = addShifted5V(d0, d8, v1, zero)
			d0, d8 = addShifted6V(d0, d8, v2, zero)
			d0, d8 = addShifted7V(d0, d8, v3, zero)
			d0, d8 = addShifted8V(d0, d8, v4, zero)
			storeA8(dst[di:], d0, n)
			d0, d8 = d8, half
			si += 8
			di += 8
		}
	}
	var t0, t8 fp88
	d0.StoreArray((*[8]uint16)(&t0))
	d8.StoreArray((*[8]uint16)(&t8))
	blurRowTail(radius, g, src[si:], srcW-x, dst[di:], dstW-x, t0, t8)
}

// directBlurXSIMD is the simd twin of directBlurX. The gauss factors are splatted once per rect in both forms — the
// vector one for the kernels, the fp88 one for the scalar tails — and so is the narrowing state, so the per-row call
// carries no setup.
func directBlurXSIMD(radius int, gauss *[5]uint16, src []uint8, srcStride, srcW int, dst []uint8, dstStride, dstW, dstH int) {
	var g [5]fp88
	var gv [5]archsimd.Uint16x8
	for i := range g {
		g[i] = fp88Splat(gauss[i])
		gv[i] = archsimd.BroadcastUint16x8(gauss[i])
	}
	n := newA8Narrow()
	si, di := 0, 0
	for y := 0; y < dstH; y++ {
		blurRowSIMD(radius, &g, &gv, src[si:], srcW, dst[di:], dstW, n)
		si += srcStride
		di += dstStride
	}
}

///////////////////////////////////////////////////////////////////////////////
// the vertical pass

// blurColumnSIMD is blurColumn for a full-width (8 lane) column strip. The vertical pass needs no cross-lane shifts at
// all — every lane is an independent column — so the whole win here is keeping blurYRadius's pipeline registers in
// vector registers for the length of the column instead of restacking an [8]fp88 array per row. That requires the
// radius to be fixed, hence the unrolled switch; each case is blurYRadius transcribed with the register indices
// resolved, including its ordering rule that regs[n] is updated from the not-yet-updated regs[n+1].
//
// Narrower strips (the ragged right edge) and any radius outside 1..4 go to the scalar blurColumn whole.
func blurColumnSIMD(radius int, g *[5]fp88, gv *[5]archsimd.Uint16x8, src []uint8, srcRB, srcH int, dst []uint8, dstRB int, n a8Narrow) {
	if radius < 1 || radius > 4 {
		blurColumn(radius, 8, g, src, srcRB, srcH, dst, dstRB)
		return
	}
	half := archsimd.BroadcastUint16x8(fpHalf)
	var regs [8]archsimd.Uint16x8
	di := 0
	switch radius {
	case 1:
		g0, g1 := gv[0], gv[1]
		r0, r1 := half, half
		si := 0
		for y := 0; y < srcH; y++ {
			s := loadA8V(src[si:])
			v0, v1 := mulhiV(s, g0), mulhiV(s, g1)
			answer := r0.Add(v1)
			r0 = r1.Add(v0)
			r1 = v1.Add(half)
			storeA8(dst[di:], answer, n)
			si += srcRB
			di += dstRB
		}
		regs[0], regs[1] = r0, r1
	case 2:
		g0, g1, g2 := gv[0], gv[1], gv[2]
		r0, r1, r2, r3 := half, half, half, half
		si := 0
		for y := 0; y < srcH; y++ {
			s := loadA8V(src[si:])
			v0, v1, v2 := mulhiV(s, g0), mulhiV(s, g1), mulhiV(s, g2)
			answer := r0.Add(v2)
			r0 = r1.Add(v1)
			r1 = r2.Add(v0)
			r2 = r3.Add(v1)
			r3 = v2.Add(half)
			storeA8(dst[di:], answer, n)
			si += srcRB
			di += dstRB
		}
		regs[0], regs[1], regs[2], regs[3] = r0, r1, r2, r3
	case 3:
		g0, g1, g2, g3 := gv[0], gv[1], gv[2], gv[3]
		r0, r1, r2, r3, r4, r5 := half, half, half, half, half, half
		si := 0
		for y := 0; y < srcH; y++ {
			s := loadA8V(src[si:])
			v0, v1, v2, v3 := mulhiV(s, g0), mulhiV(s, g1), mulhiV(s, g2), mulhiV(s, g3)
			answer := r0.Add(v3)
			r0 = r1.Add(v2)
			r1 = r2.Add(v1)
			r2 = r3.Add(v0)
			r3 = r4.Add(v1)
			r4 = r5.Add(v2)
			r5 = v3.Add(half)
			storeA8(dst[di:], answer, n)
			si += srcRB
			di += dstRB
		}
		regs[0], regs[1], regs[2], regs[3], regs[4], regs[5] = r0, r1, r2, r3, r4, r5
	default:
		g0, g1, g2, g3, g4 := gv[0], gv[1], gv[2], gv[3], gv[4]
		r0, r1, r2, r3, r4, r5, r6, r7 := half, half, half, half, half, half, half, half
		si := 0
		for y := 0; y < srcH; y++ {
			s := loadA8V(src[si:])
			v0, v1, v2, v3 := mulhiV(s, g0), mulhiV(s, g1), mulhiV(s, g2), mulhiV(s, g3)
			v4 := mulhiV(s, g4)
			answer := r0.Add(v4)
			r0 = r1.Add(v3)
			r1 = r2.Add(v2)
			r2 = r3.Add(v1)
			r3 = r4.Add(v0)
			r4 = r5.Add(v1)
			r5 = r6.Add(v2)
			r6 = r7.Add(v3)
			r7 = v4.Add(half)
			storeA8(dst[di:], answer, n)
			si += srcRB
			di += dstRB
		}
		regs[0], regs[1], regs[2], regs[3] = r0, r1, r2, r3
		regs[4], regs[5], regs[6], regs[7] = r4, r5, r6, r7
	}

	// Flush the pipeline registers: radius pairs of trailing rows, in register order.
	for i := 0; i < 2*radius; i++ {
		storeA8(dst[di:], regs[i], n)
		di += dstRB
	}
}

// directBlurYSIMD is the simd twin of directBlurY: full 8-wide column strips run the vector kernel, and the ragged
// right edge falls back to the scalar blurColumn, exactly as the scalar driver's own tail does.
func directBlurYSIMD(radius int, gauss *[5]uint16, src []uint8, srcRB, srcW, srcH int, dst []uint8, dstRB int) {
	var g [5]fp88
	var gv [5]archsimd.Uint16x8
	for i := range g {
		g[i] = fp88Splat(gauss[i])
		gv[i] = archsimd.BroadcastUint16x8(gauss[i])
	}
	n := newA8Narrow()
	x := 0
	si, di := 0, 0
	for ; x <= srcW-8; x += 8 {
		blurColumnSIMD(radius, &g, &gv, src[si:], srcRB, srcH, dst[di:], dstRB, n)
		si += 8
		di += 8
	}
	if xTail := srcW - x; xTail > 0 {
		blurColumn(radius, xTail, &g, src[si:], srcRB, srcH, dst[di:], dstRB)
	}
}
