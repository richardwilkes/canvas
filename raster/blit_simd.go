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

import "simd/archsimd"

// The goexperiment.simd blit rows, written against archsimd's 128-bit types for the same reason the span kernels are
// (see span_simd.go): it is the widest shape arm64 and amd64 share, so both vector lanes compute the same expressions
// in the same order. These are all-integer kernels; the shared machinery they build on is documented there, in
// particular the hoisted shift counts (shift16/shift32, needed because arm64 re-materializes a count vector per
// iteration otherwise) and the "gather four coverage bytes through a scalar word" idiom, which exists because
// LoadUint8x16Part is an out-of-line call.
//
// Two lane layouts recur below, both of them the vector spelling of the portable SWAR forms:
//
//   - The 8888 byte split. Viewing a Uint32x4 of four device words as a Uint16x8 puts pixel i's bytes 0 and 1 in lane
//     2i and its bytes 2 and 3 in lane 2i+1. Masking with 0x00FF then leaves channels 0 and 2 (one per lane, 8 bits of
//     headroom each) and shifting right by 8 leaves channels 1 and 3, so a pair of vectors carries all four channels of
//     four pixels with every byte in its own 16-bit lane. Each kernel's comment states the bound that keeps its
//     products inside those lanes; recombining the halves with lo | hi<<8 is exact whenever both are bytes, and where
//     the portable form ends in a 32-bit *add* (which may carry across channels) the vector form does too, on the
//     recombined words, rather than adding per lane.
//   - The 565 duplicate. lcd16Load gathers four subpixel coverage values, and interleaving that vector with itself puts
//     mask[i] in both of pixel i's lanes, so one vector supplies the even-lane channel's coverage and the odd-lane
//     channel's coverage at once. evenLanes16 then selects which channel each lane takes.
//
// Every kernel here is locked against its portable twin by TestBlitSIMDMatchesScalar, whose integer subtests enumerate
// complete domains wherever they are small enough to.

// init swaps the dispatch variables to the simd kernels the per-arch preference constants elect. Two gates the span
// kernels use are deliberately not the gate here: simdKernelsPreferred answers whether archsimd beats span_arm64.s,
// which none of these rows has to lose to, and simdKernelsSupported carries the shader pipeline's AVX2+FMA floor,
// which these all-integer rows do not need (see simdBlitSupported).
func init() {
	if !simdBlitSupported() {
		return
	}
	if preferSIMDFillWords {
		fillWordsFn = fillWordsSIMD
	}
	if preferSIMDFillBytes {
		fillBytesFn = fillBytesSIMD
	}
	if preferSIMDColor32Row {
		color32RowFn = color32RowSIMD
	}
	if preferSIMDBlitMaskTranslucentRow {
		blitMaskTranslucentRowFn = blitMaskTranslucentRowSIMD
	}
	if preferSIMDInterp256Row {
		interp256RowFn = interp256RowSIMD
	}
	if preferSIMDPremulRow {
		premulRowFn = premulRowSIMD
	}
	if preferSIMDPMBlendRow {
		pmBlendRowFn = pmBlendRowSIMD
	}
	if preferSIMDBlitRowLCD16 {
		blitRowLCD16Fn = blitRowLCD16SIMD
	}
	if preferSIMDBlitRowLCD16Opaque {
		blitRowLCD16OpaqueFn = blitRowLCD16OpaqueSIMD
	}
	if preferSIMDBlendRowLCD16Opaque {
		blendRowLCD16OpaqueFn = blendRowLCD16OpaqueSIMD
	}
}

// fillWordsSIMD fills a device span with one word. The wide body issues four independent 128-bit stores per iteration,
// which is worth a few percent over the single-store loop on a span this size — the loop is nothing but stores, so what
// is left to save is the per-iteration bookkeeping. Spans too short for it fall through to the single-store loop and
// then to the portable tail.
func fillWordsSIMD(dst []uint32, v uint32) {
	n := len(dst)
	vv := archsimd.BroadcastUint32x4(v)
	i := 0
	for ; i+16 <= n; i += 16 {
		vv.Store(dst[i:])
		vv.Store(dst[i+4:])
		vv.Store(dst[i+8:])
		vv.Store(dst[i+12:])
	}
	for ; i+4 <= n; i += 4 {
		vv.Store(dst[i:])
	}
	if i < n {
		fillWordsGeneric(dst[i:], v)
	}
}

// fillBytesSIMD fills an A8 coverage row with one byte, structured exactly like fillWordsSIMD and for the same reason.
func fillBytesSIMD(dst []uint8, v uint8) {
	n := len(dst)
	vv := archsimd.BroadcastUint8x16(v)
	i := 0
	for ; i+64 <= n; i += 64 {
		vv.Store(dst[i:])
		vv.Store(dst[i+16:])
		vv.Store(dst[i+32:])
		vv.Store(dst[i+48:])
	}
	for ; i+16 <= n; i += 16 {
		vv.Store(dst[i:])
	}
	if i < n {
		fillBytesGeneric(dst[i:], v)
	}
}

// color32RowSIMD scales a device span by invA and adds a constant premultiplied color: per byte (d*invA)>>8, then one
// 32-bit add of the packed result and the color. Four pixels per iteration, the sub-quad tail on the portable form.
//
// invA is a 0..256 scale, so every byte product is at most 255*256 = 0xFF00 and stays inside its 16-bit lane, and
// (d*invA)>>8 is at most 255 — which is both what makes the halves recombine exactly with lo | hi<<8 and why the
// portable form's & 0x00FF00FF after its shift is a no-op on the values (it only clears the SWAR word's cross-channel
// bleed, which separate lanes do not have). The final add runs on the recombined 32-bit words, so it carries between
// channels exactly where the portable form's uint32 add would.
func color32RowSIMD(dst []uint32, color, invA uint32) {
	n := len(dst)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	inv := archsimd.BroadcastUint16x8(uint16(invA))
	col := archsimd.BroadcastUint32x4(color)
	r8, l8 := rightShift16(8), leftShift16(8)
	i := 0
	for ; i+4 <= n; i += 4 {
		d16 := archsimd.LoadUint32x4(dst[i:]).ReshapeToUint16s()
		lo := shift16(d16.And(byteMask).Mul(inv), r8)
		hi := shift16(shift16(d16, r8).Mul(inv), r8)
		lo.Or(shift16(hi, l8)).ReshapeToUint32s().Add(col).Store(dst[i:])
	}
	if i < n {
		color32RowGeneric(dst[i:], color, invA)
	}
}

// blitMaskTranslucentRowSIMD blends a translucent solid color through an A8 coverage row: per pixel alphaMulQ(pm, m+1)
// + alphaMulQ(dev, 256 - ((srcA*(m+1))>>8)). Four pixels per iteration, the sub-quad tail on the portable form.
//
// Both alphaMulQ scales are in [1, 256]: m+1 by construction, and the destination's because (srcA*(m+1))>>8 <= 255 for
// srcA <= 255. So every byte product is at most 255*256 = 0xFF00 inside its 16-bit lane and every scaled byte is at
// most 255, which is what lets each half recombine with lo | hi<<8 before the two packed words are added as 32-bit
// lanes — the same add, carries included, the portable form performs.
func blitMaskTranslucentRowSIMD(dev []uint32, aa []uint8, pm, srcA uint32) {
	n := len(aa)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	one := archsimd.BroadcastUint16x8(1)
	c256 := archsimd.BroadcastUint16x8(256)
	sa := archsimd.BroadcastUint16x8(uint16(srcA))
	r8, l8 := rightShift16(8), leftShift16(8)
	pmw := archsimd.BroadcastUint32x4(pm).ReshapeToUint16s()
	pmLo := pmw.And(byteMask)
	pmHi := shift16(pmw, r8)
	i := 0
	for ; i+4 <= n; i += 4 {
		// The quad's four coverage bytes come through a scalar word, as in blitMaskOpaqueRowSIMD: the extend lifts
		// them to 16-bit lanes and the self-interleave duplicates each into the two lanes of its pixel.
		m32 := uint32(aa[i]) | uint32(aa[i+1])<<8 | uint32(aa[i+2])<<16 | uint32(aa[i+3])<<24
		mb := archsimd.BroadcastUint32x4(m32).ReshapeToUint8s().ExtendLo8ToUint16()
		m256 := mb.InterleaveLo(mb).Add(one)
		scale := c256.Sub(shift16(sa.Mul(m256), r8))
		d16 := archsimd.LoadUint32x4(dev[i:]).ReshapeToUint16s()
		srcLo := shift16(pmLo.Mul(m256), r8)
		srcHi := shift16(pmHi.Mul(m256), r8)
		dstLo := shift16(d16.And(byteMask).Mul(scale), r8)
		dstHi := shift16(shift16(d16, r8).Mul(scale), r8)
		srcLo.Or(shift16(srcHi, l8)).ReshapeToUint32s().
			Add(dstLo.Or(shift16(dstHi, l8)).ReshapeToUint32s()).Store(dev[i:])
	}
	if i < n {
		blitMaskTranslucentRowGeneric(dev[i:n], aa[i:n], pm, srcA)
	}
}

// interp256RowSIMD blends an opaque source row into dst under one 0..256 scale: per byte (s*scale + d*(256-scale))>>8,
// the row form of fastFourByteInterp256. Four pixels per iteration, the sub-quad tail on the portable form.
//
// Each lane holds a convex combination of two bytes weighted by 256, so it is at most 255*256 = 0xFF00 and never leaves
// its 16-bit lane — the same headroom argument the portable form makes for its four 16-bit fields inside a uint64 —
// and every result byte is at most 255, so the halves recombine exactly with lo | hi<<8.
func interp256RowSIMD(dst, src []uint32, scale uint32) {
	n := len(src)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	sc := archsimd.BroadcastUint16x8(uint16(scale))
	nsc := archsimd.BroadcastUint16x8(uint16(256 - scale))
	r8, l8 := rightShift16(8), leftShift16(8)
	i := 0
	for ; i+4 <= n; i += 4 {
		s16 := archsimd.LoadUint32x4(src[i:]).ReshapeToUint16s()
		d16 := archsimd.LoadUint32x4(dst[i:]).ReshapeToUint16s()
		lo := shift16(s16.And(byteMask).Mul(sc).Add(d16.And(byteMask).Mul(nsc)), r8)
		hi := shift16(shift16(s16, r8).Mul(sc).Add(shift16(d16, r8).Mul(nsc)), r8)
		lo.Or(shift16(hi, l8)).ReshapeToUint32s().Store(dst[i:])
	}
	if i < n {
		interp256RowGeneric(dst[i:], src[i:n], scale)
	}
}

// premulRowSIMD premultiplies a row of straight-alpha words: every color channel scaled by the pixel's alpha with the
// round-half-up div255 (mulDiv255Round), alpha carried across untouched. Four pixels per iteration, the sub-quad tail
// on the portable form.
//
// The lane bound is premul8888's: prod = b*a + 128 <= 255*255+128 = 0xFE81, prod>>8 <= 0xFE, and prod + (prod>>8) <=
// 0xFF7F, so nothing leaves its 16-bit lane and (prod + (prod>>8)) >> 8 is exactly mulDiv255Round's value. The alpha
// lane's product is computed and discarded, as in the portable form, which then reinstates the original alpha byte;
// the vector does that with the 0x00FFFFFF / 0xFF000000 pair. premul8888's a == 0xFF early-out needs no counterpart:
// mulDiv255Round(b, 255) == b for every byte, so the general path already returns such a pixel unchanged.
func premulRowSIMD(dst, src []uint32) {
	n := len(src)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	half := archsimd.BroadcastUint16x8(128)
	spread := archsimd.BroadcastUint32x4(0x00010001)
	rgbMask := archsimd.BroadcastUint32x4(0x00FFFFFF)
	alphaMask := archsimd.BroadcastUint32x4(0xFF000000)
	r8, l8, r24 := rightShift16(8), leftShift16(8), rightShift32(24)
	i := 0
	for ; i+4 <= n; i += 4 {
		s32 := archsimd.LoadUint32x4(src[i:])
		s16 := s32.ReshapeToUint16s()
		// The alpha byte, replicated into both 16-bit lanes of its pixel. The multiply by 0x00010001 cannot overflow
		// a 32-bit lane (alpha <= 255) and is the cheapest way to reach both halves at once.
		a := shift32(s32, r24).Mul(spread).ReshapeToUint16s()
		lo := s16.And(byteMask).Mul(a).Add(half)
		lo = shift16(lo.Add(shift16(lo, r8)), r8)
		hi := shift16(s16, r8).Mul(a).Add(half)
		hi = shift16(hi.Add(shift16(hi, r8)), r8)
		lo.Or(shift16(hi, l8)).ReshapeToUint32s().And(rgbMask).Or(s32.And(alphaMask)).Store(dst[i:])
	}
	if i < n {
		premulRowGeneric(dst[i:], src[i:n])
	}
}

// pmBlendRowSIMD blends a premultiplied source row into dst under one 0..256 scale: per channel (s*alpha256 +
// d*alphaMulInv256(srcA, alpha256)) >> 8. Four pixels per iteration, the sub-quad tail on the portable form.
//
// The destination scale is derived in 32-bit lanes, where 0xFFFF - srcA*alpha256 and the (x + (x>>8)) >> 8 divide have
// room, and is then replicated into both 16-bit lanes of its pixel; it lands in [0, 256], as does alpha256, so each of
// the two byte products is at most 255*256 = 0xFF00 and fits a 16-bit lane. Their *sum* does not (it can reach
// 0x1FE00), so the shift is split by the exact identity (P+Q)>>8 = (P>>8) + (Q>>8) + (((P&0xFF) + (Q&0xFF)) >> 8),
// whose three terms sum to at most 511 and so stay in the lane.
//
// That 511 is also why the halves recombine through 32-bit lanes rather than lo | hi<<8: a channel over 255 must spill
// into the next channel's byte exactly as the portable form's r | g<<8 | b<<16 | a<<24 spills it. Reshaping the low
// vector to 32-bit lanes yields r | b<<16 with r's ninth bit already at bit 8, and the high vector shifted left by 8
// yields g<<8 | a<<24 with the same alignment (and, like the portable form's uint32, drops alpha's ninth bit off the
// top), so OR-ing the two reproduces the packed word bit for bit.
func pmBlendRowSIMD(dst, src []uint32, alpha256 uint32) {
	n := len(src)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	a16 := archsimd.BroadcastUint16x8(uint16(alpha256))
	a32 := archsimd.BroadcastUint32x4(alpha256)
	c65535 := archsimd.BroadcastUint32x4(0xFFFF)
	spread := archsimd.BroadcastUint32x4(0x00010001)
	r8 := rightShift16(8)
	r8w, l8w, r24 := rightShift32(8), leftShift32(8), rightShift32(24)
	i := 0
	for ; i+4 <= n; i += 4 {
		s32 := archsimd.LoadUint32x4(src[i:])
		d32 := archsimd.LoadUint32x4(dst[i:])
		prod := c65535.Sub(shift32(s32, r24).Mul(a32))
		ds := shift32(prod.Add(shift32(prod, r8w)), r8w).Mul(spread).ReshapeToUint16s()
		s16, d16 := s32.ReshapeToUint16s(), d32.ReshapeToUint16s()
		lo := shiftSum8(s16.And(byteMask).Mul(a16), d16.And(byteMask).Mul(ds), byteMask, r8)
		hi := shiftSum8(shift16(s16, r8).Mul(a16), shift16(d16, r8).Mul(ds), byteMask, r8)
		lo.ReshapeToUint32s().Or(shift32(hi.ReshapeToUint32s(), l8w)).Store(dst[i:])
	}
	if i < n {
		pmBlendRowGeneric(dst[i:], src[i:n], alpha256)
	}
}

// shiftSum8 is (p + q) >> 8 for 16-bit lanes whose sum may not fit in one: the high halves are added directly and the
// low halves contribute only their carry. Exact for every p, q, since p + q = ((p>>8) + (q>>8))<<8 + (p&0xFF) +
// (q&0xFF) and the trailing pair is at most 510, so its ninth bit is the carry.
func shiftSum8(p, q, byteMask archsimd.Uint16x8, r8 shiftCount16) archsimd.Uint16x8 {
	return shift16(p, r8).Add(shift16(q, r8)).Add(shift16(p.And(byteMask).Add(q.And(byteMask)), r8))
}

// lcd16Load gathers four 565 subpixel coverage values into the low half of a vector (and, harmlessly, a copy of them
// into the high half). Assembling them into a scalar 64-bit word and broadcasting that beats LoadUint16x8Part, which is
// an out-of-line call, and unlike LoadUint16x8 it does not require four further mask values to be in bounds — glyph
// mask rows are narrow, so a kernel that could only advance eight pixels at a time would spend most rows in its tail.
func lcd16Load(mask []uint16) archsimd.Uint16x8 {
	m := uint64(mask[0]) | uint64(mask[1])<<16 | uint64(mask[2])<<32 | uint64(mask[3])<<48
	return archsimd.BroadcastUint64x2(m).ReshapeToUint16s()
}

// evenLanes16 is the mask selecting the even 16-bit lanes — the ones holding channels 0 and 2 of the four device words
// a Uint32x4 carries. It is built from a broadcast inside each kernel rather than held in a package-level variable,
// since a package-level archsimd value would execute vector code during init on CPUs that fail the gate.
func evenLanes16() archsimd.Mask16x8 {
	return archsimd.BroadcastUint32x4(0x00010000).ReshapeToUint16s().Equal(archsimd.BroadcastUint16x8(0))
}

// upscale31To32v is the lanewise upscale31To32: v + v>>4, mapping a 5-bit coverage onto 0..32.
func upscale31To32v(v archsimd.Uint16x8, r4 shiftCount16) archsimd.Uint16x8 {
	return v.Add(shift16(v, r4))
}

// blend32v is the lanewise blend32: dst + ((src-dst)*scale >> 5), with the arithmetic shift the scalar's int32 math
// performs. Every lane of src and dst must be a byte and every lane of scale in [0, 32], which keeps (src-dst)*scale
// inside [-8160, 8160] — an int16 — and the result inside [0, 255], since the shift's truncation toward negative
// infinity leaves the value between dst and src. The difference and the product are computed unsigned because the low
// 16 bits of a signed and an unsigned product agree; only the shift needs the signed view.
func blend32v(src, dst, scale archsimd.Uint16x8, r5 shiftCount16) archsimd.Uint16x8 {
	return dst.Add(shiftI16(src.Sub(dst).Mul(scale).BitsToInt16(), r5).ToBits())
}

// blitRowLCD16OpaqueSIMD blends an opaque solid color through a 565 subpixel coverage row. Four pixels per iteration,
// the sub-quad tail on the portable form. srcR/G/B must be bytes, which is what keeps blend32v inside its lane.
//
// The channel pairing follows the 8888 byte split: the low vector's even lanes hold the destination's red and its odd
// lanes the destination's blue, so its scale vector takes the red coverage on even lanes and the blue coverage on odd
// ones; the high vector holds green and alpha the same way, so its scale vector pairs green's coverage with alpha's,
// which the reference takes to be the max of the three.
//
// The kernel is branch-free where the portable form has two shortcuts, and reproduces both. A zero mask needs no
// counterpart at all: it makes every blend32 scale zero, which returns each destination channel unchanged, alpha
// included, since blend32(0xFF, dstA, 0) is dstA. A 0xFFFF mask is instead selected explicitly — it does drive every
// channel to the source, but the portable form substitutes the caller's opaqueDst word, which equals that only for a
// caller whose opaqueDst really is this color's device word (the blitter's always is; the row proc's signature does not
// require it).
func blitRowLCD16OpaqueSIMD(dst []uint32, mask []uint16, srcR, srcG, srcB, opaqueDst uint32) {
	n := len(mask)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	c31 := archsimd.BroadcastUint16x8(0x1F)
	full := archsimd.BroadcastUint32x4(0xFFFF)
	opaque := archsimd.BroadcastUint32x4(opaqueDst)
	srcLo := archsimd.BroadcastUint32x4(srcR | srcB<<16).ReshapeToUint16s()
	srcHi := archsimd.BroadcastUint32x4(srcG | 0xFF<<16).ReshapeToUint16s()
	even := evenLanes16()
	r4, r5, r6, r8, r11, l8 := rightShift16(4), rightShift16(5), rightShift16(6), rightShift16(8), rightShift16(11),
		leftShift16(8)
	i := 0
	for ; i+4 <= n; i += 4 {
		md := lcd16Load(mask[i:])
		mdup := md.InterleaveLo(md)
		mr := upscale31To32v(shift16(mdup, r11).And(c31), r4)
		mg := upscale31To32v(shift16(mdup, r6).And(c31), r4)
		mb := upscale31To32v(mdup.And(c31), r4)
		d16 := archsimd.LoadUint32x4(dst[i:]).ReshapeToUint16s()
		lo := blend32v(srcLo, d16.And(byteMask), mr.IfElse(even, mb), r5)
		hi := blend32v(srcHi, shift16(d16, r8), mg.IfElse(even, mr.Max(mg).Max(mb)), r5)
		opaque.IfElse(md.ExtendLo4ToUint32().Equal(full), lo.Or(shift16(hi, l8)).ReshapeToUint32s()).Store(dst[i:])
	}
	if i < n {
		blitRowLCD16OpaqueGeneric(dst[i:], mask[i:n], srcR, srcG, srcB, opaqueDst)
	}
}

// blitRowLCD16SIMD blends a translucent solid color through a 565 subpixel coverage row. Four pixels per iteration, the
// sub-quad tail on the portable form.
//
// It is blitRowLCD16OpaqueSIMD with two additions from the reference: each coverage is first scaled by srcA256 (which
// is in [1, 256] for a srcA in [0, 255], so a 0..32 coverage times it is at most 0x2000 in its lane and is back inside
// 0..32 after the shift), and alpha's coverage is the min of the three scaled coverages where the source alpha is below
// the destination's rather than the max (skbug.com/40037823). The comparison runs in every lane; only the odd lanes'
// answer, where the destination's alpha byte sits, is consumed. There is no 0xFFFF shortcut in this lane of the
// reference, and the zero-mask shortcut again needs no counterpart. srcA and srcR/G/B must be bytes.
func blitRowLCD16SIMD(dst []uint32, mask []uint16, srcA, srcR, srcG, srcB uint32) {
	n := len(mask)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	c31 := archsimd.BroadcastUint16x8(0x1F)
	a256 := archsimd.BroadcastUint16x8(uint16(alpha255To256(srcA)))
	sa := archsimd.BroadcastUint16x8(uint16(srcA))
	srcLo := archsimd.BroadcastUint32x4(srcR | srcB<<16).ReshapeToUint16s()
	srcHi := archsimd.BroadcastUint32x4(srcG | 0xFF<<16).ReshapeToUint16s()
	even := evenLanes16()
	r4, r5, r6, r8, r11, l8 := rightShift16(4), rightShift16(5), rightShift16(6), rightShift16(8), rightShift16(11),
		leftShift16(8)
	i := 0
	for ; i+4 <= n; i += 4 {
		md := lcd16Load(mask[i:])
		mdup := md.InterleaveLo(md)
		mr := shift16(upscale31To32v(shift16(mdup, r11).And(c31), r4).Mul(a256), r8)
		mg := shift16(upscale31To32v(shift16(mdup, r6).And(c31), r4).Mul(a256), r8)
		mb := shift16(upscale31To32v(mdup.And(c31), r4).Mul(a256), r8)
		d16 := archsimd.LoadUint32x4(dst[i:]).ReshapeToUint16s()
		dHi := shift16(d16, r8)
		ma := mr.Min(mg).Min(mb).IfElse(sa.Less(dHi), mr.Max(mg).Max(mb))
		lo := blend32v(srcLo, d16.And(byteMask), mr.IfElse(even, mb), r5)
		hi := blend32v(srcHi, dHi, mg.IfElse(even, ma), r5)
		lo.Or(shift16(hi, l8)).ReshapeToUint32s().Store(dst[i:])
	}
	if i < n {
		blitRowLCD16Generic(dst[i:], mask[i:n], srcA, srcR, srcG, srcB)
	}
}

// blendRowLCD16OpaqueSIMD blends a shaded opaque premultiplied span through a 565 subpixel coverage row. Four pixels
// per iteration, the sub-quad tail on the portable form.
//
// It is blitRowLCD16OpaqueSIMD's kernel with the constant color replaced by the span's own words and two changes at the
// edges. Alpha is not blended at all here — the portable form writes a literal 0xFF into it — so the high half's odd
// lanes are overwritten with 0xFF after the blend rather than being given a coverage, and its scale vector can simply
// carry green's coverage in both lanes. And the portable form's zero-mask shortcut *is* observable in this lane, unlike
// the solid-color one: skipping the pixel leaves the destination's own alpha in place where the straight-line result
// would have forced it to 0xFF, so a zero mask selects the untouched destination word explicitly.
func blendRowLCD16OpaqueSIMD(dst []uint32, mask []uint16, src []uint32) {
	n := len(mask)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	c31 := archsimd.BroadcastUint16x8(0x1F)
	c255 := archsimd.BroadcastUint16x8(0xFF)
	zero := archsimd.BroadcastUint32x4(0)
	even := evenLanes16()
	r4, r5, r6, r8, r11, l8 := rightShift16(4), rightShift16(5), rightShift16(6), rightShift16(8), rightShift16(11),
		leftShift16(8)
	i := 0
	for ; i+4 <= n; i += 4 {
		md := lcd16Load(mask[i:])
		mdup := md.InterleaveLo(md)
		mr := upscale31To32v(shift16(mdup, r11).And(c31), r4)
		mg := upscale31To32v(shift16(mdup, r6).And(c31), r4)
		mb := upscale31To32v(mdup.And(c31), r4)
		s16 := archsimd.LoadUint32x4(src[i:]).ReshapeToUint16s()
		d32 := archsimd.LoadUint32x4(dst[i:])
		d16 := d32.ReshapeToUint16s()
		lo := blend32v(s16.And(byteMask), d16.And(byteMask), mr.IfElse(even, mb), r5)
		hi := blend32v(shift16(s16, r8), shift16(d16, r8), mg, r5).IfElse(even, c255)
		d32.IfElse(md.ExtendLo4ToUint32().Equal(zero), lo.Or(shift16(hi, l8)).ReshapeToUint32s()).Store(dst[i:])
	}
	if i < n {
		blendRowLCD16OpaqueGeneric(dst[i:], mask[i:n], src[i:n])
	}
}
