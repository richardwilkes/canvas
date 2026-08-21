// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package imagecore

import "simd/archsimd"

// The goexperiment.simd conversion rows, written against archsimd's 128-bit types for the same reason raster's are: it
// is the widest shape arm64 and amd64 share, so both vector lanes compute the same expressions in the same order. Every
// operation below was picked for bit-exactness with the portable form, not for convenience:
//
//   - The 8888 byte split. Viewing a Uint32x4 of four pixels as a Uint16x8 puts pixel i's bytes 0 and 1 in lane 2i and
//     its bytes 2 and 3 in lane 2i+1. Masking with 0x00FF then leaves R and B (one per lane, 8 bits of headroom each)
//     and shifting right by 8 leaves G and A, so a pair of vectors carries all four channels of four pixels with every
//     byte in its own 16-bit lane. This is raster's layout, and premulWordRowSIMD's bound below shows nothing carries
//     into a neighbor, so the packed results are identical to the scalar words.
//   - div255Round(x*a) is computed as the mulDiv255Round chain (prod = x*a + 128; (prod + prod>>8) >> 8), which is the
//     same expression the scalar (v+128 + (v+128)/256)/256 evaluates, in 16-bit lanes. Exhaustively checked against
//     the scalar over every (channel, alpha) pair, on both arches.
//   - unpremulChannelRP is transcribed literally, including its association: normA = a * (1/255), invA = 1/normA,
//     normC = c * (1/255), denorm = (normC * invA) * 255, round-to-nearest-even, clamp to 255. The division is a real
//     IEEE divide (VDIVPS / FDIV, both correctly rounded), and the a == 0 guard is applied to invA *before* any
//     float-to-int conversion — that is what removes the +Inf and the 0*Inf NaN the reciprocal would otherwise hand
//     ConvertToInt32, whose out-of-range and NaN results differ per arch. archsimd's ConvertToInt32 is the truncating
//     convert, so the rounding is done first by Round (VFRINTN / VROUNDPS in round-to-nearest-even mode), which is
//     exactly math.RoundToEven's rule; the convert then only moves an already-integral in-range value across. The whole
//     chain is checked exhaustively against the scalar over all 65536 (channel, alpha) pairs, on both arches.
//   - Destination rows are byte slices, so the results are stored through ReshapeToUint8s().Store rather than through
//     a reinterpreted []uint32: a byte store needs no alignment reasoning and no unsafe, and this file's build tag
//     guarantees the little-endian lane order that makes it equal to four binary.LittleEndian.PutUint32 calls (probed).
//
// The lane shifts go through the shift16/shift32 helpers with counts hoisted out of the loop, which is a codegen
// accommodation only — see convert_simd_arm64.go. The word→byte gather is per-arch for the reason maskfilter's
// narrowing is (every pack-and-narrow method archsimd offers for it is AVX-512 on amd64); see narrowWordAlphas.
//
// Every kernel here is locked against its portable twin by TestConvertSIMDMatchesScalar.

// init swaps the dispatch variables to the simd kernels the per-arch preference constants elect. Unqualified CPUs keep
// the portable forms.
func init() {
	if !simdConvertSupported() {
		return
	}
	if preferSIMDSwizzleWordRow {
		swizzleWordRowFn = swizzleWordRowSIMD
	}
	if preferSIMDPremulWordRow {
		premulWordRowFn = premulWordRowSIMD
	}
	if preferSIMDUnpremulWordRow {
		unpremulWordRowFn = unpremulWordRowSIMD
	}
	if preferSIMDAlphaFromWordsRow {
		alphaFromWordsRowFn = alphaFromWordsRowSIMD
	}
	if preferSIMDFillBytesRow {
		fillBytesRowFn = fillBytesRowSIMD
	}
	if preferSIMDGrayToWordsRow {
		grayToWordsRowFn = grayToWordsRowSIMD
	}
}

// swizzleWordRowSIMD is the no-alpha-step 8888↔8888 lane. Without the swap it is a pure move, so the words go straight
// from a vector load to a byte store; with it, R and B are the two bytes of the 0x00FF00FF half, which a 16-bit rotate
// of the word exchanges (the halves cannot collide, each being a single byte), while G and A ride through in the
// complementary half.
func swizzleWordRowSIMD(dst []byte, src []uint32, swapRB bool) {
	n := len(src)
	i := 0
	if !swapRB {
		for ; i+4 <= n; i += 4 {
			archsimd.LoadUint32x4(src[i:]).ReshapeToUint8s().Store(dst[4*i:])
		}
	} else {
		keepGA := archsimd.BroadcastUint32x4(0xFF00FF00)
		rbMask := archsimd.BroadcastUint32x4(0x00FF00FF)
		l16, r16 := leftShift32(16), rightShift32(16)
		for ; i+4 <= n; i += 4 {
			s := archsimd.LoadUint32x4(src[i:])
			rb := s.And(rbMask)
			s.And(keepGA).Or(shift32(rb, l16)).Or(shift32(rb, r16)).ReshapeToUint8s().Store(dst[4*i:])
		}
	}
	if i < n {
		swizzleWordRowGeneric(dst[4*i:], src[i:], swapRB)
	}
}

// premulWordRowSIMD is the 8888↔8888 premultiply lane. Both 16-bit halves of the split multiply by the same vector
// shape: the pixel's alpha replicated into both lanes for the (R, B) half, and alpha in the even lane with 255 in the
// odd one for the (G, A) half — 255 because div255Round(255*a) == a for every byte a, so the alpha channel rides
// through the identical arithmetic instead of needing a separate lane-blend to protect it.
//
// The bound that keeps every byte inside its own 16-bit lane is the one raster's row kernels use: with
// prod = x*a + 128 <= 255*255 + 128 = 0xFE81, prod>>8 <= 0xFE and prod + (prod>>8) <= 0xFF7F, so nothing carries into
// a neighbor and (prod + (prod>>8)) >> 8 is exactly div255Round(x*a). Recombining with lo | hi<<8 is exact because both
// halves are bytes.
//
// The R/B exchange is a 16-bit rotate of the low half's word, under a branch on the loop-invariant swapRB: one
// perfectly-predicted test per four pixels costs less than the two extra shifts and the OR a branchless select would
// add to every iteration of both lanes.
func premulWordRowSIMD(dst []byte, src []uint32, swapRB bool) {
	n := len(src)
	byteMask := archsimd.BroadcastUint16x8(0x00FF)
	half := archsimd.BroadcastUint16x8(128)
	spread := archsimd.BroadcastUint32x4(0x00010001)
	alphaIdentity := archsimd.BroadcastUint32x4(0x00FF0000)
	r8, l8 := rightShift16(8), leftShift16(8)
	r24 := rightShift32(24)
	l16, r16 := leftShift32(16), rightShift32(16)
	i := 0
	for ; i+4 <= n; i += 4 {
		s32 := archsimd.LoadUint32x4(src[i:])
		s16 := s32.ReshapeToUint16s()
		// The alpha byte replicated into both 16-bit lanes of its pixel. The multiply by 0x00010001 cannot overflow a
		// 32-bit lane (a <= 255) and is the cheapest way to reach both halves at once.
		av := shift32(s32, r24).Mul(spread)
		mRB := av.ReshapeToUint16s()
		mGA := av.Or(alphaIdentity).ReshapeToUint16s()
		lo := s16.And(byteMask).Mul(mRB).Add(half)
		lo = shift16(lo.Add(shift16(lo, r8)), r8)
		hi := shift16(s16, r8).Mul(mGA).Add(half)
		hi = shift16(hi.Add(shift16(hi, r8)), r8)
		lo32 := lo.ReshapeToUint32s()
		if swapRB {
			lo32 = shift32(lo32, l16).Or(shift32(lo32, r16))
		}
		lo32.Or(shift16(hi, l8).ReshapeToUint32s()).ReshapeToUint8s().Store(dst[4*i:])
	}
	if i < n {
		premulWordRowGeneric(dst[4*i:], src[i:], swapRB)
	}
}

// unpremulChan4 is the lanewise unpremulChannelRP body for one already-extracted color channel: c * (1/255), times the
// caller's reciprocal, times 255, rounded to nearest-even and clamped to 255. The a == 0 case is already handled by the
// caller's inv (zeroed there), so nothing here can see a NaN or an out-of-range value: c <= 255 and inv <= 1/(1/255)
// bound the product below 65026. The clamp is the integer Min, not a float one — the scalar clamps after its cast too,
// and an unsigned integer minimum has no NaN question to answer.
func unpremulChan4(c archsimd.Uint32x4, inv, inv255, f255 archsimd.Float32x4, c255 archsimd.Uint32x4) archsimd.Uint32x4 {
	cf := c.BitsToInt32().ConvertToFloat32().Mul(inv255)
	return cf.Mul(inv).Mul(f255).Round().ConvertToInt32().ToBits().Min(c255)
}

// unpremulWordRowSIMD is the 8888↔8888 unpremultiply lane. Unlike the premultiply lane this one cannot run in 16-bit
// lanes — the arithmetic is float — so each color channel is extracted into its own 32-bit lane, converted through the
// *signed* integer convert (the unsigned one is AVX-512 on amd64; every value here is a byte, where the two agree), and
// reassembled by shifting the three results back into place. The reciprocal is computed once per quad and shared by
// the three channels, exactly as the scalar shares invA.
//
// The R/B exchange is free here: R and B live in separate vectors, so it is only a question of which left-shift count
// each one gets, and both counts are hoisted out of the loop.
func unpremulWordRowSIMD(dst []byte, src []uint32, swapRB bool) {
	const inv255f = float32(1) / 255
	n := len(src)
	byteMask := archsimd.BroadcastUint32x4(0xFF)
	alphaMask := archsimd.BroadcastUint32x4(0xFF000000)
	c255 := archsimd.BroadcastUint32x4(255)
	zerou := archsimd.BroadcastUint32x4(0)
	inv255 := archsimd.BroadcastFloat32x4(inv255f)
	f255 := archsimd.BroadcastFloat32x4(255)
	onef := archsimd.BroadcastFloat32x4(1)
	zerof := archsimd.BroadcastFloat32x4(0)
	r8, r16, r24 := rightShift32(8), rightShift32(16), rightShift32(24)
	l8 := leftShift32(8)
	lR, lB := leftShift32(0), leftShift32(16)
	if swapRB {
		lR, lB = leftShift32(16), leftShift32(0)
	}
	i := 0
	for ; i+4 <= n; i += 4 {
		s := archsimd.LoadUint32x4(src[i:])
		a := shift32(s, r24)
		// 1/(a/255), with the a == 0 lanes forced to 0 before anything downstream can see the +Inf (or the 0*Inf NaN
		// it makes of a transparent black pixel). A zeroed reciprocal produces 0, which is what the scalar's "a == 0
		// -> 0" guard returns.
		inv := onef.Div(a.BitsToInt32().ConvertToFloat32().Mul(inv255)).IfElse(a.NotEqual(zerou), zerof)
		r := unpremulChan4(s.And(byteMask), inv, inv255, f255, c255)
		g := unpremulChan4(shift32(s, r8).And(byteMask), inv, inv255, f255, c255)
		b := unpremulChan4(shift32(s, r16).And(byteMask), inv, inv255, f255, c255)
		shift32(r, lR).
			Or(shift32(g, l8)).
			Or(shift32(b, lB)).
			Or(s.And(alphaMask)).
			ReshapeToUint8s().Store(dst[4*i:])
	}
	if i < n {
		unpremulWordRowGeneric(dst[4*i:], src[i:], swapRB)
	}
}

// alphaFromWordsRowSIMD is convert_to_alpha8's 8888 lane: the top byte of sixteen source words gathered into sixteen
// destination bytes. The gather is per-arch (see narrowWordAlphas); everything else is four vector loads and one store.
func alphaFromWordsRowSIMD(dst []byte, src []uint32) {
	n := len(src)
	narrow := newWordAlphaNarrow()
	i := 0
	for ; i+16 <= n; i += 16 {
		narrowWordAlphas(
			archsimd.LoadUint32x4(src[i:]),
			archsimd.LoadUint32x4(src[i+4:]),
			archsimd.LoadUint32x4(src[i+8:]),
			archsimd.LoadUint32x4(src[i+12:]),
			narrow,
		).Store(dst[i:])
	}
	if i < n {
		alphaFromWordsRowGeneric(dst[i:], src[i:])
	}
}

// fillBytesRowSIMD fills a byte row with one value. The wide body issues four independent 128-bit stores per iteration,
// which is worth a few percent on a row this size — the loop is nothing but stores, so what is left to save is the
// per-iteration bookkeeping. Rows too short for it fall through to the single-store loop and then to the portable tail.
func fillBytesRowSIMD(dst []byte, v byte) {
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
		fillBytesRowGeneric(dst[i:], v)
	}
}

// grayExpandQuad turns the low four bytes of b into four opaque 8888 words: the byte widened into a 32-bit lane and
// multiplied by 0x00010101, which for a value that fits in a byte is exactly v | v<<8 | v<<16 with no carry between the
// three copies, then OR-ed with the opaque alpha byte. The widen is the two-step byte→u16→u32 zero-extend both arches
// offer (arm64 has no byte→word extend of its own).
func grayExpandQuad(b archsimd.Uint8x16, replicate, opaque archsimd.Uint32x4) archsimd.Uint8x16 {
	return b.ExtendLo8ToUint16().ExtendLo4ToUint32().Mul(replicate).Or(opaque).ReshapeToUint8s()
}

// grayToWordsRowSIMD is the lowp pipeline's Gray8 → 8888 lane, sixteen pixels per iteration: one byte load, then four
// quads, each reaching its four source bytes by shifting the loaded vector down (zero.ConcatShiftBytesRight(b, k) is
// "b shifted down k bytes"; the receiver supplies the concatenation's high half, so the zeros are the receiver — lane
// order probed before use).
func grayToWordsRowSIMD(dst, src []byte) {
	n := len(src)
	var zero archsimd.Uint8x16
	replicate := archsimd.BroadcastUint32x4(0x00010101)
	opaque := archsimd.BroadcastUint32x4(0xFF000000)
	i := 0
	for ; i+16 <= n; i += 16 {
		b := archsimd.LoadUint8x16(src[i:])
		grayExpandQuad(b, replicate, opaque).Store(dst[4*i:])
		grayExpandQuad(zero.ConcatShiftBytesRight(b, 4), replicate, opaque).Store(dst[4*i+16:])
		grayExpandQuad(zero.ConcatShiftBytesRight(b, 8), replicate, opaque).Store(dst[4*i+32:])
		grayExpandQuad(zero.ConcatShiftBytesRight(b, 12), replicate, opaque).Store(dst[4*i+48:])
	}
	if i < n {
		grayToWordsRowGeneric(dst[4*i:], src[i:])
	}
}
