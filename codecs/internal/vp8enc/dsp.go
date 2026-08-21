// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Encoding DSP: forward/inverse transforms, intra predictors, SSE metrics and quantization (the inverse WHT is shared
// with the decoder side).

package vp8enc

// Kernel dispatch. The four hot DSP entry points below (fTransform, iTransformOne, getSSE and quantizeBlock) each have
// a portable "Generic" implementation here and, in the goexperiment.simd build on arm64 and amd64, an archsimd kernel
// in dsp_simd.go. The entry point itself is a thin wrapper defined per build mode — dsp_portable.go for builds without
// the kernels, dsp_simd.go for builds with them — which branches on a package-level bool that init sets from the CPU
// check and the per-arch preference constants.
//
// This is deliberately NOT the function-variable dispatch the raster, maskfilter, imagecore and shaders packages use,
// and the reason is escape analysis rather than taste. Those packages hand their kernels slices of already-heap-backed
// pixel buffers, so routing the call through a package-level func variable — which escape analysis must treat as
// leaking every pointer argument — costs nothing. This package's hot kernels instead take pointers into the caller's
// *stack*: reconstructIntra16's tmp [16][16]int16 and dcTmp [16]int16, reconstructIntra4's tmp [16]int16 (160 calls
// per macroblock) and reconstructUV's tmp [8][16]int16. Switching quantizeBlock and fTransform alone to func variables
// moves all four to the heap ("go build -gcflags=-m": moved to heap: tmp/dcTmp at frame.go:140,141,176,188) and costs
// ~3% of whole-encode time in fresh allocation and GC before a single kernel has run. Branching on a bool keeps the
// call graph static, so escape analysis still sees that neither implementation leaks its arguments and every one of
// those blocks stays on the stack.
//
// The bools are variables rather than constants because the equivalence tests flip them to run a whole encode down
// each lane and compare the emitted bitstreams byte for byte (TestDSPSIMDEncodeMatchesScalar).

// bps is the stride of all the macroblock work caches.
const bps = 32

// YUV work-cache layout (yuvIn/yuvOut/yuvOut2, each yuvSize bytes with stride bps):
//
//	+----+----+
//	|YYYY|UUVV|
//	|YYYY|UUVV|
//	|YYYY|....|
//	|YYYY|....|
//	+----+----+
const (
	yuvSize = bps * 16
	yOff    = 0
	uOff    = 16
	vOff    = 16 + 8
)

// Prediction-cache layout (yuvP, predSize bytes with stride bps). Intra16 predictions are 16x16 blocks two per row;
// chroma predictions are 16x8 (U and V side by side); intra4 predictions are 4x4 blocks.
const (
	predSize = 32*bps + 16*bps + 8*bps

	i16DC16 = 0 * 16 * bps
	i16TM16 = i16DC16 + 16
	i16VE16 = 1 * 16 * bps
	i16HE16 = i16VE16 + 16

	c8DC8 = 2 * 16 * bps
	c8TM8 = c8DC8 + 1*16
	c8VE8 = 2*16*bps + 8*bps
	c8HE8 = c8VE8 + 1*16

	i4DC4 = 3*16*bps + 0
	i4TM4 = i4DC4 + 4
	i4VE4 = i4DC4 + 8
	i4HE4 = i4DC4 + 12
	i4RD4 = i4DC4 + 16
	i4VR4 = i4DC4 + 20
	i4LD4 = i4DC4 + 24
	i4VL4 = i4DC4 + 28
	i4HD4 = 3*16*bps + 4*bps
	i4HU4 = i4HD4 + 4
	i4TMP = i4HD4 + 8
)

// predI16ModeOffsets, predUVModeOffsets and predI4ModeOffsets locate each mode's prediction in yuvP, ordered by the
// mode enums above.
var (
	predI16ModeOffsets = [numPredModes]int{i16DC16, i16TM16, i16VE16, i16HE16}
	predUVModeOffsets  = [numPredModes]int{c8DC8, c8TM8, c8VE8, c8HE8}
	predI4ModeOffsets  = [numBModes]int{i4DC4, i4TM4, i4VE4, i4HE4, i4RD4, i4VR4, i4LD4, i4VL4, i4HD4, i4HU4}
)

// scan locates the 16 luma 4x4 sub-blocks inside the Y area of the work cache; scanUV the 4+4 chroma sub-blocks (U then
// V).
var scan = [16]int{
	0 + 0*bps, 4 + 0*bps, 8 + 0*bps, 12 + 0*bps,
	0 + 4*bps, 4 + 4*bps, 8 + 4*bps, 12 + 4*bps,
	0 + 8*bps, 4 + 8*bps, 8 + 8*bps, 12 + 8*bps,
	0 + 12*bps, 4 + 12*bps, 8 + 12*bps, 12 + 12*bps,
}

var scanUV = [8]int{
	0 + 0*bps, 4 + 0*bps, 0 + 4*bps, 4 + 4*bps, // U
	8 + 0*bps, 12 + 0*bps, 8 + 4*bps, 12 + 4*bps, // V
}

func clip8b(v int) uint8 {
	if v&^0xff == 0 {
		return uint8(v)
	}
	if v < 0 {
		return 0
	}
	return 255
}

//------------------------------------------------------------------------------
// Transforms (paragraph 14.4)

const (
	ac3Mul1Const = 20091
	ac3Mul2Const = 35468
)

func mul1(a int) int { return (a*ac3Mul1Const)>>16 + a }
func mul2(a int) int { return (a * ac3Mul2Const) >> 16 }

// iTransformOneGeneric reconstructs a 4x4 block: dst = clip(ref + idct(in)).
func iTransformOneGeneric(ref []uint8, in []int16, dst []uint8) {
	var c [4 * 4]int
	for i := 0; i < 4; i++ { // vertical pass
		a := int(in[i]) + int(in[i+8])
		b := int(in[i]) - int(in[i+8])
		cc := mul2(int(in[i+4])) - mul1(int(in[i+12]))
		d := mul1(int(in[i+4])) + mul2(int(in[i+12]))
		c[4*i+0] = a + d
		c[4*i+1] = b + cc
		c[4*i+2] = b - cc
		c[4*i+3] = a - d
	}
	for i := 0; i < 4; i++ { // horizontal pass
		dc := c[i] + 4
		a := dc + c[i+8]
		b := dc - c[i+8]
		cc := mul2(c[i+4]) - mul1(c[i+12])
		d := mul1(c[i+4]) + mul2(c[i+12])
		dst[0+i*bps] = clip8b(int(ref[0+i*bps]) + (a+d)>>3)
		dst[1+i*bps] = clip8b(int(ref[1+i*bps]) + (b+cc)>>3)
		dst[2+i*bps] = clip8b(int(ref[2+i*bps]) + (b-cc)>>3)
		dst[3+i*bps] = clip8b(int(ref[3+i*bps]) + (a-d)>>3)
	}
}

// fTransformGeneric computes the forward DCT of the 4x4 difference block src-ref into out.
func fTransformGeneric(src, ref []uint8, out []int16) {
	var tmp [16]int
	for i := 0; i < 4; i++ {
		d0 := int(src[i*bps+0]) - int(ref[i*bps+0]) // 9-bit dynamic range [-255,255]
		d1 := int(src[i*bps+1]) - int(ref[i*bps+1])
		d2 := int(src[i*bps+2]) - int(ref[i*bps+2])
		d3 := int(src[i*bps+3]) - int(ref[i*bps+3])
		a0 := d0 + d3 // 10b [-510,510]
		a1 := d1 + d2
		a2 := d1 - d2
		a3 := d0 - d3
		tmp[0+i*4] = (a0 + a1) * 8 // 14b [-8160,8160]
		tmp[1+i*4] = (a2*2217 + a3*5352 + 1812) >> 9
		tmp[2+i*4] = (a0 - a1) * 8
		tmp[3+i*4] = (a3*2217 - a2*5352 + 937) >> 9
	}
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[12+i] // 15b
		a1 := tmp[4+i] + tmp[8+i]
		a2 := tmp[4+i] - tmp[8+i]
		a3 := tmp[0+i] - tmp[12+i]
		out[0+i] = int16((a0 + a1 + 7) >> 4) // 12b
		v := (a2*2217 + a3*5352 + 12000) >> 16
		if a3 != 0 {
			v++
		}
		out[4+i] = int16(v)
		out[8+i] = int16((a0 - a1 + 7) >> 4)
		out[12+i] = int16((a3*2217 - a2*5352 + 51000) >> 16)
	}
}

// fTransformWHT computes the forward Walsh-Hadamard transform of the 16 DC coefficients. in is the [16][16]int16 luma
// coefficient array flattened; it reads entry 0 of each block.
//
// Neither WHT is a dispatch point. The butterflies are 32 adds, but the operands are one int16 out of every 32-byte
// block, so a vector form would spend more on the strided gather (and, for the inverse, the scatter) than the whole
// transform costs; and where fTransform runs 64 times per macroblock, these run four.
func fTransformWHT(in *[16][16]int16, out *[16]int16) {
	var tmp [16]int
	for i := 0; i < 4; i++ {
		a0 := int(in[4*i+0][0]) + int(in[4*i+2][0]) // 13b
		a1 := int(in[4*i+1][0]) + int(in[4*i+3][0])
		a2 := int(in[4*i+1][0]) - int(in[4*i+3][0])
		a3 := int(in[4*i+0][0]) - int(in[4*i+2][0])
		tmp[0+i*4] = a0 + a1 // 14b
		tmp[1+i*4] = a3 + a2
		tmp[2+i*4] = a3 - a2
		tmp[3+i*4] = a0 - a1
	}
	for i := 0; i < 4; i++ {
		a0 := tmp[0+i] + tmp[8+i] // 15b
		a1 := tmp[4+i] + tmp[12+i]
		a2 := tmp[4+i] - tmp[12+i]
		a3 := tmp[0+i] - tmp[8+i]
		b0 := a0 + a1 // 16b
		b1 := a3 + a2
		b2 := a3 - a2
		b3 := a0 - a1
		out[0+i] = int16(b0 >> 1) // 15b
		out[4+i] = int16(b1 >> 1)
		out[8+i] = int16(b2 >> 1)
		out[12+i] = int16(b3 >> 1)
	}
}

// iTransformWHT is the inverse Walsh-Hadamard transform: it distributes the dequantized second order coefficients back
// into slot 0 of each luma coefficient block.
func iTransformWHT(in *[16]int16, out *[16][16]int16) {
	var tmp [16]int
	for i := 0; i < 4; i++ {
		a0 := int(in[0+i]) + int(in[12+i])
		a1 := int(in[4+i]) + int(in[8+i])
		a2 := int(in[4+i]) - int(in[8+i])
		a3 := int(in[0+i]) - int(in[12+i])
		tmp[0+i] = a0 + a1
		tmp[8+i] = a0 - a1
		tmp[4+i] = a3 + a2
		tmp[12+i] = a3 - a2
	}
	for i := 0; i < 4; i++ {
		dc := tmp[0+i*4] + 3 // w/ rounder
		a0 := dc + tmp[3+i*4]
		a1 := tmp[1+i*4] + tmp[2+i*4]
		a2 := tmp[1+i*4] - tmp[2+i*4]
		a3 := dc - tmp[3+i*4]
		out[4*i+0][0] = int16((a0 + a1) >> 3)
		out[4*i+1][0] = int16((a3 + a2) >> 3)
		out[4*i+2][0] = int16((a0 - a1) >> 3)
		out[4*i+3][0] = int16((a3 - a2) >> 3)
	}
}

//------------------------------------------------------------------------------
// Intra predictions. left and top may be nil when the corresponding neighbors are outside the frame. left[-1] (the
// top-left corner sample) is passed separately as topLeft.
//
// None of the predictors is a dispatch point either. Measured on an M4 Max with the DSP kernels in play, the whole
// family costs about 1.2 us per macroblock — encPredLuma16 355 ns, encPredChroma8 200 ns, encPredLuma4 38 ns times
// sixteen sub-blocks — which is ~2% of a 512x384 photo encode, and only the 16x16 and 8x8 halves are vector-shaped:
// the 4x4 predictors write four-byte rows into a cache where the next mode's block starts four bytes later (see i4TM4
// and friends above), so a wider store would clobber a neighbor, and their ten diagonal patterns would each need their
// own shuffle.

func fill(dst []uint8, value uint8, size int) {
	for j := 0; j < size; j++ {
		row := dst[j*bps : j*bps+size]
		for i := range row {
			row[i] = value
		}
	}
}

func verticalPred(dst, top []uint8, size int) {
	if top != nil {
		for j := 0; j < size; j++ {
			copy(dst[j*bps:j*bps+size], top[:size])
		}
	} else {
		fill(dst, 127, size)
	}
}

func horizontalPred(dst, left []uint8, size int) {
	if left != nil {
		for j := 0; j < size; j++ {
			row := dst[j*bps : j*bps+size]
			for i := range row {
				row[i] = left[j]
			}
		}
	} else {
		fill(dst, 129, size)
	}
}

func trueMotion(dst, left []uint8, topLeft uint8, top []uint8, size int) {
	switch {
	case left != nil && top != nil:
		for y := 0; y < size; y++ {
			base := int(left[y]) - int(topLeft)
			row := dst[y*bps : y*bps+size]
			for x := 0; x < size; x++ {
				row[x] = clip8b(base + int(top[x]))
			}
		}
	case left != nil:
		horizontalPred(dst, left, size)
	case top != nil:
		// True motion without left samples (hence: with default 129 value) is equivalent to vertical prediction, just
		// copying the top samples.
		verticalPred(dst, top, size)
	default:
		fill(dst, 129, size)
	}
}

func dcMode(dst, left, top []uint8, size, round, shift int) {
	dc := 0
	switch {
	case top != nil:
		for j := 0; j < size; j++ {
			dc += int(top[j])
		}
		if left != nil { // top and left present
			for j := 0; j < size; j++ {
				dc += int(left[j])
			}
		} else { // top, but no left
			dc += dc
		}
		dc = (dc + round) >> uint(shift)
	case left != nil: // left but no top
		for j := 0; j < size; j++ {
			dc += int(left[j])
		}
		dc += dc
		dc = (dc + round) >> uint(shift)
	default: // no top, no left: nothing
		dc = 0x80
	}
	fill(dst, uint8(dc), size)
}

// encPredLuma16 computes the four 16x16 luma predictions into the prediction cache.
func encPredLuma16(yuvP, left []uint8, topLeft uint8, top []uint8) {
	dcMode(yuvP[i16DC16:], left, top, 16, 16, 5)
	verticalPred(yuvP[i16VE16:], top, 16)
	horizontalPred(yuvP[i16HE16:], left, 16)
	trueMotion(yuvP[i16TM16:], left, topLeft, top, 16)
}

// encPredChroma8 computes the four 8x8 chroma predictions (U and V side by side) into the prediction cache. left holds
// 8 U samples then 8 V samples; top likewise; topLeftU/V are the corner samples.
func encPredChroma8(yuvP, left []uint8, topLeftU, topLeftV uint8, top []uint8) {
	var leftU, leftV, topU, topV []uint8
	if left != nil {
		leftU, leftV = left[:8], left[8:16]
	}
	if top != nil {
		topU, topV = top[:8], top[8:16]
	}
	// U block
	dcMode(yuvP[c8DC8:], leftU, topU, 8, 8, 4)
	verticalPred(yuvP[c8VE8:], topU, 8)
	horizontalPred(yuvP[c8HE8:], leftU, 8)
	trueMotion(yuvP[c8TM8:], leftU, topLeftU, topU, 8)
	// V block
	dcMode(yuvP[c8DC8+8:], leftV, topV, 8, 8, 4)
	verticalPred(yuvP[c8VE8+8:], topV, 8)
	horizontalPred(yuvP[c8HE8+8:], leftV, 8)
	trueMotion(yuvP[c8TM8+8:], leftV, topLeftV, topV, 8)
}

//------------------------------------------------------------------------------
// Luma 4x4 predictions. The boundary samples are packed into one array: left samples are top[-5..-2] (bottom to top),
// top-left is top[-1], top samples are top[0..3] and top-right is top[4..7]. Here top is a slice with those 13 samples
// at indexes 0..12, so "top[i]" below is top[5+i].

func avg3(a, b, c int) uint8 { return uint8((a + 2*b + c + 2) >> 2) }
func avg2(a, b int) uint8    { return uint8((a + b + 1) >> 1) }

func encPredLuma4(yuvP, top []uint8) {
	t := top[5:] // t[i] == C top[i]; t[-1] via top[4], etc.
	x := int(top[4])
	i := int(top[3])
	j := int(top[2])
	k := int(top[1])
	l := int(top[0])
	a := int(t[0])
	b := int(t[1])
	c := int(t[2])
	d := int(t[3])
	e := int(t[4])
	f := int(t[5])
	g := int(t[6])
	h := int(t[7])

	// DC4
	{
		dc := 4
		for n := 0; n < 4; n++ {
			dc += int(t[n]) + int(top[n]) // top and left samples
		}
		fill(yuvP[i4DC4:], uint8(dc>>3), 4)
	}
	// TM4
	{
		dst := yuvP[i4TM4:]
		for y := 0; y < 4; y++ {
			base := int(top[3-y]) - x
			for xx := 0; xx < 4; xx++ {
				dst[y*bps+xx] = clip8b(base + int(t[xx]))
			}
		}
	}
	// VE4
	{
		vals := [4]uint8{avg3(x, a, b), avg3(a, b, c), avg3(b, c, d), avg3(c, d, e)}
		for n := 0; n < 4; n++ {
			copy(yuvP[i4VE4+n*bps:i4VE4+n*bps+4], vals[:])
		}
	}
	// HE4
	{
		dst := yuvP[i4HE4:]
		rows := [4]uint8{avg3(x, i, j), avg3(i, j, k), avg3(j, k, l), avg3(k, l, l)}
		for n := 0; n < 4; n++ {
			for m := 0; m < 4; m++ {
				dst[n*bps+m] = rows[n]
			}
		}
	}
	// RD4
	{
		dst := yuvP[i4RD4:]
		set := func(xx, yy int, v uint8) { dst[xx+yy*bps] = v }
		set(0, 3, avg3(j, k, l))
		v := avg3(i, j, k)
		set(0, 2, v)
		set(1, 3, v)
		v = avg3(x, i, j)
		set(0, 1, v)
		set(1, 2, v)
		set(2, 3, v)
		v = avg3(a, x, i)
		set(0, 0, v)
		set(1, 1, v)
		set(2, 2, v)
		set(3, 3, v)
		v = avg3(b, a, x)
		set(1, 0, v)
		set(2, 1, v)
		set(3, 2, v)
		v = avg3(c, b, a)
		set(2, 0, v)
		set(3, 1, v)
		set(3, 0, avg3(d, c, b))
	}
	// VR4
	{
		dst := yuvP[i4VR4:]
		set := func(xx, yy int, v uint8) { dst[xx+yy*bps] = v }
		v := avg2(x, a)
		set(0, 0, v)
		set(1, 2, v)
		v = avg2(a, b)
		set(1, 0, v)
		set(2, 2, v)
		v = avg2(b, c)
		set(2, 0, v)
		set(3, 2, v)
		set(3, 0, avg2(c, d))
		set(0, 3, avg3(k, j, i))
		set(0, 2, avg3(j, i, x))
		v = avg3(i, x, a)
		set(0, 1, v)
		set(1, 3, v)
		v = avg3(x, a, b)
		set(1, 1, v)
		set(2, 3, v)
		v = avg3(a, b, c)
		set(2, 1, v)
		set(3, 3, v)
		set(3, 1, avg3(b, c, d))
	}
	// LD4
	{
		dst := yuvP[i4LD4:]
		set := func(xx, yy int, v uint8) { dst[xx+yy*bps] = v }
		set(0, 0, avg3(a, b, c))
		v := avg3(b, c, d)
		set(1, 0, v)
		set(0, 1, v)
		v = avg3(c, d, e)
		set(2, 0, v)
		set(1, 1, v)
		set(0, 2, v)
		v = avg3(d, e, f)
		set(3, 0, v)
		set(2, 1, v)
		set(1, 2, v)
		set(0, 3, v)
		v = avg3(e, f, g)
		set(3, 1, v)
		set(2, 2, v)
		set(1, 3, v)
		v = avg3(f, g, h)
		set(3, 2, v)
		set(2, 3, v)
		set(3, 3, avg3(g, h, h))
	}
	// VL4
	{
		dst := yuvP[i4VL4:]
		set := func(xx, yy int, v uint8) { dst[xx+yy*bps] = v }
		set(0, 0, avg2(a, b))
		v := avg2(b, c)
		set(1, 0, v)
		set(0, 2, v)
		v = avg2(c, d)
		set(2, 0, v)
		set(1, 2, v)
		v = avg2(d, e)
		set(3, 0, v)
		set(2, 2, v)
		set(0, 1, avg3(a, b, c))
		v = avg3(b, c, d)
		set(1, 1, v)
		set(0, 3, v)
		v = avg3(c, d, e)
		set(2, 1, v)
		set(1, 3, v)
		v = avg3(d, e, f)
		set(3, 1, v)
		set(2, 3, v)
		set(3, 2, avg3(e, f, g))
		set(3, 3, avg3(f, g, h))
	}
	// HD4
	{
		dst := yuvP[i4HD4:]
		set := func(xx, yy int, v uint8) { dst[xx+yy*bps] = v }
		v := avg2(i, x)
		set(0, 0, v)
		set(2, 1, v)
		v = avg2(j, i)
		set(0, 1, v)
		set(2, 2, v)
		v = avg2(k, j)
		set(0, 2, v)
		set(2, 3, v)
		set(0, 3, avg2(l, k))
		set(3, 0, avg3(a, b, c))
		set(2, 0, avg3(x, a, b))
		v = avg3(i, x, a)
		set(1, 0, v)
		set(3, 1, v)
		v = avg3(j, i, x)
		set(1, 1, v)
		set(3, 2, v)
		v = avg3(k, j, i)
		set(1, 2, v)
		set(3, 3, v)
		set(1, 3, avg3(l, k, j))
	}
	// HU4
	{
		dst := yuvP[i4HU4:]
		set := func(xx, yy int, v uint8) { dst[xx+yy*bps] = v }
		set(0, 0, avg2(i, j))
		v := avg2(j, k)
		set(2, 0, v)
		set(0, 1, v)
		v = avg2(k, l)
		set(2, 1, v)
		set(0, 2, v)
		set(1, 0, avg3(i, j, k))
		v = avg3(j, k, l)
		set(3, 0, v)
		set(1, 1, v)
		v = avg3(k, l, l)
		set(3, 1, v)
		set(1, 2, v)
		lv := uint8(l)
		set(3, 2, lv)
		set(2, 2, lv)
		set(0, 3, lv)
		set(1, 3, lv)
		set(2, 3, lv)
		set(3, 3, lv)
	}
}

//------------------------------------------------------------------------------
// Metrics

func getSSEGeneric(a, b []uint8, w, h int) int64 {
	var count int64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			diff := int64(a[y*bps+x]) - int64(b[y*bps+x])
			count += diff * diff
		}
	}
	return count
}

func sse16x16(a, b []uint8) int64 { return getSSE(a, b, 16, 16) }
func sse16x8(a, b []uint8) int64  { return getSSE(a, b, 16, 8) }
func sse4x4(a, b []uint8) int64   { return getSSE(a, b, 4, 4) }

// isFlatSource16 reports whether the 16x16 source block is a single uniform color. This one is deliberately not a
// dispatch point: its scalar loop exits at the first differing sample, which on real content is the second sample of
// the block, so a vector form that must always read all sixteen rows would lose. It is also called once per
// macroblock, against the ~1500 calls per macroblock the quantizer sees.
func isFlatSource16(src []uint8) bool {
	v := src[0]
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			if src[j*bps+i] != v {
				return false
			}
		}
	}
	return true
}

// isFlat reports whether the given quantized level blocks have no more than thresh non-zero AC coefficients in total.
// Like isFlatSource16 this is deliberately not a dispatch point, and for the same reason: the thresholds it is asked
// about are 0 to 3, so the scalar loop stops within a handful of coefficients, while a vector form has to count every
// block before it can compare. Measured on an M4 Max, an archsimd version of this was -20% for the single-block case
// but +21% for eight blocks and +256% for sixteen, and it would also have cost this function its inlining in the
// portable build.
func isFlat(blocks [][16]int16, thresh int) bool {
	score := 0
	for b := range blocks {
		for i := 1; i < 16; i++ { // omit DC (first coefficient)
			if blocks[b][i] != 0 {
				score++
				if score > thresh {
					return false
				}
			}
		}
	}
	return true
}

//------------------------------------------------------------------------------
// Quantization

const qFix = 17

func quantBias(b uint32) uint32 { return b << (qFix - 8) }

// quantDiv is the only lossy step of quantization: (n*iQ + B) >> qFix.
func quantDiv(n, iq, b uint32) int {
	return int((uint64(n)*uint64(iq) + uint64(b)) >> qFix)
}

// matrix holds pre-computed quantizer parameters for one coefficient plane.
type matrix struct {
	q       [16]uint16 // quantizer steps
	iq      [16]uint32 // reciprocals, fixed point qFix
	bias    [16]uint32 // rounding bias
	zthresh [16]uint32 // value below which a coefficient is quantized to zero
	sharpen [16]uint16 // frequency boosters for slight sharpening
	// 16-bit views of iq and zthresh, which is the width the simd quantizer's lanes carry them in (both are provably
	// under 65536; see quantizeBlockSIMD and TestQuantMatrixSIMDDomain). They are filled unconditionally — expand runs
	// three times per frame, so the cost is noise, and keeping one matrix layout for both build modes keeps the
	// portable path readable.
	iq16      [16]uint16
	zthresh16 [16]uint16
}

// expand fills the derived fields from q[0]/q[1] and returns the average quantizer. mType is 0 for luma-1 (AC), 1 for
// Y2, 2 for chroma. (libwebp's ExpandMatrix.)
func (m *matrix) expand(mType int) int {
	for i := 0; i < 2; i++ {
		isAC := 0
		if i > 0 {
			isAC = 1
		}
		bias := uint32(biasMatrices[mType][isAC])
		m.iq[i] = (1 << qFix) / uint32(m.q[i])
		m.bias[i] = quantBias(bias)
		// zthresh is the exact value such that quantDiv(coeff, iq, bias) is zero iff coeff <= zthresh.
		m.zthresh[i] = ((1 << qFix) - 1 - m.bias[i]) / m.iq[i]
	}
	for i := 2; i < 16; i++ {
		m.q[i] = m.q[1]
		m.iq[i] = m.iq[1]
		m.bias[i] = m.bias[1]
		m.zthresh[i] = m.zthresh[1]
	}
	sum := 0
	for i := 0; i < 16; i++ {
		if mType == 0 { // sharpening is used for AC luma coefficients only
			m.sharpen[i] = uint16((int(freqSharpening[i]) * int(m.q[i])) >> sharpenBits)
		} else {
			m.sharpen[i] = 0
		}
		m.iq16[i] = uint16(m.iq[i])
		m.zthresh16[i] = uint16(m.zthresh[i])
		sum += int(m.q[i])
	}
	return (sum + 8) >> 4
}

// quantizeBlockGeneric quantizes the 16 coefficients of in, writing the quantized levels in zigzag order to out and the
// dequantized coefficients back into in (natural order). It returns whether any level is non-zero.
func quantizeBlockGeneric(in, out *[16]int16, m *matrix) int {
	last := -1
	for n := 0; n < 16; n++ {
		j := zigzag[n]
		sign := in[j] < 0
		coeff := uint32(int(in[j])) // magnitude plus sharpen
		if sign {
			coeff = uint32(-int(in[j]))
		}
		coeff += uint32(m.sharpen[j])
		if coeff > m.zthresh[j] {
			level := quantDiv(coeff, m.iq[j], m.bias[j])
			if level > maxLevel {
				level = maxLevel
			}
			if sign {
				level = -level
			}
			in[j] = int16(level * int(m.q[j]))
			out[n] = int16(level)
			if level != 0 {
				last = n
			}
		} else {
			out[n] = 0
			in[j] = 0
		}
	}
	if last >= 0 {
		return 1
	}
	return 0
}
