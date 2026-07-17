// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The macroblock iterator: source import with boundary replication, reconstructed-boundary bookkeeping and the intra4x4
// sub-block rotation.

package vp8enc

// iterator walks the macroblocks in raster order, holding the per-macroblock work caches.
type iterator struct {
	mb         *mbInfo
	e          *encoder
	yuvIn      []uint8   // source samples (yuvSize, stride bps)
	yuvOut     []uint8   // best reconstruction
	yuvOut2    []uint8   // scratch reconstruction
	yuvP       []uint8   // prediction cache (predSize)
	topNz      [9]int    // top non-zero context: 0..3 luma, 4..5 U, 6..7 V, 8 DC
	leftNz     [9]int    // left non-zero context; leftNz[8] is managed separately from the nz words
	i4         int       // current intra4 sub-block (0..15)
	y          int       // current macroblock position
	i4TopIdx   int       // index of the "top" sample of the current sub-block inside i4Boundary
	predsIdx   int       // index into e.preds of the current macroblock's (0,0) sub-block mode
	nzIdx      int       // index into e.nz of the current macroblock (1-based; 0 is the left sentinel)
	x          int       // current macroblock position
	i4Boundary [37]uint8 // boundary samples for intra4 prediction, snake layout
	// Left boundary samples of the current macroblock; index 0 is the top-left corner sample (C's left[-1]), samples
	// follow at 1..16 (luma) / 1..8 (chroma).
	yLeft         [17]uint8
	uvLeftScratch [16]uint8 // U then V left samples, packed for encPredChroma8
	vLeft         [9]uint8
	uLeft         [9]uint8
}

// mbInfo holds the coded state of one macroblock.
type mbInfo struct {
	typ      uint8 // 1 = intra16x16, 0 = intra4x4
	uvMode   uint8
	skip     bool // no coefficient levels coded (the coded skip flag when skipProba is in use)
	noCoeffs bool // decoder-visible "no DCT coefficients" (§15.1), gating the inner loop filter
}

// topLeftI4 records the position of the "top" sample of each intra4 sub-block inside i4Boundary.
var topLeftI4 = [16]uint8{
	17, 21, 25, 29,
	13, 17, 21, 25,
	9, 13, 17, 21,
	5, 9, 13, 17,
}

func newIterator(e *encoder) *iterator {
	it := &iterator{
		e:       e,
		yuvIn:   make([]uint8, yuvSize+8), // small slack so 4x4 sub-slices stay in bounds
		yuvOut:  make([]uint8, yuvSize+8),
		yuvOut2: make([]uint8, yuvSize+8),
		yuvP:    make([]uint8, predSize+8),
	}
	it.reset()
	return it
}

func (it *iterator) initLeft() {
	corner := uint8(127)
	if it.y > 0 {
		corner = 129
	}
	it.yLeft[0] = corner
	it.uLeft[0] = corner
	it.vLeft[0] = corner
	for i := 1; i < len(it.yLeft); i++ {
		it.yLeft[i] = 129
	}
	for i := 1; i < len(it.uLeft); i++ {
		it.uLeft[i] = 129
		it.vLeft[i] = 129
	}
	it.leftNz[8] = 0
}

func (it *iterator) initTop() {
	e := it.e
	for i := range e.topY {
		e.topY[i] = 127
	}
	for i := range e.topUV {
		e.topUV[i] = 127
	}
	for i := range e.nz {
		e.nz[i] = 0
	}
}

func (it *iterator) setRow(y int) {
	e := it.e
	it.x = 0
	it.y = y
	it.predsIdx = e.predsBase + y*4*e.predsW
	it.nzIdx = 1
	it.mb = &e.mbInfo[y*e.mbW]
	it.initLeft()
}

func (it *iterator) reset() {
	it.setRow(0)
	it.initTop()
}

// next advances to the next macroblock, returning false once past the last one.
func (it *iterator) next() bool {
	it.x++
	if it.x == it.e.mbW {
		if it.y+1 == it.e.mbH {
			return false
		}
		it.setRow(it.y + 1)
	} else {
		it.predsIdx += 4
		it.nzIdx++
		it.mb = &it.e.mbInfo[it.y*it.e.mbW+it.x]
	}
	return true
}

//------------------------------------------------------------------------------
// Source import with boundary replication

func importBlock(src []uint8, srcStride int, dst []uint8, w, h, size int) {
	for i := 0; i < h; i++ {
		copy(dst[i*bps:i*bps+w], src[i*srcStride:i*srcStride+w])
		for j := w; j < size; j++ {
			dst[i*bps+j] = dst[i*bps+w-1]
		}
	}
	for i := h; i < size; i++ {
		copy(dst[i*bps:i*bps+size], dst[(i-1)*bps:(i-1)*bps+size])
	}
}

// importMB loads the source samples of the current macroblock into yuvIn, replicating boundary pixels when the
// macroblock overlaps the right/bottom edge (the prediction boundaries carry reconstructed samples via saveBoundary).
func (it *iterator) importMB() {
	e := it.e
	x, y := it.x, it.y
	ySrcOff := (y*e.yStride + x) * 16
	uvSrcOff := (y*e.uvStride + x) * 8
	w := minInt(e.width-x*16, 16)
	h := minInt(e.height-y*16, 16)
	uvW := (w + 1) >> 1
	uvH := (h + 1) >> 1
	importBlock(e.yPlane[ySrcOff:], e.yStride, it.yuvIn[yOff:], w, h, 16)
	importBlock(e.uPlane[uvSrcOff:], e.uvStride, it.yuvIn[uOff:], uvW, uvH, 8)
	importBlock(e.vPlane[uvSrcOff:], e.uvStride, it.yuvIn[vOff:], uvW, uvH, 8)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// exportMB copies the reconstructed macroblock into the (macroblock-padded) reconstruction planes.
func (it *iterator) exportMB() {
	e := it.e
	x, y := it.x, it.y
	for j := 0; j < 16; j++ {
		copy(e.reconY[(y*16+j)*e.reconYStride+x*16:][:16], it.yuvOut[yOff+j*bps:][:16])
	}
	for j := 0; j < 8; j++ {
		copy(e.reconU[(y*8+j)*e.reconUVStride+x*8:][:8], it.yuvOut[uOff+j*bps:][:8])
		copy(e.reconV[(y*8+j)*e.reconUVStride+x*8:][:8], it.yuvOut[vOff+j*bps:][:8])
	}
}

//------------------------------------------------------------------------------
// Non-zero context conversion (bit layout: bits 0-15 are the luma 4x4 blocks, 16-19 U, 20-23 V, 24 the intra16 DC).

func bit(nz uint32, n uint) int {
	if nz&(1<<n) != 0 {
		return 1
	}
	return 0
}

func (it *iterator) nzToBytes() {
	e := it.e
	tnz, lnz := e.nz[it.nzIdx], e.nz[it.nzIdx-1]
	// Top-Y
	it.topNz[0] = bit(tnz, 12)
	it.topNz[1] = bit(tnz, 13)
	it.topNz[2] = bit(tnz, 14)
	it.topNz[3] = bit(tnz, 15)
	// Top-U
	it.topNz[4] = bit(tnz, 18)
	it.topNz[5] = bit(tnz, 19)
	// Top-V
	it.topNz[6] = bit(tnz, 22)
	it.topNz[7] = bit(tnz, 23)
	// DC
	it.topNz[8] = bit(tnz, 24)
	// Left-Y
	it.leftNz[0] = bit(lnz, 3)
	it.leftNz[1] = bit(lnz, 7)
	it.leftNz[2] = bit(lnz, 11)
	it.leftNz[3] = bit(lnz, 15)
	// Left-U
	it.leftNz[4] = bit(lnz, 17)
	it.leftNz[5] = bit(lnz, 19)
	// Left-V
	it.leftNz[6] = bit(lnz, 21)
	it.leftNz[7] = bit(lnz, 23)
	// leftNz[8] (left-DC) is special: it is not read from the nz words but carried directly.
}

func (it *iterator) bytesToNz() {
	nz := uint32(0)
	// top
	nz |= uint32(it.topNz[0])<<12 | uint32(it.topNz[1])<<13
	nz |= uint32(it.topNz[2])<<14 | uint32(it.topNz[3])<<15
	nz |= uint32(it.topNz[4])<<18 | uint32(it.topNz[5])<<19
	nz |= uint32(it.topNz[6])<<22 | uint32(it.topNz[7])<<23
	nz |= uint32(it.topNz[8]) << 24 // the top DC bit is propagated, especially for intra4
	// left
	nz |= uint32(it.leftNz[0])<<3 | uint32(it.leftNz[1])<<7
	nz |= uint32(it.leftNz[2]) << 11
	nz |= uint32(it.leftNz[4])<<17 | uint32(it.leftNz[6])<<21
	it.e.nz[it.nzIdx] = nz
}

//------------------------------------------------------------------------------
// Boundary bookkeeping

// saveBoundary stores the reconstructed right column and bottom row of the current macroblock as the left/top
// boundaries of the neighbors to come.
func (it *iterator) saveBoundary() {
	e := it.e
	x, y := it.x, it.y
	ySrc := it.yuvOut[yOff:]
	uvSrc := it.yuvOut[uOff:]
	if x < e.mbW-1 { // left boundary for the next macroblock
		for i := 0; i < 16; i++ {
			it.yLeft[1+i] = ySrc[15+i*bps]
		}
		for i := 0; i < 8; i++ {
			it.uLeft[1+i] = uvSrc[7+i*bps]
			it.vLeft[1+i] = uvSrc[15+i*bps]
		}
		// top-left corners (before the top row is overwritten)
		it.yLeft[0] = e.topY[x*16+15]
		it.uLeft[0] = e.topUV[x*16+7]
		it.vLeft[0] = e.topUV[x*16+8+7]
	}
	if y < e.mbH-1 { // top boundary for the row below
		copy(e.topY[x*16:x*16+16], ySrc[15*bps:15*bps+16])
		copy(e.topUV[x*16:x*16+16], uvSrc[7*bps:7*bps+16])
	}
}

//------------------------------------------------------------------------------
// Mode setting helpers

func (it *iterator) setIntra16Mode(mode int) {
	e := it.e
	for j := 0; j < 4; j++ {
		row := e.preds[it.predsIdx+j*e.predsW:]
		for i := 0; i < 4; i++ {
			row[i] = uint8(mode)
		}
	}
	it.mb.typ = 1
}

func (it *iterator) setIntra4Modes(modes *[16]uint8) {
	e := it.e
	for j := 0; j < 4; j++ {
		copy(e.preds[it.predsIdx+j*e.predsW:it.predsIdx+j*e.predsW+4], modes[j*4:j*4+4])
	}
	it.mb.typ = 0
}

//------------------------------------------------------------------------------
// Intra4x4 sub-block iteration. Boundary samples live in a 37-sample "snake": entries 0..16 are the left column
// bottom-to-top plus the top-left corner, 17..32 the top row, 33..36 the top-right samples.

func (it *iterator) startI4() {
	e := it.e
	it.i4 = 0
	it.i4TopIdx = int(topLeftI4[0])
	for i := 0; i < 17; i++ { // left
		it.i4Boundary[i] = it.yLeft[16-i] // yLeft[1+15-i]; i==16 is the top-left corner yLeft[0]
	}
	for i := 0; i < 16; i++ { // top
		it.i4Boundary[17+i] = e.topY[it.x*16+i]
	}
	// Top-right samples have a special case at the far right of the picture.
	if it.x < e.mbW-1 {
		for i := 16; i < 16+4; i++ {
			it.i4Boundary[17+i] = e.topY[it.x*16+i]
		}
	} else { // replicate the last valid pixel four times
		for i := 16; i < 16+4; i++ {
			it.i4Boundary[17+i] = it.i4Boundary[17+15]
		}
	}
	it.nzToBytes() // import the non-zero context
}

// rotateI4 feeds the freshly reconstructed sub-block samples back into the boundary snake and advances to the next
// sub-block, returning false after the last one.
func (it *iterator) rotateI4(yuvOut []uint8) bool {
	blk := yuvOut[scan[it.i4]:]
	top := it.i4Boundary[it.i4TopIdx:]
	// Update the cache with 7 fresh samples.
	for i := 0; i <= 3; i++ {
		it.i4Boundary[it.i4TopIdx-4+i] = blk[i+3*bps] // store future top samples
	}
	if it.i4&3 != 3 { // if not on the right sub-blocks #3, #7, #11, #15
		for i := 0; i <= 2; i++ { // store future left samples
			top[i] = blk[3+(2-i)*bps]
		}
	} else { // else replicate the top-right samples, as the spec says
		for i := 0; i <= 3; i++ {
			top[i] = top[i+4]
		}
	}
	it.i4++
	if it.i4 == 16 { // done
		return false
	}
	it.i4TopIdx = int(topLeftI4[it.i4])
	return true
}

//------------------------------------------------------------------------------
// Prediction construction

func (it *iterator) makeLuma16Preds() {
	var left []uint8
	var topLeft uint8 = 129
	if it.x > 0 {
		left = it.yLeft[1:17]
		topLeft = it.yLeft[0]
	}
	var top []uint8
	if it.y > 0 {
		top = it.e.topY[it.x*16 : it.x*16+16]
	}
	encPredLuma16(it.yuvP, left, topLeft, top)
}

func (it *iterator) makeChroma8Preds() {
	var left []uint8
	var topLeftU, topLeftV uint8 = 129, 129
	if it.x > 0 {
		copy(it.uvLeftScratch[:8], it.uLeft[1:9])
		copy(it.uvLeftScratch[8:], it.vLeft[1:9])
		left = it.uvLeftScratch[:]
		topLeftU = it.uLeft[0]
		topLeftV = it.vLeft[0]
	}
	var top []uint8
	if it.y > 0 {
		top = it.e.topUV[it.x*16 : it.x*16+16]
	}
	encPredChroma8(it.yuvP, left, topLeftU, topLeftV, top)
}

func (it *iterator) makeIntra4Preds() {
	encPredLuma4(it.yuvP, it.i4Boundary[it.i4TopIdx-5:it.i4TopIdx+8])
}
