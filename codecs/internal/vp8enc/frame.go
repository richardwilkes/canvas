// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Frame coding: rate-distortion mode decision (the basic RD lane, no trellis), the reconstruction loop, residual
// recording/coding, and the two-pass encode driver (a full statistics pass followed by the final coding pass).

package vp8enc

const maxCost = int64(0x7fffffffffffff)

// encoder is the per-frame VP8 encoder state.
type encoder struct {
	nz     []uint32 // mbW+1 non-zero context words; index 0 is the left sentinel
	reconU []uint8  // reconstruction plane; see reconY/reconV for details
	mbInfo []mbInfo
	preds  []uint8 // (4*mbW+1) x (4*mbH+1) intra mode grid, with a zeroed top row/left column
	yPlane []uint8 // Source planes (exact image size; importMB replicates the boundary).
	uPlane []uint8
	vPlane []uint8
	topUV  []uint8 // mbW*16 reconstructed top boundary (8 U + 8 V per macroblock)
	// Reconstruction planes, padded to whole macroblocks. After encodeFrame these hold exactly what a decoder
	// reconstructs (loop filter included).
	reconV          []uint8
	reconY          []uint8
	topY            []uint8   // mbW*16 reconstructed top boundary (luma)
	bwHeader        bitWriter // partition 0: headers + modes
	bwCoeffs        bitWriter // partition 1: coefficients
	proba           encProba
	dqm             segmentInfo
	dqY2DC          int
	width           int
	predsW          int
	reconUVStride   int
	reconYStride    int
	uvStride        int
	yStride         int
	baseQuant       int
	dqY1DC          int
	predsBase       int // index of the (0,0) sub-block of macroblock (0,0)
	dqY2AC          int
	dqUVDC          int
	dqUVAC          int
	filterLevel     int
	filterSharpness int
	height          int
	mbH             int
	mbW             int
	filterSimple    bool
}

func newEncoder(width, height int) *encoder {
	mbW := (width + 15) >> 4
	mbH := (height + 15) >> 4
	predsW := 4*mbW + 1
	predsH := 4*mbH + 1
	e := &encoder{
		width:         width,
		height:        height,
		mbW:           mbW,
		mbH:           mbH,
		yStride:       width,
		uvStride:      (width + 1) >> 1,
		reconY:        make([]uint8, mbW*16*mbH*16),
		reconU:        make([]uint8, mbW*8*mbH*8),
		reconV:        make([]uint8, mbW*8*mbH*8),
		reconYStride:  mbW * 16,
		reconUVStride: mbW * 8,
		preds:         make([]uint8, predsW*predsH),
		predsW:        predsW,
		predsBase:     predsW + 1,
		nz:            make([]uint32, mbW+1),
		topY:          make([]uint8, mbW*16),
		topUV:         make([]uint8, mbW*16),
		mbInfo:        make([]mbInfo, mbW*mbH),
	}
	return e
}

// modeScore accumulates the rate-distortion score of a candidate mode set.
type modeScore struct {
	d, sd     int64 // distortion and spectral distortion
	h, r      int64 // header and rate costs
	score     int64
	nz        uint32
	dcNz      bool // whether any post-inverse-WHT luma DC coefficient is non-zero (i16 only)
	modeI16   int
	modesI4   [16]uint8
	modeUV    int
	yDcLevels [16]int16
	yAcLevels [16][16]int16
	uvLevels  [8][16]int16
}

func (rd *modeScore) initScore() {
	rd.d = 0
	rd.sd = 0
	rd.r = 0
	rd.h = 0
	rd.nz = 0
	rd.score = maxCost
}

func (rd *modeScore) copyScore(src *modeScore) {
	rd.d = src.d
	rd.sd = src.sd
	rd.r = src.r
	rd.h = src.h
	rd.nz = src.nz // note that nz is not accumulated, just copied
	rd.score = src.score
}

func (rd *modeScore) addScore(src *modeScore) {
	rd.d += src.d
	rd.sd += src.sd
	rd.r += src.r
	rd.h += src.h
	rd.nz |= src.nz // here the new nz bits are accumulated
	rd.score += src.score
}

func (rd *modeScore) setRDScore(lambda int64) {
	rd.score = (rd.r+rd.h)*lambda + rdDistoMult*(rd.d+rd.sd)
}

//------------------------------------------------------------------------------
// Reconstruction: difference, transform, quantize, inverse transform, add — all at once. The output is the
// reconstructed block in yuvOut and the quantized levels in the score struct.

func (e *encoder) reconstructIntra16(it *iterator, rd *modeScore, yuvOut []uint8, mode int) uint32 {
	ref := it.yuvP[predI16ModeOffsets[mode]:]
	src := it.yuvIn[yOff:]
	dqm := &e.dqm
	nz := uint32(0)
	var tmp [16][16]int16
	var dcTmp [16]int16

	for n := 0; n < 16; n++ {
		fTransform(src[scan[n]:], ref[scan[n]:], tmp[n][:])
	}
	fTransformWHT(&tmp, &dcTmp)
	nz |= uint32(quantizeBlock(&dcTmp, &rd.yDcLevels, &dqm.y2)) << 24

	for n := 0; n < 16; n++ {
		// Zero out the first coefficient so that (a) nz is correct below and (b) finding the last non-zero coefficient
		// in setCoeffs is simplified.
		tmp[n][0] = 0
		nz |= uint32(quantizeBlock(&tmp[n], &rd.yAcLevels[n], &dqm.y1)) << uint(n)
	}

	// Transform back.
	iTransformWHT(&dcTmp, &tmp)
	// Track whether the distributed DC coefficients are all zero: the decoder derives its per-macroblock loop-filter
	// skip from the post-WHT coefficients (RFC 6386 §15.1), so a macroblock whose Y2 levels are non-zero but whose
	// inverse WHT rounds to all-zero DCs is a filtering skip even though rd.nz has the Y2 bit set.
	rd.dcNz = false
	for n := 0; n < 16; n++ {
		if tmp[n][0] != 0 {
			rd.dcNz = true
			break
		}
	}
	for n := 0; n < 16; n++ {
		iTransformOne(ref[scan[n]:], tmp[n][:], yuvOut[scan[n]:])
	}
	return nz
}

func (e *encoder) reconstructIntra4(it *iterator, levels *[16]int16, src, dst []uint8, mode int) int {
	ref := it.yuvP[predI4ModeOffsets[mode]:]
	var tmp [16]int16
	fTransform(src, ref, tmp[:])
	nz := quantizeBlock(&tmp, levels, &e.dqm.y1)
	iTransformOne(ref, tmp[:], dst)
	return nz
}

func (e *encoder) reconstructUV(it *iterator, rd *modeScore, yuvOut []uint8, mode int) uint32 {
	ref := it.yuvP[predUVModeOffsets[mode]:]
	src := it.yuvIn[uOff:]
	dqm := &e.dqm
	nz := uint32(0)
	var tmp [8][16]int16
	for n := 0; n < 8; n++ {
		fTransform(src[scanUV[n]:], ref[scanUV[n]:], tmp[n][:])
	}
	for n := 0; n < 8; n++ {
		nz |= uint32(quantizeBlock(&tmp[n], &rd.uvLevels[n], &dqm.uv)) << uint(n)
	}
	for n := 0; n < 8; n++ {
		iTransformOne(ref[scanUV[n]:], tmp[n][:], yuvOut[scanUV[n]:])
	}
	return nz << 16
}

//------------------------------------------------------------------------------
// Residual costing

func (e *encoder) getCostLuma16(it *iterator, rd *modeScore) int64 {
	var res residual
	r := int64(0)
	it.nzToBytes() // re-import the non-zero context

	// DC
	res.init(0, 1)
	res.setCoeffs(&rd.yDcLevels)
	r += int64(e.proba.getResidualCost(it.topNz[8]+it.leftNz[8], &res))

	// AC
	res.init(1, 0)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			ctx := it.topNz[x] + it.leftNz[y]
			res.setCoeffs(&rd.yAcLevels[x+y*4])
			r += int64(e.proba.getResidualCost(ctx, &res))
			nzVal := 0
			if res.last >= 0 {
				nzVal = 1
			}
			it.topNz[x] = nzVal
			it.leftNz[y] = nzVal
		}
	}
	return r
}

func (e *encoder) getCostLuma4(it *iterator, levels *[16]int16) int64 {
	x, y := it.i4&3, it.i4>>2
	var res residual
	res.init(0, 3)
	ctx := it.topNz[x] + it.leftNz[y]
	res.setCoeffs(levels)
	return int64(e.proba.getResidualCost(ctx, &res))
}

func (e *encoder) getCostUV(it *iterator, rd *modeScore) int64 {
	var res residual
	r := int64(0)
	it.nzToBytes() // re-import the non-zero context
	res.init(0, 2)
	for ch := 0; ch <= 2; ch += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				ctx := it.topNz[4+ch+x] + it.leftNz[4+ch+y]
				res.setCoeffs(&rd.uvLevels[ch*2+x+y*2])
				r += int64(e.proba.getResidualCost(ctx, &res))
				nzVal := 0
				if res.last >= 0 {
					nzVal = 1
				}
				it.topNz[4+ch+x] = nzVal
				it.leftNz[4+ch+y] = nzVal
			}
		}
	}
	return r
}

//------------------------------------------------------------------------------
// Mode decision: pick the best intra16, intra4 and chroma modes by rate-distortion score.

func (e *encoder) pickBestIntra16(it *iterator, rd *modeScore) {
	const numBlocks = 16
	dqm := &e.dqm
	lambda := dqm.lambdaI16
	src := it.yuvIn[yOff:]
	var rdTmp modeScore
	rdCur := &rdTmp
	rdBest := rd
	flat := isFlatSource16(src)

	rd.modeI16 = -1
	for mode := 0; mode < numPredModes; mode++ {
		tmpDst := it.yuvOut2[yOff:] // scratch buffer
		rdCur.modeI16 = mode

		// Reconstruct.
		rdCur.nz = e.reconstructIntra16(it, rdCur, tmpDst, mode)

		// Measure the RD score.
		rdCur.d = sse16x16(src, tmpDst)
		rdCur.sd = 0 // spectral distortion is only used at method >= 4 (not ported)
		rdCur.h = int64(fixedCostsI16[mode])
		rdCur.r = e.getCostLuma16(it, rdCur)
		if flat {
			// Refine the first impression (which was in pixel space).
			if isFlat(rdCur.yAcLevels[:numBlocks], flatnessLimitI16) {
				// The block is very flat: emphasize having a very low distortion.
				rdCur.d *= 2
				rdCur.sd *= 2
			} else {
				flat = false
			}
		}

		// Since intra16 is always examined first, *rd can be overwritten directly.
		rdCur.setRDScore(lambda)
		if mode == 0 || rdCur.score < rdBest.score {
			rdCur, rdBest = rdBest, rdCur
			it.yuvOut, it.yuvOut2 = it.yuvOut2, it.yuvOut
		}
	}
	if rdBest != rd {
		*rd = *rdBest
	}
	rd.setRDScore(dqm.lambdaMode) // finalize the score for mode decision
	it.setIntra16Mode(rd.modeI16)

	// A blocky macroblock (only DCs non-zero) with fairly high distortion: record the max delta so the minimal
	// filtering strength that smooths these blocks out can be derived later.
	if rd.nz&0x100ffff == 0x1000000 && rd.d > int64(dqm.minDisto) {
		dqm.storeMaxDelta(&rd.yDcLevels)
	}
}

// getCostModeI4 returns the mode-cost row for the current sub-block, given the surrounding prediction modes.
func (it *iterator) getCostModeI4(modes *[16]uint8) *[numBModes]uint16 {
	e := it.e
	x, y := it.i4&3, it.i4>>2
	var left, top int
	if x == 0 {
		left = int(e.preds[it.predsIdx+y*e.predsW-1])
	} else {
		left = int(modes[it.i4-1])
	}
	if y == 0 {
		top = int(e.preds[it.predsIdx-e.predsW+x])
	} else {
		top = int(modes[it.i4-4])
	}
	return &fixedCostsI4[top][left]
}

// pickBestIntra4 returns true when intra4 beats the intra16 decision already stored in rd.
func (e *encoder) pickBestIntra4(it *iterator, rd *modeScore) bool {
	dqm := &e.dqm
	lambda := dqm.lambdaI4
	src0 := it.yuvIn[yOff:]
	bestBlocks := it.yuvOut2[yOff:]
	totalHeaderBits := int64(0)
	var rdBest modeScore

	rdBest.initScore()
	rdBest.h = 211 // the cost of signaling "not intra16", VP8BitCost(0, 145)
	rdBest.setRDScore(dqm.lambdaMode)
	it.startI4()
	for {
		var rdI4 modeScore
		bestMode := -1
		src := src0[scan[it.i4]:]
		modeCosts := it.getCostModeI4(&rd.modesI4)
		bestBlock := bestBlocks[scan[it.i4]:]
		tmpDst := it.yuvP[i4TMP:] // scratch buffer

		rdI4.initScore()
		it.makeIntra4Preds()
		for mode := 0; mode < numBModes; mode++ {
			var rdTmp modeScore
			var tmpLevels [16]int16

			// Reconstruct.
			rdTmp.nz = uint32(e.reconstructIntra4(it, &tmpLevels, src, tmpDst, mode)) << uint(it.i4)

			// Compute the RD score.
			rdTmp.d = sse4x4(src, tmpDst)
			rdTmp.sd = 0
			rdTmp.h = int64(modeCosts[mode])

			// Add a flatness penalty to avoid flat areas being mispredicted by a complex mode.
			if mode > 0 && isFlat([][16]int16{tmpLevels}, flatnessLimitI4) {
				rdTmp.r = flatnessPenalty // times one block
			} else {
				rdTmp.r = 0
			}

			// Early-out check.
			rdTmp.setRDScore(lambda)
			if bestMode >= 0 && rdTmp.score >= rdI4.score {
				continue
			}

			// Finish computing the score.
			rdTmp.r += e.getCostLuma4(it, &tmpLevels)
			rdTmp.setRDScore(lambda)

			if bestMode < 0 || rdTmp.score < rdI4.score {
				rdI4.copyScore(&rdTmp)
				bestMode = mode
				tmpDst, bestBlock = bestBlock, tmpDst
				rdBest.yAcLevels[it.i4] = tmpLevels
			}
		}
		rdI4.setRDScore(dqm.lambdaMode)
		rdBest.addScore(&rdI4)
		if rdBest.score >= rd.score {
			return false
		}
		totalHeaderBits += rdI4.h // equal to modeCosts[bestMode]
		if totalHeaderBits > maxI4HeaderBits {
			return false
		}
		// Copy the selected samples to the right place if they are not there already.
		if &bestBlock[0] != &bestBlocks[scan[it.i4]] {
			copy4x4(bestBlock, bestBlocks[scan[it.i4]:])
		}
		rd.modesI4[it.i4] = uint8(bestMode)
		nzVal := 0
		if rdI4.nz != 0 {
			nzVal = 1
		}
		it.topNz[it.i4&3] = nzVal
		it.leftNz[it.i4>>2] = nzVal
		if !it.rotateI4(bestBlocks) {
			break
		}
	}

	// Finalize the state.
	rd.copyScore(&rdBest)
	it.setIntra4Modes(&rd.modesI4)
	it.yuvOut, it.yuvOut2 = it.yuvOut2, it.yuvOut
	rd.yAcLevels = rdBest.yAcLevels
	rd.dcNz = false // intra4 macroblocks have no Y2 block
	return true     // select intra4x4 over intra16x16
}

func copy4x4(src, dst []uint8) {
	for j := 0; j < 4; j++ {
		copy(dst[j*bps:j*bps+4], src[j*bps:j*bps+4])
	}
}

func (e *encoder) pickBestUV(it *iterator, rd *modeScore) {
	const numBlocks = 8
	dqm := &e.dqm
	lambda := dqm.lambdaUV
	src := it.yuvIn[uOff:]
	tmpDst := it.yuvOut2[uOff:] // scratch buffer
	dst0 := it.yuvOut[uOff:]
	dst := dst0
	var rdBest modeScore

	rd.modeUV = -1
	rdBest.initScore()
	for mode := 0; mode < numPredModes; mode++ {
		var rdUV modeScore

		// Reconstruct.
		rdUV.nz = e.reconstructUV(it, &rdUV, tmpDst, mode)

		// Compute the RD score.
		rdUV.d = sse16x8(src, tmpDst)
		rdUV.sd = 0 // not calling TDisto here: it tends to flatten areas
		rdUV.h = int64(fixedCostsUV[mode])
		rdUV.r = e.getCostUV(it, &rdUV)
		if mode > 0 && isFlat(rdUV.uvLevels[:numBlocks], flatnessLimitUV) {
			rdUV.r += flatnessPenalty * numBlocks
		}

		rdUV.setRDScore(lambda)
		if mode == 0 || rdUV.score < rdBest.score {
			rdBest.copyScore(&rdUV)
			rd.modeUV = mode
			rd.uvLevels = rdUV.uvLevels
			dst, tmpDst = tmpDst, dst
		}
	}
	it.mb.uvMode = uint8(rd.modeUV)
	rd.addScore(&rdBest)
	if &dst[0] != &dst0[0] { // copy the 16x8 block if needed
		copy16x8(dst, dst0)
	}
}

func copy16x8(src, dst []uint8) {
	for j := 0; j < 8; j++ {
		copy(dst[j*bps:j*bps+16], src[j*bps:j*bps+16])
	}
}

// decimate performs the complete mode decision and reconstruction for the current macroblock, returning whether it is
// skippable (all-zero coefficients).
func (e *encoder) decimate(it *iterator, rd *modeScore) bool {
	rd.initScore()

	// Predictions for luma16x16 and chroma8x8 can be computed upfront; luma4x4 predictions are done as-we-go.
	it.makeLuma16Preds()
	it.makeChroma8Preds()

	e.pickBestIntra16(it, rd)
	e.pickBestIntra4(it, rd)
	e.pickBestUV(it, rd)

	skipped := rd.nz == 0
	it.mb.skip = skipped
	// The decoder's loop-filter skip (§15.1: "no DCT coefficient coded for the whole macroblock") is derived from the
	// post-WHT coefficients, so the Y2 bit of rd.nz counts only when the inverse WHT actually produced a non-zero DC.
	it.mb.noCoeffs = rd.nz&0xffffff == 0 && !rd.dcNz
	return skipped
}

//------------------------------------------------------------------------------
// Residual coding and recording

func (e *encoder) codeResiduals(bw *bitWriter, it *iterator, rd *modeScore) {
	var res residual
	i16 := it.mb.typ == 1
	it.nzToBytes()

	if i16 {
		res.init(0, 1)
		res.setCoeffs(&rd.yDcLevels)
		nzVal := e.proba.putCoeffs(bw, it.topNz[8]+it.leftNz[8], &res)
		it.topNz[8] = nzVal
		it.leftNz[8] = nzVal
		res.init(1, 0)
	} else {
		res.init(0, 3)
	}

	// luma-AC
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			ctx := it.topNz[x] + it.leftNz[y]
			res.setCoeffs(&rd.yAcLevels[x+y*4])
			nzVal := e.proba.putCoeffs(bw, ctx, &res)
			it.topNz[x] = nzVal
			it.leftNz[y] = nzVal
		}
	}

	// U/V
	res.init(0, 2)
	for ch := 0; ch <= 2; ch += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				ctx := it.topNz[4+ch+x] + it.leftNz[4+ch+y]
				res.setCoeffs(&rd.uvLevels[ch*2+x+y*2])
				nzVal := e.proba.putCoeffs(bw, ctx, &res)
				it.topNz[4+ch+x] = nzVal
				it.leftNz[4+ch+y] = nzVal
			}
		}
	}
	it.bytesToNz()
}

// recordResiduals is the same walk as codeResiduals but only records statistics.
func (e *encoder) recordResiduals(it *iterator, rd *modeScore) {
	var res residual
	it.nzToBytes()

	if it.mb.typ == 1 { // i16x16
		res.init(0, 1)
		res.setCoeffs(&rd.yDcLevels)
		nzVal := e.proba.recordCoeffs(it.topNz[8]+it.leftNz[8], &res)
		it.topNz[8] = nzVal
		it.leftNz[8] = nzVal
		res.init(1, 0)
	} else {
		res.init(0, 3)
	}

	// luma-AC
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			ctx := it.topNz[x] + it.leftNz[y]
			res.setCoeffs(&rd.yAcLevels[x+y*4])
			nzVal := e.proba.recordCoeffs(ctx, &res)
			it.topNz[x] = nzVal
			it.leftNz[y] = nzVal
		}
	}

	// U/V
	res.init(0, 2)
	for ch := 0; ch <= 2; ch += 2 {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				ctx := it.topNz[4+ch+x] + it.leftNz[4+ch+y]
				res.setCoeffs(&rd.uvLevels[ch*2+x+y*2])
				nzVal := e.proba.recordCoeffs(ctx, &res)
				it.topNz[4+ch+x] = nzVal
				it.leftNz[4+ch+y] = nzVal
			}
		}
	}
	it.bytesToNz()
}

// resetAfterSkip resets the coefficient contexts after a skipped macroblock.
func (it *iterator) resetAfterSkip() {
	if it.mb.typ == 1 {
		it.e.nz[it.nzIdx] = 0 // reset all predictors
		it.leftNz[8] = 0
	} else {
		it.e.nz[it.nzIdx] &= 1 << 24 // preserve the dc_nz bit
	}
}

//------------------------------------------------------------------------------
// Encode driver: one full statistics pass (deciding the adaptive token probabilities and the skip probability), then
// the final coding pass.

func (e *encoder) encodeFrame(quality float32) {
	e.proba.initDefaults()
	e.setSegmentParams(quality)
	e.proba.calculateLevelCosts()
	e.proba.resetTokenStats()
	e.proba.nbSkip = 0

	// Pass 1: collect statistics over all macroblocks (the full frame, not a subsampled probe).
	it := newIterator(e)
	for {
		var info modeScore
		it.importMB()
		if e.decimate(it, &info) {
			e.proba.nbSkip++
		}
		e.recordResiduals(it, &info)
		it.saveBoundary()
		if !it.next() {
			break
		}
	}
	e.proba.finalizeSkipProba(e.mbW * e.mbH)
	e.proba.finalizeTokenProbas()
	e.proba.calculateLevelCosts()

	// Pass 2: final coding with the updated probabilities.
	e.bwCoeffs.init()
	it.reset()
	for {
		var info modeScore
		it.importMB()
		// Order matters: decimate first, then decide how to code the skip, if any.
		if !e.decimate(it, &info) || !e.proba.useSkipProba {
			e.codeResiduals(&e.bwCoeffs, it, &info)
		} else { // reset the predictors after a skip
			it.resetAfterSkip()
		}
		it.exportMB()
		it.saveBoundary()
		if !it.next() {
			break
		}
	}
	e.bwCoeffs.finish()

	e.adjustFilterStrength()
	e.applyLoopFilter()
}
