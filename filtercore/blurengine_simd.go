// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd && (arm64 || amd64)

package filtercore

import "simd/archsimd"

// The goexperiment.simd blur kernels, written against archsimd's 128-bit types — the widest shape arm64 and amd64
// share, and a natural fit here because an RGBA sample is exactly four lanes. Both passes vectorize across the four
// channels and nothing else: each output sample depends on the sample before it (the Gaussian pass through the
// circular buffer's rotation, the three-box pass through a genuine loop-carried prefix sum), so there is no
// across-samples axis to take without changing the arithmetic. Vectorizing across taps would change it too — floats do
// not reassociate, and a tree reduction over the window is a different sum than the scalar's ascending-tap chain — so
// the Gaussian kernel keeps the scalar's tap order exactly and only widens each tap from four scalar operations to one.
//
// Every operation below was picked for bit-exactness with the portable form, not for convenience:
//
//   - The window walk. The scalar indexes the circular buffer as (i+base)%window for ascending i. Ascending i visits
//     slots base..window-1 and then 0..base-1, both contiguous, so the kernel splits the walk into those two runs and
//     drops the modulo. That is a re-spelling of the same index sequence in the same order, not a reassociation: tap k
//     of the vector loop is tap k of the scalar loop, with the same weight, added to the same partial sum.
//   - The accumulation. "sum += buf[s]*k" and "sum*255 + 0.5" are plain Go expressions, which the language permits an
//     implementation to contract into a fused multiply-add. This build's arm64 compiler does (both disassemble to
//     FMADDS) and its amd64 compiler at the module's GOAMD64=v1 baseline does not (MULSS then ADDSS), so exprMulAdd4
//     is spelled per arch to reproduce whichever lowering the scalar twin in the same binary got. Getting that backwards
//     diverges on hostile inputs, and TestBlurEngineSIMDContractionNegativeControl proves the equivalence fuzz's corpus
//     is strong enough to see that divergence — byte-quantized output hides small ones, so the corpus is built to
//     surface them rather than merely assumed to.
//   - The clamps. "if v > 255 {255} else if v < 0 {0}" leaves NaN alone (both comparisons are false for it), which the
//     hardware Min/Max would not — they propagate NaN — so the clamps are compare + IfElse, applied in the scalar's
//     order. x.IfElse(mask, y) keeps x where mask is true.
//   - The float-to-byte. The scalar's uint8(v) disassembles to the truncating signed convert (FCVTZS / CVTTSS2SIL)
//     followed by taking the low byte, which is exactly ConvertToInt32 plus the low-byte gather. The two arches
//     disagree on what the convert returns for NaN (0 on arm64, 0x80000000 on amd64) but agree on its low byte, which
//     is all either form keeps — and after the clamp above every non-NaN lane is an in-range [0, 255] value where they
//     agree outright.
//   - The integer pipeline. Uint32x4.Add/Sub are wrapping lane operations, which is what the scalar's uint32 arithmetic
//     is, and ScaledDividerU32's "(v*factor)>>32" is an exact 32x32->64 high half (divideVec, spelled per arch).
//
// The byte gathers are per-arch for the reason imagecore's and maskfilter's are: every pack-and-narrow method archsimd
// offers for them is AVX-512 on amd64. See packLowBytes.
//
// Both kernels are locked against their portable twins by TestBlurEngineSIMDMatchesScalar, which fuzzes whole segments
// and requires bit-identical output words and bit-identical residual pass state.

// init swaps the dispatch variables to the simd kernels the per-arch preference constants elect. Unqualified CPUs keep
// the portable forms.
func init() {
	if !simdKernelsSupported() {
		return
	}
	if preferSIMDGaussianSegment {
		gaussianBlurSegmentFn = gaussianBlurSegmentSIMD
	}
	if preferSIMDThreeBoxSegment {
		threeBoxBlurSegmentFn = threeBoxBlurSegmentSIMD
	}
}

// maxGaussianWindow is the largest window makeGaussianPassMaker can hand a gaussianPass: it rejects sigma >= 2, and
// SigmaToRadius is ceil(3*sigma), so the radius tops out at 6 and the window at 2*6+1. gaussianBlurSegmentSIMD sizes
// its per-segment broadcast kernel to that bound and falls back to the portable body for anything larger, so a future
// widening of the sigma range degrades rather than corrupts.
const maxGaussianWindow = 13

// unpackWord spreads an 8888 word's four bytes into the low byte of four 32-bit lanes, in channel order — the vector
// form of the scalar's "w&0xff, (w>>8)&0xff, (w>>16)&0xff, w>>24", since on a little-endian machine those are just the
// word's bytes 0..3. Two zero-extending widens do it (byte to word, word to long); the broadcast only has to get the
// word into a vector register, because the widens read the low half. Lane order probed before use.
func unpackWord(w uint32) archsimd.Uint32x4 {
	return archsimd.BroadcastUint32x4(w).ReshapeToUint8s().ExtendLo8ToUint16().ExtendLo4ToUint32()
}

///////////////////////////////////////////////////////////////////////////////
// GaussianPass

// gaussianBlurSegmentSIMD is gaussianBlurSegmentGeneric with the per-sample window walk in vector registers: the four
// channels of one buffer slot are one Float32x4, so a tap is one 16-byte load, one weight load and one multiply-add
// instead of four of each, and the (i+base)%window index arithmetic collapses into two contiguous runs.
//
// The weights are broadcast once per segment rather than per tap: archsimd's BroadcastFloat32x4 of a loaded scalar
// costs four instructions on arm64 (Go's SSA has no loop-invariant code motion, so it rebuilds the splat every
// iteration), while a pre-splatted [4]float32 is a single 16-byte load. A segment's worth of samples pays for the
// window-sized setup many times over, and the scratch lives on the stack so nothing is allocated.
func gaussianBlurSegmentSIMD(p *gaussianPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	window := p.window
	if window > maxGaussianWindow {
		gaussianBlurSegmentGeneric(p, n, src, srcStride, dst, dstStride)
		return
	}
	base := p.base
	buf := p.srcBuffer[:window]
	var weightStore [maxGaussianWindow][4]float32
	weights := weightStore[:window]
	if dst != nil {
		for i, k := range p.kernel[:window] {
			archsimd.BroadcastFloat32x4(k).StoreArray(&weights[i])
		}
	}
	inv255 := archsimd.BroadcastFloat32x4(1.0 / 255.0)
	scale255 := archsimd.BroadcastFloat32x4(255)
	half := archsimd.BroadcastFloat32x4(0.5)
	var zero archsimd.Float32x4
	gather := newLowByteGather()
	srcCursor := int32(0)
	dstCursor := int32(0)
	for ; n > 0; n-- {
		var leadingEdge archsimd.Float32x4
		if src != nil {
			leadingEdge = unpackWord(src[srcCursor]).BitsToInt32().ConvertToFloat32().Mul(inv255)
			srcCursor += srcStride
		}
		// (base+window-1)%window, with base known to be in [0, window).
		lead := base - 1
		if lead < 0 {
			lead = window - 1
		}
		leadingEdge.StoreArray(&buf[lead])

		if dst != nil {
			var sum archsimd.Float32x4
			// Ascending tap order, split at the buffer's wrap point: taps 0..window-base-1 read slots base..window-1
			// and taps window-base..window-1 read slots 0..base-1. Each run reslices its weights to the run's own
			// length so the index is provably in range.
			tail := buf[base:]
			tailWeights := weights[:len(tail)]
			for i := range tail {
				sum = exprMulAdd4(archsimd.LoadFloat32x4Array(&tail[i]),
					archsimd.LoadFloat32x4Array(&tailWeights[i]), sum)
			}
			headWeights := weights[len(tail):]
			head := buf[:len(headWeights)]
			for i := range head {
				sum = exprMulAdd4(archsimd.LoadFloat32x4Array(&head[i]),
					archsimd.LoadFloat32x4Array(&headWeights[i]), sum)
			}
			v := exprMulAdd4(sum, scale255, half)
			v = scale255.IfElse(v.Greater(scale255), v)
			v = zero.IfElse(v.Less(zero), v)
			dst[dstCursor] = packLowBytes(v.ConvertToInt32().ToBits(), gather)
			dstCursor += dstStride
		}
		base++
		if base >= window {
			base = 0
		}
	}
	p.base = base
}

///////////////////////////////////////////////////////////////////////////////
// ThreeBoxApproxPass

// threeBoxBlurSegmentSIMD is threeBoxBlurSegmentGeneric with the four channels in one Uint32x4. The three running sums
// and the three circular buffers all hold [4]uint32 already, so each of them becomes one register and one 16-byte
// memory access where the scalar performs four; the scaled divide becomes one vector high-half multiply.
//
// The scan axis is not vectorized and cannot be: sum0 += leadingEdge, sum1 += sum0, sum2 += sum1 is a prefix sum whose
// every step needs the previous sample's result. Ganging independent rows (or columns) into the spare lanes would need
// four pass states and a caller-level rewrite of runPass and the blurPass interface — shared, portable code reshaped
// solely for one build's vector lane — so this takes the four-channel win and leaves the driver alone.
func threeBoxBlurSegmentSIMD(p *threeBoxApproxPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	sum0 := archsimd.LoadUint32x4Array(&p.sum0)
	sum1 := archsimd.LoadUint32x4Array(&p.sum1)
	sum2 := archsimd.LoadUint32x4Array(&p.sum2)
	c0, c1, c2 := p.cursor0, p.cursor1, p.cursor2
	buffer0, buffer1, buffer2 := p.buffer0, p.buffer1, p.buffer2
	len0, len1, len2 := int32(len(buffer0)), int32(len(buffer1)), int32(len(buffer2))
	divisor := newDivisorVec(p.divisorFactor)
	gather := newLowByteGather()
	srcCursor := int32(0)
	dstCursor := int32(0)
	for ; n > 0; n-- {
		var leadingEdge archsimd.Uint32x4
		if src != nil {
			leadingEdge = unpackWord(src[srcCursor])
			srcCursor += srcStride
		}

		sum0 = sum0.Add(leadingEdge)
		sum1 = sum1.Add(sum0)
		sum2 = sum2.Add(sum1)
		blurred := divideVec(sum2, divisor)

		// Reduce the sums by the trailing edges stored in the circular buffers, storing the current sums for the next
		// go-around *before* their own trailing-edge subtraction.
		trailing2 := archsimd.LoadUint32x4Array(&buffer2[c2])
		sum1.StoreArray(&buffer2[c2])
		c2++
		if c2 >= len2 {
			c2 = 0
		}
		trailing1 := archsimd.LoadUint32x4Array(&buffer1[c1])
		sum0.StoreArray(&buffer1[c1])
		c1++
		if c1 >= len1 {
			c1 = 0
		}
		trailing0 := archsimd.LoadUint32x4Array(&buffer0[c0])
		leadingEdge.StoreArray(&buffer0[c0])
		c0++
		if c0 >= len0 {
			c0 = 0
		}
		sum2 = sum2.Sub(trailing2)
		sum1 = sum1.Sub(trailing1)
		sum0 = sum0.Sub(trailing0)

		if dst != nil {
			dst[dstCursor] = packLowBytes(blurred, gather)
			dstCursor += dstStride
		}
	}
	sum0.StoreArray(&p.sum0)
	sum1.StoreArray(&p.sum1)
	sum2.StoreArray(&p.sum2)
	p.cursor0, p.cursor1, p.cursor2 = c0, c1, c2
}
