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

import (
	"math"
	"math/rand/v2"
	"reflect"
	"simd/archsimd"
	"strconv"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
)

// The input domain these kernels are fuzzed over, and why it is the whole of it.
//
// Source pixels: every blurSegment call reads 8888 words out of an N32 surface, so the *complete* input domain for the
// leading edge is the 32-bit word — the tests draw it uniformly at random, which covers every byte of every channel.
// There is no float class to hunt for on that side: the Gaussian pass turns each byte into k*(1/255) for an integer k
// in [0, 255], and the three-box pass keeps it an integer.
//
// Kernel weights: compute1DBlurKernel produces normalized exp() weights, so a real kernel is finite, non-negative and
// sums to one, which bounds every accumulator to [0, 1] and every pre-clamp output to [0.5, 255.5]. The realistic
// subtests fuzz exactly that domain (sigma swept across the whole gaussianPass range, from just above the no-blur
// cutoff to just under the three-box handover). Because that domain never produces a NaN, an infinity or a denormal,
// it also cannot by itself distinguish a fused multiply-add from an unfused one, so a second subtest replaces the
// kernel with hostile float classes — the same corpus shape the shaders package uses. That is not a reachable input;
// it is the sensitivity the equivalence claim needs, and TestBlurEngineSIMDContractionNegativeControl proves it
// delivers.

// blurSIMDHostileFloats seeds kernel weights with every float class the arithmetic can meet, so the vector/scalar
// comparison exercises NaN propagation, infinities, signed zero, denormals and rounding boundaries.
var blurSIMDHostileFloats = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, 0.5, -0.5, 0.9999999, 1.0000001, 255, 256, -255,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 1e-38, -1e-38,
	3.4e38, -3.4e38, 1e-45, 0.1, 1.0 / 3.0, 2.0 / 3.0,
}

// blurSIMDQuietFloat turns arbitrary random bits into a value the arithmetic could actually hold: signaling NaNs are
// quieted, because no IEEE operation emits one and their handling differs between the comparison-based scalar clamps
// and any hardware min/max.
func blurSIMDQuietFloat(bits uint32) float32 {
	const expMask, quietBit = 0x7F800000, 0x00400000
	if bits&expMask == expMask && bits&0x007FFFFF != 0 {
		bits |= quietBit
	}
	return math.Float32frombits(bits)
}

// blurSIMDSegment is one blurSegment call's shape. runPass issues up to three per scanline and any of them may have a
// nil src (the trailing drain), a nil dst (the leading preload), or both (the degenerate gap), so the fuzz draws all
// four combinations along with the strides the X pass (1) and the Y pass (a row stride) use.
type blurSIMDSegment struct {
	n         int32
	srcStride int32
	dstStride int32
	band      int32
	hasSrc    bool
	hasDst    bool
}

// blurSIMDRandSegment draws one segment shape: mostly short runs that cover every phase of the kernels' index
// arithmetic, one in five a realistic scanline length. band selects the source run's shape (see blurSIMDRandWords).
func blurSIMDRandSegment(rng *rand.Rand, band int32) blurSIMDSegment {
	s := blurSIMDSegment{
		n:         1 + int32(rng.IntN(67)),
		srcStride: 1 + int32(rng.IntN(4)),
		dstStride: 1 + int32(rng.IntN(4)),
		band:      band,
		hasSrc:    rng.IntN(8) != 0,
		hasDst:    rng.IntN(8) != 0,
	}
	if rng.IntN(5) == 0 {
		s.n = 256
	}
	return s
}

// blurSIMDGuard is the pattern written after a destination run; a kernel that stores past the last sample it was asked
// for corrupts it. The vector kernels write one word at a time, but they compute it from a whole register, so the last
// store is the one place an off-by-one could reach outside the run.
const blurSIMDGuard = 0x5A5A5A5A

// blurSIMDBuffers builds the source words and the two guarded destination runs for one segment. Both destinations start
// identical so a kernel that fails to write a sample is caught by the comparison rather than hidden by it.
func blurSIMDBuffers(rng *rand.Rand, s blurSIMDSegment) (src, dstA, dstB []uint32) {
	if s.hasSrc {
		src = blurSIMDRandWords(rng, int((s.n-1)*s.srcStride+1), s.band)
	}
	if s.hasDst {
		n := int((s.n-1)*s.dstStride+1) + 8
		dstA = make([]uint32, n)
		dstB = make([]uint32, n)
		for i := range dstA {
			dstA[i] = blurSIMDGuard
			dstB[i] = blurSIMDGuard
		}
	}
	return src, dstA, dstB
}

// blurSIMDRandWords fills a source run in one of three shapes. A negative band draws uniform 32-bit words, which is the
// complete input domain a real N32 surface can present. A band of zero repeats one word, the way a solid-color region
// looks, and a positive band keeps every channel within that many levels of a random center, the way a smooth region
// looks. The two flat shapes are what give the float comparison its reach: paired with a zero-sum kernel (see
// blurSIMDHostileKernel) they make the window sum a near-total cancellation of large products, so the per-product
// rounding a fused multiply-add does not perform is lifted out of the far end of the mantissa and into the output byte.
// TestBlurEngineSIMDContractionNegativeControl measures how much reach that buys: with uniform words alone a swapped
// contraction spelling changed 8 of 4096 hostile segments, and with these shapes it changes 1274 of them.
func blurSIMDRandWords(rng *rand.Rand, n int, band int32) []uint32 {
	w := make([]uint32, n)
	if band < 0 {
		for i := range w {
			w[i] = rng.Uint32()
		}
		return w
	}
	var center [4]int32
	for c := range center {
		center[c] = int32(rng.IntN(256))
	}
	for i := range w {
		var v uint32
		for c := range 4 {
			level := center[c]
			if band > 0 {
				level += int32(rng.IntN(int(2*band+1))) - band
			}
			v |= uint32(min(max(level, 0), 255)) << (8 * c)
		}
		w[i] = v
	}
	return w
}

// blurSIMDHostileKernel writes the same hostile weights into both passes' kernels and returns the source band that
// exposes them. None of the three classes is reachable from compute1DBlurKernel. The "wild" class draws from every
// float class the arithmetic can meet, exercising NaN propagation, the infinities, signed zero and denormals through
// the clamp and the float-to-byte convert, and pairs with the uniform source domain. The two "zero-sum" classes
// alternate sign at a common magnitude and are then mean-corrected to sum to zero; paired with a smooth or a solid
// source run they leave a window sum built almost entirely out of rounding, which is the only way a byte-quantized
// output can see the difference between a fused and an unfused multiply-add.
func blurSIMDHostileKernel(rng *rand.Rand, want, got []float32) (band int32) {
	switch rng.IntN(4) {
	case 0:
		for i := range want {
			k := blurSIMDQuietFloat(rng.Uint32())
			if rng.IntN(3) == 0 {
				k = blurSIMDHostileFloats[rng.IntN(len(blurSIMDHostileFloats))]
			}
			want[i], got[i] = k, k
		}
		return -1
	case 1:
		blurSIMDZeroSumKernel(rng, want, got, 0, 7)
		return 1
	default:
		blurSIMDZeroSumKernel(rng, want, got, 17, 26)
		return 0
	}
}

// blurSIMDBandFor draws a source-run shape for the subtests that are not exercising a particular one: mostly the
// complete uniform word domain, with the smooth and solid shapes mixed in because real surfaces are full of them.
func blurSIMDBandFor(rng *rand.Rand) int32 {
	switch rng.IntN(4) {
	case 0:
		return 0
	case 1:
		return 1 + int32(rng.IntN(8))
	default:
		return -1
	}
}

// blurSIMDZeroSumKernel fills both kernels with sign-alternating weights of a common random magnitude drawn from
// 2**minExp to 2**maxExp, then subtracts their mean so the weights sum to (very nearly) zero. Over a solid source run
// that makes the whole window sum rounding residue, whose size scales with the magnitude; the residue class's exponent
// range is the one that lands the resulting output byte inside [0, 255] rather than off either end of the clamp.
func blurSIMDZeroSumKernel(rng *rand.Rand, want, got []float32, minExp, maxExp int) {
	magnitude := float32(math.Exp2(float64(minExp + rng.IntN(maxExp-minExp))))
	var total float32
	for i := range want {
		k := magnitude * (0.5 + rng.Float32())
		if i&1 == 1 {
			k = -k
		}
		want[i] = k
		total += k
	}
	mean := total / float32(len(want))
	for i := range want {
		want[i] -= mean
		got[i] = want[i]
	}
}

// blurSIMDEqWords requires the two destination runs to be word-identical, guard bytes included.
func blurSIMDEqWords(t *testing.T, name string, step int, got, want []uint32) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s step %d: word %d is %08x, want %08x", name, step, i, got[i], want[i])
			}
		}
	}
}

// blurSIMDEqGaussianState requires the residual pass state to be bit-identical. The circular buffer is compared through
// Float32bits because a hostile kernel puts NaNs in it, and NaN != NaN.
func blurSIMDEqGaussianState(t *testing.T, name string, step int, got, want *gaussianPass) {
	t.Helper()
	if got.base != want.base {
		t.Fatalf("%s step %d: base is %d, want %d", name, step, got.base, want.base)
	}
	for i := range want.srcBuffer {
		for c := range 4 {
			g := math.Float32bits(got.srcBuffer[i][c])
			w := math.Float32bits(want.srcBuffer[i][c])
			if g != w {
				t.Fatalf("%s step %d: srcBuffer[%d][%d] is %08x, want %08x", name, step, i, c, g, w)
			}
		}
	}
}

// blurSIMDEqThreeBoxState requires the residual pass state to be bit-identical. Everything in it is an integer, so a
// deep comparison is exact.
func blurSIMDEqThreeBoxState(t *testing.T, name string, step int, got, want *threeBoxApproxPass) {
	t.Helper()
	if got.sum0 != want.sum0 || got.sum1 != want.sum1 || got.sum2 != want.sum2 {
		t.Fatalf("%s step %d: sums are %v/%v/%v, want %v/%v/%v", name, step,
			got.sum0, got.sum1, got.sum2, want.sum0, want.sum1, want.sum2)
	}
	if got.cursor0 != want.cursor0 || got.cursor1 != want.cursor1 || got.cursor2 != want.cursor2 {
		t.Fatalf("%s step %d: cursors are %d/%d/%d, want %d/%d/%d", name, step,
			got.cursor0, got.cursor1, got.cursor2, want.cursor0, want.cursor1, want.cursor2)
	}
	if !reflect.DeepEqual(got.buffer0, want.buffer0) || !reflect.DeepEqual(got.buffer1, want.buffer1) ||
		!reflect.DeepEqual(got.buffer2, want.buffer2) {
		t.Fatalf("%s step %d: circular buffers diverged", name, step)
	}
}

// blurSIMDGaussianPair builds two gaussianPasses for the same sigma, both started, so one can run the portable body and
// the other the vector kernel over the identical segment stream.
func blurSIMDGaussianPair(sigma float32) (want, got *gaussianPass) {
	maker := makeGaussianPassMaker(sigma)
	want = maker.makePass().(*gaussianPass)
	got = maker.makePass().(*gaussianPass)
	want.startBlur()
	got.startBlur()
	return want, got
}

// blurSIMDRunGaussian drives a run of segments through both bodies, comparing the destination words and the residual
// pass state after every one — the state carries into the next segment, so a divergence there is as fatal as a
// divergence in the output.
func blurSIMDRunGaussian(t *testing.T, name string, rng *rand.Rand, want, got *gaussianPass, steps int, band int32) {
	t.Helper()
	for step := range steps {
		s := blurSIMDRandSegment(rng, band)
		src, dstA, dstB := blurSIMDBuffers(rng, s)
		gaussianBlurSegmentGeneric(want, s.n, src, s.srcStride, dstA, s.dstStride)
		gaussianBlurSegmentSIMD(got, s.n, src, s.srcStride, dstB, s.dstStride)
		if s.hasDst {
			blurSIMDEqWords(t, name, step, dstB, dstA)
		}
		blurSIMDEqGaussianState(t, name, step, got, want)
	}
}

// TestBlurEngineSIMDMatchesScalar drives both simd blur kernels and their portable twins over identical segment streams
// and requires bit-identical output words and bit-identical residual pass state. The source domain is complete (random
// 32-bit words); the kernel-weight domain is covered both in its reachable shape and in a hostile one that gives the
// float comparison its teeth.
func TestBlurEngineSIMDMatchesScalar(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd blur kernels require; dispatch stays on the portable forms")
	}
	rng := rand.New(rand.NewPCG(9, 17))

	// Every window makeGaussianPassMaker can produce, at both ends and in the middle: radius 1 through 6.
	t.Run("gaussianRealisticKernels", func(t *testing.T) {
		for _, sigma := range []float32{0.031, 0.2, 0.33, 0.34, 0.7, 1, 1.25, 1.6667, 1.9999999} {
			want, got := blurSIMDGaussianPair(sigma)
			if want.window > maxGaussianWindow {
				t.Fatalf("sigma %v yields window %d, above the vector kernel's %d bound",
					sigma, want.window, maxGaussianWindow)
			}
			blurSIMDRunGaussian(t, "gaussian sigma="+strconv.FormatFloat(float64(sigma), 'g', -1, 32),
				rng, want, got, 96, blurSIMDBandFor(rng))
		}
	})

	// The same kernels with hostile weights. Not a reachable input (see this file's header) — this is what makes the
	// comparison sensitive to the fused/unfused choice and to NaN handling in the clamp and the float-to-byte convert.
	t.Run("gaussianHostileKernels", func(t *testing.T) {
		for _, sigma := range []float32{0.1, 0.5, 1.2, 1.9} {
			for range 256 {
				want, got := blurSIMDGaussianPair(sigma)
				band := blurSIMDHostileKernel(rng, want.kernel, got.kernel)
				blurSIMDRunGaussian(t, "gaussianHostile", rng, want, got, 16, band)
			}
		}
	})

	// A window past the vector kernel's stack-scratch bound must fall back to the portable body rather than misbehave.
	// makeGaussianPassMaker cannot build one, so the pass is assembled by hand.
	t.Run("gaussianOversizedWindowFallsBack", func(t *testing.T) {
		const window = maxGaussianWindow + 8
		build := func() *gaussianPass {
			p := &gaussianPass{
				radius:    (window - 1) / 2,
				window:    window,
				kernel:    make([]float32, window),
				srcBuffer: make([][4]float32, window),
			}
			compute1DBlurKernel(float32(window)/6, (window-1)/2, p.kernel)
			p.startBlur()
			return p
		}
		blurSIMDRunGaussian(t, "gaussianOversized", rng, build(), build(), 32, blurSIMDBandFor(rng))
	})

	// The three-box pass over its whole reachable window range: 2 (the smallest evalBlurPasses will run) through 254
	// (the largest makeThreeBoxPassMaker accepts), odd and even alike, since the parity picks a different third-buffer
	// size, border and divisor.
	t.Run("threeBox", func(t *testing.T) {
		for _, window := range []int32{2, 3, 4, 5, 8, 9, 16, 17, 63, 64, 127, 128, 253, 254} {
			sigma := float32(window) * 4 / (3 * sqrtf32(2*math.Pi))
			maker := makeThreeBoxPassMaker(sigma)
			if maker == nil {
				t.Fatalf("no three-box maker for window %d (sigma %v)", window, sigma)
			}
			want := maker.makePass().(*threeBoxApproxPass)
			got := maker.makePass().(*threeBoxApproxPass)
			want.startBlur()
			got.startBlur()
			name := "threeBox window=" + strconv.Itoa(int(maker.window))
			for step := range 64 {
				s := blurSIMDRandSegment(rng, blurSIMDBandFor(rng))
				src, dstA, dstB := blurSIMDBuffers(rng, s)
				threeBoxBlurSegmentGeneric(want, s.n, src, s.srcStride, dstA, s.dstStride)
				threeBoxBlurSegmentSIMD(got, s.n, src, s.srcStride, dstB, s.dstStride)
				if s.hasDst {
					blurSIMDEqWords(t, name, step, dstB, dstA)
				}
				blurSIMDEqThreeBoxState(t, name, step, got, want)
			}
		}
	})

	// End to end through the real driver: runPass' three-phase scanline walk, both axes, the inflated intermediate
	// buffer and the in-place Y pass. This is what actually reaches the imagefilter blur scenarios.
	t.Run("evalBlurPasses", func(t *testing.T) {
		for _, sigmas := range [][2]float32{
			{0.4, 0.4}, {1.7, 0.9}, {0.6, 3.5}, {4, 4}, {12.5, 2.25}, {0, 1.4}, {2.5, 0},
		} {
			for _, size := range [][2]int32{{1, 1}, {5, 37}, {37, 5}, {64, 48}} {
				src := imagecore.NewPixels(imagecore.MakeN32Premul(size[0], size[1]))
				for i := range src.Words {
					src.Words[i] = rng.Uint32()
				}
				makeMaker := func(sigma float32) *passMaker {
					if maker := makeGaussianPassMaker(sigma); maker != nil {
						return maker
					}
					return makeThreeBoxPassMaker(sigma)
				}
				makerX, makerY := makeMaker(sigmas[0]), makeMaker(sigmas[1])
				srcBounds := geom.IRect{Right: size[0], Bottom: size[1]}
				dstBounds := srcBounds.Inset(-SigmaToRadius(sigmas[0]), -SigmaToRadius(sigmas[1]))

				gaussianBlurSegmentFn, threeBoxBlurSegmentFn = gaussianBlurSegmentGeneric, threeBoxBlurSegmentGeneric
				want := evalBlurPasses(makerX, makerY, src, srcBounds, dstBounds)
				gaussianBlurSegmentFn, threeBoxBlurSegmentFn = gaussianBlurSegmentSIMD, threeBoxBlurSegmentSIMD
				got := evalBlurPasses(makerX, makerY, src, srcBounds, dstBounds)
				blurSIMDRestoreDispatch()

				if want == nil || got == nil {
					t.Fatalf("sigma=%v size=%v: nil result (want=%v got=%v)", sigmas, size, want == nil, got == nil)
				}
				if !reflect.DeepEqual(got.AsImage().PeekPixels(imagecore.CachingAllow).Words,
					want.AsImage().PeekPixels(imagecore.CachingAllow).Words) {
					t.Fatalf("sigma=%v size=%v: blurred pixels diverged", sigmas, size)
				}
			}
		}
	})
}

// TestBlurEngineSIMDContractionNegativeControl proves the hostile-kernel corpus can actually tell a fused multiply-add
// from an unfused one. It reruns that corpus through a copy of the Gaussian kernel whose only difference is the
// *opposite* contraction spelling from the one this arch's compiler gives the scalar body, and requires it to diverge.
// Without this, "the vector kernel matches the scalar" would be a claim the corpus was too weak to test.
func TestBlurEngineSIMDContractionNegativeControl(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd blur kernels require; dispatch stays on the portable forms")
	}
	if !blurSIMDSwappedContractionAvailable() {
		t.Skip("this CPU cannot execute the opposite contraction spelling, so the control cannot run")
	}
	rng := rand.New(rand.NewPCG(31, 37))
	diverged, total := 0, 0
	for range 256 {
		sigma := float32(0.1) + rng.Float32()*1.8
		want, got := blurSIMDGaussianPair(sigma)
		band := blurSIMDHostileKernel(rng, want.kernel, got.kernel)
		for range 16 {
			s := blurSIMDRandSegment(rng, band)
			s.hasSrc, s.hasDst = true, true
			src, dstA, dstB := blurSIMDBuffers(rng, s)
			gaussianBlurSegmentGeneric(want, s.n, src, s.srcStride, dstA, s.dstStride)
			blurSIMDGaussianSegmentSwapped(got, s.n, src, s.srcStride, dstB, s.dstStride)
			total++
			if !reflect.DeepEqual(dstB, dstA) {
				diverged++
			}
		}
	}
	// The bar is deliberately well above zero: the equivalence suite runs the same generator at the same volume, so a
	// control that only just diverges would mean the positive test could miss a swapped spelling on another seed.
	if diverged*8 < total {
		t.Fatalf("the swapped contraction spelling diverged on only %d of %d hostile segments; the equivalence fuzz "+
			"cannot reliably distinguish the two lowerings and its bit-exactness claim is undertested",
			diverged, total)
	}
	t.Logf("swapped contraction diverged on %d of %d hostile segments", diverged, total)
}

// blurSIMDGaussianSegmentSwapped is gaussianBlurSegmentSIMD's main path with exprMulAdd4 replaced by the spelling this
// arch's compiler does *not* give the scalar body. It exists only for the negative control above; it deliberately skips
// the nil-src, nil-dst and oversized-window handling the real kernel needs, because the control always drives it with
// both buffers present and a maker-built window.
func blurSIMDGaussianSegmentSwapped(p *gaussianPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	window := p.window
	base := p.base
	buf := p.srcBuffer[:window]
	var weightStore [maxGaussianWindow][4]float32
	weights := weightStore[:window]
	for i, k := range p.kernel[:window] {
		archsimd.BroadcastFloat32x4(k).StoreArray(&weights[i])
	}
	inv255 := archsimd.BroadcastFloat32x4(1.0 / 255.0)
	scale255 := archsimd.BroadcastFloat32x4(255)
	half := archsimd.BroadcastFloat32x4(0.5)
	var zero archsimd.Float32x4
	gather := newLowByteGather()
	srcCursor := int32(0)
	dstCursor := int32(0)
	for ; n > 0; n-- {
		leadingEdge := unpackWord(src[srcCursor]).BitsToInt32().ConvertToFloat32().Mul(inv255)
		srcCursor += srcStride
		lead := base - 1
		if lead < 0 {
			lead = window - 1
		}
		leadingEdge.StoreArray(&buf[lead])

		var sum archsimd.Float32x4
		tail := buf[base:]
		tailWeights := weights[:len(tail)]
		for i := range tail {
			sum = blurSIMDExprMulAdd4Swapped(archsimd.LoadFloat32x4Array(&tail[i]), archsimd.LoadFloat32x4Array(&tailWeights[i]), sum)
		}
		headWeights := weights[len(tail):]
		head := buf[:len(headWeights)]
		for i := range head {
			sum = blurSIMDExprMulAdd4Swapped(archsimd.LoadFloat32x4Array(&head[i]), archsimd.LoadFloat32x4Array(&headWeights[i]), sum)
		}
		v := blurSIMDExprMulAdd4Swapped(sum, scale255, half)
		v = scale255.IfElse(v.Greater(scale255), v)
		v = zero.IfElse(v.Less(zero), v)
		dst[dstCursor] = packLowBytes(v.ConvertToInt32().ToBits(), gather)
		dstCursor += dstStride
		base++
		if base >= window {
			base = 0
		}
	}
	p.base = base
}

// TestBlurEngineSIMDWiring locks that on qualifying hardware the goexperiment.simd build's init actually repointed the
// dispatch variables the per-arch preference table elects, so a refactor cannot silently fall back to the portable
// bodies.
func TestBlurEngineSIMDWiring(t *testing.T) {
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd blur kernels require; dispatch stays on the portable forms")
	}
	wantGaussian := reflect.ValueOf(gaussianBlurSegmentGeneric).Pointer()
	if preferSIMDGaussianSegment {
		wantGaussian = reflect.ValueOf(gaussianBlurSegmentSIMD).Pointer()
	}
	if reflect.ValueOf(gaussianBlurSegmentFn).Pointer() != wantGaussian {
		t.Errorf("gaussianBlurSegmentFn does not match preferSIMDGaussianSegment=%v", preferSIMDGaussianSegment)
	}
	wantThreeBox := reflect.ValueOf(threeBoxBlurSegmentGeneric).Pointer()
	if preferSIMDThreeBoxSegment {
		wantThreeBox = reflect.ValueOf(threeBoxBlurSegmentSIMD).Pointer()
	}
	if reflect.ValueOf(threeBoxBlurSegmentFn).Pointer() != wantThreeBox {
		t.Errorf("threeBoxBlurSegmentFn does not match preferSIMDThreeBoxSegment=%v", preferSIMDThreeBoxSegment)
	}
}

// blurSIMDRestoreDispatch puts the dispatch variables back where init left them, for the subtests that swap them to
// drive the whole engine through one lane.
func blurSIMDRestoreDispatch() {
	gaussianBlurSegmentFn = gaussianBlurSegmentGeneric
	threeBoxBlurSegmentFn = threeBoxBlurSegmentGeneric
	if simdKernelsSupported() {
		if preferSIMDGaussianSegment {
			gaussianBlurSegmentFn = gaussianBlurSegmentSIMD
		}
		if preferSIMDThreeBoxSegment {
			threeBoxBlurSegmentFn = threeBoxBlurSegmentSIMD
		}
	}
}
