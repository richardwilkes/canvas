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

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"math/rand/v2"
	"testing"
)

// The kernels are gated on simdKernelsSupported, not on the per-kernel preference constants: an arch that declines a
// kernel for speed still has to prove it computes the same values, since the constant can be flipped back.

func simdTestSkip(t *testing.T) {
	t.Helper()
	if !simdKernelsSupported() {
		t.Skip("CPU lacks the features the simd DSP kernels require")
	}
}

// simdBlockClass fills an 8-bit block with one of the hostile classes: extremes, flat runs, alternations, a single
// spike and uniform noise. Every class is drawn at every size so the vector tails and the byte gathers see them all.
func simdBlockClass(rng *rand.Rand, buf []uint8, w, h, class int) {
	for i := range buf {
		buf[i] = 0
	}
	for y := range h {
		for x := range w {
			var v uint8
			switch class {
			case 0:
				v = 0
			case 1:
				v = 255
			case 2:
				if (x+y)&1 == 0 {
					v = 255
				}
			case 3:
				v = 128
				if x == 2 && y == 1 {
					v = 255
				}
			case 4:
				if x < w/2 {
					v = 255
				}
			default:
				v = uint8(rng.IntN(256))
			}
			buf[y*bps+x] = v
		}
	}
}

// simdCoeffClass fills a 16-coefficient block with one of the hostile classes: zero, the int16 extremes, alternating
// signs, a single spike, small values around the quantizer's zero threshold and uniform noise.
func simdCoeffClass(rng *rand.Rand, blk *[16]int16, class int) {
	for i := range blk {
		switch class {
		case 0:
			blk[i] = 0
		case 1:
			blk[i] = 32767
		case 2:
			blk[i] = -32768
		case 3:
			if i&1 == 0 {
				blk[i] = 2047
			} else {
				blk[i] = -2047
			}
		case 4:
			blk[i] = 0
			if i == 5 {
				blk[i] = -1
			}
		case 5:
			blk[i] = int16(rng.IntN(9) - 4)
		case 6:
			blk[i] = int16(rng.IntN(4097) - 2048)
		case 7:
			// Straddling the zero threshold: no matrix the encoder builds has a zthresh above 255, so this class puts
			// a good share of its coefficients on either side of every one of them.
			blk[i] = int16(rng.IntN(1025) - 512)
		default:
			blk[i] = int16(rng.IntN(65536) - 32768)
		}
	}
}

const simdCoeffClasses = 9

func TestDSPSIMDMatchesScalar(t *testing.T) {
	simdTestSkip(t)
	rng := rand.New(rand.NewPCG(1, 2))

	t.Run("fTransform", func(t *testing.T) {
		src := make([]uint8, yuvSize+8)
		ref := make([]uint8, yuvSize+8)
		var want, got [16]int16
		for trial := range 4096 {
			simdBlockClass(rng, src, 4, 4, trial%6)
			simdBlockClass(rng, ref, 4, 4, (trial/6)%6)
			fTransformGeneric(src, ref, want[:])
			fTransformSIMD(src, ref, got[:])
			if want != got {
				t.Fatalf("trial %d: got %v want %v (src %v ref %v)", trial, got, want, src[:4], ref[:4])
			}
		}
	})

	t.Run("iTransformOne", func(t *testing.T) {
		ref := make([]uint8, yuvSize+8)
		wantDst := make([]uint8, yuvSize+8)
		gotDst := make([]uint8, yuvSize+8)
		var in [16]int16
		fellBack := false
		for trial := range 8192 {
			simdBlockClass(rng, ref, 4, 4, trial%6)
			simdCoeffClass(rng, &in, trial%simdCoeffClasses)
			for i := range wantDst {
				wantDst[i] = 0xA5
				gotDst[i] = 0xA5
			}
			inCopy := in
			iTransformOneGeneric(ref, in[:], wantDst)
			iTransformOneSIMD(ref, inCopy[:], gotDst)
			if !bytes.Equal(wantDst, gotDst) {
				t.Fatalf("trial %d: mismatch for in %v", trial, in)
			}
			if in != inCopy {
				t.Fatalf("trial %d: the kernel modified its input", trial)
			}
			for _, v := range in {
				if v > iTransformSafeBound || v < -iTransformSafeBound {
					fellBack = true
				}
			}
		}
		if !fellBack {
			t.Fatal("no trial exercised the out-of-range fallback")
		}
	})

	// The magnitude guard gets its own walk: magnitudes on either side of the bound and beyond it, in every combination
	// of row signs, so both sides of the branch are taken and the transition between them is exercised. The last
	// magnitude is the int16 minimum, whose absolute value is 32768 — a value no signed 16-bit lane can hold, so a
	// guard written as a signed compare would wave it through (the kernel compares unsigned for exactly this reason).
	// Whether the accepted side can wrap at all is a separate question, and an output comparison is the wrong tool for
	// it; see TestITransformSIMDBoundProof.
	t.Run("iTransformOneBoundary", func(t *testing.T) {
		ref := make([]uint8, yuvSize+8)
		wantDst := make([]uint8, yuvSize+8)
		gotDst := make([]uint8, yuvSize+8)
		for _, class := range []int{0, 1, 5} {
			simdBlockClass(rng, ref, 4, 4, class)
			for _, mag := range []int{
				iTransformSafeBound - 1, iTransformSafeBound, iTransformSafeBound + 1,
				iTransformSafeBound * 2, 32767, 32768,
			} {
				for pattern := range 16 {
					var in [16]int16
					for row := range 4 {
						v := mag
						if pattern&(1<<row) != 0 {
							v = -mag
						}
						if v > 32767 {
							v = -32768 // the magnitude that only fits as the int16 minimum
						}
						for col := range 4 {
							in[row*4+col] = int16(v)
						}
					}
					for i := range wantDst {
						wantDst[i] = 0x5A
						gotDst[i] = 0x5A
					}
					iTransformOneGeneric(ref, in[:], wantDst)
					iTransformOneSIMD(ref, in[:], gotDst)
					if !bytes.Equal(wantDst, gotDst) {
						t.Fatalf("magnitude %d pattern %d: mismatch for in %v", mag, pattern, in)
					}
				}
			}
		}
	})

	t.Run("getSSE", func(t *testing.T) {
		a := make([]uint8, yuvSize+8)
		b := make([]uint8, yuvSize+8)
		for _, size := range [][2]int{{16, 16}, {16, 8}, {4, 4}} {
			for trial := range 2048 {
				simdBlockClass(rng, a, size[0], size[1], trial%6)
				simdBlockClass(rng, b, size[0], size[1], (trial/6)%6)
				want := getSSEGeneric(a, b, size[0], size[1])
				got := getSSESIMD(a, b, size[0], size[1])
				if want != got {
					t.Fatalf("%dx%d trial %d: got %d want %d", size[0], size[1], trial, got, want)
				}
			}
		}
	})

	t.Run("quantizeBlock", func(t *testing.T) {
		for q := range 128 {
			for mType := range 3 {
				m := simdTestMatrix(q, mType)
				for trial := range 72 {
					var in [16]int16
					simdCoeffClass(rng, &in, trial%simdCoeffClasses)
					wantIn, gotIn := in, in
					var wantOut, gotOut [16]int16
					wantNZ := quantizeBlockGeneric(&wantIn, &wantOut, m)
					gotNZ := quantizeBlockSIMD(&gotIn, &gotOut, m)
					if wantIn != gotIn || wantOut != gotOut || wantNZ != gotNZ {
						t.Fatalf("q%d type%d trial %d: in %v/%v out %v/%v nz %d/%d",
							q, mType, trial, gotIn, wantIn, gotOut, wantOut, gotNZ, wantNZ)
					}
				}
			}
		}
	})
}

// simdTestMatrix builds the quantizer matrix the encoder would build for the given base quantizer index and matrix
// type, the same way setupMatrices does.
func simdTestMatrix(q, mType int) *matrix {
	var m matrix
	switch mType {
	case 0:
		m.q[0] = dcTable[q]
		m.q[1] = acTable[q]
	case 1:
		m.q[0] = dcTable[q] * 2
		m.q[1] = acTable2[q]
	default:
		m.q[0] = dcTable[clipInt(q, 0, 117)]
		m.q[1] = acTable[q]
	}
	m.expand(mType)
	return &m
}

// TestITransformSIMDBoundProof checks the arithmetic claim iTransformOneSIMD's 32-bit lanes rest on, rather than
// hoping a randomized case exhibits a wrap. It has to be checked this way: a wrapped product is 65536/8 = 8192 away
// from the true one by the time it reaches the output, which the [0, 255] clamp then swallows in almost every
// arrangement, so an output-level test can pass over a kernel that is silently wrapping.
//
// The claim is that with every input at most iTransformSafeBound in magnitude, no intermediate the kernel multiplies
// can push a product past a signed 32-bit lane. Each of the vertical pass's four outputs is monotone in each of its
// four inputs (mul1 and mul2 are), so the extremes live at the corners of the input box, and enumerating the sixteen
// sign patterns at the bound finds the true maximum.
func TestITransformSIMDBoundProof(t *testing.T) {
	worst := 0
	for pattern := range 16 {
		v := [4]int{}
		for i := range v {
			v[i] = iTransformSafeBound
			if pattern&(1<<i) != 0 {
				v[i] = -iTransformSafeBound
			}
		}
		a := v[0] + v[2]
		b := v[0] - v[2]
		cc := mul2(v[1]) - mul1(v[3])
		d := mul1(v[1]) + mul2(v[3])
		for _, c := range []int{a + d, b + cc, b - cc, a - d} {
			if c < 0 {
				c = -c
			}
			if c > worst {
				worst = c
			}
		}
	}
	// The horizontal pass multiplies two of those intermediates by each constant; the larger constant binds.
	for _, k := range []int{ac3Mul1Const, ac3Mul2Const} {
		if product := int64(worst) * int64(k); product > math.MaxInt32 {
			t.Fatalf("with inputs at %d the intermediate reaches %d, and %d * %d = %d overflows a signed 32-bit lane",
				iTransformSafeBound, worst, worst, k, product)
		}
	}
	// And the sums built from them, which are never multiplied again, must fit as well.
	if sum := int64(2)*int64(worst) + 4 + int64(mul1(worst)) + int64(mul2(worst)); sum > math.MaxInt32 {
		t.Fatalf("the horizontal pass sum reaches %d, past a signed 32-bit lane", sum)
	}
	t.Logf("inputs bounded by %d give intermediates bounded by %d; %d * %d = %d, %.0f%% of a signed 32-bit lane",
		iTransformSafeBound, worst, worst, ac3Mul2Const, worst*ac3Mul2Const,
		100*float64(worst)*float64(ac3Mul2Const)/float64(math.MaxInt32))
}

// TestQuantMatrixSIMDDomain enumerates every quantizer matrix the encoder can build and checks the three facts
// quantizeBlockSIMD's lane widths rest on: the reciprocal and the zero threshold fit in 16 bits, and the widest
// possible product plus bias fits in an unsigned 32-bit lane. The magnitude bound is 32768 (the negated int16
// minimum), which is the largest value the kernel can present to the multiply.
func TestQuantMatrixSIMDDomain(t *testing.T) {
	const maxMagnitude = 32768
	for q := range 128 {
		for mType := range 3 {
			m := simdTestMatrix(q, mType)
			for i := range 16 {
				if m.iq[i] > 0xFFFF {
					t.Fatalf("q%d type%d coeff %d: iq %d does not fit in 16 bits", q, mType, i, m.iq[i])
				}
				if m.zthresh[i] > 0xFFFF {
					t.Fatalf("q%d type%d coeff %d: zthresh %d does not fit in 16 bits", q, mType, i, m.zthresh[i])
				}
				if uint64(maxMagnitude)+uint64(m.sharpen[i]) > 0xFFFF {
					t.Fatalf("q%d type%d coeff %d: magnitude + sharpen %d overflows 16 bits", q, mType, i,
						maxMagnitude+int(m.sharpen[i]))
				}
				product := (uint64(maxMagnitude)+uint64(m.sharpen[i]))*uint64(m.iq[i]) + uint64(m.bias[i])
				if product > 0xFFFFFFFF {
					t.Fatalf("q%d type%d coeff %d: coeff*iq + bias reaches %d, past a 32-bit lane",
						q, mType, i, product)
				}
			}
		}
	}
}

// TestDSPSIMDZigzagGather recomputes the four byte-gather index vectors quantizeBlockSIMD permutes its levels with,
// straight from the zigzag table, so a hand-typed entry cannot drift.
func TestDSPSIMDZigzagGather(t *testing.T) {
	var want [4][16]uint8
	for i := range want {
		for j := range want[i] {
			want[i][j] = 0xFF
		}
	}
	for n, j := range zigzag {
		half := n / 8       // which output vector the level lands in
		src := int(j) / 8   // which input vector it comes from
		lane := int(j) % 8  // its lane there
		out := (n % 8) * 2  // its byte position in the output vector
		table := half*2 + 0 // the gather that reads the "own half" source
		if src != half {
			table = half*2 + 1
		}
		want[table][out] = uint8(lane * 2)
		want[table][out+1] = uint8(lane*2 + 1)
	}
	if want != zigzagGatherIdx {
		t.Fatalf("zigzagGatherIdx is %v, want %v", zigzagGatherIdx, want)
	}
}

// TestDSPSIMDWiring locks that init actually enabled every kernel whose per-arch preference constant elects it, so a
// refactor cannot silently fall back to the portable forms.
func TestDSPSIMDWiring(t *testing.T) {
	simdTestSkip(t)
	for _, tc := range []struct {
		name   string
		wired  bool
		prefer bool
	}{
		{"fTransform", useSIMDFTransform, preferSIMDFTransform},
		{"iTransformOne", useSIMDITransformOne, preferSIMDITransformOne},
		{"getSSE", useSIMDGetSSE, preferSIMDGetSSE},
		{"quantizeBlock", useSIMDQuantizeBlock, preferSIMDQuantizeBlock},
	} {
		if tc.prefer != tc.wired {
			t.Fatalf("%s: dispatch is %v but this arch prefers %v", tc.name, tc.wired, tc.prefer)
		}
	}
}

// TestDSPSIMDEncodeMatchesScalar is the decisive gate: it encodes a spread of content down both lanes — every kernel
// forced to its portable form, then every kernel on the simd form — and requires the two bitstreams to be identical
// byte for byte. Nothing weaker will do, because a VP8 encoder feeds every rounding decision back into the next
// macroblock through the reconstruction loop, so a single differing lane would fan out across the frame.
func TestDSPSIMDEncodeMatchesScalar(t *testing.T) {
	simdTestSkip(t)
	images := map[string]image.Image{
		"gradient":  gradientImage(97, 53),
		"solid":     solidImage(64, 48, color.NRGBA{R: 120, G: 130, B: 140, A: 255}),
		"photo":     photoLikeImage(160, 112, 5),
		"edges":     sharpEdgesImage(96, 96),
		"noise":     noiseImage(80, 64, 11),
		"tiny":      gradientImage(1, 1),
		"thin":      photoLikeImage(15, 100, 3),
		"odd-edges": sharpEdgesImage(33, 17),
	}
	for name, img := range images {
		for _, quality := range []float32{0, 4, 25, 75, 95, 100} {
			scalar := simdEncodeWith(t, img, quality, false)
			vector := simdEncodeWith(t, img, quality, true)
			if !bytes.Equal(scalar, vector) {
				t.Fatalf("%s q%v: the simd lane emitted a different bitstream (%d bytes vs %d)",
					name, quality, len(vector), len(scalar))
			}
		}
	}
}

// simdEncodeWith encodes img with every dispatch bool forced to the given lane, restoring the dispatch afterwards.
func simdEncodeWith(t *testing.T, img image.Image, quality float32, useSIMD bool) []byte {
	t.Helper()
	savedF, savedI := useSIMDFTransform, useSIMDITransformOne
	savedS, savedQ := useSIMDGetSSE, useSIMDQuantizeBlock
	defer func() {
		useSIMDFTransform, useSIMDITransformOne = savedF, savedI
		useSIMDGetSSE, useSIMDQuantizeBlock = savedS, savedQ
	}()
	useSIMDFTransform, useSIMDITransformOne = useSIMD, useSIMD
	useSIMDGetSSE, useSIMDQuantizeBlock = useSIMD, useSIMD
	data, err := Encode(img, quality)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return data
}
