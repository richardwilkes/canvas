// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package vp8enc

import (
	"math/rand"
	"testing"
)

// The reference predictors below are independent transcriptions of the DECODER's prediction functions
// (golang.org/x/image/vp8 predfunc.go, the ground truth this encoder's bitstreams are verified against), formulated
// directly on (left, topLeft, top) boundary arrays.

func refAvg2(a, b int) uint8    { return uint8((a + b + 1) / 2) }
func refAvg3(a, b, c int) uint8 { return uint8((a + 2*b + c + 2) / 4) }
func refClip255(v int) uint8    { return clamp255(v) }

// refPred4 computes the 4x4 prediction for the given mode (this encoder's mode numbering) from left[0..3] (p,q,r,s top
// to bottom), topLeft (a) and top[0..7] (b..e plus the top-right f..h overhang, using the decoder's letters).
func refPred4(mode int, left [4]uint8, topLeft uint8, top [8]uint8) (out [4][4]uint8) {
	p, q, r, s := int(left[0]), int(left[1]), int(left[2]), int(left[3])
	a := int(topLeft)
	b, c, d, e := int(top[0]), int(top[1]), int(top[2]), int(top[3])
	f, g, h := int(top[4]), int(top[5]), int(top[6])
	hh := int(top[7])
	switch mode {
	case bDCPred:
		sum := 4 + b + c + d + e + p + q + r + s
		v := uint8(sum / 8)
		for j := range out {
			for i := range out[j] {
				out[j][i] = v
			}
		}
	case bTMPred:
		for j, l := range []int{p, q, r, s} {
			for i, t := range []int{b, c, d, e} {
				out[j][i] = refClip255(l + t - a)
			}
		}
	case bVEPred:
		vals := [4]uint8{refAvg3(a, b, c), refAvg3(b, c, d), refAvg3(c, d, e), refAvg3(d, e, f)}
		for j := range out {
			out[j] = vals
		}
	case bHEPred:
		rows := [4]uint8{refAvg3(a, p, q), refAvg3(p, q, r), refAvg3(q, r, s), refAvg3(r, s, s)}
		for j := range out {
			for i := range out[j] {
				out[j][i] = rows[j]
			}
		}
	case bRDPred:
		srq, rqp, qpa := refAvg3(s, r, q), refAvg3(r, q, p), refAvg3(q, p, a)
		pab, abc, bcd, cde := refAvg3(p, a, b), refAvg3(a, b, c), refAvg3(b, c, d), refAvg3(c, d, e)
		out = [4][4]uint8{
			{pab, abc, bcd, cde},
			{qpa, pab, abc, bcd},
			{rqp, qpa, pab, abc},
			{srq, rqp, qpa, pab},
		}
	case bVRPred:
		ab, bc, cd, de := refAvg2(a, b), refAvg2(b, c), refAvg2(c, d), refAvg2(d, e)
		rqp, qpa, pab := refAvg3(r, q, p), refAvg3(q, p, a), refAvg3(p, a, b)
		abc, bcd, cde := refAvg3(a, b, c), refAvg3(b, c, d), refAvg3(c, d, e)
		out = [4][4]uint8{
			{ab, bc, cd, de},
			{pab, abc, bcd, cde},
			{qpa, ab, bc, cd},
			{rqp, pab, abc, bcd},
		}
	case bLDPred:
		abc, bcd, cde := refAvg3(b, c, d), refAvg3(c, d, e), refAvg3(d, e, f)
		def, efg, fgh := refAvg3(e, f, g), refAvg3(f, g, h), refAvg3(g, h, hh)
		ghh := refAvg3(h, hh, hh)
		out = [4][4]uint8{
			{abc, bcd, cde, def},
			{bcd, cde, def, efg},
			{cde, def, efg, fgh},
			{def, efg, fgh, ghh},
		}
	case bVLPred:
		ab, bc, cd, de := refAvg2(b, c), refAvg2(c, d), refAvg2(d, e), refAvg2(e, f)
		abc, bcd, cde := refAvg3(b, c, d), refAvg3(c, d, e), refAvg3(d, e, f)
		def, efg, fgh := refAvg3(e, f, g), refAvg3(f, g, h), refAvg3(g, h, hh)
		out = [4][4]uint8{
			{ab, bc, cd, de},
			{abc, bcd, cde, def},
			{bc, cd, de, efg},
			{bcd, cde, def, fgh},
		}
	case bHDPred:
		sr, rq, qp, pa := refAvg2(s, r), refAvg2(r, q), refAvg2(q, p), refAvg2(p, a)
		srq, rqp, qpa := refAvg3(s, r, q), refAvg3(r, q, p), refAvg3(q, p, a)
		pab, abc, bcd := refAvg3(p, a, b), refAvg3(a, b, c), refAvg3(b, c, d)
		out = [4][4]uint8{
			{pa, pab, abc, bcd},
			{qp, qpa, pa, pab},
			{rq, rqp, qp, qpa},
			{sr, srq, rq, rqp},
		}
	case bHUPred:
		pq, qr, rs := refAvg2(p, q), refAvg2(q, r), refAvg2(r, s)
		pqr, qrs, rss := refAvg3(p, q, r), refAvg3(q, r, s), refAvg3(r, s, s)
		ss := uint8(s)
		out = [4][4]uint8{
			{pq, pqr, qr, qrs},
			{qr, qrs, rs, rss},
			{rs, rss, ss, ss},
			{ss, ss, ss, ss},
		}
	}
	return out
}

// The decoder's letters for the VL/LD modes shift by one relative to the VE case (their "a" is top[0]). The tables
// above already account for this: LD/VL use b..hh = top[0..7].

func TestPredLuma4AgainstDecoderReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	yuvP := make([]uint8, predSize+8)
	for trial := 0; trial < 200; trial++ {
		// Build the 13-sample boundary window: indexes 0..3 = left bottom-to-top (s,r,q,p), 4 = top-left, 5..12 = top +
		// top-right.
		var window [13]uint8
		for i := range window {
			window[i] = uint8(rng.Intn(256))
		}
		encPredLuma4(yuvP, window[:])

		left := [4]uint8{window[3], window[2], window[1], window[0]} // p,q,r,s
		topLeft := window[4]
		var top [8]uint8
		copy(top[:], window[5:13])
		for mode := 0; mode < numBModes; mode++ {
			want := refPred4(mode, left, topLeft, top)
			for j := 0; j < 4; j++ {
				for i := 0; i < 4; i++ {
					got := yuvP[predI4ModeOffsets[mode]+j*bps+i]
					if got != want[j][i] {
						t.Fatalf("trial %d mode %d (%d,%d): got %d want %d (window %v)",
							trial, mode, i, j, got, want[j][i], window)
					}
				}
			}
		}
	}
}

// refPredNxN computes the DC/TM/VE/HE prediction for an NxN block per the decoder's predFunc8/predFunc16 family.
// left/top may be nil for missing neighbors.
func refPredNxN(mode, n int, left []uint8, topLeft uint8, top []uint8) [][]uint8 {
	out := make([][]uint8, n)
	for j := range out {
		out[j] = make([]uint8, n)
	}
	setAll := func(v uint8) {
		for j := range out {
			for i := range out[j] {
				out[j][i] = v
			}
		}
	}
	switch mode {
	case dcPred:
		switch {
		case left != nil && top != nil:
			sum := uint32(n)
			for i := 0; i < n; i++ {
				sum += uint32(top[i]) + uint32(left[i])
			}
			setAll(uint8(sum / uint32(2*n)))
		case top != nil: // DCTop
			sum := uint32(n / 2)
			for i := 0; i < n; i++ {
				sum += uint32(top[i])
			}
			setAll(uint8(sum / uint32(n)))
		case left != nil: // DCLeft
			sum := uint32(n / 2)
			for i := 0; i < n; i++ {
				sum += uint32(left[i])
			}
			setAll(uint8(sum / uint32(n)))
		default: // DCTopLeft
			setAll(0x80)
		}
	case tmPred:
		switch {
		case left != nil && top != nil:
			for j := 0; j < n; j++ {
				for i := 0; i < n; i++ {
					out[j][i] = refClip255(int(left[j]) + int(top[i]) - int(topLeft))
				}
			}
		case left != nil:
			for j := 0; j < n; j++ {
				for i := 0; i < n; i++ {
					out[j][i] = left[j]
				}
			}
		case top != nil:
			for j := 0; j < n; j++ {
				copy(out[j], top[:n])
			}
		default:
			setAll(129)
		}
	case vPred:
		if top != nil {
			for j := 0; j < n; j++ {
				copy(out[j], top[:n])
			}
		} else {
			setAll(127)
		}
	case hPred:
		if left != nil {
			for j := 0; j < n; j++ {
				for i := 0; i < n; i++ {
					out[j][i] = left[j]
				}
			}
		} else {
			setAll(129)
		}
	}
	return out
}

func TestPredLuma16AgainstDecoderReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	yuvP := make([]uint8, predSize+8)
	for _, hasLeft := range []bool{false, true} {
		for _, hasTop := range []bool{false, true} {
			for trial := 0; trial < 20; trial++ {
				var left, top []uint8
				var topLeft uint8 = 129
				if hasLeft {
					left = make([]uint8, 16)
					for i := range left {
						left[i] = uint8(rng.Intn(256))
					}
					topLeft = uint8(rng.Intn(256))
				}
				if hasTop {
					top = make([]uint8, 16)
					for i := range top {
						top[i] = uint8(rng.Intn(256))
					}
				}
				encPredLuma16(yuvP, left, topLeft, top)
				for mode := 0; mode < numPredModes; mode++ {
					want := refPredNxN(mode, 16, left, topLeft, top)
					for j := 0; j < 16; j++ {
						for i := 0; i < 16; i++ {
							got := yuvP[predI16ModeOffsets[mode]+j*bps+i]
							if got != want[j][i] {
								t.Fatalf("left=%v top=%v mode %d (%d,%d): got %d want %d",
									hasLeft, hasTop, mode, i, j, got, want[j][i])
							}
						}
					}
				}
			}
		}
	}
}

func TestPredChroma8AgainstDecoderReference(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	yuvP := make([]uint8, predSize+8)
	for _, hasLeft := range []bool{false, true} {
		for _, hasTop := range []bool{false, true} {
			for trial := 0; trial < 20; trial++ {
				var left, top []uint8
				var tlU, tlV uint8 = 129, 129
				if hasLeft {
					left = make([]uint8, 16)
					for i := range left {
						left[i] = uint8(rng.Intn(256))
					}
					tlU = uint8(rng.Intn(256))
					tlV = uint8(rng.Intn(256))
				}
				if hasTop {
					top = make([]uint8, 16)
					for i := range top {
						top[i] = uint8(rng.Intn(256))
					}
				}
				encPredChroma8(yuvP, left, tlU, tlV, top)
				for mode := 0; mode < numPredModes; mode++ {
					for half := 0; half < 2; half++ { // 0 = U, 1 = V
						var l, tp []uint8
						tl := tlU
						if half == 1 {
							tl = tlV
						}
						if left != nil {
							l = left[half*8 : half*8+8]
						}
						if top != nil {
							tp = top[half*8 : half*8+8]
						}
						want := refPredNxN(mode, 8, l, tl, tp)
						for j := 0; j < 8; j++ {
							for i := 0; i < 8; i++ {
								got := yuvP[predUVModeOffsets[mode]+half*8+j*bps+i]
								if got != want[j][i] {
									t.Fatalf("left=%v top=%v mode %d half %d (%d,%d): got %d want %d",
										hasLeft, hasTop, mode, half, i, j, got, want[j][i])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestFDCTIDCTConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	src := make([]uint8, 4*bps)
	ref := make([]uint8, 4*bps)
	dst := make([]uint8, 4*bps)
	out := make([]int16, 16)
	for trial := 0; trial < 1000; trial++ {
		for j := 0; j < 4; j++ {
			for i := 0; i < 4; i++ {
				src[j*bps+i] = uint8(rng.Intn(256))
				ref[j*bps+i] = uint8(rng.Intn(256))
			}
		}
		fTransform(src, ref, out)
		iTransformOne(ref, out, dst)
		for j := 0; j < 4; j++ {
			for i := 0; i < 4; i++ {
				d := int(dst[j*bps+i]) - int(src[j*bps+i])
				if d < -1 || d > 1 {
					t.Fatalf("trial %d (%d,%d): recon %d vs src %d exceeds ±1",
						trial, i, j, dst[j*bps+i], src[j*bps+i])
				}
			}
		}
	}
}

func TestWHTRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 1000; trial++ {
		var blocks [16][16]int16
		var want [16]int16
		for n := 0; n < 16; n++ {
			v := int16(rng.Intn(4096) - 2048) // 12-bit signed DC range
			blocks[n][0] = v
			want[n] = v
		}
		var wht [16]int16
		fTransformWHT(&blocks, &wht)
		var out [16][16]int16
		iTransformWHT(&wht, &out)
		for n := 0; n < 16; n++ {
			d := int(out[n][0]) - int(want[n])
			if d < -1 || d > 1 {
				t.Fatalf("trial %d block %d: got %d want %d (±1)", trial, n, out[n][0], want[n])
			}
		}
	}
}

func TestQuantizeBlockDequantConsistency(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var m matrix
	m.q[0] = dcTable[40]
	m.q[1] = acTable[40]
	m.expand(0)
	for trial := 0; trial < 1000; trial++ {
		var in, orig, out [16]int16
		for i := range in {
			in[i] = int16(rng.Intn(4096) - 2048)
			orig[i] = in[i]
		}
		nz := quantizeBlock(&in, &out, &m)
		anyNZ := false
		for n := 0; n < 16; n++ {
			j := zigzag[n]
			if out[n] != 0 {
				anyNZ = true
			}
			// The written-back value must be the dequantized level.
			if int(in[j]) != int(out[n])*int(m.q[j]) {
				t.Fatalf("trial %d coeff %d: dequant %d != level %d * q %d", trial, n, in[j], out[n], m.q[j])
			}
			// The level must be the correctly biased quotient of the (sharpened) magnitude.
			mag := int(orig[j])
			sign := mag < 0
			if sign {
				mag = -mag
			}
			mag += int(m.sharpen[j])
			wantLevel := quantDiv(uint32(mag), m.iq[j], m.bias[j])
			if wantLevel > maxLevel {
				wantLevel = maxLevel
			}
			if uint32(mag) <= m.zthresh[j] {
				wantLevel = 0
			}
			if sign {
				wantLevel = -wantLevel
			}
			if int(out[n]) != wantLevel {
				t.Fatalf("trial %d coeff %d: level %d want %d", trial, n, out[n], wantLevel)
			}
		}
		if (nz == 1) != anyNZ {
			t.Fatalf("trial %d: nz flag %d but anyNZ %v", trial, nz, anyNZ)
		}
	}
}

// TestQuantTableConsistency proves acTable2 is exactly the decoder-side Y2-AC expansion (RFC 6386 dequant_init: max(8,
// ac*155/100)), so the encoder's reconstruction uses the very values a decoder will use.
func TestQuantTableConsistency(t *testing.T) {
	for i := 0; i < 128; i++ {
		want := int(acTable[i]) * 155 / 100
		if want < 8 {
			want = 8
		}
		if int(acTable2[i]) != want {
			t.Fatalf("acTable2[%d] = %d, want %d", i, acTable2[i], want)
		}
	}
}
