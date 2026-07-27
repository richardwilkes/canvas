// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The in-loop deblocking filter (RFC 6386 §15), applied to the reconstruction planes after the frame is coded so that
// the encoder's reference matches the decoder output exactly. The filter math follows the spec; the driving loop
// matches golang.org/x/image/vp8's post-frame filter application (the decoder this bitstream is verified against),
// which is itself the spec's.

package vp8enc

func clamp15(x int) int {
	if x < -16 {
		return -16
	}
	if x > 15 {
		return 15
	}
	return x
}

func clamp127(x int) int {
	if x < -128 {
		return -128
	}
	if x > 127 {
		return 127
	}
	return x
}

func clamp255(x int) uint8 {
	if x < 0 {
		return 0
	}
	if x > 255 {
		return 255
	}
	return uint8(x)
}

// filter246 modifies a 2-, 4- or 6-pixel wide or high band along an edge (the normal filter, §15.3).
func filter246(pix []uint8, n, level, ilevel, hlevel, index, iStep, jStep int, fourNotSix bool) {
	for ; n > 0; n, index = n-1, index+iStep {
		p3 := int(pix[index-4*jStep])
		p2 := int(pix[index-3*jStep])
		p1 := int(pix[index-2*jStep])
		p0 := int(pix[index-1*jStep])
		q0 := int(pix[index+0*jStep])
		q1 := int(pix[index+1*jStep])
		q2 := int(pix[index+2*jStep])
		q3 := int(pix[index+3*jStep])
		if absInt(p0-q0)<<1+absInt(p1-q1)>>1 > level {
			continue
		}
		if absInt(p3-p2) > ilevel ||
			absInt(p2-p1) > ilevel ||
			absInt(p1-p0) > ilevel ||
			absInt(q1-q0) > ilevel ||
			absInt(q2-q1) > ilevel ||
			absInt(q3-q2) > ilevel {
			continue
		}
		switch {
		case absInt(p1-p0) > hlevel || absInt(q1-q0) > hlevel:
			// High edge variance: filter 2 pixels.
			a := 3*(q0-p0) + clamp127(p1-q1)
			a1 := clamp15((a + 4) >> 3)
			a2 := clamp15((a + 3) >> 3)
			pix[index-1*jStep] = clamp255(p0 + a2)
			pix[index+0*jStep] = clamp255(q0 - a1)
		case fourNotSix:
			// Filter 4 pixels.
			a := 3 * (q0 - p0)
			a1 := clamp15((a + 4) >> 3)
			a2 := clamp15((a + 3) >> 3)
			a3 := (a1 + 1) >> 1
			pix[index-2*jStep] = clamp255(p1 + a3)
			pix[index-1*jStep] = clamp255(p0 + a2)
			pix[index+0*jStep] = clamp255(q0 - a1)
			pix[index+1*jStep] = clamp255(q1 - a3)
		default:
			// Filter 6 pixels.
			a := clamp127(3*(q0-p0) + clamp127(p1-q1))
			a1 := (27*a + 63) >> 7
			a2 := (18*a + 63) >> 7
			a3 := (9*a + 63) >> 7
			pix[index-3*jStep] = clamp255(p2 + a3)
			pix[index-2*jStep] = clamp255(p1 + a2)
			pix[index-1*jStep] = clamp255(p0 + a1)
			pix[index+0*jStep] = clamp255(q0 - a1)
			pix[index+1*jStep] = clamp255(q1 - a2)
			pix[index+2*jStep] = clamp255(q2 - a3)
		}
	}
}

// filterParams computes the per-macroblock filter thresholds (§15.4). j is 1 when the macroblock's inner edges must be
// filtered.
func (e *encoder) filterParams() (level2, ilevel, hlevel int) {
	level := e.filterLevel
	if level == 0 {
		return 0, 0, 0
	}
	ilevel = level
	if e.filterSharpness > 0 {
		if e.filterSharpness > 4 {
			ilevel >>= 2
		} else {
			ilevel >>= 1
		}
		if x := 9 - e.filterSharpness; ilevel > x {
			ilevel = x
		}
	}
	if ilevel < 1 {
		ilevel = 1
	}
	level2 = 2*level + ilevel
	switch { // keyframe high-edge-variance thresholds
	case level >= 40:
		hlevel = 2
	case level >= 15:
		hlevel = 1
	default:
		hlevel = 0
	}
	return level2, ilevel, hlevel
}

// applyLoopFilter runs the deblocking filter over the reconstruction planes, exactly as the decoder does after
// reconstructing all macroblocks. Only the normal filter (§15.3) is ported: setupFilterStrength hard-codes
// e.filterSimple to false and nothing else assigns it, so putFilterHeader always writes filter_type 0 and the spec's
// simple filter (§15.2) is unreachable.
func (e *encoder) applyLoopFilter() {
	if e.filterLevel == 0 {
		return
	}
	level2, ilevel, hlevel := e.filterParams()
	for mby := 0; mby < e.mbH; mby++ {
		for mbx := 0; mbx < e.mbW; mbx++ {
			mb := &e.mbInfo[mby*e.mbW+mbx]
			e.filterMBNormal(mbx, mby, level2, ilevel, hlevel, mb.typ == 0 || !mb.noCoeffs)
		}
	}
}

func (e *encoder) filterMBNormal(mbx, mby, l, il, hl int, inner bool) {
	yIndex := (mby*e.reconYStride + mbx) * 16
	cIndex := (mby*e.reconUVStride + mbx) * 8
	if mbx > 0 {
		filter246(e.reconY, 16, l+4, il, hl, yIndex, e.reconYStride, 1, false)
		filter246(e.reconU, 8, l+4, il, hl, cIndex, e.reconUVStride, 1, false)
		filter246(e.reconV, 8, l+4, il, hl, cIndex, e.reconUVStride, 1, false)
	}
	if inner {
		filter246(e.reconY, 16, l, il, hl, yIndex+0x4, e.reconYStride, 1, true)
		filter246(e.reconY, 16, l, il, hl, yIndex+0x8, e.reconYStride, 1, true)
		filter246(e.reconY, 16, l, il, hl, yIndex+0xc, e.reconYStride, 1, true)
		filter246(e.reconU, 8, l, il, hl, cIndex+0x4, e.reconUVStride, 1, true)
		filter246(e.reconV, 8, l, il, hl, cIndex+0x4, e.reconUVStride, 1, true)
	}
	if mby > 0 {
		filter246(e.reconY, 16, l+4, il, hl, yIndex, 1, e.reconYStride, false)
		filter246(e.reconU, 8, l+4, il, hl, cIndex, 1, e.reconUVStride, false)
		filter246(e.reconV, 8, l+4, il, hl, cIndex, 1, e.reconUVStride, false)
	}
	if inner {
		filter246(e.reconY, 16, l, il, hl, yIndex+e.reconYStride*0x4, 1, e.reconYStride, true)
		filter246(e.reconY, 16, l, il, hl, yIndex+e.reconYStride*0x8, 1, e.reconYStride, true)
		filter246(e.reconY, 16, l, il, hl, yIndex+e.reconYStride*0xc, 1, e.reconYStride, true)
		filter246(e.reconU, 8, l, il, hl, cIndex+e.reconUVStride*0x4, 1, e.reconUVStride, true)
		filter246(e.reconV, 8, l, il, hl, cIndex+e.reconUVStride*0x4, 1, e.reconUVStride, true)
	}
}
