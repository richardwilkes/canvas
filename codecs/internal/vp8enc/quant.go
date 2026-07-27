// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Quality → quantizer mapping and per-segment quantization parameters, restricted to the single-segment configuration
// this encoder uses, plus the loop-filter strength setup.

package vp8enc

import "math"

// Encoder configuration constants, matching the reference encoder's defaults for the knobs this encoder keeps fixed:
// sns_strength 50, filter_strength 60, filter_sharpness 0, filter_type 1 (the normal, non-simple loop filter), method
// 3-equivalent single-pass RD (no trellis), one segment, one coefficient partition.
const (
	snsStrength      = 50
	filterStrength   = 60
	fstrengthCutoff  = 2   // filter strengths below this have close to no visual effect
	rdDistoMult      = 256 // distortion multiplier (the effective lambda balance)
	flatnessLimitI16 = 0   // I16 mode (special case)
	flatnessLimitI4  = 3   // I4 mode
	flatnessLimitUV  = 2   // UV mode
	flatnessPenalty  = 140
	// maxI4HeaderBits caps the per-macroblock intra4 mode-signaling budget (the default with an unlimited partition).
	maxI4HeaderBits = 256 * 16 * 16
)

// segmentInfo holds the quantization matrices and rate-distortion lambdas of the (single) segment.
type segmentInfo struct {
	y1, y2, uv matrix
	quant      int // base quantizer index in [0, 127]
	fstrength  int // filter strength in [0, 63]
	maxEdge    int // max edge delta, for filter strength adjustment
	minDisto   int // minimum distortion required to trigger max-edge tracking
	lambdaI16  int64
	lambdaI4   int64
	lambdaUV   int64
	lambdaMode int64
	i4Penalty  int64 // unused by the RD path; kept for parity with the distortion-only path
}

// qualityToCompression maps the user quality (0..1) to a compression factor: jpeg-like behavior, with file size
// roughly scaling as quantizer^3.
func qualityToCompression(c float64) float64 {
	var linearC float64
	if c < 0.75 {
		linearC = c * (2. / 3.)
	} else {
		linearC = 2.*c - 1.
	}
	return math.Pow(linearC, 1./3.)
}

// clampQuality clips q to [0, 100] (NaN maps to 0).
func clampQuality(q float32) float64 {
	if !(q >= 0) { // also catches NaN
		return 0
	}
	if q > 100 {
		return 100
	}
	return float64(q)
}

// setSegmentParams computes the quantizer indexes, matrices, lambdas and filter parameters for the given quality. With
// a single segment the susceptibility modulation reduces to the base compression factor (alpha_ = 0).
func (e *encoder) setSegmentParams(quality float32) {
	q := clampQuality(quality)
	cBase := qualityToCompression(q / 100.)
	e.dqm.quant = clipInt(int(127.*(1.-cBase)), 0, 127)
	e.baseQuant = e.dqm.quant

	// uv_alpha_ susceptibility is not computed (no analysis pass), so the UV AC delta is always zero (the
	// neutral-susceptibility behavior) while the DC boost below still applies.
	dqUVDC := -4 * snsStrength / 100
	dqUVDC = clipInt(dqUVDC, -15, 15) // 4-bit signed maximum

	e.dqY1DC = 0
	e.dqY2DC = 0
	e.dqY2AC = 0
	e.dqUVDC = dqUVDC
	e.dqUVAC = 0

	e.setupFilterStrength()
	e.setupMatrices()
}

func clipInt(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

// setupMatrices fills the quantization matrices and RD lambdas (tlambda disabled, as it is below method 4).
func (e *encoder) setupMatrices() {
	m := &e.dqm
	q := m.quant
	m.y1.q[0] = dcTable[clipInt(q+e.dqY1DC, 0, 127)]
	m.y1.q[1] = acTable[clipInt(q, 0, 127)]

	m.y2.q[0] = dcTable[clipInt(q+e.dqY2DC, 0, 127)] * 2
	m.y2.q[1] = acTable2[clipInt(q+e.dqY2AC, 0, 127)]

	m.uv.q[0] = dcTable[clipInt(q+e.dqUVDC, 0, 117)]
	m.uv.q[1] = acTable[clipInt(q+e.dqUVAC, 0, 127)]

	qI4 := int64(m.y1.expand(0))
	qI16 := int64(m.y2.expand(1))
	qUV := int64(m.uv.expand(2))

	m.lambdaI4 = max64(1, (3*qI4*qI4)>>7)
	m.lambdaI16 = max64(1, 3*qI16*qI16)
	m.lambdaUV = max64(1, (3*qUV*qUV)>>6)
	m.lambdaMode = max64(1, (1*qI4*qI4)>>7)
	m.i4Penalty = 1000 * qI4 * qI4

	m.minDisto = 20 * int(m.y1.q[0]) // quantization-aware minimum distortion
	m.maxEdge = 0
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// filterStrengthFromDelta returns the minimum filtering strength needed to smooth an edge step of the given delta at
// sharpness 0 (the sharpness-0 row of the level table is the identity, so this reduces to a clamp).
func filterStrengthFromDelta(delta int) int {
	if delta >= 64 {
		return 63
	}
	return delta
}

// setupFilterStrength derives the loop-filter strength from the quantizer (single segment, beta 0).
func (e *encoder) setupFilterStrength() {
	level0 := 5 * filterStrength
	// The filtering strength focuses on the quantization of AC coefficients.
	qstep := int(acTable[clipInt(e.dqm.quant, 0, 127)]) >> 2
	baseStrength := filterStrengthFromDelta(qstep)
	f := baseStrength * level0 / (256 + 0) // beta_ == 0: neutral segment complexity
	switch {
	case f < fstrengthCutoff:
		e.dqm.fstrength = 0
	case f > 63:
		e.dqm.fstrength = 63
	default:
		e.dqm.fstrength = f
	}
	e.filterLevel = e.dqm.fstrength
	e.filterSharpness = 0
	// Always the normal filter: this is the only assignment to filterSimple, and applyLoopFilter ports §15.3 alone.
	e.filterSimple = false
}

// adjustFilterStrength raises the filter strength when blocky macroblocks (only DCs non-zero, high distortion) were
// recorded during coding.
func (e *encoder) adjustFilterStrength() {
	if filterStrength <= 0 {
		return
	}
	dqm := &e.dqm
	// The '>> 3' accounts for some inverse WHT scaling.
	delta := (dqm.maxEdge * int(dqm.y2.q[1])) >> 3
	level := filterStrengthFromDelta(delta)
	if level > dqm.fstrength {
		dqm.fstrength = level
	}
	e.filterLevel = dqm.fstrength
}

// storeMaxDelta looks at the first few AC coefficients of the Y2 block to estimate the average delta between sub-4x4
// blocks.
func (dqm *segmentInfo) storeMaxDelta(dcs *[16]int16) {
	v0 := absInt(int(dcs[1]))
	v1 := absInt(int(dcs[2]))
	v2 := absInt(int(dcs[4]))
	maxV := v1
	if v0 > maxV {
		maxV = v0
	}
	if v2 > maxV {
		maxV = v2
	}
	if maxV > dqm.maxEdge {
		dqm.maxEdge = maxV
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
