// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build !goexperiment.simd || (!arm64 && !amd64)

package vp8enc

// The DSP entry points for builds with no simd kernels available: every one is its portable form, with no dispatch
// state to consult. See the dispatch note at the top of dsp.go for why these are direct calls rather than function
// variables, and dsp_simd.go for the twin of this file.

func fTransform(src, ref []uint8, out []int16) { fTransformGeneric(src, ref, out) }

func iTransformOne(ref []uint8, in []int16, dst []uint8) { iTransformOneGeneric(ref, in, dst) }

func getSSE(a, b []uint8, w, h int) int64 { return getSSEGeneric(a, b, w, h) }

func quantizeBlock(in, out *[16]int16, m *matrix) int { return quantizeBlockGeneric(in, out, m) }
