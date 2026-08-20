// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import "github.com/richardwilkes/canvas/colorcore"

// The hot per-span kernels dispatch through package-level function variables (clampSpan01Fn, storeSpanSrcFn,
// pmSrcOverRowFn, blitMaskOpaqueRowFn), so a build can substitute a vector form for the portable one: arm64 wires the
// NEON kernels in span_arm64.go/span_arm64.s, and a goexperiment.simd build's init (span_simd.go) repoints the
// variables at the archsimd kernels on qualifying hardware. Every substitute is bit-identical to the portable form
// (locked by TestSpanNEONMatchesScalar and TestSpanSIMDMatchesScalar), so the choice of lane changes throughput only,
// never rendered bytes.
//
// The portable forms live here, under Generic names and with no build tag, so every build compiles them: they are both
// the default dispatch on platforms without a vector lane and the sub-chunk tail of every vector kernel, which calls
// them rather than duplicating their arithmetic. (pmSrcOverRowGeneric and blitMaskOpaqueRowGeneric live next to their
// blitters, in blitter_sprite.go and blitter_solid.go, for the same reason.)

// The dispatch signatures. Each build's span_*.go declares one variable of each type (see span_generic.go and
// span_arm64.go), and the goexperiment.simd init assigns over them.
type (
	spanClampFn func(buf []colorcore.PMColor4f)
	spanStoreFn func(buf []colorcore.PMColor4f, span []uint32)
	spanRowFn   func(dst, src []uint32)
	spanMaskFn  func(dev []uint32, aa []uint8, pm uint32)
)

// clampSpan01Generic clamps every channel of buf to [0, 1] (the 8888 normalization clamp).
func clampSpan01Generic(buf []colorcore.PMColor4f) {
	for i := range buf {
		buf[i].R = clamp01f(buf[i].R)
		buf[i].G = clamp01f(buf[i].G)
		buf[i].B = clamp01f(buf[i].B)
		buf[i].A = clamp01f(buf[i].A)
	}
}

// storeSpanSrcGeneric stores shaded premultiplied floats as 8888 words (the BlendSrc full-coverage lane; toUnorm
// performs the clamp). It writes one word per element of span and reads the matching prefix of buf.
func storeSpanSrcGeneric(buf []colorcore.PMColor4f, span []uint32) {
	for i := range span {
		span[i] = storeWord(pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A})
	}
}
