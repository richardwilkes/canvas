// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import (
	"unsafe"

	"github.com/richardwilkes/canvas/colorcore"
)

// On arm64 the span dispatch switches to thin wrappers over the NEON kernels in span_arm64.s. They are bit-identical
// to the portable defaults (see the .s file's equivalence notes and TestSpanNEONMatchesScalar), so this substitution
// changes throughput only, never rendered bytes. Under goexperiment.simd, span_simd.go's init runs after this one
// (file-name order) and would repoint the variables at the archsimd kernels — today it declines on arm64, where these
// wrappers are the faster lane (see span_simd_arm64.go).
func init() {
	clampSpan01Fn = clampSpan01NEON
	storeSpanSrcFn = storeSpanSrcNEON
	pmSrcOverRowFn = pmSrcOverRowNEON
	blitMaskOpaqueRowFn = blitMaskOpaqueRowNEON
}

// The NEON span kernels treat a PMColor4f as a 16-byte {R,G,B,A} float quad.
var (
	_ [16 - unsafe.Sizeof(colorcore.PMColor4f{})]byte
	_ [unsafe.Sizeof(colorcore.PMColor4f{}) - 16]byte
)

// Implemented in span_arm64.s.
func clampQuads(p *float32, quads int)
func storeSpanQuads(buf *float32, span *uint32, quads int)
func pmSrcOverQuads(dst, src *uint32, quads int)
func blitMaskOpaqueQuads(dev *uint32, aa *uint8, pm uint32, quads int)

// clampSpan01NEON clamps every channel of buf to [0, 1] (the 8888 normalization clamp). Each pixel is one float quad,
// so the whole span runs in the NEON kernel with no tail.
func clampSpan01NEON(buf []colorcore.PMColor4f) {
	if len(buf) == 0 {
		return
	}
	clampQuads(&buf[0].R, len(buf))
}

// storeSpanSrcNEON stores shaded premultiplied floats as 8888 words (the BlendSrc full-coverage lane; toUnorm performs
// the clamp). Four pixels per NEON iteration; the sub-quad tail takes the scalar storeWord path, which computes the
// identical bytes.
func storeSpanSrcNEON(buf []colorcore.PMColor4f, span []uint32) {
	n := len(span)
	if q := n / 4; q > 0 {
		storeSpanQuads(&buf[0].R, &span[0], q)
	}
	if t := n &^ 3; t < n {
		storeSpanSrcGeneric(buf[t:], span[t:])
	}
}

// pmSrcOverRowNEON blends a row of premultiplied src pixels over dst with src-over-opaque-dst semantics: dst =
// satAdd8(src, mulDiv255Round(dst, 255-srcA)). Four pixels per NEON iteration; the sub-quad tail takes the portable
// SWAR path, which computes the identical bytes.
func pmSrcOverRowNEON(dst, src []uint32) {
	n := len(src)
	if q := n / 4; q > 0 {
		pmSrcOverQuads(&dst[0], &src[0], q)
	}
	if t := n &^ 3; t < n {
		pmSrcOverRowGeneric(dst[t:], src[t:n])
	}
}

// blitMaskOpaqueRowNEON blends an opaque solid color through an A8 coverage row (the glyph-mask kernel). Four pixels
// per NEON iteration; the sub-quad tail takes the portable path, which computes the identical bytes.
func blitMaskOpaqueRowNEON(dev []uint32, aa []uint8, pm uint32) {
	n := len(aa)
	if q := n / 4; q > 0 {
		blitMaskOpaqueQuads(&dev[0], &aa[0], pm, q)
	}
	if t := n &^ 3; t < n {
		blitMaskOpaqueRowGeneric(dev[t:n], aa[t:n], pm)
	}
}
