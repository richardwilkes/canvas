// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build arm64

package raster

import (
	"unsafe"

	"github.com/richardwilkes/canvas/colorcore"
)

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

// clampSpan01 clamps every channel of buf to [0, 1] (the 8888 normalization clamp). Each pixel is one float quad, so
// the whole span runs in the NEON kernel with no tail.
func clampSpan01(buf []colorcore.PMColor4f) {
	if len(buf) == 0 {
		return
	}
	clampQuads(&buf[0].R, len(buf))
}

// storeSpanSrc stores shaded premultiplied floats as 8888 words (the BlendSrc full-coverage lane; toUnorm performs the
// clamp). Four pixels per NEON iteration; the sub-quad tail takes the scalar storeWord path, which computes the
// identical bytes.
func storeSpanSrc(buf []colorcore.PMColor4f, span []uint32) {
	n := len(span)
	if q := n / 4; q > 0 {
		storeSpanQuads(&buf[0].R, &span[0], q)
	}
	for i := n &^ 3; i < n; i++ {
		span[i] = storeWord(pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A})
	}
}

// pmSrcOverRow blends a row of premultiplied src pixels over dst with src-over-opaque-dst semantics: dst = satAdd8(src,
// mulDiv255Round(dst, 255-srcA)). Four pixels per NEON iteration; the sub-quad tail takes the portable SWAR path, which
// computes the identical bytes.
func pmSrcOverRow(dst, src []uint32) {
	n := len(src)
	if q := n / 4; q > 0 {
		pmSrcOverQuads(&dst[0], &src[0], q)
	}
	if t := n &^ 3; t < n {
		pmSrcOverRowGeneric(dst[t:], src[t:n])
	}
}

// blitMaskOpaqueRow blends an opaque solid color through an A8 coverage row (the glyph-mask kernel). Four pixels per
// NEON iteration; the sub-quad tail takes the portable path, which computes the identical bytes.
func blitMaskOpaqueRow(dev []uint32, aa []uint8, pm uint32) {
	n := len(aa)
	if q := n / 4; q > 0 {
		blitMaskOpaqueQuads(&dev[0], &aa[0], pm, q)
	}
	if t := n &^ 3; t < n {
		blitMaskOpaqueRowGeneric(dev[t:n], aa[t:n], pm)
	}
}
