// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build !arm64

package raster

import "github.com/richardwilkes/canvas/colorcore"

// The hot per-span kernels dispatch through these functions so arm64 can substitute NEON forms (span_arm64.go /
// span_arm64.s) while every other platform runs the portable code. The NEON forms are bit-identical (locked by
// TestSpanNEONMatchesScalar), so the choice of lane never changes rendered bytes.

// clampSpan01 clamps every channel of buf to [0, 1] (the 8888 normalization clamp).
func clampSpan01(buf []colorcore.PMColor4f) {
	for i := range buf {
		buf[i].R = clamp01f(buf[i].R)
		buf[i].G = clamp01f(buf[i].G)
		buf[i].B = clamp01f(buf[i].B)
		buf[i].A = clamp01f(buf[i].A)
	}
}

// storeSpanSrc stores shaded premultiplied floats as 8888 words (the BlendSrc full-coverage lane; toUnorm performs the
// clamp).
func storeSpanSrc(buf []colorcore.PMColor4f, span []uint32) {
	for i := range span {
		span[i] = storeWord(pmColor4f{r: buf[i].R, g: buf[i].G, b: buf[i].B, a: buf[i].A})
	}
}

// pmSrcOverRow blends a row of premultiplied src pixels over dst with src-over-opaque-dst semantics: dst = satAdd8(src,
// mulDiv255Round(dst, 255-srcA)).
func pmSrcOverRow(dst, src []uint32) {
	pmSrcOverRowGeneric(dst, src)
}

// blitMaskOpaqueRow blends an opaque solid color through an A8 coverage row (the glyph-mask kernel).
func blitMaskOpaqueRow(dev []uint32, aa []uint8, pm uint32) {
	blitMaskOpaqueRowGeneric(dev, aa, pm)
}
