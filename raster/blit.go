// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

// The hot integer blit rows dispatch through package-level function variables, the same shape the span kernels use (see
// span.go): the portable form of each row lives next to its blitter, under a Generic name and with no build tag, and a
// goexperiment.simd build's init (blit_simd.go) repoints the variable at an archsimd kernel on qualifying hardware.
// These rows have no hand-written assembly lane, so unlike the span kernels every build starts on the portable form.
//
// Every substitute is bit-identical to the portable form (locked by TestBlitSIMDMatchesScalar, whose integer subtests
// enumerate complete domains wherever they are small enough), so the choice of lane changes throughput only, never
// rendered bytes. The Generic forms are also the sub-chunk tail of every vector kernel, which calls them rather than
// duplicating their arithmetic.

// The dispatch signatures.
type (
	// blitFillWordsFn fills a device span with one word.
	blitFillWordsFn func(dst []uint32, v uint32)
	// blitFillBytesFn fills an A8 coverage row with one byte.
	blitFillBytesFn func(dst []uint8, v uint8)
	// blitColorRowFn scales a device span by invA and adds a constant premultiplied color (color32's general lane).
	blitColorRowFn func(dst []uint32, color, invA uint32)
	// blitScaleRowFn blends a source row into a destination row under one loop-invariant 0..256 scale.
	blitScaleRowFn func(dst, src []uint32, scale uint32)
	// blitMaskRowFn blends a constant premultiplied color through an A8 coverage row.
	blitMaskRowFn func(dev []uint32, aa []uint8, pm, srcA uint32)
	// blitLCD16RowFn blends a constant unpremultiplied color through a 565 subpixel coverage row.
	blitLCD16RowFn func(dst []uint32, mask []uint16, srcA, srcR, srcG, srcB uint32)
	// blitLCD16OpaqueRowFn is blitLCD16RowFn's opaque-source specialization; opaqueDst is the color's device word.
	blitLCD16OpaqueRowFn func(dst []uint32, mask []uint16, srcR, srcG, srcB, opaqueDst uint32)
	// blitLCD16SpanRowFn blends a shaded premultiplied span through a 565 subpixel coverage row.
	blitLCD16SpanRowFn func(dst []uint32, mask []uint16, src []uint32)
)

// The dispatch variables. Each defaults to the portable form declared beside the blitter that drives it; blit_simd.go's
// init is the only assignment. premulRow reuses spanRowFn, whose shape it already has.
var (
	fillWordsFn              blitFillWordsFn      = fillWordsGeneric              // blitter_solid.go
	fillBytesFn              blitFillBytesFn      = fillBytesGeneric              // blitter_a8.go
	color32RowFn             blitColorRowFn       = color32RowGeneric             // blitter_solid.go
	blitMaskTranslucentRowFn blitMaskRowFn        = blitMaskTranslucentRowGeneric // blitter_solid.go
	interp256RowFn           blitScaleRowFn       = interp256RowGeneric           // blitter_solid.go
	premulRowFn              spanRowFn            = premulRowGeneric              // blitter_sprite.go
	pmBlendRowFn             blitScaleRowFn       = pmBlendRowGeneric             // blitter_sprite.go
	blitRowLCD16Fn           blitLCD16RowFn       = blitRowLCD16Generic           // blitter_lcd16.go
	blitRowLCD16OpaqueFn     blitLCD16OpaqueRowFn = blitRowLCD16OpaqueGeneric     // blitter_lcd16.go
	blendRowLCD16OpaqueFn    blitLCD16SpanRowFn   = blendRowLCD16OpaqueGeneric    // blitter_lcd16.go
)
