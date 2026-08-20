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

// Off arm64 the span dispatch starts on the portable kernels (span.go, blitter_sprite.go, blitter_solid.go). An amd64
// goexperiment.simd build's init (span_simd.go) repoints them at the archsimd kernels when the CPU qualifies; every
// other build runs these for the whole span.
var (
	clampSpan01Fn       spanClampFn = clampSpan01Generic
	storeSpanSrcFn      spanStoreFn = storeSpanSrcGeneric
	pmSrcOverRowFn      spanRowFn   = pmSrcOverRowGeneric
	blitMaskOpaqueRowFn spanMaskFn  = blitMaskOpaqueRowGeneric
)
