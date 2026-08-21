// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build arm64 && !goexperiment.simd

package raster

import (
	"reflect"
	"testing"
)

// TestSpanNEONWiring locks that the arm64 dispatch variables actually point at the NEON wrappers, so a refactor cannot
// silently fall back to the portable kernels. Excluded under goexperiment.simd, where init (span_simd.go) deliberately
// repoints the variables at the archsimd kernels — TestSpanSIMDWiring locks that build's dispatch instead.
func TestSpanNEONWiring(t *testing.T) {
	for name, pair := range map[string][2]any{
		"clampSpan01":       {clampSpan01Fn, spanClampFn(clampSpan01NEON)},
		"storeSpanSrc":      {storeSpanSrcFn, spanStoreFn(storeSpanSrcNEON)},
		"pmSrcOverRow":      {pmSrcOverRowFn, spanRowFn(pmSrcOverRowNEON)},
		"blitMaskOpaqueRow": {blitMaskOpaqueRowFn, spanMaskFn(blitMaskOpaqueRowNEON)},
	} {
		if reflect.ValueOf(pair[0]).Pointer() != reflect.ValueOf(pair[1]).Pointer() {
			t.Fatalf("%s: dispatch fn is not the NEON wrapper", name)
		}
	}
}
