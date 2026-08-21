// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build goexperiment.simd

package filtercore

import "simd/archsimd"

// blurSIMDExprMulAdd4Swapped is the spelling exprMulAdd4 must NOT have on arm64: a separately rounded multiply and add,
// where the compiler contracts the scalar body's "f*m + a" into FMADDS. TestBlurEngineSIMDContractionNegativeControl
// runs the hostile corpus through it and requires a divergence, which is what proves the corpus can tell the two
// lowerings apart.
func blurSIMDExprMulAdd4Swapped(f, m, a archsimd.Float32x4) archsimd.Float32x4 {
	return f.Mul(m).Add(a)
}

// blurSIMDSwappedContractionAvailable reports whether this CPU can execute the swapped spelling. On arm64 both
// spellings are baseline NEON, so the control always runs.
func blurSIMDSwappedContractionAvailable() bool { return true }
