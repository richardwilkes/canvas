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

// blurSIMDExprMulAdd4Swapped is the spelling exprMulAdd4 must NOT have on amd64: a single fused multiply-add, where the
// compiler leaves the scalar body's "f*m + a" as a separately rounded MULSS and ADDSS at this module's GOAMD64=v1
// baseline. TestBlurEngineSIMDContractionNegativeControl runs the hostile corpus through it and requires a divergence,
// which is what proves the corpus can tell the two lowerings apart.
func blurSIMDExprMulAdd4Swapped(f, m, a archsimd.Float32x4) archsimd.Float32x4 { return f.MulAdd(m, a) }

// blurSIMDSwappedContractionAvailable reports whether this CPU can execute the swapped spelling. Float32x4.MulAdd is
// VFMADD213PS, so the control needs FMA on top of the AVX2 the kernels themselves gate on — a CPU with AVX2 but no FMA
// (Rosetta 2) runs the kernels and skips the control.
func blurSIMDSwappedContractionAvailable() bool { return archsimd.X86.FMA() }
