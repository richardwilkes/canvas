// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import (
	"unsafe"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// The hot span stages dispatch through these fn variables; on arm64 they point at thin wrappers over the NEON kernels
// in stage_arm64.s. The kernels are bit-identical to the portable stages (see the .s file's equivalence notes and
// TestStageNEONMatchesScalar), so this substitution changes throughput only, never rendered bytes.
var (
	seedStageFn                 stageFn = seedStageNEON
	clampX1StageFn              stageFn = clampX1StageNEON
	matrixTranslateStageFn      stageFn = matrixTranslateStageNEON
	matrixScaleTranslateStageFn stageFn = matrixScaleTranslateStageNEON
	matrixAffineStageFn         stageFn = matrixAffineStageNEON
	gradient2StopStageFn        stageFn = gradient2StopStageNEON
	gradientEvenlyStageFn       stageFn = gradientEvenlyStageNEON
)

// The NEON kernels assume these layouts: a gradStop is a 32-byte {factor[4], bias[4]} record gathered as two quads, and
// a Color4f is a 16-byte {R,G,B,A} quad.
var (
	_ [32 - unsafe.Sizeof(gradStop{})]byte
	_ [unsafe.Sizeof(gradStop{}) - 32]byte
	_ [16 - unsafe.Sizeof(colorcore.Color4f{})]byte
	_ [unsafe.Sizeof(colorcore.Color4f{}) - 16]byte
)

// Implemented in stage_arm64.s.
func seed16(r, gn, b, a *[16]float32, fx, fy float32, iotas *[16]float32)
func clampX116(r *[16]float32)
func matrixTranslate16(r, gn *[16]float32, tx, ty float32)
func matrixScaleTranslate16(r, gn *[16]float32, sx, sy, tx, ty float32)
func matrixAffine16(r, gn *[16]float32, m *[9]float32)
func grad2Stop16(r, gn, b, a *[16]float32, cl, factor *[4]float32)
func gradEvenly16(r, gn, b, a *[16]float32, stops *gradStop, maxIdx int64, gapCount float32)

// The wrappers compute the same scalar preamble the portable stages do (seed's coordinate floats, the matrix
// coefficient reads) and hand the 16-lane register file to the NEON kernels. The kernels write all 16 lanes; lanes at
// or beyond z.n are scratch no consumer reads.

func seedStageNEON(z *lanes) {
	seed16(&z.r, &z.g, &z.b, &z.a, float32(uint32(z.dx)), float32(uint32(z.dy))+0.5, &iota05)
}

func clampX1StageNEON(z *lanes) {
	clampX116(&z.r)
}

func matrixTranslateStageNEON(z *lanes) {
	c := z.ctx.(*matrixCtx)
	matrixTranslate16(&z.r, &z.g, c.m[geom.MTransX], c.m[geom.MTransY])
}

func matrixScaleTranslateStageNEON(z *lanes) {
	c := z.ctx.(*matrixCtx)
	matrixScaleTranslate16(&z.r, &z.g,
		c.m[geom.MScaleX], c.m[geom.MScaleY], c.m[geom.MTransX], c.m[geom.MTransY])
}

func matrixAffineStageNEON(z *lanes) {
	c := z.ctx.(*matrixCtx)
	matrixAffine16(&z.r, &z.g, &c.m)
}

func gradient2StopStageNEON(z *lanes) {
	c := z.ctx.(*gradientCtx)
	grad2Stop16(&z.r, &z.g, &z.b, &z.a,
		(*[4]float32)(unsafe.Pointer(&c.cl)), (*[4]float32)(unsafe.Pointer(&c.factor)))
}

func gradientEvenlyStageNEON(z *lanes) {
	c := z.ctx.(*gradientCtx)
	gradEvenly16(&z.r, &z.g, &z.b, &z.a, &c.stops[0], int64(len(c.stops)-1), c.gapCount)
}
