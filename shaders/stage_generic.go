// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build !arm64

package shaders

// The hot span stages dispatch through these fn variables so arm64 can substitute NEON kernels (stage_arm64.go /
// stage_arm64.s) while every other platform runs the portable forms. The NEON kernels are bit-identical to the portable
// stages — the substitution never changes rendered output, only throughput — so the choice of lane is invisible to the
// differential suite.
var (
	seedStageFn                 stageFn = seedStage
	clampX1StageFn              stageFn = clampX1Stage
	matrixTranslateStageFn      stageFn = matrixTranslateStage
	matrixScaleTranslateStageFn stageFn = matrixScaleTranslateStage
	matrixAffineStageFn         stageFn = matrixAffineStage
	gradient2StopStageFn        stageFn = gradient2StopStage
	gradientEvenlyStageFn       stageFn = gradientEvenlyStage
)
