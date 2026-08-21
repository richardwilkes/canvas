// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build arm64 && !goexperiment.simd

package shaders

import (
	"reflect"
	"testing"
)

// TestStageNEONWiring locks that the arm64 dispatch variables actually point at the NEON wrappers, so a refactor cannot
// silently fall back to the scalar stages. Excluded under goexperiment.simd, where init (stage_simd.go) deliberately
// repoints the variables at the simd kernels — TestStageSIMDWiring locks that build's dispatch instead.
func TestStageNEONWiring(t *testing.T) {
	for name, pair := range map[string][2]stageFn{
		"seed":                 {seedStageFn, seedStageNEON},
		"clampX1":              {clampX1StageFn, clampX1StageNEON},
		"matrixTranslate":      {matrixTranslateStageFn, matrixTranslateStageNEON},
		"matrixScaleTranslate": {matrixScaleTranslateStageFn, matrixScaleTranslateStageNEON},
		"matrixAffine":         {matrixAffineStageFn, matrixAffineStageNEON},
		"gradient2Stop":        {gradient2StopStageFn, gradient2StopStageNEON},
		"gradientEvenly":       {gradientEvenlyStageFn, gradientEvenlyStageNEON},
	} {
		if reflect.ValueOf(pair[0]).Pointer() != reflect.ValueOf(pair[1]).Pointer() {
			t.Fatalf("%s: dispatch fn is not the NEON wrapper", name)
		}
	}
}
