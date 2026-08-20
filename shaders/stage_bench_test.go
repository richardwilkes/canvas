// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import "testing"

// The stage benchmarks drive the dispatch variables, so one binary measures whichever kernel the build wired in:
// scalar, NEON, or (under goexperiment.simd) the simd kernels. Compare builds with benchstat.

func BenchmarkMatrix4x5Stage(b *testing.B) {
	var m [20]float32
	for i := range m {
		m[i] = float32(i)*0.1 - 0.7
	}
	var z lanes
	for i := range stride {
		z.r[i] = float32(i) * 0.05
		z.g[i] = 1 - float32(i)*0.03
		z.b[i] = float32(i&3) * 0.25
		z.a[i] = 1
	}
	z.n = stride
	z.ctx = &m
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		matrix4x5StageFn(&z)
	}
}
