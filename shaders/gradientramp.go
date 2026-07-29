// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The CPU-stage evaluation lane for the GPU textured colorizer: this bakes a gradient LUT by running seed_shader →
// Scale(1/width, 1) → the gradient fill stages → the interpolated-to-dst conversion. It evaluates the same stages the
// CPU renderer uses — not re-derived interpolation math — so the GPU LUT and a CPU render of the same gradient are
// self-consistent by construction. The interpolated-to-dst conversion is exactly the final premultiplication, skipped
// when every stop is opaque (the same reduction appendBaseStages makes).

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// EvalGradientRamp evaluates the gradient color fill stages across a len(dst)×1 ramp, writing one color per texel
// sampled at the texel center: dst[i] holds the gradient color at t = (i + 0.5) / len(dst), as the seed_shader +
// Scale(1/width, 1) matrix produces. colors are the stop-fixed interpolation-space stop colors; positions must be the
// explicit stop positions when the stops are not evenly spaced (nil selects the evenly-spaced fill stage; the GPU
// caller always passes explicit positions, forcing the bake to take the searched stage). premul applies the final
// interpolated-to-dst premultiplication — pass it as !allOpaque so opaque ramps skip the no-op, matching both the
// interpolated-to-dst stage and the analytic colorizers' color-transform lane. colors must hold at least one stop, and
// positions, when non-nil, must be at least as long as colors; a single stop bakes that color across the whole ramp.
func EvalGradientRamp(colors []colorcore.Color4f, positions []float32, premul bool, dst []colorcore.PMColor4f) {
	p := borrowPipeline()
	defer RecyclePipeline(p)
	p.appendSeed()
	m := geom.ScaleMatrix(1/float32(len(dst)), 1)
	p.appendMatrix(&m)
	g := gradientBase{colors: colors, positions: positions}
	g.appendFillStages(p)
	if premul {
		p.AppendPremul()
	}
	p.ShadeSpan(0, 0, dst)
}
