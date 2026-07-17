// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The sRGB transfer function and its inverse now live in colorcore (see colorcore/srgb.go) so imagecore's
// pixel-conversion steps can share them; these delegating names keep this package's internal call sites and tests
// unchanged.

package colorfilter

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/shaders"
)

// srgbTF is the parametric sRGB transfer function.
var srgbTF = colorcore.SRGBTF

// srgbInvTF is the inverse of srgbTF, computed at init.
var srgbInvTF = colorcore.SRGBInvTF

// evalTF evaluates a parametric transfer function with the scalar approximation math.
func evalTF(tf *shaders.TransferFn, x float32) float32 {
	return colorcore.EvalSkcmsTF(tf, x)
}
