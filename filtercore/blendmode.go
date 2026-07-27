// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The Porter-Duff destination-coefficient table for the 15 coefficient-based blend modes. filtercore's one consumer is
// blendModeAffectsTransparentBlack (filterresult.go), which asks what a blend does to the dst when the src is
// transparent black, so only the destination half of each mode's coefficient pair is transcribed: the source half
// multiplies a zero source and cannot affect the answer.

package filtercore

import "github.com/richardwilkes/canvas/raster"

// blendCoeff is one Porter-Duff destination coefficient. Only the values reachable as a destination coefficient are
// enumerated (a destination coefficient never references the dst).
type blendCoeff int32

const (
	coeffZero blendCoeff = iota // 0
	coeffOne                    // 1
	coeffSC                     // src color
	coeffISC                    // inverse src color
	coeffSA                     // src alpha
	coeffISA                    // inverse src alpha
)

// gDstCoeffs is the Porter-Duff destination-coefficient table, indexed by blend mode through Screen.
var gDstCoeffs = [...]blendCoeff{
	raster.BlendClear:    coeffZero,
	raster.BlendSrc:      coeffZero,
	raster.BlendDst:      coeffOne,
	raster.BlendSrcOver:  coeffISA,
	raster.BlendDstOver:  coeffOne,
	raster.BlendSrcIn:    coeffZero,
	raster.BlendDstIn:    coeffSA,
	raster.BlendSrcOut:   coeffZero,
	raster.BlendDstOut:   coeffISA,
	raster.BlendSrcATop:  coeffISA,
	raster.BlendDstATop:  coeffSA,
	raster.BlendXor:      coeffISA,
	raster.BlendPlus:     coeffOne,
	raster.BlendModulate: coeffSC,
	raster.BlendScreen:   coeffISC,
}

// blendModeDstCoeff returns the destination coefficient for mode; ok is false for the advanced modes not in the
// coefficient table.
func blendModeDstCoeff(mode raster.BlendMode) (dst blendCoeff, ok bool) {
	if int(mode) >= len(gDstCoeffs) {
		return 0, false
	}
	return gDstCoeffs[mode], true
}
