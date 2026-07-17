// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The Porter-Duff coefficient table for the 15 coefficient-based blend modes. filtercore uses it to reason about which
// blends preserve transparent black (FilterResult.Draw) and which child bounds survive a blend (the blend image
// filter's output-bounds analysis in part 2).

package filtercore

import "github.com/richardwilkes/canvas/raster"

// blendCoeff is one Porter-Duff blend coefficient.
type blendCoeff int32

const (
	coeffZero blendCoeff = iota // 0
	coeffOne                    // 1
	coeffSC                     // src color
	coeffISC                    // inverse src color
	coeffDC                     // dst color
	coeffIDC                    // inverse dst color
	coeffSA                     // src alpha
	coeffISA                    // inverse src alpha
	coeffDA                     // dst alpha
	coeffIDA                    // inverse dst alpha
)

// gCoeffs is the Porter-Duff coefficient table, indexed by blend mode through Screen.
var gCoeffs = [...][2]blendCoeff{
	raster.BlendClear:    {coeffZero, coeffZero},
	raster.BlendSrc:      {coeffOne, coeffZero},
	raster.BlendDst:      {coeffZero, coeffOne},
	raster.BlendSrcOver:  {coeffOne, coeffISA},
	raster.BlendDstOver:  {coeffIDA, coeffOne},
	raster.BlendSrcIn:    {coeffDA, coeffZero},
	raster.BlendDstIn:    {coeffZero, coeffSA},
	raster.BlendSrcOut:   {coeffIDA, coeffZero},
	raster.BlendDstOut:   {coeffZero, coeffISA},
	raster.BlendSrcATop:  {coeffDA, coeffISA},
	raster.BlendDstATop:  {coeffIDA, coeffSA},
	raster.BlendXor:      {coeffIDA, coeffISA},
	raster.BlendPlus:     {coeffOne, coeffOne},
	raster.BlendModulate: {coeffZero, coeffSC},
	raster.BlendScreen:   {coeffOne, coeffISC},
}

// blendModeAsCoeff returns the source/destination coefficients for mode; ok is false for the advanced modes not in the
// coefficient table.
func blendModeAsCoeff(mode raster.BlendMode) (src, dst blendCoeff, ok bool) {
	if int(mode) >= len(gCoeffs) {
		return 0, 0, false
	}
	c := gCoeffs[mode]
	return c[0], c[1], true
}
