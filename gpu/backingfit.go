// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// BackingFit describes whether a backing store must match the requested dimensions exactly or may be larger, plus the
// approx-size rounding used for loose fits.

package gpu

import "github.com/richardwilkes/canvas/geom"

// BackingFit describes whether a backing store must match the requested dimensions exactly.
type BackingFit bool

// BackingFit values.
const (
	// BackingFitApprox allows the backing store to be larger than strictly necessary (rounded up by GetApproxSize) so
	// it can be recycled for other approx-fit requests.
	BackingFitApprox BackingFit = false
	// BackingFitExact requires the backing store to match the requested dimensions exactly.
	BackingFitExact BackingFit = true
)

// approxSizeAdjust rounds value up to a larger power of 2, or, above magicTol, to the midpoint between the floor and
// ceiling powers of 2 (a looser fit that still recycles well).
func approxSizeAdjust(value int32) int32 {
	const minApproxSize = 16
	const magicTol = 1024

	if value < minApproxSize {
		value = minApproxSize
	}
	if value&(value-1) == 0 { // already a power of 2
		return value
	}
	// Next power of 2 above value (value > 1 here).
	ceilPow2 := int32(1)
	for ceilPow2 < value {
		ceilPow2 <<= 1
	}
	if value <= magicTol {
		return ceilPow2
	}
	floorPow2 := ceilPow2 >> 1
	mid := floorPow2 + (floorPow2 >> 1)
	if value <= mid {
		return mid
	}
	return ceilPow2
}

// GetApproxSize maps dimensions to larger powers of 2 (or, above a tolerance, the midpoints between powers of 2) so
// approx-fit backing stores can be recycled across requests.
func GetApproxSize(size geom.ISize) geom.ISize {
	return geom.ISize{Width: approxSizeAdjust(size.Width), Height: approxSizeAdjust(size.Height)}
}
