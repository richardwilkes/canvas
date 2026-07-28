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

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

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
	// The ceiling power of 2 above 2^30 is not representable as an int32, so handle that range without searching for
	// it: the shift below would overflow to a negative value (still < value), then to 0, and 0 shifts to 0 forever.
	// Saturating at MaxInt32 keeps the "result >= value" contract; every caller's dimensions are bounded well below
	// this by ValidateSurfaceParams/clip bounds, so no allocation is actually attempted at these sizes.
	const maxPow2 = int32(1) << 30
	if value > maxPow2 {
		if mid := maxPow2 + maxPow2>>1; value <= mid {
			return mid
		}
		return math.MaxInt32
	}
	// Next power of 2 above value (1 < value <= 2^30 here).
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
