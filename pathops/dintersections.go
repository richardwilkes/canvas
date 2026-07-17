// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The sorted set of intersection results (t on each curve plus the point) with the coincidence bookkeeping. This file
// carries the storage, the sorted insert/remove machinery, and the coincidence accessors; the actual line/line,
// horizontal, and vertical intersection algorithms live in dlineintersect.go. The curve-specific entry points
// (quad/conic/cubic) arrive with their geometry elsewhere. The result arrays hold up to 13 entries, the maximum
// required across all curve/curve intersection cases this package handles.

package pathops

import "math"

// intersections accumulates the intersection results between two curves.
type intersections struct {
	pts        [13]dPoint     // the intersection points
	ts         [2][13]float64 // the t on each of the two curves
	used       int            // number of populated entries
	max        int            // capacity for this intersection kind
	isCoin     [2]uint16      // per-curve coincident-t bitfields
	nearlySame [2]bool        // true if an end pair nearly matches
	allowNear  bool           // whether near (not just exact) matches are accepted
	swap       bool           // whether curve one/two are swapped in insertSwap
}

// newIntersections returns a zeroed intersections with reset() applied; the caller must set max before inserting.
func newIntersections() *intersections {
	in := &intersections{}
	in.reset()
	in.max = 0 // require that the caller set the max
	return in
}

// reset clears the used count and coincidence bits, leaving swap and max alone.
func (in *intersections) reset() {
	in.allowNear = true
	in.used = 0
	in.isCoin[0] = 0
	in.isCoin[1] = 0
}

// setAllowNear controls whether near (not just exact) matches are accepted.
func (in *intersections) setAllowNear(nearAllowed bool) { in.allowNear = nearAllowed }

// usedCount returns the number of populated intersection entries.
func (in *intersections) usedCount() int { return in.used }

// tVal returns the t on curve row (0 or 1) at the given intersection index.
func (in *intersections) tVal(row, index int) float64 { return in.ts[row][index] }

// pt returns the intersection point at index.
func (in *intersections) pt(index int) dPoint { return in.pts[index] }

// nearlySameAt reports whether the endpoint pair at index (0 or 1) nearly matches without being exactly equal.
func (in *intersections) nearlySameAt(index int) bool { return in.nearlySame[index] }

// setMax sets the maximum number of entries this instance may hold.
func (in *intersections) setMax(maxEntries int) { in.max = maxEntries }

// flip replaces every curve-two t with 1-t.
func (in *intersections) flip() {
	for index := 0; index < in.used; index++ {
		in.ts[1][index] = 1 - in.ts[1][index]
	}
}

// hasT reports whether curve one already has an endpoint t (0 or 1) recorded.
func (in *intersections) hasT(t float64) bool {
	if in.used <= 0 {
		return false
	}
	if t == 0 {
		return in.ts[0][0] == 0
	}
	return in.ts[0][in.used-1] == 1
}

// hasOppT reports whether curve two already has t (0 or 1) recorded at an end.
func (in *intersections) hasOppT(t float64) bool {
	if in.used <= 0 {
		return false
	}
	return in.ts[1][0] == t || in.ts[1][in.used-1] == t
}

// isCoincident reports whether the entry at index is flagged as part of a coincident run.
func (in *intersections) isCoincident(index int) bool {
	return in.isCoin[0]&(uint16(1)<<uint(index)) != 0
}

// setCoincident flags the entry at index as part of a coincident run on both curves.
func (in *intersections) setCoincident(index int) {
	bit := uint16(1) << uint(index)
	in.isCoin[0] |= bit
	in.isCoin[1] |= bit
}

// insert places (one, two, pt) into the t-sorted result set, dropping exact and near-duplicate entries and maintaining
// the coincidence bitfields. Returns the insertion index, or -1 if the entry was dropped.
func (in *intersections) insert(one, two float64, pt dPoint) int {
	if in.isCoin[0] == 3 && between(in.ts[0][0], one, in.ts[0][1]) {
		// For now, don't allow a mix of coincident and non-coincident intersections.
		return -1
	}
	var index int
	for index = 0; index < in.used; index++ {
		oldOne := in.ts[0][index]
		oldTwo := in.ts[1][index]
		if one == oldOne && two == oldTwo {
			return -1
		}
		if moreRoughlyEqual(oldOne, one) && moreRoughlyEqual(oldTwo, two) {
			if (!preciselyZero(one) || preciselyZero(oldOne)) &&
				(!preciselyEqual(one, 1) || preciselyEqual(oldOne, 1)) &&
				(!preciselyZero(two) || preciselyZero(oldTwo)) &&
				(!preciselyEqual(two, 1) || preciselyEqual(oldTwo, 1)) {
				return -1
			}
			// remove this and reinsert below in case replacing would make the list unsorted
			remaining := in.used - index - 1
			copy(in.pts[index:index+remaining], in.pts[index+1:index+1+remaining])
			copy(in.ts[0][index:index+remaining], in.ts[0][index+1:index+1+remaining])
			copy(in.ts[1][index:index+remaining], in.ts[1][index+1:index+1+remaining])
			clearMask := ^((uint16(1) << uint(index)) - 1)
			in.isCoin[0] -= (in.isCoin[0] >> 1) & clearMask
			in.isCoin[1] -= (in.isCoin[1] >> 1) & clearMask
			in.used--
			break
		}
	}
	for index = 0; index < in.used; index++ {
		if in.ts[0][index] > one {
			break
		}
	}
	if in.used >= in.max {
		// This error, if it were to be handled properly, must be propagated to the caller as a failure; here it simply
		// collapses the result to empty.
		in.used = 0
		return 0
	}
	remaining := in.used - index
	if remaining > 0 {
		copy(in.pts[index+1:index+1+remaining], in.pts[index:index+remaining])
		copy(in.ts[0][index+1:index+1+remaining], in.ts[0][index:index+remaining])
		copy(in.ts[1][index+1:index+1+remaining], in.ts[1][index:index+remaining])
		clearMask := ^((uint16(1) << uint(index)) - 1)
		in.isCoin[0] += in.isCoin[0] & clearMask
		in.isCoin[1] += in.isCoin[1] & clearMask
	}
	in.pts[index] = pt
	if one < 0 || one > 1 {
		return -1
	}
	if two < 0 || two > 1 {
		return -1
	}
	in.ts[0][index] = one
	in.ts[1][index] = two
	in.used++
	return index
}

// insertNear records an intersection whose two endpoints are near but not exactly equal.
func (in *intersections) insertNear(one, two float64, pt1 dPoint) {
	slot := 0
	if one != 0 {
		slot = 1
	}
	in.nearlySame[slot] = true
	in.insert(one, two, pt1)
}

// insertSwap inserts (one, two) swapped when the swap flag is set.
func (in *intersections) insertSwap(one, two float64, pt dPoint) int {
	if in.swap {
		return in.insert(two, one, pt)
	}
	return in.insert(one, two, pt)
}

// insertCoincident inserts (swap-aware) and marks the resulting entry coincident. Returns the insertion index or -1.
func (in *intersections) insertCoincident(one, two float64, pt dPoint) int {
	index := in.insertSwap(one, two, pt)
	if index >= 0 {
		in.setCoincident(index)
	}
	return index
}

// clearCoincidence drops the coincident flag at index.
func (in *intersections) clearCoincidence(index int) {
	bit := ^(uint16(1) << uint(index))
	in.isCoin[0] &= bit
	in.isCoin[1] &= bit
}

// merge resets and records a single entry combining a's t on curve one and b's t on curve one (used by the line/line
// convergence lane of the curve solver, which computes each end against the opposite curve separately).
func (in *intersections) merge(a *intersections, aIndex int, b *intersections, bIndex int) {
	in.reset()
	in.ts[0][0] = a.ts[0][aIndex]
	in.ts[1][0] = b.ts[0][bIndex]
	in.pts[0] = a.pts[aIndex]
	in.used = 1
}

// coincidentUsedCount returns the number of entries flagged coincident on curve one.
func (in *intersections) coincidentUsedCount() int {
	if in.isCoin[0] == 0 {
		return 0
	}
	count := 0
	for index := 0; index < in.used; index++ {
		if in.isCoin[0]&(uint16(1)<<uint(index)) != 0 {
			count++
		}
	}
	return count
}

// closestTo returns the index of the intersection whose t on curve one lies in [rangeStart, rangeEnd] and whose point
// is closest to testPt, plus that squared distance. Returns (-1, math.MaxFloat32) when no intersection qualifies.
func (in *intersections) closestTo(rangeStart, rangeEnd float64, testPt dPoint) (closestIdx int, distSq float64) {
	closest := -1
	closestDist := float64(math.MaxFloat32) // SK_ScalarMax
	for index := 0; index < in.used; index++ {
		if !between(rangeStart, in.ts[0][index], rangeEnd) {
			continue
		}
		dist := testPt.distanceSquared(in.pts[index])
		if closestDist > dist {
			closestDist = dist
			closest = index
		}
	}
	return closest, closestDist
}

// mostOutside returns, among intersections whose t on curve one lies in [rangeStart, rangeEnd], the index whose point
// is furthest counterclockwise from origin. Returns -1 when none qualify.
func (in *intersections) mostOutside(rangeStart, rangeEnd float64, origin dPoint) int {
	result := -1
	for index := 0; index < in.used; index++ {
		if !between(rangeStart, in.ts[0][index], rangeEnd) {
			continue
		}
		if result < 0 {
			result = index
			continue
		}
		best := in.pts[result].sub(origin)
		test := in.pts[index].sub(origin)
		if test.crossCheck(best) < 0 {
			result = index
		}
	}
	return result
}

// removeOne deletes the entry at index, shifting later entries down and updating the coincidence bitfields.
func (in *intersections) removeOne(index int) {
	in.used--
	remaining := in.used - index
	if remaining <= 0 {
		return
	}
	copy(in.pts[index:index+remaining], in.pts[index+1:index+1+remaining])
	copy(in.ts[0][index:index+remaining], in.ts[0][index+1:index+1+remaining])
	copy(in.ts[1][index:index+remaining], in.ts[1][index+1:index+1+remaining])
	idxBit := uint16(1) << uint(index)
	clearMask := ^(idxBit - 1)
	coBit := in.isCoin[0] & idxBit
	in.isCoin[0] -= ((in.isCoin[0] >> 1) & clearMask) + coBit
	in.isCoin[1] -= ((in.isCoin[1] >> 1) & clearMask) + coBit
}
