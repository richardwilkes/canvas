// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The shared driver spine the boolean-op and simplify entry points build on — the angle-winding lookup (angleWinding),
// the undone-span scan (findUndone), the simplify chase walker (findChase), the bounds-sort of the contour list
// (sortContourList), and the whole coincidence-resolution orchestrator (handleCoincidence) with its per-contour
// calc/sort-angles and move/missing-coincidence passes. findSortableTop lives in winding.go (session 64).
//
// The chase list is a *[]*opSpanBase, with appends always going to the tail; sortContourList re-seats the head through
// the opContourHead sortedHead redirect rather than reassigning a head pointer (see opContourHead); the two markAngle
// overloads are markAngleUnary/markAngleBinary. Pathops computes in double precision, so winding accumulators are plain
// int and t values are float64.

package pathops

import "sort"

// angleWinding finds the first angle in the loop leaving [start,end], seeds the winding to its computed wind sum, and
// reports whether the loop is sortable. Returns the angle whose segment carries the winding (nil when the loop cannot
// be resolved).
func angleWinding(start, end *opSpanBase, windingPtr *int, sortablePtr *bool) *opAngle {
	segment := start.segment
	angle := segment.spanToAngle(start, end)
	if angle == nil {
		*windingPtr = skMinS32
		return nil
	}
	computeWinding := false
	firstAngle := angle
	loop := false
	unorderable := false
	winding := skMinS32
	for {
		angle = angle.next
		if angle == nil {
			return nil
		}
		unorderable = unorderable || angle.unorderable
		computeWinding = unorderable || (angle == firstAngle && loop)
		if computeWinding {
			break // if we get here, there's no winding, loop is unorderable
		}
		loop = loop || angle == firstAngle
		segment = angle.segment()
		winding = segment.windSum(angle)
		if winding != skMinS32 {
			break
		}
	}
	// if the angle loop contains an unorderable span, the angle order may be useless; directly compute the winding in
	// this case for each span
	if computeWinding {
		firstAngle = angle
		winding = skMinS32
		for {
			startSpan := angle.start
			endSpan := angle.end
			lesser := startSpan.starter(endSpan)
			testWinding := lesser.windSum
			if testWinding == skMinS32 {
				testWinding = lesser.computeWindSum()
			}
			if testWinding != skMinS32 {
				winding = testWinding
			}
			angle = angle.next
			if angle == firstAngle {
				break
			}
		}
	}
	*sortablePtr = !unorderable
	*windingPtr = winding
	return angle
}

// findUndone returns the first undone span across the contour list, or nil if every span is done.
func findUndone(contourHead *opContourHead) *opSpan {
	for contour := contourHead.listHead(); contour != nil; contour = contour.next {
		if contour.done {
			continue
		}
		if result := contour.undoneSpan(); result != nil {
			return result
		}
	}
	return nil
}

// findChase pops chase spans, seeds the winding, and advances to the next active edge, marking passed-over edges.
// Returns the segment to continue tracing from, or nil when the chase is exhausted.
func findChase(chase *[]*opSpanBase, startPtr, endPtr **opSpanBase) *opSegment {
	for len(*chase) != 0 {
		span := (*chase)[len(*chase)-1]
		*chase = (*chase)[:len(*chase)-1]
		segment := span.segment
		*startPtr = span.ptT.next.span
		done := true
		*endPtr = nil
		if last := segment.activeAngle(*startPtr, startPtr, endPtr, &done); last != nil {
			*startPtr = last.start
			*endPtr = last.end
			*chase = append(*chase, span)
			return last.segment()
		}
		if done {
			continue
		}
		// find first angle, initialize winding to computed wind sum
		var winding int
		var sortable bool
		angle := angleWinding(*startPtr, *endPtr, &winding, &sortable)
		if angle == nil {
			return nil
		}
		if winding == skMinS32 {
			continue
		}
		var sumWinding int
		if sortable {
			segment = angle.segment()
			sumWinding = segment.updateWindingReverse(angle)
		}
		var first *opSegment
		firstAngle := angle
		for {
			angle = angle.next
			if angle == firstAngle {
				break
			}
			segment = angle.segment()
			start := angle.start
			end := angle.end
			var maxWinding int
			if sortable {
				segment.setUpWinding(start, end, &maxWinding, &sumWinding)
			}
			if !segment.doneAngle(angle) {
				if first == nil && (sortable || start.starter(end).windSum != skMinS32) {
					first = segment
					*startPtr = start
					*endPtr = end
				}
				// OPTIMIZATION: should this also add to the chase?
				if sortable {
					segment.markAngleUnary(maxWinding, sumWinding, angle, nil)
				}
			}
		}
		if first != nil {
			*chase = append(*chase, span)
			return first
		}
	}
	return nil
}

// sortContourList drops empty contours, sets each contour's opposite even-odd flag, bounds-sorts the remainder, and
// re-seats the walk head to the smallest. Returns false when no contour carries geometry (the caller returns an empty
// result).
func sortContourList(contourList *opContourHead, evenOdd, oppEvenOdd bool) bool {
	var list []*opContour
	for contour := contourList.listHead(); contour != nil; contour = contour.next {
		if contour.count != 0 {
			if contour.operand {
				contour.setOppXor(evenOdd)
			} else {
				contour.setOppXor(oppEvenOdd)
			}
			list = append(list, contour)
		}
	}
	count := len(list)
	if count == 0 {
		return false
	}
	if count > 1 {
		sort.Slice(list, func(a, b int) bool { return list[a].less(list[b]) })
	}
	contour := list[0]
	contourList.sortedHead = contour
	contour.globalState().setContourHead(contourList)
	for index := 1; index < count; index++ {
		next := list[index]
		contour.setNext(next)
		contour = next
	}
	contour.setNext(nil)
	return true
}

// calcAnglesAll builds every contour's angle loops.
func calcAnglesAll(contourList *opContourHead) {
	for contour := contourList.listHead(); contour != nil; contour = contour.next {
		contour.calcAngles()
	}
}

// missingCoincidenceAll runs the per-contour missing-coincidence check on every contour, returning whether any contour
// found one.
func missingCoincidenceAll(contourList *opContourHead) bool {
	result := false
	for contour := contourList.listHead(); contour != nil; contour = contour.next {
		if contour.missingCoincidence() {
			result = true
		}
	}
	return result
}

// moveMultiplesAll runs moveMultiples on every contour, combining t values when multiple intersections land on some
// segments but not others. Returns false if any contour's pass fails.
func moveMultiplesAll(contourList *opContourHead) bool {
	for contour := contourList.listHead(); contour != nil; contour = contour.next {
		if !contour.moveMultiples() {
			return false
		}
	}
	return true
}

// moveNearbyAll runs moveNearby on every contour, moving nearby t values and points together to eliminate small or tiny
// gaps. Returns false if any contour's pass fails.
func moveNearbyAll(contourList *opContourHead) bool {
	for contour := contourList.listHead(); contour != nil; contour = contour.next {
		if !contour.moveNearby() {
			return false
		}
	}
	return true
}

// sortAnglesAll sorts the angle loop at every contour. Returns false if any contour's angles cannot be sorted.
func sortAnglesAll(contourList *opContourHead) bool {
	for contour := contourList.listHead(); contour != nil; contour = contour.next {
		if !contour.sortAngles() {
			return false
		}
	}
	return true
}

// handleCoincidence resolves all coincident runs (expand/add-missing/mark/apply/find-overlaps with the two safety-hatch
// loops), then builds and sorts the angle loops. Returns false on any unresolvable step (the caller abandons the op).
func handleCoincidence(contourList *opContourHead, coincidence *opCoincidence) bool {
	globalState := contourList.globalState()
	// match up points within the coincident runs
	if !coincidence.addExpanded() {
		return false
	}
	// combine t values when multiple intersections occur on some segments but not others
	if !moveMultiplesAll(contourList) {
		return false
	}
	// move t values and points together to eliminate small/tiny gaps
	if !moveNearbyAll(contourList) {
		return false
	}
	// add coincidence formed by pairing on curve points and endpoints
	coincidence.correctEnds()
	if !coincidence.addEndMovedSpans() {
		return false
	}
	const safetyCount = 3
	safetyHatch := safetyCount
	// look for coincidence present in A-B and A-C but missing in B-C
	for {
		var added bool
		if !coincidence.addMissing(&added) {
			return false
		}
		if !added {
			break
		}
		safetyHatch--
		if safetyHatch == 0 {
			return false
		}
		moveNearbyAll(contourList)
	}
	// check to see if, loosely, coincident ranges may be expanded
	if coincidence.expand() {
		var added bool
		if !coincidence.addMissing(&added) {
			return false
		}
		if !coincidence.addExpanded() {
			return false
		}
		if !moveMultiplesAll(contourList) {
			return false
		}
		moveNearbyAll(contourList)
	}
	// the expanded ranges may not align -- add the missing spans
	if !coincidence.addExpanded() {
		return false
	}
	// mark spans of coincident segments as coincident
	coincidence.mark()
	// look for coincidence lines and curves undetected by intersection
	if missingCoincidenceAll(contourList) {
		coincidence.expand()
		if !coincidence.addExpanded() {
			return false
		}
		if !coincidence.mark() {
			return false
		}
	} else {
		coincidence.expand()
	}
	coincidence.expand()

	overlaps := &opCoincidence{globalState: globalState}
	safetyHatch = safetyCount
	for {
		pairs := coincidence
		if !overlaps.isEmpty() {
			pairs = overlaps
		}
		// adjust the winding value to account for coincident edges
		if !pairs.apply() {
			return false
		}
		// for each coincident pair that overlaps another, when the receivers (the 1st of the pair) are different,
		// construct a new pair to resolve their mutual span
		if !pairs.findOverlaps(overlaps) {
			return false
		}
		safetyHatch--
		if safetyHatch == 0 {
			return false
		}
		if overlaps.isEmpty() {
			break
		}
	}
	calcAnglesAll(contourList)
	return sortAnglesAll(contourList)
}
