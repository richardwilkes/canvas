// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The engine tail of Simplify: bridgeWinding (the non-zero-winding output tracer, with a findChase multi-piece chase),
// bridgeXor (the even-odd tracer over findUndone, no chase), and runSimplify (the body run after the convex fast path,
// which stays in pathops.go).

package pathops

import "github.com/richardwilkes/canvas/path"

// bridgeWinding traces the active winding edges from each sortable top, emitting closed contours; unresolved pieces are
// chased via findChase.
func bridgeWinding(contourList *opContourHead, writer *pathWriter) bool {
	unsortable := false
	for {
		span := findSortableTop(contourList)
		if span == nil {
			break
		}
		current := span.segment
		start := span.next
		end := &span.opSpanBase
		var chase []*opSpanBase
		for {
			if current.activeWinding(start, end) {
				for unsortable || !current.done() {

					nextStart := start
					nextEnd := end
					next := current.findNextWinding(&chase, &nextStart, &nextEnd, &unsortable)
					if next == nil {
						break
					}
					if !current.addCurveTo(start, end, writer) {
						return false
					}
					current = next
					start = nextStart
					end = nextEnd
					if writer.isClosed() || (unsortable && start.starter(end).done) {
						break
					}
				}
				if current.activeWinding(start, end) && !writer.isClosed() {
					spanStart := start.starter(end)
					if !spanStart.done {
						if !current.addCurveTo(start, end, writer) {
							return false
						}
						current.markDone(spanStart)
					}
				}
				writer.finishContour()
			} else {
				var last *opSpanBase
				if !current.markAndChaseDone(start, end, &last) {
					return false
				}
				if last != nil && !last.chased {
					last.setChased(true)
					chase = append(chase, last)
				}
			}
			current = findChase(&chase, &start, &end)
			if current == nil {
				break
			}
		}
	}
	return true
}

// bridgeXor traces every undone edge in angle order (even-odd needs no winding computation), emitting closed contours.
// Returns true only if all edges were processed.
func bridgeXor(contourList *opContourHead, writer *pathWriter) bool {
	unsortable := false
	safetyNet := 1000000
	for {
		span := findUndone(contourList)
		if span == nil {
			break
		}
		current := span.segment
		start := span.next
		end := &span.opSpanBase
		for {
			safetyNet--
			if safetyNet < 0 {
				return false
			}
			if !unsortable && current.done() {
				break
			}
			nextStart := start
			nextEnd := end
			next := current.findNextXor(&nextStart, &nextEnd, &unsortable)
			if next == nil {
				break
			}
			if !current.addCurveTo(start, end, writer) {
				return false
			}
			current = next
			start = nextStart
			end = nextEnd
			if writer.isClosed() || (unsortable && start.starter(end).done) {
				break
			}
		}
		if !writer.isClosed() {
			spanStart := start.starter(end)
			if !spanStart.done {
				return false
			}
		}
		writer.finishContour()
	}
	return true
}

// runSimplify runs the simplification engine after the convex fast path (which stays in Simplify, pathops.go). fillType
// is the even-odd/inverse-even-odd result fill.
func runSimplify(p *path.Path, fillType path.FillType) (*path.Path, bool) {
	head := &opContourHead{}
	globalState := newOpGlobalState(head)
	coincidence := newOpCoincidence(globalState)
	// turn path into list of segments
	builder := newOpEdgeBuilder(p, head, globalState)
	if !builder.finish() {
		return nil, false
	}
	if !sortContourList(head, false, false) {
		result := path.New()
		result.SetFillType(fillType)
		return result, true
	}
	// find all intersections between segments
	addIntersections(head, coincidence)
	if !handleCoincidence(head, coincidence) {
		return nil, false
	}
	// construct closed contours
	writer := newPathWriter(fillType)
	if builder.xorMask[b2i(builder.operand)] == windingMask {
		if !bridgeWinding(head, writer) {
			return nil, false
		}
	} else {
		if !bridgeXor(head, writer) {
			return nil, false
		}
	}
	writer.assemble() // if some edges could not be resolved, assemble remaining
	return writer.nativePath(), true
}
