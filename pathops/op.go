// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The boolean-op engine tail: findChaseOp (the binary-op chase walker), bridgeOp (the output tracer that walks the
// resolved segment graph and emits closed contours), and runOp (the driver that runs the operator/fill-type algebra and
// the fast paths, which stay in pathops.go). The chase worklist is a []*opSpanBase; the fill-rule masks flow as plain
// ints (winding=-1, even-odd=1) so the bit-mask math in activeOp stays simple integer arithmetic; both operand paths
// map to *path.Path.

package pathops

import "github.com/richardwilkes/canvas/path"

// findChaseOp pops chase spans, seeds the minuend/subtrahend windings, and advances to the next active edge for a
// boolean op, marking passed-over edges. Sets *result to the next segment (nil when the chase is exhausted) and returns
// false only on an unrecoverable markAngle.
func findChaseOp(chase *[]*opSpanBase, startPtr, endPtr **opSpanBase, result **opSegment) bool {
	for len(*chase) != 0 {
		span := (*chase)[len(*chase)-1]
		*chase = (*chase)[:len(*chase)-1]
		// OPTIMIZE: prev makes this compatible with old code -- but is it necessary?
		*startPtr = span.ptT.prev().span
		segment := (*startPtr).segment
		done := true
		*endPtr = nil
		if last := segment.activeAngle(*startPtr, startPtr, endPtr, &done); last != nil {
			*startPtr = last.start
			*endPtr = last.end
			*chase = append(*chase, span)
			*result = last.segment()
			return true
		}
		if done {
			continue
		}
		var winding int
		var sortable bool
		angle := angleWinding(*startPtr, *endPtr, &winding, &sortable)
		if angle == nil {
			*result = nil
			return true
		}
		if winding == skMinS32 {
			continue
		}
		var sumMiWinding, sumSuWinding int
		if sortable {
			segment = angle.segment()
			sumMiWinding = segment.updateWindingReverse(angle)
			if sumMiWinding == skMinS32 {
				*result = nil
				return true
			}
			sumSuWinding = segment.updateOppWindingReverse(angle)
			if sumSuWinding == skMinS32 {
				*result = nil
				return true
			}
			if segment.operand() {
				sumMiWinding, sumSuWinding = sumSuWinding, sumMiWinding
			}
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
			var maxWinding, sumWinding, oppMaxWinding, oppSumWinding int
			if sortable {
				segment.setUpWindingsBinary(start, end, &sumMiWinding, &sumSuWinding,
					&maxWinding, &sumWinding, &oppMaxWinding, &oppSumWinding)
			}
			if !segment.doneAngle(angle) {
				if first == nil && (sortable || start.starter(end).windSum != skMinS32) {
					first = segment
					*startPtr = start
					*endPtr = end
				}
				// OPTIMIZATION: should this also add to the chase?
				if sortable {
					if !segment.markAngleBinary(maxWinding, sumWinding, oppMaxWinding, oppSumWinding, angle,
						nil) {
						return false
					}
				}
			}
		}
		if first != nil {
			*chase = append(*chase, span)
			*result = first
			return true
		}
	}
	*result = nil
	return true
}

// bridgeOp repeatedly finds a sortable top span and traces the active-edge chain from it, emitting closed contours into
// the writer, until every span is consumed.
func bridgeOp(contourList *opContourHead, op PathOp, xorMask, xorOpMask int, writer *pathWriter) bool {
	unsortable := false
	lastSimple := false
	simple := false
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
			if current.activeOp(start, end, xorMask, xorOpMask, op) {
				for unsortable || !current.done() {

					nextStart := start
					nextEnd := end
					lastSimple = simple
					next := current.findNextOp(&chase, &nextStart, &nextEnd, &unsortable, &simple, op, xorMask,
						xorOpMask)
					if next == nil {
						if !unsortable && writer.hasMove() && current.verb != path.VerbLine &&
							!writer.isClosed() {
							if !current.addCurveTo(start, end, writer) {
								return false
							}
						} else if lastSimple {
							if !current.addCurveTo(start, end, writer) {
								return false
							}
						}
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
			if !findChaseOp(&chase, &start, &end, &current) {
				return false
			}
			if current == nil {
				break
			}
		}
	}
	return true
}

// runOp runs the boolean-op pipeline after the fast paths (rect-intersect, empty-operand) and the operator/ fill-type
// algebra, which stay in Op (pathops.go). minuend/subtrahend and op are already remapped (any reverse-difference
// reduced to difference by swapping the operands). fillType is the even-odd/inverse-even-odd result fill.
func runOp(minuend, subtrahend *path.Path, op PathOp, fillType path.FillType) (*path.Path, bool) {
	head := &opContourHead{}
	globalState := newOpGlobalState(head)
	coincidence := newOpCoincidence(globalState)
	// turn path into list of segments
	builder := newOpEdgeBuilder(minuend, head, globalState)
	if builder.unparseable {
		return nil, false
	}
	xorMask := int(builder.xorMask[b2i(builder.operand)])
	builder.addOperand(subtrahend)
	if !builder.finish() {
		return nil, false
	}
	xorOpMask := int(builder.xorMask[b2i(builder.operand)])
	if !sortContourList(head, xorMask == int(evenOddMask), xorOpMask == int(evenOddMask)) {
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
	if !bridgeOp(head, op, xorMask, xorOpMask, writer) {
		return nil, false
	}
	writer.assemble() // if some edges could not be resolved, assemble remaining
	return writer.nativePath(), true
}
