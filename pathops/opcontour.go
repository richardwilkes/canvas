// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The contour construction surface: a contour holds a forward-linked list of segments (the first stored by value as
// head), plus opContourHead (the list head with appendContour/remove) and opContourBuilder (the line-combining front
// end the edge builder drives). This slice covers the pieces that build the data model; the walking helpers
// (calcAngles, sortAngles, toPath, missingCoincidence, …) arrive with their consumers.

package pathops

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// opContour is one closed contour's worth of segments plus the bookkeeping the boolean-op engine attaches to it. The
// zero value is the reset state (tail/next nil, count 0, done false).
type opContour struct {
	state   *opGlobalState // the shared state for the whole operation
	tail    *opSegment     // the last segment appended
	next    *opContour     // the next contour in the list
	head    opSegment      // the first segment, stored by value
	ccw     int            // ray-cast winding orientation: -1 unset, else 0/1
	count   int            // number of segments appended
	bounds  pathOpsBounds  // the union of all segment bounds
	done    bool           // set by findSortableTop once every segment is done
	operand bool           // true for the second argument to a binary operator
	reverse bool           // true when the contour must be written back reversed (fix-winding only)
	xor     bool           // true when the source path had even-odd fill
	oppXor  bool           // true when the opposite operand's path had even-odd fill
}

func (c *opContour) globalState() *opGlobalState { return c.state }

// init sets up a freshly appended contour: the shared global state, whether it is the second (subtrahend) operand, and
// whether its source path used even-odd fill.
func (c *opContour) init(state *opGlobalState, operand, isXor bool) {
	c.state = state
	c.operand = operand
	c.xor = isXor
}

// appendSegment adds a new segment to the contour's list, returning it. The first segment reuses the head field's
// storage; later segments are heap-allocated.
func (c *opContour) appendSegment() *opSegment {
	var result *opSegment
	if c.count > 0 {
		result = &opSegment{}
	} else {
		result = &c.head
	}
	c.count++
	result.setPrev(c.tail)
	if c.tail != nil {
		c.tail.setNext(result)
	}
	c.tail = result
	return result
}

// addLine appends a new line segment built from pts to the contour.
func (c *opContour) addLine(pts []geom.Point) *opSegment {
	return c.appendSegment().addLine(pts, c)
}

// addQuad appends a new quadratic-curve segment built from pts to the contour.
func (c *opContour) addQuad(pts []geom.Point) { c.appendSegment().addQuad(pts, c) }

// addConic appends a new conic segment built from pts and weight to the contour.
func (c *opContour) addConic(pts []geom.Point, weight float32) {
	c.appendSegment().addConic(pts, weight, c)
}

// addCubic appends a new cubic-curve segment built from pts to the contour.
func (c *opContour) addCubic(pts []geom.Point) { c.appendSegment().addCubic(pts, c) }

// complete finalizes the contour once all its segments have been added.
func (c *opContour) complete() { c.setBounds() }

// setBounds recomputes the contour's bounds as the union of all its segments' bounds.
func (c *opContour) setBounds() {
	segment := &c.head
	c.bounds = segment.bounds
	for segment = segment.next; segment != nil; segment = segment.next {
		c.bounds.addBounds(segment.bounds)
	}
}

// first returns the contour's first segment.
func (c *opContour) first() *opSegment { return &c.head }

// missingCoincidence runs the per-segment check across the contour, returning whether any segment found a missing
// coincident run.
func (c *opContour) missingCoincidence() bool {
	result := false
	for segment := c.first(); segment != nil; segment = segment.next {
		if segment.missingCoincidence() {
			result = true
		}
	}
	return result
}

// moveMultiples runs the per-segment merge across the contour, aborting on the first failure.
func (c *opContour) moveMultiples() bool {
	for segment := c.first(); segment != nil; segment = segment.next {
		if !segment.moveMultiples() {
			return false
		}
	}
	return true
}

// moveNearby runs the per-segment gap collapse across the contour, aborting on the first failure.
func (c *opContour) moveNearby() bool {
	for segment := c.first(); segment != nil; segment = segment.next {
		if !segment.moveNearby() {
			return false
		}
	}
	return true
}

// calcAngles builds the angle loops for every segment in the contour.
func (c *opContour) calcAngles() {
	for segment := c.first(); segment != nil; segment = segment.next {
		segment.calcAngles()
	}
}

// sortAngles sorts every segment's angle loops, aborting on failure.
func (c *opContour) sortAngles() bool {
	for segment := c.first(); segment != nil; segment = segment.next {
		if !segment.sortAngles() {
			return false
		}
	}
	return true
}

// undoneSpan returns the first undone span across the contour's segments; marks the contour done when every segment is.
func (c *opContour) undoneSpan() *opSpan {
	for testSegment := c.first(); testSegment != nil; testSegment = testSegment.next {
		if testSegment.done() {
			continue
		}
		return testSegment.undoneSpan()
	}
	c.done = true
	return nil
}

// less orders contours by the bounds' top, breaking ties by left.
func (c *opContour) less(rh *opContour) bool {
	if c.bounds.top == rh.bounds.top {
		return c.bounds.left < rh.bounds.left
	}
	return c.bounds.top < rh.bounds.top
}

// setNext links contour as the next contour after c in the list.
func (c *opContour) setNext(contour *opContour) { c.next = contour }

// setXor marks whether this contour's source path used even-odd fill.
func (c *opContour) setXor(isXor bool) { c.xor = isXor }

// setOppXor marks whether the opposite operand's path used even-odd fill.
func (c *opContour) setOppXor(isOppXor bool) { c.oppXor = isOppXor }

// setCcw records the ray-cast winding orientation (used by the fix-winding phase).
func (c *opContour) setCcw(ccw int) { c.ccw = ccw }

// isCcw returns the ray-cast winding orientation findSortableTop recorded during the fix-winding phase (-1 until set,
// then 0/1).
func (c *opContour) isCcw() int { return c.ccw }

// setReverse flags the contour to be reverse-written (fix-winding only).
func (c *opContour) setReverse() { c.reverse = true }

// reversed reports whether the contour is flagged to be reverse-written.
func (c *opContour) reversed() bool { return c.reverse }

// resetReverse clears the ccw/reverse state on this contour and every non-empty contour after it (fixWinding calls this
// on the head to reset the whole list).
func (c *opContour) resetReverse() {
	for next := c; next != nil; next = next.next {
		if next.count == 0 {
			continue
		}
		next.ccw = -1
		next.reverse = false
	}
}

// markAllDone flags every span of every segment in the contour done.
func (c *opContour) markAllDone() {
	for segment := &c.head; segment != nil; segment = segment.next {
		segment.markAllDone()
	}
}

// joinSegments links each segment's tail pt-t to the next segment's head (the last segment joins back to the head), so
// the fix-winding walk can traverse the closed contour.
func (c *opContour) joinSegments() {
	segment := &c.head
	for segment != nil {
		next := segment.next
		target := next
		if target == nil {
			target = &c.head
		}
		segment.joinEnds(target)
		segment = next
	}
}

// toPath emits every segment forward (head→tail) into the writer, then finishes and assembles the contour.
func (c *opContour) toPath(writer *pathWriter) {
	if c.count == 0 {
		return
	}
	for segment := &c.head; segment != nil; segment = segment.next {
		segment.addCurveTo(&segment.head.opSpanBase, &segment.tail, writer)
	}
	writer.finishContour()
	writer.assemble()
}

// toReversePath emits every segment backward (tail→head) from the contour tail, then finishes and assembles. Only
// called on non-empty contours (the caller guards on count).
func (c *opContour) toReversePath(writer *pathWriter) {
	for segment := c.tail; segment != nil; segment = segment.prev {
		segment.addCurveTo(&segment.tail, &segment.head.opSpanBase, writer)
	}
	writer.finishContour()
	writer.assemble()
}

// opContourHead is the head of the contour list. sortContourList wants to re-seat the walk's starting point at
// whichever contour has the smallest bounds, but a Go type embedding opContour by value cannot be re-seated to an
// arbitrary heap contour. So this type keeps the first-built contour as physical storage (the embedded opContour) and
// redirects the walk head through sortedHead, which sortContourList points at the bounds-smallest contour. Before
// sorting, sortedHead is nil and the walk starts at the embedded contour.
type opContourHead struct {
	sortedHead *opContour // sortContourList's bounds-smallest head (nil before sorting)
	opContour
}

// listHead returns the head of the contour walk: the bounds-smallest contour once sortContourList has re-seated it,
// else the first-built (embedded) contour. Every traversal that runs after sortContourList must start here rather than
// at &h.opContour.
func (h *opContourHead) listHead() *opContour {
	if h.sortedHead != nil {
		return h.sortedHead
	}
	return &h.opContour
}

// appendContour allocates a contour and links it at the list tail.
func (h *opContourHead) appendContour() *opContour {
	contour := &opContour{}
	prev := &h.opContour
	for prev.next != nil {
		prev = prev.next
	}
	prev.next = contour
	return contour
}

// remove unlinks contour (which must be the last, unless it is the head).
func (h *opContourHead) remove(contour *opContour) {
	if contour == &h.opContour {
		return
	}
	prev := &h.opContour
	for prev.next != contour {
		prev = prev.next
	}
	prev.next = nil
}

// joinAllSegments runs joinSegments on every non-empty contour in the list (fixWinding calls this before the
// fix-winding ray-cast walk).
func (h *opContourHead) joinAllSegments() {
	for next := &h.opContour; next != nil; next = next.next {
		if next.count == 0 {
			continue
		}
		next.joinSegments()
	}
}

// opContourBuilder coalesces the run of lines the edge builder emits, eliminating an immediately-reversed line pair,
// before handing them to the contour.
type opContourBuilder struct {
	contour    *opContour    // the contour segments are appended to
	lastLine   [2]geom.Point // the most recently buffered (not yet flushed) line, if lastIsLine
	lastIsLine bool          // true when lastLine holds a pending line not yet appended
}

// newOpContourBuilder returns a builder that appends coalesced segments to contour.
func newOpContourBuilder(contour *opContour) *opContourBuilder {
	return &opContourBuilder{contour: contour}
}

// addConic flushes any pending line, then appends a conic segment to the contour.
func (b *opContourBuilder) addConic(pts []geom.Point, weight float32) {
	b.flush()
	b.contour.addConic(pts, weight)
}

// addCubic flushes any pending line, then appends a cubic segment to the contour.
func (b *opContourBuilder) addCubic(pts []geom.Point) {
	b.flush()
	b.contour.addCubic(pts)
}

// addCurve dispatches by verb to the matching add method, copying the points into stable storage before adding since
// the caller's array is transient.
func (b *opContourBuilder) addCurve(verb path.Verb, pts []geom.Point, weight float32) {
	switch verb {
	case path.VerbLine:
		b.addLine(pts)
	case path.VerbQuad:
		b.addQuad(append([]geom.Point(nil), pts[:3]...))
	case path.VerbConic:
		b.addConic(append([]geom.Point(nil), pts[:3]...), weight)
	case path.VerbCubic:
		b.addCubic(append([]geom.Point(nil), pts[:4]...))
	}
}

// addLine buffers pts as the pending line; if the previous pending line was the exact opposite (a degenerate
// there-and-back pair), it cancels both instead of appending either.
func (b *opContourBuilder) addLine(pts []geom.Point) {
	if b.lastIsLine {
		if b.lastLine[0] == pts[1] && b.lastLine[1] == pts[0] {
			b.lastIsLine = false
			return
		}
		b.flush()
	}
	b.lastLine[0] = pts[0]
	b.lastLine[1] = pts[1]
	b.lastIsLine = true
}

// addQuad flushes any pending line, then appends a quadratic-curve segment to the contour.
func (b *opContourBuilder) addQuad(pts []geom.Point) {
	b.flush()
	b.contour.addQuad(pts)
}

// flush appends the pending line to the contour, if any is buffered.
func (b *opContourBuilder) flush() {
	if !b.lastIsLine {
		return
	}
	b.contour.addLine([]geom.Point{b.lastLine[0], b.lastLine[1]})
	b.lastIsLine = false
}

// setContour flushes any pending line, then retargets the builder at a new contour.
func (b *opContourBuilder) setContour(contour *opContour) {
	b.flush()
	b.contour = contour
}
