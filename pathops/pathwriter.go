// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The pathops output assembler. The bridgeOp/bridgeWinding/bridgeXor drivers trace the resolved segment graph and emit
// one edge at a time through deferredMove/deferredLine/quadTo/conicTo/cubicTo; pathWriter buffers the current contour,
// closes it when the trace loops back to its start, and stores any partial (open) contours for assemble() to stitch
// together at the end. finishContour stores the current contour by pointer and init() allocates a fresh *path.Path so a
// contour already stored in partials keeps its own storage. Sorting uses sort.Slice, which is unstable, so tie-broken
// orderings are not guaranteed stable across runs (assemble only runs for the rare open-contour case, and its result is
// area-equivalent regardless).

package pathops

import (
	"sort"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// pathWriter accumulates the output path while tracing the resolved segment graph.
type pathWriter struct {
	defers   [2]*opPtT    // [0] deferred move, [1] deferred line
	builder  *path.Path   // the accumulated output
	current  *path.Path   // contour under construction
	firstPtT *opPtT       // first in current contour
	partials []*path.Path // contours with mismatched starts and ends
	endPtTs  []*opPtT     // possible pt values for partial starts and ends
}

// newPathWriter creates a pathWriter that will produce output with the given fill type.
func newPathWriter(ft path.FillType) *pathWriter {
	w := &pathWriter{builder: path.New()}
	w.builder.SetFillType(ft)
	w.init()
	return w
}

// init resets the writer to start a new contour, allocating a fresh path so a contour already stored in partials keeps
// its own storage.
func (w *pathWriter) init() {
	w.current = path.New()
	w.firstPtT = nil
	w.defers[0] = nil
	w.defers[1] = nil
}

// hasMove reports whether no first point has been set yet for the current contour.
func (w *pathWriter) hasMove() bool { return w.firstPtT == nil }

// isClosed reports whether the last deferred point matches the current contour's first point.
func (w *pathWriter) isClosed() bool { return w.matchedLast(w.firstPtT) }

// nativePath returns the accumulated output path.
func (w *pathWriter) nativePath() *path.Path { return w.builder }

// close finalizes the current contour and appends it to the output, then resets for the next contour.
func (w *pathWriter) close() {
	if w.current.IsEmpty() {
		return
	}
	w.current.Close()
	w.builder.AddPath(w.current, path.AddPathAppend)
	w.init()
}

// conicTo flushes any deferred move/line, then appends a conic edge ending at pt2.
func (w *pathWriter) conicTo(pt1 geom.Point, pt2 *opPtT, weight float32) {
	pt2pt := w.update(pt2)
	w.current.ConicToPt(pt1, pt2pt, weight)
}

// cubicTo flushes any deferred move/line, then appends a cubic edge ending at pt3.
func (w *pathWriter) cubicTo(pt1, pt2 geom.Point, pt3 *opPtT) {
	pt3pt := w.update(pt3)
	w.current.CubicToPt(pt1, pt2, pt3pt)
}

// quadTo flushes any deferred move/line, then appends a quadratic edge ending at pt2.
func (w *pathWriter) quadTo(pt1 geom.Point, pt2 *opPtT) {
	pt2pt := w.update(pt2)
	w.current.QuadToPt(pt1, pt2pt)
}

// deferredLine records pt as the end of a pending line edge without writing it immediately, so consecutive collinear
// line edges can be merged into one. Returns false if pt closes the contour (the caller should finish it instead).
func (w *pathWriter) deferredLine(pt *opPtT) bool {
	if w.defers[0] == pt {
		// Caller should have preflighted this: adding a degenerate (zero-length) line.
		return true
	}
	if pt.containsPtT(w.defers[0]) {
		// Caller should have preflighted this: adding a degenerate (zero-length) line.
		return true
	}
	if w.matchedLast(pt) {
		return false
	}
	if w.defers[1] != nil && w.changedSlopes(pt) {
		w.lineTo()
		w.defers[0] = w.defers[1]
	}
	w.defers[1] = pt
	return true
}

// deferredMove records pt as the start of a new contour, finishing the previous contour first if pt doesn't match its
// last deferred point.
func (w *pathWriter) deferredMove(pt *opPtT) {
	if w.defers[1] == nil {
		w.firstPtT = pt
		w.defers[0] = pt
		return
	}
	if !w.matchedLast(pt) {
		w.finishContour()
		w.firstPtT = pt
		w.defers[0] = pt
	}
}

// finishContour flushes any pending deferred line, then either closes the current contour (if its ends match) or
// stashes it as a partial (open) contour for assemble() to stitch later.
func (w *pathWriter) finishContour() {
	if !w.matchedLast(w.defers[0]) {
		if w.defers[1] == nil {
			return
		}
		w.lineTo()
	}
	if w.current.IsEmpty() {
		return
	}
	if w.isClosed() {
		w.close()
	} else {
		w.endPtTs = append(w.endPtTs, w.firstPtT, w.defers[1])
		w.partials = append(w.partials, w.current)
		w.init()
	}
}

// lineTo writes the pending deferred line edge to the current contour, moving to the first point first if the contour
// is still empty.
func (w *pathWriter) lineTo() {
	if w.current.IsEmpty() {
		w.moveTo()
	}
	w.current.LineToPt(w.defers[1].pt)
}

// matchedLast reports whether test is (or shares a pt-t loop with) the last deferred point.
func (w *pathWriter) matchedLast(test *opPtT) bool {
	if test == w.defers[1] {
		return true
	}
	if test == nil {
		return false
	}
	if w.defers[1] == nil {
		return false
	}
	return test.containsPtT(w.defers[1])
}

// moveTo writes a move to the current contour's first point.
func (w *pathWriter) moveTo() { w.current.MoveToPt(w.firstPtT.pt) }

// update flushes any pending deferred move/line and returns the point to write next: if the last point to be written
// matches the current contour's first point, it substitutes that first point to avoid writing a degenerate lineTo when
// the contour is closed.
func (w *pathWriter) update(pt *opPtT) geom.Point {
	if w.defers[1] == nil {
		w.moveTo()
	} else if !w.matchedLast(w.defers[0]) {
		w.lineTo()
	}
	result := pt.pt
	if w.firstPtT != nil && result != w.firstPtT.pt && w.firstPtT.containsPtT(pt) {
		result = w.firstPtT.pt
	}
	w.defers[0] = pt // set both to know that there is not a pending deferred line
	w.defers[1] = pt
	return result
}

// someAssemblyRequired finishes the current contour and reports whether any partial (open) contours remain that need
// assemble() to stitch together.
func (w *pathWriter) someAssemblyRequired() bool {
	w.finishContour()
	return len(w.endPtTs) != 0
}

// changedSlopes reports whether extending the pending deferred line to ptT would change its direction, meaning the line
// so far must be flushed before continuing.
func (w *pathWriter) changedSlopes(ptT *opPtT) bool {
	if w.matchedLast(w.defers[0]) {
		return false
	}
	deferDx := w.defers[1].pt.X - w.defers[0].pt.X
	deferDy := w.defers[1].pt.Y - w.defers[0].pt.Y
	lineDx := ptT.pt.X - w.defers[1].pt.X
	lineDy := ptT.pt.Y - w.defers[1].pt.Y
	return deferDx*lineDy != deferDy*lineDx
}

// assemble checks the start and end of each partial contour; if they differ, it matches them up, connects the closest,
// and reassembles the pieces into new closed contours.
func (w *pathWriter) assemble() {
	if !w.someAssemblyRequired() {
		return
	}
	// The partials are consumed into w.builder below (or abandoned, when the link walk cannot continue), so they must
	// not survive into the next contour: opContour.toPath/toReversePath call assemble() once per contour on a writer
	// shared across the whole path, and stale partials would be re-walked and their geometry emitted a second time.
	defer func() {
		w.partials = nil
		w.endPtTs = nil
	}()
	runs := w.endPtTs // starts, ends of partial contours (shares backing with w.endPtTs)
	endCount := len(w.endPtTs)
	// lengthen any partial contour adjacent to a simple segment
	for pIndex := 0; pIndex < endCount; pIndex++ {
		curPtT := runs[pIndex]
		partWriter := newPathWriter(path.FillWinding) // the default fill type
		for zeroOrOne(curPtT.t) {

			spanBase := curPtT.span
			var start *opSpanBase
			step := -1
			if curPtT.t != 0 {
				start = &spanBase.prev.opSpanBase
				step = 1
			} else {
				start = spanBase.upCast().next
			}
			seg := spanBase.segment
			nextSegment := seg.isSimple(&start, &step)
			if nextSegment == nil {
				break
			}
			var opSpanEnd *opSpanBase
			if start.t() != 0 {
				opSpanEnd = &start.prev.opSpanBase
			} else {
				opSpanEnd = start.upCast().next
			}
			if start.starter(opSpanEnd).alreadyAdded {
				break
			}
			nextSegment.addCurveTo(start, opSpanEnd, partWriter)
			curPtT = &opSpanEnd.ptT
			runs[pIndex] = curPtT
		}
		partWriter.finishContour()
		if len(partWriter.partials) == 0 {
			continue
		}
		// if pIndex is even, reverse and prepend to fPartials; otherwise, append
		partial := w.partials[pIndex>>1]
		part := partWriter.partials[0]
		if pIndex&1 != 0 {
			partial.AddPath(part, path.AddPathExtend)
		} else {
			reverse := path.New()
			reverse.ReverseAddPath(part)
			reverse.AddPath(partial, path.AddPathExtend)
			w.partials[pIndex>>1] = reverse
		}
	}
	linkCount := endCount / 2 // number of partial contours
	sLink := make([]int, linkCount)
	eLink := make([]int, linkCount)
	for rIndex := 0; rIndex < linkCount; rIndex++ {
		sLink[rIndex] = skMaxS32
		eLink[rIndex] = skMaxS32
	}
	entries := endCount * (endCount - 1) / 2 // folded triangle
	distances := make([]float64, 0, entries)
	sortedDist := make([]int, 0, entries)
	distLookup := make([]int, 0, entries)
	rRow := 0
	dIndex := 0
	for rIndex := 0; rIndex < endCount-1; rIndex++ {
		oPtT := runs[rIndex]
		for iIndex := rIndex + 1; iIndex < endCount; iIndex++ {
			iPtT := runs[iIndex]
			dx := float64(iPtT.pt.X - oPtT.pt.X)
			dy := float64(iPtT.pt.Y - oPtT.pt.Y)
			dist := dx*dx + dy*dy
			distLookup = append(distLookup, rRow+iIndex)
			distances = append(distances, dist) // oStart distance from iStart
			sortedDist = append(sortedDist, dIndex)
			dIndex++
		}
		rRow += endCount
	}
	sort.Slice(sortedDist, func(a, b int) bool { return distances[sortedDist[a]] < distances[sortedDist[b]] })
	remaining := linkCount // number of start/end pairs
	for rIndex := 0; rIndex < entries; rIndex++ {
		pair := distLookup[sortedDist[rIndex]]
		row := pair / endCount
		col := pair - row*endCount
		ndxOne := row >> 1
		endOne := row&1 != 0
		linkOne := sLink
		if endOne {
			linkOne = eLink
		}
		if linkOne[ndxOne] != skMaxS32 {
			continue
		}
		ndxTwo := col >> 1
		endTwo := col&1 != 0
		linkTwo := sLink
		if endTwo {
			linkTwo = eLink
		}
		if linkTwo[ndxTwo] != skMaxS32 {
			continue
		}
		flip := endOne == endTwo
		if flip {
			linkOne[ndxOne] = ^ndxTwo
			linkTwo[ndxTwo] = ^ndxOne
		} else {
			linkOne[ndxOne] = ndxTwo
			linkTwo[ndxTwo] = ndxOne
		}
		remaining--
		if remaining == 0 {
			break
		}
	}
	rIndex := 0
	for {
		forward := true
		first := true
		sIndex := sLink[rIndex]
		sLink[rIndex] = skMaxS32
		var eIndex int
		if sIndex < 0 {
			eIndex = sLink[^sIndex]
			sLink[^sIndex] = skMaxS32
		} else {
			eIndex = eLink[sIndex]
			eLink[sIndex] = skMaxS32
		}
		for {
			contour := w.partials[rIndex]
			if !first {
				if _, ok := w.builder.LastPt(); !ok {
					return
				}
				// TODO: if there is a gap between the open path written so far and the path to come, connect by
				// following segments rather than introducing a diagonal.
			}
			if forward {
				mode := path.AddPathExtend
				if first {
					mode = path.AddPathAppend
				}
				w.builder.AddPath(contour, mode)
			} else {
				w.builder.ReversePathTo(contour)
			}
			first = false
			var cmp int
			if (rIndex != eIndex) != forward { // (rIndex != eIndex) ^ forward
				cmp = eIndex
			} else {
				cmp = ^eIndex
			}
			if sIndex == cmp {
				w.builder.Close()
				break
			}
			if forward {
				eIndex = eLink[rIndex]
				eLink[rIndex] = skMaxS32
				if eIndex >= 0 {
					sLink[eIndex] = skMaxS32
				} else {
					eLink[^eIndex] = skMaxS32
				}
			} else {
				eIndex = sLink[rIndex]
				sLink[rIndex] = skMaxS32
				if eIndex >= 0 {
					eLink[eIndex] = skMaxS32
				} else {
					sLink[^eIndex] = skMaxS32
				}
			}
			rIndex = eIndex
			if rIndex < 0 {
				forward = !forward
				rIndex = ^rIndex
			}
		}
		for rIndex = 0; rIndex < linkCount; rIndex++ {
			if sLink[rIndex] != skMaxS32 {
				break
			}
		}
		if rIndex >= linkCount {
			break
		}
	}
}
