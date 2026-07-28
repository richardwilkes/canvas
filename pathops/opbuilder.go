// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Engine helpers for the incremental path-op builder: oneContour, reversePath, and fixWinding. Builder.Add and
// Builder.Resolve (the accumulate/resolve facade) live in pathops.go; this file holds the fixWinding machinery the
// all-union optimization branch of Resolve needs to convert a simplified (even-odd) operand back to winding form before
// accumulation.

package pathops

import "github.com/richardwilkes/canvas/path"

// oneContour reports whether the path has a single contour: true when only the first verb in the raw verb stream is a
// move.
func oneContour(p *path.Path) bool {
	it := path.NewRawIter(p)
	index := 0
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			break
		}
		if index >= 1 && verb == path.VerbMove {
			return false
		}
		index++
	}
	return true
}

// reversePath returns p's single contour reversed: move to the last point, reverse-append the contour, close.
func reversePath(p *path.Path) *path.Path {
	lastPt, _ := p.LastPt()
	temp := path.New()
	temp.MoveToPt(lastPt)
	temp.ReversePathTo(p)
	temp.Close()
	return temp
}

// fixWinding converts a simplified (even-odd) path to winding form so that, when several such paths are summed and
// re-simplified, overlapping unions reinforce (winding) rather than cancel (even-odd). It reverses contours whose
// ray-cast orientation disagrees with their nesting parity.
//
// Returns the (possibly rebuilt) path and true on success; false only when the input is unparseable or an edge-built
// model has a single surviving contour with an indeterminate orientation.
func fixWinding(p *path.Path) (*path.Path, bool) {
	fillType := p.FillType()
	switch fillType {
	case path.FillInverseEvenOdd:
		fillType = path.FillInverseWinding
	case path.FillEvenOdd:
		fillType = path.FillWinding
	}
	if oneContour(p) {
		dir := p.ComputeFirstDirection()
		if dir != path.FirstDirectionUnknown {
			if dir == path.FirstDirectionCW {
				p = reversePath(p)
			}
			p.SetFillType(fillType)
			return p, true
		}
	}
	head := &opContourHead{}
	globalState := newOpGlobalState(head)
	builder := newOpEdgeBuilder(p, head, globalState)
	if builder.unparseable || !builder.finish() {
		return p, false
	}
	if head.count == 0 {
		return p, true
	}
	if head.next == nil {
		return p, false
	}
	head.joinAllSegments()
	head.resetReverse()
	writePath := false
	globalState.setPhase(opPhaseFixWinding)
	for {
		topSpan := findSortableTop(head)
		if topSpan == nil {
			break
		}
		topContour := topSpan.segment.contour
		// A contour's nesting parity (odd = inside an odd number of other contours) must match its ray-cast winding
		// orientation; if it disagrees, reverse it so the winding rule fills consistently.
		if (globalState.nested & 1) != b2i(topContour.isCcw() != 0) {
			topContour.setReverse()
			writePath = true
		}
		topContour.markAllDone()
		globalState.clearNested()
	}
	if !writePath {
		p.SetFillType(fillType)
		return p, true
	}
	woundPath := newPathWriter(fillType)
	for test := head.listHead(); test != nil; test = test.next {
		if test.count == 0 {
			continue
		}
		if test.reversed() {
			test.toReversePath(woundPath)
		} else {
			test.toPath(woundPath)
		}
	}
	return woundPath.nativePath(), true
}
