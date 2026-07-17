// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file implements a clip element deque with incremental finite-bound and gen-ID tracking, covering the subset of
// clip operations the PDF device needs. The canvas Device interface only ever clips with rects and paths
// (ClipRect/ClipPath/ClipRegion→path/ReplaceClip), so this only carries empty/rect/path elements and the
// intersect/difference/replace ops; rounded-rect and clip-shader element kinds are unreachable here and are not
// represented, and an oval clip stays a path element rather than being special-cased — observably identical for PDF,
// whose emitter lowers every non-rect element to a device-space path anyway. Path-element conservative containment
// (used only by quickContains's under-reporting fast path) returns false, the safe conservative answer the API contract
// explicitly permits.

package pdf

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// Reserved generation IDs for special clip states: invalid (not yet assigned), empty, and wide-open (unclipped).
const (
	invalidGenID  uint32 = 0
	emptyGenID    uint32 = 1
	wideOpenGenID uint32 = 2
)

// nextGenID is the atomic source of fresh generation IDs; 0-2 are reserved for the special states above.
var nextGenID atomic.Uint32

func init() { nextGenID.Store(3) }

// getNextGenID returns a fresh generation ID, skipping the reserved low values.
func getNextGenID() uint32 {
	for {
		id := nextGenID.Add(1) - 1
		if id >= 3 {
			return id
		}
	}
}

// clipBoundsType classifies how a clipElement's finiteBound should be interpreted.
type clipBoundsType uint8

const (
	// normalBounds: the finite bound encloses all writeable pixels.
	normalBounds clipBoundsType = iota
	// insideOutBounds: the finite bound encloses the un-writeable pixels; the true bound is infinite.
	insideOutBounds
)

// clipDeviceType is the shape kind stored by a clipElement (rounded-rect and shader clip kinds are not represented
// here; see the file comment above).
type clipDeviceType uint8

const (
	clipEmpty clipDeviceType = iota
	clipRect
	clipPath
)

// clipElement is one entry in the clip stack: a rect or path clip, its combine op, and cached finite-bound/gen-ID state
// used to answer bounds and containment queries in O(1).
type clipElement struct {
	deviceSpacePath       *path.Path // valid when deviceSpaceType == clipPath
	deviceSpaceRect       geom.Rect  // valid when deviceSpaceType == clipRect
	finiteBound           geom.Rect
	saveCount             int
	op                    raster.ClipOp
	deviceSpaceType       clipDeviceType
	finiteBoundType       clipBoundsType
	genID                 uint32
	doAA                  bool
	isReplace             bool
	isIntersectionOfRects bool
}

// newRectElement creates a clip element for a rectangle clip at the given save level.
func newRectElement(saveCount int, rect geom.Rect, m *geom.Matrix, op raster.ClipOp, doAA bool) *clipElement {
	e := &clipElement{}
	e.initRect(saveCount, rect, m, op, doAA)
	return e
}

// newPathElement creates a clip element for a path clip at the given save level.
func newPathElement(saveCount int, p *path.Path, m *geom.Matrix, op raster.ClipOp, doAA bool) *clipElement {
	e := &clipElement{}
	e.initPath(saveCount, p, m, op, doAA)
	return e
}

// newReplaceRectElement creates a clip element that replaces the entire clip stack with a single rectangle, rather than
// combining with what came before.
func newReplaceRectElement(saveCount int, rect geom.Rect, doAA bool) *clipElement {
	e := &clipElement{}
	e.deviceSpaceRect = rect
	e.deviceSpaceType = clipRect
	e.initCommon(saveCount, raster.ClipIntersect, doAA)
	e.isReplace = true
	return e
}

func (e *clipElement) initCommon(saveCount int, op raster.ClipOp, doAA bool) {
	e.saveCount = saveCount
	e.op = op
	e.doAA = doAA
	e.isReplace = false
	// Default inside-out + empty bounds: nothing is known to be outside the clip.
	e.finiteBoundType = insideOutBounds
	e.finiteBound = geom.Rect{}
	e.isIntersectionOfRects = false
	e.genID = invalidGenID
}

func (e *clipElement) initRect(saveCount int, rect geom.Rect, m *geom.Matrix, op raster.ClipOp, doAA bool) {
	if m.RectStaysRect() {
		devRect, _ := m.MapRect(rect)
		e.deviceSpaceRect = devRect
		e.deviceSpaceType = clipRect
		e.initCommon(saveCount, op, doAA)
		return
	}
	rp := (&path.Path{}).AddRect(rect, geom.DirectionCW)
	e.initAsPath(saveCount, rp, m, op, doAA)
}

func (e *clipElement) initPath(saveCount int, p *path.Path, m *geom.Matrix, op raster.ClipOp, doAA bool) {
	if !p.IsInverseFillType() {
		if r, ok := p.IsRect(); ok {
			e.initRect(saveCount, r, m, op, doAA)
			return
		}
	}
	e.initAsPath(saveCount, p, m, op, doAA)
}

func (e *clipElement) initAsPath(saveCount int, p *path.Path, m *geom.Matrix, op raster.ClipOp, doAA bool) {
	dp := p.Clone()
	dp.Transform(m)
	e.deviceSpacePath = dp
	e.deviceSpaceType = clipPath
	e.initCommon(saveCount, op, doAA)
}

// setEmpty turns the element into an empty (nothing-visible) clip.
func (e *clipElement) setEmpty() {
	e.deviceSpaceType = clipEmpty
	e.finiteBound = geom.Rect{}
	e.finiteBoundType = normalBounds
	e.isIntersectionOfRects = false
	e.deviceSpaceRect = geom.Rect{}
	e.deviceSpacePath = nil
	e.genID = emptyGenID
}

// isInverseFilled reports whether the element is a path clip with inverse fill (it excludes its interior rather than
// including it).
func (e *clipElement) isInverseFilled() bool {
	return e.deviceSpaceType == clipPath && e.deviceSpacePath.IsInverseFillType()
}

// bounds returns the element's device-space bounding rect.
func (e *clipElement) bounds() geom.Rect {
	switch e.deviceSpaceType {
	case clipRect:
		return e.deviceSpaceRect
	case clipPath:
		return e.deviceSpacePath.Bounds()
	default:
		return geom.Rect{}
	}
}

// contains reports whether rect lies entirely within the element. Path elements always answer conservatively (false),
// which is the safe under-report the quickContains contract permits.
func (e *clipElement) contains(rect geom.Rect) bool {
	switch e.deviceSpaceType {
	case clipRect:
		return e.deviceSpaceRect.ContainsRect(rect)
	default:
		return false
	}
}

// asDeviceSpacePath returns the element's shape as a device-space path, converting a rect element to an equivalent
// rectangular path.
func (e *clipElement) asDeviceSpacePath() *path.Path {
	switch e.deviceSpaceType {
	case clipRect:
		return (&path.Path{}).AddRect(e.deviceSpaceRect, geom.DirectionCW)
	case clipPath:
		return e.deviceSpacePath
	default:
		return &path.Path{}
	}
}

// canBeIntersectedInPlace reports whether a new clip op at the given save level can be folded into this element in
// place, rather than pushing a new element onto the stack.
func (e *clipElement) canBeIntersectedInPlace(saveCount int, op raster.ClipOp) bool {
	if e.deviceSpaceType == clipEmpty && (op == raster.ClipDifference || op == raster.ClipIntersect) {
		return true
	}
	return e.saveCount == saveCount && op == raster.ClipIntersect &&
		(e.op == raster.ClipIntersect || e.isReplace)
}

// rectRectIntersectAllowed reports whether this rect element can be intersected in place with newR without changing
// antialiasing behavior at the edges.
func (e *clipElement) rectRectIntersectAllowed(newR geom.Rect, newAA bool) bool {
	if e.doAA == newAA {
		return true
	}
	if !e.deviceSpaceRect.Intersects(newR) {
		return true
	}
	if e.deviceSpaceRect.ContainsRect(newR) {
		return true
	}
	return false
}

// Combination codes describing how the current and previous elements' fill types (normal vs. inside-out) pair up, used
// to pick the right finite-bound update rule.
const (
	prevCurCombo       = 0
	prevInvCurCombo    = 1
	invPrevCurCombo    = 2
	invPrevInvCurCombo = 3
)

// combineBoundsDiff updates e's finite bound for a difference (subtract) op, given the combination code and the
// previous element's finite bound.
func (e *clipElement) combineBoundsDiff(combination int, prevFinite geom.Rect) {
	switch combination {
	case invPrevInvCurCombo:
		e.finiteBoundType = normalBounds
	case invPrevCurCombo:
		e.finiteBound.Join(prevFinite)
		e.finiteBoundType = insideOutBounds
	case prevInvCurCombo:
		if !e.finiteBound.Intersect(prevFinite) {
			e.finiteBound = geom.Rect{}
			e.genID = emptyGenID
		}
		e.finiteBoundType = normalBounds
	case prevCurCombo:
		e.finiteBound = prevFinite
	}
}

// combineBoundsIntersection updates e's finite bound for an intersect op, given the combination code and the previous
// element's finite bound.
func (e *clipElement) combineBoundsIntersection(combination int, prevFinite geom.Rect) {
	switch combination {
	case invPrevInvCurCombo:
		e.finiteBound.Join(prevFinite)
		e.finiteBoundType = insideOutBounds
	case invPrevCurCombo:
		// Only the current clip remains writeable; nothing to do.
	case prevInvCurCombo:
		e.finiteBound = prevFinite
		e.finiteBoundType = normalBounds
	case prevCurCombo:
		if !e.finiteBound.Intersect(prevFinite) {
			e.setEmpty()
		}
	}
}

// updateBoundAndGenID assigns e a fresh generation ID and recomputes its finite bound from its own shape combined with
// the prior element's bound (prior is nil if e is the first element).
func (e *clipElement) updateBoundAndGenID(prior *clipElement) {
	e.genID = getNextGenID()
	e.isIntersectionOfRects = false
	switch e.deviceSpaceType {
	case clipRect:
		e.finiteBound = e.deviceSpaceRect
		e.finiteBoundType = normalBounds
		if e.isReplace ||
			(e.op == raster.ClipIntersect && prior == nil) ||
			(e.op == raster.ClipIntersect && prior.isIntersectionOfRects &&
				prior.rectRectIntersectAllowed(e.deviceSpaceRect, e.doAA)) {
			e.isIntersectionOfRects = true
		}
	case clipPath:
		e.finiteBound = e.deviceSpacePath.Bounds()
		if e.deviceSpacePath.IsInverseFillType() {
			e.finiteBoundType = insideOutBounds
		} else {
			e.finiteBoundType = normalBounds
		}
	case clipEmpty:
		// Should not be reached with an empty element.
	}

	var prevFinite geom.Rect
	var prevType clipBoundsType
	if prior == nil {
		prevFinite = geom.Rect{}
		prevType = insideOutBounds
	} else {
		prevFinite = prior.finiteBound
		prevType = prior.finiteBoundType
	}

	combination := prevCurCombo
	if e.finiteBoundType == insideOutBounds {
		combination |= 0x01
	}
	if prevType == insideOutBounds {
		combination |= 0x02
	}

	if !e.isReplace {
		switch e.op {
		case raster.ClipDifference:
			e.combineBoundsDiff(combination, prevFinite)
		case raster.ClipIntersect:
			e.combineBoundsIntersection(combination, prevFinite)
		}
	}
}

// ClipStack is a stack of clip elements with save/restore semantics, tracking enough state to answer bounds and
// containment queries without rasterizing.
type ClipStack struct {
	deque     []*clipElement
	saveCount int
}

// NewClipStack returns an empty (wide-open) clip stack.
func NewClipStack() *ClipStack { return &ClipStack{} }

func (cs *ClipStack) back() *clipElement {
	if len(cs.deque) == 0 {
		return nil
	}
	return cs.deque[len(cs.deque)-1]
}

// SaveCount returns the current save level.
func (cs *ClipStack) SaveCount() int { return cs.saveCount }

// Save increments the save level; clips pushed after this call are discarded by the matching Restore.
func (cs *ClipStack) Save() { cs.saveCount++ }

// Restore pops the save level and discards any clip elements pushed since the matching Save.
func (cs *ClipStack) Restore() {
	cs.saveCount--
	cs.restoreTo(cs.saveCount)
}

func (cs *ClipStack) restoreTo(saveCount int) {
	for len(cs.deque) > 0 {
		e := cs.back()
		if e.saveCount <= saveCount {
			break
		}
		cs.deque = cs.deque[:len(cs.deque)-1]
	}
}

// ClipRect pushes a rectangle clip, combined with the current clip stack using op.
func (cs *ClipStack) ClipRect(rect geom.Rect, matrix *geom.Matrix, op raster.ClipOp, doAA bool) {
	cs.pushElement(newRectElement(cs.saveCount, rect, matrix, op, doAA))
}

// ClipPath pushes a path clip, combined with the current clip stack using op.
func (cs *ClipStack) ClipPath(p *path.Path, matrix *geom.Matrix, op raster.ClipOp, doAA bool) {
	cs.pushElement(newPathElement(cs.saveCount, p, matrix, op, doAA))
}

// ReplaceClip discards the current clip stack and replaces it with a single rectangle clip.
func (cs *ClipStack) ReplaceClip(devRect geom.Rect, doAA bool) {
	cs.pushElement(newReplaceRectElement(cs.saveCount, devRect, doAA))
}

// pushElement adds element to the stack, folding it into the top element in place when possible (e.g. intersecting two
// rects at the same save level) instead of growing the stack.
func (cs *ClipStack) pushElement(element *clipElement) {
	var prior *clipElement
	priorIdx := len(cs.deque) - 1
	if priorIdx >= 0 {
		prior = cs.deque[priorIdx]
	}
	if prior != nil {
		if element.isReplace {
			cs.restoreTo(cs.saveCount - 1)
			prior = cs.back()
		} else if prior.canBeIntersectedInPlace(cs.saveCount, element.op) {
			switch prior.deviceSpaceType {
			case clipEmpty:
				return
			case clipRect:
				if element.deviceSpaceType == clipRect {
					if prior.rectRectIntersectAllowed(element.deviceSpaceRect, element.doAA) {
						isectRect := prior.deviceSpaceRect
						if !isectRect.Intersect(element.deviceSpaceRect) {
							prior.setEmpty()
							return
						}
						prior.deviceSpaceRect = isectRect
						prior.doAA = element.doAA
						prior.updateBoundAndGenID(cs.priorOf(priorIdx))
						return
					}
				} else {
					if !prior.bounds().Intersects(element.bounds()) {
						prior.setEmpty()
						return
					}
				}
			default:
				if !prior.bounds().Intersects(element.bounds()) {
					prior.setEmpty()
					return
				}
			}
		}
	}
	cs.deque = append(cs.deque, element)
	element.updateBoundAndGenID(prior)
}

// priorOf returns the element below index idx (deque back's predecessor), or nil.
func (cs *ClipStack) priorOf(idx int) *clipElement {
	if idx-1 < 0 {
		return nil
	}
	return cs.deque[idx-1]
}

// getBounds returns the topmost element's finite bound, its interpretation, and whether it is known to be a plain rect
// intersection (a wide-open bound if the stack is empty).
func (cs *ClipStack) getBounds() (bound geom.Rect, boundType clipBoundsType, isIntersectionOfRects bool) {
	e := cs.back()
	if e == nil {
		return geom.Rect{}, insideOutBounds, false
	}
	return e.finiteBound, e.finiteBoundType, e.isIntersectionOfRects
}

// bounds returns the clip's bounding rect clamped to deviceBounds.
func (cs *ClipStack) bounds(deviceBounds geom.IRect) geom.Rect {
	r, bt, _ := cs.getBounds()
	db := deviceBounds.ToRect()
	if bt == insideOutBounds {
		return db
	}
	if r.Intersect(db) {
		return r
	}
	return geom.Rect{}
}

// isEmpty reports whether the clip excludes every pixel within deviceBounds.
func (cs *ClipStack) isEmpty(deviceBounds geom.IRect) bool { return cs.bounds(deviceBounds).IsEmpty() }

// getTopmostGenID returns the generation ID of the current clip state, or wideOpenGenID if nothing is clipped.
func (cs *ClipStack) getTopmostGenID() uint32 {
	e := cs.back()
	if e == nil {
		return wideOpenGenID
	}
	if e.finiteBoundType == insideOutBounds && e.finiteBound.IsEmpty() {
		return wideOpenGenID
	}
	return e.genID
}

// isWideOpen reports whether the clip stack currently excludes nothing.
func (cs *ClipStack) isWideOpen() bool { return cs.getTopmostGenID() == wideOpenGenID }

// quickContains conservatively reports whether devRect lies entirely within the clip, without rasterizing; false
// negatives are allowed, false positives are not.
func (cs *ClipStack) quickContains(devRect geom.Rect) bool {
	return cs.isWideOpen() || cs.internalQuickContains(devRect)
}

// internalQuickContains implements quickContains's walk over the element stack from top to bottom.
func (cs *ClipStack) internalQuickContains(rect geom.Rect) bool {
	for i := len(cs.deque) - 1; i >= 0; i-- {
		e := cs.deque[i]
		if e.op != raster.ClipIntersect && !e.isReplace {
			return false
		}
		if e.isInverseFilled() {
			if e.bounds().Intersects(rect) {
				return false
			}
		} else if !e.contains(rect) {
			return false
		}
		if e.isReplace {
			break
		}
	}
	return true
}

// isAnyAA reports whether any element in the stack requests antialiasing.
func (cs *ClipStack) isAnyAA() bool {
	for _, e := range cs.deque {
		if e.doAA {
			return true
		}
	}
	return false
}

// elements returns the elements bottom-to-top. Callers must not mutate the returned slice.
func (cs *ClipStack) elements() []*clipElement { return cs.deque }
