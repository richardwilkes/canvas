// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The non-antialiased path scan converter (plus its linked-list helpers): walks a sorted edge list one scanline at a
// time, blitting spans between edge crossings per the fill rule.

package raster

import (
	"math"
	"slices"
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

const (
	edgeHeadY = int32(math.MinInt32)
	edgeTailY = int32(math.MaxInt32)
)

///////////////////////////////////////////////////////////////////////////////
// Doubly-linked edge list helpers

func removeEdge(edge *Edge) {
	edge.Prev.Next = edge.Next
	edge.Next.Prev = edge.Prev
}

func insertEdgeAfter(edge, afterMe *Edge) {
	edge.Prev = afterMe
	edge.Next = afterMe.Next
	afterMe.Next.Prev = edge
	afterMe.Next = edge
}

// backwardInsertEdgeBasedOnX ripples the edge backwards until it is x-sorted again.
func backwardInsertEdgeBasedOnX(edge *Edge) {
	x := edge.X
	prev := edge.Prev
	for prev.Prev != nil && prev.X > x {
		prev = prev.Prev
	}
	if prev.Next != edge {
		removeEdge(edge)
		insertEdgeAfter(edge, prev)
	}
}

// backwardInsertStart, from prev, searches backwards for the point to begin a new edge insertion.
func backwardInsertStart(prev *Edge, x Fixed) *Edge {
	for prev.Prev != nil && prev.X > x {
		prev = prev.Prev
	}
	return prev
}

// insertNewEdges: newEdge is the first edge with FirstY >= currY; insert all edges starting at currY into x-sorted
// position.
func insertNewEdges(newEdge *Edge, currY int32) {
	if newEdge.FirstY != currY {
		return
	}
	prev := newEdge.Prev
	if prev.X <= newEdge.X {
		return
	}
	// find first x pos to insert
	start := backwardInsertStart(prev, newEdge.X)
	// insert the lot, fixing up the links as we go
	for {
		next := newEdge.Next
		inserted := false
		for {
			if start.Next == newEdge {
				inserted = true
				break
			}
			after := start.Next
			if after.X >= newEdge.X {
				break
			}
			start = after
		}
		if !inserted {
			removeEdge(newEdge)
			insertEdgeAfter(newEdge, start)
		}
		start = newEdge
		newEdge = next
		if newEdge.FirstY != currY {
			break
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

// prePostProc is called at the start (isStart true) and end of each scanline.
type prePostProc func(y int32, isStart bool)

// walkEdges is the general winding/even-odd walker.
func walkEdges(prevHead *Edge, fillType path.FillType, blitter Blitter, startY, stopY int32, proc prePostProc, rightClip int32) {
	currY := startY
	var windingMask int
	if fillType.IsEvenOdd() {
		windingMask = 1
	} else {
		windingMask = -1
	}

	for {
		w := 0
		var left int32
		currE := prevHead.Next
		prevX := prevHead.X

		if proc != nil {
			proc(currY, true) // pre-proc
		}

		for currE.FirstY <= currY {
			x := FixedRoundToInt(currE.X)

			if w&windingMask == 0 { // we're starting interval
				left = x
			}

			w += int(currE.Winding)

			if w&windingMask == 0 { // we finished an interval
				if width := x - left; width > 0 {
					blitter.BlitH(left, currY, width)
				}
			}

			next := currE.Next
			var newX Fixed

			if currE.LastY == currY { // are we done with this edge?
				resort := false
				if currE.hasNextSegment() && currE.nextSegment() {
					newX = currE.X
					resort = true
				} else {
					removeEdge(currE)
				}
				if resort {
					if newX < prevX { // ripple currE backwards until it is x-sorted
						backwardInsertEdgeBasedOnX(currE)
					} else {
						prevX = newX
					}
				}
			} else {
				newX = currE.X + currE.DxDy
				currE.X = newX
				if newX < prevX { // ripple currE backwards until it is x-sorted
					backwardInsertEdgeBasedOnX(currE)
				} else {
					prevX = newX
				}
			}
			currE = next
		}

		if w&windingMask != 0 { // was our right-edge culled away?
			if width := rightClip - left; width > 0 {
				blitter.BlitH(left, currY, width)
			}
		}

		if proc != nil {
			proc(currY, false) // post-proc
		}

		currY++
		if currY >= stopY {
			break
		}
		// now currE points to the first edge with a Yint larger than currY
		insertNewEdges(currE, currY)
	}
}

// updateEdge returns true if we're NOT done with this edge.
func updateEdge(edge *Edge, lastY int32) bool {
	if lastY != edge.LastY {
		return true
	}
	if !edge.hasNextSegment() {
		return false
	}
	return edge.nextSegment()
}

// walkSimpleEdges is the two-edge (left/right) walker for convex fills; it needs Y to only change once (looser than
// convex in X). The malformed-input checks below are silent returns rather than panics.
func walkSimpleEdges(prevHead *Edge, blitter Blitter, startY, stopY int32) {
	leftE := prevHead.Next
	riteE := leftE.Next
	currE := riteE.Next

	// our edge choppers for curves can result in the initial edges not lining up, so we take the max.
	localTop := max(leftE.FirstY, riteE.FirstY)
	if localTop < startY {
		return
	}

	for localTop < stopY {
		localBot := min(leftE.LastY, riteE.LastY)
		localBot = min(localBot, stopY-1)
		if localTop > localBot {
			return
		}

		left := leftE.X
		dLeft := leftE.DxDy
		rite := riteE.X
		dRite := riteE.DxDy
		count := localBot - localTop
		if count < 0 {
			return
		}

		if dLeft == 0 && dRite == 0 {
			l := FixedRoundToInt(left)
			r := FixedRoundToInt(rite)
			if l > r {
				l, r = r, l
			}
			if l < r {
				count++
				blitter.BlitRect(l, localTop, r-l, count)
			}
			localTop = localBot + 1
		} else {
			for {
				l := FixedRoundToInt(left)
				r := FixedRoundToInt(rite)
				if l > r {
					l, r = r, l
				}
				if l < r {
					blitter.BlitH(l, localTop, r-l)
				}
				// Either/both of these adds might overflow, since we perform this step even if (later) we determine
				// that we are done with the edge (see updateEdge). Wrapping on overflow is intentional and harmless
				// here.
				left += dLeft
				rite += dRite
				localTop++
				count--
				if count < 0 {
					break
				}
			}
		}

		leftE.X = left
		riteE.X = rite

		if !updateEdge(leftE, localBot) {
			if currE.FirstY >= stopY {
				return // we're done
			}
			leftE = currE
			currE = currE.Next
			if leftE.FirstY != localTop {
				return
			}
		}
		if !updateEdge(riteE, localBot) {
			if currE.FirstY >= stopY {
				return // we're done
			}
			riteE = currE
			currE = currE.Next
			if riteE.FirstY != localTop {
				return
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

// inverseBlitter overrides BlitH, calling its proxy blitter with the inverse of the spans it is given (clipped to the
// left/right of the clip rect). Used to implement inverse fill types.
type inverseBlitter struct {
	blitter Blitter
	firstX  int32
	lastX   int32
	prevX   int32
}

func (ib *inverseBlitter) setBlitter(blitter Blitter, clip geom.IRect) {
	ib.blitter = blitter
	ib.firstX = clip.Left
	ib.lastX = clip.Right
}

func (ib *inverseBlitter) prepost(y int32, isStart bool) {
	if isStart {
		ib.prevX = ib.firstX
	} else {
		if invWidth := ib.lastX - ib.prevX; invWidth > 0 {
			ib.blitter.BlitH(ib.prevX, y, invWidth)
		}
	}
}

// BlitH implements Blitter.
func (ib *inverseBlitter) BlitH(x, y, width int32) {
	if invWidth := x - ib.prevX; invWidth > 0 {
		ib.blitter.BlitH(ib.prevX, y, invWidth)
	}
	ib.prevX = x + width
}

// The remaining entry points are never called on an inverse blitter.

// BlitAntiH implements Blitter.
func (ib *inverseBlitter) BlitAntiH(_, _ int32, _ []Alpha, _ []int16) {}

// BlitV implements Blitter.
func (ib *inverseBlitter) BlitV(_, _, _ int32, _ Alpha) {}

// BlitRect implements Blitter.
func (ib *inverseBlitter) BlitRect(_, _, _, _ int32) {}

// BlitAntiRect implements Blitter.
func (ib *inverseBlitter) BlitAntiRect(_, _, _, _ int32, _, _ Alpha) {}

// BlitMask implements Blitter.
func (ib *inverseBlitter) BlitMask(_ *Mask, _ geom.IRect) {}

// BlitAntiH2 implements Blitter.
func (ib *inverseBlitter) BlitAntiH2(_, _ int32, _, _ Alpha) {}

// BlitAntiV2 implements Blitter.
func (ib *inverseBlitter) BlitAntiV2(_, _ int32, _, _ Alpha) {}

///////////////////////////////////////////////////////////////////////////////

// sortEdges sorts by (FirstY, X) and links the list, returning first and last. slices.SortFunc rather than sort.Slice
// avoids sort.Slice's per-call reflect Swapper allocation.
func sortEdges(list []*Edge) (first, last *Edge) {
	slices.SortFunc(list, func(a, b *Edge) int {
		switch {
		case a.FirstY < b.FirstY:
			return -1
		case a.FirstY > b.FirstY:
			return 1
		case a.X < b.X:
			return -1
		case a.X > b.X:
			return 1
		default:
			return 0
		}
	})
	for i := 1; i < len(list); i++ {
		list[i-1].Next = list[i]
		list[i].Prev = list[i-1]
	}
	return list[0], list[len(list)-1]
}

// fillPath builds the edge list for p and walks it, blitting into blitter over [startY, stopY).
func fillPath(p *path.Path, clipRect geom.IRect, blitter Blitter, startY, stopY int32, pathContainedInClip bool) {
	builder := getBasicEdgeBuilder()
	defer putBasicEdgeBuilder(builder)
	var clipArg *geom.IRect
	if !pathContainedInClip {
		clipArg = &clipRect
	}
	count := builder.buildEdges(p, clipArg)
	list := builder.list

	if count == 0 {
		if p.IsInverseFillType() {
			// Since we are in inverse-fill, our caller has already drawn above our top (startY) and will draw below our
			// bottom (stopY). Thus we need to restrict our drawing to the intersection of the clip and those two
			// limits.
			rect := clipRect
			if rect.Top < startY {
				rect.Top = startY
			}
			if rect.Bottom > stopY {
				rect.Bottom = stopY
			}
			if !rect.IsEmpty() {
				blitter.BlitRect(rect.Left, rect.Top, rect.Width(), rect.Height())
			}
		}
		return
	}

	// The sentinel edges live on the pooled builder — their addresses enter the linked list, which would heap-allocate
	// them as locals on every fill.
	headEdge := &builder.head
	tailEdge := &builder.tail
	// this returns the first and last edge after they're sorted into a dlink list
	edge, last := sortEdges(list)

	headEdge.Prev = nil
	headEdge.Next = edge
	headEdge.FirstY = edgeHeadY
	headEdge.X = Fixed(math.MinInt32)
	edge.Prev = headEdge

	tailEdge.Prev = last
	tailEdge.Next = nil
	tailEdge.FirstY = edgeTailY
	last.Next = tailEdge

	// now edge is the head of the sorted linklist
	if !pathContainedInClip && startY < clipRect.Top {
		startY = clipRect.Top
	}
	if !pathContainedInClip && stopY > clipRect.Bottom {
		stopY = clipRect.Bottom
	}

	var proc prePostProc

	if p.IsInverseFillType() {
		// The inverse-blitter wrapper lives on the pooled builder — its address enters the blitter chain, which would
		// heap-allocate it as a local on every fill.
		ib := &builder.inv
		ib.setBlitter(blitter, clipRect)
		blitter = ib
		proc = ib.prepost
	}

	// count >= 2 is required as the convex walker does not handle missing right edges
	if p.IsConvex() && proc == nil && count >= 2 {
		walkSimpleEdges(headEdge, blitter, startY, stopY)
	} else {
		walkEdges(headEdge, p.FillType(), blitter, startY, stopY, proc, clipRect.Right)
	}
}

// blitAboveClip blits the portion of clip above avoid (for inverse fills).
func blitAboveClip(blitter Blitter, avoid geom.IRect, clip *Region) {
	cr := clip.Bounds()
	tmp := geom.IRectLTRB(cr.Left, cr.Top, cr.Right, avoid.Top)
	if !tmp.IsEmpty() {
		blitRectRegion(blitter, tmp, clip)
	}
}

// blitBelowClip blits the portion of clip below avoid (for inverse fills).
func blitBelowClip(blitter Blitter, avoid geom.IRect, clip *Region) {
	cr := clip.Bounds()
	tmp := geom.IRectLTRB(cr.Left, avoid.Bottom, cr.Right, cr.Bottom)
	if !tmp.IsEmpty() {
		blitRectRegion(blitter, tmp, clip)
	}
}

///////////////////////////////////////////////////////////////////////////////

// scanClipper decides whether the blitter needs a clipping wrapper and whether the scan converter still needs the clip
// rect. It is initialized in place by init (rather than constructed and returned by value) because blitter may point at
// rectBlitter/rgnBlitter within the same scanClipper, so the storage must be caller-owned and stay at a stable address
// for the fill's duration. needClipRect is a plain bool (not a pointer whose value is never read, only its presence) so
// the clip bounds don't escape to the heap.
type scanClipper struct {
	rgnBlitter   RgnClipBlitter
	blitter      Blitter // nil means blit nothing
	rectBlitter  RectClipBlitter
	needClipRect bool // true when the scan converter must still clip to the clip's rect bounds
}

// init sets up the clipper, filling c in place.
func (c *scanClipper) init(blitter Blitter, clip *Region, ir geom.IRect, skipRejectTest, irPreClipped bool) {
	*c = scanClipper{}
	if clip == nil {
		c.blitter = blitter
		return
	}
	bounds := clip.Bounds()
	c.needClipRect = true
	if !skipRejectTest && !bounds.Intersects(ir) { // completely clipped out
		return
	}
	if clip.IsRect() {
		if !irPreClipped && bounds.ContainsRect(ir) {
			c.needClipRect = false
		} else if irPreClipped || bounds.Left > ir.Left || bounds.Right < ir.Right {
			// only need a wrapper blitter if we're horizontally clipped
			c.rectBlitter.Init(blitter, bounds)
			blitter = &c.rectBlitter
		}
	} else {
		c.rgnBlitter.Init(blitter, clip)
		blitter = &c.rgnBlitter
	}
	c.blitter = blitter
}

// scanClipper.blitter can be a self-referential pointer into one of the receiver's own embedded wrapper blitters
// (&c.rectBlitter / &c.rgnBlitter), so the struct is forced to the heap whenever the caller reads that field back and
// hands it to the (non-inlined) scan converter. Pooling keeps every path fill from allocating one; a sync.Pool (rather
// than a single shared instance) keeps concurrent FillPathParallel bands independent. Each clipper is fully consumed by
// the synchronous fill before putScanClipper returns it.
var scanClipperPool = sync.Pool{New: func() any { return new(scanClipper) }}

func getScanClipper(blitter Blitter, clip *Region, ir geom.IRect, skipRejectTest, irPreClipped bool) *scanClipper {
	c := scanClipperPool.Get().(*scanClipper)
	c.init(blitter, clip, ir, skipRejectTest, irPreClipped)
	return c
}

func putScanClipper(c *scanClipper) {
	*c = scanClipper{} // drop the external device-blitter and region references before reuse
	scanClipperPool.Put(c)
}

///////////////////////////////////////////////////////////////////////////////

// kConservativeRoundBias nudges float-to-int rect rounding a little larger, so we don't "think" a path's bounds are
// inside a clip when (due to numeric drift in the scan converter) we might walk beyond the predicted limits. Value
// determined by trial and error; cubics are the main reason for the slop.
const kConservativeRoundBias = 0.5 + 1.5/float64(FDot6One)

// doubleSaturateToInt converts x to int32 with saturation.
func doubleSaturateToInt(x float64) int32 {
	if !(x < math.MaxInt32) {
		return math.MaxInt32
	}
	if !(x > math.MinInt32) {
		return math.MinInt32
	}
	return int32(x)
}

// roundDownToInt is used for a rect's top/left, biased so the rounded int is smaller than a normal round would give (a
// more conservative, larger int-bounds).
func roundDownToInt(x float32) int32 {
	xx := float64(x)
	xx -= kConservativeRoundBias
	return doubleSaturateToInt(math.Ceil(xx))
}

// roundUpToInt is used for a rect's right/bottom.
func roundUpToInt(x float32) int32 {
	xx := float64(x)
	xx += kConservativeRoundBias
	return doubleSaturateToInt(math.Floor(xx))
}

// conservativeRoundToInt rounds src outward, conservatively, to an integer rect.
func conservativeRoundToInt(src geom.Rect) geom.IRect {
	return geom.IRectLTRB(
		roundDownToInt(src.Left),
		roundDownToInt(src.Top),
		roundUpToInt(src.Right),
		roundUpToInt(src.Bottom),
	)
}

// makeLargeS32 returns a rect of ±(1<<29), which round-trips float Rect <-> IRect safely.
func makeLargeS32() geom.Rect {
	const large = 1 << 29
	return geom.RectLTRB(-large, -large, large, large)
}

// FillPath fills the path, without antialiasing, into blitter, restricted to a rectangular clip (origClip).
func FillPath(p *path.Path, origClip geom.IRect, blitter Blitter) {
	FillPathRegion(p, NewRegionRect(origClip), blitter)
}

// clipToLimitRegion trims the clip so coordinates fit Fixed; returns the (possibly reduced) clip and whether reduction
// occurred.
func clipToLimitRegion(orig *Region) (*Region, bool) {
	const limit = 32767 >> 1
	limitR := geom.IRectLTRB(-limit, -limit, limit, limit)
	if limitR.ContainsRect(orig.Bounds()) {
		return orig, false
	}
	reduced := &Region{}
	reduced.OpRegion(orig, NewRegionRect(limitR), RegionIntersect)
	return reduced, true
}

// PathRequiresTiling reports whether bounds exceeds the scan converter's fixed-point coordinate range, requiring the
// caller to tile the fill.
func PathRequiresTiling(bounds geom.IRect) bool {
	_, reduced := clipToLimitRegion(NewRegionRect(bounds))
	return reduced
}

// FillPathRegion fills the path, without antialiasing, into blitter, restricted to a region clip (origClip).
func FillPathRegion(p *path.Path, origClip *Region, blitter Blitter) {
	// Our edges are fixed-point, and don't like the bounds of the clip to exceed that. Here we trim the clip just so we
	// don't overflow later on.
	clip, reduced := clipToLimitRegion(origClip)
	if reduced && clip.IsEmpty() {
		return
	}

	bounds := p.Bounds()
	irPreClipped := false
	if !makeLargeS32().ContainsRect(bounds) {
		if !bounds.Intersect(makeLargeS32()) {
			bounds = geom.Rect{}
		}
		irPreClipped = true
	}

	ir := conservativeRoundToInt(bounds)
	if ir.IsEmpty() {
		if p.IsInverseFillType() {
			blitRegion(blitter, clip)
		}
		return
	}

	clipper := getScanClipper(blitter, clip, ir, p.IsInverseFillType(), irPreClipped)
	defer putScanClipper(clipper)

	blitter = clipper.blitter
	if blitter == nil {
		return
	}
	// we have to keep our calls to blitter in sorted order, so we must blit the above section first, then the middle,
	// then the bottom.
	if p.IsInverseFillType() {
		blitAboveClip(blitter, ir, clip)
	}
	fillPath(p, clip.Bounds(), blitter, ir.Top, ir.Bottom, !clipper.needClipRect)
	if p.IsInverseFillType() {
		blitBelowClip(blitter, ir, clip)
	}
}
