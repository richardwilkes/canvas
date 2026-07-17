// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Region: a compressed run-length-encoded set of integer rectangles, the non-AA clip primitive. Region layout (all
// int32 "runs"):
//
//	TOP [ Bottom, IntervalCount, [Left, Right]..., X-Sentinel ]... Y-Sentinel
//
// runs slices are immutable once built and shared freely; every mutation constructs a new slice.

package raster

import (
	"math"
	"slices"

	"github.com/richardwilkes/canvas/geom"
)

// regionSentinel terminates a run list.
const regionSentinel int32 = 0x7FFFFFFF

// rectRegionRuns is the run count for a simple rect: TOP, BOTTOM, INTERVALCOUNT(1), LEFT, RIGHT, X-SENTINEL,
// Y-SENTINEL.
const rectRegionRuns = 7

// RegionOp is the set of boolean ops a Region can be combined with.
type RegionOp uint8

// RegionOp values.
const (
	RegionDifference RegionOp = iota
	RegionIntersect
	RegionUnion
	RegionXor
	RegionReverseDifference
	RegionReplace
)

// regionRunHead holds the run-length-encoded row data backing a complex Region (slices are shared immutably).
type regionRunHead struct {
	runs          []int32
	ySpanCount    int32
	intervalCount int32
}

// Region is a hard-edged (non-AA) clip: a compressed run-length-encoded set of integer rectangles. The zero value is an
// empty region. A Region with a nil head and non-empty bounds is a rectangle; complex regions carry their runs in head.
type Region struct {
	head   *regionRunHead
	bounds geom.IRect
}

// NewRegionRect returns a region set to r.
func NewRegionRect(r geom.IRect) *Region {
	rgn := &Region{}
	rgn.SetRect(r)
	return rgn
}

// IsEmpty reports whether the region contains no pixels.
func (rgn *Region) IsEmpty() bool { return rgn.head == nil && rgn.bounds.IsEmpty() }

// IsRect reports whether the region is a simple rectangle.
func (rgn *Region) IsRect() bool { return rgn.head == nil && !rgn.bounds.IsEmpty() }

// IsComplex reports whether the region is more than a simple rect.
func (rgn *Region) IsComplex() bool { return rgn.head != nil }

// Bounds returns the region's bounding rect.
func (rgn *Region) Bounds() geom.IRect { return rgn.bounds }

// ComputeRegionComplexity returns the number of intervals in the region: 0 for empty, 1 for a rect, or the interval
// count for a complex region.
func (rgn *Region) ComputeRegionComplexity() int {
	switch {
	case rgn.IsEmpty():
		return 0
	case rgn.IsRect():
		return 1
	}
	return int(rgn.head.intervalCount)
}

// SetEmpty clears the region to empty and returns false.
func (rgn *Region) SetEmpty() bool {
	rgn.bounds = geom.IRect{}
	rgn.head = nil
	return false
}

// SetRect sets the region to a simple rectangle.
func (rgn *Region) SetRect(r geom.IRect) bool {
	if r.IsEmpty() || r.Right == regionSentinel || r.Bottom == regionSentinel {
		return rgn.SetEmpty()
	}
	rgn.bounds = r
	rgn.head = nil
	return true
}

// SetRegion sets the region to match src, sharing the runs.
func (rgn *Region) SetRegion(src *Region) bool {
	if rgn != src {
		rgn.bounds = src.bounds
		rgn.head = src.head
	}
	return !rgn.IsEmpty()
}

// buildRectRuns fills runs with the run-length encoding of a single-rect region.
func buildRectRuns(bounds geom.IRect, runs *[rectRegionRuns]int32) {
	runs[0] = bounds.Top
	runs[1] = bounds.Bottom
	runs[2] = 1 // 1 interval for this scanline
	runs[3] = bounds.Left
	runs[4] = bounds.Right
	runs[5] = regionSentinel
	runs[6] = regionSentinel
}

// runsAreARect reports whether runs (of the given count) encode a simple rect, returning it if so.
func runsAreARect(runs []int32, count int) (geom.IRect, bool) {
	if count == rectRegionRuns {
		return geom.IRectLTRB(runs[3], runs[0], runs[4], runs[1]), true
	}
	return geom.IRect{}, false
}

// setRunsTrimmed trims empty top/bottom spans, then adopts the runs into the region. The runs slice is scratch and may
// be modified.
func (rgn *Region) setRunsTrimmed(runs []int32) bool {
	count := len(runs)
	if count <= 2 {
		return rgn.SetEmpty()
	}
	start := 0
	stop := count
	if count > rectRegionRuns {
		if runs[start+3] == regionSentinel { // skip empty initial span
			newTop := runs[start+1] // prev bottom
			start += 3
			runs[start] = newTop
		}
		// now check for a trailing empty span
		if runs[stop-5] == regionSentinel { // eek, stop[-4] was a bottom with no x-runs
			runs[stop-4] = regionSentinel // kill empty last span
			stop -= 3
		}
		count = stop - start
		if count <= 2 {
			return rgn.SetEmpty()
		}
	}

	trimmed := runs[start:stop]
	if bounds, isRect := runsAreARect(trimmed, count); isRect {
		return rgn.SetRect(bounds)
	}

	// if we get here, we need to become a complex region
	head := &regionRunHead{runs: slices.Clone(trimmed)}
	rgn.head = head
	rgn.bounds = head.computeRunBounds()

	// Our computed bounds might be too large, so we have to check here.
	if rgn.bounds.IsEmpty() {
		return rgn.SetEmpty()
	}
	return true
}

// computeRunBounds computes interval counts and bounds from the run data.
func (h *regionRunHead) computeRunBounds() geom.IRect {
	runs := h.runs
	i := 0
	top := runs[i]
	i++

	var bot int32
	ySpanCount := int32(0)
	intervalCount := int32(0)
	left := int32(math.MaxInt32)
	rite := int32(math.MinInt32)

	for {
		bot = runs[i]
		i++
		ySpanCount++

		intervals := runs[i]
		i++
		if intervals > 0 {
			if l := runs[i]; left > l {
				left = l
			}
			i += int(intervals) * 2
			if r := runs[i-1]; rite < r {
				rite = r
			}
			intervalCount += intervals
		}
		i++ // skip x-sentinel

		if runs[i] >= regionSentinel {
			break
		}
	}

	h.ySpanCount = ySpanCount
	h.intervalCount = intervalCount
	return geom.IRectLTRB(left, top, rite, bot)
}

// skipEntireScanline, given the index of a scanline (starting at its Bottom value), returns the index of the next
// scanline.
func skipEntireScanline(runs []int32, i int) int {
	intervals := runs[i+1]
	// skip the entire line [B N [L R] S]
	return i + 1 + 1 + int(intervals)*2 + 1
}

// findScanline returns the index of the scanline containing y (y must be within bounds). The returned index is the
// scanline's Bottom value.
func (h *regionRunHead) findScanline(y int32) int {
	runs := h.runs
	i := 1 // skip top-Y
	for {
		if y < runs[i] {
			return i
		}
		i = skipEntireScanline(runs, i)
	}
}

// ContainsPoint reports whether (x, y) is inside the region.
func (rgn *Region) ContainsPoint(x, y int32) bool {
	if !rgn.bounds.ContainsPoint(x, y) {
		return false
	}
	if rgn.IsRect() {
		return true
	}
	runs := rgn.head.runs
	i := rgn.head.findScanline(y) + 2 // skip Bottom and IntervalCount
	// Just walk this scanline, checking each interval; the x-sentinel appears as a left and aborts.
	for {
		if x < runs[i] {
			return false
		}
		if x < runs[i+1] {
			return true
		}
		i += 2
	}
}

// scanlineBottom returns a scanline's bottom Y value.
func scanlineBottom(runs []int32, i int) int32 { return runs[i] }

// scanlineNext returns the index of the scanline following the one at i.
func scanlineNext(runs []int32, i int) int {
	// skip [B N [L R]... S]
	return i + 2 + int(runs[i+1])*2 + 1
}

// scanlineContains reports whether the scanline at i fully covers [left, rite).
func scanlineContains(runs []int32, i int, left, rite int32) bool {
	i += 2 // skip Bottom and IntervalCount
	for {
		if left < runs[i] {
			return false
		}
		if rite <= runs[i+1] {
			return true
		}
		i += 2
	}
}

// scanlineIntersects reports whether the scanline at i overlaps [left, rite).
func scanlineIntersects(runs []int32, i int, left, rite int32) bool {
	i += 2 // skip Bottom and IntervalCount
	for {
		if rite <= runs[i] {
			return false
		}
		if left < runs[i+1] {
			return true
		}
		i += 2
	}
}

// ContainsRect reports whether r is fully covered by the region.
func (rgn *Region) ContainsRect(r geom.IRect) bool {
	if !rgn.bounds.ContainsRect(r) {
		return false
	}
	if rgn.IsRect() {
		return true
	}
	runs := rgn.head.runs
	i := rgn.head.findScanline(r.Top)
	for {
		if !scanlineContains(runs, i, r.Left, r.Right) {
			return false
		}
		if r.Bottom <= scanlineBottom(runs, i) {
			return true
		}
		i = scanlineNext(runs, i)
	}
}

// ContainsRegion reports whether other is fully covered by the region.
func (rgn *Region) ContainsRegion(other *Region) bool {
	if rgn.IsEmpty() || other.IsEmpty() || !rgn.bounds.ContainsRect(other.bounds) {
		return false
	}
	if rgn.IsRect() {
		return true
	}
	if other.IsRect() {
		return rgn.ContainsRect(other.bounds)
	}
	// A contains B is equivalent to B - A == 0
	return !regionOper(other, rgn, RegionDifference, nil)
}

// QuickContains reports true only when the region is a rect containing r.
func (rgn *Region) QuickContains(r geom.IRect) bool {
	return rgn.IsRect() && !r.IsEmpty() && rgn.bounds.ContainsRect(r)
}

// QuickReject reports whether r is empty or definitely outside the region.
func (rgn *Region) QuickReject(r geom.IRect) bool {
	return rgn.IsEmpty() || r.IsEmpty() || !rgn.bounds.Intersects(r)
}

// Intersects reports whether r overlaps the region.
func (rgn *Region) Intersects(r geom.IRect) bool {
	if rgn.IsEmpty() || r.IsEmpty() {
		return false
	}
	sect := rgn.bounds
	if !sect.Intersect(r) {
		return false
	}
	if rgn.IsRect() {
		return true
	}
	runs := rgn.head.runs
	i := rgn.head.findScanline(sect.Top)
	for {
		if scanlineIntersects(runs, i, sect.Left, sect.Right) {
			return true
		}
		if sect.Bottom <= scanlineBottom(runs, i) {
			return false
		}
		i = scanlineNext(runs, i)
	}
}

// IntersectsRegion reports whether other overlaps the region.
func (rgn *Region) IntersectsRegion(other *Region) bool {
	if rgn.IsEmpty() || other.IsEmpty() {
		return false
	}
	if !rgn.bounds.Intersects(other.bounds) {
		return false
	}
	weAreARect := rgn.IsRect()
	theyAreARect := other.IsRect()
	if weAreARect && theyAreARect {
		return true
	}
	if weAreARect {
		return other.Intersects(rgn.bounds)
	}
	if theyAreARect {
		return rgn.Intersects(other.bounds)
	}
	// both of us are complex
	return regionOper(rgn, other, RegionIntersect, nil)
}

// Equal reports whether the region and other represent the same set of pixels.
func (rgn *Region) Equal(other *Region) bool {
	if rgn == other {
		return true
	}
	if rgn.bounds != other.bounds {
		return false
	}
	ah, bh := rgn.head, other.head
	if ah == bh { // catches empties and rects being equal
		return true
	}
	if ah == nil || bh == nil {
		return false
	}
	return slices.Equal(ah.runs, bh.runs)
}

// pinOffset32 clamps offset so that minV+offset and maxV+offset both stay within the int32 range.
func pinOffset32(minV, maxV, offset int32) int32 {
	const lo = int32(math.MinInt32)
	const hi = int32(math.MaxInt32)
	if int64(minV)+int64(offset) < int64(lo) {
		offset = lo - minV
	}
	if int64(maxV)+int64(offset) > int64(hi) {
		offset = hi - maxV
	}
	return offset
}

// Translate offsets the region in place by (dx, dy).
func (rgn *Region) Translate(dx, dy int32) {
	if rgn.IsEmpty() {
		return
	}
	dx = pinOffset32(rgn.bounds.Left, rgn.bounds.Right, dx)
	dy = pinOffset32(rgn.bounds.Top, rgn.bounds.Bottom, dy)
	if rgn.IsRect() {
		rgn.SetRect(rgn.bounds.Offset(dx, dy))
		return
	}
	src := rgn.head.runs
	dst := make([]int32, len(src))
	si := 0
	di := 0
	dst[di] = src[si] + dy // top
	si++
	di++
	for {
		bottom := src[si]
		si++
		if bottom == regionSentinel {
			break
		}
		dst[di] = bottom + dy
		di++
		dst[di] = src[si] // copy intervalCount
		si++
		di++
		for {
			x := src[si]
			si++
			if x == regionSentinel {
				break
			}
			dst[di] = x + dx
			di++
			dst[di] = src[si] + dx
			si++
			di++
		}
		dst[di] = regionSentinel // x sentinel
		di++
	}
	dst[di] = regionSentinel // y sentinel

	rgn.head = &regionRunHead{
		runs:          dst,
		ySpanCount:    rgn.head.ySpanCount,
		intervalCount: rgn.head.intervalCount,
	}
	rgn.bounds = rgn.bounds.Offset(dx, dy)
}

// getRuns returns the region's runs, materializing rect/empty forms into tmp.
func (rgn *Region) getRuns(tmp *[rectRegionRuns]int32) (runs []int32, intervals int32) {
	switch {
	case rgn.IsEmpty():
		tmp[0] = regionSentinel
		return tmp[:1], 0
	case rgn.IsRect():
		buildRectRuns(rgn.bounds, tmp)
		return tmp[:], 1
	default:
		return rgn.head.runs, rgn.head.intervalCount
	}
}

///////////////////////////////////////////////////////////////////////////////
// The boolean op machine (Oper)

// runArray is an int32 buffer that only ever grows.
type runArray struct {
	buf []int32
}

func (r *runArray) resizeToAtLeast(count int) {
	if count > len(r.buf) {
		count += count >> 1 // leave at least 50% extra space for future growth
		nbuf := make([]int32, count)
		copy(nbuf, r.buf)
		r.buf = nbuf
	}
}

// spanRec merges two scanlines' interval lists.
type spanRec struct {
	aRuns, bRuns []int32
	aIdx, bIdx   int
	aLeft, aRite int32
	bLeft, bRite int32
	left, rite   int32
	inside       int32
}

func (s *spanRec) init(aRuns []int32, aIdx int, bRuns []int32, bIdx int) {
	s.aLeft = aRuns[aIdx]
	s.aRite = aRuns[aIdx+1]
	s.bLeft = bRuns[bIdx]
	s.bRite = bRuns[bIdx+1]
	s.aRuns = aRuns
	s.aIdx = aIdx + 2
	s.bRuns = bRuns
	s.bIdx = bIdx + 2
}

func (s *spanRec) done() bool {
	return s.aLeft == regionSentinel && s.bLeft == regionSentinel
}

func (s *spanRec) next() {
	var inside, left, rite int32
	aFlush := false
	bFlush := false

	aLeft := s.aLeft
	aRite := s.aRite
	bLeft := s.bLeft
	bRite := s.bRite

	switch {
	case aLeft < bLeft:
		inside = 1
		left = aLeft
		if aRite <= bLeft { // [...] <...>
			rite = aRite
			aFlush = true
		} else { // [...<..]...> or [...<...>...]
			rite = bLeft
			aLeft = bLeft
		}
	case bLeft < aLeft:
		inside = 2
		left = bLeft
		if bRite <= aLeft { // [...] <...>
			rite = bRite
			bFlush = true
		} else { // [...<..]...> or [...<...>...]
			rite = aLeft
			bLeft = aLeft
		}
	default: // aLeft == bLeft
		inside = 3
		left = aLeft
		if aRite <= bRite {
			rite = aRite
			bLeft = aRite
			aFlush = true
		}
		if bRite <= aRite {
			rite = bRite
			aLeft = bRite
			bFlush = true
		}
	}

	if aFlush {
		aLeft = s.aRuns[s.aIdx]
		aRite = s.aRuns[s.aIdx+1]
		s.aIdx += 2
	}
	if bFlush {
		bLeft = s.bRuns[s.bIdx]
		bRite = s.bRuns[s.bIdx+1]
		s.bIdx += 2
	}

	s.aLeft = aLeft
	s.aRite = aRite
	s.bLeft = bLeft
	s.bRite = bRite
	s.left = left
	s.rite = rite
	s.inside = inside
}

func distanceToSentinel(runs []int32, i int) int {
	start := i
	for runs[i] != regionSentinel {
		i += 2
	}
	return i - start
}

// operateOnSpan merges one scanline from each region's interval list into array under the boolean op implied by
// minIn/maxIn, appending the result at dstOffset.
func operateOnSpan(aRuns []int32, aIdx int, bRuns []int32, bIdx int, array *runArray, dstOffset int, minIn, maxIn int32) int {
	// This is a worst-case for this span plus two for TWO terminating sentinels.
	array.resizeToAtLeast(dstOffset + distanceToSentinel(aRuns, aIdx) + distanceToSentinel(bRuns, bIdx) + 2)
	dst := dstOffset

	var rec spanRec
	firstInterval := true
	rec.init(aRuns, aIdx, bRuns, bIdx)

	for !rec.done() {
		rec.next()

		left := rec.left
		rite := rec.rite

		// add left,rite to our dst buffer (checking for coincidence), skip if equal
		if uint32(rec.inside-minIn) <= uint32(maxIn-minIn) && left < rite {
			if firstInterval || array.buf[dst-1] < left {
				array.buf[dst] = left
				array.buf[dst+1] = rite
				dst += 2
				firstInterval = false
			} else {
				// update the right edge
				array.buf[dst-1] = rite
			}
		}
	}
	array.buf[dst] = regionSentinel
	dst++
	return dst
}

// gOpMinMax gives the inside-count range that should be kept for each op (indexed by RegionOp
// Difference/Intersect/Union/Xor).
var gOpMinMax = [4]struct{ min, max int32 }{
	{min: 1, max: 1}, // Difference
	{min: 3, max: 3}, // Intersection
	{min: 1, max: 3}, // Union
	{min: 1, max: 2}, // XOR
}

// rgnOper accumulates the output scanlines of a region boolean op, collapsing adjacent identical rows.
type rgnOper struct {
	array    *runArray
	min, max int32
	startDst int
	prevDst  int
	prevLen  int
	top      int32
}

func newRgnOper(top int32, array *runArray, op RegionOp) rgnOper {
	return rgnOper{
		array:    array,
		min:      gOpMinMax[op].min,
		max:      gOpMinMax[op].max,
		startDst: 0,
		prevDst:  1,
		prevLen:  0, // will never match a length from operateOnSpan
		top:      top,
	}
}

func (o *rgnOper) addSpan(bottom int32, aRuns []int32, aIdx int, bRuns []int32, bIdx int) {
	// skip X values and slots for the next Y+intervalCount
	start := o.prevDst + o.prevLen + 2
	// start points to beginning of dst interval
	stop := operateOnSpan(aRuns, aIdx, bRuns, bIdx, o.array, start, o.min, o.max)
	length := stop - start

	sameAsPrev := o.prevLen == length
	if sameAsPrev && length != 1 {
		sameAsPrev = slices.Equal(o.array.buf[o.prevDst:o.prevDst+length-1],
			o.array.buf[start:start+length-1])
	}
	if sameAsPrev {
		// update Y value
		o.array.buf[o.prevDst-2] = bottom
	} else { // accept the new span
		if length == 1 && o.prevLen == 0 {
			o.top = bottom // just update our bottom
		} else {
			o.array.buf[start-2] = bottom
			o.array.buf[start-1] = int32(length >> 1)
			o.prevDst = start
			o.prevLen = length
		}
	}
}

func (o *rgnOper) flush() int {
	o.array.buf[o.startDst] = o.top
	// Previously reserved enough for TWO sentinels.
	o.array.buf[o.prevDst+o.prevLen] = regionSentinel
	return o.prevDst - o.startDst + o.prevLen + 1
}

func (o *rgnOper) isEmpty() bool { return o.prevLen == 0 }

// quickExitTrueCount is the sentinel return value from operate when quickExit finds a non-empty result.
const quickExitTrueCount = -1

// gEmptyScanline is a stand-in scanline used by operate when one side has no matching row; gSentinelIdx indexes the
// sentinel within it.
var gEmptyScanline = []int32{
	0, // fake bottom value
	0, // zero intervals
	regionSentinel,
	// just need a 2nd value, since spanRec.init() reads 2 values, even though if the first value is the sentinel, it
	// ignores the 2nd value.
	0,
}

const gSentinelIdx = 2

// operate runs the boolean op over both regions' scanlines.
func operate(aRunsIn, bRunsIn []int32, dst *runArray, op RegionOp, quickExit bool) int {
	aIdx := 0
	bIdx := 0
	aTop := aRunsIn[aIdx]
	aIdx++
	aBot := aRunsIn[aIdx]
	aIdx++
	bTop := bRunsIn[bIdx]
	bIdx++
	bBot := bRunsIn[bIdx]
	bIdx++

	aIdx++ // skip the intervalCount
	bIdx++ // skip the intervalCount

	// Now aIdx and bIdx are at their intervals (or sentinel).

	oper := newRgnOper(min(aTop, bTop), dst, op)

	prevBot := regionSentinel // so we fail the first test

	for aBot < regionSentinel || bBot < regionSentinel {
		var top int32
		var bot int32
		run0Runs := gEmptyScanline
		run0Idx := gSentinelIdx
		run1Runs := gEmptyScanline
		run1Idx := gSentinelIdx
		aFlush := false
		bFlush := false

		switch {
		case aTop < bTop:
			top = aTop
			run0Runs, run0Idx = aRunsIn, aIdx
			if aBot <= bTop { // [...] <...>
				bot = aBot
				aFlush = true
			} else { // [...<..]...> or [...<...>...]
				bot = bTop
				aTop = bTop
			}
		case bTop < aTop:
			top = bTop
			run1Runs, run1Idx = bRunsIn, bIdx
			if bBot <= aTop { // [...] <...>
				bot = bBot
				bFlush = true
			} else { // [...<..]...> or [...<...>...]
				bot = aTop
				bTop = aTop
			}
		default: // aTop == bTop
			top = aTop
			run0Runs, run0Idx = aRunsIn, aIdx
			run1Runs, run1Idx = bRunsIn, bIdx
			if aBot <= bBot {
				bot = aBot
				bTop = aBot
				aFlush = true
			}
			if bBot <= aBot {
				bot = bBot
				aTop = bBot
				bFlush = true
			}
		}

		if top > prevBot {
			oper.addSpan(top, gEmptyScanline, gSentinelIdx, gEmptyScanline, gSentinelIdx)
		}
		oper.addSpan(bot, run0Runs, run0Idx, run1Runs, run1Idx)

		if quickExit && !oper.isEmpty() {
			return quickExitTrueCount
		}

		if aFlush {
			aIdx = skipIntervalsAt(aRunsIn, aIdx)
			aTop = aBot
			aBot = aRunsIn[aIdx]
			aIdx++
			aIdx++ // skip uninitialized intervalCount
			if aBot == regionSentinel {
				aTop = aBot
			}
		}
		if bFlush {
			bIdx = skipIntervalsAt(bRunsIn, bIdx)
			bTop = bBot
			bBot = bRunsIn[bIdx]
			bIdx++
			bIdx++ // skip uninitialized intervalCount
			if bBot == regionSentinel {
				bTop = bBot
			}
		}

		prevBot = bot
	}
	return oper.flush()
}

// skipIntervalsAt returns the index past the current scanline's intervals; idx is at the intervals, and the count sits
// one slot back.
func skipIntervalsAt(runs []int32, idx int) int {
	intervals := runs[idx-1]
	return idx + int(intervals)*2 + 1
}

func setEmptyCheck(result *Region) bool {
	if result != nil {
		return result.SetEmpty()
	}
	return false
}

func setRectCheck(result *Region, rect geom.IRect) bool {
	if result != nil {
		return result.SetRect(rect)
	}
	return !rect.IsEmpty()
}

func setRegionCheck(result, rgn *Region) bool {
	if result != nil {
		return result.SetRegion(rgn)
	}
	return !rgn.IsEmpty()
}

// regionOper combines rgnaOrig and rgnbOrig under op, writing to result. result may be nil (query-only, with quick
// exit).
func regionOper(rgnaOrig, rgnbOrig *Region, op RegionOp, result *Region) bool {
	if op == RegionReplace {
		return setRegionCheck(result, rgnbOrig)
	}

	rgna := rgnaOrig
	rgnb := rgnbOrig

	// collapse difference and reverse-difference into just difference
	if op == RegionReverseDifference {
		rgna, rgnb = rgnb, rgna
		op = RegionDifference
	}

	aEmpty := rgna.IsEmpty()
	bEmpty := rgnb.IsEmpty()
	aRect := rgna.IsRect()
	bRect := rgnb.IsRect()

	switch op {
	case RegionDifference:
		if aEmpty {
			return setEmptyCheck(result)
		}
		if bEmpty || !rgna.bounds.Intersects(rgnb.bounds) {
			return setRegionCheck(result, rgna)
		}
		if bRect && rgnb.bounds.ContainsRect(rgna.bounds) {
			return setEmptyCheck(result)
		}
	case RegionIntersect:
		bounds := rgna.bounds
		if aEmpty || bEmpty || !bounds.Intersect(rgnb.bounds) {
			return setEmptyCheck(result)
		}
		if aRect && bRect {
			return setRectCheck(result, bounds)
		}
		if aRect && rgna.bounds.ContainsRect(rgnb.bounds) {
			return setRegionCheck(result, rgnb)
		}
		if bRect && rgnb.bounds.ContainsRect(rgna.bounds) {
			return setRegionCheck(result, rgna)
		}
	case RegionUnion:
		if aEmpty {
			return setRegionCheck(result, rgnb)
		}
		if bEmpty {
			return setRegionCheck(result, rgna)
		}
		if aRect && rgna.bounds.ContainsRect(rgnb.bounds) {
			return setRegionCheck(result, rgna)
		}
		if bRect && rgnb.bounds.ContainsRect(rgna.bounds) {
			return setRegionCheck(result, rgnb)
		}
	case RegionXor:
		if aEmpty {
			return setRegionCheck(result, rgnb)
		}
		if bEmpty {
			return setRegionCheck(result, rgna)
		}
	default:
	}

	var tmpA, tmpB [rectRegionRuns]int32
	aRuns, _ := rgna.getRuns(&tmpA)
	bRuns, _ := rgnb.getRuns(&tmpB)

	array := runArray{buf: make([]int32, 256)}
	count := operate(aRuns, bRuns, &array, op, result == nil)

	if result != nil {
		return result.setRunsTrimmed(array.buf[:count])
	}
	return count == quickExitTrueCount || count > 2
}

// OpRegion combines rgna and rgnb under op, storing the result in rgn.
func (rgn *Region) OpRegion(rgna, rgnb *Region, op RegionOp) bool {
	return regionOper(rgna, rgnb, op, rgn)
}

// Op combines the region with other under op: rgn = rgn op other.
func (rgn *Region) Op(other *Region, op RegionOp) bool {
	return regionOper(rgn, other, op, rgn)
}

// OpRect combines the region with rect under op: rgn = rgn op rect.
func (rgn *Region) OpRect(rect geom.IRect, op RegionOp) bool {
	tmp := NewRegionRect(rect)
	return regionOper(rgn, tmp, op, rgn)
}

// SetRects sets the region to the union of rects.
func (rgn *Region) SetRects(rects []geom.IRect) bool {
	if len(rects) == 0 {
		rgn.SetEmpty()
	} else {
		rgn.SetRect(rects[0])
		for _, r := range rects[1:] {
			rgn.OpRect(r, RegionUnion)
		}
	}
	return !rgn.IsEmpty()
}

///////////////////////////////////////////////////////////////////////////////
// Iterators

// RegionIterator returns the region's rects in Y-X order.
type RegionIterator struct {
	rgn  *Region
	runs []int32 // nil for the rect case
	idx  int
	rect geom.IRect
	done bool
}

// NewRegionIterator returns an iterator over rgn's rects.
func NewRegionIterator(rgn *Region) *RegionIterator {
	it := &RegionIterator{}
	it.Reset(rgn)
	return it
}

// Reset reinitializes the iterator to walk rgn's rects from the start.
func (it *RegionIterator) Reset(rgn *Region) {
	it.rgn = rgn
	if rgn.IsEmpty() {
		it.done = true
		return
	}
	it.done = false
	if rgn.IsRect() {
		it.rect = rgn.bounds
		it.runs = nil
		return
	}
	runs := rgn.head.runs
	it.rect = geom.IRectLTRB(runs[3], runs[0], runs[4], runs[1])
	it.runs = runs
	it.idx = 5 // now idx points at the 2nd interval (or x-sentinel)
}

// Done reports whether the iterator has exhausted the region's rects.
func (it *RegionIterator) Done() bool { return it.done }

// Rect returns the current rect.
func (it *RegionIterator) Rect() geom.IRect { return it.rect }

// Next advances to the next rect.
func (it *RegionIterator) Next() {
	if it.done {
		return
	}
	if it.runs == nil { // rect case
		it.done = true
		return
	}
	runs := it.runs
	i := it.idx

	if runs[i] < regionSentinel { // valid X value
		it.rect.Left = runs[i]
		it.rect.Right = runs[i+1]
		i += 2
	} else { // we're at the end of a line
		i++
		if runs[i] < regionSentinel { // valid Y value
			intervals := runs[i+1]
			if intervals == 0 { // empty line
				it.rect.Top = runs[i]
				i += 3
			} else {
				it.rect.Top = it.rect.Bottom
			}
			it.rect.Bottom = runs[i]
			it.rect.Left = runs[i+2]
			it.rect.Right = runs[i+3]
			i += 4
		} else { // end of rgn
			it.done = true
		}
	}
	it.idx = i
}

// RegionCliperator iterates the region's rects intersected with clip. The iterator is embedded by value (not by
// pointer) so a stack-allocated cliperator needs no separate heap object; hot per-draw loops can keep one on the stack
// via Init.
type RegionCliperator struct {
	iter RegionIterator
	clip geom.IRect
	rect geom.IRect
	done bool
}

// NewRegionCliperator returns an iterator over rgn's rects intersected with clip.
func NewRegionCliperator(rgn *Region, clip geom.IRect) *RegionCliperator {
	c := &RegionCliperator{}
	c.Init(rgn, clip)
	return c
}

// Init resets the cliperator in place to iterate rgn's rects intersected with clip. It lets a caller keep the
// cliperator on the stack, avoiding the heap allocation NewRegionCliperator makes for the returned object.
func (c *RegionCliperator) Init(rgn *Region, clip geom.IRect) {
	c.iter.Reset(rgn)
	c.clip = clip
	c.rect = geom.IRect{}
	c.done = true
	for !c.iter.Done() {
		r := c.iter.Rect()
		if r.Top >= clip.Bottom {
			break
		}
		c.rect = clip
		if c.rect.Intersect(r) {
			c.done = false
			break
		}
		c.iter.Next()
	}
}

// Done reports whether the cliperator has exhausted the intersected rects.
func (c *RegionCliperator) Done() bool { return c.done }

// Rect returns the current (clipped) rect.
func (c *RegionCliperator) Rect() geom.IRect { return c.rect }

// Next advances to the next intersected rect.
func (c *RegionCliperator) Next() {
	if c.done {
		return
	}
	c.done = true
	c.iter.Next()
	for !c.iter.Done() {
		r := c.iter.Rect()
		if r.Top >= c.clip.Bottom {
			break
		}
		c.rect = c.clip
		if c.rect.Intersect(r) {
			c.done = false
			break
		}
		c.iter.Next()
	}
}

// RegionSpanerator iterates a single scanline's spans clipped to [left, right).
type RegionSpanerator struct {
	runs        []int32 // nil means we're a rect
	idx         int
	left, right int32
	done        bool
}

// NewRegionSpanerator returns an iterator over scanline y's spans, clipped to [left, right).
func NewRegionSpanerator(rgn *Region, y, left, right int32) *RegionSpanerator {
	s := &RegionSpanerator{done: true}
	r := rgn.Bounds()
	if !rgn.IsEmpty() && y >= r.Top && y < r.Bottom && right > r.Left && left < r.Right {
		if rgn.IsRect() {
			s.left = max(left, r.Left)
			s.right = min(right, r.Right)
			s.runs = nil
			s.done = false
		} else {
			runs := rgn.head.runs
			i := rgn.head.findScanline(y) + 2 // skip Bottom and IntervalCount
			for runs[i] < right {

				if runs[i+1] <= left { // to the left of the span, so continue
					i += 2
					continue
				}
				// runs[i..i+1] intersects the span
				s.runs = runs
				s.idx = i
				s.left = left
				s.right = right
				s.done = false
				break
			}
		}
	}
	return s
}

// Next returns the next span, or ok == false when exhausted.
func (s *RegionSpanerator) Next() (left, right int32, ok bool) {
	if s.done {
		return 0, 0, false
	}
	if s.runs == nil { // we're a rect
		s.done = true // ok, now we're done
		return s.left, s.right, true
	}
	if s.runs[s.idx] >= s.right {
		s.done = true
		return 0, 0, false
	}
	left = max(s.left, s.runs[s.idx])
	right = min(s.right, s.runs[s.idx+1])
	s.idx += 2
	return left, right, true
}
