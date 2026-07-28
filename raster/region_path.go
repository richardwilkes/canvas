// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// rgnBuilder and Region.SetPath: rasterize a path (without AA) directly into region runs by collecting the non-AA scan
// converter's BlitH spans.

package raster

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// rgnBuilder accumulates scanline spans into region runs as a non-AA path fill blits them. Scanline records in storage
// mimic a region row, nearly: a region row is [Bottom IntervalCount [L R]... Sentinel] while a stored scanline is
// [LastY XCount [L R]... (slot)] — same length, transmuted by copyToRgn.
type rgnBuilder struct {
	storage []int32

	currScanline int // index of the current scanline record; -1 signals empty collection
	prevScanline int // -1 signals first scanline
	currXIdx     int // index of the next available X slot in the current scanline
	top          int32
}

func (b *rgnBuilder) lastY(s int) int32  { return b.storage[s] }
func (b *rgnBuilder) xCount(s int) int32 { return b.storage[s+1] }
func (b *rgnBuilder) firstX(s int) int   { return s + 2 }
func (b *rgnBuilder) nextScanline(s int) int {
	// add final +1 for the x-sentinel
	return s + 2 + int(b.storage[s+1]) + 1
}

// init allocates the working storage; returns false if it would overflow.
func (b *rgnBuilder) init(maxHeight, maxTransitions int, pathIsInverse bool) bool {
	if maxHeight < 0 || maxTransitions < 0 {
		return false
	}
	if pathIsInverse {
		// allow for additional X transitions to "invert" each scanline: [ L' ... ... R' ]
		maxTransitions += 2
	}
	// compute the count with +1 and +3 slop for the working buffer
	count := int64(maxHeight+1) * int64(3+maxTransitions)
	if pathIsInverse {
		// allow for two "empty" rows for the top and bottom: [ Y, 1, L, R, S] == 5 (*2)
		count += 10
	}
	if count < 0 || count > math.MaxInt32 {
		return false
	}
	b.storage = make([]int32, count)
	b.currScanline = -1 // signal empty collection
	b.prevScanline = -1 // signal first scanline
	return true
}

func (b *rgnBuilder) collapsWithPrev() bool {
	if b.prevScanline >= 0 &&
		b.lastY(b.prevScanline)+1 == b.lastY(b.currScanline) &&
		b.xCount(b.prevScanline) == b.xCount(b.currScanline) {
		n := int(b.xCount(b.currScanline))
		pi := b.firstX(b.prevScanline)
		ci := b.firstX(b.currScanline)
		same := true
		for i := 0; i < n; i++ {
			if b.storage[pi+i] != b.storage[ci+i] {
				same = false
				break
			}
		}
		if same {
			// update the height of prevScanline
			b.storage[b.prevScanline] = b.lastY(b.currScanline)
			return true
		}
	}
	return false
}

// BlitH implements Blitter, accumulating a scanline span into the builder's storage.
func (b *rgnBuilder) BlitH(x, y, width int32) {
	if b.currScanline < 0 { // first time
		b.top = y
		b.currScanline = 0
		b.storage[0] = y // lastY
		b.currXIdx = b.firstX(0)
	} else if y > b.lastY(b.currScanline) {
		// if we get here, we're done with currScanline
		b.storage[b.currScanline+1] = int32(b.currXIdx - b.firstX(b.currScanline)) // xCount

		prevLastY := b.lastY(b.currScanline)
		if !b.collapsWithPrev() {
			b.prevScanline = b.currScanline
			b.currScanline = b.nextScanline(b.currScanline)
		}
		if y-1 > prevLastY { // insert empty run
			b.storage[b.currScanline] = y - 1
			b.storage[b.currScanline+1] = 0
			b.currScanline = b.nextScanline(b.currScanline)
		}
		// setup for the new curr line
		b.storage[b.currScanline] = y
		b.currXIdx = b.firstX(b.currScanline)
	}
	// check if we should extend the current run, or add a new one
	if b.currXIdx > b.firstX(b.currScanline) && b.storage[b.currXIdx-1] == x {
		b.storage[b.currXIdx-1] = x + width
	} else {
		b.storage[b.currXIdx] = x
		b.storage[b.currXIdx+1] = x + width
		b.currXIdx += 2
	}
}

// done flushes the last scanline once the fill is complete.
func (b *rgnBuilder) done() {
	if b.currScanline >= 0 {
		b.storage[b.currScanline+1] = int32(b.currXIdx - b.firstX(b.currScanline)) // xCount
		if !b.collapsWithPrev() {                                                  // flush the last line
			b.currScanline = b.nextScanline(b.currScanline)
		}
	}
}

// computeRunCount returns the number of int32 slots the final region's run array needs.
func (b *rgnBuilder) computeRunCount() int {
	if b.currScanline < 0 {
		return 0
	}
	return 2 + b.currScanline
}

// copyToRect returns the collected scanlines as a rect (only valid when computeRunCount() == rectRegionRuns).
func (b *rgnBuilder) copyToRect() geom.IRect {
	// A rect's scanline is [bottom intervals left right sentinel] == 5
	return geom.IRectLTRB(b.storage[2], b.top, b.storage[3], b.lastY(0)+1)
}

// copyToRgn packs the collected scanlines into runs in region-run-array form.
func (b *rgnBuilder) copyToRgn(runs []int32) {
	i := 0
	runs[i] = b.top
	i++
	line := 0
	stop := b.currScanline
	for {
		runs[i] = b.lastY(line) + 1
		i++
		count := int(b.xCount(line))
		runs[i] = int32(count >> 1) // intervalCount
		i++
		if count > 0 {
			copy(runs[i:i+count], b.storage[b.firstX(line):b.firstX(line)+count])
			i += count
		}
		runs[i] = regionSentinel
		i++
		line = b.nextScanline(line)
		if line >= stop {
			break
		}
	}
	runs[i] = regionSentinel
}

// The remaining Blitter methods. The non-AA fill emits only BlitH and BlitRect on the region builder — BlitRect from
// walkSimpleEdges' zero-slope fast path and from blitAboveClip/blitBelowClip via blitRectRegion — so BlitRect below has
// a real body while the rest are no-ops.

// BlitAntiH implements Blitter.
func (b *rgnBuilder) BlitAntiH(_, _ int32, _ []Alpha, _ []int16) {}

// BlitV implements Blitter.
func (b *rgnBuilder) BlitV(_, _, _ int32, _ Alpha) {}

// BlitRect implements Blitter.
func (b *rgnBuilder) BlitRect(x, y, width, height int32) {
	// The scan converter's rect fast path lowers through blitRect; feed it back as rows.
	for r := y; r < y+height; r++ {
		b.BlitH(x, r, width)
	}
}

// BlitAntiRect implements Blitter.
func (b *rgnBuilder) BlitAntiRect(_, _, _, _ int32, _, _ Alpha) {}

// BlitMask implements Blitter.
func (b *rgnBuilder) BlitMask(_ *Mask, _ geom.IRect) {}

// BlitAntiH2 implements Blitter.
func (b *rgnBuilder) BlitAntiH2(_, _ int32, _, _ Alpha) {}

// BlitAntiV2 implements Blitter.
func (b *rgnBuilder) BlitAntiV2(_, _ int32, _, _ Alpha) {}

///////////////////////////////////////////////////////////////////////////////

// verbToInitialLastIndex returns the index of the verb's last point (0 for moves and closes).
func verbToInitialLastIndex(verb path.Verb) int {
	switch verb {
	case path.VerbLine:
		return 1
	case path.VerbQuad, path.VerbConic:
		return 2
	case path.VerbCubic:
		return 3
	default: // move, close
		return 0
	}
}

func verbToMaxEdges(verb path.Verb) int {
	switch verb {
	case path.VerbLine:
		return 1
	case path.VerbQuad, path.VerbConic:
		return 2
	case path.VerbCubic:
		return 3
	default:
		return 0
	}
}

// countPathRuntypeValues returns the worst-case edge count and Y bounds for a path. Returns 0 maxEdges when the path
// has only moves and closes (itop/ibot are then meaningless).
func countPathRuntypeValues(p *path.Path) (maxEdges int, itop, ibot int32) {
	top := float32(math.MaxInt16)
	bot := float32(math.MinInt16)

	iter := path.NewIter(p, true)
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		maxEdges += verbToMaxEdges(verb)

		lastIndex := verbToInitialLastIndex(verb)
		if lastIndex > 0 {
			for i := 1; i <= lastIndex; i++ {
				if top > pts[i].Y {
					top = pts[i].Y
				} else if bot < pts[i].Y {
					bot = pts[i].Y
				}
			}
		} else if verb == path.VerbMove {
			if top > pts[0].Y {
				top = pts[0].Y
			} else if bot < pts[0].Y {
				bot = pts[0].Y
			}
		}
	}
	if maxEdges == 0 {
		return 0, 0, 0 // we have only moves+closes
	}
	return maxEdges, geom.RoundToInt(top), geom.RoundToInt(bot)
}

// countRuntypeValues returns the worst-case transition count and Y bounds for the region.
func (rgn *Region) countRuntypeValues() (maxT int, itop, ibot int32) {
	if rgn.IsRect() {
		maxT = 2
	} else {
		maxT = int(rgn.head.intervalCount) * 2
	}
	return maxT, rgn.bounds.Top, rgn.bounds.Bottom
}

func checkInverseOnEmptyReturn(dst *Region, p *path.Path, clip *Region) bool {
	if p.IsInverseFillType() {
		return dst.SetRegion(clip)
	}
	return dst.SetEmpty()
}

// SetPath sets the region to the area described by the path, clipped to clip (which supplies the bounds for inverse
// fills).
func (rgn *Region) SetPath(p *path.Path, clip *Region) bool {
	if clip.IsEmpty() || !p.IsFinite() || p.IsEmpty() {
		// This treats non-finite paths as empty as well, so this returns empty or 'clip' if it's inverse-filled. If
		// clip is also empty, the path's fill type doesn't really matter and this region ends up empty.
		return checkInverseOnEmptyReturn(rgn, p, clip)
	}

	// Our builder is very fragile, and can't be called with spans/rects out of Y->X order. To ensure this, we only
	// "fill" clipped to a rect (the clip's bounds), and if the clip is more complex than that, we just post-intersect
	// the result with the clip.
	clipBounds := clip.Bounds()
	if clip.IsComplex() {
		if !rgn.SetPath(p, NewRegionRect(clipBounds)) {
			return false
		}
		return rgn.Op(clip, RegionIntersect)
	}

	// The scan converter has limits on the coordinate range of the clipping region. If it's too big, tile the clip
	// bounds and union the pieces back together.
	if PathRequiresTiling(clipBounds) {
		const kTileSize = 32767 >> 1 // Limit so coords can fit into a 16.16 fixed-point value
		pathBounds := p.Bounds().RoundOut()

		rgn.SetEmpty()

		// Note: with large integers some intermediate calculations can overflow, but the end results will still be in
		// integer range; use int64 for the intermediates.
		for top := int64(clipBounds.Top); top < int64(clipBounds.Bottom); top += kTileSize {
			bot := min(top+kTileSize, int64(clipBounds.Bottom))
			for left := int64(clipBounds.Left); left < int64(clipBounds.Right); left += kTileSize {
				right := min(left+kTileSize, int64(clipBounds.Right))

				tileClipBounds := geom.IRectLTRB(int32(left), int32(top), int32(right), int32(bot))
				if !pathBounds.Intersects(tileClipBounds) {
					continue
				}

				// Shift coordinates so the top left is (0,0) during scan conversion and then translate the region
				// afterwards.
				tileClipBounds = tileClipBounds.Offset(int32(-left), int32(-top))
				var m geom.Matrix
				m.SetAll(1, 0, float32(-left), 0, 1, float32(-top), 0, 0, 1)
				shifted := path.New()
				p.TransformTo(&m, shifted)
				tile := &Region{}
				tile.SetPath(shifted, NewRegionRect(tileClipBounds))
				tile.Translate(int32(left), int32(top))
				rgn.Op(tile, RegionUnion)
			}
		}
		// During tiling we only applied the bounds of the tile; now that we have a full region, apply the original
		// clip.
		return rgn.Op(clip, RegionIntersect)
	}

	// compute worst-case rgn-size for the path
	pathTransitions, pathTop, pathBot := countPathRuntypeValues(p)
	if pathTransitions == 0 {
		return checkInverseOnEmptyReturn(rgn, p, clip)
	}

	clipTransitions, clipTop, clipBot := clip.countRuntypeValues()

	top := max(pathTop, clipTop)
	bot := min(pathBot, clipBot)
	if top >= bot {
		return checkInverseOnEmptyReturn(rgn, p, clip)
	}

	var builder rgnBuilder
	if !builder.init(int(bot-top), max(pathTransitions, clipTransitions), p.IsInverseFillType()) {
		// can't allocate working space, so return false
		return rgn.SetEmpty()
	}

	FillPathRegion(p, clip, &builder)
	builder.done()

	count := builder.computeRunCount()
	switch count {
	case 0:
		return rgn.SetEmpty()
	case rectRegionRuns:
		return rgn.SetRect(builder.copyToRect())
	default:
		runs := make([]int32, count)
		builder.copyToRgn(runs)
		head := &regionRunHead{runs: runs}
		rgn.head = head
		rgn.bounds = head.computeRunBounds()
		return true
	}
}
