// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// RgnClipBlitter and the region blit helpers: restrict all blits to a region before forwarding.

package raster

import "github.com/richardwilkes/canvas/geom"

// blitRectRegion blits rect through b, clipped span by span to clip.
func blitRectRegion(b Blitter, rect geom.IRect, clip *Region) {
	for it := NewRegionCliperator(clip, rect); !it.Done(); it.Next() {
		cr := it.Rect()
		b.BlitRect(cr.Left, cr.Top, cr.Width(), cr.Height())
	}
}

// blitRegion blits every span of clip through b.
func blitRegion(b Blitter, clip *Region) {
	visitRegionSpans(clip, func(r geom.IRect) {
		b.BlitRect(r.Left, r.Top, r.Width(), r.Height())
	})
}

// visitRegionSpans calls visitor with each span of rgn, in Y -> X ascending order (multi-interval scanlines are emitted
// per row).
func visitRegionSpans(rgn *Region, visitor func(geom.IRect)) {
	if rgn.IsEmpty() {
		return
	}
	if rgn.IsRect() {
		visitor(rgn.Bounds())
		return
	}
	p := rgn.head.runs
	i := 0
	top := p[i]
	i++
	bot := p[i]
	i++
	for {
		pairCount := p[i]
		i++
		if pairCount == 1 {
			visitor(geom.IRectLTRB(p[i], top, p[i+1], bot))
			i += 2
		} else if pairCount > 1 {
			// we have to loop repeated in Y, sending each interval in Y -> X order
			for y := top; y < bot; y++ {
				for k := 0; k < int(pairCount); k++ {
					visitor(geom.IRectLTRB(p[i+2*k], y, p[i+2*k+1], y+1))
				}
			}
			i += int(pairCount) * 2
		}
		i++ // skip sentinel

		// read next bottom or sentinel
		top = bot
		bot = p[i]
		i++
		if bot == regionSentinel {
			return
		}
	}
}

// RgnClipBlitter wraps another blitter, restricting every blit to a Region.
type RgnClipBlitter struct {
	blitter Blitter
	rgn     *Region
}

// Init sets up the blitter to restrict blitter's blits to rgn.
func (r *RgnClipBlitter) Init(blitter Blitter, rgn *Region) {
	r.blitter = blitter
	r.rgn = rgn
}

// BlitH implements Blitter.
func (r *RgnClipBlitter) BlitH(x, y, width int32) {
	span := NewRegionSpanerator(r.rgn, y, x, x+width)
	for {
		left, right, ok := span.Next()
		if !ok {
			return
		}
		r.blitter.BlitH(left, y, right-left)
	}
}

// BlitAntiH implements Blitter: breaks the runs at the region's span boundaries and zeroes the alphas of the excluded
// gaps.
func (r *RgnClipBlitter) BlitAntiH(x, y int32, aa []Alpha, runs []int16) {
	width := computeAntiWidth(runs)
	span := NewRegionSpanerator(r.rgn, y, x, x+width)

	prevRite := x
	for {
		left, right, ok := span.Next()
		if !ok {
			break
		}
		AlphaRunsBreak(runs, aa, int(left-x), int(right-left))

		// now zero before left
		if left > prevRite {
			index := prevRite - x
			aa[index] = 0 // skip runs after right
			runs[index] = int16(left - prevRite)
		}
		prevRite = right
	}

	if prevRite > x {
		runs[prevRite-x] = 0

		off := int32(0)
		if x < 0 {
			skip := int32(runs[0])
			off = skip
			x += skip
		}
		r.blitter.BlitAntiH(x, y, aa[off:], runs[off:])
	}
}

// BlitV implements Blitter.
func (r *RgnClipBlitter) BlitV(x, y, height int32, alpha Alpha) {
	bounds := geom.IRectLTRB(x, y, x+1, y+height)
	for it := NewRegionCliperator(r.rgn, bounds); !it.Done(); it.Next() {
		cr := it.Rect()
		r.blitter.BlitV(x, cr.Top, cr.Height(), alpha)
	}
}

// BlitRect implements Blitter.
func (r *RgnClipBlitter) BlitRect(x, y, width, height int32) {
	bounds := geom.IRectLTRB(x, y, x+width, y+height)
	for it := NewRegionCliperator(r.rgn, bounds); !it.Done(); it.Next() {
		cr := it.Rect()
		r.blitter.BlitRect(cr.Left, cr.Top, cr.Width(), cr.Height())
	}
}

// BlitAntiRect implements Blitter.
func (r *RgnClipBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	// The *true* width of the rectangle to blit is width + 2.
	bounds := geom.IRectLTRB(x, y, x+width+2, y+height)
	for it := NewRegionCliperator(r.rgn, bounds); !it.Done(); it.Next() {
		cr := it.Rect()
		effectiveLeftAlpha := leftAlpha
		if cr.Left != x {
			effectiveLeftAlpha = 255
		}
		effectiveRightAlpha := rightAlpha
		if cr.Right != x+width+2 {
			effectiveRightAlpha = 255
		}
		switch {
		case effectiveLeftAlpha == 255 && effectiveRightAlpha == 255:
			r.blitter.BlitRect(cr.Left, cr.Top, cr.Width(), cr.Height())
		case cr.Width() == 1:
			if cr.Left == x {
				r.blitter.BlitV(cr.Left, cr.Top, cr.Height(), effectiveLeftAlpha)
			} else {
				r.blitter.BlitV(cr.Left, cr.Top, cr.Height(), effectiveRightAlpha)
			}
		default:
			r.blitter.BlitAntiRect(cr.Left, cr.Top, cr.Width()-2, cr.Height(),
				effectiveLeftAlpha, effectiveRightAlpha)
		}
	}
}

// BlitMask implements Blitter.
func (r *RgnClipBlitter) BlitMask(mask *Mask, clip geom.IRect) {
	for it := NewRegionCliperator(r.rgn, clip); !it.Done(); it.Next() {
		r.blitter.BlitMask(mask, it.Rect())
	}
}

// BlitAntiH2 implements Blitter via the default lowering.
func (r *RgnClipBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(r, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (r *RgnClipBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(r, x, y, a0, a1)
}
