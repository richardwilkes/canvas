// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Antialiased hairlines (the four specialized hair blitters and doAntiHairLine's pixel-column walker), AA rect fills
// through the FDot8 (24.8) machinery (AntiFillXRect/AntiFillRect), and the AA rect frame stroker (AntiFrameRect).

package raster

import (
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// outsetBeforeClipTest: the worst-case bounds estimate for the horizontal/vertical hair cases can undervalue by a
// pixel; outsetting before the containment test keeps the clip installed in those cases rather than drawing out of
// bounds.
const outsetBeforeClipTest = true

const hlineStackBuffer = 100

// hlineScratch holds callHLineBlitter's runs/coverage buffers. They are handed to the Blitter interface as slices,
// which the compiler must treat as escaping (and some wrapper blitters genuinely mutate them in place), so a plain
// stack array would heap-allocate on every AA rect-fill / hairline scanline. Pooling keeps those hot lanes
// allocation-free; a sync.Pool (rather than a single shared instance) keeps concurrent FillPathParallel bands
// independent.
type hlineScratch struct {
	runs [hlineStackBuffer + 1]int16
	aa   [hlineStackBuffer]Alpha
}

var hlineScratchPool = sync.Pool{New: func() any { return new(hlineScratch) }}

// smallDot6Scale computes (value * dot6) >> 6 with dot6 in [0, 64].
func smallDot6Scale(value, dot6 int32) int32 {
	return (value * dot6) >> 6
}

// callHLineBlitter blits count pixels at (x, y) with a constant alpha, via BlitAntiH in chunks of at most
// hlineStackBuffer.
func callHLineBlitter(blitter Blitter, x, y, count int32, alpha uint32) {
	s := hlineScratchPool.Get().(*hlineScratch)
	runs := s.runs[:]
	aa := s.aa[:]

	for count > 0 {
		// In theory, we should be able to just do this once (outside of the loop), since aa[] and runs[] are supposed
		// to be const when we call the blitter. In reality, some wrapper-blitters (e.g. RgnClipBlitter) modify the
		// buffers in-place. Hence the need to be defensive here and reseed the aa value.
		aa[0] = Alpha(alpha)

		n := count
		if n > hlineStackBuffer {
			n = hlineStackBuffer
		}
		runs[0] = int16(n)
		runs[n] = 0
		blitter.BlitAntiH(x, y, aa, runs)
		x += n
		count -= n
	}
	hlineScratchPool.Put(s)
}

// antiHairBlitter is implemented by the four hair-line orientation strategies. Each drawCap/drawLine returns the
// updated fractional ordinate.
type antiHairBlitter interface {
	drawCap(x int32, f, slope Fixed, mod64 int32) Fixed
	drawLine(x, stopx int32, f, slope Fixed) Fixed
}

// hlineAntiHairBlitter draws completely horizontal lines.
type hlineAntiHairBlitter struct {
	blitter Blitter
}

func (h *hlineAntiHairBlitter) drawCap(x int32, fy, _ Fixed, mod64 int32) Fixed {
	fy += FixedOne / 2

	y := int32(fy >> 16)
	a := int32((fy >> 8) & 0xFF)

	// lower line
	ma := smallDot6Scale(a, mod64)
	if ma != 0 {
		callHLineBlitter(h.blitter, x, y, 1, uint32(ma))
	}

	// upper line
	ma = smallDot6Scale(255-a, mod64)
	if ma != 0 {
		callHLineBlitter(h.blitter, x, y-1, 1, uint32(ma))
	}

	return fy - FixedOne/2
}

func (h *hlineAntiHairBlitter) drawLine(x, stopx int32, fy, _ Fixed) Fixed {
	count := stopx - x
	fy += FixedOne / 2

	y := int32(fy >> 16)
	a := int32((fy >> 8) & 0xFF)

	// lower line
	if a != 0 {
		callHLineBlitter(h.blitter, x, y, count, uint32(a))
	}

	// upper line
	a = 255 - a
	if a != 0 {
		callHLineBlitter(h.blitter, x, y-1, count, uint32(a))
	}

	return fy - FixedOne/2
}

// horishAntiHairBlitter draws mostly horizontal lines.
type horishAntiHairBlitter struct {
	blitter Blitter
}

func (h *horishAntiHairBlitter) drawCap(x int32, fy, dy Fixed, mod64 int32) Fixed {
	fy += FixedOne / 2

	lowerY := int32(fy >> 16)
	a := int32((fy >> 8) & 0xFF)
	a0 := smallDot6Scale(255-a, mod64)
	a1 := smallDot6Scale(a, mod64)
	h.blitter.BlitAntiV2(x, lowerY-1, Alpha(a0), Alpha(a1))

	return fy + dy - FixedOne/2
}

func (h *horishAntiHairBlitter) drawLine(x, stopx int32, fy, dy Fixed) Fixed {
	fy += FixedOne / 2
	for {
		lowerY := int32(fy >> 16)
		a := int32((fy >> 8) & 0xFF)
		h.blitter.BlitAntiV2(x, lowerY-1, Alpha(255-a), Alpha(a))
		fy += dy
		x++
		if x >= stopx {
			break
		}
	}

	return fy - FixedOne/2
}

// vlineAntiHairBlitter draws completely vertical lines.
type vlineAntiHairBlitter struct {
	blitter Blitter
}

func (v *vlineAntiHairBlitter) drawCap(y int32, fx, _ Fixed, mod64 int32) Fixed {
	fx += FixedOne / 2

	x := int32(fx >> 16)
	a := int32((fx >> 8) & 0xFF)

	ma := smallDot6Scale(a, mod64)
	if ma != 0 {
		v.blitter.BlitV(x, y, 1, Alpha(ma))
	}
	ma = smallDot6Scale(255-a, mod64)
	if ma != 0 {
		v.blitter.BlitV(x-1, y, 1, Alpha(ma))
	}

	return fx - FixedOne/2
}

func (v *vlineAntiHairBlitter) drawLine(y, stopy int32, fx, _ Fixed) Fixed {
	fx += FixedOne / 2

	x := int32(fx >> 16)
	a := int32((fx >> 8) & 0xFF)

	if a != 0 {
		v.blitter.BlitV(x, y, stopy-y, Alpha(a))
	}
	a = 255 - a
	if a != 0 {
		v.blitter.BlitV(x-1, y, stopy-y, Alpha(a))
	}

	return fx - FixedOne/2
}

// vertishAntiHairBlitter draws mostly vertical lines.
type vertishAntiHairBlitter struct {
	blitter Blitter
}

func (v *vertishAntiHairBlitter) drawCap(y int32, fx, dx Fixed, mod64 int32) Fixed {
	fx += FixedOne / 2

	x := int32(fx >> 16)
	a := int32((fx >> 8) & 0xFF)
	v.blitter.BlitAntiH2(x-1, y, Alpha(smallDot6Scale(255-a, mod64)), Alpha(smallDot6Scale(a, mod64)))

	return fx + dx - FixedOne/2
}

func (v *vertishAntiHairBlitter) drawLine(y, stopy int32, fx, dx Fixed) Fixed {
	fx += FixedOne / 2
	for {
		x := int32(fx >> 16)
		a := int32((fx >> 8) & 0xFF)
		v.blitter.BlitAntiH2(x-1, y, Alpha(255-a), Alpha(a))
		fx += dx
		y++
		if y >= stopy {
			break
		}
	}

	return fx - FixedOne/2
}

// fastFixDiv computes (a << 16) / b, valid when a fits in 16 bits.
func fastFixDiv(a, b FDot6) Fixed {
	return Fixed(int32(a) << 16 / int32(b))
}

// anyBadInts reports whether any argument is 0x80000000 (integer NaN, typically from converting a huge float), which
// the hairline math cannot negate.
func anyBadInts(a, b, c, d int32) bool {
	badInt := func(x int32) int32 { return x & -x }
	return (badInt(a)|badInt(b)|badInt(c)|badInt(d))>>31 != 0
}

// contribution64 returns the fractional part of a dot6 ordinate, with multiples of 64 returning 64 rather than 0.
func contribution64(ordinate FDot6) int32 {
	return int32((ordinate-1)&63) + 1
}

// doAntiHairLine draws one antialiased hairline segment. clip is nil when the segment is known to be fully inside.
func doAntiHairLine(x0, y0, x1, y1 FDot6, clip *geom.IRect, blitter Blitter) {
	// check for integer NaN (0x80000000) which we can't handle (can't negate it). It appears typically from a huge
	// float (inf or nan) being converted to int. If we see it, just don't draw.
	if anyBadInts(int32(x0), int32(y0), int32(x1), int32(y1)) {
		return
	}

	// The caller must clip the line to [-32767.0 ... 32767.0] ahead of time (in dot6 format).

	if FixedAbs(Fixed(x1-x0)) > Fixed(511<<6) || FixedAbs(Fixed(y1-y0)) > Fixed(511<<6) {
		// instead of (x0 + x1) >> 1, we shift each separately. This is less precise, but avoids overflowing the
		// intermediate result if the values are huge. A better fix might be to clip the original pts directly (i.e. do
		// the divide), so we don't spend time subdividing huge lines at all.
		hx := (x0 >> 1) + (x1 >> 1)
		hy := (y0 >> 1) + (y1 >> 1)
		doAntiHairLine(x0, y0, hx, hy, clip, blitter)
		doAntiHairLine(hx, hy, x1, y1, clip, blitter)
		return
	}

	var scaleStart, scaleStop int32
	var istart, istop int32
	var fstart, slope Fixed
	var hairBlitter antiHairBlitter

	// The chosen orientation strategy is constructed after the clip decides the final blitter.
	var mode int // 0 = hline, 1 = horish, 2 = vline, 3 = vertish

	if FixedAbs(Fixed(x1-x0)) > FixedAbs(Fixed(y1-y0)) { // mostly horizontal
		if x0 > x1 { // we want to go left-to-right
			x0, x1 = x1, x0
			y0, y1 = y1, y0
		}

		istart = FDot6Floor(x0)
		istop = FDot6Ceil(x1)
		fstart = FDot6ToFixed(y0)
		if y0 == y1 { // completely horizontal, take fast case
			slope = 0
			mode = 0
		} else {
			slope = fastFixDiv(y1-y0, x1-x0)
			fstart += (slope*Fixed(32-(x0&63)) + 32) >> 6
			mode = 1
		}

		if istop-istart == 1 {
			// we are within a single pixel
			scaleStart = int32(x1 - x0)
			scaleStop = 0
		} else {
			scaleStart = 64 - int32(x0&63)
			scaleStop = int32(x1 & 63)
		}

		if clip != nil {
			if istart >= clip.Right || istop <= clip.Left {
				return
			}
			if istart < clip.Left {
				fstart += slope * Fixed(clip.Left-istart)
				istart = clip.Left
				scaleStart = 64
				if istop-istart == 1 {
					// we are within a single pixel
					scaleStart = contribution64(x1)
					scaleStop = 0
				}
			}
			if istop > clip.Right {
				istop = clip.Right
				scaleStop = 0 // so we don't draw this last column
			}

			if istart == istop {
				return
			}
			// now test if our Y values are completely inside the clip
			var top, bottom int32
			if slope >= 0 { // T2B
				top = FixedFloorToInt(fstart - FixedHalf)
				bottom = FixedCeilToInt(fstart + Fixed(istop-istart-1)*slope + FixedHalf)
			} else { // B2T
				bottom = FixedCeilToInt(fstart + FixedHalf)
				top = FixedFloorToInt(fstart + Fixed(istop-istart-1)*slope - FixedHalf)
			}
			if outsetBeforeClipTest {
				top--
				bottom++
			}
			if top >= clip.Bottom || bottom <= clip.Top {
				return
			}
			if clip.Top <= top && clip.Bottom >= bottom {
				clip = nil
			}
		}
	} else { // mostly vertical
		if y0 > y1 { // we want to go top-to-bottom
			x0, x1 = x1, x0
			y0, y1 = y1, y0
		}

		istart = FDot6Floor(y0)
		istop = FDot6Ceil(y1)
		fstart = FDot6ToFixed(x0)
		if x0 == x1 {
			if y0 == y1 { // are we zero length?
				return // nothing to do
			}
			slope = 0
			mode = 2
		} else {
			slope = fastFixDiv(x1-x0, y1-y0)
			fstart += (slope*Fixed(32-(y0&63)) + 32) >> 6
			mode = 3
		}

		if istop-istart == 1 {
			// we are within a single pixel
			scaleStart = int32(y1 - y0)
			scaleStop = 0
		} else {
			scaleStart = 64 - int32(y0&63)
			scaleStop = int32(y1 & 63)
		}

		if clip != nil {
			if istart >= clip.Bottom || istop <= clip.Top {
				return
			}
			if istart < clip.Top {
				fstart += slope * Fixed(clip.Top-istart)
				istart = clip.Top
				scaleStart = 64
				if istop-istart == 1 {
					// we are within a single pixel
					scaleStart = contribution64(y1)
					scaleStop = 0
				}
			}
			if istop > clip.Bottom {
				istop = clip.Bottom
				scaleStop = 0 // so we don't draw this last row
			}

			if istart == istop {
				return
			}

			// now test if our X values are completely inside the clip
			var left, right int32
			if slope >= 0 { // L2R
				left = FixedFloorToInt(fstart - FixedHalf)
				right = FixedCeilToInt(fstart + Fixed(istop-istart-1)*slope + FixedHalf)
			} else { // R2L
				right = FixedCeilToInt(fstart + FixedHalf)
				left = FixedFloorToInt(fstart + Fixed(istop-istart-1)*slope - FixedHalf)
			}
			if outsetBeforeClipTest {
				left--
				right++
			}
			if left >= clip.Right || right <= clip.Left {
				return
			}
			if clip.Left <= left && clip.Right >= right {
				clip = nil
			}
		}
	}

	var rectClipper RectClipBlitter
	if clip != nil {
		rectClipper.Init(blitter, *clip)
		blitter = &rectClipper
	}

	switch mode {
	case 0:
		hairBlitter = &hlineAntiHairBlitter{blitter: blitter}
	case 1:
		hairBlitter = &horishAntiHairBlitter{blitter: blitter}
	case 2:
		hairBlitter = &vlineAntiHairBlitter{blitter: blitter}
	default:
		hairBlitter = &vertishAntiHairBlitter{blitter: blitter}
	}

	fstart = hairBlitter.drawCap(istart, fstart, slope, scaleStart)
	istart++
	fullSpans := istop - istart
	if scaleStop > 0 {
		fullSpans--
	}
	if fullSpans > 0 {
		fstart = hairBlitter.drawLine(istart, istart+fullSpans, fstart, slope)
	}
	if scaleStop > 0 {
		hairBlitter.drawCap(istop-1, fstart, slope, scaleStop)
	}
}

// AntiHairLineRgn draws len(pts)-1 antialiased hairline segments, clipped to the region (nil = no clip).
func AntiHairLineRgn(pts []geom.Point, clip *Region, blitter Blitter) {
	if clip != nil && clip.IsEmpty() {
		return
	}

	const maxCoord = float32(32767)
	fixedBounds := geom.RectLTRB(-maxCoord, -maxCoord, maxCoord, maxCoord)

	var clipBounds geom.Rect
	if clip != nil {
		clipBounds = clip.Bounds().ToRect()
		// We perform integral clipping later on, but we do a scalar clip first to ensure that our coordinates are
		// expressible in fixed/integers.
		//
		// antialiased hairlines can draw up to 1/2 of a pixel outside of their bounds, so we need to outset the clip
		// before calling the clipper. To make the numerics safer, we outset by a whole pixel, since the 1/2 pixel
		// boundary is important to the antihair blitter, we don't want to risk numerical fate by chopping on that edge.
		clipBounds = clipBounds.Outset(1, 1)
	}

	for i := 0; i < len(pts)-1; i++ {
		var seg, clipped [2]geom.Point

		// We have to pre-clip the line to fit in a Fixed, so we just chop the line.
		seg[0] = pts[i]
		seg[1] = pts[i+1]
		if !geom.IntersectLine(&seg, fixedBounds, &clipped) {
			continue
		}

		if clip != nil && !geom.IntersectLine(&clipped, clipBounds, &clipped) {
			continue
		}

		x0 := FloatToFDot6(clipped[0].X)
		y0 := FloatToFDot6(clipped[0].Y)
		x1 := FloatToFDot6(clipped[1].X)
		y1 := FloatToFDot6(clipped[1].Y)

		if clip != nil {
			left := min(x0, x1)
			top := min(y0, y1)
			right := max(x0, x1)
			bottom := max(y0, y1)

			ir := geom.IRectLTRB(FDot6Floor(left)-1, FDot6Floor(top)-1, FDot6Ceil(right)+1, FDot6Ceil(bottom)+1)

			if clip.QuickReject(ir) {
				continue
			}
			if !clip.QuickContains(ir) {
				for c := NewRegionCliperator(clip, ir); !c.Done(); c.Next() {
					r := c.Rect()
					doAntiHairLine(x0, y0, x1, y1, &r, blitter)
				}
				continue
			}
			// fall through to no-clip case
		}
		doAntiHairLine(x0, y0, x1, y1, nil, blitter)
	}
}

// AntiHairLine draws antialiased hairline segments restricted to a raster clip.
func AntiHairLine(pts []geom.Point, clip *Clip, blitter Blitter) {
	if clip.IsBW() {
		AntiHairLineRgn(pts, clip.BWRgn(), blitter)
		return
	}

	var clipRgn *Region

	r := boundsOrEmpty(pts)

	var wrap AAClipBlitterWrapper
	if !clip.QuickContains(r.RoundOut().Inset(-1, -1)) {
		wrap.Init(clip, blitter)
		blitter = wrap.Blitter()
		clipRgn = wrap.Rgn()
	}
	AntiHairLineRgn(pts, clipRgn, blitter)
}

// boundsOrEmpty returns the bounds of pts, or the empty rect if pts is empty or contains a non-finite value.
func boundsOrEmpty(pts []geom.Point) geom.Rect {
	var r geom.Rect
	if !r.SetBounds(pts) {
		return geom.Rect{}
	}
	return r
}

// AntiHairRect draws an antialiased hairline rectangle outline.
func AntiHairRect(rect geom.Rect, clip *Clip, blitter Blitter) {
	var pts [5]geom.Point
	pts[0] = geom.Point{X: rect.Left, Y: rect.Top}
	pts[1] = geom.Point{X: rect.Right, Y: rect.Top}
	pts[2] = geom.Point{X: rect.Right, Y: rect.Bottom}
	pts[3] = geom.Point{X: rect.Left, Y: rect.Bottom}
	pts[4] = pts[0]
	AntiHairLine(pts[:], clip, blitter)
}

///////////////////////////////////////////////////////////////////////////////
// FDot8 (24.8) AA rect machinery

// fDot8 is a 24.8 integer fixed-point value.
type fDot8 int32

func fixedToFDot8(x Fixed) fDot8 {
	return fDot8((x + 0x80) >> 8)
}

// doScanline blits one scanline of an AA-filled rect at the given alpha, splitting off the fractional left/right edge
// columns.
func doScanline(l fDot8, top int32, r fDot8, alpha uint32, blitter Blitter) {
	if (l >> 8) == ((r - 1) >> 8) { // 1x1 pixel
		blitter.BlitV(int32(l>>8), top, 1, Alpha(alphaMul(alpha, uint32(r-l))))
		return
	}

	left := int32(l >> 8)

	if l&0xFF != 0 {
		blitter.BlitV(left, top, 1, Alpha(alphaMul(alpha, uint32(256-(l&0xFF)))))
		left++
	}

	rite := int32(r >> 8)
	width := rite - left
	if width > 0 {
		callHLineBlitter(blitter, left, top, width, alpha)
	}
	if r&0xFF != 0 {
		blitter.BlitV(rite, top, 1, Alpha(alphaMul(alpha, uint32(r&0xFF))))
	}
}

// antiFillDot8 AA-fills the rect [l, t, r, b) (in 24.8 fixed point), splitting off the fractional top and bottom edge
// rows.
func antiFillDot8(l, t, r, b fDot8, blitter Blitter, fillInner bool) {
	// check for empty now that we're in our reduced precision space
	if l >= r || t >= b {
		return
	}
	top := int32(t >> 8)
	if top == int32((b-1)>>8) { // just one scanline high
		doScanline(l, top, r, uint32(b-t-1), blitter)
		return
	}

	if t&0xFF != 0 {
		doScanline(l, top, r, uint32(256-(t&0xFF)), blitter)
		top++
	}

	bot := int32(b >> 8)
	height := bot - top
	if height > 0 {
		left := int32(l >> 8)
		if left == int32((r-1)>>8) { // just 1-pixel wide
			blitter.BlitV(left, top, height, Alpha(r-l-1))
		} else {
			if l&0xFF != 0 {
				blitter.BlitV(left, top, height, Alpha(256-(l&0xFF)))
				left++
			}
			rite := int32(r >> 8)
			width := rite - left
			if width > 0 && fillInner {
				blitter.BlitRect(left, top, width, height)
			}
			if r&0xFF != 0 {
				blitter.BlitV(rite, top, height, Alpha(r&0xFF))
			}
		}
	}

	if b&0xFF != 0 {
		doScanline(l, bot, r, uint32(b&0xFF), blitter)
	}
}

// antiFillXRectBlit AA-fills xr with no clipping.
func antiFillXRectBlit(xr XRect, blitter Blitter) {
	antiFillDot8(fixedToFDot8(xr.Left), fixedToFDot8(xr.Top),
		fixedToFDot8(xr.Right), fixedToFDot8(xr.Bottom), blitter, true)
}

// AntiFillXRectRegion AA-fills xr restricted to an optional region clip.
func AntiFillXRectRegion(xr XRect, clip *Region, blitter Blitter) {
	if clip == nil {
		antiFillXRectBlit(xr, blitter)
		return
	}
	outerBounds := xr.RoundOut()

	if clip.IsRect() {
		clipBounds := clip.Bounds()

		if clipBounds.ContainsRect(outerBounds) {
			antiFillXRectBlit(xr, blitter)
		} else {
			// this keeps our original edges fractional
			tmpR := XRectFromIRect(clipBounds)
			if tmpR.Intersect(xr) {
				antiFillXRectBlit(tmpR, blitter)
			}
		}
		return
	}
	for c := NewRegionCliperator(clip, outerBounds); !c.Done(); c.Next() {
		// this keeps our original edges fractional
		tmpR := XRectFromIRect(c.Rect())
		if tmpR.Intersect(xr) {
			antiFillXRectBlit(tmpR, blitter)
		}
	}
}

// AntiFillXRect AA-fills xr restricted to a raster clip.
func AntiFillXRect(xr XRect, clip *Clip, blitter Blitter) {
	if clip.IsBW() {
		AntiFillXRectRegion(xr, clip.BWRgn(), blitter)
		return
	}
	outerBounds := xr.RoundOut()

	if clip.QuickContains(outerBounds) {
		AntiFillXRectRegion(xr, nil, blitter)
	} else {
		var wrapper AAClipBlitterWrapper
		wrapper.Init(clip, blitter)
		AntiFillXRectRegion(xr, wrapper.Rgn(), wrapper.Blitter())
	}
}

// antiFillRectBlit AA-fills r with no clipping. r has already been clipped, so it is safe to convert to fixed point
// without overflow.
func antiFillRectBlit(r geom.Rect, blitter Blitter) {
	antiFillXRectBlit(XRectFromRect(r), blitter)
}

// AntiFillRectRegion AA-fills origR restricted to an optional region clip. We repeat the clipping logic of
// AntiFillXRect because the float rect might overflow if we blindly converted it to an XRect: r is clipped into one or
// more smaller float rects first.
func AntiFillRectRegion(origR geom.Rect, clip *Region, blitter Blitter) {
	if clip != nil {
		newR := clip.Bounds().ToRect()
		if !newR.Intersect(origR) {
			return
		}

		outerBounds := newR.RoundOut()

		if clip.IsRect() {
			antiFillRectBlit(newR, blitter)
		} else {
			for c := NewRegionCliperator(clip, outerBounds); !c.Done(); c.Next() {
				newR = c.Rect().ToRect()
				if newR.Intersect(origR) {
					antiFillRectBlit(newR, blitter)
				}
			}
		}
		return
	}
	antiFillRectBlit(origR, blitter)
}

// AntiFillRect AA-fills r restricted to a raster clip.
func AntiFillRect(r geom.Rect, clip *Clip, blitter Blitter) {
	if clip.IsBW() {
		AntiFillRectRegion(r, clip.BWRgn(), blitter)
		return
	}
	var wrap AAClipBlitterWrapper
	wrap.Init(clip, blitter)
	AntiFillRectRegion(r, wrap.Rgn(), wrap.Blitter())
}

///////////////////////////////////////////////////////////////////////////////
// AA frame (stroked rect)

// fillCheckRect calls BlitRect if the rectangle is non-empty.
func fillCheckRect(l, t, r, b int32, blitter Blitter) {
	if l < r && t < b {
		blitter.BlitRect(l, t, r-l, b-t)
	}
}

func scalarToFDot8(x float32) fDot8 {
	return fDot8(x * 256)
}

func fDot8Floor(x fDot8) int32 {
	return int32(x >> 8)
}

func fDot8Ceil(x fDot8) int32 {
	return int32((x + 0xFF) >> 8)
}

// invAlphaMul computes 1 - (1 - a)*(1 - b), with precise rounding (not just a plain alpha multiply) so that values like
// a=228, b=252 don't overflow the result.
func invAlphaMul(a, b uint32) uint32 {
	return a + b - uint32(colorcore.MulDiv255Round(uint8(a), uint8(b)))
}

// innerScanline blits one scanline of the inner edge of an AA stroke frame, using the inverse coverage bias.
func innerScanline(l fDot8, top int32, r fDot8, alpha uint32, blitter Blitter) {
	if (l >> 8) == ((r - 1) >> 8) { // 1x1 pixel
		widClamp := int32(r - l)
		// border case clamp 256 to 255 instead of going through call_hline_blitter (skbug/4406)
		widClamp -= widClamp >> 8
		blitter.BlitV(int32(l>>8), top, 1, Alpha(invAlphaMul(alpha, uint32(widClamp))))
		return
	}

	left := int32(l >> 8)
	if l&0xFF != 0 {
		blitter.BlitV(left, top, 1, Alpha(invAlphaMul(alpha, uint32(l&0xFF))))
		left++
	}

	rite := int32(r >> 8)
	width := rite - left
	if width > 0 {
		callHLineBlitter(blitter, left, top, width, alpha)
	}

	if r&0xFF != 0 {
		blitter.BlitV(rite, top, 1, Alpha(invAlphaMul(alpha, uint32(^r&0xFF))))
	}
}

// innerStrokeDot8 strokes the inner edge of an AA frame around [l, t, r, b) (in 24.8 fixed point), using the inverse
// coverage bias.
func innerStrokeDot8(l, t, r, b fDot8, blitter Blitter) {
	top := int32(t >> 8)
	if top == int32((b-1)>>8) { // just one scanline high
		// We want the inverse of B-T, since we're the inner-stroke
		alpha := 256 - int32(b-t)
		if alpha != 0 {
			innerScanline(l, top, r, uint32(alpha), blitter)
		}
		return
	}

	if t&0xFF != 0 {
		innerScanline(l, top, r, uint32(t&0xFF), blitter)
		top++
	}

	bot := int32(b >> 8)
	height := bot - top
	if height > 0 {
		if l&0xFF != 0 {
			blitter.BlitV(int32(l>>8), top, height, Alpha(l&0xFF))
		}
		if r&0xFF != 0 {
			blitter.BlitV(int32(r>>8), top, height, Alpha(^r&0xFF))
		}
	}

	if b&0xFF != 0 {
		innerScanline(l, bot, r, uint32(^b&0xFF), blitter)
	}
}

// alignThinStroke: for sub-unit strokes whose edges fall within the same pixel, snap the outer edge to the pixel
// boundary so the frame logic neither double-blits nor miscomputes coverage.
func alignThinStroke(edge1, edge2 *fDot8) {
	if fDot8Floor(*edge1) == fDot8Floor(*edge2) {
		*edge2 -= *edge1 & 0xFF
		*edge1 &^= 0xFF
	}
}

// AntiFrameRectRegion draws an antialiased, centered frame (miter-join stroke) around r, restricted to an optional
// region clip.
func AntiFrameRectRegion(r geom.Rect, strokeSize geom.Point, clip *Region, blitter Blitter) {
	rx := strokeSize.X / 2
	ry := strokeSize.Y / 2

	// If we're empty on either axis, we remove the outset amount, to be sure we stroke the same way a polygon would
	// (i.e. it would just see a "line" and not extend it for the miter join).
	if r.Width() == 0 {
		ry = 0
	}
	if r.Height() == 0 {
		rx = 0
	}

	// outset by the radius
	outerL := scalarToFDot8(r.Left - rx)
	outerT := scalarToFDot8(r.Top - ry)
	outerR := scalarToFDot8(r.Right + rx)
	outerB := scalarToFDot8(r.Bottom + ry)

	// set outer to the outer rect of the outer section
	outer := geom.IRectLTRB(fDot8Floor(outerL), fDot8Floor(outerT), fDot8Ceil(outerR), fDot8Ceil(outerB))

	var clipper blitterClipper
	if clip != nil {
		if clip.QuickReject(outer) {
			return
		}
		if !clip.ContainsRect(outer) {
			blitter = clipper.apply(blitter, clip, &outer)
		}
		// now we can ignore clip for the rest of the function
	}

	// in case we lost a bit with diameter/2
	rx = strokeSize.X - rx
	ry = strokeSize.Y - ry

	// inset by the radius
	innerL := scalarToFDot8(r.Left + rx)
	innerT := scalarToFDot8(r.Top + ry)
	innerR := scalarToFDot8(r.Right - rx)
	innerB := scalarToFDot8(r.Bottom - ry)

	// For sub-unit strokes, tweak the hulls such that one of the edges coincides with the pixel edge. This ensures that
	// the general rect stroking logic below
	//   a) doesn't blit the same scanline twice
	//   b) computes the correct coverage when both edges fall within the same pixel
	if strokeSize.X < 1 || strokeSize.Y < 1 {
		alignThinStroke(&outerL, &innerL)
		alignThinStroke(&outerT, &innerT)
		alignThinStroke(&innerR, &outerR)
		alignThinStroke(&innerB, &outerB)
	}

	// stroke the outer hull
	antiFillDot8(outerL, outerT, outerR, outerB, blitter, false)

	// set outer to the outer rect of the middle section
	outer = geom.IRectLTRB(fDot8Ceil(outerL), fDot8Ceil(outerT), fDot8Floor(outerR), fDot8Floor(outerB))

	if innerL >= innerR || innerT >= innerB {
		fillCheckRect(outer.Left, outer.Top, outer.Right, outer.Bottom, blitter)
	} else {
		// set inner to the inner rect of the middle section
		inner := geom.IRectLTRB(fDot8Floor(innerL), fDot8Floor(innerT), fDot8Ceil(innerR), fDot8Ceil(innerB))

		// draw the frame in 4 pieces
		fillCheckRect(outer.Left, outer.Top, outer.Right, inner.Top, blitter)
		fillCheckRect(outer.Left, inner.Top, inner.Left, inner.Bottom, blitter)
		fillCheckRect(inner.Right, inner.Top, outer.Right, inner.Bottom, blitter)
		fillCheckRect(outer.Left, inner.Bottom, outer.Right, outer.Bottom, blitter)

		// now stroke the inner rect, which is similar to antifilldot8() except that it treats the fractional
		// coordinates with the inverse bias (since its inner).
		innerStrokeDot8(innerL, innerT, innerR, innerB, blitter)
	}
}

// AntiFrameRect draws an antialiased, centered frame (miter-join stroke) around r, restricted to a raster clip.
func AntiFrameRect(r geom.Rect, strokeSize geom.Point, clip *Clip, blitter Blitter) {
	if clip.IsBW() {
		AntiFrameRectRegion(r, strokeSize, clip.BWRgn(), blitter)
		return
	}
	var wrap AAClipBlitterWrapper
	wrap.Init(clip, blitter)
	AntiFrameRectRegion(r, strokeSize, wrap.Rgn(), wrap.Blitter())
}
