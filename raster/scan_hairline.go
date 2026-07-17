// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Non-antialiased hairlines (horiline/vertline DDA), the hairline rect (HairRect) and frame (FrameRect) converters, and
// the hairline path walkers (butt/square/round caps over lines, quads, conics and cubics) shared by the AA and non-AA
// hair entry points.

package raster

import (
	"math"
	"math/bits"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// HairRgnProc draws len(pts)-1 hairline segments through an optional region clip.
type HairRgnProc func(pts []geom.Point, clip *Region, blitter Blitter)

// horiline is a mostly-horizontal hairline DDA over x.
func horiline(x, stopx int32, fy, dy Fixed, blitter Blitter) {
	for ; x < stopx; x++ {
		blitter.BlitH(x, int32(fy>>16), 1)
		fy += dy
	}
}

// vertline is a mostly-vertical hairline DDA over y.
func vertline(y, stopy int32, fx, dx Fixed, blitter Blitter) {
	for ; y < stopy; y++ {
		blitter.BlitH(int32(fx>>16), y, 1)
		fx += dx
	}
}

// HairLineRgn draws non-AA hairline segments through an optional region clip.
func HairLineRgn(pts []geom.Point, clip *Region, origBlitter Blitter) {
	var clipper blitterClipper

	const maxCoord = float32(32767)
	fixedBounds := geom.RectLTRB(-maxCoord, -maxCoord, maxCoord, maxCoord)

	var clipBounds geom.Rect
	if clip != nil {
		clipBounds = clip.Bounds().ToRect()
	}

	for i := 0; i < len(pts)-1; i++ {
		blitter := origBlitter

		var seg, clipped [2]geom.Point
		seg[0] = pts[i]
		seg[1] = pts[i+1]

		// We have to pre-clip the line to fit in a Fixed, so we just chop the line.
		if !geom.IntersectLine(&seg, fixedBounds, &clipped) {
			continue
		}

		// Perform a clip in scalar space, so we catch huge values which might be missed after we convert to FDot6
		// (overflow).
		if clip != nil && !geom.IntersectLine(&clipped, clipBounds, &clipped) {
			continue
		}

		x0 := FloatToFDot6(clipped[0].X)
		y0 := FloatToFDot6(clipped[0].Y)
		x1 := FloatToFDot6(clipped[1].X)
		y1 := FloatToFDot6(clipped[1].Y)

		if clip != nil {
			// now perform clipping again, as the rounding to dot6 can wiggle us. our rects are really dot6 rects, but
			// since we've already used lineclipper, we know they will fit in 32bits (26.6)
			bounds := clip.Bounds()

			clipR := geom.IRectLTRB(bounds.Left<<6, bounds.Top<<6, bounds.Right<<6, bounds.Bottom<<6)
			ptsR := geom.IRectLTRB(int32(x0), int32(y0), int32(x1), int32(y1)).Sorted()

			// outset the right and bottom, to account for how hairlines are actually drawn, which may hit the pixel to
			// the right or below of the coordinate
			ptsR.Right += int32(FDot6One)
			ptsR.Bottom += int32(FDot6One)

			if !ptsR.Intersects(clipR) {
				continue
			}
			if !clip.IsRect() || !clipR.ContainsRect(ptsR) {
				blitter = clipper.apply(origBlitter, clip, nil)
			}
		}

		dx := x1 - x0
		dy := y1 - y0

		if FixedAbs(Fixed(dx)) > FixedAbs(Fixed(dy)) { // mostly horizontal
			if x0 > x1 { // we want to go left-to-right
				x0, x1 = x1, x0
				y0 = y1 // only the starting y matters below, so the other half of the swap is elided
			}
			ix0 := FDot6Round(x0)
			ix1 := FDot6Round(x1)
			if ix0 == ix1 { // too short to draw
				continue
			}
			slope := FixedDiv(Fixed(dy), Fixed(dx))
			startY := FDot6ToFixed(y0) + (slope*Fixed((32-x0)&63))>>6

			horiline(ix0, ix1, startY, slope, blitter)
		} else { // mostly vertical
			if y0 > y1 { // we want to go top-to-bottom
				x0 = x1 // only the starting x matters below, so the other half of the swap is elided
				y0, y1 = y1, y0
			}
			iy0 := FDot6Round(y0)
			iy1 := FDot6Round(y1)
			if iy0 == iy1 { // too short to draw
				continue
			}
			slope := FixedDiv(Fixed(dx), Fixed(dy))
			startX := FDot6ToFixed(x0) + (slope*Fixed((32-y0)&63))>>6

			vertline(iy0, iy1, startX, slope, blitter)
		}
	}
}

// HairRect draws a non-AA hairline rectangle. We don't just draw 4 lines, 'cause that can leave a gap in the
// bottom-right and double-hit the top-left.
func HairRect(rect geom.Rect, clip *Clip, blitter Blitter) {
	var wrapper AAClipBlitterWrapper
	var clipper blitterClipper
	// Create the enclosing bounds of the hairrect. i.e. we will stroke the interior of r.
	r := geom.IRectLTRB(geom.FloorToInt(rect.Left), geom.FloorToInt(rect.Top),
		geom.FloorToInt(rect.Right+1), geom.FloorToInt(rect.Bottom+1))

	// Note: r might be crazy big, if rect was huge, possibly getting pinned to max/min s32. We need to trim it back to
	// something reasonable before we can query its width etc. since r.fRight - r.fLeft might wrap around to negative
	// even if fRight > fLeft.
	//
	// We outset the clip bounds by 1 before intersecting, since r is being stroked and not filled so we don't want to
	// pin an edge of it to the clip. The intersect's job is mostly to just get the actual edge values into a reasonable
	// range (e.g. so width() can't overflow).
	if !r.Intersect(clip.Bounds().Inset(-1, -1)) {
		return
	}

	if clip.QuickReject(r) {
		return
	}
	if !clip.QuickContains(r) {
		var clipRgn *Region
		if clip.IsBW() {
			clipRgn = clip.BWRgn()
		} else {
			wrapper.Init(clip, blitter)
			clipRgn = wrapper.Rgn()
			blitter = wrapper.Blitter()
		}
		blitter = clipper.apply(blitter, clipRgn, nil)
	}

	width := r.Width()
	height := r.Height()

	if width == 0 || height == 0 {
		return
	}
	if width <= 2 || height <= 2 {
		blitter.BlitRect(r.Left, r.Top, width, height)
		return
	}
	// if we get here, we know we have 4 segments to draw
	blitter.BlitH(r.Left, r.Top, width)               // top
	blitter.BlitRect(r.Left, r.Top+1, 1, height-2)    // left
	blitter.BlitRect(r.Right-1, r.Top+1, 1, height-2) // right
	blitter.BlitH(r.Left, r.Bottom-1, width)          // bottom
}

///////////////////////////////////////////////////////////////////////////////
// hair curve subdivision

const (
	maxCubicSubdivideLevel = 9
	maxQuadSubdivideLevel  = 5
)

// computeIntQuadDist estimates, in whole pixels, how far the quad's control point is from the line connecting its
// endpoints.
func computeIntQuadDist(pts []geom.Point) uint32 {
	// compute the vector between the control point ([1]) and the middle of the line connecting the start and end ([0]
	// and [2])
	dx := (pts[0].X+pts[2].X)/2 - pts[1].X
	dy := (pts[0].Y+pts[2].Y)/2 - pts[1].Y
	// we want everyone to be positive
	dx = geom.ScalarAbs(dx)
	dy = geom.ScalarAbs(dy)
	// convert to whole pixel values (use ceiling to be conservative). assign to unsigned so we can safely add 1/2 of
	// the smaller and still fit in uint32, since CeilToInt returns 31 bits at most.
	idx := uint32(geom.CeilToInt(dx))
	idy := uint32(geom.CeilToInt(dy))
	// use the cheap approx for distance
	if idx > idy {
		return idx + (idy >> 1)
	}
	return idy + (idx >> 1)
}

// computeQuadLevel computes the subdivision level for hairQuadSubdivide: quadratics approach the line connecting their
// start and end points 4x closer with each subdivision, so this computes the number of subdivisions needed to get that
// distance to be less than a pixel.
func computeQuadLevel(pts []geom.Point) int {
	d := computeIntQuadDist(pts)
	level := (33 - bits.LeadingZeros32(d)) >> 1
	// safety check on level (from the previous version)
	if level > maxQuadSubdivideLevel {
		level = maxQuadSubdivideLevel
	}
	return level
}

// hairQuadSubdivide flattens the quad into 1<<level line segments and hands them to the line proc.
func hairQuadSubdivide(pts []geom.Point, clip *Region, blitter Blitter, level int, lineproc HairRgnProc) {
	lines := int32(1) << level
	dt := float32(1) / float32(lines)

	var tmp [(1 << maxQuadSubdivideLevel) + 1]geom.Point

	// Quadratic Bezier coefficients: C = P0, B = 2(P1-P0), A = P2 - 2P1 + P0.
	cx, cy := pts[0].X, pts[0].Y
	bx, by := 2*(pts[1].X-pts[0].X), 2*(pts[1].Y-pts[0].Y)
	ax, ay := pts[2].X-2*pts[1].X+pts[0].X, pts[2].Y-2*pts[1].Y+pts[0].Y

	tmp[0] = pts[0]
	t := float32(0)
	for i := int32(1); i < lines; i++ {
		t += dt
		tmp[i] = geom.Point{X: (ax*t+bx)*t + cx, Y: (ay*t+by)*t + cy}
	}
	tmp[lines] = pts[2]
	lineproc(tmp[:lines+1], clip, blitter)
}

// computeNocheckBounds returns the point bounds with no finite/empty checks.
func computeNocheckBounds(pts []geom.Point) geom.Rect {
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := minX, minY
	for _, p := range pts[1:] {
		minX = min32(minX, p.X)
		minY = min32(minY, p.Y)
		maxX = max32(maxX, p.X)
		maxY = max32(maxY, p.Y)
	}
	return geom.RectLTRB(minX, minY, maxX, maxY)
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func isInverted(r geom.Rect) bool {
	return r.Left > r.Right || r.Top > r.Bottom
}

// geometricOverlap reports whether a and b overlap. Can't call Rect.Intersects, since it cares about empty, and we
// don't (since we're tracking something to be stroked, so empty can still draw something, e.g. a horizontal line).
func geometricOverlap(a, b geom.Rect) bool {
	return a.Left < b.Right && b.Left < a.Right && a.Top < b.Bottom && b.Top < a.Bottom
}

// geometricContains reports whether outer contains inner (see geometricOverlap for why not Rect.Contains).
func geometricContains(outer, inner geom.Rect) bool {
	return inner.Right <= outer.Right && inner.Left >= outer.Left &&
		inner.Bottom <= outer.Bottom && inner.Top >= outer.Top
}

// hairQuad does per-segment culling against the cached inset/outset clip rects, then subdivision.
func hairQuad(pts []geom.Point, clip *Region, insetClip, outsetClip *geom.Rect, blitter Blitter, level int, lineproc HairRgnProc) {
	if insetClip != nil {
		bounds := computeNocheckBounds(pts[:3])
		if !geometricOverlap(*outsetClip, bounds) {
			return
		} else if geometricContains(*insetClip, bounds) {
			clip = nil
		}
	}

	hairQuadSubdivide(pts, clip, blitter, level, lineproc)
}

// computeCubicSegs computes the number of line segments (a power of two) needed to flatten the cubic within tolerance.
func computeCubicSegs(pts []geom.Point) int32 {
	const oneThird = float32(1.0 / 3.0)
	const twoThird = float32(2.0 / 3.0)

	p13x := oneThird*pts[3].X + twoThird*pts[0].X
	p13y := oneThird*pts[3].Y + twoThird*pts[0].Y
	p23x := oneThird*pts[0].X + twoThird*pts[3].X
	p23y := oneThird*pts[0].Y + twoThird*pts[3].Y

	diff := max32(
		max32(geom.ScalarAbs(pts[1].X-p13x), geom.ScalarAbs(pts[1].Y-p13y)),
		max32(geom.ScalarAbs(pts[2].X-p23x), geom.ScalarAbs(pts[2].Y-p23y)),
	)
	tol := float32(1.0 / 8)

	for i := 0; i < maxCubicSubdivideLevel; i++ {
		if diff < tol {
			return 1 << i
		}
		tol *= 4
	}
	return 1 << maxCubicSubdivideLevel
}

// lt90 reports whether the angle p0-pivot-p2 is at most 90 degrees.
func lt90(p0, pivot, p2 geom.Point) bool {
	return p0.Sub(pivot).Dot(p2.Sub(pivot)) >= 0
}

// quickCubicNicenessCheck reports whether the off-curve points are "inside" the limits of the on-curve pts.
func quickCubicNicenessCheck(pts []geom.Point) bool {
	return lt90(pts[1], pts[0], pts[3]) &&
		lt90(pts[2], pts[0], pts[3]) &&
		lt90(pts[1], pts[3], pts[0]) &&
		lt90(pts[2], pts[3], pts[0])
}

func float32IsFinite(x float32) bool {
	return math.Float32bits(x)&0x7F800000 != 0x7F800000
}

// hairCubicSubdivide flattens into computed segments; if any evaluated point is non-finite, draw nothing.
func hairCubicSubdivide(pts []geom.Point, clip *Region, blitter Blitter, lineproc HairRgnProc) {
	lines := computeCubicSegs(pts)
	if lines == 1 {
		tmp := [2]geom.Point{pts[0], pts[3]}
		lineproc(tmp[:], clip, blitter)
		return
	}

	// Cubic Bezier coefficients: A = P3 + 3(P1 - P2) - P0, B = 3(P2 - 2P1 + P0), C = 3(P1 - P0), D = P0.
	ax := pts[3].X + 3*(pts[1].X-pts[2].X) - pts[0].X
	ay := pts[3].Y + 3*(pts[1].Y-pts[2].Y) - pts[0].Y
	bx := 3 * (pts[2].X - 2*pts[1].X + pts[0].X)
	by := 3 * (pts[2].Y - 2*pts[1].Y + pts[0].Y)
	cx := 3 * (pts[1].X - pts[0].X)
	cy := 3 * (pts[1].Y - pts[0].Y)
	dx := pts[0].X
	dy := pts[0].Y

	dt := float32(1) / float32(lines)
	t := float32(0)

	var tmp [(1 << maxCubicSubdivideLevel) + 1]geom.Point
	tmp[0] = pts[0]
	isFinite := true
	for i := int32(1); i < lines; i++ {
		t += dt
		px := ((ax*t+bx)*t+cx)*t + dx
		py := ((ay*t+by)*t+cy)*t + dy
		isFinite = isFinite && float32IsFinite(px) && float32IsFinite(py)
		tmp[i] = geom.Point{X: px, Y: py}
	}
	if isFinite {
		tmp[lines] = pts[3]
		lineproc(tmp[:lines+1], clip, blitter)
	} // else some point(s) are non-finite, so don't draw
}

// hairCubic culls, then subdivides directly when the control points are "nice", else chops at max curvature first.
func hairCubic(pts []geom.Point, clip *Region, insetClip, outsetClip *geom.Rect, blitter Blitter, lineproc HairRgnProc) {
	if insetClip != nil {
		bounds := computeNocheckBounds(pts[:4])
		if !geometricOverlap(*outsetClip, bounds) {
			return
		} else if geometricContains(*insetClip, bounds) {
			clip = nil
		}
	}

	if quickCubicNicenessCheck(pts) {
		hairCubicSubdivide(pts, clip, blitter, lineproc)
	} else {
		var tmp [13]geom.Point
		count := geom.ChopCubicAtMaxCurvature(pts, tmp[:])
		for i := 0; i < count; i++ {
			hairCubicSubdivide(tmp[i*3:i*3+4], clip, blitter, lineproc)
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// hair path walking

// capOutsets holds the per-cap extension amounts used by extendPts: the area of a circle is PI*R*R; for a unit circle R
// = 1/2, and the cap covers half of that.
const (
	squareCapOutset = float32(0.5)
	roundCapOutset  = float32(math.Pi / 8)
)

// extendPts extends the points in the direction of the starting or ending tangent by the cap outset. If there's no
// distance between the end point and the control point, use the next control point to create a tangent. If the curve is
// degenerate, move the cap out 1/2 unit horizontally.
func extendPts(capOutset float32, prevVerb, nextVerb path.Verb, hasPrev, hasNext bool, pts []geom.Point) {
	if hasPrev && prevVerb == path.VerbMove {
		first := 0
		ctrl := 0
		controls := len(pts) - 1
		var tangent geom.Point
		for {
			ctrl++
			tangent = pts[first].Sub(pts[ctrl])
			if !tangent.IsZero() {
				break
			}
			controls--
			if controls <= 0 {
				break
			}
		}
		if tangent.IsZero() {
			tangent = geom.Point{X: 1, Y: 0}
			controls = len(pts) - 1 // If all points are equal, move all but one
		} else {
			tangent.Normalize()
		}
		for { // If the end point and control points are equal, loop to move them in tandem.
			pts[first].X += tangent.X * capOutset
			pts[first].Y += tangent.Y * capOutset
			first++
			controls++
			if controls >= len(pts) {
				break
			}
		}
	}
	if !hasNext || nextVerb == path.VerbMove || nextVerb == path.VerbClose {
		last := len(pts) - 1
		ctrl := last
		controls := len(pts) - 1
		var tangent geom.Point
		for {
			ctrl--
			tangent = pts[last].Sub(pts[ctrl])
			if !tangent.IsZero() {
				break
			}
			controls--
			if controls <= 0 {
				break
			}
		}
		if tangent.IsZero() {
			tangent = geom.Point{X: -1, Y: 0}
			controls = len(pts) - 1
		} else {
			tangent.Normalize()
		}
		for {
			pts[last].X += tangent.X * capOutset
			pts[last].Y += tangent.Y * capOutset
			last--
			controls++
			if controls >= len(pts) {
				break
			}
		}
	}
}

// HairCap identifies the cap style for the hairline path walkers.
type HairCap uint8

// HairCap values.
const (
	HairCapButt HairCap = iota
	HairCapRound
	HairCapSquare
)

// hairPath walks a path, drawing hairline segments for each verb with the given cap style.
func hairPath(p *path.Path, rclip *Clip, blitter Blitter, capStyle HairCap, lineproc HairRgnProc) {
	if p.IsEmpty() {
		return
	}

	var wrap AAClipBlitterWrapper
	var clip *Region
	var insetStorage, outsetStorage geom.Rect
	var insetClip, outsetClip *geom.Rect

	capOut := int32(1)
	if capStyle != HairCapButt {
		capOut = 2
	}
	ibounds := p.Bounds().RoundOut().Inset(-capOut, -capOut)
	if rclip.QuickReject(ibounds) {
		return
	}
	if !rclip.QuickContains(ibounds) {
		if rclip.IsBW() {
			clip = rclip.BWRgn()
		} else {
			wrap.Init(rclip, blitter)
			blitter = wrap.Blitter()
			clip = wrap.Rgn()
		}

		// We now cache two scalar rects, to use for culling per-segment (e.g. cubic). Since we're hairlining, the
		// "bounds" of the control points isn't necessarily the limit of where a segment can draw (it might draw up to 1
		// pixel beyond in aa-hairs).
		//
		// Compute the pt-bounds per segment is easy, so we do that, and then inversely adjust the culling bounds so we
		// can just do a straight compare per segment.
		//
		// insetClip is use for quick-accept (i.e. the segment is not clipped), so we inset it from the clip-bounds
		// (since segment bounds can be off by 1).
		//
		// outsetClip is used for quick-reject (i.e. the segment is entirely outside), so we outset it from the
		// clip-bounds.
		insetStorage = clip.Bounds().ToRect()
		outsetStorage = insetStorage.Outset(1, 1)
		insetStorage = insetStorage.Outset(-1, -1)
		if isInverted(insetStorage) {
			// our bounds checks assume the rects are never inverted. If insetting has created that, we assume that the
			// area is too small to safely perform a quick-accept, so we just mark the rect as empty (so the
			// quick-accept check will always fail).
			insetStorage = geom.Rect{}
		}
		if rclip.IsRect() {
			insetClip = &insetStorage
		}
		outsetClip = &outsetStorage
	}

	var pts [4]geom.Point
	var firstPt, lastPt geom.Point

	capOutset := squareCapOutset
	if capStyle == HairCapRound {
		capOutset = roundCapOutset
	}

	var prevVerb path.Verb
	hasPrevVerb := false
	iter := path.NewRawIter(p)
	for {
		verb, srcPts, weight, ok := iter.Next()
		if !ok {
			break
		}
		nextVerb, hasNextVerb := iter.PeekVerb()
		switch verb {
		case path.VerbMove:
			firstPt = srcPts[0]
			lastPt = srcPts[0]
		case path.VerbLine:
			pts[0] = srcPts[0]
			pts[1] = srcPts[1]
			if capStyle != HairCapButt {
				extendPts(capOutset, prevVerb, nextVerb, hasPrevVerb, hasNextVerb, pts[:2])
			}
			lineproc(pts[:2], clip, blitter)
			lastPt = pts[1]
		case path.VerbQuad:
			copy(pts[:3], srcPts[:3])
			if capStyle != HairCapButt {
				extendPts(capOutset, prevVerb, nextVerb, hasPrevVerb, hasNextVerb, pts[:3])
			}
			hairQuad(pts[:3], clip, insetClip, outsetClip, blitter, computeQuadLevel(pts[:3]), lineproc)
			lastPt = pts[2]
		case path.VerbConic:
			copy(pts[:3], srcPts[:3])
			if capStyle != HairCapButt {
				extendPts(capOutset, prevVerb, nextVerb, hasPrevVerb, hasNextVerb, pts[:3])
			}
			// how close should the quads be to the original conic?
			const tol = float32(1.0 / 4)
			quadPts, count := path.ConicToQuads(pts[:3], weight, tol)
			for i := 0; i < count; i++ {
				q := quadPts[i*2 : i*2+3]
				hairQuad(q, clip, insetClip, outsetClip, blitter, computeQuadLevel(q), lineproc)
			}
			lastPt = pts[2]
		case path.VerbCubic:
			copy(pts[:4], srcPts[:4])
			if capStyle != HairCapButt {
				extendPts(capOutset, prevVerb, nextVerb, hasPrevVerb, hasNextVerb, pts[:4])
			}
			hairCubic(pts[:4], clip, insetClip, outsetClip, blitter, lineproc)
			lastPt = pts[3]
		case path.VerbClose:
			pts[0] = lastPt
			pts[1] = firstPt
			if capStyle != HairCapButt && hasPrevVerb && prevVerb == path.VerbMove {
				// cap moveTo/close to match svg expectations for degenerate segments
				extendPts(capOutset, prevVerb, nextVerb, hasPrevVerb, hasNextVerb, pts[:2])
			}
			lineproc(pts[:2], clip, blitter)
		}
		if capStyle != HairCapButt {
			if hasPrevVerb && prevVerb == path.VerbMove &&
				verb >= path.VerbLine && verb <= path.VerbCubic {
				firstPt = pts[0] // the curve moved the initial point, so close to it instead
			}
			prevVerb = verb
			hasPrevVerb = true
		}
	}
}

// HairPath draws a hairline path (non-AA, butt caps).
func HairPath(p *path.Path, clip *Clip, blitter Blitter) {
	hairPath(p, clip, blitter, HairCapButt, HairLineRgn)
}

// AntiHairPath draws a hairline path (AA, butt caps).
func AntiHairPath(p *path.Path, clip *Clip, blitter Blitter) {
	hairPath(p, clip, blitter, HairCapButt, AntiHairLineRgn)
}

// HairSquarePath draws a hairline path with square caps (non-AA).
func HairSquarePath(p *path.Path, clip *Clip, blitter Blitter) {
	hairPath(p, clip, blitter, HairCapSquare, HairLineRgn)
}

// AntiHairSquarePath draws a hairline path with square caps (AA).
func AntiHairSquarePath(p *path.Path, clip *Clip, blitter Blitter) {
	hairPath(p, clip, blitter, HairCapSquare, AntiHairLineRgn)
}

// HairRoundPath draws a hairline path with round caps (non-AA).
func HairRoundPath(p *path.Path, clip *Clip, blitter Blitter) {
	hairPath(p, clip, blitter, HairCapRound, HairLineRgn)
}

// AntiHairRoundPath draws a hairline path with round caps (AA).
func AntiHairRoundPath(p *path.Path, clip *Clip, blitter Blitter) {
	hairPath(p, clip, blitter, HairCapRound, AntiHairLineRgn)
}

///////////////////////////////////////////////////////////////////////////////

// FrameRect draws a non-AA centered frame (miter-join stroke) around r, drawn as four filled rects.
func FrameRect(r geom.Rect, strokeSize geom.Point, clip *Clip, blitter Blitter) {
	if strokeSize.X < 0 || strokeSize.Y < 0 {
		return
	}

	dx := strokeSize.X
	dy := strokeSize.Y
	rx := dx / 2
	ry := dy / 2

	outer := geom.RectLTRB(r.Left-rx, r.Top-ry, r.Right+rx, r.Bottom+ry)

	if r.Width() <= dx || r.Height() <= dy {
		// If we're empty on either axis, we remove the outset amount, to be sure we stroke the same way a polygon would
		// (i.e. it would just see a "line" and not extend it for the miter join).
		if r.Width() == 0 {
			outer.Top = r.Top
			outer.Bottom = r.Bottom
		}
		if r.Height() == 0 {
			outer.Left = r.Left
			outer.Right = r.Right
		}
		FillRectRasterClip(outer, clip, blitter)
		return
	}

	tmp := geom.RectLTRB(outer.Left, outer.Top, outer.Right, outer.Top+dy)
	FillRectRasterClip(tmp, clip, blitter)
	tmp.Top = outer.Bottom - dy
	tmp.Bottom = outer.Bottom
	FillRectRasterClip(tmp, clip, blitter)

	tmp = geom.RectLTRB(outer.Left, outer.Top+dy, outer.Left+dx, outer.Bottom-dy)
	FillRectRasterClip(tmp, clip, blitter)
	tmp.Left = outer.Right - dx
	tmp.Right = outer.Right
	FillRectRasterClip(tmp, clip, blitter)
}

// HairLine draws hairline segments restricted to a raster clip.
func HairLine(pts []geom.Point, clip *Clip, blitter Blitter) {
	if clip.IsBW() {
		HairLineRgn(pts, clip.BWRgn(), blitter)
		return
	}

	var clipRgn *Region

	r := boundsOrEmpty(pts).Outset(0.5, 0.5)

	var wrap AAClipBlitterWrapper
	if !clip.QuickContains(r.RoundOut()) {
		wrap.Init(clip, blitter)
		blitter = wrap.Blitter()
		clipRgn = wrap.Rgn()
	}
	HairLineRgn(pts, clipRgn, blitter)
}
