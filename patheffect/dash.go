// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The dash path effect: dash-interval math (calcDashParameters and friends) and the dash path effect itself
// (dashEffect), including the cull optimization for lines/rects, the special-line fast path that emits the stroked
// quads directly, and the AsPoints acceleration consumed by the dashed-line draw lane.

package patheffect

import (
	"math"

	"github.com/richardwilkes/canvas/contour"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// maxDashCount is the largest number of dash segments a filter will produce before giving up, to bound memory pressure
// for extreme path-length/dash-length ratios.
const maxDashCount = float32(1000000)

func isEven(x int) bool {
	return x&1 == 0
}

// findFirstInterval returns the remaining length and index of the interval that phase falls within.
func findFirstInterval(intervals []float32, phase float32) (dashLength float32, index int) {
	for i, gap := range intervals {
		if phase > gap || (phase == gap && gap != 0) {
			phase -= gap
		} else {
			return gap - phase, i
		}
	}
	// If we get here, phase "appears" to be larger than our length. This shouldn't happen with perfect precision, but
	// we can accumulate errors during the initial length computation (rounding can make our sum be too big or too
	// small. In that event, we just have to eat the error here.
	return intervals[0], 0
}

// calcDashParameters derives the initial dash length/index and total interval length for phase and intervals, along
// with phase normalized into [0, intervalLength).
func calcDashParameters(phase float32, intervals []float32) (initialDashLength float32, initialDashIndex int, intervalLength, adjustedPhase float32) {
	length := float32(0)
	for _, interval := range intervals {
		length += interval
	}
	intervalLength = length
	// Adjust phase to be between 0 and len, "flipping" phase if negative. e.g., if len is 100, then phase of -20 (or
	// -120) is equivalent to 80
	if phase < 0 {
		phase = -phase
		if phase > length {
			phase = geom.ScalarMod(phase, length)
		}
		phase = length - phase

		// Due to finite precision, it's possible that phase == len, even after the subtract (if len >>> phase), so fix
		// that here. This fixes http://crbug.com/124652 .
		if phase == length {
			phase = 0
		}
	} else if phase >= length {
		phase = geom.ScalarMod(phase, length)
	}
	adjustedPhase = phase

	initialDashLength, initialDashIndex = findFirstInterval(intervals, phase)
	return initialDashLength, initialDashIndex, intervalLength, adjustedPhase
}

// validDashPath reports whether intervals form a usable dash pattern (even count >= 2, all non-negative, positive total
// length) and phase is finite.
func validDashPath(phase float32, intervals []float32) bool {
	if len(intervals) < 2 || len(intervals)&1 != 0 {
		return false
	}
	length := float32(0)
	for _, interval := range intervals {
		if interval < 0 {
			return false
		}
		length += interval
	}
	// watch out for values that might make us go out of bounds
	return length > 0 && scalarIsFinite(phase) && scalarIsFinite(length)
}

func scalarIsFinite(x float32) bool {
	return math.Float32bits(x)&0x7F800000 != 0x7F800000
}

// outsetForStroke outsets rect by half the stroke width (scaled by the miter limit for miter joins), so a cull test
// against the outset rect accounts for the stroke's extent.
func outsetForStroke(rect geom.Rect, rec *stroke.Rec) geom.Rect {
	radius := 0.5 * rec.Width()
	if radius == 0 {
		radius = 1 // hairlines
	}
	if rec.Join() == stroke.JoinMiter {
		radius *= rec.Miter()
	}
	return rect.Outset(radius, radius)
}

// ptCoord returns a pointer to the point's X (idx 0) or Y (idx 1).
func ptCoord(pt *geom.Point, idx int) *float32 {
	if idx == 0 {
		return &pt.X
	}
	return &pt.Y
}

// adjustZeroLengthLine bumps out the end of a zero-length line by a tiny amount to draw endcaps. The bump factor is
// sized so that the distance between the two points computes as non-zero.
func adjustZeroLengthLine(pts *[2]geom.Point) {
	pts[1].X += max(1.001, pts[1].X) * geom.ScalarNearlyZeroTol
}

// clipLine chops a horizontal or vertical line to the bounds, keeping the result in phase with the dash.
func clipLine(pts *[2]geom.Point, bounds geom.Rect, intervalLength, priorPhase float32) bool {
	dxy := pts[1].Sub(pts[0])

	// only horizontal or vertical lines
	if dxy.X != 0 && dxy.Y != 0 {
		return false
	}
	xyOffset := 0 // 0 to adjust horizontal, 1 to adjust vertical
	if dxy.Y != 0 {
		xyOffset = 1
	}

	minXY := *ptCoord(&pts[0], xyOffset)
	maxXY := *ptCoord(&pts[1], xyOffset)
	swapped := maxXY < minXY
	if swapped {
		minXY, maxXY = maxXY, minXY
	}

	var leftTop, rightBottom float32
	if xyOffset == 0 {
		leftTop, rightBottom = bounds.Left, bounds.Right
	} else {
		leftTop, rightBottom = bounds.Top, bounds.Bottom
	}
	if maxXY < leftTop || minXY > rightBottom {
		return false
	}

	// Now we actually perform the chop, removing the excess to the left/top and right/bottom of the bounds (keeping our
	// new line "in phase" with the dash, hence the (mod intervalLength).

	if minXY < leftTop {
		minXY = leftTop - geom.ScalarMod(leftTop-minXY, intervalLength)
		if !swapped {
			minXY -= priorPhase // for rectangles, adjust by prior phase
		}
	}
	if maxXY > rightBottom {
		maxXY = rightBottom + geom.ScalarMod(maxXY-rightBottom, intervalLength)
		if swapped {
			maxXY += priorPhase // for rectangles, adjust by prior phase
		}
	}

	if swapped {
		minXY, maxXY = maxXY, minXY
	}
	*ptCoord(&pts[0], xyOffset) = minXY
	*ptCoord(&pts[1], xyOffset) = maxXY

	if minXY == maxXY {
		adjustZeroLengthLine(pts)
	}
	return true
}

// cullPath handles only lines and rects. If it returns true, builder is the new smaller path, otherwise builder may
// have been changed but should be ignored.
func cullPath(srcPath *path.Path, rec *stroke.Rec, cullRect *geom.Rect, intervalLength float32, builder *path.Path) bool {
	if cullRect == nil {
		if pts, ok := srcPath.IsLine(); ok && pts[0] == pts[1] {
			adjustZeroLengthLine(&pts)
			builder.MoveToPt(pts[0])
			builder.LineToPt(pts[1])
			return true
		}
		return false
	}

	bounds := outsetForStroke(*cullRect, rec)

	if pts, ok := srcPath.IsLine(); ok {
		if clipLine(&pts, bounds, intervalLength, 0) {
			builder.MoveToPt(pts[0])
			builder.LineToPt(pts[1])
			return true
		}
		return false
	}

	if _, isRect := srcPath.IsRect(); isRect {
		// We'll break the rect into four lines, culling each separately.
		iter := path.NewIter(srcPath, false)

		_, _, _ = iter.Next() // the move

		var accum float64 // Sum of unculled edge lengths to keep the phase correct.
		//                   Intentionally a double to minimize the risk of overflow and drift.
		for {
			verb, itPts, ok := iter.Next()
			if !ok || verb != path.VerbLine {
				break
			}
			// Notice this vector v and accum work with the original unclipped length.
			v := itPts[1].Sub(itPts[0])

			pts := [2]geom.Point{itPts[0], itPts[1]}
			if clipLine(&pts, bounds, intervalLength, float32(math.Mod(accum, float64(intervalLength)))) {
				// pts[0] may have just been changed by clip_line(). If that's not where we ended the previous lineTo(),
				// we need to moveTo() there.
				if last, hasLast := builder.LastPt(); !hasLast || last != pts[0] {
					builder.MoveToPt(pts[0])
				}
				builder.LineToPt(pts[1])
			}

			// We either just traveled v.fX horizontally or v.fY vertically.
			accum += float64(geom.ScalarAbs(v.X + v.Y))
		}
		return !builder.IsEmpty()
	}

	return false
}

// specialLineRec is the fast path that emits the stroked dash quads for a simple line directly, leaving the stroke rec
// set to fill.
type specialLineRec struct {
	pts        [2]geom.Point
	tangent    geom.Point
	normal     geom.Point
	pathLength float32
}

func (r *specialLineRec) init(src *path.Path, rec *stroke.Rec, intervalCount int, intervalLength float32) bool {
	if rec.IsHairlineStyle() {
		return false
	}
	pts, ok := src.IsLine()
	if !ok {
		return false
	}
	r.pts = pts

	// can relax this in the future, if we handle square and round caps
	if rec.Cap() != stroke.CapButt {
		return false
	}

	pathLength := r.pts[0].DistanceTo(r.pts[1])

	r.tangent = r.pts[1].Sub(r.pts[0])
	if r.tangent.IsZero() {
		return false
	}

	r.pathLength = pathLength
	r.tangent = r.tangent.Scaled(geom.IEEEFloatDivide(1, pathLength))
	if !r.tangent.IsFinite() {
		return false
	}
	r.normal = r.tangent.RotateCCW()
	r.normal = r.normal.Scaled(0.5 * rec.Width())

	// now estimate how many quads will be added to the path
	//     resulting segments = pathLen * intervalCount / intervalLen
	//     resulting points = 4 * segments
	ptCount := pathLength * float32(intervalCount) / intervalLength
	ptCount = min(ptCount, maxDashCount)
	if ptCount != ptCount { // NaN
		return false
	}

	// we will take care of the stroking
	rec.SetFillStyle()
	return true
}

func (r *specialLineRec) addSegment(d0, d1 float32, dst *path.Path) {
	// clamp the segment to our length
	if d1 > r.pathLength {
		d1 = r.pathLength
	}

	x0 := r.pts[0].X + r.tangent.X*d0
	x1 := r.pts[0].X + r.tangent.X*d1
	y0 := r.pts[0].Y + r.tangent.Y*d0
	y1 := r.pts[0].Y + r.tangent.Y*d1

	pts := [4]geom.Point{
		{X: x0 + r.normal.X, Y: y0 + r.normal.Y}, // moveTo
		{X: x1 + r.normal.X, Y: y1 + r.normal.Y}, // lineTo
		{X: x1 - r.normal.X, Y: y1 - r.normal.Y}, // lineTo
		{X: x0 - r.normal.X, Y: y0 - r.normal.Y}, // lineTo
	}
	dst.AddPoly(pts[:], false)
}

// dashInternalFilter applies the dash pattern to src, allowing the stroke rec's style to be applied.
func dashInternalFilter(dst, src *path.Path, rec *stroke.Rec, cullRect *geom.Rect, intervals []float32, initialDashLength float32, initialDashIndex int, intervalLength, startPhase float32) bool {
	count := len(intervals)

	// we do nothing if the src wants to be filled
	style := rec.Style()
	if style == stroke.StyleFill || style == stroke.StyleStrokeAndFill {
		return false
	}

	dashCount := float32(0)

	// The maxDashCount bail-out below throws away everything this call produced, so when dst already holds output from
	// an earlier effect (MakeSum hands the same dst to both of its children) the dashed segments have to accumulate in a
	// scratch path that can be dropped on its own. Otherwise the bail-out would erase the caller's contribution too,
	// violating the "appending the result to dst" contract in stroke.PathEffect. dst is empty in the common case, so the
	// scratch path (and the copy it costs) is only paid for when it is actually needed.
	out := dst
	if !dst.IsEmpty() {
		out = path.Borrow()
		defer path.Recycle(out)
	}

	builder := &path.Path{}
	srcPtr := src
	if cullPath(src, rec, cullRect, intervalLength, builder) {
		// if rect is closed, starts in a dash, and ends in a dash, add the initial join potentially a better fix is
		// described here: skbug.com/40038693
		if _, isRect := src.IsRect(); isRect && src.IsLastContourClosed() && isEven(initialDashIndex) {
			pathLength := contour.NewPathMeasure(src, false, rec.ResScale()).Length()
			endPhase := geom.ScalarMod(pathLength+startPhase, intervalLength)
			index := 0
			for endPhase > intervals[index] {
				endPhase -= intervals[index]
				index++
				if index == count {
					// We have run out of intervals. endPhase "should" never get to this point, but it could if the
					// subtracts underflowed. Hence we will pin it as if it perfectly ran through the intervals. See
					// crbug.com/875494 (and skbug.com/40039544)
					endPhase = 0
					break
				}
			}
			// if dash ends inside "on", or ends at beginning of "off"
			if isEven(index) == (endPhase > 0) {
				midPoint := src.Point(0)
				// get vector at end of rect
				last := src.CountPoints() - 1
				for midPoint == src.Point(last) {
					last--
				}
				// get vector at start of rect
				next := 1
				for midPoint == src.Point(next) {
					next++
				}
				v := midPoint.Sub(src.Point(last))
				const tinyOffset = geom.ScalarNearlyZeroTol
				// scale vector to make start of tiny right angle
				v = v.Scaled(tinyOffset)
				builder.MoveToPt(midPoint.Sub(v))
				builder.LineToPt(midPoint)
				v = midPoint.Sub(src.Point(next))
				// scale vector to make end of tiny right angle
				v = v.Scaled(tinyOffset)
				builder.LineToPt(midPoint.Sub(v))
			}
		}

		srcPtr = builder
	}

	var lineRec specialLineRec
	specialLine := lineRec.init(srcPtr, rec, count>>1, intervalLength)

	meas := contour.NewPathMeasure(srcPtr, false, rec.ResScale())

	for {
		skipFirstSegment := meas.IsClosed()
		addedSegment := false
		length := meas.Length()
		index := initialDashIndex

		// Since the path length / dash length ratio may be arbitrarily large, we can exert significant memory pressure
		// while attempting to build the filtered path. To avoid this, we simply give up dashing beyond a certain
		// threshold (see maxDashCount).
		dashCount += length * float32(count>>1) / intervalLength
		if dashCount > maxDashCount {
			out.Rewind()
			return false
		}

		// Using double precision to avoid looping indefinitely due to single precision rounding (for extreme
		// path_length/dash_length ratios). See test_infinite_dash() unittest.
		distance := float64(0)
		dlen := float64(initialDashLength)

		for distance < float64(length) {
			addedSegment = false
			if isEven(index) && !skipFirstSegment {
				addedSegment = true

				if specialLine {
					lineRec.addSegment(geom.DoubleToScalar(distance), geom.DoubleToScalar(distance+dlen), out)
				} else {
					meas.GetSegment(geom.DoubleToScalar(distance), geom.DoubleToScalar(distance+dlen), out, true)
				}
			}
			distance += dlen

			// clear this so we only respect it the first time around
			skipFirstSegment = false

			// wrap around our intervals array if necessary
			index++
			if index == count {
				index = 0
			}

			// fetch our next dlen
			dlen = float64(intervals[index])
		}

		// extend if we ended on a segment and we need to join up with the (skipped) initial segment
		if meas.IsClosed() && isEven(initialDashIndex) && initialDashLength >= 0 {
			meas.GetSegment(0, initialDashLength, out, !addedSegment)
		}
		if !meas.NextContour() {
			break
		}
	}

	if out != dst {
		dst.AddPath(out, path.AddPathAppend)
	}
	return true
}

///////////////////////////////////////////////////////////////////////////////
// dash path effect

// dashEffect is the dash path effect implementation.
type dashEffect struct {
	intervals         []float32
	phase             float32
	initialDashLength float32
	intervalLength    float32
	initialDashIndex  int
}

// MakeDash creates a dash path effect: intervals are the on/off distances (even count, >= 2, all >= 0, at least one >
// 0); phase is the offset into the intervals at which to start. Returns nil for invalid inputs.
func MakeDash(intervals []float32, phase float32) stroke.PathEffect {
	if !validDashPath(phase, intervals) {
		return nil
	}
	d := &dashEffect{intervals: append([]float32(nil), intervals...)}
	d.initialDashLength, d.initialDashIndex, d.intervalLength, d.phase = calcDashParameters(phase, d.intervals)
	return d
}

// DashIntervals exports the effect's intervals (E.1: the SDFT strike-spec construction rescales dash intervals into the
// distance-field strike's space). The returned slice must not be mutated.
func (d *dashEffect) DashIntervals() []float32 { return d.intervals }

// DashPhase exports the effect's phase (see DashIntervals).
func (d *dashEffect) DashPhase() float32 { return d.phase }

func (d *dashEffect) FilterPath(dst, src *path.Path, rec *stroke.Rec, cullRect *geom.Rect, _ *geom.Matrix) bool {
	return dashInternalFilter(dst, src, rec, cullRect, d.intervals,
		d.initialDashLength, d.initialDashIndex, d.intervalLength, d.phase)
}

func (d *dashEffect) ComputeFastBounds(*geom.Rect) bool {
	// Dashing a path returns a subset of the input path so just return true and leave bounds unmodified
	return true
}

// scalarIsInt reports whether x has no fractional part.
func scalarIsInt(x float32) bool {
	return x == geom.FloorToScalar(x)
}

// cullLine attempts to trim the line to minimally cover the cull rect (currently only works for horizontal and vertical
// lines). Return true if processing should continue.
func cullLine(pts *[2]geom.Point, rec *stroke.Rec, ctm *geom.Matrix, cullRect *geom.Rect, intervalLength float32) bool {
	if cullRect == nil {
		return false
	}

	dx := pts[1].X - pts[0].X
	dy := pts[1].Y - pts[0].Y

	if (dx != 0 && dy != 0) || (dx == 0 && dy == 0) {
		return false
	}

	bounds := outsetForStroke(*cullRect, rec)

	// cullRect is in device space while pts are in the local coordinate system defined by the ctm. We want our answer
	// in the local coordinate system.
	inv, ok := ctm.Invert()
	if !ok {
		return false
	}
	bounds, _ = inv.MapRect(bounds)

	if dx != 0 {
		minX := pts[0].X
		maxX := pts[1].X
		if dx < 0 {
			minX, maxX = maxX, minX
		}
		if maxX <= bounds.Left || minX >= bounds.Right {
			return false
		}

		// Now we actually perform the chop, removing the excess to the left and right of the bounds (keeping our new
		// line "in phase" with the dash, hence the (mod intervalLength).
		if minX < bounds.Left {
			minX = bounds.Left - geom.ScalarMod(bounds.Left-minX, intervalLength)
		}
		if maxX > bounds.Right {
			maxX = bounds.Right + geom.ScalarMod(maxX-bounds.Right, intervalLength)
		}
		if dx < 0 {
			minX, maxX = maxX, minX
		}
		pts[0].X = minX
		pts[1].X = maxX
	} else {
		minY := pts[0].Y
		maxY := pts[1].Y
		if dy < 0 {
			minY, maxY = maxY, minY
		}
		if maxY <= bounds.Top || minY >= bounds.Bottom {
			return false
		}

		// Now we actually perform the chop, removing the excess to the top and bottom of the bounds (keeping our new
		// line "in phase" with the dash, hence the (mod intervalLength).
		if minY < bounds.Top {
			minY = bounds.Top - geom.ScalarMod(bounds.Top-minY, intervalLength)
		}
		if maxY > bounds.Bottom {
			maxY = bounds.Bottom + geom.ScalarMod(maxY-bounds.Bottom, intervalLength)
		}
		if dy < 0 {
			minY, maxY = maxY, minY
		}
		pts[0].Y = minY
		pts[1].Y = maxY
	}

	return true
}

// AsPoints implements the point-representation acceleration for a dashed line. Currently more restrictive than it needs
// to be: it requires a two-interval integer on==off pattern on a line with butt or round caps under a rect-stays-rect
// matrix.
func (d *dashEffect) AsPoints(results *stroke.PointData, src *path.Path, rec *stroke.Rec, ctm *geom.Matrix, cullRect *geom.Rect) bool {
	// width < 0 -> fill && width == 0 -> hairline so requiring width > 0 rules both out
	if rec.Width() <= 0 {
		return false
	}

	// TODO: this next test could be eased up. We could allow any number of intervals as long as all the ons match and
	// all the offs match. Additionally, they do not necessarily need to be integers. We cannot allow arbitrary
	// intervals since we want the returned points to be uniformly sized.
	if len(d.intervals) != 2 ||
		!geom.ScalarNearlyEqual(d.intervals[0], d.intervals[1]) ||
		!scalarIsInt(d.intervals[0]) ||
		!scalarIsInt(d.intervals[1]) {
		return false
	}

	pts, ok := src.IsLine()
	if !ok {
		return false
	}

	// TODO: this test could be eased up to allow circles
	if rec.Cap() != stroke.CapButt {
		return false
	}

	// TODO: this test could be eased up for circles. Rotations could be allowed.
	if !ctm.RectStaysRect() {
		return false
	}

	// See if the line can be limited to something plausible.
	if !cullLine(&pts, rec, ctm, cullRect, d.intervalLength) {
		return false
	}

	length := pts[1].DistanceTo(pts[0])

	tangent := pts[1].Sub(pts[0])
	if tangent.IsZero() {
		return false
	}

	tangent = tangent.Scaled(geom.ScalarInvert(length))

	// TODO: make this test for horizontal & vertical lines more robust
	isXAxis := true
	switch {
	case geom.ScalarNearlyEqual(1, tangent.X) || geom.ScalarNearlyEqual(-1, tangent.X):
		results.Size = geom.Point{X: 0.5 * d.intervals[0], Y: 0.5 * rec.Width()}
	case geom.ScalarNearlyEqual(1, tangent.Y) || geom.ScalarNearlyEqual(-1, tangent.Y):
		results.Size = geom.Point{X: 0.5 * rec.Width(), Y: 0.5 * d.intervals[0]}
		isXAxis = false
	case rec.Cap() != stroke.CapRound:
		// Angled lines don't have axis-aligned boxes.
		return false
	}

	results.Flags = 0
	clampedInitialDashLength := min(length, d.initialDashLength)

	if rec.Cap() == stroke.CapRound {
		results.Flags |= stroke.CirclesPointFlag
	}

	numPoints := 0
	len2 := length
	if clampedInitialDashLength > 0 || d.initialDashIndex == 0 {
		if d.initialDashIndex == 0 {
			if clampedInitialDashLength > 0 {
				if clampedInitialDashLength >= d.intervals[0] {
					numPoints++ // partial first dash
				}
				len2 -= clampedInitialDashLength
			}
			len2 -= d.intervals[1] // also skip first space
			if len2 < 0 {
				len2 = 0
			}
		} else {
			len2 -= clampedInitialDashLength // skip initial partial empty
		}
	}
	// Too many midpoints can cause numPoints to overflow or otherwise cause the points allocation below to OOM. Cap it
	// to a sane value.
	numIntervals := len2 / d.intervalLength
	if !scalarIsFinite(numIntervals) || numIntervals > maxDashCount {
		return false
	}
	numMidPoints := int(geom.FloorToScalar(numIntervals))
	numPoints += numMidPoints
	len2 -= float32(numMidPoints) * d.intervalLength
	partialLast := false
	if len2 > 0 {
		if len2 < d.intervals[0] {
			partialLast = true
		} else {
			numMidPoints++
			numPoints++
		}
	}

	results.Points = make([]geom.Point, numPoints)

	distance := float32(0)
	curPt := 0

	if clampedInitialDashLength > 0 || d.initialDashIndex == 0 {
		if d.initialDashIndex == 0 {
			if clampedInitialDashLength > 0 {
				// partial first block
				x := pts[0].X + tangent.X*(0.5*clampedInitialDashLength)
				y := pts[0].Y + tangent.Y*(0.5*clampedInitialDashLength)
				var halfWidth, halfHeight float32
				if isXAxis {
					halfWidth = 0.5 * clampedInitialDashLength
					halfHeight = 0.5 * rec.Width()
				} else {
					halfWidth = 0.5 * rec.Width()
					halfHeight = 0.5 * clampedInitialDashLength
				}
				if clampedInitialDashLength < d.intervals[0] {
					// This one will not be like the others
					results.First.Reset()
					results.First.AddRect(geom.RectLTRB(x-halfWidth, y-halfHeight, x+halfWidth, y+halfHeight),
						geom.DirectionCW)
				} else {
					results.Points[curPt] = geom.Point{X: x, Y: y}
					curPt++
				}

				distance += clampedInitialDashLength
			}

			distance += d.intervals[1] // skip over the next blank block too
		} else {
			distance += clampedInitialDashLength
		}
	}

	if numMidPoints != 0 {
		distance += 0.5 * d.intervals[0]

		for i := 0; i < numMidPoints; i++ {
			results.Points[curPt] = geom.Point{
				X: pts[0].X + tangent.X*distance,
				Y: pts[0].Y + tangent.Y*distance,
			}
			curPt++
			distance += d.intervalLength
		}

		distance -= 0.5 * d.intervals[0]
	}

	if partialLast {
		// partial final block
		temp := length - distance
		x := pts[0].X + tangent.X*(distance+0.5*temp)
		y := pts[0].Y + tangent.Y*(distance+0.5*temp)
		var halfWidth, halfHeight float32
		if isXAxis {
			halfWidth = 0.5 * temp
			halfHeight = 0.5 * rec.Width()
		} else {
			halfWidth = 0.5 * rec.Width()
			halfHeight = 0.5 * temp
		}
		results.Last.Reset()
		results.Last.AddRect(geom.RectLTRB(x-halfWidth, y-halfHeight, x+halfWidth, y+halfHeight),
			geom.DirectionCW)
	}

	return true
}
