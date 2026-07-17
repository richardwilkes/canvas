// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The cap and join procs the path stroker composes.

package stroke

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// Cap identifies how a stroke's open ends are drawn. The values match sk_stroke_cap_t.
type Cap uint8

// Cap values.
const (
	CapButt Cap = iota
	CapRound
	CapSquare
)

// Join identifies how a stroke's segment joins are drawn. The values match sk_stroke_join_t.
type Join uint8

// Join values.
const (
	JoinMiter Join = iota
	JoinRound
	JoinBevel
)

// capProc appends a cap to sink joining pivot+normal (the current last point) to stop, where normal points to the
// "outside" half of the stroke. otherIsLine is true when the segment being capped is a line.
type capProc func(sink *path.Path, pivot, normal, stop geom.Point, otherIsLine bool)

// joinProc appends a join at pivot between the segment with unit normal beforeUnitNormal and the one with
// afterUnitNormal.
type joinProc func(outer, inner *path.Path, beforeUnitNormal, pivot geom.Point,
	afterUnitNormal geom.Point, radius, invMiterLimit float32, prevIsLine, currIsLine bool)

func buttCapper(sink *path.Path, _, _, stop geom.Point, _ bool) {
	sink.LineTo(stop.X, stop.Y)
}

func roundCapper(sink *path.Path, pivot, normal, stop geom.Point, _ bool) {
	parallel := normal.RotateCW()
	projectedCenter := pivot.Add(parallel)
	sink.ConicToPt(projectedCenter.Add(normal), projectedCenter, geom.ScalarRoot2Over2)
	sink.ConicToPt(projectedCenter.Sub(normal), stop, geom.ScalarRoot2Over2)
}

func squareCapper(sink *path.Path, pivot, normal, stop geom.Point, extendLastPt bool) {
	parallel := normal.RotateCW()
	if extendLastPt {
		sink.SetLastPt(pivot.X+normal.X+parallel.X, pivot.Y+normal.Y+parallel.Y)
		sink.LineTo(pivot.X-normal.X+parallel.X, pivot.Y-normal.Y+parallel.Y)
	} else {
		sink.LineTo(pivot.X+normal.X+parallel.X, pivot.Y+normal.Y+parallel.Y)
		sink.LineTo(pivot.X-normal.X+parallel.X, pivot.Y-normal.Y+parallel.Y)
		sink.LineTo(stop.X, stop.Y)
	}
}

///////////////////////////////////////////////////////////////////////////////

// isClockwise reports whether the turn from before to after is clockwise.
func isClockwise(before, after geom.Point) bool {
	return before.X*after.Y > before.Y*after.X
}

// angleType classifies the angle between two unit normals for join selection.
type angleType uint8

const (
	angleNearly180 angleType = iota
	angleSharp
	angleShallow
	angleNearlyLine
)

// dot2AngleType classifies the angle between two unit normals from their dot product.
func dot2AngleType(dot float32) angleType {
	if dot >= 0 { // shallow or line
		if geom.ScalarNearlyZero(1 - dot) {
			return angleNearlyLine
		}
		return angleShallow
	}
	// sharp or 180
	if geom.ScalarNearlyZero(1 + dot) {
		return angleNearly180
	}
	return angleSharp
}

// handleInnerJoin closes the inner edge of a join at pivot.
func handleInnerJoin(inner *path.Path, pivot, after geom.Point) {
	// In the degenerate case that the stroke radius is larger than our segments just connecting the two inner segments
	// may "show through" as a funny diagonal. To pseudo-fix this, we go through the pivot point. This adds an extra
	// point/edge, but I can't see a cheap way to know when this is not needed.
	inner.LineTo(pivot.X, pivot.Y)
	inner.LineTo(pivot.X-after.X, pivot.Y-after.Y)
}

func bluntJoiner(outer, inner *path.Path, beforeUnitNormal, pivot, afterUnitNormal geom.Point, radius, _ float32, _, _ bool) {
	after := afterUnitNormal.Scaled(radius)
	if !isClockwise(beforeUnitNormal, afterUnitNormal) {
		outer, inner = inner, outer
		after = after.Negated()
	}
	outer.LineTo(pivot.X+after.X, pivot.Y+after.Y)
	handleInnerJoin(inner, pivot, after)
}

func roundJoiner(outer, inner *path.Path, beforeUnitNormal, pivot, afterUnitNormal geom.Point, radius, _ float32, _, _ bool) {
	dotProd := beforeUnitNormal.Dot(afterUnitNormal)
	if dot2AngleType(dotProd) == angleNearlyLine {
		return
	}

	before := beforeUnitNormal
	after := afterUnitNormal
	dir := geom.RotationCW

	if !isClockwise(before, after) {
		outer, inner = inner, outer
		before = before.Negated()
		after = after.Negated()
		dir = geom.RotationCCW
	}

	var matrix geom.Matrix
	matrix.SetScale(radius, radius)
	matrix.PostTranslate(pivot.X, pivot.Y)
	var conics [geom.MaxConicsForArc]geom.Conic
	count := geom.BuildUnitArc(before, after, dir, &matrix, &conics)
	if count > 0 {
		for i := 0; i < count; i++ {
			outer.ConicToPt(conics[i].Pts[1], conics[i].Pts[2], conics[i].W)
		}
		after = after.Scaled(radius)
		handleInnerJoin(inner, pivot, after)
	}
}

// kOneOverSqrt2 is 1/sqrt(2), used to special-case right-angle miter joins.
const kOneOverSqrt2 = float32(0.707106781)

func miterJoiner(outer, inner *path.Path, beforeUnitNormal, pivot, afterUnitNormal geom.Point, radius, invMiterLimit float32, prevIsLine, currIsLine bool) {
	// negate the dot since we're using normals instead of tangents
	dotProd := beforeUnitNormal.Dot(afterUnitNormal)
	angle := dot2AngleType(dotProd)
	before := beforeUnitNormal
	after := afterUnitNormal
	var mid geom.Point

	if angle == angleNearlyLine {
		return
	}

	// doMiter selects the miter lane; otherwise this falls through to a blunt (beveled) join.
	doMiter := false
	ccw := false

	if angle != angleNearly180 {
		ccw = !isClockwise(before, after)
		if ccw {
			outer, inner = inner, outer
			before = before.Negated()
			after = after.Negated()
		}

		/* Before we enter the world of square-roots and divides, check if we're trying to join an
		   upright right angle (common case for stroking rectangles). If so, special case that (for
		   speed an accuracy). Note: we only need to check one normal if dot==0 */
		if dotProd == 0 && invMiterLimit <= kOneOverSqrt2 {
			mid = before.Add(after).Scaled(radius)
			doMiter = true
		} else {
			/*  midLength = radius / sinHalfAngle
			    if (midLength > miterLimit * radius) abort
			    if (radius / sinHalf > miterLimit * radius) abort
			    if (1 / sinHalf > miterLimit) abort
			    if (1 / miterLimit > sinHalf) abort
			    My dotProd is opposite sign, since it is built from normals and not tangents
			    hence 1 + dot instead of 1 - dot in the formula
			*/
			sinHalfAngle := geom.ScalarSqrt((1 + dotProd) / 2)
			if sinHalfAngle < invMiterLimit {
				currIsLine = false
			} else {
				// choose the most accurate way to form the initial mid-vector
				if angle == angleSharp {
					mid = geom.Point{X: after.Y - before.Y, Y: before.X - after.X}
					if ccw {
						mid = mid.Negated()
					}
				} else {
					mid = geom.Point{X: before.X + after.X, Y: before.Y + after.Y}
				}
				mid.SetLength(radius / sinHalfAngle)
				doMiter = true
			}
		}
	} else {
		currIsLine = false
		// the nearly-180 case skips the swap-on-ccw step above
	}

	if doMiter {
		if prevIsLine {
			outer.SetLastPt(pivot.X+mid.X, pivot.Y+mid.Y)
		} else {
			outer.LineTo(pivot.X+mid.X, pivot.Y+mid.Y)
		}
	}

	after = after.Scaled(radius)
	if !currIsLine {
		outer.LineTo(pivot.X+after.X, pivot.Y+after.Y)
	}
	handleInnerJoin(inner, pivot, after)
}

///////////////////////////////////////////////////////////////////////////////

// capFactory returns the cap proc for the given cap style.
func capFactory(strokeCap Cap) capProc {
	switch strokeCap {
	case CapRound:
		return roundCapper
	case CapSquare:
		return squareCapper
	default:
		return buttCapper
	}
}

// joinFactory returns the join proc for the given join style.
func joinFactory(join Join) joinProc {
	switch join {
	case JoinRound:
		return roundJoiner
	case JoinBevel:
		return bluntJoiner
	default:
		return miterJoiner
	}
}
