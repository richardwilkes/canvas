// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// w0PlaneDistance is how far from the w=0 plane the perspective clip keeps geometry.
const w0PlaneDistance = 1.0 / (1 << 14)

// Transform transforms the path's points (and conic weights, under perspective) in place. Perspective transforms may
// change verbs and increase their number (cubics are subdivided, quads become conics).
func (p *Path) Transform(matrix *geom.Matrix) {
	p.transform(matrix, p)
}

// TransformTo is like Transform, but the source is unmodified and the result is written to dst.
func (p *Path) TransformTo(matrix *geom.Matrix, dst *Path) {
	p.transform(matrix, dst)
}

// subdivideCubicTo recursively chops the cubic in half 'level' times before appending, so perspective mapping of the
// control points stays a good approximation.
func subdivideCubicTo(dst *Path, pts []geom.Point, level int) {
	level--
	if level >= 0 {
		var tmp [7]geom.Point
		geom.ChopCubicAtHalf(pts, tmp[:])
		subdivideCubicTo(dst, tmp[0:4], level)
		subdivideCubicTo(dst, tmp[3:7], level)
	} else {
		dst.CubicToPt(pts[1], pts[2], pts[3])
	}
}

func (p *Path) transform(matrix *geom.Matrix, dst *Path) {
	if matrix.IsIdentity() {
		if dst != nil && dst != p {
			dst.copyFrom(p)
		}
		return
	}

	if dst == nil {
		dst = p
	}

	if matrix.HasPerspective() {
		src := p
		if clipped, didClip := perspectiveClip(p, matrix); didClip {
			src = clipped
		}

		tmp := New()
		tmp.fillType = p.fillType

		iter := NewIter(src, false)
		for {
			verb, pts, ok := iter.Next()
			if !ok {
				break
			}
			switch verb {
			case VerbMove:
				tmp.MoveToPt(pts[0])
			case VerbLine:
				tmp.LineToPt(pts[1])
			case VerbQuad:
				// promote the quad to a conic
				tmp.ConicToPt(pts[1], pts[2], geom.TransformW(pts[:3], 1, matrix))
			case VerbConic:
				tmp.ConicToPt(pts[1], pts[2], geom.TransformW(pts[:3], iter.ConicWeight(), matrix))
			case VerbCubic:
				subdivideCubicTo(tmp, pts[:4], 2)
			case VerbClose:
				tmp.Close()
			}
		}
		// Map the accumulated points through the matrix (with perspective division) as the final step.
		matrix.MapPointsInPlace(tmp.points)
		tmp.boundsValid = false
		*dst = *tmp
		return
	}

	convexity := p.convexity

	// The non-perspective transform path: map points directly and try to carry lazy caches forward.
	srcBoundsValid := p.boundsValid
	srcFinite := p.finite
	srcBounds := p.bounds
	srcIsA := p.isa
	srcIsADir := p.isaDir
	srcIsAStart := p.isaStart

	if dst != p {
		dst.verbs = append(dst.verbs[:0], p.verbs...)
		dst.conicWeights = append(dst.conicWeights[:0], p.conicWeights...)
		if cap(dst.points) < len(p.points) {
			dst.points = make([]geom.Point, len(p.points))
		} else {
			dst.points = dst.points[:len(p.points)]
		}
	}
	matrix.MapPoints(dst.points, p.points)

	canXformBounds := srcBoundsValid && matrix.RectStaysRect() && len(p.points) > 1

	// Here we optimize the bounds computation, by noting if the bounds are already known, and if so, we just transform
	// those as well and mark them as "known", rather than force the transformed path to have to recompute them.
	//
	// Special gotchas if the path is effectively empty (<= 1 point) or if it is non-finite. In those cases bounds need
	// to stay empty, regardless of the matrix.
	if canXformBounds {
		if srcFinite {
			mapped, _ := matrix.MapRect(srcBounds)
			dst.setBounds(mapped)
			if !dst.finite {
				dst.bounds = geom.Rect{}
			}
		} else {
			dst.boundsValid = true
			dst.finite = false
			dst.bounds = geom.Rect{}
		}
	} else {
		dst.boundsValid = false
	}

	dst.segmentMask = p.segmentMask

	// It's an oval/rrect only if rect stays rect.
	if srcIsA != isAGeneral && matrix.RectStaysRect() {
		dst.isa = srcIsA
		dst.isaDir, dst.isaStart = transformDirAndStart(matrix, srcIsA == isARRect, srcIsADir, srcIsAStart)
	} else {
		dst.isa = isAGeneral
	}

	if p != dst {
		dst.lastMoveIdx = p.lastMoveIdx
		dst.fillType = p.fillType
		dst.isVolatile = p.isVolatile
	}
	// The transformed geometry gets a fresh identity: a new generation ID, assigned lazily.
	dst.genID = 0

	// Due to finite/fragile float numerics, we can't assume that a convex path remains convex after a transformation,
	// so mark it as unknown here. However, some transformations are thought to be safe: axis-aligned values under
	// scale/translate.
	if convexity.IsConvex() {
		if !matrix.IsScaleTranslate() || !isAxisAligned(dst.points) {
			// Not safe to still assume we're convex...
			convexity = ConvexityUnknown
		} else {
			det2x2 := matrix.Get(geom.MScaleX)*matrix.Get(geom.MScaleY) -
				matrix.Get(geom.MSkewX)*matrix.Get(geom.MSkewY)
			switch {
			case det2x2 < 0:
				convexity = oppositeConvexDirection(convexity)
			case det2x2 > 0:
				// we keep our direction
			default: // det2x2 == 0
				convexity = ConvexityConvexDegenerate
			}
		}
	}
	dst.convexity = convexity
}

// isAxisAligned is a conservative (quick) test to see if all segments are axis-aligned, looking only at the raw points.
func isAxisAligned(pts []geom.Point) bool {
	for i := 1; i < len(pts); i++ {
		if pts[i-1].X != pts[i].X && pts[i-1].Y != pts[i].Y {
			return false
		}
	}
	return true
}

// transformDirAndStart computes how an oval/rrect's direction and start index change under a rect-stays-rect matrix.
func transformDirAndStart(matrix *geom.Matrix, isRRect bool, dir geom.PathDirection, start uint8) (newDir geom.PathDirection, newStart uint8) {
	inStart := uint(start)
	isCCW := dir == geom.DirectionCCW

	rm := uint(0)
	if isRRect {
		// Degenerate rrect indices to oval indices and remember the remainder. Ovals have one index per side whereas
		// rrects have two.
		rm = inStart & 1
		inStart /= 2
	}
	// antiDiag: is the antidiagonal non-zero (otherwise the diagonal is zero). topNeg: is the non-zero value in the top
	// row (either scaleX or skewX) negative. sameSign: are the two non-zero diagonal or antidiagonal values the same
	// sign.
	var antiDiag, topNeg, sameSign uint
	if matrix.Get(geom.MScaleX) != 0 {
		antiDiag = 0b00
		if matrix.Get(geom.MScaleX) > 0 {
			topNeg = 0b00
			if matrix.Get(geom.MScaleY) > 0 {
				sameSign = 0b01
			}
		} else {
			topNeg = 0b10
			if matrix.Get(geom.MScaleY) <= 0 {
				sameSign = 0b01
			}
		}
	} else {
		antiDiag = 0b01
		if matrix.Get(geom.MSkewX) > 0 {
			topNeg = 0b00
			if matrix.Get(geom.MSkewY) > 0 {
				sameSign = 0b01
			}
		} else {
			topNeg = 0b10
			if matrix.Get(geom.MSkewY) <= 0 {
				sameSign = 0b01
			}
		}
	}
	var outStart uint
	if sameSign != antiDiag {
		// This is a rotation (and maybe scale). The direction is unchanged.
		outStart = (inStart + 4 - (topNeg | antiDiag)) % 4
		if isRRect {
			outStart = 2*outStart + rm
		}
	} else {
		// This is a mirror (and maybe scale). The direction is reversed.
		isCCW = !isCCW
		outStart = (6 + (topNeg | antiDiag) - inStart) % 4
		if isRRect {
			if rm != 0 {
				outStart = 2 * outStart
			} else {
				outStart = 2*outStart + 1
			}
		}
	}

	outDir := geom.DirectionCW
	if isCCW {
		outDir = geom.DirectionCCW
	}
	return outDir, uint8(outStart)
}

//////////////////////////////////////////////////////////////////////////////
// Perspective clip

// halfPlane represents the line a*x + b*y + c = 0, used to clip a path against the w=0 plane under perspective.
type halfPlane struct {
	a, b, c float32
}

func (h *halfPlane) eval(x, y float32) float32 {
	return h.a*x + h.b*y + h.c
}

func (h *halfPlane) normalize() bool {
	a := float64(h.a)
	b := float64(h.b)
	c := float64(h.c)
	dmag := math.Sqrt(a*a + b*b)
	// length of initial plane normal is zero
	if dmag == 0 {
		h.a = 0
		h.b = 0
		h.c = 1
		return true
	}
	dscale := 1.0 / dmag
	a *= dscale
	b *= dscale
	c *= dscale
	// check if we're not finite, or normal is zero-length
	if math.IsInf(a, 0) || math.IsNaN(a) || math.IsInf(b, 0) || math.IsNaN(b) ||
		math.IsInf(c, 0) || math.IsNaN(c) || (a == 0 && b == 0) {
		h.a = 0
		h.b = 0
		h.c = 1
		return false
	}
	h.a = float32(a)
	h.b = float32(b)
	h.c = float32(c)
	return true
}

type halfPlaneResult uint8

const (
	halfPlaneAllNegative halfPlaneResult = iota
	halfPlaneAllPositive
	halfPlaneMixed
)

func (h *halfPlane) test(bounds geom.Rect) halfPlaneResult {
	// check whether the diagonal aligned with the normal crosses the plane
	var diagMin, diagMax geom.Point
	if h.a >= 0 {
		diagMin.X = bounds.Left
		diagMax.X = bounds.Right
	} else {
		diagMin.X = bounds.Right
		diagMax.X = bounds.Left
	}
	if h.b >= 0 {
		diagMin.Y = bounds.Top
		diagMax.Y = bounds.Bottom
	} else {
		diagMin.Y = bounds.Bottom
		diagMax.Y = bounds.Top
	}
	test := h.eval(diagMin.X, diagMin.Y)
	sign := test * h.eval(diagMax.X, diagMax.Y)
	if sign > 0 {
		// the path is either all on one side of the half-plane or the other
		if test < 0 {
			return halfPlaneAllNegative
		}
		return halfPlaneAllPositive
	}
	return halfPlaneMixed
}

// MapRect maps src through matrix, correctly for all matrix types. For non-perspective matrices it is
// geom.Matrix.MapRect; for perspective matrices it routes the rect through a path transform (build a rect path,
// transform it, take its bounds), which clips against the w = 1/16384 half-plane — behavior geom.Matrix.MapRect cannot
// reproduce on its own, since the clip machinery lives above the geom layer. The boolean reports whether the result is
// exactly the image of src.
func MapRect(matrix *geom.Matrix, src geom.Rect) (geom.Rect, bool) {
	if !matrix.HasPerspective() {
		return matrix.MapRect(src)
	}
	rectPath := New()
	rectPath.AddRect(src, geom.DirectionCW)
	rectPath.Transform(matrix)
	return rectPath.Bounds(), false
}

// clipAgainstPlane rotates the path so the (pre-normalized) half-plane becomes the horizontal line y = 0, clips against
// the upper half plane with the edge clipper, and rotates the result back. Returns ok == false if the rotation could
// not be inverted or a non-finite intermediate appeared, in which case the caller treats the path as clipped out.
func clipAgainstPlane(p *Path, plane *halfPlane) (*Path, bool) {
	var mx geom.Matrix
	p0 := geom.Point{X: -plane.a * plane.c, Y: -plane.b * plane.c}
	mx.SetAll(plane.b, plane.a, p0.X,
		-plane.a, plane.b, p0.Y,
		0, 0, 1)
	inv, ok := mx.Invert()
	if !ok {
		return nil, false
	}

	rotated := New()
	p.TransformTo(&inv, rotated)
	if !rotated.IsFinite() {
		return nil, false
	}

	big := geom.ScalarMax
	clip := geom.Rect{Left: -big, Top: 0, Right: big, Bottom: big}

	result := New()
	prev := geom.Point{}

	ClipPath(rotated, clip, false, func(clipper *geom.EdgeClipper, newCtr bool) {
		addLineTo := false
		var pts [4]geom.Point
		for {
			verb, more := clipper.Next(pts[:])
			if !more {
				break
			}
			if newCtr {
				result.MoveToPt(pts[0])
				prev = pts[0]
				newCtr = false
			}

			if addLineTo || pts[0] != prev {
				result.LineToPt(pts[0])
			}

			switch verb {
			case geom.ClipVerbLine:
				result.LineToPt(pts[1])
				prev = pts[1]
			case geom.ClipVerbQuad:
				result.QuadToPt(pts[1], pts[2])
				prev = pts[2]
			case geom.ClipVerbCubic:
				result.CubicToPt(pts[1], pts[2], pts[3])
				prev = pts[3]
			}
			addLineTo = true
		}
	})

	result.SetFillType(p.FillType())
	result.Transform(&mx)
	if !result.IsFinite() {
		return nil, false
	}
	return result, true
}

// perspectiveClip clips the path against the w = 1/16384 plane if needed (to not blow up under a perspective matrix),
// returning the clipped path and true. Paths entirely on the visible side return (nil, false): no clipping needed. The
// result may be empty (fully clipped out, or the clip failed).
func perspectiveClip(p *Path, matrix *geom.Matrix) (*Path, bool) {
	if !matrix.HasPerspective() {
		return nil, false
	}

	plane := halfPlane{
		a: matrix.Get(geom.MPersp0),
		b: matrix.Get(geom.MPersp1),
		c: matrix.Get(geom.MPersp2) - w0PlaneDistance,
	}
	if plane.normalize() {
		switch plane.test(p.Bounds()) {
		case halfPlaneAllPositive:
			return nil, false
		case halfPlaneMixed:
			if clipped, ok := clipAgainstPlane(p, &plane); ok {
				return clipped, true
			}
			return New(), true // clipped out (or failed)
		default:
		}
	}
	// clipped out (or failed)
	return New(), true
}
