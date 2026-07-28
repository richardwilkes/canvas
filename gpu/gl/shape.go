// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Shape is the data-only union of the specialized geometries the GPU backend handles (rect, round rect, line, point,
// path) with lossless simplification — plus round-rect helpers the clip stack consumes (corner containment / contains /
// inner bounds / conservative intersect / is-simple-circular), specialized to geom.RRect's uniform-radii storage (the
// only form publicly reachable). Trims: an arc shape type is dropped (no public entry point creates arc shapes; the
// drawArc lanes go through ovals/paths instead); path→rrect detection is absent from the path package, so round-rect
// clip paths stay Type Path and take the mask lanes instead of the analytic rrect FP (visually equivalent); rect
// detection from paths reports no start index, so simplified rects carry the default winding params (winding params
// only matter for the styling lanes, and the clip stack always simplifies with ignore-winding, which resets them
// anyway).

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// ShapeType identifies which geometry a Shape currently holds (an arc type is trimmed; see the file comment).
type ShapeType uint8

// ShapeType values.
const (
	ShapeTypeEmpty ShapeType = iota
	ShapeTypePoint
	ShapeTypeRect
	ShapeTypeRRect
	ShapeTypePath
	ShapeTypeLine
)

// Shape simplification flags.
const (
	// ShapeSimplifyFill: the original shape would have been implicitly filled when drawn or clipped, so simpler closed
	// shape types can be considered and 0-area shapes (points and lines) can turn into empty.
	ShapeSimplifyFill uint32 = 0b001
	// ShapeSimplifyIgnoreWinding: simplifications that would impact a directional stroke or path effect can still be
	// taken.
	ShapeSimplifyIgnoreWinding uint32 = 0b010
	// ShapeSimplifyMakeCanonical: the geometry is updated to have sorted coordinates (rects, lines).
	ShapeSimplifyMakeCanonical uint32 = 0b100

	ShapeSimplifyAll uint32 = 0b111
)

// Shape defaults.
const (
	shapeDefaultDir   = geom.DirectionCW
	shapeDefaultStart = uint8(0)
)

// shapeDefaultFillType is the fill rule used for non-path shapes' equivalent path form.
const shapeDefaultFillType = path.FillEvenOdd

// Shape is a data-only union of the specialized geometries the GPU backend draws. The zero value is the empty shape.
type Shape struct {
	path     *path.Path
	point    geom.Point
	rect     geom.Rect
	rrect    geom.RRect
	line     [2]geom.Point
	typ      ShapeType
	start    uint8 // winding start index; restricted to rrects and simpler, so < 8
	cw       bool
	inverted bool
}

// MakeShapeRect builds a rect shape.
func MakeShapeRect(rect geom.Rect) Shape {
	var s Shape
	s.SetRect(rect)
	return s
}

// MakeShapeRRect builds a round-rect shape.
func MakeShapeRRect(rrect geom.RRect) Shape {
	var s Shape
	s.SetRRect(rrect)
	return s
}

// MakeShapePath builds a path shape. The shape clones the path so later mutations of the argument cannot alter it.
func MakeShapePath(p *path.Path) Shape {
	var s Shape
	s.SetPath(p)
	return s
}

// MakeShapePoint builds a point shape.
func MakeShapePoint(pt geom.Point) Shape {
	var s Shape
	s.SetPoint(pt)
	return s
}

// MakeShapeLine builds a line-segment shape.
func MakeShapeLine(p1, p2 geom.Point) Shape {
	var s Shape
	s.SetLine(p1, p2)
	return s
}

// IsEmpty reports whether the shape is the empty shape.
func (s *Shape) IsEmpty() bool { return s.typ == ShapeTypeEmpty }

// IsPoint reports whether the shape is a point.
func (s *Shape) IsPoint() bool { return s.typ == ShapeTypePoint }

// IsRect reports whether the shape is a rect.
func (s *Shape) IsRect() bool { return s.typ == ShapeTypeRect }

// IsRRect reports whether the shape is a round-rect.
func (s *Shape) IsRRect() bool { return s.typ == ShapeTypeRRect }

// IsPath reports whether the shape is a path.
func (s *Shape) IsPath() bool { return s.typ == ShapeTypePath }

// IsLine reports whether the shape is a line segment.
func (s *Shape) IsLine() bool { return s.typ == ShapeTypeLine }

// Type returns the shape's type. The type queries and this method reflect the shape type provided when assigned;
// simplification may change the type.
func (s *Shape) Type() ShapeType { return s.typ }

// Inverted reports whether the shape's fill is inverted.
func (s *Shape) Inverted() bool {
	if s.IsPath() {
		return s.path.IsInverseFillType()
	}
	return s.inverted
}

// Dir returns the shape's winding direction.
func (s *Shape) Dir() geom.PathDirection {
	if s.cw {
		return geom.DirectionCW
	}
	return geom.DirectionCCW
}

// StartIndex returns the shape's winding start index.
func (s *Shape) StartIndex() uint8 { return s.start }

// setPathWindingParams sets the shape's winding direction and start index.
func (s *Shape) setPathWindingParams(dir geom.PathDirection, start uint8) {
	s.cw = dir == geom.DirectionCW
	s.start = start
}

// SetInverted sets whether the shape's fill is inverted.
func (s *Shape) SetInverted(inverted bool) {
	if s.IsPath() {
		if inverted != s.path.IsInverseFillType() {
			s.path.ToggleInverseFillType()
		}
	} else {
		s.inverted = inverted
	}
}

// Point returns the shape's point. It panics if the shape is not a point.
func (s *Shape) Point() geom.Point {
	if !s.IsPoint() {
		panic("shape is not a point")
	}
	return s.point
}

// Rect returns the shape's rect. It panics if the shape is not a rect.
func (s *Shape) Rect() geom.Rect {
	if !s.IsRect() {
		panic("shape is not a rect")
	}
	return s.rect
}

// RRect returns the shape's round-rect. It panics if the shape is not a round-rect.
func (s *Shape) RRect() geom.RRect {
	if !s.IsRRect() {
		panic("shape is not a rrect")
	}
	return s.rrect
}

// Path returns the shape's path. It panics if the shape is not a path.
func (s *Shape) Path() *path.Path {
	if !s.IsPath() {
		panic("shape is not a path")
	}
	return s.path
}

// Line returns the shape's line segment endpoints. It panics if the shape is not a line segment.
func (s *Shape) Line() [2]geom.Point {
	if !s.IsLine() {
		panic("shape is not a line")
	}
	return s.line
}

// setType switches the shape to typ, dropping any previously held path.
func (s *Shape) setType(typ ShapeType) {
	if s.IsPath() && typ != ShapeTypePath {
		s.inverted = s.path.IsInverseFillType()
		s.path = nil
	}
	s.typ = typ
}

// resetTo resets the shape to typ with default winding params and no inversion.
func (s *Shape) resetTo(typ ShapeType) {
	s.setType(typ)
	s.setPathWindingParams(shapeDefaultDir, shapeDefaultStart)
	s.SetInverted(false)
}

// Reset clears the shape back to empty.
func (s *Shape) Reset() { s.resetTo(ShapeTypeEmpty) }

// SetPoint replaces the shape's contents with a point.
func (s *Shape) SetPoint(pt geom.Point) {
	s.resetTo(ShapeTypePoint)
	s.point = pt
}

// SetRect replaces the shape's contents with a rect.
func (s *Shape) SetRect(rect geom.Rect) {
	s.resetTo(ShapeTypeRect)
	s.rect = rect
}

// SetRRect replaces the shape's contents with a round rect.
func (s *Shape) SetRRect(rrect geom.RRect) {
	s.resetTo(ShapeTypeRRect)
	s.rrect = rrect
}

// SetLine replaces the shape's contents with a line segment.
func (s *Shape) SetLine(p1, p2 geom.Point) {
	s.resetTo(ShapeTypeLine)
	s.line = [2]geom.Point{p1, p2}
}

// SetPath replaces the shape's contents with a path. The path is cloned (see MakeShapePath); the source's generation ID
// is primed first so the clone shares the source's identity (two shapes built from the same source path key
// identically).
func (s *Shape) SetPath(p *path.Path) {
	_ = p.GenerationID()
	s.setType(ShapeTypePath)
	s.path = p.Clone()
	// Must also set these since we didn't call resetTo like the other setters.
	s.setPathWindingParams(shapeDefaultDir, shapeDefaultStart)
	s.inverted = p.IsInverseFillType()
}

// simplifyPath simplifies a path shape to a simpler type when possible, returning whether the original shape was
// closed. Path→rrect detection is trimmed (see the file comment); ovals and rects are detected.
func (s *Shape) simplifyPath(flags uint32) bool {
	if !s.IsPath() {
		panic("simplifyPath on non-path")
	}
	if s.path.IsEmpty() {
		s.setType(ShapeTypeEmpty)
		return false
	}
	if pts, ok := s.path.IsLine(); ok {
		s.simplifyLine(pts[0], pts[1], flags)
		return false
	}
	if bounds, ok := s.path.IsOval(); ok {
		// The oval detection reports no direction/start; use the defaults (winding params only matter for the styling
		// lanes).
		s.simplifyRRect(geom.MakeRRect(bounds, bounds.Width()*0.5, bounds.Height()*0.5),
			shapeDefaultDir, shapeDefaultStart, flags)
		return true
	}
	// The detection reports direction but no start index, so the "simple rect" and "any closed rect" cases collapse
	// into one lane here, with the closed/simple-fill gates preserved.
	if rect, isClosed, dir, ok := s.path.IsRectWithDirection(); ok &&
		(isClosed || flags&ShapeSimplifyFill != 0) {
		s.simplifyRect(rect, dir, shapeDefaultStart, flags)
		return true
	}
	// No further simplification for a path.
	return false
}

// simplifyRRect simplifies a round-rect shape, collapsing to a rect if it has no rounding.
func (s *Shape) simplifyRRect(rrect geom.RRect, dir geom.PathDirection, start uint8, flags uint32) {
	if rrect.IsEmpty() || rrect.Type == geom.RRectRect {
		// Change index from rrect to rect.
		start = ((start + 1) / 2) % 4
		s.simplifyRect(rrect.Rect, dir, start, flags)
	} else if !s.IsRRect() {
		s.setType(ShapeTypeRRect)
		s.rrect = rrect
		s.setPathWindingParams(dir, start)
		// A round rect is already canonical, so there's nothing more to do.
	}
}

// simplifyRect simplifies a rect shape, collapsing to a line or point if it has zero area.
func (s *Shape) simplifyRect(rect geom.Rect, dir geom.PathDirection, start uint8, flags uint32) {
	if rect.Width() == 0 || rect.Height() == 0 {
		switch {
		case flags&ShapeSimplifyFill != 0:
			// A zero area, filled shape so go straight to empty.
			s.setType(ShapeTypeEmpty)
		case (rect.Width() == 0) != (rect.Height() == 0):
			// A line; choose the first point that best matches the starting index.
			p1 := geom.Point{X: rect.Left, Y: rect.Top}
			p2 := geom.Point{X: rect.Right, Y: rect.Bottom}
			if start >= 2 && flags&ShapeSimplifyIgnoreWinding == 0 {
				p1, p2 = p2, p1
			}
			s.simplifyLine(p1, p2, flags)
		default:
			// A point (all edges are equal, so start+dir doesn't affect the choice).
			s.simplifyPoint(geom.Point{X: rect.Left, Y: rect.Top}, flags)
		}
		return
	}
	if !s.IsRect() {
		s.setType(ShapeTypeRect)
		s.rect = rect
		s.setPathWindingParams(dir, start)
	}
	if flags&ShapeSimplifyMakeCanonical != 0 {
		s.rect = s.rect.Sorted()
	}
}

// simplifyLine simplifies a line shape, collapsing to a point if degenerate.
func (s *Shape) simplifyLine(p1, p2 geom.Point, flags uint32) {
	switch {
	case flags&ShapeSimplifyFill != 0:
		s.setType(ShapeTypeEmpty)
	case p1 == p2:
		s.simplifyPoint(p1, 0)
	default:
		if !s.IsLine() {
			s.setType(ShapeTypeLine)
			s.line = [2]geom.Point{p1, p2}
		}
		if flags&ShapeSimplifyMakeCanonical != 0 {
			// Sort the end points.
			if s.line[1].Y < s.line[0].Y ||
				(s.line[1].Y == s.line[0].Y && s.line[1].X < s.line[0].X) {
				s.line[0], s.line[1] = s.line[1], s.line[0]
			}
		}
	}
}

// simplifyPoint simplifies a point shape.
func (s *Shape) simplifyPoint(pt geom.Point, flags uint32) {
	if flags&ShapeSimplifyFill != 0 {
		s.setType(ShapeTypeEmpty)
	} else if !s.IsPoint() {
		s.setType(ShapeTypePoint)
		s.point = pt
	}
}

// Simplify reduces the shape to its simplest equivalent form given flags, returning true if the shape was originally
// closed based on type (or detected type within a path), even if the final simplification results in a point, line, or
// empty.
func (s *Shape) Simplify(flags uint32) bool {
	wasClosed := false
	switch s.typ {
	case ShapeTypeEmpty:
		// Do nothing.
	case ShapeTypePoint:
		s.simplifyPoint(s.point, flags)
	case ShapeTypeLine:
		s.simplifyLine(s.line[0], s.line[1], flags)
	case ShapeTypeRect:
		s.simplifyRect(s.rect, s.Dir(), s.start, flags)
		wasClosed = true
	case ShapeTypeRRect:
		s.simplifyRRect(s.rrect, s.Dir(), s.start, flags)
		wasClosed = true
	case ShapeTypePath:
		wasClosed = s.simplifyPath(flags)
	}
	if flags&ShapeSimplifyIgnoreWinding != 0 || (s.typ != ShapeTypeRect && s.typ != ShapeTypeRRect) {
		// Reset winding parameters if we don't need them anymore.
		s.setPathWindingParams(shapeDefaultDir, shapeDefaultStart)
	}
	return wasClosed
}

// StateKey packs the shape type, winding direction, start index, and invertedness into a value suitable for use in a
// resource key. This does not include any geometry coordinates.
func (s *Shape) StateKey() uint32 {
	// Use the path's full fill type instead of just whether or not it's inverted.
	var key uint32
	if s.IsPath() {
		key = uint32(s.path.FillType())
	} else if s.inverted {
		key = 1
	}
	key |= uint32(s.typ) << 2 // fill type was 2 bits
	key |= uint32(s.start) << 5
	if s.cw { // start was 3 bits, total 8 bits so far
		key |= 1 << 8
	}
	return key
}

// ConservativeContainsRect reports whether the given bounding box is completely inside the shape, treated as a filled,
// closed shape. For paths this tests the four rect corners with the path package's winding containment when the path is
// convex (any conservative answer is legal here — false negatives only skip an optimization).
func (s *Shape) ConservativeContainsRect(rect geom.Rect) bool {
	switch s.typ {
	case ShapeTypeRect:
		return s.rect.ContainsRect(rect)
	case ShapeTypeRRect:
		return rrectContainsRect(s.rrect, rect)
	case ShapeTypePath:
		if !s.path.IsConvex() || s.path.IsInverseFillType() {
			return false
		}
		return s.path.Contains(rect.Left, rect.Top) && s.path.Contains(rect.Right, rect.Top) &&
			s.path.Contains(rect.Right, rect.Bottom) && s.path.Contains(rect.Left, rect.Bottom)
	default:
		// Empty, point, and line have zero area.
		return false
	}
}

// ConservativeContainsPoint reports whether pt is inside the shape, treated as a filled, closed shape.
func (s *Shape) ConservativeContainsPoint(pt geom.Point) bool {
	switch s.typ {
	case ShapeTypeRect:
		return s.rect.ContainsPoint(pt.X, pt.Y)
	case ShapeTypeRRect:
		return rrectContainsPoint(s.rrect, pt)
	case ShapeTypePath:
		return s.path.Contains(pt.X, pt.Y)
	default:
		return false
	}
}

// Closed reports whether the shape is closed (or would be treated as closed when filled).
func (s *Shape) Closed() bool {
	switch s.typ {
	case ShapeTypeEmpty, ShapeTypeRect, ShapeTypeRRect:
		return true
	case ShapeTypePath:
		return pathIsClosedSingleContour(s.path)
	default: // point, line
		return false
	}
}

// pathIsClosedSingleContour reports whether p has exactly one contour whose final verb is a close.
func pathIsClosedSingleContour(p *path.Path) bool {
	verbCount := p.CountVerbs()
	if verbCount == 0 {
		return false
	}
	moveCount := 0
	i := 0
	it := path.NewRawIter(p)
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			return false
		}
		switch verb {
		case path.VerbMove:
			moveCount++
			if moveCount > 1 {
				return false
			}
		case path.VerbClose:
			return i == verbCount-1
		default:
		}
		i++
	}
}

// Convex reports whether the shape is known to be convex, assuming no other styles. If simpleFill is true, contours are
// assumed implicitly closed when drawn or used.
func (s *Shape) Convex(simpleFill bool) bool {
	switch s.typ {
	case ShapeTypeEmpty, ShapeTypeRect, ShapeTypeRRect:
		return true
	case ShapeTypePath:
		// A path's convexity really means "is this path convex were it to be closed". Convex paths may only have one
		// contour, hence checking the last contour's closedness is sufficient.
		return (simpleFill || s.path.IsLastContourClosed()) && s.path.IsConvex()
	default: // point, line
		return false
	}
}

// Bounds returns the shape's bounding rect. Bounds where left == right or top == bottom can indicate a line or point
// shape; a truly empty shape returns inverted bounds.
func (s *Shape) Bounds() geom.Rect {
	switch s.typ {
	case ShapeTypeEmpty:
		return geom.RectLTRB(1, 1, -1, -1)
	case ShapeTypePoint:
		return geom.RectLTRB(s.point.X, s.point.Y, s.point.X, s.point.Y)
	case ShapeTypeRect:
		return s.rect.Sorted()
	case ShapeTypeRRect:
		return s.rrect.Rect
	case ShapeTypePath:
		return s.path.Bounds()
	default: // line
		return geom.RectLTRB(s.line[0].X, s.line[0].Y, s.line[1].X, s.line[1].Y).Sorted()
	}
}

// SegmentMask returns the segment masks describing the shape were it converted to a path.
func (s *Shape) SegmentMask() uint8 {
	switch s.typ {
	case ShapeTypeEmpty:
		return 0
	case ShapeTypeRRect:
		switch {
		case s.rrect.IsEmpty() || s.rrect.Type == geom.RRectRect:
			return path.SegmentLine
		case s.rrect.Type == geom.RRectOval:
			return path.SegmentConic
		default:
			return path.SegmentConic | path.SegmentLine
		}
	case ShapeTypePath:
		return s.path.SegmentMasks()
	default: // point, line, rect
		return path.SegmentLine
	}
}

// PeekPath returns the shape's backing path without cloning when the shape is a path, or nil otherwise. Read-only:
// path.Path deep-copies on clone, so this is the zero-cost lane for queries (verb counts and the like) that neither
// mutate nor retain the path. Callers that mutate or hold the result must use AsPath.
func (s *Shape) PeekPath() *path.Path {
	if s.typ == ShapeTypePath {
		return s.path
	}
	return nil
}

// AsPath converts the shape into a path describing the same geometry (as if simply filled). The result is always
// freshly owned by the caller (built here, or cloned from the backing path), so callers may mutate it freely; read-only
// queries should prefer PeekPath.
func (s *Shape) AsPath() *path.Path {
	out := &path.Path{}
	if !s.IsPath() {
		// When not a path, we need to set the fill type on the path to match invertedness. All the non-path geometries
		// produce equivalent shapes with either even-odd or winding, so we can use the default fill type.
		if s.inverted {
			out.SetFillType(path.FillInverseEvenOdd)
		} else {
			out.SetFillType(shapeDefaultFillType)
		}
	}
	switch s.typ {
	case ShapeTypeEmpty:
	case ShapeTypePoint:
		// A plain moveTo() or moveTo+close() does not match the expected path for a point that is being dashed (dashing
		// needs to see a zero-length segment).
		out.MoveToPt(s.point)
		out.LineToPt(s.point)
	case ShapeTypeRect:
		out.AddRectWithStart(s.rect, s.Dir(), uint(s.start))
	case ShapeTypeRRect:
		out.AddRRectWithStart(s.rrect, s.Dir(), uint(s.start))
	case ShapeTypePath:
		return s.path.Clone()
	case ShapeTypeLine:
		out.MoveToPt(s.line[0])
		out.LineToPt(s.line[1])
	}
	return out
}

//////////////////////////////////////////////////////////////////////////////
// Round-rect helpers, specialized to the uniform-radii geom.RRect.

// rrectRadii returns the corner radii for the uniform storage (every corner has the same radii; ovals report half the
// dimensions and rects zero, matching how MakeRRect classifies).
func rrectRadii(rr geom.RRect) geom.Point {
	switch rr.Type {
	case geom.RRectOval:
		return geom.Point{X: rr.Rect.Width() * 0.5, Y: rr.Rect.Height() * 0.5}
	case geom.RRectSimple:
		return geom.Point{X: rr.RadiusX, Y: rr.RadiusY}
	default:
		return geom.Point{}
	}
}

// rrectIsSimpleCircular reports whether rr is a simple round rect with equal x/y radii.
func rrectIsSimpleCircular(rr geom.RRect) bool {
	return rr.Type == geom.RRectSimple && rr.RadiusX == rr.RadiusY
}

// rrectCheckCornerContainment reports whether (x, y) is inside rr's corner curves, for uniform radii.
func rrectCheckCornerContainment(rr geom.RRect, x, y float32) bool {
	var canonicalX, canonicalY float32
	r := rrectRadii(rr)
	if rr.Type == geom.RRectOval {
		canonicalX = x - rr.Rect.CenterX()
		canonicalY = y - rr.Rect.CenterY()
	} else {
		switch {
		case x < rr.Rect.Left+r.X && y < rr.Rect.Top+r.Y:
			// UL corner
			canonicalX = x - (rr.Rect.Left + r.X)
			canonicalY = y - (rr.Rect.Top + r.Y)
		case x > rr.Rect.Right-r.X && y < rr.Rect.Top+r.Y:
			// UR corner
			canonicalX = x - (rr.Rect.Right - r.X)
			canonicalY = y - (rr.Rect.Top + r.Y)
		case x > rr.Rect.Right-r.X && y > rr.Rect.Bottom-r.Y:
			// LR corner
			canonicalX = x - (rr.Rect.Right - r.X)
			canonicalY = y - (rr.Rect.Bottom - r.Y)
		case x < rr.Rect.Left+r.X && y > rr.Rect.Bottom-r.Y:
			// LL corner
			canonicalX = x - (rr.Rect.Left + r.X)
			canonicalY = y - (rr.Rect.Bottom - r.Y)
		default:
			// Not in any of the corners.
			return true
		}
	}
	// A point is in an ellipse (in standard position) if b^2*x^2 + a^2*y^2 <= (ab)^2.
	dist := canonicalX*canonicalX*r.Y*r.Y + canonicalY*canonicalY*r.X*r.X
	return dist <= (r.X*r.Y)*(r.X*r.Y)
}

// rrectContainsPoint reports whether pt is inside rr.
func rrectContainsPoint(rr geom.RRect, pt geom.Point) bool {
	return rr.Rect.ContainsPoint(pt.X, pt.Y) && rrectCheckCornerContainment(rr, pt.X, pt.Y)
}

// rrectContainsRect reports whether rect is entirely inside rr.
func rrectContainsRect(rr geom.RRect, rect geom.Rect) bool {
	if !rr.Rect.ContainsRect(rect) {
		return false
	}
	if rr.Type == geom.RRectRect {
		return true
	}
	// At this point we know all four corners of 'rect' are inside the bounds; check that they are inside the rrect's
	// corner curves.
	return rrectCheckCornerContainment(rr, rect.Left, rect.Top) &&
		rrectCheckCornerContainment(rr, rect.Right, rect.Top) &&
		rrectCheckCornerContainment(rr, rect.Right, rect.Bottom) &&
		rrectCheckCornerContainment(rr, rect.Left, rect.Bottom)
}

// rrectInnerBounds returns the largest of the three candidate rects inscribed within rr (horizontal inset, vertical
// inset, corner-curve inscribed), for uniform radii.
func rrectInnerBounds(rr geom.RRect) geom.Rect {
	if rr.Type == geom.RRectRect || rr.IsEmpty() {
		return rr.Rect
	}
	innerBounds := rr.Rect
	r := rrectRadii(rr)
	// With uniform radii, every edge's maximum inset is just the radius on that axis.
	dw := r.X + r.X
	dh := r.Y + r.Y

	// Area removed by shifting left/right, by shifting top/bottom, and by insetting all edges to the corner ellipses'
	// maximal inscribed point (sqrt(2)/2 * (rx, ry), so shifts scale by 1 - sqrt(2)/2). The scale is slightly increased
	// so that the shifted corners land safely inside the curves; sitting exactly on them lets numerical error push a
	// corner marginally outside, which fails the containment the inner bounds assert.
	horizArea := (innerBounds.Width() - dw) * innerBounds.Height()
	vertArea := (innerBounds.Height() - dh) * innerBounds.Width()
	const scale = 1 - float32(math.Sqrt2)/2 + 1e-5
	innerArea := (innerBounds.Width() - scale*dw) * (innerBounds.Height() - scale*dh)

	switch {
	case horizArea > vertArea && horizArea > innerArea:
		innerBounds.Left += r.X
		innerBounds.Right -= r.X
	case vertArea > innerArea:
		innerBounds.Top += r.Y
		innerBounds.Bottom -= r.Y
	case innerArea > 0:
		innerBounds.Left += scale * r.X
		innerBounds.Right -= scale * r.X
		innerBounds.Top += scale * r.Y
		innerBounds.Bottom -= scale * r.Y
	default:
		return geom.Rect{}
	}
	return innerBounds
}

// rrectCorner returns the coordinate of the rect matching the corner index (UL, UR, LR, LL).
func rrectCorner(r geom.Rect, corner int) geom.Point {
	switch corner {
	case 0:
		return geom.Point{X: r.Left, Y: r.Top}
	case 1:
		return geom.Point{X: r.Right, Y: r.Top}
	case 2:
		return geom.Point{X: r.Right, Y: r.Bottom}
	default:
		return geom.Point{X: r.Left, Y: r.Bottom}
	}
}

// rrectInsideCorner reports whether shape A's extreme point is contained within shape B's extreme point, relative to
// the corner location.
func rrectInsideCorner(corner int, a, b geom.Point) bool {
	switch corner {
	case 0: // UL
		return a.X >= b.X && a.Y >= b.Y
	case 1: // UR
		return a.X <= b.X && a.Y >= b.Y
	case 2: // LR
		return a.X <= b.X && a.Y <= b.Y
	default: // LL
		return a.X >= b.X && a.Y <= b.Y
	}
}

// rrectConservativeIntersect conservatively intersects two round rects for uniform-radii inputs. A true intersection
// can require per-corner radii mixes that geom.RRect cannot store; those results report empty (not-an-rrect), which is
// a legal conservative answer for both consumers (the containment test compares against a uniform rrect, which a mixed
// result could never equal, and the combine lane simply keeps both elements).
func rrectConservativeIntersect(a, b geom.RRect) geom.RRect {
	var radii [4]geom.Point

	getIntersectionRadii := func(r geom.Rect, corner int) bool {
		test := rrectCorner(r, corner)
		aCorner := rrectCorner(a.Rect, corner)
		bCorner := rrectCorner(b.Rect, corner)
		aRadii := rrectRadii(a)
		bRadii := rrectRadii(b)
		switch {
		case test == aCorner && aCorner == bCorner:
			// The round rects share a corner anchor, so pick A or B such that its X and Y radii are both larger than
			// the other rrect's, or fail if neither has the max radii.
			switch {
			case aRadii.X >= bRadii.X && aRadii.Y >= bRadii.Y:
				radii[corner] = aRadii
				return true
			case bRadii.X >= aRadii.X && bRadii.Y >= aRadii.Y:
				radii[corner] = bRadii
				return true
			default:
				return false
			}
		case test == aCorner:
			// Test that A's ellipse is contained by B. This is a non-trivial function to evaluate, so it is restricted
			// to when the corners have the same radii; otherwise use the more conservative test that A's extreme point
			// is contained in B.
			radii[corner] = aRadii
			if aRadii == bRadii {
				return rrectInsideCorner(corner, aCorner, bCorner) // A inside B
			}
			return rrectCheckCornerContainment(b, aCorner.X, aCorner.Y)
		case test == bCorner:
			// Mirror of the above.
			radii[corner] = bRadii
			if bRadii == aRadii {
				return rrectInsideCorner(corner, bCorner, aCorner) // B inside A
			}
			return rrectCheckCornerContainment(a, bCorner.X, bCorner.Y)
		default:
			// This is a corner formed by two straight edges of A and B, so confirm that it is contained in both (if
			// not, then the intersection can't be a round rect).
			radii[corner] = geom.Point{}
			return rrectCheckCornerContainment(a, test.X, test.Y) &&
				rrectCheckCornerContainment(b, test.X, test.Y)
		}
	}

	rect := a.Rect
	if !rect.Intersect(b.Rect) {
		return geom.RRect{}
	}
	for c := 0; c < 4; c++ {
		if !getIntersectionRadii(rect, c) {
			return geom.RRect{} // Resulting intersection is not a rrect.
		}
	}
	// The uniform storage can only represent all four corners sharing one radii pair.
	if radii[0] != radii[1] || radii[0] != radii[2] || radii[0] != radii[3] {
		return geom.RRect{}
	}
	// Check for radius overlap along the four edges: if the combined radii from two adjacent corners don't fit, the
	// intersection shape is definitively not a round rect.
	if radii[0].X+radii[0].X > rect.Width() || radii[0].Y+radii[0].Y > rect.Height() {
		return geom.RRect{}
	}
	return geom.MakeRRect(rect, radii[0].X, radii[0].Y)
}

// rrectsEqual reports whether two round rects are equal over the normalized (type, rect, radii) tuple.
func rrectsEqual(a, b geom.RRect) bool {
	return a.Type == b.Type && a.Rect == b.Rect && rrectRadii(a) == rrectRadii(b)
}
