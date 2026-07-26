// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package path provides the Path type: verb/point/conic-weight storage, the full builder op set (including the three
// arcTo forms), shape adders, bounds/tight-bounds/contains queries, and transform. Structure uses plain Go slices (no
// copy-on-write sharing), with well-defined semantics for verb streams, point arrays, conic weights, lazy caches, and
// degenerate-input handling.
package path

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
)

// FillType selects the rule for deciding which regions of a path are "inside" when filling.
type FillType uint8

const (
	// FillWinding computes "inside" by a non-zero sum of signed edge crossings.
	FillWinding FillType = iota
	// FillEvenOdd computes "inside" by an odd number of edge crossings.
	FillEvenOdd
	// FillInverseWinding is FillWinding, but draws outside of the path.
	FillInverseWinding
	// FillInverseEvenOdd is FillEvenOdd, but draws outside of the path.
	FillInverseEvenOdd
)

// IsInverse reports whether ft is one of the inverse fill types.
func (ft FillType) IsInverse() bool {
	return ft&2 != 0
}

// IsEvenOdd reports whether ft uses the even-odd rule (as opposed to winding).
func (ft FillType) IsEvenOdd() bool {
	return ft&1 != 0
}

// Verb identifies one segment-construction operation in a path's verb stream.
type Verb uint8

const (
	// VerbMove starts a new contour at one point.
	VerbMove Verb = iota
	// VerbLine adds a line with one point.
	VerbLine
	// VerbQuad adds a quadratic bezier with two points.
	VerbQuad
	// VerbConic adds a conic (rational quad) with two points and a weight.
	VerbConic
	// VerbCubic adds a cubic bezier with three points.
	VerbCubic
	// VerbClose closes the current contour with no points.
	VerbClose
)

// pointsForVerb returns the number of points consumed from the point array by each verb.
func (v Verb) pointsForVerb() int {
	switch v {
	case VerbMove, VerbLine:
		return 1
	case VerbQuad, VerbConic:
		return 2
	case VerbCubic:
		return 3
	default:
		return 0
	}
}

// AddPathMode selects how AddPath joins the source path's contours onto the destination.
type AddPathMode uint8

const (
	// AddPathAppend appends the source path's contours as new contours.
	AddPathAppend AddPathMode = iota
	// AddPathExtend extends the last contour of the destination with the first contour of the source, connecting them
	// with a line.
	AddPathExtend
)

// ArcSize selects which of the two possible elliptical arcs an SVG-style arcTo should take.
type ArcSize uint8

const (
	// ArcSizeSmall selects the smaller of the arc pair.
	ArcSizeSmall ArcSize = iota
	// ArcSizeLarge selects the larger of the arc pair.
	ArcSizeLarge
)

// Segment mask bits, identifying which verb types are present in a path.
const (
	SegmentLine  = 1 << 0
	SegmentQuad  = 1 << 1
	SegmentConic = 1 << 2
	SegmentCubic = 1 << 3
)

// Convexity classifies a path's shape as convex (with a winding direction) or concave. The unknown state is the zero
// value so that the zero Path is usable without initialization.
type Convexity uint8

const (
	// ConvexityUnknown means the convexity has not been computed (or the path was edited since).
	ConvexityUnknown Convexity = iota
	// ConvexityConvexCW is a convex path wound clockwise.
	ConvexityConvexCW
	// ConvexityConvexCCW is a convex path wound counter-clockwise.
	ConvexityConvexCCW
	// ConvexityConvexDegenerate is convex but with no determinable direction.
	ConvexityConvexDegenerate
	// ConvexityConcave is a non-convex path.
	ConvexityConcave
)

// IsConvex reports whether the convexity value is one of the convex states.
func (c Convexity) IsConvex() bool {
	return c == ConvexityConvexCW || c == ConvexityConvexCCW || c == ConvexityConvexDegenerate
}

// directionToConvexity converts a winding direction into the corresponding convex Convexity value.
func directionToConvexity(dir geom.PathDirection) Convexity {
	if dir == geom.DirectionCW {
		return ConvexityConvexCW
	}
	return ConvexityConvexCCW
}

// oppositeConvexDirection returns the convex Convexity value with the opposite winding direction, leaving non-convex
// values unchanged.
func oppositeConvexDirection(c Convexity) Convexity {
	switch c {
	case ConvexityConvexCW:
		return ConvexityConvexCCW
	case ConvexityConvexCCW:
		return ConvexityConvexCW
	default:
		return c
	}
}

// isAType records which simple shape (if any) the path is known to be.
type isAType uint8

const (
	isAGeneral isAType = iota
	isAOval
	isARRect
)

// Path is a sequence of verbs with associated points and conic weights, plus a fill type and lazily-computed bounds and
// convexity caches. The zero value is an empty path with a winding fill.
//
// A Path is not safe for concurrent use unless it is immutable and every one of its lazy caches has been primed, since
// each is written by the first reader that needs it: the bounds and finiteness cache (call Bounds or IsFinite), the
// convexity cache (call GetConvexity or IsConvex) and the generation ID (call GenerationID). Priming only some of them
// leaves the rest to race.
type Path struct {
	points       []geom.Point
	verbs        []Verb
	conicWeights []float32

	bounds geom.Rect

	// lastMoveIdx tracks the index of the current contour's moveTo point, biased by +1 so the zero value means -1
	// (which signals "inject a moveTo before the next segment"). Negative (real) values are the bitwise complement of
	// the contour's move index, set by Close.
	lastMoveIdx int

	// genID is the lazily-assigned generation ID: 0 = not yet assigned (or dirtied by an edit), 1 = the shared
	// empty-path ID, values >= 2 come from a package-wide counter. Fill-type changes do not affect the generation ID.
	genID uint32

	fillType    FillType
	convexity   Convexity
	segmentMask uint8
	boundsValid bool
	finite      bool
	isVolatile  bool
	isa         isAType
	isaDir      geom.PathDirection
	isaStart    uint8
}

// nextPathGenID hands out generation IDs; 0 and 1 (kEmptyGenID) are reserved.
var nextPathGenID atomic.Uint32

func init() { nextPathGenID.Store(1) }

// GenerationID returns a value that changes whenever the path's verb or point data mutates (fill-type changes
// excluded). Empty paths share the ID 1. The ID is assigned lazily; like the other lazy caches (Bounds), the first call
// counts as a mutation for the concurrency rules, so prime it before sharing a path between goroutines.
func (p *Path) GenerationID() uint32 {
	if len(p.verbs) == 0 {
		return 1
	}
	if p.genID == 0 {
		p.genID = nextPathGenID.Add(1)
		for p.genID < 2 { // skip the reserved values on wraparound
			p.genID = nextPathGenID.Add(1)
		}
	}
	return p.genID
}

// SetVolatile marks the path as transient so downstream caches (the GPU mask/geometry caches) do not key resources off
// of it.
func (p *Path) SetVolatile(isVolatile bool) *Path {
	p.isVolatile = isVolatile
	return p
}

// IsVolatile reports whether the path was marked transient via SetVolatile.
func (p *Path) IsVolatile() bool { return p.isVolatile }

// New returns an empty path with a winding fill.
func New() *Path {
	return &Path{}
}

// lastMoveToIndex returns the index of the current contour's moveTo point, or -1 if a moveTo needs to be injected
// before the next segment.
func (p *Path) lastMoveToIndex() int {
	return p.lastMoveIdx - 1
}

// setLastMoveToIndex sets the index of the current contour's moveTo point (see lastMoveToIndex).
func (p *Path) setLastMoveToIndex(v int) {
	p.lastMoveIdx = v + 1
}

// Clone returns a deep copy of the path (there is no copy-on-write sharing, so this copies the storage).
func (p *Path) Clone() *Path {
	c := *p
	c.points = append([]geom.Point(nil), p.points...)
	c.verbs = append([]Verb(nil), p.verbs...)
	c.conicWeights = append([]float32(nil), p.conicWeights...)
	return &c
}

// copyFrom replaces p's contents with a deep copy of src.
func (p *Path) copyFrom(src *Path) {
	if p == src {
		return
	}
	p.points = append(p.points[:0], src.points...)
	p.verbs = append(p.verbs[:0], src.verbs...)
	p.conicWeights = append(p.conicWeights[:0], src.conicWeights...)
	p.bounds = src.bounds
	p.lastMoveIdx = src.lastMoveIdx
	p.fillType = src.fillType
	p.convexity = src.convexity
	p.segmentMask = src.segmentMask
	p.boundsValid = src.boundsValid
	p.finite = src.finite
	p.isVolatile = src.isVolatile
	p.genID = src.genID // same geometry, same identity
	p.isa = src.isa
	p.isaDir = src.isaDir
	p.isaStart = src.isaStart
}

// resetFields restores the path's scalar fields (fill type, convexity, moveTo index) to their initial values, without
// touching the verb/point/weight storage.
func (p *Path) resetFields() {
	p.setLastMoveToIndex(-1)
	p.fillType = FillWinding
	p.convexity = ConvexityUnknown
}

// resetStorage clears the verb/point/weight storage, optionally retaining the underlying arrays' capacity for reuse.
func (p *Path) resetStorage(keepCapacity bool) {
	if keepCapacity {
		p.points = p.points[:0]
		p.verbs = p.verbs[:0]
		p.conicWeights = p.conicWeights[:0]
	} else {
		p.points = nil
		p.verbs = nil
		p.conicWeights = nil
	}
	p.boundsValid = false
	p.segmentMask = 0
	p.isa = isAGeneral
	p.genID = 0
}

// Reset removes all verbs, points and weights, sets the fill type to winding, and releases internal storage.
func (p *Path) Reset() *Path {
	p.resetStorage(false)
	p.resetFields()
	return p
}

// Rewind is like Reset, but retains internal storage for reuse.
func (p *Path) Rewind() *Path {
	p.resetStorage(true)
	p.resetFields()
	return p
}

// FillType returns the path's fill rule.
func (p *Path) FillType() FillType {
	return p.fillType
}

// SetFillType sets the path's fill rule.
func (p *Path) SetFillType(ft FillType) {
	p.fillType = ft
}

// IsInverseFillType reports whether the path's fill type is one of the inverse variants.
func (p *Path) IsInverseFillType() bool {
	return p.fillType.IsInverse()
}

// ToggleInverseFillType flips the path's fill type between its normal and inverse variant.
func (p *Path) ToggleInverseFillType() {
	p.fillType ^= 2
}

// IsEmpty reports whether the path has no verbs.
func (p *Path) IsEmpty() bool {
	return len(p.verbs) == 0
}

// IsLastContourClosed reports whether the most recently added contour ends with a close verb.
func (p *Path) IsLastContourClosed() bool {
	if len(p.verbs) == 0 {
		return false
	}
	return p.verbs[len(p.verbs)-1] == VerbClose
}

// CountPoints returns the total number of points in the path.
func (p *Path) CountPoints() int {
	return len(p.points)
}

// Point returns the point at index, or (0, 0) when out of range.
func (p *Path) Point(index int) geom.Point {
	if index >= 0 && index < len(p.points) {
		return p.points[index]
	}
	return geom.Point{}
}

// Points copies min(len(dst), CountPoints()) points into dst and returns the total number of points in the path.
func (p *Path) Points(dst []geom.Point) int {
	copy(dst, p.points)
	return len(p.points)
}

// CountVerbs returns the total number of verbs in the path.
func (p *Path) CountVerbs() int {
	return len(p.verbs)
}

// SegmentMasks returns the union of Segment* bits for the verb types present.
func (p *Path) SegmentMasks() uint8 {
	return p.segmentMask
}

// hasOnlyMoveTos reports whether the path contains no segment verbs (only moveTos, if anything).
func (p *Path) hasOnlyMoveTos() bool {
	return p.segmentMask == 0
}

// LastPt returns the last point and true, or the zero point and false when the path has no points.
func (p *Path) LastPt() (geom.Point, bool) {
	if n := len(p.points); n > 0 {
		return p.points[n-1], true
	}
	return geom.Point{}, false
}

// SetLastPt replaces the path's last point, or starts a moveTo at (x, y) if the path has no points.
func (p *Path) SetLastPt(x, y float32) {
	if n := len(p.points); n == 0 {
		p.MoveTo(x, y)
	} else {
		p.points[n-1] = geom.Pt(x, y)
		// Moving a point changes the geometry, so the bounds, the simple-shape identity and everything dirtyAfterEdit
		// covers (convexity, generation ID) must all be invalidated.
		p.boundsValid = false
		p.isa = isAGeneral
		p.dirtyAfterEdit()
	}
}

// IsLine reports whether the path is exactly one moveTo followed by one lineTo, in which case the two points are
// returned.
func (p *Path) IsLine() (line [2]geom.Point, ok bool) {
	if len(p.verbs) == 2 && p.verbs[0] == VerbMove && p.verbs[1] == VerbLine {
		return [2]geom.Point{p.points[0], p.points[1]}, true
	}
	return line, false
}

// IsOval reports whether the path was built by a single AddOval (or equivalent), returning the oval's bounds.
func (p *Path) IsOval() (bounds geom.Rect, ok bool) {
	if p.isa == isAOval {
		return p.Bounds(), true
	}
	return bounds, false
}

// Bounds returns the (lazily computed and cached) bounds of all points, including points belonging to trailing moveTo
// verbs. Non-finite points make the bounds empty.
func (p *Path) Bounds() geom.Rect {
	if !p.boundsValid {
		p.finite = p.bounds.SetBounds(p.points)
		p.boundsValid = true
	}
	return p.bounds
}

// IsFinite reports whether all of the path's points are finite.
func (p *Path) IsFinite() bool {
	p.Bounds()
	return p.finite
}

// setBounds installs a precomputed bounds value directly (used by transform's bounds fast path).
func (p *Path) setBounds(r geom.Rect) {
	p.bounds = r
	p.boundsValid = true
	p.finite = r.IsFinite()
}

//////////////////////////////////////////////////////////////////////////////
// Construction

// dirtyAfterEdit invalidates the convexity cache and clears the cached generation ID (which GenerationID lazily
// reassigns on its next call) after a mutation.
func (p *Path) dirtyAfterEdit() *Path {
	p.convexity = ConvexityUnknown
	p.genID = 0
	return p
}

// growForVerb appends the verb (and, for a conic, its weight), updates the segment mask, and dirties the bounds and
// shape caches. The caller appends the points.
func (p *Path) growForVerb(verb Verb, weight float32) {
	switch verb {
	case VerbLine:
		p.segmentMask |= SegmentLine
	case VerbQuad:
		p.segmentMask |= SegmentQuad
	case VerbConic:
		p.segmentMask |= SegmentConic
	case VerbCubic:
		p.segmentMask |= SegmentCubic
	default:
	}
	p.boundsValid = false
	p.isa = isAGeneral
	p.genID = 0
	p.verbs = append(p.verbs, verb)
	if verb == VerbConic {
		p.conicWeights = append(p.conicWeights, weight)
	}
}

// MoveTo begins a contour at (x, y). If the previous verb was also a moveTo, its point is replaced rather than a new
// verb appended (each contour holds only the last moveTo specified).
func (p *Path) MoveTo(x, y float32) *Path {
	if len(p.verbs) > 0 && p.verbs[len(p.verbs)-1] == VerbMove {
		p.points[len(p.points)-1] = geom.Pt(x, y)
		p.boundsValid = false
	} else {
		p.setLastMoveToIndex(len(p.points))
		p.growForVerb(VerbMove, 0)
		p.points = append(p.points, geom.Pt(x, y))
	}
	return p.dirtyAfterEdit()
}

// MoveToPt is MoveTo with a point argument.
func (p *Path) MoveToPt(pt geom.Point) *Path {
	return p.MoveTo(pt.X, pt.Y)
}

// RMoveTo begins a contour at the last point offset by (dx, dy).
func (p *Path) RMoveTo(dx, dy float32) *Path {
	var pt geom.Point
	if count := len(p.points); count > 0 {
		if p.lastMoveToIndex() >= 0 {
			pt = p.points[count-1]
		} else {
			pt = p.points[^p.lastMoveToIndex()]
		}
	}
	return p.MoveTo(pt.X+dx, pt.Y+dy)
}

// injectMoveToIfNeeded adds a moveTo (repeating the previous contour's start point, or (0, 0) for an empty path) when a
// segment verb arrives after a close or on an empty path.
func (p *Path) injectMoveToIfNeeded() {
	if p.lastMoveToIndex() < 0 {
		var x, y float32
		if len(p.verbs) != 0 {
			pt := p.points[^p.lastMoveToIndex()]
			x = pt.X
			y = pt.Y
		}
		p.MoveTo(x, y)
	}
}

// LineTo appends a line from the current point to (x, y).
func (p *Path) LineTo(x, y float32) *Path {
	p.injectMoveToIfNeeded()
	p.growForVerb(VerbLine, 0)
	p.points = append(p.points, geom.Pt(x, y))
	return p.dirtyAfterEdit()
}

// LineToPt is LineTo with a point argument.
func (p *Path) LineToPt(pt geom.Point) *Path {
	return p.LineTo(pt.X, pt.Y)
}

// RLineTo appends a line from the current point offset by (dx, dy).
func (p *Path) RLineTo(dx, dy float32) *Path {
	p.injectMoveToIfNeeded() // This can change the result of LastPt.
	pt, _ := p.LastPt()
	return p.LineTo(pt.X+dx, pt.Y+dy)
}

// QuadTo appends a quadratic Bezier from the current point through control point (x1, y1) to (x2, y2).
func (p *Path) QuadTo(x1, y1, x2, y2 float32) *Path {
	p.injectMoveToIfNeeded()
	p.growForVerb(VerbQuad, 0)
	p.points = append(p.points, geom.Pt(x1, y1), geom.Pt(x2, y2))
	return p.dirtyAfterEdit()
}

// QuadToPt is QuadTo with point arguments.
func (p *Path) QuadToPt(p1, p2 geom.Point) *Path {
	return p.QuadTo(p1.X, p1.Y, p2.X, p2.Y)
}

// RQuadTo appends a quadratic Bezier using control/end points relative to the current point.
func (p *Path) RQuadTo(dx1, dy1, dx2, dy2 float32) *Path {
	p.injectMoveToIfNeeded() // This can change the result of LastPt.
	pt, _ := p.LastPt()
	return p.QuadTo(pt.X+dx1, pt.Y+dy1, pt.X+dx2, pt.Y+dy2)
}

// ConicTo appends a conic (rational quadratic Bezier) with control point (x1, y1), end point (x2, y2), and weight w. A
// weight <= 0 (or NaN) reduces to a line to (x2, y2); a non-finite weight becomes two lines; a weight of one becomes a
// quad.
func (p *Path) ConicTo(x1, y1, x2, y2, w float32) *Path {
	switch {
	case !(w > 0): // catches <= 0 and NaN
		p.LineTo(x2, y2)
	case !geom.IsFinite(w):
		p.LineTo(x1, y1)
		p.LineTo(x2, y2)
	case w == 1:
		p.QuadTo(x1, y1, x2, y2)
	default:
		p.injectMoveToIfNeeded()
		p.growForVerb(VerbConic, w)
		p.points = append(p.points, geom.Pt(x1, y1), geom.Pt(x2, y2))
		p.dirtyAfterEdit()
	}
	return p
}

// ConicToPt is ConicTo with point arguments.
func (p *Path) ConicToPt(p1, p2 geom.Point, w float32) *Path {
	return p.ConicTo(p1.X, p1.Y, p2.X, p2.Y, w)
}

// RConicTo appends a conic using control/end points relative to the current point.
func (p *Path) RConicTo(dx1, dy1, dx2, dy2, w float32) *Path {
	p.injectMoveToIfNeeded() // This can change the result of LastPt.
	pt, _ := p.LastPt()
	return p.ConicTo(pt.X+dx1, pt.Y+dy1, pt.X+dx2, pt.Y+dy2, w)
}

// CubicTo appends a cubic Bezier from the current point through control points (x1, y1) and (x2, y2) to (x3, y3).
func (p *Path) CubicTo(x1, y1, x2, y2, x3, y3 float32) *Path {
	p.injectMoveToIfNeeded()
	p.growForVerb(VerbCubic, 0)
	p.points = append(p.points, geom.Pt(x1, y1), geom.Pt(x2, y2), geom.Pt(x3, y3))
	return p.dirtyAfterEdit()
}

// CubicToPt is CubicTo with point arguments.
func (p *Path) CubicToPt(p1, p2, p3 geom.Point) *Path {
	return p.CubicTo(p1.X, p1.Y, p2.X, p2.Y, p3.X, p3.Y)
}

// RCubicTo appends a cubic Bezier using control/end points relative to the current point.
func (p *Path) RCubicTo(dx1, dy1, dx2, dy2, dx3, dy3 float32) *Path {
	p.injectMoveToIfNeeded() // This can change the result of LastPt.
	pt, _ := p.LastPt()
	return p.CubicTo(pt.X+dx1, pt.Y+dy1, pt.X+dx2, pt.Y+dy2, pt.X+dx3, pt.Y+dy3)
}

// Close appends a close verb unless the path is empty or the last verb is already a close, and flags that the next
// segment verb must inject a moveTo back to the contour start.
func (p *Path) Close() *Path {
	if len(p.verbs) > 0 && p.verbs[len(p.verbs)-1] != VerbClose {
		p.growForVerb(VerbClose, 0)
	}
	// Signal that we need a moveTo to follow us (unless we're done).
	if idx := p.lastMoveToIndex(); idx >= 0 {
		p.setLastMoveToIndex(^idx)
	}
	return p
}

// Equal reports whether two paths have the same fill type, verbs, points and conic weights, using a bit-identical float
// compare (so NaN points never compare equal, and 0 == -0).
func Equal(a, b *Path) bool {
	if a == b {
		return true
	}
	if a.fillType != b.fillType ||
		len(a.verbs) != len(b.verbs) ||
		len(a.points) != len(b.points) ||
		len(a.conicWeights) != len(b.conicWeights) {
		return false
	}
	for i := range a.verbs {
		if a.verbs[i] != b.verbs[i] {
			return false
		}
	}
	for i := range a.points {
		if a.points[i] != b.points[i] {
			return false
		}
	}
	for i := range a.conicWeights {
		if a.conicWeights[i] != b.conicWeights[i] {
			return false
		}
	}
	return true
}
