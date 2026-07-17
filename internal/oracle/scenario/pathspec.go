// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package scenario

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// PathOpKind identifies one recorded path-building operation. The set matches the sk_path_* builder entry points
// exactly (note: the public surface has no rquad_to, so neither does this).
type PathOpKind uint8

// PathOpKind values.
const (
	OpMoveTo PathOpKind = iota
	OpLineTo
	OpQuadTo
	OpConicTo
	OpCubicTo
	OpRMoveTo
	OpRLineTo
	OpRConicTo
	OpRCubicTo
	OpClose
	OpArcToRotated  // SVG endpoint arc: F[0..2]=rx,ry,angle, Size, Dir, F[3..4]=x,y
	OpRArcToRotated // relative SVG endpoint arc
	OpArcToOval     // F[0]=startAngle, F[1]=sweepAngle, Bool=forceMoveTo, Rect=oval
	OpArcToTangent  // F[0..4]=x1,y1,x2,y2,radius
	OpAddRect       // Rect, Dir
	OpAddOval       // Rect, Dir
	OpAddCircle     // F[0..2]=x,y,r, Dir
	OpAddRoundRect  // Rect, F[0..1]=rx,ry, Dir
	OpAddPoly       // Pts, Bool=close
	OpAddPath       // Sub, Mode
	OpAddPathOffset // Sub, F[0..1]=dx,dy, Mode
	OpAddPathMatrix // Sub, M, Mode
	OpAddPathReverse
)

// PathOp is one recorded operation; which fields are meaningful depends on Kind.
type PathOp struct {
	Sub  *PathSpec
	M    *geom.Matrix
	Pts  []geom.Point
	Rect geom.Rect
	F    [6]float32
	Kind PathOpKind
	Dir  geom.PathDirection
	Size path.ArcSize
	Mode path.AddPathMode
	Bool bool
}

// PathSpec is a declaratively recorded path, replayable into any backend's path type.
type PathSpec struct {
	Ops  []PathOp
	Fill path.FillType
}

// NewPathSpec returns an empty spec ready for chained recording.
func NewPathSpec() *PathSpec { return &PathSpec{} }

func (s *PathSpec) add(op *PathOp) *PathSpec {
	s.Ops = append(s.Ops, *op)
	return s
}

// SetFill records the fill type the built path uses.
func (s *PathSpec) SetFill(ft path.FillType) *PathSpec { s.Fill = ft; return s }

// MoveTo records a move_to.
func (s *PathSpec) MoveTo(x, y float32) *PathSpec {
	return s.add(&PathOp{Kind: OpMoveTo, F: [6]float32{x, y}})
}

// LineTo records a line_to.
func (s *PathSpec) LineTo(x, y float32) *PathSpec {
	return s.add(&PathOp{Kind: OpLineTo, F: [6]float32{x, y}})
}

// QuadTo records a quad_to.
func (s *PathSpec) QuadTo(x1, y1, x2, y2 float32) *PathSpec {
	return s.add(&PathOp{Kind: OpQuadTo, F: [6]float32{x1, y1, x2, y2}})
}

// ConicTo records a conic_to.
func (s *PathSpec) ConicTo(x1, y1, x2, y2, w float32) *PathSpec {
	return s.add(&PathOp{Kind: OpConicTo, F: [6]float32{x1, y1, x2, y2, w}})
}

// CubicTo records a cubic_to.
func (s *PathSpec) CubicTo(x1, y1, x2, y2, x3, y3 float32) *PathSpec {
	return s.add(&PathOp{Kind: OpCubicTo, F: [6]float32{x1, y1, x2, y2, x3, y3}})
}

// RMoveTo records a relative move_to.
func (s *PathSpec) RMoveTo(dx, dy float32) *PathSpec {
	return s.add(&PathOp{Kind: OpRMoveTo, F: [6]float32{dx, dy}})
}

// RLineTo records a relative line_to.
func (s *PathSpec) RLineTo(dx, dy float32) *PathSpec {
	return s.add(&PathOp{Kind: OpRLineTo, F: [6]float32{dx, dy}})
}

// RConicTo records a relative conic_to.
func (s *PathSpec) RConicTo(dx1, dy1, dx2, dy2, w float32) *PathSpec {
	return s.add(&PathOp{Kind: OpRConicTo, F: [6]float32{dx1, dy1, dx2, dy2, w}})
}

// RCubicTo records a relative cubic_to.
func (s *PathSpec) RCubicTo(dx1, dy1, dx2, dy2, dx3, dy3 float32) *PathSpec {
	return s.add(&PathOp{Kind: OpRCubicTo, F: [6]float32{dx1, dy1, dx2, dy2, dx3, dy3}})
}

// Close records a close of the current contour.
func (s *PathSpec) Close() *PathSpec { return s.add(&PathOp{Kind: OpClose}) }

// ArcToRotated records an SVG-style rotated elliptical arc.
func (s *PathSpec) ArcToRotated(rx, ry, angle float32, large path.ArcSize, sweep geom.PathDirection, x, y float32) *PathSpec {
	return s.add(&PathOp{Kind: OpArcToRotated, F: [6]float32{rx, ry, angle, x, y}, Size: large, Dir: sweep})
}

// RArcToRotated records a relative SVG-style rotated elliptical arc.
func (s *PathSpec) RArcToRotated(rx, ry, angle float32, large path.ArcSize, sweep geom.PathDirection, dx, dy float32) *PathSpec {
	return s.add(&PathOp{Kind: OpRArcToRotated, F: [6]float32{rx, ry, angle, dx, dy}, Size: large, Dir: sweep})
}

// ArcToOval records an arc along the given oval.
func (s *PathSpec) ArcToOval(oval geom.Rect, startAngle, sweepAngle float32, forceMoveTo bool) *PathSpec {
	return s.add(&PathOp{Kind: OpArcToOval, Rect: oval, F: [6]float32{startAngle, sweepAngle}, Bool: forceMoveTo})
}

// ArcToTangent records a tangent-to-tangent arc of the given radius.
func (s *PathSpec) ArcToTangent(x1, y1, x2, y2, radius float32) *PathSpec {
	return s.add(&PathOp{Kind: OpArcToTangent, F: [6]float32{x1, y1, x2, y2, radius}})
}

// AddRect records adding a rect contour.
func (s *PathSpec) AddRect(r geom.Rect, dir geom.PathDirection) *PathSpec {
	return s.add(&PathOp{Kind: OpAddRect, Rect: r, Dir: dir})
}

// AddOval records adding an oval contour.
func (s *PathSpec) AddOval(r geom.Rect, dir geom.PathDirection) *PathSpec {
	return s.add(&PathOp{Kind: OpAddOval, Rect: r, Dir: dir})
}

// AddCircle records adding a circle contour.
func (s *PathSpec) AddCircle(x, y, radius float32, dir geom.PathDirection) *PathSpec {
	return s.add(&PathOp{Kind: OpAddCircle, F: [6]float32{x, y, radius}, Dir: dir})
}

// AddRoundRect records adding a round-rect contour with uniform corner radii.
func (s *PathSpec) AddRoundRect(r geom.Rect, rx, ry float32, dir geom.PathDirection) *PathSpec {
	return s.add(&PathOp{Kind: OpAddRoundRect, Rect: r, F: [6]float32{rx, ry}, Dir: dir})
}

// AddPoly records adding a polygon contour, optionally closed.
func (s *PathSpec) AddPoly(pts []geom.Point, closePath bool) *PathSpec {
	return s.add(&PathOp{Kind: OpAddPoly, Pts: pts, Bool: closePath})
}

// AddPath records appending a sub-spec.
func (s *PathSpec) AddPath(sub *PathSpec, mode path.AddPathMode) *PathSpec {
	return s.add(&PathOp{Kind: OpAddPath, Sub: sub, Mode: mode})
}

// AddPathOffset records appending a sub-spec translated by (dx, dy).
func (s *PathSpec) AddPathOffset(sub *PathSpec, dx, dy float32, mode path.AddPathMode) *PathSpec {
	return s.add(&PathOp{Kind: OpAddPathOffset, Sub: sub, F: [6]float32{dx, dy}, Mode: mode})
}

// AddPathMatrix records appending a sub-spec transformed by m.
func (s *PathSpec) AddPathMatrix(sub *PathSpec, m *geom.Matrix, mode path.AddPathMode) *PathSpec {
	return s.add(&PathOp{Kind: OpAddPathMatrix, Sub: sub, M: m, Mode: mode})
}

// AddPathReverse records appending a sub-spec with its verbs reversed.
func (s *PathSpec) AddPathReverse(sub *PathSpec) *PathSpec {
	return s.add(&PathOp{Kind: OpAddPathReverse, Sub: sub})
}

// BuildGo replays the spec into the Go port's path type.
func (s *PathSpec) BuildGo() *path.Path {
	p := path.New()
	for _, op := range s.Ops {
		switch op.Kind {
		case OpMoveTo:
			p.MoveTo(op.F[0], op.F[1])
		case OpLineTo:
			p.LineTo(op.F[0], op.F[1])
		case OpQuadTo:
			p.QuadTo(op.F[0], op.F[1], op.F[2], op.F[3])
		case OpConicTo:
			p.ConicTo(op.F[0], op.F[1], op.F[2], op.F[3], op.F[4])
		case OpCubicTo:
			p.CubicTo(op.F[0], op.F[1], op.F[2], op.F[3], op.F[4], op.F[5])
		case OpRMoveTo:
			p.RMoveTo(op.F[0], op.F[1])
		case OpRLineTo:
			p.RLineTo(op.F[0], op.F[1])
		case OpRConicTo:
			p.RConicTo(op.F[0], op.F[1], op.F[2], op.F[3], op.F[4])
		case OpRCubicTo:
			p.RCubicTo(op.F[0], op.F[1], op.F[2], op.F[3], op.F[4], op.F[5])
		case OpClose:
			p.Close()
		case OpArcToRotated:
			p.ArcToRotated(op.F[0], op.F[1], op.F[2], op.Size, op.Dir, op.F[3], op.F[4])
		case OpRArcToRotated:
			p.RArcToRotated(op.F[0], op.F[1], op.F[2], op.Size, op.Dir, op.F[3], op.F[4])
		case OpArcToOval:
			p.ArcToOval(op.Rect, op.F[0], op.F[1], op.Bool)
		case OpArcToTangent:
			p.ArcToTangent(op.F[0], op.F[1], op.F[2], op.F[3], op.F[4])
		case OpAddRect:
			p.AddRect(op.Rect, op.Dir)
		case OpAddOval:
			p.AddOval(op.Rect, op.Dir)
		case OpAddCircle:
			p.AddCircle(op.F[0], op.F[1], op.F[2], op.Dir)
		case OpAddRoundRect:
			p.AddRoundRect(op.Rect, op.F[0], op.F[1], op.Dir)
		case OpAddPoly:
			p.AddPoly(op.Pts, op.Bool)
		case OpAddPath:
			p.AddPath(op.Sub.BuildGo(), op.Mode)
		case OpAddPathOffset:
			p.AddPathOffset(op.Sub.BuildGo(), op.F[0], op.F[1], op.Mode)
		case OpAddPathMatrix:
			p.AddPathMatrix(op.Sub.BuildGo(), op.M, op.Mode)
		case OpAddPathReverse:
			p.ReverseAddPath(op.Sub.BuildGo())
		}
	}
	p.SetFillType(s.Fill)
	return p
}
