// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The 2D line/path lattice effects: stamps lines or a path over the lattice cells covered by the input path, in the
// coordinate system defined by the lattice matrix.

package patheffect

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stroke"
)

// grid2D is the shared 2D lattice state: the lattice matrix, its inverse, and the per-cell callback supplied by the
// concrete effect (a path stamp, or a line span for line2DEffect).
type grid2D struct {
	matrix    geom.Matrix
	inverse   geom.Matrix
	inverible bool
}

func newGrid2D(matrix *geom.Matrix) grid2D {
	g := grid2D{matrix: *matrix}
	g.inverse, g.inverible = matrix.Invert()
	return g
}

// filterPath calls nextSpan for each covered lattice span.
func (g *grid2D) filterPath(dst, src *path.Path, nextSpan func(x, y, ucount int32, dst *path.Path)) bool {
	if !g.inverible {
		return false
	}

	tmp := &path.Path{}
	src.TransformTo(&g.inverse, tmp)
	ir := tmp.Bounds().Round()
	if !ir.IsEmpty() {
		var rgn raster.Region
		rgn.SetPath(tmp, raster.NewRegionRect(ir))
		for it := raster.NewRegionIterator(&rgn); !it.Done(); it.Next() {
			rect := it.Rect()
			for y := rect.Top; y < rect.Bottom; y++ {
				nextSpan(rect.Left, y, rect.Width(), dst)
			}
		}
	}
	return true
}

///////////////////////////////////////////////////////////////////////////////
// 2D line lattice effect

// line2DEffect is the 2D line lattice effect implementation.
type line2DEffect struct {
	noAsPoints
	grid  grid2D
	width float32
}

// MakeLine2D creates horizontal lattice lines of the given stroke width, in the coordinate system defined by matrix.
// Returns nil for a negative width.
func MakeLine2D(width float32, matrix *geom.Matrix) stroke.PathEffect {
	if !(width >= 0) {
		return nil
	}
	return &line2DEffect{grid: newGrid2D(matrix), width: width}
}

func (e *line2DEffect) FilterPath(dst, src *path.Path, rec *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	if e.grid.filterPath(dst, src, e.nextSpan) {
		rec.SetStrokeStyle(e.width, false)
		return true
	}
	return false
}

// nextSpan draws one horizontal lattice line spanning ucount cells starting at (u, v).
func (e *line2DEffect) nextSpan(u, v, ucount int32, dst *path.Path) {
	if ucount > 1 {
		src := [2]geom.Point{
			{X: float32(u) + 0.5, Y: float32(v) + 0.5},
			{X: float32(u+ucount) + 0.5, Y: float32(v) + 0.5},
		}
		var dstP [2]geom.Point
		e.grid.matrix.MapPoints(dstP[:], src[:])

		dst.MoveToPt(dstP[0])
		dst.LineToPt(dstP[1])
	}
}

func (e *line2DEffect) ComputeFastBounds(*geom.Rect) bool {
	// For simplicity, assume fast bounds cannot be computed
	return false
}

///////////////////////////////////////////////////////////////////////////////
// 2D path lattice effect

// path2DEffect is the 2D path lattice effect implementation.
type path2DEffect struct {
	noAsPoints
	path *path.Path
	grid grid2D
}

// MakePath2D creates an effect that stamps p at each lattice cell covered by the filtered path, in the coordinate
// system defined by matrix.
func MakePath2D(matrix *geom.Matrix, p *path.Path) stroke.PathEffect {
	return &path2DEffect{grid: newGrid2D(matrix), path: p.Clone()}
}

func (e *path2DEffect) FilterPath(dst, src *path.Path, _ *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	return e.grid.filterPath(dst, src, e.nextSpan)
}

// nextSpan stamps e.path once per lattice cell in the span.
func (e *path2DEffect) nextSpan(x, y, ucount int32, dst *path.Path) {
	if !e.grid.inverible {
		return
	}
	src := geom.Point{X: float32(x) + 0.5, Y: float32(y) + 0.5}
	for ucount > 0 {
		loc := e.grid.matrix.MapPoint(src)
		dst.AddPathOffset(e.path, loc.X, loc.Y, path.AddPathAppend)
		src.X++
		ucount--
	}
}

func (e *path2DEffect) ComputeFastBounds(*geom.Rect) bool {
	// For simplicity, assume fast bounds cannot be computed
	return false
}
