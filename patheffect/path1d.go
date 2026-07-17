// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The 1D path stamp effect: stamps a path along the contours of the input at a fixed advance, translated, rotated, or
// morphed to follow the path.

package patheffect

import (
	"github.com/richardwilkes/canvas/contour"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// Path1DStyle selects how the stamped path is transformed at each position along the input contour.
type Path1DStyle uint8

// Path1DStyle values.
const (
	Path1DTranslate Path1DStyle = iota // translate the shape to each position
	Path1DRotate                       // rotate the shape about its center
	Path1DMorph                        // transform each point, and turn lines into curves
)

// maxReasonable1DIterations bounds the stepping loop: since we are stepping by a float, the loop might go on forever
// (or nearly so); this governor limits pathological values from looping too long (and allocating too much memory).
const maxReasonable1DIterations = 100000

// path1DEffect is the 1D path stamp effect implementation, including the contour walk.
type path1DEffect struct {
	noAsPoints
	path          *path.Path
	advance       float32
	initialOffset float32 // computed from phase
	style         Path1DStyle
}

// MakePath1D creates a 1D path stamp effect: stamp p along the filtered path every advance units, starting at phase.
// Returns nil for advance <= 0, non-finite advance/phase, or an empty path.
func MakePath1D(p *path.Path, advance, phase float32, style Path1DStyle) stroke.PathEffect {
	if advance <= 0 || !scalarIsFinite(advance) || !scalarIsFinite(phase) || p.IsEmpty() {
		return nil
	}

	// cleanup their phase parameter, inverting it so that it becomes an offset along the path (to match the
	// interpretation in PostScript)
	if phase < 0 {
		phase = -phase
		if phase > advance {
			phase = geom.ScalarMod(phase, advance)
		}
	} else {
		if phase > advance {
			phase = geom.ScalarMod(phase, advance)
		}
		phase = advance - phase
	}
	// now catch the edge case where phase == advance (within epsilon)
	if phase >= advance {
		phase = 0
	}

	return &path1DEffect{
		path:          p.Clone(),
		advance:       advance,
		initialOffset: phase,
		style:         style,
	}
}

func (e *path1DEffect) FilterPath(dst, src *path.Path, rec *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	rec.SetFillStyle()

	// Walk each contour, stamping at every advance.
	meas := contour.NewPathMeasure(src, false, 1)
	for {
		governor := maxReasonable1DIterations
		length := meas.Length()
		distance := e.initialOffset // starting position along this contour
		for distance < length {
			governor--
			if governor < 0 {
				break
			}
			delta := e.next(dst, distance, meas)
			if delta <= 0 {
				break
			}
			distance += delta
		}
		if governor < 0 {
			return false
		}
		if !meas.NextContour() {
			break
		}
	}
	return true
}

// morphPoints maps each point in src to follow meas at dist + point.X, rotated to the path's tangent.
func morphPoints(dst, src []geom.Point, meas *contour.PathMeasure, dist float32) bool {
	for i := range src {
		sx := src[i].X
		sy := src[i].Y

		pos, tangent, ok := meas.GetPosTan(dist+sx, true, true)
		if !ok {
			return false
		}

		var matrix geom.Matrix
		matrix.SetSinCos(tangent.Y, tangent.X)
		matrix.PreTranslate(-sx, 0)
		matrix.PostTranslate(pos.X, pos.Y)
		dst[i] = matrix.MapPoint(geom.Point{X: sx, Y: sy})
	}
	return true
}

// morphPath maps each verb of src to follow meas, converting lines to quads so the morph can curve them. TODO: needs
// differentially more subdivisions when the follow-path is curvy.
func morphPath(dst, src *path.Path, meas *contour.PathMeasure, dist float32) {
	iter := path.NewIter(src, false)
	var dstP [3]geom.Point
	var scratch [3]geom.Point

	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			if morphPoints(dstP[:1], pts[:1], meas, dist) {
				dst.MoveToPt(dstP[0])
			}
		case path.VerbLine:
			scratch[0] = pts[0]
			scratch[1] = geom.Point{X: (pts[0].X + pts[1].X) * 0.5, Y: (pts[0].Y + pts[1].Y) * 0.5}
			scratch[2] = pts[1]
			// now we look like a quad
			if morphPoints(dstP[:2], scratch[1:3], meas, dist) {
				dst.QuadToPt(dstP[0], dstP[1])
			}
		case path.VerbQuad:
			if morphPoints(dstP[:2], pts[1:3], meas, dist) {
				dst.QuadToPt(dstP[0], dstP[1])
			}
		case path.VerbConic:
			if morphPoints(dstP[:2], pts[1:3], meas, dist) {
				dst.ConicToPt(dstP[0], dstP[1], iter.ConicWeight())
			}
		case path.VerbCubic:
			if morphPoints(dstP[:3], pts[1:4], meas, dist) {
				dst.CubicToPt(dstP[0], dstP[1], dstP[2])
			}
		case path.VerbClose:
			dst.Close()
		}
	}
}

// next stamps the path once at distance and returns the advance to the next stamp position.
func (e *path1DEffect) next(dst *path.Path, distance float32, meas *contour.PathMeasure) float32 {
	switch e.style {
	case Path1DTranslate:
		if pos, _, ok := meas.GetPosTan(distance, true, false); ok {
			dst.AddPathOffset(e.path, pos.X, pos.Y, path.AddPathAppend)
		}
	case Path1DRotate:
		var matrix geom.Matrix
		if meas.GetMatrix(distance, &matrix, contour.GetPosAndTanMatrixFlag) {
			dst.AddPathMatrix(e.path, &matrix, path.AddPathAppend)
		}
	case Path1DMorph:
		morphPath(dst, e.path, meas, distance)
	}
	return e.advance
}

func (e *path1DEffect) ComputeFastBounds(*geom.Rect) bool {
	// For simplicity, assume fast bounds cannot be computed
	return false
}
