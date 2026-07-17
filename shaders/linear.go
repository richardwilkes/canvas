// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// LinearGradient: a gradient interpolating along the line from one point to another.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// LinearGradient is a gradient interpolating along the line from start to end.
type LinearGradient struct {
	gradientBase
	start, end geom.Point
}

// linearPtsToUnit returns the matrix mapping the segment p0-p1 onto the unit x-axis.
func linearPtsToUnit(p0, p1 geom.Point) geom.Matrix {
	vec := geom.Point{X: p1.X - p0.X, Y: p1.Y - p0.Y}
	mag := vec.Length()
	inv := geom.ScalarInvert(mag)
	if mag == 0 {
		inv = 0
	}
	vec.X *= inv
	vec.Y *= inv
	var m geom.Matrix
	m.SetSinCosPivot(-vec.Y, vec.X, p0.X, p0.Y)
	m.PostTranslate(-p0.X, -p0.Y)
	m.PostScale(inv, inv)
	return m
}

// NewLinearGradient builds a linear gradient shader from byte colors; localMatrix may be nil. A nil result indicates
// the gradient cannot be constructed (e.g. invalid points or degenerate matrix).
func NewLinearGradient(p0, p1 geom.Point, colors []colorcore.Color, pos []float32, tileMode TileMode, localMatrix *geom.Matrix) Shader {
	d := geom.Point{X: p1.X - p0.X, Y: p1.Y - p0.Y}
	if !geom.IsFinite(d.Length()) {
		return nil
	}
	if !validGradient(colors, tileMode) {
		return nil
	}
	c4f := colors4f(colors)
	if len(colors) == 1 {
		return NewColor4f(c4f[0])
	}
	if localMatrix != nil {
		if _, ok := localMatrix.Invert(); !ok {
			return nil
		}
	}
	if geom.ScalarNearlyZeroTolerance(d.Length(), gradDegenerateThreshold) {
		// Degenerate gradient: the two half planes' divider becomes undefined, so use the end color for a stable
		// solution.
		return withLocalMatrixPtr(makeDegenerateGradient(c4f, pos, tileMode), localMatrix)
	}
	g := &LinearGradient{start: p0, end: p1}
	g.initGradientBase(c4f, pos, tileMode, linearPtsToUnit(p0, p1))
	return withLocalMatrixPtr(g, localMatrix)
}

// withLocalMatrixPtr applies an optional local matrix, the tail every gradient factory shares.
func withLocalMatrixPtr(s Shader, localMatrix *geom.Matrix) Shader {
	if localMatrix == nil {
		return s
	}
	return NewWithLocalMatrix(s, *localMatrix)
}

// Start returns the gradient's start point; End returns its end point.
func (g *LinearGradient) Start() geom.Point { return g.start }

// End returns the gradient's end point.
func (g *LinearGradient) End() geom.Point { return g.end }

// IsOpaque implements Shader by delegating to the shared gradient base logic.
func (g *LinearGradient) IsOpaque() bool { return g.isOpaqueBase() }

// appendStages runs the base gradient stages; a linear gradient needs no extra coordinate stage.
func (g *LinearGradient) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	return g.appendBaseStages(p, &m, func(_ *Pipeline) {
		// No extra stage needed for linear gradients.
	})
}
