// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// RadialGradient: a gradient radiating outward from a center point.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// RadialGradient is a gradient radiating outward from a center point.
type RadialGradient struct {
	gradientBase
	center geom.Point
	radius float32
}

// radToUnit returns the matrix mapping the gradient's circle to the unit circle at the origin.
func radToUnit(center geom.Point, radius float32) geom.Matrix {
	inv := geom.ScalarInvert(radius)
	var m geom.Matrix
	m.SetTranslate(-center.X, -center.Y)
	m.PostScale(inv, inv)
	return m
}

// NewRadialGradient builds a radial gradient shader from byte colors; localMatrix may be nil.
func NewRadialGradient(center geom.Point, radius float32, colors []colorcore.Color, pos []float32, tileMode TileMode, localMatrix *geom.Matrix) Shader {
	if radius < 0 {
		return nil
	}
	if !validGradient(colors, pos, tileMode) {
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
	if geom.ScalarNearlyZeroTolerance(radius, gradDegenerateThreshold) {
		// Degenerate gradient optimization, and no special logic needed for clamped radial gradient
		return withLocalMatrixPtr(makeDegenerateGradient(c4f, pos, tileMode), localMatrix)
	}
	g := &RadialGradient{center: center, radius: radius}
	g.initGradientBase(c4f, pos, tileMode, radToUnit(center, radius))
	return withLocalMatrixPtr(g, localMatrix)
}

// Center returns the gradient's center; Radius returns its radius.
func (g *RadialGradient) Center() geom.Point { return g.center }

// Radius returns the gradient's radius.
func (g *RadialGradient) Radius() float32 { return g.radius }

// IsOpaque implements Shader by delegating to the shared gradient base logic.
func (g *RadialGradient) IsOpaque() bool { return g.isOpaqueBase() }

// appendStages runs the base gradient stages, then appends the xy_to_radius stage.
func (g *RadialGradient) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	return g.appendBaseStages(p, &m, func(p *Pipeline) {
		p.appendXYToRadius()
	})
}
