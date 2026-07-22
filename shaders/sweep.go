// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SweepGradient: an angular gradient sweeping around a center point.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// SweepGradient is an angular gradient sweeping around a center point.
type SweepGradient struct {
	gradientBase
	center geom.Point
	tBias  float32
	tScale float32
}

// NewSweepGradient builds a sweep gradient shader from byte colors; angles are in degrees, localMatrix may be nil.
func NewSweepGradient(center geom.Point, colors []colorcore.Color, pos []float32, tileMode TileMode, startAngle, endAngle float32, localMatrix *geom.Matrix) Shader {
	if !validGradient(colors, pos, tileMode) {
		return nil
	}
	c4f := colors4f(colors)
	if len(colors) == 1 {
		return NewColor4f(c4f[0])
	}
	if !geom.IsFinite(startAngle, endAngle) || startAngle > endAngle {
		return nil
	}
	if localMatrix != nil {
		if _, ok := localMatrix.Invert(); !ok {
			return nil
		}
	}

	if geom.ScalarNearlyEqualTolerance(startAngle, endAngle, gradDegenerateThreshold) {
		// Degenerate gradient, which should follow default degenerate behavior unless it is clamped and the angle is
		// greater than 0.
		if tileMode == TileClamp && endAngle > gradDegenerateThreshold {
			// The first color is repeated from 0 to the angle, then a hardstop switches to the last color (all other
			// colors are compressed to the infinitely thin interpolation region).
			clampColors := []colorcore.Color{colors[0], colors[0], colors[len(colors)-1]}
			return NewSweepGradient(center, clampColors, []float32{0, 1, 1}, tileMode, 0, endAngle,
				localMatrix)
		}
		return withLocalMatrixPtr(makeDegenerateGradient(c4f, pos, tileMode), localMatrix)
	}

	if startAngle <= 0 && endAngle >= 360 {
		// If the t-range includes [0,1], then we can always use clamping (presumably faster).
		tileMode = TileClamp
	}

	t0 := startAngle / 360
	t1 := endAngle / 360
	g := &SweepGradient{center: center, tBias: -t0, tScale: 1 / (t1 - t0)}
	var ptsToUnit geom.Matrix
	ptsToUnit.SetTranslate(-center.X, -center.Y)
	g.initGradientBase(c4f, pos, tileMode, ptsToUnit)
	return withLocalMatrixPtr(g, localMatrix)
}

// Center returns the sweep's center; TBias/TScale are the angular remapping coefficients.
func (g *SweepGradient) Center() geom.Point { return g.center }

// TBias returns the sweep's t bias (-t0).
func (g *SweepGradient) TBias() float32 { return g.tBias }

// TScale returns the sweep's t scale (1/(t1-t0)).
func (g *SweepGradient) TScale() float32 { return g.tScale }

// IsOpaque implements Shader by delegating to the shared gradient base logic.
func (g *SweepGradient) IsOpaque() bool { return g.isOpaqueBase() }

// appendStages runs the base gradient stages, then appends the sweep-specific stages: xy_to_unit_angle then the t remap
// Scale(tScale, 1) * Translate(tBias, 0).
func (g *SweepGradient) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	return g.appendBaseStages(p, &m, func(p *Pipeline) {
		p.appendXYToUnitAngle()
		remap := geom.ScaleMatrix(g.tScale, 1)
		remap.PreTranslate(g.tBias, 0)
		p.appendMatrix(&remap)
	})
}
