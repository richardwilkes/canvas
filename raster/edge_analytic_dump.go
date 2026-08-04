// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package raster

import "github.com/richardwilkes/canvas/geom"

// AnalyticEdgeSegment is one line segment of an analytic curve edge, in the field order the oracle shim's dump uses: X,
// DX, UpperX, UpperY, LowerY, DY, Winding.
type AnalyticEdgeSegment = [7]int32

func recordAnalyticEdge(e *AnalyticEdge) AnalyticEdgeSegment {
	return AnalyticEdgeSegment{
		int32(e.X), int32(e.DX), int32(e.UpperX), int32(e.UpperY),
		int32(e.LowerY), int32(e.DY), int32(e.Winding),
	}
}

// AnalyticQuadSegments returns the raw line-segment stream an AnalyticQuadraticEdge produces for the given control
// points (no walker interleaving), with the piece's endpoints snapped to a 1/(1<<snapAccuracy) grid.
// Differential-test instrumentation, matching the oracle shim's oracle_analytic_quad_segments; pass SkiaSnapAccuracy to
// replay the stream Skia produces, or analyticSnapAccuracy's value for the port's own.
func AnalyticQuadSegments(pts []geom.Point, snapAccuracy int) []AnalyticEdgeSegment {
	var edge AnalyticQuadraticEdge
	if !edge.setQuadratic(pts, snapAccuracy) {
		return nil
	}
	var out []AnalyticEdgeSegment
	for {
		out = append(out, recordAnalyticEdge(&edge.AnalyticEdge))
		if edge.CurveCount <= 0 || !edge.updateCurve() {
			return out
		}
	}
}

// AnalyticCubicSegments is AnalyticQuadSegments for cubics.
func AnalyticCubicSegments(pts []geom.Point, snapAccuracy int) []AnalyticEdgeSegment {
	var edge AnalyticCubicEdge
	if !edge.setCubic(pts, snapAccuracy) {
		return nil
	}
	var out []AnalyticEdgeSegment
	for {
		out = append(out, recordAnalyticEdge(&edge.AnalyticEdge))
		if edge.CurveCount >= 0 || !edge.updateCurve() {
			return out
		}
	}
}
