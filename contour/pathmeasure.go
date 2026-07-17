// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The single-contour-at-a-time convenience wrapper over the contour measure iterator, which is the form the path
// effects consume.

package contour

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// PathMeasure measures a path one contour at a time.
type PathMeasure struct {
	iter    *Iter
	contour *Measure
}

// NewPathMeasure returns a measure positioned at the path's first contour.
func NewPathMeasure(p *path.Path, forceClosed bool, resScale float32) *PathMeasure {
	m := &PathMeasure{iter: NewIter(p, forceClosed, resScale)}
	m.contour = m.iter.Next()
	return m
}

// SetPath assigns a new path (resScale 1), restarting at its first contour.
func (m *PathMeasure) SetPath(p *path.Path, forceClosed bool) {
	m.iter.Reset(p, forceClosed, 1)
	m.contour = m.iter.Next()
}

// Length returns the length of the current contour, or 0 if none.
func (m *PathMeasure) Length() float32 {
	if m.contour != nil {
		return m.contour.Length()
	}
	return 0
}

// GetPosTan returns the position and/or normalized tangent at the given distance along the current contour.
func (m *PathMeasure) GetPosTan(distance float32, wantPos, wantTan bool) (pos, tangent geom.Point, ok bool) {
	if m.contour == nil {
		return pos, tangent, false
	}
	return m.contour.GetPosTan(distance, wantPos, wantTan)
}

// GetMatrix builds the matrix positioning (and optionally orienting) geometry at the given distance along the current
// contour.
func (m *PathMeasure) GetMatrix(distance float32, matrix *geom.Matrix, flags MatrixFlags) bool {
	return m.contour != nil && m.contour.GetMatrix(distance, matrix, flags)
}

// GetSegment appends the current contour's slice from startD to stopD to dst.
func (m *PathMeasure) GetSegment(startD, stopD float32, dst *path.Path, startWithMoveTo bool) bool {
	return m.contour != nil && m.contour.GetSegment(startD, stopD, dst, startWithMoveTo)
}

// IsClosed reports whether the current contour is closed.
func (m *PathMeasure) IsClosed() bool {
	return m.contour != nil && m.contour.IsClosed()
}

// NextContour moves to the next contour, reporting whether one exists.
func (m *PathMeasure) NextContour() bool {
	m.contour = m.iter.Next()
	return m.contour != nil
}

// CurrentMeasure returns the current contour's measure (nil if none).
func (m *PathMeasure) CurrentMeasure() *Measure {
	return m.contour
}
