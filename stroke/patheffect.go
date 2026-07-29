// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The PathEffect contract. The interface lives here (below the concrete effects) so that FillPathWithPaint and the
// canvas paint can consume effects without depending on the patheffect package.

package stroke

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// PointData holds everything needed to draw the point primitives returned by an AsPoints call.
type PointData struct {
	Points []geom.Point // the center point of each generated point
	First  path.Path    // if not empty, contains geometry for first point
	Last   path.Path    // if not empty, contains geometry for last point
	Size   geom.Point   // the size to draw the points
	Flags  uint32       // flags that impact the drawing of the points
}

// PointData flags. CirclesPointFlag is the only flag defined; a "use path" or "use clip" flag is not supported here.
// None of the effects implemented in this module set it — dash's AsPoints, the only AsPoints implementation, accepts
// butt caps only, so it always reports square points — but the canvas honors it, so an effect supplied by a caller can
// request the circle form.
const (
	CirclesPointFlag uint32 = 0x01 // draw points as circles (instead of rects)
)

// PathEffect is the interface implemented by path effects (dash, corner rounding, etc.).
//
// FilterPath applies the effect to src, appending the result to dst and returning true on success. rec may be modified
// (e.g. dash's special line case and 2D-line set it to fill/stroke). cullRect may be nil. The output is always in src's
// coordinate system regardless of ctm.
//
// AsPoints reports whether applying this effect to src yields a set of points, and if so fills in results. Only the
// dash effect implements it.
//
// ComputeFastBounds updates bounds (in/out) with a conservative bounds of the effect's output; returns false when fast
// bounds cannot be computed. A nil bounds performs a dry run, answering only whether bounds could be computed.
type PathEffect interface {
	FilterPath(dst, src *path.Path, rec *Rec, cullRect *geom.Rect, ctm *geom.Matrix) bool
	AsPoints(results *PointData, src *path.Path, rec *Rec, ctm *geom.Matrix, cullRect *geom.Rect) bool
	ComputeFastBounds(bounds *geom.Rect) bool
}
