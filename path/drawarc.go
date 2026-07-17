// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// DrawArcIsConvex reports whether the path CreateDrawArcPath would build for the given sweep angle, useCenter, and
// fill/path-effect configuration is convex.
func DrawArcIsConvex(sweepAngle float32, useCenter, isFillNoPathEffect bool) bool {
	abs := float32(math.Abs(float64(sweepAngle)))
	if isFillNoPathEffect && abs >= 360 {
		// This gets converted to an oval.
		return true
	}
	if useCenter {
		// This is a pie wedge. It's convex if the angle is <= 180.
		return abs <= 180
	}
	// When the angle exceeds 360 this wraps back on top of itself. Otherwise it is a circle clipped to a secant, i.e.
	// convex.
	return abs <= 360
}

// CreateDrawArcPath builds the path that a drawArc canvas call renders. oval must be non-empty and sweepAngle non-zero.
func CreateDrawArcPath(oval geom.Rect, startAngle, sweepAngle float32, useCenter, isFillNoPathEffect bool) *Path {
	// We cap the number of total rotations. This keeps the resulting paths simpler. More important, it prevents values
	// so large that the loops below never terminate (once ULP > 360).
	if float32(math.Abs(float64(sweepAngle))) > 3600 {
		sweepAngle = float32(math.Copysign(3600, float64(sweepAngle))) +
			float32(math.Mod(float64(sweepAngle), 360))
	}

	p := &Path{}

	if isFillNoPathEffect && float32(math.Abs(float64(sweepAngle))) >= 360 {
		p.AddOval(oval, geom.DirectionCW)
		return p
	}

	if useCenter {
		p.MoveTo(oval.CenterX(), oval.CenterY())
	}
	firstDir := firstDirectionCCW
	if sweepAngle > 0 {
		firstDir = firstDirectionCW
	}
	convex := DrawArcIsConvex(sweepAngle, useCenter, isFillNoPathEffect)
	// Arc to mods at 360 and drawArc is not supposed to.
	forceMoveTo := !useCenter
	for sweepAngle <= -360 {
		p.ArcToOval(oval, startAngle, -180, forceMoveTo)
		startAngle -= 180
		p.ArcToOval(oval, startAngle, -180, false)
		startAngle -= 180
		forceMoveTo = false
		sweepAngle += 360
	}
	for sweepAngle >= 360 {
		p.ArcToOval(oval, startAngle, 180, forceMoveTo)
		startAngle += 180
		p.ArcToOval(oval, startAngle, 180, false)
		startAngle += 180
		forceMoveTo = false
		sweepAngle -= 360
	}
	p.ArcToOval(oval, startAngle, sweepAngle, forceMoveTo)
	if useCenter {
		p.Close()
	}

	if convex {
		p.SetConvexity(firstDirectionToConvexity(firstDir))
	} else {
		p.SetConvexity(ConvexityConcave)
	}
	return p
}
