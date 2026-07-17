// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The discrete path effect: breaks the path into segments of the given length and randomly displaces them perpendicular
// to the tangent.

package patheffect

import (
	"github.com/richardwilkes/canvas/contour"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// lcgRandom generates pseudo random 32-bit numbers using a fast linear congruential equation.
type lcgRandom struct {
	seed uint32
}

// See "Numerical Recipes in C", 1992 page 284 for these constants.
const (
	lcgMul = 1664525
	lcgAdd = 1013904223
)

func (r *lcgRandom) nextU() uint32 {
	r.seed = r.seed*lcgMul + lcgAdd
	return r.seed
}

// nextSScalar1 returns the next pseudo random number as a scalar in [-1..1): a signed 16.16 fixed-point value (produced
// by an arithmetic >>15 of the raw 32-bit value) converted to float.
func (r *lcgRandom) nextSScalar1() float32 {
	fixed := int32(r.nextU()) >> 15
	return float32(fixed) * (1.0 / 65536)
}

// perterb displaces p along the CCW-rotated tangent by scale.
func perterb(p *geom.Point, tangent geom.Point, scale float32) {
	normal := tangent.RotateCCW()
	normal.SetLength(scale)
	*p = p.Add(normal)
}

// discreteEffect is the discrete path effect implementation.
type discreteEffect struct {
	noAsPoints
	segLength  float32
	perterb    float32
	seedAssist uint32
}

// MakeDiscrete creates a discrete path effect: segLength is the segment subdivision length, deviation the random
// displacement magnitude, seedAssist an optional extra seed (0 by default). Returns nil for non-finite inputs or
// segLength <= nearly-zero.
func MakeDiscrete(segLength, deviation float32, seedAssist uint32) stroke.PathEffect {
	if !scalarIsFinite(segLength) || !scalarIsFinite(deviation) {
		return nil
	}
	if segLength <= geom.ScalarNearlyZeroTol {
		return nil
	}
	return &discreteEffect{segLength: segLength, perterb: deviation, seedAssist: seedAssist}
}

func (d *discreteEffect) FilterPath(dst, src *path.Path, rec *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	doFill := rec.IsFillStyle()

	meas := contour.NewPathMeasure(src, doFill, 1)

	// Caller may supply their own seed assist, which by default is 0
	seed := d.seedAssist ^ uint32(geom.RoundToInt(meas.Length()))

	rand := lcgRandom{seed: seed ^ ((seed << 16) | (seed >> 16))}
	scale := d.perterb

	for {
		length := meas.Length()

		fillBump := 2
		if doFill {
			fillBump = 3
		}
		if d.segLength*float32(fillBump) > length {
			meas.GetSegment(0, length, dst, true) // too short for us to mangle
		} else {
			n := int(geom.RoundToInt(length / d.segLength))
			const maxReasonableIterations = 100000
			n = min(n, maxReasonableIterations)
			delta := length / float32(n)
			distance := float32(0)

			if meas.IsClosed() {
				n--
				distance += delta / 2
			}

			if p, v, ok := meas.GetPosTan(distance, true, true); ok {
				perterb(&p, v, rand.nextSScalar1()*scale)
				dst.MoveToPt(p)
			}
			for n--; n >= 0; n-- {
				distance += delta
				if p, v, ok := meas.GetPosTan(distance, true, true); ok {
					perterb(&p, v, rand.nextSScalar1()*scale)
					dst.LineToPt(p)
				}
			}
			if meas.IsClosed() {
				dst.Close()
			}
		}
		if !meas.NextContour() {
			break
		}
	}
	return true
}

func (d *discreteEffect) ComputeFastBounds(bounds *geom.Rect) bool {
	if bounds != nil {
		maxOutset := geom.ScalarAbs(d.perterb)
		*bounds = bounds.Outset(maxOutset, maxOutset)
	}
	return true
}
