// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

// Target is the kind of rendering glyphs are being hinted for. Instructions can ask about it and adapt.
type Target uint8

// The possible Target values.
const (
	// TargetMono is aliased, one bit per pixel. It runs the interpreter without backward compatibility.
	TargetMono Target = iota
	// TargetNormal is grayscale anti-aliasing.
	TargetNormal
	// TargetLight is grayscale anti-aliasing with light hinting.
	TargetLight
	// TargetLCD is subpixel anti-aliasing for horizontal stripes.
	TargetLCD
	// TargetVerticalLCD is subpixel anti-aliasing for vertical stripes.
	TargetVerticalLCD
)

func (t Target) isSmooth() bool { return t != TargetMono }

func (t Target) isGrayscaleClearType() bool { return t == TargetNormal || t == TargetLight }

func (t Target) isVerticalLCD() bool { return t == TargetVerticalLCD }

// Point flag bits. onCurve is the font format's; the touched markers are the interpreter's own.
const (
	flagOnCurve  = 0x01
	flagTouchedX = 0x10
	flagTouchedY = 0x20
	flagTouched  = flagTouchedX | flagTouchedY
)

type coordAxis uint8

const (
	axisBoth coordAxis = iota
	axisX
	axisY
)

func (a coordAxis) touchedMarker() uint8 {
	switch a {
	case axisX:
		return flagTouchedX
	case axisY:
		return flagTouchedY
	default:
		return flagTouched
	}
}

type zonePointer uint8

const (
	zoneTwilight zonePointer = iota
	zoneGlyph
)

type roundMode uint8

const (
	roundGrid roundMode = iota
	roundHalfGrid
	roundDoubleGrid
	roundDownToGrid
	roundUpToGrid
	roundOff
	roundSuper
	roundSuper45
)

type roundState struct {
	mode      roundMode
	threshold int32
	phase     int32
	period    int32
}

func (r *roundState) round(distance int32) int32 {
	bias := r.threshold - r.phase
	neg := -distance
	switch r.mode {
	case roundHalfGrid:
		if distance >= 0 {
			return max(floor26(distance)+32, 0)
		}
		return min(-(floor26(neg) + 32), 0)
	case roundGrid:
		if distance >= 0 {
			return max(round26(distance), 0)
		}
		return min(-round26(neg), 0)
	case roundDoubleGrid:
		if distance >= 0 {
			return max(roundPad(distance, 32), 0)
		}
		return min(-roundPad(neg, 32), 0)
	case roundDownToGrid:
		if distance >= 0 {
			return max(floor26(distance), 0)
		}
		return min(-floor26(neg), 0)
	case roundUpToGrid:
		if distance >= 0 {
			return max(ceil26(distance), 0)
		}
		return min(-ceil26(neg), 0)
	case roundSuper:
		if distance >= 0 {
			val := ((distance + bias) & -r.period) + r.phase
			if val < 0 {
				return r.phase
			}
			return val
		}
		val := -((bias - distance) & -r.period) - r.phase
		if val > 0 {
			return -r.phase
		}
		return val
	case roundSuper45:
		if r.period == 0 {
			return distance
		}
		if distance >= 0 {
			val := (distance+bias)/r.period*r.period + r.phase
			if val < 0 {
				return r.phase
			}
			return val
		}
		val := -((bias - distance) / r.period * r.period) - r.phase
		if val > 0 {
			return -r.phase
		}
		return val
	default:
		return distance
	}
}

// retainedState is the part of the graphics state the control value program leaves behind for every glyph program.
type retainedState struct {
	target            Target
	autoFlip          bool
	scanControl       bool
	isRotated         bool
	isStretched       bool
	instructControl   uint8
	deltaBase         uint16
	deltaShift        uint16
	controlValueCutin int32
	minDistance       int32
	scanType          int32
	singleWidthCutin  int32
	singleWidth       int32
	scale             int32
	ppem              int32
}

func newRetainedState(scale, ppem int32, target Target) retainedState {
	return retainedState{
		autoFlip: true,
		// 17/16 of a pixel.
		controlValueCutin: 68,
		deltaBase:         9,
		deltaShift:        3,
		minDistance:       64,
		target:            target,
		scale:             scale,
		ppem:              ppem,
	}
}

type zone struct {
	// unscaled is empty for the twilight zone, whose unscaled points are all at the origin.
	unscaled []Point
	original []Point
	points   []Point
	flags    []uint8
	contours []uint16
}

type graphicsState struct {
	zones [2]zone
	retainedState
	projVector            Point
	dualProjVector        Point
	freedomVector         Point
	roundState            roundState
	fdotp                 int32
	rp0, rp1, rp2         int
	loopCounter           uint32
	projAxis              coordAxis
	dualProjAxis          coordAxis
	freedomAxis           coordAxis
	zp0, zp1, zp2         zonePointer
	isComposite           bool
	backwardCompatibility bool
	didIUPX               bool
	didIUPY               bool
}

// reset puts everything but the retained state, the zones and the composite flag back to its default.
func (gs *graphicsState) reset() {
	vector := Point{X: 0x4000}
	*gs = graphicsState{
		zones:                 gs.zones,
		retainedState:         gs.retainedState,
		isComposite:           gs.isComposite,
		projVector:            vector,
		dualProjVector:        vector,
		freedomVector:         vector,
		fdotp:                 0x4000,
		roundState:            roundState{mode: roundGrid, period: 64},
		loopCounter:           1,
		zp0:                   zoneGlyph,
		zp1:                   zoneGlyph,
		zp2:                   zoneGlyph,
		backwardCompatibility: true,
	}
	gs.updateProjectionState()
}

func (gs *graphicsState) resetRetained() {
	gs.retainedState = newRetainedState(gs.scale, gs.ppem, gs.target)
}

func (gs *graphicsState) unscaledToPixels() int32 {
	if gs.isComposite {
		return 1 << 16
	}
	return gs.scale
}

func (gs *graphicsState) updateProjectionState() {
	const one = 0x4000
	switch {
	case gs.freedomVector.X == one:
		gs.fdotp = gs.projVector.X
	case gs.freedomVector.Y == one:
		gs.fdotp = gs.projVector.Y
	default:
		gs.fdotp = (gs.projVector.X*gs.freedomVector.X + gs.projVector.Y*gs.freedomVector.Y) >> 14
	}
	gs.projAxis = axisOf(gs.projVector)
	gs.dualProjAxis = axisOf(gs.dualProjVector)
	gs.freedomAxis = axisBoth
	if gs.fdotp == one {
		gs.freedomAxis = axisOf(gs.freedomVector)
	}
	// At small sizes, fdotp can become too small resulting in overflows and spikes.
	if abs32(gs.fdotp) < 0x400 {
		gs.fdotp = one
	}
}

func axisOf(v Point) coordAxis {
	switch {
	case v.X == 0x4000:
		return axisX
	case v.Y == 0x4000:
		return axisY
	default:
		return axisBoth
	}
}

func (gs *graphicsState) project(v1, v2 Point) int32 {
	switch gs.projAxis {
	case axisX:
		return v1.X - v2.X
	case axisY:
		return v1.Y - v2.Y
	default:
		return dot14(v1.X-v2.X, v1.Y-v2.Y, gs.projVector.X, gs.projVector.Y)
	}
}

// dualProject serves both scaled and unscaled points: the arithmetic is the same.
func (gs *graphicsState) dualProject(v1, v2 Point) int32 {
	switch gs.dualProjAxis {
	case axisX:
		return v1.X - v2.X
	case axisY:
		return v1.Y - v2.Y
	default:
		return dot14(v1.X-v2.X, v1.Y-v2.Y, gs.dualProjVector.X, gs.dualProjVector.Y)
	}
}

func (gs *graphicsState) round(distance int32) int32 { return gs.roundState.round(distance) }

func (gs *graphicsState) zone(zp zonePointer) *zone { return &gs.zones[zp] }

func (gs *graphicsState) inBounds(zp zonePointer, index int) bool {
	return index >= 0 && index < len(gs.zones[zp].points)
}

// Zone accessors. An index the font got wrong raises a hintError, which ends the program.

func (z *zone) point(index int) Point {
	if index < 0 || index >= len(z.points) {
		fail(errInvalidPointIndex)
	}
	return z.points[index]
}

func (z *zone) pointRef(index int) *Point {
	if index < 0 || index >= len(z.points) {
		fail(errInvalidPointIndex)
	}
	return &z.points[index]
}

func (z *zone) originalPoint(index int) Point {
	if index < 0 || index >= len(z.original) {
		fail(errInvalidPointIndex)
	}
	return z.original[index]
}

func (z *zone) originalRef(index int) *Point {
	if index < 0 || index >= len(z.original) {
		fail(errInvalidPointIndex)
	}
	return &z.original[index]
}

func (z *zone) unscaledPoint(index int) Point {
	if index < 0 || index >= len(z.unscaled) {
		return Point{}
	}
	return z.unscaled[index]
}

func (z *zone) contour(index int) int {
	if index < 0 || index >= len(z.contours) {
		fail(errInvalidContourIndex)
	}
	return int(z.contours[index])
}

func (z *zone) flag(index int) *uint8 {
	if index < 0 || index >= len(z.flags) {
		fail(errInvalidPointIndex)
	}
	return &z.flags[index]
}

func (z *zone) touch(index int, axis coordAxis) { *z.flag(index) |= axis.touchedMarker() }

func (z *zone) untouch(index int, axis coordAxis) { *z.flag(index) &^= axis.touchedMarker() }

func (z *zone) isTouched(index int, axis coordAxis) bool {
	return *z.flag(index)&axis.touchedMarker() != 0
}

func coord(p *Point, axis coordAxis) *int32 {
	if axis == axisX {
		return &p.X
	}
	return &p.Y
}

// iup interpolates the untouched points of every contour between the touched ones, along one axis.
func (z *zone) iup(axis coordAxis) {
	point := 0
	for i := range z.contours {
		endPoint := z.contour(i)
		firstPoint := point
		if endPoint >= len(z.points) {
			endPoint = max(len(z.points)-1, 0)
		}
		for point <= endPoint && !z.isTouched(point, axis) {
			point++
		}
		if point > endPoint {
			continue
		}
		firstTouched := point
		curTouched := point
		point++
		for point <= endPoint {
			if z.isTouched(point, axis) {
				z.iupInterpolate(axis, curTouched+1, point-1, curTouched, point)
				curTouched = point
			}
			point++
		}
		if curTouched == firstTouched {
			z.iupShift(axis, firstPoint, endPoint, curTouched)
		} else {
			z.iupInterpolate(axis, curTouched+1, endPoint, curTouched, firstTouched)
			if firstTouched > 0 {
				z.iupInterpolate(axis, firstPoint, firstTouched-1, curTouched, firstTouched)
			}
		}
	}
}

func (z *zone) iupShift(axis coordAxis, p1, p2, p int) {
	if p1 > p2 || p1 > p || p > p2 {
		return
	}
	pt, org := z.point(p), z.originalPoint(p)
	delta := *coord(&pt, axis) - *coord(&org, axis)
	if delta == 0 {
		return
	}
	if p2 >= len(z.points) {
		fail(errInvalidPointRange)
	}
	for i := p1; i <= p2; i++ {
		if i != p {
			*coord(&z.points[i], axis) += delta
		}
	}
}

func (z *zone) iupInterpolate(axis coordAxis, p1, p2, ref1, ref2 int) {
	if p1 > p2 {
		return
	}
	if ref1 >= len(z.points) || ref2 >= len(z.points) {
		return
	}
	u1, u2 := z.unscaledPoint(ref1), z.unscaledPoint(ref2)
	orus1, orus2 := *coord(&u1, axis), *coord(&u2, axis)
	if orus1 > orus2 {
		orus1, orus2 = orus2, orus1
		ref1, ref2 = ref2, ref1
	}
	o1, o2 := z.originalPoint(ref1), z.originalPoint(ref2)
	c1, c2 := z.point(ref1), z.point(ref2)
	org1, org2 := *coord(&o1, axis), *coord(&o2, axis)
	cur1, cur2 := *coord(&c1, axis), *coord(&c2, axis)
	delta1 := cur1 - org1
	delta2 := cur2 - org2
	if p2 >= len(z.original) || p2 >= len(z.unscaled) || p2 >= len(z.points) {
		fail(errInvalidPointRange)
	}
	if cur1 == cur2 || orus1 == orus2 {
		for i := p1; i <= p2; i++ {
			a := *coord(&z.original[i], axis)
			switch {
			case a <= org1:
				a += delta1
			case a >= org2:
				a += delta2
			default:
				a = cur1
			}
			*coord(&z.points[i], axis) = a
		}
		return
	}
	scale := divFixed(cur2-cur1, orus2-orus1)
	for i := p1; i <= p2; i++ {
		a := *coord(&z.original[i], axis)
		switch {
		case a <= org1:
			a += delta1
		case a >= org2:
			a += delta2
		default:
			a = cur1 + mulFixed(*coord(&z.unscaled[i], axis)-orus1, scale)
		}
		*coord(&z.points[i], axis) = a
	}
}

func (gs *graphicsState) moveOriginal(zp zonePointer, index int, distance int32) {
	point := gs.zone(zp).originalRef(index)
	switch gs.freedomAxis {
	case axisX:
		point.X += distance
	case axisY:
		point.Y += distance
	default:
		if gs.freedomVector.X != 0 {
			point.X += mulDiv(distance, gs.freedomVector.X, gs.fdotp)
		}
		if gs.freedomVector.Y != 0 {
			point.Y += mulDiv(distance, gs.freedomVector.Y, gs.fdotp)
		}
	}
}

// movePoint moves a point along the freedom vector and marks it touched. In backward compatibility mode x is never
// adjusted, and y is not adjusted once IUP has run in both directions: horizontal adjustment is what subpixel
// rendering does better on its own. See https://learn.microsoft.com/en-us/typography/cleartype/truetypecleartype
func (gs *graphicsState) movePoint(zp zonePointer, index int, distance int32) {
	backCompat := gs.backwardCompatibility
	backCompatAndDidIUP := backCompat && gs.didIUPX && gs.didIUPY
	z := gs.zone(zp)
	point := z.pointRef(index)
	switch gs.freedomAxis {
	case axisX:
		if !backCompat {
			point.X += distance
		}
		z.touch(index, axisX)
	case axisY:
		if !backCompatAndDidIUP {
			point.Y += distance
		}
		z.touch(index, axisY)
	default:
		if gs.freedomVector.X != 0 {
			if !backCompat {
				point.X += mulDiv(distance, gs.freedomVector.X, gs.fdotp)
			}
			z.touch(index, axisX)
		}
		if gs.freedomVector.Y != 0 {
			if !backCompatAndDidIUP {
				point.Y += mulDiv(distance, gs.freedomVector.Y, gs.fdotp)
			}
			z.touch(index, axisY)
		}
	}
}

func (gs *graphicsState) moveZP2Point(index int, dx, dy int32, touch bool) {
	backCompat := gs.backwardCompatibility
	backCompatAndDidIUP := backCompat && gs.didIUPX && gs.didIUPY
	z := gs.zone(gs.zp2)
	if gs.freedomVector.X != 0 {
		if !backCompat {
			z.pointRef(index).X += dx
		}
		if touch {
			z.touch(index, axisX)
		}
	}
	if gs.freedomVector.Y != 0 {
		if !backCompatAndDidIUP {
			z.pointRef(index).Y += dy
		}
		if touch {
			z.touch(index, axisY)
		}
	}
}

type pointDisplacement struct {
	index  int
	dx, dy int32
	zone   zonePointer
}

func (gs *graphicsState) pointDisplacement(opcode uint8) pointDisplacement {
	zp, index := gs.zp1, gs.rp2
	if opcode&1 != 0 {
		zp, index = gs.zp0, gs.rp1
	}
	z := gs.zone(zp)
	distance := gs.project(z.point(index), z.originalPoint(index))
	return pointDisplacement{
		zone:  zp,
		index: index,
		dx:    mulDiv(distance, gs.freedomVector.X, gs.fdotp),
		dy:    mulDiv(distance, gs.freedomVector.Y, gs.fdotp),
	}
}
