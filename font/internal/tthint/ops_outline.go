// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

// The instructions that move points.

func (e *engine) opFLIPPT() {
	gs := &e.gs
	count := gs.loopCounter
	gs.loopCounter = 1
	if gs.backwardCompatibility && gs.didIUPX && gs.didIUPY {
		for range count {
			e.pop()
		}
		return
	}
	z := gs.zone(zoneGlyph)
	for range count {
		*z.flag(e.popIndex()) ^= flagOnCurve
	}
}

func (e *engine) setOnCurveForRange(on bool) {
	gs := &e.gs
	high := e.popIndex()
	low := e.popIndex()
	if high == -1 {
		fail(errInvalidPointIndex)
	}
	high++
	if gs.backwardCompatibility && gs.didIUPX && gs.didIUPY {
		return
	}
	flags := gs.zone(zoneGlyph).flags
	if low < 0 || high < 0 || low > high || high > len(flags) {
		fail(errInvalidPointRange)
	}
	for i := low; i < high; i++ {
		if on {
			flags[i] |= flagOnCurve
		} else {
			flags[i] &^= flagOnCurve
		}
	}
}

func (e *engine) opSHP(opcode uint8) {
	gs := &e.gs
	disp := gs.pointDisplacement(opcode)
	count := gs.loopCounter
	gs.loopCounter = 1
	for range count {
		gs.moveZP2Point(e.popIndex(), disp.dx, disp.dy, true)
	}
}

func (e *engine) opSHC(opcode uint8) {
	gs := &e.gs
	contour := e.popIndex()
	z := gs.zone(gs.zp2)
	if contour < 0 || contour >= len(z.contours) {
		return
	}
	disp := gs.pointDisplacement(opcode)
	start := 0
	if contour != 0 {
		start = z.contour(contour-1) + 1
	}
	end := len(z.points)
	if gs.zp2 != zoneTwilight {
		end = z.contour(contour) + 1
	}
	for i := start; i < end; i++ {
		if disp.zone != gs.zp2 || disp.index != i {
			gs.moveZP2Point(i, disp.dx, disp.dy, true)
		}
	}
}

func (e *engine) opSHZ(opcode uint8) {
	gs := &e.gs
	e.popZone()
	disp := gs.pointDisplacement(opcode)
	z := gs.zone(gs.zp2)
	end := 0
	if gs.zp2 == zoneTwilight {
		end = len(z.points)
	} else if len(z.contours) != 0 {
		end = int(z.contours[len(z.contours)-1]) + 1
	}
	for i := range end {
		if disp.zone != gs.zp2 || disp.index != i {
			gs.moveZP2Point(i, disp.dx, disp.dy, false)
		}
	}
}

func (e *engine) opSHPIX() {
	gs := &e.gs
	inTwilight := gs.zp0 == zoneTwilight || gs.zp1 == zoneTwilight || gs.zp2 == zoneTwilight
	amount := e.pop()
	dx := mul14(amount, gs.freedomVector.X)
	dy := mul14(amount, gs.freedomVector.Y)
	count := gs.loopCounter
	gs.loopCounter = 1
	didIUP := gs.didIUPX && gs.didIUPY
	for range count {
		p := e.popIndex()
		if gs.backwardCompatibility {
			if inTwilight ||
				(!didIUP && ((gs.isComposite && gs.freedomVector.Y != 0) || gs.zone(gs.zp2).isTouched(p, axisY))) {
				gs.moveZP2Point(p, dx, dy, true)
			}
		} else {
			gs.moveZP2Point(p, dx, dy, true)
		}
	}
}

func (e *engine) opMSIRP(opcode uint8) {
	gs := &e.gs
	distance := e.pop()
	point := e.popIndex()
	if !gs.inBounds(gs.zp1, point) || !gs.inBounds(gs.zp0, gs.rp0) {
		return
	}
	if gs.zp1 == zoneTwilight {
		*gs.zone(gs.zp1).pointRef(point) = gs.zone(gs.zp0).originalPoint(gs.rp0)
		gs.moveOriginal(gs.zp1, point, distance)
		*gs.zone(gs.zp1).pointRef(point) = gs.zone(gs.zp1).originalPoint(point)
	}
	d := gs.project(gs.zone(gs.zp1).point(point), gs.zone(gs.zp0).point(gs.rp0))
	gs.movePoint(gs.zp1, point, distance-d)
	gs.rp1 = gs.rp0
	gs.rp2 = point
	if opcode&1 != 0 {
		gs.rp0 = point
	}
}

func (e *engine) opMDAP(opcode uint8) {
	gs := &e.gs
	p := e.popIndex()
	if !gs.inBounds(gs.zp0, p) {
		gs.rp0 = p
		gs.rp1 = p
		return
	}
	var distance int32
	if opcode&1 != 0 {
		cur := gs.project(gs.zone(gs.zp0).point(p), Point{})
		distance = gs.round(cur) - cur
	}
	gs.movePoint(gs.zp0, p, distance)
	gs.rp0 = p
	gs.rp1 = p
}

func (e *engine) opMIAP(opcode uint8) {
	gs := &e.gs
	cvtEntry := e.popIndex()
	point := e.popIndex()
	distance := e.cvtGet(cvtEntry)
	if gs.zp0 == zoneTwilight {
		z := gs.zone(gs.zp0)
		original := z.originalRef(point)
		original.X = mul14(distance, gs.freedomVector.X)
		original.Y = mul14(distance, gs.freedomVector.Y)
		*z.pointRef(point) = *original
	}
	originalDistance := gs.project(gs.zone(gs.zp0).point(point), Point{})
	if opcode&1 != 0 {
		if abs32(distance-originalDistance) > gs.controlValueCutin {
			distance = originalDistance
		}
		distance = gs.round(distance)
	}
	gs.movePoint(gs.zp0, point, distance-originalDistance)
	gs.rp0 = point
	gs.rp1 = point
}

func (e *engine) opMDRP(opcode uint8) {
	gs := &e.gs
	p := e.popIndex()
	if !gs.inBounds(gs.zp1, p) || !gs.inBounds(gs.zp0, gs.rp0) {
		gs.rp1 = gs.rp0
		gs.rp2 = p
		if opcode&16 != 0 {
			gs.rp0 = p
		}
		return
	}
	z0, z1 := gs.zone(gs.zp0), gs.zone(gs.zp1)
	var originalDistance int32
	if gs.zp0 == zoneTwilight || gs.zp1 == zoneTwilight {
		originalDistance = gs.dualProject(z1.originalPoint(p), z0.originalPoint(gs.rp0))
	} else {
		dist := gs.dualProject(z1.unscaledPoint(p), z0.unscaledPoint(gs.rp0))
		originalDistance = mulFixed(dist, gs.unscaledToPixels())
	}
	cutin := gs.singleWidthCutin
	value := gs.singleWidth
	if cutin > 0 && originalDistance < value+cutin && originalDistance > value-cutin {
		if originalDistance >= 0 {
			originalDistance = value
		} else {
			originalDistance = -value
		}
	}
	distance := originalDistance
	if opcode&4 != 0 {
		distance = gs.round(originalDistance)
	}
	if opcode&8 != 0 {
		if originalDistance >= 0 {
			distance = max(distance, gs.minDistance)
		} else {
			distance = min(distance, -gs.minDistance)
		}
	}
	originalDistance = gs.project(z1.point(p), z0.point(gs.rp0))
	gs.movePoint(gs.zp1, p, distance-originalDistance)
	gs.rp1 = gs.rp0
	gs.rp2 = p
	if opcode&16 != 0 {
		gs.rp0 = p
	}
}

func (e *engine) opMIRP(opcode uint8) {
	gs := &e.gs
	n := int(e.pop() + 1)
	p := e.popIndex()
	if !gs.inBounds(gs.zp1, p) || !gs.inBounds(gs.zp0, gs.rp0) || n < 0 || n > len(e.cvt) {
		gs.rp1 = gs.rp0
		if opcode&16 != 0 {
			gs.rp0 = p
		}
		gs.rp2 = p
		return
	}
	var cvtDistance int32
	if n != 0 {
		cvtDistance = e.cvtGet(n - 1)
	}
	cutin := gs.singleWidthCutin
	value := gs.singleWidth
	if abs32(cvtDistance-value) < cutin {
		if cvtDistance >= 0 {
			cvtDistance = value
		} else {
			cvtDistance = -value
		}
	}
	z0, z1 := gs.zone(gs.zp0), gs.zone(gs.zp1)
	if gs.zp1 == zoneTwilight {
		p2 := z0.originalPoint(gs.rp0)
		p1 := z1.originalRef(p)
		p1.X = p2.X + mulFixed(cvtDistance, gs.freedomVector.X)
		p1.Y = p2.Y + mulFixed(cvtDistance, gs.freedomVector.Y)
		*z1.pointRef(p) = *p1
	}
	originalDistance := gs.dualProject(z1.originalPoint(p), z0.originalPoint(gs.rp0))
	currentDistance := gs.project(z1.point(p), z0.point(gs.rp0))
	if gs.autoFlip && (originalDistance^cvtDistance) < 0 {
		cvtDistance = -cvtDistance
	}
	distance := cvtDistance
	if opcode&4 != 0 {
		if gs.zp0 == gs.zp1 && abs32(cvtDistance-originalDistance) > gs.controlValueCutin {
			cvtDistance = originalDistance
		}
		distance = gs.round(cvtDistance)
	}
	if opcode&8 != 0 {
		if originalDistance >= 0 {
			distance = max(distance, gs.minDistance)
		} else {
			distance = min(distance, -gs.minDistance)
		}
	}
	gs.movePoint(gs.zp1, p, distance-currentDistance)
	gs.rp1 = gs.rp0
	if opcode&16 != 0 {
		gs.rp0 = p
	}
	gs.rp2 = p
}

func (e *engine) opISECT() {
	gs := &e.gs
	b1 := e.popIndex()
	b0 := e.popIndex()
	a1 := e.popIndex()
	a0 := e.popIndex()
	point := e.popIndex()
	pa0, pa1 := gs.zone(gs.zp1).point(a0), gs.zone(gs.zp1).point(a1)
	pb0, pb1 := gs.zone(gs.zp0).point(b0), gs.zone(gs.zp0).point(b1)
	dbx := pb1.X - pb0.X
	dby := pb1.Y - pb0.Y
	dax := pa1.X - pa0.X
	day := pa1.Y - pa0.Y
	dx := pb0.X - pa0.X
	dy := pb0.Y - pa0.Y
	discriminant := mulDiv(dax, -dby, 0x40) + mulDiv(day, dbx, 0x40)
	dotProduct := mulDiv(dax, dbx, 0x40) + mulDiv(day, dby, 0x40)
	target := gs.zone(gs.zp2).pointRef(point)
	if abs32(discriminant)*19 > abs32(dotProduct) {
		v := mulDiv(dx, -dby, 0x40) + mulDiv(dy, dbx, 0x40)
		target.X = pa0.X + mulDiv(v, dax, discriminant)
		target.Y = pa0.Y + mulDiv(v, day, discriminant)
	} else {
		// The lines are nearly parallel, so take the middle of the four points.
		target.X = (pa0.X + pa1.X + pb0.X + pb1.X) / 4
		target.Y = (pa0.Y + pa1.Y + pb0.Y + pb1.Y) / 4
	}
	gs.zone(gs.zp2).touch(point, axisBoth)
}

func (e *engine) opIP() {
	gs := &e.gs
	count := gs.loopCounter
	gs.loopCounter = 1
	if !gs.inBounds(gs.zp0, gs.rp1) || !gs.inBounds(gs.zp1, gs.rp2) {
		return
	}
	inTwilight := gs.zp0 == zoneTwilight || gs.zp1 == zoneTwilight || gs.zp2 == zoneTwilight
	z0, z1 := gs.zone(gs.zp0), gs.zone(gs.zp1)
	var orusBase Point
	var oldRange int32
	if inTwilight {
		orusBase = z0.originalPoint(gs.rp1)
		oldRange = gs.dualProject(z1.originalPoint(gs.rp2), orusBase)
	} else {
		orusBase = z0.unscaledPoint(gs.rp1)
		oldRange = gs.dualProject(z1.unscaledPoint(gs.rp2), orusBase)
	}
	curBase := z0.point(gs.rp1)
	curRange := gs.project(z1.point(gs.rp2), curBase)
	for range count {
		point := e.popIndex()
		if !gs.inBounds(gs.zp2, point) {
			continue
		}
		z2 := gs.zone(gs.zp2)
		var originalDistance int32
		if inTwilight {
			originalDistance = gs.dualProject(z2.originalPoint(point), orusBase)
		} else {
			originalDistance = gs.dualProject(z2.unscaledPoint(point), orusBase)
		}
		curDistance := gs.project(z2.point(point), curBase)
		var newDistance int32
		if originalDistance != 0 {
			if oldRange != 0 {
				newDistance = mulDiv(originalDistance, curRange, oldRange)
			} else {
				newDistance = originalDistance
			}
		}
		gs.movePoint(gs.zp2, point, newDistance-curDistance)
	}
}

func (e *engine) opIUP(opcode uint8) {
	gs := &e.gs
	axis := axisY
	if opcode&1 != 0 {
		axis = axisX
	}
	run := true
	if gs.backwardCompatibility {
		if gs.didIUPX && gs.didIUPY {
			run = false
		}
		if axis == axisX {
			gs.didIUPX = true
		} else {
			gs.didIUPY = true
		}
	}
	if run {
		gs.zone(zoneGlyph).iup(axis)
	}
}
