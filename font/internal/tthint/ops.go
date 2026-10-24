// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

// The TrueType instruction set. Instructions that come in numbered variants are named for the first.
const (
	opSVTCA0    = 0x00
	opSPVTL0    = 0x06
	opSPVFS     = 0x0A
	opSFVFS     = 0x0B
	opGPV       = 0x0C
	opGFV       = 0x0D
	opSFVTPV    = 0x0E
	opISECT     = 0x0F
	opSRP0      = 0x10
	opSRP1      = 0x11
	opSRP2      = 0x12
	opSZP0      = 0x13
	opSZP1      = 0x14
	opSZP2      = 0x15
	opSZPS      = 0x16
	opSLOOP     = 0x17
	opRTG       = 0x18
	opRTHG      = 0x19
	opSMD       = 0x1A
	opELSE      = 0x1B
	opJMPR      = 0x1C
	opSCVTCI    = 0x1D
	opSSWCI     = 0x1E
	opSSW       = 0x1F
	opDUP       = 0x20
	opPOP       = 0x21
	opCLEAR     = 0x22
	opSWAP      = 0x23
	opDEPTH     = 0x24
	opCINDEX    = 0x25
	opMINDEX    = 0x26
	opALIGNPTS  = 0x27
	opUTP       = 0x29
	opLOOPCALL  = 0x2A
	opCALL      = 0x2B
	opFDEF      = 0x2C
	opENDF      = 0x2D
	opMDAP0     = 0x2E
	opIUP0      = 0x30
	opSHP0      = 0x32
	opSHC0      = 0x34
	opSHZ0      = 0x36
	opSHPIX     = 0x38
	opIP        = 0x39
	opMSIRP0    = 0x3A
	opALIGNRP   = 0x3C
	opRTDG      = 0x3D
	opMIAP0     = 0x3E
	opNPUSHB    = 0x40
	opNPUSHW    = 0x41
	opWS        = 0x42
	opRS        = 0x43
	opWCVTP     = 0x44
	opRCVT      = 0x45
	opGC0       = 0x46
	opSCFS      = 0x48
	opMD0       = 0x49
	opMPPEM     = 0x4B
	opMPS       = 0x4C
	opFLIPON    = 0x4D
	opFLIPOFF   = 0x4E
	opDEBUG     = 0x4F
	opLT        = 0x50
	opLTEQ      = 0x51
	opGT        = 0x52
	opGTEQ      = 0x53
	opEQ        = 0x54
	opNEQ       = 0x55
	opODD       = 0x56
	opEVEN      = 0x57
	opIF        = 0x58
	opEIF       = 0x59
	opAND       = 0x5A
	opOR        = 0x5B
	opNOT       = 0x5C
	opDELTAP1   = 0x5D
	opSDB       = 0x5E
	opSDS       = 0x5F
	opADD       = 0x60
	opSUB       = 0x61
	opDIV       = 0x62
	opMUL       = 0x63
	opABS       = 0x64
	opNEG       = 0x65
	opFLOOR     = 0x66
	opCEILING   = 0x67
	opROUND0    = 0x68
	opNROUND0   = 0x6C
	opWCVTF     = 0x70
	opDELTAP2   = 0x71
	opDELTAP3   = 0x72
	opDELTAC1   = 0x73
	opDELTAC2   = 0x74
	opDELTAC3   = 0x75
	opSROUND    = 0x76
	opS45ROUND  = 0x77
	opJROT      = 0x78
	opJROF      = 0x79
	opROFF      = 0x7A
	opRUTG      = 0x7C
	opRDTG      = 0x7D
	opSANGW     = 0x7E
	opAA        = 0x7F
	opFLIPPT    = 0x80
	opFLIPRGON  = 0x81
	opFLIPRGOFF = 0x82
	opSCANCTRL  = 0x85
	opSDPVTL0   = 0x86
	opGETINFO   = 0x88
	opIDEF      = 0x89
	opROLL      = 0x8A
	opMAX       = 0x8B
	opMIN       = 0x8C
	opSCANTYPE  = 0x8D
	opINSTCTRL  = 0x8E
	opPUSHB0    = 0xB0
	opPUSHW0    = 0xB8
	opMDRP0     = 0xC0
	opMIRP0     = 0xE0
)

func b2i(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

func (e *engine) dispatch(ins *instruction) {
	gs := &e.gs
	op := ins.opcode
	switch op {
	case 0x00, 0x01, 0x02, 0x03, 0x04, 0x05:
		e.opSVTCA(op)
	case 0x06, 0x07, 0x08, 0x09:
		e.opSVTL(op)
	case opSPVFS:
		y := int32(int16(e.pop()))
		x := int32(int16(e.pop()))
		vector := gs.projVector
		if x != 0 || y != 0 {
			vector = normalize14(x, y)
		}
		gs.projVector = vector
		gs.dualProjVector = vector
		gs.updateProjectionState()
	case opSFVFS:
		y := int32(int16(e.pop()))
		x := int32(int16(e.pop()))
		if x != 0 || y != 0 {
			gs.freedomVector = normalize14(x, y)
		}
		gs.updateProjectionState()
	case opGPV:
		e.push(gs.projVector.X)
		e.push(gs.projVector.Y)
	case opGFV:
		e.push(gs.freedomVector.X)
		e.push(gs.freedomVector.Y)
	case opSFVTPV:
		gs.freedomVector = gs.projVector
		gs.updateProjectionState()
	case opISECT:
		e.opISECT()
	case opSRP0:
		gs.rp0 = e.popIndex()
	case opSRP1:
		gs.rp1 = e.popIndex()
	case opSRP2:
		gs.rp2 = e.popIndex()
	case opSZP0:
		gs.zp0 = e.popZone()
	case opSZP1:
		gs.zp1 = e.popZone()
	case opSZP2:
		gs.zp2 = e.popZone()
	case opSZPS:
		zp := e.popZone()
		gs.zp0, gs.zp1, gs.zp2 = zp, zp, zp
	case opSLOOP:
		n := e.pop()
		if n < 0 {
			fail(errNegativeLoopCounter)
		}
		gs.loopCounter = min(uint32(n), 0xFFFF)
	case opRTG:
		gs.roundState.mode = roundGrid
	case opRTHG:
		gs.roundState.mode = roundHalfGrid
	case opSMD:
		gs.minDistance = e.pop()
	case opELSE:
		e.opElse()
	case opJMPR:
		e.doJump(true)
	case opSCVTCI:
		gs.controlValueCutin = e.pop()
	case opSSWCI:
		gs.singleWidthCutin = e.pop()
	case opSSW:
		gs.singleWidth = mulFixed(e.pop(), gs.scale)
	case opDUP:
		if e.sp > 0 {
			e.push(e.stack[e.sp-1])
		} else {
			e.push(0)
		}
	case opPOP:
		e.pop()
	case opCLEAR:
		e.sp = 0
	case opSWAP:
		a := e.pop()
		b := e.pop()
		e.push(a)
		e.push(b)
	case opDEPTH:
		e.push(int32(e.sp))
	case opCINDEX:
		top := e.sp - 1
		if top < 0 {
			fail(errValueStackUnderflow)
		}
		index := int(e.stack[top])
		if index < 0 || index > top {
			fail(errValueStackUnderflow)
		}
		e.stack[top] = e.stack[top-index]
	case opMINDEX:
		top := e.sp - 1
		if top < 1 {
			fail(errValueStackUnderflow)
		}
		index := int(e.stack[top])
		if index < 0 || index > top {
			fail(errValueStackUnderflow)
		}
		element := top - index
		value := e.stack[element]
		copy(e.stack[element:], e.stack[element+1:e.sp])
		e.stack[top-1] = value
		e.sp--
	case opALIGNPTS:
		p2 := e.popIndex()
		p1 := e.popIndex()
		distance := gs.project(gs.zone(gs.zp0).point(p2), gs.zone(gs.zp1).point(p1)) / 2
		gs.movePoint(gs.zp1, p1, distance)
		gs.movePoint(gs.zp0, p2, -distance)
	case opUTP:
		p := e.popIndex()
		switch {
		case gs.freedomVector.X != 0 && gs.freedomVector.Y != 0:
			gs.zone(gs.zp0).untouch(p, axisBoth)
		case gs.freedomVector.X != 0:
			gs.zone(gs.zp0).untouch(p, axisX)
		case gs.freedomVector.Y != 0:
			gs.zone(gs.zp0).untouch(p, axisY)
		}
	case opLOOPCALL:
		e.opLoopCall()
	case opCALL:
		e.callFunction(1, e.pop())
	case opFDEF:
		e.doDef(e.functions, e.pop())
	case opENDF:
		e.leave()
	case opMDAP0, opMDAP0 + 1:
		e.opMDAP(op)
	case opIUP0, opIUP0 + 1:
		e.opIUP(op)
	case opSHP0, opSHP0 + 1:
		e.opSHP(op)
	case opSHC0, opSHC0 + 1:
		e.opSHC(op)
	case opSHZ0, opSHZ0 + 1:
		e.opSHZ(op)
	case opSHPIX:
		e.opSHPIX()
	case opIP:
		e.opIP()
	case opMSIRP0, opMSIRP0 + 1:
		e.opMSIRP(op)
	case opALIGNRP:
		count := gs.loopCounter
		gs.loopCounter = 1
		for range count {
			p := e.popIndex()
			distance := gs.project(gs.zone(gs.zp1).point(p), gs.zone(gs.zp0).point(gs.rp0))
			gs.movePoint(gs.zp1, p, -distance)
		}
	case opRTDG:
		gs.roundState.mode = roundDoubleGrid
	case opMIAP0, opMIAP0 + 1:
		e.opMIAP(op)
	case opNPUSHB, opNPUSHW:
		e.pushInline(ins)
	case opWS:
		value := e.pop()
		e.storageSet(e.popIndex(), value)
	case opRS:
		location := e.popIndex()
		var value int32
		if location >= 0 && location < len(e.storage) {
			value = e.storage[location]
		}
		e.push(value)
	case opWCVTP:
		value := e.pop()
		e.cvtSet(e.popIndex(), value)
	case opRCVT:
		location := e.popIndex()
		var value int32
		if location >= 0 && location < len(e.cvt) {
			value = e.cvt[location]
		}
		e.push(value)
	case opGC0, opGC0 + 1:
		p := e.popIndex()
		switch {
		case !gs.inBounds(gs.zp2, p):
			e.push(0)
		case op&1 != 0:
			e.push(gs.dualProject(gs.zone(gs.zp2).originalPoint(p), Point{}))
		default:
			e.push(gs.project(gs.zone(gs.zp2).point(p), Point{}))
		}
	case opSCFS:
		value := e.pop()
		p := e.popIndex()
		projection := gs.project(gs.zone(gs.zp2).point(p), Point{})
		gs.movePoint(gs.zp2, p, value-projection)
		if gs.zp2 == zoneTwilight {
			twilight := gs.zone(zoneTwilight)
			*twilight.originalRef(p) = twilight.point(p)
		}
	case opMD0, opMD0 + 1:
		e.opMD(op)
	case opMPPEM:
		e.push(gs.ppem)
	case opMPS:
		// FreeType computes ppem * 64 * 72 / resolution, and the resolution here is always 72.
		e.push(int32(min(int64(gs.ppem)*64, 0x7FFFFFFF)))
	case opFLIPON:
		gs.autoFlip = true
	case opFLIPOFF:
		gs.autoFlip = false
	case opDEBUG:
		e.pop()
	case opLT:
		b, a := e.pop(), e.pop()
		e.push(b2i(a < b))
	case opLTEQ:
		b, a := e.pop(), e.pop()
		e.push(b2i(a <= b))
	case opGT:
		b, a := e.pop(), e.pop()
		e.push(b2i(a > b))
	case opGTEQ:
		b, a := e.pop(), e.pop()
		e.push(b2i(a >= b))
	case opEQ:
		b, a := e.pop(), e.pop()
		e.push(b2i(a == b))
	case opNEQ:
		b, a := e.pop(), e.pop()
		e.push(b2i(a != b))
	case opODD:
		e.push(b2i(gs.round(e.pop())&64 == 64))
	case opEVEN:
		e.push(b2i(gs.round(e.pop())&64 == 0))
	case opIF:
		e.opIf()
	case opEIF:
	case opAND:
		b, a := e.pop(), e.pop()
		e.push(b2i(a != 0 && b != 0))
	case opOR:
		b, a := e.pop(), e.pop()
		e.push(b2i(a != 0 || b != 0))
	case opNOT:
		e.push(b2i(e.pop() == 0))
	case opDELTAP1, opDELTAP2, opDELTAP3:
		e.opDELTAP(op)
	case opSDB:
		gs.deltaBase = uint16(e.pop())
	case opSDS:
		n := e.pop()
		if uint32(n) > 6 {
			fail(errInvalidStackValue)
		}
		gs.deltaShift = uint16(n)
	case opADD:
		b, a := e.pop(), e.pop()
		e.push(a + b)
	case opSUB:
		b, a := e.pop(), e.pop()
		e.push(a - b)
	case opDIV:
		b, a := e.pop(), e.pop()
		if b == 0 {
			fail(errDivideByZero)
		}
		e.push(mulDivNoRound(a, 64, b))
	case opMUL:
		b, a := e.pop(), e.pop()
		e.push(mulDiv(a, b, 64))
	case opABS:
		e.push(abs32(e.pop()))
	case opNEG:
		e.push(-e.pop())
	case opFLOOR:
		e.push(floor26(e.pop()))
	case opCEILING:
		e.push(ceil26(e.pop()))
	case opROUND0, opROUND0 + 1, opROUND0 + 2, opROUND0 + 3:
		e.push(gs.round(e.pop()))
	case opNROUND0, opNROUND0 + 1, opNROUND0 + 2, opNROUND0 + 3:
		// "No round" does nothing.
	case opWCVTF:
		value := e.pop()
		e.cvtSet(e.popIndex(), mulFixed(value, gs.scale))
	case opDELTAC1, opDELTAC2, opDELTAC3:
		e.opDELTAC(op)
	case opSROUND:
		e.superRound(0x4000, e.pop())
		gs.roundState.mode = roundSuper
	case opS45ROUND:
		e.superRound(0x2D41, e.pop())
		gs.roundState.mode = roundSuper45
	case opJROT:
		e.doJump(e.pop() != 0)
	case opJROF:
		e.doJump(e.pop() == 0)
	case opROFF:
		gs.roundState.mode = roundOff
	case opRUTG:
		gs.roundState.mode = roundUpToGrid
	case opRDTG:
		gs.roundState.mode = roundDownToGrid
	case opSANGW:
		e.pop()
	case opAA:
		// Unsupported everywhere; does nothing.
	case opFLIPPT:
		e.opFLIPPT()
	case opFLIPRGON:
		e.setOnCurveForRange(true)
	case opFLIPRGOFF:
		e.setOnCurveForRange(false)
	case opSCANCTRL:
		e.opSCANCTRL()
	case opSDPVTL0, opSDPVTL0 + 1:
		index1 := e.popIndex()
		index2 := e.popIndex()
		parallel := op&1 == 0
		gs.dualProjVector = lineVector(gs.zone(gs.zp1).originalPoint(index2), gs.zone(gs.zp2).originalPoint(index1),
			parallel)
		gs.projVector = lineVector(gs.zone(gs.zp1).point(index2), gs.zone(gs.zp2).point(index1), parallel)
		gs.updateProjectionState()
	case opGETINFO:
		e.opGETINFO()
	case opIDEF:
		e.doDef(e.instructions, e.pop())
	case opROLL:
		a := e.pop()
		b := e.pop()
		c := e.pop()
		e.push(b)
		e.push(a)
		e.push(c)
	case opMAX:
		b, a := e.pop(), e.pop()
		e.push(max(a, b))
	case opMIN:
		b, a := e.pop(), e.pop()
		e.push(min(a, b))
	case opSCANTYPE:
		gs.scanType = e.pop() & 0xFFFF
	case opINSTCTRL:
		e.opINSTCTRL()
	default:
		// GETVARIATION and GETDATA land here with everything else undefined: this package does not hint variable
		// fonts, and for any other font they are whatever the font defines them to be.
		switch {
		case op >= opMIRP0:
			e.opMIRP(op)
		case op >= opMDRP0:
			e.opMDRP(op)
		case op >= opPUSHB0:
			e.pushInline(ins)
		default:
			e.opUnknown(op)
		}
	}
}

func (e *engine) popZone() zonePointer {
	switch e.pop() {
	case 0:
		return zoneTwilight
	case 1:
		return zoneGlyph
	default:
		fail(errInvalidZoneIndex)
		return zoneGlyph
	}
}

func (e *engine) opSVTCA(opcode uint8) {
	gs := &e.gs
	x := int32(opcode&1) << 14
	y := x ^ 0x4000
	if opcode < 4 {
		gs.projVector = Point{X: x, Y: y}
		gs.dualProjVector = gs.projVector
	}
	if opcode&2 == 0 {
		gs.freedomVector = Point{X: x, Y: y}
	}
	gs.updateProjectionState()
}

func (e *engine) opSVTL(opcode uint8) {
	gs := &e.gs
	index1 := e.popIndex()
	index2 := e.popIndex()
	p1 := gs.zone(gs.zp1).point(index2)
	p2 := gs.zone(gs.zp2).point(index1)
	vector := lineVector(p1, p2, opcode&1 == 0)
	if opcode < 8 {
		gs.projVector = vector
		gs.dualProjVector = vector
	} else {
		gs.freedomVector = vector
	}
	gs.updateProjectionState()
}

func lineVector(p1, p2 Point, parallel bool) Point {
	a := p1.X - p2.X
	b := p1.Y - p2.Y
	if a == 0 && b == 0 {
		a = 0x4000
	} else if !parallel {
		a, b = -b, a
	}
	return normalize14(a, b)
}

func (e *engine) superRound(gridPeriod, selector int32) {
	rs := &e.gs.roundState
	var period int32
	switch selector & 0xC0 {
	case 0:
		period = gridPeriod / 2
	case 0x40:
		period = gridPeriod
	case 0x80:
		period = gridPeriod * 2
	default:
		period = gridPeriod
	}
	var phase int32
	switch selector & 0x30 {
	case 0:
		phase = 0
	case 0x10:
		phase = period / 4
	case 0x20:
		phase = period / 2
	default:
		phase = period * 3 / 4
	}
	threshold := period - 1
	if selector&0x0F != 0 {
		threshold = ((selector & 0x0F) - 4) * period / 8
	}
	rs.period = period >> 8
	rs.phase = phase >> 8
	rs.threshold = threshold >> 8
}

func (e *engine) opINSTCTRL() {
	gs := &e.gs
	selector := uint32(e.pop())
	value := uint32(e.pop())
	if selector < 1 || selector > 3 {
		return
	}
	selectorFlag := uint32(1) << (selector - 1)
	if value != 0 && value != selectorFlag {
		return
	}
	switch {
	case e.initial == programControlValue:
		gs.instructControl &^= uint8(selectorFlag)
		gs.instructControl |= uint8(value)
	case e.initial == programGlyph && selector == 3:
		gs.backwardCompatibility = value != 4
	}
}

func (e *engine) opSCANCTRL() {
	gs := &e.gs
	n := e.pop()
	threshold := n & 0xFF
	switch threshold {
	case 0xFF:
		gs.scanControl = true
	case 0:
		gs.scanControl = false
	default:
		if n&0x100 != 0 && gs.ppem <= threshold {
			gs.scanControl = true
		}
		if n&0x200 != 0 && gs.isRotated {
			gs.scanControl = true
		}
		if n&0x400 != 0 && gs.isStretched {
			gs.scanControl = true
		}
		if n&0x800 != 0 && gs.ppem > threshold {
			gs.scanControl = false
		}
		if n&0x1000 != 0 && gs.isRotated {
			gs.scanControl = false
		}
		if n&0x2000 != 0 && gs.isStretched {
			gs.scanControl = false
		}
	}
}

func (e *engine) opGETINFO() {
	gs := &e.gs
	selector := e.pop()
	var result int32
	// Interpreter version (selector bit 0, result bits 0-7).
	if selector&(1<<0) != 0 {
		result = 40
	}
	// Glyph rotated (selector bit 1, result bit 8).
	if selector&(1<<1) != 0 && gs.isRotated {
		result |= 1 << 8
	}
	// Glyph stretched (selector bit 2, result bit 9).
	if selector&(1<<2) != 0 && gs.isStretched {
		result |= 1 << 9
	}
	// Font variations (selector bit 3, result bit 10) are never reported: see the package comment.
	if gs.target.isSmooth() {
		// Subpixel hinting, i.e. ClearType enabled (selector bit 6, result bit 13). Always.
		if selector&(1<<6) != 0 {
			result |= 1 << 13
		}
		// Vertical LCD subpixels (selector bit 8, result bit 15).
		if selector&(1<<8) != 0 && gs.target.isVerticalLCD() {
			result |= 1 << 15
		}
		// Subpixel positioned (selector bit 10, result bit 17). Always.
		if selector&(1<<10) != 0 {
			result |= 1 << 17
		}
		// Symmetrical smoothing (selector bit 11, result bit 18). Always, as in FreeType.
		if selector&(1<<11) != 0 {
			result |= 1 << 18
		}
		// ClearType hinting with grayscale rendering (selector bit 12, result bit 19).
		if selector&(1<<12) != 0 && gs.target.isGrayscaleClearType() {
			result |= 1 << 19
		}
	}
	e.push(result)
}

func (e *engine) opMD(opcode uint8) {
	gs := &e.gs
	p1 := e.popIndex()
	p2 := e.popIndex()
	if !gs.inBounds(gs.zp0, p2) || !gs.inBounds(gs.zp1, p1) {
		e.push(0)
		return
	}
	z0, z1 := gs.zone(gs.zp0), gs.zone(gs.zp1)
	switch {
	case opcode&1 != 0:
		// Measured in the grid-fitted outline.
		e.push(gs.project(z0.point(p2), z1.point(p1)))
	case gs.zp0 == zoneTwilight || gs.zp1 == zoneTwilight:
		e.push(gs.dualProject(z0.originalPoint(p2), z1.originalPoint(p1)))
	default:
		// Measured in the original unscaled outline.
		e.push(mulFixed(gs.dualProject(z0.unscaledPoint(p2), z1.unscaledPoint(p1)), gs.unscaledToPixels()))
	}
}

// deltaStep decodes the magnitude nibble of a DELTA exception into a 26.6 distance.
func (e *engine) deltaStep(b int32) int32 {
	b = (b & 0xF) - 8
	if b >= 0 {
		b++
	}
	return b * (1 << (6 - int32(e.gs.deltaShift)))
}

func (e *engine) opDELTAP(opcode uint8) {
	gs := &e.gs
	ppem := uint32(gs.ppem)
	pointCount := len(gs.zone(gs.zp0).points)
	// Each exception takes two values, so the count is limited to what the stack holds: popping an empty stack
	// yields zeroes rather than an error, and would otherwise loop for as long as the font asked.
	n := e.popCount()
	n = min(n, e.sp/2)
	bias := uint32(gs.deltaBase)
	switch opcode {
	case opDELTAP2:
		bias += 16
	case opDELTAP3:
		bias += 32
	}
	backCompat := gs.backwardCompatibility
	didIUP := gs.didIUPX && gs.didIUPY
	for range n {
		point := e.popIndex()
		b := e.pop()
		// Some popular fonts contain invalid DELTAP instructions, so out of bounds points are ignored.
		if point < 0 || point >= pointCount {
			continue
		}
		if ppem != (uint32(b)&0xF0)>>4+bias {
			continue
		}
		distance := e.deltaStep(b)
		if backCompat {
			if !didIUP && ((gs.isComposite && gs.freedomVector.Y != 0) || gs.zone(gs.zp0).isTouched(point, axisY)) {
				gs.movePoint(gs.zp0, point, distance)
			}
		} else {
			gs.movePoint(gs.zp0, point, distance)
		}
	}
}

func (e *engine) opDELTAC(opcode uint8) {
	gs := &e.gs
	ppem := uint32(gs.ppem)
	n := e.popCount()
	n = min(n, e.sp/2)
	bias := uint32(gs.deltaBase)
	switch opcode {
	case opDELTAC2:
		bias += 16
	case opDELTAC3:
		bias += 32
	}
	for range n {
		index := e.popIndex()
		b := e.pop()
		if ppem != (uint32(b)&0xF0)>>4+bias {
			continue
		}
		value := e.cvtGet(index)
		if !e.cvtSet(index, value+e.deltaStep(b)) {
			fail(errInvalidCvtIndex)
		}
	}
}
