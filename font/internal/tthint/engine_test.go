// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

import (
	"bytes"
	"errors"
	"slices"
	"testing"
)

// testEngine returns an engine over a glyph zone holding the given points and a twilight zone of four, with one
// control value per entry of cvt, ready to run program as a glyph program for a smooth target.
func testEngine(points []Point, cvt []int32, program []byte) *engine {
	e := &engine{
		bytecode: [3][]byte{nil, nil, program},
		cvt:      cvt,
		stack:    make([]int32, 64),
		cvtLen:   len(cvt),
	}
	e.gs.retainedState = newRetainedState(1<<16, 16, TargetLCD)
	e.gs.zones[zoneTwilight] = zone{
		original: make([]Point, 4),
		points:   make([]Point, 4),
		flags:    make([]uint8, 4),
		contours: []uint16{4},
	}
	e.gs.zones[zoneGlyph] = zone{
		unscaled: slices.Clone(points),
		original: slices.Clone(points),
		points:   points,
		flags:    make([]uint8, len(points)),
		contours: []uint16{uint16(len(points))},
	}
	e.setLoopLimit()
	return e
}

// TestMIRPPlacesATwilightPointAlongTheFreedomVector checks that MIRP into the twilight zone puts the point's original
// position at the control value distance along the freedom vector, which is a 2.14 vector and not a 16.16 one.
func TestMIRPPlacesATwilightPointAlongTheFreedomVector(t *testing.T) {
	e := testEngine(make([]Point, 1), []int32{0, 128}, []byte{
		opPUSHB0, 0, opSZPS, // every zone pointer to the twilight zone
		0x00,               // SVTCA[y]
		opPUSHB0 + 1, 1, 1, // point 1, control value 1
		opMIRP0 + 4, // MIRP, rounding
	})
	if err := e.runProgram(programGlyph); err != nil {
		t.Fatal(err)
	}
	twilight := e.gs.zone(zoneTwilight)
	want := Point{Y: 128}
	if got := twilight.original[1]; got != want {
		t.Errorf("original position %v, want %v", got, want)
	}
	if got := twilight.points[1]; got != want {
		t.Errorf("position %v, want %v", got, want)
	}
}

// TestMoveAlongAHorizontalProjection checks the one move backward compatibility lets through in both directions: along
// a diagonal freedom vector, measured along a horizontal projection vector. With the projection vector off the
// horizontal the same move is suppressed.
func TestMoveAlongAHorizontalProjection(t *testing.T) {
	for _, horizontal := range []bool{true, false} {
		e := testEngine([]Point{{X: 100, Y: 50}}, nil, nil)
		gs := &e.gs
		gs.reset()
		gs.freedomVector = normalize14(1, 1)
		gs.projVector = normalize14(-1, 0)
		if !horizontal {
			gs.projVector = normalize14(-2, 1)
		}
		gs.updateProjectionState()
		gs.didIUPX, gs.didIUPY = true, true
		gs.movePoint(zoneGlyph, 0, 10)
		moved := gs.zone(zoneGlyph).points[0] != Point{X: 100, Y: 50}
		if moved != horizontal {
			t.Errorf("horizontal projection %v: point moved %v", horizontal, moved)
		}
	}
}

// TestLoopIterationsCountAgainstTheBudget checks that a program cannot multiply its work by the loop counter: the
// iterations of a repeating instruction are charged like instructions, so asking for more of them than the budget of a
// run ends it.
func TestLoopIterationsCountAgainstTheBudget(t *testing.T) {
	// SLOOP 32767; ALIGNRP on an empty stack, which aligns point 0 with itself each time round.
	once := []byte{opPUSHW0, 0x7F, 0xFF, opSLOOP, opALIGNRP}
	e := testEngine(make([]Point, 1), nil, once)
	if err := e.runProgram(programGlyph); err != nil {
		t.Errorf("one full loop: %v", err)
	}
	e = testEngine(make([]Point, 1), nil, bytes.Repeat(once, maxRunInstructions/0x7FFF+1))
	if err := e.runProgram(programGlyph); !errors.Is(err, errExceededExecutionBudget) {
		t.Errorf("more iterations than the budget: got %v, want %v", err, errExceededExecutionBudget)
	}
}

// TestPhantomPointsSurviveAFailedProgram checks that a glyph program which ends in an error outside backward
// compatibility still hands back the phantom points it had moved, like the rest of the outline.
func TestPhantomPointsSurviveAFailedProgram(t *testing.T) {
	// A monochrome target runs without backward compatibility.
	in, err := testFont(t, "Roboto-Regular.ttf").NewInstance(16, TargetMono)
	if err != nil {
		t.Fatal(err)
	}
	s, _ := in.pool.Get().(*scratch)
	defer in.pool.Put(s)
	// One point and the four phantom points; the program shifts the advance phantom point a pixel to the right and
	// then runs an opcode nothing defines.
	points := make([]Point, 1+phantomPointCount)
	points[2].X = 640
	program := []byte{opPUSHB0 + 1, 2, 64, opSHPIX, 0x28}
	keep := in.hint(s, slices.Clone(points), slices.Clone(points), points, make([]uint8, len(points)), []uint16{0},
		program, false)
	if !keep {
		t.Error("the phantom points of a failed program were dropped")
	}
	if points[2].X != 704 {
		t.Errorf("advance phantom point at %d, want 704", points[2].X)
	}
}

// TestAATakesItsOperand checks that the obsolete AA instruction still pops the value it was given, so that what
// follows finds the stack it expects.
func TestAATakesItsOperand(t *testing.T) {
	e := testEngine(make([]Point, 1), nil, []byte{opPUSHB0 + 1, 5, 7, opAA, opSRP0})
	if err := e.runProgram(programGlyph); err != nil {
		t.Fatal(err)
	}
	if e.gs.rp0 != 5 {
		t.Errorf("rp0 is %d, want 5: AA left its operand for SRP0", e.gs.rp0)
	}
}

// TestSDPVTLOnCoincidentOriginalPoints checks that SDPVTL asked for a perpendicular gives it up for both vectors once
// the two points turn out to coincide in the original outline: the dual projection vector is the x axis, and the
// projection vector runs along the line through the points' current positions, not across it.
func TestSDPVTLOnCoincidentOriginalPoints(t *testing.T) {
	e := testEngine(make([]Point, 2), nil, []byte{opPUSHB0 + 1, 0, 1, opSDPVTL0 + 1})
	e.gs.zones[zoneGlyph].points[1] = Point{Y: 64}
	if err := e.runProgram(programGlyph); err != nil {
		t.Fatal(err)
	}
	if want := (Point{X: 0x4000}); e.gs.dualProjVector != want {
		t.Errorf("dual projection vector %v, want %v", e.gs.dualProjVector, want)
	}
	if want := (Point{Y: -0x4000}); e.gs.projVector != want {
		t.Errorf("projection vector %v, want %v", e.gs.projVector, want)
	}
}

// TestMSIRPStartsATwilightPointFromTheReferencePoint checks that MSIRP into the twilight zone measures its distance
// from the reference point's original position, wherever the twilight point was before.
func TestMSIRPStartsATwilightPointFromTheReferencePoint(t *testing.T) {
	e := testEngine([]Point{{Y: 700}}, nil, []byte{
		opPUSHB0, 0, opSZP1,
		0x00,                // SVTCA[y]
		opPUSHB0 + 1, 1, 64, // point 1, a distance of one pixel
		opMSIRP0,
	})
	twilight := e.gs.zone(zoneTwilight)
	twilight.original[1] = Point{X: 3, Y: 999}
	if err := e.runProgram(programGlyph); err != nil {
		t.Fatal(err)
	}
	want := Point{Y: 764}
	if twilight.original[1] != want || twilight.points[1] != want {
		t.Errorf("twilight point at %v, originally %v, want both %v", twilight.points[1], twilight.original[1], want)
	}
}
