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
	"errors"
	"sync"
)

// Instance is a Font prepared for one size and rendering target: its font and control value programs have been run,
// and what they left behind is what every glyph program starts from. It is immutable and safe for concurrent use.
type Instance struct {
	font             *Font
	pool             sync.Pool
	functions        []definition
	instructions     []definition
	cvt              []int32
	storage          []int32
	twilightScaled   []Point
	twilightOriginal []Point
	twilightFlags    []uint8
	graphics         retainedState
	scale            int32
}

// scratch is the memory one glyph load needs, pooled per Instance.
type scratch struct {
	simple           simpleGlyph
	stack            []int32
	cvt              []int32
	storage          []int32
	twilightScaled   []Point
	twilightOriginal []Point
	twilightFlags    []uint8
	unscaled         []Point
	original         []Point
	scaled           []Point
	flags            []uint8
	contours         []uint16
	engine           engine
}

// ErrHintingDisabled is returned by NewInstance when the font's own control value program turns hinting off at the
// requested size, which fonts do for sizes their instructions were not written for.
var ErrHintingDisabled = errors.New("the font disables its instructions at this size")

// NewInstance prepares the font for hinting at ppem pixels per em, which need not be a whole number unless the font
// insists. It fails if the font's setup programs do, in which case the font is best drawn unhinted at this size.
func (f *Font) NewInstance(ppem float32, target Target) (*Instance, error) {
	if !(ppem > 0) || ppem > 0x4000 {
		return nil, errors.New("invalid size")
	}
	size := int32(ppem * 64)
	if f.forceIntegerPPEM {
		size = round26(size)
	}
	scale := divFixed(size, int32(f.unitsPerEm))
	// The integer ppem instructions see is rounded in fixed point, as FreeType does.
	roundedPPEM := (mulFixed(scale, int32(f.unitsPerEm)) + 32) >> 6
	in := &Instance{
		font:             f,
		scale:            scale,
		functions:        make([]definition, f.maxFunctionDefs),
		instructions:     make([]definition, f.maxInstrDefs),
		cvt:              make([]int32, len(f.cvt)),
		storage:          make([]int32, f.maxStorage),
		twilightScaled:   make([]Point, f.maxTwilight),
		twilightOriginal: make([]Point, f.maxTwilight),
		twilightFlags:    make([]uint8, f.maxTwilight),
	}
	// Control values are held in 26.6 already, hence the odd scale.
	cvtScale := scale >> 6
	for i, v := range f.cvt {
		in.cvt[i] = mulFixed(int32(v)*64, cvtScale)
	}
	e := engine{
		bytecode:     [3][]byte{f.fpgm, f.prep, nil},
		functions:    in.functions,
		instructions: in.instructions,
		cvt:          in.cvt,
		storage:      in.storage,
		stack:        make([]int32, f.maxStack),
		cvtLen:       len(f.cvt),
		defsWritable: true,
	}
	e.gs.retainedState = newRetainedState(scale, roundedPPEM, target)
	e.gs.zones[zoneTwilight] = zone{
		original: in.twilightOriginal,
		points:   in.twilightScaled,
		flags:    in.twilightFlags,
		contours: []uint16{uint16(len(in.twilightScaled))},
	}
	e.setLoopLimit()
	if err := e.runProgram(programFont); err != nil {
		return nil, err
	}
	if err := e.runProgram(programControlValue); err != nil {
		return nil, err
	}
	in.graphics = e.gs.retainedState
	// Bit 0 of the instruction control flags disables the glyph programs entirely.
	if in.graphics.instructControl&1 != 0 {
		return nil, ErrHintingDisabled
	}
	in.pool.New = func() any {
		return &scratch{
			stack:            make([]int32, f.maxStack),
			cvt:              make([]int32, len(in.cvt)),
			storage:          make([]int32, len(in.storage)),
			twilightScaled:   make([]Point, len(in.twilightScaled)),
			twilightOriginal: make([]Point, len(in.twilightOriginal)),
			twilightFlags:    make([]uint8, len(in.twilightFlags)),
		}
	}
	return in, nil
}

// backwardCompatibility reports whether glyph programs start out in backward compatibility mode.
func (in *Instance) backwardCompatibility() bool {
	if in.graphics.target.isSmooth() {
		return in.graphics.instructControl&4 == 0
	}
	return false
}

// hint runs a glyph program over the given points, which end with the four phantom points. Errors end the program
// and are otherwise ignored, leaving the points as far as it got. It reports whether the phantom points the program
// left are to be kept, which they are when it ran to its end outside backward compatibility mode.
func (in *Instance) hint(s *scratch, unscaled, original, scaled []Point, flags []uint8, contours []uint16,
	bytecode []byte, isComposite bool,
) (keepPhantom bool) {
	copy(s.twilightScaled, in.twilightScaled)
	copy(s.twilightOriginal, in.twilightOriginal)
	copy(s.twilightFlags, in.twilightFlags)
	e := &s.engine
	*e = engine{
		bytecode:       [3][]byte{in.font.fpgm, in.font.prep, bytecode},
		functions:      in.functions,
		instructions:   in.instructions,
		cvt:            in.cvt,
		cvtScratch:     s.cvt,
		storage:        in.storage,
		storageScratch: s.storage,
		stack:          s.stack,
		cvtLen:         len(in.font.cvt),
	}
	e.gs.retainedState = in.graphics
	e.gs.isComposite = isComposite
	e.gs.zones[zoneTwilight] = zone{
		original: s.twilightOriginal,
		points:   s.twilightScaled,
		flags:    s.twilightFlags,
		contours: []uint16{uint16(len(s.twilightScaled))},
	}
	e.gs.zones[zoneGlyph] = zone{
		unscaled: unscaled,
		original: original,
		points:   scaled,
		flags:    flags,
		contours: contours,
	}
	e.setLoopLimit()
	// An error here is the font's, and the outline is used as far as its program got.
	err := e.runProgram(programGlyph)
	return err == nil && !e.gs.backwardCompatibility
}
