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
	"runtime"
)

// hintError is what an instruction raises to end its program. It travels as a panic from wherever in the instruction
// it was detected to runProgram, which is the only place that recovers one, so that the instructions themselves read
// as the straight-line code they are in the reference.
type hintError struct {
	err error
}

var (
	errUnexpectedEndOfBytecode  = errors.New("unexpected end of bytecode")
	errInvalidPointIndex        = errors.New("invalid point index")
	errInvalidPointRange        = errors.New("invalid point range")
	errInvalidContourIndex      = errors.New("invalid contour index")
	errInvalidCvtIndex          = errors.New("invalid control value index")
	errInvalidZoneIndex         = errors.New("invalid zone index")
	errInvalidStackValue        = errors.New("invalid stack value")
	errInvalidJump              = errors.New("invalid jump")
	errInvalidDefinition        = errors.New("call of an undefined function")
	errUnhandledOpcode          = errors.New("unhandled opcode")
	errValueStackOverflow       = errors.New("value stack overflow")
	errValueStackUnderflow      = errors.New("value stack underflow")
	errCallStackOverflow        = errors.New("call stack overflow")
	errCallStackUnderflow       = errors.New("call stack underflow")
	errDefinitionInGlyphProgram = errors.New("definition in a glyph program")
	errNestedDefinition         = errors.New("nested definition")
	errTooManyDefinitions       = errors.New("too many definitions")
	errNegativeLoopCounter      = errors.New("negative loop counter")
	errDivideByZero             = errors.New("divide by zero")
	errExceededExecutionBudget  = errors.New("exceeded execution budget")
	errInternal                 = errors.New("instruction failed")
)

func fail(err error) {
	panic(hintError{err: err})
}

type program uint8

const (
	programFont program = iota
	programControlValue
	programGlyph
)

// maxRunInstructions bounds a single program run.
const maxRunInstructions = 1_000_000

// maxCallDepth is the depth of FreeType's call stack.
const maxCallDepth = 32

type definition struct {
	start, end int
	key        int32
	program    program
	active     bool
}

type callRecord struct {
	definition    definition
	returnPC      int
	currentCount  uint32
	callerProgram program
}

type instruction struct {
	inline  []byte
	pc      int
	opcode  uint8
	isWords bool
}

type engine struct {
	bytecode     [3][]byte
	code         []byte
	functions    []definition
	instructions []definition
	// cvt and storage are what is read. For a glyph program they start out as the instance's own, shared and not to be
	// written, and are swapped for the scratch copy on the first write.
	cvt            []int32
	cvtScratch     []int32
	storage        []int32
	storageScratch []int32
	stack          []int32
	gs             graphicsState
	callStack      [maxCallDepth]callRecord
	pc             int
	sp             int
	callDepth      int
	loopLimit      int
	backwardJumps  int
	loopCalls      int
	skipped        int
	cvtLen         int
	initial        program
	current        program
	// defsWritable is true only while the font and control value programs run.
	defsWritable bool
}

// runProgram runs one of the three programs from its start, reporting the error that ended it, if one did.
func (e *engine) runProgram(p program) (err error) {
	defer func() {
		if r := recover(); r != nil {
			switch failure := r.(type) {
			case hintError:
				err = failure.err
			case runtime.Error:
				// Indexes are checked before use, so this should not happen, but a malformed font must cost no more
				// than an unhinted glyph.
				err = errInternal
			default:
				panic(r)
			}
		}
	}()
	e.reset(p)
	count := 0
	for {
		ins, ok := e.decode()
		if !ok {
			return nil
		}
		e.dispatch(&ins)
		count++
		if count > maxRunInstructions {
			fail(errExceededExecutionBudget)
		}
	}
}

func (e *engine) reset(p program) {
	e.initial = p
	e.current = p
	e.code = e.bytecode[p]
	e.pc = 0
	e.sp = 0
	e.callDepth = 0
	e.gs.reset()
	e.backwardJumps = 0
	e.loopCalls = 0
	e.skipped = 0
	switch p {
	case programFont:
		clear(e.functions)
		clear(e.instructions)
	case programControlValue:
		e.gs.backwardCompatibility = false
	case programGlyph:
		// Instruct control bit 1 says the retained graphics state goes back to its defaults.
		if e.gs.instructControl&2 != 0 {
			e.gs.resetRetained()
		}
		if e.gs.target.isSmooth() {
			e.gs.backwardCompatibility = e.gs.instructControl&4 == 0
		} else {
			e.gs.backwardCompatibility = false
		}
	}
}

// setLoopLimit computes the bound on loop calls and backward jumps, as FreeType does.
func (e *engine) setLoopLimit() {
	if n := len(e.gs.zones[zoneGlyph].points); n != 0 {
		e.loopLimit = max(n*10, 50) + max(e.cvtLen/10, 50)
	} else {
		e.loopLimit = 300 + 22*e.cvtLen
	}
}

// decode reads the instruction at the program counter and advances past it. It reports false at the end of the code.
func (e *engine) decode() (instruction, bool) {
	if e.pc < 0 || e.pc >= len(e.code) {
		return instruction{}, false
	}
	opcode := e.code[e.pc]
	length := 1
	countLen := 0
	switch {
	case opcode == opNPUSHB || opcode == opNPUSHW:
		if e.pc+1 >= len(e.code) {
			fail(errUnexpectedEndOfBytecode)
		}
		size := 1
		if opcode == opNPUSHW {
			size = 2
		}
		length = size*int(e.code[e.pc+1]) + 2
		countLen = 1
	case opcode >= opPUSHB0 && opcode <= opPUSHB0+7:
		length = int(opcode-opPUSHB0) + 2
	case opcode >= opPUSHW0 && opcode <= opPUSHW0+7:
		length = 2*int(opcode-opPUSHW0) + 3
	}
	ins := instruction{opcode: opcode, pc: e.pc}
	inlineStart := e.pc + 1 + countLen
	next := e.pc + length
	if next > inlineStart {
		if next > len(e.code) {
			fail(errUnexpectedEndOfBytecode)
		}
		ins.inline = e.code[inlineStart:next]
		ins.isWords = opcode == opNPUSHW || (opcode >= opPUSHW0 && opcode <= opPUSHW0+7)
	}
	e.pc = next
	return ins, true
}

func (e *engine) decodeNextOpcode() uint8 {
	ins, ok := e.decode()
	if !ok {
		fail(errUnexpectedEndOfBytecode)
	}
	return ins.opcode
}

// The value stack. Popping an empty stack yields zero, as FreeType's does outside its pedantic mode.

func (e *engine) push(value int32) {
	if e.sp >= len(e.stack) {
		fail(errValueStackOverflow)
	}
	e.stack[e.sp] = value
	e.sp++
}

func (e *engine) pop() int32 {
	if e.sp == 0 {
		return 0
	}
	e.sp--
	return e.stack[e.sp]
}

// popIndex pops a value used as an index. A negative one stays negative, which no bounds check accepts.
func (e *engine) popIndex() int { return int(e.pop()) }

func (e *engine) popCount() int { return int(max(e.pop(), 0)) }

func (e *engine) pushInline(ins *instruction) {
	if ins.isWords {
		n := len(ins.inline) / 2
		if e.sp+n > len(e.stack) {
			fail(errValueStackOverflow)
		}
		for i := range n {
			e.stack[e.sp+i] = int32(int16(uint16(ins.inline[i*2])<<8 | uint16(ins.inline[i*2+1])))
		}
		e.sp += n
		return
	}
	n := len(ins.inline)
	if e.sp+n > len(e.stack) {
		fail(errValueStackOverflow)
	}
	for i, b := range ins.inline {
		e.stack[e.sp+i] = int32(b)
	}
	e.sp += n
}

// Control flow.

func (e *engine) skippingInstruction() {
	e.skipped++
	if e.skipped > maxRunInstructions {
		fail(errExceededExecutionBudget)
	}
}

func (e *engine) opIf() {
	if e.pop() != 0 {
		return
	}
	// The condition is false, so skip to the matching ELSE or EIF, stepping over nested IFs.
	depth := 1
	for out := false; !out; {
		opcode := e.decodeNextOpcode()
		e.skippingInstruction()
		switch opcode {
		case opIF:
			depth++
		case opELSE:
			out = depth == 1
		case opEIF:
			depth--
			out = depth == 0
		}
	}
}

func (e *engine) opElse() {
	for depth := 1; depth != 0; {
		opcode := e.decodeNextOpcode()
		e.skippingInstruction()
		switch opcode {
		case opIF:
			depth++
		case opEIF:
			depth--
		}
	}
}

func (e *engine) doJump(test bool) {
	// The offset is relative to the jump instruction and the decoder is already past it.
	offset := e.pop() - 1
	if !test {
		return
	}
	if offset < 0 {
		if offset == -1 {
			// This would loop in place forever.
			fail(errInvalidJump)
		}
		e.backwardJumps++
		if e.backwardJumps > e.loopLimit {
			fail(errExceededExecutionBudget)
		}
	}
	e.pc += int(offset)
}

// Function and instruction definitions.

func allocateDefinition(defs []definition, key int32) *definition {
	// First see if the key can be used as an index, which it can for any well-behaved font's functions.
	index := -1
	if key >= 0 && int(key) < len(defs) && (!defs[key].active || defs[key].key == key) {
		index = int(key)
	} else {
		// Otherwise walk backward for an active entry with the key, remembering the last inactive one.
		for i := len(defs) - 1; i >= 0; i-- {
			if defs[i].active {
				if defs[i].key == key {
					index = i
					break
				}
			} else if index < 0 {
				index = i
			}
		}
		if index < 0 {
			fail(errTooManyDefinitions)
		}
	}
	defs[index] = definition{program: programFont, key: key, active: true}
	return &defs[index]
}

func findDefinition(defs []definition, key int32) (definition, bool) {
	if key >= 0 && int(key) < len(defs) && defs[key].active && defs[key].key == key {
		return defs[key], true
	}
	for i := len(defs) - 1; i >= 0; i-- {
		if defs[i].active && defs[i].key == key {
			return defs[i], true
		}
	}
	return definition{}, false
}

func (e *engine) doDef(defs []definition, key int32) {
	if e.initial == programGlyph || !e.defsWritable {
		fail(errDefinitionInGlyphProgram)
	}
	def := allocateDefinition(defs, key)
	start := e.pc
	for {
		ins, ok := e.decode()
		if !ok {
			fail(errUnexpectedEndOfBytecode)
		}
		switch ins.opcode {
		case opFDEF, opIDEF:
			fail(errNestedDefinition)
		case opENDF:
			*def = definition{program: e.current, start: start, end: ins.pc + 1, key: key, active: true}
			return
		}
	}
}

func (e *engine) enter(def definition, count uint32) {
	if e.callDepth >= maxCallDepth {
		fail(errCallStackOverflow)
	}
	e.callStack[e.callDepth] = callRecord{
		callerProgram: e.current,
		returnPC:      e.pc,
		currentCount:  count,
		definition:    def,
	}
	e.callDepth++
	e.current = def.program
	e.code = e.bytecode[def.program]
	e.pc = def.start
}

func (e *engine) leave() {
	if e.callDepth == 0 {
		fail(errCallStackUnderflow)
	}
	record := &e.callStack[e.callDepth-1]
	if record.currentCount > 1 {
		// A loop call with iterations remaining.
		record.currentCount--
		e.pc = record.definition.start
		return
	}
	e.callDepth--
	e.current = record.callerProgram
	e.code = e.bytecode[record.callerProgram]
	e.pc = record.returnPC
}

func (e *engine) callFunction(count uint32, key int32) {
	if count == 0 {
		return
	}
	def, ok := findDefinition(e.functions, key)
	if !ok {
		fail(errInvalidDefinition)
	}
	e.enter(def, count)
}

// opUnknown runs an instruction the font defined for itself with IDEF, if it did.
func (e *engine) opUnknown(opcode uint8) {
	def, ok := findDefinition(e.instructions, int32(opcode))
	if !ok {
		fail(errUnhandledOpcode)
	}
	e.enter(def, 1)
}

func (e *engine) opLoopCall() {
	f := e.pop()
	count := e.pop()
	if count <= 0 {
		return
	}
	e.loopCalls += int(count)
	if e.loopCalls > e.loopLimit {
		fail(errExceededExecutionBudget)
	}
	e.callFunction(uint32(count), f)
}

// The control value table and storage area.

func (e *engine) cvtGet(index int) int32 {
	if index < 0 || index >= len(e.cvt) {
		fail(errInvalidCvtIndex)
	}
	return e.cvt[index]
}

// cvtSet reports whether the index was valid.
func (e *engine) cvtSet(index int, value int32) bool {
	if e.cvtScratch != nil {
		copy(e.cvtScratch, e.cvt)
		e.cvt = e.cvtScratch
		e.cvtScratch = nil
	}
	if index < 0 || index >= len(e.cvt) {
		return false
	}
	e.cvt[index] = value
	return true
}

func (e *engine) storageSet(index int, value int32) {
	if e.storageScratch != nil {
		copy(e.storageScratch, e.storage)
		e.storage = e.storageScratch
		e.storageScratch = nil
	}
	if index >= 0 && index < len(e.storage) {
		e.storage[index] = value
	}
}
