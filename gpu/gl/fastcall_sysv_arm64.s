// Copyright 2026 Richard Wilkes
//
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
//
// glcall9: the B.2 fixed-arity GL trampoline for arm64 SysV-style platforms (darwin/linux share
// the AAPCS64 integer-argument convention). Structurally a trimmed port of purego's syscall15X
// (sys_arm64.s, Apache-2.0): integer registers only (float-carrying GL entry points ride
// purego.RegisterFunc), one 8-byte stack slot for the 9th argument (the only stack argument any
// SyscallN-routed GL entry point has — see fastcall_sysv.go), and no errno read-back. It must run
// on the system stack with the C calling convention: it is only ever reached through
// runtime.cgocall(glcall9ABI0, &glcall9Args{...}), which enters via asmcgocall on g0. On entry
// R0 holds the *glcall9Args block; the result is stored back into its r1 field. The struct field
// offsets come from the compiler-generated go_asm.h, so a layout change in fastcall_sysv.go
// breaks this file's build instead of miscalling.

//go:build darwin || linux

#include "textflag.h"
#include "go_asm.h"
#include "funcdata.h"

// One 8-byte slot for the 9th argument at 0(RSP), an 8-byte margin beyond the callee's incoming
// argument area, then the saved block pointer; 32 keeps the required 16-byte SP alignment.
#define STACK_SIZE 32
#define PTR_ADDRESS (STACK_SIZE - 8)

GLOBL ·glcall9ABI0(SB), NOPTR|RODATA, $8
DATA ·glcall9ABI0(SB)/8, $glcall9(SB)
TEXT glcall9(SB), NOSPLIT, $0
	SUB  $STACK_SIZE, RSP
	MOVD R0, PTR_ADDRESS(RSP)      // stash the argument-block pointer across the call
	MOVD R0, R9

	MOVD glcall9Args_a1(R9), R0
	MOVD glcall9Args_a2(R9), R1
	MOVD glcall9Args_a3(R9), R2
	MOVD glcall9Args_a4(R9), R3
	MOVD glcall9Args_a5(R9), R4
	MOVD glcall9Args_a6(R9), R5
	MOVD glcall9Args_a7(R9), R6
	MOVD glcall9Args_a8(R9), R7
	MOVD glcall9Args_a9(R9), R10
	MOVD R10, 0(RSP)               // 9th argument spills to the stack (8-byte slot; see the
	                               // darwin/arm64 packing note in fastcall_sysv.go)

	MOVD glcall9Args_fn(R9), R10
	BL   (R10)

	MOVD PTR_ADDRESS(RSP), R2
	ADD  $STACK_SIZE, RSP
	MOVD R0, glcall9Args_r1(R2)
	RET
