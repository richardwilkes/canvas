// Copyright 2026 Richard Wilkes
//
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
//
// glcall9: the B.2 fixed-arity GL trampoline for amd64 SysV platforms (darwin/linux). Structurally
// a trimmed port of purego's syscall15X (sys_amd64.s, Apache-2.0): integer registers only
// (float-carrying GL entry points ride purego.RegisterFunc), three 8-byte stack slots for
// arguments 7-9, AL zeroed for the SysV variadic convention (inert for the non-variadic GL entry
// points), and no errno read-back. It must run on the system stack with the C calling convention:
// it is only ever reached through runtime.cgocall(glcall9ABI0, &glcall9Args{...}), which enters
// via asmcgocall on g0. On entry DI holds the *glcall9Args block; the result is stored back into
// its r1 field. The struct field offsets come from the compiler-generated go_asm.h, so a layout
// change in fastcall_sysv.go breaks this file's build instead of miscalling.

//go:build darwin || linux

#include "textflag.h"
#include "go_asm.h"
#include "funcdata.h"

// Three 8-byte slots for arguments 7-9 at 0/8/16(SP), a 16-byte margin beyond the callee's
// incoming argument area, then the saved block pointer; 48 keeps the required 16-byte alignment
// at the CALL (entry SP is 8 mod 16, PUSHQ BP makes it 0).
#define STACK_SIZE 48
#define PTR_ADDRESS (STACK_SIZE - 8)

GLOBL ·glcall9ABI0(SB), NOPTR|RODATA, $8
DATA ·glcall9ABI0(SB)/8, $glcall9(SB)
TEXT glcall9(SB), NOSPLIT|NOFRAME, $0
	PUSHQ BP
	MOVQ  SP, BP
	SUBQ  $STACK_SIZE, SP
	MOVQ  DI, PTR_ADDRESS(SP)      // stash the argument-block pointer across the call
	MOVQ  DI, R11

	MOVQ  glcall9Args_a1(R11), DI
	MOVQ  glcall9Args_a2(R11), SI
	MOVQ  glcall9Args_a3(R11), DX
	MOVQ  glcall9Args_a4(R11), CX
	MOVQ  glcall9Args_a5(R11), R8
	MOVQ  glcall9Args_a6(R11), R9
	MOVQ  glcall9Args_a7(R11), R10
	MOVQ  R10, 0(SP)
	MOVQ  glcall9Args_a8(R11), R10
	MOVQ  R10, 8(SP)
	MOVQ  glcall9Args_a9(R11), R10
	MOVQ  R10, 16(SP)
	XORL  AX, AX                   // variadic convention: no vector-register arguments

	MOVQ  glcall9Args_fn(R11), R10
	CALL  R10

	MOVQ  PTR_ADDRESS(SP), DI
	MOVQ  AX, glcall9Args_r1(DI)
	ADDQ  $STACK_SIZE, SP
	POPQ  BP
	RET
