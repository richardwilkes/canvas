// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The B.2 fixed-arity GL call lane for the SysV platforms (darwin/linux on amd64/arm64): glCall dispatches a resolved
// GL proc through the in-repo glcall9 assembly trampoline (fastcall_sysv_*.s) via runtime.cgocall, replacing
// purego.SyscallN for the all-integer wrappers in interface.go. purego.SyscallN costs two heap allocations per call —
// the escaping variadic []uintptr at the call site (forced by its //go:uintptrescapes tag) and its internal 25-slot
// syscall15Args struct, which escapes because purego's runtime.cgocall linkname carries no //go:noescape — measured at
// ~240-256 B/call and ~85% of the panel frame's allocations. This lane is fixed-arity (no slice), keeps its argument
// block on the goroutine stack (see the runtimeCgocall contract below), loads integer registers only, and skips
// purego's errno read-back (GL never reports through errno), so a GL call performs zero heap allocations and one fewer
// libc call on darwin.
//
// ABI notes (the two known purego holes are inherited deliberately, so the glgen classifier's routing stays valid
// unchanged):
//   - Integer-class arguments only: any entry point with a float-by-value parameter is bound via purego.RegisterFunc by
//     the generator, exactly as before.
//   - The 9th argument spills to a full 8-byte stack slot. On darwin/arm64 Apple packs stack arguments at natural size,
//     so this is correct for an 8-byte 9th argument (glTexImage2D & co. pass a data pointer/offset there) and, because
//     the 9th is necessarily the *last* argument at this arity, also for a sub-8-byte one (the callee reads the low
//     bytes of the zero-extended slot on little-endian — the same "single trailing stack argument" case the SyscallN
//     path relied on, e.g. glMapTexSubImage2D's GLenum access). Signatures where a sub-8-byte stack argument is
//     followed by more arguments are already routed to RegisterFunc by the generator.
//
// Arity 9 is the maximum across all SyscallN-routed GL entry points (glTexImage2D, glTexSubImage2D,
// glCompressedTexSubImage2D, glMapTexSubImage2D); TestGLCallArityCoverage pins that so a future GL function-table
// regeneration cannot silently exceed the trampoline's arity.

//go:build (darwin || linux) && (amd64 || arm64)

package gl

import (
	"unsafe"

	_ "github.com/ebitengine/purego" // keeps the fakecgo runtime shim linked for nocgo builds
)

// glcall9Args is the argument block the glcall9 trampoline consumes. The field offsets are baked into fastcall_sysv_*.s
// through the generated go_asm.h, so a layout change breaks the build rather than miscalling.
type glcall9Args struct {
	fn                                 uintptr
	a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr
	r1                                 uintptr
}

// glcall9ABI0 holds the C-ABI entry address of the glcall9 trampoline (bound by the DATA directive in
// fastcall_sysv_*.s), in the same style as purego's syscall15XABI0.
var glcall9ABI0 uintptr

// runtimeCgocall is runtime.cgocall, the same pull purego uses (its go_runtime.go), but tagged //go:noescape so the
// argument block stays on the goroutine stack. That is sound because:
//   - cgocall does not retain arg beyond the call (it ends with KeepAlive(arg));
//   - while the C code runs the goroutine is in _Gsyscall (cgocall calls entersyscall), and the runtime never moves a
//     stack whose goroutine is in a syscall (copystack/shrinkstack bail out when gp.syscallsp != 0), so the raw pointer
//     the trampoline holds cannot be invalidated;
//   - any stack growth in cgocall's own prologue happens before the pointer leaves Go, and arg is a real unsafe.Pointer
//     parameter, so the stack copier adjusts it.
//
// The runtime's own darwin libc shims (syscall_syscall9 & co.) rely on exactly this stack-resident-block pattern. In
// CGO_ENABLED=0 builds runtime.cgocall works because importing purego links its fakecgo runtime shim, which flips
// runtime.iscgo and installs the cgo hooks.
//
//go:noescape
//go:linkname runtimeCgocall runtime.cgocall
func runtimeCgocall(fn uintptr, arg unsafe.Pointer) int32

// glCall invokes the GL proc fn with up to nine integer-class arguments through the glcall9 trampoline and returns the
// proc's integer result. Callers pad unused trailing arguments with zero; a C callee never reads registers or stack
// slots beyond its declared parameters, so the padding is inert. The //go:uintptrescapes tag is the S105 GC-stack
// contract, identical to the tag on the purego.SyscallN this lane replaces: a Go pointer converted at a glCall call
// site (e.g. uintptr(unsafe.Pointer(buffers)) in a generated wrapper) is forced to escape and stays alive and pinned
// for the duration of the call.
//
//go:uintptrescapes
func glCall(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) uintptr {
	args := glcall9Args{
		fn: fn, a1: a1, a2: a2, a3: a3, a4: a4, a5: a5, a6: a6, a7: a7, a8: a8,
		a9: a9,
	}
	runtimeCgocall(glcall9ABI0, unsafe.Pointer(&args))
	return args.r1
}
