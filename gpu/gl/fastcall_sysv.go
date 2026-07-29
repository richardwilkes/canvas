// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The fixed-arity GL call lane for the SysV platforms (darwin/linux on amd64/arm64): glCall dispatches a resolved
// GL proc through the in-repo glcall9 assembly trampoline (fastcall_sysv_*.s) via runtime.cgocall, replacing
// purego.SyscallN for the all-integer wrappers in interface.go. purego.SyscallN costs two heap allocations per call —
// the escaping variadic []uintptr at the call site (forced by its //go:uintptrescapes tag) and its internal 25-slot
// syscall15Args struct, which escapes because purego's runtime.cgocall linkname carries no //go:noescape — measured at
// ~240-256 B/call and ~85% of the panel frame's allocations. This lane is fixed-arity (no slice), reuses one
// heap-resident argument block per Functions (see the glCall contract below), loads integer registers only, and skips
// purego's errno read-back (GL never reports through errno), so a GL call performs zero heap allocations in steady
// state and one fewer libc call on darwin.
//
// ABI notes — two constraints every glCall-dispatched wrapper in interface.go must satisfy. Both are holes inherited
// deliberately from the purego.SyscallN path this lane replaced, so the routing that path already used stays valid
// unchanged. interface.go is maintained by hand, so nothing enforces them as a wrapper is written; the tests are what
// hold the line:
//   - Integer-class arguments only: an entry point with a float-by-value parameter must instead be bound through
//     purego.RegisterFunc, as one of the fnXxx fields on Functions (see initRegistered).
//   - The 9th argument spills to a full 8-byte stack slot. On darwin/arm64 Apple packs stack arguments at natural size,
//     so this is correct for an 8-byte 9th argument (glTexImage2D & co. pass a data pointer/offset there) and, because
//     the 9th is necessarily the *last* argument at this arity, also for a sub-8-byte one (the callee reads the low
//     bytes of the zero-extended slot on little-endian — the same "single trailing stack argument" case the SyscallN
//     path relied on, e.g. glMapTexSubImage2D's GLenum access). A signature with a sub-8-byte stack argument followed
//     by further arguments must go through RegisterFunc as well.
//
// Arity 9 is the maximum across all glCall-dispatched GL entry points (glTexImage2D, glTexSubImage2D,
// glCompressedTexSubImage2D, glMapTexSubImage2D); TestGLCallArityCoverage pins that so a later addition to the proc
// table cannot silently exceed the trampoline's arity.

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

// glCallState is this lane's per-Functions call state: the reusable, heap-resident glcall9Args block glCall marshals
// through. One block per Functions rather than one per call keeps the lane allocation-free in steady state; see glCall
// for why the block must not live on the goroutine stack.
type glCallState struct {
	args *glcall9Args
}

// newGLCallArgs allocates a glcall9Args block on the heap. The //go:noinline tag is the guarantee, not an optimization
// hint: a function that cannot be inlined and returns a pointer to a value it allocated cannot place that value on its
// own frame, so the block is heap-resident regardless of where the owning Functions lives. That matters because a
// Functions is not always heap-allocated itself — interface_test.go builds one as a bare &Functions{} that escape
// analysis may well keep on the stack.
//
//go:noinline
func newGLCallArgs() *glcall9Args { return new(glcall9Args) }

// runtimeCgocall is runtime.cgocall, the same pull purego uses (its go_runtime.go), tagged //go:noescape so the
// argument block is passed as a raw pointer with no escape penalty. cgocall does not retain arg beyond the call (it
// ends with KeepAlive(arg)), so the only requirement is that the pointee stay put for the duration — which glCall
// satisfies by keeping the block on the heap.
//
// In CGO_ENABLED=0 builds runtime.cgocall works because importing purego links its fakecgo runtime shim, which flips
// runtime.iscgo and installs the cgo hooks.
//
//go:noescape
//go:linkname runtimeCgocall runtime.cgocall
func runtimeCgocall(fn uintptr, arg unsafe.Pointer) int32

// glCall invokes the GL proc fn with up to nine integer-class arguments through the glcall9 trampoline and returns the
// proc's integer result. Callers pad unused trailing arguments with zero; a C callee never reads registers or stack
// slots beyond its declared parameters, so the padding is inert. The //go:uintptrescapes tag is the GC-stack contract,
// identical to the tag on the purego.SyscallN this lane replaces: a Go pointer converted at a glCall call site (e.g.
// uintptr(unsafe.Pointer(buffers)) in an interface.go wrapper) is forced to escape and stays alive and pinned for the
// duration of the call.
//
// The argument block is deliberately NOT a local. glcall9 stashes the block pointer across its BL to the GL proc and
// writes the result back through it afterwards, so the block must not move while the proc runs. A stack-resident block
// is safe only while the goroutine stays in _Gsyscall, which holds for a real driver but not for a Go callee: a
// purego.NewCallback proc (every fake driver in this package's tests) re-enters Go through cgocallbackg, which calls
// exitsyscall and runs on this goroutine's stack. If that re-entry needs to grow the stack, copystack moves the frame
// and the trampoline writes the result to the abandoned copy — the caller then reads a stale zero. That was the cause
// of a class of sporadic CI failure: glGetString returning "" (an unparsable GL_VERSION, so NewContextInfo returned
// nil) and glClientWaitSync returning 0 instead of ALREADY_SIGNALED (a fence that never reported signaled). Keeping
// the block on the heap removes the hazard for every callee, Go or C.
//
// The block is shared per Functions, which assumes GL calls on one Functions are neither concurrent nor re-entrant.
// Both hold: a GL context is current on exactly one thread (gltest locks it), and nothing in this package installs a
// Go callback that GL can invoke mid-call — glDebugMessageCallback is resolved but never bound to one. Binding a Go
// debug callback that itself issues GL calls would need a per-depth block.
//
//go:uintptrescapes
func (f *Functions) glCall(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) uintptr {
	args := f.callState.args
	if args == nil {
		args = newGLCallArgs()
		f.callState.args = args
	}
	*args = glcall9Args{
		fn: fn, a1: a1, a2: a2, a3: a3, a4: a4, a5: a5, a6: a6, a7: a7, a8: a8,
		a9: a9,
	}
	runtimeCgocall(glcall9ABI0, unsafe.Pointer(args))
	return args.r1
}
