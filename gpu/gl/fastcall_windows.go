// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The fixed-arity GL call lane for Windows: syscall.Syscall15 is the public, fixed-arity, allocation-free stdcall
// dispatcher the runtime provides — it is exactly what purego.SyscallN wraps on Windows (its syscall_windows.go),
// minus purego's escaping variadic []uintptr at the call site. Windows already marshals more cheaply than the SysV
// platforms because purego's internal syscall15Args struct never exists there; calling Syscall15 directly removes the
// remaining per-call slice allocation. Trailing arguments are padded with zero,
// which is inert: the dispatcher always passes 15 slots (just as purego always did) and a C callee never reads beyond
// its declared parameters.

//go:build windows

package gl

import "syscall"

// glCallState is this lane's per-Functions call state. There is none: Syscall15 takes its arguments in registers and
// returns the result the same way, so no argument block is exposed to the callee and nothing has to be kept off the
// goroutine stack (contrast the SysV lane's glcall9 trampoline, which writes its result back through a block pointer).
type glCallState struct{} //nolint:unused // referenced only as Functions.callState's type, which this lane never touches

// glCall invokes the GL proc fn with up to nine integer-class arguments and returns the proc's integer result. The
// //go:uintptrescapes tag is the GC-stack contract, identical to the tag on the purego.SyscallN this lane replaces: a
// Go pointer converted at a glCall call site (e.g. uintptr(unsafe.Pointer(buffers)) in an interface.go wrapper) is
// forced to escape and stays alive and pinned for the duration of the call.
//
//go:uintptrescapes
func (*Functions) glCall(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) uintptr {
	r1, _, _ := syscall.Syscall15(fn, 15, a1, a2, a3, a4, a5, a6, a7, a8, a9, 0, 0, 0, 0, 0, 0) //nolint:staticcheck,errcheck // SA1019: the fixed-arity form is the point; the Errno return is meaningless for GL procs, which report through glGetError
	return r1
}
