// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Resolves the native GL entry points on Windows using the standard syscall package (no purego dlfcn on Windows).
// opengl32.dll only exports the GL 1.1 core; everything newer comes from wglGetProcAddress, which requires a current
// WGL context. Desktop trim: ES dispatch (ANGLE) is not supported; an ES context yields nil.

package gl

import (
	"syscall"
	"unsafe"
)

// MakeNativeInterface assembles an Interface for the GL context current on this thread by resolving entry points from
// opengl32.dll and wglGetProcAddress.
func MakeNativeInterface() *Interface {
	module, err := syscall.LoadLibrary("opengl32.dll")
	if err != nil {
		return nil
	}
	wglGetCurrentContext, err := syscall.GetProcAddress(module, "wglGetCurrentContext")
	if err != nil || wglGetCurrentContext == 0 {
		return nil
	}
	if rawCall(wglGetCurrentContext) == 0 {
		return nil
	}
	wglGetProcAddress, err := syscall.GetProcAddress(module, "wglGetProcAddress")
	if err != nil || wglGetProcAddress == 0 {
		return nil
	}
	return MakeAssembledInterface(func(name string) uintptr {
		if addr, gpaErr := syscall.GetProcAddress(module, name); gpaErr == nil && addr != 0 {
			return addr
		}
		cname := cString(name)
		// rawCall, not syscall.SyscallN directly: the //go:uintptrescapes forwarder is what heap-forces the converted
		// pointer for the duration of the call, exactly as the linux leg does. SyscallN's own //go:uintptrkeepalive
		// covers only conversions written at its own call site and needs an all-nosplit caller chain, which this closure
		// is not; cString's non-constant allocation happens to heap-allocate today, but that is incidental.
		addr := rawCall(wglGetProcAddress, uintptr(unsafe.Pointer(cname)))
		keepAlive(cname)
		return addr
	})
}
