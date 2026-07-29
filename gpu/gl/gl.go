// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package gl is the GPU backend's OpenGL binding layer and the module's only foreign-function boundary: GL entry points
// are resolved to raw proc addresses and invoked without cgo, via purego on macOS/Linux and std syscall on Windows. An
// Interface is a proc table (Functions) plus the context's standard and extension set, assembled from a GetProc
// resolver and validated against the version/extension matrix. Per the desktop trim, only desktop core-profile OpenGL
// is supported: ES and WebGL contexts are recognized but rejected at assembly.
//
// The proc table and its wrappers (interface.go) are maintained by hand. Adding or changing an entry point means
// observing the call-lane routing rules documented on Functions there and in fastcall_sysv.go's ABI notes; the tests in
// fastcall_test.go are what enforce them.

package gl

import (
	"runtime"
	"unsafe"
)

// Standard identifies the flavor of the GL implementation behind a context.
type Standard int32

// Standard values.
const (
	StandardNone Standard = iota
	StandardGL
	StandardGLES
	StandardWebGL
)

// String returns the display name of the standard.
func (s Standard) String() string {
	switch s {
	case StandardGL:
		return "OpenGL"
	case StandardGLES:
		return "OpenGL ES"
	case StandardWebGL:
		return "WebGL"
	default:
		return "none"
	}
}

// Version packs a GL major/minor version as (major << 16) | minor.
type Version uint32

// InvalidVersion represents an unrecognized or unparsed GL version.
const InvalidVersion Version = 0

// Ver packs a major/minor pair into a Version.
func Ver(major, minor uint32) Version {
	return Version(major<<16 | (minor & 0xFFFF))
}

// Major returns the major component of the version.
func (v Version) Major() uint32 { return uint32(v) >> 16 }

// Minor returns the minor component of the version.
func (v Version) Minor() uint32 { return uint32(v) & 0xFFFF }

func (v Version) String() string {
	if v == InvalidVersion {
		return "invalid"
	}
	return itoa(v.Major()) + "." + itoa(v.Minor())
}

func itoa(v uint32) string {
	if v == 0 {
		return "0"
	}
	var buf [10]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// Sync is an opaque driver-owned fence handle.
type Sync uintptr

// GetProc resolves a GL entry point by name (e.g. "glActiveTexture") to its address in the current context, returning 0
// when unavailable.
type GetProc func(name string) uintptr

// glBool converts a Go bool to a GLboolean argument word.
func glBool(b bool) uintptr {
	if b {
		return 1
	}
	return 0
}

// ptrFromUintptr converts a C-owned pointer returned in a register (never a Go pointer) to an unsafe.Pointer. Isolated
// in a helper so the conversion site is auditable.
func ptrFromUintptr(p uintptr) unsafe.Pointer {
	//nolint:govet // the value is a C pointer that never points into the Go heap
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// goString copies a NUL-terminated C string (e.g. a glGetString result) into a Go string. A zero pointer yields "".
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	b := (*byte)(ptrFromUintptr(p))
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(b), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(b, n))
}

// cString copies a Go string into a NUL-terminated byte buffer for a C string argument. The caller must keep the result
// alive across the foreign call (see keepAlive).
func cString(s string) *byte {
	buf := make([]byte, len(s)+1)
	copy(buf, s)
	return &buf[0]
}

// keepAlive pins p until this call site, so a buffer passed to a foreign call as a raw uintptr cannot be collected
// mid-call.
//
//go:noinline
func keepAlive(p *byte) {
	runtime.KeepAlive(p)
}

// BindAttribLocationString is BindAttribLocation with a Go string name.
func (f *Functions) BindAttribLocationString(program, index uint32, name string) {
	cname := cString(name)
	f.BindAttribLocation(program, index, cname)
	keepAlive(cname)
}

// GetUniformLocationString is GetUniformLocation with a Go string name.
func (f *Functions) GetUniformLocationString(program uint32, name string) int32 {
	cname := cString(name)
	loc := f.GetUniformLocation(program, cname)
	keepAlive(cname)
	return loc
}

// ShaderSourceStrings uploads Go strings as the shader source, passing explicit lengths so no NUL terminators are
// needed.
func (f *Functions) ShaderSourceStrings(shader uint32, srcs []string) {
	if len(srcs) == 0 {
		f.ShaderSource(shader, 0, 0, nil)
		return
	}
	ptrs := make([]*byte, len(srcs))
	lens := make([]int32, len(srcs))
	for i, s := range srcs {
		b := []byte(s)
		if len(b) == 0 {
			b = []byte{0}
			lens[i] = 0
		} else {
			lens[i] = int32(len(b))
		}
		ptrs[i] = &b[0]
	}
	f.ShaderSource(shader, int32(len(srcs)), uintptr(unsafe.Pointer(&ptrs[0])), &lens[0])
	runtime.KeepAlive(ptrs)
	runtime.KeepAlive(lens)
}

// GetShaderInfoLogGo returns the shader info log as a Go string.
func (f *Functions) GetShaderInfoLogGo(shader uint32) string {
	var length int32
	f.GetShaderiv(shader, INFO_LOG_LENGTH, &length)
	if length <= 1 {
		return ""
	}
	buf := make([]byte, length)
	var written int32
	f.GetShaderInfoLog(shader, length, &written, &buf[0])
	return string(buf[:written])
}

// GetProgramInfoLogGo returns the program info log as a Go string.
func (f *Functions) GetProgramInfoLogGo(program uint32) string {
	var length int32
	f.GetProgramiv(program, INFO_LOG_LENGTH, &length)
	if length <= 1 {
		return ""
	}
	buf := make([]byte, length)
	var written int32
	f.GetProgramInfoLog(program, length, &written, &buf[0])
	return string(buf[:written])
}

// GetStringGo calls glGetString and returns the result as a Go string ("" when GL returns NULL).
func (f *Functions) GetStringGo(name uint32) string {
	return goString(f.GetString(name))
}

// GetStringiGo calls glGetStringi and returns the result as a Go string ("" when GL returns NULL).
func (f *Functions) GetStringiGo(name, index uint32) string {
	return goString(f.GetStringi(name, index))
}
