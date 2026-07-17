// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// FFI allocation probe: purego.SyscallN heap-allocates its argument struct and the caller-side variadic []uintptr on
// every GL call — the FFI-variadic tier that was ~85% of the panels frame until the fixed-arity glCall lane
// (fastcall_*.go) closed it. This probe keeps three per-call characterizations live on the *identical* dlsym'd proc
// addresses: the production wrapper (the glCall lane — must stay zero-alloc on the trampoline platforms and never worse
// than SyscallN anywhere), a raw purego.SyscallN (the comparison baseline), and a purego.RegisterFunc-bound typed func
// (the standing verdict that reflection-marshaled calls allocate strictly more, which is why the 12 float-carrying
// entry points ride it only for ABI correctness). This is an in-package test so it can read the unexported proc-address
// fields the assembler resolved (f.vertexAttribPointer, …).

package gl

import (
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/gpu/gl/gltest"
)

// probeFunctions brings up a live GL context (skipping when none is available, like the other GL live tests) and
// returns its resolved proc table.
func probeFunctions(tb testing.TB) *Functions {
	tb.Helper()
	ctx, err := gltest.New()
	if err != nil {
		tb.Skipf("no GL context available: %v", err)
	}
	tb.Cleanup(ctx.Destroy)
	intf := MakeNativeInterface()
	if intf == nil {
		tb.Fatal("MakeNativeInterface returned nil with a current context")
		return nil // unreachable; Fatal does not return, but staticcheck cannot see that through testing.TB
	}
	return &intf.Functions
}

// probeSetupGeometry binds a VAO with an array buffer and an element buffer so that the probed
// glVertexAttribPointer(index 0, offset 0) and glDrawElements(offset 0) calls reference valid bound-buffer offsets. The
// FFI marshaling (the thing this probe measures) happens Go-side before the driver runs, so a GL error from a missing
// bound program would not change the allocation counts; binding real buffers just keeps the calls clean.
func probeSetupGeometry(tb testing.TB, f *Functions) {
	tb.Helper()
	var vao, vbo, ebo uint32
	f.GenVertexArrays(1, &vao)
	f.BindVertexArray(vao)
	f.GenBuffers(1, &vbo)
	f.BindBuffer(ARRAY_BUFFER, vbo)
	verts := []float32{0, 0, 1, 0, 0, 1}
	f.BufferData(ARRAY_BUFFER, len(verts)*4, uintptr(unsafe.Pointer(&verts[0])), STATIC_DRAW)
	f.GenBuffers(1, &ebo)
	f.BindBuffer(ELEMENT_ARRAY_BUFFER, ebo)
	idx := []uint16{0, 1, 2}
	f.BufferData(ELEMENT_ARRAY_BUFFER, len(idx)*2, uintptr(unsafe.Pointer(&idx[0])), STATIC_DRAW)
	f.EnableVertexAttribArray(0)
	f.GetError() // drain any accumulated error; irrelevant to allocation, keeps the state tidy
}

// TestFFIAllocProbe reports allocs/op for three all-integer procs called three ways: through the production wrapper
// (the B.2 fixed-arity glCall lane), through a raw purego.SyscallN of the identical proc address (the pre-B.2
// production path, kept for the head-to-head), and through a purego.RegisterFunc-bound typed func. Run with -v to see
// the table:
//
//	go test ./gpu/gl/ -run TestFFIAllocProbe -v
func TestFFIAllocProbe(t *testing.T) {
	f := probeFunctions(t)
	probeSetupGeometry(t, f)

	// RegisterFunc-bound typed funcs of the identical proc addresses (bound once — RegisterFunc itself is
	// reflection-heavy at bind time; we measure the per-*call* cost).
	var rfScissor func(int32, int32, int32, int32)
	purego.RegisterFunc(&rfScissor, f.scissor)
	var rfVAP func(uint32, int32, uint32, uint8, int32, uintptr)
	purego.RegisterFunc(&rfVAP, f.vertexAttribPointer)
	var rfDraw func(uint32, int32, uint32, uintptr)
	purego.RegisterFunc(&rfDraw, f.drawElements)

	cases := []struct {
		wrapper      func()
		syscallN     func()
		registerFunc func()
		name         string
	}{
		{
			name:         "Scissor(4 int)",
			wrapper:      func() { f.Scissor(0, 0, 64, 64) },
			syscallN:     func() { purego.SyscallN(f.scissor, 0, 0, 64, 64) },
			registerFunc: func() { rfScissor(0, 0, 64, 64) },
		},
		{
			name:         "VertexAttribPointer(6 int)",
			wrapper:      func() { f.VertexAttribPointer(0, 2, FLOAT, false, 0, 0) },
			syscallN:     func() { purego.SyscallN(f.vertexAttribPointer, 0, 2, FLOAT, 0, 0, 0) },
			registerFunc: func() { rfVAP(0, 2, FLOAT, 0, 0, 0) },
		},
		{
			name:         "DrawElements(4 int)",
			wrapper:      func() { f.DrawElements(TRIANGLES, 3, UNSIGNED_SHORT, 0) },
			syscallN:     func() { purego.SyscallN(f.drawElements, TRIANGLES, 3, UNSIGNED_SHORT, 0) },
			registerFunc: func() { rfDraw(TRIANGLES, 3, UNSIGNED_SHORT, 0) },
		},
	}

	t.Logf("%-28s %12s %14s %18s", "proc", "glCall a/op", "SyscallN a/op", "RegisterFunc a/op")
	for _, c := range cases {
		gc := testing.AllocsPerRun(5000, c.wrapper)
		sn := testing.AllocsPerRun(5000, c.syscallN)
		rf := testing.AllocsPerRun(5000, c.registerFunc)
		t.Logf("%-28s %12.2f %14.2f %18.2f", c.name, gc, sn, rf)
		// The B.2 verdict: the fixed-arity glCall lane must never be more expensive than the purego.SyscallN path it
		// replaced (it is 0 allocs/op on the trampoline platforms and equal on the fallback platforms, where glCall is
		// SyscallN).
		if gc > sn {
			t.Errorf("%s: the glCall lane costs %.2f allocs/op vs SyscallN's %.2f — the fixed-arity "+
				"path regressed", c.name, gc, sn)
		}
		// The verdict this probe exists to keep live: RegisterFunc is not a cheaper allocation path than SyscallN (it
		// boxes each argument into a heap reflect.Value on top of the same arg-struct escape, so allocs scale with
		// arity — measured on darwin/arm64 against purego v0.10.1 at 6 vs 2 allocs/op for a 4-int glDrawElements and 8
		// vs 2 for a 6-int glVertexAttribPointer). If this ever flips — a future purego/Go making RegisterFunc-bound
		// calls allocation-free — it re-opens a low-risk route for the remaining float-carrying entry points, and this
		// failure is the signal to re-evaluate.
		if rf <= sn {
			t.Errorf("%s: RegisterFunc %.2f allocs/op is no longer worse than SyscallN %.2f — the "+
				"cheap path may have re-opened; re-evaluate routing hot entry points through RegisterFunc",
				c.name, rf, sn)
		}
	}
}

// The benchmarks give B/op alongside allocs/op for the same head-to-head. The GLCall pair measures the production
// wrappers (the B.2 fixed-arity lane); the SyscallN pair measures a raw purego.SyscallN of the identical proc address
// (the pre-B.2 production path). Run:
//
//	go test ./gpu/gl/ -run '^$' -bench BenchmarkFFIProbe -benchmem
func BenchmarkFFIProbeGLCallVertexAttribPointer(b *testing.B) {
	f := probeFunctions(b)
	probeSetupGeometry(b, f)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.VertexAttribPointer(0, 2, FLOAT, false, 0, 0)
	}
}

func BenchmarkFFIProbeSyscallNVertexAttribPointer(b *testing.B) {
	f := probeFunctions(b)
	probeSetupGeometry(b, f)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		purego.SyscallN(f.vertexAttribPointer, 0, 2, FLOAT, 0, 0, 0)
	}
}

func BenchmarkFFIProbeRegisterFuncVertexAttribPointer(b *testing.B) {
	f := probeFunctions(b)
	probeSetupGeometry(b, f)
	var rfVAP func(uint32, int32, uint32, uint8, int32, uintptr)
	purego.RegisterFunc(&rfVAP, f.vertexAttribPointer)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rfVAP(0, 2, FLOAT, 0, 0, 0)
	}
}

func BenchmarkFFIProbeGLCallDrawElements(b *testing.B) {
	f := probeFunctions(b)
	probeSetupGeometry(b, f)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.DrawElements(TRIANGLES, 3, UNSIGNED_SHORT, 0)
	}
}

func BenchmarkFFIProbeSyscallNDrawElements(b *testing.B) {
	f := probeFunctions(b)
	probeSetupGeometry(b, f)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		purego.SyscallN(f.drawElements, TRIANGLES, 3, UNSIGNED_SHORT, 0)
	}
}

func BenchmarkFFIProbeRegisterFuncDrawElements(b *testing.B) {
	f := probeFunctions(b)
	probeSetupGeometry(b, f)
	var rfDraw func(uint32, int32, uint32, uintptr)
	purego.RegisterFunc(&rfDraw, f.drawElements)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rfDraw(TRIANGLES, 3, UNSIGNED_SHORT, 0)
	}
}
