// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The vertical-slice spike: on a real desktop-GL context created the way unison will hand us one, assemble and validate
// the interface without cgo, clear a framebuffer, draw one solid rect through a hand-rolled proto-pipeline (one VBO,
// one program, one draw), read the pixels back, and verify — plus the FFI-overhead measurements for the no-cgo call
// path.

package gl_test

import (
	"testing"
	"unsafe"

	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/gpu/gl/gltest"
)

func ptr[T any](p *T) unsafe.Pointer { return unsafe.Pointer(p) }

// glEnv is the shared spike fixture: a live context plus a validated interface.
type glEnv struct {
	ctx  *gltest.Context
	intf *gl.Interface
}

func newGLEnv(tb testing.TB) *glEnv {
	tb.Helper()
	ctx, err := gltest.New()
	if err != nil {
		tb.Skipf("no GL context available: %v", err)
	}
	intf := gl.MakeNativeInterface()
	if intf == nil {
		ctx.Destroy()
		tb.Fatal("MakeNativeInterface returned nil with a current context")
	}
	env := &glEnv{ctx: ctx, intf: intf}
	tb.Cleanup(env.destroy)
	return env
}

func (e *glEnv) destroy() {
	if e.ctx != nil {
		e.ctx.Destroy()
		e.ctx = nil
	}
}

func (e *glEnv) checkNoError(tb testing.TB, stage string) {
	tb.Helper()
	if err := e.intf.Functions.GetError(); err != gl.NO_ERROR {
		tb.Fatalf("%s: glGetError = 0x%04X", stage, err)
	}
}

func (e *glEnv) glslVersionDecl(tb testing.TB) string {
	tb.Helper()
	ver := gl.GetVersionFromString(e.intf.Functions.GetStringGo(gl.VERSION))
	if ver >= gl.Ver(4, 1) {
		return "#version 410 core\n"
	}
	return "#version 150\n"
}

func (e *glEnv) compileShader(tb testing.TB, kind uint32, src string) uint32 {
	tb.Helper()
	f := &e.intf.Functions
	shader := f.CreateShader(kind)
	f.ShaderSourceStrings(shader, []string{src})
	f.CompileShader(shader)
	var status int32
	f.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == 0 {
		tb.Fatalf("shader compile failed:\n%s\nsource:\n%s", f.GetShaderInfoLogGo(shader), src)
	}
	return shader
}

// makeOffscreenFBO builds a 64x64 RGBA8 renderbuffer-backed FBO and binds it. Rendering to a wrapped FBO id is exactly
// what unison's backend-render-target path does; here we own the FBO since the context is headless.
func (e *glEnv) makeOffscreenFBO(tb testing.TB, size int32) (fbo, rb uint32) {
	tb.Helper()
	f := &e.intf.Functions
	f.GenRenderbuffers(1, &rb)
	f.BindRenderbuffer(gl.RENDERBUFFER, rb)
	f.RenderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, size, size)
	f.GenFramebuffers(1, &fbo)
	f.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	f.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb)
	if status := f.CheckFramebufferStatus(gl.FRAMEBUFFER); status != gl.FRAMEBUFFER_COMPLETE {
		tb.Fatalf("framebuffer incomplete: 0x%04X", status)
	}
	f.Viewport(0, 0, size, size)
	e.checkNoError(tb, "fbo setup")
	return fbo, rb
}

func TestSpikeInterfaceAssemblesAndValidates(t *testing.T) {
	env := newGLEnv(t)
	f := &env.intf.Functions

	if env.intf.Standard != gl.StandardGL {
		t.Fatalf("standard = %v, want OpenGL", env.intf.Standard)
	}
	verStr := f.GetStringGo(gl.VERSION)
	ver := gl.GetVersionFromString(verStr)
	if ver < gl.Ver(3, 2) {
		t.Fatalf("GL version %v below the desktop core-profile floor (GL_VERSION %q)", ver, verStr)
	}
	if !env.intf.Validate() {
		t.Fatal("interface failed validation")
	}
	t.Logf("GL_VERSION    %s", verStr)
	t.Logf("GL_RENDERER   %s", f.GetStringGo(gl.RENDERER))
	t.Logf("GL_VENDOR     %s", f.GetStringGo(gl.VENDOR))
	t.Logf("GLSL          %s", f.GetStringGo(gl.SHADING_LANGUAGE_VERSION))
	t.Logf("extensions    %d", len(env.intf.Extensions.All()))
	env.checkNoError(t, "info queries")
}

func TestValidateDetectsMissingProc(t *testing.T) {
	env := newGLEnv(t)
	if !env.intf.Validate() {
		t.Fatal("fresh interface failed validation")
	}
	broken := gl.MakeNativeInterface()
	if broken == nil {
		t.Fatal("second MakeNativeInterface returned nil")
	}
	broken.ZeroRequiredProcForTest()
	if broken.Validate() {
		t.Fatal("Validate passed with a required proc missing")
	}
	var empty gl.Interface
	if empty.Validate() {
		t.Fatal("Validate passed on a zero Interface")
	}
}

func TestSpikeClearDrawReadback(t *testing.T) {
	env := newGLEnv(t)
	f := &env.intf.Functions
	const size = 64
	env.makeOffscreenFBO(t, size)

	// Clear to a known color; ClearColor goes through the RegisterFunc float path.
	f.ClearColor(0.2, 0.4, 0.6, 1)
	f.Clear(gl.COLOR_BUFFER_BIT)
	env.checkNoError(t, "clear")

	// One program: solid color from a uniform set through the mixed int/float RegisterFunc path.
	decl := env.glslVersionDecl(t)
	vs := env.compileShader(t, gl.VERTEX_SHADER, decl+
		"in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n")
	fs := env.compileShader(t, gl.FRAGMENT_SHADER, decl+
		"uniform vec4 uColor;\nout vec4 fragColor;\nvoid main() { fragColor = uColor; }\n")
	program := f.CreateProgram()
	f.AttachShader(program, vs)
	f.AttachShader(program, fs)
	f.BindAttribLocationString(program, 0, "pos")
	f.LinkProgram(program)
	var linked int32
	f.GetProgramiv(program, gl.LINK_STATUS, &linked)
	if linked == 0 {
		t.Fatalf("program link failed:\n%s", f.GetProgramInfoLogGo(program))
	}
	f.DeleteShader(vs)
	f.DeleteShader(fs)
	env.checkNoError(t, "program build")

	// One VBO under one VAO: a triangle strip covering the left half of the target (x in [-1, 0]).
	var vao, vbo uint32
	f.GenVertexArrays(1, &vao)
	f.BindVertexArray(vao)
	f.GenBuffers(1, &vbo)
	f.BindBuffer(gl.ARRAY_BUFFER, vbo)
	verts := []float32{-1, -1, 0, -1, -1, 1, 0, 1}
	f.BufferData(gl.ARRAY_BUFFER, len(verts)*4, uintptr(ptr(&verts[0])), gl.STATIC_DRAW)
	f.EnableVertexAttribArray(0)
	f.VertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
	env.checkNoError(t, "geometry upload")

	// One draw.
	f.UseProgram(program)
	loc := f.GetUniformLocationString(program, "uColor")
	if loc < 0 {
		t.Fatal("uColor uniform not found")
	}
	f.Uniform4f(loc, 1, 0, 0, 1)
	f.DrawArrays(gl.TRIANGLE_STRIP, 0, 4)
	env.checkNoError(t, "draw")

	// Read back and verify: left half red, right half the clear color.
	pixels := make([]byte, size*size*4)
	f.PixelStorei(gl.PACK_ALIGNMENT, 1)
	f.ReadPixels(0, 0, size, size, gl.RGBA, gl.UNSIGNED_BYTE, uintptr(ptr(&pixels[0])))
	env.checkNoError(t, "readback")

	checkPixel := func(x, y int, want [4]byte, what string) {
		t.Helper()
		i := (y*size + x) * 4
		got := [4]byte{pixels[i], pixels[i+1], pixels[i+2], pixels[i+3]}
		for c := 0; c < 4; c++ {
			d := int(got[c]) - int(want[c])
			if d < -1 || d > 1 {
				t.Fatalf("%s: pixel (%d,%d) = %v, want %v (±1)", what, x, y, got, want)
			}
		}
	}
	red := [4]byte{255, 0, 0, 255}
	clearColor := [4]byte{51, 102, 153, 255}
	for _, y := range []int{1, size / 2, size - 2} {
		checkPixel(2, y, red, "left half (drawn rect)")
		checkPixel(size/2-2, y, red, "left half (drawn rect)")
		checkPixel(size/2+2, y, clearColor, "right half (clear)")
		checkPixel(size-2, y, clearColor, "right half (clear)")
	}

	f.DeleteBuffers(1, &vbo)
	f.DeleteVertexArrays(1, &vao)
	f.DeleteProgram(program)
	env.checkNoError(t, "cleanup")
}

// TestSpikeGLInfoDump prints the caps-dump block that gets checked into gldumps/. Run with -v to see it; the checked-in
// copies are regenerated by hand when drivers change.
func TestSpikeGLInfoDump(t *testing.T) {
	env := newGLEnv(t)
	f := &env.intf.Functions
	t.Logf("GL_VERSION: %s", f.GetStringGo(gl.VERSION))
	t.Logf("GL_RENDERER: %s", f.GetStringGo(gl.RENDERER))
	t.Logf("GL_VENDOR: %s", f.GetStringGo(gl.VENDOR))
	t.Logf("GL_SHADING_LANGUAGE_VERSION: %s", f.GetStringGo(gl.SHADING_LANGUAGE_VERSION))
	var maxTex, maxRB, maxSamples int32
	f.GetIntegerv(gl.MAX_TEXTURE_SIZE, &maxTex)
	f.GetIntegerv(gl.MAX_RENDERBUFFER_SIZE, &maxRB)
	f.GetIntegerv(gl.MAX_SAMPLES, &maxSamples)
	t.Logf("MAX_TEXTURE_SIZE: %d", maxTex)
	t.Logf("MAX_RENDERBUFFER_SIZE: %d", maxRB)
	t.Logf("MAX_SAMPLES: %d", maxSamples)
	for _, ext := range env.intf.Extensions.All() {
		t.Logf("EXT: %s", ext)
	}
}

// The FFI-overhead gate: measure per-call cost of the SyscallN fast path (0/1/4 integer args) and the RegisterFunc
// float path, then relate them to the worst-case per-frame GL call volume.
func BenchmarkFFISyscallGetError(b *testing.B) {
	env := newGLEnv(b)
	f := &env.intf.Functions
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		f.GetError()
	}
}

func BenchmarkFFISyscallIsTexture(b *testing.B) {
	env := newGLEnv(b)
	f := &env.intf.Functions
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		f.IsTexture(0)
	}
}

func BenchmarkFFISyscallScissor(b *testing.B) {
	env := newGLEnv(b)
	f := &env.intf.Functions
	env.makeOffscreenFBO(b, 64)
	f.Enable(gl.SCISSOR_TEST)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		f.Scissor(0, 0, 64, 64)
	}
	b.StopTimer()
	f.Disable(gl.SCISSOR_TEST)
}

func BenchmarkFFIRegisterFuncClearColor(b *testing.B) {
	env := newGLEnv(b)
	f := &env.intf.Functions
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		f.ClearColor(0, 0, 0, 1)
	}
}

func BenchmarkFFIRegisterFuncUniform4f(b *testing.B) {
	env := newGLEnv(b)
	f := &env.intf.Functions
	env.makeOffscreenFBO(b, 64)
	decl := env.glslVersionDecl(b)
	vs := env.compileShader(b, gl.VERTEX_SHADER, decl+
		"in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n")
	fs := env.compileShader(b, gl.FRAGMENT_SHADER, decl+
		"uniform vec4 uColor;\nout vec4 fragColor;\nvoid main() { fragColor = uColor; }\n")
	program := f.CreateProgram()
	f.AttachShader(program, vs)
	f.AttachShader(program, fs)
	f.LinkProgram(program)
	f.UseProgram(program)
	loc := f.GetUniformLocationString(program, "uColor")
	if loc < 0 {
		b.Fatal("uColor uniform not found")
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		f.Uniform4f(loc, 1, 0, 0, 1)
	}
}
