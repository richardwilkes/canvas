// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Headless WGL context creation via the standard library's syscall package (no cgo, no purego — Windows resolves GL
// through std syscall, matching gpu/gl/native_windows.go). This is the Windows counterpart of context_darwin.go's CGL
// leg and context_linux.go's GLX leg: unison creates its GL contexts through WGL on Windows, so the test context is
// built the same way. It creates an offscreen core-profile context on a hidden 1x1 window's device context — no window
// is ever shown, and the GPU backend renders to framebuffer objects, so the window's back buffer is unused — which
// matches how this package's GPU test suites use the context.
//
// WGL bootstrap: a modern (core-profile) context can only be created through wglCreateContextAttribsARB, which is
// itself an extension entry point that opengl32.dll does not export — it must be resolved via wglGetProcAddress, and
// that in turn requires a current legacy context. So init() creates a throwaway legacy context first, resolves
// wglCreateContextAttribsARB, then creates the real 4.1-then-3.2 core context (the desktop-trim range) and deletes the
// legacy one, mirroring the darwin/linux 4.1-then-3.2 selection.
//
// The GL stack: a stock Windows runner's opengl32.dll is the Microsoft software rasterizer, which exposes only GL 1.1
// and no WGL_ARB_create_context — so wglGetProcAddress("wglCreateContextAttribsARB") returns NULL, init() returns an
// error, and the GPU tests skip (graceful degradation, exactly as on a headless host). A usable core-profile GL comes
// from a Mesa3D opengl32.dll drop-in (llvmpipe software GL 3.3/4.x) placed next to the executable — opengl32.dll is not
// a KnownDLL, so the app-directory copy is loaded in preference to System32's. See the "Provision Mesa3D" steps in
// .github/workflows/build.yml and capture-goldens.yml for the CI-provisioning notes.
//
// Software-renderer selection: unlike CGL (which takes a renderer-id pixel-format attribute), WGL has no "use the
// software rasterizer" attribute, so CANVAS_GLTEST_RENDERER=software (honored by the darwin leg) is a no-op here,
// exactly as on Linux — which opengl32.dll is loaded governs (Mesa's drop-in is the software renderer).

package gltest

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// Win32 window-class / window-style constants (winuser.h).
const (
	csOwnDC = 0x0020     // CS_OWNDC: a stable private DC per window (recommended for OpenGL).
	wsPopup = 0x80000000 // WS_POPUP: a frameless window; never shown, so it stays invisible.
)

// PIXELFORMATDESCRIPTOR flags and types (wingdi.h).
const (
	pfdDrawToWindow  = 0x00000004
	pfdSupportOpenGL = 0x00000020
	pfdDoubleBuffer  = 0x00000001
	pfdTypeRGBA      = 0
	pfdMainPlane     = 0
)

// WGL_ARB_create_context attribute values (wglext.h). These are part of the stable WGL ABI.
const (
	wglContextMajorVersionARB   = 0x2091
	wglContextMinorVersionARB   = 0x2092
	wglContextProfileMaskARB    = 0x9126
	wglContextCoreProfileBitARB = 0x00000001
)

// pixelFormatDescriptor mirrors the Win32 PIXELFORMATDESCRIPTOR (wingdi.h) byte-for-byte: 40 bytes on every arch (a
// WORD/WORD/DWORD header, 20 packed BYTE fields, then three DWORD masks). The layout is fixed by the ABI, so the field
// order and sizes must not change.
type pixelFormatDescriptor struct {
	nSize           uint16
	nVersion        uint16
	dwFlags         uint32
	iPixelType      byte
	cColorBits      byte
	cRedBits        byte
	cRedShift       byte
	cGreenBits      byte
	cGreenShift     byte
	cBlueBits       byte
	cBlueShift      byte
	cAlphaBits      byte
	cAlphaShift     byte
	cAccumBits      byte
	cAccumRedBits   byte
	cAccumGreenBits byte
	cAccumBlueBits  byte
	cAccumAlphaBits byte
	cDepthBits      byte
	cStencilBits    byte
	cAuxBuffers     byte
	iLayerType      byte
	bReserved       byte
	dwLayerMask     uint32
	dwVisibleMask   uint32
	dwDamageMask    uint32
}

// wndClassExW mirrors the Win32 WNDCLASSEXW (winuser.h): 80 bytes on 64-bit. The layout is fixed by the ABI.
type wndClassExW struct {
	cbSize        uint32
	style         uint32
	lpfnWndProc   uintptr
	cbClsExtra    int32
	cbWndExtra    int32
	hInstance     uintptr
	hIcon         uintptr
	hCursor       uintptr
	hbrBackground uintptr
	lpszMenuName  *uint16
	lpszClassName *uint16
	hIconSm       uintptr
}

// classCounter makes each context's window-class name unique within the process, so repeated New()/Destroy() cycles
// never collide on RegisterClassExW (which fails if a class name is already registered).
var classCounter uint64

// The window procedure is stateless (it forwards everything to DefWindowProcW), so a single process-wide callback is
// created once and shared by every context, rather than leaking a syscall.NewCallback slot per New().
var (
	wndProcOnce        sync.Once
	wndProcCB          uintptr
	defWindowProcWAddr uintptr
)

func windowProcCallback() uintptr {
	wndProcOnce.Do(func() {
		user32, err := syscall.LoadLibrary("user32.dll")
		if err != nil {
			return
		}
		if defWindowProcWAddr, err = syscall.GetProcAddress(user32, "DefWindowProcW"); err != nil {
			return
		}
		wndProcCB = syscall.NewCallback(func(hwnd, msg, wParam, lParam uintptr) uintptr {
			if defWindowProcWAddr == 0 {
				return 0
			}
			r, _, _ := syscall.SyscallN(defWindowProcWAddr, hwnd, msg, wParam, lParam)
			return r
		})
	})
	return wndProcCB
}

type platformContext struct {
	// Resolved user32 / gdi32 / opengl32 entry points.
	getModuleHandleW  uintptr
	registerClassExW  uintptr
	unregisterClassW  uintptr
	createWindowExW   uintptr
	destroyWindow     uintptr
	getDC             uintptr
	releaseDC         uintptr
	choosePixelFormat uintptr
	setPixelFormat    uintptr
	wglCreateContext  uintptr
	wglMakeCurrent    uintptr
	wglDeleteContext  uintptr
	wglGetProcAddress uintptr

	hInstance uintptr
	className *uint16 // kept alive for UnregisterClassW
	classAtom uintptr
	hwnd      uintptr // HWND (hidden window)
	hdc       uintptr // HDC (its device context)
	hglrc     uintptr // HGLRC (the core-profile context)
	locked    bool
}

// winCall invokes a resolved Win32/WGL entry point with integer/pointer arguments only (no by-value floats), which
// covers every function used here; syscall.SyscallN places every argument in the integer registers and has no way to
// reach the float bank, the same constraint glxRawCall carries on the Linux leg.
//
// The //go:uintptrescapes tag is the GC-stack contract, matching the tag glxRawCall carries on the Linux leg and the
// one gpu/gl puts on glCall: syscall.SyscallN's own //go:uintptrkeepalive only covers conversions written at its own
// call site and needs an all-nosplit chain, and this splittable forwarder passes an already-built []uintptr with no
// pointer provenance left. Without the tag, a Go pointer converted at a winCall call site (e.g.
// uintptr(unsafe.Pointer(&wc)) below) could stay on the stack and be moved by a stack growth before the Win32 call
// reads it, leaving the driver writing to a stale address; runtime.KeepAlive cannot substitute, since it keeps an
// object alive without pinning it in place.
//
//go:uintptrescapes
func winCall(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := syscall.SyscallN(fn, args...)
	return r
}

func (p *platformContext) init() error {
	runtime.LockOSThread()
	p.locked = true
	ok := false
	defer func() {
		if !ok {
			p.destroy()
		}
	}()

	kernel32, err := syscall.LoadLibrary("kernel32.dll")
	if err != nil {
		return fmt.Errorf("gltest: LoadLibrary kernel32: %w", err)
	}
	user32, err := syscall.LoadLibrary("user32.dll")
	if err != nil {
		return fmt.Errorf("gltest: LoadLibrary user32: %w", err)
	}
	gdi32, err := syscall.LoadLibrary("gdi32.dll")
	if err != nil {
		return fmt.Errorf("gltest: LoadLibrary gdi32: %w", err)
	}
	// opengl32.dll may be the stock Microsoft GL 1.1 rasterizer or a Mesa3D drop-in; either exports the WGL entry
	// points below. The app-directory copy (Mesa) wins over System32's because opengl32 is not a KnownDLL.
	opengl32, err := syscall.LoadLibrary("opengl32.dll")
	if err != nil {
		return fmt.Errorf("gltest: LoadLibrary opengl32: %w", err)
	}
	for _, proc := range []struct {
		dst    *uintptr
		module syscall.Handle
		name   string
	}{
		{dst: &p.getModuleHandleW, module: kernel32, name: "GetModuleHandleW"},
		{dst: &p.registerClassExW, module: user32, name: "RegisterClassExW"},
		{dst: &p.unregisterClassW, module: user32, name: "UnregisterClassW"},
		{dst: &p.createWindowExW, module: user32, name: "CreateWindowExW"},
		{dst: &p.destroyWindow, module: user32, name: "DestroyWindow"},
		{dst: &p.getDC, module: user32, name: "GetDC"},
		{dst: &p.releaseDC, module: user32, name: "ReleaseDC"},
		{dst: &p.choosePixelFormat, module: gdi32, name: "ChoosePixelFormat"},
		{dst: &p.setPixelFormat, module: gdi32, name: "SetPixelFormat"},
		{dst: &p.wglCreateContext, module: opengl32, name: "wglCreateContext"},
		{dst: &p.wglMakeCurrent, module: opengl32, name: "wglMakeCurrent"},
		{dst: &p.wglDeleteContext, module: opengl32, name: "wglDeleteContext"},
		{dst: &p.wglGetProcAddress, module: opengl32, name: "wglGetProcAddress"},
	} {
		if *proc.dst, err = syscall.GetProcAddress(proc.module, proc.name); err != nil {
			return fmt.Errorf("gltest: GetProcAddress %s: %w", proc.name, err)
		}
	}

	wndProc := windowProcCallback()
	if wndProc == 0 {
		return errors.New("gltest: could not create a window procedure callback")
	}
	p.hInstance = winCall(p.getModuleHandleW, 0)

	// Register a unique CS_OWNDC class so the window gets a stable private DC (the pixel format set on it sticks for
	// the context's lifetime). A per-process-unique name means repeated New()/Destroy() never collide.
	classNameStr := fmt.Sprintf("canvasGLTest_%d", atomic.AddUint64(&classCounter, 1))
	if p.className, err = syscall.UTF16PtrFromString(classNameStr); err != nil {
		return fmt.Errorf("gltest: UTF16 class name: %w", err)
	}
	wc := wndClassExW{
		cbSize:        uint32(unsafe.Sizeof(wndClassExW{})),
		style:         csOwnDC,
		lpfnWndProc:   wndProc,
		hInstance:     p.hInstance,
		lpszClassName: p.className,
	}
	p.classAtom = winCall(p.registerClassExW, uintptr(unsafe.Pointer(&wc)))
	runtime.KeepAlive(&wc)
	runtime.KeepAlive(p.className)
	if p.classAtom == 0 {
		return errors.New("gltest: RegisterClassExW failed")
	}

	// A hidden 1x1 popup window: never shown, and its back buffer is unused (the GPU backend renders to FBOs); it
	// exists only to own a GL-capable device context.
	p.hwnd = winCall(p.createWindowExW,
		0,                                    // dwExStyle
		uintptr(unsafe.Pointer(p.className)), // lpClassName
		0,                                    // lpWindowName (NULL)
		wsPopup,                              // dwStyle (no WS_VISIBLE)
		0, 0, 1, 1,                           // x, y, width, height
		0,           // hWndParent
		0,           // hMenu
		p.hInstance, // hInstance
		0)           // lpParam
	runtime.KeepAlive(p.className)
	if p.hwnd == 0 {
		return errors.New("gltest: CreateWindowExW failed")
	}
	if p.hdc = winCall(p.getDC, p.hwnd); p.hdc == 0 {
		return errors.New("gltest: GetDC failed")
	}

	// A double-buffered RGBA8 (+depth24/stencil8) pixel format. The GPU backend renders to its own FBOs, so the depth
	// and stencil here are unused, but they are requested for parity with the CGL/GLX legs.
	pfd := pixelFormatDescriptor{
		nSize:        uint16(unsafe.Sizeof(pixelFormatDescriptor{})),
		nVersion:     1,
		dwFlags:      pfdDrawToWindow | pfdSupportOpenGL | pfdDoubleBuffer,
		iPixelType:   pfdTypeRGBA,
		cColorBits:   32,
		cAlphaBits:   8,
		cDepthBits:   24,
		cStencilBits: 8,
		iLayerType:   pfdMainPlane,
	}
	format := winCall(p.choosePixelFormat, p.hdc, uintptr(unsafe.Pointer(&pfd)))
	if format == 0 {
		runtime.KeepAlive(&pfd)
		return errors.New("gltest: ChoosePixelFormat found no OpenGL-capable format")
	}
	if winCall(p.setPixelFormat, p.hdc, format, uintptr(unsafe.Pointer(&pfd))) == 0 {
		runtime.KeepAlive(&pfd)
		return errors.New("gltest: SetPixelFormat failed")
	}
	runtime.KeepAlive(&pfd)

	// Bootstrap: a throwaway legacy context so wglGetProcAddress can resolve wglCreateContextAttribsARB.
	legacy := winCall(p.wglCreateContext, p.hdc)
	if legacy == 0 {
		return errors.New("gltest: wglCreateContext (legacy bootstrap) failed")
	}
	if winCall(p.wglMakeCurrent, p.hdc, legacy) == 0 {
		winCall(p.wglDeleteContext, legacy)
		return errors.New("gltest: wglMakeCurrent (legacy bootstrap) failed")
	}
	procName := append([]byte("wglCreateContextAttribsARB"), 0)
	createContextAttribs := winCall(p.wglGetProcAddress, uintptr(unsafe.Pointer(&procName[0])))
	runtime.KeepAlive(procName)
	if createContextAttribs == 0 {
		// No WGL_ARB_create_context: the stock GL 1.1 software rasterizer, which cannot make a core profile. Release +
		// delete the legacy context and fail (→ skip); a Mesa3D opengl32.dll drop-in is required.
		winCall(p.wglMakeCurrent, 0, 0)
		winCall(p.wglDeleteContext, legacy)
		return errors.New("gltest: wglCreateContextAttribsARB unavailable (no core-profile GL; is a Mesa3D " +
			"opengl32.dll drop-in present?)")
	}

	// Prefer a 4.1 core profile (what unison gets on modern desktop GL), fall back to 3.2 core — the desktop trim
	// range, matching the darwin CGL and Linux GLX legs' 4.1-then-3.2 selection.
	for _, v := range [][2]int32{{4, 1}, {3, 2}} {
		ctxAttribs := []int32{
			wglContextMajorVersionARB, v[0],
			wglContextMinorVersionARB, v[1],
			wglContextProfileMaskARB, wglContextCoreProfileBitARB,
			0,
		}
		p.hglrc = winCall(createContextAttribs, p.hdc, 0, uintptr(unsafe.Pointer(&ctxAttribs[0])))
		runtime.KeepAlive(ctxAttribs)
		if p.hglrc != 0 {
			break
		}
	}
	// Switch off the legacy context and delete it whether or not the core context came up.
	winCall(p.wglMakeCurrent, 0, 0)
	winCall(p.wglDeleteContext, legacy)
	if p.hglrc == 0 {
		return errors.New("gltest: wglCreateContextAttribsARB failed for both 4.1 and 3.2 core profiles")
	}
	if winCall(p.wglMakeCurrent, p.hdc, p.hglrc) == 0 {
		return errors.New("gltest: wglMakeCurrent (core profile) failed")
	}
	ok = true
	return nil
}

func (p *platformContext) destroy() {
	if p.wglMakeCurrent != 0 {
		winCall(p.wglMakeCurrent, 0, 0)
	}
	if p.hglrc != 0 {
		winCall(p.wglDeleteContext, p.hglrc)
		p.hglrc = 0
	}
	if p.hdc != 0 && p.hwnd != 0 && p.releaseDC != 0 {
		winCall(p.releaseDC, p.hwnd, p.hdc)
		p.hdc = 0
	}
	if p.hwnd != 0 && p.destroyWindow != 0 {
		winCall(p.destroyWindow, p.hwnd)
		p.hwnd = 0
	}
	if p.classAtom != 0 && p.className != nil && p.unregisterClassW != 0 {
		winCall(p.unregisterClassW, uintptr(unsafe.Pointer(p.className)), p.hInstance)
		runtime.KeepAlive(p.className)
		p.classAtom = 0
	}
	if p.locked {
		runtime.UnlockOSThread()
		p.locked = false
	}
}
