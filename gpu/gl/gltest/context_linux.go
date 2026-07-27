// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Headless GLX context creation via purego (no cgo). This is the Linux counterpart of context_darwin.go's CGL leg:
// unison creates its GL contexts through GLX on Linux (see gpu/gl/native_linux.go, which resolves entry points through
// glXGetProcAddress and requires a current GLX context), so the test context is built the same way. It creates an
// offscreen core-profile context bound to a 1x1 pbuffer — no window is needed, all rendering goes to framebuffer
// objects — which matches how the GPU backend uses the context.
//
// It needs a running X server (Xvfb in CI) and a GL stack that offers GLX (Mesa; llvmpipe when LIBGL_ALWAYS_SOFTWARE=1
// is set, the non-hardware GL stack CI runners use). When no display or no core-profile context is available, init
// returns an error and the GPU tests skip, exactly as they do on a headless host.
//
// Software-renderer selection: unlike CGL (which takes a renderer-id pixel-format attribute), GLX has no "use the
// software rasterizer" attribute — Mesa selects llvmpipe from the LIBGL_ALWAYS_SOFTWARE=1 environment variable, which
// CI sets for the whole job. CANVAS_GLTEST_RENDERER=software (honored by the darwin leg) is therefore a no-op here; the
// env var governs. This is documented on gpu/gl/gltest and in the CI wiring.

package gltest

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// GLX attribute/enum values (GL/glx.h and GL/glxext.h). These are part of the stable GLX ABI.
const (
	glxRenderType   = 0x8011
	glxRGBABit      = 0x00000001
	glxDrawableType = 0x8010
	glxPbufferBit   = 0x00000004
	glxRedSize      = 8
	glxGreenSize    = 9
	glxBlueSize     = 10
	glxAlphaSize    = 11
	glxDepthSize    = 12
	glxStencilSize  = 13
	glxDoubleBuffer = 5
	glxPbufferWidth = 0x8041
	glxPbufferHt    = 0x8040

	glxContextMajorVersionARB   = 0x2091
	glxContextMinorVersionARB   = 0x2092
	glxContextProfileMaskARB    = 0x9126
	glxContextCoreProfileBitARB = 0x00000001

	glxNone = 0 // X.h's None / GLX list terminator.
	xTrue   = 1 // Xlib Bool True (direct-rendering request).
)

type platformContext struct {
	// Resolved libGL / libX11 entry points.
	xOpenDisplay       uintptr
	xCloseDisplay      uintptr
	xDefaultScreen     uintptr
	xFree              uintptr
	xSetErrorHandler   uintptr
	chooseFBConfig     uintptr
	getProcAddress     uintptr
	createPbuffer      uintptr
	destroyPbuffer     uintptr
	makeContextCurrent uintptr
	destroyContext     uintptr

	display uintptr // Display*
	pbuffer uintptr // GLXPbuffer (an XID)
	context uintptr // GLXContext
	locked  bool
}

// glxRawCall invokes a resolved C entry point with integer/pointer arguments only (no by-value floats), which covers
// every GLX/Xlib function used here. The restriction is purego.SyscallN's: it mirrors the first eight argument slots
// into both the integer and the float register banks, which is only ABI-correct when no float argument needs a
// different bank position, so it cannot call a mixed integer/float signature. Such a signature has to be bound through
// purego.RegisterFunc instead, as gpu/gl does for its float-carrying entry points.
//
// The //go:uintptrescapes tag is the GC-stack contract, matching the tag gpu/gl puts on glCall and its uintptr-taking
// wrappers: purego.SyscallN carries the tag itself, but it only covers conversions written at its own call site, and
// this forwarder passes an already-built []uintptr with no pointer provenance left. Without the tag here, a Go pointer
// converted at a glxRawCall call site (e.g. uintptr(unsafe.Pointer(&nConfigs)) below) could stay on the stack and be
// moved by a stack growth before the C call reads it, leaving the driver writing to a stale address.
//
//go:uintptrescapes
func glxRawCall(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}

// ptrFromUintptr converts a C-owned pointer returned in a register (never a Go pointer) to an unsafe.Pointer, isolated
// in a helper so the conversion site is auditable — mirroring gpu/gl.ptrFromUintptr.
func ptrFromUintptr(p uintptr) unsafe.Pointer {
	//nolint:govet // the value is a C pointer (an X/GLX object) that never points into the Go heap
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// The X error handler installed during setup is stateless (it swallows the error and returns 0), so a single
// process-wide callback is created once and shared by every context, rather than leaking a purego.NewCallback slot per
// New(). purego never releases callback slots and hard-caps them at 2000, panicking once they are exhausted, so a
// per-context callback would eventually take down a process that cycles New()/Destroy() enough times. This mirrors the
// windows leg's shared window procedure.
var (
	xErrorHandlerOnce sync.Once
	xErrorHandlerCB   uintptr
)

func xErrorHandlerCallback() uintptr {
	xErrorHandlerOnce.Do(func() {
		xErrorHandlerCB = purego.NewCallback(func(_ /* Display* */, _ /* XErrorEvent* */ uintptr) uintptr { return 0 })
	})
	return xErrorHandlerCB
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

	glLib, err := purego.Dlopen("libGL.so.1", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
	if err != nil {
		if glLib, err = purego.Dlopen("libGL.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL); err != nil {
			return fmt.Errorf("gltest: dlopen libGL: %w", err)
		}
	}
	x11Lib, err := purego.Dlopen("libX11.so.6", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
	if err != nil {
		if x11Lib, err = purego.Dlopen("libX11.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL); err != nil {
			return fmt.Errorf("gltest: dlopen libX11: %w", err)
		}
	}
	for _, proc := range []struct {
		dst  *uintptr
		lib  uintptr
		name string
	}{
		{dst: &p.xOpenDisplay, lib: x11Lib, name: "XOpenDisplay"},
		{dst: &p.xCloseDisplay, lib: x11Lib, name: "XCloseDisplay"},
		{dst: &p.xDefaultScreen, lib: x11Lib, name: "XDefaultScreen"},
		{dst: &p.xFree, lib: x11Lib, name: "XFree"},
		{dst: &p.xSetErrorHandler, lib: x11Lib, name: "XSetErrorHandler"},
		{dst: &p.chooseFBConfig, lib: glLib, name: "glXChooseFBConfig"},
		{dst: &p.createPbuffer, lib: glLib, name: "glXCreatePbuffer"},
		{dst: &p.destroyPbuffer, lib: glLib, name: "glXDestroyPbuffer"},
		{dst: &p.makeContextCurrent, lib: glLib, name: "glXMakeContextCurrent"},
		{dst: &p.destroyContext, lib: glLib, name: "glXDestroyContext"},
	} {
		if *proc.dst, err = purego.Dlsym(proc.lib, proc.name); err != nil {
			return fmt.Errorf("gltest: dlsym %s: %w", proc.name, err)
		}
	}
	// glXGetProcAddressARB resolves the GLX_ARB_create_context entry point below (the modern create path).
	if p.getProcAddress, err = purego.Dlsym(glLib, "glXGetProcAddressARB"); err != nil || p.getProcAddress == 0 {
		if p.getProcAddress, err = purego.Dlsym(glLib, "glXGetProcAddress"); err != nil || p.getProcAddress == 0 {
			return fmt.Errorf("gltest: dlsym glXGetProcAddressARB: %w", err)
		}
	}

	// Ignore X protocol errors during setup so an unsupported context/pbuffer request returns NULL instead of the
	// default Xlib handler calling exit(). glXCreateContextAttribsARB is the classic offender: a rejected
	// version/profile raises a GLXBadFBConfig/BadMatch X error, which would otherwise terminate the process.
	prevHandler := glxRawCall(p.xSetErrorHandler, xErrorHandlerCallback())
	defer glxRawCall(p.xSetErrorHandler, prevHandler)

	// Open the default X display (Xvfb's in CI, via DISPLAY). No server => headless, so skip.
	p.display = glxRawCall(p.xOpenDisplay, 0)
	if p.display == 0 {
		return errors.New("gltest: XOpenDisplay(NULL) returned NULL (no X server; is DISPLAY set / Xvfb running?)")
	}
	screen := glxRawCall(p.xDefaultScreen, p.display)

	// Choose a pbuffer-capable, RGBA8 (+depth24/stencil8) framebuffer config. The pbuffer only exists so the context
	// can be made current; the GPU backend renders to its own FBOs, so the config's depth/stencil are unused.
	fbAttribs := []int32{
		glxDrawableType, glxPbufferBit,
		glxRenderType, glxRGBABit,
		glxRedSize, 8, glxGreenSize, 8, glxBlueSize, 8, glxAlphaSize, 8,
		glxDepthSize, 24, glxStencilSize, 8,
		glxDoubleBuffer, 0,
		glxNone,
	}
	var nConfigs int32
	configs := glxRawCall(p.chooseFBConfig, p.display, screen,
		uintptr(unsafe.Pointer(&fbAttribs[0])), uintptr(unsafe.Pointer(&nConfigs)))
	runtime.KeepAlive(fbAttribs)
	runtime.KeepAlive(&nConfigs)
	if configs == 0 || nConfigs == 0 {
		return errors.New("gltest: glXChooseFBConfig found no pbuffer-capable RGBA config")
	}
	fbConfig := *(*uintptr)(ptrFromUintptr(configs)) // configs[0]; the array is XFree'd once we hold the element.
	glxRawCall(p.xFree, configs)
	if fbConfig == 0 {
		return errors.New("gltest: glXChooseFBConfig returned a null config")
	}

	createCtxName := append([]byte("glXCreateContextAttribsARB"), 0)
	createContextAttribs := glxRawCall(p.getProcAddress, uintptr(unsafe.Pointer(&createCtxName[0])))
	runtime.KeepAlive(createCtxName)
	if createContextAttribs == 0 {
		return errors.New("gltest: glXCreateContextAttribsARB unavailable (no GLX_ARB_create_context)")
	}

	// Prefer a 4.1 core profile (what unison gets on modern desktop GL), fall back to 3.2 core — the desktop trim
	// range, matching the darwin CGL leg's 4.1-then-3.2 selection.
	for _, v := range [][2]int32{{4, 1}, {3, 2}} {
		ctxAttribs := []int32{
			glxContextMajorVersionARB, v[0],
			glxContextMinorVersionARB, v[1],
			glxContextProfileMaskARB, glxContextCoreProfileBitARB,
			glxNone,
		}
		p.context = glxRawCall(createContextAttribs, p.display, fbConfig, 0, xTrue,
			uintptr(unsafe.Pointer(&ctxAttribs[0])))
		runtime.KeepAlive(ctxAttribs)
		if p.context != 0 {
			break
		}
	}
	if p.context == 0 {
		return errors.New("gltest: glXCreateContextAttribsARB failed for both 4.1 and 3.2 core profiles")
	}

	// A 1x1 pbuffer solely to make the context current; all rendering targets FBOs.
	pbAttribs := []int32{glxPbufferWidth, 1, glxPbufferHt, 1, glxNone}
	p.pbuffer = glxRawCall(p.createPbuffer, p.display, fbConfig, uintptr(unsafe.Pointer(&pbAttribs[0])))
	runtime.KeepAlive(pbAttribs)
	if p.pbuffer == 0 {
		return errors.New("gltest: glXCreatePbuffer failed")
	}
	if glxRawCall(p.makeContextCurrent, p.display, p.pbuffer, p.pbuffer, p.context) == 0 {
		return errors.New("gltest: glXMakeContextCurrent failed")
	}
	ok = true
	return nil
}

func (p *platformContext) destroy() {
	if p.display != 0 {
		if p.context != 0 {
			// Release the current context before destroying it (glXMakeContextCurrent with no drawable/context).
			glxRawCall(p.makeContextCurrent, p.display, 0, 0, 0)
			glxRawCall(p.destroyContext, p.display, p.context)
			p.context = 0
		}
		if p.pbuffer != 0 {
			glxRawCall(p.destroyPbuffer, p.display, p.pbuffer)
			p.pbuffer = 0
		}
		glxRawCall(p.xCloseDisplay, p.display)
		p.display = 0
	}
	if p.locked {
		runtime.UnlockOSThread()
		p.locked = false
	}
}
