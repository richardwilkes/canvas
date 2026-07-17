// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Headless CGL context creation via purego (no cgo). CGL is the only macOS GL context API reachable without
// Objective-C, and it happily creates offscreen core-profile contexts: no drawable is attached, so all rendering goes
// to framebuffer objects — which is exactly what this package's GPU test suites need.

package gltest

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

const openGLFrameworkPath = "/System/Library/Frameworks/OpenGL.framework/OpenGL"

// CGLPixelFormatAttribute values (CGLTypes.h).
const (
	cglPFAOpenGLProfile  = 99
	cglPFARendererID     = 70
	cglOGLPVersion32Core = 0x3200
	cglOGLPVersion41Core = 0x4100
	cglErrorNone         = 0

	// kCGLRendererGenericFloatID (CGLRenderers.h): Apple's software renderer, selectable via
	// CANVAS_GLTEST_RENDERER=software to approximate non-hardware GL stacks (CI VMs) locally.
	cglRendererGenericFloatID = 0x00020400
)

type platformContext struct {
	choosePixelFormat  uintptr
	destroyPixelFormat uintptr
	createContext      uintptr
	destroyContext     uintptr
	setCurrentContext  uintptr
	pixelFormat        uintptr
	context            uintptr
	locked             bool
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

	lib, err := purego.Dlopen(openGLFrameworkPath, purego.RTLD_LAZY)
	if err != nil {
		return fmt.Errorf("gltest: dlopen OpenGL framework: %w", err)
	}
	for _, proc := range []struct {
		dst  *uintptr
		name string
	}{
		{dst: &p.choosePixelFormat, name: "CGLChoosePixelFormat"},
		{dst: &p.destroyPixelFormat, name: "CGLDestroyPixelFormat"},
		{dst: &p.createContext, name: "CGLCreateContext"},
		{dst: &p.destroyContext, name: "CGLDestroyContext"},
		{dst: &p.setCurrentContext, name: "CGLSetCurrentContext"},
	} {
		if *proc.dst, err = purego.Dlsym(lib, proc.name); err != nil {
			return fmt.Errorf("gltest: dlsym %s: %w", proc.name, err)
		}
	}

	// Prefer a 4.1 core profile (what unison gets on modern macOS); fall back to 3.2 core.
	// CANVAS_GLTEST_RENDERER=software pins Apple's software renderer so driver-stack differences (CI VMs render on
	// non-hardware GL) can be approximated locally.
	var extraAttribs []int32
	if os.Getenv("CANVAS_GLTEST_RENDERER") == "software" {
		extraAttribs = []int32{cglPFARendererID, cglRendererGenericFloatID}
	}
	for _, profile := range []int32{cglOGLPVersion41Core, cglOGLPVersion32Core} {
		attribs := append([]int32{cglPFAOpenGLProfile, profile}, extraAttribs...)
		attribs = append(attribs, 0)
		var npix int32
		r, _, _ := purego.SyscallN(p.choosePixelFormat,
			uintptr(unsafe.Pointer(&attribs[0])), uintptr(unsafe.Pointer(&p.pixelFormat)), uintptr(unsafe.Pointer(&npix)))
		if int32(r) == cglErrorNone && p.pixelFormat != 0 {
			break
		}
		p.pixelFormat = 0
	}
	if p.pixelFormat == 0 {
		return fmt.Errorf("gltest: CGLChoosePixelFormat found no core-profile pixel format")
	}
	if r, _, _ := purego.SyscallN(p.createContext, p.pixelFormat, 0, uintptr(unsafe.Pointer(&p.context))); int32(r) != cglErrorNone || p.context == 0 {
		return fmt.Errorf("gltest: CGLCreateContext failed (CGLError %d)", int32(r))
	}
	if r, _, _ := purego.SyscallN(p.setCurrentContext, p.context); int32(r) != cglErrorNone {
		return fmt.Errorf("gltest: CGLSetCurrentContext failed (CGLError %d)", int32(r))
	}
	ok = true
	return nil
}

func (p *platformContext) destroy() {
	if p.context != 0 {
		purego.SyscallN(p.setCurrentContext, 0)
		purego.SyscallN(p.destroyContext, p.context)
		p.context = 0
	}
	if p.pixelFormat != 0 {
		purego.SyscallN(p.destroyPixelFormat, p.pixelFormat)
		p.pixelFormat = 0
	}
	if p.locked {
		runtime.UnlockOSThread()
		p.locked = false
	}
}
