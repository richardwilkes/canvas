// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

// cstr returns a pointer (as uintptr) to a NUL-terminated copy of s that stays alive for the test process. Only for the
// fake-driver callbacks below.
var cstrPins [][]byte

func cstr(s string) uintptr {
	b := append([]byte(s), 0)
	cstrPins = append(cstrPins, b)
	return uintptr(unsafe.Pointer(&b[0]))
}

// fakeDriver builds NewCallback-backed getString/getStringi/getIntegerv procs, exercising the same FFI boundary Init
// uses against a real driver.
type fakeDriver struct {
	version    string
	extensions []string
	indexed    bool
}

// As with the shared caps procs in caps_test.go: purego callbacks are never freed and the process has a hard cap on
// them, so the extension procs are created once and dispatch to whichever driver procs() installed here last.
var (
	extFakeCurrent   *fakeDriver
	extFakeProcsOnce sync.Once
	extGetString     uintptr
	extGetStringi    uintptr
	extGetIntegerv   uintptr
)

func (d *fakeDriver) procs() (getString, getStringi, getIntegerv uintptr) {
	extFakeProcsOnce.Do(func() {
		extGetString = purego.NewCallback(func(name uintptr) uintptr {
			drv := extFakeCurrent
			switch uint32(name) {
			case VERSION:
				return cstr(drv.version)
			case EXTENSIONS:
				if drv.indexed {
					return 0
				}
				all := ""
				for i, e := range drv.extensions {
					if i > 0 {
						all += " "
					}
					all += e
				}
				return cstr(all)
			}
			return 0
		})
		extGetStringi = purego.NewCallback(func(name, index uintptr) uintptr {
			drv := extFakeCurrent
			if uint32(name) == EXTENSIONS && drv.indexed && int(index) < len(drv.extensions) {
				return cstr(drv.extensions[int(index)])
			}
			return 0
		})
		extGetIntegerv = purego.NewCallback(func(pname, params uintptr) uintptr {
			if uint32(pname) == NUM_EXTENSIONS {
				*(*int32)(ptrFromUintptr(params)) = int32(len(extFakeCurrent.extensions))
			}
			return 0
		})
	})
	extFakeCurrent = d
	return extGetString, extGetStringi, extGetIntegerv
}

func TestExtensionsInitIndexed(t *testing.T) {
	d := &fakeDriver{
		version:    "4.1 Fake Driver",
		extensions: []string{"GL_Z_last", "GL_ARB_sync", "GL_A_first"},
		indexed:    true,
	}
	getString, getStringi, getIntegerv := d.procs()
	var e Extensions
	if !e.Init(StandardGL, getString, getStringi, getIntegerv, 0, EGLNoDisplay) {
		t.Fatal("Init failed")
	}
	if !e.IsInitialized() {
		t.Fatal("not initialized after Init")
	}
	for _, ext := range d.extensions {
		if !e.Has(ext) {
			t.Errorf("missing %s", ext)
		}
	}
	if e.Has("GL_absent") {
		t.Error("Has(absent) = true")
	}
	all := e.All()
	for i := 1; i < len(all); i++ {
		if all[i-1] >= all[i] {
			t.Fatalf("extension list not sorted: %v", all)
		}
	}
}

func TestExtensionsInitNonIndexed(t *testing.T) {
	// A pre-3.0 version string forces the space-separated GL_EXTENSIONS path.
	d := &fakeDriver{
		version:    "2.1 Fake Driver",
		extensions: []string{"GL_EXT_framebuffer_object", "GL_ARB_texture_storage"},
	}
	getString, getStringi, getIntegerv := d.procs()
	var e Extensions
	if !e.Init(StandardGL, getString, getStringi, getIntegerv, 0, EGLNoDisplay) {
		t.Fatal("Init failed")
	}
	for _, ext := range d.extensions {
		if !e.Has(ext) {
			t.Errorf("missing %s", ext)
		}
	}
}

func TestExtensionsInitRejectsBadVersion(t *testing.T) {
	d := &fakeDriver{version: "garbage"}
	getString, getStringi, getIntegerv := d.procs()
	var e Extensions
	if e.Init(StandardGL, getString, getStringi, getIntegerv, 0, EGLNoDisplay) {
		t.Fatal("Init succeeded with an unparseable version")
	}
	if e.Init(StandardGL, 0, getStringi, getIntegerv, 0, EGLNoDisplay) {
		t.Fatal("Init succeeded with no getString proc")
	}
}

func TestExtensionsAddRemove(t *testing.T) {
	var e Extensions
	e.Add("GL_B")
	e.Add("GL_A")
	e.Add("GL_C")
	e.Add("GL_B") // duplicate is a no-op
	if got := e.All(); len(got) != 3 || got[0] != "GL_A" || got[1] != "GL_B" || got[2] != "GL_C" {
		t.Fatalf("All() = %v", got)
	}
	if !e.Remove("GL_B") || e.Has("GL_B") {
		t.Fatal("Remove failed")
	}
	if e.Remove("GL_B") {
		t.Fatal("Remove of absent extension reported success")
	}
	if !e.Has("GL_A") || !e.Has("GL_C") {
		t.Fatal("Remove disturbed other entries")
	}
}
