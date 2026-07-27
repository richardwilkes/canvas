// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the generated GL entry-point wrappers' argument marshaling.

package gl

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Captured arguments of the last glWindowRectanglesEXT dispatch, plus the shared callback that records them (purego
// never frees callbacks, so the callback is created once for the life of the test binary).
var (
	recWindowRectsArgs [3]uintptr
	recWindowRectsProc uintptr
)

func windowRectanglesCaptureProc() uintptr {
	if recWindowRectsProc == 0 {
		recWindowRectsProc = purego.NewCallback(func(mode, count, box uintptr) uintptr {
			recWindowRectsArgs = [3]uintptr{mode, count, box}
			return 0
		})
	}
	return recWindowRectsProc
}

// TestWindowRectanglesPassesBoxPointer pins the box argument as a pointer: glWindowRectanglesEXT(GLenum, GLsizei,
// const GLint *box) dereferences its third argument, so passing an int32 by value would have the driver treat the
// value as an address.
func TestWindowRectanglesPassesBoxPointer(t *testing.T) {
	f := &Functions{}
	f.windowRectangles = windowRectanglesCaptureProc()

	// The disable form the GPU backend uses: no rectangles, so no array.
	recWindowRectsArgs = [3]uintptr{}
	f.WindowRectangles(EXCLUSIVE, 0, nil)
	if recWindowRectsArgs[0] != uintptr(EXCLUSIVE) || recWindowRectsArgs[1] != 0 ||
		recWindowRectsArgs[2] != 0 {
		t.Fatalf("disable dispatch = %v, want (EXCLUSIVE, 0, nil)", recWindowRectsArgs)
	}

	// Two rectangles of four GLints each: the callee must receive the array's address, and reading through it must
	// yield the caller's values.
	boxes := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	f.WindowRectangles(INCLUSIVE, int32(len(boxes)/4), &boxes[0])
	if recWindowRectsArgs[0] != uintptr(INCLUSIVE) || recWindowRectsArgs[1] != 2 {
		t.Fatalf("dispatch = %v, want (INCLUSIVE, 2, &boxes[0])", recWindowRectsArgs)
	}
	if recWindowRectsArgs[2] != uintptr(unsafe.Pointer(&boxes[0])) {
		t.Fatalf("box argument = %#x, want the array address %#x", recWindowRectsArgs[2],
			uintptr(unsafe.Pointer(&boxes[0])))
	}
	got := *(*[8]int32)(ptrFromUintptr(recWindowRectsArgs[2]))
	if got != [8]int32{1, 2, 3, 4, 5, 6, 7, 8} {
		t.Fatalf("box array read back as %v, want the caller's values", got)
	}
	runtime.KeepAlive(boxes)

	if got := f.callCounts[idxWindowRectangles]; got != 2 {
		t.Fatalf("call count = %d, want 2", got)
	}
}
