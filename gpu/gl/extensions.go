// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The sorted extension-name set queried from the context at interface-assembly time. Init receives raw proc addresses
// since assembly runs before a Functions table exists.

package gl

import (
	"sort"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
)

// EGLNoDisplay is the sentinel for "no EGL display".
const EGLNoDisplay uintptr = 0

// Extensions is a sorted list of extension names supporting binary-search lookup.
type Extensions struct {
	strings     []string
	initialized bool
}

// IsInitialized reports whether Init has succeeded.
func (e *Extensions) IsInitialized() bool { return e.initialized }

// Init populates the extension set. getString/getStringi/getIntegerv/queryString are raw proc addresses (zero when
// unavailable); eglDisplay is the display passed to queryString.
func (e *Extensions) Init(standard Standard, getString, getStringi, getIntegerv, queryString, eglDisplay uintptr) bool {
	e.initialized = false
	e.strings = nil

	if getString == 0 {
		return false
	}

	verString := goString(rawCall(getString, VERSION))
	version := GetVersionFromString(verString)
	if version == InvalidVersion {
		return false
	}

	indexed := false
	switch standard {
	case StandardGL, StandardGLES:
		// glGetStringi and indexed extensions were added in version 3.0 of desktop GL and ES.
		indexed = version >= Ver(3, 0)
	case StandardWebGL:
		indexed = version >= Ver(2, 0)
	}

	if indexed {
		if getStringi == 0 || getIntegerv == 0 {
			return false
		}
		var extensionCnt int32
		rawCall(getIntegerv, NUM_EXTENSIONS, uintptr(unsafe.Pointer(&extensionCnt)))
		e.strings = make([]string, extensionCnt)
		for i := int32(0); i < extensionCnt; i++ {
			e.strings[i] = goString(rawCall(getStringi, EXTENSIONS, uintptr(i)))
		}
	} else {
		all := goString(rawCall(getString, EXTENSIONS))
		if all == "" {
			return false
		}
		e.strings = strings.Fields(all)
	}
	if queryString != 0 {
		e.strings = append(e.strings, strings.Fields(goString(rawCall(queryString, eglDisplay, EGL_EXTENSIONS)))...)
	}
	sort.Strings(e.strings)
	e.initialized = true
	return true
}

// Has reports whether ext is present in the extension set.
func (e *Extensions) Has(ext string) bool {
	idx := sort.SearchStrings(e.strings, ext)
	return idx < len(e.strings) && e.strings[idx] == ext
}

// Remove deletes ext from the extension set, reporting whether it was present.
func (e *Extensions) Remove(ext string) bool {
	idx := sort.SearchStrings(e.strings, ext)
	if idx >= len(e.strings) || e.strings[idx] != ext {
		return false
	}
	e.strings = append(e.strings[:idx], e.strings[idx+1:]...)
	return true
}

// Add inserts ext into the extension set if not already present.
func (e *Extensions) Add(ext string) {
	idx := sort.SearchStrings(e.strings, ext)
	if idx >= len(e.strings) || e.strings[idx] != ext {
		e.strings = append(e.strings, "")
		copy(e.strings[idx+1:], e.strings[idx:])
		e.strings[idx] = ext
	}
}

// All returns the sorted extension names (for caps dumps and tests).
func (e *Extensions) All() []string {
	out := make([]string, len(e.strings))
	copy(out, e.strings)
	return out
}

// rawCall invokes a proc address before a Functions table exists (extension/version queries during assembly).
// Integer/pointer arguments only. The //go:uintptrescapes tag is the same GC-stack contract the interface.go wrappers
// carry: a Go pointer handed in as uintptr(unsafe.Pointer(x)) at a rawCall call site (e.g. Extensions.Init's &
// extensionCnt) must be kept alive and heap-resident for the duration of the call. SyscallN's own tag does not cover it
// — the pragma only applies to conversions written at the tagged function's direct call site — so without this tag the
// pointee stays on the stack, and any stack growth during the call (SyscallN's own prologue, or a purego-callback fake
// driver in the tests re-entering Go) moves the stack and strands the callee's write on the old copy.
//
//go:uintptrescapes
func rawCall(fn uintptr, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(fn, args...)
	return r
}
