// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Resolves the native GL entry points on Linux through glXGetProcAddress, requiring a current GLX context. Desktop
// trim: unison creates contexts via GLX on Linux, so no EGL native interface maker is provided here.

package gl

import (
	"unsafe"

	"github.com/ebitengine/purego"
)

// MakeNativeInterface assembles an Interface for the GL context current on this thread by resolving entry points
// through glXGetProcAddress.
func MakeNativeInterface() *Interface {
	lib, err := purego.Dlopen("libGL.so.1", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
	if err != nil {
		if lib, err = purego.Dlopen("libGL.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL); err != nil {
			return nil
		}
	}
	getCurrentContext, err := purego.Dlsym(lib, "glXGetCurrentContext")
	if err != nil || getCurrentContext == 0 {
		return nil
	}
	if rawCall(getCurrentContext) == 0 {
		return nil
	}
	getProcAddress, err := purego.Dlsym(lib, "glXGetProcAddress")
	if err != nil || getProcAddress == 0 {
		if getProcAddress, err = purego.Dlsym(lib, "glXGetProcAddressARB"); err != nil || getProcAddress == 0 {
			return nil
		}
	}
	return MakeAssembledInterface(func(name string) uintptr {
		// Avoid calling glXGetProcAddress() for EGL procs. We don't expect it to ever succeed, but sometimes it returns
		// non-null anyway.
		if len(name) >= 3 && name[:3] == "egl" {
			return 0
		}
		cname := cString(name)
		addr := rawCall(getProcAddress, uintptr(unsafe.Pointer(cname)))
		keepAlive(cname)
		return addr
	})
}
