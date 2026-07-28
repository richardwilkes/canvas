// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Resolves the native GL entry points on macOS, using purego's dlopen/dlsym to load the system OpenGL framework. The
// handle is kept open for the process lifetime, so libGL can never be unloaded out from under resolved procs — and the
// OpenGL framework is never unloaded on macOS regardless.
//
// Two deliberate differences from the linux/windows loaders: there is no explicit current-context check (dlsym resolves
// without one, and MakeAssembledGLInterface already returns nil when glGetString(GL_VERSION) comes back empty because
// no context is current), and the standard sniff in MakeAssembledInterface is skipped, since OpenGL.framework only ever
// hosts desktop GL — a CGL context is never ES or WebGL.

package gl

import "github.com/ebitengine/purego"

const macGLPath = "/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/libGL.dylib"

// MakeNativeInterface assembles an Interface for the GL context current on this thread by resolving entry points from
// the system GL library.
func MakeNativeInterface() *Interface {
	lib, err := purego.Dlopen(macGLPath, purego.RTLD_LAZY)
	if err != nil {
		return nil
	}
	return MakeAssembledGLInterface(func(name string) uintptr {
		addr, dlerr := purego.Dlsym(lib, name)
		if dlerr != nil {
			return 0
		}
		return addr
	})
}
