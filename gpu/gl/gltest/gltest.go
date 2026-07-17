// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package gltest creates headless desktop-GL contexts for the GPU test suites, without cgo. It stands in for the
// context unison provides in production ("using unison-created contexts"): tests bring their own context exactly the
// way unison will, then hand the current context to gpu/gl. The darwin (CGL), Linux (GLX over Xvfb/llvmpipe), and
// Windows (WGL over a Mesa3D opengl32.dll drop-in) legs are all implemented; when no core-profile context can be
// created (headless host, or a Windows runner with only the stock GL 1.1 opengl32.dll), New returns an error and the
// GPU tests skip.
package gltest

// Context is a headless GL context that is current on the calling goroutine's OS thread from New until Destroy. The OS
// thread is locked for that whole span, since GL binds the context to the thread.
type Context struct {
	platform platformContext
}

// New creates a desktop core-profile GL context (the highest of 4.1/3.2 available) and makes it current. The calling
// goroutine is locked to its OS thread until Destroy.
func New() (*Context, error) {
	c := &Context{}
	if err := c.platform.init(); err != nil {
		return nil, err
	}
	return c, nil
}

// Destroy releases the context and unlocks the OS thread.
func (c *Context) Destroy() {
	c.platform.destroy()
}
