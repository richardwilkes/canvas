// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import "github.com/richardwilkes/canvas/gpu"

// ContextInfo holds information about an OpenGL context including the GL version, driver identity, the GLSL generation
// to target, and the caps derived from all of it.
type ContextInfo struct {
	Interface *Interface
	Caps      *Caps
	// Options retains the context options the context was created with.
	Options        *gpu.ContextOptions
	DriverInfo     DriverInfo
	GLSLGeneration gpu.GLSLGeneration
}

// NewContextInfo validates iface, derives the driver info and GLSL generation, and builds the caps. A GL context must
// be current on this thread. Returns nil on failed validation, an unparsable GL_VERSION, or an unparsable GLSL version.
func NewContextInfo(iface *Interface, options *gpu.ContextOptions) *ContextInfo {
	if iface == nil || !iface.Validate() {
		return nil
	}
	if options == nil {
		options = gpu.DefaultContextOptions()
	}
	info := &ContextInfo{Interface: iface, Options: options}
	info.DriverInfo = GetDriverInfo(iface)
	if info.DriverInfo.Version == InvalidVersion {
		return nil
	}
	generation, ok := glslGeneration(&info.DriverInfo)
	if !ok {
		return nil
	}
	info.GLSLGeneration = generation
	info.Caps = newCaps(options, info, iface)
	return info
}

// Standard returns the GL standard (desktop GL, ES, or WebGL) this context targets.
func (c *ContextInfo) Standard() Standard { return c.Interface.Standard }

// Version returns the context's GL version.
func (c *ContextInfo) Version() Version { return c.DriverInfo.Version }

// Vendor returns the context's vendor (sans the ANGLE preference, per the trim).
func (c *ContextInfo) Vendor() Vendor { return c.DriverInfo.Vendor }

// Renderer returns the context's renderer (sans the ANGLE preference, per the trim).
func (c *ContextInfo) Renderer() Renderer { return c.DriverInfo.Renderer }

// Driver returns the driver running the GL implementation, which is not necessarily related to the vendor (e.g. an
// Intel GPU driven by Mesa).
func (c *ContextInfo) Driver() Driver { return c.DriverInfo.Driver }

// DriverVersion returns the driver's version.
func (c *ContextInfo) DriverVersion() DriverVersion { return c.DriverInfo.DriverVersion }

// HasExtension reports whether ext is present in this context's extension set.
func (c *ContextInfo) HasExtension(ext string) bool { return c.Interface.HasExtension(ext) }

// glslGeneration computes the GLSL generation to target (desktop branch).
func glslGeneration(info *DriverInfo) (gpu.GLSLGeneration, bool) {
	// Workaround: some drivers report a GLSL version ahead of their GL version, so pin the GLSL version to the GL
	// version. GLSL versions have an extra digit on their minor level, so the GL minor revision is scaled up to be
	// comparable.
	ver := Ver(info.Version.Major(), 10*info.Version.Minor())
	if info.GLSLVersion < ver {
		ver = info.GLSLVersion
	}
	if info.GLSLVersion == InvalidVersion {
		return 0, false
	}
	switch {
	case ver >= Ver(4, 20):
		return gpu.GLSL420, true
	case ver >= Ver(4, 0):
		return gpu.GLSL400, true
	case ver >= Ver(3, 30):
		return gpu.GLSL330, true
	case ver >= Ver(1, 50):
		return gpu.GLSL150, true
	case ver >= Ver(1, 40):
		return gpu.GLSL140, true
	case ver >= Ver(1, 30):
		return gpu.GLSL130, true
	default:
		return gpu.GLSL110, true
	}
}
