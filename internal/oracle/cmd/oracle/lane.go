// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/richardwilkes/canvas/internal/oracle/gorender"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// The golden lanes. Each lane is a rendering backend configuration with its own per-platform golden set under
// goldens/<lane>/<GOOS_GOARCH>/.
const (
	laneRaster   = "raster"
	laneGPU      = "gpu"
	laneGPUDMSAA = "gpudmsaa"
)

// exitNoGLContext is the exit code soak and bless use when the GPU backend cannot bring up a GL context. Distinct from
// the generic failure code 1 so the capture-goldens workflow can treat "this leg has no GL stack" as a loud skip of the
// lane rather than a job failure.
const exitNoGLContext = 3

// errNoGLContext marks a session-creation failure caused by the GPU backend having no GL context. main maps it to
// exitNoGLContext.
var errNoGLContext = errors.New("no GL context")

var errNoBothGPUFlags = errors.New("-gpu and -dmsaa are mutually exclusive")

// laneName maps the soak flags onto a lane name.
func laneName(useGPU, useDMSAA bool) string {
	switch {
	case useDMSAA:
		return laneGPUDMSAA
	case useGPU:
		return laneGPU
	default:
		return laneRaster
	}
}

// laneSession is one rendering session for a lane: a render function plus, for the GPU lanes, the fresh GL context
// behind it and that context's identity strings (empty for raster, which has no environment exposure).
//
// Sessions exist because GPU-lane output is reproducible per *fresh-context corpus pass*, not per repeated render on a
// warm context: re-rendering with the GL backend's context-lifetime caches warm shifts blur-family scenarios by a max
// channel delta of 1 on software rasterizers. The gates each create one fresh context and render scenario.All() in
// order, so that is the shape soak proves and bless guards: a fresh-context in-order corpus pass reproduces itself —
// bit-identically for raster, within the ±1 LSB driver envelope for the GPU lanes (see the soak doc comment). Renders
// must run inline on the creating goroutine (the GL context is thread-bound), and dispose must be called before
// creating another session.
type laneSession struct {
	render     func(scenario.Scenario) []byte
	dispose    func()
	glRenderer string
	glVersion  string
}

// pinSoftwareRenderer pins CANVAS_GLTEST_RENDERER=software for the GL lanes, so a capture renders on the same stack
// the golden gates do. Every GPU gate pins it — TestGoGPUvsSelfCapturedGolden and its wrapped-FBO twin
// (gorender/gpugolden_test.go) as much as TestGoGPUDMSAAvsSelfCapturedGolden — because comparing across GL stacks is
// meaningless under exact1: hardware and software rasterizers differ structurally at AA edges, and DMSAA output
// additionally depends on the driver's MSAA sample positions and resolve filter. Pinning only the gpudmsaa lane would
// let `oracle bless -lane gpu` on a machine with hardware GL capture a set no gate can ever render, caught only after
// a full two-pass capture by the gate's GL_RENDERER manifest guard.
//
// On darwin the variable selects kCGLRendererGenericFloatID; on Linux and Windows it is a no-op (their software stacks
// are provisioned externally — llvmpipe under Xvfb / the Mesa3D drop-in), so setting it for every GL lane is safe. The
// raster lane is pure Go with no GL exposure, so it is left alone.
func pinSoftwareRenderer(lane string) error {
	if lane == laneRaster {
		return nil
	}
	return os.Setenv("CANVAS_GLTEST_RENDERER", "software")
}

// newLaneSession opens a fresh rendering session for one full corpus pass. For the GPU lanes it pins the software GL
// stack (see pinSoftwareRenderer) and creates the headless GL context; a context-creation failure comes back wrapping
// errNoGLContext so main can exit with exitNoGLContext (the capture workflow's loud-skip signal).
func newLaneSession(lane string) (*laneSession, error) {
	if lane == laneRaster {
		return &laneSession{render: gorender.RenderScenarioRaster, dispose: func() {}}, nil
	}
	if err := pinSoftwareRenderer(lane); err != nil {
		return nil, err
	}
	g, err := gorender.NewGPUContext()
	if err != nil {
		return nil, fmt.Errorf("lane %s: GPU backend unavailable (%w): %w", lane, errNoGLContext, err)
	}
	s := &laneSession{dispose: g.Dispose, glRenderer: g.RendererString(), glVersion: g.VersionString()}
	if lane == laneGPUDMSAA {
		s.render = func(sc scenario.Scenario) []byte { return gorender.RenderScenarioGPUDMSAA(g, sc) }
	} else {
		s.render = func(sc scenario.Scenario) []byte { return gorender.RenderScenarioGPU(g, sc) }
	}
	return s, nil
}

// blessDir derives the golden directory for a lane's per-platform set, relative to the oracle module root:
// goldens/<lane>/<GOOS_GOARCH>. The path is derived, never user-supplied, so a capture can only land where the gates
// will look for it.
func blessDir(lane, goos, goarch string) string {
	return filepath.Join("goldens", lane, goos+"_"+goarch)
}
