// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the GLSL blend-function registry: every emitted body must only call functions the same emission actually
// defines.

package gl

import (
	"regexp"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/raster"
)

var (
	// A definition is a return type at the start of a line, immediately followed by the function name and its
	// parameter list. Body lines are either indented or start with something other than a bare type + name + '('.
	glslFuncDefRE  = regexp.MustCompile(`(?m)^(?:float|vec2|vec3|vec4)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	glslFuncCallRE = regexp.MustCompile(`([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
)

// checkGLSLBlendDefinitions reports every function the emitted GLSL calls without defining. Only the emitter's own
// functions are considered: GLSL builtins never start with "blend_" or the '_' prefix this package gives its private
// helpers.
func checkGLSLBlendDefinitions(t *testing.T, what, src string) {
	t.Helper()
	defined := make(map[string]bool)
	for _, m := range glslFuncDefRE.FindAllStringSubmatch(src, -1) {
		defined[m[1]] = true
	}
	for _, m := range glslFuncCallRE.FindAllStringSubmatch(src, -1) {
		called := m[1]
		if !strings.HasPrefix(called, "blend_") && !strings.HasPrefix(called, "_") {
			continue
		}
		if !defined[called] {
			t.Errorf("%s calls undefined %s():\n%s", what, called, src)
		}
	}
}

// TestGLSLBlendFunctionBodiesResolve emits the dedicated function for every blend mode plus each shared uniform-driven
// function, and checks that everything their bodies call is defined by the same emission. ensureBlendHelper renames two
// registry entries as it emits them (blend_darken_mode → blend_darken, blend_overlay_flip → blend_overlay), so a body
// naming the registry key rather than the emitted name would fail to compile with an undefined identifier. No
// end-to-end shader test can catch that for darken/lighten: getReducedBlendModeInfo gives them uniform data, so
// GLSLBlendExpression always takes the shared path and the dedicated functions are unreachable today.
func TestGLSLBlendFunctionBodiesResolve(t *testing.T) {
	g := newRecordingGpu(t)
	for mode := raster.BlendClear; mode <= raster.BlendLuminosity; mode++ {
		fs := newProgramBuilder(g, nil, nil).fs
		name := fs.ensureBlendFunction(mode)
		src := string(fs.shaderStrings[sectionDefinitions])
		if !glslFuncDefRE.MatchString(src) {
			t.Fatalf("blend mode %d emitted no definitions", mode)
		}
		checkGLSLBlendDefinitions(t, "blend mode "+name, src)
	}
	for _, fn := range []string{
		"blend_porter_duff", glslBlendOverlayFn, glslBlendDarkenFn,
		glslBlendHSLCFn,
	} {
		fs := newProgramBuilder(g, nil, nil).fs
		name := fs.ensureSharedBlendFunction(fn)
		checkGLSLBlendDefinitions(t, "shared function "+name,
			string(fs.shaderStrings[sectionDefinitions]))
	}
}
