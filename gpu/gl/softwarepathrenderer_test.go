// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for the software path renderer's stencil contract.

package gl

import "testing"

// TestSoftwarePathRendererStencilSupport pins the renderer's stencil report to the base-class default its own
// OnStencilPath assumes: reporting anything else makes PathRendererStencilPath (which only rejects
// StencilSupportNone) hand the renderer a stencil draw it can only panic on.
func TestSoftwarePathRendererStencilSupport(t *testing.T) {
	r := NewSoftwarePathRenderer(nil, false)
	shape := MakeStyledShapePath(starPath(), SimpleFillStyle(), DoSimplifyYes)
	if got := PathRendererGetStencilSupport(r, &shape); got != StencilSupportNone {
		t.Fatalf("stencil support = %d, want StencilSupportNone (%d)", got, StencilSupportNone)
	}

	// The wrapper must reject the draw itself; reaching OnStencilPath's own panic would mean the guard was bypassed.
	defer func() {
		msg, _ := recover().(string)
		switch msg {
		case "stencilPath on a renderer without stencil support":
		case "":
			t.Error("PathRendererStencilPath must reject the SW renderer")
		default:
			t.Errorf("panicked with %q; the support guard did not reject the draw first", msg)
		}
	}()
	PathRendererStencilPath(r, &StencilPathArgs{Shape: &shape})
}
