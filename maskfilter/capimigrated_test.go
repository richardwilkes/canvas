// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Constructor checks for the public mask-filter entry points.

package maskfilter

import (
	"testing"

	"github.com/richardwilkes/canvas/shaders"
)

// TestNewShaderConstructor verifies NewShader: a valid shader yields a mask filter, a nil shader yields nil.
func TestNewShaderConstructor(t *testing.T) {
	if mf := NewShader(shaders.NewColor(0xFF808080)); mf == nil {
		t.Error("shader mask filter with a valid shader: want non-nil")
	}
	if mf := NewShader(nil); mf != nil {
		t.Error("shader mask filter with a nil shader: want nil")
	}
}

// TestNewClipConstructor verifies NewClip builds the clip table filter.
func TestNewClipConstructor(t *testing.T) {
	if mf := NewClip(64, 192); mf == nil {
		t.Error("clip table mask filter: want non-nil")
	}
}
