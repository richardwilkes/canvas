// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Null- and failure-contract checks for the public context and interface constructors — the guards unison relies on
// when its GL environment is degraded.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/gpu/gl"
)

// TestMakeGLDirectContextNilInterface verifies the null contracts of the context assembly path: a nil interface yields
// a nil context info and a nil direct context instead of a panic.
func TestMakeGLDirectContextNilInterface(t *testing.T) {
	if gl.NewContextInfo(nil, nil) != nil {
		t.Error("NewContextInfo(nil) should be nil")
	}
	if gl.MakeGLDirectContext(nil, nil) != nil {
		t.Error("MakeGLDirectContext(nil) should be nil")
	}
}

// TestMakeAssembledInterfaceUnresolvable verifies MakeAssembledInterface's first resolution guard: a get-proc that
// cannot resolve glGetString yields nil (no GL is issued).
func TestMakeAssembledInterfaceUnresolvable(t *testing.T) {
	if gl.MakeAssembledInterface(func(string) uintptr { return 0 }) != nil {
		t.Error("MakeAssembledInterface with an unresolvable get-proc should be nil")
	}
}

// TestFormatFromEnum pins FormatFromEnum for the formats the wrapped-FBO path passes through (unison hands RGBA8), plus
// the unknown-enum fallback.
func TestFormatFromEnum(t *testing.T) {
	cases := []struct {
		enum uint32
		want gl.Format
	}{
		{enum: gl.RGBA8, want: gl.FormatRGBA8},
		{enum: gl.BGRA8, want: gl.FormatBGRA8},
		{enum: gl.R8, want: gl.FormatR8},
		{enum: 0xDEAD, want: gl.FormatUnknown},
	}
	for _, c := range cases {
		if got := gl.FormatFromEnum(c.enum); got != c.want {
			t.Errorf("FormatFromEnum(0x%04X) = %v, want %v", c.enum, got, c.want)
		}
	}
}
