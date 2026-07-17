// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import "testing"

func TestGetVersionFromString(t *testing.T) {
	cases := []struct {
		in   string
		want Version
	}{
		{in: "", want: InvalidVersion},
		{in: "4.1 Metal - 90.5", want: Ver(4, 1)},                                   // macOS
		{in: "4.1 ATI-4.14.1", want: Ver(4, 1)},                                     // macOS AMD
		{in: "3.2.0 NVIDIA 310.32", want: Ver(3, 2)},                                // NVIDIA proprietary
		{in: "4.5 (Core Profile) Mesa 23.0.4", want: Ver(4, 5)},                     // Mesa core profile
		{in: "3.1 Mesa 20.3.5", want: Ver(3, 1)},                                    // Mesa via the mesa pattern
		{in: "OpenGL ES 3.2 v1.r38p1", want: Ver(3, 2)},                             // ES
		{in: "OpenGL ES-CM 1.1 Apple A8 GPU", want: Ver(1, 1)},                      // ES 1 (chars then version)
		{in: "OpenGL ES 2.0 (WebGL 1.0 (OpenGL ES 2.0 Chromium))", want: Ver(1, 0)}, // WebGL takes the inner pair
		{in: "garbage", want: InvalidVersion},
		{in: "OpenGL ES", want: InvalidVersion},
	}
	for _, c := range cases {
		if got := GetVersionFromString(c.in); got != c.want {
			t.Errorf("GetVersionFromString(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestGetStandardInUseFromString(t *testing.T) {
	cases := []struct {
		in   string
		want Standard
	}{
		{in: "", want: StandardNone},
		{in: "4.1 Metal - 90.5", want: StandardGL},
		{in: "3.2.0 NVIDIA 310.32", want: StandardGL},
		{in: "4.5 (Core Profile) Mesa 23.0.4", want: StandardGL},
		{in: "OpenGL ES 3.2 v1.r38p1", want: StandardGLES},
		{in: "OpenGL ES-CM 1.1", want: StandardNone}, // ES 1 is unsupported
		{in: "OpenGL ES 2.0 (WebGL 1.0 (OpenGL ES 2.0 Chromium))", want: StandardWebGL},
		{in: "garbage", want: StandardNone},
	}
	for _, c := range cases {
		if got := GetStandardInUseFromString(c.in); got != c.want {
			t.Errorf("GetStandardInUseFromString(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestVersionPacking(t *testing.T) {
	if v := Ver(4, 1); v.Major() != 4 || v.Minor() != 1 || v.String() != "4.1" {
		t.Errorf("Ver(4,1) roundtrip failed: %v", v)
	}
	if Ver(3, 0) >= Ver(4, 1) || Ver(4, 0) >= Ver(4, 1) || Ver(4, 2) <= Ver(4, 1) {
		t.Error("version ordering broken")
	}
}

func TestScanFormatSemantics(t *testing.T) {
	// %d skips leading whitespace like sscanf.
	if v, n := scanFormat("  7 . 3", "%d.%d"); n != 1 || v[0] != 7 {
		// The '.' literal must match immediately after the int; " . 3" fails at the literal.
		t.Errorf("scanFormat whitespace/literal semantics: got %v n=%d", v, n)
	}
	// A ' ' in the format matches zero or more whitespace bytes.
	if _, n := scanFormat("OpenGLES 2.0", "OpenGL ES %d.%d"); n != 2 {
		t.Errorf("format space should match empty run: n=%d", n)
	}
	// Negative values parse (sscanf %d accepts a sign).
	if v, n := scanFormat("-2.5", "%d.%d"); n != 2 || v[0] != -2 || v[1] != 5 {
		t.Errorf("signed parse: got %v n=%d", v, n)
	}
}
