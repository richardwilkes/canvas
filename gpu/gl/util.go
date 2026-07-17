// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GL version-string parsing: GetStandardInUseFromString and GetVersionFromString extract the GL standard (desktop, ES,
// or WebGL) and version number from the string returned by glGetString(GL_VERSION). Both are built on scanFormat, a
// small sscanf-style matcher (' ' matches a run of whitespace, %d skips whitespace then reads an optional sign and
// digits, %c reads one byte verbatim) that reports how many conversions completed, so each pattern's expected-count
// check tells the caller whether the match held.

package gl

// scanFormat is the mini-sscanf used by the version parsers. Supported verbs: %d (int32) and %c (any byte). A ' ' in
// the format matches any run of whitespace, including none. Returns the number of conversions completed before the
// first mismatch or end of input.
func scanFormat(s, format string) (values []int, count int) {
	si := 0
	fi := 0
	for fi < len(format) {
		switch {
		case format[fi] == '%' && fi+1 < len(format) && format[fi+1] == 'd':
			fi += 2
			for si < len(s) && isSpace(s[si]) {
				si++
			}
			start := si
			if si < len(s) && (s[si] == '-' || s[si] == '+') {
				si++
			}
			digits := si
			for si < len(s) && s[si] >= '0' && s[si] <= '9' {
				si++
			}
			if si == digits {
				return values, count
			}
			v := 0
			neg := s[start] == '-'
			for i := digits; i < si; i++ {
				v = v*10 + int(s[i]-'0')
			}
			if neg {
				v = -v
			}
			values = append(values, v)
			count++
		case format[fi] == '%' && fi+1 < len(format) && format[fi+1] == 'c':
			fi += 2
			if si >= len(s) {
				return values, count
			}
			values = append(values, int(s[si]))
			si++
			count++
		case format[fi] == ' ':
			fi++
			for si < len(s) && isSpace(s[si]) {
				si++
			}
		default:
			if si >= len(s) || s[si] != format[fi] {
				return values, count
			}
			si++
			fi++
		}
	}
	return values, count
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'
}

// GetStandardInUseFromString parses the string returned by glGetString(GL_VERSION) and reports which GL standard
// (desktop GL, GLES, or WebGL) it identifies.
func GetStandardInUseFromString(versionString string) Standard {
	if versionString == "" {
		return StandardNone
	}
	// Check for desktop.
	if _, n := scanFormat(versionString, "%d.%d"); n == 2 {
		return StandardGL
	}
	// WebGL might look like "OpenGL ES 2.0 (WebGL 1.0 (OpenGL ES 2.0 Chromium))".
	if _, n := scanFormat(versionString, "OpenGL ES %d.%d (WebGL %d.%d"); n == 4 {
		return StandardWebGL
	}
	// Check for ES 1: no longer supported.
	if _, n := scanFormat(versionString, "OpenGL ES-%c%c %d.%d"); n == 4 {
		return StandardNone
	}
	// Check for ES2.
	if _, n := scanFormat(versionString, "OpenGL ES %d.%d"); n == 2 {
		return StandardGLES
	}
	return StandardNone
}

// GetVersionFromString parses the string returned by glGetString(GL_VERSION) and returns the GL version it identifies,
// handling desktop, ES, WebGL, and Mesa-suffixed formats.
func GetVersionFromString(versionString string) Version {
	if versionString == "" {
		return InvalidVersion
	}
	// Check for mesa.
	if v, n := scanFormat(versionString, "%d.%d Mesa %d.%d"); n == 4 {
		return Ver(uint32(v[0]), uint32(v[1]))
	}
	if v, n := scanFormat(versionString, "%d.%d"); n == 2 {
		return Ver(uint32(v[0]), uint32(v[1]))
	}
	// WebGL might look like "OpenGL ES 2.0 (WebGL 1.0 (OpenGL ES 2.0 Chromium))".
	if v, n := scanFormat(versionString, "OpenGL ES %d.%d (WebGL %d.%d"); n == 4 {
		return Ver(uint32(v[2]), uint32(v[3]))
	}
	if v, n := scanFormat(versionString, "OpenGL ES-%c%c %d.%d"); n == 4 {
		return Ver(uint32(v[2]), uint32(v[3]))
	}
	if v, n := scanFormat(versionString, "OpenGL ES %d.%d"); n == 2 {
		return Ver(uint32(v[0]), uint32(v[1]))
	}
	return InvalidVersion
}
