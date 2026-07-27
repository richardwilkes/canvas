// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gorender

// Driver quirks that are proven to live below the port: rendering differences a given GL stack produces on its own,
// which no app-level change can remove. They live here rather than in the golden gates because capture (oracle
// soak/bless) and gating must agree about them exactly — a quirk excused by the gate but treated as nondeterminism by
// the capture path makes capture succeed or fail by luck, which is what happened to darwin_arm64's gpu lane on
// 2026-07-27. Every entry must carry the evidence that put it here; nothing goes in this file to make a red leg green.

// AppleSoftwareRenderer is the GL_RENDERER string of Apple's software GL stack, the capture stack of the darwin golden
// sets.
const AppleSoftwareRenderer = "Apple Software Renderer"

// bimodalOnAppleSoftware lists scenarios the Apple software renderer draws in one of two bit-exact flavors per GL
// session (sticky per session and per machine state, flipping between sessions/machines at a load-dependent rate, with
// identical GL_RENDERER and GL_VERSION strings in both). Proven driver-internal: output is independent of all
// app-allocation content, and the GL command stream is byte-identical across flips; llvmpipe renders the same
// scenarios deterministically. The affected scenes put antialiased edges under a perspective divide, which distributes
// edge/sample crossings essentially uniformly against the pixel grid — so whichever internal precision mode the
// driver picks, the few percent of crossings within its epsilon flip wholesale (measured: ~1,000-4,000 px at channel
// deltas up to 209), far beyond the ±1 envelope and unfixable by nudging geometry.
//
// On golden sets captured on that renderer these scenarios are reported but not pixel-gated; every other set (the
// llvmpipe lanes) gates them fully, and the exact-cover check still requires their goldens to exist everywhere. The
// same pathology under 4x MSAA is why darwin_arm64 has no DMSAA set at all (there the flip count made even capture
// unreliable).
var bimodalOnAppleSoftware = map[string]bool{
	"clip-persp": true,
}

// DriverBimodal reports whether scenarioName is a known driver-bimodal scenario on the GL stack named by glRenderer,
// i.e. one whose beyond-envelope pass-to-pass differences are the driver picking a different internal precision mode
// rather than a rendering bug or app-level nondeterminism.
//
// Both the golden gates and the capture path (oracle soak/bless) consult this: the gates report such a scenario
// without pixel-gating it, and capture reports it without counting it as nondeterminism. Anything else is a hard
// failure in both, so this is the single place where a beyond-envelope difference is ever forgiven.
//
// The empty renderer string (the raster lane, which has no GL context) matches nothing, so raster stays strictly
// bit-exact with no lane check at the call sites.
func DriverBimodal(glRenderer, scenarioName string) bool {
	return glRenderer == AppleSoftwareRenderer && bimodalOnAppleSoftware[scenarioName]
}
