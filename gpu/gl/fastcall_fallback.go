// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The B.2 GL call lane fallback: platforms without an in-repo trampoline (anything that is not Windows and not
// darwin/linux on amd64/arm64) keep the original purego.SyscallN path, so the generated wrappers stay correct
// everywhere purego runs — only the per-call allocation win is forgone. Trailing zero padding is inert: extra integer
// arguments land in registers or stack slots a C callee never reads.

//go:build !windows && !((darwin || linux) && (amd64 || arm64))

package gl

import "github.com/ebitengine/purego"

// glCall invokes the GL proc fn with up to nine integer-class arguments and returns the proc's integer result.
// purego.SyscallN carries //go:uintptrescapes itself, but the tag is repeated here so this lane's S105 contract matches
// the fixed-arity lanes exactly.
//
//go:uintptrescapes
func glCall(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) uintptr {
	r1, _, _ := purego.SyscallN(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9)
	return r1
}
