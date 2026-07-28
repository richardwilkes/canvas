// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the fixed-arity glCall lane (fastcall_*.go): a live-GL functional check of the trampoline against
// purego.SyscallN on identical proc addresses (including the 9-argument stack-slot path and the zero-allocation
// guarantee), and a source-level audit that pins the lane's arity contract so a future GL function-table regeneration
// cannot silently violate it.

package gl

import (
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/gpu/gl/gltest"
)

// TestGLCallLaneLive drives the glCall lane against a live context: results must agree with purego.SyscallN on the same
// proc address, out-parameters and the 9th-argument stack slot must land (glTexImage2D with a pixel upload), and a call
// must not allocate on the trampoline platforms (on fallback platforms glCall is SyscallN, so the allocation assertion
// is skipped there by construction — this file only builds where the repo builds, and the trampoline covers every
// platform the library supports).
func TestGLCallLaneLive(t *testing.T) {
	ctx, err := gltest.New()
	if err != nil {
		t.Skipf("no GL context available: %v", err)
	}
	defer ctx.Destroy()
	intf := MakeNativeInterface()
	if intf == nil {
		t.Fatal("MakeNativeInterface returned nil with a current context")
	}
	f := &intf.Functions

	// Result agreement: glGetString(GL_VERSION) via both lanes must return the same C pointer.
	r1, _, _ := purego.SyscallN(f.getString, uintptr(VERSION))
	r2 := glCall(f.getString, uintptr(VERSION), 0, 0, 0, 0, 0, 0, 0, 0)
	if r1 != r2 {
		t.Fatalf("getString mismatch: SyscallN %#x vs glCall %#x", r1, r2)
	}

	// Pointer out-parameter: glGetIntegerv must write through the second argument.
	var v1, v2 int32
	purego.SyscallN(f.getIntegerv, uintptr(MAX_TEXTURE_SIZE), uintptr(unsafe.Pointer(&v1)))
	glCall(f.getIntegerv, uintptr(MAX_TEXTURE_SIZE), uintptr(unsafe.Pointer(&v2)), 0, 0, 0, 0, 0,
		0, 0)
	if v1 == 0 || v1 != v2 {
		t.Fatalf("getIntegerv mismatch: SyscallN %d vs glCall %d", v1, v2)
	}

	// The 9-argument path (the trampoline's single stack slot): glTexImage2D uploads through its 9th argument; a
	// mis-marshaled call would raise GL_INVALID_* or upload garbage dimensions.
	var tex uint32
	f.GenTextures(1, &tex)
	f.BindTexture(TEXTURE_2D, tex)
	pix := make([]byte, 4*4*4)
	for i := range pix {
		pix[i] = byte(i)
	}
	f.TexImage2D(TEXTURE_2D, 0, RGBA8, 4, 4, 0, RGBA, UNSIGNED_BYTE,
		uintptr(unsafe.Pointer(&pix[0])))
	if e := f.GetError(); e != NO_ERROR {
		t.Fatalf("glTexImage2D through the glCall lane raised GL error %#x", e)
	}
	var w int32
	f.GetTexLevelParameteriv(TEXTURE_2D, 0, TEXTURE_WIDTH, &w)
	if w != 4 {
		t.Fatalf("uploaded texture width = %d, want 4 — the 9th-argument stack slot mis-marshaled", w)
	}
	f.DeleteTextures(1, &tex)

	if allocs := testing.AllocsPerRun(1000, func() { f.GetError() }); allocs != 0 {
		t.Fatalf("a glCall-lane wrapper allocates %v per call, want 0", allocs)
	}
}

// TestGLCallArityCoverage audits the interface.go source: every glCall dispatch must carry exactly nine arguments after
// the proc address — the trampoline's fixed arity, zero-padded. Nine is also the safety boundary of the lane's single
// 8-byte stack slot on darwin/arm64: the 9th argument is necessarily the last, so even a sub-8-byte parameter there is
// read correctly from the low bytes of the zero-extended slot (the "single trailing stack argument" case;
// glMapTexSubImage2D's GLenum access exercises it). A tenth argument would break that reasoning, which is why the
// generator refuses to emit one (glCallMaxArgs) and this audit pins the output.
func TestGLCallArityCoverage(t *testing.T) {
	src, err := os.ReadFile("interface.go")
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	lines := strings.Split(string(src), "\n")
	for i := 0; i < len(lines); i++ {
		idx := strings.Index(lines[i], "glCall(f.")
		if idx < 0 {
			continue
		}
		calls++
		// A dispatch may be wrapped across physical lines, each continued one ending in a comma. Rejoin them into the
		// single logical line the comma count below expects.
		line := lines[i]
		for strings.HasSuffix(strings.TrimSpace(line), ",") && i+1 < len(lines) {
			i++
			line += " " + strings.TrimSpace(lines[i])
		}
		// Count the dispatch's top-level commas (arguments after the proc address).
		depth, commas := 0, 0
		for _, ch := range line[idx+len("glCall("):] {
			switch {
			case ch == '(':
				depth++
			case ch == ')':
				if depth == 0 {
					goto counted
				}
				depth--
			case ch == ',' && depth == 0:
				commas++
			}
		}
	counted:
		if commas != 9 {
			t.Errorf("glCall with %d args (want exactly 9, zero-padded): %s", commas,
				strings.TrimSpace(line))
		}
	}
	if calls < 100 {
		t.Fatalf("audited only %d glCall dispatches — the source parse is broken", calls)
	}
	t.Logf("audited %d glCall dispatches, all at the fixed arity of 9", calls)
}
