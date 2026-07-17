// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests — most importantly the system-font Default() manager, which the rest of the suite
// avoids for hermeticity.

package fontmgr

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/font"
)

// TestDefaultManagerEnumeration verifies the process-wide default manager builds from the system font index
// (sk_fontmgr_ref_default) and its enumeration entry points stay consistent.
func TestDefaultManagerEnumeration(t *testing.T) {
	mgr := Default()
	if mgr == nil {
		t.Fatal("default font manager is nil")
	}
	if Default() != mgr {
		t.Fatal("Default must return the same manager every time")
	}
	count := mgr.CountFamilies()
	if count == 0 {
		t.Skip("no system font families available")
	}
	for _, i := range []int{0, count / 2, count - 1} {
		if mgr.FamilyName(i) == "" {
			t.Errorf("family[%d] has an empty name", i)
		}
	}
	// Out-of-range indices report the empty name rather than panicking.
	if mgr.FamilyName(count) != "" || mgr.FamilyName(-1) != "" {
		t.Error("out-of-range family index should yield an empty name")
	}
}

// TestDefaultManagerMatching verifies the match entry points on the default (system) manager: an unknown family yields
// an empty, non-nil style set; a real family matches; style matching resolves a face; and character coverage matching
// finds a face that really covers the probe rune.
func TestDefaultManagerMatching(t *testing.T) {
	mgr := Default()
	empty := mgr.MatchFamily("This Family Does Not Exist 12345")
	if empty == nil {
		t.Fatal("MatchFamily must not return nil for an unknown family")
	}
	if empty.Count() != 0 {
		t.Fatalf("unknown family set has %d faces, want 0", empty.Count())
	}

	if mgr.CountFamilies() == 0 {
		t.Skip("no system font families available")
	}
	fam := mgr.FamilyName(0)
	set := mgr.MatchFamily(fam)
	if set == nil || set.Count() == 0 {
		t.Fatalf("MatchFamily(%q) yielded no faces", fam)
	}
	if _, name := set.Style(0); name == "" {
		t.Errorf("style 0 of %q has an empty style name", fam)
	}
	if set.CreateTypeface(0) == nil {
		t.Errorf("CreateTypeface(0) for %q returned nil", fam)
	}
	if set.MatchStyle(font.NormalStyle()) == nil {
		t.Errorf("MatchStyle(normal) for %q returned nil", fam)
	}
	if mgr.MatchFamilyStyle(fam, font.BoldStyle()) == nil {
		t.Errorf("MatchFamilyStyle(%q, bold) returned nil", fam)
	}

	// A plain ASCII letter should be covered by some system family.
	tf := mgr.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'A')
	if tf == nil {
		t.Skip("no system family covers 'A'")
	}
	if tf.UnicharToGlyph('A') == 0 {
		t.Error("matched face does not actually cover 'A'")
	}
	// A bcp47 hint list is threaded through (most-significant last); still resolves a CJK sample if covered.
	if cjk := mgr.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"zh-Hans"}, 0x4E2D); cjk != nil {
		if cjk.UnicharToGlyph(0x4E2D) == 0 {
			t.Error("matched face does not actually cover U+4E2D")
		}
	}
}

// TestDefaultManagerMakeFromData verifies sk_fontmgr_create_from_data's semantics on the manager: valid font bytes
// parse to a typeface, garbage yields nil.
func TestDefaultManagerMakeFromData(t *testing.T) {
	mgr := Default()
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatalf("read test font: %v", err)
	}
	tf := mgr.MakeFromData(data, 0)
	if tf == nil {
		t.Fatal("MakeFromData returned nil for a valid face")
	}
	if got := tf.FamilyName(); got != "Roboto" {
		t.Errorf("family name = %q, want Roboto", got)
	}
	if bad := mgr.MakeFromData([]byte("not a font"), 0); bad != nil {
		t.Error("MakeFromData should return nil for garbage bytes")
	}
}
