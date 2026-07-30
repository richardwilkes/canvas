// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Checks for the public font-manager entry points — most importantly the system-font Default() manager. Reaching
// Default scans the system font directories and writes an index cache, so the tests that do are opt-in
// (requireSystemFonts) and the suite stays hermetic by default, exactly as fontmgr_test.go's header claims.

package fontmgr

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/font"
)

// TestDefaultManagerEnumeration verifies the process-wide default manager builds from the system font index and its
// enumeration entry points stay consistent.
func TestDefaultManagerEnumeration(t *testing.T) {
	requireSystemFonts(t)
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
	requireSystemFonts(t)
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

	// A plain ASCII letter must be covered by some system family: this machine has font families (the skip above), so a
	// nil here means the whole cross-family character-fallback lane (matchCoveringTiered/matchCovering) is broken —
	// MatchFamilyStyleCharacter's primary regression mode, and skipping on it would take the assertions below with it.
	tf := mgr.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'A')
	if tf == nil {
		t.Fatal("no system family covers 'A'")
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

// TestMakeFromData verifies MakeFromData's semantics: valid font bytes parse to a typeface, garbage yields nil. It
// parses only the bytes it is handed and never consults the manager's inventory, so an empty manager answers exactly
// as the system one does — which is what keeps this case off the system font directories.
func TestMakeFromData(t *testing.T) {
	mgr := NewFromData()
	if got := mgr.CountFamilies(); got != 0 {
		t.Fatalf("the empty manager holds %d families; MakeFromData must not need an inventory", got)
	}
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
	// A collection index past the end of the file is nil too, rather than a face from another index.
	if bad := mgr.MakeFromData(data, 1); bad != nil {
		t.Error("MakeFromData should return nil for an out-of-range collection index")
	}
}
