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
		// The opt-in exists to exercise the scan, so an empty inventory is the failure it is here to report:
		// fontscan.SystemFonts erroring, or handing back footprints whose families are all empty, would otherwise
		// report SKIP and take every assertion below with it.
		t.Fatal("the system scan found no font families")
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

	count := mgr.CountFamilies()
	if count == 0 {
		t.Fatal("the system scan found no font families")
	}
	// Every enumerated family has to yield faces and a typeface — that much is the enumeration filter's own promise
	// and holds for all of them. A non-empty *subfamily* name is not: name IDs 17 and 2 are optional, so whether the
	// alphabetically-first family on this machine happens to carry one is a property of that machine's fonts rather
	// than of this package. The style-name assertion therefore runs over whichever family first offers one, and the
	// failure is "no system family names any of its styles", which really would be this package's doing.
	named, namedFamily := false, ""
	for i := range count {
		fam := mgr.FamilyName(i)
		set := mgr.MatchFamily(fam)
		if set == nil || set.Count() == 0 {
			t.Fatalf("MatchFamily(%q) yielded no faces", fam)
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
		if !named {
			if _, name := set.Style(0); name != "" {
				named, namedFamily = true, fam
			}
		}
	}
	if !named {
		t.Errorf("none of the %d system families names the style of its first face", count)
	} else {
		t.Logf("first family naming a style: %q", namedFamily)
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
	// A bcp47 hint list is threaded through (most-significant last). Each tag restricts the candidate set and then falls
	// through — to the next less-significant tag and finally to the unrestricted scan — so a hint can redirect the
	// choice but must never lose one: whatever the unhinted lane resolves, the hinted lane has to resolve too. Which
	// face a hint redirects to is machine-specific, so the redirection itself is pinned hermetically in
	// TestManagerMatchCharacterBCP47Fallthrough; what only the system manager can show is that real footprint language
	// sets do not make the tag pass fail closed.
	const cjk = 0x4E2D
	plain := mgr.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, cjk)
	if plain == nil {
		// Not every machine ships a CJK font; the unhinted probe is what tells that apart from a broken tag pass.
		t.Logf("no system family covers U+%04X, so the hinted lane has nothing to lose", cjk)
	} else {
		for _, tags := range [][]string{{"zh-Hans"}, {"ja", "zh-Hans"}, {""}, {"@@@"}} {
			hinted := mgr.MatchFamilyStyleCharacter("", font.NormalStyle(), tags, cjk)
			if hinted == nil {
				t.Errorf("bcp47 %v lost the match the unhinted scan found; the tag pass must fall through", tags)
				continue
			}
			if hinted.UnicharToGlyph(cjk) == 0 {
				t.Errorf("bcp47 %v matched a face that does not cover U+%04X", tags, cjk)
			}
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
