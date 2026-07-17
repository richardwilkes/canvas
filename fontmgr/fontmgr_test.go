// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Manager/StyleSet behavior tests over the checked-in test fonts (font/testdata), so the corpus is hermetic: no system
// fonts are touched. Coverage facts about the corpus (from the fonts' cmaps):
//
//	Roboto-Regular.ttf    family "Roboto"      (400,5,upright) "Regular"  covers NUL, CR, ASCII (215 runes)
//	DejaVuSans.subset.ttf family "DejaVu Sans" (400,5,upright) "Book"     covers 'H' 'a' 'x'
//	test.ttc[0]           family "Test"        (400,5,upright) "Regular"  covers '!' '"' '0'-'4' 'A'
//	test.ttc[1]           family "Test"        (700,5,upright) "Bold"     covers '!'
//
// Roboto's NUL/CR entries map real glyphs. The other three faces additionally carry a cmap4 sentinel segment mapping
// U+FFFF to glyph 0 (.notdef): go-text's scanner counts it into the footprint rune set, but it maps no real glyph — the
// character-fallback verification cases below rely on that.
//
// The system manager (Default) is exercised only by the opt-in system-scan test at the bottom of this file; the live
// oracle probe that once compared it against the C library's platform host was removed along with that library.

package fontmgr

import (
	"os"
	"sync"
	"testing"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/fontscan"
	"github.com/go-text/typesetting/language"
	"github.com/richardwilkes/canvas/font"
)

// newTestFaceRec builds a faceRec for one face of a testdata font, deriving the grouping key and rune coverage from the
// file exactly as the fontscan footprints would.
func newTestFaceRec(t *testing.T, file string, index int, langs ...string) *faceRec {
	t.Helper()
	path := "../font/testdata/" + file
	info, err := font.DescribeFaceFile(path, index)
	if err != nil {
		t.Fatalf("%s[%d]: describe: %v", file, index, err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close() //nolint:errcheck // read-only file; close errors are irrelevant
	lds, err := opentype.NewLoaders(f)
	if err != nil {
		t.Fatal(err)
	}
	ft, err := tsfont.NewFont(lds[index])
	if err != nil {
		t.Fatal(err)
	}
	var runes fontscan.RuneSet
	iter := ft.Cmap.Iter()
	for iter.Next() {
		r, _ := iter.Char()
		runes.Add(r)
	}
	var langSet fontscan.LangSet
	for _, lang := range langs {
		id, ok := language.NewLangID(language.NewLanguage(lang))
		if !ok {
			t.Fatalf("unknown test language %q", lang)
		}
		langSet.Add(id)
	}
	return &faceRec{
		key:    tsfont.NormalizeFamily(info.Family),
		path:   path,
		index:  index,
		runes:  runes,
		langs:  langSet,
		approx: info.Style,
	}
}

// newTestManager builds the standard four-face corpus manager: Roboto tagged "en", DejaVu Sans tagged "fr".
func newTestManager(t *testing.T) *Manager {
	t.Helper()
	return newManager([]*faceRec{
		newTestFaceRec(t, "Roboto-Regular.ttf", 0, "en"),
		newTestFaceRec(t, "DejaVuSans.subset.ttf", 0, "fr"),
		newTestFaceRec(t, "test.ttc", 0),
		newTestFaceRec(t, "test.ttc", 1),
	})
}

func styleTriple(s font.Style) (weight, width, slant int) {
	return s.Weight(), s.Width(), int(s.Slant())
}

func TestManagerEnumeration(t *testing.T) {
	m := newTestManager(t)
	if got := m.CountFamilies(); got != 3 {
		t.Fatalf("CountFamilies = %d, want 3", got)
	}
	// Families sort by normalized key: dejavusans < roboto < test.
	want := []string{"DejaVu Sans", "Roboto", "Test"}
	for i, name := range want {
		if got := m.FamilyName(i); got != name {
			t.Errorf("FamilyName(%d) = %q, want %q", i, got, name)
		}
	}
	for _, idx := range []int{-1, 3} {
		if got := m.FamilyName(idx); got != "" {
			t.Errorf("FamilyName(%d) = %q, want \"\"", idx, got)
		}
	}
	// Every enumerated name round-trips through MatchFamily (the iteration invariant unison relies on: FamilyName(i) →
	// MatchFamily(name) finds the same family).
	for i := range m.CountFamilies() {
		if got := m.MatchFamily(m.FamilyName(i)).Count(); got < 1 {
			t.Errorf("MatchFamily(FamilyName(%d)) count = %d, want >= 1", i, got)
		}
	}
}

func TestManagerMatchFamily(t *testing.T) {
	m := newTestManager(t)
	if got := m.MatchFamily("Test").Count(); got != 2 {
		t.Errorf("MatchFamily(Test) count = %d, want 2", got)
	}
	// Normalized lookup: case- and space-insensitive.
	for _, name := range []string{"test", "TEST", "Te st"} {
		if got := m.MatchFamily(name).Count(); got != 2 {
			t.Errorf("MatchFamily(%q) count = %d, want 2", name, got)
		}
	}
	// Unknown and empty families produce an empty, non-nil set (MatchFamily never returns nil).
	for _, name := range []string{"Non Existing Family Name", ""} {
		set := m.MatchFamily(name)
		if set == nil {
			t.Fatalf("MatchFamily(%q) = nil", name)
		}
		if got := set.Count(); got != 0 {
			t.Errorf("MatchFamily(%q) count = %d, want 0", name, got)
		}
		if tf := set.MatchStyle(font.NormalStyle()); tf != nil {
			t.Errorf("empty set MatchStyle != nil")
		}
	}
}

func TestManagerStyleSet(t *testing.T) {
	m := newTestManager(t)
	set := m.MatchFamily("Test")
	wantStyles := []struct {
		name                string
		weight, width, slnt int
	}{
		{name: "Regular", weight: 400, width: 5, slnt: 0},
		{name: "Bold", weight: 700, width: 5, slnt: 0},
	}
	for i, want := range wantStyles {
		style, name := set.Style(i)
		w, wd, sl := styleTriple(style)
		if w != want.weight || wd != want.width || sl != want.slnt || name != want.name {
			t.Errorf("Style(%d) = (%d,%d,%d) %q, want (%d,%d,%d) %q", i, w, wd, sl, name,
				want.weight, want.width, want.slnt, want.name)
		}
		tf := set.CreateTypeface(i)
		if tf == nil {
			t.Fatalf("CreateTypeface(%d) = nil", i)
		}
		// The listed style always equals the created typeface's style (both derive from the same OS/2 read).
		if tf.Style() != style {
			t.Errorf("CreateTypeface(%d).Style() = %v, want %v", i, tf.Style(), style)
		}
		if got := tf.FamilyName(); got != "Test" {
			t.Errorf("CreateTypeface(%d).FamilyName() = %q, want Test", i, got)
		}
	}
	for _, idx := range []int{-1, 2} {
		if tf := set.CreateTypeface(idx); tf != nil {
			t.Errorf("CreateTypeface(%d) != nil", idx)
		}
	}
	// MatchStyle: exact weights match; thin prefers the lighter face (CSS3 weight rules).
	for _, c := range []struct {
		pattern    font.Style
		wantWeight int
	}{
		{pattern: font.NormalStyle(), wantWeight: 400},
		{pattern: font.BoldStyle(), wantWeight: 700},
		{pattern: font.NewStyle(font.WeightThin, font.WidthNormal, font.SlantUpright), wantWeight: 400},
		{pattern: font.NewStyle(font.WeightBlack, font.WidthNormal, font.SlantUpright), wantWeight: 700},
		{pattern: font.BoldItalicStyle(), wantWeight: 700}, // no italics in the set: weight decides
	} {
		tf := set.MatchStyle(c.pattern)
		if tf == nil {
			t.Fatalf("MatchStyle(%v) = nil", c.pattern)
		}
		if got := tf.Style().Weight(); got != c.wantWeight {
			t.Errorf("MatchStyle weight = %d, want %d", got, c.wantWeight)
		}
	}
}

func TestManagerMatchFamilyStyle(t *testing.T) {
	m := newTestManager(t)
	if tf := m.MatchFamilyStyle("Test", font.BoldStyle()); tf == nil || tf.Style().Weight() != 700 {
		t.Errorf("MatchFamilyStyle(Test, bold) = %v", tf)
	}
	if tf := m.MatchFamilyStyle("Roboto", font.BoldStyle()); tf == nil || tf.Style().Weight() != 400 {
		t.Errorf("MatchFamilyStyle(Roboto, bold) should return the only face")
	}
	// Unknown families are nil (the CoreText host's failed CTFontDescriptorCreateMatchingFontDescriptor, the fontconfig
	// host's FontFamilyNameMatches rejection).
	if tf := m.MatchFamilyStyle("Non Existing Family Name", font.NormalStyle()); tf != nil {
		t.Errorf("MatchFamilyStyle(unknown) != nil")
	}
	if tf := m.MatchFamilyStyle("ῢ ΰ ῤ ῦ ῧ Ῠ Ῡ Ὺ Ύ Ῥ ῲ ῳ ῴ ῶ ῷ Ὸ Ό Ὼ Ώ ῼ", font.NormalStyle()); tf != nil {
		t.Errorf("MatchFamilyStyle(case-folding torture name) != nil")
	}
	// The empty name requests the platform default family; none of the defaults exist in the test corpus, so the first
	// family (DejaVu Sans) answers.
	tf := m.MatchFamilyStyle("", font.NormalStyle())
	if tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("MatchFamilyStyle(\"\") = %v, want the first family", tf)
	}
	// An empty manager returns nil for everything.
	empty := newManager(nil)
	if tf = empty.MatchFamilyStyle("", font.NormalStyle()); tf != nil {
		t.Errorf("empty manager MatchFamilyStyle != nil")
	}
	if got := empty.CountFamilies(); got != 0 {
		t.Errorf("empty manager CountFamilies = %d", got)
	}
}

func TestManagerMatchFamilyStyleCharacter(t *testing.T) {
	m := newTestManager(t)
	match := func(family string, style font.Style, bcp47 []string, ch int32) *font.Typeface {
		t.Helper()
		return m.MatchFamilyStyleCharacter(family, style, bcp47, ch)
	}
	// The named family wins when it covers the character; style picks the face within it.
	if tf := match("Test", font.NormalStyle(), nil, '!'); tf == nil || tf.Style().Weight() != 400 {
		t.Errorf("(Test, normal, '!') = %v, want the regular face", tf)
	}
	if tf := match("Test", font.BoldStyle(), nil, '!'); tf == nil || tf.Style().Weight() != 700 {
		t.Errorf("(Test, bold, '!') = %v, want the bold face", tf)
	}
	// 'A' is only in the regular face: the bold request still lands on the covering face.
	if tf := match("Test", font.BoldStyle(), nil, 'A'); tf == nil || tf.Style().Weight() != 400 {
		t.Errorf("(Test, bold, 'A') = %v, want the regular face", tf)
	}
	// The family does not cover 'x': fall back to the global scan. Roboto and DejaVu tie on style, and the
	// family-sorted candidate order makes DejaVu Sans (first) win.
	if tf := match("Test", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("(Test, normal, 'x') = %v, want DejaVu Sans", tf)
	}
	// No family: same global scan.
	if tf := match("", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("(\"\", normal, 'x') = %v, want DejaVu Sans", tf)
	}
	// BCP-47: the most significant tag is last; 'x' is covered by both tagged fonts.
	if tf := match("", font.NormalStyle(), []string{"fr", "en"}, 'x'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [fr en] = %v, want Roboto (en most significant)", tf)
	}
	if tf := match("", font.NormalStyle(), []string{"en", "fr"}, 'x'); tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("bcp47 [en fr] = %v, want DejaVu Sans (fr most significant)", tf)
	}
	// Unmatched tags fall through to the next most significant, then to the unrestricted scan.
	if tf := match("", font.NormalStyle(), []string{"en", "ja"}, 'x'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [en ja] = %v, want Roboto (ja unmatched, en next)", tf)
	}
	if tf := match("", font.NormalStyle(), []string{"ja"}, 'x'); tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("bcp47 [ja] = %v, want DejaVu Sans (unrestricted scan)", tf)
	}
	// A derived language tag maps to its primary ("fr-CA" → "fr").
	if tf := match("", font.NormalStyle(), []string{"fr-CA"}, 'x'); tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("bcp47 [fr-CA] = %v, want DejaVu Sans", tf)
	}
	// Roboto maps NUL to a real glyph, so even character 0 resolves (the coverage sets decide, no special-casing of
	// control characters).
	if tf := match("", font.NormalStyle(), nil, 0); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("(\"\", normal, NUL) = %v, want Roboto", tf)
	}
	// No font covers U+4E2D: nil per the documented contract (the CoreText host would return the family font instead).
	if tf := match("", font.NormalStyle(), nil, 0x4E2D); tf != nil {
		t.Errorf("uncovered character = %v, want nil", tf)
	}
	// Footprint-only coverage is not coverage: the DejaVu subset and both Test faces carry the cmap4 sentinel segment
	// mapping U+FFFF to glyph 0, which the footprint rune sets count but no host does (FreeType's charcode iteration
	// skips glyph-0 entries, so fontconfig charsets never contain them; the CoreText/DirectWrite cmap lookups yield the
	// missing glyph). The verified answer is nil, in the global scan and within a named family alike.
	if tf := match("", font.NormalStyle(), nil, 0xFFFF); tf != nil {
		t.Errorf("footprint-only coverage (U+FFFF sentinel) = %v, want nil", tf)
	}
	if tf := match("Test", font.NormalStyle(), nil, 0xFFFF); tf != nil {
		t.Errorf("footprint-only coverage in named family = %v, want nil", tf)
	}
	// A face whose footprint claims a rune its cmap does not really map (the fontconfig-leg CI machines'
	// DejaVuSans-ExtraLight maps NUL to glyph 0) is skipped in rank order and the genuinely covering face answers, even
	// though the liar ranks first (family-sorted tie-break, as in the 'x' cases above).
	liar := newTestFaceRec(t, "DejaVuSans.subset.ttf", 0)
	liar.runes.Add(0)
	m2 := newManager([]*faceRec{liar, newTestFaceRec(t, "Roboto-Regular.ttf", 0)})
	if tf := m2.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 0); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("lying footprint for NUL = %v, want the genuinely covering Roboto", tf)
	}
	// Invalid code points are nil regardless of coverage.
	for _, ch := range []int32{0xD800, 0xDFFF, 0x110000, 0x1FFFFF, -1} {
		if tf := match("Test", font.NormalStyle(), nil, ch); tf != nil {
			t.Errorf("invalid character %#x = %v, want nil", ch, tf)
		}
	}
}

func TestManagerUnloadableFace(t *testing.T) {
	// A face whose file has vanished since the scan: metadata falls back to the footprint aspect, typeface creation
	// fails, and style matching skips it in rank order rather than failing.
	ghost := &faceRec{
		key:    "test",
		path:   "../font/testdata/does-not-exist.ttf",
		approx: font.BoldStyle(),
	}
	realFace := newTestFaceRec(t, "test.ttc", 0)
	m := newManager([]*faceRec{ghost, realFace})
	set := m.MatchFamily("Test")
	if got := set.Count(); got != 2 {
		t.Fatalf("count = %d, want 2", got)
	}
	if style, name := set.Style(0); style != font.BoldStyle() || name != "" {
		t.Errorf("ghost Style() = %v %q, want approx bold and no name", style, name)
	}
	if tf := set.CreateTypeface(0); tf != nil {
		t.Errorf("ghost CreateTypeface != nil")
	}
	// The bold pattern scores the ghost first, but the loadable regular face must answer.
	tf := set.MatchStyle(font.BoldStyle())
	if tf == nil || tf.Style().Weight() != 400 {
		t.Errorf("MatchStyle(bold) = %v, want the loadable regular face", tf)
	}
	// A family made only of unloadable faces still reports its normalized key as the display name.
	ghostOnly := newManager([]*faceRec{{key: "ghost", path: "nope.ttf"}})
	if got := ghostOnly.FamilyName(0); got != "ghost" {
		t.Errorf("ghost family name = %q, want key fallback", got)
	}
}

func TestManagerConcurrentAccess(t *testing.T) {
	// Manager is documented thread-safe; the lazy per-face loads must be too (exercised under -race).
	m := newTestManager(t)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range m.CountFamilies() {
				name := m.FamilyName(i)
				set := m.MatchFamily(name)
				for j := range set.Count() {
					set.Style(j)
					set.CreateTypeface(j)
				}
				m.MatchFamilyStyle(name, font.BoldStyle())
				m.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"en"}, 'x')
			}
		}()
	}
	wg.Wait()
}

func TestManagerHiddenFamilies(t *testing.T) {
	// A hidden (dot-prefixed) family: excluded from enumeration, still matchable by name, and reachable through
	// character fallback only when no visible font covers.
	hidden := newTestFaceRec(t, "Roboto-Regular.ttf", 0)
	hidden.key = ".hiddensans" // keys are always in normalized (lowercase, space-free) form
	visible := newTestFaceRec(t, "DejaVuSans.subset.ttf", 0)
	m := newManager([]*faceRec{hidden, visible})
	if got := m.CountFamilies(); got != 1 {
		t.Fatalf("CountFamilies = %d, want 1 (hidden excluded)", got)
	}
	if got := m.FamilyName(0); got != "DejaVu Sans" {
		t.Errorf("FamilyName(0) = %q, want DejaVu Sans", got)
	}
	if got := m.MatchFamily(".Hidden Sans").Count(); got != 1 {
		t.Errorf("MatchFamily(.Hidden Sans) count = %d, want 1 (hidden families stay matchable)", got)
	}
	// 'x' is covered by both; the visible font wins. NUL is covered only by the hidden Roboto, which then answers as
	// the last tier.
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != "DejaVu Sans" {
		t.Errorf("fallback for 'x' = %v, want the visible DejaVu Sans", tf)
	}
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 0); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("fallback for NUL = %v, want the hidden face", tf)
	}
}

func TestStyleFromAspect(t *testing.T) {
	cases := []struct {
		aspect              tsfont.Aspect
		weight, width, slnt int
	}{
		{aspect: tsfont.Aspect{}, weight: 400, width: 5, slnt: 0}, // zero values default to normal
		{aspect: tsfont.Aspect{Style: tsfont.StyleItalic, Weight: tsfont.WeightBold, Stretch: tsfont.StretchNormal}, weight: 700, width: 5, slnt: 1},
		{aspect: tsfont.Aspect{Weight: tsfont.WeightThin, Stretch: tsfont.StretchUltraCondensed}, weight: 100, width: 1, slnt: 0},
		{aspect: tsfont.Aspect{Weight: tsfont.WeightBlack, Stretch: tsfont.StretchUltraExpanded}, weight: 900, width: 9, slnt: 0},
		{aspect: tsfont.Aspect{Stretch: 0.8}, weight: 400, width: 3, slnt: 0},  // nearest class: condensed (0.75 vs 0.875)
		{aspect: tsfont.Aspect{Stretch: 1.05}, weight: 400, width: 5, slnt: 0}, // nearest class: normal
	}
	for _, c := range cases {
		got := styleFromAspect(c.aspect)
		w, wd, sl := styleTriple(got)
		if w != c.weight || wd != c.width || sl != c.slnt {
			t.Errorf("styleFromAspect(%+v) = (%d,%d,%d), want (%d,%d,%d)", c.aspect, w, wd, sl,
				c.weight, c.width, c.slnt)
		}
	}
}

// TestDefaultManagerSystem exercises the system scan. It is opt-in (CANVAS_FONTMGR_SYSTEM=1) so the unit suite stays
// hermetic; it is the only coverage the system manager has now that the C-library comparison probe is gone.
func TestDefaultManagerSystem(t *testing.T) {
	if os.Getenv("CANVAS_FONTMGR_SYSTEM") == "" {
		t.Skip("set CANVAS_FONTMGR_SYSTEM=1 to scan system fonts")
	}
	m := Default()
	n := m.CountFamilies()
	if n == 0 {
		t.Fatal("no system font families")
	}
	t.Logf("system families: %d", n)
	for i := range n {
		name := m.FamilyName(i)
		if name == "" {
			t.Errorf("FamilyName(%d) empty", i)
			continue
		}
		if got := m.MatchFamily(name).Count(); got < 1 {
			t.Errorf("MatchFamily(%q) count = %d", name, got)
		}
	}
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'A'); tf == nil {
		t.Error("no system font covers 'A'")
	}
}
