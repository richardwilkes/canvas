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
//	DejaVuSans.subset.ttf family "Corpus Sans" (400,5,upright) "Book"     covers 'H' 'a' 'x'
//	test.ttc[0]           family "Test"        (400,5,upright) "Regular"  covers '!' '"' '0'-'4' 'A'
//	test.ttc[1]           family "Test"        (700,5,upright) "Bold"     covers '!'
//
// The sans face is really DejaVu Sans, renamed in memory because that name is a platform default family (see
// corpusSansFamily); no corpus family is a default on any GOOS, which is what TestCorpusHoldsNoPlatformDefaultFamily
// guards.
//
// Roboto's NUL/CR entries map real glyphs. The other three faces additionally carry a cmap4 sentinel segment mapping
// U+FFFF to glyph 0 (.notdef): go-text's scanner counts it into the footprint rune set, but it maps no real glyph — the
// character-fallback verification cases below rely on that.
//
// The system manager (Default) scans every system font directory and serializes an index cache into the user's cache
// dir, so nothing here calls it by default: every test that does is gated on CANVAS_FONTMGR_SYSTEM through
// requireSystemFonts, in this file and in capimigrated_test.go alike. The live oracle probe that once compared the
// system manager against the C library's platform host was removed along with that library.

package fontmgr

import (
	"bytes"
	"encoding/binary"
	"os"
	"sync"
	"testing"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/fontscan"
	"github.com/go-text/typesetting/language"
	"github.com/richardwilkes/canvas/font"
)

// newTestFaceRec builds a file-backed faceRec for one face of a testdata font, deriving the grouping key and rune
// coverage from the file exactly as the fontscan footprints would.
func newTestFaceRec(t *testing.T, file string, index int, langs ...string) *faceRec {
	t.Helper()
	path := "../font/testdata/" + file
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	f := newTestDataFaceRec(t, data, index, langs...)
	f.data, f.path = nil, path
	return f
}

// newTestDataFaceRec is newTestFaceRec over in-memory font bytes: the same derivation, against the data lane of every
// lazy read rather than the file lane.
func newTestDataFaceRec(t *testing.T, data []byte, index int, langs ...string) *faceRec {
	t.Helper()
	info, err := font.DescribeFaceData(data, index)
	if err != nil {
		t.Fatalf("face %d: describe: %v", index, err)
	}
	lds, err := opentype.NewLoaders(bytes.NewReader(data))
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
		data:   data,
		index:  index,
		runes:  runes,
		langs:  langSet,
		approx: info.Style,
	}
}

// corpusSansFamily is the family the corpus's sans face carries. The face really is DejaVu Sans, which is the *first*
// entry of defaultFamilies() on every non-darwin, non-windows GOOS: under its own name it lands in the default-family
// tier on the Linux legs, so the platform, not the test, decides which branch of lookupOrDefault and
// matchCoveringTiered answers — leaving the first-visible-family fallback and the family-sorted tie-break unexercised
// there. Renaming it to a name no platform defaults to puts the same branch under test on every GOOS. The replacement
// is the same length as the original (the rename overwrites the name strings in place) and still normalizes below
// "roboto", so the corpus's family order is unchanged.
const corpusSansFamily = "Corpus Sans"

// newSansFaceRec builds the corpus's sans face: DejaVuSans.subset.ttf renamed to corpusSansFamily. The rename lives in
// memory — the checked-in font is untouched — so the face is data-backed, which also keeps the data lane of the lazy
// reads under test alongside the file lane the rest of the corpus uses.
func newSansFaceRec(t *testing.T, langs ...string) *faceRec {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/DejaVuSans.subset.ttf")
	if err != nil {
		t.Fatal(err)
	}
	f := newTestDataFaceRec(t, renameSfntFamily(t, data, "DejaVu Sans", corpusSansFamily), 0, langs...)
	if want := tsfont.NormalizeFamily(corpusSansFamily); f.key != want {
		t.Fatalf("the renamed sans face keys as %q, want %q", f.key, want)
	}
	return f
}

// sfntTable returns tag's table bytes within a single-font sfnt file, as a slice aliasing data.
func sfntTable(t *testing.T, data []byte, tag string) []byte {
	t.Helper()
	for i := range int(binary.BigEndian.Uint16(data[4:])) {
		rec := 12 + i*16
		if string(data[rec:rec+4]) != tag {
			continue
		}
		off := int(binary.BigEndian.Uint32(data[rec+8:]))
		return data[off : off+int(binary.BigEndian.Uint32(data[rec+12:]))]
	}
	t.Fatalf("%s table not found", tag)
	return nil
}

// renameSfntFamily returns a copy of a single-font sfnt whose family-name records — the name-table IDs the family
// precedence in font.DescribeFace reads, 21, 16 and 1 — currently reading from are rewritten to to. Both names must be
// ASCII and the same length, so every name string keeps its byte length in the UTF-16BE and single-byte encodings
// alike and no offset in the table moves; records sharing a rewritten string (the full name and unique ID commonly
// alias the family name) are rewritten with it.
func renameSfntFamily(t *testing.T, data []byte, from, to string) []byte {
	t.Helper()
	if len(from) != len(to) {
		t.Fatalf("%q and %q differ in length; the rewrite must not move a name-table offset", from, to)
	}
	out := bytes.Clone(data)
	tab := sfntTable(t, out, "name")
	strOff := int(binary.BigEndian.Uint16(tab[4:]))
	renamed := 0
	for i := range int(binary.BigEndian.Uint16(tab[2:])) {
		rec := 6 + i*12
		switch binary.BigEndian.Uint16(tab[rec+6:]) { // nameID
		case 1, 16, 21:
		default:
			continue
		}
		// Platform 1 (Macintosh) encodes one byte per character; the Unicode and Windows platforms use UTF-16BE.
		wide := binary.BigEndian.Uint16(tab[rec:]) != 1
		off := strOff + int(binary.BigEndian.Uint16(tab[rec+10:]))
		s := tab[off : off+int(binary.BigEndian.Uint16(tab[rec+8:]))]
		if !bytes.Equal(s, encodeSfntName(from, wide)) {
			continue
		}
		copy(s, encodeSfntName(to, wide))
		renamed++
	}
	if renamed == 0 {
		t.Fatalf("no family-name record reads %q", from)
	}
	return out
}

// encodeSfntName encodes an ASCII name-table string, either as UTF-16BE or as one byte per character.
func encodeSfntName(s string, wide bool) []byte {
	if !wide {
		return []byte(s)
	}
	out := make([]byte, 0, len(s)*2)
	for _, r := range s {
		out = binary.BigEndian.AppendUint16(out, uint16(r))
	}
	return out
}

// readTestFontData returns the raw bytes of a font in the shared corpus, for the in-memory (data-backed) faces.
func readTestFontData(t *testing.T, file string) []byte {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/" + file)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// newTestManager builds the standard four-face corpus manager: Roboto tagged "en", Corpus Sans tagged "fr".
func newTestManager(t *testing.T) *Manager {
	t.Helper()
	return newManager([]*faceRec{
		newTestFaceRec(t, "Roboto-Regular.ttf", 0, "en"),
		newSansFaceRec(t, "fr"),
		newTestFaceRec(t, "test.ttc", 0),
		newTestFaceRec(t, "test.ttc", 1),
	})
}

func TestCorpusHoldsNoPlatformDefaultFamily(t *testing.T) {
	// The branch split the whole corpus rests on: no corpus family may be one of this platform's default families, or
	// the default-family branches of lookupOrDefault and matchCoveringTiered answer the empty-family and
	// character-fallback cases instead of the first-visible-family fallback and the family-sorted tie-break those cases
	// are written for — silently, and only on the platforms whose defaults collide (see corpusSansFamily).
	m := newTestManager(t)
	for _, name := range defaultFamilies() {
		if fam := m.byKey[tsfont.NormalizeFamily(name)]; fam != nil {
			t.Errorf("the corpus holds %q, one of this platform's default families %v", name, defaultFamilies())
		}
	}
}

// familyOf names a matched typeface's family for a failure message (a %v verb on the typeface itself dumps the whole
// parsed font, font-file bytes included).
func familyOf(tf *font.Typeface) string {
	if tf == nil {
		return "nil"
	}
	return tf.FamilyName()
}

func styleTriple(s font.Style) (weight, width, slant int) {
	return s.Weight(), s.Width(), int(s.Slant())
}

func TestManagerEnumeration(t *testing.T) {
	m := newTestManager(t)
	if got := m.CountFamilies(); got != 3 {
		t.Fatalf("CountFamilies = %d, want 3", got)
	}
	// Families sort by normalized key: corpussans < roboto < test.
	want := []string{corpusSansFamily, "Roboto", "Test"}
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
	// Unknown families produce an empty, non-nil set (MatchFamily never returns nil).
	set := m.MatchFamily("Non Existing Family Name")
	if set == nil {
		t.Fatal("MatchFamily(unknown) = nil")
	}
	if got := set.Count(); got != 0 {
		t.Errorf("MatchFamily(unknown) count = %d, want 0", got)
	}
	if tf := set.MatchStyle(font.NormalStyle()); tf != nil {
		t.Errorf("empty set MatchStyle != nil")
	}
	// An empty name requests the platform default family, as MatchFamilyStyle does with it (see
	// TestManagerMatchFamilyEmptyNameResolvesTheDefault); only a manager with no families at all answers it empty.
	if got := m.MatchFamily("").Count(); got == 0 {
		t.Error(`MatchFamily("") resolved no default family`)
	}
	if set = newManager(nil).MatchFamily(""); set == nil || set.Count() != 0 {
		t.Errorf(`empty manager MatchFamily("") = %v, want an empty, non-nil set`, set)
	}
}

func TestManagerMatchFamilyEmptyNameResolvesTheDefault(t *testing.T) {
	// MatchFamily and MatchFamilyStyle document the same contract for an empty (a host's null) family name — the
	// default system family — so both must answer out of the same family. Keying a corpus face to the platform's own
	// first default name puts lookupOrDefault's default-family branch, rather than its first-visible-family fallback,
	// under the test on every GOOS.
	defaults := defaultFamilies()
	dflt := newTestFaceRec(t, "test.ttc", 1) // the bold face, so the answering face is identifiable by style name
	dflt.key = tsfont.NormalizeFamily(defaults[0])
	m := newManager([]*faceRec{newTestFaceRec(t, "Roboto-Regular.ttf", 0), dflt})
	set := m.MatchFamily("")
	if got := set.Count(); got != 1 {
		t.Fatalf(`MatchFamily("") count = %d, want 1 (the default family's one face)`, got)
	}
	if _, name := set.Style(0); name != "Bold" {
		t.Errorf(`MatchFamily("") holds the %q face, want the default family's Bold face`, name)
	}
	viaSet := set.MatchStyle(font.NormalStyle())
	viaStyle := m.MatchFamilyStyle("", font.NormalStyle())
	if viaSet == nil || viaSet != viaStyle {
		t.Errorf(`MatchFamily("").MatchStyle() = %s, MatchFamilyStyle("") = %s: the two disagree`,
			familyOf(viaSet), familyOf(viaStyle))
	}
	// They keep agreeing in the four-face corpus, which holds no default family on any platform, so the other branch —
	// the first-visible-family fallback — is what answers there.
	m = newTestManager(t)
	viaSet, viaStyle = m.MatchFamily("").MatchStyle(font.NormalStyle()), m.MatchFamilyStyle("", font.NormalStyle())
	if viaSet == nil || viaSet != viaStyle {
		t.Errorf(`corpus MatchFamily("").MatchStyle() = %s, MatchFamilyStyle("") = %s`,
			familyOf(viaSet), familyOf(viaStyle))
	}
}

func TestManagerCharacterFallbackScoresDefaultsInOrder(t *testing.T) {
	// defaultFamilies is a search order, not one pool: the first default family present answers whenever it covers the
	// character, even when a later default family holds the better style match. Keying corpus faces to the platform's
	// own default names — over a corpus that holds none of them under its own names — makes the tiers testable on every
	// GOOS.
	defaults := defaultFamilies()
	first := newTestFaceRec(t, "Roboto-Regular.ttf", 0) // regular weight: a near miss for a bold request
	first.key = tsfont.NormalizeFamily(defaults[0])
	second := newTestFaceRec(t, "test.ttc", 1) // the exact bold match, covering '!'
	second.key = tsfont.NormalizeFamily(defaults[1])
	m := newManager([]*faceRec{newSansFaceRec(t), second, first})
	if tf := m.MatchFamilyStyleCharacter("", font.BoldStyle(), nil, '!'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("bold '!' = %s, want Roboto, the first default family's regular face (the second default is an exact "+
			"style match, but it is second)", familyOf(tf))
	}
	// The non-default families still rank below every default: Corpus Sans covers 'x' and ties Roboto on style, but the
	// default-family tier answers first.
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("normal 'x' = %s, want the default family's Roboto over the non-default Corpus Sans", familyOf(tf))
	}
	// A default family that does not cover the character drops out and the next default in the order answers — still
	// ahead of the non-default families that cover it too.
	first = newTestFaceRec(t, "test.ttc", 1) // covers '!' and nothing else
	first.key = tsfont.NormalizeFamily(defaults[0])
	second = newTestFaceRec(t, "Roboto-Regular.ttf", 0)
	second.key = tsfont.NormalizeFamily(defaults[1])
	m = newManager([]*faceRec{newSansFaceRec(t), second, first})
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("normal 'x' = %s, want the second default family's Roboto (the first does not cover it)", familyOf(tf))
	}
}

func TestNewFromData(t *testing.T) {
	// The in-memory manager: every lazy read — display metadata, the coverage probe, the typeface load — has to answer
	// out of the blob the face came with, since none of these faces has a file behind it.
	m := NewFromData(readTestFontData(t, "Roboto-Regular.ttf"), readTestFontData(t, "test.ttc"),
		[]byte("not a font"), nil)
	// Unparsable blobs are skipped, and the collection contributes both of its faces to one family.
	if got := m.CountFamilies(); got != 2 {
		t.Fatalf("CountFamilies = %d, want 2 (Roboto and Test)", got)
	}
	for i, want := range []string{"Roboto", "Test"} {
		if got := m.FamilyName(i); got != want {
			t.Errorf("FamilyName(%d) = %q, want %q", i, got, want)
		}
	}
	for _, f := range m.all {
		if f.path != "" {
			t.Fatalf("data-backed face %q carries the path %q", f.key, f.path)
		}
		if len(f.data) == 0 {
			t.Fatalf("data-backed face %q carries no data", f.key)
		}
	}
	set := m.MatchFamily("Test")
	if got := set.Count(); got != 2 {
		t.Fatalf("MatchFamily(Test) count = %d, want 2", got)
	}
	for i, want := range []struct {
		name   string
		weight int
	}{{name: "Regular", weight: 400}, {name: "Bold", weight: 700}} {
		style, name := set.Style(i)
		if name != want.name || style.Weight() != want.weight {
			t.Errorf("Style(%d) = %q weight %d, want %q weight %d", i, name, style.Weight(), want.name, want.weight)
		}
	}
	tf := m.MatchFamilyStyle("Test", font.BoldStyle())
	if tf == nil || tf.FamilyName() != "Test" || tf.Style().Weight() != 700 {
		t.Errorf("MatchFamilyStyle(Test, bold) = %s, want the bold Test face parsed out of the blob", familyOf(tf))
	}
	// Character fallback runs over the coverage NewFromData read from each cmap: only Roboto has 'x' here.
	if tf = m.MatchFamilyStyleCharacter("Test", font.NormalStyle(), nil, 'x'); tf == nil ||
		tf.FamilyName() != "Roboto" {
		t.Errorf("fallback for 'x' = %s, want Roboto", familyOf(tf))
	}
	// That coverage is exact where a scan footprint over-claims: the test.ttc faces' cmap4 sentinel segment maps U+FFFF
	// to glyph 0, so no face here even claims the rune.
	for _, f := range m.all {
		if f.runes.Contains(0xFFFF) {
			t.Errorf("face %q claims the U+FFFF .notdef sentinel", f.key)
		}
	}
	if tf = m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 0xFFFF); tf != nil {
		t.Errorf("fallback for the U+FFFF sentinel = %s, want nil", familyOf(tf))
	}
	// Nothing usable supplied: an empty manager, never nil.
	if got := NewFromData([]byte("not a font")).CountFamilies(); got != 0 {
		t.Errorf("garbage-only CountFamilies = %d, want 0", got)
	}
	if got := NewFromData().CountFamilies(); got != 0 {
		t.Errorf("no-blob CountFamilies = %d, want 0", got)
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
	// The empty name requests the platform default family; no corpus family is one of the defaults on any GOOS
	// (TestCorpusHoldsNoPlatformDefaultFamily), so lookupOrDefault's first-visible-family fallback answers with Corpus
	// Sans everywhere.
	tf := m.MatchFamilyStyle("", font.NormalStyle())
	if tf == nil || tf.FamilyName() != corpusSansFamily {
		t.Errorf("MatchFamilyStyle(\"\") = %s, want the first family", familyOf(tf))
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
	// The family does not cover 'x': fall back to the global scan. Roboto and Corpus Sans tie on style, and the
	// family-sorted candidate order makes Corpus Sans (first) win.
	if tf := match("Test", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != corpusSansFamily {
		t.Errorf("(Test, normal, 'x') = %s, want Corpus Sans", familyOf(tf))
	}
	// No family: same global scan.
	if tf := match("", font.NormalStyle(), nil, 'x'); tf == nil || tf.FamilyName() != corpusSansFamily {
		t.Errorf("(\"\", normal, 'x') = %s, want Corpus Sans", familyOf(tf))
	}
	// BCP-47: the most significant tag is last; 'x' is covered by both tagged fonts.
	if tf := match("", font.NormalStyle(), []string{"fr", "en"}, 'x'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [fr en] = %v, want Roboto (en most significant)", tf)
	}
	if tf := match("", font.NormalStyle(), []string{"en", "fr"}, 'x'); tf == nil || tf.FamilyName() != corpusSansFamily {
		t.Errorf("bcp47 [en fr] = %s, want Corpus Sans (fr most significant)", familyOf(tf))
	}
	// Unmatched tags fall through to the next most significant, then to the unrestricted scan.
	if tf := match("", font.NormalStyle(), []string{"en", "ja"}, 'x'); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [en ja] = %v, want Roboto (ja unmatched, en next)", tf)
	}
	if tf := match("", font.NormalStyle(), []string{"ja"}, 'x'); tf == nil || tf.FamilyName() != corpusSansFamily {
		t.Errorf("bcp47 [ja] = %s, want Corpus Sans (unrestricted scan)", familyOf(tf))
	}
	// A derived language tag maps to its primary ("fr-CA" → "fr").
	if tf := match("", font.NormalStyle(), []string{"fr-CA"}, 'x'); tf == nil || tf.FamilyName() != corpusSansFamily {
		t.Errorf("bcp47 [fr-CA] = %s, want Corpus Sans", familyOf(tf))
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
	// Footprint-only coverage is not coverage: the sans face and both Test faces carry the cmap4 sentinel segment
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
	liar := newSansFaceRec(t)
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

// langSetOf builds the fontscan language set for the given BCP-47 tags.
func langSetOf(t *testing.T, langs ...string) fontscan.LangSet {
	t.Helper()
	var set fontscan.LangSet
	for _, lang := range langs {
		id, ok := language.NewLangID(language.NewLanguage(lang))
		if !ok {
			t.Fatalf("unknown test language %q", lang)
		}
		set.Add(id)
	}
	return set
}

func TestManagerMatchCharacterBCP47Fallthrough(t *testing.T) {
	// A BCP-47 tag whose candidates all fail the cmap verification (or fail to load) must not fail closed: the search
	// continues with the next less-significant tag and then the unrestricted scan, exactly as the named-family pass
	// does. The contract promises nil only when no known font covers the character.
	roboto := func() *faceRec { return newTestFaceRec(t, "Roboto-Regular.ttf", 0, "en") }
	// A liar: the footprint claims NUL (as the fontconfig-leg CI machines' DejaVuSans-ExtraLight does) but its cmap
	// maps NUL to glyph 0. Roboto maps NUL to a real glyph.
	newLiar := func(langs ...string) *faceRec {
		f := newSansFaceRec(t, langs...)
		f.runes.Add(0)
		return f
	}
	m := newManager([]*faceRec{newLiar("fr"), roboto()})
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"en", "fr"}, 0); tf == nil ||
		tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [en fr] with a lying fr candidate = %v, want Roboto (fall through to en)", tf)
	}
	// The genuinely covering face carries no language tag at all: the unrestricted scan must still find it.
	m = newManager([]*faceRec{newLiar("fr"), newTestFaceRec(t, "Roboto-Regular.ttf", 0)})
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"fr"}, 0); tf == nil ||
		tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [fr] with a lying fr candidate = %v, want Roboto (unrestricted scan)", tf)
	}
	// Same for a candidate that cannot be loaded: the footprint claims 'x' but the file has vanished.
	ghost := &faceRec{
		key:    "ghost",
		path:   "../font/testdata/does-not-exist.ttf",
		runes:  newSansFaceRec(t).runes,
		langs:  langSetOf(t, "fr"),
		approx: font.NormalStyle(),
	}
	m = newManager([]*faceRec{ghost, roboto()})
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"en", "fr"}, 'x'); tf == nil ||
		tf.FamilyName() != "Roboto" {
		t.Errorf("bcp47 [en fr] with an unloadable fr candidate = %v, want Roboto (fall through to en)", tf)
	}
	// Falling through never invents coverage: when nothing genuinely covers the character the answer is still nil.
	m = newManager([]*faceRec{newLiar("fr"), newTestFaceRec(t, "test.ttc", 0, "en")})
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"en", "fr"}, 0); tf != nil {
		t.Errorf("bcp47 [en fr] with no genuine coverage = %v, want nil", tf)
	}
	// A tag with a genuinely covering candidate still wins over the less significant ones (no over-eager fall-through).
	m = newManager([]*faceRec{newSansFaceRec(t, "fr"), roboto()})
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), []string{"en", "fr"}, 'x'); tf == nil ||
		tf.FamilyName() != corpusSansFamily {
		t.Errorf("bcp47 [en fr] = %s, want Corpus Sans (fr most significant and covering)", familyOf(tf))
	}
}

// loadedTypefaces counts the manager's faces whose full typeface has been loaded, and is therefore cached — with the
// whole font file's bytes inside it — for the life of the manager. Safe to read directly: the caller is the only
// goroutine touching the manager.
func loadedTypefaces(m *Manager) int {
	n := 0
	for _, f := range m.all {
		if f.tf != nil {
			n++
		}
	}
	return n
}

func TestManagerCharacterFallbackLoadsOnlyTheAnswer(t *testing.T) {
	// Character-fallback candidates are verified from their cmap alone (faceRec.covers). Verifying by loading instead
	// retained one fully parsed font — file bytes included — per rejected candidate for the life of the manager, and the
	// footprint rune sets over-claim exactly the characters that reject the most candidates (U+0000/U+FFFF .notdef
	// sentinels), so a single call could retain most of a system inventory.
	m := newTestManager(t)
	// U+FFFF: the sans face and both test.ttc faces claim it in their footprints via the cmap4 sentinel segment and
	// none of them maps it. The answer is nil, and no candidate may be loaded to establish that.
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 0xFFFF); tf != nil {
		t.Fatalf("footprint-only coverage (U+FFFF) = %v, want nil", tf)
	}
	if n := loadedTypefaces(m); n != 0 {
		t.Errorf("%d typefaces cached rejecting 3 lying footprints, want 0", n)
	}
	// The same walk with a real answer at the end of it: the higher-ranked liar (family-sorted, so it is scored first)
	// must be rejected without a load, and only the answering face ends up cached.
	liar := newSansFaceRec(t)
	liar.runes.Add(0)
	roboto := newTestFaceRec(t, "Roboto-Regular.ttf", 0)
	m2 := newManager([]*faceRec{liar, roboto})
	if tf := m2.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 0); tf == nil || tf.FamilyName() != "Roboto" {
		t.Fatalf("NUL = %v, want the genuinely covering Roboto", tf)
	}
	if liar.tf != nil {
		t.Error("the rejected candidate's typeface was loaded and cached")
	}
	if roboto.tf == nil {
		t.Error("the answering candidate's typeface was not cached")
	}
	if n := loadedTypefaces(m2); n != 1 {
		t.Errorf("%d typefaces cached, want 1 (the answer only)", n)
	}
	// The named-family pass verifies the same way: "Test" claims U+FFFF in both its faces and maps it in neither.
	m3 := newTestManager(t)
	if tf := m3.MatchFamilyStyleCharacter("Test", font.NormalStyle(), nil, 0xFFFF); tf != nil {
		t.Fatalf("footprint-only coverage in a named family = %v, want nil", tf)
	}
	if n := loadedTypefaces(m3); n != 0 {
		t.Errorf("%d typefaces cached rejecting a named family's lying footprints, want 0", n)
	}
}

func TestFaceRecCovers(t *testing.T) {
	// covers answers from the cmap without loading the typeface, and its single-entry memo must not let one rune's
	// verdict answer for another.
	f := newTestFaceRec(t, "Roboto-Regular.ttf", 0)
	for i := range 2 {
		if !f.covers('A') {
			t.Errorf("pass %d: Roboto does not cover 'A'", i)
		}
		if f.covers(0x4E2D) {
			t.Errorf("pass %d: Roboto covers U+4E2D", i)
		}
		if !f.covers('A') {
			t.Errorf("pass %d: Roboto does not cover 'A' on re-probe", i)
		}
	}
	if f.tf != nil {
		t.Error("covers loaded the typeface")
	}
	// The in-memory branch (a faceRec with data rather than a path) answers identically.
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	inMem := &faceRec{key: "roboto", data: data}
	if !inMem.covers('A') || inMem.covers(0x4E2D) {
		t.Error("in-memory face disagrees with the file-backed one")
	}
	// An unloadable face covers nothing rather than reporting the footprint's claim.
	ghost := &faceRec{key: "ghost", path: "../font/testdata/does-not-exist.ttf"}
	if ghost.covers('A') {
		t.Error("a vanished file reported coverage")
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
	visible := newSansFaceRec(t)
	m := newManager([]*faceRec{hidden, visible})
	if got := m.CountFamilies(); got != 1 {
		t.Fatalf("CountFamilies = %d, want 1 (hidden excluded)", got)
	}
	if got := m.FamilyName(0); got != corpusSansFamily {
		t.Errorf("FamilyName(0) = %q, want %q", got, corpusSansFamily)
	}
	if got := m.MatchFamily(".Hidden Sans").Count(); got != 1 {
		t.Errorf("MatchFamily(.Hidden Sans) count = %d, want 1 (hidden families stay matchable)", got)
	}
	// 'x' is covered by both; the visible font wins. NUL is covered only by the hidden Roboto, which then answers as
	// the last tier.
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 'x'); tf == nil ||
		tf.FamilyName() != corpusSansFamily {
		t.Errorf("fallback for 'x' = %s, want the visible Corpus Sans", familyOf(tf))
	}
	if tf := m.MatchFamilyStyleCharacter("", font.NormalStyle(), nil, 0); tf == nil || tf.FamilyName() != "Roboto" {
		t.Errorf("fallback for NUL = %v, want the hidden face", tf)
	}
	// An empty family name lands on lookupOrDefault's first-*visible*-family fallback, no corpus family being one of
	// this platform's defaults. The hidden family sorts before the visible one (a dot leads every letter), so this is
	// the arrangement that tells that fallback apart from the all-families one behind it — over a corpus whose only
	// family is visible, either answers the same face.
	for _, c := range []struct {
		got  *font.Typeface
		name string
	}{
		{got: m.MatchFamilyStyle("", font.NormalStyle()), name: `MatchFamilyStyle("")`},
		{got: m.MatchFamily("").MatchStyle(font.NormalStyle()), name: `MatchFamily("").MatchStyle()`},
	} {
		if c.got == nil || c.got.FamilyName() != corpusSansFamily {
			t.Errorf("%s = %s, want the first visible family, not the hidden one sorting ahead of it", c.name,
				familyOf(c.got))
		}
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

// requireSystemFonts skips unless the system-font tests have been opted into. Default's first call runs
// fontscan.SystemFonts, which walks every system font directory and serializes a font_index_v*.cache file into the
// user's cache directory, so a test that reaches it is neither hermetic nor free of side effects and its runtime
// depends on the machine's font inventory. Every test that calls Default gates on this.
func requireSystemFonts(t *testing.T) {
	t.Helper()
	if os.Getenv("CANVAS_FONTMGR_SYSTEM") == "" {
		t.Skip("set CANVAS_FONTMGR_SYSTEM=1 to scan system fonts")
	}
}

// TestDefaultManagerSystem exercises the system scan end to end. Like the other system-font tests it is opt-in, so the
// unit suite stays hermetic.
func TestDefaultManagerSystem(t *testing.T) {
	requireSystemFonts(t)
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
