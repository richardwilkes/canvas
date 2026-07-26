// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The font manager surface: family enumeration, family/style/character matching, style sets, and load-from-data, built
// over go-text/typesetting's fontscan index. Font metadata is host-independent: footprints (file location, normalized
// family, rune coverage, language coverage, aspect) group into families; per-face display names, style names, and font
// styles come from a lightweight table read (font.DescribeFaceFile) on demand; style matching uses the CSS3
// style-matching scoring in matchStyleCSS3; character fallback probes the footprints' cmap-derived rune coverage with
// BCP-47 hints ordered least-to-most significant, then verifies the chosen face really maps the character through its
// own cmap before answering.
//
// This is one host-independent implementation where a platform library would have three hosts (CoreText, fontconfig,
// DirectWrite), so its behavior is the documented font-manager contract plus the majority host behavior rather than any
// single host byte-for-byte. The inventory is fontscan's on-disk scan, so it can include families a host has not
// activated and miss host-virtual families; hidden (dot-prefixed) families are excluded from enumeration but stay
// matchable by name and reachable through character fallback; enumeration is ordered by normalized family name; family
// lookup is case- and space-insensitive with no fontconfig-style alias substitution; matchStyleCSS3 is the one
// style-distance algorithm on every platform; and when nothing covers a character, matchFamilyStyleCharacter returns
// nil per the documented contract.

package fontmgr

import (
	"runtime"
	"sort"
	"strings"
	"sync"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/fontscan"
	"github.com/go-text/typesetting/language"
	"github.com/richardwilkes/canvas/font"
)

// Manager is an immutable view of a font collection. All methods are safe for concurrent use. The zero value is an
// empty manager (no families); use Default for the system manager.
type Manager struct {
	byKey    map[string]*Family
	families []*Family // sorted by normalized family key
	visible  []*Family // families minus the hidden (dot-prefixed) ones — the enumeration list
	all      []*faceRec
}

// Family is one font family: the set of faces whose (normalized) family name matches.
type Family struct {
	key      string
	name     string
	faces    []*faceRec
	nameOnce sync.Once
}

// faceRec is one face known to the manager. Display metadata (family/style names, the OS/2-derived style) and the
// typeface itself load lazily and are cached for the life of the manager (cache aggressively).
type faceRec struct {
	tf       *font.Typeface
	key      string
	path     string
	data     []byte
	info     font.FaceInfo
	runes    fontscan.RuneSet
	langs    fontscan.LangSet
	index    int
	approx   font.Style
	infoOK   bool
	infoOnce sync.Once
	tfOnce   sync.Once
}

// faceInfo returns the lazily-loaded lightweight description (family/style names + style).
func (f *faceRec) faceInfo() (font.FaceInfo, bool) {
	f.infoOnce.Do(func() {
		var err error
		if f.path != "" {
			f.info, err = font.DescribeFaceFile(f.path, f.index)
		} else {
			f.info, err = font.DescribeFaceData(f.data, f.index)
		}
		f.infoOK = err == nil
	})
	return f.info, f.infoOK
}

// style returns the face's font style: the OS/2-derived style from the table read when available (always equal to the
// style of the typeface this face creates), else the footprint's approximate aspect.
func (f *faceRec) style() font.Style {
	if info, ok := f.faceInfo(); ok {
		return info.Style
	}
	return f.approx
}

func (f *faceRec) styleName() string {
	info, _ := f.faceInfo()
	return info.StyleName
}

// typeface loads (and caches) the full typeface for this face; nil if the file no longer parses.
func (f *faceRec) typeface() *font.Typeface {
	f.tfOnce.Do(func() {
		var err error
		if f.path != "" {
			f.tf, err = font.NewTypefaceFromFile(f.path, f.index)
		} else {
			f.tf, err = font.NewTypefaceFromData(f.data, f.index)
		}
		if err != nil {
			f.tf = nil
		}
	})
	return f.tf
}

// Name returns the family's display name: the first face's described family (which normalizes back to the grouping key
// by construction), falling back to the normalized key if no face parses.
func (f *Family) Name() string {
	f.nameOnce.Do(func() {
		for _, face := range f.faces {
			if info, ok := face.faceInfo(); ok && info.Family != "" {
				f.name = info.Family
				return
			}
		}
		f.name = f.key
	})
	return f.name
}

// discardLogger silences fontscan's informational logging.
type discardLogger struct{}

func (discardLogger) Printf(string, ...any) {}

var (
	defaultOnce sync.Once
	defaultMgr  *Manager
)

// Default returns the process-wide system font manager. The first call scans the system fonts through fontscan (an
// on-disk index cache makes later runs cheap); if the scan fails the manager is empty.
func Default() *Manager {
	defaultOnce.Do(func() {
		fps, err := fontscan.SystemFonts(discardLogger{}, "")
		if err != nil {
			defaultMgr = newManager(nil)
			return
		}
		recs := make([]*faceRec, 0, len(fps))
		for i := range fps {
			fp := &fps[i]
			if fp.Family == "" {
				continue
			}
			recs = append(recs, &faceRec{
				key:    fp.Family,
				path:   fp.Location.File,
				index:  int(fp.Location.Index),
				runes:  fp.Runes,
				langs:  fp.Langs,
				approx: styleFromAspect(fp.Aspect),
			})
		}
		defaultMgr = newManager(recs)
	})
	return defaultMgr
}

// newManager groups faces into families (sorted by normalized key; faces keep their scan order within a family).
func newManager(recs []*faceRec) *Manager {
	m := &Manager{byKey: make(map[string]*Family)}
	for _, rec := range recs {
		fam := m.byKey[rec.key]
		if fam == nil {
			fam = &Family{key: rec.key}
			m.byKey[rec.key] = fam
			m.families = append(m.families, fam)
		}
		fam.faces = append(fam.faces, rec)
	}
	sort.Slice(m.families, func(i, j int) bool { return m.families[i].key < m.families[j].key })
	for _, fam := range m.families {
		m.all = append(m.all, fam.faces...)
		// Hidden families (macOS's dot-prefixed UI fonts) stay matchable by name and reachable through character
		// fallback, but are excluded from enumeration, matching CTFontManagerCopyAvailableFontFamilyNames.
		if !strings.HasPrefix(fam.key, ".") {
			m.visible = append(m.visible, fam)
		}
	}
	return m
}

// CountFamilies returns the number of enumerable families.
func (m *Manager) CountFamilies() int { return len(m.visible) }

// FamilyName returns the display name of the index-th family, "" out of range.
func (m *Manager) FamilyName(index int) string {
	if index < 0 || index >= len(m.visible) {
		return ""
	}
	return m.visible[index].Name()
}

// MatchFamily returns the style set for the named family (normalized, so the lookup is case- and space-insensitive like
// the platform hosts'). Unknown or empty names yield an empty set, never nil.
func (m *Manager) MatchFamily(familyName string) *StyleSet {
	if familyName == "" {
		return &StyleSet{}
	}
	fam := m.byKey[tsfont.NormalizeFamily(familyName)]
	if fam == nil {
		return &StyleSet{}
	}
	return &StyleSet{faces: fam.faces}
}

// MatchFamilyStyle returns the best style match within the named family, nil when the family is unknown (both the
// CoreText and fontconfig hosts return null after their family-name check). An empty family name requests the platform
// default family, as the hosts do for a null familyName.
func (m *Manager) MatchFamilyStyle(familyName string, style font.Style) *font.Typeface {
	fam := m.lookupOrDefault(familyName)
	if fam == nil {
		return nil
	}
	return (&StyleSet{faces: fam.faces}).MatchStyle(style)
}

func (m *Manager) lookupOrDefault(familyName string) *Family {
	if familyName != "" {
		return m.byKey[tsfont.NormalizeFamily(familyName)]
	}
	for _, name := range defaultFamilies() {
		if fam := m.byKey[tsfont.NormalizeFamily(name)]; fam != nil {
			return fam
		}
	}
	if len(m.visible) != 0 {
		return m.visible[0]
	}
	if len(m.families) != 0 {
		return m.families[0]
	}
	return nil
}

// defaultFamilies is the platform default-family search order used when no family name is given, approximating each
// host's default font (CoreText: Helvetica; DirectWrite: Segoe UI; fontconfig: its sans-serif alias).
func defaultFamilies() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"Helvetica", "Helvetica Neue", "Arial"}
	case "windows":
		return []string{"Segoe UI", "Tahoma", "Arial"}
	default:
		return []string{"DejaVu Sans", "Liberation Sans", "Noto Sans", "FreeSans", "Arial"}
	}
}

// MatchFamilyStyleCharacter finds a typeface covering character, preferring the named family, then fonts covering the
// most significant matching BCP-47 language (bcp47[0] is the least significant, bcp47[len-1] the most), then the
// closest style. Returns nil for invalid code points (surrogates, out of range — the CoreText host's CFString
// rejection) and when no known font covers the character (the fontconfig host's FontContainsCharacter check; the
// CoreText host instead returns the family font). "Covers" means the face's cmap maps the character to a real glyph:
// candidates whose footprint claims the rune but whose cmap yields glyph 0 are skipped (matchCovering), so the returned
// face always maps character through its own cmap.
func (m *Manager) MatchFamilyStyleCharacter(familyName string, style font.Style, bcp47 []string, character int32) *font.Typeface {
	if character < 0 || character > 0x10FFFF || (character >= 0xD800 && character <= 0xDFFF) {
		return nil
	}
	r := character
	// The named family wins when any of its faces covers the character (the CoreText host resolves the styled family
	// font first and CTFontCreateForString returns it unchanged when it covers the string).
	if familyName != "" {
		if fam := m.byKey[tsfont.NormalizeFamily(familyName)]; fam != nil {
			if tf := matchCovering(fam.faces, style, r, false); tf != nil {
				return tf
			}
		}
	}
	// BCP-47 pass: most significant tag last. Each tag (scanning back to front) restricts the candidate set to the fonts
	// claiming both the rune and the language; the first tag whose restricted set yields a verified covering face wins.
	// A tag with candidates that all fail verification (a lying footprint) or fail to load falls through to the next
	// less-significant tag and finally to the unrestricted scan, as the named-family pass above does — the contract
	// promises nil only when no known font covers the character.
	for i := len(bcp47) - 1; i >= 0; i-- {
		id, ok := language.NewLangID(language.NewLanguage(bcp47[i]))
		if !ok {
			continue
		}
		var candidates []*faceRec
		for _, f := range m.all {
			if f.runes.Contains(r) && f.langs.Contains(id) {
				candidates = append(candidates, f)
			}
		}
		if len(candidates) != 0 {
			if tf := m.matchCoveringTiered(candidates, style, r); tf != nil {
				return tf
			}
		}
	}
	return m.matchCoveringTiered(m.all, style, r)
}

// matchCoveringTiered is the cross-family fallback selection: default-family faces are preferred over other visible
// families, which are preferred over hidden (dot-prefixed) families — approximating the platform cascade lists, which
// never surface the hidden UI fonts fontscan's raw index includes — with the CSS3 style score ordering within each
// tier.
func (m *Manager) matchCoveringTiered(faces []*faceRec, style font.Style, r rune) *font.Typeface {
	var tier [3][]*faceRec
	for _, name := range defaultFamilies() {
		if fam := m.byKey[tsfont.NormalizeFamily(name)]; fam != nil {
			tier[0] = append(tier[0], fam.faces...)
		}
	}
	for _, f := range faces {
		if !f.runes.Contains(r) {
			continue
		}
		if strings.HasPrefix(f.key, ".") {
			tier[2] = append(tier[2], f)
		} else {
			tier[1] = append(tier[1], f)
		}
	}
	// The default families only participate when they are part of the candidate set (they may have been excluded by a
	// BCP-47 restriction or may not cover r; matchCovering re-checks coverage).
	if len(tier[0]) != 0 {
		allowed := make(map[*faceRec]bool, len(tier[1]))
		for _, f := range tier[1] {
			allowed[f] = true
		}
		kept := tier[0][:0]
		for _, f := range tier[0] {
			if allowed[f] {
				kept = append(kept, f)
			}
		}
		tier[0] = kept
	}
	for _, candidates := range tier {
		if tf := matchCovering(candidates, style, r, true); tf != nil {
			return tf
		}
	}
	return nil
}

// matchCovering returns the best css3-scored face among those covering r, nil when none cover (or none load —
// lower-scored covering faces are tried in rank order, as in StyleSet.MatchStyle). approx selects the footprint-derived
// style (no I/O — used for the cross-family fallback scans) over the exact table-read style (used within a single
// family).
//
// The footprint rune sets overcount: go-text's scanner counts cmap entries that map to glyph 0 (fontforge-built fonts
// commonly carry U+0000 and U+FFFF segments mapping to .notdef — ubuntu's DejaVuSans-ExtraLight does), and no host
// treats a .notdef mapping as coverage: FreeType's charcode iteration skips glyph-0 entries, so fontconfig charsets
// never contain them, and the CoreText/DirectWrite cmap lookups yield the missing glyph. The loaded face's own cmap is
// therefore the final word — a ranked candidate only answers when it really maps r.
func matchCovering(faces []*faceRec, pattern font.Style, r rune, approx bool) *font.Typeface {
	var covering []*faceRec
	for _, f := range faces {
		if f.runes.Contains(r) {
			covering = append(covering, f)
		}
	}
	styleAt := func(i int) font.Style {
		if approx {
			return covering[i].approx
		}
		return covering[i].style()
	}
	for _, idx := range rankStylesCSS3(pattern, len(covering), styleAt) {
		if tf := covering[idx].typeface(); tf != nil && tf.UnicharToGlyph(r) != 0 {
			return tf
		}
	}
	return nil
}

// MakeFromData loads a typeface from font data; nil when the data is not a usable font or the collection index is out
// of range.
func (m *Manager) MakeFromData(data []byte, index int) *font.Typeface {
	tf, err := font.NewTypefaceFromData(data, index)
	if err != nil {
		return nil
	}
	return tf
}
