// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Lightweight per-face description for the font manager: reads a face's family/style names and font style directly from
// the OS/2, head, and name tables, without instantiating the font. The family-name precedence follows
// go-text/typesetting's font.Describe (the WWS-aware rule fontscan uses to build its index families), so a display name
// always normalizes back to the fontscan footprint's family key; the style-name precedence is the same rule over the
// subfamily IDs (22 → 17 → 2). The font style reuses the OS/2 mapping the typeface derives at load time (computeStyle),
// so a style set's listed styles always equal the styles of the typefaces it creates.
//
// FaceCoversRune* is the same idea for one cmap question — does this face really map this character? — so the font
// manager's character fallback can reject a candidate without parsing (and caching) the whole font.

package font

import (
	"bytes"
	"errors"
	"fmt"
	"os"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/font/opentype/tables"
)

// FaceInfo is the lightweight description of one face in a font file: the display family name, the style (subfamily)
// name, and the resolved Style.
type FaceInfo struct {
	Family    string
	StyleName string
	Style     Style
}

// DescribeFaceFile reads the description of face index within the font file at path without a full font parse (only the
// OS/2, head, and name tables are touched).
func DescribeFaceFile(path string, index int) (FaceInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return FaceInfo{}, err
	}
	defer f.Close() //nolint:errcheck // read-only file; close errors are irrelevant
	return describeFace(f, index)
}

// DescribeFaceData is DescribeFaceFile over in-memory font data.
func DescribeFaceData(data []byte, index int) (FaceInfo, error) {
	return describeFace(bytes.NewReader(data), index)
}

func describeFace(src opentype.Resource, index int) (FaceInfo, error) {
	lds, err := opentype.NewLoaders(src)
	if err != nil {
		return FaceInfo{}, fmt.Errorf("font: parse: %w", err)
	}
	if index < 0 || index >= len(lds) {
		return FaceInfo{}, fmt.Errorf("font: collection index %d out of range (%d faces)", index, len(lds))
	}
	ld := lds[index]

	var head tables.Head
	raw, err := ld.RawTable(opentype.MustNewTag("head"))
	if err != nil {
		return FaceInfo{}, errors.New("font: missing head table")
	}
	if head, _, err = tables.ParseHead(raw); err != nil {
		return FaceInfo{}, fmt.Errorf("font: head: %w", err)
	}
	var os2 *tables.Os2
	if raw, err = ld.RawTable(opentype.MustNewTag("OS/2")); err == nil {
		if parsed, _, err2 := tables.ParseOs2(raw); err2 == nil {
			os2 = &parsed
		}
	}
	var info FaceInfo
	info.Style = computeStyle(os2, head)
	if raw, err = ld.RawTable(opentype.MustNewTag("name")); err == nil {
		if name, _, err2 := tables.ParseName(raw); err2 == nil {
			info.Family, info.StyleName = familyAndStyleNames(name, os2)
		}
	}
	return info, nil
}

// fsSelectionWWS is OS/2 fsSelection bit 8: the typographic family and subfamily names already delimit a WWS (weight,
// width, slope) family, so the explicit WWS names add nothing.
const fsSelectionWWS = 0x100

// familyAndStyleNames resolves a face's display family and style (subfamily) names from its name table, applying the
// WWS-aware precedence rules of typesetting's font.Describe (the same rule FreeType uses for face->family_name, which
// is what SkTypeface::getFamilyName reports): when OS/2 fsSelection bit 8 is set the typographic names already delimit
// the family, so the order is 16 → 1 and 17 → 2; otherwise the explicit WWS names win, giving 21 → 16 → 1 and
// 22 → 17 → 2. Typeface and FaceInfo share this one rule so a typeface's FamilyName always equals the family the font
// manager lists it under and keys MatchFamily by. tables.Name.Name applies the standard record-selection rules (favor
// Windows English, then Mac English/Roman, then Unicode).
func familyAndStyleNames(name tables.Name, os2 *tables.Os2) (family, styleName string) {
	if os2 != nil && os2.FsSelection&fsSelectionWWS != 0 {
		return firstName(name, 16, 1), firstName(name, 17, 2)
	}
	return firstName(name, 21, 16, 1), firstName(name, 22, 17, 2)
}

// FaceCoversRuneFile reports whether face index of the font file at path maps r through its own cmap to a real
// (nonzero) glyph. It answers exactly what NewTypefaceFromFile(path, index).UnicharToGlyph(r) != 0 answers, but touches
// only the 'cmap' and OS/2 tables and retains nothing: the font manager's character fallback verifies candidates with
// it, so rejecting a candidate costs a table read rather than a full font parse whose result is then cached for the life
// of the process. Anything that makes the answer unknowable — an unopenable or unparsable file, an out-of-range
// collection index, a missing or unusable cmap — reports false, matching the treatment of a face whose typeface fails to
// load.
func FaceCoversRuneFile(path string, index int, r rune) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close() //nolint:errcheck // read-only file; close errors are irrelevant
	return faceCoversRune(f, index, r)
}

// FaceCoversRuneData is FaceCoversRuneFile over in-memory font data.
func FaceCoversRuneData(data []byte, index int, r rune) bool {
	return faceCoversRune(bytes.NewReader(data), index, r)
}

// FaceRunesData reports every rune face index of data maps to a real (nonzero) glyph through its own cmap, in the
// cmap's iteration order. It is FaceCoversRuneData asked about every character at once, reading the same two tables and
// retaining nothing else: a font manager built over in-memory data uses it for the rune coverage a system scan would
// take from its index, and gets an exact set where the scanned footprints over-claim (they count cmap entries that map
// to glyph 0). Anything that makes the answer unknowable — unparsable data, an out-of-range collection index, a missing
// or unusable cmap — yields nil, matching FaceCoversRuneData's treatment. There is deliberately no file-backed twin:
// the file lane's coverage comes from the scan index.
func FaceRunesData(data []byte, index int) []rune {
	cmap, ok := faceCmap(bytes.NewReader(data), index)
	if !ok {
		return nil
	}
	var runes []rune
	iter := cmap.Iter()
	for iter.Next() {
		if r, gid := iter.Char(); gid != 0 && gid <= 0xFFFF {
			runes = append(runes, r)
		}
	}
	return runes
}

// faceCoversRune resolves r the way Typeface.UnicharToGlyph does, through the same cmap (see faceCmap): a glyph ID no
// sfnt glyph store can hold counts as unmapped.
func faceCoversRune(src opentype.Resource, index int, r rune) bool {
	if r < 0 || r > 0x10FFFF {
		return false
	}
	cmap, ok := faceCmap(src, index)
	if !ok {
		return false
	}
	gid, found := cmap.Lookup(r)
	return found && gid != 0 && gid <= 0xFFFF
}

// faceCmap returns the cmap face index of src resolves characters through, chosen the way typesetting's NewFont chooses
// it: the OS/2 font page selects the encoding and the chosen subtable is sanitized through ProcessCmap. ok is false
// when the face has no usable cmap.
func faceCmap(src opentype.Resource, index int) (cmap tsfont.Cmap, ok bool) {
	lds, err := opentype.NewLoaders(src)
	if err != nil || index < 0 || index >= len(lds) {
		return nil, false
	}
	ld := lds[index]
	// NewFont reads OS/2 for the font page with every error ignored; ParseOs2 fills nothing when it fails, so treating a
	// failed parse as a missing table (both leaving the zero table's FPNone) selects the same encoding it does.
	var fontPage tables.FontPage
	if raw, err2 := ld.RawTable(opentype.MustNewTag("OS/2")); err2 == nil {
		if os2, _, err3 := tables.ParseOs2(raw); err3 == nil {
			fontPage = os2.FontPage()
		}
	}
	raw, err := ld.RawTable(opentype.MustNewTag("cmap"))
	if err != nil {
		return nil, false
	}
	parsed, _, err := tables.ParseCmap(raw)
	if err != nil {
		return nil, false
	}
	best, _, err := tsfont.ProcessCmap(parsed, fontPage)
	if err != nil {
		return nil, false
	}
	return best, true
}

func firstName(name tables.Name, ids ...tables.NameID) string {
	for _, id := range ids {
		if s := name.Name(id); s != "" {
			return s
		}
	}
	return ""
}

// NewTypefaceFromFile parses face index of the font file at path by loading the file into memory and deferring to
// NewTypefaceFromData.
func NewTypefaceFromFile(path string, index int) (*Typeface, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return NewTypefaceFromData(data, index)
}
