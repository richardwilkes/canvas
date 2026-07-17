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

package font

import (
	"bytes"
	"errors"
	"fmt"
	"os"

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
			// The WWS-aware precedence rules from typesetting's font.Describe: when OS/2 fsSelection bit 8 (WWS) is
			// set, the typographic names already delimit the family; otherwise prefer the explicit WWS names. This
			// keeps display names consistent with fontscan's index grouping.
			wws := os2 != nil && os2.FsSelection&0x100 != 0
			if wws {
				info.Family = firstName(name, 16, 1)
				info.StyleName = firstName(name, 17, 2)
			} else {
				info.Family = firstName(name, 21, 16, 1)
				info.StyleName = firstName(name, 22, 17, 2)
			}
		}
	}
	return info, nil
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
