// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Typeface and FaceInfo must name a face identically: the font manager lists a face under FaceInfo.Family and keys
// MatchFamily by the normalization of that name, so a typeface reporting a different FamilyName would be unreachable
// under the name it is listed with. Every font in testdata carries only name ID 1, which cannot tell the precedence
// rules apart, so these cases are synthesized.

package font

import (
	"testing"

	"github.com/go-text/typesetting/font/opentype/tables"
)

func TestFamilyAndStyleNamePrecedence(t *testing.T) {
	base := readTestFont(t, "Roboto-Regular.ttf")
	full := map[tables.NameID]string{
		1: "Legacy Family", 2: "Legacy Style",
		16: "Typographic Family", 17: "Typographic Style",
		21: "WWS Family", 22: "WWS Style",
	}
	for _, c := range []struct {
		names     map[tables.NameID]string
		desc      string
		family    string
		styleName string
		wws       bool
	}{
		{
			desc:  "the WWS names win when the WWS bit is clear",
			names: full, family: "WWS Family", styleName: "WWS Style",
		},
		{
			desc:  "the WWS bit means the typographic names already delimit the family",
			names: full, wws: true, family: "Typographic Family", styleName: "Typographic Style",
		},
		{
			desc: "the typographic names are next when there are no WWS names",
			names: map[tables.NameID]string{
				1: "Legacy Family", 2: "Legacy Style", 16: "Typographic Family", 17: "Typographic Style",
			},
			family: "Typographic Family", styleName: "Typographic Style",
		},
		{
			desc:   "the legacy names are the last resort",
			names:  map[tables.NameID]string{1: "Legacy Family", 2: "Legacy Style"},
			family: "Legacy Family", styleName: "Legacy Style",
		},
	} {
		data := sfntWithTables(t, base, map[string][]byte{
			"name": synthNameTable(c.names),
			"OS/2": os2WithFsSelectionBits(t, base, fsSelectionWWS, c.wws),
		})
		info, err := DescribeFaceData(data, 0)
		if err != nil {
			t.Fatalf("%s: DescribeFaceData: %v", c.desc, err)
		}
		if info.Family != c.family || info.StyleName != c.styleName {
			t.Errorf("%s: FaceInfo = (%q, %q), want (%q, %q)",
				c.desc, info.Family, info.StyleName, c.family, c.styleName)
		}
		tf, err := NewTypefaceFromData(data, 0)
		if err != nil {
			t.Fatalf("%s: NewTypefaceFromData: %v", c.desc, err)
		}
		if tf.FamilyName() != info.Family {
			t.Errorf("%s: Typeface.FamilyName() = %q, but the font manager lists the face as %q",
				c.desc, tf.FamilyName(), info.Family)
		}
	}
}
