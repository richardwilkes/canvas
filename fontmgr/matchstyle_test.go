// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A 36-set / 277-case corpus exercising matchStyleCSS3 (style names camel-cased). The invalidFontStyle sentinel marks
// cases expected to produce a null match (empty set).

package fontmgr

import (
	"testing"

	"github.com/richardwilkes/canvas/font"
)

var (
	invalidFontStyle = font.NewStyle(101, font.WidthNormal, font.SlantUpright)

	condensedNormal100 = font.NewStyle(font.WeightThin, font.WidthCondensed, font.SlantUpright)
	condensedNormal900 = font.NewStyle(font.WeightBlack, font.WidthCondensed, font.SlantUpright)
	condensedItalic100 = font.NewStyle(font.WeightThin, font.WidthCondensed, font.SlantItalic)
	condensedItalic900 = font.NewStyle(font.WeightBlack, font.WidthCondensed, font.SlantItalic)
	condensedObliqu100 = font.NewStyle(font.WeightThin, font.WidthCondensed, font.SlantOblique)
	condensedObliqu900 = font.NewStyle(font.WeightBlack, font.WidthCondensed, font.SlantOblique)
	expandedNormal100  = font.NewStyle(font.WeightThin, font.WidthExpanded, font.SlantUpright)
	expandedNormal900  = font.NewStyle(font.WeightBlack, font.WidthExpanded, font.SlantUpright)
	expandedItalic100  = font.NewStyle(font.WeightThin, font.WidthExpanded, font.SlantItalic)
	expandedItalic900  = font.NewStyle(font.WeightBlack, font.WidthExpanded, font.SlantItalic)
	expandedObliqu100  = font.NewStyle(font.WeightThin, font.WidthExpanded, font.SlantOblique)
	expandedObliqu900  = font.NewStyle(font.WeightBlack, font.WidthExpanded, font.SlantOblique)

	normalNormal100 = font.NewStyle(font.WeightThin, font.WidthNormal, font.SlantUpright)
	normalNormal300 = font.NewStyle(font.WeightLight, font.WidthNormal, font.SlantUpright)
	normalNormal400 = font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantUpright)
	normalNormal500 = font.NewStyle(font.WeightMedium, font.WidthNormal, font.SlantUpright)
	normalNormal600 = font.NewStyle(font.WeightSemiBold, font.WidthNormal, font.SlantUpright)
	normalNormal900 = font.NewStyle(font.WeightBlack, font.WidthNormal, font.SlantUpright)
)

// TestMatchStyleCSS3 exercises matchStyleCSS3 against the corpus above.
func TestMatchStyleCSS3(t *testing.T) {
	for ti, test := range matchStyleCSS3Tests {
		for ci, c := range test.cases {
			pattern, want := c[0], c[1]
			idx := matchStyleCSS3(pattern, len(test.set), func(i int) font.Style { return test.set[i] })
			if idx < 0 {
				if want != invalidFontStyle {
					t.Errorf("test %d case %d: no match, want (%d,%d,%d)", ti, ci,
						want.Weight(), want.Width(), want.Slant())
				}
				continue
			}
			if got := test.set[idx]; got != want {
				t.Errorf("test %d case %d: pattern (%d,%d,%d) matched (%d,%d,%d), want (%d,%d,%d)", ti, ci,
					pattern.Weight(), pattern.Width(), pattern.Slant(),
					got.Weight(), got.Width(), got.Slant(),
					want.Weight(), want.Width(), want.Slant())
			}
		}
	}
}

var matchStyleCSS3Tests = []struct {
	set   []font.Style
	cases [][2]font.Style
}{
	{
		set: []font.Style{normalNormal500, normalNormal400},
		cases: [][2]font.Style{
			{normalNormal400, normalNormal400},
			{normalNormal500, normalNormal500},
		},
	},
	{
		set: []font.Style{normalNormal500, normalNormal300},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal300},
			{normalNormal400, normalNormal500},
			{normalNormal500, normalNormal500},
		},
	},
	{
		set: []font.Style{
			condensedNormal100, condensedNormal900, condensedItalic100, condensedItalic900,
			expandedNormal100, expandedNormal900, expandedItalic100, expandedItalic900,
		},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedItalic100, expandedNormal100, expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{condensedNormal900, condensedItalic900, expandedNormal900, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedNormal900, expandedNormal100, expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{condensedNormal100, expandedNormal100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal100},
		},
	},
	{
		set: []font.Style{condensedNormal900, expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal900},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal900},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{condensedItalic100, condensedItalic900, expandedItalic100, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedItalic100, expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{condensedItalic900, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic900},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedItalic900},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedNormal900, condensedItalic100, condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal100},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic100},
		},
	},
	{
		set: []font.Style{condensedNormal900, condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedNormal900},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedItalic900},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedNormal100},
			{expandedItalic900, condensedNormal900},
		},
	},
	{
		set: []font.Style{condensedNormal100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal100},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal100},
			{expandedItalic100, condensedNormal100},
			{expandedItalic900, condensedNormal100},
		},
	},
	{
		set: []font.Style{condensedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal900},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, condensedNormal900},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedNormal900},
			{expandedItalic900, condensedNormal900},
		},
	},
	{
		set: []font.Style{condensedItalic100, condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedItalic100},
			{expandedNormal900, condensedItalic900},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{condensedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, condensedItalic100},
			{expandedNormal900, condensedItalic100},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic100},
		},
	},
	{
		set: []font.Style{condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic900},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedItalic900},
			{expandedNormal900, condensedItalic900},
			{expandedItalic100, condensedItalic900},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedNormal900, expandedItalic100, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{condensedObliqu100, expandedItalic100},
			{condensedObliqu900, expandedItalic900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
			{expandedObliqu100, expandedItalic100},
			{expandedObliqu900, expandedItalic900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal100},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{expandedNormal900, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal900},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedItalic900},
			{condensedItalic900, expandedItalic900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedNormal100},
			{condensedItalic900, expandedNormal900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{expandedNormal100},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal100},
			{condensedItalic100, expandedNormal100},
			{condensedItalic900, expandedNormal100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal100},
		},
	},
	{
		set: []font.Style{expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal900},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedNormal900},
			{condensedItalic900, expandedNormal900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal900},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{expandedItalic100, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedItalic100},
			{condensedNormal900, expandedItalic900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, expandedItalic100},
			{condensedNormal900, expandedItalic100},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic100},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedItalic900},
			{condensedNormal900, expandedItalic900},
			{condensedItalic100, expandedItalic900},
			{condensedItalic900, expandedItalic900},
			{expandedNormal100, expandedItalic900},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{normalNormal100, normalNormal900},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal100},
			{normalNormal400, normalNormal100},
			{normalNormal500, normalNormal100},
			{normalNormal600, normalNormal900},
		},
	},
	{
		set: []font.Style{normalNormal100, normalNormal400, normalNormal900},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal100},
			{normalNormal400, normalNormal400},
			{normalNormal500, normalNormal400},
			{normalNormal600, normalNormal900},
		},
	},
	{
		set: []font.Style{normalNormal100, normalNormal500, normalNormal900},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal100},
			{normalNormal400, normalNormal500},
			{normalNormal500, normalNormal500},
			{normalNormal600, normalNormal900},
		},
	},
	{
		set: []font.Style{},
		cases: [][2]font.Style{
			{normalNormal300, invalidFontStyle},
			{normalNormal400, invalidFontStyle},
			{normalNormal500, invalidFontStyle},
			{normalNormal600, invalidFontStyle},
		},
	},
	{
		set: []font.Style{
			expandedNormal100, expandedNormal900, expandedItalic100, expandedItalic900,
			expandedObliqu100, expandedObliqu900,
		},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{condensedObliqu100, expandedObliqu100},
			{condensedObliqu900, expandedObliqu900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
			{expandedObliqu100, expandedObliqu100},
			{expandedObliqu900, expandedObliqu900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedNormal900, expandedObliqu100, expandedObliqu900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedObliqu100},
			{condensedItalic900, expandedObliqu900},
			{condensedObliqu100, expandedObliqu100},
			{condensedObliqu900, expandedObliqu900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedObliqu100},
			{expandedItalic900, expandedObliqu900},
			{expandedObliqu100, expandedObliqu100},
			{expandedObliqu900, expandedObliqu900},
		},
	},
	{
		set: []font.Style{expandedItalic100, expandedItalic900, expandedObliqu100, expandedObliqu900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedObliqu100},
			{condensedNormal900, expandedObliqu900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{condensedObliqu100, expandedObliqu100},
			{condensedObliqu900, expandedObliqu900},
			{expandedNormal100, expandedObliqu100},
			{expandedNormal900, expandedObliqu900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
			{expandedObliqu100, expandedObliqu100},
			{expandedObliqu900, expandedObliqu900},
		},
	},
}
