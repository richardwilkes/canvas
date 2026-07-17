// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Style's underlying representation: weight/width/slant packed into a single int32, so callers can traffic in a single
// compact numeric value.

package font

// Slant identifies whether a font style is upright, italic, or oblique.
type Slant int32

// Slant values.
const (
	SlantUpright Slant = iota
	SlantItalic
	SlantOblique
)

// Weight constants.
const (
	WeightInvisible  = 0
	WeightThin       = 100
	WeightExtraLight = 200
	WeightLight      = 300
	WeightNormal     = 400
	WeightMedium     = 500
	WeightSemiBold   = 600
	WeightBold       = 700
	WeightExtraBold  = 800
	WeightBlack      = 900
	WeightExtraBlack = 1000
)

// Width constants.
const (
	WidthUltraCondensed = 1
	WidthExtraCondensed = 2
	WidthCondensed      = 3
	WidthSemiCondensed  = 4
	WidthNormal         = 5
	WidthSemiExpanded   = 6
	WidthExpanded       = 7
	WidthExtraExpanded  = 8
	WidthUltraExpanded  = 9
)

// Style packs a font's weight, width, and slant into a single value: weight + width<<16 + slant<<24, with each
// component pinned to its legal range at construction.
type Style int32

// NewStyle builds a Style from a weight, width, and slant, clamping each to its legal range.
func NewStyle(weight, width int, slant Slant) Style {
	return Style(pinInt(weight, WeightInvisible, WeightExtraBlack) +
		pinInt(width, WidthUltraCondensed, WidthUltraExpanded)<<16 +
		pinInt(int(slant), int(SlantUpright), int(SlantOblique))<<24)
}

// NormalStyle returns the normal-weight, normal-width, upright style.
func NormalStyle() Style { return NewStyle(WeightNormal, WidthNormal, SlantUpright) }

// BoldStyle returns the bold-weight, normal-width, upright style.
func BoldStyle() Style { return NewStyle(WeightBold, WidthNormal, SlantUpright) }

// ItalicStyle returns the normal-weight, normal-width, italic style.
func ItalicStyle() Style { return NewStyle(WeightNormal, WidthNormal, SlantItalic) }

// BoldItalicStyle returns the bold-weight, normal-width, italic style.
func BoldItalicStyle() Style { return NewStyle(WeightBold, WidthNormal, SlantItalic) }

// Weight returns the style's weight component.
func (s Style) Weight() int { return int(int32(s) & 0xFFFF) }

// Width returns the style's width component.
func (s Style) Width() int { return int((int32(s) >> 16) & 0xFF) }

// Slant returns the style's slant component.
func (s Style) Slant() Slant { return Slant((int32(s) >> 24) & 0xFF) }

func pinInt(v, lo, hi int) int {
	return max(lo, min(v, hi))
}
