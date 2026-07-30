// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Font-wide metrics: ascent/descent, line spacing, and the underline/strikeout geometry a font exposes.

package font

// Metrics flags: which optional fields of Metrics were actually populated by the font.
const (
	MetricsFlagUnderlineThicknessIsValid uint32 = 1 << 0
	MetricsFlagUnderlinePositionIsValid  uint32 = 1 << 1
	MetricsFlagStrikeoutThicknessIsValid uint32 = 1 << 2
	MetricsFlagStrikeoutPositionIsValid  uint32 = 1 << 3
	// MetricsFlagBoundsInvalid is the inverse of its siblings: it is set when Top/Bottom/XMin/XMax hold no bounding box
	// the font reported (the empty typeface, or a font with no hhea table), so a consumer must treat those four as
	// unknown rather than as an empty box.
	MetricsFlagBoundsInvalid uint32 = 1 << 4
)

// Metrics holds font-wide layout metrics. All values are in the font's device space (scaled by the font size); y-down,
// so values above the baseline (Top, Ascent, XMin's partner YMax, ...) are typically negative.
type Metrics struct {
	Flags              uint32  // which metrics are valid
	Top                float32 // greatest extent above origin of any glyph bounding box, typically negative
	Ascent             float32 // distance to reserve above baseline, typically negative
	Descent            float32 // distance to reserve below baseline, typically positive
	Bottom             float32 // greatest extent below origin of any glyph bounding box, typically positive
	Leading            float32 // distance to add between lines, typically positive or zero
	AvgCharWidth       float32 // average character width, zero if unknown
	MaxCharWidth       float32 // maximum character width, zero if unknown
	XMin               float32 // greatest extent to left of origin of any glyph bounding box, typically negative
	XMax               float32 // greatest extent to right of origin of any glyph bounding box, typically positive
	XHeight            float32 // height of lower-case 'x', zero if unknown
	CapHeight          float32 // height of an upper-case letter, zero if unknown
	UnderlineThickness float32 // underline thickness
	UnderlinePosition  float32 // distance from baseline to top of stroke, typically positive
	StrikeoutThickness float32 // strikeout thickness
	StrikeoutPosition  float32 // distance from baseline to bottom of stroke, typically negative
}

// scale multiplies every scalar metric by s.
func (m *Metrics) scale(s float32) {
	m.Top *= s
	m.Ascent *= s
	m.Descent *= s
	m.Bottom *= s
	m.Leading *= s
	m.AvgCharWidth *= s
	m.MaxCharWidth *= s
	m.XMin *= s
	m.XMax *= s
	m.XHeight *= s
	m.CapHeight *= s
	m.UnderlineThickness *= s
	m.UnderlinePosition *= s
	m.StrikeoutThickness *= s
	m.StrikeoutPosition *= s
}
