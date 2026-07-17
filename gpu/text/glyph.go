// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Glyph is the GPU-side glyph record pairing a packed glyph ID with its atlas locator. This package is home to the
// text-rendering GPU layer (the SubRun system and its StrikeCache). FormatFromGlyph maps the mask formats the scaler
// produces (A8 coverage — which also absorbs bilevel and 3D masks, SDF->A8 for distance-field text, ARGB32 color, and
// LCD16->A565 for subpixel text) to the atlas mask format.

package text

import (
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/gpu"
)

// Glyph is the GPU-side record for one glyph: its packed ID and where it lives in the atlas.
type Glyph struct {
	// PackedID is set at creation and never reassigned.
	PackedID     font.PackedGlyphID
	AtlasLocator gpu.AtlasLocator
}

// NewGlyph returns a new Glyph for the given packed glyph ID.
func NewGlyph(packedID font.PackedGlyphID) *Glyph { return &Glyph{PackedID: packedID} }

// FormatFromGlyph maps the scaler's mask format to the atlas mask format.
func FormatFromGlyph(format font.MaskFormat) gpu.MaskFormat {
	switch format {
	case font.MaskA8, font.MaskSDF:
		return gpu.MaskFormatA8
	case font.MaskARGB32:
		return gpu.MaskFormatARGB
	case font.MaskLCD16:
		return gpu.MaskFormatA565
	default:
		panic("unreachable mask format")
	}
}
