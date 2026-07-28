// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Text scenarios: rotated and scaled text is the SDFT demand trigger, so the corpus carries scenes exercising
// the distance-field window (device text size in [162, glyphsAsPathsFontSize]) under rotation and scaling, plus a
// rotated direct-mask scene below the window. The scenarios draw with an embedded copy of Roboto-Regular (the same
// fixture the probe tests use) so every render rasterizes identical outlines through the library's own scaler, making
// text output deterministic per platform and gated in every lane against the per-platform golden sets.

package scenario

import (
	_ "embed"

	"github.com/richardwilkes/canvas/colorcore"
)

// FontData is the corpus text fixture: Roboto-Regular (Apache-2.0; see font/testdata/README.md), embedded so every
// consumer (CLI, gorender tests) resolves it independent of the working directory.
//
//go:embed testdata/Roboto-Regular.ttf
var FontData []byte

// The color-emoji corpus fixtures: the Skia-authored sbix and COLRv0/CPAL test fonts (BSD-3-Clause; see
// testdata/README.md), each mapping U+1F600 to a color glyph. Embedded for the same reason as FontData.
var (
	//go:embed testdata/sbix.ttf
	fontDataSbix []byte
	//go:embed testdata/colr.ttf
	fontDataCOLR []byte
)

// FontID selects a corpus font for Canvas.DrawSimpleTextFont.
type FontID int32

// FontID values.
const (
	FontRoboto FontID = iota
	FontSbix
	FontCOLR
)

// FontDataFor returns the embedded font bytes for a FontID.
func FontDataFor(f FontID) []byte {
	switch f {
	case FontSbix:
		return fontDataSbix
	case FontCOLR:
		return fontDataCOLR
	default:
		return FontData
	}
}

func init() {
	Register(Scenario{
		Name:   "text-sdf-rotated",
		Width:  256,
		Height: 256,
		Draw: func(c Canvas) {
			// 200px text is inside the default SDFT window (min 162; max >= 256 on every OS) and the 30-degree rotation
			// is the distance-field payoff case.
			c.Clear(colorcore.Color(0xFFFFFFFF))
			c.Save()
			c.Translate(128, 128)
			c.RotateRadians(30 * 3.14159265358979 / 180)
			c.Translate(-128, -128)
			c.DrawSimpleText("Sk", 32, 200, 200, Fill(colorcore.Color(0xFF102040)))
			c.Restore()
		},
	})
	Register(Scenario{
		Name:   "text-sdf-scaled",
		Width:  256,
		Height: 256,
		Draw: func(c Canvas) {
			// A 100px font under a 2x canvas scale: the approximate device text size (200px) selects the SDFT lane
			// through the view matrix rather than the font size.
			c.Clear(colorcore.Color(0xFFFFFFFF))
			c.Save()
			c.Scale(2, 2)
			c.DrawSimpleText("Go", 4, 110, 100, Fill(colorcore.Color(0xFF203010)))
			c.Restore()
		},
	})
	Register(Scenario{
		Name:   "text-direct-rotated",
		Width:  256,
		Height: 256,
		Draw: func(c Canvas) {
			// 32px rotated text stays below the SDFT window: the direct-mask lane with the rotation baked into the
			// strike, the regression gate that small text does not change lanes.
			c.Clear(colorcore.Color(0xFFFFFFFF))
			c.Save()
			c.Translate(24, 200)
			c.RotateRadians(-25 * 3.14159265358979 / 180)
			c.DrawSimpleText("rotated direct", 0, 0, 32, Fill(colorcore.Color(0xFF000000)))
			c.Restore()
		},
	})
}
