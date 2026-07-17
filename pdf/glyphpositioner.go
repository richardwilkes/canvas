// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// glyphPositioner turns a sequence of positioned glyphs into the PDF text operators (Tm, Td, and hex-encoded Tj
// strings), coalescing glyphs that fall exactly at the expected advance into a single Tj run.

package pdf

import (
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stream"
)

// glyphPositioner accumulates positioned glyphs and emits the corresponding PDF text-showing operators.
type glyphPositioner struct {
	content                stream.WStream
	pdfFont                *pdfFont
	currentMatrixOrigin    geom.Point
	xAdvance               float32
	viewersAgreeOnAdvances bool
	viewersAgreeOnXAdvance bool
	textSkewX              float32
	inText                 bool
	initialized            bool
}

// newGlyphPositioner creates a glyphPositioner that writes to content, starting text at origin with the given
// synthetic-italic skew.
func newGlyphPositioner(content stream.WStream, textSkewX float32, origin geom.Point) *glyphPositioner {
	return &glyphPositioner{
		content:                content,
		currentMatrixOrigin:    origin,
		textSkewX:              textSkewX,
		viewersAgreeOnAdvances: true,
		viewersAgreeOnXAdvance: true,
	}
}

// flush closes an open Tj string.
func (g *glyphPositioner) flush() {
	if g.inText {
		writeText(g.content, "> Tj\n")
		g.inText = false
	}
}

// setFont switches the active font, flushing any open Tj run first.
func (g *glyphPositioner) setFont(f *pdfFont) {
	g.flush()
	g.pdfFont = f
	// Reader 2020.013.20064 incorrectly advances some Type3 fonts (crbug.com/1226960).
	convertedToType3 := f.fontType == font.FontTypeOther
	thousandEM := f.typeface.UnitsPerEm() == 1000
	g.viewersAgreeOnAdvances = thousandEM || !convertedToType3
}

// writeGlyph positions and appends one glyph, opening a new Tm/Td as needed and starting or continuing the current Tj
// run.
func (g *glyphPositioner) writeGlyph(glyph uint16, advanceWidth float32, xy geom.Point) {
	if !g.initialized {
		// Flip the text about the x-axis to account for the origin swap and include the passed parameters.
		writeText(g.content, "1 0 ")
		writeScalar(g.content, -g.textSkewX)
		writeText(g.content, " -1 ")
		writeScalar(g.content, g.currentMatrixOrigin.X)
		writeText(g.content, " ")
		writeScalar(g.content, g.currentMatrixOrigin.Y)
		writeText(g.content, " Tm\n")
		g.currentMatrixOrigin = geom.Point{}
		g.initialized = true
	}
	position := geom.Pt(xy.X-g.currentMatrixOrigin.X, xy.Y-g.currentMatrixOrigin.Y)
	if !g.viewersAgreeOnXAdvance || position != geom.Pt(g.xAdvance, 0) {
		g.flush()
		writeScalar(g.content, position.X-position.Y*g.textSkewX)
		writeText(g.content, " ")
		writeScalar(g.content, -position.Y)
		writeText(g.content, " Td ")
		g.currentMatrixOrigin = xy
		g.xAdvance = 0
		g.viewersAgreeOnXAdvance = true
	}
	g.xAdvance += advanceWidth
	if !g.viewersAgreeOnAdvances {
		g.viewersAgreeOnXAdvance = false
	}
	if !g.inText {
		writeText(g.content, "<")
		g.inText = true
	}
	if g.pdfFont.multiByteGlyphs() {
		writeUInt16BE(g.content, glyph)
	} else {
		writeUInt8(g.content, uint8(glyph))
	}
}
