// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package pdf

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stream"
)

// TestDocumentDrawPaintAndClear drives a full-page paint fill through the page canvas: a full-page Clear (an opaque src
// draw that the fast-path reduces to src-over) plus an explicit DrawPaint, then verifies the document serializes
// structurally and the drawn page carries more content than an empty one.
func TestDocumentDrawPaintAndClear(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, &Metadata{Title: "DrawPaint"})
	c := doc.BeginPageCanvas(120, 90)
	if c == nil {
		t.Fatal("BeginPageCanvas returned nil")
	}
	// Clear: opaque color, kSrc — checkFastPathIsSrcOver's solid lane folds it to kSrcOver.
	c.Clear(0xFF204080)
	// DrawPaint proper: fills the clip with the paint.
	p := canvas.NewPaint()
	p.Color = 0x80FF8040
	c.DrawPaint(p)
	// A src-blend translucent paint is NOT reducible to src-over and keeps its mode.
	p2 := canvas.NewPaint()
	p2.Color = 0x40104080
	p2.BlendMode = raster.BlendSrc
	c.DrawRect(geom.RectLTRB(10, 10, 60, 40), p2)
	doc.EndPage()
	doc.Close()

	data := out.Bytes()
	validatePDF(t, data)

	var emptyOut stream.MemoryWStream
	empty := NewDocument(&emptyOut, &Metadata{Title: "DrawPaint"})
	empty.BeginPageCanvas(120, 90)
	empty.EndPage()
	empty.Close()
	if len(data) <= len(emptyOut.Bytes()) {
		t.Error("draw-paint page is not larger than an empty page")
	}
	if !bytes.Contains(data, []byte("/Type /Catalog")) {
		t.Error("output missing the document catalog")
	}
}
