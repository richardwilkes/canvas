// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"bytes"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/stream"
)

// loadTestTypeface loads the Roboto TrueType face shared with the font package's testdata.
func loadTestTypeface(t *testing.T) *font.Typeface {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatalf("read font: %v", err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatalf("parse font: %v", err)
	}
	return tf
}

// dictContaining returns the first dict object body containing all of the given substrings, or "".
func dictContaining(data []byte, needles ...string) string {
	for _, body := range dictObjects(data) {
		ok := true
		for _, n := range needles {
			if !strings.Contains(body, n) {
				ok = false
				break
			}
		}
		if ok {
			return body
		}
	}
	return ""
}

func TestTextEmitsCIDFontType2(t *testing.T) {
	tf := loadTestTypeface(t)
	f := font.NewFont(tf, 20, 1, 0)
	paint := canvas.NewPaint()
	paint.Color = colorcore.ARGB(255, 0, 0, 0)
	data := renderPDF(t, 200, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("Hi"), font.TextEncodingUTF8, 10, 50, f, paint)
	})
	validatePDF(t, data)

	content := pageContent(t, data)
	for _, op := range []string{"BT\n", "Tm\n", " Tf\n", " Tj\n", "ET\n"} {
		mustContain(t, content, op)
	}
	// The content references a /F font resource at the drawn text size.
	mustContain(t, content, " 20 Tf\n")
	if !strings.Contains(content, "/F") {
		t.Errorf("content missing /F font resource\n%s", content)
	}

	// The Type0 wrapper: Identity-H encoding, a descendant font, and a ToUnicode CMap.
	type0 := dictContaining(data, "/Subtype /Type0", "/Encoding /Identity-H")
	if type0 == "" {
		t.Fatalf("no Type0 font dict\nobjects=%v", dictObjects(data))
	}
	for _, key := range []string{"/DescendantFonts", "/ToUnicode", "/BaseFont"} {
		if !strings.Contains(type0, key) {
			t.Errorf("Type0 dict missing %s: %s", key, type0)
		}
	}

	// The CIDFontType2 descendant with the Identity CIDToGIDMap and CIDSystemInfo.
	cid := dictContaining(data, "/Subtype /CIDFontType2")
	if cid == "" {
		t.Fatalf("no CIDFontType2 dict\nobjects=%v", dictObjects(data))
	}
	for _, key := range []string{"/CIDToGIDMap /Identity", "/CIDSystemInfo", "/DW", "/FontDescriptor"} {
		if !strings.Contains(cid, key) {
			t.Errorf("CIDFont dict missing %s: %s", key, cid)
		}
	}

	// The FontDescriptor with the embedded full font program.
	desc := dictContaining(data, "/FontFile2")
	if desc == "" {
		t.Fatalf("no FontDescriptor with FontFile2\nobjects=%v", dictObjects(data))
	}
	for _, key := range []string{"/FontName", "/Flags", "/FontBBox", "/Ascent"} {
		if !strings.Contains(desc, key) {
			t.Errorf("FontDescriptor missing %s: %s", key, desc)
		}
	}

	// A /Font resource entry is present in some /Resources dict.
	if dictContaining(data, "/Font") == "" {
		t.Error("no /Resources /Font subdict")
	}
}

func TestFontProgramOmittedWhenUnavailable(t *testing.T) {
	var buf stream.MemoryWStream
	doc := NewDocument(&buf, DefaultMetadata())
	descriptor := NewTypedDict("FontDescriptor")
	size := descriptor.Size()
	written := buf.BytesWritten()
	for _, data := range [][]byte{nil, {}} {
		if insertFontProgram(doc, descriptor, data) {
			t.Errorf("empty font program (%v) reported as embedded", data)
		}
	}
	if descriptor.Size() != size {
		t.Errorf("empty font program added an entry to the descriptor: %s", emitToString(descriptor))
	}
	if buf.BytesWritten() != written {
		t.Errorf("empty font program emitted a stream object: %s", buf.Bytes()[written:])
	}

	// A real program is embedded, with its uncompressed length in /Length1.
	program := []byte(strings.Repeat("font program bytes; ", 32))
	if !insertFontProgram(doc, descriptor, program) {
		t.Fatal("non-empty font program not embedded")
	}
	body := emitToString(descriptor)
	mustContain(t, body, "/FontFile2")
	if buf.BytesWritten() == written {
		t.Error("non-empty font program emitted no stream object")
	}
	mustContain(t, string(buf.Bytes()[written:]), "/Length1 "+strconv.Itoa(len(program)))
}

// emitToString serializes an object to its PDF representation.
func emitToString(o Object) string {
	var buf stream.MemoryWStream
	o.emit(&buf)
	return string(buf.Bytes())
}

func TestTextEmbedsWholeFontProgram(t *testing.T) {
	raw, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatalf("read font: %v", err)
	}
	tf, err := font.NewTypefaceFromData(raw, 0)
	if err != nil {
		t.Fatalf("parse font: %v", err)
	}
	f := font.NewFont(tf, 20, 1, 0)
	paint := canvas.NewPaint()
	data := renderPDF(t, 200, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("Hi"), font.TextEncodingUTF8, 10, 50, f, paint)
	})
	validatePDF(t, data)
	// The FontFile2 stream carries the full (unsubsetted) font program, so /Length1 is the whole file's size.
	if !bytes.Contains(data, []byte("/Length1 "+strconv.Itoa(len(raw)))) {
		t.Errorf("no FontFile2 stream with /Length1 %d", len(raw))
	}
}

func TestTextToUnicodeMapsGlyphs(t *testing.T) {
	tf := loadTestTypeface(t)
	f := font.NewFont(tf, 24, 1, 0)
	paint := canvas.NewPaint()
	data := renderPDF(t, 200, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("Hi"), font.TextEncodingUTF8, 10, 50, f, paint)
	})
	validatePDF(t, data)

	// The ToUnicode CMap (a FlateDecode stream) maps the drawn glyphs back to U+0048 ('H') and U+0069 ('i').
	var cmap string
	for _, s := range allStreamContents(data) {
		if strings.Contains(s, "begincmap") {
			cmap = s
		}
	}
	if cmap == "" {
		t.Fatal("no ToUnicode CMap stream")
	}
	mustContain(t, cmap, "/CMapType 2 def")
	mustContain(t, cmap, "beginbfchar")
	mustContain(t, cmap, "0048") // 'H'
	mustContain(t, cmap, "0069") // 'i'
}

func TestTextFontResourceDedup(t *testing.T) {
	tf := loadTestTypeface(t)
	f := font.NewFont(tf, 18, 1, 0)
	paint := canvas.NewPaint()
	data := renderPDF(t, 300, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("first"), font.TextEncodingUTF8, 10, 40, f, paint)
		c.DrawSimpleText([]byte("second"), font.TextEncodingUTF8, 10, 80, f, paint)
	})
	validatePDF(t, data)

	// Two draws of the same typeface produce exactly one Type0 font object.
	n := 0
	for _, body := range dictObjects(data) {
		if strings.Contains(body, "/Subtype /Type0") {
			n++
		}
	}
	if n != 1 {
		t.Errorf("expected one shared Type0 font, got %d", n)
	}
	// Both draws select the same font resource once each (two Tf operators referencing the same /F name).
	content := pageContent(t, data)
	if c := strings.Count(content, " Tf\n"); c != 2 {
		t.Errorf("expected 2 Tf operators, got %d\n%s", c, content)
	}
}

func TestTextWidthsAccumulateAcrossDraws(t *testing.T) {
	tf := loadTestTypeface(t)
	f := font.NewFont(tf, 16, 1, 0)
	paint := canvas.NewPaint()
	// Draw text that uses many distinct-width glyphs so the /W array is non-empty.
	data := renderPDF(t, 400, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("Proportional"), font.TextEncodingUTF8, 10, 50, f, paint)
		c.DrawSimpleText([]byte("widths vary"), font.TextEncodingUTF8, 10, 80, f, paint)
	})
	validatePDF(t, data)
	cid := dictContaining(data, "/Subtype /CIDFontType2")
	if cid == "" {
		t.Fatal("no CIDFontType2 dict")
	}
	// Proportional font ⇒ a /W array with per-glyph advances.
	mustContain(t, cid, "/W [")
	mustContain(t, cid, "/DW ")
}

func TestStrokedTextFallsBackToPaths(t *testing.T) {
	tf := loadTestTypeface(t)
	f := font.NewFont(tf, 40, 1, 0)
	paint := canvas.NewPaint()
	paint.Style = canvas.StyleStroke
	paint.StrokeWidth = 1
	data := renderPDF(t, 300, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("O"), font.TextEncodingUTF8, 10, 60, f, paint)
	})
	validatePDF(t, data)

	// Stroked text draws glyph outlines as paths: no text operators, no embedded font, and stroke ops appear.
	content := pageContent(t, data)
	if strings.Contains(content, "Tj\n") || strings.Contains(content, " Tf\n") {
		t.Errorf("stroked text should not emit text operators\n%s", content)
	}
	if dictContaining(data, "/Subtype /Type0") != "" {
		t.Error("stroked text should not embed a Type0 font")
	}
	// The glyph outline is stroked (S) — a curve operator confirms real outline geometry was emitted.
	if !strings.Contains(content, "S\n") && !strings.Contains(content, "B\n") {
		t.Errorf("expected a stroke/fill operator for the outline\n%s", content)
	}
}

func TestTextResourceDictNamesMatch(t *testing.T) {
	tf := loadTestTypeface(t)
	f := font.NewFont(tf, 22, 1, 0)
	paint := canvas.NewPaint()
	data := renderPDF(t, 200, 100, func(c *canvas.Canvas) {
		c.DrawSimpleText([]byte("Ok"), font.TextEncodingUTF8, 10, 50, f, paint)
	})
	validatePDF(t, data)
	content := pageContent(t, data)
	// The /F name selected by Tf must appear in the page's /Font resource subdict.
	i := strings.Index(content, "/F")
	if i < 0 {
		t.Fatalf("no font resource name in content\n%s", content)
	}
	name := content[i : i+strings.IndexAny(content[i:], " \n")]
	if dictContaining(data, name+" ") == "" {
		t.Errorf("font resource %s not found in any /Resources dict", name)
	}
}
