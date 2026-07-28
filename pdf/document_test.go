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
	"compress/zlib"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stream"
)

// parsedPDF is a minimal structural view of a serialized PDF, used only for test validation.
type parsedPDF struct {
	objectOffsets map[int]int // object number -> byte offset of its "N 0 obj"
	trailer       string
	data          []byte
	xrefOffset    int
	xrefCount     int
}

var objHeaderRE = regexp.MustCompile(`(?m)^(\d+) 0 obj\n`)

// validatePDF parses and structurally validates a PDF byte stream: header, object headers, the xref table (offsets must
// point at the matching object header), and the trailer. It fails the test on any violation.
func validatePDF(t *testing.T, data []byte) *parsedPDF {
	t.Helper()
	const header = "%PDF-1.4\n%\xD3\xEB\xE9\xE1\n"
	if !bytes.HasPrefix(data, []byte(header)) {
		t.Fatalf("bad header: %q", data[:min(len(data), 20)])
	}
	if !bytes.HasSuffix(data, []byte("\n%%EOF\n")) {
		t.Fatal("missing end-of-file trailer")
	}

	p := &parsedPDF{data: data, objectOffsets: map[int]int{}}
	for _, m := range objHeaderRE.FindAllSubmatchIndex(data, -1) {
		num, err := strconv.Atoi(string(data[m[2]:m[3]]))
		if err != nil {
			t.Fatalf("bad object number %q: %v", data[m[2]:m[3]], err)
		}
		p.objectOffsets[num] = m[0]
	}

	// startxref <offset>\n%%EOF
	sx := bytes.LastIndex(data, []byte("startxref\n"))
	if sx < 0 {
		t.Fatal("missing startxref")
	}
	if _, err := fmt.Sscanf(string(data[sx:]), "startxref\n%d", &p.xrefOffset); err != nil {
		t.Fatalf("bad startxref: %v", err)
	}
	if p.xrefOffset < 0 || p.xrefOffset >= len(data) {
		t.Fatalf("startxref offset %d out of range", p.xrefOffset)
	}
	if !bytes.HasPrefix(data[p.xrefOffset:], []byte("xref\n0 ")) {
		t.Fatalf("startxref does not point at an xref table")
	}

	// Parse the xref table: "xref\n0 <count>\n" then <count> 20-byte entries.
	xref := data[p.xrefOffset:]
	var count int
	if _, err := fmt.Sscanf(string(xref), "xref\n0 %d\n", &count); err != nil {
		t.Fatalf("bad xref header: %v", err)
	}
	p.xrefCount = count
	lineStart := bytes.IndexByte(xref, '\n') + 1
	lineStart += bytes.IndexByte(xref[lineStart:], '\n') + 1 // skip "0 <count>" line
	entries := xref[lineStart:]
	// Entry 0 is the free head.
	if !bytes.HasPrefix(entries, []byte("0000000000 65535 f \n")) {
		t.Fatalf("bad free-list head entry: %q", entries[:min(20, len(entries))])
	}
	for i := 1; i < count; i++ {
		entry := entries[i*20 : i*20+20]
		var offset int
		if _, err := fmt.Sscanf(string(entry), "%d 00000 n", &offset); err != nil {
			t.Fatalf("bad xref entry %q: %v", entry, err)
		}
		wantOffset, ok := p.objectOffsets[i]
		if !ok {
			t.Fatalf("xref references object %d that has no header", i)
		}
		if offset != wantOffset {
			t.Errorf("object %d: xref offset %d != actual header offset %d", i, offset, wantOffset)
		}
	}
	if len(p.objectOffsets) != count-1 {
		t.Errorf("object count mismatch: %d headers vs xref count %d (minus free)", len(p.objectOffsets), count-1)
	}

	// Trailer.
	ti := bytes.LastIndex(data, []byte("trailer\n"))
	if ti < 0 {
		t.Fatal("missing trailer")
	}
	p.trailer = string(data[ti+len("trailer\n") : sx])
	if !strings.Contains(p.trailer, fmt.Sprintf("/Size %d", count)) {
		t.Errorf("trailer /Size mismatch; trailer=%q count=%d", p.trailer, count)
	}
	for _, key := range []string{"/Root", "/Info"} {
		if !strings.Contains(p.trailer, key) {
			t.Errorf("trailer missing %s: %q", key, p.trailer)
		}
	}
	return p
}

// object returns the body of object num (between "obj\n" and "\nendobj").
func (p *parsedPDF) object(num int) string {
	off, ok := p.objectOffsets[num]
	if !ok {
		return ""
	}
	rest := p.data[off:]
	start := bytes.Index(rest, []byte(" obj\n"))
	end := bytes.Index(rest, []byte("\nendobj\n"))
	if start < 0 || end < 0 {
		return ""
	}
	return string(rest[start+len(" obj\n") : end])
}

func TestDocumentNoPages(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	doc.Close()
	if out.BytesWritten() != 0 {
		t.Errorf("a document with no pages should write nothing, wrote %d bytes", out.BytesWritten())
	}
	// Close is idempotent.
	doc.Close()
	if out.BytesWritten() != 0 {
		t.Error("second Close wrote output")
	}
}

func TestDocumentSinglePage(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	page := doc.BeginPage(612, 792)
	writeText(page.Content(), "q Q\n") // trivial content, too short to compress
	doc.EndPage()
	doc.Close()

	p := validatePDF(t, out.Bytes())

	// The info dict is object 1 (emitted first, on the first BeginPage).
	if info := p.object(1); !strings.Contains(info, "/Producer (Skia/PDF m142)") {
		t.Errorf("object 1 is not the info dict: %q", info)
	}
	// Some page object must carry the expected MediaBox and be of Type Page.
	foundPage := false
	for num := range p.objectOffsets {
		body := p.object(num)
		if strings.Contains(body, "/Type /Page\n") && strings.Contains(body, "/MediaBox [0 0 612 792]") {
			foundPage = true
		}
	}
	if !foundPage {
		t.Error("no page object with MediaBox [0 0 612 792]")
	}
	// The catalog must reference the page tree.
	foundCatalog := false
	for num := range p.objectOffsets {
		if strings.Contains(p.object(num), "/Type /Catalog") {
			foundCatalog = true
		}
	}
	if !foundCatalog {
		t.Error("no /Catalog object")
	}
}

func TestDocumentCompressedContent(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	page := doc.BeginPage(100, 100)
	content := strings.Repeat("0 0 100 100 re f\n", 50) // compressible and well over the savings threshold
	writeText(page.Content(), content)
	doc.EndPage()
	doc.Close()

	p := validatePDF(t, out.Bytes())

	// Find the FlateDecode content stream and decompress it back to the original bytes.
	found := false
	for num := range p.objectOffsets {
		body := p.object(num)
		if !strings.Contains(body, "/Filter /FlateDecode") {
			continue
		}
		found = true
		si := strings.Index(body, " stream\n")
		ei := strings.LastIndex(body, "\nendstream")
		if si < 0 || ei < 0 {
			t.Fatalf("malformed stream object: %q", body)
		}
		compressed := body[si+len(" stream\n") : ei]
		r, err := zlib.NewReader(bytes.NewReader([]byte(compressed)))
		if err != nil {
			t.Fatalf("zlib.NewReader: %v", err)
		}
		decoded, err := io.ReadAll(r)
		if err != nil {
			t.Fatalf("inflate content: %v", err)
		}
		if string(decoded) != content {
			t.Error("decompressed content does not match original")
		}
		// The /Length must equal the compressed byte count.
		if !strings.Contains(body, fmt.Sprintf("/Length %d", len(compressed))) {
			t.Errorf("stream /Length does not match compressed size %d: %q", len(compressed), body)
		}
	}
	if !found {
		t.Error("expected a FlateDecode content stream")
	}
}

func TestDocumentMultiPageTree(t *testing.T) {
	// 20 pages exercises the >8 branching factor of generate_page_tree (multiple internal /Pages nodes).
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	const nPages = 20
	for i := 0; i < nPages; i++ {
		doc.BeginPage(200, 300)
		doc.EndPage()
	}
	doc.Close()

	p := validatePDF(t, out.Bytes())

	// Count leaf pages and internal Pages nodes; the total descendant Count on internal nodes must be consistent with a
	// tree over nPages leaves.
	leafPages, internalNodes, rootCountSum := 0, 0, 0
	for num := range p.objectOffsets {
		body := p.object(num)
		switch {
		case strings.Contains(body, "/Type /Page\n"):
			leafPages++
		case strings.Contains(body, "/Type /Pages"):
			internalNodes++
			// Root nodes have no /Parent; sum their Counts to reach nPages.
			if !strings.Contains(body, "/Parent") {
				idx := strings.Index(body, "/Count ")
				if idx < 0 {
					t.Fatalf("object %d: root Pages node has no /Count", num)
				}
				var c int
				if _, err := fmt.Sscanf(body[idx:], "/Count %d", &c); err != nil {
					t.Fatalf("bad /Count in object %d: %v", num, err)
				}
				rootCountSum += c
			}
		}
	}
	if leafPages != nPages {
		t.Errorf("leaf page count = %d, want %d", leafPages, nPages)
	}
	if internalNodes < 2 {
		t.Errorf("expected multiple internal Pages nodes for %d pages, got %d", nPages, internalNodes)
	}
	if rootCountSum != nPages {
		t.Errorf("root /Count total = %d, want %d", rootCountSum, nPages)
	}
}

func TestDocumentAbort(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	doc.BeginPage(100, 100)
	doc.EndPage()
	doc.Abort()
	doc.Close() // must be a no-op after Abort

	data := out.Bytes()
	// Abort writes the already-emitted header/info/page bytes but no xref/trailer.
	if bytes.Contains(data, []byte("trailer")) {
		t.Error("aborted document must not contain a trailer")
	}
	if bytes.Contains(data, []byte("%%EOF")) {
		t.Error("aborted document must not be finalized with an end-of-file marker")
	}
}

func TestDocumentMetadataInTrailer(t *testing.T) {
	var out stream.MemoryWStream
	md := DefaultMetadata()
	md.Title = "Report"
	md.Lang = "en-US"
	doc := NewDocument(&out, md)
	doc.BeginPage(50, 50)
	doc.EndPage()
	doc.Close()

	p := validatePDF(t, out.Bytes())
	// Title set -> ViewerPreferences with DisplayDocTitle, and a /Lang in the catalog.
	foundViewerPrefs, foundLang := false, false
	for num := range p.objectOffsets {
		body := p.object(num)
		if strings.Contains(body, "/Type /Catalog") {
			foundViewerPrefs = strings.Contains(body, "/ViewerPreferences") &&
				strings.Contains(body, "/DisplayDocTitle true")
			foundLang = strings.Contains(body, "/Lang (en-US)")
		}
	}
	if !foundViewerPrefs {
		t.Error("catalog missing ViewerPreferences/DisplayDocTitle")
	}
	if !foundLang {
		t.Error("catalog missing /Lang")
	}
}

// TestDocumentCloseFinishesOpenPage closes a document whose only page was begun but never ended. The page's object
// number was reserved by BeginPage, so Close must finish the page rather than leave a dangling in-use xref entry.
func TestDocumentCloseFinishesOpenPage(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	page := doc.BeginPage(612, 792)
	writeText(page.Content(), "q Q\n")
	doc.Close() // no EndPage

	p := validatePDF(t, out.Bytes()) // catches the offset-0 xref entry and the /Size overcount
	foundPage := false
	for num := range p.objectOffsets {
		body := p.object(num)
		if strings.Contains(body, "/Type /Page\n") && strings.Contains(body, "/MediaBox [0 0 612 792]") {
			foundPage = true
		}
	}
	if !foundPage {
		t.Error("the unfinished page was dropped instead of being emitted")
	}
	if !bytes.Contains(out.Bytes(), []byte("q Q\n")) {
		t.Error("the unfinished page's content stream was dropped")
	}
	// The finished page must be indistinguishable from one closed the normal way.
	var want stream.MemoryWStream
	doc2 := NewDocument(&want, DefaultMetadata())
	writeText(doc2.BeginPage(612, 792).Content(), "q Q\n")
	doc2.EndPage()
	doc2.Close()
	if !bytes.Equal(out.Bytes(), want.Bytes()) {
		t.Error("closing with an open page differs from an explicit EndPage")
	}
}

// TestDocumentCloseFinishesOpenPageAfterCompletedPage covers the mixed case: a completed page followed by an open one.
// The trailing page must be appended to the page tree, not dropped.
func TestDocumentCloseFinishesOpenPageAfterCompletedPage(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	doc.BeginPage(100, 100)
	doc.EndPage()
	doc.BeginPage(200, 200)
	doc.Close() // no EndPage for the second page

	p := validatePDF(t, out.Bytes())
	leaves := 0
	for num := range p.objectOffsets {
		if strings.Contains(p.object(num), "/Type /Page\n") {
			leaves++
		}
	}
	if leaves != 2 {
		t.Errorf("leaf page count = %d, want 2", leaves)
	}
}

// TestDocumentCloseFinishesOpenCanvasPage repeats the check for a device-backed page, whose content and resources come
// from the recorded draws rather than the caller-filled stream.
func TestDocumentCloseFinishesOpenCanvasPage(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	c := doc.BeginPageCanvas(120, 90)
	c.Clear(0xFF204080)
	doc.Close() // no EndPage

	p := validatePDF(t, out.Bytes())
	foundPage := false
	for num := range p.objectOffsets {
		if strings.Contains(p.object(num), "/Type /Page\n") {
			foundPage = true
		}
	}
	if !foundPage {
		t.Error("the unfinished canvas page was dropped instead of being emitted")
	}
}

// TestDocumentAbortDiscardsOpenPage confirms Close after Abort still writes nothing, even with a page left open: Abort
// drops the in-progress page rather than deferring it to Close.
func TestDocumentAbortDiscardsOpenPage(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	doc.BeginPage(100, 100)
	doc.Abort()
	doc.Close()

	data := out.Bytes()
	if bytes.Contains(data, []byte("trailer")) || bytes.Contains(data, []byte("%%EOF")) {
		t.Error("aborted document was finalized by Close")
	}
}

// TestDocumentWritesNothingAfterClose pins the closed-document contract for every writer, not just Close: once the
// xref/trailer/%%EOF are out, nothing may be appended. An object emitted then would be unreachable trailing bytes,
// since the xref table naming it is already written and the follow-up Close returns without writing a new one.
func TestDocumentWritesNothingAfterClose(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	writeText(doc.BeginPage(100, 100).Content(), "q Q\n")
	doc.EndPage()
	doc.Close()
	finished := append([]byte(nil), out.Bytes()...)

	if p := doc.BeginPage(200, 200); p != nil {
		t.Error("BeginPage after Close returned a page")
	}
	if c := doc.BeginPageCanvas(200, 200); c != nil {
		t.Error("BeginPageCanvas after Close returned a canvas")
	}
	if ref := doc.Emit(NewDict()); ref.IsValid() {
		t.Error("Emit after Close returned a valid reference")
	}
	if ref := doc.EmitAt(NewDict(), doc.ReserveRef()); ref.IsValid() {
		t.Error("EmitAt after Close returned a valid reference")
	}
	if ref := doc.StreamOut(nil, []byte("past the end of the file"), false); ref.IsValid() {
		t.Error("StreamOut after Close returned a valid reference")
	}
	doc.EndPage()
	doc.Close()

	if data := out.Bytes(); !bytes.Equal(data, finished) {
		t.Errorf("%d bytes were appended after %%%%EOF: %q", len(data)-len(finished), data[len(finished):])
	}
	validatePDF(t, out.Bytes())
	if !bytes.HasSuffix(out.Bytes(), []byte("%%EOF\n")) {
		t.Error("the closed document no longer ends with its end-of-file marker")
	}
}

// TestDocumentWritesNothingAfterAbort is the Abort half of the same contract. Abort drops the page list, so a BeginPage
// afterwards would take the first-page path again and write a second %PDF header (resetting the offset map's base) into
// a document whose doc comment promises nothing further is written.
func TestDocumentWritesNothingAfterAbort(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	writeText(doc.BeginPage(100, 100).Content(), "q Q\n")
	doc.EndPage()
	doc.Abort()
	aborted := append([]byte(nil), out.Bytes()...)

	if p := doc.BeginPage(50, 50); p != nil {
		t.Error("BeginPage after Abort returned a page")
	}
	if c := doc.BeginPageCanvas(50, 50); c != nil {
		t.Error("BeginPageCanvas after Abort returned a canvas")
	}
	if ref := doc.Emit(NewDict()); ref.IsValid() {
		t.Error("Emit after Abort returned a valid reference")
	}
	if ref := doc.StreamOut(nil, []byte("past the end of the file"), false); ref.IsValid() {
		t.Error("StreamOut after Abort returned a valid reference")
	}
	doc.EndPage()
	doc.Close()

	data := out.Bytes()
	if !bytes.Equal(data, aborted) {
		t.Errorf("%d bytes were written after Abort: %q", len(data)-len(aborted), data[len(aborted):])
	}
	if n := bytes.Count(data, []byte("%PDF-1.4")); n != 1 {
		t.Errorf("file header count = %d, want 1", n)
	}
}

// contentRectInUserSpace composes every "cm" operator preceding the first "re" in a content stream and returns the rect
// that "re" covers in PDF user space — the same 72dpi space the page's /MediaBox is expressed in.
func contentRectInUserSpace(t *testing.T, content string) geom.Rect {
	t.Helper()
	parse := func(s string) float32 {
		v, err := strconv.ParseFloat(s, 32)
		if err != nil {
			t.Fatalf("bad numeric operand %q: %v", s, err)
		}
		return float32(v)
	}
	ctm := geom.IdentityMatrix()
	for _, line := range strings.Split(content, "\n") {
		fields := strings.Fields(line)
		switch {
		case len(fields) == 7 && fields[6] == "cm":
			// PDF's [a b c d e f] is (scaleX, skewY, skewX, scaleY, transX, transY).
			var cm geom.Matrix
			cm.SetAll(parse(fields[0]), parse(fields[2]), parse(fields[4]),
				parse(fields[1]), parse(fields[3]), parse(fields[5]), 0, 0, 1)
			ctm.PreConcat(&cm)
		case len(fields) == 5 && fields[4] == "re":
			x, y := parse(fields[0]), parse(fields[1])
			r, _ := ctm.MapRect(geom.Rect{Left: x, Top: y, Right: x + parse(fields[2]), Bottom: y + parse(fields[3])})
			return r.Sorted()
		}
	}
	t.Fatalf("no %q operator in content:\n%s", "re", content)
	return geom.Rect{}
}

// TestDocumentRasterDPIScalesPageCanvas pins that a device-backed page's canvas takes coordinates in the page's nominal
// 72dpi point space whatever the RasterDPI: content drawn at the page's point size must cover the whole MediaBox, not
// 72/RasterDPI of it. The page device works at the rasterized scale (so image-filter layer devices get the extra
// resolution), and the initial transform prepended to the content stream divides that scale back out.
func TestDocumentRasterDPIScalesPageCanvas(t *testing.T) {
	const w, h = 612, 792
	for _, dpi := range []float32{72, 144, 288, 300} {
		t.Run(fmt.Sprintf("dpi%g", dpi), func(t *testing.T) {
			md := DefaultMetadata()
			md.RasterDPI = dpi
			var out stream.MemoryWStream
			doc := NewDocument(&out, md)
			c := doc.BeginPageCanvas(w, h)
			paint := canvas.NewPaint()
			paint.Color = colorcore.ARGB(255, 200, 30, 40)
			c.DrawRect(geom.RectLTRB(0, 0, w, h), paint)
			doc.EndPage()
			doc.Close()

			data := out.Bytes()
			validatePDF(t, data)
			// The MediaBox is always the nominal point size; only the device pixel grid behind it changes.
			mustContain(t, dictContaining(data, "/MediaBox"), "/MediaBox [0 0 612 792]")
			got := contentRectInUserSpace(t, pageContent(t, data))
			want := geom.RectLTRB(0, 0, w, h)
			const tol = 0.02 // 300dpi's 72/300 scale is not exact in float32
			if abs32(got.Left-want.Left) > tol || abs32(got.Top-want.Top) > tol ||
				abs32(got.Right-want.Right) > tol || abs32(got.Bottom-want.Bottom) > tol {
				t.Errorf("full-page rect covers %v in user space, want the MediaBox %v", got, want)
			}
		})
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// TestDocumentEmitAtRejectsInvalidRef covers EmitAt with a reference that points at no object. Object numbers are
// 1-based, so a non-positive one used to index offsets[-1] and panic; it must be rejected instead, leaving the stream
// untouched and the document still serializable.
func TestDocumentEmitAtRejectsInvalidRef(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	page := doc.BeginPage(100, 100) // writes the header and the info dict
	writeText(page.Content(), "q Q\n")
	before := out.BytesWritten()
	for _, ref := range []IndirectReference{{}, {value: -1}, {value: -1000}} {
		got := doc.EmitAt(NewDict(), ref)
		if got.IsValid() {
			t.Errorf("EmitAt(%d) returned valid reference %d", ref.value, got.value)
		}
	}
	if out.BytesWritten() != before {
		t.Errorf("EmitAt with an invalid reference wrote %d bytes", out.BytesWritten()-before)
	}
	doc.EndPage()
	doc.Close()
	validatePDF(t, out.Bytes())
}

// TestDocumentBeginPageFinishesOpenPage covers a BeginPage with no matching EndPage for the previous page. The open
// page's object number was already reserved, so it must be finished rather than dropped — otherwise its content
// disappears and the xref table gains an in-use entry pointing at byte 0.
func TestDocumentBeginPageFinishesOpenPage(t *testing.T) {
	var out stream.MemoryWStream
	doc := NewDocument(&out, DefaultMetadata())
	writeText(doc.BeginPage(100, 100).Content(), "q Q\n")
	writeText(doc.BeginPage(200, 200).Content(), "n\n") // no EndPage for the first page
	doc.EndPage()
	doc.Close()

	p := validatePDF(t, out.Bytes()) // catches the offset-0 xref entry and the /Size overcount
	leaves := 0
	for num := range p.objectOffsets {
		if strings.Contains(p.object(num), "/Type /Page\n") {
			leaves++
		}
	}
	if leaves != 2 {
		t.Errorf("leaf page count = %d, want 2", leaves)
	}
	for _, box := range []string{"/MediaBox [0 0 100 100]", "/MediaBox [0 0 200 200]"} {
		if dictContaining(out.Bytes(), box) == "" {
			t.Errorf("no page with %s", box)
		}
	}
	if !bytes.Contains(out.Bytes(), []byte("q Q\n")) {
		t.Error("the unfinished page's content stream was dropped")
	}
	// The result must be indistinguishable from the same two pages closed the normal way.
	var want stream.MemoryWStream
	doc2 := NewDocument(&want, DefaultMetadata())
	writeText(doc2.BeginPage(100, 100).Content(), "q Q\n")
	doc2.EndPage()
	writeText(doc2.BeginPage(200, 200).Content(), "n\n")
	doc2.EndPage()
	doc2.Close()
	if !bytes.Equal(out.Bytes(), want.Bytes()) {
		t.Error("an implicit EndPage from BeginPage differs from an explicit one")
	}
}

func TestDocumentRasterDPIClamp(t *testing.T) {
	md := DefaultMetadata()
	md.RasterDPI = 0 // clamped to 72
	var out stream.MemoryWStream
	doc := NewDocument(&out, md)
	if doc.Metadata().RasterDPI != 72 {
		t.Errorf("RasterDPI not clamped: %g", doc.Metadata().RasterDPI)
	}
	md.RasterDPI = -5
	doc2 := NewDocument(&stream.MemoryWStream{}, md)
	if doc2.Metadata().RasterDPI != 72 {
		t.Errorf("negative RasterDPI not clamped: %g", doc2.Metadata().RasterDPI)
	}
	// EncodingQuality clamp.
	md.EncodingQuality = -3
	doc3 := NewDocument(&stream.MemoryWStream{}, md)
	if doc3.Metadata().EncodingQuality != 0 {
		t.Errorf("negative EncodingQuality not clamped: %d", doc3.Metadata().EncodingQuality)
	}
}
