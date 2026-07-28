// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The PDF document serializer: the indirect-object catalog and cross-reference (xref) machinery, stream emission with
// optional FlateDecode compression, the file header/trailer, the page tree, and the document lifecycle
// (BeginPage/EndPage/Close/Abort). Stream serialization is single-threaded; all jobs run inline.
//
// Pages carry either a caller-supplied content stream and resource dict (Page.Content / Page.Resources) or, via
// BeginPageCanvas, a content stream the PDF Device generates from canvas draw ops.

package pdf

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stream"
)

// ---- OffsetMap ---------------------------------------------------------------------------------------

// offsetMap tracks the byte offset of each indirect object so the xref table can be written at close.
type offsetMap struct {
	offsets    []int
	baseOffset int64
	hasBase    bool
}

// markStartOfDocument records the stream position of the "%PDF" header.
func (m *offsetMap) markStartOfDocument(s stream.WStream) {
	m.baseOffset = s.BytesWritten()
	m.hasBase = true
}

// markStartOfObject records object referenceNumber's offset relative to the header. Object numbers are 1-based; callers
// screen out non-positive numbers (Document.EmitAt rejects an invalid IndirectReference) before reaching here.
func (m *offsetMap) markStartOfObject(referenceNumber int, s stream.WStream) {
	index := referenceNumber - 1
	if index >= len(m.offsets) {
		grown := make([]int, index+1)
		copy(grown, m.offsets)
		m.offsets = grown
	}
	m.offsets[index] = int(s.BytesWritten() - m.baseOffset)
}

// objectCount returns the number of indirect objects, including the special zeroth free object.
func (m *offsetMap) objectCount() int { return len(m.offsets) + 1 }

// emitCrossReferenceTable writes the xref table and returns its file offset.
func (m *offsetMap) emitCrossReferenceTable(s stream.WStream) int {
	xRefFileOffset := int(s.BytesWritten() - m.baseOffset)
	writeText(s, "xref\n0 ")
	writeDecAsText(s, m.objectCount())
	writeText(s, "\n0000000000 65535 f \n")
	for _, offset := range m.offsets {
		writeBigDecAsText(s, int64(offset), 10)
		writeText(s, " 00000 n \n")
	}
	return xRefFileOffset
}

// skpdfMagic is the four-byte binary marker recommended after the %PDF header to signal to naive tools that the file
// contains binary data; each byte has its high bit set. These particular bytes match SkPDF's, so the output stays
// byte-for-byte compatible with it; document_test.go pins them.
const skpdfMagic = "\xD3\xEB\xE9\xE1"

func serializeHeader(m *offsetMap, s stream.WStream) {
	m.markStartOfDocument(s)
	writeText(s, "%PDF-1.4\n%"+skpdfMagic+"\n")
}

func beginIndirectObject(m *offsetMap, ref IndirectReference, s stream.WStream) {
	m.markStartOfObject(int(ref.value), s)
	writeDecAsText(s, int(ref.value))
	writeText(s, " 0 obj\n") // Generation number is always 0.
}

func endIndirectObject(s stream.WStream) { writeText(s, "\nendobj\n") }

func serializeFooter(m *offsetMap, s stream.WStream, infoDict, docCatalog IndirectReference, uuid UUID) {
	xRefFileOffset := m.emitCrossReferenceTable(s)
	trailerDict := NewDict()
	trailerDict.InsertInt("Size", int32(m.objectCount()))
	trailerDict.InsertRef("Root", docCatalog)
	trailerDict.InsertRef("Info", infoDict)
	if !uuid.isZero() {
		trailerDict.InsertObject("ID", MakePdfID(uuid, uuid))
	}
	writeText(s, "trailer\n")
	trailerDict.emit(s)
	writeText(s, "\nstartxref\n")
	writeBigDecAsText(s, int64(xRefFileOffset), 0)
	writeText(s, "\n%%EOF\n")
}

// ---- Document -----------------------------------------------------------------------------------------

// Document writes a PDF to a stream as pages are added. Not safe for concurrent use.
type Document struct {
	stream stream.WStream
	// gradientPatternMap caches gradient pattern objects: identical gradient shaders share one pattern object. The key
	// is the serialization of the gradient shader's parameters (gradientKey).
	gradientPatternMap map[string]IndirectReference
	fonts              map[uint32]*pdfFont
	toUnicodeMaps      map[uint32][]int32
	// Font canonicalization: the advanced metrics, reverse cmap, and CIDFontType2 resource are each one per typeface
	// per document, keyed by typeface unique ID. Fonts accumulate glyph usage across every draw and are emitted at
	// Close.
	typefaceMetrics map[uint32]*font.AdvancedMetrics
	strokeGSMap     map[strokeGSKey]IndirectReference
	current         *Page
	// pdfBitmapMap caches serialized images: a keyed image is serialized as an Image XObject at most once; repeated
	// draws of the same (sub)image reuse the cached indirect reference.
	pdfBitmapMap map[bitmapKey]IndirectReference
	// Graphic-state canonicalization maps: identical paints share one /ExtGState indirect object.
	fillGSMap map[fillGSKey]IndirectReference
	// imageShaderMap caches tiling pattern objects: identical image shaders share one tiling pattern object. The key is
	// the serialization of the image shader's parameters (imageShaderKeyString).
	imageShaderMap map[string]IndirectReference
	pages          []*Dict
	pageRefs       []IndirectReference
	offsetMap      offsetMap
	metadata       Metadata
	// invertFunction is the shared "{1 exch sub}" Type 4 transfer function, lazily emitted the first time an inverted
	// SMask needs it (zero value = not yet emitted).
	invertFunction IndirectReference
	xmp            IndirectReference
	infoDict       IndirectReference
	// noSmaskGraphicState is the shared "/SMask /None" /ExtGState used to turn a soft mask back off after a masked
	// form-XObject draw. Lazily emitted the first time drawFormXObjectWithMask needs it (zero value = not yet emitted).
	noSmaskGraphicState    IndirectReference
	nextObjectNumber       int32
	inverseRasterScale     float32
	rasterScale            float32
	nextFontSubsetTagValue uint32
	uuid                   UUID
	closed                 bool
}

// NewDocument applies the rasterDPI (<=0 → 72) and encodingQuality (<0 → 0) clamps and prepares the object catalog.
// Nothing is written until the first page begins.
func NewDocument(w stream.WStream, metadata *Metadata) *Document {
	if metadata == nil {
		metadata = DefaultMetadata()
	}
	rasterDPI := metadata.RasterDPI
	if rasterDPI <= 0 {
		rasterDPI = defaultRasterDPI
	}
	d := &Document{
		stream:             w,
		metadata:           *metadata,
		rasterScale:        rasterDPI / defaultRasterDPI,
		inverseRasterScale: defaultRasterDPI / rasterDPI,
		nextObjectNumber:   1,
		fillGSMap:          map[fillGSKey]IndirectReference{},
		strokeGSMap:        map[strokeGSKey]IndirectReference{},
		pdfBitmapMap:       map[bitmapKey]IndirectReference{},
		gradientPatternMap: map[string]IndirectReference{},
		imageShaderMap:     map[string]IndirectReference{},
		typefaceMetrics:    map[uint32]*font.AdvancedMetrics{},
		toUnicodeMaps:      map[uint32][]int32{},
		fonts:              map[uint32]*pdfFont{},
	}
	d.metadata.RasterDPI = rasterDPI
	if d.metadata.EncodingQuality < 0 {
		d.metadata.EncodingQuality = 0
	}
	return d
}

// Metadata returns the document's (clamped) metadata.
func (d *Document) Metadata() *Metadata { return &d.metadata }

// ReserveRef reserves an indirect object number; every returned reference must be passed to EmitAt exactly once. (Emit
// is the shorthand for the common case: it reserves its own reference and emits under it.)
func (d *Document) ReserveRef() IndirectReference {
	ref := IndirectReference{value: d.nextObjectNumber}
	d.nextObjectNumber++
	return ref
}

// Emit serializes object under a freshly reserved reference.
func (d *Document) Emit(object Object) IndirectReference {
	return d.EmitAt(object, d.ReserveRef())
}

// EmitAt serializes object under ref and returns ref. A reference that does not point at an object (see
// IndirectReference.IsValid) is rejected: nothing is written and the zero, invalid reference is returned.
func (d *Document) EmitAt(object Object, ref IndirectReference) IndirectReference {
	if !ref.IsValid() {
		return IndirectReference{}
	}
	beginIndirectObject(&d.offsetMap, ref, d.stream)
	object.emit(d.stream)
	endIndirectObject(d.stream)
	return ref
}

// emitStream emits dict, then " stream\n", the writer's bytes, and "\nendstream", as a single indirect object.
func (d *Document) emitStream(dict *Dict, content []byte, ref IndirectReference) {
	beginIndirectObject(&d.offsetMap, ref, d.stream)
	dict.emit(d.stream)
	writeText(d.stream, " stream\n")
	d.stream.Write(content)
	writeText(d.stream, "\nendstream")
	endIndirectObject(d.stream)
}

// minimumFlateSavings is the minimum byte savings required to justify adding a /Filter /FlateDecode entry to the stream
// dict.
const minimumFlateSavings = len("/Filter /FlateDecode ")

// StreamOut emits content as a stream object, FlateDecode-compressing it when compression is enabled, the level is not
// None, and doing so actually saves space. dict may be nil (a fresh dictionary is used). Returns the object's
// reference.
func (d *Document) StreamOut(dict *Dict, content []byte, compress bool) IndirectReference {
	ref := d.ReserveRef()
	if dict == nil {
		dict = NewDict()
	}
	if d.metadata.CompressionLevel != CompressionNone && compress &&
		int64(len(content)) > int64(minimumFlateSavings) {
		var compressed stream.MemoryWStream
		deflate := NewDeflateWStream(&compressed, int(d.metadata.CompressionLevel))
		deflate.Write(content)
		deflate.Finalize()
		if int64(len(content)) > compressed.BytesWritten()+int64(minimumFlateSavings) {
			content = append([]byte(nil), compressed.Bytes()...)
			dict.InsertName("Filter", "FlateDecode")
		}
	}
	dict.InsertInt("Length", int32(len(content)))
	d.emitStream(dict, content, ref)
	return ref
}

// ---- page lifecycle ---------------------------------------------------------------------------------

// Page is an in-progress PDF page: a content stream plus a resource dictionary that the caller (the PDF device, or
// the application directly) fills between BeginPage and EndPage.
type Page struct {
	doc              *Document
	content          *stream.MemoryWStream
	resources        *Dict
	device           *Device
	width            float32
	height           float32
	pageSize         geom.ISize
	initialTransform geom.Matrix
}

// Content returns the page's content stream; write PDF content operators here before EndPage.
func (p *Page) Content() stream.WStream { return p.content }

// Resources returns the page's /Resources dictionary.
func (p *Page) Resources() *Dict { return p.resources }

// InitialTransform is the device-initial matrix mapping the canvas's top-left, raster-scaled space to PDF's
// bottom-left, 72dpi space. The PDF device consumes this.
func (p *Page) InitialTransform() geom.Matrix { return p.initialTransform }

// PageSize is the rasterized page size in device pixels (round(width*rasterScale)).
func (p *Page) PageSize() geom.ISize { return p.pageSize }

// BeginPage starts a new page. A page left open by a missing EndPage is finished first, as upstream does. On the first
// page it writes the file header and emits the document information dictionary (and, in PDF/A mode, the XMP metadata).
// It returns the Page whose content and resources the caller fills before calling EndPage.
func (d *Document) BeginPage(width, height float32) *Page {
	// The open page's object number is already reserved, so overwriting it would drop its content and leave an in-use
	// xref entry pointing at byte 0.
	d.EndPage()
	if len(d.pages) == 0 {
		serializeHeader(&d.offsetMap, d.stream)
		d.infoDict = d.Emit(MakeDocumentInformationDict(&d.metadata))
		if d.metadata.PDFA {
			d.uuid = CreateUUID(&d.metadata)
			// Same UUID for Document ID and Instance ID: this is the first (only) revision.
			d.xmp = MakeXMPObject(&d.metadata, d.uuid, d.uuid, d)
		}
	}
	// By scaling the page at the device level, layer devices are created at the rasterized scale.
	pageSize := geom.ISize{
		Width:  geom.RoundToInt(width * d.rasterScale),
		Height: geom.RoundToInt(height * d.rasterScale),
	}
	// The canvas uses the top-left as the origin, PDF the bottom-left; this matrix corrects that and applies the raster
	// scale.
	var initialTransform geom.Matrix
	initialTransform.SetScaleTranslate(d.inverseRasterScale, -d.inverseRasterScale,
		0, d.inverseRasterScale*float32(pageSize.Height))
	p := &Page{
		doc:              d,
		content:          stream.NewMemoryWStream(),
		resources:        NewDict(),
		width:            width,
		height:           height,
		pageSize:         pageSize,
		initialTransform: initialTransform,
	}
	d.pageRefs = append(d.pageRefs, d.ReserveRef())
	d.current = p
	return p
}

// BeginPageCanvas starts a page backed by a real PDF Device and returns a canvas that records draw ops into that
// device's content stream. The canvas takes its coordinates in the page's nominal 72dpi point space, whatever the
// metadata's RasterDPI. EndPage serializes the device's content and /Resources.
func (d *Document) BeginPageCanvas(width, height float32) *canvas.Canvas {
	p := d.BeginPage(width, height)
	p.device = newDevice(p.pageSize, d, p.initialTransform)
	c := canvas.New(p.device)
	// The device (and any layer device it spawns) works at the rasterized scale, while the caller draws in nominal
	// points; this scale bridges the two. The initial transform prepended to the content stream divides it back out on
	// the way to PDF's 72dpi user space, so without it every draw would be shrunk by 72/RasterDPI.
	c.Scale(d.rasterScale, d.rasterScale)
	return c
}

// EndPage emits the current page's content stream and builds the Page dictionary (MediaBox, Resources, Contents,
// StructParents, Tabs).
func (d *Document) EndPage() {
	p := d.current
	if p == nil {
		return
	}
	page := NewTypedDict("Page")
	mediaW := float32(p.pageSize.Width) * d.inverseRasterScale
	mediaH := float32(p.pageSize.Height) * d.inverseRasterScale
	// A device-backed page (BeginPageCanvas) supplies its content and resources from the recorded draws; a raw page
	// (BeginPage) uses the caller-filled content stream and resource dict.
	resources := p.resources
	content := p.content.Bytes()
	if p.device != nil {
		resources = p.device.makeResourceDict()
		content = p.device.contentBytes()
	}
	page.InsertObject("Resources", resources)
	page.InsertObject("MediaBox", rectToArray(geom.Rect{Right: mediaW, Bottom: mediaH}))
	page.InsertRef("Contents", d.StreamOut(nil, content, true))
	// The StructParents identifier for each page is its 0-based page index.
	page.InsertInt("StructParents", int32(len(d.pages)))
	// Tabs is PDF 1.5, but setting it checks an accessibility box.
	page.InsertName("Tabs", "S")
	d.pages = append(d.pages, page)
	d.current = nil
}

// generatePageTree builds a balanced tree of /Pages nodes (branching factor maxNodeSize = 8), built bottom up, skipping
// internal nodes with a single child.
func generatePageTree(doc *Document, pages []*Dict, pageRefs []IndirectReference) IndirectReference {
	type node struct {
		dict                 *Dict
		reservedRef          IndirectReference
		pageObjectDescendant int
	}
	layer := func(vec []node) []node {
		const maxNodeSize = 8
		n := len(vec)
		resultLen := (n-1)/maxNodeSize + 1
		result := make([]node, 0, resultLen)
		index := 0
		for i := 0; i < resultLen; i++ {
			if n != 1 && index+1 == n { // No need to create a new node.
				result = append(result, vec[index])
				index++
				continue
			}
			parent := doc.ReserveRef()
			kidsList := NewArray()
			descendantCount := 0
			for j := 0; j < maxNodeSize && index < n; j++ {
				child := vec[index]
				index++
				child.dict.InsertRef("Parent", parent)
				kidsList.AppendRef(doc.EmitAt(child.dict, child.reservedRef))
				descendantCount += child.pageObjectDescendant
			}
			next := NewTypedDict("Pages")
			next.InsertInt("Count", int32(descendantCount))
			next.InsertObject("Kids", kidsList)
			result = append(result, node{dict: next, reservedRef: parent, pageObjectDescendant: descendantCount})
		}
		return result
	}
	currentLayer := make([]node, 0, len(pages))
	for i := range pages {
		currentLayer = append(currentLayer, node{dict: pages[i], reservedRef: pageRefs[i], pageObjectDescendant: 1})
	}
	currentLayer = layer(currentLayer)
	for len(currentLayer) > 1 {
		currentLayer = layer(currentLayer)
	}
	root := currentLayer[0]
	return doc.EmitAt(root.dict, root.reservedRef)
}

// Close finishes any page left open by a BeginPage with no matching EndPage, builds the document catalog (page tree,
// viewer preferences, language), emits it, and writes the xref table and trailer. Closing a document with no pages, or
// an already-closed/aborted document, writes nothing.
func (d *Document) Close() {
	if d.closed {
		return
	}
	// BeginPage already reserved the page's object number, so an unfinished page must still be emitted; leaving it out
	// would put an in-use xref entry with offset 0 in the table and silently drop the page's content.
	d.EndPage()
	d.closed = true
	if len(d.pages) == 0 {
		return
	}
	docCatalog := NewTypedDict("Catalog")
	if d.metadata.PDFA {
		docCatalog.InsertRef("Metadata", d.xmp)
		// The sRGB /OutputIntents ICC profile is deferred: PDF/A isn't reachable through the oracle's public metadata
		// API, so this completeness gap is unobservable.
	}
	docCatalog.InsertRef("Pages", generatePageTree(d, d.pages, d.pageRefs))

	// If ViewerPreferences DisplayDocTitle isn't set to true, accessibility checks fail.
	if d.metadata.Title != "" {
		viewerPrefs := NewTypedDict("ViewerPreferences")
		viewerPrefs.InsertBool("DisplayDocTitle", true)
		docCatalog.InsertObject("ViewerPreferences", viewerPrefs)
	}
	if d.metadata.Lang != "" {
		docCatalog.InsertTextString("Lang", d.metadata.Lang)
	}

	docCatalogRef := d.Emit(docCatalog)

	// Emit every font used across the document.
	d.emitFonts()

	serializeFooter(&d.offsetMap, d.stream, d.infoDict, docCatalogRef, d.uuid)
}

// Abort discards all pending work and writes nothing further. The stream is left with whatever was already emitted (the
// header/info dict/content streams of completed pages).
func (d *Document) Abort() {
	d.closed = true
	d.current = nil
	d.pages = nil
	d.pageRefs = nil
}
