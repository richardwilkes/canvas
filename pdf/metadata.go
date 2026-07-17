// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Metadata and DateTime types for PDF document metadata: the document information dictionary (/Info), the document /ID,
// and the PDF/A XMP metadata packet. The oracle's public metadata API has no PDF/A toggle, so the UUID/XMP path can't
// be differentially verified (kept for completeness). CreateUUID hashes the current time, so it is non-reproducible by
// design — it is only meaningful in PDF/A mode, where per-run uniqueness is what's wanted.

package pdf

import (
	"crypto/md5" //nolint:gosec // not used for security: PDF document IDs are conventionally MD5 (mirrors Skia)
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"time"
)

// milestone is the version number embedded in the default Producer string.
const milestone = 142

// defaultProducer is the default value for the /Producer document metadata entry.
var defaultProducer = fmt.Sprintf("Skia/PDF m%d", milestone)

// defaultRasterDPI is the default resolution, in dots per inch, used to rasterize content that has no vector
// representation.
const defaultRasterDPI float32 = 72.0

// CompressionLevel is the zlib deflate level used for stream compression, or None to disable it.
type CompressionLevel int

// CompressionLevel values.
const (
	CompressionDefault     CompressionLevel = -1
	CompressionNone        CompressionLevel = 0
	CompressionLowButFast  CompressionLevel = 1
	CompressionAverage     CompressionLevel = 6
	CompressionHighButSlow CompressionLevel = 9
)

// DateTime is a PDF date/time value. The zero value represents "no date".
type DateTime struct {
	TimeZoneMinutes int16
	Year            uint16
	Month           uint8
	DayOfWeek       uint8
	Day             uint8
	Hour            uint8
	Minute          uint8
	Second          uint8
}

func (dt DateTime) isZero() bool { return dt == DateTime{} }

// ToISO8601 formats dt as "YYYY-mm-ddTHH:MM:SS[+|-]ZZ:ZZ".
func (dt DateTime) ToISO8601() string {
	timeZoneMinutes := int(dt.TimeZoneMinutes)
	timezoneSign := byte('+')
	if timeZoneMinutes < 0 {
		timezoneSign = '-'
	}
	timeZoneHours := absInt(timeZoneMinutes) / 60
	timeZoneMinutes = absInt(timeZoneMinutes) % 60
	return fmt.Sprintf("%04d-%02d-%02dT%02d:%02d:%02d%c%02d:%02d",
		dt.Year, dt.Month, dt.Day, dt.Hour, dt.Minute, dt.Second,
		timezoneSign, timeZoneHours, timeZoneMinutes)
}

// pdfDate formats dt in PDF date-string form: "D:YYYYMMDDHHMMSS[+|-]HH'MM'".
func pdfDate(dt DateTime) string {
	timeZoneMinutes := int(dt.TimeZoneMinutes)
	timezoneSign := byte('+')
	if timeZoneMinutes < 0 {
		timezoneSign = '-'
	}
	timeZoneHours := absInt(timeZoneMinutes) / 60
	timeZoneMinutes = absInt(timeZoneMinutes) % 60
	return fmt.Sprintf("D:%04d%02d%02d%02d%02d%02d%c%02d'%02d'",
		dt.Year, dt.Month, dt.Day, dt.Hour, dt.Minute, dt.Second,
		timezoneSign, timeZoneHours, timeZoneMinutes)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// Metadata holds the document metadata and encoding options for a PDF document. The zero value does not carry the
// intended defaults (Go has no field initializers); use DefaultMetadata to get a Metadata with its Producer, RasterDPI,
// EncodingQuality, and CompressionLevel fields already set.
type Metadata struct {
	Title            string
	Author           string
	Subject          string
	Keywords         string
	Creator          string
	Producer         string
	Lang             string
	CompressionLevel CompressionLevel
	EncodingQuality  int
	RasterDPI        float32
	Modified         DateTime
	Creation         DateTime
	PDFA             bool
}

// DefaultMetadata returns a Metadata with Producer, RasterDPI, EncodingQuality, and CompressionLevel set to their
// default values.
func DefaultMetadata() *Metadata {
	return &Metadata{
		Producer:         defaultProducer,
		RasterDPI:        defaultRasterDPI,
		EncodingQuality:  101,
		CompressionLevel: CompressionDefault,
	}
}

// metadataKeys defines the /Info dictionary's key order and how to read each value from a Metadata.
var metadataKeys = []struct {
	get func(*Metadata) string
	key string
}{
	{key: "Title", get: func(m *Metadata) string { return m.Title }},
	{key: "Author", get: func(m *Metadata) string { return m.Author }},
	{key: "Subject", get: func(m *Metadata) string { return m.Subject }},
	{key: "Keywords", get: func(m *Metadata) string { return m.Keywords }},
	{key: "Creator", get: func(m *Metadata) string { return m.Creator }},
	{key: "Producer", get: func(m *Metadata) string { return m.Producer }},
}

// MakeDocumentInformationDict builds the /Info document information dictionary from metadata, omitting any field that
// is empty or a zero DateTime.
func MakeDocumentInformationDict(metadata *Metadata) *Dict {
	dict := NewDict()
	for _, kv := range metadataKeys {
		if value := kv.get(metadata); value != "" {
			dict.InsertTextString(kv.key, value)
		}
	}
	if !metadata.Creation.isZero() {
		dict.InsertTextString("CreationDate", pdfDate(metadata.Creation))
	}
	if !metadata.Modified.isZero() {
		dict.InsertTextString("ModDate", pdfDate(metadata.Modified))
	}
	return dict
}

// UUID is a 16-byte universally unique identifier.
type UUID [16]byte

func (u UUID) isZero() bool { return u == UUID{} }

// CreateUUID derives a UUID from the current time and the document's metadata fields. The hash mixes the current time,
// so the result is non-reproducible by design (used only in PDF/A mode).
func CreateUUID(metadata *Metadata) UUID {
	h := md5.New() //nolint:gosec // not used for security: PDF document IDs are conventionally MD5 (mirrors Skia)
	h.Write([]byte("org.skia.pdf\n"))
	var msec [8]byte
	binary.LittleEndian.PutUint64(msec[:], math.Float64bits(float64(time.Now().UnixNano())/1e6))
	h.Write(msec[:])
	var now DateTime
	getDateTime(&now)
	h.Write(dateTimeBytes(now))
	h.Write(dateTimeBytes(metadata.Creation))
	h.Write(dateTimeBytes(metadata.Modified))
	for _, kv := range metadataKeys {
		h.Write([]byte(kv.key))
		h.Write([]byte{0o37})
		h.Write([]byte(kv.get(metadata)))
		h.Write([]byte{0o36})
	}
	var digest [16]byte
	copy(digest[:], h.Sum(nil))
	// See RFC 4122, page 6-7. (This deliberately derives digest[8] from digest[6] rather than digest[8] itself — a
	// long-standing quirk that output-compatibility requires preserving.)
	digest[6] = (digest[6] & 0x0F) | 0x30
	digest[8] = (digest[6] & 0x3F) | 0x80
	return UUID(digest)
}

func dateTimeBytes(dt DateTime) []byte {
	return []byte{
		byte(dt.TimeZoneMinutes), byte(uint16(dt.TimeZoneMinutes) >> 8),
		byte(dt.Year), byte(dt.Year >> 8),
		dt.Month, dt.DayOfWeek, dt.Day, dt.Hour, dt.Minute, dt.Second,
	}
}

// getDateTime fills dt with the current UTC time.
func getDateTime(dt *DateTime) {
	now := time.Now().UTC()
	dt.TimeZoneMinutes = 0
	dt.Year = uint16(now.Year())
	dt.Month = uint8(now.Month())
	dt.DayOfWeek = uint8(now.Weekday())
	dt.Day = uint8(now.Day())
	dt.Hour = uint8(now.Hour())
	dt.Minute = uint8(now.Minute())
	dt.Second = uint8(now.Second())
}

// MakePdfID returns the document /ID array: two byte strings holding the raw UUID bytes.
func MakePdfID(doc, instance UUID) *Array {
	array := NewArray()
	array.AppendByteString(string(doc[:]))
	array.AppendByteString(string(instance[:]))
	return array
}

// hexLower holds the lowercase hexadecimal digit characters.
const hexLower = "0123456789abcdef"

// uuidToString formats uuid in the standard 8-4-4-4-12 hyphenated hex form.
func uuidToString(uuid UUID) string {
	var b strings.Builder
	hexify := func(start, count int) {
		for i := 0; i < count; i++ {
			v := uuid[start+i]
			b.WriteByte(hexLower[v>>4])
			b.WriteByte(hexLower[v&0xF])
		}
	}
	hexify(0, 4)
	b.WriteByte('-')
	hexify(4, 2)
	b.WriteByte('-')
	hexify(6, 2)
	b.WriteByte('-')
	hexify(8, 2)
	b.WriteByte('-')
	hexify(10, 6)
	return b.String()
}

// countXMLEscapeSize returns the number of extra bytes escapeXML would add when escaping input.
func countXMLEscapeSize(input string) int {
	extra := 0
	for i := 0; i < len(input); i++ {
		switch input[i] {
		case '&':
			extra += 4
		case '<':
			extra += 3
		}
	}
	return extra
}

// escapeXML escapes input for inclusion in XML text ("&"→"&amp;", "<"→"&lt;") and wraps the result in before/after. An
// empty input returns "" with no wrapping.
func escapeXML(input, before, after string) string {
	if input == "" {
		return input
	}
	var b strings.Builder
	b.WriteString(before)
	for i := 0; i < len(input); i++ {
		switch input[i] {
		case '&':
			b.WriteString("&amp;")
		case '<':
			b.WriteString("&lt;")
		default:
			b.WriteByte(input[i])
		}
	}
	b.WriteString(after)
	return b.String()
}

// MakeXMPObject emits an uncompressed /Metadata XML stream object through the document. Only used in PDF/A mode (not
// reachable through the oracle's public metadata API).
func MakeXMPObject(metadata *Metadata, doc, instance UUID, d *Document) IndirectReference {
	dict := NewTypedDict("Metadata")
	dict.InsertName("Subtype", "XML")
	// The standard requires this packet be greppable, so it is never compressed.
	return d.StreamOut(dict, []byte(makeXMPBytes(metadata, doc, instance)), false)
}

// xmpTemplate is the XMP packet's printf-style template, with %s placeholders filled in by makeXMPBytes.
const xmpTemplate = "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n" +
	"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"\n" +
	" x:xmptk=\"Adobe XMP Core 5.4-c005 78.147326, " +
	"2012/08/23-13:03:03\">\n" +
	"<rdf:RDF " +
	"xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n" +
	"<rdf:Description rdf:about=\"\"\n" +
	" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n" +
	" xmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n" +
	" xmlns:xmpMM=\"http://ns.adobe.com/xap/1.0/mm/\"\n" +
	" xmlns:pdf=\"http://ns.adobe.com/pdf/1.3/\"\n" +
	" xmlns:pdfaid=\"http://www.aiim.org/pdfa/ns/id/\">\n" +
	"<pdfaid:part>2</pdfaid:part>\n" +
	"<pdfaid:conformance>B</pdfaid:conformance>\n" +
	"%s" + // ModifyDate
	"%s" + // CreateDate
	"%s" + // xmp:CreatorTool
	"<dc:format>application/pdf</dc:format>\n" +
	"%s" + // dc:title
	"%s" + // dc:description
	"%s" + // author
	"%s" + // keywords
	"<xmpMM:DocumentID>uuid:%s</xmpMM:DocumentID>\n" +
	"<xmpMM:InstanceID>uuid:%s</xmpMM:InstanceID>\n" +
	"%s" + // pdf:Producer
	"%s" + // pdf:Keywords
	"</rdf:Description>\n" +
	"</rdf:RDF>\n" +
	"</x:xmpmeta>\n" +
	"<?xpacket end=\"w\"?>\n"

// makeXMPBytes builds the XMP packet contents by filling in xmpTemplate.
func makeXMPBytes(metadata *Metadata, doc, instance UUID) string {
	var creationDate, modificationDate string
	if !metadata.Creation.isZero() {
		creationDate = fmt.Sprintf("<xmp:CreateDate>%s</xmp:CreateDate>\n", metadata.Creation.ToISO8601())
	}
	if !metadata.Modified.isZero() {
		modificationDate = fmt.Sprintf("<xmp:ModifyDate>%s</xmp:ModifyDate>\n", metadata.Modified.ToISO8601())
	}
	title := escapeXML(metadata.Title,
		"<dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">", "</rdf:li></rdf:Alt></dc:title>\n")
	author := escapeXML(metadata.Author,
		"<dc:creator><rdf:Seq><rdf:li>", "</rdf:li></rdf:Seq></dc:creator>\n")
	subject := escapeXML(metadata.Subject,
		"<dc:description><rdf:Alt><rdf:li xml:lang=\"x-default\">", "</rdf:li></rdf:Alt></dc:description>\n")
	keywords1 := escapeXML(metadata.Keywords,
		"<dc:subject><rdf:Bag><rdf:li>", "</rdf:li></rdf:Bag></dc:subject>\n")
	keywords2 := escapeXML(metadata.Keywords, "<pdf:Keywords>", "</pdf:Keywords>\n")
	producer := escapeXML(metadata.Producer, "<pdf:Producer>", "</pdf:Producer>\n")
	creator := escapeXML(metadata.Creator, "<xmp:CreatorTool>", "</xmp:CreatorTool>\n")
	documentID := uuidToString(doc)
	instanceID := uuidToString(instance)
	return fmt.Sprintf(xmpTemplate, modificationDate, creationDate, creator, title, subject, author,
		keywords1, documentID, instanceID, producer, keywords2)
}
