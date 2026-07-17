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
	"strings"
	"testing"
)

func sampleDateTime() DateTime {
	// 2026-07-06 12:30:45, timezone -08:00 (-480 minutes).
	return DateTime{TimeZoneMinutes: -480, Year: 2026, Month: 7, Day: 6, Hour: 12, Minute: 30, Second: 45}
}

func TestPDFDate(t *testing.T) {
	if got, want := pdfDate(sampleDateTime()), "D:20260706123045-08'00'"; got != want {
		t.Errorf("pdfDate = %q, want %q", got, want)
	}
	// Positive timezone with non-zero minutes.
	dt := DateTime{TimeZoneMinutes: 330, Year: 2000, Month: 1, Day: 2, Hour: 3, Minute: 4, Second: 5}
	if got, want := pdfDate(dt), "D:20000102030405+05'30'"; got != want {
		t.Errorf("pdfDate(+5:30) = %q, want %q", got, want)
	}
}

func TestToISO8601(t *testing.T) {
	if got, want := sampleDateTime().ToISO8601(), "2026-07-06T12:30:45-08:00"; got != want {
		t.Errorf("ToISO8601 = %q, want %q", got, want)
	}
	// Dates need no XML escaping: ISO8601 output contains no '&' or '<' characters.
	if countXMLEscapeSize(sampleDateTime().ToISO8601()) != 0 {
		t.Error("ISO8601 date should need no XML escaping")
	}
}

func TestMakeDocumentInformationDict(t *testing.T) {
	md := DefaultMetadata()
	md.Title = "My Title"
	md.Author = "Jane"
	md.Creation = sampleDateTime()
	got := string(emitToBytes(MakeDocumentInformationDict(md)))
	// Order is fixed: Title, Author, (Subject, Keywords, Creator skipped when empty), Producer, then dates.
	want := "<</Title (My Title)\n/Author (Jane)\n/Producer (Skia/PDF m142)\n" +
		"/CreationDate (D:20260706123045-08'00')>>"
	if got != want {
		t.Errorf("info dict =\n%q\nwant\n%q", got, want)
	}
}

func TestMakeDocumentInformationDictEmpty(t *testing.T) {
	// A zero Metadata has no fields set, so the info dict is empty.
	md := Metadata{}
	if got, want := string(emitToBytes(MakeDocumentInformationDict(&md))), "<<>>"; got != want {
		t.Errorf("empty info dict = %q, want %q", got, want)
	}
}

func TestUUIDToStringAndPdfID(t *testing.T) {
	uuid := UUID{0x81, 0xb1, 0x4a, 0xaf, 0xa3, 0x13, 0xdb, 0x63, 0xdb, 0xd6, 0xf9, 0x81, 0xe4, 0x9f, 0x94, 0xf4}
	if got, want := uuidToString(uuid), "81b14aaf-a313-db63-dbd6-f981e49f94f4"; got != want {
		t.Errorf("uuidToString = %q, want %q", got, want)
	}
	// MakePdfId emits two identical byte strings holding the raw UUID bytes (hex, since non-printable).
	got := string(emitToBytes(MakePdfID(uuid, uuid)))
	want := "[<81B14AAFA313DB63DBD6F981E49F94F4> <81B14AAFA313DB63DBD6F981E49F94F4>]"
	if got != want {
		t.Errorf("MakePdfID = %q, want %q", got, want)
	}
}

func TestEscapeXML(t *testing.T) {
	if got := escapeXML("", "<a>", "</a>"); got != "" {
		t.Errorf("empty input should stay empty, got %q", got)
	}
	if got, want := escapeXML("A&B<C", "<t>", "</t>"), "<t>A&amp;B&lt;C</t>"; got != want {
		t.Errorf("escapeXML = %q, want %q", got, want)
	}
}

func TestMakeXMPBytes(t *testing.T) {
	md := DefaultMetadata()
	md.Title = "Doc & Title"
	md.Modified = sampleDateTime()
	uuid := UUID{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xmp := makeXMPBytes(md, uuid, uuid)
	// Structural checks: packet delimiters, escaped title, embedded UUID, producer.
	for _, sub := range []string{
		"<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>",
		"<pdfaid:part>2</pdfaid:part>",
		"Doc &amp; Title",
		"<xmp:ModifyDate>2026-07-06T12:30:45-08:00</xmp:ModifyDate>",
		"<xmpMM:DocumentID>uuid:01020304-0506-0708-090a-0b0c0d0e0f10</xmpMM:DocumentID>",
		"<pdf:Producer>Skia/PDF m142</pdf:Producer>",
		"<?xpacket end=\"w\"?>",
	} {
		if !strings.Contains(xmp, sub) {
			t.Errorf("XMP missing %q", sub)
		}
	}
}
