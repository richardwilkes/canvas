// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Synthetic font builders for the table-level tests: the corpus in testdata carries none of the tables whose absence is
// indistinguishable from a table the loader never reads (name IDs 16/21, fvar, PCLT), and none of the container formats
// the sfnt reader accepts but a PDF cannot embed (WOFF). Each helper patches a real font rather than assembling one from
// nothing, so everything the parser needs is still there and only the table under test differs.

package font

import (
	"encoding/binary"
	"os"
	"sort"
	"testing"
	"unicode/utf16"

	"github.com/go-text/typesetting/font/opentype/tables"
)

// rawSfntTable returns the bytes of tag within the single-face sfnt in data.
func rawSfntTable(t *testing.T, data []byte, tag string) []byte {
	t.Helper()
	numTables := int(binary.BigEndian.Uint16(data[4:]))
	for i := range numTables {
		record := data[12+16*i:]
		if string(record[:4]) != tag {
			continue
		}
		offset := int(binary.BigEndian.Uint32(record[8:]))
		length := int(binary.BigEndian.Uint32(record[12:]))
		return data[offset : offset+length]
	}
	t.Fatalf("no %q table", tag)
	return nil
}

// sfntWithTables returns a copy of the single-face sfnt in data with each of the given tables added, replacing any table
// already present under the same tag; a nil replacement drops the tag instead (sfntWithoutTables). The whole file is
// reassembled — a fresh tag-ordered directory plus 4-byte-aligned table data — so the result is a font the sfnt reader
// accepts. The per-table checksums and head.checkSumAdjustment are carried over unchanged (a replaced table gets zero):
// nothing in the parse path verifies either.
func sfntWithTables(t *testing.T, data []byte, extra map[string][]byte) []byte {
	t.Helper()
	type table struct {
		tag      string
		bytes    []byte
		checkSum uint32
	}
	numTables := int(binary.BigEndian.Uint16(data[4:]))
	list := make([]table, 0, numTables+len(extra))
	for i := range numTables {
		record := data[12+16*i:]
		tag := string(record[:4])
		if _, replaced := extra[tag]; replaced {
			continue
		}
		offset := int(binary.BigEndian.Uint32(record[8:]))
		length := int(binary.BigEndian.Uint32(record[12:]))
		list = append(list, table{
			tag: tag, bytes: data[offset : offset+length],
			checkSum: binary.BigEndian.Uint32(record[4:]),
		})
	}
	for tag, raw := range extra {
		if len(tag) != 4 {
			t.Fatalf("table tag %q is not four bytes", tag)
		}
		if raw == nil { // dropped rather than replaced: the original was already skipped above
			continue
		}
		list = append(list, table{tag: tag, bytes: raw})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].tag < list[j].tag })

	out := make([]byte, 12+16*len(list))
	binary.BigEndian.PutUint32(out, binary.BigEndian.Uint32(data)) // sfntVersion
	binary.BigEndian.PutUint16(out[4:], uint16(len(list)))
	entrySelector := uint16(0)
	for 1<<(entrySelector+1) <= len(list) {
		entrySelector++
	}
	searchRange := uint16(16) << entrySelector
	binary.BigEndian.PutUint16(out[6:], searchRange)
	binary.BigEndian.PutUint16(out[8:], entrySelector)
	binary.BigEndian.PutUint16(out[10:], uint16(len(list))*16-searchRange)
	for i, tbl := range list {
		for len(out)%4 != 0 { // tables must start on a 4-byte boundary
			out = append(out, 0)
		}
		offset := len(out)
		out = append(out, tbl.bytes...)
		// The appends may have moved the backing array, so re-slice the record only now.
		record := out[12+16*i:]
		copy(record, tbl.tag)
		binary.BigEndian.PutUint32(record[4:], tbl.checkSum)
		binary.BigEndian.PutUint32(record[8:], uint32(offset))
		binary.BigEndian.PutUint32(record[12:], uint32(len(tbl.bytes)))
	}
	return out
}

// sfntWithoutTables returns a copy of the single-face sfnt in data with each of the given tables removed, reassembled
// by sfntWithTables. It is how the tests reach the code paths a table's absence selects, since every font in testdata
// carries the tables the parser reads.
func sfntWithoutTables(t *testing.T, data []byte, tags ...string) []byte {
	t.Helper()
	drop := make(map[string][]byte, len(tags))
	for _, tag := range tags {
		drop[tag] = nil
	}
	return sfntWithTables(t, data, drop)
}

// synthNameTable builds a version 0 'name' table holding each entry as a Windows / Unicode BMP / US-English (3, 1,
// 0x409) record, the combination tables.Name.Name prefers. The records are sorted by name ID, as the directory requires
// (only the name ID varies here).
func synthNameTable(names map[tables.NameID]string) []byte {
	ids := make([]int, 0, len(names))
	for id := range names {
		ids = append(ids, int(id))
	}
	sort.Ints(ids)
	storage := 6 + 12*len(ids)
	total := storage
	encodings := make([][]byte, len(ids))
	for i, id := range ids {
		for _, unit := range utf16.Encode([]rune(names[tables.NameID(id)])) {
			encodings[i] = binary.BigEndian.AppendUint16(encodings[i], unit)
		}
		total += len(encodings[i])
	}
	out := make([]byte, storage, total)
	binary.BigEndian.PutUint16(out[2:], uint16(len(ids)))
	binary.BigEndian.PutUint16(out[4:], uint16(storage))
	for i, id := range ids {
		encoded := encodings[i]
		record := out[6+12*i:]
		binary.BigEndian.PutUint16(record, 3)         // platformID: Windows
		binary.BigEndian.PutUint16(record[2:], 1)     // encodingID: Unicode BMP
		binary.BigEndian.PutUint16(record[4:], 0x409) // languageID: US English
		binary.BigEndian.PutUint16(record[6:], uint16(id))
		binary.BigEndian.PutUint16(record[8:], uint16(len(encoded)))
		binary.BigEndian.PutUint16(record[10:], uint16(len(out)-storage))
		out = append(out, encoded...)
	}
	return out
}

// synthCmapFormat13 builds a 'cmap' table holding one format-13 subtable under platform 3 / encoding 10 (Windows UCS-4),
// the first 32-bit encoding ProcessCmap prefers and therefore the subtable a face carrying it resolves through. Each
// group is {startCharCode, endCharCode, glyphID}; nothing on the parse path checks that the codes are ordered or in
// range, which is what lets the malformed-group cases exist at all.
func synthCmapFormat13(groups ...[3]uint32) []byte {
	const subtableOffset = 12
	out := make([]byte, subtableOffset, subtableOffset+16+12*len(groups))
	binary.BigEndian.PutUint16(out[2:], 1)  // numTables
	binary.BigEndian.PutUint16(out[4:], 3)  // platformID: Windows
	binary.BigEndian.PutUint16(out[6:], 10) // encodingID: UCS-4
	binary.BigEndian.PutUint32(out[8:], subtableOffset)
	out = binary.BigEndian.AppendUint16(out, 13) // format
	out = binary.BigEndian.AppendUint16(out, 0)  // reserved
	out = binary.BigEndian.AppendUint32(out, uint32(16+12*len(groups)))
	out = binary.BigEndian.AppendUint32(out, 0) // language
	out = binary.BigEndian.AppendUint32(out, uint32(len(groups)))
	for _, g := range groups {
		out = binary.BigEndian.AppendUint32(out, g[0])
		out = binary.BigEndian.AppendUint32(out, g[1])
		out = binary.BigEndian.AppendUint32(out, g[2])
	}
	return out
}

// os2FsSelectionOffset is the byte offset of fsSelection within the OS/2 table.
const os2FsSelectionOffset = 62

// os2WithFsSelectionBits returns a copy of data's OS/2 table with bits set or cleared in fsSelection.
func os2WithFsSelectionBits(t *testing.T, data []byte, bits uint16, set bool) []byte {
	t.Helper()
	out := append([]byte(nil), rawSfntTable(t, data, "OS/2")...)
	if len(out) < os2FsSelectionOffset+2 {
		t.Fatalf("OS/2 table is %d bytes, too short to hold fsSelection", len(out))
	}
	fsSelection := binary.BigEndian.Uint16(out[os2FsSelectionOffset:])
	if set {
		fsSelection |= bits
	} else {
		fsSelection &^= bits
	}
	binary.BigEndian.PutUint16(out[os2FsSelectionOffset:], fsSelection)
	return out
}

// synthFvarTable builds an 'fvar' table declaring axisCount weight axes and no named instances — enough for a parse that
// reports the font as variable, which is all the advanced metrics ask of it.
func synthFvarTable(axisCount int) []byte {
	const axisSize = 20
	out := make([]byte, 16+axisSize*axisCount)
	binary.BigEndian.PutUint16(out, 1)      // majorVersion
	binary.BigEndian.PutUint16(out[4:], 16) // axesArrayOffset
	binary.BigEndian.PutUint16(out[6:], 2)  // reserved, permanently 2
	binary.BigEndian.PutUint16(out[8:], uint16(axisCount))
	binary.BigEndian.PutUint16(out[10:], axisSize)   // axisSize
	binary.BigEndian.PutUint16(out[14:], axisSize+4) // instanceSize (axisCount * Fixed + 4)
	for i := range axisCount {
		axis := out[16+axisSize*i:]
		copy(axis, "wght")
		binary.BigEndian.PutUint32(axis[4:], 100<<16)  // minValue
		binary.BigEndian.PutUint32(axis[8:], 400<<16)  // defaultValue
		binary.BigEndian.PutUint32(axis[12:], 900<<16) // maxValue
	}
	return out
}

// synthPCLTTable builds a 54-byte 'PCLT' table carrying the given capHeight and serifStyle; every other field is left
// zero, since the advanced metrics read only those two.
func synthPCLTTable(capHeight uint16, serifStyle uint8) []byte {
	out := make([]byte, 54)
	binary.BigEndian.PutUint32(out, 1<<16) // version 1.0
	binary.BigEndian.PutUint16(out[pcltCapHeightOffset:], capHeight)
	out[pcltSerifStyleOffset] = serifStyle
	return out
}

// woffWrap repackages the single-face sfnt in data as a WOFF file with every table stored uncompressed. WOFF is a
// container the sfnt reader accepts and a PDF cannot embed, which is what FontFlagAltDataFormat reports.
func woffWrap(t *testing.T, data []byte) []byte {
	t.Helper()
	const headerSize, entrySize = 44, 20
	numTables := int(binary.BigEndian.Uint16(data[4:]))
	out := make([]byte, headerSize+entrySize*numTables)
	copy(out, "wOFF")
	binary.BigEndian.PutUint32(out[4:], binary.BigEndian.Uint32(data)) // flavor: the wrapped sfntVersion
	binary.BigEndian.PutUint16(out[12:], uint16(numTables))
	binary.BigEndian.PutUint32(out[16:], uint32(len(data))) // totalSfntSize
	binary.BigEndian.PutUint16(out[20:], 1)                 // majorVersion
	for i := range numTables {
		record := data[12+16*i:]
		offset := int(binary.BigEndian.Uint32(record[8:]))
		length := int(binary.BigEndian.Uint32(record[12:]))
		for len(out)%4 != 0 { // WOFF tables are 4-byte aligned too
			out = append(out, 0)
		}
		tableOffset := len(out)
		out = append(out, data[offset:offset+length]...)
		entry := out[headerSize+entrySize*i:]
		copy(entry, record[:4])                                                     // tag
		binary.BigEndian.PutUint32(entry[4:], uint32(tableOffset))                  // offset
		binary.BigEndian.PutUint32(entry[8:], uint32(length))                       // compLength, == origLength: stored raw
		binary.BigEndian.PutUint32(entry[12:], uint32(length))                      // origLength
		binary.BigEndian.PutUint32(entry[16:], binary.BigEndian.Uint32(record[4:])) // origChecksum
	}
	binary.BigEndian.PutUint32(out[8:], uint32(len(out))) // length: the whole WOFF file
	return out
}

// readTestFont returns the raw bytes of a font in testdata.
func readTestFont(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
