// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

import (
	"bytes"
	"encoding/binary"
)

// Fonts that need their instructions to draw correct contours, often called "tricky" fonts: their outlines are
// assembled by the glyph programs, horizontal moves included, so they run without backward compatibility. This is a
// port of the detection in read-fonts, Skrifa's companion crate in the same repository
// (read-fonts/src/tables/glyf/hint_reliant.rs), which takes its lists of family names and table checksums from
// FreeType.

// hintReliantNames are the family names, matched as substrings of the name FreeType would report for the font. A '?'
// stands for any character outside ASCII.
var hintReliantNames = [...]string{
	"cpop",               // dftt-p7.ttf; version 1.00, 1992 [DLJGyShoMedium]
	"DFGirl-W6-WIN-BF",   // dftt-h6.ttf; version 1.00, 1993
	"DFGothic-EB",        // DynaLab Inc. 1992-1995
	"DFGyoSho-Lt",        // DynaLab Inc. 1992-1995
	"DFHei",              // DynaLab Inc. 1992-1995 [DFHei-Bd-WIN-HK-BF]
	"DFHSGothic-W5",      // DynaLab Inc. 1992-1995
	"DFHSMincho-W3",      // DynaLab Inc. 1992-1995
	"DFHSMincho-W7",      // DynaLab Inc. 1992-1995
	"DFKaiSho-SB",        // dfkaisb.ttf
	"DFKaiShu",           // covers "DFKaiShu-Md-HK-BF", maybe DynaLab Inc.
	"DFKai-SB",           // kaiu.ttf; version 3.00, 1998 [DFKaiShu-SB-Estd-BF]
	"DFMing",             // DynaLab Inc. 1992-1995 [DFMing-Md-WIN-HK-BF]
	"DLC",                // dftt-m7.ttf; version 1.00, 1993 [DLCMingBold]
	"HuaTianKaiTi?",      // htkt2.ttf
	"HuaTianSongTi?",     // htst3.ttf
	"Ming(for ISO10646)", // hkscsiic.ttf; version 0.12, 2007 [Ming]
	"MingLiU",            // mingliu.ttf
	"MingMedium",         // dftt-m5.ttf; version 1.00, 1993 [DLCMingMedium]
	"PMingLiU",           // mingliu.ttc; version 3.21, 2001
	"MingLi43",           // mingli.ttf; version 1.00, 1992
}

// tableID identifies a table by its checksum and length.
type tableID struct {
	checksum, length uint32
}

// fontID identifies a font by its three hinting tables.
type fontID struct {
	cvt, fpgm, prep tableID
}

// hintReliantIDs are the fonts recognized by their hinting tables, whatever they are named.
var hintReliantIDs = [...]fontID{
	{ // MingLiU 1995
		cvt:  tableID{checksum: 0x05BCF058, length: 0x000002E4},
		fpgm: tableID{checksum: 0x28233BF1, length: 0x000087C4},
		prep: tableID{checksum: 0xA344A1EA, length: 0x000001E1},
	},
	{ // MingLiU 1996-
		cvt:  tableID{checksum: 0x05BCF058, length: 0x000002E4},
		fpgm: tableID{checksum: 0x28233BF1, length: 0x000087C4},
		prep: tableID{checksum: 0xA344A1EB, length: 0x000001E1},
	},
	{ // DFGothic-EB
		cvt:  tableID{checksum: 0x12C3EBB2, length: 0x00000350},
		fpgm: tableID{checksum: 0xB680EE64, length: 0x000087A7},
		prep: tableID{checksum: 0xCE939563, length: 0x00000758},
	},
	{ // DFGyoSho-Lt
		cvt:  tableID{checksum: 0x11E5EAD4, length: 0x00000350},
		fpgm: tableID{checksum: 0xCE5956E9, length: 0x0000BC85},
		prep: tableID{checksum: 0x8272F416, length: 0x00000045},
	},
	{ // DFHei-Md-HK-BF
		cvt:  tableID{checksum: 0x1257EB46, length: 0x00000350},
		fpgm: tableID{checksum: 0xF699D160, length: 0x0000715F},
		prep: tableID{checksum: 0xD222F568, length: 0x000003BC},
	},
	{ // DFHSGothic-W5
		cvt:  tableID{checksum: 0x1262EB4E, length: 0x00000350},
		fpgm: tableID{checksum: 0xE86A5D64, length: 0x00007940},
		prep: tableID{checksum: 0x7850F729, length: 0x000005FF},
	},
	{ // DFHSMincho-W3
		cvt:  tableID{checksum: 0x122DEB0A, length: 0x00000350},
		fpgm: tableID{checksum: 0x3D16328A, length: 0x0000859B},
		prep: tableID{checksum: 0xA93FC33B, length: 0x000002CB},
	},
	{ // DFHSMincho-W7
		cvt:  tableID{checksum: 0x125FEB26, length: 0x00000350},
		fpgm: tableID{checksum: 0xA5ACC982, length: 0x00007EE1},
		prep: tableID{checksum: 0x90999196, length: 0x0000041F},
	},
	{ // DFKaiShu
		cvt:  tableID{checksum: 0x11E5EAD4, length: 0x00000350},
		fpgm: tableID{checksum: 0x5A30CA3B, length: 0x00009063},
		prep: tableID{checksum: 0x13A42602, length: 0x0000007E},
	},
	{ // DFKaiShu, variant
		cvt:  tableID{checksum: 0x11E5EAD4, length: 0x00000350},
		fpgm: tableID{checksum: 0xA6E78C01, length: 0x00008998},
		prep: tableID{checksum: 0x13A42602, length: 0x0000007E},
	},
	{ // DFKaiShu-Md-HK-BF
		cvt:  tableID{checksum: 0x11E5EAD4, length: 0x00000360},
		fpgm: tableID{checksum: 0x9DB282B2, length: 0x0000C06E},
		prep: tableID{checksum: 0x53E6D7CA, length: 0x00000082},
	},
	{ // DFMing-Bd-HK-BF
		cvt:  tableID{checksum: 0x1243EB18, length: 0x00000350},
		fpgm: tableID{checksum: 0xBA0A8C30, length: 0x000074AD},
		prep: tableID{checksum: 0xF3D83409, length: 0x0000037B},
	},
	{ // DLCLiShu
		cvt:  tableID{checksum: 0x07DCF546, length: 0x00000308},
		fpgm: tableID{checksum: 0x40FE7C90, length: 0x00008E2A},
		prep: tableID{checksum: 0x608174B5, length: 0x0000007A},
	},
	{ // DLCHayBold
		cvt:  tableID{checksum: 0xEB891238, length: 0x00000308},
		fpgm: tableID{checksum: 0xD2E4DCD4, length: 0x0000676F},
		prep: tableID{checksum: 0x8EA5F293, length: 0x000003B8},
	},
	{ // HuaTianKaiTi
		cvt:  tableID{checksum: 0xFFFBFFFC, length: 0x00000008},
		fpgm: tableID{checksum: 0x9C9E48B8, length: 0x0000BEA2},
		prep: tableID{checksum: 0x70020112, length: 0x00000008},
	},
	{ // HuaTianSongTi
		cvt:  tableID{checksum: 0xFFFBFFFC, length: 0x00000008},
		fpgm: tableID{checksum: 0x0A5A0483, length: 0x00017C39},
		prep: tableID{checksum: 0x70020112, length: 0x00000008},
	},
	{ // NEC fadpop7.ttf
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x40C92555, length: 0x000000E5},
		prep: tableID{checksum: 0xA39B58E3, length: 0x0000117C},
	},
	{ // NEC fadrei5.ttf
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x33C41652, length: 0x000000E5},
		prep: tableID{checksum: 0x26D6C52A, length: 0x00000F6A},
	},
	{ // NEC fangot7.ttf
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x6DB1651D, length: 0x0000019D},
		prep: tableID{checksum: 0x6C6E4B03, length: 0x00002492},
	},
	{ // NEC fangyo5.ttf
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x40C92555, length: 0x000000E5},
		prep: tableID{checksum: 0xDE51FAD0, length: 0x0000117C},
	},
	{ // NEC fankyo5.ttf
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x85E47664, length: 0x000000E5},
		prep: tableID{checksum: 0xA6C62831, length: 0x00001CAA},
	},
	{ // NEC fanrgo5.ttf
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x2D891CFD, length: 0x0000019D},
		prep: tableID{checksum: 0xA0604633, length: 0x00001DE8},
	},
	{ // NEC fangot5.ttc
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x40AA774C, length: 0x000001CB},
		prep: tableID{checksum: 0x9B5CAA96, length: 0x00001F9A},
	},
	{ // NEC fanmin3.ttc
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x0D3DE9CB, length: 0x00000141},
		prep: tableID{checksum: 0xD4127766, length: 0x00002280},
	},
	{ // NEC FA-Gothic, 1996
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x4A692698, length: 0x000001F0},
		prep: tableID{checksum: 0x340D4346, length: 0x00001FCA},
	},
	{ // NEC FA-Minchou, 1996
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0xCD34C604, length: 0x00000166},
		prep: tableID{checksum: 0x6CF31046, length: 0x000022B0},
	},
	{ // NEC FA-RoundGothicB, 1996
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0x5DA75315, length: 0x0000019D},
		prep: tableID{checksum: 0x40745A5F, length: 0x000022E0},
	},
	{ // NEC FA-RoundGothicM, 1996
		cvt:  tableID{checksum: 0x00000000, length: 0x00000000},
		fpgm: tableID{checksum: 0xF055FC48, length: 0x000001C2},
		prep: tableID{checksum: 0x3900DED3, length: 0x00001E18},
	},
	{ // MINGLI.TTF, 1992
		cvt:  tableID{checksum: 0x00170003, length: 0x00000060},
		fpgm: tableID{checksum: 0xDBB4306E, length: 0x000058AA},
		prep: tableID{checksum: 0xD643482A, length: 0x00000035},
	},
	{ // DFHei-Bd-WIN-HK-BF, issue #1087
		cvt:  tableID{checksum: 0x1269EB58, length: 0x00000350},
		fpgm: tableID{checksum: 0x5CD5957A, length: 0x00006A4E},
		prep: tableID{checksum: 0xF758323A, length: 0x00000380},
	},
	{ // DFMing-Md-WIN-HK-BF, issue #1087
		cvt:  tableID{checksum: 0x122FEB0B, length: 0x00000350},
		fpgm: tableID{checksum: 0x7F10919A, length: 0x000070A9},
		prep: tableID{checksum: 0x7CD7E7B7, length: 0x0000025C},
	},
}

// isHintReliant reports whether the font with these tables needs its instructions to draw correct contours.
func isHintReliant(table func(tag string) []byte) bool {
	if hintReliantName(table("name"), table("OS/2")) {
		return true
	}
	// The control values are summed as the whole entries that are read from them.
	cvt := table("cvt ")
	id := fontID{cvt: tableIDOf(cvt[:len(cvt)&^1]), fpgm: tableIDOf(table("fpgm")), prep: tableIDOf(table("prep"))}
	for i := range hintReliantIDs {
		if hintReliantIDs[i] == id {
			return true
		}
	}
	return false
}

// tableIDOf computes a table's identity: the sfnt checksum, which sums the data as big-endian 32-bit words with the
// last one padded with zeros.
func tableIDOf(data []byte) tableID {
	length := uint32(len(data))
	var sum uint32
	for len(data) >= 4 {
		sum += binary.BigEndian.Uint32(data)
		data = data[4:]
	}
	if len(data) != 0 {
		var last [4]byte
		copy(last[:], data)
		sum += binary.BigEndian.Uint32(last[:])
	}
	return tableID{checksum: sum, length: length}
}

// Name table constants.
const (
	nameIDFamily            = 1
	nameIDTypographicFamily = 16
	nameIDWWSFamily         = 21
	os2SelectionWWS         = 1 << 8
)

// hintReliantName reports whether the family name FreeType would report for the font is on the list.
func hintReliantName(name, os2 []byte) bool {
	// With the OS/2 WWS bit set the typographic family comes first, and without it the WWS family does.
	ids := []uint16{nameIDWWSFamily, nameIDTypographicFamily, nameIDFamily}
	if len(os2) >= 64 && binary.BigEndian.Uint16(os2[62:])&os2SelectionWWS != 0 {
		ids = ids[1:]
	}
	for _, id := range ids {
		if family, ok := asciiName(name, id); ok {
			return matchesHintReliantName(family)
		}
	}
	return false
}

// nameRecord is one entry of the name table's directory.
type nameRecord struct {
	platform, encoding, language, length, offset uint16
}

// asciiName returns the name with the given ID as FreeType reads it: the record it would choose, cut at the first NUL,
// with '?' in place of each character outside the range 32 to 127. A Windows record is used if it is English or if
// there is no Macintosh record; otherwise a Macintosh English or Roman record; a Unicode record if there is neither.
func asciiName(name []byte, id uint16) ([]byte, bool) {
	if len(name) < 6 {
		return nil, false
	}
	count := int(binary.BigEndian.Uint16(name[2:]))
	storage := int(binary.BigEndian.Uint16(name[4:]))
	var appleEnglish, appleRoman, windows, unicode *nameRecord
	windowsIsEnglish := false
	for i := range count {
		raw := name[6:]
		if len(raw) < 12*(i+1) {
			break
		}
		raw = raw[12*i:]
		if binary.BigEndian.Uint16(raw[6:]) != id {
			continue
		}
		record := &nameRecord{
			platform: binary.BigEndian.Uint16(raw),
			encoding: binary.BigEndian.Uint16(raw[2:]),
			language: binary.BigEndian.Uint16(raw[4:]),
			length:   binary.BigEndian.Uint16(raw[8:]),
			offset:   binary.BigEndian.Uint16(raw[10:]),
		}
		if record.length == 0 {
			continue
		}
		switch record.platform {
		case 0, 2:
			unicode = record
		case 1:
			switch {
			case record.language == 0:
				appleEnglish = record
			case record.encoding == 0:
				appleRoman = record
			}
		case 3:
			english := record.language&0x3FF == 0x009
			if (windows == nil || english) && (record.encoding == 0 || record.encoding == 1 || record.encoding == 10) {
				windows = record
				windowsIsEnglish = english
			}
		}
	}
	apple := appleEnglish
	if apple == nil {
		apple = appleRoman
	}
	record, step := unicode, 2
	switch {
	case windows != nil && (apple == nil || windowsIsEnglish):
		record = windows
	case apple != nil:
		record, step = apple, 1
	}
	if record == nil {
		return nil, false
	}
	start := storage + int(record.offset)
	end := start + int(record.length)
	if end > len(name) {
		return nil, false
	}
	data := name[start:end]
	ascii := make([]byte, 0, len(data)/step)
	for ; len(data) >= step; data = data[step:] {
		code := uint16(data[0])
		if step == 2 {
			code = binary.BigEndian.Uint16(data)
		}
		if code == 0 {
			break
		}
		if code < 32 || code > 127 {
			code = '?'
		}
		ascii = append(ascii, byte(code))
	}
	return ascii, true
}

// matchesHintReliantName reports whether name contains one of the listed family names. The random tag of a font
// embedded in a PDF, six capitals and a plus, is skipped first.
func matchesHintReliantName(name []byte) bool {
	if len(name) >= 8 && name[6] == '+' {
		tagged := true
		for _, c := range name[:6] {
			tagged = tagged && c >= 'A' && c <= 'Z'
		}
		if tagged {
			name = name[7:]
		}
	}
	for _, tricky := range hintReliantNames {
		if bytes.Contains(name, []byte(tricky)) {
			return true
		}
	}
	return false
}
