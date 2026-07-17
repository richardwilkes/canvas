// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Typeface's surface (family name, font style, units-per-em, fixed pitch, text→glyph conversion), built over
// go-text/typesetting's font parser: the sfnt tables are read directly rather than through a platform text-rendering
// host, following the rules closest to raw table data. The deltas that follow from that are: rendering is always
// unhinted with linear metrics (hinting is recorded but only HintingNone/HintingSlight are honored, and
// forceAutoHinting/embeddedBitmaps are recorded but never honored — outline fonts only); font metrics follow the
// FreeType table recipe (hhea, or OS/2 typo metrics when fsSelection UseTypoMetrics is set, for ascent/descent/leading;
// the head bbox for top/bottom; post for underline; OS/2 for x-height, cap-height, avgCharWidth, and strikeout; 'x'/'H'
// control-box synthesis for missing heights), where a platform host such as CoreText sources several of those
// differently; glyph bounds are the outline control box (or the styled-path bounds when a stroke or path effect
// applies) mapped through the text matrix and rounded out; text→glyph conversion keeps per-char cmap semantics for
// every input; and the empty typeface has no glyphs, upem 0, fixed pitch, and normal style.

package font

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/font/opentype/tables"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// Typeface is an immutable font face. The zero value is not useful; use NewTypefaceFromData or EmptyTypeface. A nil
// *Typeface passed to NewFont is replaced by the empty typeface, which has no glyphs.
type Typeface struct {
	os2             *tables.Os2
	face            *tsfont.Face // nil only for the empty typeface
	post            *tables.Post
	hhea            *tables.Hhea
	familyName      string
	postScriptName  string // name ID 6 (FT_Get_Postscript_Name)
	data            []byte // the raw font-file bytes (openStream's asset)
	palette         []colorcore.Color
	head            tables.Head
	nGlyphs         int
	collectionIndex int // the TTC face index within data
	upem            int
	boundsVal       geom.Rect
	boundsOnce      sync.Once
	faceMu          sync.Mutex
	uniqueID        uint32
	style           Style
	sCapHgt         int16  // OS/2 v2+ sCapHeight, 0 if absent
	sxHght          int16  // OS/2 v2+ sxHeight, 0 if absent
	fsType          uint16 // OS/2 fsType (embedding permissions)
	italicAngle     int16  // floor of the post table's 16.16 italicAngle
	hasCOLR         bool
	hasBitmaps      bool
	fixedPitch      bool
	hasGlyf         bool // a 'glyf' table is present (TrueType outlines)
	hasCFF          bool // a 'CFF ' table is present (PostScript/CFF outlines)
}

// emptyTypeface is the singleton empty typeface: no glyphs, empty family name, normal style, fixed pitch.
var emptyTypeface = &Typeface{familyName: "", style: NormalStyle(), fixedPitch: true, uniqueID: nextTypefaceID()}

// EmptyTypeface returns the singleton typeface with no glyphs.
func EmptyTypeface() *Typeface { return emptyTypeface }

// typefaceIDCounter backs UniqueID.
var typefaceIDCounter atomic.Uint32

// nextTypefaceID returns a fresh nonzero typeface ID.
func nextTypefaceID() uint32 {
	for {
		if id := typefaceIDCounter.Add(1); id != 0 {
			return id
		}
	}
}

// UniqueID returns a process-unique nonzero ID identifying this typeface instance (strike keys and caches hang off it).
func (t *Typeface) UniqueID() uint32 { return t.uniqueID }

// faceHAdvance returns the design-space horizontal advance for gid under the face lock.
func (t *Typeface) faceHAdvance(gid opentype.GID) float32 {
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	return t.face.HorizontalAdvance(gid)
}

// faceGlyphExtents returns the glyph extents (control box) for gid under the face lock.
func (t *Typeface) faceGlyphExtents(gid opentype.GID) (tsfont.GlyphExtents, bool) {
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	return t.face.GlyphExtents(gid)
}

// faceGlyphData returns the glyph data (outline segments) for gid under the face lock.
func (t *Typeface) faceGlyphData(gid opentype.GID) tsfont.GlyphData {
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	return t.face.GlyphData(gid)
}

// faceGlyphOutline returns gid's raw 'glyf'/'CFF ' outline under the face lock, bypassing GlyphData's bitmap/SVG/COLR
// preference. The COLRv1 walker needs this to load the outlines of glyphs that are themselves COLR base glyphs.
func (t *Typeface) faceGlyphOutline(gid opentype.GID) (tsfont.GlyphOutline, bool) {
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	return t.face.GlyphDataOutline(tables.GlyphID(gid))
}

// faceNominalGlyph maps ch through the face's cmap under the face lock (the Face-level lookup caches).
func (t *Typeface) faceNominalGlyph(ch rune) (opentype.GID, bool) {
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	return t.face.NominalGlyph(ch)
}

// GlyphMaskNeedsCurrentColor reports whether the typeface has a COLR table: COLRv0 layers may paint with the current
// paint color (palette index 0xFFFF), so the foreground color must enter the scaler rec.
func (t *Typeface) GlyphMaskNeedsCurrentColor() bool { return t.hasCOLR }

// faceColorPaint returns the COLR paint for gid: a COLRv0 layer list (tables.PaintColrLayersResolved) or the root of a
// COLRv1 paint graph. typesetting's Search prefers a v1 BaseGlyphList record over a v0 baseGlyph record when both
// exist, as the spec directs ("give preference to the version 1 color glyph").
func (t *Typeface) faceColorPaint(gid opentype.GID) (tables.PaintTable, bool) {
	if !t.hasCOLR {
		return nil, false
	}
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	return t.face.COLR.Search(tables.GlyphID(gid))
}

// faceColorV0Layers returns the COLRv0 layer records for gid, or ok=false when the glyph has no v0 layers
// (COLRv1-preferred glyphs answer false here and render through the v1 walker instead).
func (t *Typeface) faceColorV0Layers(gid opentype.GID) ([]tables.Layer, bool) {
	paint, ok := t.faceColorPaint(gid)
	if !ok {
		return nil, false
	}
	layers, ok := paint.(tables.PaintColrLayersResolved)
	if !ok {
		return nil, false
	}
	return layers, true
}

// colrTable returns the parsed COLR table, or nil. The table is parsed eagerly at font load and immutable afterwards,
// so reads (LayerList/ClipList lookups in the COLRv1 walker) are safe without faceMu.
func (t *Typeface) colrTable() *tables.COLR1 {
	if !t.hasCOLR || t.face == nil {
		return nil
	}
	return t.face.COLR
}

// PaletteColor returns the CPAL palette-0 color for index, or the foreground color for the special index 0xFFFF (the
// layer paints with the current text color). ok=false for out-of-range indices (a malformed font; the layer is
// skipped).
func (t *Typeface) PaletteColor(index uint16, foreground colorcore.Color) (colorcore.Color, bool) {
	if index == 0xFFFF {
		return foreground, true
	}
	if int(index) >= len(t.palette) {
		return 0, false
	}
	return t.palette[index], true
}

// facePpem sets the face's ppem (invalidating typesetting's extents cache only on change) and must be called with
// faceMu held.
func (t *Typeface) facePpem(ppem uint16) {
	if x, y := t.face.Ppem(); x != ppem || y != ppem {
		t.face.SetPpem(ppem, ppem)
	}
}

// faceBitmapGlyph returns the bitmap strike glyph (sbix/CBDT/EBDT) and its font-unit extents for gid at the requested
// ppem (typesetting's chooseStrike picks the smallest strike >= ppem, else the largest — the same selection as the
// FreeType host's chooseBitmapStrike). ok=false when the face has no bitmap for the glyph.
func (t *Typeface) faceBitmapGlyph(gid opentype.GID, ppem uint16) (tsfont.GlyphBitmap, tsfont.GlyphExtents, bool) {
	if !t.hasBitmaps {
		return tsfont.GlyphBitmap{}, tsfont.GlyphExtents{}, false
	}
	t.faceMu.Lock()
	defer t.faceMu.Unlock()
	t.facePpem(ppem)
	bm, ok := t.face.GlyphDataBitmap(tables.GlyphID(gid))
	if !ok {
		return tsfont.GlyphBitmap{}, tsfont.GlyphExtents{}, false
	}
	ext, ok := t.face.GlyphExtents(gid)
	if !ok {
		return tsfont.GlyphBitmap{}, tsfont.GlyphExtents{}, false
	}
	return bm, ext, true
}

// Bounds returns the font-wide bounds at 1pt, computed from the font metrics at a big size (2048) and scaled back down,
// empty when the face reports no bounds.
func (t *Typeface) Bounds() geom.Rect {
	t.boundsOnce.Do(func() {
		if t.face == nil || t.upem <= 0 || t.hhea == nil {
			return
		}
		const textSize = 2048
		const invTextSize = 1.0 / textSize
		st := strike{t: t, size: textSize, scaleX: 1, frameWidth: -1}
		m := st.fontMetrics()
		t.boundsVal = geom.RectLTRB(
			m.XMin*invTextSize, m.Top*invTextSize, m.XMax*invTextSize, m.Bottom*invTextSize,
		)
	})
	return t.boundsVal
}

// NewTypefaceFromData parses an sfnt font (TTF/OTF/TTC) from data, selecting face index within a collection. It returns
// an error for unparsable data or an out-of-range index.
func NewTypefaceFromData(data []byte, index int) (*Typeface, error) {
	lds, err := opentype.NewLoaders(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("font: parse: %w", err)
	}
	if index < 0 || index >= len(lds) {
		return nil, fmt.Errorf("font: collection index %d out of range (%d faces)", index, len(lds))
	}
	ld := lds[index]
	f, err := tsfont.NewFont(ld)
	if err != nil {
		return nil, fmt.Errorf("font: parse: %w", err)
	}
	t := &Typeface{face: tsfont.NewFace(f), uniqueID: nextTypefaceID(), data: data, collectionIndex: index}

	// Font-outline format: classified by which outline table is present ('glyf' → TrueType, 'CFF ' → CFF). Only glyf
	// faces embed as CIDFontType2; everything else falls back to drawing glyphs as paths in the PDF backend.
	if raw, err2 := ld.RawTable(opentype.MustNewTag("glyf")); err2 == nil && len(raw) > 0 {
		t.hasGlyf = true
	}
	if raw, err2 := ld.RawTable(opentype.MustNewTag("CFF ")); err2 == nil && len(raw) > 0 {
		t.hasCFF = true
	}

	raw, err := ld.RawTable(opentype.MustNewTag("head"))
	if err != nil {
		return nil, errors.New("font: missing head table")
	}
	if t.head, _, err = tables.ParseHead(raw); err != nil {
		return nil, fmt.Errorf("font: head: %w", err)
	}
	t.upem = int(t.head.Upem())

	if raw, err = ld.RawTable(opentype.MustNewTag("maxp")); err == nil {
		if maxp, _, err2 := tables.ParseMaxp(raw); err2 == nil {
			t.nGlyphs = int(maxp.NumGlyphs)
		}
	}
	if raw, err = ld.RawTable(opentype.MustNewTag("hhea")); err == nil {
		if hhea, _, err2 := tables.ParseHhea(raw); err2 == nil {
			t.hhea = &hhea
		}
	}
	if raw, err = ld.RawTable(opentype.MustNewTag("post")); err == nil {
		if post, _, err2 := tables.ParsePost(raw); err2 == nil {
			t.post = &post
			t.fixedPitch = post.IsFixedPitch != 0
			// post.italicAngle is unexported; read it from the raw table (offset 4, a signed 16.16 Fixed) and floor it
			// to an int.
			if len(raw) >= 8 {
				t.italicAngle = int16(int32(binary.BigEndian.Uint32(raw[4:])) >> 16)
			}
		}
	}
	if raw, err = ld.RawTable(opentype.MustNewTag("OS/2")); err == nil {
		// os2.fSType is unexported; fsType lives at byte offset 8 (after version, xAvgCharWidth, usWeightClass,
		// usWidthClass), read directly for the PDF canEmbed check.
		if len(raw) >= 10 {
			t.fsType = binary.BigEndian.Uint16(raw[8:])
		}
		if os2, _, err2 := tables.ParseOs2(raw); err2 == nil {
			t.os2 = &os2
			// OS/2 v2+ appends (after usWinDescent): ulCodePageRange1/2 (8 bytes), sxHeight, sCapHeight, usDefaultChar,
			// usBreakChar, usMaxContext. typesetting keeps those in HigherVersionData.
			if os2.Version >= 2 && len(os2.HigherVersionData) >= 12 {
				t.sxHght = int16(binary.BigEndian.Uint16(os2.HigherVersionData[8:]))
				t.sCapHgt = int16(binary.BigEndian.Uint16(os2.HigherVersionData[10:]))
			}
		}
	}
	// Color-glyph tables: COLR presence is a nonempty-table check (typesetting only keeps the parsed table when it is
	// valid — same outcome for well-formed fonts); the CPAL palette-0 colors convert BGRA color records to Colors;
	// bitmap strikes come from BitmapSizes (sbix needs hhea, per typesetting's own gate).
	if raw, err = ld.RawTable(opentype.MustNewTag("COLR")); err == nil && len(raw) > 0 {
		t.hasCOLR = true
	}
	if cpal := t.face.CPAL; len(cpal) > 0 {
		t.palette = make([]colorcore.Color, len(cpal[0]))
		for i, rec := range cpal[0] {
			t.palette[i] = colorcore.ARGB(rec.Alpha, rec.Red, rec.Green, rec.Blue)
		}
	}
	t.hasBitmaps = len(t.face.BitmapSizes()) > 0

	if raw, err = ld.RawTable(opentype.MustNewTag("name")); err == nil {
		if name, _, err2 := tables.ParseName(raw); err2 == nil {
			// Prefer the typographic family (name ID 16) and fall back to the font family (name ID 1); tables.Name.Name
			// applies the standard record-selection rules (favor Windows English, then Mac English/Roman, then
			// Unicode).
			if t.familyName = name.Name(16); t.familyName == "" {
				t.familyName = name.Name(1)
			}
			t.postScriptName = name.Name(6) // the font's PostScript name
		}
	}
	t.style = computeStyle(t.os2, t.head)
	return t, nil
}

// computeStyle derives the Style from OS/2 (weight/width/slant), falling back to head.macStyle.
func computeStyle(os2 *tables.Os2, head tables.Head) Style {
	weight := WeightNormal
	width := WidthNormal
	slant := SlantUpright
	if os2 != nil && os2.Version != 0xFFFF {
		if os2.USWeightClass != 0 {
			weight = int(os2.USWeightClass)
		}
		if os2.USWidthClass != 0 {
			width = int(os2.USWidthClass)
		}
		if os2.FsSelection&0x01 != 0 { // ITALIC
			slant = SlantItalic
		}
		if os2.Version >= 4 && os2.FsSelection&0x200 != 0 { // OBLIQUE
			slant = SlantOblique
		}
	} else {
		if head.MacStyle&0x01 != 0 { // bold
			weight = WeightBold
		}
		if head.MacStyle&0x02 != 0 { // italic
			slant = SlantItalic
		}
	}
	return NewStyle(weight, width, slant)
}

// FamilyName returns the typeface's family name.
func (t *Typeface) FamilyName() string { return t.familyName }

// Style returns the typeface's font style.
func (t *Typeface) Style() Style { return t.style }

// UnitsPerEm returns the typeface's units-per-em (0 for the empty typeface).
func (t *Typeface) UnitsPerEm() int { return t.upem }

// IsFixedPitch reports whether the typeface is a fixed-pitch (monospace) font.
func (t *Typeface) IsFixedPitch() bool { return t.fixedPitch }

// CountGlyphs returns the number of glyphs in the typeface.
func (t *Typeface) CountGlyphs() int { return t.nGlyphs }

// UnicharToGlyph maps a single Unicode code point to a glyph ID, 0 when unmapped. Typeface's contract is thread-safety,
// so the lookup goes through the read-only Font cmap (safe for concurrent use), not the Face-level lookup, which
// memoizes into an unsynchronized per-Face cache (go-text documents Faces as NOT safe for concurrent use).
func (t *Typeface) UnicharToGlyph(unichar int32) uint16 {
	if t.face == nil || unichar < 0 || unichar > 0x10FFFF {
		return 0
	}
	gid, ok := t.face.Font.NominalGlyph(unichar)
	if !ok || gid > 0xFFFF {
		return 0
	}
	return uint16(gid)
}

// UnicharsToGlyphs maps each Unicode code point in unichars to a glyph ID.
func (t *Typeface) UnicharsToGlyphs(unichars []int32, glyphs []uint16) {
	n := min(len(unichars), len(glyphs))
	for i := range n {
		glyphs[i] = t.UnicharToGlyph(unichars[i])
	}
}

// TextToGlyphs decodes text (raw bytes in the given encoding, UTF-16/32 native little-endian) and maps each character
// to a glyph ID. It returns the number of glyphs the text represents; glyphs are written only when the slice is large
// enough. Invalid UTF-8/UTF-16 input returns -1.
func (t *Typeface) TextToGlyphs(text []byte, encoding TextEncoding, glyphs []uint16) int {
	if len(text) == 0 {
		return 0
	}
	count := CountTextElements(text, encoding)
	if count < 0 || count > len(glyphs) {
		return count
	}
	if encoding == TextEncodingGlyphID {
		for i := range count {
			glyphs[i] = uint16(text[2*i]) | uint16(text[2*i+1])<<8
		}
		return count
	}
	i := 0
	for g := range count {
		var uni int32
		switch encoding {
		case TextEncodingUTF8:
			uni, i = nextUTF8(text, i)
		case TextEncodingUTF16:
			uni, i = nextUTF16(text, i)
		default: // TextEncodingUTF32
			uni = int32(binary.LittleEndian.Uint32(text[i:]))
			i += 4
		}
		glyphs[g] = t.UnicharToGlyph(uni)
	}
	return count
}

// TextEncoding identifies the byte encoding of a text buffer passed to TextToGlyphs and friends.
type TextEncoding int32

// TextEncoding values.
const (
	TextEncodingUTF8 TextEncoding = iota
	TextEncodingUTF16
	TextEncodingUTF32
	TextEncodingGlyphID
)

// CountTextElements returns the number of text elements (unichars or glyph IDs) in text under encoding. For
// UTF-8/UTF-16 it returns -1 on invalid input (strict validation); UTF-32 and glyph IDs are counted by size alone.
func CountTextElements(text []byte, encoding TextEncoding) int {
	switch encoding {
	case TextEncodingUTF8:
		return countUTF8(text)
	case TextEncodingUTF16:
		return countUTF16(text)
	case TextEncodingUTF32:
		return len(text) >> 2
	case TextEncodingGlyphID:
		return len(text) >> 1
	}
	return 0
}
