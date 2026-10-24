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
	"encoding/binary"
	"errors"
)

// Composite glyph component flags.
const (
	compArgsAreWords      = 0x0001
	compArgsAreXYValues   = 0x0002
	compRoundXYToGrid     = 0x0004
	compHaveScale         = 0x0008
	compMoreComponents    = 0x0020
	compHaveXYScale       = 0x0040
	compHaveTwoByTwo      = 0x0080
	compHaveInstructions  = 0x0100
	compUseMyMetrics      = 0x0200
	compScaledOffset      = 0x0800
	compUnscaledOffset    = 0x1000
	compHaveTransformMask = compHaveScale | compHaveXYScale | compHaveTwoByTwo
)

// Limits on what a glyph may ask for, so that a malformed font cannot make loading one unbounded.
const (
	maxRecursionDepth = 64
	maxOutlinePoints  = 1 << 16
	maxComponents     = 1 << 12
)

// Font is the part of a TrueType font that hinting needs. It is immutable and safe for concurrent use.
type Font struct {
	glyf, loca, hmtx []byte
	fpgm, prep       []byte
	cvt              []int16
	numGlyphs        int
	numHMetrics      int
	unitsPerEm       uint16
	maxFunctionDefs  uint16
	maxInstrDefs     uint16
	maxTwilight      uint16
	maxStack         uint16
	maxStorage       uint16
	typoAscender     int16
	typoDescender    int16
	longLoca         bool
	forceIntegerPPEM bool
	hasVariations    bool
	hasInstructions  bool
}

// ErrNoOutlines is returned by NewFont for a font that has no TrueType outlines to hint.
var ErrNoOutlines = errors.New("font has no TrueType outlines")

// NewFont reads what hinting needs from a font's tables. table returns the content of the table with the given
// four-character tag, or nil if the font has none; the returned slices are retained and must not be modified.
func NewFont(table func(tag string) []byte) (*Font, error) {
	head := table("head")
	maxp := table("maxp")
	f := &Font{
		glyf: table("glyf"),
		loca: table("loca"),
		hmtx: table("hmtx"),
		fpgm: table("fpgm"),
		prep: table("prep"),
	}
	if len(head) < 54 || len(maxp) < 6 || len(f.glyf) == 0 || len(f.loca) == 0 {
		return nil, ErrNoOutlines
	}
	f.unitsPerEm = binary.BigEndian.Uint16(head[18:])
	if f.unitsPerEm == 0 {
		return nil, ErrNoOutlines
	}
	// Bit 3 of the head flags asks for ppem to be rounded to an integer before scaling.
	f.forceIntegerPPEM = binary.BigEndian.Uint16(head[16:])&(1<<3) != 0
	f.longLoca = binary.BigEndian.Uint16(head[50:]) == 1
	f.numGlyphs = int(binary.BigEndian.Uint16(maxp[4:]))
	var maxInstructions uint16
	if len(maxp) >= 32 {
		f.maxTwilight = binary.BigEndian.Uint16(maxp[16:])
		f.maxStorage = binary.BigEndian.Uint16(maxp[18:])
		f.maxFunctionDefs = binary.BigEndian.Uint16(maxp[20:])
		f.maxInstrDefs = binary.BigEndian.Uint16(maxp[22:])
		f.maxStack = binary.BigEndian.Uint16(maxp[24:])
		maxInstructions = binary.BigEndian.Uint16(maxp[26:])
	}
	// Room for the four phantom points in the twilight zone, and slack on the stack, as FreeType allows.
	f.maxTwilight = uint16(min(int(f.maxTwilight)+4, 0xFFFF))
	f.maxStack = uint16(min(int(f.maxStack)+32, 0xFFFF))
	if hhea := table("hhea"); len(hhea) >= 36 {
		f.numHMetrics = int(binary.BigEndian.Uint16(hhea[34:]))
	}
	if os2 := table("OS/2"); len(os2) >= 72 {
		f.typoAscender = int16(binary.BigEndian.Uint16(os2[68:]))
		f.typoDescender = int16(binary.BigEndian.Uint16(os2[70:]))
	}
	if cvt := table("cvt "); len(cvt) >= 2 {
		f.cvt = make([]int16, len(cvt)/2)
		for i := range f.cvt {
			f.cvt[i] = int16(binary.BigEndian.Uint16(cvt[i*2:]))
		}
	}
	f.hasVariations = len(table("gvar")) != 0
	f.hasInstructions = maxInstructions != 0 || len(f.fpgm) != 0 || len(f.prep) != 0
	return f, nil
}

// HasInstructions reports whether the font carries any instructions at all. One that does not is left for whatever
// the caller does with unhinted fonts.
func (f *Font) HasInstructions() bool { return f.hasInstructions }

// HasVariations reports whether the font is a variable font. Only its default instance can be hinted here.
func (f *Font) HasVariations() bool { return f.hasVariations }

// UnitsPerEm returns the size of the font's design grid.
func (f *Font) UnitsPerEm() int { return int(f.unitsPerEm) }

// hMetrics returns a glyph's advance width and left side bearing in font units.
func (f *Font) hMetrics(gid int) (advance, lsb int32) {
	if f.numHMetrics == 0 || gid < 0 {
		return 0, 0
	}
	if gid < f.numHMetrics {
		if off := gid * 4; off+4 <= len(f.hmtx) {
			advance = int32(binary.BigEndian.Uint16(f.hmtx[off:]))
			lsb = int32(int16(binary.BigEndian.Uint16(f.hmtx[off+2:])))
		}
		return advance, lsb
	}
	if off := (f.numHMetrics - 1) * 4; off+2 <= len(f.hmtx) {
		advance = int32(binary.BigEndian.Uint16(f.hmtx[off:]))
	}
	if off := f.numHMetrics*4 + (gid-f.numHMetrics)*2; off+2 <= len(f.hmtx) {
		lsb = int32(int16(binary.BigEndian.Uint16(f.hmtx[off:])))
	}
	return advance, lsb
}

// glyphData returns the bytes of a glyph's record in glyf: empty for a glyph with no outline, and ok false for one
// the font does not have or describes wrongly.
func (f *Font) glyphData(gid int) (data []byte, ok bool) {
	if gid < 0 || gid >= f.numGlyphs {
		return nil, false
	}
	var start, end int
	if f.longLoca {
		if (gid+2)*4 > len(f.loca) {
			return nil, false
		}
		start = int(binary.BigEndian.Uint32(f.loca[gid*4:]))
		end = int(binary.BigEndian.Uint32(f.loca[gid*4+4:]))
	} else {
		if (gid+2)*2 > len(f.loca) {
			return nil, false
		}
		start = int(binary.BigEndian.Uint16(f.loca[gid*2:])) * 2
		end = int(binary.BigEndian.Uint16(f.loca[gid*2+2:])) * 2
	}
	if start > end || end > len(f.glyf) {
		return nil, false
	}
	return f.glyf[start:end], true
}

// glyphHeader is the fixed start of a glyph record.
type glyphHeader struct {
	contours               int
	xMin, yMin, xMax, yMax int16
}

func parseGlyphHeader(data []byte) (h glyphHeader, body []byte, ok bool) {
	if len(data) < 10 {
		return h, nil, false
	}
	h.contours = int(int16(binary.BigEndian.Uint16(data)))
	h.xMin = int16(binary.BigEndian.Uint16(data[2:]))
	h.yMin = int16(binary.BigEndian.Uint16(data[4:]))
	h.xMax = int16(binary.BigEndian.Uint16(data[6:]))
	h.yMax = int16(binary.BigEndian.Uint16(data[8:]))
	return h, data[10:], true
}

// simpleGlyph is a decoded simple glyph: points in font units, absolute.
type simpleGlyph struct {
	endPts       []uint16
	instructions []byte
	points       []Point
	onCurve      []bool
}

// parseSimpleGlyph decodes the body of a simple glyph with the given number of contours into g, reusing its slices.
func parseSimpleGlyph(body []byte, contours int, g *simpleGlyph) bool {
	g.endPts = g.endPts[:0]
	g.points = g.points[:0]
	g.onCurve = g.onCurve[:0]
	g.instructions = nil
	if len(body) < contours*2+2 {
		return false
	}
	last := -1
	for i := range contours {
		end := int(binary.BigEndian.Uint16(body[i*2:]))
		if end < last {
			return false
		}
		last = end
		g.endPts = append(g.endPts, uint16(end))
	}
	body = body[contours*2:]
	insLen := int(binary.BigEndian.Uint16(body))
	body = body[2:]
	if len(body) < insLen {
		return false
	}
	g.instructions = body[:insLen]
	body = body[insLen:]
	count := last + 1
	if count > maxOutlinePoints {
		return false
	}
	// Flags, with runs compressed by the repeat bit.
	const (
		onCurve   = 0x01
		xShort    = 0x02
		yShort    = 0x04
		repeat    = 0x08
		xSameOrPo = 0x10
		ySameOrPo = 0x20
	)
	flags := make([]uint8, 0, count)
	for len(flags) < count {
		if len(body) == 0 {
			return false
		}
		flag := body[0]
		body = body[1:]
		flags = append(flags, flag)
		if flag&repeat != 0 {
			if len(body) == 0 {
				return false
			}
			n := int(body[0])
			body = body[1:]
			for ; n > 0 && len(flags) < count; n-- {
				flags = append(flags, flag)
			}
		}
	}
	var x int32
	for _, flag := range flags {
		switch {
		case flag&xShort != 0:
			if len(body) < 1 {
				return false
			}
			d := int32(body[0])
			body = body[1:]
			if flag&xSameOrPo == 0 {
				d = -d
			}
			x += d
		case flag&xSameOrPo == 0:
			if len(body) < 2 {
				return false
			}
			x += int32(int16(binary.BigEndian.Uint16(body)))
			body = body[2:]
		}
		g.points = append(g.points, Point{X: x})
		g.onCurve = append(g.onCurve, flag&onCurve != 0)
	}
	var y int32
	for i, flag := range flags {
		switch {
		case flag&yShort != 0:
			if len(body) < 1 {
				return false
			}
			d := int32(body[0])
			body = body[1:]
			if flag&ySameOrPo == 0 {
				d = -d
			}
			y += d
		case flag&ySameOrPo == 0:
			if len(body) < 2 {
				return false
			}
			y += int32(int16(binary.BigEndian.Uint16(body)))
			body = body[2:]
		}
		g.points[i].Y = y
	}
	return true
}

// component is one part of a composite glyph.
type component struct {
	glyph int
	flags uint16
	// arg1 and arg2 are an offset in font units, or a pair of point indices: base, then component.
	arg1, arg2 int32
	// The transform, in 2.14.
	xx, yx, xy, yy int16
}

// parseCompositeGlyph decodes the body of a composite glyph.
func parseCompositeGlyph(body []byte) (components []component, instructions []byte, ok bool) {
	for {
		if len(body) < 4 {
			return nil, nil, false
		}
		c := component{
			flags: binary.BigEndian.Uint16(body),
			glyph: int(binary.BigEndian.Uint16(body[2:])),
			xx:    0x4000,
			yy:    0x4000,
		}
		body = body[4:]
		if c.flags&compArgsAreWords != 0 {
			if len(body) < 4 {
				return nil, nil, false
			}
			if c.flags&compArgsAreXYValues != 0 {
				c.arg1 = int32(int16(binary.BigEndian.Uint16(body)))
				c.arg2 = int32(int16(binary.BigEndian.Uint16(body[2:])))
			} else {
				c.arg1 = int32(binary.BigEndian.Uint16(body))
				c.arg2 = int32(binary.BigEndian.Uint16(body[2:]))
			}
			body = body[4:]
		} else {
			if len(body) < 2 {
				return nil, nil, false
			}
			if c.flags&compArgsAreXYValues != 0 {
				c.arg1 = int32(int8(body[0]))
				c.arg2 = int32(int8(body[1]))
			} else {
				c.arg1 = int32(body[0])
				c.arg2 = int32(body[1])
			}
			body = body[2:]
		}
		switch {
		case c.flags&compHaveScale != 0:
			if len(body) < 2 {
				return nil, nil, false
			}
			c.xx = int16(binary.BigEndian.Uint16(body))
			c.yy = c.xx
			body = body[2:]
		case c.flags&compHaveXYScale != 0:
			if len(body) < 4 {
				return nil, nil, false
			}
			c.xx = int16(binary.BigEndian.Uint16(body))
			c.yy = int16(binary.BigEndian.Uint16(body[2:]))
			body = body[4:]
		case c.flags&compHaveTwoByTwo != 0:
			if len(body) < 8 {
				return nil, nil, false
			}
			c.xx = int16(binary.BigEndian.Uint16(body))
			c.yx = int16(binary.BigEndian.Uint16(body[2:]))
			c.xy = int16(binary.BigEndian.Uint16(body[4:]))
			c.yy = int16(binary.BigEndian.Uint16(body[6:]))
			body = body[8:]
		}
		components = append(components, c)
		if len(components) > maxComponents {
			return nil, nil, false
		}
		if c.flags&compMoreComponents == 0 {
			if c.flags&compHaveInstructions != 0 && len(body) >= 2 {
				n := int(binary.BigEndian.Uint16(body))
				if len(body) >= 2+n {
					instructions = body[2 : 2+n]
				}
			}
			return components, instructions, true
		}
	}
}
