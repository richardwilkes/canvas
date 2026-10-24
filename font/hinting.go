// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package font

import (
	"encoding/binary"
	"math"
	"sync"

	"github.com/richardwilkes/canvas/font/internal/tthint"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// Hinting: with HintingFull, the TrueType instructions a typeface carries are run by the tthint package, which
// reproduces FreeType's default interpreter: glyphs are fitted to the pixel grid vertically only, so advances and
// layout are unchanged. It applies only to a strike that is an upright uniform scale, a TrueType typeface with
// instructions that is not a variable font, and a size the font's own programs agree to run at. Anywhere else the
// unhinted outline is drawn.

// maxHintInstances bounds the per-typeface cache of prepared sizes, so that animating a font size cannot grow it
// without limit.
const maxHintInstances = 64

type hintKey struct {
	ppemBits uint32
	target   tthint.Target
}

// typefaceHinting is a Typeface's hinting state: the font's tables, read on first use, and the sizes prepared so far.
type typefaceHinting struct {
	font      *tthint.Font
	instances map[hintKey]*tthint.Instance
	once      sync.Once
	mu        sync.Mutex
}

// hintFont returns what the typeface offers for hinting, or nil if it cannot be hinted.
func (t *Typeface) hintFont() *tthint.Font {
	t.hinting.once.Do(func() {
		if !t.hasGlyf || t.hasFvar {
			return
		}
		table := sfntTables(t.data, t.collectionIndex)
		if table == nil {
			return
		}
		f, err := tthint.NewFont(table)
		if err != nil || !f.HasInstructions() || f.HasVariations() {
			return
		}
		t.hinting.font = f
	})
	return t.hinting.font
}

// hintInstance returns the typeface prepared for hinting at a size, or nil if it cannot be hinted there. Either answer
// is cached, since preparing a size runs the font's setup programs and the LCD lane has a strike per text color.
func (t *Typeface) hintInstance(ppem float32, target tthint.Target) *tthint.Instance {
	f := t.hintFont()
	if f == nil {
		return nil
	}
	key := hintKey{ppemBits: math.Float32bits(ppem), target: target}
	t.hinting.mu.Lock()
	defer t.hinting.mu.Unlock()
	if in, ok := t.hinting.instances[key]; ok {
		return in
	}
	in, err := f.NewInstance(ppem, target)
	if err != nil {
		in = nil
	}
	if len(t.hinting.instances) >= maxHintInstances {
		clear(t.hinting.instances)
	}
	if t.hinting.instances == nil {
		t.hinting.instances = make(map[hintKey]*tthint.Instance)
	}
	t.hinting.instances[key] = in
	return in
}

// sfntTables returns a lookup of the tables of face index in sfnt data, as views into data, or nil if data is not a
// plain sfnt (a WOFF, say, whose tables are compressed).
func sfntTables(data []byte, index int) func(tag string) []byte {
	if len(data) < 12 {
		return nil
	}
	offset := 0
	if string(data[:4]) == "ttcf" {
		count := int(binary.BigEndian.Uint32(data[8:]))
		if index < 0 || index >= count || len(data) < 12+4*(index+1) {
			return nil
		}
		offset = int(binary.BigEndian.Uint32(data[12+4*index:]))
		if offset < 0 || offset+12 > len(data) {
			return nil
		}
	}
	switch binary.BigEndian.Uint32(data[offset:]) {
	case 0x00010000, 0x74727565: // TrueType outlines: version 1.0, or Apple's 'true'
	default:
		return nil
	}
	count := int(binary.BigEndian.Uint16(data[offset+4:]))
	directory := data[offset+12:]
	if len(directory) < 16*count {
		return nil
	}
	tables := make(map[string][]byte, count)
	for i := range count {
		record := directory[16*i:]
		start := uint64(binary.BigEndian.Uint32(record[8:]))
		length := uint64(binary.BigEndian.Uint32(record[12:]))
		if start+length > uint64(len(data)) {
			continue
		}
		tables[string(record[:4])] = data[start : start+length : start+length]
	}
	return func(tag string) []byte { return tables[tag] }
}

// hinterFor returns the prepared size a context hints with, or nil if it draws unhinted outlines.
func hinterFor(typeface *Typeface, rec *ScalerRec, single *geom.Matrix) *tthint.Instance {
	if rec.Flags&recFlagHinted == 0 {
		return nil
	}
	sx, sy := single.Get(geom.MScaleX), single.Get(geom.MScaleY)
	if single.Get(geom.MSkewX) != 0 || single.Get(geom.MSkewY) != 0 || sx != sy || !(sy > 0) || single.HasPerspective() {
		return nil
	}
	target := tthint.TargetNormal
	if rec.Format == MaskLCD16 {
		target = tthint.TargetLCD
		if rec.Flags&recFlagLCDVertical != 0 {
			target = tthint.TargetVerticalLCD
		}
	}
	return typeface.hintInstance(sy, target)
}

var hintOutlinePool = sync.Pool{New: func() any { return new(tthint.Outline) }}

// hintedOutline returns gid's hinted outline, to be handed back to hintOutlinePool, or nil if the glyph is not hinted:
// the context does not hint, the glyph has no outline, or the font describes it wrongly.
func (c *ScalerContext) hintedOutline(gid uint16) *tthint.Outline {
	if c.hinter == nil {
		return nil
	}
	outline, _ := hintOutlinePool.Get().(*tthint.Outline)
	if !c.hinter.Outline(gid, outline) || len(outline.Points) == 0 {
		hintOutlinePool.Put(outline)
		return nil
	}
	return outline
}

// hintedPoint converts a hinted point, in 26.6 pixels with y up, to device space.
func hintedPoint(p tthint.Point) geom.Point {
	return geom.Pt(float32(p.X)/64, -float32(p.Y)/64)
}

// hintedPath builds the device-space path of gid's hinted outline, or returns nil if the glyph is not hinted.
func (c *ScalerContext) hintedPath(gid uint16) *path.Path {
	outline := c.hintedOutline(gid)
	if outline == nil {
		return nil
	}
	defer hintOutlinePool.Put(outline)
	p := path.New()
	start := 0
	for _, end := range outline.ContourEnds {
		stop := int(end) + 1
		if stop > len(outline.Points) {
			break
		}
		if stop-start >= 2 {
			appendQuadraticContour(p, outline.Points[start:stop], outline.OnCurve[start:stop])
		}
		start = stop
	}
	return p
}

// appendQuadraticContour adds one closed TrueType contour to p. Consecutive off-curve points imply an on-curve point
// midway between them; a contour that begins off the curve starts from its last point, or from the implied point
// between the two when that is off the curve too.
func appendQuadraticContour(p *path.Path, points []tthint.Point, onCurve []bool) {
	n := len(points)
	mid := func(a, b geom.Point) geom.Point { return geom.Pt((a.X+b.X)/2, (a.Y+b.Y)/2) }
	var start geom.Point
	first := 0
	switch {
	case onCurve[0]:
		start = hintedPoint(points[0])
		first = 1
	case onCurve[n-1]:
		n--
		start = hintedPoint(points[n])
	default:
		start = mid(hintedPoint(points[0]), hintedPoint(points[n-1]))
	}
	p.MoveToPt(start)
	var control geom.Point
	haveControl := false
	for i := first; i < n; i++ {
		pt := hintedPoint(points[i])
		switch {
		case !onCurve[i]:
			if haveControl {
				p.QuadToPt(control, mid(control, pt))
			}
			control = pt
			haveControl = true
		case haveControl:
			p.QuadToPt(control, pt)
			haveControl = false
		default:
			p.LineToPt(pt)
		}
	}
	if haveControl {
		p.QuadToPt(control, start)
	}
	p.Close()
}

// hintedBounds returns the control box of gid's hinted outline in device space, ok false if the glyph is not hinted.
// Hinting moves edges by up to a pixel, so the design extents would clip the mask.
func (c *ScalerContext) hintedBounds(gid uint16) (bounds geom.Rect, ok bool) {
	outline := c.hintedOutline(gid)
	if outline == nil {
		return geom.Rect{}, false
	}
	defer hintOutlinePool.Put(outline)
	first := outline.Points[0]
	xMin, xMax, yMin, yMax := first.X, first.X, first.Y, first.Y
	for _, p := range outline.Points[1:] {
		xMin = min(xMin, p.X)
		xMax = max(xMax, p.X)
		yMin = min(yMin, p.Y)
		yMax = max(yMax, p.Y)
	}
	return geom.RectLTRB(float32(xMin)/64, -float32(yMax)/64, float32(xMax)/64, -float32(yMin)/64), true
}
