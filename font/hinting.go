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

// Hinting: with HintingFull, the TrueType instructions a typeface carries are run by the tthint package, a port of
// Skrifa's interpreter: glyphs are fitted to the pixel grid vertically only. It applies only to a strike that is an
// upright uniform scale, a TrueType typeface with instructions that is not a variable font, and a size the font's own
// programs agree to run at. Anywhere else the unhinted outline is drawn.
//
// Advances follow Skia's FreeType and Fontations backends. With linear metrics (Font.SetLinearMetrics) a hinted glyph
// keeps its design advance at the requested size, so layout is exactly what it is unhinted. Without them, which is the
// default, its advance is the one the hinter reports, rounded to a whole pixel (hintedAdvance). The two differ most
// for a font that asks for whole sizes, as nearly all hinted fonts do: at 12.5 pixels its glyphs are fitted for 13, and
// only the hinted advance is spaced for 13 too.

// maxHintInstances bounds the per-typeface cache of prepared sizes, so that animating a font size cannot grow it
// without limit.
const maxHintInstances = 64

type hintKey struct {
	ppemBits uint32
	target   tthint.Target
}

// maxHintedBoxes bounds the control boxes remembered for one prepared size.
const maxHintedBoxes = 4096

// typefaceHinting is a Typeface's hinting state: the font's tables, read on first use, and the sizes prepared so far.
type typefaceHinting struct {
	font  *tthint.Font
	sizes map[hintKey]*hintedSize
	once  sync.Once
	mu    sync.Mutex
}

// hintedSize is a typeface prepared for hinting at one size, with the control boxes of the glyphs hinted so far. The
// boxes are kept here, not with a strike, because the LCD lane has a strike per text color and measuring has none: all
// of them want the same boxes.
type hintedSize struct {
	instance *tthint.Instance
	boxes    map[uint16]hintedBox
	mu       sync.Mutex
}

// hintedBox is what is kept of a hinted glyph: the control box of its outline and its advance, in 26.6 pixels, y up.
// loaded is false for a glyph the font describes wrongly, which has neither. hinted is false for that glyph and for
// one with no outline, which has an advance and no box.
type hintedBox struct {
	xMin, yMin, xMax, yMax int32
	advance                int32
	loaded                 bool
	hinted                 bool
}

// boxOf returns the box and advance of an outline, which is nil for a glyph the font describes wrongly.
func boxOf(outline *tthint.Outline) hintedBox {
	if outline == nil {
		return hintedBox{}
	}
	if len(outline.Points) == 0 {
		return hintedBox{advance: outline.Advance, loaded: true}
	}
	first := outline.Points[0]
	box := hintedBox{
		xMin: first.X, yMin: first.Y, xMax: first.X, yMax: first.Y,
		advance: outline.Advance, loaded: true, hinted: true,
	}
	for _, p := range outline.Points[1:] {
		box.xMin = min(box.xMin, p.X)
		box.xMax = max(box.xMax, p.X)
		box.yMin = min(box.yMin, p.Y)
		box.yMax = max(box.yMax, p.Y)
	}
	return box
}

// hintedAdvance returns the advance to use in place of the linear one, in pixels, and whether there is one: the
// hinter's, rounded to a whole pixel as FreeType rounds a hinted advance and as Skia rounds Skrifa's. An advance that
// rounds to nothing is not used.
func (b hintedBox) hintedAdvance() (advance float32, ok bool) {
	if !b.loaded {
		return 0, false
	}
	pixels := (b.advance + 32) >> 6
	return float32(pixels), pixels != 0
}

// rect converts the box to pixels with y down.
func (b hintedBox) rect() geom.Rect {
	return geom.RectLTRB(float32(b.xMin)/64, -float32(b.yMax)/64, float32(b.xMax)/64, -float32(b.yMin)/64)
}

// box returns gid's control box and advance, computing it the first time it is asked for. compute runs
// outside the lock, so two callers may both hint a glyph nobody has yet; they get the same answer.
func (h *hintedSize) box(gid uint16, compute func() hintedBox) hintedBox {
	h.mu.Lock()
	box, ok := h.boxes[gid]
	h.mu.Unlock()
	if ok {
		return box
	}
	box = compute()
	h.mu.Lock()
	if len(h.boxes) >= maxHintedBoxes {
		clear(h.boxes)
	}
	if h.boxes == nil {
		h.boxes = make(map[uint16]hintedBox)
	}
	h.boxes[gid] = box
	h.mu.Unlock()
	return box
}

// measuredBox returns gid's control box and advance for a caller with no scaler context of its own to hint with.
func (h *hintedSize) measuredBox(gid uint16) hintedBox {
	return h.box(gid, func() hintedBox {
		outline, _ := hintOutlinePool.Get().(*tthint.Outline)
		defer hintOutlinePool.Put(outline)
		if !h.instance.Outline(gid, outline) {
			return hintedBox{}
		}
		return boxOf(outline)
	})
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

// hintedSize returns the typeface prepared for hinting at a size, or nil if it cannot be hinted there. Either answer is
// cached, since preparing a size runs the font's setup programs and the LCD lane has a strike per text color.
//
// The programs run outside the lock, so that preparing a new size on one goroutine does not hold up lookups of the
// sizes already prepared on the others. Two goroutines may then prepare the same new size; the first to finish is kept.
func (t *Typeface) hintedSize(ppem float32, target tthint.Target) *hintedSize {
	f := t.hintFont()
	if f == nil {
		return nil
	}
	key := hintKey{ppemBits: math.Float32bits(ppem), target: target}
	t.hinting.mu.Lock()
	size, ok := t.hinting.sizes[key]
	t.hinting.mu.Unlock()
	if ok {
		return size
	}
	if in, err := f.NewInstance(ppem, target); err == nil {
		size = &hintedSize{instance: in}
	}
	t.hinting.mu.Lock()
	defer t.hinting.mu.Unlock()
	if existing, found := t.hinting.sizes[key]; found {
		return existing
	}
	if len(t.hinting.sizes) >= maxHintInstances {
		clear(t.hinting.sizes)
	}
	if t.hinting.sizes == nil {
		t.hinting.sizes = make(map[hintKey]*hintedSize)
	}
	t.hinting.sizes[key] = size
	return size
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
func hinterFor(typeface *Typeface, rec *ScalerRec, single *geom.Matrix) *hintedSize {
	if rec.Flags&recFlagHinted == 0 {
		return nil
	}
	return hintedSizeFor(typeface, rec, single)
}

// hintedSizeFor returns the prepared size a rec with the given single matrix would hint with if it asked for hinting,
// or nil if it cannot be hinted: the strike is not an upright uniform scale, or the typeface is not hinted at the size.
func hintedSizeFor(typeface *Typeface, rec *ScalerRec, single *geom.Matrix) *hintedSize {
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
	return typeface.hintedSize(sy, target)
}

var hintOutlinePool = sync.Pool{New: func() any { return new(tthint.Outline) }}

// maxPendingOutlines bounds the outlines a context keeps for glyphs whose paths have not been asked for.
const maxPendingOutlines = 256

// pendingOutlines are the outlines a context has hinted for glyphs it has not drawn yet. A glyph is asked for its
// bounds and later for its path, usually with the bounds of the rest of its run in between, and keeping the outline
// until then is what lets its program run once. nil stands for a glyph that is not hinted.
type pendingOutlines struct {
	outlines map[uint16]*tthint.Outline
	mu       sync.Mutex
}

// hintedOutline returns gid's hinted outline, which has no points if the glyph has no outline, or nil if the font
// describes the glyph wrongly. The caller holds c.outlines.mu, and until it lets go the outline is its to read.
func (c *ScalerContext) hintedOutline(gid uint16) *tthint.Outline {
	pending := &c.outlines
	if outline, ok := pending.outlines[gid]; ok {
		return outline
	}
	outline, _ := hintOutlinePool.Get().(*tthint.Outline)
	if !c.hinter.instance.Outline(gid, outline) {
		hintOutlinePool.Put(outline)
		outline = nil
	}
	if len(pending.outlines) >= maxPendingOutlines {
		// Glyphs that were measured and never drawn.
		for _, old := range pending.outlines {
			if old != nil {
				hintOutlinePool.Put(old)
			}
		}
		clear(pending.outlines)
	}
	if pending.outlines == nil {
		pending.outlines = make(map[uint16]*tthint.Outline)
	}
	pending.outlines[gid] = outline
	return outline
}

// releaseHintedOutline gives up gid's outline once its path has been built from it. The caller holds c.outlines.mu.
func (c *ScalerContext) releaseHintedOutline(gid uint16) {
	if outline, ok := c.outlines.outlines[gid]; ok {
		delete(c.outlines.outlines, gid)
		if outline != nil {
			hintOutlinePool.Put(outline)
		}
	}
}

// hintedBox returns gid's control box and advance, the zero value if the context does not hint.
func (c *ScalerContext) hintedBox(gid uint16) hintedBox {
	if c.hinter == nil {
		return hintedBox{}
	}
	c.outlines.mu.Lock()
	defer c.outlines.mu.Unlock()
	return c.hinter.box(gid, func() hintedBox { return boxOf(c.hintedOutline(gid)) })
}

// hintedPoint converts a hinted point, in 26.6 pixels with y up, to device space.
func hintedPoint(p tthint.Point) geom.Point {
	return geom.Pt(float32(p.X)/64, -float32(p.Y)/64)
}

// hintedPath builds the device-space path of gid's hinted outline, or returns nil if the glyph is not hinted.
func (c *ScalerContext) hintedPath(gid uint16) *path.Path {
	if c.hinter == nil {
		return nil
	}
	c.outlines.mu.Lock()
	defer c.outlines.mu.Unlock()
	defer c.releaseHintedOutline(gid)
	outline := c.hintedOutline(gid)
	if outline == nil || len(outline.Points) == 0 {
		return nil
	}
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
	box := c.hintedBox(gid)
	if !box.hinted {
		return geom.Rect{}, false
	}
	return box.rect(), true
}
