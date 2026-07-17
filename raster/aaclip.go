// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// AAClip: a run-length-encoded 8-bit-coverage clip — the antialiased counterpart of Region. Rows are packed [count,
// alpha] byte pairs spanning exactly the bounds width; identical adjacent rows are collapsed through per-row YOffset
// entries whose y is the last row covered. The head is an immutable-once-built struct shared by pointer; only freshly
// constructed heads (inside Builder.finish and setRegion) are ever mutated.

package raster

import (
	"bytes"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// ClipOp is the set of ops allowed for canvas/raster clip modification.
type ClipOp uint8

// ClipOp values.
const (
	ClipDifference ClipOp = iota
	ClipIntersect
)

// aaYOffset is one row record: y is the (bounds-relative) last scanline this row covers; offset is the byte offset of
// the row's run data.
type aaYOffset struct {
	y      int32
	offset uint32
}

// aaRunHead holds the run-length-encoded row data backing an AAClip.
type aaRunHead struct {
	yOffsets []aaYOffset
	data     []uint8
}

// aaComputeRowSizeForWidth returns the byte size of a full-width run row: 2 bytes per segment, where each segment can
// store up to 255 for count.
func aaComputeRowSizeForWidth(width int32) int {
	segments := 0
	for width > 0 {
		segments++
		width -= min(width, 255)
	}
	return segments * 2
}

// aaAllocRectRunHead builds a single-row run head covering bounds at full (0xFF) coverage.
func aaAllocRectRunHead(bounds geom.IRect) *aaRunHead {
	width := bounds.Width()
	head := &aaRunHead{
		yOffsets: []aaYOffset{{y: bounds.Height() - 1, offset: 0}},
		data:     make([]uint8, 0, aaComputeRowSizeForWidth(width)),
	}
	for width > 0 {
		n := min(width, 255)
		head.data = append(head.data, uint8(n), 0xFF)
		width -= n
	}
	return head
}

// AAClip is an antialiased run-length-encoded coverage clip. The zero value is an empty clip.
type AAClip struct {
	head   *aaRunHead // nil == empty
	bounds geom.IRect
}

// IsEmpty reports whether the clip contains no pixels.
func (c *AAClip) IsEmpty() bool { return c.head == nil }

// Bounds returns the clip's bounding rect.
func (c *AAClip) Bounds() geom.IRect { return c.bounds }

// SetEmpty clears the clip to empty and returns false.
func (c *AAClip) SetEmpty() bool {
	c.head = nil
	c.bounds = geom.IRect{}
	return false
}

// SetRect sets the clip to a hard-edged rect with full coverage.
func (c *AAClip) SetRect(bounds geom.IRect) bool {
	if bounds.IsEmpty() {
		return c.SetEmpty()
	}
	c.bounds = bounds
	c.head = aaAllocRectRunHead(bounds)
	return true
}

// IsRect returns true iff the clip is not empty, and is just a hard-edged rect (no partial alpha). If true, Bounds()
// can be used in place of this clip.
func (c *AAClip) IsRect() bool {
	if c.IsEmpty() {
		return false
	}
	if len(c.head.yOffsets) != 1 {
		return false
	}
	yoff := &c.head.yOffsets[0]
	if yoff.y != c.bounds.Bottom-c.bounds.Top-1 {
		return false
	}
	row := c.head.data[yoff.offset:]
	width := c.bounds.Width()
	for width > 0 {
		if row[1] != 0xFF {
			return false
		}
		width -= int32(row[0])
		row = row[2:]
	}
	return true
}

// SetRegion sets the clip to match a Region's coverage (full alpha inside, none outside).
func (c *AAClip) SetRegion(rgn *Region) bool {
	if rgn.IsEmpty() {
		return c.SetEmpty()
	}
	if rgn.IsRect() {
		return c.SetRect(rgn.Bounds())
	}

	bounds := rgn.Bounds()
	offsetX := bounds.Left
	offsetY := bounds.Top

	var yArray []aaYOffset
	var xArray []uint8

	appendXRun := func(value uint8, count int32) {
		for count > 0 {
			n := min(count, 255)
			xArray = append(xArray, uint8(n), value)
			count -= n
		}
	}

	prevRight := int32(0)
	prevBot := int32(0)
	haveCurrY := false

	for it := NewRegionIterator(rgn); !it.Done(); it.Next() {
		r := it.Rect()
		bot := r.Bottom - offsetY
		if bot > prevBot {
			if haveCurrY {
				// flush current row
				appendXRun(0, bounds.Width()-prevRight)
			}
			// did we introduce an empty-gap from the prev row?
			top := r.Top - offsetY
			if top > prevBot {
				yArray = append(yArray, aaYOffset{y: top - 1, offset: uint32(len(xArray))})
				appendXRun(0, bounds.Width())
			}
			// create a new record for this Y value
			yArray = append(yArray, aaYOffset{y: bot - 1, offset: uint32(len(xArray))})
			haveCurrY = true
			prevRight = 0
			prevBot = bot
		}

		x := r.Left - offsetX
		appendXRun(0, x-prevRight)

		w := r.Right - r.Left
		appendXRun(0xFF, w)
		prevRight = x + w
	}
	// flush last row
	appendXRun(0, bounds.Width()-prevRight)

	c.bounds = bounds
	c.head = &aaRunHead{yOffsets: yArray, data: xArray}
	return true
}

// SetPath sets the clip to the path's coverage restricted to clip.
func (c *AAClip) SetPath(p *path.Path, clip geom.IRect, doAA bool) bool {
	if clip.IsEmpty() {
		return c.SetEmpty()
	}

	var ibounds geom.IRect
	// Since we assert that the builder blitter will never blit outside the intersection of clip and ibounds, we create
	// the builder with the snug bounds.
	if p.IsInverseFillType() {
		ibounds = clip
	} else {
		ibounds = p.Bounds().RoundOut()
		// It's possible the bounds of our path might exceed the int32 range in width, but since our clip is within that
		// range (otherwise IsEmpty above would catch it), we can use the plain left/right comparison safely here;
		// blitPath will intersect the two bounds before drawing.
		if ibounds.Right <= ibounds.Left || ibounds.Bottom <= ibounds.Top || !ibounds.Intersect(clip) {
			return c.SetEmpty()
		}
	}

	builder := newAAClipBuilder(ibounds)
	return builder.blitPath(c, p, doAA)
}

///////////////////////////////////////////////////////////////////////////////
// ops

// Op combines this clip with other under op (intersect or difference).
func (c *AAClip) Op(other *AAClip, op ClipOp) bool {
	if c.IsEmpty() {
		// Once the clip is empty, it cannot become un-empty.
		return false
	}

	bounds := c.bounds
	switch op {
	case ClipDifference:
		if other.IsEmpty() || !c.bounds.Intersects(other.bounds) {
			// this remains unmodified and isn't empty
			return true
		}
	case ClipIntersect:
		if other.IsEmpty() || !bounds.Intersect(other.bounds) {
			// the intersected clip becomes empty
			return c.SetEmpty()
		}
	}

	builder := newAAClipBuilder(bounds)
	return builder.applyClipOp(c, other, op)
}

// OpIRect combines this clip with a hard-edged rect under op.
func (c *AAClip) OpIRect(rect geom.IRect, op ClipOp) bool {
	// It can be expensive to build a local aaclip before applying the op, so we first see if we can restrict the bounds
	// of new rect to our current bounds, or note that the new rect subsumes our current clip.
	pixelBounds := c.bounds
	switch {
	case !pixelBounds.Intersect(rect):
		// No change or clip becomes empty depending on 'op'
		if op == ClipDifference {
			return !c.IsEmpty()
		}
		return c.SetEmpty()
	case pixelBounds == c.bounds:
		// Wholly inside 'rect', so clip becomes empty or remains unchanged
		if op == ClipDifference {
			return c.SetEmpty()
		}
		return !c.IsEmpty()
	case op == ClipIntersect && c.QuickContains(pixelBounds):
		// We become just the remaining rectangle
		return c.SetRect(pixelBounds)
	default:
		var clip AAClip
		clip.SetRect(rect)
		return c.Op(&clip, op)
	}
}

// OpRect combines this clip with a (possibly fractional) rect under op, optionally antialiased.
func (c *AAClip) OpRect(rect geom.Rect, op ClipOp, doAA bool) bool {
	if !doAA {
		return c.OpIRect(rect.Round(), op)
	}
	// Tighten bounds for "path" aaclip of the rect
	pixelBounds := c.bounds
	switch {
	case !pixelBounds.Intersect(rect.RoundOut()):
		// No change or clip becomes empty depending on 'op'
		if op == ClipDifference {
			return !c.IsEmpty()
		}
		return c.SetEmpty()
	case rect.ContainsRect(c.bounds.ToRect()):
		// Wholly inside 'rect', so clip becomes empty or remains unchanged
		if op == ClipDifference {
			return c.SetEmpty()
		}
		return !c.IsEmpty()
	case op == ClipIntersect && c.QuickContains(pixelBounds):
		// We become just the rect intersected with pixel bounds (preserving fractional coords for AA edges).
		return c.SetPath(path.New().AddRect(rect, geom.DirectionCW), pixelBounds, true)
	default:
		var rectClip AAClip
		clipTo := pixelBounds
		if op == ClipDifference {
			clipTo = c.bounds
		}
		rectClip.SetPath(path.New().AddRect(rect, geom.DirectionCW), clipTo, true)
		return c.Op(&rectClip, op)
	}
}

// Translate sets dst to this clip offset by (dx, dy). The run head is shared between src and dst since translation only
// changes bounds.
func (c *AAClip) Translate(dx, dy int32, dst *AAClip) bool {
	if dst == nil {
		return !c.IsEmpty()
	}
	if c.IsEmpty() {
		return dst.SetEmpty()
	}
	if c != dst {
		dst.head = c.head
		dst.bounds = c.bounds
	}
	dst.bounds = dst.bounds.Offset(dx, dy)
	return true
}

///////////////////////////////////////////////////////////////////////////////
// row lookup & containment

// findRow returns the run data for the row containing y, or nil when y is outside the bounds. lastY is the last
// device-space scanline sharing that data.
func (c *AAClip) findRow(y int32) (row []uint8, lastY int32) {
	if y < c.bounds.Top || y >= c.bounds.Bottom {
		return nil, 0
	}
	y -= c.bounds.Top // our yoffset values are relative to the top

	i := 0
	for c.head.yOffsets[i].y < y {
		i++
	}
	return c.head.data[c.head.yOffsets[i].offset:], c.bounds.Top + c.head.yOffsets[i].y
}

// findX advances run data to the run containing x. initialCount is the number of pixels remaining in that run at x.
func (c *AAClip) findX(data []uint8, x int32) (row []uint8, initialCount int32) {
	x -= c.bounds.Left
	for {
		n := int32(data[0])
		if x < n {
			return data, n - x
		}
		data = data[2:]
		x -= n
	}
}

// QuickContains reports whether the clip is at full coverage over all of r.
func (c *AAClip) QuickContains(r geom.IRect) bool {
	return c.quickContains(r.Left, r.Top, r.Right, r.Bottom)
}

func (c *AAClip) quickContains(left, top, right, bottom int32) bool {
	if c.IsEmpty() {
		return false
	}
	if !c.bounds.ContainsRect(geom.IRectLTRB(left, top, right, bottom)) {
		return false
	}

	// Note: this compares the row band's last scanline against the exclusive bottom, so a band must extend one row past
	// the query rect for the quick answer to be "yes" — a conservative quirk that callers' fallback paths rely on; keep
	// it exact.
	row, lastY := c.findRow(top)
	if lastY < bottom {
		return false
	}
	// now just need to check in X
	row, count := c.findX(row, left)

	rectWidth := right - left
	for row[1] == 0xFF {
		if count >= rectWidth {
			return true
		}
		rectWidth -= count
		row = row[2:]
		count = int32(row[0])
	}
	return false
}

// CopyToMask allocates a mask the size of the clip and expands the run data into it.
func (c *AAClip) CopyToMask() *Mask {
	mask := &Mask{}
	if c.IsEmpty() {
		return mask
	}

	expandRowToMask := func(dst, row []uint8, width int32) {
		for width > 0 {
			n := int32(row[0])
			for i := int32(0); i < n; i++ {
				dst[i] = row[1]
			}
			dst = dst[n:]
			row = row[2:]
			width -= n
		}
	}

	mask.Bounds = c.bounds
	mask.RowBytes = c.bounds.Width()
	mask.Image = make([]uint8, int(mask.RowBytes)*int(c.bounds.Height()))

	iter := newAAIter(c)
	dst := mask.Image
	width := c.bounds.Width()

	y := c.bounds.Top
	for !iter.done {
		for {
			expandRowToMask(dst, iter.data(), width)
			dst = dst[mask.RowBytes:]
			y++
			if y >= iter.bottom {
				break
			}
		}
		iter.next()
	}
	return mask
}

///////////////////////////////////////////////////////////////////////////////
// bounds trimming (only ever run on freshly built heads)

// countLeftRightZeros returns the number of zeros on the left and right edges of the RLE row. If the row is all zeros,
// returns width in both.
func countLeftRightZeros(row []uint8, width int32) (leftZ, riteZ int32) {
	zeros := int32(0)
	for width > 0 {
		if row[1] != 0 {
			break
		}
		n := int32(row[0])
		zeros += n
		row = row[2:]
		width -= n
	}
	leftZ = zeros

	if width == 0 {
		// this line is completely empty; return 'width' in both variables
		return leftZ, leftZ
	}

	zeros = 0
	for width > 0 {
		n := int32(row[0])
		if row[1] == 0 {
			zeros += n
		} else {
			zeros = 0
		}
		row = row[2:]
		width -= n
	}
	return leftZ, zeros
}

// trimLeftRight shrinks the clip bounds to remove uniform all-zero columns on the left and right edges, expressed as
// offset adjustment plus boundary-run rewrites since the head is freshly built and unshared.
func (c *AAClip) trimLeftRight() bool {
	if c.IsEmpty() {
		return false
	}

	width := c.bounds.Width()
	head := c.head

	// After this loop, leftZeros & riteZeros will contain the minimum number of zeros on the left and right of the
	// clip. This information can be used to shrink the bounding box.
	leftZeros := width
	riteZeros := width
	for i := range head.yOffsets {
		l, r := countLeftRightZeros(head.data[head.yOffsets[i].offset:], width)
		if l < leftZeros {
			leftZeros = l
		}
		if r < riteZeros {
			riteZeros = r
		}
		if leftZeros|riteZeros == 0 {
			// no trimming to do
			return true
		}
	}

	if width == leftZeros {
		return c.SetEmpty()
	}

	c.bounds.Left += leftZeros
	c.bounds.Right -= riteZeros

	// For now we don't realloc the storage (for time), we just shrink in place: play tricks with the per-row offset for
	// each row (and rewrite boundary run counts in place).
	for i := range head.yOffsets {
		yoff := &head.yOffsets[i]
		row := head.data[yoff.offset:]
		// trim the left
		leftZ := leftZeros
		for leftZ > 0 {
			n := int32(row[0])
			if n > leftZ {
				row[0] = uint8(n - leftZ)
				break
			}
			yoff.offset += 2
			leftZ -= n
			row = row[2:]
		}
		if riteZeros > 0 {
			// walk the (possibly left-trimmed) row to the end, then back up to trim riteZeros
			row = head.data[yoff.offset:]
			w := width - leftZeros
			idx := 0
			for w > 0 {
				w -= int32(row[idx])
				idx += 2
			}
			riteZ := riteZeros
			for {
				idx -= 2
				n := int32(row[idx])
				if n > riteZ {
					row[idx] = uint8(n - riteZ)
					break
				}
				riteZ -= n
				if riteZ <= 0 {
					break
				}
			}
		}
	}
	return true
}

// rowIsAllZeros reports whether a run row has zero coverage across its full width.
func rowIsAllZeros(row []uint8, width int32) bool {
	for width > 0 {
		if row[1] != 0 {
			return false
		}
		width -= int32(row[0])
		row = row[2:]
	}
	return true
}

// trimTopBottom shrinks the clip bounds to remove all-zero rows from the top and bottom.
func (c *AAClip) trimTopBottom() bool {
	if c.IsEmpty() {
		return false
	}

	width := c.bounds.Width()
	head := c.head

	// Look to trim away empty rows from the top.
	skip := 0
	for skip < len(head.yOffsets) {
		if !rowIsAllZeros(head.data[head.yOffsets[skip].offset:], width) {
			break
		}
		skip++
	}
	if skip == len(head.yOffsets) {
		return c.SetEmpty()
	}
	if skip > 0 {
		// adjust the y values and fBounds.Top as we remove the skipped rows
		dy := head.yOffsets[skip-1].y + 1
		head.yOffsets = head.yOffsets[skip:]
		for i := range head.yOffsets {
			head.yOffsets[i].y -= dy
		}
		c.bounds.Top += dy
	}

	// Look to trim away empty rows from the bottom. We know that we have at least one non-zero row, so we can just walk
	// backwards without checking for running past the start.
	last := len(head.yOffsets) - 1
	for rowIsAllZeros(head.data[head.yOffsets[last].offset:], width) {
		last--
	}
	if last < len(head.yOffsets)-1 {
		c.bounds.Bottom = c.bounds.Top + head.yOffsets[last].y + 1
		head.yOffsets = head.yOffsets[:last+1]
	}
	return true
}

// trimBounds finalizes the clip's bounds after building. Since rows are built from top to bottom, it's possible the
// bounds' bottom is bigger than the last scanline of data, so it is trimmed back up.
func (c *AAClip) trimBounds() bool {
	if c.IsEmpty() {
		return false
	}
	lastY := c.head.yOffsets[len(c.head.yOffsets)-1]
	c.bounds.Bottom = c.bounds.Top + lastY.y + 1
	return c.trimTopBottom() && c.trimLeftRight()
}

///////////////////////////////////////////////////////////////////////////////
// iterators

const aaMaxInt32 = int32(math.MaxInt32)

// aaRowIter iterates the [left, right) alpha spans of one row.
type aaRowIter struct {
	row         []uint8
	idx         int
	left        int32
	right       int32
	boundsRight int32
	alpha       uint8
	isDone      bool
}

func newAARowIter(row []uint8, bounds geom.IRect) aaRowIter {
	it := aaRowIter{row: row, left: bounds.Left, boundsRight: bounds.Right}
	if row != nil {
		it.right = bounds.Left + int32(row[0])
		it.alpha = row[1]
	} else {
		it.isDone = true
		it.right = aaMaxInt32
	}
	return it
}

func (it *aaRowIter) next() {
	if !it.isDone {
		it.left = it.right
		if it.right == it.boundsRight {
			it.isDone = true
			it.right = aaMaxInt32
			it.alpha = 0
		} else {
			it.idx += 2
			it.right += int32(it.row[it.idx])
			it.alpha = it.row[it.idx+1]
		}
	}
}

// aaIter iterates the [top, bottom) row bands of a clip.
type aaIter struct {
	head   *aaRunHead
	curr   int
	top    int32
	bottom int32
	done   bool
}

// newAAIter returns an iterator positioned at the first row band of clip.
func newAAIter(clip *AAClip) aaIter {
	if clip.head == nil {
		// an empty clip: an already finished iterator
		return aaIter{done: true, top: aaMaxInt32, bottom: aaMaxInt32}
	}
	return aaIter{
		head:   clip.head,
		top:    clip.bounds.Top,
		bottom: clip.bounds.Top + clip.head.yOffsets[0].y + 1,
	}
}

func (it *aaIter) data() []uint8 {
	if it.done {
		// once the iterator finishes, data is nil; operateY relies on that to treat the exhausted side as all-zero
		// coverage
		return nil
	}
	return it.head.data[it.head.yOffsets[it.curr].offset:]
}

func (it *aaIter) next() {
	if !it.done {
		next := it.curr + 1
		it.top = it.bottom
		if next >= len(it.head.yOffsets) {
			it.done = true
			it.bottom = aaMaxInt32
		} else {
			it.bottom += it.head.yOffsets[next].y - it.head.yOffsets[it.curr].y
			it.curr = next
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Builder

// aaBuilderRow accumulates one row's run data while building an AAClip.
type aaBuilderRow struct {
	data  []uint8
	y     int32
	width int32
}

// aaClipBuilder accumulates rows of run data (from path scan conversion or clip-op merging) and packs them into an
// AAClip.
type aaClipBuilder struct {
	currRow *aaBuilderRow
	rows    []*aaBuilderRow
	bounds  geom.IRect
	prevY   int32
	width   int32
	minY    int32
}

func newAAClipBuilder(bounds geom.IRect) *aaClipBuilder {
	return &aaClipBuilder{
		bounds: bounds,
		prevY:  -1,
		width:  bounds.Width(),
		minY:   bounds.Top,
	}
}

// aaAppendRun appends a run of count pixels at alpha to data, splitting into multiple 255-max segments as needed.
func aaAppendRun(data []uint8, alpha uint8, count int32) []uint8 {
	for {
		n := min(count, 255)
		data = append(data, uint8(n), alpha)
		count -= n
		if count <= 0 {
			return data
		}
	}
}

func (b *aaClipBuilder) addRun(x, y int32, alpha uint8, count int32) {
	x -= b.bounds.Left
	y -= b.bounds.Top

	row := b.currRow
	if y != b.prevY {
		b.prevY = y
		row = b.flushRow(true)
		row.y = y
		row.width = 0
		b.currRow = row
	}

	if gap := x - row.width; gap > 0 {
		row.data = aaAppendRun(row.data, 0, gap)
		row.width += gap
	}

	row.data = aaAppendRun(row.data, alpha, count)
	row.width += count
}

func (b *aaClipBuilder) addColumn(x, y int32, alpha uint8, height int32) {
	b.addRun(x, y, alpha, 1)
	b.flushRowH(b.currRow)
	y -= b.bounds.Top
	b.currRow.y = y + height - 1
}

func (b *aaClipBuilder) addRectRun(x, y, width, height int32) {
	b.addRun(x, y, 0xFF, width)

	// we assume the rect must be all we'll see for these scanlines, so we ensure our row goes all the way to our right
	b.flushRowH(b.currRow)

	y -= b.bounds.Top
	b.currRow.y = y + height - 1
}

func (b *aaClipBuilder) addAntiRectRun(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	// No matter whether leftAlpha is 0 or positive, we should always consider [x, x+1] as the left-most column and
	// [x+1, x+1+width] as the rect with full alpha.
	//
	// Conceptually we're always adding 3 runs, but we should merge or omit them if possible.
	switch {
	case leftAlpha == 0xFF:
		width++
	case leftAlpha > 0:
		b.addRun(x, y, leftAlpha, 1)
		x++
	default:
		// leftAlpha is 0, ignore the left column
		x++
	}
	if rightAlpha == 0xFF {
		width++
	}
	if width > 0 {
		b.addRun(x, y, 0xFF, width)
	}
	if rightAlpha > 0 && rightAlpha < 255 {
		b.addRun(x+width, y, rightAlpha, 1)
	}

	// if we never called addRun, we might not have a currRow yet
	if b.currRow != nil {
		// we assume the rect must be all we'll see for these scanlines, so we ensure our row goes all the way to our
		// right
		b.flushRowH(b.currRow)

		y -= b.bounds.Top
		b.currRow.y = y + height - 1
	}
}

// flushRowH pads the row with zeros out to the full width.
func (b *aaClipBuilder) flushRowH(row *aaBuilderRow) {
	if row.width < b.width {
		row.data = aaAppendRun(row.data, 0, b.width-row.width)
		row.width = b.width
	}
}

// flushRow flushes the last row, collapses it into its predecessor if their data match, and optionally returns a fresh
// row ready for more runs.
func (b *aaClipBuilder) flushRow(readyForAnother bool) *aaBuilderRow {
	var next *aaBuilderRow
	count := len(b.rows)
	if count > 0 {
		b.flushRowH(b.rows[count-1])
	}
	if count > 1 {
		// are our last two runs the same?
		prev := b.rows[count-2]
		curr := b.rows[count-1]
		if bytes.Equal(prev.data, curr.data) {
			prev.y = curr.y
			if readyForAnother {
				curr.data = curr.data[:0]
				next = curr
			} else {
				b.rows = b.rows[:count-1]
			}
		} else if readyForAnother {
			next = &aaBuilderRow{}
			b.rows = append(b.rows, next)
		}
	} else if readyForAnother {
		next = &aaBuilderRow{}
		b.rows = append(b.rows, next)
	}
	return next
}

// finish packs the accumulated rows into target's run head and trims it.
func (b *aaClipBuilder) finish(target *AAClip) bool {
	b.flushRow(false)

	dataSize := 0
	for _, row := range b.rows {
		dataSize += len(row.data)
	}
	if dataSize == 0 {
		return target.SetEmpty()
	}

	adjustY := b.minY - b.bounds.Top
	b.bounds.Top = b.minY

	head := &aaRunHead{
		yOffsets: make([]aaYOffset, len(b.rows)),
		data:     make([]uint8, 0, dataSize),
	}
	for i, row := range b.rows {
		head.yOffsets[i] = aaYOffset{y: row.y - adjustY, offset: uint32(len(head.data))}
		head.data = append(head.data, row.data...)
	}

	target.bounds = b.bounds
	target.head = head
	return target.trimBounds()
}

///////////////////////////////////////////////////////////////////////////////
// Builder boolean-op machinery

// aaAlphaProc combines two coverage values under a clip op.
type aaAlphaProc func(alphaA, alphaB uint8) uint8

// operateX merges one row from each side over the builder bounds' width.
func (b *aaClipBuilder) operateX(lastY int32, iterA, iterB *aaRowIter, proc aaAlphaProc) {
	advanceRowIter := func(iter *aaRowIter, iterLeft, iterRite *int32, rite int32) {
		if rite == *iterRite {
			iter.next()
			*iterLeft = iter.left
			*iterRite = iter.right
		}
	}

	leftA := iterA.left
	riteA := iterA.right
	leftB := iterB.left
	riteB := iterB.right

	prevRite := b.bounds.Left

	for {
		var alphaA, alphaB uint8
		var left, rite int32

		switch {
		case leftA < leftB:
			left = leftA
			alphaA = iterA.alpha
			if riteA <= leftB {
				rite = riteA
			} else {
				leftA = leftB
				rite = leftA
			}
		case leftB < leftA:
			left = leftB
			alphaB = iterB.alpha
			if riteB <= leftA {
				rite = riteB
			} else {
				leftB = leftA
				rite = leftB
			}
		default:
			left = leftA // or leftB, since leftA == leftB
			rite = min(riteA, riteB)
			leftA = rite
			leftB = rite
			alphaA = iterA.alpha
			alphaB = iterB.alpha
		}

		if left >= b.bounds.Right {
			break
		}
		if rite > b.bounds.Right {
			rite = b.bounds.Right
		}

		if left >= b.bounds.Left {
			b.addRun(left, lastY, proc(alphaA, alphaB), rite-left)
			prevRite = rite
		}

		advanceRowIter(iterA, &leftA, &riteA, rite)
		advanceRowIter(iterB, &leftB, &riteB, rite)

		if iterA.isDone && iterB.isDone {
			break
		}
	}

	if prevRite < b.bounds.Right {
		b.addRun(prevRite, lastY, 0, b.bounds.Right-prevRite)
	}
}

// operateY walks both clips' row bands, combining rows with the op's alpha proc.
func (b *aaClipBuilder) operateY(clipA, clipB *AAClip, op ClipOp) {
	var proc aaAlphaProc
	if op == ClipDifference {
		proc = func(a, alphaB uint8) uint8 { return colorcore.MulDiv255Round(a, 0xFF-alphaB) }
	} else {
		proc = colorcore.MulDiv255Round
	}

	iterA := newAAIter(clipA)
	iterB := newAAIter(clipB)

	topA := iterA.top
	botA := iterA.bottom
	topB := iterB.top
	botB := iterB.bottom

	advanceIter := func(iter *aaIter, iterTop, iterBot *int32, bot int32) {
		if bot == *iterBot {
			iter.next()
			*iterTop = *iterBot
			*iterBot = iter.bottom
		}
	}

	for {
		var rowA, rowB []uint8
		var top, bot int32

		switch {
		case topA < topB:
			top = topA
			rowA = iterA.data()
			if botA <= topB {
				bot = botA
			} else {
				topA = topB
				bot = topA
			}
		case topB < topA:
			top = topB
			rowB = iterB.data()
			if botB <= topA {
				bot = botB
			} else {
				topB = topA
				bot = topB
			}
		default:
			top = topA // or topB, since topA == topB
			bot = min(botA, botB)
			topA = bot
			topB = bot
			rowA = iterA.data()
			rowB = iterB.data()
		}

		if top >= b.bounds.Bottom {
			break
		}

		if bot > b.bounds.Bottom {
			bot = b.bounds.Bottom
		}

		if rowA == nil && rowB == nil {
			b.addRun(b.bounds.Left, bot-1, 0, b.bounds.Width())
		} else if top >= b.bounds.Top {
			boundsA := b.bounds
			if rowA != nil {
				boundsA = clipA.bounds
			}
			boundsB := b.bounds
			if rowB != nil {
				boundsB = clipB.bounds
			}
			rowIterA := newAARowIter(rowA, boundsA)
			rowIterB := newAARowIter(rowB, boundsB)
			b.operateX(bot-1, &rowIterA, &rowIterB, proc)
		}

		advanceIter(&iterA, &topA, &botA, bot)
		advanceIter(&iterB, &topB, &botB, bot)

		if iterA.done && iterB.done {
			break
		}
	}
}

// applyClipOp merges other into target under op and packs the result.
func (b *aaClipBuilder) applyClipOp(target, other *AAClip, op ClipOp) bool {
	b.operateY(target, other, op)
	return b.finish(target)
}

// blitPath scan-converts p into the builder via a dedicated blitter, then packs the result into target.
func (b *aaClipBuilder) blitPath(target *AAClip, p *path.Path, doAA bool) bool {
	blitter := newAAClipBuilderBlitter(b)
	clip := NewRegionRect(b.bounds)

	if doAA {
		antiFillPathRegion(p, clip, blitter, true)
	} else {
		FillPathRegion(p, clip, blitter)
	}

	blitter.finish()
	return b.finish(target)
}

///////////////////////////////////////////////////////////////////////////////
// Builder's blitter

// aaClipBuilderBlitter adapts scan-converter blits into builder runs, in scanline order.
type aaClipBuilderBlitter struct {
	builder *aaClipBuilder
	left    int32 // cache of builder's bounds' left edge
	right   int32
	minY    int32
	lastY   int32
}

func newAAClipBuilderBlitter(builder *aaClipBuilder) *aaClipBuilderBlitter {
	return &aaClipBuilderBlitter{
		builder: builder,
		left:    builder.bounds.Left,
		right:   builder.bounds.Right,
		minY:    aaMaxInt32,
		lastY:   -aaMaxInt32, // sentinel
	}
}

func (bl *aaClipBuilderBlitter) finish() {
	if bl.minY < aaMaxInt32 {
		bl.builder.minY = bl.minY
	}
}

// recordMinY tracks the top scanline, in case the scan converter skipped some number of scanlines at the top of the
// bounds it was given. This allows the builder, during its finish, to trim its bounds down to the "real" top.
func (bl *aaClipBuilderBlitter) recordMinY(y int32) {
	if y < bl.minY {
		bl.minY = y
	}
}

// checkForYGap injects an explicit empty scanline if we see a gap of 1 or more empty scanlines while building in
// Y-order.
func (bl *aaClipBuilderBlitter) checkForYGap(y int32) {
	if bl.lastY > -aaMaxInt32 {
		if y-bl.lastY > 1 {
			bl.builder.addRun(bl.left, y-1, 0, bl.right-bl.left)
		}
	}
	bl.lastY = y
}

// BlitV implements Blitter. Clips must be evaluated in scan-line order, so blitV is unwelcome — but an AAClip can be
// clipped down to a single pixel wide, so it must be supported (given AntiRect semantics: minimum width is 2).
func (bl *aaClipBuilderBlitter) BlitV(x, y, height int32, alpha Alpha) {
	if height == 1 {
		// We're still in scan-line order if height is 1. This is useful for analytic AA.
		alphas := [2]Alpha{alpha, 0}
		runs := [2]int16{1, 0}
		bl.BlitAntiH(x, y, alphas[:], runs[:])
	} else {
		bl.recordMinY(y)
		bl.builder.addColumn(x, y, alpha, height)
		bl.lastY = y + height - 1
	}
}

// BlitRect implements Blitter.
func (bl *aaClipBuilderBlitter) BlitRect(x, y, width, height int32) {
	bl.recordMinY(y)
	bl.checkForYGap(y)
	bl.builder.addRectRun(x, y, width, height)
	bl.lastY = y + height - 1
}

// BlitAntiRect implements Blitter.
func (bl *aaClipBuilderBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	bl.recordMinY(y)
	bl.checkForYGap(y)
	bl.builder.addAntiRectRun(x, y, width, height, leftAlpha, rightAlpha)
	bl.lastY = y + height - 1
}

// BlitMask implements Blitter. The builder's forceRLE dispatch means the mask route is never taken.
func (bl *aaClipBuilderBlitter) BlitMask(_ *Mask, _ geom.IRect) {
	panic("aaClipBuilderBlitter: did not expect to get called here")
}

// BlitH implements Blitter.
func (bl *aaClipBuilderBlitter) BlitH(x, y, width int32) {
	bl.recordMinY(y)
	bl.checkForYGap(y)
	bl.builder.addRun(x, y, 0xFF, width)
}

// BlitAntiH implements Blitter.
func (bl *aaClipBuilderBlitter) BlitAntiH(x, y int32, alpha []Alpha, runs []int16) {
	bl.recordMinY(y)
	bl.checkForYGap(y)
	off := 0
	for {
		count := int32(runs[off])
		if count <= 0 {
			return
		}

		// The scan converter's buffer can be the width of the device, so we may have to trim the run to our bounds. The
		// analytic AA is too sensitive to precision errors to assert the trimmed spans are exactly alpha==0; we just
		// trim.
		localX := x
		localCount := count
		if x < bl.left {
			gap := bl.left - x
			localX += gap
			localCount -= gap
		}
		if right := x + count; right > bl.right {
			localCount -= right - bl.right
		}

		if localCount > 0 {
			bl.builder.addRun(localX, y, alpha[off], localCount)
		}
		// Next run
		off += int(count)
		x += count
	}
}

// BlitAntiH2 implements Blitter via the default lowering.
func (bl *aaClipBuilderBlitter) BlitAntiH2(x, y int32, a0, a1 Alpha) {
	BlitAntiH2Via(bl, x, y, a0, a1)
}

// BlitAntiV2 implements Blitter via the default lowering.
func (bl *aaClipBuilderBlitter) BlitAntiV2(x, y int32, a0, a1 Alpha) {
	BlitAntiV2Via(bl, x, y, a0, a1)
}
