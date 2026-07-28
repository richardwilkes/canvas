// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The backend-shared atlas types and token machinery: IRect16, MaskFormat, AtlasGenerationCounter, PlotLocator,
// AtlasLocator, PlotEvictionCallback, BulkUsePlotUpdater, Plot, and PlotList, plus Token/TokenTracker. These back the
// GPU draw-op atlas in gpu/gl. Plot's pixel copy has no RGBA<->BGRA swizzle lane since this codebase's device/N32
// format is RGBA8888 on every platform; the intrusive-list plumbing is an explicit intrusive list (PlotList); and
// MaskFormat.ColorType maps each mask format directly to the ColorType it uploads as.

package gpu

import (
	"github.com/richardwilkes/canvas/geom"
)

// Token is a generic token used to sequence operations. The zero value is the invalid token.
type Token struct {
	sequenceNumber uint64
}

// InvalidToken returns the invalid token.
func InvalidToken() Token { return Token{} }

// LessThan reports whether t sequences before other.
func (t Token) LessThan(other Token) bool { return t.sequenceNumber < other.sequenceNumber }

// LessThanOrEqual reports whether t sequences before or at the same point as other.
func (t Token) LessThanOrEqual(other Token) bool {
	return t.sequenceNumber <= other.sequenceNumber
}

// Next returns the token immediately following t.
func (t Token) Next() Token { return Token{sequenceNumber: t.sequenceNumber + 1} }

// Value returns the raw value for debugging and comparison.
func (t Token) Value() uint64 { return t.sequenceNumber }

// InInterval reports whether t lies in the [start, end] inclusive interval.
func (t Token) InInterval(start, end Token) bool {
	return start.LessThanOrEqual(t) && t.LessThanOrEqual(end)
}

// TokenTracker encapsulates the incrementing and distribution of Tokens. The issue methods are meant to be called only
// by the flush state and tests.
type TokenTracker struct {
	currentDrawToken  Token
	currentFlushToken Token
}

// NextFlushToken returns one beyond the last flushed token — the ID for the current batch of work being recorded.
func (t *TokenTracker) NextFlushToken() Token { return t.currentFlushToken.Next() }

// CurrentFlushToken returns the most recently issued flush token.
func (t *TokenTracker) CurrentFlushToken() Token { return t.currentFlushToken }

// NextDrawToken returns the token that would be issued by the next draw.
func (t *TokenTracker) NextDrawToken() Token { return t.currentDrawToken.Next() }

// IssueDrawToken advances and returns the current draw token.
func (t *TokenTracker) IssueDrawToken() Token {
	t.currentDrawToken = t.currentDrawToken.Next()
	return t.currentDrawToken
}

// IssueFlushToken advances and returns the current flush token.
func (t *TokenTracker) IssueFlushToken() Token {
	t.currentFlushToken = t.currentFlushToken.Next()
	return t.currentFlushToken
}

// IRect16 is an integer rectangle with 16-bit coordinates, used for compact atlas bookkeeping.
type IRect16 struct {
	Left, Top, Right, Bottom int16
}

// IRect16MakeXYWH returns the rectangle at (x, y) with the given width and height.
func IRect16MakeXYWH(x, y, w, h int16) IRect16 {
	return IRect16{Left: x, Top: y, Right: x + w, Bottom: y + h}
}

// Width returns the rectangle's width.
func (r IRect16) Width() int { return int(r.Right) - int(r.Left) }

// Height returns the rectangle's height.
func (r IRect16) Height() int { return int(r.Bottom) - int(r.Top) }

// Offset translates the rectangle by (dx, dy).
func (r *IRect16) Offset(dx, dy int16) {
	r.Left += dx
	r.Top += dy
	r.Right += dx
	r.Bottom += dy
}

// MaskFormat identifies the pixel format of a glyph mask stored in the font atlas. Important that these are 0-based
// (they index the atlas-manager arrays).
type MaskFormat int32

// MaskFormat values.
const (
	// MaskFormatA8: 1-byte-per-pixel coverage.
	MaskFormatA8 MaskFormat = iota
	// MaskFormatA565: 2-bytes-per-pixel, RGB represent 3-channel LCD coverage (LCD16 glyphs).
	MaskFormatA565
	// MaskFormatARGB: 4-bytes-per-pixel color format.
	MaskFormatARGB

	// MaskFormatCount is the number of valid MaskFormat values.
	MaskFormatCount = int(MaskFormatARGB) + 1
)

// BytesPerPixel returns the storage size in bytes of one pixel of this mask format.
func (f MaskFormat) BytesPerPixel() int { return 1 << int(f) }

// ColorType returns the gpu.ColorType the format's atlas stores texels as.
func (f MaskFormat) ColorType() ColorType {
	switch f {
	case MaskFormatA8:
		return ColorTypeAlpha8
	case MaskFormatA565:
		return ColorTypeBGR565
	case MaskFormatARGB:
		return ColorTypeRGBA8888
	default:
		panic("invalid mask format")
	}
}

// AtlasGenerationCounter issues monotonically increasing atlas generation numbers. The zero value is ready to use (the
// first Next() returns 1).
type AtlasGenerationCounter struct {
	generation uint64
}

// InvalidAtlasGeneration is the generation value that never matches a real atlas generation.
const InvalidAtlasGeneration uint64 = 0

// Next returns the next atlas generation number.
func (c *AtlasGenerationCounter) Next() uint64 {
	c.generation++
	return c.generation
}

// PlotLocator constants: restricted by the space they occupy in the locator; MaxMultitexturePages is also limited by
// being crammed into the glyph UVs, and MaxPlots by BulkUsePlotUpdater's bitfield.
const (
	MaxMultitexturePages = 4
	MaxPlots             = 32
)

// PlotLocator identifies a page/plot/generation triple: a portion of a glyph image location in the atlas. The zero
// value is the invalid locator.
type PlotLocator struct {
	genID     uint64 // 48 bits of usable range
	plotIndex uint32 // 8 bits of usable range
	pageIndex uint32 // 8 bits of usable range
}

// MakePlotLocator returns a PlotLocator for the given page, plot, and generation.
func MakePlotLocator(pageIdx, plotIdx uint32, generation uint64) PlotLocator {
	if pageIdx >= MaxMultitexturePages || plotIdx >= MaxPlots || generation >= 1<<48 {
		panic("invalid plot locator")
	}
	return PlotLocator{genID: generation, plotIndex: plotIdx, pageIndex: pageIdx}
}

// IsValid reports whether this is a real (non-zero) plot locator.
func (p PlotLocator) IsValid() bool {
	return p.genID != InvalidAtlasGeneration || p.plotIndex != 0 || p.pageIndex != 0
}

// MakeInvalid resets the locator to the invalid value.
func (p *PlotLocator) MakeInvalid() { *p = PlotLocator{} }

// PageIndex returns the atlas page this locator refers to.
func (p PlotLocator) PageIndex() uint32 { return p.pageIndex }

// PlotIndex returns the plot within the page this locator refers to.
func (p PlotLocator) PlotIndex() uint32 { return p.plotIndex }

// GenID returns the generation this locator was issued at.
func (p PlotLocator) GenID() uint64 { return p.genID }

// AtlasLocator holds atlas position information as a left-top/right-bottom pair of encoded UV coordinates; bits 13 and
// 14 of the U coordinates hold the atlas page index. This encoding has the nice property that width = uvs[2] - uvs[0]
// (the page bits subtract to zero).
type AtlasLocator struct {
	plotLocator PlotLocator
	uvs         [4]uint16
}

// UVs returns the raw encoded UV coordinates.
func (a *AtlasLocator) UVs() [4]uint16 { return a.uvs }

// InvalidatePlotLocator marks the locator's plot as invalid.
func (a *AtlasLocator) InvalidatePlotLocator() { a.plotLocator.MakeInvalid() }

// PlotLocator returns the plot locator this atlas locator resolves to.
func (a *AtlasLocator) PlotLocator() PlotLocator { return a.plotLocator }

// PageIndex returns the atlas page this locator refers to.
func (a *AtlasLocator) PageIndex() uint32 { return a.plotLocator.PageIndex() }

// PlotIndex returns the plot within the page this locator refers to.
func (a *AtlasLocator) PlotIndex() uint32 { return a.plotLocator.PlotIndex() }

// GenID returns the generation this locator's plot was issued at.
func (a *AtlasLocator) GenID() uint64 { return a.plotLocator.GenID() }

// TopLeft returns the top-left texel coordinate the locator refers to, with the page-index bits masked out.
func (a *AtlasLocator) TopLeft() geom.IPoint {
	return geom.IPoint{X: int32(a.uvs[0] & 0x1FFF), Y: int32(a.uvs[1])}
}

// Width returns the width in texels of the region the locator refers to.
func (a *AtlasLocator) Width() uint16 { return a.uvs[2] - a.uvs[0] }

// Height returns the height in texels of the region the locator refers to.
func (a *AtlasLocator) Height() uint16 { return a.uvs[3] - a.uvs[1] }

// InsetSrc shrinks the locator's region by padding texels on every side.
func (a *AtlasLocator) InsetSrc(padding int) {
	if 2*padding > int(a.Width()) || 2*padding > int(a.Height()) {
		panic("insetSrc padding exceeds locator size")
	}
	a.uvs[0] += uint16(padding)
	a.uvs[1] += uint16(padding)
	a.uvs[2] -= uint16(padding)
	a.uvs[3] -= uint16(padding)
}

// UpdatePlotLocator sets the locator's plot and re-encodes the page-index bits into the UVs.
func (a *AtlasLocator) UpdatePlotLocator(p PlotLocator) {
	a.plotLocator = p
	if a.plotLocator.PageIndex() > 3 {
		panic("page index exceeds the UV encoding")
	}
	page := uint16(a.plotLocator.PageIndex()) << 13
	a.uvs[0] = (a.uvs[0] & 0x1FFF) | page
	a.uvs[2] = (a.uvs[2] & 0x1FFF) | page
}

// UpdateRect re-encodes the locator's texel rectangle, preserving the page-index bits.
func (a *AtlasLocator) UpdateRect(rect IRect16) {
	if rect.Left > rect.Right || rect.Right > 0x1FFF {
		panic("rect does not fit the UV encoding")
	}
	a.uvs[0] = (a.uvs[0] & 0xE000) | uint16(rect.Left)
	a.uvs[1] = uint16(rect.Top)
	a.uvs[2] = (a.uvs[2] & 0xE000) | uint16(rect.Right)
	a.uvs[3] = uint16(rect.Bottom)
}

// PlotEvictionCallback is notified whenever an atlas evicts a specific PlotLocator, so listeners can process the
// eviction.
type PlotEvictionCallback interface {
	Evict(PlotLocator)
}

// BulkUsePlotUpdater can be handed back to an atlas for updating plots in bulk. The current max number of plots per
// page an atlas can handle is 32.
type BulkUsePlotUpdater struct {
	plotsToUpdate      []BulkUsePlotData
	plotAlreadyUpdated [MaxMultitexturePages]uint32
}

// BulkUsePlotData records one plot slated for a bulk update.
type BulkUsePlotData struct {
	PageIndex uint32
	PlotIndex uint32
}

// Add records atlasLocator's plot as used, returning false if it was already recorded.
func (b *BulkUsePlotUpdater) Add(atlasLocator *AtlasLocator) bool {
	plotIdx := atlasLocator.PlotIndex()
	pageIdx := atlasLocator.PageIndex()
	if b.find(pageIdx, plotIdx) {
		return false
	}
	b.set(pageIdx, plotIdx)
	return true
}

// Reset clears all recorded plot updates.
func (b *BulkUsePlotUpdater) Reset() {
	b.plotsToUpdate = b.plotsToUpdate[:0]
	b.plotAlreadyUpdated = [MaxMultitexturePages]uint32{}
}

// Count returns the number of recorded plot updates.
func (b *BulkUsePlotUpdater) Count() int { return len(b.plotsToUpdate) }

// PlotData returns the plot update recorded at index.
func (b *BulkUsePlotUpdater) PlotData(index int) *BulkUsePlotData {
	return &b.plotsToUpdate[index]
}

func (b *BulkUsePlotUpdater) find(pageIdx, index uint32) bool {
	if index >= MaxPlots {
		panic("plot index out of range")
	}
	return (b.plotAlreadyUpdated[pageIdx]>>index)&1 != 0
}

func (b *BulkUsePlotUpdater) set(pageIdx, index uint32) {
	b.plotAlreadyUpdated[pageIdx] |= 1 << index
	b.plotsToUpdate = append(b.plotsToUpdate, BulkUsePlotData{PageIndex: pageIdx, PlotIndex: index})
}

// Plot is one cell of an atlas's backing texture: the backing texture is broken into a spatial grid of Plots, which
// track subimage placement via their Rectanizer.
type Plot struct {
	generationCounter   *AtlasGenerationCounter
	next                *Plot // Intrusive PlotList links.
	list                *PlotList
	prev                *Plot
	data                []byte
	rectanizer          RectanizerSkyline
	plotLocator         PlotLocator
	flushesSinceLastUse int
	x                   int
	genID               uint64
	bytesPerPixel       int
	lastUse             Token
	width               int
	height              int
	offsetY             int
	y                   int
	lastUpload          Token
	offsetX             int        // the offset of the plot in the backing texture
	dirtyRect           geom.IRect // area in the Plot that needs to be uploaded
	plotIndex           uint32
	colorType           ColorType
	pageIndex           uint32
	isFull              bool
}

// NewPlot returns a new Plot at the given page/plot indices and backing-texture offset.
func NewPlot(pageIndex, plotIndex int, generationCounter *AtlasGenerationCounter, offX, offY, width, height int, colorType ColorType, bpp int) *Plot {
	// We expect the allocated dimensions to be a multiple of 4 bytes.
	if (width*bpp)&0x3 != 0 {
		panic("plot width is not 4-byte aligned")
	}
	// The padding for faster uploads only works for 1, 2 and 4 byte texels.
	if bpp == 3 || bpp > 4 {
		panic("unsupported bytes per pixel")
	}
	p := &Plot{
		pageIndex:         uint32(pageIndex),
		plotIndex:         uint32(plotIndex),
		generationCounter: generationCounter,
		genID:             generationCounter.Next(),
		width:             width,
		height:            height,
		x:                 offX,
		y:                 offY,
		rectanizer:        MakeRectanizerSkyline(width, height),
		offsetX:           offX * width,
		offsetY:           offY * height,
		colorType:         colorType,
		bytesPerPixel:     bpp,
	}
	p.plotLocator = MakePlotLocator(p.pageIndex, p.plotIndex, p.genID)
	return p
}

// PageIndex returns the atlas page this plot belongs to.
func (p *Plot) PageIndex() uint32 { return p.pageIndex }

// PlotIndex returns a unique id for the plot relative to the owning atlas and page.
func (p *Plot) PlotIndex() uint32 { return p.plotIndex }

// GenID returns the plot's current generation: incremented when the plot is evicted due to an atlas spill; used to know
// if a particular subimage is still present in the atlas.
func (p *Plot) GenID() uint64 { return p.genID }

// PlotLocator returns the plot's current locator. Panics if the plot has never been given a valid locator.
func (p *Plot) PlotLocator() PlotLocator {
	if !p.plotLocator.IsValid() {
		panic("invalid plot locator")
	}
	return p.plotLocator
}

// Bpp returns the plot's bytes per pixel.
func (p *Plot) Bpp() int { return p.bytesPerPixel }

// AddRect attempts to reserve width x height texels in the plot's rectanizer, updating atlasLocator on success. To add
// data to the Plot, first call AddRect to see if it's possible; if successful, render into DataAt or copy with
// CopySubImage.
func (p *Plot) AddRect(width, height int, atlasLocator *AtlasLocator) bool {
	if width > p.width || height > p.height {
		panic("rect larger than plot")
	}
	locX, locY, ok := p.rectanizer.AddRect(width, height)
	if !ok {
		return false
	}
	rect := IRect16MakeXYWH(int16(locX), int16(locY), int16(width), int16(height))
	p.dirtyRect.Join(geom.IRectLTRB(int32(rect.Left), int32(rect.Top), int32(rect.Right),
		int32(rect.Bottom)))
	rect.Offset(int16(p.offsetX), int16(p.offsetY))
	atlasLocator.UpdateRect(rect)
	return true
}

// DataAt returns a slice of the backing store beginning at the locator's top-left texel (rows are bytesPerPixel*width
// apart).
func (p *Plot) DataAt(atlasLocator *AtlasLocator) []byte {
	if p.data == nil {
		p.data = make([]byte, p.bytesPerPixel*p.width*p.height)
	}
	topLeft := atlasLocator.TopLeft()
	// Assert we're accessing the correct Plot.
	if int(topLeft.X) < p.offsetX || int(topLeft.X) >= p.offsetX+p.width ||
		int(topLeft.Y) < p.offsetY || int(topLeft.Y) >= p.offsetY+p.height {
		panic("locator outside this plot")
	}
	x := int(topLeft.X) - p.offsetX
	y := int(topLeft.Y) - p.offsetY
	return p.data[p.bytesPerPixel*(p.width*y+x):]
}

// CopySubImage copies image into the plot at atlasLocator's position. image rows are tightly packed
// (width*bytesPerPixel). There is no RGBA<->BGRA swizzle lane: this codebase's N32 format is RGBA8888 everywhere.
func (p *Plot) CopySubImage(atlasLocator *AtlasLocator, image []byte) {
	dataPtr := p.DataAt(atlasLocator)
	width := int(atlasLocator.Width())
	height := int(atlasLocator.Height())
	rowBytes := width * p.bytesPerPixel
	plotRB := p.width * p.bytesPerPixel
	for i := range height {
		copy(dataPtr[i*plotRB:i*plotRB+rowBytes], image[i*rowBytes:(i+1)*rowBytes])
	}
}

// AddSubImage combines AddRect and CopySubImage: reserves space for width x height and copies image into it, returning
// false if the plot is full or has no room.
func (p *Plot) AddSubImage(width, height int, image []byte, atlasLocator *AtlasLocator) bool {
	if p.isFull || !p.AddRect(width, height, atlasLocator) {
		return false
	}
	p.CopySubImage(atlasLocator, image)
	return true
}

// LastUploadToken returns the token of the plot's last upload: if it hasn't been flushed to the GPU yet, a new upload
// can piggy-back on it.
func (p *Plot) LastUploadToken() Token { return p.lastUpload }

// LastUseToken returns the token of the plot's last use: if it has flushed through the GPU then the plot can be reused.
func (p *Plot) LastUseToken() Token { return p.lastUse }

// SetLastUploadToken records the token of the plot's most recent upload.
func (p *Plot) SetLastUploadToken(token Token) { p.lastUpload = token }

// SetLastUseToken records the token of the plot's most recent use.
func (p *Plot) SetLastUseToken(token Token) { p.lastUse = token }

// FlushesSinceLastUsed returns how many flushes have occurred since the plot was last used.
func (p *Plot) FlushesSinceLastUsed() int { return p.flushesSinceLastUse }

// ResetFlushesSinceLastUsed zeroes the flushes-since-last-used counter.
func (p *Plot) ResetFlushesSinceLastUsed() { p.flushesSinceLastUse = 0 }

// IncFlushesSinceLastUsed increments the flushes-since-last-used counter.
func (p *Plot) IncFlushesSinceLastUsed() { p.flushesSinceLastUse++ }

// NeedsUpload reports whether the plot has dirty data awaiting upload.
func (p *Plot) NeedsUpload() bool { return !p.dirtyRect.IsEmpty() }

// PrepareForUpload returns the data slice positioned at the dirty rect's origin (rows plotWidth*bpp apart) and the
// dirty rect in atlas coordinates, then clears the dirty state. Returns a nil slice when no backing data was ever
// allocated.
func (p *Plot) PrepareForUpload() (data []byte, offsetRect geom.IRect) {
	// We should only be issuing uploads if we are dirty.
	if p.dirtyRect.IsEmpty() {
		panic("prepareForUpload on a clean plot")
	}
	if p.data == nil {
		p.dirtyRect = geom.IRect{}
		p.isFull = false
		return nil, geom.IRect{}
	}
	rowBytes := p.bytesPerPixel * p.width
	// Clamp to 4-byte aligned boundaries.
	clearBits := int32(0x3 / p.bytesPerPixel)
	p.dirtyRect.Left &^= clearBits
	p.dirtyRect.Right += clearBits
	p.dirtyRect.Right &^= clearBits
	if p.dirtyRect.Right > int32(p.width) {
		panic("dirty rect exceeds plot width")
	}
	offset := rowBytes*int(p.dirtyRect.Top) + p.bytesPerPixel*int(p.dirtyRect.Left)
	data = p.data[offset:]
	offsetRect = p.dirtyRect.Offset(int32(p.offsetX), int32(p.offsetY))
	p.dirtyRect = geom.IRect{}
	p.isFull = false
	return data, offsetRect
}

// ResetRects re-initializes the Plot for reuse. The client should process any eviction callbacks first. If freeData is
// true the backing data is released as well (only when the caller knows it won't be adding to the Plot immediately
// afterwards).
func (p *Plot) ResetRects(freeData bool) {
	p.rectanizer.Reset()
	p.genID = p.generationCounter.Next()
	p.plotLocator = MakePlotLocator(p.pageIndex, p.plotIndex, p.genID)
	p.lastUpload = InvalidToken()
	p.lastUse = InvalidToken()

	if freeData {
		p.data = nil
	} else if p.data != nil {
		clear(p.data)
	}
	p.dirtyRect = geom.IRect{}
	p.isFull = false
}

// MarkFullIfUsed marks the plot full if it has any dirty (used) area.
func (p *Plot) MarkFullIfUsed() { p.isFull = !p.dirtyRect.IsEmpty() }

// IsEmpty reports whether the plot's rectanizer has never placed a rectangle.
func (p *Plot) IsEmpty() bool { return p.rectanizer.PercentFull() == 0 }

// HasAllocation reports whether the plot's backing data has been allocated.
func (p *Plot) HasAllocation() bool { return p.data != nil }

// Clone returns a fresh Plot with the same position and format, to take the place of the current plot in the atlas.
func (p *Plot) Clone() *Plot {
	return NewPlot(int(p.pageIndex), int(p.plotIndex), p.generationCounter, p.x, p.y, p.width,
		p.height, p.colorType, p.bytesPerPixel)
}

// PlotList is an intrusive doubly-linked LRU list of Plots (MRU at head, LRU at tail).
type PlotList struct {
	head *Plot
	tail *Plot
}

// Head returns the most recently used plot.
func (l *PlotList) Head() *Plot { return l.head }

// Tail returns the least recently used plot.
func (l *PlotList) Tail() *Plot { return l.tail }

// AddToHead inserts p at the head (most-recently-used end) of the list.
func (l *PlotList) AddToHead(p *Plot) {
	if p.list != nil {
		panic("plot already in a list")
	}
	p.prev = nil
	p.next = l.head
	if l.head != nil {
		l.head.prev = p
	}
	l.head = p
	if l.tail == nil {
		l.tail = p
	}
	p.list = l
}

// Remove detaches p from the list.
func (l *PlotList) Remove(p *Plot) {
	if p.list != l {
		panic("plot not in this list")
	}
	if p.prev != nil {
		p.prev.next = p.next
	} else {
		l.head = p.next
	}
	if p.next != nil {
		p.next.prev = p.prev
	} else {
		l.tail = p.prev
	}
	p.prev = nil
	p.next = nil
	p.list = nil
}

// Reset detaches every plot from the list.
func (l *PlotList) Reset() {
	for p := l.head; p != nil; {
		next := p.next
		p.prev = nil
		p.next = nil
		p.list = nil
		p = next
	}
	l.head = nil
	l.tail = nil
}

// Next exposes the intrusive next pointer for head-to-tail (MRU-to-LRU) iteration.
func (p *Plot) Next() *Plot { return p.next }
