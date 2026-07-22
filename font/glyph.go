// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The glyph model: PackedGlyphID (a glyph ID plus 2-bit sub-pixel position fields), RoundingSpec (the sub-pixel
// quantization rules for horizontal text), and the glyph itself — metrics, the mask image (A8 coverage, ARGB32 color,
// or LCD16), and the device-space outline path, all filled lazily by the strike.
//
// Reachable-set trims: outline glyphs are A8 by default; ARGB32 arrives from the color-glyph scaler lanes (sbix/CBDT
// PNG bitmaps and COLR layers — color emoji); LCD16 arrives when the Go API selects EdgingSubpixelAntiAlias on a
// surface with known pixel geometry (the public surface still has no edging setter, so text drawn through it stays A8).
// A BW format remains unreachable (EdgingAlias renders A8). Drawable glyphs are not produced by any reachable scaler
// and are omitted.

package font

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// PackedGlyphID packs a glyph ID with its sub-pixel position: bits [1:0] sub-pixel x, [17:2] glyph ID, [19:18]
// sub-pixel y.
type PackedGlyphID uint32

// Bit layout constants.
const (
	packedGlyphIDLen  = 16
	packedSubPixelLen = 2

	packedSubPixelXShift = 0
	packedGlyphIDShift   = packedSubPixelLen
	packedSubPixelYShift = packedGlyphIDLen + packedSubPixelLen
	packedEndData        = packedGlyphIDLen + 2*packedSubPixelLen

	packedGlyphIDMask     = 1<<packedGlyphIDLen - 1
	packedSubPixelPosMask = 1<<packedSubPixelLen - 1
	packedMaskAll         = 1<<packedEndData - 1
)

// SubpixelRound is the rounding constant for sub-pixel positioning (1/8).
const SubpixelRound = 1.0 / (1 << (packedSubPixelLen + 1))

// packedXYFieldMask is the packed-ID bit mask for the sub-pixel fields, per axis.
var packedXYFieldMask = geom.IPoint{
	X: packedSubPixelPosMask << packedSubPixelXShift,
	Y: packedSubPixelPosMask << packedSubPixelYShift,
}

// PackGlyphID packs a bare glyph ID with zero sub-pixel position.
func PackGlyphID(glyphID uint16) PackedGlyphID {
	return PackedGlyphID(uint32(glyphID) << packedGlyphIDShift)
}

// PackGlyphIDPoint packs a glyph ID with the sub-pixel position derived from pt: pt must already carry the rounding
// constant; mask selects which axes contribute sub-pixel bits.
func PackGlyphIDPoint(glyphID uint16, pt geom.Point, mask geom.IPoint) PackedGlyphID {
	const magicX = 1 << (packedSubPixelLen + packedSubPixelXShift)
	const magicY = 1 << (packedSubPixelLen + packedSubPixelYShift)
	x := float64(pt.X)
	y := float64(pt.Y)
	// Extend the modulo-1 range to [0,2) so truncation and floor agree; the mask removes the +1.
	x = x - math.Floor(x) + 1.0
	y = y - math.Floor(y) + 1.0
	subX := int32(x*magicX) & mask.X
	subY := int32(y*magicY) & mask.Y
	return PackedGlyphID(uint32(glyphID)<<packedGlyphIDShift | uint32(subX) | uint32(subY))
}

// GlyphID returns the bare glyph ID.
func (p PackedGlyphID) GlyphID() uint16 {
	return uint16((p >> packedGlyphIDShift) & packedGlyphIDMask)
}

// subPixelField extracts the 2-bit field at the given shift.
func (p PackedGlyphID) subPixelField(shift uint32) uint32 {
	return (uint32(p) >> shift) & packedSubPixelPosMask
}

// SubXOffset returns the sub-pixel x offset in pixels (0, 0.25, 0.5, 0.75).
func (p PackedGlyphID) SubXOffset() float32 {
	return float32(p.subPixelField(packedSubPixelXShift)) * 0.25
}

// SubYOffset returns the sub-pixel y offset in pixels.
func (p PackedGlyphID) SubYOffset() float32 {
	return float32(p.subPixelField(packedSubPixelYShift)) * 0.25
}

// AxisAlignment identifies which axis of a glyph position is rounded to whole pixels.
type AxisAlignment uint32

// AxisAlignment values.
const (
	AxisAlignmentNone AxisAlignment = iota
	AxisAlignmentX
	AxisAlignmentY
)

// RoundingSpec is the rounding constant added to a device position before flooring, and the packed-ID field mask
// selecting which axes keep sub-pixel bits.
type RoundingSpec struct {
	HalfAxisSampleFreq      geom.Point
	IgnorePositionFieldMask geom.IPoint
}

// NewRoundingSpec builds the rounding spec for the given subpixel/axis-alignment configuration.
func NewRoundingSpec(isSubpixel bool, axisAlignment AxisAlignment) RoundingSpec {
	return RoundingSpec{
		HalfAxisSampleFreq:      roundingHalfAxisSampleFreq(isSubpixel, axisAlignment),
		IgnorePositionFieldMask: roundingIgnorePositionFieldMask(isSubpixel, axisAlignment),
	}
}

func roundingHalfAxisSampleFreq(isSubpixel bool, axisAlignment AxisAlignment) geom.Point {
	if !isSubpixel {
		return geom.Pt(0.5, 0.5)
	}
	switch axisAlignment {
	case AxisAlignmentX:
		return geom.Pt(SubpixelRound, 0.5)
	case AxisAlignmentY:
		return geom.Pt(0.5, SubpixelRound)
	default:
		return geom.Pt(SubpixelRound, SubpixelRound)
	}
}

func roundingIgnorePositionFieldMask(isSubpixel bool, axisAlignment AxisAlignment) geom.IPoint {
	var mask geom.IPoint
	if isSubpixel && axisAlignment != AxisAlignmentY {
		mask.X = ^int32(0)
	}
	if isSubpixel && axisAlignment != AxisAlignmentX {
		mask.Y = ^int32(0)
	}
	return geom.IPoint{X: mask.X & packedXYFieldMask.X, Y: mask.Y & packedXYFieldMask.Y}
}

// maxGlyphWidth is the largest glyph width that gets an allocated image; the image stays nil at or above this.
const maxGlyphWidth = 1 << 13

// maxGlyphHeight is the largest glyph height that gets an allocated image; the image stays nil at or above this.
const maxGlyphHeight = 1 << 13

// MaskFormat identifies a glyph mask's pixel layout.
type MaskFormat uint8

// MaskFormat values.
const (
	MaskA8     MaskFormat = iota // 8-bit coverage
	MaskARGB32                   // premultiplied color pixels in the device (raster.Pixmap) word layout
	MaskLCD16                    // 565 words: 3-channel LCD coverage
	MaskSDF                      // 8-bit signed distance field
)

// bytesPerPixel returns the per-pixel byte width for the format.
func (f MaskFormat) bytesPerPixel() int32 {
	switch f {
	case MaskARGB32:
		return 4
	case MaskLCD16:
		return 2
	default:
		return 1
	}
}

// Glyph holds metrics plus the lazily-filled mask image and outline path. All fields are managed by the owning Strike
// under its lock.
type Glyph struct {
	// Outline path in device (strike) space. nil until PrepareForPath, and nil forever when the glyph has no outline.
	// pathIsHairline records that the path should be drawn as a hairline stroke rather than filled.
	pathVal *path.Path
	// Cached intercept computations.
	intercepts []*glyphIntercept
	// Mask image. A8 glyphs fill Image (Width*Height bytes, row-major, rowBytes == Width); ARGB32 glyphs fill Image32
	// (Width*Height premultiplied device words, row-major, matching raster.Pixmap's layout); LCD16 glyphs fill Image16
	// (Width*Height 565 words, row-major). All are nil until PrepareForImage, and nil forever for empty or too-large
	// glyphs.
	Image   []uint8
	Image32 []uint32
	Image16 []uint16
	// Metrics, valid after creation (generateMetrics + bounds resolution).
	AdvanceX float32
	AdvanceY float32
	Left     int32
	Top      int32
	Width    int32
	Height   int32
	packedID PackedGlyphID
	// scalerBits records which scaler lane produced this glyph (the color lanes record themselves here so getImage
	// regenerates through the same lane).
	scalerBits     uint8
	imageDone      bool
	pathIsHairline bool
	pathDone       bool
	// Format is the mask format for the reachable formats (A8 coverage, ARGB32 color).
	Format MaskFormat
}

// glyphIntercept caches a bounds/interval pair from a font-intercept computation.
type glyphIntercept struct {
	bounds   [2]float32
	interval [2]float32
}

// PackedID returns the glyph's packed ID.
func (g *Glyph) PackedID() PackedGlyphID { return g.packedID }

// IsEmpty reports whether the glyph has zero width or height.
func (g *Glyph) IsEmpty() bool { return g.Width == 0 || g.Height == 0 }

// imageTooLarge reports whether the glyph is too large in either dimension to allocate an image.
func (g *Glyph) imageTooLarge() bool { return g.Width >= maxGlyphWidth || g.Height >= maxGlyphHeight }

// RowBytes returns the byte stride of one mask row for the glyph's format.
func (g *Glyph) RowBytes() int32 { return g.Width * g.Format.bytesPerPixel() }

// ImageSize returns the mask image size in bytes.
func (g *Glyph) ImageSize() int {
	if g.IsEmpty() || g.imageTooLarge() {
		return 0
	}
	return int(g.RowBytes()) * int(g.Height)
}

// HasImage reports whether PrepareForImage produced a mask (whichever plane the format uses).
func (g *Glyph) HasImage() bool {
	switch g.Format {
	case MaskARGB32:
		return g.Image32 != nil
	case MaskLCD16:
		return g.Image16 != nil
	default:
		return g.Image != nil
	}
}

// Pixmap returns the ARGB32 image as a raster.Pixmap positioned at the origin (the wrap the color-glyph draw lanes
// use). Valid only for ARGB32 glyphs with an image.
func (g *Glyph) Pixmap() raster.Pixmap {
	return raster.Pixmap{Pix: g.Image32, Width: g.Width, Height: g.Height, RowPixels: g.Width}
}

// IRect returns the glyph bounds as an integer rect.
func (g *Glyph) IRect() geom.IRect {
	return geom.IRectLTRB(g.Left, g.Top, g.Left+g.Width, g.Top+g.Height)
}

// Rect returns the glyph bounds as a float rect.
func (g *Glyph) Rect() geom.Rect {
	return geom.RectLTRB(float32(g.Left), float32(g.Top), float32(g.Left+g.Width), float32(g.Top+g.Height))
}

// MaxDimension returns the larger of the glyph's width and height.
func (g *Glyph) MaxDimension() int32 { return max(g.Width, g.Height) }

// Mask returns the glyph's mask positioned at pos; pos must be integral. For ARGB32 glyphs the Image plane is nil (the
// color pixels live in Image32/Pixmap) — only Bounds is meaningful, and the painter routes those to the sprite lane
// before any blitter sees the mask. LCD16 glyphs carry their 565 plane in Image16 with the matching raster.MaskLCD16
// format tag.
func (g *Glyph) Mask(pos geom.Point) raster.Mask {
	b := g.IRect().Offset(int32(math.Floor(float64(pos.X))), int32(math.Floor(float64(pos.Y))))
	format := raster.MaskA8
	if g.Format == MaskLCD16 {
		format = raster.MaskLCD16
	}
	return raster.Mask{
		Image: g.Image, Image16: g.Image16, Bounds: b, RowBytes: g.RowBytes(),
		Format: format,
	}
}

// Path returns the glyph's device-space outline, or nil (valid only after PrepareForPath).
func (g *Glyph) Path() *path.Path { return g.pathVal }

// PathIsHairline reports whether the path should be drawn as a hairline (valid only after PrepareForPath).
func (g *Glyph) PathIsHairline() bool { return g.pathIsHairline }

// zeroBounds clears the bounds only (advance survives).
func (g *Glyph) zeroBounds() {
	g.Left = 0
	g.Top = 0
	g.Width = 0
	g.Height = 0
}

// setPath installs the glyph's path.
func (g *Glyph) setPath(p *path.Path, hairline bool) {
	if !g.pathDone {
		g.pathVal = p
		g.pathIsHairline = hairline
		g.pathDone = true
	}
}

// approximatePathBytes estimates the path's memory for the strike budget.
func approximatePathBytes(p *path.Path) int {
	if p == nil {
		return 0
	}
	return 64 + p.CountPoints()*8 + p.CountVerbs()
}
