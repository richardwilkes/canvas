// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The drawing-strike scaler: the rec built by MakeRecAndEffects (font + paint + device matrix), the glyph
// metrics/path/image generation, and the strike-level mask filter application. There is exactly one glyph host: the
// typeface's sfnt outlines through go-text/typesetting, rendered by the raster package — the unhinted FreeType-style
// recipe. The color glyph lanes use the same host: COLRv0 layers filled as device-space paths with CPAL/foreground
// colors, and sbix/CBDT PNG strikes decoded and scaled through the image shader, both into ARGB32 masks.
//
// Reachable-set trims: outline glyphs are A8 by default; color glyphs are ARGB32; LCD16 arrives when the font's edging
// is subpixel-AA and the destination's pixel geometry is known, rendered through the LCD lane (the 4x horizontal
// oversample + pack4xHToMask FIR, since the only rasterizer is the path rasterizer) with the mask-gamma pre-blend. A8
// masks stay linear coverage (the pre-blend applies only to LCD recs); that rule is hoisted into rec construction, so
// only LCD16 recs carry a luminance color (LumBits) — A8 strikes stay color-independent, where keying every rec on the
// canonical color would fragment strikes with no pixel difference. The device gamma and contrast rec fields have no
// reachable variation (the surface-props text contrast/gamma constructor is not exposed) and stay the defaults inside
// maskgamma.go. Embolden (fake bold) and the embedded-bitmap request are recorded on the Font but have no lane here
// (there is no synthetic-bold generator, and bitmap strikes are decoded whenever the typeface carries them), so neither
// reaches the rec: keying strikes on a request that changes no pixel would only fragment the cache.

package font

import (
	"math"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/font/opentype/tables"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stroke"
)

// Scaler rec flags (the reachable subset).
const (
	recFlagFrameAndFill  = 1 << 0 // stroke-and-fill style
	recFlagSubpixel      = 1 << 1 // subpixel glyph positioning
	recFlagLinearMetrics = 1 << 2 // linearly scaled metrics
	recFlagBaselineSnap  = 1 << 3 // snap the baseline to pixels
	recFlagLCDVertical   = 1 << 4 // vertical LCD stripes: else horizontal
	recFlagLCDBGROrder   = 1 << 5 // BGR LCD subpixel order: else RGB
	recFlagGenA8FromLCD  = 1 << 6 // filter the path-generated A8 as the LCD lane would
	recFlagAliased       = 1 << 7 // rasterize hard-edged coverage: EdgingAlias
)

// PixelGeometry describes the destination's subpixel layout, with the same ordinals as surface.PixelGeometry (the font
// package cannot import surface — the import chain runs the other way — so the enum is duplicated here).
type PixelGeometry uint8

// PixelGeometry values.
const (
	PixelGeometryUnknown PixelGeometry = iota
	PixelGeometryRGBH
	PixelGeometryBGRH
	PixelGeometryRGBV
	PixelGeometryBGRV
)

// DeviceProps carries the surface properties MakeRecAndEffects consumes: the destination's pixel geometry (which gates
// LCD16) and the device-independent-fonts flag (which disables LCD — that flag conventionally routes text through SDFT,
// whose fonts use plain AA edging; the outcome is encoded directly here until E.1 lands SDFT). A nil *DeviceProps means
// no device: unknown geometry.
type DeviceProps struct {
	PixelGeometry             PixelGeometry
	UseDeviceIndependentFonts bool
}

// Beyond this size LCD doesn't appreciably improve quality but always costs more RAM and draws slower, so it is capped.
const maxSizeForLCDText = 48

// ScalerRec is the scaler configuration record. It is comparable and forms the strike key together with the effects.
type ScalerRec struct {
	TypefaceID uint32
	TextSize   float32
	PreScaleX  float32
	PreSkewX   float32
	// Post2x2 is the device-matrix 2x2 in row-major order: [scaleX, skewX, skewY, scaleY].
	Post2x2    [4]float32
	FrameWidth float32 // >= 0 when stroking; -1 for fill
	MiterLimit float32
	StrokeJoin stroke.Join
	StrokeCap  stroke.Cap
	Flags      uint16
	Hinting    Hinting
	// Format is the mask format for the outline lane: MaskA8 or MaskLCD16 (the color lanes override it per glyph). The
	// zero value is MaskA8, keeping every pre-E.4 rec construction site an A8 rec.
	Format MaskFormat
	// LumBits is the canonical luminance color keying the mask pre-blend. Nonzero only for LCD16 recs (non-LCD recs
	// ignore the pre-blend, hoisted here so A8 strikes stay color-independent; see the file comment).
	LumBits colorcore.Color
	// ForegroundColor is the paint color, entering the rec (and therefore the strike key) only when the typeface's
	// glyph masks need it (COLR layers with palette index 0xFFFF); black otherwise.
	ForegroundColor colorcore.Color
}

// ScalerEffects carries the path and mask effects that accompany a rec.
type ScalerEffects struct {
	PathEffect stroke.PathEffect
	MaskFilter maskfilter.MaskFilter
}

// ScalerPaint carries the paint fields MakeRecAndEffects consumes (the slice of the paint relevant to text).
type ScalerPaint struct {
	PathEffect stroke.PathEffect
	MaskFilter maskfilter.MaskFilter
	Style      stroke.PaintStyle
	Width      float32
	MiterLimit float32
	Cap        stroke.Cap
	Join       stroke.Join
	// Color is the paint color; it reaches the rec only for typefaces whose glyph masks need the current color. The
	// zero value is transparent; a nil *ScalerPaint means the default paint, whose color is black.
	Color colorcore.Color
	// LumColor is the color the mask pre-blend is keyed on. The canvas paint computes it (paint color, run through the
	// color filter; 50% gray for shader paints) because the scaler paint does not carry the shader/filter objects.
	// Consumed only by LCD16 recs.
	LumColor colorcore.Color
}

// skRound1024 rounds to 1/1024 precision: limited fractional precision for the post 2x2, consolidating strikes whose
// matrices differ only slightly.
func skRound1024(x float32) float32 {
	return float32(math.Floor(float64(x)*1024+0.5)) / 1024
}

// tooBigForLCD reports whether the effective text size exceeds maxSizeForLCDText, disabling the LCD lane.
func tooBigForLCD(rec *ScalerRec, checkPost2x2 bool) bool {
	if checkPost2x2 {
		area := rec.Post2x2[0]*rec.Post2x2[3] - rec.Post2x2[2]*rec.Post2x2[1]
		area *= rec.TextSize * rec.TextSize
		return area > maxSizeForLCDText*maxSizeForLCDText
	}
	return rec.TextSize > maxSizeForLCDText
}

// MakeRecAndEffects builds the scaler rec and effects from the font, paint, and device matrix (paint may be nil,
// meaning a default paint; props may be nil, meaning no device — unknown pixel geometry).
func MakeRecAndEffects(f *Font, paint *ScalerPaint, deviceMatrix *geom.Matrix, props *DeviceProps) (ScalerRec, ScalerEffects) {
	var rec ScalerRec
	var effects ScalerEffects

	rec.TypefaceID = f.typeface.UniqueID()
	rec.TextSize = f.size
	rec.PreScaleX = f.scaleX
	rec.PreSkewX = f.skewX

	checkPost2x2 := false
	mask := deviceMatrix.Type()
	if mask&geom.TypeScale != 0 {
		rec.Post2x2[0] = skRound1024(deviceMatrix.Get(geom.MScaleX))
		rec.Post2x2[3] = skRound1024(deviceMatrix.Get(geom.MScaleY))
		checkPost2x2 = true
	} else {
		rec.Post2x2[0] = 1
		rec.Post2x2[3] = 1
	}
	if mask&geom.TypeAffine != 0 {
		rec.Post2x2[1] = skRound1024(deviceMatrix.Get(geom.MSkewX))
		rec.Post2x2[2] = skRound1024(deviceMatrix.Get(geom.MSkewY))
		checkPost2x2 = true
	}

	if paint != nil && paint.Style != stroke.PaintStyleFill && paint.Width >= 0 {
		rec.FrameWidth = paint.Width
		rec.MiterLimit = paint.MiterLimit
		rec.StrokeJoin = paint.Join
		rec.StrokeCap = paint.Cap
		if paint.Style == stroke.PaintStyleStrokeAndFill {
			rec.Flags |= recFlagFrameAndFill
		}
	} else {
		rec.FrameWidth = -1
	}

	// Every float above came straight from caller-supplied values, so scrub NaN out before the rec becomes a strike key
	// (and before the gates below read TextSize and Post2x2).
	rec.canonicalizeKeyFloats()

	// Mask-format selection: alias and antialias edging both render A8 (there is no BW mask format — aliased edging
	// rasterizes hard-edged coverage into the A8 plane instead, recFlagAliased); subpixel-AA requests LCD16, subject to
	// the gates below, including the device-independent-fonts disable (see DeviceProps).
	rec.Format = MaskA8
	if f.edging == EdgingAlias {
		rec.Flags |= recFlagAliased
	}
	if f.edging == EdgingSubpixelAntiAlias {
		rec.Format = MaskLCD16
		switch {
		case tooBigForLCD(&rec, checkPost2x2):
			rec.Format = MaskA8
			rec.Flags |= recFlagGenA8FromLCD
		case props == nil || props.PixelGeometry == PixelGeometryUnknown ||
			props.UseDeviceIndependentFonts:
			// Eeek, can't support LCD.
			rec.Format = MaskA8
			rec.Flags |= recFlagGenA8FromLCD
		default:
			switch props.PixelGeometry {
			case PixelGeometryRGBH:
				// Our default, do nothing.
			case PixelGeometryBGRH:
				rec.Flags |= recFlagLCDBGROrder
			case PixelGeometryRGBV:
				rec.Flags |= recFlagLCDVertical
			case PixelGeometryBGRV:
				rec.Flags |= recFlagLCDVertical | recFlagLCDBGROrder
			}
		}
	}

	// Only LCD16 recs consume the pre-blend (see the file comment), so only they carry the canonical color in the key.
	if rec.Format == MaskLCD16 {
		lum := colorcore.RGB(0, 0, 0)
		if paint != nil {
			lum = paint.LumColor
		}
		rec.LumBits = MaskGammaCanonicalColor(lum)
	}

	if f.Subpixel() {
		rec.Flags |= recFlagSubpixel
	}
	if f.flags&flagLinearMetrics != 0 {
		rec.Flags |= recFlagLinearMetrics
	}
	if f.flags&flagBaselineSnap != 0 {
		rec.Flags |= recFlagBaselineSnap
	}
	rec.Hinting = f.hinting

	// The paint color enters the rec (and the strike key) only when the typeface's glyph masks may paint with it, so
	// ordinary fonts never fragment strikes per color.
	if f.typeface.GlyphMaskNeedsCurrentColor() {
		rec.ForegroundColor = colorcore.RGB(0, 0, 0)
		if paint != nil {
			rec.ForegroundColor = paint.Color
		}
	}

	if paint != nil {
		effects.PathEffect = paint.PathEffect
		effects.MaskFilter = paint.MaskFilter
	}
	return rec, effects
}

// canonicalizeKeyFloats replaces any NaN among the rec's float fields with a well-defined finite value. The rec is half
// of the strike cache's map key and a struct holding a NaN never equals itself, so a poisoned rec would miss on every
// lookup and never match the delete in removeStrike: the strike and every glyph mask it generated would be retained
// forever while the cache's accounting reported them freed, beyond the reach of either budget. Infinities need no such
// treatment — they are self-equal, so they key correctly, and computeScale's non-finite gate already collapses them to
// a degenerate strike. (Upstream Skia is immune because its key is a memcmp'd byte blob rather than a comparable
// struct.)
func (r *ScalerRec) canonicalizeKeyFloats() {
	// A NaN size, pre-scale, or pre-skew becomes zero, which computeScale then reports as singular, so the glyphs come
	// out empty rather than nonsensical.
	r.TextSize = zeroNaN(r.TextSize)
	r.PreScaleX = zeroNaN(r.PreScaleX)
	r.PreSkewX = zeroNaN(r.PreSkewX)
	for i := range r.Post2x2 {
		r.Post2x2[i] = zeroNaN(r.Post2x2[i])
	}
	// FrameWidth's sentinel for "not stroking" is -1; zero would mean a hairline frame.
	if math.IsNaN(float64(r.FrameWidth)) {
		r.FrameWidth = -1
	}
	r.MiterLimit = zeroNaN(r.MiterLimit)
}

// zeroNaN returns v, or zero when v is NaN.
func zeroNaN(v float32) float32 {
	if math.IsNaN(float64(v)) {
		return 0
	}
	return v
}

// localMatrix returns the text matrix: size and pre-scale/skew, without the device 2x2.
func (r *ScalerRec) localMatrix() geom.Matrix {
	var m geom.Matrix
	m.SetAll(r.TextSize*r.PreScaleX, r.TextSize*r.PreSkewX, 0, 0, r.TextSize, 0, 0, 0, 1)
	return m
}

// matrixFrom2x2 returns the device 2x2 as a full matrix.
func (r *ScalerRec) matrixFrom2x2() geom.Matrix {
	var m geom.Matrix
	m.SetAll(r.Post2x2[0], r.Post2x2[1], 0, r.Post2x2[2], r.Post2x2[3], 0, 0, 0, 1)
	return m
}

// singleMatrix returns the combined text matrix: localMatrix post-concatenated with the device 2x2.
func (r *ScalerRec) singleMatrix() geom.Matrix {
	local := r.localMatrix()
	post := r.matrixFrom2x2()
	local.PostConcat(&post)
	return local
}

// computeScale extracts the scale from the single matrix: remove rotation via a Givens rotation and return the
// remaining scales' magnitudes. ok=false is the singular gate (scales nearly zero, or non-finite) — glyphs then have
// zero advances/bounds.
func (r *ScalerRec) computeScale() (geom.Point, bool) {
	a := r.singleMatrix()
	sx := float64(a.Get(geom.MScaleX))
	sy := float64(a.Get(geom.MScaleY))
	kx := float64(a.Get(geom.MSkewX))
	ky := float64(a.Get(geom.MSkewY))
	if kx != 0 || ky != 0 || sx < 0 || sy < 0 {
		// GA = G * A where G rotates A's mapped x-axis onto the x-axis (a Givens rotation).
		hx := sx
		hy := ky
		h := math.Hypot(hx, hy)
		if h == 0 || !isFinite(h) {
			return geom.Point{}, false
		}
		c := hx / h
		s := hy / h
		gaScaleX := c*sx + s*ky
		gaScaleY := -s*kx + c*sy
		if !isFinite(gaScaleX) || !isFinite(gaScaleY) ||
			math.Abs(gaScaleX) <= nearlyZero || math.Abs(gaScaleY) <= nearlyZero {
			return geom.Point{}, false
		}
		return geom.Pt(float32(math.Abs(gaScaleX)), float32(math.Abs(gaScaleY))), true
	}
	if !isFinite(sx) || !isFinite(sy) || !isFinite(kx) ||
		math.Abs(sx) <= nearlyZero || math.Abs(sy) <= nearlyZero {
		return geom.Point{}, false
	}
	return geom.Pt(float32(math.Abs(sx)), float32(math.Abs(sy))), true
}

// singular reports whether the single matrix is effectively singular.
func (r *ScalerRec) singular() bool {
	_, ok := r.computeScale()
	return !ok
}

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// computeAxisAlignmentForHText reports which axis horizontal text can snap to under this rec's 2x2.
func (r *ScalerRec) computeAxisAlignmentForHText() AxisAlignment {
	if r.Flags&recFlagBaselineSnap == 0 {
		return AxisAlignmentNone
	}
	if r.Post2x2[2] == 0 {
		return AxisAlignmentX
	}
	if r.Post2x2[0] == 0 {
		return AxisAlignmentY
	}
	return AxisAlignmentNone
}

// isSubpixel reports whether subpixel positioning is enabled.
func (r *ScalerRec) isSubpixel() bool { return r.Flags&recFlagSubpixel != 0 }

// ScalerContext generates glyph metrics, paths, and images for one rec against the sfnt outline host.
type ScalerContext struct {
	// preBlend is the mask-gamma pre-blend: non-applicable with a mask filter (the pre-blend is not applied to filtered
	// text) or for non-LCD recs (LumBits is only set for LCD16).
	preBlend   maskPreBlend
	pathEffect stroke.PathEffect
	maskFilter maskfilter.MaskFilter
	typeface   *Typeface
	rec        ScalerRec
	single     geom.Matrix // cached rec.singleMatrix()
	strikePpem uint16      // requested bitmap-strike ppem derived from the y scale
	// generateImageFromPath: styled glyphs must render through their (styled) path.
	generateImageFromPath bool
	isSing                bool // cached rec.singular()
}

// NewScalerContext builds a scaler context for the rec and effects.
func NewScalerContext(typeface *Typeface, rec ScalerRec, effects ScalerEffects) *ScalerContext {
	scale, ok := rec.computeScale()
	var preBlend maskPreBlend
	if effects.MaskFilter == nil && rec.Format == MaskLCD16 {
		preBlend = getMaskPreBlend(rec.LumBits)
	}
	return &ScalerContext{
		rec:                   rec,
		typeface:              typeface,
		pathEffect:            effects.PathEffect,
		maskFilter:            effects.MaskFilter,
		generateImageFromPath: rec.FrameWidth >= 0 || effects.PathEffect != nil,
		preBlend:              preBlend,
		single:                rec.singleMatrix(),
		isSing:                !ok,
		strikePpem:            strikePpemFor(scale.Y),
	}
}

// strikePpemFor computes the requested bitmap-strike size: the y scale is quantized to 26.6 (truncating) and compared
// against the strikes' integer ppems, so the effective request is the ceiling of the truncated value, clamped to uint16
// (the smallest strike >= the request wins, else the largest — typesetting's chooseStrike implements that rule).
func strikePpemFor(scaleY float32) uint16 {
	if !(scaleY > 0) {
		return 0
	}
	fdot6 := float64(scaleY) * 64
	if fdot6 >= (math.MaxUint16+1)*64 {
		return math.MaxUint16
	}
	t := int64(fdot6) // truncating 26.6 quantization
	ppem := t >> 6
	if t&63 != 0 {
		ppem++
	}
	if ppem <= 0 {
		return 1
	}
	if ppem > math.MaxUint16 {
		return math.MaxUint16
	}
	return uint16(ppem)
}

// Rec returns the context's rec.
func (c *ScalerContext) Rec() *ScalerRec { return &c.rec }

// Typeface returns the context's typeface.
func (c *ScalerContext) Typeface() *Typeface { return c.typeface }

// mapDesignPoint maps a design-space point (font units, y up) through the single matrix into device space (y down).
func (c *ScalerContext) mapDesignPoint(x, y float32) geom.Point {
	upem := float32(c.typeface.upem)
	return c.single.MapXY(x/upem, -y/upem)
}

// generatePath builds the glyph outline in device space (the single matrix applied), contours closed. Returns nil when
// the glyph has no outline.
func (c *ScalerContext) generatePath(gid uint16) *path.Path {
	t := c.typeface
	if t.face == nil || t.upem <= 0 || int(gid) >= t.nGlyphs || c.isSing {
		return nil
	}
	return glyphOutlinePath(t, gid, c.mapDesignPoint)
}

// scalerBits values: which scaler lane produced a glyph.
const (
	scalerBitsNone   uint8 = iota
	scalerBitsCOLRv0       // COLRv0 layers filled as paths
	scalerBitsCOLRv1       // COLRv1 paint graph interpreted onto the mask (colorglyph_v1.go)
	scalerBitsBitmap       // sbix/CBDT/EBDT PNG strike scaled through the image shader
)

// glyphMetrics is what generateMetrics reports back to makeGlyph.
type glyphMetrics struct {
	advanceX, advanceY float32
	bounds             geom.Rect
	maskFormat         MaskFormat
	extraBits          uint8
	neverRequestPath   bool
}

// generateMetrics computes a glyph's metrics through its lane: the COLRv0 layer union, the bitmap-strike quad, or the
// outline control box, plus the linear advance through the single matrix (unhinted; bitmap-font advances stay linear
// too).
func (c *ScalerContext) generateMetrics(packedID PackedGlyphID) glyphMetrics {
	var mx glyphMetrics
	gid := packedID.GlyphID()
	t := c.typeface

	// Linear advance through the single matrix.
	adv := t.faceHAdvance(opentype.GID(gid)) / float32(t.upem)
	advVec := c.single.MapVector(geom.Pt(adv, 0))
	mx.advanceX = advVec.X
	mx.advanceY = advVec.Y

	// COLR lanes (checked first). COLRv1 paint graphs (preferred over v0 records when both exist, per the spec's search
	// order) take their bounding box from the ClipList when present, else a measuring traversal of the paint graph;
	// COLRv0 layer lists union the layer outlines' control boxes in device space.
	if paint, ok := t.faceColorPaint(opentype.GID(gid)); ok {
		var bounds geom.Rect
		if layers, isV0 := paint.(tables.PaintColrLayersResolved); isV0 {
			for _, layer := range layers {
				if layerPath := glyphOutlinePath(t, layer.GlyphID, c.mapDesignPoint); layerPath != nil {
					if b := layerPath.Bounds(); !b.IsEmpty() {
						bounds.Join(b)
					}
				}
			}
			mx.extraBits = scalerBitsCOLRv0
		} else {
			bounds = c.colrV1Bounds(gid)
			mx.extraBits = scalerBitsCOLRv1
		}
		mx.maskFormat = MaskARGB32
		mx.neverRequestPath = true
		mx.bounds = c.offsetBoundsIfSubpixel(packedID, bounds, c.rec.isSubpixel())
		return mx
	}

	// Bitmap-strike lane (sbix/CBDT/EBDT PNG only — B&W strikes and sbix jpg/tif graphic types fall through to the
	// outline lane): the strike glyph's font-unit extents are mapped through the single matrix to produce the
	// strike-pixel bounds.
	if t.hasBitmaps {
		if bm, ext, ok := t.faceBitmapGlyph(opentype.GID(gid), c.strikePpem); ok && bm.Format == tsfont.PNG {
			bounds := c.mapExtentsQuad(ext)
			mx.maskFormat = MaskARGB32
			mx.extraBits = scalerBitsBitmap
			mx.neverRequestPath = true
			// Bitmap glyphs only occur for effectively non-scalable faces, where sub-pixel positioning of the resampled
			// bitmap is always allowed.
			mx.bounds = c.offsetBoundsIfSubpixel(packedID, bounds, c.rec.isSubpixel())
			return mx
		}
	}

	// The outline control-box lane: map the extents quad through the single matrix, offset by the sub-pixel position,
	// then the LCD outset (updateGlyphBoundsIfLCD).
	mx.maskFormat = c.rec.Format
	ext, ok := t.faceGlyphExtents(opentype.GID(gid))
	if ok && (ext.Width != 0 || ext.Height != 0) {
		mx.bounds = c.offsetBoundsIfSubpixel(packedID, c.mapExtentsQuad(ext), c.rec.isSubpixel())
		mx.bounds = c.updateGlyphBoundsIfLCD(mx.maskFormat, mx.bounds)
	}
	return mx
}

// updateGlyphBoundsIfLCD gives LCD16 masks one extra pixel on each side in the sampling direction (the FIR filter
// spills coverage into the neighbors).
func (c *ScalerContext) updateGlyphBoundsIfLCD(format MaskFormat, r geom.Rect) geom.Rect {
	if format != MaskLCD16 || r.IsEmpty() {
		return r
	}
	r = r.RoundOutRect()
	if c.rec.Flags&recFlagLCDVertical != 0 {
		r.Top--
		r.Bottom++
	} else {
		r.Left--
		r.Right++
	}
	return r
}

// mapExtentsQuad maps a font-unit extents box through the single matrix and returns the bounding rect of the mapped
// quad.
func (c *ScalerContext) mapExtentsQuad(ext tsfont.GlyphExtents) geom.Rect {
	p0 := c.mapDesignPoint(ext.XBearing, ext.YBearing)
	p1 := c.mapDesignPoint(ext.XBearing+ext.Width, ext.YBearing)
	p2 := c.mapDesignPoint(ext.XBearing, ext.YBearing+ext.Height)
	p3 := c.mapDesignPoint(ext.XBearing+ext.Width, ext.YBearing+ext.Height)
	return geom.RectLTRB(
		min(p0.X, p1.X, p2.X, p3.X),
		min(p0.Y, p1.Y, p2.Y, p3.Y),
		max(p0.X, p1.X, p2.X, p3.X),
		max(p0.Y, p1.Y, p2.Y, p3.Y),
	)
}

// offsetBoundsIfSubpixel offsets the bounds by the packed sub-pixel position.
func (c *ScalerContext) offsetBoundsIfSubpixel(packedID PackedGlyphID, r geom.Rect, subpixel bool) geom.Rect {
	if subpixel && !r.IsEmpty() {
		dx := packedID.SubXOffset()
		dy := packedID.SubYOffset()
		r.Left += dx
		r.Right += dx
		r.Top += dy
		r.Bottom += dy
	}
	return r
}

// makeGlyph builds a glyph: metrics, then bounds — from the styled path when generateImageFromPath (and the lane
// requests paths), else from the lane's reported bounds — then the mask-filter bounds adjustment.
func (c *ScalerContext) makeGlyph(packedID PackedGlyphID) *Glyph {
	g := &Glyph{packedID: packedID}
	gid := packedID.GlyphID()
	t := c.typeface

	if t.face == nil || t.upem <= 0 || int(gid) >= t.nGlyphs || c.isSing {
		g.setPath(nil, false)
		return g
	}

	mx := c.generateMetrics(packedID)
	g.AdvanceX = mx.advanceX
	g.AdvanceY = mx.advanceY
	g.Format = mx.maskFormat
	g.scalerBits = mx.extraBits

	if c.generateImageFromPath && !mx.neverRequestPath {
		c.internalGetPath(g)
		if devPath := g.Path(); devPath != nil {
			doVert := c.rec.Flags&recFlagLCDVertical != 0
			a8LCD := c.rec.Flags&recFlagGenA8FromLCD != 0
			generateMetricsFromPath(g, devPath, doVert, a8LCD, g.pathIsHairline)
		}
	} else {
		if !mx.bounds.IsEmpty() {
			saturateGlyphBounds(g, mx.bounds)
		}
		if mx.neverRequestPath {
			g.setPath(nil, false)
		}
	}

	// If either dimension is empty, zap the image bounds of the glyph.
	if g.Width == 0 || g.Height == 0 {
		g.zeroBounds()
		return g
	}

	// The SDF pad: the SDF lane is the rec format itself (MakeSDFTMaskSpec), so the bounds are outset by
	// DistanceFieldPad here directly; the format was already MaskSDF from generateMetrics. Color lanes (COLR/bitmap)
	// override the format per glyph and take no pad.
	if g.Format == MaskSDF {
		g.Left -= DistanceFieldPad
		g.Top -= DistanceFieldPad
		g.Width += 2 * DistanceFieldPad
		g.Height += 2 * DistanceFieldPad
	}

	// The mask-filter bounds pass. On an ARGB32 source only the blur filter succeeds (the box blur accepts color masks,
	// extracting the alpha plane; the table/shader filters guard on A8 and return false), and a successful filter's dst
	// is A8 — the glyph then adopts that format.
	if c.maskFilter != nil && (g.Format != MaskARGB32 || maskfilter.AcceptsColorMask(c.maskFilter)) {
		// Only want the bounds from the filter.
		src := raster.Mask{Bounds: g.IRect(), RowBytes: g.RowBytes()}
		m := c.rec.matrixFrom2x2()
		if dst, _, ok := c.maskFilter.FilterMask(&src, &m); ok {
			if dst.Bounds.IsEmpty() {
				g.zeroBounds()
				return g
			}
			saturateGlyphBoundsI(g, dst.Bounds)
			g.Format = MaskA8
		}
	}
	return g
}

// saturateGlyphBounds stores the rounded-out bounds into the glyph, saturating to 16-bit limits.
func saturateGlyphBounds(g *Glyph, r geom.Rect) {
	out := r.RoundOutRect()
	g.Left = int32(satInt16(out.Left))
	g.Top = int32(satInt16(out.Top))
	g.Width = int32(satUint16(out.Right - out.Left))
	g.Height = int32(satUint16(out.Bottom - out.Top))
}

// saturateGlyphBoundsI is saturateGlyphBounds for integer bounds.
func saturateGlyphBoundsI(g *Glyph, r geom.IRect) {
	g.Left = int32(satInt16I(r.Left))
	g.Top = int32(satInt16I(r.Top))
	g.Width = int32(satUint16I(int64(r.Right) - int64(r.Left)))
	g.Height = int32(satUint16I(int64(r.Bottom) - int64(r.Top)))
}

func satInt16(v float32) int16 {
	if v >= math.MaxInt16 {
		return math.MaxInt16
	}
	if v <= math.MinInt16 {
		return math.MinInt16
	}
	return int16(v)
}

func satUint16(v float32) uint16 {
	if v >= math.MaxUint16 {
		return math.MaxUint16
	}
	if v <= 0 {
		return 0
	}
	return uint16(v)
}

func satInt16I(v int32) int16 {
	if v >= math.MaxInt16 {
		return math.MaxInt16
	}
	if v <= math.MinInt16 {
		return math.MinInt16
	}
	return int16(v)
}

func satUint16I(v int64) uint16 {
	if v >= math.MaxUint16 {
		return math.MaxUint16
	}
	if v <= 0 {
		return 0
	}
	return uint16(v)
}

// generateMetricsFromPath computes bounds from the device path's control bounds, with the LCD-sampling and hairline
// outsets. The fromLCD outset applies to LCD16 masks and to A8 masks generated through the LCD filter lane
// (recFlagGenA8FromLCD).
func generateMetricsFromPath(g *Glyph, devPath *path.Path, verticalLCD, a8FromLCD, hairline bool) {
	bounds := devPath.Bounds()
	if !bounds.IsEmpty() {
		fromLCD := g.Format == MaskLCD16 || (g.Format == MaskA8 && a8FromLCD)
		needExtraWidth := (fromLCD && !verticalLCD) || hairline
		needExtraHeight := (fromLCD && verticalLCD) || hairline
		if needExtraWidth {
			bounds = bounds.RoundOutRect()
			bounds.Left--
			bounds.Right++
		}
		if needExtraHeight {
			bounds = bounds.RoundOutRect()
			bounds.Top--
			bounds.Bottom++
		}
	}
	saturateGlyphBounds(g, bounds)
}

// internalGetPath resolves the glyph's path: the device outline, sub-pixel offset, then path effect and stroke applied
// in local space (the inverse of the post 2x2).
func (c *ScalerContext) internalGetPath(g *Glyph) {
	if g.pathDone {
		return
	}
	devPath := c.generatePath(g.packedID.GlyphID())
	if devPath == nil {
		g.setPath(nil, false)
		return
	}

	if c.rec.isSubpixel() {
		dx := g.packedID.SubXOffset()
		dy := g.packedID.SubYOffset()
		if dx != 0 || dy != 0 {
			var offset geom.Matrix
			offset.SetTranslate(dx, dy)
			devPath.Transform(&offset)
		}
	}

	if c.rec.FrameWidth < 0 && c.pathEffect == nil {
		g.setPath(devPath, false)
		return
	}

	// Need the path in user-space, with only the point-size applied, so that stroking and effects operate the same way
	// they would if the user had extracted the path themself.
	matrix := c.rec.matrixFrom2x2()
	inverse, ok := matrix.Invert()
	if !ok {
		g.setPath(path.New(), false)
		return
	}
	localPath := path.New()
	devPath.TransformTo(&inverse, localPath)

	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	if c.rec.FrameWidth >= 0 {
		rec.SetStrokeStyle(c.rec.FrameWidth, c.rec.Flags&recFlagFrameAndFill != 0)
		// Glyphs are always closed contours, so cap type is ignored.
		rec.SetStrokeParams(c.rec.StrokeCap, c.rec.StrokeJoin, c.rec.MiterLimit)
	}

	if c.pathEffect != nil {
		dst := path.New()
		if c.pathEffect.FilterPath(dst, localPath, &rec, nil, &matrix) {
			localPath = dst
		}
	}

	if rec.NeedToApply() {
		dst := path.New()
		if rec.ApplyToPath(dst, localPath) {
			localPath = dst
		}
	}

	localPath.Transform(&matrix)
	g.setPath(localPath, rec.IsHairlineStyle())
}

// getImage renders the glyph's mask through its lane, then applies the mask filter if present. The glyph's image plane
// must already be allocated (ImageSize bytes, per Format).
func (c *ScalerContext) getImage(g *Glyph) {
	if c.maskFilter == nil {
		c.generateImage(g)
		return
	}

	// Need the original bounds/format, sans the mask filter.
	mf := c.maskFilter
	c.maskFilter = nil
	unfiltered := c.makeGlyph(g.packedID)
	c.maskFilter = mf

	if unfiltered.IsEmpty() || unfiltered.ImageSize() == 0 {
		clearBytes(g.Image)
		clear16(g.Image16)
		return
	}
	allocGlyphImage(unfiltered)
	c.generateImage(unfiltered)

	if unfiltered.Format == MaskARGB32 {
		if !maskfilter.AcceptsColorMask(mf) {
			// filterMask returns false on a MaskARGB32 source for this filter, so getImage copies the unfiltered
			// mask — the bounds pass was skipped too, so the rects and formats line up.
			if unfiltered.IRect() == g.IRect() && g.Format == MaskARGB32 {
				copy(g.Image32, unfiltered.Image32)
			}
			return
		}
		// The box blur on ARGB32 extracts the alpha plane and blurs it into an A8 dst; the glyph's format was flipped
		// to A8 by the bounds pass.
		srcMask := raster.Mask{
			Image:    alphaPlane(unfiltered),
			Bounds:   unfiltered.IRect(),
			RowBytes: unfiltered.Width,
		}
		m := c.rec.matrixFrom2x2()
		dst, _, ok := mf.FilterMask(&srcMask, &m)
		if !ok || dst == nil || dst.Image == nil {
			// Filter did nothing (a blur whose CTM-adjusted sigma falls under the no-blur cutoff returns false). The
			// bounds pass failed the same way, so the glyph kept its ARGB32 format and unfiltered bounds: copy the
			// unfiltered color mask, as the LCD16 and A8 lanes below do. Anything else leaves the freshly allocated
			// (zeroed) plane.
			if unfiltered.IRect() == g.IRect() && g.Format == MaskARGB32 {
				copy(g.Image32, unfiltered.Image32)
			}
			return
		}
		clearBytes(g.Image)
		copyMaskIntersection(g, dst)
		return
	}

	if unfiltered.Format == MaskLCD16 {
		// The filter pass runs on the mask's alpha plane (the average of the expanded r/g/b coverages); a successful
		// filter's dst is A8, adopted by the bounds pass.
		srcMask := raster.Mask{
			Image:    alphaPlane16(unfiltered),
			Bounds:   unfiltered.IRect(),
			RowBytes: unfiltered.Width,
		}
		m := c.rec.matrixFrom2x2()
		dst, _, ok := mf.FilterMask(&srcMask, &m)
		if !ok || dst == nil || dst.Image == nil {
			// Filter did nothing; the glyph kept its LCD16 format, so copy the unfiltered plane when the bounds line
			// up.
			if unfiltered.IRect() == g.IRect() && g.Format == MaskLCD16 {
				copy(g.Image16, unfiltered.Image16)
			} else {
				clear16(g.Image16)
				clearBytes(g.Image)
			}
			return
		}
		clearBytes(g.Image)
		copyMaskIntersection(g, dst)
		return
	}

	srcMask := raster.Mask{
		Image:    unfiltered.Image,
		Bounds:   unfiltered.IRect(),
		RowBytes: unfiltered.RowBytes(),
	}
	m := c.rec.matrixFrom2x2()
	dst, _, ok := mf.FilterMask(&srcMask, &m)
	if !ok || dst == nil || dst.Image == nil {
		// Filter did nothing; copy the unfiltered mask if the bounds line up, else clear.
		if unfiltered.IRect() == g.IRect() {
			copy(g.Image, unfiltered.Image)
		} else {
			clearBytes(g.Image)
			copyMaskIntersection(g, &srcMask)
		}
		return
	}
	clearBytes(g.Image)
	copyMaskIntersection(g, dst)
}

// allocGlyphImage allocates the glyph's image plane for its format (zeroed, as Go allocations are).
func allocGlyphImage(g *Glyph) {
	switch g.Format {
	case MaskARGB32:
		g.Image32 = make([]uint32, int(g.Width)*int(g.Height))
	case MaskLCD16:
		g.Image16 = make([]uint16, int(g.Width)*int(g.Height))
	default:
		g.Image = make([]uint8, g.ImageSize())
	}
}

// alphaPlane extracts the alpha channel of an ARGB32 glyph image as an A8 plane (the top byte of each premul word).
func alphaPlane(g *Glyph) []uint8 {
	out := make([]uint8, int(g.Width)*int(g.Height))
	for i, w := range g.Image32[:len(out)] {
		out[i] = uint8(w >> 24)
	}
	return out
}

// alphaPlane16 extracts an A8 plane from an LCD16 glyph image (the average of the 5/6/5 coverages expanded to 8 bits).
func alphaPlane16(g *Glyph) []uint8 {
	out := make([]uint8, int(g.Width)*int(g.Height))
	for i, w := range g.Image16[:len(out)] {
		r := uint32(w>>11) & 0x1F
		gg := uint32(w>>5) & 0x3F
		b := uint32(w) & 0x1F
		r = r<<3 | r>>2
		gg = gg<<2 | gg>>4
		b = b<<3 | b>>2
		out[i] = uint8((r + gg + b) / 3)
	}
	return out
}

func clear16(b []uint16) {
	for i := range b {
		b[i] = 0
	}
}

// generateImage dispatches on the glyph's lane: the color lanes render through their tables; everything else renders
// the (possibly styled) device path as A8 coverage.
func (c *ScalerContext) generateImage(g *Glyph) {
	switch g.scalerBits {
	case scalerBitsCOLRv0:
		c.renderCOLRv0(g)
	case scalerBitsCOLRv1:
		c.renderCOLRv1(g)
	case scalerBitsBitmap:
		c.renderBitmap(g)
	default:
		c.generateMask(g, g.IRect())
	}
}

// copyMaskIntersection copies the intersection of src's bounds and the glyph's bounds into the glyph image.
func copyMaskIntersection(g *Glyph, src *raster.Mask) {
	dstBounds := g.IRect()
	inter := src.Bounds
	if !inter.Intersect(dstBounds) {
		return
	}
	srcRB := int(src.RowBytes)
	dstRB := int(g.RowBytes())
	w := int(inter.Width())
	for y := inter.Top; y < inter.Bottom; y++ {
		srcOff := int(y-src.Bounds.Top)*srcRB + int(inter.Left-src.Bounds.Left)
		dstOff := int(y-dstBounds.Top)*dstRB + int(inter.Left-dstBounds.Left)
		copy(g.Image[dstOff:dstOff+w], src.Image[srcOff:srcOff+w])
	}
}

// generateMask zeroes the buffer, then draws the device path — filled, or as a hairline when the styled path is a
// hairline — translated to mask-local coordinates through the A8 rasterizer, anti-aliased unless the rec asked for
// aliased edging (recFlagAliased, hard 0-or-255 coverage). The LCD lanes (LCD16, and A8 through the LCD filter for
// styled glyphs when recFlagGenA8FromLCD is set; unstyled A8-from-LCD masks come straight from the rasterizer) draw
// into a 4x-oversampled intermediate and downsample through pack4xHToMask with the pre-blend.
func (c *ScalerContext) generateMask(g *Glyph, bounds geom.IRect) {
	clearBytes(g.Image)
	clear16(g.Image16)
	c.internalGetPath(g)
	devPath := g.Path()
	if devPath == nil {
		// No outline host exists in the library other than paths; nothing to draw.
		return
	}

	verticalLCD := c.rec.Flags&recFlagLCDVertical != 0
	doBGR := c.rec.Flags&recFlagLCDBGROrder != 0
	a8FromLCD := c.rec.Flags&recFlagGenA8FromLCD != 0
	hairline := g.pathIsHairline

	// The SDF lane: rasterize A8 coverage at the bounds inset by the pad, then generate the distance field into
	// the glyph's padded plane.
	if g.Format == MaskSDF {
		inner := bounds
		inner.Left += DistanceFieldPad
		inner.Top += DistanceFieldPad
		inner.Right -= DistanceFieldPad
		inner.Bottom -= DistanceFieldPad
		w, h := int(inner.Width()), int(inner.Height())
		if w <= 0 || h <= 0 {
			return
		}
		a8 := make([]uint8, w*h)
		mask := raster.Mask{Image: a8, Bounds: inner, RowBytes: int32(w)}
		maskfilter.RenderPathIntoMask(&mask, devPath, !hairline)
		GenerateDistanceFieldFromA8Image(g.Image, a8, w, h, w)
		return
	}

	fromLCD := g.Format == MaskLCD16 ||
		(g.Format == MaskA8 && a8FromLCD && c.generateImageFromPath)
	if !fromLCD {
		mask := raster.Mask{Image: g.Image, Bounds: bounds, RowBytes: bounds.Width()}
		if c.rec.Flags&recFlagAliased != 0 {
			// Aliased edging: hard coverage, so the mask lane draws the same hard edges the path lane does for this
			// font (Font.HasSomeAntiAliasing gates that one) instead of anti-aliasing regardless of the request.
			maskfilter.RenderPathIntoMaskAliased(&mask, devPath, !hairline)
		} else {
			maskfilter.RenderPathIntoMask(&mask, devPath, !hairline)
		}
		return
	}

	// The LCD lane: rasterize A8 at 4x in the sampling direction. The matrix maps the mask's pixel columns [left+1,
	// right-1) onto sample columns [0, 4*(width-2)) — the outset pixels added by the metrics pass receive the filter
	// spill.
	srcW := int(bounds.Width())
	srcH := int(bounds.Height())
	var dstW, dstH int
	var m geom.Matrix
	if verticalLCD {
		dstW = 4*srcH - 8
		dstH = srcW
		m.SetAll(0, 4, -float32(bounds.Top+1)*4, 1, 0, -float32(bounds.Left), 0, 0, 1)
	} else {
		dstW = 4*srcW - 8
		dstH = srcH
		m.SetAll(4, 0, -float32(bounds.Left+1)*4, 0, 1, -float32(bounds.Top), 0, 0, 1)
	}
	if dstW <= 0 || dstH <= 0 {
		return
	}

	pathToUse := devPath
	fill := !hairline
	if hairline {
		// LCD hairline doesn't line up with the pixels, so do it the expensive way: stroke at width 1 and fill the
		// result.
		rec := stroke.NewStrokeRec(stroke.InitStyleFill)
		rec.SetStrokeStyle(1.0, false)
		rec.SetStrokeParams(stroke.CapButt, stroke.JoinRound, 0.0)
		if rec.NeedToApply() {
			stroked := path.New()
			if rec.ApplyToPath(stroked, devPath) {
				pathToUse = stroked
				fill = true
			}
		}
	}

	oversampled := path.New()
	pathToUse.TransformTo(&m, oversampled)
	intermediate := raster.Mask{
		Image:    make([]uint8, dstW*dstH),
		Bounds:   geom.IRectWH(int32(dstW), int32(dstH)),
		RowBytes: int32(dstW),
	}
	maskfilter.RenderPathIntoMask(&intermediate, oversampled, fill)
	pack4xHToMask(intermediate.Image, dstW, dstH, g, &c.preBlend, doBGR, verticalLCD)
}

// pack4xHToMask coefficients: one 12-tap FIR per subpixel over the 4x samples, determined by a Gaussian where 5 samples
// = 3 std deviations (0x110 'contrast'). The red subpixel is centered inside the first sample (at 1/6 pixel) and is
// shifted; green is centered between two samples (1/2 pixel), so symmetric; blue is centered inside the last sample
// (5/6 pixel), the red row reversed.
var pack4xCoefficients = [3][12]uint32{
	{0x03, 0x0b, 0x1c, 0x33, 0x40, 0x39, 0x24, 0x10, 0x05, 0x01, 0x00, 0x00},
	{0x00, 0x02, 0x08, 0x16, 0x2b, 0x3d, 0x3d, 0x2b, 0x16, 0x08, 0x02, 0x00},
	{0x00, 0x00, 0x01, 0x05, 0x10, 0x24, 0x39, 0x40, 0x33, 0x1c, 0x0b, 0x03},
}

// pack4xHToMask downsamples a 4x-oversampled A8 raster into the glyph's LCD16 plane (or its A8 plane when the rec
// generates A8 from the LCD lane), applying the mask pre-blend per channel. src is sampleWidth×height with tight rows;
// doVert transposes the write (x and y swap when writing to dst).
func pack4xHToMask(src []uint8, sampleWidth, height int, g *Glyph, preBlend *maskPreBlend, doBGR, doVert bool) {
	toA8 := g.Format == MaskA8
	dstW := int(g.Width)
	for y := range height {
		srcRow := src[y*sampleWidth : (y+1)*sampleWidth]
		outX, outY := 0, y
		if doVert {
			outX, outY = y, 0
		}
		for sampleX := -4; sampleX < sampleWidth+4; sampleX += 4 {
			var fir [3]uint32
			sampleIndex := max(0, sampleX-4)
			coeffIndex := sampleIndex - (sampleX - 4)
			for ; sampleIndex < min(sampleX+8, sampleWidth); sampleIndex++ {
				value := uint32(srcRow[sampleIndex])
				fir[0] += pack4xCoefficients[0][coeffIndex] * value
				fir[1] += pack4xCoefficients[1][coeffIndex] * value
				fir[2] += pack4xCoefficients[2][coeffIndex] * value
				coeffIndex++
			}
			for i := range fir {
				fir[i] /= 0x100
				fir[i] = min(fir[i], 255)
			}
			r, gg, b := fir[0], fir[1], fir[2]
			if doBGR {
				r, b = b, r
			}
			di := outY*dstW + outX
			if toA8 {
				a := (r + gg + b) / 3
				if preBlend.isApplicable() {
					a = uint32(preBlend.g[a])
				}
				g.Image[di] = uint8(a)
			} else {
				if preBlend.isApplicable() {
					r = uint32(preBlend.r[r])
					gg = uint32(preBlend.g[gg])
					b = uint32(preBlend.b[b])
				}
				g.Image16[di] = pack888ToRGB16(r, gg, b)
			}
			if doVert {
				outY++
			} else {
				outX++
			}
		}
	}
}

// pack888ToRGB16 packs 8-bit channels into a 5/6/5 word.
func pack888ToRGB16(r, g, b uint32) uint16 {
	return uint16((r>>3)<<11 | (g>>2)<<5 | b>>3)
}

func clearBytes(b []uint8) {
	for i := range b {
		b[i] = 0
	}
}

// getFontMetrics computes the font-wide metrics through the measuring strike's recipe, in device space: the scale
// applied is computeScale's y — the single matrix's vertical scale with rotation removed, which is what upstream scales
// every font metric by — so the metrics live in the same space as the advances and glyph bounds the same strike reports
// (both of which go through singleMatrix). Measuring from the local text matrix alone left them in source space, so a
// device matrix that doubled every glyph left the ascent untouched.
func (c *ScalerContext) getFontMetrics() Metrics {
	scale, ok := c.rec.computeScale()
	size := scale.Y
	if !ok {
		// The singular gate. Zero size drives strike.degenerate(), whose documented fallback reports the metrics at
		// size 1 — the same answer the no-device measuring strike gives for a singular text matrix.
		size = 0
	}
	st := strike{
		t: c.typeface,
		// computeScale's y already carries the whole matrix's vertical scale, so the pre-scale and pre-skew must not be
		// applied a second time; they only reach fontMetrics through the degeneracy check anyway.
		size:       size,
		scaleX:     1,
		frameWidth: -1,
	}
	return st.fontMetrics()
}
