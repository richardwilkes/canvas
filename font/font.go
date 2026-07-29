// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Font's fields: size/scaleX/skewX, the flag set, and the measuring entry points (metrics, measure, widths,
// x-positions, text→glyphs). Drawing arrives with the canvas text phase.

package font

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

// Hinting selects the requested hinting level. only HintingNone and HintingSlight change nothing here (rendering is
// always unhinted); HintingNormal/HintingFull are accepted and recorded but not honored.
type Hinting uint8

// Hinting values.
const (
	HintingNone Hinting = iota
	HintingSlight
	HintingNormal
	HintingFull
)

// Edging selects the anti-aliasing treatment applied to a glyph's mask.
type Edging uint8

// Edging values.
const (
	EdgingAlias Edging = iota
	EdgingAntiAlias
	EdgingSubpixelAntiAlias
)

// Private flag bits.
const (
	flagForceAutoHinting = 1 << 0
	flagEmbeddedBitmaps  = 1 << 1
	flagSubpixel         = 1 << 2
	flagLinearMetrics    = 1 << 3
	flagEmbolden         = 1 << 4
	flagBaselineSnap     = 1 << 5
)

// Defaults for a newly constructed Font: 12pt size, baseline-snapping enabled, anti-aliased edging, and normal hinting.
const (
	defaultSize    = 12
	defaultFlags   = flagBaselineSnap
	defaultEdging  = EdgingAntiAlias
	defaultHinting = HintingNormal
)

// Font holds a typeface plus size/scale/skew and rendering flags: everything needed to lay out and measure text.
type Font struct {
	typeface *Typeface
	size     float32
	scaleX   float32
	skewX    float32
	flags    uint8
	edging   Edging
	hinting  Hinting
}

// NewFont builds a Font from typeface, size, scaleX, and skewX: a nil typeface becomes the empty typeface, negative and
// NaN sizes clamp to zero.
func NewFont(typeface *Typeface, size, scaleX, skewX float32) *Font {
	if typeface == nil {
		typeface = EmptyTypeface()
	}
	return &Font{
		typeface: typeface,
		size:     validSize(size),
		scaleX:   scaleX,
		skewX:    skewX,
		flags:    defaultFlags,
		edging:   defaultEdging,
		hinting:  defaultHinting,
	}
}

// validSize clamps a requested text size to a usable value: negatives and NaN both become zero. Go's max propagates
// NaN, so it cannot be used here (upstream's std::max<SkScalar>(0, size) returns 0 for NaN); a NaN size would reach
// ScalerRec.TextSize, and a strike key holding NaN never equals itself, so its cache entry could never be found again
// nor deleted. See ScalerRec.canonicalizeKeyFloats.
func validSize(size float32) float32 {
	if !(size > 0) { // the negated comparison is false for NaN too
		return 0
	}
	return size
}

// Typeface returns the font's typeface (never nil).
func (f *Font) Typeface() *Typeface { return f.typeface }

// Size returns the text size.
func (f *Font) Size() float32 { return f.size }

// ScaleX returns the horizontal scale factor.
func (f *Font) ScaleX() float32 { return f.scaleX }

// SkewX returns the horizontal skew factor.
func (f *Font) SkewX() float32 { return f.skewX }

// SetSize sets the text size, clamping negative and NaN values to zero.
func (f *Font) SetSize(size float32) { f.size = validSize(size) }

// SetScaleX sets the horizontal scale factor.
func (f *Font) SetScaleX(scale float32) { f.scaleX = scale }

// SetSkewX sets the horizontal skew factor.
func (f *Font) SetSkewX(skew float32) { f.skewX = skew }

func (f *Font) setFlag(mask uint8, on bool) {
	if on {
		f.flags |= mask
	} else {
		f.flags &^= mask
	}
}

// SetForceAutoHinting sets whether auto-hinting is forced (recorded, never honored: this library renders outline fonts
// only).
func (f *Font) SetForceAutoHinting(on bool) { f.setFlag(flagForceAutoHinting, on) }

// ForceAutoHinting reports whether auto-hinting is forced.
func (f *Font) ForceAutoHinting() bool { return f.flags&flagForceAutoHinting != 0 }

// SetLinearMetrics sets whether font and glyph metrics are requested to be linearly scalable. Rendering here is always
// unhinted, so the request only enters the scaler rec (and therefore the strike key); it changes no metric.
func (f *Font) SetLinearMetrics(on bool) { f.setFlag(flagLinearMetrics, on) }

// LinearMetrics reports whether linearly scalable font and glyph metrics are requested.
func (f *Font) LinearMetrics() bool { return f.flags&flagLinearMetrics != 0 }

// SetSubpixel sets whether subpixel positioning is requested.
func (f *Font) SetSubpixel(on bool) { f.setFlag(flagSubpixel, on) }

// Subpixel reports whether subpixel positioning is requested.
func (f *Font) Subpixel() bool { return f.flags&flagSubpixel != 0 }

// SetHinting sets the requested hinting level (only HintingNone/HintingSlight are honored; the library is always
// unhinted with linear metrics).
func (f *Font) SetHinting(h Hinting) { f.hinting = h }

// Hinting returns the recorded hinting.
func (f *Font) Hinting() Hinting { return f.hinting }

// SetEdging sets the anti-aliasing treatment.
func (f *Font) SetEdging(e Edging) { f.edging = e }

// Edging returns the recorded edging.
func (f *Font) Edging() Edging { return f.edging }

// UnicharToGlyph maps a single Unicode code point to a glyph ID via the typeface's cmap.
func (f *Font) UnicharToGlyph(unichar int32) uint16 { return f.typeface.UnicharToGlyph(unichar) }

// UnicharsToGlyphs maps each Unicode code point in unichars to a glyph ID via the typeface's cmap.
func (f *Font) UnicharsToGlyphs(unichars []int32, glyphs []uint16) {
	f.typeface.UnicharsToGlyphs(unichars, glyphs)
}

// TextToGlyphs decodes text in the given encoding and maps each character to a glyph ID.
func (f *Font) TextToGlyphs(text []byte, encoding TextEncoding, glyphs []uint16) int {
	return f.typeface.TextToGlyphs(text, encoding, glyphs)
}

// glyphsFromText returns the glyph run for the text, or nil for empty/invalid input.
func (f *Font) glyphsFromText(text []byte, encoding TextEncoding) []uint16 {
	if len(text) == 0 {
		return nil
	}
	if encoding == TextEncodingGlyphID {
		glyphs := make([]uint16, len(text)>>1)
		for i := range glyphs {
			glyphs[i] = uint16(text[2*i]) | uint16(text[2*i+1])<<8
		}
		return glyphs
	}
	count := CountTextElements(text, encoding)
	if count <= 0 {
		return nil
	}
	glyphs := make([]uint16, count)
	f.typeface.TextToGlyphs(text, encoding, glyphs)
	return glyphs
}

// MeasureText returns the advance width of the text, optionally its joined glyph bounds, with paint (stroke style and
// path effect) applied to the bounds as the scaler does. text is raw bytes in the given encoding; paint may be nil.
func (f *Font) MeasureText(text []byte, encoding TextEncoding, bounds *geom.Rect, paint *stroke.PaintSpec) float32 {
	glyphs := f.glyphsFromText(text, encoding)
	if len(glyphs) == 0 {
		if bounds != nil {
			*bounds = geom.Rect{}
		}
		return 0
	}
	st, strikeToSourceScale := makeCanonicalized(f, paint)
	var width float32
	if bounds != nil {
		*bounds = st.glyphBounds(glyphs[0])
		width = st.glyphAdvance(glyphs[0])
		for _, g := range glyphs[1:] {
			r := st.glyphBounds(g)
			r.Left += width
			r.Right += width
			bounds.Join(r)
			width += st.glyphAdvance(g)
		}
	} else {
		for _, g := range glyphs {
			width += st.glyphAdvance(g)
		}
	}
	if strikeToSourceScale != 1 {
		width *= strikeToSourceScale
		if bounds != nil {
			bounds.Left *= strikeToSourceScale
			bounds.Top *= strikeToSourceScale
			bounds.Right *= strikeToSourceScale
			bounds.Bottom *= strikeToSourceScale
		}
	}
	return width
}

// GlyphWidths fills widths with the advance width of each glyph (unpainted).
func (f *Font) GlyphWidths(glyphs []uint16, widths []float32) {
	st, strikeToSourceScale := makeCanonicalized(f, nil)
	n := min(len(glyphs), len(widths))
	for i := range n {
		widths[i] = st.glyphAdvance(glyphs[i]) * strikeToSourceScale
	}
}

// GetXPos fills xpos with the successive x-coordinate origin of each glyph, starting at origin and advancing by each
// glyph's advance width.
func (f *Font) GetXPos(glyphs []uint16, xpos []float32, origin float32) {
	st, strikeToSourceScale := makeCanonicalized(f, nil)
	loc := origin
	n := min(len(glyphs), len(xpos))
	for i := range n {
		xpos[i] = loc
		loc += st.glyphAdvance(glyphs[i]) * strikeToSourceScale
	}
}

// setupForAsPaths reconfigures the font (and optionally the paint) for canonical-size path measurement/drawing,
// returning the strike-to-source scale.
func (f *Font) setupForAsPaths(paint *ScalerPaint) float32 {
	// flagLinearMetrics deliberately survives (upstream's flagsToIgnore covers only these two): path measurement is
	// already unhinted and linear, so clearing it would only fragment the strike key.
	const flagsToIgnore = flagEmbeddedBitmaps | flagForceAutoHinting
	f.flags = (f.flags &^ uint8(flagsToIgnore)) | flagSubpixel
	f.hinting = HintingNone
	if f.edging == EdgingSubpixelAntiAlias {
		f.edging = EdgingAntiAlias
	}
	if paint != nil {
		paint.Style = stroke.PaintStyleFill
		paint.PathEffect = nil
	}
	textSize := f.size
	f.size = canonicalTextSizeForPaths
	return textSize / canonicalTextSizeForPaths
}

// GetFontBounds returns the typeface's bounds mapped through the font's text matrix (scale by (size*scaleX, size), then
// skew by skewX).
func GetFontBounds(f *Font) geom.Rect {
	var m geom.Matrix
	m.SetScale(f.size*f.scaleX, f.size)
	var skew geom.Matrix
	skew.SetSkew(f.skewX, 0)
	m.PostConcat(&skew)
	r, _ := m.MapRect(f.typeface.Bounds())
	return r
}

// DrawTextPositions fills out with successive glyph origins from a no-device strike (actual size, no canonicalization),
// starting at origin. It writes min(len(glyphs), len(out)) origins, as the sibling GetPos does.
func DrawTextPositions(f *Font, glyphs []uint16, origin geom.Point, out []geom.Point) {
	st := strike{t: f.typeface, size: f.size, scaleX: f.scaleX, skewX: f.skewX, frameWidth: -1}
	sum := origin
	n := min(len(glyphs), len(out))
	for i := range n {
		out[i] = sum
		sum.X += st.glyphAdvance(glyphs[i])
	}
}

// HasSomeAntiAliasing reports whether the current edging applies any anti-aliasing.
func (f *Font) HasSomeAntiAliasing() bool {
	return f.edging == EdgingAntiAlias || f.edging == EdgingSubpixelAntiAlias
}

// GetPos fills pos with successive glyph origins starting at origin, advancing by each glyph's advance vector.
func (f *Font) GetPos(glyphs []uint16, pos []geom.Point, origin geom.Point) {
	st, strikeToSourceScale := makeCanonicalized(f, nil)
	sum := origin
	n := min(len(glyphs), len(pos))
	for i := range n {
		pos[i] = sum
		sum.X += st.glyphAdvance(glyphs[i]) * strikeToSourceScale
	}
}

// GlyphBounds fills bounds with the per-glyph bounds in source space (unpainted).
func (f *Font) GlyphBounds(glyphs []uint16, bounds []geom.Rect) {
	st, strikeToSourceScale := makeCanonicalized(f, nil)
	n := min(len(glyphs), len(bounds))
	for i := range n {
		r := st.glyphBounds(glyphs[i])
		if strikeToSourceScale != 1 {
			r.Left *= strikeToSourceScale
			r.Top *= strikeToSourceScale
			r.Right *= strikeToSourceScale
			r.Bottom *= strikeToSourceScale
		}
		bounds[i] = r
	}
}

// GlyphBoundsWithPaint fills bounds with the per-glyph bounds in source space with the paint's style applied (the
// glyphrun_source_bounds tight fallback).
func GlyphBoundsWithPaint(f *Font, glyphs []uint16, paint *stroke.PaintSpec, bounds []geom.Rect) {
	st, strikeToSourceScale := makeCanonicalized(f, paint)
	n := min(len(glyphs), len(bounds))
	for i := range n {
		r := st.glyphBounds(glyphs[i])
		if strikeToSourceScale != 1 {
			r.Left *= strikeToSourceScale
			r.Top *= strikeToSourceScale
			r.Right *= strikeToSourceScale
			r.Bottom *= strikeToSourceScale
		}
		bounds[i] = r
	}
}

// Metrics fills metrics (which may be nil) and returns the recommended line spacing (descent - ascent + leading).
func (f *Font) Metrics(metrics *Metrics) float32 {
	st, strikeToSourceScale := makeCanonicalized(f, nil)
	m := st.fontMetrics()
	if strikeToSourceScale != 1 {
		m.scale(strikeToSourceScale)
	}
	if metrics != nil {
		*metrics = m
	}
	return m.Descent - m.Ascent + m.Leading
}
