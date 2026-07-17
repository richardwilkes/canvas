// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Style holds the fill/stroke information a StyledShape is rendered with, plus the style-key machinery the
// mask/geometry caches build on. Path effects are devolved to styled paths on the CPU before anything reaches the GPU
// device, so Style carries no path effect: HasPathEffect and IsDashed are always false, and the key has no dash
// portion, only a stroke-rec portion.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// Style holds fill/stroke information without a path effect. The zero value is a simple fill.
type Style struct {
	rec stroke.Rec
}

// SimpleFillStyle returns a Style representing a plain fill.
func SimpleFillStyle() Style {
	return Style{rec: stroke.NewStrokeRec(stroke.InitStyleFill)}
}

// SimpleHairlineStyle returns a Style representing a hairline stroke.
func SimpleHairlineStyle() Style {
	return Style{rec: stroke.NewStrokeRec(stroke.InitStyleHairline)}
}

// MakeStyle wraps an existing stroke rec as a Style.
func MakeStyle(rec stroke.Rec) Style { return Style{rec: rec} }

// StyleApply selects which portions of a style key to compute.
type StyleApply uint8

// StyleApply values.
const (
	StyleApplyPathEffectOnly StyleApply = iota
	StyleApplyPathEffectAndStrokeRec
)

// Style key flags: optional flags for computing keys that may remove unnecessary variation in the key due to style
// settings that don't affect particular classes of geometry.
const (
	// StyleKeyClosed: the shape being styled has no open contours.
	StyleKeyClosed uint32 = 0x1
	// StyleKeyNoJoins: the shape being styled doesn't have any joins and so isn't affected by join type.
	StyleKeyNoJoins uint32 = 0x2
)

// Rec returns the underlying stroke rec.
func (s *Style) Rec() *stroke.Rec { return &s.rec }

// ResetToInitStyle resets the style to a plain fill or hairline stroke.
func (s *Style) ResetToInitStyle(fillOrHairline stroke.InitStyle) {
	if fillOrHairline == stroke.InitStyleFill {
		s.rec.SetFillStyle()
	} else {
		s.rec.SetHairlineStyle()
	}
}

// IsSimpleFill reports whether this style is a fill with no path effect.
func (s *Style) IsSimpleFill() bool { return s.rec.IsFillStyle() }

// IsSimpleHairline reports whether this style is a hairline stroke.
func (s *Style) IsSimpleHairline() bool { return s.rec.IsHairlineStyle() }

// HasPathEffect always reports false: this Style never carries a path effect.
func (s *Style) HasPathEffect() bool { return false }

// HasNonDashPathEffect always reports false: this Style never carries a path effect.
func (s *Style) HasNonDashPathEffect() bool { return false }

// IsDashed always reports false: this Style never carries a dash path effect.
func (s *Style) IsDashed() bool { return false }

// Applies reports whether this style alters the geometry it is applied to: hairline or fill styles without path effects
// make no alterations.
func (s *Style) Applies() bool {
	return !s.rec.IsFillStyle() && !s.rec.IsHairlineStyle()
}

// MatrixToScaleFactor derives the scale factor a stroker should use from a view matrix. MaxScale returns -1 if the
// matrix has perspective; in that case a scale factor of 1 is used.
func MatrixToScaleFactor(matrix *geom.Matrix) float32 {
	return geom.ScalarAbs(matrix.MaxScale())
}

// ApplyToPath succeeds only when the stroke rec changes the geometry, writing the result into dst and returning its
// fill-or-hairline disposition. resScale controls geometric approximations made by the stroker and is typically
// computed from the view matrix.
func (s *Style) ApplyToPath(dst, src *path.Path, resScale float32) (fillOrHairline stroke.InitStyle, ok bool) {
	rec := s.rec
	rec.SetResScale(resScale)
	if !rec.NeedToApply() {
		// Nothing to do for path effect or stroke, fail.
		return stroke.InitStyleFill, false
	}
	if !rec.ApplyToPath(dst, src) {
		return stroke.InitStyleFill, false
	}
	dst.SetVolatile(true)
	return stroke.InitStyleFill, true
}

// AdjustBounds computes the bounds of a path with the style applied, given the bounds of the path itself.
func (s *Style) AdjustBounds(src geom.Rect) geom.Rect {
	radius := s.rec.InflationRadius()
	return src.Outset(radius, radius)
}

// StyleKeySize returns the key length for a style. There is no path effect, so the key never fails and only the
// stroke-rec portion contributes; a simple fill has a zero-sized key.
func StyleKeySize(style *Style, apply StyleApply, flags uint32) int {
	_ = flags
	size := 0
	if apply == StyleApplyPathEffectOnly {
		return size
	}
	if style.rec.NeedToApply() {
		// One for res scale, one for style/cap/join, one for miter limit, and one for width.
		size += 4
	}
	return size
}

// WriteStyleKey writes StyleKeySize(style, apply, flags) values into key describing the stroke-rec portion of the
// style; there is no dash portion since styles here never carry a path effect.
func WriteStyleKey(key []uint32, style *Style, apply StyleApply, scale float32, flags uint32) {
	i := 0
	if apply == StyleApplyPathEffectAndStrokeRec && style.rec.NeedToApply() {
		key[i] = math.Float32bits(scale)
		i++
		const styleBits = 2
		const joinBits = 2
		const joinShift = styleBits
		const capShift = joinShift + joinBits

		// The cap type only matters for unclosed shapes. A path effect could unclose the shape before it is stroked,
		// but that lane is unreachable here since styles never carry a path effect.
		strokeCap := stroke.CapButt // the default cap
		if flags&StyleKeyClosed == 0 {
			strokeCap = style.rec.Cap()
		}
		miter := float32(-1)
		join := stroke.JoinMiter // the default join

		// Dashing will not insert joins but other path effects may (unreachable here).
		if flags&StyleKeyNoJoins == 0 {
			join = style.rec.Join()
			// Miter limit only affects miter joins.
			if join == stroke.JoinMiter {
				miter = style.rec.Miter()
			}
		}

		key[i] = uint32(style.rec.Style()) | uint32(join)<<joinShift | uint32(strokeCap)<<capShift
		i++
		key[i] = math.Float32bits(miter)
		i++
		key[i] = math.Float32bits(style.rec.Width())
		i++
	}
	if i != StyleKeySize(style, apply, flags) {
		panic("style key size mismatch")
	}
}
