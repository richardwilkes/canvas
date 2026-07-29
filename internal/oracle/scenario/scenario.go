// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package scenario defines the declarative drawing corpus rendered for differential testing. Scenarios draw through the
// Canvas interface using only operations reachable through the public surface, with paints and paths described as plain
// data so a backend can materialize them natively. This package is cgo-free.
//
// The gating references are the library's own per-platform self-captured sets under ../goldens, regenerated
// deliberately via `oracle bless` and the capture-goldens workflow whenever rendering changes intentionally.
package scenario

import (
	"sort"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// Enum ordinals are frozen: they mirror the C API the corpus was built against (whose values are Skia's own) and the
// goldens depend on them, so they must not be renumbered. The `sk_*` names below record which C type each mirrors.

// BlendMode mirrors sk_blend_mode_t.
type BlendMode int32

// BlendMode values.
const (
	BlendClear BlendMode = iota
	BlendSrc
	BlendDst
	BlendSrcOver
	BlendDstOver
	BlendSrcIn
	BlendDstIn
	BlendSrcOut
	BlendDstOut
	BlendSrcATop
	BlendDstATop
	BlendXor
	BlendPlus
	BlendModulate
	BlendScreen
	BlendOverlay
	BlendDarken
	BlendLighten
	BlendColorDodge
	BlendColorBurn
	BlendHardLight
	BlendSoftLight
	BlendDifference
	BlendExclusion
	BlendMultiply
	BlendHue
	BlendSaturation
	BlendColor
	BlendLuminosity
)

// Style mirrors sk_paint_style_t.
type Style int32

// Style values.
const (
	StyleFill Style = iota
	StyleStroke
	StyleStrokeAndFill
)

// StrokeCap mirrors sk_stroke_cap_t.
type StrokeCap int32

// StrokeCap values.
const (
	CapButt StrokeCap = iota
	CapRound
	CapSquare
)

// StrokeJoin mirrors sk_stroke_join_t.
type StrokeJoin int32

// StrokeJoin values.
const (
	JoinMiter StrokeJoin = iota
	JoinRound
	JoinBevel
)

// ClipOp mirrors sk_clip_op_t.
type ClipOp int32

// ClipOp values.
const (
	ClipDifference ClipOp = iota
	ClipIntersect
)

// PointMode mirrors sk_point_mode_t.
type PointMode int32

// PointMode values.
const (
	PointModePoints PointMode = iota
	PointModeLines
	PointModePolygon
)

// TileMode mirrors sk_tile_mode_t.
type TileMode int32

// TileMode values.
const (
	TileClamp TileMode = iota
	TileRepeat
	TileMirror
	TileDecal
)

// ShaderKind selects the ShaderSpec variant.
type ShaderKind int32

// ShaderKind values.
const (
	ShaderLinearGradient ShaderKind = iota
	ShaderRadialGradient
	ShaderSweepGradient
	ShaderTwoPointConicalGradient
	ShaderColor
	ShaderBlend
)

// ShaderSpec declaratively describes a paint shader: the four gradient families, the single-color shader, and the
// blend-of-two-shaders combinator. Each variant reads only its documented fields.
type ShaderSpec struct {
	Dst, Src   *ShaderSpec // blend shader inputs
	Local      *geom.Matrix
	Colors     []colorcore.Color
	Pos        []float32     // nil for even spacing
	Pts        [2]geom.Point // linear: start/end; two-point conical: start/end centers
	Center     geom.Point    // radial, sweep
	Kind       ShaderKind
	Tile       TileMode
	Radius     float32         // radial; two-point conical: start radius
	Radius2    float32         // two-point conical: end radius
	StartAngle float32         // sweep, degrees (0 with EndAngle 0 means the full 0..360 sweep)
	EndAngle   float32         // sweep, degrees
	Color      colorcore.Color // color shader
	Blend      BlendMode       // blend shader
}

// Paint declaratively describes a paint. Always start from NewPaint: the zero value has blend mode Clear (enum
// ordinal 0), not the SrcOver default. The effect/filter spec types (all nil by default) live in effects.go.
type Paint struct {
	Shader      *ShaderSpec
	ImageFilter *ImageFilterSpec
	MaskFilter  *MaskFilterSpec
	ColorFilter *ColorFilterSpec
	PathEffect  *PathEffectSpec
	StrokeWidth float32
	Join        StrokeJoin
	Cap         StrokeCap
	StrokeMiter float32
	Color       colorcore.Color
	Blend       BlendMode
	Style       Style
	AA          bool
	Dither      bool
}

// NewPaint returns a Paint matching a default-constructed Skia's Paint: opaque black fill, SrcOver, butt cap, miter
// join with limit 4, no AA.
func NewPaint() Paint {
	return Paint{
		Color:       colorcore.Color(0xFF000000),
		Style:       StyleFill,
		Blend:       BlendSrcOver,
		StrokeMiter: 4,
		Cap:         CapButt,
		Join:        JoinMiter,
	}
}

// Fill returns a default paint filled with the given color, antialiased.
func Fill(c colorcore.Color) Paint {
	p := NewPaint()
	p.Color = c
	p.AA = true
	return p
}

// Stroke returns a default paint stroking with the given color and width, antialiased.
func Stroke(c colorcore.Color, width float32) Paint {
	p := NewPaint()
	p.Color = c
	p.Style = StyleStroke
	p.StrokeWidth = width
	p.AA = true
	return p
}

// Canvas is the publicly reachable drawing surface a scenario draws on. gorender's adapter over the library's
// *canvas.Canvas is its only implementation now; the removed C oracle once provided a second one.
type Canvas interface {
	Clear(c colorcore.Color)
	DrawColor(c colorcore.Color, mode BlendMode)
	DrawPaint(p Paint)
	DrawRect(r geom.Rect, p Paint)
	DrawOval(r geom.Rect, p Paint)
	DrawCircle(cx, cy, radius float32, p Paint)
	DrawRoundRect(r geom.Rect, rx, ry float32, p Paint)
	DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, p Paint)
	DrawLine(x0, y0, x1, y1 float32, p Paint)
	DrawPoints(mode PointMode, pts []geom.Point, p Paint)
	DrawPath(spec *PathSpec, p Paint)
	ClipRect(r geom.Rect, op ClipOp, aa bool)
	ClipPath(spec *PathSpec, op ClipOp, aa bool)
	Translate(dx, dy float32)
	Scale(sx, sy float32)
	RotateRadians(radians float32)
	Skew(sx, sy float32)
	Concat(m *geom.Matrix)
	Save() int
	SaveLayerAlpha(bounds *geom.Rect, alpha uint8) int
	// SaveLayer is the paint-carrying saveLayer: the layer's restore draws through the paint's alpha, blend mode,
	// color filter, and image filter. p nil means a plain layer.
	SaveLayer(bounds *geom.Rect, p *Paint) int
	Restore()
	// DrawSimpleText draws UTF-8 text at the baseline origin with the corpus font (FontData) at the given point size —
	// Skia's Font defaults otherwise (kAntiAlias edging, normal hinting, no subpixel), which both backends share. Added
	// with the text scenarios.
	DrawSimpleText(text string, x, y, size float32, p Paint)
	// DrawSimpleTextFont is DrawSimpleText with an explicit corpus font (text.go's FontID table); the emoji scenarios
	// select the color test fonts through it.
	DrawSimpleTextFont(text string, x, y, size float32, f FontID, p Paint)
	// DrawAtlas draws sprites from an atlas passed as raw RGBA8888-premul pixels (tightly packed, top-left origin), so
	// each backend materializes its native image type from identical bytes: sprite i maps the atlas sub-rect tex[i]
	// onto xforms[i]'s quad, optionally modulated by colors[i] (nil for none) through mode. linear selects linear
	// filtering (nearest otherwise); cull is the optional quick-reject hint. This is materialized through
	// Canvas.DrawAtlas.
	DrawAtlas(atlasW, atlasH int32, atlasRGBA []byte, xforms []geom.RSXform, tex []geom.Rect,
		colors []colorcore.Color, mode BlendMode, linear bool, cull *geom.Rect, p Paint)
}

// Scenario is one differential test case: a named drawing routine at a fixed size. Every scenario renders in every
// lane: the golden sets are per-platform (goldens/<lane>/<GOOS_GOARCH>/), so platform-dependent output — text via the
// library's own rasterizer, composite rounding differences between architectures — is in scope for all of them.
type Scenario struct {
	Draw   func(c Canvas)
	Name   string
	Width  int
	Height int
}

var registry = map[string]Scenario{}

// Register adds a scenario to the corpus; duplicate names panic (they would silently shadow goldens).
func Register(s Scenario) {
	if _, ok := registry[s.Name]; ok {
		panic("duplicate scenario name: " + s.Name)
	}
	registry[s.Name] = s
}

// All returns the corpus sorted by name.
func All() []Scenario {
	out := make([]Scenario, 0, len(registry))
	for _, s := range registry {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ByName returns the named scenario.
func ByName(name string) (Scenario, bool) {
	s, ok := registry[name]
	return s, ok
}
