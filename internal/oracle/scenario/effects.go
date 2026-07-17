// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The declarative effect/filter specs added as the corpus grew: path effects, color filters, the blur mask
// filter, and image filters, each described as plain data so a backend materializes them natively (the port through
// patheffect/colorfilter/maskfilter/imagefilter; the C oracle, while it existed, through the public-surface
// constructors). Every enum whose values cross the public boundary mirrors the ordinals of the removed capi/sk_capi.h,
// so backends may cast directly — the same contract as the scenario.go enums.

package scenario

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// PathEffectKind selects the PathEffectSpec variant.
type PathEffectKind int32

// PathEffectKind values.
const (
	PathEffectDash PathEffectKind = iota
	PathEffectCorner
	PathEffectDiscrete
	PathEffectTrim
	PathEffect1DPath
	PathEffect2DLine
	PathEffect2DPath
	PathEffectSum
	PathEffectCompose
)

// TrimMode mirrors sk_path_effect_trim_mode_t.
type TrimMode int32

// TrimMode values.
const (
	TrimNormal TrimMode = iota
	TrimInverted
)

// Path1DStyle mirrors sk_path_effect_1d_style_t.
type Path1DStyle int32

// Path1DStyle values.
const (
	Path1DTranslate Path1DStyle = iota
	Path1DRotate
	Path1DMorph
)

// PathEffectSpec declaratively describes a paint path effect; each variant reads only its documented fields.
type PathEffectSpec struct {
	Path      *PathSpec       // 1d-path (the stamp), 2d-path (the tile)
	Matrix    *geom.Matrix    // 2d-line, 2d-path (the lattice matrix)
	A, B      *PathEffectSpec // sum: first/second; compose: outer/inner
	Intervals []float32       // dash: on/off lengths (even count)
	Kind      PathEffectKind
	Phase     float32     // dash, 1d-path
	Radius    float32     // corner
	SegLength float32     // discrete
	Deviation float32     // discrete
	Seed      uint32      // discrete (seedAssist; both engines share Skia's LCG)
	Start     float32     // trim
	Stop      float32     // trim
	Trim      TrimMode    // trim
	Advance   float32     // 1d-path
	Style1D   Path1DStyle // 1d-path
	Width     float32     // 2d-line
}

// ColorFilterKind selects the ColorFilterSpec variant.
type ColorFilterKind int32

// ColorFilterKind values.
const (
	ColorFilterMatrix ColorFilterKind = iota
	ColorFilterBlend
	ColorFilterLighting
	ColorFilterLuma
	ColorFilterHighContrast
	ColorFilterCompose
)

// HighContrastInvertStyle mirrors sk_high_contrast_config_invert_style_t.
type HighContrastInvertStyle int32

// HighContrastInvertStyle values.
const (
	InvertNone HighContrastInvertStyle = iota
	InvertBrightness
	InvertLightness
)

// ColorFilterSpec declaratively describes a paint color filter; each variant reads only its documented fields.
type ColorFilterSpec struct {
	Matrix       *[20]float32     // matrix (row-major 4x5, unpremul float like Skia's ColorMatrix)
	Outer, Inner *ColorFilterSpec // compose
	Kind         ColorFilterKind
	Color        colorcore.Color         // blend ("mode" filter)
	Blend        BlendMode               // blend
	Mul, Add     colorcore.Color         // lighting
	InvertStyle  HighContrastInvertStyle // high-contrast
	Contrast     float32                 // high-contrast
	Grayscale    bool                    // high-contrast
}

// BlurStyle mirrors sk_blur_style_t.
type BlurStyle int32

// BlurStyle values.
const (
	BlurNormal BlurStyle = iota
	BlurSolid
	BlurOuter
	BlurInner
)

// MaskFilterSpec declaratively describes a paint mask filter. Only the blur mask filter is in the corpus: it is the one
// with a dedicated GPU lane on both sides (the table/clip/gamma/shader mask filters are raster-only constructs covered
// by the probe differentials, and a corpus scenario renders through both lanes).
type MaskFilterSpec struct {
	Style      BlurStyle
	Sigma      float32
	RespectCTM bool
}

// ImageFilterKind selects the ImageFilterSpec variant — one per sk_imagefilter_new_* constructor reachable without an
// Skia's Image input (the image-source leaf needs image plumbing the corpus does not carry).
type ImageFilterKind int32

// ImageFilterKind values.
const (
	FilterBlur ImageFilterKind = iota
	FilterDropShadow
	FilterDropShadowOnly
	FilterOffset
	FilterColorFilter
	FilterCompose
	FilterMerge
	FilterArithmetic
	FilterDilate
	FilterErode
	FilterDisplacement
	FilterConvolution
	FilterMatrixTransform
	FilterMagnifier
	FilterDistantLitDiffuse
	FilterPointLitDiffuse
	FilterSpotLitDiffuse
	FilterDistantLitSpecular
	FilterPointLitSpecular
	FilterSpotLitSpecular
	FilterTile
)

// ColorChannel mirrors sk_color_channel_t (displacement-map channel selectors).
type ColorChannel int32

// ColorChannel values.
const (
	ChannelR ColorChannel = iota
	ChannelG
	ChannelB
	ChannelA
)

// Point3 mirrors sk_point3_t (lighting-filter light geometry).
type Point3 struct{ X, Y, Z float32 }

// ImageFilterSpec declaratively describes a paint image filter DAG; each variant reads only its documented fields. A
// nil Input/A/B leg means "the filtered source" (the draw itself), exactly as a null sk_image_filter_t* does.
type ImageFilterSpec struct {
	Input       *ImageFilterSpec   // single-input variants
	Crop        *geom.Rect         // crop rect, where the constructor takes one
	ColorFilter *ColorFilterSpec   // color-filter
	A, B        *ImageFilterSpec   // compose: outer/inner; arithmetic: background/foreground; displacement: displacement/color
	Matrix      *geom.Matrix       // matrix-transform
	Inputs      []*ImageFilterSpec // merge
	Kernel      []float32          // convolution (row-major, KernelW*KernelH)

	Lens             geom.Rect  // magnifier
	SrcRect, DstRect geom.Rect  // tile
	K                [4]float32 // arithmetic k1..k4
	Direction        Point3     // distant-lit
	Location         Point3     // point-lit, spot-lit
	Target           Point3     // spot-lit

	Kind                 ImageFilterKind
	SigmaX, SigmaY       float32         // blur, drop-shadow
	Tile                 TileMode        // blur, convolution
	Dx, Dy               float32         // drop-shadow, offset
	Color                colorcore.Color // drop-shadow
	RadiusX, RadiusY     int32           // dilate, erode
	XChannel, YChannel   ColorChannel    // displacement
	Scale                float32         // displacement
	KernelW, KernelH     int32           // convolution
	Gain, Bias           float32         // convolution
	OffsetX, OffsetY     int32           // convolution kernel offset
	Zoom, Inset          float32         // magnifier
	LightColor           colorcore.Color // lighting
	SurfaceScale, KD, KS float32         // lighting (KD diffuse, KS specular)
	Shininess            float32         // specular
	SpecExp, CutoffAngle float32         // spot-lit falloff exponent + cone cutoff

	Linear        bool // matrix-transform, magnifier: linear (vs nearest) sampling
	EnforcePM     bool // arithmetic
	ConvolveAlpha bool // convolution
}
