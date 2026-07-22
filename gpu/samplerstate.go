// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SamplerState describes how a texture is sampled: wrap mode, filter, mipmap mode, and anisotropy. The key packing in
// AsKey must stay stable since the GL sampler-object cache and program keys depend on it.

package gpu

// FilterMode selects how a texture is sampled between texel centers.
type FilterMode uint8

// FilterMode values.
const (
	FilterModeNearest FilterMode = iota // single sample point (nearest neighbor)
	FilterModeLinear                    // interpolate between 2x2 sample points (bilinear)

	// FilterModeCount is the number of valid FilterMode values.
	FilterModeCount = int(FilterModeLinear) + 1
)

// MipmapMode selects how a texture is sampled across mip levels.
type MipmapMode uint8

// MipmapMode values.
const (
	MipmapModeNone    MipmapMode = iota // ignore mipmap levels, sample from the "base"
	MipmapModeNearest                   // sample from the nearest level
	MipmapModeLinear                    // interpolate between the two nearest levels

	// MipmapModeCount is the number of valid MipmapMode values.
	MipmapModeCount = int(MipmapModeLinear) + 1
)

// WrapMode selects how out-of-range texture coordinates are handled.
type WrapMode uint8

// WrapMode values.
const (
	WrapModeClamp WrapMode = iota
	WrapModeRepeat
	WrapModeMirrorRepeat
	WrapModeClampToBorder

	// WrapModeCount is the number of valid WrapMode values.
	WrapModeCount = int(WrapModeClampToBorder) + 1
)

// MaxMaxAniso is the largest legal value for SamplerState.MaxAniso.
const MaxMaxAniso = 1024

// SamplerState describes a complete texture sampling configuration. The zero value is a reasonable default
// (clamp/clamp, nearest, no mipmap) except MaxAniso, which callers should set via the Make functions rather than a
// struct literal (the intended default is 1, not 0).
type SamplerState struct {
	WrapModeX  WrapMode
	WrapModeY  WrapMode
	Filter     FilterMode
	MipmapMode MipmapMode
	MaxAniso   int32
}

// MakeSamplerState returns a non-anisotropic SamplerState with the given wrap modes, filter, and mipmap mode.
func MakeSamplerState(wrapX, wrapY WrapMode, filter FilterMode, mm MipmapMode) SamplerState {
	return SamplerState{WrapModeX: wrapX, WrapModeY: wrapY, Filter: filter, MipmapMode: mm, MaxAniso: 1}
}

// SamplerStateFilter returns a SamplerState with clamp wrapping and the given filter and mipmap mode.
func SamplerStateFilter(filter FilterMode, mm MipmapMode) SamplerState {
	return MakeSamplerState(WrapModeClamp, WrapModeClamp, filter, mm)
}

// SamplerStateAniso returns an anisotropically-filtered SamplerState, clamping maxAniso to [1, MaxMaxAniso] and
// selecting mipmap sampling when viewIsMipped.
func SamplerStateAniso(wrapX, wrapY WrapMode, maxAniso int32, viewIsMipped Mipmapped) SamplerState {
	if maxAniso < 1 {
		maxAniso = 1
	} else if maxAniso > MaxMaxAniso {
		maxAniso = MaxMaxAniso
	}
	mm := MipmapModeNone
	if viewIsMipped {
		mm = MipmapModeLinear
	}
	return SamplerState{
		WrapModeX:  wrapX,
		WrapModeY:  wrapY,
		Filter:     FilterModeLinear,
		MipmapMode: mm,
		MaxAniso:   maxAniso,
	}
}

// IsRepeatedX reports whether the X wrap mode repeats the texture.
func (s SamplerState) IsRepeatedX() bool {
	return s.WrapModeX == WrapModeRepeat || s.WrapModeX == WrapModeMirrorRepeat
}

// IsRepeatedY reports whether the Y wrap mode repeats the texture.
func (s SamplerState) IsRepeatedY() bool {
	return s.WrapModeY == WrapModeRepeat || s.WrapModeY == WrapModeMirrorRepeat
}

// IsRepeated reports whether either axis repeats the texture.
func (s SamplerState) IsRepeated() bool { return s.IsRepeatedX() || s.IsRepeatedY() }

// Mipmapped reports whether this sampler state samples across mip levels.
func (s SamplerState) Mipmapped() Mipmapped { return s.MipmapMode != MipmapModeNone }

// IsAniso reports whether anisotropic filtering is enabled.
func (s SamplerState) IsAniso() bool { return s.MaxAniso > 1 }

// AsKey packs the state into an integer key. How aniso participates depends on whether the underlying API defines aniso
// as orthogonal to the other filter settings or as a replacement for them.
func (s SamplerState) AsKey(anisoIsOrthogonal bool) uint32 {
	// Bit widths sized to the value counts above: wrap 2, maxAniso 11, filter 1, mipmap 2.
	const (
		wrapXShift      = 0
		wrapYShift      = 2
		maxAnisoShift   = 4
		filterShift     = 15
		mipmapModeShift = 16
	)
	key := uint32(s.WrapModeX)<<wrapXShift | uint32(s.WrapModeY)<<wrapYShift |
		uint32(s.MaxAniso)<<maxAnisoShift
	if s.MaxAniso == 1 || anisoIsOrthogonal {
		key |= uint32(s.Filter)<<filterShift | uint32(s.MipmapMode)<<mipmapModeShift
	}
	return key
}
