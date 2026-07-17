// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// TextureParameters shadows a GL texture object's parameter state, used to skip redundant glTexParameter calls.
// Parameters are considered invalid on all textures after a context reset; that is tracked with a reset timestamp
// rather than walking every texture.

package gl

import "math"

// TextureResetTimestamp identifies a context-reset generation; a texture's cached parameters are stale if their stamped
// timestamp doesn't match the context's current one.
type TextureResetTimestamp uint64

// SamplerOverriddenState is texture parameter state that is overridden when a non-zero sampler object is bound.
type SamplerOverriddenState struct {
	MinFilter uint32
	MagFilter uint32
	WrapS     uint32
	WrapT     uint32
	MinLOD    float32
	MaxLOD    float32
	MaxAniso  float32
	// BorderColorInvalid: we always want the border color to be transparent black, so just track whether it has been
	// invalidated and is no longer the default.
	BorderColorInvalid bool
}

// MakeSamplerOverriddenState returns the OpenGL default values.
func MakeSamplerOverriddenState() SamplerOverriddenState {
	return SamplerOverriddenState{
		MinFilter: NEAREST_MIPMAP_LINEAR,
		MagFilter: LINEAR,
		WrapS:     REPEAT,
		WrapT:     REPEAT,
		MinLOD:    -1000,
		MaxLOD:    1000,
		MaxAniso:  1,
	}
}

func (s *SamplerOverriddenState) invalidate() {
	s.MinFilter = ^uint32(0)
	s.MagFilter = ^uint32(0)
	s.WrapS = ^uint32(0)
	s.WrapT = ^uint32(0)
	s.MinLOD = float32(math.NaN())
	s.MaxLOD = float32(math.NaN())
	s.MaxAniso = -1
	s.BorderColorInvalid = true
}

// NonsamplerState is texture parameter state not overridden by a bound sampler object.
type NonsamplerState struct {
	BaseMipmapLevel int32
	MaxMipmapLevel  int32
	SwizzleIsRGBA   bool
}

// MakeNonsamplerState returns the OpenGL default values.
func MakeNonsamplerState() NonsamplerState {
	return NonsamplerState{BaseMipmapLevel: 0, MaxMipmapLevel: 1000, SwizzleIsRGBA: true}
}

func (s *NonsamplerState) invalidate() {
	s.SwizzleIsRGBA = false
	s.BaseMipmapLevel = math.MinInt32 // any never-used sentinel value works
	s.MaxMipmapLevel = math.MinInt32
}

// TextureParameters shadows a GL texture object's parameters. It is shared, via pointer, between a wrapped backend
// texture and any Texture created from it.
type TextureParameters struct {
	samplerOverriddenState SamplerOverriddenState
	nonsamplerState        NonsamplerState
	resetTimestamp         TextureResetTimestamp
}

// NewTextureParameters returns parameters with an expired timestamp: they are considered invalid the first time the
// texture is used unless Set is called.
func NewTextureParameters() *TextureParameters {
	return &TextureParameters{}
}

// Invalidate marks both the sampler-overridable and non-sampler parameter shadows as stale.
func (p *TextureParameters) Invalidate() {
	p.samplerOverriddenState.invalidate()
	p.nonsamplerState.invalidate()
}

// ResetTimestamp returns the context-reset generation this texture's cached parameters were last set under.
func (p *TextureParameters) ResetTimestamp() TextureResetTimestamp { return p.resetTimestamp }

// SamplerOverriddenState returns the current sampler-overridable parameter shadow.
func (p *TextureParameters) SamplerOverriddenState() *SamplerOverriddenState {
	return &p.samplerOverriddenState
}

// NonsamplerState returns the current non-sampler parameter shadow.
func (p *TextureParameters) NonsamplerState() *NonsamplerState { return &p.nonsamplerState }

// Set stores the current parameter state and the timestamp it was set under. samplerState is optional because it isn't
// tracked when sampler objects are in use.
func (p *TextureParameters) Set(samplerState *SamplerOverriddenState, nonsamplerState NonsamplerState, currTimestamp TextureResetTimestamp) {
	if samplerState != nil {
		p.samplerOverriddenState = *samplerState
	}
	p.nonsamplerState = nonsamplerState
	p.resetTimestamp = currTimestamp
}
