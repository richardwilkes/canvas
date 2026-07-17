// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

// FilterMode selects nearest-neighbor or bilinear image sampling.
type FilterMode int32

// FilterMode values.
const (
	FilterNearest FilterMode = iota
	FilterLinear
)

// MipmapMode selects how mipmap levels are sampled and blended.
type MipmapMode int32

// MipmapMode values.
const (
	MipmapNone MipmapMode = iota
	MipmapNearest
	MipmapLinear
)

// CubicResampler holds the B/C parameters of a bicubic (Mitchell-Netravali family) resampling filter.
type CubicResampler struct {
	B, C float32
}

// SamplingOptions describes how an image shader samples its source pixels.
type SamplingOptions struct {
	MaxAniso int32
	UseCubic bool
	Cubic    CubicResampler
	Filter   FilterMode
	Mipmap   MipmapMode
}

// isAniso reports whether anisotropic filtering is requested.
func (s SamplingOptions) isAniso() bool { return s.MaxAniso != 0 }

// anisoFallback returns the linear-filtering sampling options substituted when anisotropic filtering is requested but
// not supported.
func anisoFallback(imageIsMipped bool) SamplingOptions {
	mm := MipmapNone
	if imageIsMipped {
		mm = MipmapLinear
	}
	return SamplingOptions{Filter: FilterLinear, Mipmap: mm}
}

// cubicResamplerWeights returns the 16-element column-major weight table the bicubic stages consume (w[col*4+row]).
func cubicResamplerWeights(b, c float32) [16]float32 {
	// row-major matrix rows
	m := [4][4]float32{
		{(1.0 / 6) * b, -(3.0/6)*b - c, (3.0/6)*b + 2*c, -(1.0/6)*b - c},
		{1 - (2.0/6)*b, 0, -3 + (12.0/6)*b + c, 2 - (9.0/6)*b - c},
		{(1.0 / 6) * b, (3.0/6)*b + c, 3 - (15.0/6)*b - 2*c, -2 + (9.0/6)*b + c},
		{0, 0, -c, (1.0/6)*b + c},
	}
	var w [16]float32
	for col := range 4 {
		for row := range 4 {
			w[col*4+row] = m[row][col]
		}
	}
	return w
}
