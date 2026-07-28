// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package imgdiff compares RGBA8888-premul pixel buffers under the named threshold profiles below and renders diff
// heatmaps and side-by-side images for failures. It is cgo-free, like the rest of the harness.
//
// The gating profiles are (near-)bit-exact: the golden references are the port's own per-platform output, so the gates
// ask "did the output change at all?", not "are two renderers visually equivalent?". Exact gates the raster lane;
// Exact1 gates the GPU lanes, tolerating only the ±1 LSB driver-internal wobble of software GL rasterizers. The looser
// CPU/Text/GPU profiles date from when the reference was the C Skia library — a foreign renderer whose output
// legitimately differed at AA edges (SIMD lane order, analytic-AA implementation details, float contraction; more so
// on GPU, where drivers vary) — and no golden gate uses them: they remain for `oracle diff` comparisons against the
// archived Skia renders in ../goldens-skia, and GPU also bounds gorender's atlas CPU-vs-GPU self-consistency
// cross-check between the port's two live backends.
package imgdiff

import (
	"fmt"
	"image"
	"image/color"
)

// Profile is a named threshold set. A pixel is "differing" when any channel differs by more than MaxChannelDelta; a
// comparison passes when the fraction of differing pixels is at most MaxDiffFraction.
type Profile struct {
	Name            string
	MaxChannelDelta uint8
	MaxDiffFraction float64
}

// The threshold profiles. Exact (delta 0) gates the raster lane against its self-captured per-platform goldens. Exact1
// is "exact modulo ±1 LSB" — every channel delta must be ≤ 1 and zero pixels may exceed it — and gates the GPU lanes:
// software GL rasterizers wobble ±1 intermittently between GL sessions, proven driver-internal (identical inputs and
// GL command streams still produce ±1-differing output; see the oracle soak command's doc comment), while real breaks
// measure ≥32 LSB. CPU, Text, and GPU are the cross-renderer tolerances of the era when the C Skia library was the
// reference, kept for archive comparisons: Text is looser than CPU because glyph masks came from a platform scaler,
// and GPU looser still because drivers vary.
var (
	Exact  = Profile{Name: "exact", MaxChannelDelta: 0, MaxDiffFraction: 0}
	Exact1 = Profile{Name: "exact1", MaxChannelDelta: 1, MaxDiffFraction: 0}
	CPU    = Profile{Name: "cpu", MaxChannelDelta: 2, MaxDiffFraction: 0.005}
	Text   = Profile{Name: "text", MaxChannelDelta: 2, MaxDiffFraction: 0.02}
	GPU    = Profile{Name: "gpu", MaxChannelDelta: 4, MaxDiffFraction: 0.01}
)

// ProfileByName returns the named profile.
func ProfileByName(name string) (Profile, bool) {
	for _, p := range []Profile{Exact, Exact1, CPU, Text, GPU} {
		if p.Name == name {
			return p, true
		}
	}
	return Profile{}, false
}

// Result reports one comparison.
type Result struct {
	Profile       Profile
	Width, Height int
	DiffPixels    int   // pixels with any channel delta > profile.MaxChannelDelta
	AnyDiffPixels int   // pixels with any channel delta > 0
	MaxDelta      uint8 // largest channel delta anywhere
}

// DiffFraction is the fraction of pixels exceeding the profile's channel tolerance.
func (r *Result) DiffFraction() float64 {
	if r.Width*r.Height == 0 {
		return 0
	}
	return float64(r.DiffPixels) / float64(r.Width*r.Height)
}

// Pass reports whether the comparison met the profile.
func (r *Result) Pass() bool {
	return r.DiffFraction() <= r.Profile.MaxDiffFraction
}

func (r Result) String() string {
	return fmt.Sprintf("%dx%d: %d px differ (>%d/255) = %.4f%% (limit %.4f%%), %d px differ at all, max delta %d",
		r.Width, r.Height, r.DiffPixels, r.Profile.MaxChannelDelta, r.DiffFraction()*100,
		r.Profile.MaxDiffFraction*100, r.AnyDiffPixels, r.MaxDelta)
}

func absDelta(a, b byte) uint8 {
	if a > b {
		return a - b
	}
	return b - a
}

// Compare diffs two same-sized RGBA8888 buffers (4 bytes per pixel, tightly packed).
func Compare(a, b []byte, width, height int, profile Profile) (Result, error) {
	n := width * height * 4
	if len(a) != n || len(b) != n {
		return Result{}, fmt.Errorf("imgdiff: buffer sizes %d/%d do not match %dx%dx4=%d", len(a), len(b), width, height, n)
	}
	res := Result{Width: width, Height: height, Profile: profile}
	for i := 0; i < n; i += 4 {
		var maxD uint8
		for j := range 4 {
			if d := absDelta(a[i+j], b[i+j]); d > maxD {
				maxD = d
			}
		}
		if maxD > 0 {
			res.AnyDiffPixels++
			if maxD > res.MaxDelta {
				res.MaxDelta = maxD
			}
			if maxD > profile.MaxChannelDelta {
				res.DiffPixels++
			}
		}
	}
	return res, nil
}

// Heatmap renders per-pixel max channel deltas as an image: black where identical, then a blue→yellow→red ramp (delta 1
// is already clearly visible; each band's label in heatColor is its inclusive upper bound, so red means delta > 64).
func Heatmap(a, b []byte, width, height int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			i := (y*width + x) * 4
			var maxD uint8
			for j := range 4 {
				if d := absDelta(a[i+j], b[i+j]); d > maxD {
					maxD = d
				}
			}
			img.SetNRGBA(x, y, heatColor(maxD))
		}
	}
	return img
}

func heatColor(delta uint8) color.NRGBA {
	switch {
	case delta == 0:
		return color.NRGBA{A: 0xFF}
	case delta <= 2:
		return color.NRGBA{B: 0xFF, G: 0x60, A: 0xFF} // small: blue
	case delta <= 8:
		return color.NRGBA{R: 0x30, G: 0xC0, B: 0xFF, A: 0xFF} // cyan
	case delta <= 32:
		return color.NRGBA{R: 0xFF, G: 0xD0, A: 0xFF} // yellow
	case delta <= 64:
		return color.NRGBA{R: 0xFF, G: 0x80, A: 0xFF} // orange
	default:
		return color.NRGBA{R: 0xFF, A: 0xFF} // red
	}
}

// SideBySide composites a | b | heatmap horizontally with a 2px separator, for CI failure artifacts.
func SideBySide(a, b []byte, width, height int) *image.NRGBA {
	const sep = 2
	out := image.NewNRGBA(image.Rect(0, 0, width*3+sep*2, height))
	put := func(xoff int, px []byte) {
		for y := range height {
			for x := range width {
				i := (y*width + x) * 4
				// Premul bytes shown as straight alpha: good enough for a debugging artifact.
				out.SetNRGBA(x+xoff, y, color.NRGBA{R: px[i], G: px[i+1], B: px[i+2], A: px[i+3]})
			}
		}
	}
	put(0, a)
	put(width+sep, b)
	hm := Heatmap(a, b, width, height)
	for y := range height {
		for x := range width {
			out.Set(x+(width+sep)*2, y, hm.NRGBAAt(x, y))
		}
	}
	for y := range height {
		for s := range sep {
			out.SetNRGBA(width+s, y, color.NRGBA{R: 0x80, G: 0x80, B: 0x80, A: 0xFF})
			out.SetNRGBA(width*2+sep+s, y, color.NRGBA{R: 0x80, G: 0x80, B: 0x80, A: 0xFF})
		}
	}
	return out
}
