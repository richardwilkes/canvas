// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The Clip-aware fill entry points: each dispatches to the region-clipped scan converter directly for BW clips,
// or through an AAClipBlitter for AA clips.

package raster

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

func blitIRect(blitter Blitter, r geom.IRect) {
	blitter.BlitRect(r.Left, r.Top, r.Width(), r.Height())
}

// FillIRectRegion fills r restricted to clip, without antialiasing. A nil clip fills the rect unrestricted.
func FillIRectRegion(r geom.IRect, clip *Region, blitter Blitter) {
	if r.IsEmpty() {
		return
	}
	if clip != nil {
		if clip.IsRect() {
			clipBounds := clip.Bounds()
			if clipBounds.ContainsRect(r) {
				blitIRect(blitter, r)
			} else {
				rr := r
				if rr.Intersect(clipBounds) {
					blitIRect(blitter, rr)
				}
			}
		} else {
			for c := NewRegionCliperator(clip, r); !c.Done(); c.Next() {
				blitIRect(blitter, c.Rect())
			}
		}
	} else {
		blitIRect(blitter, r)
	}
}

// FillRectRegion fills the rect, rounding its coordinates, without antialiasing.
func FillRectRegion(r geom.Rect, clip *Region, blitter Blitter) {
	FillIRectRegion(r.Round(), clip, blitter)
}

// FillXRectRegion rounds the Fixed coordinates and fills without antialiasing.
func FillXRectRegion(xr XRect, clip *Region, blitter Blitter) {
	FillIRectRegion(xr.Round(), clip, blitter)
}

// FillXRect fills xr restricted to a raster clip, without antialiasing.
func FillXRect(xr XRect, clip *Clip, blitter Blitter) {
	if clip.IsEmpty() || xr.IsEmpty() {
		return
	}
	if clip.IsBW() {
		FillXRectRegion(xr, clip.BWRgn(), blitter)
		return
	}
	var wrapper AAClipBlitterWrapper
	wrapper.Init(clip, blitter)
	FillXRectRegion(xr, wrapper.Rgn(), wrapper.Blitter())
}

// FillIRectRasterClip fills r restricted to a raster clip, without antialiasing.
func FillIRectRasterClip(r geom.IRect, clip *Clip, blitter Blitter) {
	if clip.IsEmpty() || r.IsEmpty() {
		return
	}
	if clip.IsBW() {
		FillIRectRegion(r, clip.BWRgn(), blitter)
		return
	}
	var wrapper AAClipBlitterWrapper
	wrapper.Init(clip, blitter)
	FillIRectRegion(r, wrapper.Rgn(), wrapper.Blitter())
}

// FillRectRasterClip fills r restricted to a raster clip, without antialiasing.
func FillRectRasterClip(r geom.Rect, clip *Clip, blitter Blitter) {
	if clip.IsEmpty() || r.IsEmpty() {
		return
	}
	if clip.IsBW() {
		FillRectRegion(r, clip.BWRgn(), blitter)
		return
	}
	var wrapper AAClipBlitterWrapper
	wrapper.Init(clip, blitter)
	FillRectRegion(r, wrapper.Rgn(), wrapper.Blitter())
}

// FillPathRasterClip fills the path, without antialiasing, restricted to the raster clip.
func FillPathRasterClip(p *path.Path, clip *Clip, blitter Blitter) {
	if clip.IsEmpty() {
		return
	}
	if clip.IsBW() {
		FillPathRegion(p, clip.BWRgn(), blitter)
		return
	}
	tmp := NewRegionRect(clip.Bounds())
	var aaBlitter AAClipBlitter
	aaBlitter.Init(blitter, clip.AARgn())
	FillPathRegion(p, tmp, &aaBlitter)
}

// AntiFillPathRasterClip fills the path with analytic antialiasing, restricted to the raster clip.
func AntiFillPathRasterClip(p *path.Path, clip *Clip, blitter Blitter) {
	if clip.IsEmpty() {
		return
	}
	if clip.IsBW() {
		antiFillPathRegion(p, clip.BWRgn(), blitter, false)
		return
	}
	tmp := NewRegionRect(clip.Bounds())
	var aaBlitter AAClipBlitter
	aaBlitter.Init(blitter, clip.AARgn())
	antiFillPathRegion(p, tmp, &aaBlitter, true) // forceRLE: AAClipBlitter can BlitMask, but RLE is used here
}
