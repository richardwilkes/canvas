// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Renders a filtered path draw into a mask: compute the mask bounds it needs (computeMaskBounds), render the device
// path into a zeroed A8 mask (drawIntoMask), and the A8 blitter (a8Blitter) that rendering draws through, with the
// default srcover paint.

package maskfilter

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// computeMaskBounds computes the mask bounds a filtered path draw needs, clipped to clipBounds plus the filter's
// margin.
func computeMaskBounds(devPathBounds geom.Rect, clipBounds geom.IRect, mf MaskFilter, ctm *geom.Matrix) (geom.IRect, bool) {
	// init our bounds from the path
	bounds := devPathBounds.Outset(0.5, 0.5).RoundOut()

	srcM := &raster.Mask{Bounds: bounds}
	_, margin, ok := mf.FilterMask(srcM, ctm)
	if !ok {
		return geom.IRect{}, false
	}

	// trim the bounds to reflect the clip (plus whatever slop the filter needs). Guard against gigantic margins from
	// wacky filters.
	const maxMargin = 128
	if !bounds.Intersect(clipBounds.Inset(-min32(margin.X, maxMargin), -min32(margin.Y, maxMargin))) {
		return geom.IRect{}, false
	}
	return bounds, true
}

// DrawToMask renders devPath's coverage into a fresh zero-initialized A8 mask whose bounds cover the path clipped to
// clipBounds plus the filter's margin (the form FilterPath consumes). doFill selects fill vs. hairline-stroke
// rendering.
func DrawToMask(devPath *path.Path, clipBounds geom.IRect, mf MaskFilter, ctm *geom.Matrix, doFill bool) (*raster.Mask, bool) {
	if devPath.IsEmpty() {
		return nil, false
	}

	// Using infinite bounds for inverse fills lets computeMaskBounds clip to clipBounds outset by whatever extra
	// margin the mask filter requires.
	pathBounds := devPath.Bounds()
	if devPath.IsInverseFillType() {
		inf := float32(math.Inf(1))
		pathBounds = geom.RectLTRB(-inf, -inf, inf, inf)
	}
	bounds, ok := computeMaskBounds(pathBounds, clipBounds, mf, ctm)
	if !ok {
		return nil, false
	}

	mask := &raster.Mask{Bounds: bounds, RowBytes: bounds.Width()}
	size := computeImageSize(bounds, mask.RowBytes)
	if size == 0 {
		// we're too big to allocate the mask, abort
		return nil, false
	}
	mask.Image = make([]uint8, size)
	drawIntoMask(mask, devPath, doFill)
	return mask, true
}

// RenderPathIntoMask exposes drawIntoMask for the glyph scaler's A8 rendering lane: draw devPath's AA coverage into
// mask, which must already be allocated and zeroed; doFill=false renders the path as an AA hairline.
func RenderPathIntoMask(mask *raster.Mask, devPath *path.Path, doFill bool) {
	drawIntoMask(mask, devPath, doFill)
}

// drawIntoMask translates the path to mask-local coordinates and renders it antialiased through the A8 srcover blitter.
func drawIntoMask(mask *raster.Mask, devPath *path.Path, doFill bool) {
	dx := float32(-mask.Bounds.Left)
	dy := float32(-mask.Bounds.Top)
	var translate geom.Matrix
	translate.SetTranslate(dx, dy)

	local := &path.Path{}
	devPath.TransformTo(&translate, local)
	if !local.IsFinite() {
		return
	}

	blitter := newA8Blitter(mask)
	clip := raster.NewRasterClipRect(geom.IRectWH(mask.Bounds.Width(), mask.Bounds.Height()))
	if doFill {
		raster.AntiFillPathRasterClip(local, clip, blitter)
	} else {
		raster.AntiHairPath(local, clip, blitter)
	}
}

///////////////////////////////////////////////////////////////////////////////
// a8Blitter (srcover, srcAlpha = 255 — the default paint every mask-render path uses)

// a8Blitter implements the srcover row procs with a constant source alpha of 255. It addresses the mask in mask-local
// coordinates (the shared local view rebases Bounds to the origin).
type a8Blitter struct {
	dev raster.Mask
}

// newA8Blitter wraps mask's storage with origin-based bounds.
func newA8Blitter(mask *raster.Mask) *a8Blitter {
	return &a8Blitter{dev: raster.Mask{
		Image:    mask.Image,
		Bounds:   geom.IRectWH(mask.Bounds.Width(), mask.Bounds.Height()),
		RowBytes: mask.RowBytes,
	}}
}

// div255 is the exact rounding divide-by-255.
func div255(prod uint32) uint8 {
	return uint8((prod + 128) * 257 >> 16)
}

// u8Lerp linearly interpolates between a and b by t/255.
func u8Lerp(a, b, t uint8) uint8 {
	return div255((255-uint32(t))*uint32(a) + uint32(t)*uint32(b))
}

// srcoverP composes src over dst for one A8 channel.
func srcoverP(src, dst uint8) uint8 {
	return src + div255((255-uint32(src))*uint32(dst))
}

// BlitH implements raster.Blitter (the srcover bw row proc with src = 255).
func (a *a8Blitter) BlitH(x, y, width int32) {
	i := maskAddr8(&a.dev, x, y)
	row := a.dev.Image[i : i+int(width)]
	for j := range row {
		row[j] = srcoverP(255, row[j])
	}
}

// BlitAntiH implements raster.Blitter (A8_row_aa with canFoldAA = true for srcover).
func (a *a8Blitter) BlitAntiH(x, y int32, antialias []raster.Alpha, runs []int16) {
	base := maskAddr8(&a.dev, x, y)
	off := 0
	for {
		count := int(runs[off])
		if count == 0 {
			return
		}
		switch aa := antialias[off]; aa {
		case 0:
		case 0xFF:
			row := a.dev.Image[base+off : base+off+count]
			for j := range row {
				row[j] = srcoverP(255, row[j])
			}
		default:
			src := div255(255 * uint32(aa))
			row := a.dev.Image[base+off : base+off+count]
			for j := range row {
				row[j] = srcoverP(src, row[j])
			}
		}
		off += count
	}
}

// BlitV implements raster.Blitter.
func (a *a8Blitter) BlitV(x, y, height int32, alpha raster.Alpha) {
	i := maskAddr8(&a.dev, x, y)
	switch alpha {
	case 0:
	case 0xFF:
		for h := int32(0); h < height; h++ {
			a.dev.Image[i] = srcoverP(255, a.dev.Image[i])
			i += int(a.dev.RowBytes)
		}
	default:
		src := div255(255 * uint32(alpha))
		for h := int32(0); h < height; h++ {
			a.dev.Image[i] = srcoverP(src, a.dev.Image[i])
			i += int(a.dev.RowBytes)
		}
	}
}

// BlitRect implements raster.Blitter.
func (a *a8Blitter) BlitRect(x, y, width, height int32) {
	i := maskAddr8(&a.dev, x, y)
	for h := int32(0); h < height; h++ {
		row := a.dev.Image[i : i+int(width)]
		for j := range row {
			row[j] = srcoverP(255, row[j])
		}
		i += int(a.dev.RowBytes)
	}
}

// BlitMask implements raster.Blitter for A8.
func (a *a8Blitter) BlitMask(mask *raster.Mask, clip geom.IRect) {
	src := maskAddr8(mask, clip.Left, clip.Top)
	dst := maskAddr8(&a.dev, clip.Left, clip.Top)
	width := int(clip.Width())
	for h := clip.Height(); h > 0; h-- {
		for i := 0; i < width; i++ {
			d := a.dev.Image[dst+i]
			a.dev.Image[dst+i] = u8Lerp(d, srcoverP(255, d), mask.Image[src+i])
		}
		src += int(mask.RowBytes)
		dst += int(a.dev.RowBytes)
	}
}

// BlitAntiRect implements raster.Blitter via the default lowering.
func (a *a8Blitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha raster.Alpha) {
	raster.BlitAntiRectViaVH(a, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitAntiH2 implements raster.Blitter via the default lowering.
func (a *a8Blitter) BlitAntiH2(x, y int32, a0, a1 raster.Alpha) {
	raster.BlitAntiH2Via(a, x, y, a0, a1)
}

// BlitAntiV2 implements raster.Blitter via the default lowering.
func (a *a8Blitter) BlitAntiV2(x, y int32, a0, a1 raster.Alpha) {
	raster.BlitAntiV2Via(a, x, y, a0, a1)
}
