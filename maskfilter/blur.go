// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The blur mask filter, with its sigma-vs-CTM handling and the rect/rrect nine-patch fast paths. There is no mask cache
// — every patch is computed on demand; outputs are identical, only repeated-draw cost differs.

package maskfilter

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// blurMaskFilter is the blur mask filter implementation.
type blurMaskFilter struct {
	sigma      float32
	style      BlurStyle
	respectCTM bool
}

// NewBlur creates a blur mask filter; nil unless sigma is finite and positive.
func NewBlur(style BlurStyle, sigma float32, respectCTM bool) MaskFilter {
	if geom.IsFinite(sigma) && sigma > 0 && style <= BlurInner {
		return &blurMaskFilter{sigma: sigma, style: style, respectCTM: respectCTM}
	}
	return nil
}

// AcceptsColorMask reports whether filterMask on an ARGB32 (color glyph) source would succeed for this filter: only the
// blur filter blurs color masks, by extracting the alpha plane (its dst is A8); the table and shader mask filters guard
// on A8 masks and return false. The color-glyph scaler lanes consult this instead of carrying a format on raster.Mask.
func AcceptsColorMask(mf MaskFilter) bool {
	_, ok := mf.(*blurMaskFilter)
	return ok
}

// Sigma, Style, and RespectCTM return the descriptor fields — the GPU side consumes these.
func (f *blurMaskFilter) Sigma() float32 { return f.sigma }

// Style returns the blur style.
func (f *blurMaskFilter) Style() BlurStyle { return f.style }

// RespectCTM reports whether the sigma is interpreted in local space (true) or device space.
func (f *blurMaskFilter) RespectCTM() bool { return f.respectCTM }

// ignoreXform reports whether the sigma should be used as-is, ignoring the CTM.
func (f *blurMaskFilter) ignoreXform() bool { return !f.respectCTM }

// ComputeXformedSigma returns the sigma after accounting for the CTM; the GPU mask-filter draw lane consumes it.
func (f *blurMaskFilter) ComputeXformedSigma(ctm *geom.Matrix) float32 {
	return f.computeXformedSigma(ctm)
}

// computeXformedSigma maps the filter's sigma through ctm (unless ignoreXform), clamped to the maximum supported blur
// radius.
func (f *blurMaskFilter) computeXformedSigma(ctm *geom.Matrix) float32 {
	const kMaxBlurSigma = float32(128)
	xformedSigma := f.sigma
	if !f.ignoreXform() {
		xformedSigma = ctm.MapRadius(f.sigma)
	}
	if xformedSigma > kMaxBlurSigma {
		return kMaxBlurSigma
	}
	return xformedSigma
}

// FilterMask implements MaskFilter: blurs src by the CTM-adjusted sigma.
func (f *blurMaskFilter) FilterMask(src *raster.Mask, ctm *geom.Matrix) (*raster.Mask, geom.IPoint, bool) {
	return boxBlur(src, f.computeXformedSigma(ctm), f.style)
}

// filterRectMask blurs the rect r analytically. dst is a caller-owned (reused) mask header.
func (f *blurMaskFilter) filterRectMask(r geom.Rect, ctm *geom.Matrix, createMode maskCreateMode, scratch *blurScratch, dst *raster.Mask) (geom.IPoint, bool) {
	return blurRect(f.computeXformedSigma(ctm), r, f.style, createMode, scratch, dst)
}

// ComputeFastBounds implements MaskFilter: pad by 3 sigma.
func (f *blurMaskFilter) ComputeFastBounds(src geom.Rect) geom.Rect {
	pad := 3.0 * f.sigma
	return src.Outset(pad, pad)
}

// rectExceeds reports whether any edge of r lies beyond v (or v's negation) or exceeds it in size.
func rectExceeds(r geom.Rect, v float32) bool {
	return r.Left < -v || r.Top < -v || r.Right > v || r.Bottom > v ||
		r.Width() > v || r.Height() > v
}

// prepareToDrawIntoMask allocates an align-4 row-byte zeroed A8 mask over bounds.RoundOut().
func prepareToDrawIntoMask(bounds geom.Rect) (*raster.Mask, bool) {
	mask := &raster.Mask{Bounds: bounds.RoundOut()}
	mask.RowBytes = (mask.Bounds.Width() + 3) &^ 3
	size := computeImageSize(mask.Bounds, mask.RowBytes)
	if size == 0 {
		return nil, false
	}
	mask.Image = make([]uint8, size)
	return mask, true
}

// drawIntoMaskLocal renders drawer's geometry, translated to mask-local coordinates, through the A8 srcover blitter
// with an antialiased fill — the draw_into_mask template's fixed environment.
func drawIntoMaskLocal(mask *raster.Mask, drawer func(translate *geom.Matrix, clip *raster.Clip, blitter raster.Blitter)) {
	dx := -mask.Bounds.Left
	dy := -mask.Bounds.Top
	var translate geom.Matrix
	translate.SetTranslate(float32(dx), float32(dy))
	clip := raster.NewRasterClipRect(geom.IRectWH(mask.Bounds.Width(), mask.Bounds.Height()))
	drawer(&translate, clip, newA8Blitter(mask))
}

// drawRectsIntoMask renders one or two rects into a fresh mask: one rect antifills directly; a nested pair fills
// even-odd through the path scan converter, matching how an antialiased fill paint renders a rect or path under a
// translate-only CTM.
func drawRectsIntoMask(rects []geom.Rect) (*raster.Mask, bool) {
	mask, ok := prepareToDrawIntoMask(rects[0])
	if !ok {
		return nil, false
	}
	drawIntoMaskLocal(mask, func(translate *geom.Matrix, clip *raster.Clip, blitter raster.Blitter) {
		if len(rects) == 1 {
			dev, _ := translate.MapRect(rects[0])
			raster.AntiFillRect(dev, clip, blitter)
			return
		}
		p := &path.Path{}
		p.AddRect(rects[0], geom.DirectionCW)
		p.AddRect(rects[1], geom.DirectionCW)
		p.SetFillType(path.FillEvenOdd)
		dev := &path.Path{}
		p.TransformTo(translate, dev)
		raster.AntiFillPathRasterClip(dev, clip, blitter)
	})
	return mask, true
}

// drawRRectIntoMask renders rrect into a fresh mask.
func drawRRectIntoMask(rrect geom.RRect) (*raster.Mask, bool) {
	mask, ok := prepareToDrawIntoMask(rrect.Rect)
	if !ok {
		return nil, false
	}
	drawIntoMaskLocal(mask, func(translate *geom.Matrix, clip *raster.Clip, blitter raster.Blitter) {
		p := &path.Path{}
		p.AddRRect(rrect, geom.DirectionCW)
		dev := &path.Path{}
		p.TransformTo(translate, dev)
		raster.AntiFillPathRasterClip(dev, clip, blitter)
	})
	return mask, true
}

// filterRRectToNine attempts the nine-patch fast path for a blurred round rect (uncached).
func (f *blurMaskFilter) filterRRectToNine(rrect geom.RRect, ctm *geom.Matrix) (ninePatch, bool) {
	switch rrect.Type {
	case geom.RRectEmpty, geom.RRectRect, geom.RRectOval:
		// Nothing to draw, a different special case, and unhandled ovals respectively.
		return ninePatch{}, false
	default:
	}

	// TODO: report correct metrics for innerstyle, where we do not grow the total bounds, but we do need an inset the
	// size of our blur-radius
	if f.style == BlurInner {
		return ninePatch{}, false
	}

	// TODO: take clipBounds into account to limit our coordinates up front; for now, just skip too-large src rects (to
	// take the old code path).
	if rectExceeds(rrect.Rect, 32767) {
		return ninePatch{}, false
	}

	// Figure out how much we need to expand the mask (the margin) for the blurred area: a bounds-only filterMask over
	// the rrect's bounds.
	srcM := &raster.Mask{Bounds: rrect.Rect.RoundOut()}
	dstM, margin, ok := f.FilterMask(srcM, ctm)
	if !ok {
		return ninePatch{}, false
	}

	// The uniform-radii subset means all four corners share (rx, ry).
	leftUnstretched := geom.CeilToInt(rrect.RadiusX) + margin.X
	rightUnstretched := geom.CeilToInt(rrect.RadiusX) + margin.X

	// Extra space in the middle to ensure an unchanging piece for stretching.
	const stretchSize = int32(1)

	totalSmallWidth := leftUnstretched + rightUnstretched + stretchSize
	if float32(totalSmallWidth) >= rrect.Rect.Width() {
		// There is no valid piece to stretch.
		return ninePatch{}, false
	}

	topUnstretched := geom.CeilToInt(rrect.RadiusY) + margin.Y
	botUnstretched := geom.CeilToInt(rrect.RadiusY) + margin.Y

	totalSmallHeight := topUnstretched + botUnstretched + stretchSize
	if float32(totalSmallHeight) >= rrect.Rect.Height() {
		// There is no valid piece to stretch.
		return ninePatch{}, false
	}

	// Now make that scaled down nine patch rrect.
	smallR := geom.RectLTRB(0, 0, float32(totalSmallWidth), float32(totalSmallHeight))
	smallRR := geom.MakeRRect(smallR, rrect.RadiusX, rrect.RadiusY)

	// Blit the small rrect into a buffer and blur it (the mask cache is not ported).
	small, ok := drawRRectIntoMask(smallRR)
	if !ok {
		return ninePatch{}, false
	}
	filterM, _, ok := f.FilterMask(small, ctm)
	if !ok {
		return ninePatch{}, false
	}

	// The bounds of the blurred mask are at -margin, so offset it back to 0,0.
	bounds := filterM.Bounds
	bounds = geom.IRectWH(bounds.Width(), bounds.Height())
	return ninePatch{
		mask:      raster.Mask{Image: filterM.Image, Bounds: bounds, RowBytes: filterM.RowBytes},
		outerRect: dstM.Bounds,
		center:    geom.IPoint{X: margin.X + leftUnstretched, Y: margin.Y + topUnstretched},
	}, true
}

// filterRectsToNine attempts the nine-patch fast path for one or two blurred rects (uncached). scratch supplies the
// pooled analytic-blur buffers for the single-rect fast path (blurRect); the nested-rect path blurs through boxBlur,
// which allocates its own image.
func (f *blurMaskFilter) filterRectsToNine(rects []geom.Rect, ctm *geom.Matrix, scratch *blurScratch) (ninePatch, FilterReturn) {
	// TODO: report correct metrics for innerstyle, where we do not grow the total bounds, but we do need an inset the
	// size of our blur-radius
	if f.style == BlurInner || f.style == BlurOuter {
		return ninePatch{}, FilterUnimplemented
	}

	// TODO: take clipBounds into account to limit our coordinates up front; for now, just skip too-large src rects (to
	// take the old code path).
	if rectExceeds(rects[0], 32767) {
		return ninePatch{}, FilterUnimplemented
	}

	srcM := &raster.Mask{Bounds: rects[0].RoundOut()}
	var dstM *raster.Mask
	var ok bool
	if len(rects) == 1 {
		// special case for fast rect blur: don't actually do the blur the first time, just compute the correct size.
		// blurRect fills the reused scratch.boundsMask header (no per-draw alloc).
		if _, ok = f.filterRectMask(rects[0], ctm, justComputeBounds, scratch, &scratch.boundsMask); ok {
			dstM = &scratch.boundsMask
		}
	} else {
		dstM, _, ok = f.FilterMask(srcM, ctm)
	}
	if !ok {
		return ninePatch{}, FilterFalse
	}

	// smallR is the smallest version of 'rect' that will still guarantee that we get the same blur results on all
	// edges, plus 1 center row/col that is representative of the stretchable edges.
	var smallR [2]geom.Rect
	var rectCount int
	var center geom.IPoint

	// +2 is from +1 for each edge (to account for possible fractional edges)
	smallW := dstM.Bounds.Width() - srcM.Bounds.Width() + 2
	smallH := dstM.Bounds.Height() - srcM.Bounds.Height() + 2
	var innerIR geom.IRect

	if len(rects) == 1 {
		rectCount = 1
		innerIR = srcM.Bounds
		center = geom.IPoint{X: smallW, Y: smallH}
	} else {
		rectCount = 2
		innerIR = rects[1].RoundIn()
		center = geom.IPoint{
			X: smallW + (innerIR.Left - srcM.Bounds.Left),
			Y: smallH + (innerIR.Top - srcM.Bounds.Top),
		}
	}

	// +1 so we get a clean, stretchable, center row/col
	smallW++
	smallH++

	// we want the inset amounts to be integral, so we don't change any fractional phase on the fRight or fBottom of our
	// smallR.
	dx := float32(innerIR.Width() - smallW)
	dy := float32(innerIR.Height() - smallH)
	if dx < 0 || dy < 0 {
		// we're too small, relative to our blur, to break into nine-patch, so we ask to have our normal filterMask() be
		// called.
		return ninePatch{}, FilterUnimplemented
	}

	smallR[0] = geom.RectLTRB(rects[0].Left, rects[0].Top, rects[0].Right-dx, rects[0].Bottom-dy)
	if smallR[0].Width() < 2 || smallR[0].Height() < 2 {
		return ninePatch{}, FilterUnimplemented
	}
	if rectCount == 2 {
		smallR[1] = geom.RectLTRB(rects[1].Left, rects[1].Top, rects[1].Right-dx, rects[1].Bottom-dy)
	}

	var filterM *raster.Mask
	if rectCount == 2 {
		src, drawOK := drawRectsIntoMask(smallR[:2])
		if !drawOK {
			return ninePatch{}, FilterFalse
		}
		if filterM, _, ok = f.FilterMask(src, ctm); !ok {
			return ninePatch{}, FilterFalse
		}
	} else {
		// blurRect fills the reused scratch.imageMask header (no per-draw alloc); its image field aliases
		// scratch.image, which the returned nine-patch mask then references.
		if _, ok = f.filterRectMask(smallR[0], ctm, computeBoundsAndRenderImage, scratch, &scratch.imageMask); !ok {
			return ninePatch{}, FilterFalse
		}
		filterM = &scratch.imageMask
	}
	bounds := geom.IRectWH(filterM.Bounds.Width(), filterM.Bounds.Height())
	return ninePatch{
		mask:      raster.Mask{Image: filterM.Image, Bounds: bounds, RowBytes: filterM.RowBytes},
		outerRect: dstM.Bounds,
		center:    center,
	}, FilterTrue
}
