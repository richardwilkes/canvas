// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// swMaskHelper renders clip elements into an A8 coverage mask on the CPU and uploads it as a texture for the clip
// stack's software-mask fallback. Rects, round rects, and paths all render through their path form via the CPU AA/BW
// scan converters, using "replace" (kSrc) row procs. The texture is uploaded through the lazy-upload proxy lane with an
// exact backing fit.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// swMaskHelper is a CPU-rasterized A8 coverage mask, along with the clip and translation state used to render shapes
// into it.
type swMaskHelper struct {
	clip      *raster.Clip
	pixels    []uint8
	translate geom.Point
	dims      geom.ISize
}

// init allocates a zeroed A8 buffer covering resultBounds, with draws translated so the bound's upper-left corner is at
// the origin.
func (h *swMaskHelper) init(resultBounds geom.IRect) bool {
	h.translate = geom.Point{X: -float32(resultBounds.Left), Y: -float32(resultBounds.Top)}
	h.dims = geom.ISize{Width: resultBounds.Width(), Height: resultBounds.Height()}
	if h.dims.Width <= 0 || h.dims.Height <= 0 {
		return false
	}
	h.pixels = make([]uint8, int(h.dims.Width)*int(h.dims.Height))
	h.clip = raster.NewRasterClipRect(geom.IRectWH(h.dims.Width, h.dims.Height))
	return true
}

// clear fills the entire mask with alpha.
func (h *swMaskHelper) clear(alpha uint8) {
	for i := range h.pixels {
		h.pixels[i] = alpha
	}
}

// drawShape draws the shape with the given coverage value using "replace" (kSrc) semantics.
func (h *swMaskHelper) drawShape(shape *Shape, matrix *geom.Matrix, aa gpu.AA, alpha uint8) {
	translatedMatrix := *matrix
	translatedMatrix.PostTranslate(h.translate.X, h.translate.Y)

	if shape.Inverted() {
		if shape.IsEmpty() || shape.IsLine() || shape.IsPoint() {
			// These shapes are empty for simple fills, so when inverted, cover everything.
			h.drawPaint(alpha)
			return
		}
		// Else fall through to the path draw, which handles the inverse fill type.
	} else if shape.IsEmpty() || shape.IsLine() || shape.IsPoint() {
		// These shapes do not cover any pixels for simple fills.
		return
	}

	// Rects, rrects, and paths all render through their path form, using the same scan converters.
	p := shape.AsPath()
	p.Transform(&translatedMatrix)
	if !p.IsFinite() {
		return
	}
	blitter := &a8SrcBlitter{pixels: h.pixels, rowBytes: int(h.dims.Width), src: alpha}
	if aa == gpu.AAYes {
		raster.AntiFillPathRasterClip(p, h.clip, blitter)
	} else {
		raster.FillPathRasterClip(p, h.clip, blitter)
	}
}

// drawPaint fills the whole mask with the given value using replace semantics.
func (h *swMaskHelper) drawPaint(alpha uint8) {
	h.clear(alpha)
}

// drawStyledShape draws a styled shape into the mask. Only fill and hairline styles can reach the mask lanes (the SW
// path renderer requires !style.Applies(); style application happens upstream in drawShapeUsingPathRenderer), so this
// reduces to the fill and hairline draw lanes.
func (h *swMaskHelper) drawStyledShape(shape *StyledShape, matrix *geom.Matrix, aa gpu.AA, alpha uint8) {
	if shape.Style().Applies() {
		panic("styles must have been applied before the mask draw")
	}
	if shape.Style().IsSimpleHairline() {
		translated := *matrix
		translated.PostTranslate(h.translate.X, h.translate.Y)
		p := path.Borrow()
		defer path.Recycle(p)
		shape.AsPath().TransformTo(&translated, p)
		if !p.IsFinite() {
			return
		}
		blitter := &a8SrcBlitter{pixels: h.pixels, rowBytes: int(h.dims.Width), src: alpha}
		// AA is applied here per the paint; hairline coverage modulation belongs to the GPU ops' per-vertex coverage,
		// while this rasterizes the geometric hairline itself.
		if aa == gpu.AAYes {
			raster.AntiHairPath(p, h.clip, blitter)
		} else {
			raster.HairPath(p, h.clip, blitter)
		}
		return
	}
	h.drawShape(shape.Shape(), matrix, aa, alpha)
}

// toTextureView wraps the rendered mask in an uncached lazy-upload texture proxy view.
func (h *swMaskHelper) toTextureView(ctx *DirectContext) SurfaceProxyView {
	return a8DataToTextureView(ctx, h.pixels, int(h.dims.Width), h.dims,
		"ClipStack_RenderSwMask")
}

// a8DataToTextureView wraps CPU A8 coverage data in an uncached lazy-upload texture proxy view (shared by the SW clip
// masks, the SW path-draw masks, and the CPU mask-filter draw lane).
func a8DataToTextureView(ctx *DirectContext, data []uint8, rowBytes int, dims geom.ISize, label string) SurfaceProxyView {
	caps := ctx.GLCaps()
	format := caps.GetDefaultBackendFormat(gpu.ColorTypeAlpha8, false /* renderable */)
	if format == FormatUnknown {
		return SurfaceProxyView{}
	}
	swizzle := caps.ReadSwizzle(format, gpu.ColorTypeAlpha8)
	proxy := ctx.ProxyProvider().CreateLazyProxy(
		func(rp *ResourceProvider, desc *LazySurfaceDesc) LazyCallbackResult {
			tex := rp.CreateTextureWithData(desc.Dimensions, desc.Format, desc.TextureType,
				gpu.ColorTypeAlpha8, gpu.RenderableNo, 1, desc.Budgeted, gpu.MipmappedNo,
				[]gpu.MipLevel{{Pixels: data, RowBytes: rowBytes}}, desc.Label)
			return MakeLazyCallbackResult(tex)
		},
		format, dims, gpu.MipmappedNo, gpu.MipmapStatusNotAllocated, gpu.SurfaceFlags(0),
		gpu.BackingFitExact, gpu.BudgetedYes, UseAllocatorYes, label,
	)
	if proxy == nil {
		return SurfaceProxyView{}
	}
	return MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, swizzle)
}

//////////////////////////////////////////////////////////////////////////////
// An A8 blitter with "replace" (kSrc) row procs: full-coverage spans write src directly; partial coverage blends dst
// toward src by the coverage amount.

type a8SrcBlitter struct {
	pixels   []uint8
	rowBytes int
	src      uint8
}

func (b *a8SrcBlitter) addr(x, y int32) int { return int(y)*b.rowBytes + int(x) }

// a8Div255 divides prod by 255 with round-to-nearest, using integer arithmetic only.
func a8Div255(prod uint32) uint8 {
	return uint8((prod + 128) * 257 >> 16)
}

// a8Lerp linearly interpolates between a and bb by t/255.
func a8Lerp(a, bb, t uint8) uint8 {
	return a8Div255((255-uint32(t))*uint32(a) + uint32(t)*uint32(bb))
}

// BlitH implements raster.Blitter.
func (b *a8SrcBlitter) BlitH(x, y, width int32) {
	i := b.addr(x, y)
	row := b.pixels[i : i+int(width)]
	for j := range row {
		row[j] = b.src
	}
}

// BlitAntiH implements raster.Blitter.
func (b *a8SrcBlitter) BlitAntiH(x, y int32, antialias []raster.Alpha, runs []int16) {
	base := b.addr(x, y)
	off := 0
	for {
		count := int(runs[off])
		if count == 0 {
			return
		}
		switch aa := antialias[off]; aa {
		case 0:
		case 0xFF:
			row := b.pixels[base+off : base+off+count]
			for j := range row {
				row[j] = b.src
			}
		default:
			row := b.pixels[base+off : base+off+count]
			for j := range row {
				row[j] = a8Lerp(row[j], b.src, aa)
			}
		}
		off += count
	}
}

// BlitV implements raster.Blitter.
func (b *a8SrcBlitter) BlitV(x, y, height int32, alpha raster.Alpha) {
	i := b.addr(x, y)
	switch aa := alpha; aa {
	case 0:
	case 0xFF:
		for h := int32(0); h < height; h++ {
			b.pixels[i] = b.src
			i += b.rowBytes
		}
	default:
		for h := int32(0); h < height; h++ {
			b.pixels[i] = a8Lerp(b.pixels[i], b.src, aa)
			i += b.rowBytes
		}
	}
}

// BlitRect implements raster.Blitter.
func (b *a8SrcBlitter) BlitRect(x, y, width, height int32) {
	i := b.addr(x, y)
	for h := int32(0); h < height; h++ {
		row := b.pixels[i : i+int(width)]
		for j := range row {
			row[j] = b.src
		}
		i += b.rowBytes
	}
}

// BlitMask implements raster.Blitter.
func (b *a8SrcBlitter) BlitMask(mask *raster.Mask, clip geom.IRect) {
	src := int(clip.Top-mask.Bounds.Top)*int(mask.RowBytes) + int(clip.Left-mask.Bounds.Left)
	dst := b.addr(clip.Left, clip.Top)
	width := int(clip.Width())
	for h := clip.Height(); h > 0; h-- {
		for i := 0; i < width; i++ {
			b.pixels[dst+i] = a8Lerp(b.pixels[dst+i], b.src, mask.Image[src+i])
		}
		src += int(mask.RowBytes)
		dst += b.rowBytes
	}
}

// BlitAntiRect implements raster.Blitter via the default lowering.
func (b *a8SrcBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha raster.Alpha) {
	raster.BlitAntiRectViaVH(b, x, y, width, height, leftAlpha, rightAlpha)
}

// BlitAntiH2 implements raster.Blitter via the default lowering.
func (b *a8SrcBlitter) BlitAntiH2(x, y int32, a0, a1 raster.Alpha) {
	raster.BlitAntiH2Via(b, x, y, a0, a1)
}

// BlitAntiV2 implements raster.Blitter via the default lowering.
func (b *a8SrcBlitter) BlitAntiV2(x, y int32, a0, a1 raster.Alpha) {
	raster.BlitAntiV2Via(b, x, y, a0, a1)
}
