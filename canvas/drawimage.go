// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The image draws: the bitmap/sprite lowering, the device drawImageRect, the sprite-eligibility test, the canvas entry
// points (including the image-filter lane via aboutToDraw, and DrawImageNine → the lattice draw), and the device
// lattice draw over the lattice iterator.

package canvas

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// SrcRectConstraint controls how strictly an image draw's sampling is confined to the source rect; values match
// Skia's SkCanvas::SrcRectConstraint ordinals.
type SrcRectConstraint int32

// SrcRectConstraint values.
const (
	ConstraintStrict SrcRectConstraint = iota
	ConstraintFast
)

// treatAsSprite reports whether the matrix maps the w×h image onto integer pixel bounds so the draw can use the sprite
// blitters (with subpixel tolerance when anti-aliasing).
func treatAsSprite(mat *geom.Matrix, w, h int32, sampling shaders.SamplingOptions, isAntiAlias bool) bool {
	// Cubic resampling with B != 0 alters pixels even under an identity matrix, so never sprite it.
	if sampling.UseCubic && sampling.Cubic.B != 0 {
		return false
	}

	const antiAliasSubpixelBits = 4
	subpixelBits := 0
	if isAntiAlias {
		subpixelBits = antiAliasSubpixelBits
	}

	// quick reject on affine or perspective
	if mat.Type()&^uint8(geom.TypeScale|geom.TypeTranslate) != 0 {
		return false
	}

	// Don't snap to pixels when asking for linear filtering with a subpixel translation.
	tx := mat.Get(geom.MTransX)
	ty := mat.Get(geom.MTransY)
	if sampling.Filter == shaders.FilterLinear &&
		(tx != float32(int32(tx)) || ty != float32(int32(ty))) {
		return false
	}

	// quick success check
	if subpixelBits == 0 && mat.Type()&^uint8(geom.TypeTranslate) == 0 {
		return true
	}

	// mapRect supports negative scales, so eliminate those first
	if mat.Get(geom.MScaleX) < 0 || mat.Get(geom.MScaleY) < 0 {
		return false
	}

	dst, _ := path.MapRect(mat, geom.RectWH(float32(w), float32(h)))
	isrc := geom.IRectWH(w, h)

	// just apply the translate to isrc
	isrc = isrc.Offset(scalarRoundToInt(tx), scalarRoundToInt(ty))

	if subpixelBits > 0 {
		isrc.Left <<= subpixelBits
		isrc.Top <<= subpixelBits
		isrc.Right <<= subpixelBits
		isrc.Bottom <<= subpixelBits
		scale := float32(int32(1) << subpixelBits)
		dst.Left *= scale
		dst.Top *= scale
		dst.Right *= scale
		dst.Bottom *= scale
	}

	return isrc == dst.Round()
}

// scalarRoundToInt rounds half up to an int32.
func scalarRoundToInt(v float32) int32 {
	return int32(math.Floor(float64(v) + 0.5))
}

// imageDrawScratch holds the per-draw temporaries the CPU image-draw path would otherwise allocate fresh on every draw:
// the wrapping Image and base ImageShader that makePaintWithImage builds, plus drawBitmap's CTM∘prematrix concatenation
// and drawRectFull's image-lane shading matrix. Homed on a pooled (heap-resident) scratch so their addresses — which
// flow into retaining callees (blitter construction, the draw struct's ctm) — no longer force a per-draw heap
// allocation. Pooled (not a shared instance) so nested/concurrent draws stay independent.
type imageDrawScratch struct {
	image    *imagecore.Image    // lazily allocated once per scratch, re-init per draw via SetFromPixels
	shader   shaders.ImageShader // reused base clamp/clamp image shader
	matrix   geom.Matrix         // drawBitmap's CTM∘prematrix concatenation
	combined geom.Matrix         // drawRectFull's image-lane shading matrix (CTM∘paintMatrix)
}

var imageDrawScratchPool = sync.Pool{New: func() any { return &imageDrawScratch{} }}

func acquireImageDrawScratch() *imageDrawScratch {
	s := imageDrawScratchPool.Get().(*imageDrawScratch)
	if s.image == nil {
		s.image = &imagecore.Image{}
	}
	return s
}

func releaseImageDrawScratch(s *imageDrawScratch) {
	// Drop the pixel/image references so the pool does not pin the source pixels while idle.
	s.image.ClearScratch()
	s.shader = shaders.ImageShader{}
	imageDrawScratchPool.Put(s)
}

// makePaintWithImage replaces the paint's shader with a clamp/clamp image shader over the pixels (never copied),
// composing alpha-only images with the paint's shader through a DstIn blend. The base Image and ImageShader are reused
// from the scratch (no per-draw allocation); only the rare local-matrix wrap (non-nil matrix) and alpha-only blend wrap
// allocate.
func (scratch *imageDrawScratch) makePaintWithImage(origPaint *Paint, px *imagecore.Pixels, sampling shaders.SamplingOptions, matrix *geom.Matrix) Paint {
	scratch.image.SetFromPixels(px)
	var s shaders.Shader
	if shaders.SetImage(&scratch.shader, scratch.image, shaders.TileClamp, shaders.TileClamp, sampling) {
		s = &scratch.shader
		if matrix != nil && !matrix.IsIdentity() {
			s = shaders.NewWithLocalMatrix(s, *matrix)
		}
	}
	if px.Info.ColorType == imagecore.ColorTypeAlpha8 && origPaint.Shader != nil && s != nil {
		// Alpha images+shaders output the texture's alpha multiplied by the shader's color: DstIn with the source image
		// and dst shader (MakeBlend takes dst first, src second).
		s = shaders.NewBlend(raster.BlendDstIn, origPaint.Shader, s)
	}
	paint := *origPaint
	paint.Shader = s
	return paint
}

// clippedOutMatrix maps the bitmap bounds through the matrix and quick-rejects against the clip.
func clippedOutMatrix(matrix *geom.Matrix, rc *raster.Clip, w, h int32) bool {
	dstR, _ := path.MapRect(matrix, geom.RectWH(float32(w), float32(h)))
	return rc.QuickReject(dstR.RoundOut())
}

// drawBitmap draws the pixels: the sprite fast path for integer-translate draws, else the clamp/clamp image shader over
// a rect.
func (d *draw) drawBitmap(px *imagecore.Pixels, prematrix *geom.Matrix, dstBounds *geom.Rect, sampling shaders.SamplingOptions, origPaint *Paint) {
	if d.rc.IsEmpty() || px.Info.Width == 0 || px.Info.Height == 0 ||
		px.Info.ColorType == imagecore.ColorTypeUnknown {
		return
	}

	paint := origPaint
	if paint.Style != StyleFill {
		p := *paint
		p.Style = StyleFill
		paint = &p
	}

	// The matrix, the wrapping image/shader, and drawRectFull's combined shading matrix are homed on the pooled scratch
	// so their addresses (which flow into retaining callees) do not heap-allocate per draw; the scratch is fully
	// consumed by the synchronous draw below before it is recycled.
	scratch := acquireImageDrawScratch()
	defer releaseImageDrawScratch(scratch)

	matrix := &scratch.matrix
	matrix.SetConcat(d.ctm, prematrix)

	if clippedOutMatrix(matrix, d.rc, px.Info.Width, px.Info.Height) {
		return
	}

	if px.Info.ColorType != imagecore.ColorTypeAlpha8 &&
		treatAsSprite(matrix, px.Info.Width, px.Info.Height, sampling, paint.AntiAlias) {
		if pm, ok := px.PixmapView(); ok {
			ix := scalarRoundToInt(matrix.Get(geom.MTransX))
			iy := scalarRoundToInt(matrix.Get(geom.MTransY))
			if clipHandlesSprite(d.rc, ix, iy, &pm) {
				// The sprite lane cannot apply mask or color filters.
				if paint.MaskFilter == nil && paint.ColorFilter == nil {
					blitter := raster.ChooseSprite(d.dst, &pm, ix, iy, paint.Color.A(),
						paint.BlendMode, px.Info.IsOpaque())
					raster.FillIRectRasterClip(geom.IRectXYWH(ix, iy, pm.Width, pm.Height), d.rc, blitter)
					return
				}
			}
		}
		// if the sprite blitter was not possible, fall through to the shader case
	}

	paintWithShader := scratch.makePaintWithImage(paint, px, sampling, nil)
	srcBounds := geom.RectWH(float32(px.Info.Width), float32(px.Info.Height))
	if dstBounds != nil {
		d.drawRectFull(srcBounds, &paintWithShader, prematrix, dstBounds, &scratch.combined)
	} else {
		dr := draw{dst: d.dst, ctm: matrix, rc: d.rc}
		dr.drawRect(srcBounds, &paintWithShader)
	}
}

// canApplyDstMatrixAsCTM reports whether the dst matrix can fold into the CTM: mask-filter sigmas depend on the CTM, so
// only translations qualify when a mask filter is present.
func canApplyDstMatrixAsCTM(m *geom.Matrix, paint *Paint) bool {
	if paint.MaskFilter == nil {
		return true
	}
	return m.Type() <= geom.TypeTranslate
}

// deviceDrawBitmap is the tiling wrapper over drawBitmap. The tiler bounds are the local-space dst rect when the caller
// has one; otherwise (only reachable with a huge device) the prematrix-mapped bitmap bounds grown by the paint.
func (d *BitmapDevice) deviceDrawBitmap(px *imagecore.Pixels, prematrix *geom.Matrix, dstOrNil *geom.Rect, sampling shaders.SamplingOptions, paint *Paint) {
	if !d.clipMayNeedTiling() {
		dr := d.draw()
		dr.drawBitmap(px, prematrix, dstOrNil, sampling, paint)
		return
	}
	bounds := dstOrNil
	var storage geom.Rect
	if bounds == nil && tilerNeedsTiling(d) {
		mapped, _ := path.MapRect(prematrix, geom.RectWH(float32(px.Info.Width), float32(px.Info.Height)))
		bounds = paintBounds(mapped, paint, &storage)
	}
	var tiler drawTiler
	tiler.init(d, bounds)
	for dr := tiler.next(); dr != nil; dr = tiler.next() {
		dr.drawBitmap(px, prematrix, dstOrNil, sampling, paint)
	}
}

// DrawImageRect implements Device. A raster device consumes CPU pixels, so a texture-backed drawable is read back
// through MakeNonTextureImage (identity for a raster image); the GPU-native lane lives in the GPU device.
func (d *BitmapDevice) DrawImageRect(img imagecore.DrawableImage, src *geom.Rect, dst geom.Rect, sampling shaders.SamplingOptions, paint *Paint, constraint SrcRectConstraint) {
	rasterImg := img.MakeNonTextureImage()
	if rasterImg == nil {
		return
	}
	px := rasterImg.PeekPixels(imagecore.CachingAllow)
	if px == nil {
		return
	}

	bitmapBounds := geom.RectWH(float32(px.Info.Width), float32(px.Info.Height))
	tmpSrc := bitmapBounds
	if src != nil {
		tmpSrc = *src
	}

	// Compute the matrix mapping the src rect onto the dst rect.
	matrix := rectToRectFill(tmpSrc, dst)

	dstPtr := &dst
	bitmapPtr := px

	// Clip tmpSrc to the bitmap bounds and recompute dst if src was clipped.
	srcIsSubset := false
	if src != nil {
		if !bitmapBounds.ContainsRect(*src) {
			if !tmpSrc.Intersect(bitmapBounds) {
				return // nothing to draw
			}
			// recompute dst, based on the smaller tmpSrc
			tmpDst, _ := path.MapRect(&matrix, tmpSrc)
			if !tmpDst.IsFinite() {
				return
			}
			dstPtr = &tmpDst
		}
		srcIsSubset = !tmpSrc.ContainsRect(bitmapBounds)
	}

	// When src is smaller than the bitmap and we are filtering, we don't know how much more of the bitmap we need:
	// use a shader with the dst bounds.
	useShader := srcIsSubset && constraint == ConstraintFast && sampling != (shaders.SamplingOptions{})

	if !useShader {
		if srcIsSubset {
			// We may need to clamp to the borders of the src rect within the bitmap: extract a subset.
			srcIR := tmpSrc.RoundOut()
			bmIR := geom.IRectWH(px.Info.Width, px.Info.Height)
			if !srcIR.Intersect(bmIR) {
				return
			}
			bitmapPtr = px.Subset(srcIR.Left, srcIR.Top, srcIR.Right, srcIR.Bottom)

			// Since we did an extract, adjust the matrix accordingly.
			var dx, dy float32
			if srcIR.Left > 0 {
				dx = float32(srcIR.Left)
			}
			if srcIR.Top > 0 {
				dy = float32(srcIR.Top)
			}
			if dx != 0 || dy != 0 {
				matrix.PreTranslate(dx, dy)
			}

			extracted := geom.RectWH(float32(bitmapPtr.Info.Width), float32(bitmapPtr.Info.Height))
			if extracted != tmpSrc {
				useShader = true // fractional src: cannot just drawBitmap
			}
		}
		if !useShader {
			// We can go faster by just calling drawBitmap, which concats the matrix with the CTM and tries drawSprite
			// if it can.
			if canApplyDstMatrixAsCTM(&matrix, paint) {
				d.deviceDrawBitmap(bitmapPtr, &matrix, dstPtr, sampling, paint)
				return
			}
		}
	}

	// Construct a shader and call the device DrawRect with the dst (which tiles past 8191px). The wrapping image/shader
	// are homed on the pooled scratch, fully consumed by the synchronous drawRect below before it is recycled.
	scratch := acquireImageDrawScratch()
	defer releaseImageDrawScratch(scratch)
	paintWithShader := scratch.makePaintWithImage(paint, bitmapPtr, sampling, &matrix)
	if paintWithShader.Shader == nil {
		return
	}
	paintWithShader.Style = StyleFill
	d.DrawRect(*dstPtr, &paintWithShader)
}

///////////////////////////////////////////////////////////////////////////////
// canvas entry points

// cleanPaintForDrawImage normalizes the paint for image drawing: fill style, no path effect.
func cleanPaintForDrawImage(paint *Paint) Paint {
	if paint == nil {
		return *NewPaint()
	}
	cleaned := *paint
	cleaned.Style = StyleFill
	cleaned.PathEffect = nil
	return cleaned
}

// cleanSamplingForConstraint drops sampling features the strict constraint cannot honor (mipmaps, anisotropy).
func cleanSamplingForConstraint(sampling shaders.SamplingOptions, constraint SrcRectConstraint) shaders.SamplingOptions {
	if constraint == ConstraintStrict {
		if sampling.Mipmap != shaders.MipmapNone {
			return shaders.SamplingOptions{Filter: sampling.Filter}
		}
		if sampling.MaxAniso != 0 {
			return shaders.SamplingOptions{Filter: shaders.FilterLinear}
		}
	}
	return sampling
}

// imageRectScratch homes Canvas.DrawImageRect's two per-draw temporaries — the src rect and the cleaned working paint —
// whose addresses cross the polymorphic Device.DrawImageRect interface method (&src into the device call, and &paint
// into aboutToDraw, which returns it as the working paint pointer that then also flows into the device call). Every
// Device implementation — BitmapDevice, the gpu/gl device, and the pdf device — consumes both synchronously,
// dereferencing *src and copying *paint into locals before returning, so it is safe to recycle the scratch once the
// device draw (and any image-filter-layer restore) returns. Homing them on a pooled (heap-resident) scratch keeps the
// address-taken values off the per-draw stack→heap path. Pooled (not a shared instance) so nested/concurrent draws stay
// independent.
type imageRectScratch struct {
	paint Paint
	src   geom.Rect
}

var imageRectScratchPool = sync.Pool{New: func() any { return &imageRectScratch{} }}

func acquireImageRectScratch() *imageRectScratch {
	return imageRectScratchPool.Get().(*imageRectScratch)
}

func releaseImageRectScratch(s *imageRectScratch) {
	// Drop the paint's shader/color-filter/mask-filter/image-filter references so an idle pooled scratch does not pin
	// them (src is a plain value with no references).
	s.paint = Paint{}
	imageRectScratchPool.Put(s)
}

// DrawImageRect draws the src rect of img mapped onto dst. img is the polymorphic drawable (a raster image or a
// texture-backed one); the pipeline reasons about it via the image-level queries and hands it to the device, which
// draws it natively (GPU) or rasterizes it (CPU/PDF).
func (c *Canvas) DrawImageRect(img imagecore.DrawableImage, src, dst geom.Rect, sampling shaders.SamplingOptions, paint *Paint, constraint SrcRectConstraint) {
	if img == nil || !src.IsFinite() || !dst.IsFinite() {
		return
	}
	dst = dst.Sorted()

	scratch := acquireImageRectScratch()
	defer releaseImageRectScratch(scratch)
	scratch.src = src
	scratch.paint = cleanPaintForDrawImage(paint)
	realSampling := cleanSamplingForConstraint(sampling, constraint)

	if c.internalQuickReject(dst, &scratch.paint) {
		return
	}

	if dp, restore, ok := c.aboutToDraw(&scratch.paint, &dst); ok {
		c.topDevice().DrawImageRect(img, &scratch.src, dst, realSampling, dp, constraint)
		if restore != nil {
			restore()
		}
	}
}

// DrawImage draws the whole image with its top-left corner at (x, y).
func (c *Canvas) DrawImage(img *imagecore.Image, x, y float32, sampling shaders.SamplingOptions, paint *Paint) {
	if img == nil {
		return
	}
	w := float32(img.Width())
	h := float32(img.Height())
	c.DrawImageRect(img, geom.RectWH(w, h), geom.RectLTRB(x, y, x+w, y+h), sampling, paint, ConstraintFast)
}

// cleanPaintForLattice normalizes the paint for lattice drawing: no mask filter, no AA.
func cleanPaintForLattice(paint *Paint) Paint {
	if paint == nil {
		return *NewPaint()
	}
	cleaned := *paint
	cleaned.MaskFilter = nil
	cleaned.AntiAlias = false
	return cleaned
}

// DrawImageNine draws the image as a nine-patch: the center rect divides the image into nine patches, each stretched
// onto the corresponding region of dst (a two-div lattice draw).
func (c *Canvas) DrawImageNine(img *imagecore.Image, center geom.IRect, dst geom.Rect, filter shaders.FilterMode, paint *Paint) {
	if img == nil || dst.IsEmpty() {
		return
	}

	lattice := latticeSpec{
		xDivs:  []int32{center.Left, center.Right},
		yDivs:  []int32{center.Top, center.Bottom},
		bounds: geom.IRectWH(img.Width(), img.Height()),
	}
	if !latticeValid(img.Width(), img.Height(), &lattice) {
		// The fallback is an ordinary whole-image draw, not a set of lattice patches, so it keeps the caller's paint:
		// cleanPaintForLattice's mask-filter drop and AA clear exist only to keep adjoining patches from seaming.
		c.DrawImageRect(img, geom.RectWH(float32(img.Width()), float32(img.Height())), dst,
			shaders.SamplingOptions{Filter: filter}, paint, ConstraintStrict)
		return
	}

	// The lattice draw proper.
	latticePaint := cleanPaintForLattice(paint)
	realPaint := cleanPaintForDrawImage(&latticePaint)
	if c.internalQuickReject(dst, &realPaint) {
		return
	}
	dp, restoreLayer, ok := c.aboutToDraw(&realPaint, &dst)
	if !ok {
		return
	}
	realPaint = *dp
	dev := c.topDevice()

	// Draw each lattice patch.
	iter := newLatticeIter(&lattice, dst)
	var srcR geom.IRect
	var dstR geom.Rect
	for iter.next(&srcR, &dstR) {
		// The fast path for single-pixel patches: read the pixel (BGRA unpremul, packed as an unpremultiplied ARGB
		// color) and fill dstR with it, modulating the paint alpha.
		if srcR.Width() <= 1 && srcR.Height() <= 1 {
			if col, ok2 := readPixelAsARGB(img, srcR.Left, srcR.Top); ok2 {
				if col != 0 || realPaint.BlendMode != raster.BlendSrcOver {
					paintCopy := realPaint
					alpha := alphaMul(col>>24, uint32(realPaint.Color.A())+1)
					paintCopy.Color = colorcore.Color(col).WithAlpha(uint8(alpha))
					dev.DrawRect(dstR, &paintCopy)
				}
				continue
			}
		}
		sr := srcR.ToRect()
		dev.DrawImageRect(img, &sr, dstR,
			shaders.SamplingOptions{Filter: filter}, &realPaint, ConstraintStrict)
	}
	if restoreLayer != nil {
		restoreLayer()
	}
}

// rectToRectFill returns the scale+translate matrix mapping src onto dst (identity when src is empty). The translate
// terms `dst.Left - src.Left * sx` are computed with FMAs — the single rounding matters: a 1-ulp difference moves the
// legacy sampler's 4-bit filter weights at texel-boundary pixels.
func rectToRectFill(src, dst geom.Rect) geom.Matrix {
	m := geom.IdentityMatrix()
	if src.IsEmpty() {
		return m
	}
	sx := dst.Width() / src.Width()
	sy := dst.Height() / src.Height()
	tx := float32(math.FMA(float64(-src.Left), float64(sx), float64(dst.Left)))
	ty := float32(math.FMA(float64(-src.Top), float64(sy), float64(dst.Top)))
	m.SetScaleTranslate(sx, sy, tx, ty)
	return m
}

// alphaMul scales value by an alpha in the 0..256 range.
func alphaMul(value, alpha256 uint32) uint32 {
	return value * alpha256 >> 8
}

// readPixelAsARGB reads one pixel as an unpremultiplied packed ARGB color (a BGRA_8888/unpremul 1x1 ReadPixels).
func readPixelAsARGB(img *imagecore.Image, x, y int32) (uint32, bool) {
	info, ok := imagecore.MakeInfo(1, 1, imagecore.ColorTypeBGRA8888, imagecore.AlphaTypeUnpremul)
	if !ok {
		return 0, false
	}
	var buf [4]byte
	if !img.ReadPixels(info, buf[:], 4, x, y, imagecore.CachingAllow) {
		return 0, false
	}
	// BGRA little-endian bytes = B,G,R,A; the packed color is A<<24|R<<16|G<<8|B.
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0]), true
}
