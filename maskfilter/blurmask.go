// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Blur math for A8 masks — the only format this package's mask pipeline produces: BoxBlur (the mask blur filter engine
// plus the four blur styles' post-passes) and the analytic rect blur (blurRect with its profile machinery), which the
// nine-patch fast paths consume.

package maskfilter

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// BlurStyle selects which parts of a blurred mask are fuzzy vs. solid/absent.
type BlurStyle uint8

// BlurStyle values.
const (
	BlurNormal BlurStyle = iota // fuzzy inside and outside
	BlurSolid                   // solid inside, fuzzy outside
	BlurOuter                   // nothing inside, fuzzy outside
	BlurInner                   // fuzzy inside, nothing outside
)

// mergeSrcWithBlur computes, for A8 masks: dst = blur * (src+1)/256.
func mergeSrcWithBlur(dst []uint8, dstRB int, src []uint8, srcRB int, blur []uint8, blurRB, sw, sh int) {
	di, si, bi := 0, 0, 0
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			dst[di+x] = uint8((uint32(blur[bi+x]) * (uint32(src[si+x]) + 1)) >> 8) // alpha multiply
		}
		di += dstRB
		si += srcRB
		bi += blurRB
	}
}

// clampSolidWithOrig computes, for A8 masks: dst = s + d - s*d/255.
func clampSolidWithOrig(dst []uint8, dstRB int, src []uint8, srcRB, sw, sh int) {
	di, si := 0, 0
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			s := dst[di+x]
			d := src[si+x]
			dst[di+x] = uint8(uint32(d) + uint32(s) - uint32(colorcore.MulDiv255Round(d, s)))
		}
		di += dstRB
		si += srcRB
	}
}

// clampOuterWithOrig computes, for A8 masks: dst *= (255 - src)/255 where src != 0.
func clampOuterWithOrig(dst []uint8, dstRB int, src []uint8, srcRB, sw, sh int) {
	di, si := 0, 0
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			if s := src[si+x]; s != 0 {
				dst[di+x] = uint8((uint32(dst[di+x]) * (uint32(255-s) + 1)) >> 8) // alpha multiply
			}
		}
		di += dstRB
		si += srcRB
	}
}

// boxBlur runs the discrete box blur over src for A8 sources, applying the style's post-pass.
func boxBlur(src *raster.Mask, sigma float32, style BlurStyle) (dst *raster.Mask, margin geom.IPoint, ok bool) {
	blurFilter := newMaskBlurFilter(float64(sigma), float64(sigma))
	if blurFilter.hasNoBlur() {
		// If there is no effective blur most styles will just produce the original mask. However, kOuter will produce
		// an empty mask.
		if style == BlurOuter {
			dst = &raster.Mask{Bounds: geom.IRect{}, RowBytes: 0}
			return dst, geom.IPoint{}, true
		}
		return nil, geom.IPoint{}, false
	}
	dst = &raster.Mask{}
	border := blurFilter.blur(src, dst)

	if src.Image != nil && dst.Image == nil {
		// The call to blur() failed to set our destination image up (e.g. an overflow).
		return nil, geom.IPoint{}, false
	}

	margin = border

	if src.Image == nil {
		if style == BlurInner {
			dst.Bounds = src.Bounds // restore trimmed bounds
			dst.RowBytes = dst.Bounds.Width()
		}
		return dst, margin, true
	}

	sw := int(src.Bounds.Width())
	sh := int(src.Bounds.Height())
	switch style {
	case BlurNormal:
	case BlurSolid:
		start := int(border.X) + int(border.Y)*int(dst.RowBytes)
		clampSolidWithOrig(dst.Image[start:], int(dst.RowBytes), src.Image, int(src.RowBytes), sw, sh)
	case BlurOuter:
		start := int(border.X) + int(border.Y)*int(dst.RowBytes)
		clampOuterWithOrig(dst.Image[start:], int(dst.RowBytes), src.Image, int(src.RowBytes), sw, sh)
	case BlurInner:
		// now we allocate the "real" dst, mirror the size of src
		blur := *dst
		*dst = raster.Mask{Bounds: src.Bounds, RowBytes: src.Bounds.Width()}
		size := computeImageSize(dst.Bounds, dst.RowBytes)
		if size == 0 {
			return nil, geom.IPoint{}, false // too big to allocate, abort
		}
		dst.Image = make([]uint8, size)
		blurStart := int(border.X) + int(border.Y)*int(blur.RowBytes)
		mergeSrcWithBlur(dst.Image, int(dst.RowBytes), src.Image, int(src.RowBytes),
			blur.Image[blurStart:], int(blur.RowBytes), sw, sh)
	}
	return dst, margin, true
}

///////////////////////////////////////////////////////////////////////////////
// the analytic rect blur

// gaussianIntegral computes the indefinite integral of a box convolved with itself three times, evaluated at x (in
// units of window halves), pre-flipped so 1 means fully outside.
func gaussianIntegral(x float32) float32 {
	if x > 1.5 {
		return 0.0
	}
	if x < -1.5 {
		return 1.0
	}
	x2 := x * x
	x3 := x2 * x
	if x > 0.5 {
		return 0.5625 - (x3/6.0 - 3.0*x2*0.25 + 1.125*x)
	}
	if x > -0.5 {
		return 0.5 - (0.75*x - x3/3.0)
	}
	return 0.4375 + (-x3/6.0 - 3.0*x2*0.25 - 1.125*x)
}

// computeBlurProfile fills a pre-inverted 255-scale profile of a blurred half-plane. size must be ceil(6*sigma).
func computeBlurProfile(profile []uint8, size int, sigma float32) {
	center := size >> 1
	invr := 1.0 / (2 * sigma)
	profile[0] = 255
	for x := 1; x < size; x++ {
		scaledX := (float32(center) - float32(x) - 0.5) * invr
		gi := gaussianIntegral(scaledX)
		profile[x] = 255 - uint8(255.0*gi)
	}
}

// profileLookup returns the blur profile's coverage value for a scanline position.
func profileLookup(profile []uint8, loc, blurredWidth, sharpWidth int) uint8 {
	// how far are we from the original edge?
	dx := abs(((loc<<1)+1)-blurredWidth) - sharpWidth
	ox := dx >> 1
	if ox < 0 {
		ox = 0
	}
	return profile[ox]
}

// computeBlurredScanline fills pixels with the blurred coverage profile for one scanline of width.
func computeBlurredScanline(pixels, profile []uint8, width int, sigma float32) {
	profileSize := int(geom.CeilToInt(6 * sigma))
	sw := width - profileSize
	// nearest odd number less than the profile size represents the center of the (2x scaled) profile
	center := (profileSize &^ 1) - 1
	w := sw - center
	for x := 0; x < width; x++ {
		if profileSize <= sw {
			pixels[x] = profileLookup(profile, x, width, w)
		} else {
			span := float32(sw) / (2 * sigma)
			giX := 1.5 - (float32(x)+0.5)/(2*sigma)
			pixels[x] = uint8(255 * (gaussianIntegral(giX) - gaussianIntegral(giX+span)))
		}
	}
}

// maskCreateMode selects how much work blurRect does: just report the resulting bounds, or also render the blurred
// image.
type maskCreateMode uint8

const (
	justComputeBounds maskCreateMode = iota
	computeBoundsAndRenderImage
)

// blurRect computes the analytic blur of an axis-aligned rectangle, used by the nine-patch rect fast path. scratch
// supplies the pooled profile/scanline/output buffers (progress §14.1); the returned dst.Image aliases scratch.image
// and is valid until the scratch is recycled (which the FilterRects caller does after drawNine). blurRect fills the
// caller-provided dst; dst is a reused header on the pooled blurScratch, so the analytic rect blur allocates no mask
// header per draw. Every field is (re)set here, so a recycled header's stale contents are never observed.
func blurRect(sigma float32, src geom.Rect, style BlurStyle, createMode maskCreateMode, scratch *blurScratch, dst *raster.Mask) (margin geom.IPoint, ok bool) {
	profileSize := int(geom.CeilToInt(6 * sigma))
	if profileSize <= 0 {
		return geom.IPoint{}, false // no blur to compute
	}
	pad := profileSize / 2
	margin = geom.IPoint{X: int32(pad), Y: int32(pad)}

	*dst = raster.Mask{
		Bounds: geom.IRect{
			Left:   geom.RoundToInt(src.Left - float32(pad)),
			Top:    geom.RoundToInt(src.Top - float32(pad)),
			Right:  geom.RoundToInt(src.Right + float32(pad)),
			Bottom: geom.RoundToInt(src.Bottom + float32(pad)),
		},
	}
	dst.RowBytes = dst.Bounds.Width()

	sw := int(geom.FloorToInt(src.Width()))
	sh := int(geom.FloorToInt(src.Height()))

	if createMode == justComputeBounds {
		if style == BlurInner {
			dst.Bounds = src.Round() // restore trimmed bounds
			dst.RowBytes = int32(sw)
		}
		return margin, true
	}

	scratch.profile = growScratch(scratch.profile, profileSize)
	profile := scratch.profile
	computeBlurProfile(profile, profileSize, sigma)

	size := computeImageSize(dst.Bounds, dst.RowBytes)
	if size == 0 {
		return geom.IPoint{}, false // too big to allocate, abort
	}
	scratch.image = growScratch(scratch.image, size)
	dp := scratch.image
	dst.Image = dp

	dstHeight := int(dst.Bounds.Height())
	dstWidth := int(dst.Bounds.Width())

	scratch.hScan = growScratch(scratch.hScan, dstWidth)
	horizontalScanline := scratch.hScan
	scratch.vScan = growScratch(scratch.vScan, dstHeight)
	verticalScanline := scratch.vScan
	computeBlurredScanline(horizontalScanline, profile, dstWidth, sigma)
	computeBlurredScanline(verticalScanline, profile, dstHeight, sigma)

	out := 0
	for y := 0; y < dstHeight; y++ {
		for x := 0; x < dstWidth; x++ {
			dp[out] = colorcore.MulDiv255Round(horizontalScanline[x], verticalScanline[y])
			out++
		}
	}

	switch style {
	case BlurInner:
		// now we allocate the "real" dst, mirror the size of src
		srcSize := int(src.Width() * src.Height())
		if srcSize == 0 {
			return geom.IPoint{}, false // too big to allocate, abort
		}
		inner := make([]uint8, srcSize)
		for y := 0; y < sh; y++ {
			copy(inner[y*sw:y*sw+sw], dp[(y+pad)*dstWidth+pad:])
		}
		dst.Image = inner
		dst.Bounds = src.Round() // restore trimmed bounds
		dst.RowBytes = int32(sw)
	case BlurOuter:
		for y := pad; y < dstHeight-pad; y++ {
			row := dp[y*dstWidth+pad : y*dstWidth+pad+sw]
			for i := range row {
				row[i] = 0
			}
		}
	case BlurSolid:
		for y := pad; y < dstHeight-pad; y++ {
			row := dp[y*dstWidth+pad : y*dstWidth+pad+sw]
			for i := range row {
				row[i] = 0xFF
			}
		}
	default:
		// normal and solid styles are the same for analytic rect blurs
	}
	return margin, true
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
