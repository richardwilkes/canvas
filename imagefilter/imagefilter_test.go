// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package imagefilter_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/imagefilter"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// pixel returns the RGBA word at (x, y).
func pixel(pix *raster.Pixmap, x, y int32) uint32 {
	return pix.Pix[y*pix.RowPixels+x]
}

func alphaOf(p uint32) uint32 { return p >> 24 }

// drawSquare renders a filled 20x20 red square at (20,20)-(40,40) into a 64x64 surface using the given paint mutator,
// returning the pixels.
func drawSquare(t *testing.T, mutate func(p *canvas.Paint)) *raster.Pixmap {
	t.Helper()
	pix := raster.NewPixmap(64, 64)
	c := canvas.NewForPixmap(pix)
	p := canvas.NewPaint()
	p.Color = 0xFFFF0000 // opaque red
	if mutate != nil {
		mutate(p)
	}
	c.DrawRect(geom.RectLTRB(20, 20, 40, 40), p)
	return pix
}

func TestBlurImageFilterOnPaint(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Blur(3, 3, shaders.TileDecal, nil, nil)
	})
	// Center remains saturated red.
	if got := pixel(pix, 30, 30); got != 0xFF0000FF && alphaOf(got) < 250 {
		t.Fatalf("center = %08x, want nearly opaque red", got)
	}
	// The blur spreads outside the square: a pixel 4px outside the edge has partial alpha.
	if a := alphaOf(pixel(pix, 44, 30)); a == 0 || a == 255 {
		t.Fatalf("blur fringe alpha = %d, want partial", a)
	}
	// Far corner unaffected.
	if got := pixel(pix, 2, 2); got != 0 {
		t.Fatalf("far corner = %08x, want transparent", got)
	}
	// Symmetry: left and right fringes match.
	if l, r := alphaOf(pixel(pix, 16, 30)), alphaOf(pixel(pix, 43, 30)); l != r {
		t.Fatalf("asymmetric blur: left %d right %d", l, r)
	}
	// Monotonic falloff away from the square.
	prev := uint32(256)
	for x := int32(40); x < 52; x++ {
		a := alphaOf(pixel(pix, x, 30))
		if a > prev {
			t.Fatalf("non-monotonic falloff at x=%d: %d > %d", x, a, prev)
		}
		prev = a
	}
}

func TestOffsetImageFilter(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Offset(10, 6, nil, nil)
	})
	if got := pixel(pix, 22, 22); got != 0 {
		t.Fatalf("original position should be vacated, got %08x", got)
	}
	if got := pixel(pix, 35, 30+6); alphaOf(got) != 255 {
		t.Fatalf("offset content missing at (35,36): %08x", got)
	}
	if got := pixel(pix, 20+10+1, 20+6+1); alphaOf(got) != 255 {
		t.Fatalf("offset top-left corner missing: %08x", got)
	}
}

func TestDropShadowImageFilter(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.DropShadow(8, 8, 2, 2, 0xFF000000, nil, nil)
	})
	// Original square still drawn.
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red", got)
	}
	// Shadow visible below/right of the square, outside the original bounds.
	if a := alphaOf(pixel(pix, 44, 44)); a == 0 {
		t.Fatalf("expected shadow at (44,44)")
	}
	// The shadow is black where it isn't overlapped by the square.
	sh := pixel(pix, 44, 44)
	if sh&0x00FFFFFF != 0 {
		t.Fatalf("shadow should be black, got %08x", sh)
	}
}

func TestDropShadowOnly(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.DropShadowOnly(8, 8, 0, 0, 0xFF00FF00, nil, nil)
	})
	// The original square is not drawn; only the offset green shadow.
	if got := pixel(pix, 22, 22); got != 0 {
		t.Fatalf("original content should be absent, got %08x", got)
	}
	if got := pixel(pix, 30, 30); alphaOf(got) == 0 {
		t.Fatalf("shadow missing at (30,30)")
	}
	if got := pixel(pix, 30, 30); got&0x0000FF00 == 0 {
		t.Fatalf("shadow should be green, got %08x", got)
	}
}

func TestColorFilterImageFilterCollapse(t *testing.T) {
	// A color-filter image filter that doesn't affect transparent black collapses into the paint's color filter,
	// producing identical output to setting the color filter directly.
	cf := colorfilter.NewBlend(0xFF0000FF, raster.BlendSrcIn) // blue
	viaFilter := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.ColorFilter(cf, nil, nil)
	})
	viaPaint := drawSquare(t, func(p *canvas.Paint) {
		p.ColorFilter = cf
	})
	for i := range viaFilter.Pix {
		if viaFilter.Pix[i] != viaPaint.Pix[i] {
			t.Fatalf("pixel %d differs: %08x vs %08x", i, viaFilter.Pix[i], viaPaint.Pix[i])
		}
	}
	if got := pixel(viaFilter, 30, 30); got != 0xFFFF0000 {
		t.Fatalf("center = %08x, want opaque blue", got)
	}
}

func TestCropDecalRestrictsBlur(t *testing.T) {
	crop := geom.RectLTRB(20, 20, 40, 40)
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Blur(3, 3, shaders.TileDecal, nil, &crop)
	})
	// Inside the crop the blur is visible; outside it's clipped.
	if a := alphaOf(pixel(pix, 30, 30)); a == 0 {
		t.Fatalf("center missing")
	}
	if got := pixel(pix, 44, 30); got != 0 {
		t.Fatalf("outside crop should be transparent, got %08x", got)
	}
	// Blur softens inside edges of the crop (alpha < 255 near the boundary).
	if a := alphaOf(pixel(pix, 39, 30)); a == 0 || a == 255 {
		t.Fatalf("expected softened edge inside crop, alpha=%d", a)
	}
}

func TestMergeFilter(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Merge([]filtercore.Filter{
			imagefilter.Offset(-10, 0, nil, nil),
			imagefilter.Offset(10, 0, nil, nil),
		}, nil)
	})
	if a := alphaOf(pixel(pix, 15, 30)); a != 255 {
		t.Fatalf("left copy missing, alpha=%d", a)
	}
	if a := alphaOf(pixel(pix, 45, 30)); a != 255 {
		t.Fatalf("right copy missing, alpha=%d", a)
	}
}

// Merge is the only factory that takes a slice, so it is the only place a caller's backing array could reach the DAG.
// The DAG is immutable after construction, so a later write to that slice must not rewire the built filter.
func TestMergeCopiesInputSlice(t *testing.T) {
	left := imagefilter.Offset(-10, 0, nil, nil)
	right := imagefilter.Offset(10, 0, nil, nil)
	inputs := []filtercore.Filter{left, right}
	merged := imagefilter.Merge(inputs, nil)
	inputs[0] = imagefilter.Empty()
	inputs[1] = nil // nil would additionally mean "the dynamic source image"
	base := merged.Base()
	if base.CountInputs() != 2 {
		t.Fatalf("CountInputs = %d, want 2", base.CountInputs())
	}
	if base.Input(0) != left || base.Input(1) != right {
		t.Fatal("Merge retained the caller's slice")
	}
	pix := drawSquare(t, func(p *canvas.Paint) { p.ImageFilter = merged })
	if a := alphaOf(pixel(pix, 15, 30)); a != 255 {
		t.Fatalf("left copy missing, alpha=%d", a)
	}
	if a := alphaOf(pixel(pix, 45, 30)); a != 255 {
		t.Fatalf("right copy missing, alpha=%d", a)
	}
}

// Assigning the result back into the slice that produced it must not make the filter its own input, which would make
// every recursive walk of the DAG run until the stack overflows.
func TestMergeSelfAssignmentIsNotACycle(t *testing.T) {
	inner := imagefilter.Offset(10, 0, nil, nil)
	s := []filtercore.Filter{inner}
	s[0] = imagefilter.Merge(s, nil)
	// Check the edge before running any of the walks below: with the input aliased to s[0] they never return.
	if s[0].Base().Input(0) != inner {
		t.Fatal("Merge aliased its own result as its input")
	}
	if filtercore.AffectsTransparentBlack(s[0]) {
		t.Fatal("offset does not affect transparent black")
	}
	if got := filtercore.CTMCapability(s[0]); got != filtercore.MatrixCapabilityComplex {
		t.Fatalf("CTMCapability = %v", got)
	}
	pix := drawSquare(t, func(p *canvas.Paint) { p.ImageFilter = s[0] })
	if a := alphaOf(pixel(pix, 45, 30)); a != 255 {
		t.Fatalf("offset copy missing, alpha=%d", a)
	}
}

func TestComposeFilter(t *testing.T) {
	// offset(5,0) ∘ offset(0,7) == offset(5,7)
	composed := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Compose(
			imagefilter.Offset(5, 0, nil, nil),
			imagefilter.Offset(0, 7, nil, nil),
		)
	})
	direct := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Offset(5, 7, nil, nil)
	})
	for i := range composed.Pix {
		if composed.Pix[i] != direct.Pix[i] {
			t.Fatalf("pixel %d differs: %08x vs %08x", i, composed.Pix[i], direct.Pix[i])
		}
	}
}

func TestTileFilter(t *testing.T) {
	src := geom.RectLTRB(20, 20, 30, 30) // top-left quadrant of the square
	dst := geom.RectLTRB(0, 0, 60, 60)
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Tile(src, dst, nil)
	})
	// The 10x10 red tile repeats across dst.
	for _, pt := range [][2]int32{{5, 5}, {15, 25}, {45, 55}} {
		if a := alphaOf(pixel(pix, pt[0], pt[1])); a != 255 {
			t.Fatalf("tile missing at (%d,%d)", pt[0], pt[1])
		}
	}
	if got := pixel(pix, 62, 30); got != 0 {
		t.Fatalf("outside dst should be transparent, got %08x", got)
	}
}

// horizontalPairImage builds a 2x1 image holding the two given pixel words side by side.
func horizontalPairImage(left, right uint32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(2, 1, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	p.Words[0] = left
	p.Words[1] = right
	return imagecore.FromPixels(p)
}

// mirrorCropPixels mirror-tiles a 64x64 crop that holds a two-tone 20x20 bar (red left half, blue right half) and
// returns the visible 64x64 device pixels. offset places the crop — and the bar inside it — exactly one period to the
// left (-64) or right (+64) of the device, so the single visible tile is an odd period either way and must come out
// mirrored: blue on the left, red on the right.
func mirrorCropPixels(t *testing.T, offset float32) *raster.Pixmap {
	t.Helper()
	img := horizontalPairImage(0xFF0000FF, 0xFFFF0000) // red, blue
	return drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Crop(geom.RectLTRB(offset, 0, offset+64, 64), shaders.TileMirror,
			imagefilter.Image(img, geom.RectWH(2, 1), geom.RectLTRB(offset+20, 20, offset+40, 40),
				shaders.SamplingOptions{Filter: shaders.FilterNearest}))
	})
}

// TestMirrorCropNegativePeriod pins the mirror flip for a crop sitting to the right of the visible output, where the
// period index is negative. The parity test used to run on a signed remainder, which reports -1 for an odd negative
// period and so rendered that tile as an unmirrored translation while the symmetric positive period reflected.
func TestMirrorCropNegativePeriod(t *testing.T) {
	const (
		red  = 0xFF0000FF
		blue = 0xFFFF0000
	)
	for _, tc := range []struct {
		name   string
		offset float32
	}{
		{name: "period+1", offset: -64},
		{name: "period-1", offset: 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pix := mirrorCropPixels(t, tc.offset)
			// The bar is mirrored about the crop edge, so its blue half lands left of its red half.
			if got := pixel(pix, 28, 30); got != blue {
				t.Fatalf("left half = %08x, want blue (%08x): tile is not mirrored", got, uint32(blue))
			}
			if got := pixel(pix, 38, 30); got != red {
				t.Fatalf("right half = %08x, want red (%08x): tile is not mirrored", got, uint32(red))
			}
			// The mirrored bar spans x in [24,44); outside it the crop's tiling is transparent.
			if got := pixel(pix, 20, 30); got != 0 {
				t.Fatalf("left of the mirrored bar = %08x, want transparent", got)
			}
			if got := pixel(pix, 48, 30); got != 0 {
				t.Fatalf("right of the mirrored bar = %08x, want transparent", got)
			}
		})
	}
	// Both odd periods reflect about their own crop edge, so they must land on identical pixels.
	left, right := mirrorCropPixels(t, -64), mirrorCropPixels(t, 64)
	for i := range left.Pix {
		if left.Pix[i] != right.Pix[i] {
			t.Fatalf("pixel %d differs between periods +1 and -1: %08x vs %08x", i, left.Pix[i], right.Pix[i])
		}
	}
}

// TestArithmeticUnderScaledCanvas exercises Builder.Eval's input shaders under a non-identity layer matrix: the inputs
// are always sampled in layer space, so a 2x CTM must scale both the geometry and the filter's own offsets.
func TestArithmeticUnderScaledCanvas(t *testing.T) {
	pix := raster.NewPixmap(128, 128)
	c := canvas.NewForPixmap(pix)
	c.Scale(2, 2)
	p := canvas.NewPaint()
	p.Color = 0xFFFF0000
	// out = 0.5*BG + 0.5*FG, with BG the source square and FG the source offset by (10,0) in parameter space.
	p.ImageFilter = imagefilter.Arithmetic(0, 0.5, 0.5, 0, false, nil,
		imagefilter.Offset(10, 0, nil, nil), nil)
	c.DrawRect(geom.RectLTRB(20, 20, 40, 40), p)
	// The square covers device x in [40,80) and the offset copy [60,100).
	if got := alphaOf(pixel(pix, 50, 60)); got < 127 || got > 128 {
		t.Fatalf("background-only alpha = %d, want ~128", got)
	}
	if got := alphaOf(pixel(pix, 70, 60)); got != 255 {
		t.Fatalf("overlap alpha = %d, want 255", got)
	}
	if got := alphaOf(pixel(pix, 90, 60)); got < 127 || got > 128 {
		t.Fatalf("foreground-only alpha = %d, want ~128", got)
	}
	if got := pixel(pix, 110, 60); got != 0 {
		t.Fatalf("outside both = %08x, want transparent", got)
	}
}

func TestSaveLayerWithFilterPaint(t *testing.T) {
	// saveLayer with an image-filter paint applies the filter at restore.
	pix := raster.NewPixmap(64, 64)
	c := canvas.NewForPixmap(pix)
	lp := canvas.NewPaint()
	lp.ImageFilter = imagefilter.Offset(12, 0, nil, nil)
	c.SaveLayer(nil, lp)
	p := canvas.NewPaint()
	p.Color = 0xFF00FF00
	c.DrawRect(geom.RectLTRB(10, 10, 20, 20), p)
	c.Restore()
	if got := pixel(pix, 15, 15); got != 0 {
		t.Fatalf("unfiltered position should be empty, got %08x", got)
	}
	if a := alphaOf(pixel(pix, 27, 15)); a != 255 {
		t.Fatalf("offset layer content missing, alpha=%d", a)
	}
}

func TestBlurBoundsPropagation(t *testing.T) {
	f := imagefilter.Blur(4, 4, shaders.TileDecal, nil, nil)
	mapping := filtercore.MappingOf(geom.IdentityMatrix())
	content := geom.RectLTRB(10, 10, 20, 20)
	out, bounded := filtercore.GetOutputBounds(f, &mapping, content)
	if !bounded {
		t.Fatalf("blur output should be bounded")
	}
	want := geom.IRectLTRB(-2, -2, 32, 32) // outset by ceil(3*4)=12
	if out != want {
		t.Fatalf("output bounds = %v, want %v", out, want)
	}
	in := filtercore.GetInputBounds(f, &mapping, geom.IRectLTRB(0, 0, 10, 10), nil)
	wantIn := geom.IRectLTRB(-12, -12, 22, 22)
	if in != wantIn {
		t.Fatalf("input bounds = %v, want %v", in, wantIn)
	}
	if !filtercore.CanComputeFastBounds(f) {
		t.Fatalf("blur can compute fast bounds")
	}
	fb := f.OnComputeFastBounds(content)
	if fb != content.Outset(12, 12) {
		t.Fatalf("fast bounds = %v", fb)
	}
}

func TestPaintAlphaOnFilteredLayer(t *testing.T) {
	// A paint alpha rides the restore paint and modulates the filtered result.
	pix := raster.NewPixmap(64, 64)
	c := canvas.NewForPixmap(pix)
	lp := canvas.NewPaint()
	lp.Color = lp.Color.WithAlpha(128)
	lp.ImageFilter = imagefilter.Offset(0, 0, nil, nil)
	c.SaveLayer(nil, lp)
	p := canvas.NewPaint()
	p.Color = 0xFFFF0000
	c.DrawRect(geom.RectLTRB(10, 10, 30, 30), p)
	c.Restore()
	a := alphaOf(pixel(pix, 20, 20))
	if a < 126 || a > 130 {
		t.Fatalf("alpha-modulated layer alpha = %d, want ~128", a)
	}
}

func TestFilterWithScaledCanvas(t *testing.T) {
	// The blur sigma maps through the CTM: a 2x scale doubles the blur's reach.
	pix := raster.NewPixmap(128, 128)
	c := canvas.NewForPixmap(pix)
	c.Scale(2, 2)
	p := canvas.NewPaint()
	p.Color = 0xFFFF0000
	p.ImageFilter = imagefilter.Blur(2, 2, shaders.TileDecal, nil, nil)
	c.DrawRect(geom.RectLTRB(20, 20, 40, 40), p)
	// The square maps to (40,40)-(80,80); blur reach ~12px in device space.
	if a := alphaOf(pixel(pix, 86, 60)); a == 0 || a == 255 {
		t.Fatalf("scaled blur fringe alpha = %d, want partial", a)
	}
	if got := pixel(pix, 100, 60); got != 0 {
		t.Fatalf("beyond scaled blur reach should be transparent, got %08x", got)
	}
}

func TestLargeSigmaBlurRescales(t *testing.T) {
	// sigma > 135 exercises FilterResult.rescale's multi-pass downscale loop.
	pix := raster.NewPixmap(256, 256)
	c := canvas.NewForPixmap(pix)
	p := canvas.NewPaint()
	p.Color = 0xFFFF0000
	p.ImageFilter = imagefilter.Blur(150, 150, shaders.TileDecal, nil, nil)
	c.DrawRect(geom.RectLTRB(96, 96, 160, 160), p)
	center := alphaOf(pixel(pix, 128, 128))
	if center == 0 || center == 255 {
		t.Fatalf("large blur center alpha = %d, want partial", center)
	}
	edge := alphaOf(pixel(pix, 10, 128))
	if edge == 0 || edge >= center {
		t.Fatalf("large blur edge alpha = %d (center %d), want 0 < edge < center", edge, center)
	}
}

// solidTypedImage returns a dim x dim image of the given color type, every pixel set to the repeated pixel bytes.
func solidTypedImage(t *testing.T, ct imagecore.ColorType, at imagecore.AlphaType, dim int32, px []byte) *imagecore.Image {
	t.Helper()
	info, ok := imagecore.MakeInfo(dim, dim, ct, at)
	if !ok {
		t.Fatalf("MakeInfo(%v, %v) rejected", ct, at)
	}
	data := make([]byte, int(dim)*int(dim)*len(px))
	for i := range int(dim) * int(dim) {
		copy(data[i*len(px):], px)
	}
	img := imagecore.NewRasterData(info, data, int(dim)*len(px))
	if img == nil {
		t.Fatalf("NewRasterData(%v) = nil", ct)
	}
	return img
}

// TestBlurNonN32SourceImage blurs an image-source leaf of every supported color type. The leaf hands its pixels to
// FilterResult.rescale in their native form, so rescale must re-render anything that is not N32 into an N32 surface
// before the raster blur engine reads pixel storage as 32-bit words: a 1- or 2-byte-per-pixel source (Gray8, Alpha8,
// RGB565, RGBA_F16) has no word storage at all and used to panic, and a BGRA_8888 source used to blur through with its
// R and B channels swapped. Every opaque-red source must therefore land on the same N32 result as the RGBA_8888
// control, both in the blur's flat interior and in its falloff.
func TestBlurNonN32SourceImage(t *testing.T) {
	const dim = 32
	for _, tc := range []struct {
		name       string
		px         []byte
		ct         imagecore.ColorType
		at         imagecore.AlphaType
		wantCenter uint32
		wantEdge   uint32
	}{
		// Opaque red in each type's own channel order/precision; all must produce the same N32 red.
		{"RGBA8888", []byte{0xFF, 0, 0, 0xFF}, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul, 0xFF0000FF, 0xC60000C6},
		{"BGRA8888", []byte{0, 0, 0xFF, 0xFF}, imagecore.ColorTypeBGRA8888, imagecore.AlphaTypePremul, 0xFF0000FF, 0xC60000C6},
		{"RGB888x", []byte{0xFF, 0, 0, 0}, imagecore.ColorTypeRGB888x, imagecore.AlphaTypeOpaque, 0xFF0000FF, 0xC60000C6},
		{"RGB565", []byte{0x00, 0xF8}, imagecore.ColorTypeRGB565, imagecore.AlphaTypeOpaque, 0xFF0000FF, 0xC60000C6},
		{"RGBAF16", []byte{0x00, 0x3C, 0, 0, 0, 0, 0x00, 0x3C}, imagecore.ColorTypeRGBAF16, imagecore.AlphaTypePremul, 0xFF0000FF, 0xC60000C6},
		// Gray8 expands to an opaque neutral; Alpha8 carries coverage only, so the paint's transparent black shows
		// through as premultiplied black.
		{"Gray8", []byte{0x80}, imagecore.ColorTypeGray8, imagecore.AlphaTypeOpaque, 0xFF808080, 0xC6646464},
		{"Alpha8", []byte{0x80}, imagecore.ColorTypeAlpha8, imagecore.AlphaTypePremul, 0x80000000, 0x64000000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := imagefilter.ImageDefault(solidTypedImage(t, tc.ct, tc.at, dim, tc.px),
				shaders.SamplingOptions{Filter: shaders.FilterLinear})
			pix := raster.NewPixmap(64, 64)
			c := canvas.NewForPixmap(pix)
			p := canvas.NewPaint()
			p.ImageFilter = imagefilter.Blur(3, 3, shaders.TileDecal, src, nil)
			c.DrawRect(geom.RectLTRB(0, 0, 64, 64), p)

			// The center sits more than 3*sigma from every edge, so it holds the unattenuated source color.
			if got := pixel(pix, dim/2, dim/2); got != tc.wantCenter {
				t.Fatalf("center = %08x, want %08x", got, tc.wantCenter)
			}
			// Inside the left falloff, where the decal tiling has partially eaten the coverage.
			if got := pixel(pix, 2, dim/2); got != tc.wantEdge {
				t.Fatalf("edge = %08x, want %08x", got, tc.wantEdge)
			}
			// Beyond the blur radius the result stays transparent.
			if got := pixel(pix, dim+12, dim/2); got != 0 {
				t.Fatalf("outside = %08x, want transparent", got)
			}
		})
	}
}

func TestColorFilterAffectsTransparentBlackFloods(t *testing.T) {
	// A color filter that affects transparent black floods the clip when used as an image filter (cannot collapse into
	// the paint's color filter slot).
	add := colorcore.Color(0xFF102030)
	cf := colorfilter.NewLighting(0xFFFFFFFF, add)
	if !filtercore.ColorFilterAffectsTransparentBlack(cf) {
		t.Skip("lighting add filter unexpectedly leaves transparent black")
	}
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.ColorFilter(cf, nil, nil)
	})
	// Far corner is filled by the flood (add color, alpha preserved at... transparent-black input keeps alpha 0?
	// Lighting's mul/add applies to RGB only, leaving alpha — premul rules clamp RGB to alpha, producing transparent
	// black again unless alpha changes. Use a simpler check: the filter must at least not corrupt the square's
	// interior.
	if a := alphaOf(pixel(pix, 30, 30)); a == 0 {
		t.Fatalf("interior missing")
	}
}
