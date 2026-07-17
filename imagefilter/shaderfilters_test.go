// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Integration tests for the part-2 shader-evaluating filters (morphology, displacement, lighting, magnifier, matrix
// convolution, arithmetic) and the image-source leaf, exercised end-to-end through canvas paint image filters.

package imagefilter_test

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/imagefilter"
	"github.com/richardwilkes/canvas/shaders"
)

func TestDilateImageFilter(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Dilate(4, 4, nil, nil)
	})
	// The 20..40 square dilates to 16..44.
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red", got)
	}
	if got := pixel(pix, 17, 30); got != 0xFF0000FF {
		t.Fatalf("dilated left edge = %08x, want opaque red", got)
	}
	if got := pixel(pix, 42, 42); got != 0xFF0000FF {
		t.Fatalf("dilated corner = %08x, want opaque red", got)
	}
	if got := pixel(pix, 14, 30); got != 0 {
		t.Fatalf("outside dilation = %08x, want transparent", got)
	}
}

func TestErodeImageFilter(t *testing.T) {
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Erode(4, 4, nil, nil)
	})
	// The 20..40 square erodes to 24..36.
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red", got)
	}
	if got := pixel(pix, 25, 25); got != 0xFF0000FF {
		t.Fatalf("inside eroded bounds = %08x, want opaque red", got)
	}
	if got := pixel(pix, 22, 30); got != 0 {
		t.Fatalf("eroded edge = %08x, want transparent", got)
	}
}

func TestDilateLargeRadiusUsesSparsePasses(t *testing.T) {
	// radius > 14 exercises the sparse kernel-doubling passes after the linear pass.
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Dilate(18, 18, nil, nil)
	})
	// The 20..40 square dilates to 2..58; every pixel of it must be fully red (morphology preserves hard edges).
	for _, pt := range [][2]int32{{3, 30}, {30, 3}, {57, 30}, {30, 57}, {30, 30}} {
		if got := pixel(pix, pt[0], pt[1]); got != 0xFF0000FF {
			t.Fatalf("(%d,%d) = %08x, want opaque red", pt[0], pt[1], got)
		}
	}
	if got := pixel(pix, 0, 30); got != 0 {
		t.Fatalf("outside dilation = %08x, want transparent", got)
	}
}

func TestMorphologyZeroRadiusIsIdentity(t *testing.T) {
	base := drawSquare(t, nil)
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Erode(0, 0, nil, nil)
	})
	for i := range pix.Pix {
		if pix.Pix[i] != base.Pix[i] {
			t.Fatalf("pixel %d differs: %08x vs %08x", i, pix.Pix[i], base.Pix[i])
		}
	}
	if imagefilter.Dilate(-1, 2, nil, nil) != nil {
		t.Fatal("negative radius must produce a nil filter")
	}
}

func TestDisplacementMapImageFilter(t *testing.T) {
	// Both inputs default to the source (the red square). Inside the square the unpremul color is (1,0,0,1): with R
	// selecting X and G selecting Y and scale 8, interior samples shift by (+4,-4); outside the square the transparent
	// displacement shifts by (-4,-4).
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.DisplacementMap(imagefilter.ChannelR, imagefilter.ChannelG, 8,
			nil, nil, nil)
	})
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red (sample shifted within square)", got)
	}
	// (42,42) is outside the square; its displaced sample (38,38) is inside.
	if got := pixel(pix, 42, 42); got != 0xFF0000FF {
		t.Fatalf("(42,42) = %08x, want opaque red via displacement", got)
	}
	// (18,30) samples (14,26): still transparent.
	if got := pixel(pix, 18, 30); got != 0 {
		t.Fatalf("(18,30) = %08x, want transparent", got)
	}
	// Invalid channel or non-finite scale produce nil filters.
	if imagefilter.DisplacementMap(imagefilter.ColorChannel(7), imagefilter.ChannelG, 1,
		nil, nil, nil) != nil {
		t.Fatal("invalid channel must produce a nil filter")
	}
}

func TestLightingDistantDiffuse(t *testing.T) {
	// A distant light shining straight down on a flat surface: interior normals are (0,0,1), the diffuse coefficient is
	// 1, and the (unbounded) lighting output floods the clip with the light's color at full alpha.
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.DistantLitDiffuse(geom.Point3{X: 0, Y: 0, Z: 1}, 0xFFFFFFFF,
			2, 1, nil, nil)
	})
	if got := pixel(pix, 30, 30); got != 0xFFFFFFFF {
		t.Fatalf("interior = %08x, want opaque white", got)
	}
	// The lighting equation is defined on the whole plane (flat transparent alpha → same normal).
	if got := pixel(pix, 2, 2); got != 0xFFFFFFFF {
		t.Fatalf("far corner = %08x, want opaque white flood", got)
	}
}

func TestLightingSpotSpecularGeometry(t *testing.T) {
	// A tight spotlight aimed at the square's center: pixels outside the cone receive no specular highlight
	// (transparent black since specular alpha = max component), pixels at the target do.
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.SpotLitSpecular(
			geom.Point3{X: 30, Y: 30, Z: 20}, // location above the center
			geom.Point3{X: 30, Y: 30, Z: 0},  // target
			1, 15, 0xFFFFFFFF, 1, 1, 4, nil, nil,
		)
	})
	center := alphaOf(pixel(pix, 30, 30))
	if center == 0 {
		t.Fatal("center of spotlight must be lit")
	}
	if got := alphaOf(pixel(pix, 60, 60)); got != 0 {
		t.Fatalf("far outside the cone = alpha %d, want 0", got)
	}
	// Validation: cutoff cosine outside [-1,1] comes from cos() so it's always valid, but a non-finite falloff must
	// fail.
	inf := float32(math.Inf(1))
	if imagefilter.SpotLitDiffuse(geom.Point3{}, geom.Point3{X: 1}, inf, 10, 0xFFFFFFFF, 1, 1,
		nil, nil) != nil {
		t.Fatal("non-finite falloff must produce a nil filter")
	}
	if imagefilter.DistantLitDiffuse(geom.Point3{Z: 1}, 0xFFFFFFFF, 1, -1, nil, nil) != nil {
		t.Fatal("negative kd must produce a nil filter")
	}
}

func TestMagnifierImageFilter(t *testing.T) {
	lens := geom.RectLTRB(16, 16, 48, 48)
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Magnifier(lens, 2, 8,
			shaders.SamplingOptions{Filter: shaders.FilterLinear}, nil, nil)
	})
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red", got)
	}
	// The zoomed square edge extends outward: (41,32) samples inside the square.
	if got := alphaOf(pixel(pix, 41, 32)); got != 255 {
		t.Fatalf("(41,32) alpha = %d, want 255 (zoomed square edge)", got)
	}
	// Outside the lens bounds the output is cropped away.
	if got := pixel(pix, 50, 32); got != 0 {
		t.Fatalf("outside lens = %08x, want transparent", got)
	}

	// Zoom <= 1 is a no-op (returns the input, here nil).
	if imagefilter.Magnifier(lens, 1, 8, shaders.SamplingOptions{}, nil, nil) != nil {
		t.Fatal("zoom <= 1 must collapse to the input")
	}
	// Invalid lens produces nil.
	if imagefilter.Magnifier(geom.Rect{}, 2, 8, shaders.SamplingOptions{}, nil, nil) != nil {
		t.Fatal("empty lens must produce a nil filter")
	}
}

func TestMagnifierZeroInsetTransformLane(t *testing.T) {
	// A zero inset reduces the magnifier to a rect->rect transform and crop.
	lens := geom.RectLTRB(16, 16, 48, 48)
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Magnifier(lens, 2, 0,
			shaders.SamplingOptions{Filter: shaders.FilterLinear}, nil, nil)
	})
	// The lens shows the 24..40 source region scaled 2x: (41,32) maps to source (36.75, 32.25), inside the square.
	if got := alphaOf(pixel(pix, 41, 32)); got != 255 {
		t.Fatalf("(41,32) alpha = %d, want 255", got)
	}
	if got := pixel(pix, 49, 32); got != 0 {
		t.Fatalf("outside lens = %08x, want transparent", got)
	}
}

func TestMatrixConvolutionBoxBlur(t *testing.T) {
	kernel := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.MatrixConvolution(geom.ISize{Width: 3, Height: 3}, kernel,
			1.0/9.0, 0, geom.IPoint{X: 1, Y: 1}, shaders.TileDecal, true, nil, nil)
	})
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red", got)
	}
	// One pixel outside the right edge averages 3 of 9 red taps.
	got := alphaOf(pixel(pix, 40, 30))
	if got < 80 || got > 90 {
		t.Fatalf("edge alpha = %d, want ~85 (3/9 coverage)", got)
	}
	if gotFar := pixel(pix, 44, 30); gotFar != 0 {
		t.Fatalf("outside kernel reach = %08x, want transparent", gotFar)
	}
}

func TestMatrixConvolutionQuantizedKernel(t *testing.T) {
	// 7x7 = 49 taps >= 28 exercises the unorm8-quantized kernel path; a box kernel quantizes losslessly, so the
	// interior stays exactly red.
	kernel := make([]float32, 49)
	for i := range kernel {
		kernel[i] = 1
	}
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.MatrixConvolution(geom.ISize{Width: 7, Height: 7}, kernel,
			1.0/49.0, 0, geom.IPoint{X: 3, Y: 3}, shaders.TileDecal, true, nil, nil)
	})
	if got := pixel(pix, 30, 30); got != 0xFF0000FF {
		t.Fatalf("center = %08x, want opaque red", got)
	}
}

func TestMatrixConvolutionValidation(t *testing.T) {
	k := []float32{1}
	if imagefilter.MatrixConvolution(geom.ISize{Width: 0, Height: 1}, k, 1, 0, geom.IPoint{},
		shaders.TileDecal, true, nil, nil) != nil {
		t.Fatal("zero-width kernel must produce a nil filter")
	}
	if imagefilter.MatrixConvolution(geom.ISize{Width: 1, Height: 1}, k, 1, 0,
		geom.IPoint{X: 1, Y: 0}, shaders.TileDecal, true, nil, nil) != nil {
		t.Fatal("out-of-range kernel offset must produce a nil filter")
	}
	if imagefilter.MatrixConvolution(geom.ISize{Width: 32, Height: 9}, make([]float32, 288), 1, 0,
		geom.IPoint{}, shaders.TileDecal, true, nil, nil) != nil {
		t.Fatal("over-large kernel must produce a nil filter")
	}
}

func TestArithmeticImageFilter(t *testing.T) {
	// out = 0.5*FG + 0.5*BG with FG = source offset by (10,0) and BG = source.
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Arithmetic(0, 0.5, 0.5, 0, false,
			nil, imagefilter.Offset(10, 0, nil, nil), nil)
	})
	if got := alphaOf(pixel(pix, 25, 30)); got < 127 || got > 128 {
		t.Fatalf("background-only alpha = %d, want ~128", got)
	}
	if got := alphaOf(pixel(pix, 35, 30)); got != 255 {
		t.Fatalf("overlap alpha = %d, want 255", got)
	}
	if got := alphaOf(pixel(pix, 45, 30)); got < 127 || got > 128 {
		t.Fatalf("foreground-only alpha = %d, want ~128", got)
	}
	if got := pixel(pix, 55, 30); got != 0 {
		t.Fatalf("outside both = %08x, want transparent", got)
	}
}

func TestArithmeticFloodsWithK4(t *testing.T) {
	// k4 != 0 affects transparent black, flooding the clip; with enforcePremul the RGB clamps to the alpha.
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Arithmetic(0, 1, 0, 0.5, true, nil, nil, nil)
	})
	if got := pixel(pix, 2, 2); got != 0x80808080 {
		t.Fatalf("flooded background = %08x, want 0x80808080", got)
	}
	if got := pixel(pix, 30, 30); got != 0xFF8080FF {
		t.Fatalf("center = %08x, want red + 0.5 flood", got)
	}
}

func TestArithmeticCollapses(t *testing.T) {
	// k = (0,1,0,0) collapses to Src (the foreground); compare against the direct offset filter.
	direct := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Offset(10, 6, nil, nil)
	})
	collapsed := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Arithmetic(0, 1, 0, 0, false,
			imagefilter.Blur(5, 5, shaders.TileDecal, nil, nil), // background is discarded
			imagefilter.Offset(10, 6, nil, nil), nil)
	})
	for i := range direct.Pix {
		if direct.Pix[i] != collapsed.Pix[i] {
			t.Fatalf("pixel %d differs: %08x vs %08x", i, collapsed.Pix[i], direct.Pix[i])
		}
	}
	inf := float32(math.Inf(1))
	if imagefilter.Arithmetic(inf, 0, 0, 0, false, nil, nil, nil) != nil {
		t.Fatal("non-finite coefficients must produce a nil filter")
	}
}

// solidImage builds a w x h image filled with the given RGBA word.
func solidImage(w, h int32, rgba uint32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for i := range p.Words {
		p.Words[i] = rgba
	}
	return imagecore.FromPixels(p)
}

func TestImageSourceFilter(t *testing.T) {
	img := solidImage(4, 4, 0xFF00FF00) // opaque green
	pix := drawSquare(t, func(p *canvas.Paint) {
		// The DAG never references the source, so the square disappears and the image lands at dst.
		p.ImageFilter = imagefilter.Image(img, geom.RectWH(4, 4), geom.RectLTRB(10, 10, 20, 20),
			shaders.SamplingOptions{Filter: shaders.FilterLinear})
	})
	if got := pixel(pix, 15, 15); got != 0xFF00FF00 {
		t.Fatalf("inside dst = %08x, want opaque green", got)
	}
	if got := pixel(pix, 30, 30); got != 0 {
		t.Fatalf("source content must be discarded, got %08x", got)
	}
	if got := pixel(pix, 8, 15); got != 0 {
		t.Fatalf("outside dst = %08x, want transparent", got)
	}
}

func TestImageSourceFractionalSrc(t *testing.T) {
	img := solidImage(4, 4, 0xFF00FF00)
	pix := drawSquare(t, func(p *canvas.Paint) {
		// A fractional src rect takes the strict drawImageRect render lane.
		p.ImageFilter = imagefilter.Image(img, geom.RectLTRB(0.5, 0.5, 3.5, 3.5),
			geom.RectLTRB(24, 24, 40, 40), shaders.SamplingOptions{Filter: shaders.FilterLinear})
	})
	if got := pixel(pix, 32, 32); got != 0xFF00FF00 {
		t.Fatalf("inside dst = %08x, want opaque green", got)
	}
	if got := pixel(pix, 20, 20); got != 0 {
		t.Fatalf("outside dst = %08x, want transparent", got)
	}
}

func TestImageSourceEmptyInputs(t *testing.T) {
	img := solidImage(4, 4, 0xFF00FF00)
	// Empty src, empty dst, or a nil image collapse to the empty filter (transparent output).
	for _, f := range []interface{ OnAffectsTransparentBlack() bool }{
		imagefilter.Image(nil, geom.RectWH(4, 4), geom.RectWH(4, 4), shaders.SamplingOptions{}),
		imagefilter.Image(img, geom.Rect{}, geom.RectWH(4, 4), shaders.SamplingOptions{}),
		imagefilter.Image(img, geom.RectWH(4, 4), geom.Rect{}, shaders.SamplingOptions{}),
		// src entirely outside the image bounds:
		imagefilter.Image(img, geom.RectLTRB(10, 10, 12, 12), geom.RectWH(4, 4),
			shaders.SamplingOptions{}),
	} {
		if f == nil {
			t.Fatal("expected an Empty() filter, not nil")
		}
	}
	pix := drawSquare(t, func(p *canvas.Paint) {
		p.ImageFilter = imagefilter.Image(img, geom.Rect{}, geom.RectWH(4, 4),
			shaders.SamplingOptions{})
	})
	for i := range pix.Pix {
		if pix.Pix[i] != 0 {
			t.Fatalf("pixel %d = %08x, want fully transparent output", i, pix.Pix[i])
		}
	}
}
