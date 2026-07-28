// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the GPU image-shader fragment-processor lane (paintconvert.go's makeImageShaderFP): image shaders
// converted via the paint-to-FP conversion and rendered through the fill-rect op, compared against the CPU image-shader
// pipeline (the Go-GPU vs Go-CPU self-consistency lane). Before this lane existed a paint carrying an image shader
// failed fragment-processor conversion and the draw was a silent no-op. Skips when no GL context is available.

package gl_test

import (
	"fmt"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// rgbaTestImage builds a small asymmetric premul RGBA image (red varies with x, green with y, a constant blue and full
// alpha) so a vertical/horizontal flip or an x/y transpose in the texture sampling would diverge from the CPU
// image-shader pipeline rather than match it.
func rgbaTestImage(w, h int32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			r := uint32(30 + x*50)
			g := uint32(20 + y*50)
			p.Words[y*p.RowElems+x] = r | g<<8 | 0x60<<16 | 0xFF<<24
		}
	}
	return imagecore.FromPixels(p)
}

// alphaTestImage builds a small A8 (alpha-only) image whose coverage varies with x and y.
func alphaTestImage(w, h int32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeAlpha8, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			p.Bytes[y*p.RowElems+x] = byte(40 + x*40 + y*20)
		}
	}
	return imagecore.FromPixels(p)
}

// TestLiveImageShaderSwatches renders an RGBA image shader with each tile mode (and a magnifying local matrix) through
// the paint-to-FP conversion plus the fill-rect op and compares interior and exterior texels against the CPU
// image-shader pipeline. Nearest sampling at pixel centers samples exact texels, so clamp/repeat/mirror/decal and the
// local-matrix mapping all reproduce the CPU result.
func TestLiveImageShaderSwatches(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 16
	ctm := geom.IdentityMatrix()
	img := rgbaTestImage(4, 4)
	nearest := shaders.SamplingOptions{}
	scale := geom.ScaleMatrix(2, 2)

	cases := []struct {
		shader shaders.Shader
		name   string
	}{
		{name: "clamp", shader: shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, nearest, nil)},
		{name: "repeat", shader: shaders.NewImage(img, shaders.TileRepeat, shaders.TileRepeat, nearest, nil)},
		{name: "mirror", shader: shaders.NewImage(img, shaders.TileMirror, shaders.TileMirror, nearest, nil)},
		{name: "decal", shader: shaders.NewImage(img, shaders.TileDecal, shaders.TileDecal, nearest, nil)},
		{name: "clamp-x-repeat-y", shader: shaders.NewImage(img, shaders.TileClamp, shaders.TileRepeat, nearest,
			nil)},
		{name: "clamp-scaled", shader: shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, nearest,
			&scale)},
	}
	// A mix of interior ([0,4) texels) and exterior samples that exercise tiling / clamp / decal outside the image, all
	// landing clearly inside a texel (never on an integer boundary).
	samples := [][2]int32{{1, 1}, {2, 3}, {5, 1}, {1, 9}, {10, 10}, {3, 6}}
	for _, tc := range cases {
		if tc.shader == nil {
			t.Fatalf("%s: image-shader construction failed", tc.name)
		}
		sdc := newLiveSDC(t, dc, size)
		sdc.Clear([4]float32{0, 0, 0, 0})
		gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
			Color:     colorcore.Color4f{R: 1, G: 1, B: 1, A: 1},
			BlendMode: raster.BlendSrc,
			Shader:    tc.shader,
		}, ctm)
		if !ok {
			t.Fatalf("%s: MakePaint failed (image shader unrepresentable on GPU)", tc.name)
		}
		sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: size, Bottom: size})
		data := readSDC(t, sdc)
		for _, s := range samples {
			want := cpuShadePixel(t, tc.shader, ctm, 0xFFFFFFFF, s[0], s[1])
			checkPixelTol(t, data, size*4, s[0], s[1], want, 2,
				fmt.Sprintf("%s (%d,%d)", tc.name, s[0], s[1]))
		}
		sweepGLErrors(t, dc.Gpu(), tc.name)
		sdc.Release()
	}
}

// TestLiveImageShaderAlphaOnly renders an alpha-only image shader tinted by an opaque colored paint. The GPU lane wraps
// the texture effect in a dst-in blend against the input color; the CPU leg tints the same way, applying appendSetRGB
// from the paint color after the image's own appendStages runs. Both must agree.
func TestLiveImageShaderAlphaOnly(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	_ = env
	const size = 16
	ctm := geom.IdentityMatrix()
	img := alphaTestImage(4, 4)
	paint := colorcore.Color(0xFF3060C0) // opaque, so paint alpha does not scale the result
	shader := shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, shaders.SamplingOptions{},
		nil)
	if shader == nil {
		t.Fatal("alpha-only image-shader construction failed")
	}

	sdc := newLiveSDC(t, dc, size)
	sdc.Clear([4]float32{0, 0, 0, 0})
	gpuPaint, ok := gl.MakePaint(sdc, &gl.PaintParams{
		Color:     colorcore.Color4fFromColor(paint),
		BlendMode: raster.BlendSrc,
		Shader:    shader,
	}, ctm)
	if !ok {
		t.Fatal("MakePaint failed for the alpha-only image shader")
	}
	sdc.FillRectWithPaint(gpuPaint, &ctm, geom.Rect{Right: size, Bottom: size})
	data := readSDC(t, sdc)
	for _, s := range [][2]int32{{0, 0}, {1, 2}, {3, 1}, {6, 6}, {2, 3}} {
		want := cpuShadePixel(t, shader, ctm, paint, s[0], s[1])
		checkPixelTol(t, data, size*4, s[0], s[1], want, 2,
			fmt.Sprintf("alpha-only (%d,%d)", s[0], s[1]))
	}
	sweepGLErrors(t, dc.Gpu(), "alpha-only image shader")
	sdc.Release()
}
