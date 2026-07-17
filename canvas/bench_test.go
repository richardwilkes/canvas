// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// CPU benchmark suite: testing.B benchmarks of the canvas-level operations unison actually issues per frame —
// fill/stroke path, styled rect/rrect, gradient fill, blur mask, text run, image scale. Each reports allocs/op
// (b.ReportAllocs), because per-frame heap traffic is the most likely Go-specific perf failure mode and is a
// first-class benchmark output. These are pure-Go (no cgo), so they run in the shipped module; the cgo baseline
// comparison lives in internal/oracle/probe. Run: go test -run '^$' -bench . -benchmem ./canvas/

package canvas

import (
	"fmt"
	"os"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/textblob"
)

// benchCanvas returns a fresh canvas drawing into an owned N32 surface of the given size.
func benchCanvas(w, h int32) *Canvas {
	return NewForPixmap(raster.NewPixmap(w, h))
}

// benchPath builds a deterministic 3-contour mixed-segment path spanning the given box, matching the shape used by the
// raster-level fill benchmarks so the numbers are comparable across layers.
func benchPath(w, h float32) *path.Path {
	p := path.New()
	// A pseudo-random-but-fixed walk; no rng dependency so the shape is stable across runs.
	xs := []float32{0.10, 0.85, 0.30, 0.95, 0.20, 0.70, 0.05, 0.60, 0.45}
	ys := []float32{0.15, 0.20, 0.90, 0.55, 0.35, 0.95, 0.50, 0.10, 0.75}
	for c := 0; c < 3; c++ {
		base := c * 3
		p.MoveTo(xs[base%len(xs)]*w, ys[base%len(ys)]*h)
		for s := 1; s <= 8; s++ {
			i := (base + s) % len(xs)
			j := (base + s*2) % len(ys)
			switch s % 3 {
			case 0:
				p.LineTo(xs[i]*w, ys[j]*h)
			case 1:
				p.QuadTo(xs[j]*w, ys[i]*h, xs[i]*w, ys[j]*h)
			default:
				k := (base + s*3) % len(xs)
				p.CubicTo(xs[j]*w, ys[i]*h, xs[k]*w, ys[k]*h, xs[i]*w, ys[j]*h)
			}
		}
		p.Close()
	}
	return p
}

func benchFillPaint(aa bool) *Paint {
	p := NewPaint()
	p.Color = 0x80336699
	p.AntiAlias = aa
	return p
}

// BenchmarkDrawRectFillNonAA is the highest-frequency UI op: a solid non-AA filled rect.
func BenchmarkDrawRectFillNonAA(b *testing.B) {
	c := benchCanvas(256, 256)
	paint := benchFillPaint(false)
	r := geom.RectLTRB(8, 8, 248, 248)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawRect(r, paint)
	}
}

// BenchmarkDrawRectFillAA is the same rect with analytic AA (the default styled-fill path).
func BenchmarkDrawRectFillAA(b *testing.B) {
	c := benchCanvas(256, 256)
	paint := benchFillPaint(true)
	r := geom.RectLTRB(8.4, 8.4, 247.6, 247.6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawRect(r, paint)
	}
}

// BenchmarkDrawRRectFillAA covers unison's most common styled primitive (FillRRectOp note): a filled, antialiased
// rounded rectangle.
func BenchmarkDrawRRectFillAA(b *testing.B) {
	c := benchCanvas(256, 256)
	paint := benchFillPaint(true)
	r := geom.RectLTRB(8.4, 8.4, 247.6, 247.6)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawRoundRect(r, 24, 24, paint)
	}
}

// BenchmarkDrawPathFillAA fills a moderately complex 3-contour mixed-curve path with AA.
func BenchmarkDrawPathFillAA(b *testing.B) {
	c := benchCanvas(256, 256)
	paint := benchFillPaint(true)
	p := benchPath(256, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawPath(p, paint)
	}
}

// BenchmarkStrokePathAA strokes the same path (stroker → AA fill).
func BenchmarkStrokePathAA(b *testing.B) {
	c := benchCanvas(256, 256)
	paint := NewPaint()
	paint.Color = 0xFF204080
	paint.AntiAlias = true
	paint.Style = StyleStroke
	paint.StrokeWidth = 3
	paint.Join = JoinRound
	paint.Cap = CapRound
	p := benchPath(256, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawPath(p, paint)
	}
}

// BenchmarkGradientRectFill fills a rect with a multi-stop linear gradient (dominant shader in UI).
func BenchmarkGradientRectFill(b *testing.B) {
	c := benchCanvas(256, 256)
	sh := shaders.NewLinearGradient(geom.Pt(8, 8), geom.Pt(248, 248),
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.5, 1},
		shaders.TileClamp, nil)
	if sh == nil {
		b.Fatal("nil gradient shader")
	}
	paint := NewPaint()
	paint.AntiAlias = true
	paint.Shader = sh
	r := geom.RectLTRB(8, 8, 248, 248)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawRect(r, paint)
	}
}

// BenchmarkBlurRect draws a normal-blurred filled rect (mask filter → the drop-shadow/soft-edge case).
func BenchmarkBlurRect(b *testing.B) {
	c := benchCanvas(256, 256)
	mf := maskfilter.NewBlur(maskfilter.BlurNormal, 6, true)
	if mf == nil {
		b.Fatal("nil blur mask filter")
	}
	paint := NewPaint()
	paint.Color = 0xC0202020
	paint.AntiAlias = true
	paint.MaskFilter = mf
	r := geom.RectLTRB(48, 48, 208, 208)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawRect(r, paint)
	}
}

func benchFont(b *testing.B, size float32) *font.Font {
	b.Helper()
	data, err := os.ReadFile("../font/testdata/Roboto-Regular.ttf")
	if err != nil {
		b.Fatalf("read font: %v", err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		b.Fatalf("typeface: %v", err)
	}
	return font.NewFont(tf, size, 1, 0)
}

// BenchmarkTextRun draws a positioned text blob (the strike/mask/blit path).
func BenchmarkTextRun(b *testing.B) {
	c := benchCanvas(320, 64)
	f := benchFont(b, 18)
	blob := textblob.MakeFromText([]byte("The quick brown fox"), f, font.TextEncodingUTF8)
	if blob == nil {
		b.Fatal("nil text blob")
	}
	paint := NewPaint()
	paint.Color = 0xFF101010
	paint.AntiAlias = true
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawTextBlob(blob, 8, 40, paint)
	}
}

// benchImage builds a small opaque RGBA image with a deterministic pattern.
func benchImage(w, h int32) *imagecore.Image {
	src := raster.NewPixmap(w, h)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			r := uint32(x*255/w) & 0xFF
			g := uint32(y*255/h) & 0xFF
			bl := uint32((x + y) & 0xFF)
			src.Pix[y*src.RowPixels+x] = 0xFF000000 | r<<16 | g<<8 | bl
		}
	}
	return imagecore.Copy(src)
}

// BenchmarkGradientPaintFill fills the whole device with a multi-stop linear gradient via DrawPaint (the full-device
// background-gradient lane — the drawPaint counterpart of BenchmarkGradientRectFill, exercising the row-band
// parallelism added for full-device shaded fills).
func BenchmarkGradientPaintFill(b *testing.B) {
	c := benchCanvas(256, 256)
	sh := shaders.NewLinearGradient(geom.Pt(0, 0), geom.Pt(256, 256),
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.5, 1},
		shaders.TileClamp, nil)
	if sh == nil {
		b.Fatal("nil gradient shader")
	}
	paint := NewPaint()
	paint.AntiAlias = true
	paint.Shader = sh
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawPaint(paint)
	}
}

// BenchmarkImagePaintFill fills the whole device with a clamp/linear image shader via DrawPaint (the full-device
// image-background lane).
func BenchmarkImagePaintFill(b *testing.B) {
	c := benchCanvas(256, 256)
	img := benchImage(64, 64)
	sh := shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp,
		shaders.SamplingOptions{Filter: shaders.FilterLinear}, nil)
	if sh == nil {
		b.Fatal("nil image shader")
	}
	paint := NewPaint()
	paint.Shader = sh
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawPaint(paint)
	}
}

// benchAtlasSprites builds the deterministic sprite arrays for the DrawAtlas benchmarks: n rotated unit-scale sprites
// cycling through four 16x16 tiles of a 64x64 atlas.
func benchAtlasSprites(n int) (xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color) {
	xforms = make([]geom.RSXform, n)
	tex = make([]geom.Rect, n)
	colors = make([]colorcore.Color, n)
	for i := range n {
		xforms[i] = geom.RSXformFromRadians(1, float32(i)*0.13,
			float32(8+(i*23)%224), float32(8+(i*41)%224), 8, 8)
		tex[i] = geom.RectXYWH(float32(16*(i%4)), float32(16*((i/4)%4)), 16, 16)
		colors[i] = colorcore.Color(0xFF000000 | uint32((i*37)&0xFF)<<16 | uint32((i*53)&0xFF)<<8 |
			uint32((i*71)&0xFF))
	}
	return xforms, tex, colors
}

// BenchmarkDrawAtlas measures the CPU per-sprite lowering: allocs/op must stay O(1) per call — the per-sprite scratch
// work makes each sprite allocation-free. Sub-benchmarks cover 100/1000 sprites, with and without per-sprite color
// modulation.
func BenchmarkDrawAtlas(b *testing.B) {
	img := benchImage(64, 64)
	sampling := shaders.SamplingOptions{Filter: shaders.FilterLinear}
	for _, n := range []int{100, 1000} {
		xforms, tex, colors := benchAtlasSprites(n)
		b.Run(fmt.Sprintf("Plain%d", n), func(b *testing.B) {
			c := benchCanvas(256, 256)
			paint := NewPaint()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				c.DrawAtlas(img, xforms, tex, nil, raster.BlendSrcOver, sampling, nil, paint)
			}
		})
		b.Run(fmt.Sprintf("Colors%d", n), func(b *testing.B) {
			c := benchCanvas(256, 256)
			paint := NewPaint()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				c.DrawAtlas(img, xforms, tex, colors, raster.BlendModulate, sampling, nil, paint)
			}
		})
	}
}

// BenchmarkImageScale draws a small image scaled up with linear sampling (the image-shader path).
func BenchmarkImageScale(b *testing.B) {
	c := benchCanvas(256, 256)
	img := benchImage(64, 64)
	paint := NewPaint()
	sampling := shaders.SamplingOptions{Filter: shaders.FilterLinear}
	src := geom.RectWH(64, 64)
	dst := geom.RectLTRB(16, 16, 240, 240)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.DrawImageRect(img, src, dst, sampling, paint, ConstraintStrict)
	}
}
