// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package shaders

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
)

// testImage builds a 2x2 opaque image: red, green / blue, white.
func testImage(t *testing.T) *imagecore.Image {
	t.Helper()
	info, ok := imagecore.MakeInfo(2, 2, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeOpaque)
	if !ok {
		t.Fatal("bad info")
	}
	p := imagecore.NewPixels(info)
	p.Words[0] = 0xFF0000FF // red
	p.Words[1] = 0xFF00FF00 // green
	p.Words[2] = 0xFFFF0000 // blue
	p.Words[3] = 0xFFFFFFFF // white
	return imagecore.FromPixels(p)
}

func shadePixel(t *testing.T, s Shader, ctm geom.Matrix, alpha colorcore.Color, x, y int32) colorcore.PMColor4f {
	t.Helper()
	p := Compile(s, ctm, alpha)
	if p == nil {
		t.Fatal("compile failed")
	}
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(x, y, out[:])
	return out[0]
}

func near1(a, b float32) bool {
	d := a - b
	return d < 1.5/255 && d > -1.5/255
}

func TestImageShaderNearestIdentity(t *testing.T) {
	img := testImage(t)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, nil)
	cases := []struct {
		x, y    int32
		r, g, b float32
	}{
		{x: 0, y: 0, r: 1, g: 0, b: 0},
		{x: 1, y: 0, r: 0, g: 1, b: 0},
		{x: 0, y: 1, r: 0, g: 0, b: 1},
		{x: 1, y: 1, r: 1, g: 1, b: 1},
		{x: 5, y: 5, r: 1, g: 1, b: 1}, // clamped
	}
	for _, c := range cases {
		got := shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, c.x, c.y)
		if !near1(got.R, c.r) || !near1(got.G, c.g) || !near1(got.B, c.b) || !near1(got.A, 1) {
			t.Errorf("(%d,%d) = %+v, want rgb %v %v %v", c.x, c.y, got, c.r, c.g, c.b)
		}
	}
}

func TestImageShaderBilerpMidpoint(t *testing.T) {
	img := testImage(t)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{Filter: FilterLinear}, nil)
	// Scale up 2x: device pixel (1,1) has center (1.5,1.5) → image space (0.75,0.75). The bilerp kernel samples the 2x2
	// neighborhood with fx = fy = fract(0.75+0.5) = 0.25: weights (0,0)=0.5625 red, (1,0)=0.1875 green, (0,1)=0.1875
	// blue, (1,1)=0.0625 white, so R = 0.5625+0.0625 = 0.625, G = B = 0.1875+0.0625 = 0.25.
	var ctm geom.Matrix
	ctm.SetScale(2, 2)
	got := shadePixel(t, s, ctm, 0xFF000000, 1, 1)
	if !near1(got.R, 0.625) || !near1(got.G, 0.25) || !near1(got.B, 0.25) || !near1(got.A, 1) {
		t.Errorf("bilerp midpoint = %+v", got)
	}
}

func TestImageShaderTileModes(t *testing.T) {
	img := testImage(t)
	// Repeat: pixel (2,0) maps back to (0,0) → red.
	s := NewImage(img, TileRepeat, TileRepeat, SamplingOptions{}, nil)
	got := shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 2, 0)
	if !near1(got.R, 1) || !near1(got.G, 0) {
		t.Errorf("repeat = %+v", got)
	}
	// Mirror: pixel (2,0) reflects to (1,0) → green; pixel (3,0) → (0,0) red.
	s = NewImage(img, TileMirror, TileMirror, SamplingOptions{}, nil)
	got = shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 2, 0)
	if !near1(got.G, 1) || !near1(got.R, 0) {
		t.Errorf("mirror(2) = %+v", got)
	}
	got = shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 3, 0)
	if !near1(got.R, 1) {
		t.Errorf("mirror(3) = %+v", got)
	}
	// Decal: outside is transparent black.
	s = NewImage(img, TileDecal, TileDecal, SamplingOptions{}, nil)
	got = shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 2, 0)
	if got.R != 0 || got.G != 0 || got.B != 0 || got.A != 0 {
		t.Errorf("decal outside = %+v", got)
	}
	got = shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 1, 1)
	if !near1(got.R, 1) || !near1(got.A, 1) {
		t.Errorf("decal inside = %+v", got)
	}
}

func TestImageShaderAlphaOnlyTint(t *testing.T) {
	info, _ := imagecore.MakeInfo(1, 1, imagecore.ColorTypeAlpha8, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	p.Bytes[0] = 0x80
	img := imagecore.FromPixels(p)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, nil)
	// Paint color red: output = red tinted by alpha 0x80, premultiplied.
	got := shadePixel(t, s, geom.IdentityMatrix(), 0xFFFF0000, 0, 0)
	if !near1(got.A, 128.0/255) || !near1(got.R, 128.0/255) || !near1(got.G, 0) {
		t.Errorf("a8 tint = %+v", got)
	}
}

func TestImageShaderUnpremulSource(t *testing.T) {
	info, _ := imagecore.MakeInfo(1, 1, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeUnpremul)
	p := imagecore.NewPixels(info)
	p.Words[0] = 0x80FFFFFF // white at alpha 0x80, unpremul
	img := imagecore.FromPixels(p)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, nil)
	got := shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 0, 0)
	// premultiplied: rgb = 1 * (128/255)
	if !near1(got.R, 128.0/255) || !near1(got.A, 128.0/255) {
		t.Errorf("unpremul source = %+v", got)
	}
}

func TestImageShaderGrayAnd565(t *testing.T) {
	ginfo, _ := imagecore.MakeInfo(1, 1, imagecore.ColorTypeGray8, imagecore.AlphaTypeOpaque)
	gp := imagecore.NewPixels(ginfo)
	gp.Bytes[0] = 0x40
	s := NewImage(imagecore.FromPixels(gp), TileClamp, TileClamp, SamplingOptions{}, nil)
	got := shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 0, 0)
	if !near1(got.R, 64.0/255) || !near1(got.G, 64.0/255) || !near1(got.B, 64.0/255) || !near1(got.A, 1) {
		t.Errorf("gray = %+v", got)
	}

	cinfo, _ := imagecore.MakeInfo(1, 1, imagecore.ColorTypeRGB565, imagecore.AlphaTypeOpaque)
	cp := imagecore.NewPixels(cinfo)
	cp.U16s[0] = 31 << 11 // pure red
	s = NewImage(imagecore.FromPixels(cp), TileClamp, TileClamp, SamplingOptions{}, nil)
	got = shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, 0, 0)
	if !near1(got.R, 1) || !near1(got.G, 0) || !near1(got.A, 1) {
		t.Errorf("565 = %+v", got)
	}
}

func TestImageShaderCubicIsSmooth(t *testing.T) {
	img := testImage(t)
	s := NewImage(img, TileClamp, TileClamp,
		SamplingOptions{UseCubic: true, Cubic: CubicResampler{B: 1.0 / 3, C: 1.0 / 3}}, nil)
	var ctm geom.Matrix
	ctm.SetScale(4, 4)
	got := shadePixel(t, s, ctm, 0xFF000000, 3, 3) // near the center of the upscale
	if got.A < 0.99 || got.R < 0.2 || got.R > 0.8 {
		t.Errorf("cubic = %+v", got)
	}
	// Invalid cubic parameters return nil.
	if NewImage(img, TileClamp, TileClamp,
		SamplingOptions{UseCubic: true, Cubic: CubicResampler{B: 2, C: 0}}, nil) != nil {
		t.Error("invalid cubic accepted")
	}
}

func TestLegacyImageContextSelection(t *testing.T) {
	img := testImage(t)
	nearest := SamplingOptions{}
	// Eligible: nearest, clamp/clamp, scale-translate matrix.
	s := NewImage(img, TileClamp, TileClamp, nearest, nil)
	if MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF) == nil {
		t.Fatal("legacy context expected")
	}
	// Decal → pipeline.
	s = NewImage(img, TileDecal, TileDecal, nearest, nil)
	if MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF) != nil {
		t.Fatal("decal must not take the legacy lane")
	}
	// Mixed tile modes → pipeline.
	s = NewImage(img, TileClamp, TileRepeat, nearest, nil)
	if MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF) != nil {
		t.Fatal("mixed tiles must not take the legacy lane")
	}
	// Cubic → pipeline.
	s = NewImage(img, TileClamp, TileClamp,
		SamplingOptions{UseCubic: true, Cubic: CubicResampler{B: 1.0 / 3, C: 1.0 / 3}}, nil)
	if MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF) != nil {
		t.Fatal("cubic must not take the legacy lane")
	}
	// A local matrix wrapper unwraps.
	var lm geom.Matrix
	lm.SetScale(2, 2)
	s = NewImage(img, TileClamp, TileClamp, nearest, &lm)
	if MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF) == nil {
		t.Fatal("local-matrix wrapper should still take the legacy lane")
	}
}

func TestLegacyShadeSpanMatchesPixels(t *testing.T) {
	img := testImage(t)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, nil)
	ctx := MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF)
	if ctx == nil {
		t.Fatal("no legacy context")
	}
	if !ctx.OpaqueAlpha() {
		t.Fatal("opaque image at alpha 255 should set kOpaqueAlpha")
	}
	var row [4]uint32
	ctx.ShadeSpan32(0, 0, row[:2])
	if row[0] != 0xFF0000FF || row[1] != 0xFF00FF00 {
		t.Fatalf("row0 = %08x %08x", row[0], row[1])
	}
	ctx.ShadeSpan32(0, 1, row[:2])
	if row[0] != 0xFFFF0000 || row[1] != 0xFFFFFFFF {
		t.Fatalf("row1 = %08x %08x", row[0], row[1])
	}
	// Clamped read past the right edge repeats the edge pixel.
	ctx.ShadeSpan32(1, 0, row[:3])
	if row[1] != 0xFF00FF00 || row[2] != 0xFF00FF00 {
		t.Fatalf("clamped = %08x %08x %08x", row[0], row[1], row[2])
	}
	// Paint alpha scales the output (alphaMulQLegacy with 128 -> 129/256 scale).
	ctx = MakeLegacyImageContext(s, geom.IdentityMatrix(), 0x80)
	if ctx == nil || ctx.OpaqueAlpha() {
		t.Fatal("translucent paint should clear kOpaqueAlpha")
	}
	ctx.ShadeSpan32(0, 0, row[:1])
	want := alphaMulQLegacy(0xFF0000FF, alpha255To256(0x80))
	if row[0] != want {
		t.Fatalf("alpha-scaled = %08x want %08x", row[0], want)
	}
}

func TestLegacyBilerpScaleAgreesWithPipeline(t *testing.T) {
	// The legacy 4-bit bilerp and the float pipeline agree within a few /255 on smooth content.
	info, _ := imagecore.MakeInfo(4, 4, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for y := range 4 {
		for x := range 4 {
			v := uint32(60*x + 15*y)
			p.Words[y*4+x] = 0xFF000000 | v | v<<8 | v<<16
		}
	}
	img := imagecore.FromPixels(p)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{Filter: FilterLinear}, nil)
	var ctm geom.Matrix
	ctm.SetScale(1.7, 1.3)
	ctx := MakeLegacyImageContext(s, ctm, 0xFF)
	if ctx == nil {
		t.Fatal("no legacy context")
	}
	pl := Compile(s, ctm, 0xFF000000)
	if pl == nil {
		t.Fatal("no pipeline")
	}
	var legacy [6]uint32
	var flt [6]colorcore.PMColor4f
	ctx.ShadeSpan32(0, 2, legacy[:])
	pl.ShadeSpan(0, 2, flt[:])
	for i := range legacy {
		lr := int32(legacy[i] & 0xFF)
		fr := int32(flt[i].R*255 + 0.5)
		if d := lr - fr; d < -6 || d > 6 {
			t.Errorf("pixel %d: legacy r=%d float r=%d", i, lr, fr)
		}
	}
}

// TestLegacyContextReuseResetsShaderProc locks the pooling-reuse invariant: a BitmapProcContext taken from the pool and
// re-set up for a scale+bilerp draw must not inherit the translate-fast-path shaderProc a prior nearest/translate draw
// installed (setup only conditionally assigns shaderProc, so it must clear it up front). Otherwise ShadeSpan32 would
// shade the reused context through the stale proc. sync.Pool reuse is nondeterministic, so this exercises the setup
// re-init directly on one context object.
func TestLegacyContextReuseResetsShaderProc(t *testing.T) {
	img := testImage(t)

	// First use: nearest, opaque, identity → the translate fast path installs a shaderProc.
	s1 := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, nil)
	ctx := MakeLegacyImageContext(s1, geom.IdentityMatrix(), 0xFF)
	if ctx == nil {
		t.Fatal("no legacy context for the nearest/translate case")
	}
	if ctx.state.shaderProc == nil {
		t.Fatal("nearest/opaque/clamp/translate setup should install a shaderProc")
	}

	// Reuse the same context object for a scale+bilerp draw, exactly as pool reuse does.
	var ctm geom.Matrix
	ctm.SetScale(1.7, 1.3)
	inv, ok := ctm.Invert()
	if !ok {
		t.Fatal("scale matrix not invertible")
	}
	pm, pmOK := img.PeekPixels(imagecore.CachingAllow).PixmapView()
	if !pmOK {
		t.Fatal("no pixmap view")
	}
	if !ctx.state.setup(pm, &inv, 0xFF, SamplingOptions{Filter: FilterLinear}, TileClamp, TileClamp) {
		t.Fatal("reused setup failed")
	}
	if ctx.state.shaderProc != nil {
		t.Fatal("scale+bilerp setup must clear the stale translate-path shaderProc")
	}

	// The reused context must shade identically to a brand-new one built the same way.
	fresh := MakeLegacyImageContext(NewImage(img, TileClamp, TileClamp, SamplingOptions{Filter: FilterLinear}, nil), ctm, 0xFF)
	if fresh == nil {
		t.Fatal("no fresh legacy context")
	}
	var got, want [5]uint32
	ctx.ShadeSpan32(0, 1, got[:])
	fresh.ShadeSpan32(0, 1, want[:])
	if got != want {
		t.Fatalf("reused context span %v != fresh %v", got, want)
	}
}

func TestPerlinCollapsesAndDeterminism(t *testing.T) {
	// 0 octaves collapse per the C.
	s := NewFractalNoise(0.1, 0.1, 0, 42, 0, 0)
	if cs, ok := s.(*ColorShader); !ok || !cs.IsConstant() {
		t.Fatal("fractal 0-octave should collapse to a constant shader")
	}
	s = NewTurbulence(0.1, 0.1, 0, 42, 0, 0)
	if cs, ok := s.(*ColorShader); !ok || !cs.IsConstant() {
		t.Fatal("turbulence 0-octave should collapse")
	}
	// Invalid inputs return nil.
	if NewFractalNoise(-1, 0.1, 1, 0, 0, 0) != nil {
		t.Fatal("negative frequency accepted")
	}

	// Determinism: two shaders with the same seed produce identical spans.
	a := NewFractalNoise(0.05, 0.08, 3, 7, 0, 0)
	b := NewFractalNoise(0.05, 0.08, 3, 7, 0, 0)
	pa := Compile(a, geom.IdentityMatrix(), 0xFF000000)
	pb := Compile(b, geom.IdentityMatrix(), 0xFF000000)
	var sa, sb [16]colorcore.PMColor4f
	pa.ShadeSpan(3, 9, sa[:])
	pb.ShadeSpan(3, 9, sb[:])
	if sa != sb {
		t.Fatal("same seed must produce identical noise")
	}
	// Premul invariant: rgb <= a.
	for _, c := range sa {
		if c.R > c.A || c.G > c.A || c.B > c.A {
			t.Fatalf("not premultiplied: %+v", c)
		}
		if c.A < 0 || c.A > 1 {
			t.Fatalf("alpha out of range: %+v", c)
		}
	}
	// A different seed differs somewhere.
	cshader := NewFractalNoise(0.05, 0.08, 3, 8, 0, 0)
	pc := Compile(cshader, geom.IdentityMatrix(), 0xFF000000)
	var sc [16]colorcore.PMColor4f
	pc.ShadeSpan(3, 9, sc[:])
	if sa == sc {
		t.Fatal("different seeds produced identical noise")
	}
}

func TestPerlinStitchWrapsAtTile(t *testing.T) {
	// With stitching, the pattern must repeat with the tile period.
	s := NewTurbulence(0.1, 0.1, 2, 5, 20, 20)
	p := Compile(s, geom.IdentityMatrix(), 0xFF000000)
	var row0, row20 [8]colorcore.PMColor4f
	p.ShadeSpan(0, 0, row0[:])
	p.ShadeSpan(20, 20, row20[:]) // one tile away in both axes
	for i := range row0 {
		if !near1(row0[i].R, row20[i].R) || !near1(row0[i].A, row20[i].A) {
			t.Fatalf("pixel %d: %+v vs %+v", i, row0[i], row20[i])
		}
	}
}

func TestImageShaderOnRasterPixmapImage(t *testing.T) {
	// Images wrapped from surface snapshots (raster.Pixmap) shade identically.
	pm := raster.NewPixmap(2, 1)
	pm.Pix[0] = 0xFF102030
	pm.Pix[1] = 0xFF405060
	img := imagecore.FromPixmap(*pm)
	s := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, nil)
	ctx := MakeLegacyImageContext(s, geom.IdentityMatrix(), 0xFF)
	if ctx == nil {
		t.Fatal("no legacy context")
	}
	var row [2]uint32
	ctx.ShadeSpan32(0, 0, row[:])
	if row[0] != 0xFF102030 || row[1] != 0xFF405060 {
		t.Fatalf("row = %08x %08x", row[0], row[1])
	}
}

func TestAsImage(t *testing.T) {
	img := testImage(t)

	// A bare image shader reports itself with its (optimized) tile modes and an identity local matrix.
	s := NewImage(img, TileRepeat, TileClamp, SamplingOptions{}, nil)
	gotImg, tmx, tmy, lm, ok := AsImage(s)
	if !ok {
		t.Fatal("AsImage on an image shader returned ok=false")
	}
	if gotImg != img {
		t.Error("AsImage returned a different image")
	}
	if tmx != TileRepeat || tmy != TileClamp {
		t.Errorf("AsImage tile modes = %v,%v, want repeat,clamp", tmx, tmy)
	}
	if !lm.IsIdentity() {
		t.Errorf("AsImage local matrix = %v, want identity", lm.As9())
	}

	// A local-matrix wrapper reports the wrapper's matrix (AsImage forwards through the wrapper).
	m := geom.TranslateMatrix(3, 7)
	s2 := NewImage(img, TileClamp, TileClamp, SamplingOptions{}, &m)
	gotImg2, _, _, lm2, ok := AsImage(s2)
	if !ok || gotImg2 != img {
		t.Fatal("AsImage on a local-matrix-wrapped image shader failed")
	}
	if lm2.Get(geom.MTransX) != 3 || lm2.Get(geom.MTransY) != 7 {
		t.Errorf("AsImage local matrix translate = (%v,%v), want (3,7)",
			lm2.Get(geom.MTransX), lm2.Get(geom.MTransY))
	}

	// A non-image shader reports ok=false.
	if _, _, _, _, ok = AsImage(NewColor(colorcore.ARGB(255, 1, 2, 3))); ok {
		t.Error("AsImage on a color shader returned ok=true")
	}
}

// f16TestImage builds a 2x4 premultiplied F16 image whose pixels are distinct per row and column: red steps 0.25 per
// row, green is 0.25 in column 0 and 0.75 in column 1, blue is 0.5 everywhere, alpha 1. The half bits are written
// directly (every value is exactly representable) so the test does not depend on a float-to-half converter.
func f16TestImage(t *testing.T) *imagecore.Image {
	t.Helper()
	info, ok := imagecore.MakeInfo(2, 4, imagecore.ColorTypeRGBAF16, imagecore.AlphaTypePremul)
	if !ok {
		t.Fatal("bad info")
	}
	p := imagecore.NewPixels(info)
	for y := range int32(4) {
		for x := range int32(2) {
			i := y*p.RowElems + 4*x
			p.U16s[i] = f16Half(0.25 * float32(y+1))
			p.U16s[i+1] = f16Half(0.25)
			if x == 1 {
				p.U16s[i+1] = f16Half(0.75)
			}
			p.U16s[i+2] = f16Half(0.5)
			p.U16s[i+3] = f16Half(1)
		}
	}
	return imagecore.FromPixels(p)
}

// f16Half returns the half bits for the quarter-step values the F16 fixtures use (0.25, 0.5, 0.75, 1).
func f16Half(v float32) uint16 {
	switch v {
	case 0.25:
		return 0x3400
	case 0.5:
		return 0x3800
	case 0.75:
		return 0x3A00
	case 1:
		return 0x3C00
	default:
		panic("unexpected half value")
	}
}

// TestImageShaderF16Gather covers the F16 gather's element addressing: RowElems already counts the four elements per
// pixel, so the row term must not be scaled a second time. Before the fix every row but the first read the wrong pixels
// and rows at or past height/4 indexed past the end of the storage.
func TestImageShaderF16Gather(t *testing.T) {
	s := NewImage(f16TestImage(t), TileClamp, TileClamp, SamplingOptions{}, nil)
	for y := range int32(4) {
		for x := range int32(2) {
			wantG := float32(0.25)
			if x == 1 {
				wantG = 0.75
			}
			got := shadePixel(t, s, geom.IdentityMatrix(), 0xFF000000, x, y)
			if !near1(got.R, 0.25*float32(y+1)) || !near1(got.G, wantG) ||
				!near1(got.B, 0.5) || !near1(got.A, 1) {
				t.Errorf("f16 (%d,%d) = %+v, want rgb %v %v 0.5", x, y, got, 0.25*float32(y+1), wantG)
			}
		}
	}
}

// TestImageShaderF16BilerpSpansRows checks the F16 gather through the (unfused) bilinear sampler, whose four taps land
// on two different rows — the row stride has to be right for the vertical pair to blend the neighboring rows.
func TestImageShaderF16BilerpSpansRows(t *testing.T) {
	s := NewImage(f16TestImage(t), TileClamp, TileClamp, SamplingOptions{Filter: FilterLinear}, nil)
	// Device pixel (0,1) has center (0.5,1.5), so fx = fract(1.0) = 0 (column 0 only, clamped) and fy = fract(2.0) = 0,
	// which weights rows 0 and 1 evenly: R = (0.25+0.5)/2 = 0.375.
	var ctm geom.Matrix
	ctm.SetTranslate(0, 0.5)
	got := shadePixel(t, s, ctm, 0xFF000000, 0, 1)
	if !near1(got.R, 0.375) || !near1(got.G, 0.25) || !near1(got.B, 0.5) || !near1(got.A, 1) {
		t.Errorf("f16 bilerp across rows = %+v, want r 0.375", got)
	}
}

// TestFilterDecalKeepsChildSampling covers the nested-image sampling decision: FilterDecalShader applies the CTM before
// running its child, and the child's tweakSampling must still see the real device transform. With the total matrix
// dropped to identity the child downgraded linear filtering to nearest, so a 2x upscale returned the source texel
// instead of the bilerp of the four neighbors.
func TestFilterDecalKeepsChildSampling(t *testing.T) {
	img := testImage(t)
	inner := NewImage(img, TileClamp, TileClamp, SamplingOptions{Filter: FilterLinear}, nil)
	// Bounds wide enough that every decal weight saturates to 1, leaving the child's color untouched.
	s := NewFilterDecal(inner, geom.Rect{Left: -10, Top: -10, Right: 10, Bottom: 10})
	var ctm geom.Matrix
	ctm.SetScale(2, 2)
	// Same expectation as TestImageShaderBilerpMidpoint: the decal wrapper must not change the sampling.
	got := shadePixel(t, s, ctm, 0xFF000000, 1, 1)
	if !near1(got.R, 0.625) || !near1(got.G, 0.25) || !near1(got.B, 0.25) || !near1(got.A, 1) {
		t.Errorf("decal-wrapped bilerp = %+v, want rgb 0.625 0.25 0.25", got)
	}
}
