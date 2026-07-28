// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// LCD unit tests: the LCD16 rec gates, the mask-gamma LUT construction, the pack4xHToMask FIR filter pinned on
// synthetic coverage patterns, and the LCD16 glyph-image lane.

package font

import (
	"math"
	"os"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

func lcdTestFont(t *testing.T, size float32) *Font {
	t.Helper()
	data, err := os.ReadFile("testdata/Roboto-Regular.ttf")
	if err != nil {
		t.Fatal(err)
	}
	tf, err := NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	f := NewFont(tf, size, 1, 0)
	f.SetEdging(EdgingSubpixelAntiAlias)
	return f
}

// TestMakeRecAndEffectsLCDGates pins the mask-format selection and the MakeRecAndEffects LCD16 gates: pixel geometry,
// the device-independent-fonts flag, and the maximum size for LCD text.
func TestMakeRecAndEffectsLCDGates(t *testing.T) {
	f := lcdTestFont(t, 24)
	identity := geom.IdentityMatrix()
	rec := func(props *DeviceProps, deviceMatrix *geom.Matrix) ScalerRec {
		r, _ := MakeRecAndEffects(f, &ScalerPaint{LumColor: colorcore.RGB(10, 20, 30)}, deviceMatrix, props)
		return r
	}

	cases := []struct {
		props     *DeviceProps
		name      string
		lcdFlags  uint16
		format    MaskFormat
		a8FromLCD bool
	}{
		{name: "nil props", props: nil, format: MaskA8, lcdFlags: 0, a8FromLCD: true},
		{name: "unknown", props: &DeviceProps{}, format: MaskA8, lcdFlags: 0, a8FromLCD: true},
		{name: "RGB_H", props: &DeviceProps{PixelGeometry: PixelGeometryRGBH}, format: MaskLCD16, lcdFlags: 0, a8FromLCD: false},
		{name: "BGR_H", props: &DeviceProps{PixelGeometry: PixelGeometryBGRH}, format: MaskLCD16, lcdFlags: recFlagLCDBGROrder, a8FromLCD: false},
		{name: "RGB_V", props: &DeviceProps{PixelGeometry: PixelGeometryRGBV}, format: MaskLCD16, lcdFlags: recFlagLCDVertical, a8FromLCD: false},
		{
			name: "BGR_V", props: &DeviceProps{PixelGeometry: PixelGeometryBGRV}, format: MaskLCD16,
			lcdFlags: recFlagLCDVertical | recFlagLCDBGROrder, a8FromLCD: false,
		},
		{
			name: "DIF", props: &DeviceProps{PixelGeometry: PixelGeometryRGBH, UseDeviceIndependentFonts: true},
			format: MaskA8, lcdFlags: 0, a8FromLCD: true,
		},
	}
	for _, c := range cases {
		r := rec(c.props, &identity)
		if r.Format != c.format {
			t.Errorf("%s: format = %d, want %d", c.name, r.Format, c.format)
		}
		if got := r.Flags & (recFlagLCDVertical | recFlagLCDBGROrder); got != c.lcdFlags {
			t.Errorf("%s: lcd flags = %#x, want %#x", c.name, got, c.lcdFlags)
		}
		if got := r.Flags&recFlagGenA8FromLCD != 0; got != c.a8FromLCD {
			t.Errorf("%s: GenA8FromLCD = %v, want %v", c.name, got, c.a8FromLCD)
		}
		// LumBits only enters LCD16 recs, canonicalized to 3 bits per channel.
		if c.format == MaskLCD16 {
			want := MaskGammaCanonicalColor(colorcore.RGB(10, 20, 30))
			if r.LumBits != want {
				t.Errorf("%s: LumBits = %#x, want %#x", c.name, r.LumBits, want)
			}
		} else if r.LumBits != 0 {
			t.Errorf("%s: LumBits = %#x, want 0", c.name, r.LumBits)
		}
	}

	// too_big_for_lcd: SK_MAX_SIZE_FOR_LCDTEXT is 48. Text size 49 disables LCD; a device scale that pushes the area
	// over the cap does too (checkPost2x2).
	rgbh := &DeviceProps{PixelGeometry: PixelGeometryRGBH}
	big := lcdTestFont(t, 49)
	r, _ := MakeRecAndEffects(big, nil, &identity, rgbh)
	if r.Format != MaskA8 || r.Flags&recFlagGenA8FromLCD == 0 {
		t.Errorf("size 49: format = %d flags = %#x, want A8 + GenA8FromLCD", r.Format, r.Flags)
	}
	var scaled geom.Matrix
	scaled.SetScale(3, 3) // 24 * 3 = 72 > 48
	r = rec(rgbh, &scaled)
	if r.Format != MaskA8 || r.Flags&recFlagGenA8FromLCD == 0 {
		t.Errorf("scaled past cap: format = %d flags = %#x, want A8 + GenA8FromLCD", r.Format, r.Flags)
	}

	// kAntiAlias edging is A8 with no LCD flags regardless of geometry.
	aa := lcdTestFont(t, 24)
	aa.SetEdging(EdgingAntiAlias)
	r, _ = MakeRecAndEffects(aa, nil, &identity, rgbh)
	if r.Format != MaskA8 || r.Flags&(recFlagGenA8FromLCD|recFlagLCDVertical|recFlagLCDBGROrder) != 0 {
		t.Errorf("kAntiAlias: format = %d flags = %#x, want plain A8", r.Format, r.Flags)
	}
	if r.LumBits != 0 {
		t.Errorf("kAntiAlias: LumBits = %#x, want 0", r.LumBits)
	}
}

// TestMaskGammaCanonicalColor pins MaskGammaCanonicalColor's 3-bit-per-channel quantization.
func TestMaskGammaCanonicalColor(t *testing.T) {
	// scale255Lum3(i) for i in 0..7: i<<5 | i<<2 | i>>1.
	scale := func(i uint32) uint8 { return uint8(i<<5 | i<<2 | i>>1) }
	wantTable := [8]uint8{0x00, 0x24, 0x49, 0x6D, 0x92, 0xB6, 0xDB, 0xFF}
	for i := uint32(0); i < 8; i++ {
		if scale(i) != wantTable[i] {
			t.Fatalf("scale255<3>(%d) = %#x, want %#x", i, scale(i), wantTable[i])
		}
		if got := scale255Lum3(i); got != uint32(wantTable[i]) {
			t.Errorf("scale255Lum3(%d) = %#x, want %#x", i, got, wantTable[i])
		}
	}
	// Channels quantize independently through the top 3 bits.
	if got := MaskGammaCanonicalColor(colorcore.RGB(0x10, 0x20, 0x30)); got != colorcore.RGB(0, 0x24, 0x24) {
		t.Errorf("CanonicalColor(10,20,30) = %#x", got)
	}
	if got := MaskGammaCanonicalColor(colorcore.RGB(0xFF, 0x80, 0x00)); got != colorcore.RGB(0xFF, 0x92, 0) {
		t.Errorf("CanonicalColor(FF,80,00) = %#x", got)
	}
}

// referenceLUT is an independent transcription of the correcting-LUT formula (float32, sRGB curve on both sides), used
// to pin buildCorrectingLUT's output.
func referenceLUT(srcI uint32, contrast float32) [256]uint8 {
	toLuma := func(l float64) float64 {
		if l <= 0.04045 {
			return l / 12.92
		}
		return math.Pow((l+0.055)/1.055, 2.4)
	}
	fromLuma := func(l float64) float64 {
		if l <= 0.0031308 {
			return l * 12.92
		}
		return 1.055*math.Pow(l, 1/2.4) - 0.055
	}
	var table [256]uint8
	src := float32(srcI) / 255
	linSrc := float32(toLuma(float64(src)))
	dst := 1 - src
	linDst := float32(toLuma(float64(dst)))
	adjusted := contrast * linDst
	if math.Abs(float64(src-dst)) < 1.0/256.0 {
		for i := range 256 {
			srca := float32(i)/255 + (1-float32(i)/255)*adjusted*(float32(i)/255)
			table[i] = uint8(math.Floor(float64(255*srca) + 0.5))
		}
		return table
	}
	for i := range 256 {
		raw := float32(i) / 255
		srca := raw + (1-raw)*adjusted*raw
		linOut := linSrc*srca + (1-srca)*linDst
		out := float32(fromLuma(float64(linOut)))
		result := (out - dst) / (src - dst)
		table[i] = uint8(math.Floor(float64(255*result) + 0.5))
	}
	return table
}

// TestMaskGammaLUT pins the LUT construction against the reference transcription and the structural properties the
// blitters rely on.
func TestMaskGammaLUT(t *testing.T) {
	maskGammaOnce.Do(maskGammaInit)
	for i := uint32(0); i < maskGammaNumTables; i++ {
		want := referenceLUT(scale255Lum3(i), maskGammaContrast)
		for v := range 256 {
			if maskGammaTables[i][v] != want[v] {
				t.Fatalf("table %d entry %d = %d, want %d", i, v, maskGammaTables[i][v], want[v])
			}
		}
		// Endpoints: zero coverage stays zero, full coverage stays full.
		if maskGammaTables[i][0] != 0 || maskGammaTables[i][255] != 255 {
			t.Errorf("table %d endpoints = %d, %d", i, maskGammaTables[i][0], maskGammaTables[i][255])
		}
	}
	// Each table is monotone non-decreasing (a coverage remapping), and the black-src table (black text on the white
	// dst guess) lightens midtones: blending in the sRGB-encoded domain over-darkens black text, so the correcting LUT
	// reduces mid coverage even after the contrast boost.
	for i := range maskGammaTables {
		for v := 1; v < 256; v++ {
			if maskGammaTables[i][v] < maskGammaTables[i][v-1] {
				t.Fatalf("table %d not monotone at %d", i, v)
			}
		}
	}
	black := &maskGammaTables[0]
	if black[128] >= 128 {
		t.Errorf("black-src LUT at 128 = %d, want < 128", black[128])
	}
	// preBlend keys on the top 3 bits of each channel.
	pb := getMaskPreBlend(colorcore.RGB(0xFF, 0x00, 0x91))
	if !pb.isApplicable() {
		t.Fatal("preblend not applicable")
	}
	if pb.r != &maskGammaTables[7] || pb.g != &maskGammaTables[0] || pb.b != &maskGammaTables[4] {
		t.Error("preblend selected wrong tables")
	}
}

// TestPack4xHToMaskImpulse pins the FIR filter on impulse and step patterns, hand-computed from pack4xHToMask's
// coefficient table.
func TestPack4xHToMaskImpulse(t *testing.T) {
	// One mask row of width 4 needs (4-2)*4 = 8 samples. Impulse at sample 0 with value 255.
	g := &Glyph{Width: 4, Height: 1, Format: MaskLCD16}
	g.Image16 = make([]uint16, 4)
	src := make([]uint8, 8)
	src[0] = 255
	var noBlend maskPreBlend
	pack4xHToMask(src, 8, 1, g, &noBlend, false, false)

	// Output pixel p covers samples [4p-8, 4p+4) of the padded stream (sample_x = -4 + 4p... the loop starts at -4),
	// with coefficient index sample - (sample_x - 4). For sample 0 (value 255):
	//   p=0: sample_x=-4, coeff_index = 0 - (-8) = 8 → coeffs {r,g,b} = {0x05, 0x16, 0x33}
	//   p=1: sample_x= 0, coeff_index = 4          → {0x40, 0x2b, 0x10}
	//   p=2: sample_x= 4, coeff_index = 0          → {0x03, 0x00, 0x00}
	//   p=3: sample_x= 8, out of reach             → 0
	expect := func(rc, gc, bc uint32) uint16 {
		r := min(rc*255/0x100, 255)
		gg := min(gc*255/0x100, 255)
		b := min(bc*255/0x100, 255)
		return pack888ToRGB16(r, gg, b)
	}
	want := []uint16{expect(0x05, 0x16, 0x33), expect(0x40, 0x2b, 0x10), expect(0x03, 0x00, 0x00), 0}
	for i, w := range want {
		if g.Image16[i] != w {
			t.Errorf("impulse pixel %d = %#04x, want %#04x", i, g.Image16[i], w)
		}
	}

	// Step response: all samples 255 → each 12-tap kernel sums to 0x110 ("adding up to more than one simulates ink
	// spread"), so pixels whose kernels are fully inside the samples saturate to 0xFFFF; the outset edge pixels see
	// only part of each kernel. For width 7 the samples span (7-2)*4 = 20 and the fully-covered pixels are 2..4.
	g2 := &Glyph{Width: 7, Height: 1, Format: MaskLCD16}
	g2.Image16 = make([]uint16, 7)
	full := make([]uint8, 20)
	for i := range full {
		full[i] = 255
	}
	pack4xHToMask(full, 20, 1, g2, &noBlend, false, false)
	for i := 2; i <= 4; i++ {
		if g2.Image16[i] != 0xFFFF {
			t.Errorf("step interior pixel %d = %#04x, want 0xFFFF", i, g2.Image16[i])
		}
	}
	if g2.Image16[0] == 0 || g2.Image16[0] == 0xFFFF {
		t.Errorf("step edge pixel = %#04x, want partial coverage", g2.Image16[0])
	}

	// doBGR swaps the channel order: red and blue swap.
	g3 := &Glyph{Width: 4, Height: 1, Format: MaskLCD16}
	g3.Image16 = make([]uint16, 4)
	pack4xHToMask(src, 8, 1, g3, &noBlend, true, false)
	for i := range g.Image16 {
		r0 := g.Image16[i] >> 11 & 0x1F
		b0 := g.Image16[i] & 0x1F
		r1 := g3.Image16[i] >> 11 & 0x1F
		b1 := g3.Image16[i] & 0x1F
		if r0 != b1 || b0 != r1 || g.Image16[i]>>5&0x3F != g3.Image16[i]>>5&0x3F {
			t.Errorf("BGR pixel %d: rgb=%#04x bgr=%#04x not channel-swapped", i, g.Image16[i], g3.Image16[i])
		}
	}

	// toA8: the same filter averaged across channels.
	g4 := &Glyph{Width: 4, Height: 1, Format: MaskA8}
	g4.Image = make([]uint8, 4)
	pack4xHToMask(src, 8, 1, g4, &noBlend, false, false)
	wantA8 := []uint8{
		(min(0x05*255/0x100, 255) + min(0x16*255/0x100, 255) + min(0x33*255/0x100, 255)) / 3,
		(min(0x40*255/0x100, 255) + min(0x2b*255/0x100, 255) + min(0x10*255/0x100, 255)) / 3,
		(min(0x03*255/0x100, 255) + 0 + 0) / 3,
		0,
	}
	for i, w := range wantA8 {
		if g4.Image[i] != w {
			t.Errorf("toA8 pixel %d = %d, want %d", i, g4.Image[i], w)
		}
	}

	// doVert transposes: a 1-column mask (height 4) fed the same samples lands down the column.
	g5 := &Glyph{Width: 1, Height: 4, Format: MaskLCD16}
	g5.Image16 = make([]uint16, 4)
	pack4xHToMask(src, 8, 1, g5, &noBlend, false, true)
	for i := range want {
		if g5.Image16[i] != want[i] {
			t.Errorf("vertical pixel %d = %#04x, want %#04x", i, g5.Image16[i], want[i])
		}
	}
}

// TestLCDGlyphImageLane exercises the full scaler LCD lane on a real glyph: format, the ±1 horizontal bounds outset,
// real subpixel fringes, and the pre-blend keying on the luminance color.
func TestLCDGlyphImageLane(t *testing.T) {
	f := lcdTestFont(t, 24)
	identity := geom.IdentityMatrix()
	rgbh := &DeviceProps{PixelGeometry: PixelGeometryRGBH}

	lcdSpec := MakeMaskSpec(f, &ScalerPaint{LumColor: colorcore.RGB(0, 0, 0)}, &identity, rgbh)
	lcdStrike := lcdSpec.FindOrCreateStrike()
	gid := f.Typeface().UnicharToGlyph('H')
	lcdGlyph, action := lcdStrike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept {
		t.Fatalf("LCD glyph action = %d, want accept", action)
	}
	if lcdGlyph.Format != MaskLCD16 || lcdGlyph.Image16 == nil {
		t.Fatalf("LCD glyph format = %d image16 = %v", lcdGlyph.Format, lcdGlyph.Image16 != nil)
	}

	aaFont := lcdTestFont(t, 24)
	aaFont.SetEdging(EdgingAntiAlias)
	aaSpec := MakeMaskSpec(aaFont, nil, &identity, rgbh)
	aaStrike := aaSpec.FindOrCreateStrike()
	aaGlyph, _ := aaStrike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if aaGlyph.Format != MaskA8 {
		t.Fatalf("A8 glyph format = %d", aaGlyph.Format)
	}

	// updateGlyphBoundsIfLCD: one extra pixel on each side horizontally.
	if lcdGlyph.Left != aaGlyph.Left-1 || lcdGlyph.Width != aaGlyph.Width+2 {
		t.Errorf("LCD bounds left/width = %d/%d, A8 = %d/%d (want ±1 outset)",
			lcdGlyph.Left, lcdGlyph.Width, aaGlyph.Left, aaGlyph.Width)
	}
	if lcdGlyph.Top != aaGlyph.Top || lcdGlyph.Height != aaGlyph.Height {
		t.Errorf("LCD vertical bounds differ from A8")
	}

	// The mask must carry actual subpixel fringes (channel inequality at glyph edges).
	fringe := 0
	for _, w := range lcdGlyph.Image16 {
		r := w >> 11 & 0x1F
		g := w >> 6 & 0x1F // compare at 5 bits
		b := w & 0x1F
		if r != g || g != b {
			fringe++
		}
	}
	if fringe == 0 {
		t.Error("LCD16 mask has no channel fringing")
	}

	// The pre-blend keys the strike on the canonical luminance color: white text gets a different strike and different
	// mask bytes than black text.
	whiteSpec := MakeMaskSpec(f, &ScalerPaint{LumColor: colorcore.RGB(255, 255, 255)}, &identity, rgbh)
	whiteStrike := whiteSpec.FindOrCreateStrike()
	if whiteStrike == lcdStrike {
		t.Fatal("white and black LCD strikes should differ (LumBits in the key)")
	}
	whiteGlyph, _ := whiteStrike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	same := true
	for i := range whiteGlyph.Image16 {
		if whiteGlyph.Image16[i] != lcdGlyph.Image16[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("white and black pre-blends produced identical LCD masks")
	}

	// A8 strikes stay color-independent (LumBits zero for non-LCD recs).
	aaSpec2 := MakeMaskSpec(aaFont, &ScalerPaint{LumColor: colorcore.RGB(255, 255, 255)}, &identity, rgbh)
	if aaSpec2.Rec != aaSpec.Rec {
		t.Error("A8 recs fragment on luminance color")
	}
}

// TestLCDVerticalGlyph checks the vertical-geometry transpose: the outset moves to the vertical axis.
func TestLCDVerticalGlyph(t *testing.T) {
	f := lcdTestFont(t, 24)
	identity := geom.IdentityMatrix()
	gid := f.Typeface().UnicharToGlyph('H')

	vSpec := MakeMaskSpec(f, nil, &identity, &DeviceProps{PixelGeometry: PixelGeometryRGBV})
	vStrike := vSpec.FindOrCreateStrike()
	vGlyph, action := vStrike.DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))
	if action != GlyphActionAccept || vGlyph.Format != MaskLCD16 {
		t.Fatalf("vertical LCD glyph: action %d format %d", action, vGlyph.Format)
	}

	aaFont := lcdTestFont(t, 24)
	aaFont.SetEdging(EdgingAntiAlias)
	aaSpec := MakeMaskSpec(aaFont, nil, &identity, nil)
	aaGlyph, _ := aaSpec.FindOrCreateStrike().DigestFor(ActionDirectMaskCPU, PackGlyphID(gid))

	if vGlyph.Top != aaGlyph.Top-1 || vGlyph.Height != aaGlyph.Height+2 {
		t.Errorf("vertical LCD bounds top/height = %d/%d, A8 = %d/%d (want ±1 outset)",
			vGlyph.Top, vGlyph.Height, aaGlyph.Top, aaGlyph.Height)
	}
	if vGlyph.Left != aaGlyph.Left || vGlyph.Width != aaGlyph.Width {
		t.Errorf("vertical LCD horizontal bounds differ from A8")
	}
	fringe := 0
	for _, w := range vGlyph.Image16 {
		if w>>11&0x1F != w&0x1F {
			fringe++
		}
	}
	if fringe == 0 {
		t.Error("vertical LCD16 mask has no fringing")
	}
}
