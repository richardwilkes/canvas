// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package colorcore

import (
	"math"
	"testing"
)

func TestColorAccessors(t *testing.T) {
	c := ARGB(0x12, 0x34, 0x56, 0x78)
	if c != 0x12345678 {
		t.Fatalf("ARGB packed %#x", uint32(c))
	}
	if c.A() != 0x12 || c.R() != 0x34 || c.G() != 0x56 || c.B() != 0x78 {
		t.Error("accessors wrong")
	}
	if RGB(1, 2, 3) != 0xFF010203 {
		t.Error("RGB should be opaque")
	}
	if c.WithAlpha(0xFF) != 0xFF345678 {
		t.Error("WithAlpha wrong")
	}
}

func TestMulDiv255Round(t *testing.T) {
	// The integer sequence must equal round(a*b/255) exactly for all inputs.
	for a := 0; a <= 255; a++ {
		for b := 0; b <= 255; b++ {
			want := uint8(math.Floor(float64(a)*float64(b)/255 + 0.5))
			if got := MulDiv255Round(uint8(a), uint8(b)); got != want {
				t.Fatalf("MulDiv255Round(%d,%d) = %d, want %d", a, b, got, want)
			}
		}
	}
}

func TestPreMultiply(t *testing.T) {
	// Opaque colors pass through.
	c := ARGB(255, 10, 20, 30)
	if pm := c.PreMultiply(); pm != PMColor(c) {
		t.Errorf("opaque premul changed value: %#x", uint32(pm))
	}
	// Transparent black.
	if pm := ARGB(0, 255, 255, 255).PreMultiply(); pm != PMColorARGB(0, 0, 0, 0) {
		t.Errorf("transparent premul = %#x", uint32(pm))
	}
	// Half alpha rounds to nearest: 255*128/255 = 128 exactly, 128*128/255 rounds to 64.
	pm := ARGB(128, 255, 0, 128).PreMultiply()
	if pm.A() != 128 || pm.R() != 128 || pm.G() != 0 || pm.B() != 64 {
		t.Errorf("half-alpha premul = %#x", uint32(pm))
	}
}

func TestUnpremulScaleMatchesSkiaTable(t *testing.T) {
	// Golden values for the first 24 scale entries (255<<24)/alpha with round-to-nearest division.
	want := []uint32{
		0x00000000, 0xFF000000, 0x7F800000, 0x55000000, 0x3FC00000, 0x33000000, 0x2A800000, 0x246DB6DB,
		0x1FE00000, 0x1C555555, 0x19800000, 0x172E8BA3, 0x15400000, 0x139D89D9, 0x1236DB6E, 0x11000000,
		0x0FF00000, 0x0F000000, 0x0E2AAAAB, 0x0D6BCA1B, 0x0CC00000, 0x0C249249, 0x0B9745D1, 0x0B1642C8,
	}
	for a, w := range want {
		if got := unpremulScale(uint8(a)); got != w {
			t.Fatalf("unpremulScale(%d) = %#08X, want %#08X", a, got, w)
		}
	}
}

func TestPremulUnpremulRoundTrip(t *testing.T) {
	// Unpremultiply(Premultiply(c)) must reproduce the original for every (alpha, component) pair where the component
	// survives quantization exactly; verify the weaker guaranteed property everywhere: re-premultiplying the round-trip
	// result reproduces the premultiplied value.
	for a := 1; a <= 255; a++ {
		for v := 0; v <= 255; v++ {
			pm := PreMultiplyARGB(uint8(a), uint8(v), 0, 0)
			back := pm.UnPreMultiply()
			pm2 := back.PreMultiply()
			if pm2 != pm {
				t.Fatalf("premul(unpremul(premul)) not stable: a=%d v=%d pm=%#x back=%#x pm2=%#x",
					a, v, uint32(pm), uint32(back), uint32(pm2))
			}
		}
	}
	// Alpha 0 unpremultiplies to transparent.
	if PMColorARGB(0, 0, 0, 0).UnPreMultiply() != 0 {
		t.Error("alpha-0 unpremul should be 0")
	}
}

func TestColor4fRoundTrip(t *testing.T) {
	// Byte -> float -> byte must be identity for all 256 values per channel.
	for v := 0; v <= 255; v++ {
		c := ARGB(uint8(v), uint8(v), uint8(255-v), uint8(v/2))
		f := Color4fFromColor(c)
		if got := f.ToColor(); got != c {
			t.Fatalf("Color4f round trip %#x -> %+v -> %#x", uint32(c), f, uint32(got))
		}
	}
	// Out-of-range pins.
	f := Color4f{R: 2, G: -1, B: 0.5, A: 1}
	c := f.ToColor()
	if c.R() != 255 || c.G() != 0 || c.A() != 255 {
		t.Errorf("pinned conversion = %#x", uint32(c))
	}
}

func TestColor4fPremul(t *testing.T) {
	f := Color4f{R: 1, G: 0.5, B: 0.25, A: 0.5}
	pm := f.Premul()
	if pm.R != 0.5 || pm.G != 0.25 || pm.B != 0.125 || pm.A != 0.5 {
		t.Errorf("premul = %+v", pm)
	}
	back := pm.Unpremul()
	if math.Abs(float64(back.R-f.R)) > 1e-6 || math.Abs(float64(back.G-f.G)) > 1e-6 {
		t.Errorf("unpremul = %+v", back)
	}
	if (PMColor4f{}).Unpremul() != (Color4f{}) {
		t.Error("zero-alpha unpremul should be zero")
	}
}

func TestSRGBTransferRoundTrip(t *testing.T) {
	for i := 0; i <= 100; i++ {
		x := float32(i) / 100
		lin := SRGBToLinear(x)
		back := LinearToSRGB(lin)
		if math.Abs(float64(back-x)) > 1e-5 {
			t.Fatalf("sRGB round trip %g -> %g -> %g", x, lin, back)
		}
	}
	// Known anchor values.
	if SRGBToLinear(0) != 0 || SRGBToLinear(1) != 1 {
		t.Error("sRGB endpoints wrong")
	}
	if got := SRGBToLinear(0.5); math.Abs(float64(got)-0.21404114) > 1e-6 {
		t.Errorf("srgbToLinear(0.5) = %g", got)
	}
}
