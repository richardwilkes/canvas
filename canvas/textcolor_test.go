// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Canvas integration tests for color emoji: the ARGB32 sprite lane in paintMasks, the clip lanes, and the
// rescaled-bitmap third stage. The test fonts render U+1F600 as a smiley whose opaque face center is (255, 204, 0).

package canvas

import (
	"os"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/raster"
)

func loadEmojiFont(t *testing.T, name string, size float32) *font.Font {
	t.Helper()
	data, err := os.ReadFile("../font/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	tf, err := font.NewTypefaceFromData(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	return font.NewFont(tf, size, 1, 0)
}

// pixelAt returns the RGBA bytes at (x, y).
func pixelAt(pix *raster.Pixmap, x, y int32) [4]byte {
	buf := pix.RGBA8888Bytes()
	off := (y*pix.Width + x) * 4
	return [4]byte{buf[off], buf[off+1], buf[off+2], buf[off+3]}
}

const emojiUTF8 = "\U0001F600"

func TestDrawColorEmojiSprite(t *testing.T) {
	for _, name := range []string{"sbix.ttf", "cbdt.ttf", "colr.ttf"} {
		f := loadEmojiFont(t, name, 64)
		c, pix := newTextCanvas(80, 80)
		paint := NewPaint()
		paint.AntiAlias = true
		c.DrawSimpleText([]byte(emojiUTF8), font.TextEncodingUTF8, 10, 66, f, paint)

		ink := inkBounds(pix)
		if ink.IsEmpty() {
			t.Fatalf("%s: no ink", name)
		}
		// The glyph occupies roughly [10, 66-52 .. 62, 66] (colr's shape is inset within its em box).
		if ink.Left < 9 || ink.Top < 12 || ink.Right > 64 || ink.Bottom > 67 {
			t.Errorf("%s: ink %v outside expected box", name, ink)
		}
		// The face center is the opaque (255, 204, 0) color, not the black an A8 mask would produce.
		center := pixelAt(pix, (ink.Left+ink.Right)/2, (ink.Top+ink.Bottom)/2)
		if center != [4]byte{255, 204, 0, 255} {
			t.Errorf("%s: center %v, want {255 204 0 255}", name, center)
		}
	}
}

func TestDrawColorEmojiScaledCTM(t *testing.T) {
	f := loadEmojiFont(t, "sbix.ttf", 64)
	c, pix := newTextCanvas(80, 80)
	c.Scale(0.5, 0.5)
	paint := NewPaint()
	c.DrawSimpleText([]byte(emojiUTF8), font.TextEncodingUTF8, 20, 132, f, paint)

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	// Device-space glyph is ~26px square at (10, 40)..(36, 66).
	if ink.Width() < 20 || ink.Width() > 30 || ink.Height() < 20 || ink.Height() > 30 {
		t.Errorf("ink %v, want ~26px square", ink)
	}
	center := pixelAt(pix, (ink.Left+ink.Right)/2, (ink.Top+ink.Bottom)/2)
	if center[0] < 200 || center[1] < 150 || center[2] > 80 {
		t.Errorf("center %v, want the face color family", center)
	}
}

func TestDrawColorEmojiPerspective(t *testing.T) {
	// A perspective CTM skips the direct-mask stage entirely; the rescaled-bitmap stage must draw.
	f := loadEmojiFont(t, "cbdt.ttf", 32)
	c, pix := newTextCanvas(80, 80)
	var m geom.Matrix
	m.SetAll(1, 0, 4, 0, 1, 30, 0.0004, 0, 1)
	c.Concat(&m)
	paint := NewPaint()
	c.DrawSimpleText([]byte(emojiUTF8), font.TextEncodingUTF8, 10, 30, f, paint)

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink through the perspective (rescaled bitmap) stage")
	}
	center := pixelAt(pix, (ink.Left+ink.Right)/2, (ink.Top+ink.Bottom)/2)
	if center[0] < 180 || center[2] > 100 {
		t.Errorf("center %v, want the face color family", center)
	}
}

func TestDrawColorEmojiRegionClip(t *testing.T) {
	// A BW multi-rect clip drives paintMasks' region lane; the sprite must still draw, clipped.
	f := loadEmojiFont(t, "sbix.ttf", 64)
	c, pix := newTextCanvas(80, 80)
	c.ClipRect(geom.RectLTRB(0, 0, 30, 80), raster.ClipIntersect, false)
	inner := geom.RectLTRB(20, 0, 25, 80)
	c.ClipRect(inner, raster.ClipDifference, false)
	paint := NewPaint()
	c.DrawSimpleText([]byte(emojiUTF8), font.TextEncodingUTF8, 10, 66, f, paint)

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	if ink.Right > 30 {
		t.Errorf("ink %v leaks past the clip at x=30", ink)
	}
	// The difference band [20, 25) stays white.
	for y := int32(20); y < 60; y++ {
		if p := pixelAt(pix, 22, y); p != [4]byte{255, 255, 255, 255} {
			t.Fatalf("pixel inside the difference band drawn: %v at y=%d", p, y)
		}
	}
}

func TestDrawColorEmojiBlur(t *testing.T) {
	// Blur mask filters flip color glyphs to blurred A8 silhouettes drawn with the paint color.
	f := loadEmojiFont(t, "sbix.ttf", 48)
	c, pix := newTextCanvas(80, 80)
	paint := NewPaint()
	paint.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
	paint.Color = colorcore.ARGB(0xFF, 0, 0, 0xFF) // blue
	c.DrawSimpleText([]byte(emojiUTF8), font.TextEncodingUTF8, 16, 60, f, paint)

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	center := pixelAt(pix, (ink.Left+ink.Right)/2, (ink.Top+ink.Bottom)/2)
	if center[2] != 255 || center[0] != 0 {
		t.Errorf("center %v, want pure paint blue (A8 silhouette)", center)
	}
	// No face-colored pixels anywhere.
	buf := pix.RGBA8888Bytes()
	for i := 0; i < len(buf); i += 4 {
		if buf[i] == 255 && buf[i+1] == 204 && buf[i+2] == 0 {
			t.Fatal("found face-colored pixel; blur must not draw the color image")
		}
	}
}

func TestDrawColorEmojiPaintAlpha(t *testing.T) {
	// drawSprite applies the paint alpha to color glyphs.
	f := loadEmojiFont(t, "sbix.ttf", 64)
	c, pix := newTextCanvas(80, 80)
	paint := NewPaint()
	paint.Color = paint.Color.WithAlpha(128)
	c.DrawSimpleText([]byte(emojiUTF8), font.TextEncodingUTF8, 10, 66, f, paint)

	ink := inkBounds(pix)
	if ink.IsEmpty() {
		t.Fatal("no ink")
	}
	center := pixelAt(pix, (ink.Left+ink.Right)/2, (ink.Top+ink.Bottom)/2)
	// Half of (255, 204, 0) over white: roughly (255, 229, 127).
	if center[0] != 255 || center[1] < 220 || center[1] > 240 || center[2] < 115 || center[2] > 140 {
		t.Errorf("center %v, want half-alpha face over white", center)
	}
}
