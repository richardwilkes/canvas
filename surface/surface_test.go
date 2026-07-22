// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package surface

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

func fillRect(c *canvas.Canvas, r geom.Rect, color colorcore.Color) {
	p := canvas.NewPaint()
	p.Color = color
	c.DrawRect(r, p)
}

// TestSurfaceCanvasDrawsIntoBackingStore verifies the canvas↔surface wiring.
func TestSurfaceCanvasDrawsIntoBackingStore(t *testing.T) {
	s := NewRasterN32Premul(20, 10, nil)
	if s == nil {
		t.Fatal("surface creation failed")
	}
	s.Canvas().Clear(0xFF00FF00)
	if s.Pixmap().Pix[0] == 0 {
		t.Fatalf("canvas draw did not reach the surface pixels")
	}
	if s.Canvas().SurfaceBaseValue() != canvas.SurfaceBase(s) {
		t.Fatalf("canvas surface backref not installed")
	}
}

// TestSurfaceInvalidSizes verifies nil returns for degenerate dimensions.
func TestSurfaceInvalidSizes(t *testing.T) {
	if NewRasterN32Premul(0, 10, nil) != nil || NewRasterN32Premul(10, -1, nil) != nil {
		t.Fatalf("degenerate surface sizes should return nil")
	}
	if WrapPixels(nil, nil) != nil {
		t.Fatalf("nil pixmap should return nil")
	}
}

// TestWrapPixelsStrideValidation verifies WrapPixels honors RowPixels when validating the backing slice, rejecting
// pixmaps whose stride would make rendering read or write past the buffer.
func TestWrapPixelsStrideValidation(t *testing.T) {
	// A Width*Height-sized buffer with RowPixels > Width would overflow at y*RowPixels+x; it must be rejected even
	// though len(Pix) >= Width*Height.
	understrided := &raster.Pixmap{Pix: make([]uint32, 8*8), Width: 8, Height: 8, RowPixels: 16}
	if WrapPixels(understrided, nil) != nil {
		t.Fatalf("pixmap with RowPixels > Width and a Width*Height buffer must be rejected")
	}

	// One word short of the strided requirement ((Height-1)*RowPixels+Width) must still be rejected.
	const w, h, stride = 8, 8, 16
	short := &raster.Pixmap{Pix: make([]uint32, (h-1)*stride+w-1), Width: w, Height: h, RowPixels: stride}
	if WrapPixels(short, nil) != nil {
		t.Fatalf("pixmap one word short of the strided size must be rejected")
	}

	// Exactly the strided requirement must be accepted, and drawing must reach the last row without overflowing.
	ok := &raster.Pixmap{Pix: make([]uint32, (h-1)*stride+w), Width: w, Height: h, RowPixels: stride}
	s := WrapPixels(ok, nil)
	if s == nil {
		t.Fatalf("pixmap sized exactly for the strided layout must be accepted")
	}
	s.Canvas().Clear(0xFF00FF00)
	if ok.Pix[(h-1)*stride+w-1] == 0 {
		t.Fatalf("clear did not reach the last strided pixel")
	}

	// RowPixels < Width is a malformed stride and must be rejected.
	narrow := &raster.Pixmap{Pix: make([]uint32, 8*8), Width: 8, Height: 8, RowPixels: 4}
	if WrapPixels(narrow, nil) != nil {
		t.Fatalf("pixmap with RowPixels < Width must be rejected")
	}
}

// TestSnapshotCopyOnWrite verifies the surface's snapshot semantics: the image's pixels are frozen at snapshot time;
// the surface keeps its content and mutates a copy.
func TestSnapshotCopyOnWrite(t *testing.T) {
	s := NewRasterN32Premul(16, 16, nil)
	s.Canvas().Clear(0xFFFF0000)

	img := s.MakeImageSnapshot()
	if img.Width() != 16 || img.Height() != 16 {
		t.Fatalf("snapshot dimensions wrong")
	}
	before := img.Pixmap().Pix[0]

	// Same-state second snapshot shares the image (the cached snapshot is returned).
	if s.MakeImageSnapshot() != img {
		t.Fatalf("second snapshot without mutation should be the same image")
	}

	// Mutate the surface: image must be unaffected, surface must retain prior content plus the draw.
	fillRect(s.Canvas(), geom.RectLTRB(0, 0, 16, 8), 0xFF0000FF)
	if img.Pixmap().Pix[0] != before {
		t.Fatalf("snapshot pixels changed after surface draw")
	}
	if s.Pixmap().Pix[0] == before {
		t.Fatalf("surface pixels unchanged after draw")
	}
	// Bottom half retained the pre-draw content (kRetain copy-on-write).
	bottom := s.Pixmap().Pix[15*16]
	if bottom != before {
		t.Fatalf("surface lost retained content: %08x vs %08x", bottom, before)
	}

	// A new snapshot now differs from the first.
	img2 := s.MakeImageSnapshot()
	if img2 == img {
		t.Fatalf("post-mutation snapshot should be a new image")
	}
}

// TestWrapPixelsSnapshotCopies verifies raster-direct surfaces deep-copy at snapshot time and keep drawing into the
// caller's memory.
func TestWrapPixelsSnapshotCopies(t *testing.T) {
	pix := raster.NewPixmap(8, 8)
	s := WrapPixels(pix, nil)
	if s == nil {
		t.Fatal("WrapPixels failed")
	}
	s.Canvas().Clear(0xFF123456)
	if pix.Pix[0] == 0 {
		t.Fatalf("raster-direct surface did not draw into caller memory")
	}

	img := s.MakeImageSnapshot()
	was := img.Pixmap().Pix[0]
	s.Canvas().Clear(0xFF654321)
	if img.Pixmap().Pix[0] != was {
		t.Fatalf("raster-direct snapshot not isolated from later draws")
	}
	if &pix.Pix[0] != &s.Pixmap().Pix[0] {
		t.Fatalf("raster-direct surface abandoned the caller's memory")
	}
}

// TestMakeSurfaceCompatible verifies the compatible-surface factory.
func TestMakeSurfaceCompatible(t *testing.T) {
	props := Props{Flags: 1, PixelGeometry: PixelGeometryRGBH}
	s := NewRasterN32Premul(10, 10, &props)
	s2 := s.MakeSurface(4, 6)
	if s2 == nil || s2.Width() != 4 || s2.Height() != 6 {
		t.Fatalf("compatible surface has wrong size")
	}
	if s2.Props() != props {
		t.Fatalf("compatible surface did not inherit props")
	}
}
