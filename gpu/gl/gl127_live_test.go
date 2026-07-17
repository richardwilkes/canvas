// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for GPU images & surfaces: TextureFromImage upload + readback round trips, MakeNonTextureImage,
// the RenderTargetSurface snapshot/compatible-surface/ readback lanes, and the snapshot-is-frozen (copy-on-snapshot)
// guarantee. Skips when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
)

// rgba8888Premul builds the imagecore info for a w×h RGBA8888-premul readback buffer.
func rgba8888Premul(w, h int32) imagecore.ImageInfo {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	return info
}

// sourceRGBA reads an imagecore image out as RGBA8888-premul bytes (the round-trip reference).
func sourceRGBA(t *testing.T, img *imagecore.Image) []byte {
	t.Helper()
	buf := make([]byte, int(img.Width())*int(img.Height())*4)
	if !img.ReadPixels(rgba8888Premul(img.Width(), img.Height()), buf, int(img.Width())*4, 0, 0,
		imagecore.CachingAllow) {
		t.Fatal("source ReadPixels failed")
	}
	return buf
}

func assertBytesEqual(t *testing.T, got, want []byte, tol int, tag string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", tag, len(got), len(want))
	}
	for i := range got {
		d := int(got[i]) - int(want[i])
		if d < -tol || d > tol {
			t.Fatalf("%s: byte %d = %d, want %d (tol %d)", tag, i, got[i], want[i], tol)
		}
	}
}

// assertUniformRGBA checks every pixel of an RGBA8888 buffer equals want within tol.
func assertUniformRGBA(t *testing.T, data []byte, want [4]byte, tol int, tag string) {
	t.Helper()
	for off := 0; off+4 <= len(data); off += 4 {
		for c := 0; c < 4; c++ {
			d := int(data[off+c]) - int(want[c])
			if d < -tol || d > tol {
				t.Fatalf("%s: pixel byte %d = %d, want %d", tag, off+c, data[off+c], want[c])
			}
		}
	}
}

func readTexImageRGBA(t *testing.T, tex *gl.TextureImage) []byte {
	t.Helper()
	buf := make([]byte, int(tex.Width())*int(tex.Height())*4)
	if !tex.ReadPixels(rgba8888Premul(tex.Width(), tex.Height()), buf, int(tex.Width())*4, 0, 0,
		imagecore.CachingAllow) {
		t.Fatal("TextureImage.ReadPixels failed")
	}
	return buf
}

func readSurfaceRGBA(t *testing.T, s *gl.RenderTargetSurface) []byte {
	t.Helper()
	buf := make([]byte, int(s.Width())*int(s.Height())*4)
	if !s.ReadPixels(rgba8888Premul(s.Width(), s.Height()), buf, int(s.Width())*4, 0, 0) {
		t.Fatal("RenderTargetSurface.ReadPixels failed")
	}
	return buf
}

func TestLiveTextureFromImageReadback(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	img := newCheckerImage(t) // 4x4 red/white non-uniform pattern (orientation-sensitive)
	tex := gl.TextureFromImage(dc, img, gpu.MipmappedNo, gpu.BudgetedYes)
	if tex == nil {
		t.Fatal("TextureFromImage returned nil")
	}
	defer tex.Release()
	if tex.Width() != 4 || tex.Height() != 4 {
		t.Fatalf("dims = %dx%d, want 4x4", tex.Width(), tex.Height())
	}
	if !tex.IsTextureBacked() || tex.IsAlphaOnly() {
		t.Fatal("expected an opaque texture-backed image")
	}
	if tex.ColorType() != gpu.ColorTypeRGBA8888 {
		t.Fatalf("color type = %v, want RGBA8888", tex.ColorType())
	}

	want := sourceRGBA(t, img)
	// Round trip: source -> texture -> readback must be byte-exact (opaque, top-left, no flip).
	assertBytesEqual(t, readTexImageRGBA(t, tex), want, 0, "texture readback")

	// MakeNonTextureImage reads back into a CPU image; its pixels must match the source too.
	nonTex := tex.MakeNonTextureImage()
	if nonTex == nil {
		t.Fatal("MakeNonTextureImage returned nil")
	}
	assertBytesEqual(t, sourceRGBA(t, nonTex), want, 0, "non-texture image")
}

func TestLiveTextureFromImageAlpha8(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	info, _ := imagecore.MakeInfo(4, 4, imagecore.ColorTypeAlpha8, imagecore.AlphaTypePremul)
	data := make([]byte, 16)
	for i := range data {
		data[i] = byte(i * 15)
	}
	img := imagecore.NewRasterData(info, data, 4)
	if img == nil {
		t.Fatal("alpha image creation failed")
	}

	tex := gl.TextureFromImage(dc, img, gpu.MipmappedNo, gpu.BudgetedYes)
	if tex == nil {
		t.Fatal("TextureFromImage(alpha) returned nil")
	}
	defer tex.Release()
	if !tex.IsAlphaOnly() || tex.ColorType() != gpu.ColorTypeAlpha8 {
		t.Fatalf("expected an alpha-only image, got color type %v", tex.ColorType())
	}

	got := make([]byte, 16)
	if !tex.ReadPixels(info, got, 4, 0, 0, imagecore.CachingAllow) {
		t.Fatal("alpha ReadPixels failed")
	}
	assertBytesEqual(t, got, data, 0, "alpha readback")
}

func TestLiveTextureFromImageNil(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	if gl.TextureFromImage(dc, nil, gpu.MipmappedNo, gpu.BudgetedYes) != nil {
		t.Error("TextureFromImage(nil image) should be nil")
	}
}

func TestLiveRenderTargetSurfaceSnapshot(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	const w, h = 16, 16
	s := gl.MakeRenderTargetSurface(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: w, Height: h}, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "snap-test", nil)
	if s == nil {
		t.Fatal("MakeRenderTargetSurface returned nil")
	}
	if s.Width() != w || s.Height() != h {
		t.Fatalf("surface dims = %dx%d", s.Width(), s.Height())
	}

	red := [4]byte{255, 0, 0, 255}
	blue := [4]byte{0, 0, 255, 255}

	// Snapshot the red content, then draw blue over the surface with no intervening read of the snapshot — the snapshot
	// must still be frozen at red (the copy is baked at snapshot time, independent of later draw/read ordering).
	s.Canvas().Clear(colorcore.Red)
	snap1 := s.MakeImageSnapshot()
	if snap1 == nil {
		t.Fatal("MakeImageSnapshot returned nil")
	}
	defer snap1.Release()

	s.Canvas().Clear(colorcore.Blue)
	assertUniformRGBA(t, readSurfaceRGBA(t, s), blue, 1, "surface after blue clear")
	assertUniformRGBA(t, readTexImageRGBA(t, snap1), red, 1, "snapshot1 stays frozen at red")

	snap2 := s.MakeImageSnapshot()
	if snap2 == nil {
		t.Fatal("second MakeImageSnapshot returned nil")
	}
	defer snap2.Release()
	assertUniformRGBA(t, readTexImageRGBA(t, snap2), blue, 1, "snapshot2")
}

func TestLiveRenderTargetSurfaceMakeSurface(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	s := gl.MakeRenderTargetSurface(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 16, Height: 16}, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "compat-parent", nil)
	if s == nil {
		t.Fatal("parent surface nil")
	}
	s.Canvas().Clear(colorcore.Red)

	s2 := s.MakeSurface(8, 8)
	if s2 == nil {
		t.Fatal("MakeSurface returned nil")
	}
	if s2.Width() != 8 || s2.Height() != 8 {
		t.Fatalf("compatible surface dims = %dx%d, want 8x8", s2.Width(), s2.Height())
	}
	s2.Canvas().Clear(colorcore.Green)

	green := [4]byte{0, 255, 0, 255}
	red := [4]byte{255, 0, 0, 255}
	assertUniformRGBA(t, readSurfaceRGBA(t, s2), green, 1, "compatible surface")
	// The compatible surface is independent; the parent is untouched.
	assertUniformRGBA(t, readSurfaceRGBA(t, s), red, 1, "parent after compatible surface drawn")
}

func TestLiveRenderTargetSurfaceReadPixelsConvert(t *testing.T) {
	_, dc := newLiveDirectContext(t)

	s := gl.MakeRenderTargetSurface(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: 8, Height: 8}, 1,
		gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "convert-test", nil)
	if s == nil {
		t.Fatal("surface nil")
	}
	s.Canvas().Clear(colorcore.Red)

	// Read back into BGRA8888: RGBA red [255,0,0,255] becomes BGRA [0,0,255,255].
	bgra, _ := imagecore.MakeInfo(8, 8, imagecore.ColorTypeBGRA8888, imagecore.AlphaTypePremul)
	buf := make([]byte, 8*8*4)
	if !s.ReadPixels(bgra, buf, 8*4, 0, 0) {
		t.Fatal("BGRA ReadPixels failed")
	}
	assertUniformRGBA(t, buf, [4]byte{0, 0, 255, 255}, 1, "BGRA readback")
}

func TestLiveSurfaceFromBackendRenderTarget(t *testing.T) {
	env, dc := newLiveDirectContext(t)

	const size = 16
	fbo, _ := env.makeOffscreenFBO(t, size)

	s := gl.NewRenderTargetSurfaceFromBackendRenderTarget(dc, gpu.ColorTypeRGBA8888,
		geom.ISize{Width: size, Height: size}, gl.FormatRGBA8, 1, 0, fbo, gpu.OriginBottomLeft, nil)
	if s == nil {
		t.Fatal("NewRenderTargetSurfaceFromBackendRenderTarget returned nil")
	}
	s.Canvas().Clear(colorcore.Magenta)

	magenta := [4]byte{255, 0, 255, 255}
	assertUniformRGBA(t, readSurfaceRGBA(t, s), magenta, 1, "wrapped-FBO surface")

	// Snapshot of a wrapped FBO copies the render target into a texture image.
	snap := s.MakeImageSnapshot()
	if snap == nil {
		t.Fatal("wrapped-FBO snapshot returned nil")
	}
	defer snap.Release()
	assertUniformRGBA(t, readTexImageRGBA(t, snap), magenta, 1, "wrapped-FBO snapshot")
}
