// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live tests migrated from the retired façade suite: the wrapped-FBO bottom-left-origin lanes the façade's gpu_live
// tests exercised, now driven through the underlying API exactly as unison drives it. The oracle's wrapped-FBO golden
// lane covers the top-left wrapped-FBO surface path; these cover the bottom-left origin flip against an asymmetric
// source image, where a wrong flip diverges.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

// migratedGradImage builds a w×h opaque RGBA_8888 image whose red varies with x and green with y, so a
// vertical/horizontal flip or an x/y transpose in the view mapping diverges instead of matching spuriously.
func migratedGradImage(w, h int32) *imagecore.Image {
	info, _ := imagecore.MakeInfo(w, h, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	p := imagecore.NewPixels(info)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			r := uint32(20 + x*30)
			g := uint32(10 + y*30)
			p.Words[y*p.RowElems+x] = r | g<<8 | 0x80<<16 | 0xFF<<24
		}
	}
	return imagecore.FromPixels(p)
}

// readRTSurface reads a wrapped surface back as tightly packed premul RGBA bytes.
func readRTSurface(t *testing.T, rt *gl.RenderTargetSurface, size int32) []byte {
	t.Helper()
	info, _ := imagecore.MakeInfo(size, size, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	buf := make([]byte, int(size)*int(size)*4)
	if !rt.ReadPixels(info, buf, int(size)*4, 0, 0) {
		t.Fatal("RenderTargetSurface.ReadPixels failed")
	}
	return buf
}

// imagePixels reads a CPU image's premul RGBA bytes.
func imagePixels(t *testing.T, img *imagecore.Image) []byte {
	t.Helper()
	px := img.PeekPixels(imagecore.CachingAllow)
	if px == nil {
		t.Fatal("PeekPixels failed")
	}
	out := make([]byte, 0, len(px.Words)*4)
	for y := int32(0); y < px.Info.Height; y++ {
		for x := int32(0); x < px.Info.Width; x++ {
			w := px.Words[y*px.RowElems+x]
			out = append(out, byte(w), byte(w>>8), byte(w>>16), byte(w>>24))
		}
	}
	return out
}

func assertBytesClose(t *testing.T, got, want []byte, tol int, tag string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", tag, len(got), len(want))
	}
	for i := range got {
		if d := int(got[i]) - int(want[i]); d < -tol || d > tol {
			t.Fatalf("%s: byte %d (pixel %d ch %d) = %d, want %d±%d", tag, i, i/4, i%4, got[i], want[i], tol)
		}
	}
}

// TestLiveWrappedFBOBottomLeftDrawLanes drives the wrapped-FBO surface path at bottom-left origin (unison's window
// origin) through real draws:
//   - an uploaded texture image drawn onto the surface reproduces the asymmetric source (the program's rt-adjust flip
//     and the texture op's bottom-left normalization);
//   - a snapshot of that surface (a bottom-left-origin texture image) used as an image shader reproduces it again (the
//     texture effect's origin-correcting coord matrix);
//   - a compatible surface (makeSurface) derives from the wrapped target;
//   - after ResetContext and ResetGLTextureBindings the context stays usable for another draw.
func TestLiveWrappedFBOBottomLeftDrawLanes(t *testing.T) {
	env, dc := newLiveDirectContext(t)
	const size = int32(8)
	src := migratedGradImage(size, size)
	want := imagePixels(t, src)
	full := geom.RectWH(float32(size), float32(size))

	makeSurface := func() *gl.RenderTargetSurface {
		t.Helper()
		fbo, _ := env.makeOffscreenFBO(t, size)
		dc.ResetContext(gl.AllBackendState) // the FBO was created behind the context's back
		rt := gl.NewRenderTargetSurfaceFromBackendRenderTarget(dc, gpu.ColorTypeRGBA8888,
			geom.ISize{Width: size, Height: size}, gl.FormatFromEnum(gl.RGBA8), 1, 0, fbo,
			gpu.OriginBottomLeft, nil)
		if rt == nil {
			t.Fatal("NewRenderTargetSurfaceFromBackendRenderTarget returned nil")
		}
		return rt
	}

	// Part A: native draw of an uploaded texture image onto the bottom-left surface. The green clear makes a coverage
	// gap fail loudly (the gradient is opaque, so src-over fully replaces it).
	tex := gl.TextureFromImage(dc, src, gpu.MipmappedNo, gpu.BudgetedYes)
	if tex == nil {
		t.Fatal("TextureFromImage returned nil")
	}
	rtA := makeSurface()
	cA := rtA.Canvas()
	cA.Clear(0xFF00FF00)
	cA.DrawImageRect(tex, full, full, shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	dc.FlushAndSubmit(true)
	assertBytesClose(t, readRTSurface(t, rtA, size), want, 2, "texture draw (bottom-left origin)")

	// Part B: the surface snapshot is a bottom-left-origin texture image; shading a rect with it must origin-correct
	// through the texture effect.
	snap := rtA.MakeImageSnapshot()
	if snap == nil {
		t.Fatal("MakeImageSnapshot returned nil")
	}
	shader := shaders.NewImageDrawable(snap, shaders.TileClamp, shaders.TileClamp,
		shaders.SamplingOptions{}, nil)
	if shader == nil {
		t.Fatal("NewImageDrawable returned nil")
	}
	rtB := makeSurface()
	cB := rtB.Canvas()
	cB.Clear(0xFF00FF00)
	paint := canvas.NewPaint()
	paint.Color = 0xFFFFFFFF
	paint.Shader = shader
	cB.DrawRect(full, paint)
	dc.FlushAndSubmit(true)
	assertBytesClose(t, readRTSurface(t, rtB, size), want, 2, "snapshot shader draw (bottom-left origin)")

	// Part B2: the snapshot drawn as an image (not a shader) routes through the texture op's bottom-left source
	// normalization; it must land upright as well.
	rtB2 := makeSurface()
	cB2 := rtB2.Canvas()
	cB2.Clear(0xFF00FF00)
	cB2.DrawImageRect(snap, full, full, shaders.SamplingOptions{}, nil, canvas.ConstraintFast)
	dc.FlushAndSubmit(true)
	assertBytesClose(t, readRTSurface(t, rtB2, size), want, 2, "snapshot image draw (bottom-left origin)")
	rtB2.SDC().Release()

	// Part C: a compatible surface derives from the wrapped target with the same color type and origin.
	compat := rtB.MakeSurface(4, 6)
	if compat == nil {
		t.Fatal("MakeSurface returned nil")
	}
	if compat.Width() != 4 || compat.Height() != 6 {
		t.Errorf("compatible surface is %dx%d, want 4x6", compat.Width(), compat.Height())
	}
	compat.SDC().Release()

	// Part D: the state-reset lanes must leave the context usable — a following draw still lands.
	dc.ResetContext(gl.AllBackendState)
	dc.ResetGLTextureBindings()
	cB.Clear(0xFF0000FF)
	dc.FlushAndSubmit(false)
	post := readRTSurface(t, rtB, size)
	for i := 0; i+4 <= len(post); i += 4 {
		if post[i] != 0 || post[i+1] != 0 || post[i+2] != 255 || post[i+3] != 255 {
			t.Fatalf("post-reset clear pixel %d = %v, want opaque blue", i/4, post[i:i+4])
		}
	}
	sweepGLErrors(t, dc.Gpu(), "bottom-left wrapped-FBO lanes")

	rtA.SDC().Release()
	rtB.SDC().Release()
}
