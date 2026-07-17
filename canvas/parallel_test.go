// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package canvas

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// parallelGradientPaint is a multi-stop linear gradient (the ShaderBlitter lane).
func parallelGradientPaint(aa bool) *Paint {
	sh := shaders.NewLinearGradient(geom.Pt(8, 8), geom.Pt(248, 248),
		[]colorcore.Color{0xFFFF0000, 0xFF00FF00, 0xFF0000FF}, []float32{0, 0.5, 1},
		shaders.TileClamp, nil)
	p := NewPaint()
	p.AntiAlias = aa
	p.Shader = sh
	return p
}

// parallelImagePaint is a clamp/clamp linear-sampled image shader (the LegacyShaderBlitter lane).
func parallelImagePaint(aa bool) *Paint {
	img := benchImage(64, 64)
	sh := shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp,
		shaders.SamplingOptions{Filter: shaders.FilterLinear}, nil)
	p := NewPaint()
	p.AntiAlias = aa
	p.Shader = sh
	return p
}

// assertParallelByteIdentical fills devRect both serially (the whole rect through one blitter, exactly the production
// serial rect-fill lane) and via the concurrent row bands, and requires the two device buffers to be byte-for-byte
// identical — the correctness claim behind routing large shaded rect fills through row bands. It also confirms the
// paint actually takes a bandable shader lane, and that the fill produced output.
func assertParallelByteIdentical(t *testing.T, paint *Paint, devRect geom.Rect, aa bool, workers int) {
	t.Helper()
	const w, h = 256, 256
	ident := geom.IdentityMatrix()
	full := geom.IRectWH(w, h)

	probe := chooseBlitter(raster.NewPixmap(w, h), paint, &ident)
	bandable := isBandableShaderBlitter(probe)
	recycleBlitter(probe)
	if !bandable {
		t.Fatalf("paint did not choose a bandable shader blitter (%T)", probe)
	}

	bounds := raster.RectFillBandBounds(devRect.Top, devRect.Bottom, workers)
	if len(bounds) < 3 {
		t.Fatalf("expected >=2 bands for %v (workers=%d), got %v", devRect, workers, bounds)
	}

	pmSerial := raster.NewPixmap(w, h)
	var rcSerial raster.Clip
	rcSerial.SetRect(full)
	fillRectShadedBand(pmSerial, &rcSerial, devRect, paint, &ident, aa)

	pmParallel := raster.NewPixmap(w, h)
	var rcParallel raster.Clip
	rcParallel.SetRect(full)
	fillRectShadedParallel(pmParallel, &rcParallel, devRect, paint, &ident, aa, bounds)

	empty := true
	for i := range pmSerial.Pix {
		if pmSerial.Pix[i] != pmParallel.Pix[i] {
			t.Fatalf("mismatch at (%d,%d): serial %08x parallel %08x (aa=%v workers=%d rect=%v)",
				int32(i)%pmSerial.RowPixels, int32(i)/pmSerial.RowPixels,
				pmSerial.Pix[i], pmParallel.Pix[i], aa, workers, devRect)
		}
		if pmSerial.Pix[i] != 0 {
			empty = false
		}
	}
	if empty {
		t.Fatalf("serial fill produced no output for %v", devRect)
	}
}

// TestRectFillBandBoundsByteIdentical exercises the byte-exactness claim across both shader lanes (gradient and image),
// both AA and non-AA, several worker counts (so the band seams land at different scanlines), and rects with integer,
// fractional, and offset edges.
func TestRectFillBandBoundsByteIdentical(t *testing.T) {
	rects := []geom.Rect{
		geom.RectLTRB(8, 8, 248, 248),
		geom.RectLTRB(8.4, 8.4, 247.6, 247.6),
		geom.RectLTRB(0.3, 0.7, 255.9, 255.1),
		geom.RectLTRB(16, 3.25, 240, 252.75),
	}
	for _, aa := range []bool{true, false} {
		for _, mk := range []func(bool) *Paint{parallelGradientPaint, parallelImagePaint} {
			for _, r := range rects {
				for _, workers := range []int{2, 3, 5, 8} {
					assertParallelByteIdentical(t, mk(aa), r.Sorted(), aa, workers)
				}
			}
		}
	}
}

// TestDrawRectShadedParallelMatchesSerial drives the full public canvas path (whose rect-fill lane now bands large
// shaded fills) and requires it to match a serial reference fill byte-for-byte, confirming the dispatch is wired
// correctly and stays byte-exact.
func TestDrawRectShadedParallelMatchesSerial(t *testing.T) {
	const w, h = 256, 256
	ident := geom.IdentityMatrix()
	rect := geom.RectLTRB(8.4, 8.4, 247.6, 247.6)
	for _, aa := range []bool{true, false} {
		paint := parallelGradientPaint(aa)

		pmPublic := raster.NewPixmap(w, h)
		NewForPixmap(pmPublic).DrawRect(rect, paint)

		pmRef := raster.NewPixmap(w, h)
		var rc raster.Clip
		rc.SetRect(geom.IRectWH(w, h))
		fillRectShadedBand(pmRef, &rc, rect.Sorted(), paint, &ident, aa)

		if !bytes.Equal(pmPublic.RGBA8888Bytes(), pmRef.RGBA8888Bytes()) {
			t.Fatalf("aa=%v: public parallel DrawRect differs from serial reference", aa)
		}
	}
}

// TestDrawRectShadedParallelDeterministic renders the same large shaded scene twice through the public path and
// requires identical output — the concurrent band fill must introduce no nondeterminism (run under -race for the
// accompanying data-race check).
func TestDrawRectShadedParallelDeterministic(t *testing.T) {
	const w, h = 256, 256
	rect := geom.RectLTRB(8, 8, 248, 248)
	for _, aa := range []bool{true, false} {
		for _, mk := range []func(bool) *Paint{parallelGradientPaint, parallelImagePaint} {
			first := raster.NewPixmap(w, h)
			NewForPixmap(first).DrawRect(rect, mk(aa))
			for i := 0; i < 4; i++ {
				again := raster.NewPixmap(w, h)
				NewForPixmap(again).DrawRect(rect, mk(aa))
				if !bytes.Equal(first.RGBA8888Bytes(), again.RGBA8888Bytes()) {
					t.Fatalf("aa=%v: parallel public DrawRect is nondeterministic on run %d", aa, i)
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// drawPaint (full-device shaded fill) band lane

// paintClipCase is a named Clip shape used to prove the drawPaint band fill is byte-exact and race-free for every
// clip kind, not just the simple rect the drawRectFull lane is limited to.
type paintClipCase struct {
	rc   *raster.Clip
	name string
}

// paintClips builds the clip shapes that cover FillIRectRasterClip's dispatch: a full-device rect and an inset rect
// (both BW rect clips), a complex BW region (a rect with a rectangular hole), and an AA clip (an antialiased oval).
// Every one is per-row independent, so integer-scanline banding must reproduce the serial fill exactly, and every
// clip's data is immutable so concurrent bands may share it (verified under -race).
func paintClips(w, h int32) []paintClipCase {
	full := geom.IRectWH(w, h)

	rectClip := raster.NewRasterClip()
	rectClip.SetRect(full)

	insetClip := raster.NewRasterClip()
	insetClip.SetRect(geom.IRectLTRB(4, 5, w-6, h-7))

	complexClip := raster.NewRasterClipRect(full)
	complexClip.OpIRect(geom.IRectLTRB(w/4, h/4, 3*w/4, 3*h/4), raster.ClipDifference)

	oval := path.New()
	oval.AddOval(geom.RectLTRB(2.5, 3.5, float32(w)-2.5, float32(h)-3.5), geom.DirectionCW)
	aaClip := raster.NewRasterClipPath(oval, full, true)

	return []paintClipCase{
		{name: "full-rect", rc: rectClip},
		{name: "inset-rect", rc: insetClip},
		{name: "complex", rc: complexClip},
		{name: "aa-oval", rc: aaClip},
	}
}

// assertPaintBandByteIdentical fills the whole device rect (the drawPaint lane) both serially — the exact production
// serial path (one blitter, FillIRectRasterClip over the full rect) — and via the concurrent integer-scanline row bands
// under the given clip, and requires the two device buffers to be byte-for-byte identical. Because FillIRectRasterClip
// is per-row independent for any clip, this holds for rect, complex, and AA clips alike (unlike the drawRectFull
// AA-rect lane, which is gated to rect clips). It also confirms the paint takes a bandable shader lane and that the
// fill produced output.
func assertPaintBandByteIdentical(t *testing.T, paint *Paint, w, h int32, clip *raster.Clip, workers int) {
	t.Helper()
	ident := geom.IdentityMatrix()
	full := geom.IRectWH(w, h)

	probe := chooseBlitter(raster.NewPixmap(w, h), paint, &ident)
	bandable := isBandableShaderBlitter(probe)
	recycleBlitter(probe)
	if !bandable {
		t.Fatalf("paint did not choose a bandable shader blitter (%T)", probe)
	}

	bounds := raster.IRectFillBandBounds(full.Top, full.Bottom, workers)
	if len(bounds) < 3 {
		t.Fatalf("expected >=2 bands for %dx%d (workers=%d), got %v", w, h, workers, bounds)
	}

	pmSerial := raster.NewPixmap(w, h)
	fillIRectShadedBand(pmSerial, clip, full, paint, &ident)

	pmParallel := raster.NewPixmap(w, h)
	fillIRectShadedParallel(pmParallel, clip, full, paint, &ident, bounds)

	empty := true
	for i := range pmSerial.Pix {
		if pmSerial.Pix[i] != pmParallel.Pix[i] {
			t.Fatalf("mismatch at (%d,%d): serial %08x parallel %08x (workers=%d w=%d h=%d)",
				int32(i)%pmSerial.RowPixels, int32(i)/pmSerial.RowPixels,
				pmSerial.Pix[i], pmParallel.Pix[i], workers, w, h)
		}
		if pmSerial.Pix[i] != 0 {
			empty = false
		}
	}
	if empty {
		t.Fatalf("serial fill produced no output (w=%d h=%d workers=%d)", w, h, workers)
	}
}

// TestDrawPaintBandByteIdentical exercises the drawPaint (full-device shaded fill) byte-exactness claim across both
// shader lanes (gradient and image), every clip kind (rect, inset rect, complex, AA), several device heights (so band
// seams land at different scanlines), and several worker counts. Run under -race, the concurrent bands' shared reads of
// the immutable clip are also verified race-free.
func TestDrawPaintBandByteIdentical(t *testing.T) {
	sizes := []struct{ w, h int32 }{
		{w: 200, h: 128}, {w: 256, h: 200}, {w: 160, h: 256}, {w: 131, h: 257},
	}
	for _, mk := range []func(bool) *Paint{parallelGradientPaint, parallelImagePaint} {
		for _, sz := range sizes {
			for _, clip := range paintClips(sz.w, sz.h) {
				for _, workers := range []int{2, 3, 5, 8} {
					assertPaintBandByteIdentical(t, mk(false), sz.w, sz.h, clip.rc, workers)
				}
			}
		}
	}
}

// TestDrawPaintShadedParallelMatchesSerial drives the full public canvas path (DrawPaint, whose lane now bands large
// full-device shaded fills) unclipped and requires it to match a serial reference fill byte-for-byte, confirming the
// dispatch is wired correctly and stays byte-exact.
func TestDrawPaintShadedParallelMatchesSerial(t *testing.T) {
	const w, h = 200, 200
	ident := geom.IdentityMatrix()
	full := geom.IRectWH(w, h)
	for _, mk := range []func(bool) *Paint{parallelGradientPaint, parallelImagePaint} {
		paint := mk(false)

		pmPublic := raster.NewPixmap(w, h)
		NewForPixmap(pmPublic).DrawPaint(paint)

		pmRef := raster.NewPixmap(w, h)
		rc := raster.NewRasterClip()
		rc.SetRect(full)
		fillIRectShadedBand(pmRef, rc, full, paint, &ident)

		if !bytes.Equal(pmPublic.RGBA8888Bytes(), pmRef.RGBA8888Bytes()) {
			t.Fatalf("public parallel DrawPaint differs from serial reference")
		}
	}
}

// TestDrawPaintShadedParallelDeterministic renders the same full-device shaded scene several times through the public
// DrawPaint path — unclipped, under a rect clip, and under an antialiased oval clip — and requires identical output
// each time: the concurrent band fill must introduce no nondeterminism, and the bands sharing the device's (complex/AA)
// clip must be race-free (run under -race).
func TestDrawPaintShadedParallelDeterministic(t *testing.T) {
	const w, h = 200, 200
	clips := []func(*Canvas){
		func(*Canvas) {}, // unclipped (full-device rect clip)
		func(cv *Canvas) { cv.ClipRect(geom.RectLTRB(6, 7, w-5, h-9), raster.ClipIntersect, false) },
		func(cv *Canvas) {
			p := path.New()
			p.AddOval(geom.RectLTRB(4.5, 5.5, w-4.5, h-5.5), geom.DirectionCW)
			cv.ClipPath(p, raster.ClipIntersect, true) // forces an AA clip
		},
	}
	for _, mk := range []func(bool) *Paint{parallelGradientPaint, parallelImagePaint} {
		for ci, setClip := range clips {
			first := raster.NewPixmap(w, h)
			cv := NewForPixmap(first)
			setClip(cv)
			cv.DrawPaint(mk(false))
			for i := 0; i < 4; i++ {
				again := raster.NewPixmap(w, h)
				cv2 := NewForPixmap(again)
				setClip(cv2)
				cv2.DrawPaint(mk(false))
				if !bytes.Equal(first.RGBA8888Bytes(), again.RGBA8888Bytes()) {
					t.Fatalf("clip %d: parallel public DrawPaint is nondeterministic on run %d", ci, i)
				}
			}
		}
	}
}
