// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the banded shaded path-fill lane. Unlike the rect lanes, path banding is *not* byte-exact against the
// serial fill (band-clipped scan conversion re-derives edges per band), so the locks here are: the parallel machinery
// is deterministic and equals the same bands filled sequentially; the public path stays within a characterized
// divergence envelope of the serial fill; and unison's dominant shaded shape (the rrect) happens to reproduce the
// serial fill exactly.

package canvas

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// bandTestPaths builds the shapes the C.1 lane is characterized against: unison's rounded rect, an off-integer oval,
// and a self-intersecting mixed-curve torture path.
func bandTestPaths(w, h float32) map[string]*path.Path {
	shapes := map[string]*path.Path{}

	rr := path.New()
	rr.AddRRect(geom.MakeRRect(geom.RectLTRB(8.4, 8.4, w-8.4, h-8.4), 24, 24), geom.DirectionCW)
	shapes["rrect"] = rr

	ov := path.New()
	ov.AddOval(geom.RectLTRB(10.3, 5.7, w-10.8, h-5.1), geom.DirectionCW)
	shapes["oval"] = ov

	star := path.New()
	xs := []float32{0.10, 0.85, 0.30, 0.95, 0.20, 0.70, 0.05, 0.60, 0.45}
	ys := []float32{0.15, 0.20, 0.90, 0.55, 0.35, 0.95, 0.50, 0.10, 0.75}
	for c := 0; c < 3; c++ {
		base := c * 3
		star.MoveTo(xs[base%9]*w, ys[base%9]*h)
		for s := 1; s <= 8; s++ {
			i := (base + s) % 9
			j := (base + s*2) % 9
			switch s % 3 {
			case 0:
				star.LineTo(xs[i]*w, ys[j]*h)
			case 1:
				star.QuadTo(xs[j]*w, ys[i]*h, xs[i]*w, ys[j]*h)
			default:
				k := (base + s*3) % 9
				star.CubicTo(xs[j]*w, ys[i]*h, xs[k]*w, ys[k]*h, xs[i]*w, ys[j]*h)
			}
		}
		star.Close()
	}
	shapes["star"] = star
	return shapes
}

// renderPathSerial fills p through the exact production serial lane (one blitter over the raster clip).
func renderPathSerial(p *path.Path, paint *Paint, w, h int32, aa bool) *raster.Pixmap {
	ident := geom.IdentityMatrix()
	pm := raster.NewPixmap(w, h)
	var rc raster.Clip
	rc.SetRect(geom.IRectWH(w, h))
	blitter := chooseBlitter(pm, paint, &ident)
	if aa {
		raster.AntiFillPathRasterClip(p, &rc, blitter)
	} else {
		raster.FillPathRasterClip(p, &rc, blitter)
	}
	recycleBlitter(blitter)
	return pm
}

// warmShaderState performs the serial chooseBlitter fillPathShadedParallel's contract requires of its caller
// (drawDevPath does this inherently): it compiles one pipeline for the paint, resolving the shader tree's
// lazily-memoized matrix masks before any concurrent bands read them.
func warmShaderState(paint *Paint, w, h int32) {
	ident := geom.IdentityMatrix()
	probe := chooseBlitter(raster.NewPixmap(w, h), paint, &ident)
	recycleBlitter(probe)
}

// renderPathBanded fills p through fillPathShadedParallel over the same clip/band split drawDevPath uses.
func renderPathBanded(t *testing.T, p *path.Path, paint *Paint, w, h int32, aa bool) *raster.Pixmap {
	t.Helper()
	warmShaderState(paint, w, h)
	ident := geom.IdentityMatrix()
	clip := geom.IRectWH(w, h)
	ir := p.Bounds().RoundOut()
	top := max(ir.Top, clip.Top)
	bottom := min(ir.Bottom, clip.Bottom)
	bounds := raster.IRectFillBandBounds(top, bottom, 0)
	if bounds == nil {
		t.Fatalf("expected the %dx%d path to band", w, h)
	}
	pm := raster.NewPixmap(w, h)
	fillPathShadedParallel(pm, p, paint, &ident, aa, clip, bounds)
	return pm
}

// TestFillPathShadedParallelDeterministic locks that the concurrent banded shaded fill produces exactly the bytes of
// the same bands filled sequentially through per-band blitters — the concurrency itself introduces no nondeterminism
// (run under -race for the data-race check).
func TestFillPathShadedParallelDeterministic(t *testing.T) {
	const w, h = 256, 256
	ident := geom.IdentityMatrix()
	clip := geom.IRectWH(w, h)
	for name, p := range bandTestPaths(w, h) {
		for _, aa := range []bool{true, false} {
			for _, mk := range []func(bool) *Paint{parallelGradientPaint, parallelImagePaint} {
				paint := mk(aa)
				warmShaderState(paint, w, h)
				ir := p.Bounds().RoundOut()
				top := max(ir.Top, clip.Top)
				bottom := min(ir.Bottom, clip.Bottom)
				bounds := raster.IRectFillBandBounds(top, bottom, 0)
				if bounds == nil {
					t.Fatalf("%s: expected bands", name)
				}
				want := raster.NewPixmap(w, h)
				for i := 0; i+1 < len(bounds); i++ {
					fillPathShadedBand(want, p, paint, &ident, aa,
						geom.IRectLTRB(clip.Left, bounds[i], clip.Right, bounds[i+1]))
				}
				got := raster.NewPixmap(w, h)
				fillPathShadedParallel(got, p, paint, &ident, aa, clip, bounds)
				if !bytes.Equal(want.RGBA8888Bytes(), got.RGBA8888Bytes()) {
					t.Fatalf("%s aa=%v: concurrent path banding nondeterministic", name, aa)
				}
			}
		}
	}
}

// TestFillPathShadedParallelDivergenceEnvelope characterizes the banded-vs-serial divergence the C.1 lane accepts (and
// that the re-profiled oracle budgets rest on): the diff fraction stays under 5% even on the self-intersecting curve
// torture path (measured 4.4% there — every differing pixel is a partial-coverage edge pixel whose band-clipped curve
// subdivision landed differently; simple shapes measure 0-0.4%), and unison's rrect must reproduce the serial fill
// byte-exactly.
func TestFillPathShadedParallelDivergenceEnvelope(t *testing.T) {
	const w, h = 256, 256
	for name, p := range bandTestPaths(w, h) {
		for _, aa := range []bool{true, false} {
			paint := parallelGradientPaint(aa)
			serial := renderPathSerial(p, paint, w, h, aa)
			banded := renderPathBanded(t, p, paint, w, h, aa)
			diff := 0
			for i := range serial.Pix {
				if serial.Pix[i] != banded.Pix[i] {
					diff++
				}
			}
			frac := float64(diff) / float64(w*h)
			if name == "rrect" && diff != 0 {
				t.Errorf("rrect aa=%v: expected byte-exact banding, got %d differing pixels", aa, diff)
			}
			if frac > 0.05 {
				t.Errorf("%s aa=%v: banded fill diverges from serial on %.2f%% of pixels (cap 5%%)",
					name, aa, frac*100)
			}
		}
	}
}

// TestDrawPathShadedBandedWiringAndDeterminism drives the public DrawPath and locks (1) the banded lane is actually
// taken for a tall shaded path under a rect clip — the public bytes equal the banded reference, which differs from the
// serial fill for this path — and (2) repeated public draws are identical (determinism through the full dispatch).
func TestDrawPathShadedBandedWiringAndDeterminism(t *testing.T) {
	const w, h = 256, 256
	shapes := bandTestPaths(w, h)
	star := shapes["star"]
	paint := parallelGradientPaint(true)

	banded := renderPathBanded(t, star, paint, w, h, true)
	serial := renderPathSerial(star, paint, w, h, true)
	if bytes.Equal(banded.RGBA8888Bytes(), serial.RGBA8888Bytes()) {
		t.Fatal("banded and serial star renders are identical; the wiring check below would be vacuous")
	}

	first := raster.NewPixmap(w, h)
	NewForPixmap(first).DrawPath(star, paint)
	if !bytes.Equal(first.RGBA8888Bytes(), banded.RGBA8888Bytes()) {
		t.Fatal("public DrawPath did not take the banded shaded path lane")
	}
	for i := 0; i < 4; i++ {
		again := raster.NewPixmap(w, h)
		NewForPixmap(again).DrawPath(star, paint)
		if !bytes.Equal(first.RGBA8888Bytes(), again.RGBA8888Bytes()) {
			t.Fatalf("public DrawPath banded lane nondeterministic on run %d", i)
		}
	}

	// A solid paint must stay on the serial lane (C.1 is scoped to shaded fills).
	solid := NewPaint()
	solid.Color = 0x80336699
	solid.AntiAlias = true
	pmSolid := raster.NewPixmap(w, h)
	NewForPixmap(pmSolid).DrawPath(star, solid)
	pmSolidRef := renderPathSerial(star, solid, w, h, true)
	if !bytes.Equal(pmSolid.RGBA8888Bytes(), pmSolidRef.RGBA8888Bytes()) {
		t.Fatal("solid path fill no longer matches the serial lane")
	}
}
