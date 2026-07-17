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
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// TestImageDrawScratchReuseByteIdentical locks the pooled imageDrawScratch invariant: reusing a scratch after it built
// the paint for one image (through every wrap lane) must produce a shader observationally identical to a fresh
// scratch's for the next image, i.e. SetFromPixels/SetImage fully re-initialize the reused Image/ImageShader with no
// leftover state. Deterministic: single goroutine, so sync.Pool hands the just-released scratch back on the next Get.
func TestImageDrawScratchReuseByteIdentical(t *testing.T) {
	// imgA (48x48) poisons the scratch through the wrap lanes: cubic sampling plus a non-identity local matrix
	// (LocalMatrixShader wrap). imgB (40x24) is drawn plainly (base ImageShader only, nil matrix).
	imgB := benchImage(40, 24)
	pxB := imgB.PeekPixels(imagecore.CachingAllow)
	if pxB == nil {
		t.Fatal("nil pixels for imgB")
	}
	imgA := benchImage(48, 48)
	pxA := imgA.PeekPixels(imagecore.CachingAllow)
	if pxA == nil {
		t.Fatal("nil pixels for imgA")
	}
	cubic := shaders.SamplingOptions{UseCubic: true, Cubic: shaders.CubicResampler{B: 1.0 / 3, C: 1.0 / 3}}
	linear := shaders.SamplingOptions{Filter: shaders.FilterLinear}
	var lm geom.Matrix
	lm.SetTranslate(3, 5)

	// Reference: a fresh scratch builds imgB's paint. Its descriptor is the absolute expectation the reused scratch
	// must match (also cross-checked against imgB's own info/linear sampling below, so the assertions have teeth
	// against a reset that is dropped symmetrically on both scratches).
	fresh := acquireImageDrawScratch()
	ref := fresh.makePaintWithImage(NewPaint(), pxB, linear, nil)
	refIS, ok := ref.Shader.(*shaders.ImageShader)
	if !ok {
		t.Fatalf("fresh scratch base shader = %T, want *shaders.ImageShader", ref.Shader)
	}
	if di := refIS.DrawableImage(); di.Width() != 40 || di.Height() != 24 {
		t.Fatalf("fresh scratch image = %dx%d, want 40x24", di.Width(), di.Height())
	}
	if s := refIS.Sampling(); s != linear {
		t.Fatalf("fresh scratch sampling = %+v, want %+v", s, linear)
	}
	releaseImageDrawScratch(fresh)

	// Poison a scratch with imgA (cubic + local-matrix wrap), recycle it, then reuse for imgB.
	poison := acquireImageDrawScratch()
	if p := poison.makePaintWithImage(NewPaint(), pxA, cubic, &lm); p.Shader == nil {
		t.Fatal("poison paint shader unexpectedly nil")
	}
	releaseImageDrawScratch(poison)
	reused := acquireImageDrawScratch()
	got := reused.makePaintWithImage(NewPaint(), pxB, linear, nil)

	// The reused base shader must be a plain *ImageShader (not imgA's LocalMatrixShader wrap) carrying imgB's absolute
	// descriptor — not imgA's leftover 48x48 dims or cubic sampling.
	gotIS, ok := got.Shader.(*shaders.ImageShader)
	if !ok {
		t.Fatalf("reused base shader = %T, want *shaders.ImageShader (wrap leaked from poison)", got.Shader)
	}
	if di := gotIS.DrawableImage(); di.Width() != 40 || di.Height() != 24 {
		t.Errorf("reused image = %dx%d, want 40x24 (SetFromPixels leaked imgA's dims)", di.Width(), di.Height())
	}
	if s := gotIS.Sampling(); s != linear {
		t.Errorf("reused sampling = %+v, want %+v (SetImage leaked imgA's cubic sampling)", s, linear)
	}
	if tmx, tmy := gotIS.TileModes(); tmx != shaders.TileClamp || tmy != shaders.TileClamp {
		t.Errorf("reused tile modes = (%v,%v), want (clamp,clamp)", tmx, tmy)
	}
	releaseImageDrawScratch(reused)
}

// TestImageRectScratchReleaseClearsPaint locks the pooled imageRectScratch hygiene invariant: releaseImageRectScratch
// must drop the working paint's shader/color-filter/etc. references so an idle pooled scratch does not pin them.
// releaseImageRectScratch zeroes the paint in place before returning the scratch to the pool, so the reference we still
// hold observes the cleared state directly — the test asserts the clearing logic on that instance rather than depending
// on sync.Pool handing the same scratch back on the next Get, which it does not guarantee (e.g. under -race, where the
// victim cache can drop pooled items). Has teeth — fails if the release's `s.paint = Paint{}` clear is dropped.
func TestImageRectScratchReleaseClearsPaint(t *testing.T) {
	s := acquireImageRectScratch()
	s.src = geom.RectWH(4, 4)
	s.paint = Paint{
		Shader:      shaders.NewColor(colorcore.Color(0xFF112233)),
		ColorFilter: colorfilter.NewBlend(colorcore.Color(0xFF445566), raster.BlendSrcOver),
	}
	releaseImageRectScratch(s)

	// s still points at the just-released scratch; release cleared it in place (before the Put), so its paint must no
	// longer pin the shader/color-filter references. No Get is issued — the invariant holds regardless of which
	// instance the pool would hand back next.
	if s.paint.Shader != nil || s.paint.ColorFilter != nil {
		t.Errorf("release left dangling paint references: Shader=%v ColorFilter=%v",
			s.paint.Shader, s.paint.ColorFilter)
	}
}

// TestImageRectScratchReuseByteIdentical locks the pooled imageRectScratch reuse invariant end to end: a DrawImageRect
// whose paint carries a shader + color filter (so the released scratch's paint held references) followed by a plain
// DrawImageRect must render byte-identically to the plain draw alone. The pool hands the poisoned-then-cleared scratch
// to the plain draw, so the whole-value `scratch.paint = cleanPaintForDrawImage(paint)` overwrite must leave no
// leftover paint state in the output — teeth against a future field-update refactor that leaks a paint field through
// the pool.
func TestImageRectScratchReuseByteIdentical(t *testing.T) {
	img := benchImage(8, 8)
	src := geom.RectWH(8, 8)
	dst := geom.RectLTRB(2, 2, 14, 14)
	sampling := shaders.SamplingOptions{Filter: shaders.FilterLinear}

	// Reference: the plain image draw with a fresh canvas/paint.
	refPix := raster.NewPixmap(16, 16)
	NewForPixmap(refPix).DrawImageRect(img, src, dst, sampling, NewPaint(), ConstraintFast)

	// Pre-allocate the "got" surface so nothing allocates between the poison draw's scratch release and the plain
	// draw's acquire, maximizing the chance the pool returns the poisoned-then-cleared scratch.
	gotPix := raster.NewPixmap(16, 16)
	gotCanvas := NewForPixmap(gotPix)

	poison := NewPaint()
	poison.Shader = shaders.NewColor(colorcore.Color(0xFF00FF00))
	poison.ColorFilter = colorfilter.NewBlend(colorcore.Color(0xFFFF0000), raster.BlendSrcOver)
	NewForPixmap(raster.NewPixmap(16, 16)).DrawImageRect(img, src, dst, sampling, poison, ConstraintFast)

	gotCanvas.DrawImageRect(img, src, dst, sampling, NewPaint(), ConstraintFast)

	for i := range refPix.Pix {
		if gotPix.Pix[i] != refPix.Pix[i] {
			t.Fatalf("pixel %d after poisoned-scratch reuse = %08x, want %08x (paint state leaked through the pool)",
				i, gotPix.Pix[i], refPix.Pix[i])
		}
	}
}
