// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the TriangulatingPathRenderer: non-AA fills of all four fill types matched against the Go CPU
// rasterizer (Go-GPU vs Go-CPU self-consistency), the coverage-AA alpha-ramp lane, and the ThreadSafeCache vertex-data
// lanes (repeat-draw hits, the tolerance-mismatch re-triangulation with its is-newer-better replacement, and entry
// aging). Skips when no GL context is available.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

func TestLiveTriangulatingFills(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128

	fillTypes := []struct {
		name string
		ft   path.FillType
	}{
		{name: "winding", ft: path.FillWinding},
		{name: "evenodd", ft: path.FillEvenOdd},
		{name: "inverse-winding", ft: path.FillInverseWinding},
		{name: "inverse-evenodd", ft: path.FillInverseEvenOdd},
	}
	for _, tc := range fillTypes {
		scene := func(c *canvas.Canvas) {
			bg := canvas.NewPaint()
			bg.Color = 0xFFFFFFFF
			c.DrawPaint(bg)
			p := liveStarPath()
			p.SetFillType(tc.ft)
			red := canvas.NewPaint()
			red.Color = 0xFFDD2200
			red.AntiAlias = false // keyed + non-AA selects TriangulatingPathRenderer
			c.DrawPath(p, red)
		}

		cpuPix := raster.NewPixmap(w, h)
		scene(canvas.NewForPixmap(cpuPix))

		sdc := newLiveDrawSDC(t, dc, w, h)
		dev := gl.NewDevice(sdc)
		scene(canvas.New(dev))
		gpuData := readSDC(t, sdc)
		sdc.Release()

		// Non-AA triangle rasterization differs from the CPU scan converter only where edges land within rounding
		// distance of pixel centers.
		compareCPUGPU(t, gpuData, cpuPix, w, h, w*h/200, "triangulated "+tc.name)
	}
}

func TestLiveTriangulatingCoverageAA(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()
	sdc.Clear([4]float32{1, 1, 1, 1})

	identity := geom.IdentityMatrix()
	shape := gl.MakeStyledShapePath(liveStarPath(), gl.SimpleFillStyle(), gl.DoSimplifyNo)
	sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0.8, G: 0.1, B: 0, A: 1}), gpu.AAYes,
		&identity, &shape)

	data := readSDC(t, sdc)
	rb := w * 4
	probeAt(t, data, rb, 50, 48, [4]byte{204, 26, 0, 255}, 2, "aa star interior")
	probeAt(t, data, rb, 8, 100, [4]byte{255, 255, 255, 255}, 1, "aa star exterior")

	// The one-pixel coverage ramp must produce genuinely partial pixels along the spike edges.
	partial := false
	for y := int32(50); y < 80 && !partial; y++ {
		for x := int32(10); x < 50; x++ {
			off := int(y)*rb + int(x)*4
			g := data[off+1]
			if g > 40 && g < 215 {
				partial = true
				break
			}
		}
	}
	if !partial {
		t.Fatal("no partially covered pixel found along the star edge under coverage AA")
	}
}

func TestLiveTriangulationCaching(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const w, h = 128, 128
	sdc := newLiveDrawSDC(t, dc, w, h)
	defer sdc.Release()

	tsc := dc.ThreadSafeCache()
	*tsc.Stats() = gl.ThreadSafeCacheStats{}

	blue := colorcore.PMColor4f{R: 0, G: 0.2, B: 0.9, A: 1}
	// A concave curved path (kept small so the 4x-scaled draw stays in bounds): curve tolerance participates in the
	// cache-match rule only for non-linear triangulations.
	crescent := &path.Path{}
	crescent.MoveTo(5, 15)
	crescent.QuadTo(16, 30, 27, 15)
	crescent.QuadTo(16, 22, 5, 15)
	crescent.Close()

	draw := func(m geom.Matrix) {
		shape := gl.MakeStyledShapePath(crescent, gl.SimpleFillStyle(), gl.DoSimplifyNo)
		sdc.Clear([4]float32{1, 1, 1, 1})
		sdc.DrawShape(nil, livePaint(blue), gpu.AANo, &m, &shape)
		_ = readSDC(t, sdc)
	}

	identity := geom.IdentityMatrix()
	draw(identity)
	if s := tsc.Stats(); s.Misses != 1 || s.Adds != 1 || s.Hits != 0 || s.Replacements != 0 {
		t.Fatalf("after first draw: %+v", *s)
	}
	if tsc.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", tsc.NumEntries())
	}

	// The identical draw reuses the cached triangulation.
	draw(identity)
	if s := tsc.Stats(); s.Hits != 1 || s.Adds != 1 || s.Replacements != 0 {
		t.Fatalf("after repeat draw: %+v", *s)
	}

	// A translated draw shares the key (the triangulation is in source space).
	draw(geom.TranslateMatrix(40, 12))
	if s := tsc.Stats(); s.Hits != 2 || s.Adds != 1 || s.Replacements != 0 {
		t.Fatalf("after translated draw: %+v", *s)
	}

	// Scaling up shrinks the source-space tolerance past the cache-match rule: the entry is found but rejected, a finer
	// triangulation is generated, and is-newer-better replaces the coarser incumbent (same key — still one entry).
	draw(geom.ScaleMatrix(4, 4))
	if s := tsc.Stats(); s.Hits != 3 || s.Adds != 1 || s.Replacements != 1 {
		t.Fatalf("after scaled draw: %+v", *s)
	}
	if tsc.NumEntries() != 1 {
		t.Fatalf("entries = %d, want 1", tsc.NumEntries())
	}

	// The finer triangulation satisfies the identity draw too (its tolerance is smaller).
	draw(identity)
	if s := tsc.Stats(); s.Hits != 4 || s.Adds != 1 || s.Replacements != 1 {
		t.Fatalf("after re-draw at identity: %+v", *s)
	}

	// After the flushes complete no op holds the payload: purging all unlocked resources drops the uniquely held entry
	// through the resource-cache hook.
	dc.PurgeUnlockedResources(gpu.PurgeAllResources)
	if tsc.NumEntries() != 0 {
		t.Fatalf("entries = %d after purge, want 0", tsc.NumEntries())
	}

	// And the next draw regenerates it.
	draw(identity)
	if s := tsc.Stats(); s.Adds != 2 {
		t.Fatalf("after post-purge draw: %+v", *s)
	}
}

func TestLiveTriangulatingInverseClipBounds(t *testing.T) {
	// An inverse fill keyed on the clip bounds: the triangulation includes the clip-bounds contour, so different
	// surfaces (clip bounds) must not share entries.
	_, dc := newLiveDirectContext(t)
	tsc := dc.ThreadSafeCache()
	*tsc.Stats() = gl.ThreadSafeCacheStats{}

	inv := liveStarPath()
	inv.SetFillType(path.FillInverseWinding)

	drawOn := func(w, h int32) {
		sdc := newLiveDrawSDC(t, dc, w, h)
		defer sdc.Release()
		sdc.Clear([4]float32{1, 1, 1, 1})
		identity := geom.IdentityMatrix()
		shape := gl.MakeStyledShapePath(inv, gl.SimpleFillStyle(), gl.DoSimplifyNo)
		sdc.DrawShape(nil, livePaint(colorcore.PMColor4f{R: 0, G: 0.5, B: 0, A: 1}), gpu.AANo,
			&identity, &shape)
		data := readSDC(t, sdc)
		rb := int(w) * 4
		probeAt(t, data, rb, 50, 48, [4]byte{255, 255, 255, 255}, 1, "inverse: star uncovered")
		probeAt(t, data, rb, 8, 100, [4]byte{0, 128, 0, 255}, 1, "inverse: outside covered")
	}
	drawOn(128, 128)
	drawOn(96, 112)
	if s := tsc.Stats(); s.Adds != 2 {
		t.Fatalf("inverse fills on different clip bounds must not share keys: %+v", *s)
	}
}
