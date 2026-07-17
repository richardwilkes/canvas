// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pure-Go unit coverage of the draw tiler: the tile walk order, seam placement, the needsTiling predicates, and
// empty-tile skipping via draw bounds. Strips (rather than squares) keep the big devices cheap: a 20000x16 pixmap is
// ~1.25MB.
//
// These are now the tiler's whole gate. A seam-parity differential against the C library once complemented them
// (rendering big strips of hairlines, strokes and gradients through both tilers), and went with the rest of the C
// dependency. What it uniquely covered was AA coverage at a per-tile re-clip boundary for *path-lowered* draws, which
// TestDrawTilerSeamEquivalence deliberately cannot cover (see its comment); the tiler machinery it shared with the
// rect-fill and glyph lanes is still gated here and by internal/oracle/gorender's TestTilerTextSeamWindow.

package canvas

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// tileStep records one tile yielded by the walk.
type tileStep struct {
	origin geom.IPoint
	w, h   int32
}

// walkTiles runs the tiler over dev with the given local-space bounds and returns the non-empty tiles in walk order
// (nil when the tiler took the untiled path — the single yielded draw is the root device).
func walkTiles(t *testing.T, dev *BitmapDevice, bounds *geom.Rect) (steps []tileStep, tiled bool) {
	t.Helper()
	var tiler drawTiler
	tiler.init(dev, bounds)
	for dr := tiler.next(); dr != nil; dr = tiler.next() {
		if !tiler.needsTiling {
			return nil, false
		}
		steps = append(steps, tileStep{origin: tiler.origin, w: dr.dst.Width, h: dr.dst.Height})
	}
	return steps, true
}

func TestDrawTilerNeedsTilingPredicates(t *testing.T) {
	small := NewBitmapDevice(raster.NewPixmap(tilerMaxDim, 64)) // 8191 wide: the largest untiled device
	if tilerNeedsTiling(small) {
		t.Errorf("8191px device must not need tiling")
	}
	if small.clipMayNeedTiling() {
		t.Errorf("8191px clip must not pass the quick check")
	}

	big := NewBitmapDevice(raster.NewPixmap(tilerMaxDim+1, 64)) // 8192 wide: one too big
	if !tilerNeedsTiling(big) {
		t.Errorf("8192px device must need tiling")
	}
	if !big.clipMayNeedTiling() {
		t.Errorf("8192px wide-open clip must pass the quick check")
	}

	// A clip restricted inside the safe range cancels the quick check even on a huge device.
	big.ClipRect(geom.RectWH(4000, 16), raster.ClipIntersect, false)
	if big.clipMayNeedTiling() {
		t.Errorf("clip within 8191px must not pass the quick check")
	}
	steps, tiled := walkTiles(t, big, nil)
	if tiled || steps != nil {
		t.Errorf("tiler must take the untiled path under a small clip, got %d tiles", len(steps))
	}
}

func TestDrawTilerWalkOrderHorizontal(t *testing.T) {
	dev := NewBitmapDevice(raster.NewPixmap(20000, 16))
	steps, tiled := walkTiles(t, dev, nil)
	if !tiled {
		t.Fatalf("expected tiling")
	}
	// The expected walk: x-major from srcBounds' top-left, stepping tilerMaxDim; seams at 8191 and 16382.
	want := []tileStep{
		{origin: geom.IPoint{X: 0, Y: 0}, w: 8191, h: 16},
		{origin: geom.IPoint{X: 8191, Y: 0}, w: 8191, h: 16},
		{origin: geom.IPoint{X: 16382, Y: 0}, w: 20000 - 16382, h: 16},
	}
	if len(steps) != len(want) {
		t.Fatalf("got %d tiles, want %d: %+v", len(steps), len(want), steps)
	}
	for i, s := range steps {
		if s != want[i] {
			t.Errorf("tile %d: got %+v, want %+v", i, s, want[i])
		}
	}
}

func TestDrawTilerWalkOrderVertical(t *testing.T) {
	dev := NewBitmapDevice(raster.NewPixmap(16, 20000))
	steps, tiled := walkTiles(t, dev, nil)
	if !tiled {
		t.Fatalf("expected tiling")
	}
	want := []tileStep{
		{origin: geom.IPoint{X: 0, Y: 0}, w: 16, h: 8191},
		{origin: geom.IPoint{X: 0, Y: 8191}, w: 16, h: 8191},
		{origin: geom.IPoint{X: 0, Y: 16382}, w: 16, h: 20000 - 16382},
	}
	if len(steps) != len(want) {
		t.Fatalf("got %d tiles, want %d: %+v", len(steps), len(want), steps)
	}
	for i, s := range steps {
		if s != want[i] {
			t.Errorf("tile %d: got %+v, want %+v", i, s, want[i])
		}
	}
}

func TestDrawTilerBoundsAnchorAndSkip(t *testing.T) {
	dev := NewBitmapDevice(raster.NewPixmap(20000, 16))

	// Bounds that fit within kMaxDim of the origin cancel tiling after the srcBounds recheck: one untiled draw against
	// the root device.
	fits := geom.RectLTRB(100, 2, 8000, 14)
	if steps, tiled := walkTiles(t, dev, &fits); tiled {
		t.Errorf("bounds within 8191 must cancel tiling, got %d tiles", len(steps))
	}

	// Bounds entirely past kMaxDim tighten srcBounds so the walk is anchored at the bounds' left edge: one tile at
	// x=10000 (not the device-anchored grid), skipping everything before it. The tile's height is the device
	// intersection (16-2=14) — the subset is clipped by the pixmap, not srcBounds.
	narrow := geom.RectLTRB(10000, 2, 10100, 14)
	steps, tiled := walkTiles(t, dev, &narrow)
	if !tiled {
		t.Fatalf("expected tiling for bounds past 8191")
	}
	want := []tileStep{{origin: geom.IPoint{X: 10000, Y: 2}, w: 8191, h: 14}}
	if len(steps) != 1 || steps[0] != want[0] {
		t.Errorf("got %+v, want %+v", steps, want)
	}

	// Bounds that straddle the recheck limit: [8000, 8600) still has Right past 8191, so tiling stays on, but the walk
	// anchored at x=8000 covers the whole srcBounds in its first 8191-wide tile and the done test fires immediately —
	// exactly one tile.
	straddle := geom.RectLTRB(8000, 0, 8600, 16)
	steps, tiled = walkTiles(t, dev, &straddle)
	if !tiled {
		t.Fatalf("expected tiling for straddling bounds")
	}
	if len(steps) != 1 || steps[0].origin != (geom.IPoint{X: 8000, Y: 0}) {
		t.Errorf("straddling bounds: got %+v, want single tile at x=8000", steps)
	}

	// Bounds that do not intersect the clip at all draw nothing.
	dev.PushClipStack()
	dev.ClipRect(geom.RectLTRB(0, 0, 9000, 16), raster.ClipIntersect, false)
	outside := geom.RectLTRB(15000, 0, 16000, 16)
	var tiler drawTiler
	tiler.init(dev, &outside)
	if dr := tiler.next(); dr != nil {
		t.Errorf("bounds outside the clip must yield no draws")
	}
	dev.PopClipStack()
}

// TestDrawTilerSeamEquivalence verifies seam placement pixel-wise without the oracle: scenes drawn on a tiled strip
// must match an untiled window render of the same scene translated into view, for draws whose rasterization is
// invariant under integer translation and clip masking — rect fills (AA and non-AA, the AA rect lane's analytic
// coverage is per-rect) and translucent blends over them. Path-lowered draws (strokes, rrects, hairlines) are
// deliberately absent: the AA supersampler's coverage rounding depends on where the clip crops the path, so a tiled
// strip and an untiled window legitimately differ at those boundary pixels and this test cannot speak to them. They
// were covered by the C-parity differential, which tiled both sides identically and so sidestepped the problem; it is
// gone with the C dependency, and that lane is now uncovered (see the file comment).
func TestDrawTilerSeamEquivalence(t *testing.T) {
	const w, h = 20000, 24
	scene := func(c *Canvas) {
		c.Clear(colorcore.Color(0xFFFFFFFF))
		fill := NewPaint()
		fill.Color = colorcore.Color(0xFF3366CC)
		fill.AntiAlias = true
		// AA rects straddling both seams (8191, 16382) at fractional offsets.
		c.DrawRect(geom.RectLTRB(8191-40.25, 3.5, 8191+40.75, 20.25), fill)
		c.DrawRect(geom.RectLTRB(16382-15.5, 1.25, 16382+90.5, 22.75), fill)
		// A translucent non-AA rect layered across the first seam (integer region fill + blend).
		blend := NewPaint()
		blend.Color = colorcore.Color(0x8022AA44)
		c.DrawRect(geom.RectLTRB(8100, 6, 8300, 18), blend)
	}

	strip := raster.NewPixmap(w, h)
	c := NewForPixmap(strip)
	scene(c)

	// Windows re-render the scene untiled (256 wide ≤ 8191) with the content translated into view; the strip's pixels
	// in that window must match byte for byte.
	const winW = 256
	for _, x0 := range []int32{8191 - 128, 16382 - 128, 0, w - winW} {
		win := raster.NewPixmap(winW, h)
		wc := NewForPixmap(win)
		wc.Translate(float32(-x0), 0)
		scene(wc)
		for y := int32(0); y < h; y++ {
			for x := int32(0); x < winW; x++ {
				got := strip.Pix[int(y)*int(strip.RowPixels)+int(x0+x)]
				want := win.Pix[int(y)*int(win.RowPixels)+int(x)]
				if got != want {
					t.Fatalf("window at x0=%d: pixel (%d,%d) tiled=%08x untiled=%08x", x0, x, y, got, want)
				}
			}
		}
	}
}
