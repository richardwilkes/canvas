// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The draw tiler: the CPU scan converters are only safe to ~15-bit device coordinates, so draws against devices or
// clips extending past 8191px are re-issued once per 8191x8191 tile with a shifted CTM and a re-derived clip. Devices
// that fit in 8191px per side — everything unison creates — never construct a tiler at all: the wrapped device methods
// gate on clipMayNeedTiling and take the exact pre-tiler draw path otherwise, keeping that fast path allocation-free
// and byte-identical to the pre-H code.

package canvas

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// tilerMaxDim is the largest safe tile dimension: 8K is 1 too big, since 8K << supersample == 32768, which overflows
// the scan converters' 16.16 fixed-point coordinates.
const tilerMaxDim = 8192 - 1

// tilerNeedsTiling reports whether the device itself extends past the scan converters' safe range. Draw methods use it
// to decide whether computing draw bounds is worthwhile.
func tilerNeedsTiling(d *BitmapDevice) bool {
	return d.pix.Width > tilerMaxDim || d.pix.Height > tilerMaxDim
}

// clipMayNeedTiling is the cheap pre-tiler gate: tiling can only be required when the device clip extends past
// tilerMaxDim. The clip is always within the device bounds, so a ≤8191px device never passes this test. This predicate
// sits on every CPU draw; the wrapped device methods branch to the pre-tiler fast path when it is false.
func (d *BitmapDevice) clipMayNeedTiling() bool {
	clipR := d.rcStack.RC().Bounds()
	return clipR.Right > tilerMaxDim || clipR.Bottom > tilerMaxDim
}

// paintBounds returns the draw's local-space bounds grown by the paint (stroke width, mask/image filters), or nil when
// the paint defeats fast bounds — the tiler then has to visit every tile. It is only computed after clipMayNeedTiling
// passes, keeping the fast path free of computeFastBounds.
func paintBounds(rect geom.Rect, paint *Paint, storage *geom.Rect) *geom.Rect {
	if !paint.canComputeFastBounds() {
		return nil
	}
	*storage = paint.computeFastBounds(rect)
	return storage
}

// drawTiler re-issues a draw once per 8191x8191 tile: constructed per draw (only when clipMayNeedTiling passed) with
// optional local-space draw bounds; next yields one per-tile draw at a time, skipping tiles whose re-derived clip is
// empty. The walk order and step expressions are load-bearing — seam placement is observable.
type drawTiler struct {
	// dr is handed out by next (both tiled and untiled).
	dr     draw
	dev    *BitmapDevice
	tileRC raster.Clip
	// The per-tile state, used only when needsTiling.
	tileDst     raster.Pixmap
	tileMatrix  geom.Matrix
	srcBounds   geom.IRect // set only when needsTiling
	origin      geom.IPoint
	done        bool
	needsTiling bool
}

// init prepares the tiler for one draw. bounds is in local coordinates (the tiler maps it into device space); nil means
// the tiler has to visit everywhere the clip reaches.
func (t *drawTiler) init(dev *BitmapDevice, bounds *geom.Rect) {
	t.dev = dev
	t.done = false

	// The quick check (the callers already passed clipMayNeedTiling, but the tiler stays self-contained — the
	// cancel/done logic below can still turn tiling off).
	clipR := dev.rcStack.RC().Bounds()
	t.needsTiling = clipR.Right > tilerMaxDim || clipR.Bottom > tilerMaxDim
	if t.needsTiling {
		if bounds != nil {
			// Make sure we round first, and then intersect. We can't rely on promoting the clipR to floats (and then
			// intersecting with devBounds) since promoting int --> float can make the float larger than the int; our
			// RoundOut is saturating, which is fine for this use case. path.MapRect handles the full matrix, including
			// the perspective rect-path transform.
			mapped, _ := path.MapRect(dev.LocalToDevice(), *bounds)
			t.srcBounds = mapped.RoundOut()
			if t.srcBounds.Intersect(clipR) {
				// Check again, now that we have computed srcBounds.
				t.needsTiling = t.srcBounds.Right > tilerMaxDim || t.srcBounds.Bottom > tilerMaxDim
			} else {
				t.needsTiling = false
				t.done = true
			}
		} else {
			t.srcBounds = clipR
		}
	}

	if t.needsTiling {
		// dr.dst and dr.ctm are reset each time in stepAndSetupTileDraw.
		t.dr.rc = &t.tileRC
		// we'll step/increase it before using it
		t.origin = geom.IPoint{X: t.srcBounds.Left - tilerMaxDim, Y: t.srcBounds.Top}
	} else {
		// don't reference srcBounds, as it may not have been set
		t.dr.dst = dev.pix
		t.dr.ctm = &dev.localToDevice
		t.dr.rc = dev.rcStack.RC()
		t.origin = geom.IPoint{}
	}
}

// next returns the draw for the next non-empty tile, or nil when done. When tiling is not needed it yields the device's
// plain draw exactly once.
func (t *drawTiler) next() *draw {
	if t.done {
		return nil
	}
	if t.needsTiling {
		for {
			t.stepAndSetupTileDraw() // might set the clip to empty and done to true
			if t.done || !t.tileRC.IsEmpty() {
				break
			}
		}
		// if we exit the loop and we're still empty, we're (past) done
		if t.tileRC.IsEmpty() {
			return nil
		}
	} else {
		t.done = true // only draw untiled once
	}
	return &t.dr
}

// stepAndSetupTileDraw advances to the next tile and rebuilds the tile's pixmap subset, shifted CTM, and re-derived
// clip.
func (t *drawTiler) stepAndSetupTileDraw() {
	// We do srcBounds.Right - kMaxDim instead of origin.X + kMaxDim to avoid overflow.
	if t.origin.X >= t.srcBounds.Right-tilerMaxDim { // too far
		t.origin.X = t.srcBounds.Left
		t.origin.Y += tilerMaxDim
	} else {
		t.origin.X += tilerMaxDim
	}
	// done = next origin will be invalid.
	t.done = t.origin.X >= t.srcBounds.Right-tilerMaxDim &&
		t.origin.Y >= t.srcBounds.Bottom-tilerMaxDim

	bounds := geom.IRectXYWH(t.origin.X, t.origin.Y, tilerMaxDim, tilerMaxDim)
	// srcBounds is inside the clip, which is inside the device, so the intersection cannot be empty.
	t.dev.pix.ExtractSubset(&t.tileDst, bounds)
	t.dr.dst = &t.tileDst
	// now don't use bounds, since tileDst has the clipped dimensions.

	t.tileMatrix = t.dev.localToDevice
	t.tileMatrix.PostTranslate(float32(-t.origin.X), float32(-t.origin.Y))
	t.dr.ctm = &t.tileMatrix
	t.dev.rcStack.RC().Translate(-t.origin.X, -t.origin.Y, &t.tileRC)
	t.tileRC.OpIRect(geom.IRectWH(t.tileDst.Width, t.tileDst.Height), raster.ClipIntersect)
}
