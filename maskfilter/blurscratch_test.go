// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package maskfilter

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// TestGrowScratch locks the two growScratch lanes: reuse the backing array when the capacity already suffices
// (resliced, not reallocated), and allocate fresh only when it does not.
func TestGrowScratch(t *testing.T) {
	fresh := growScratch[uint8](nil, 4)
	if len(fresh) != 4 || cap(fresh) < 4 {
		t.Fatalf("nil grow: len=%d cap=%d, want len 4", len(fresh), cap(fresh))
	}

	big := make([]uint8, 8)
	reused := growScratch(big, 3)
	if len(reused) != 3 {
		t.Fatalf("reslice len=%d, want 3", len(reused))
	}
	if &reused[0] != &big[0] {
		t.Fatal("growScratch reallocated when the backing array had capacity")
	}

	grown := growScratch(reused, 16)
	if len(grown) != 16 {
		t.Fatalf("grow len=%d, want 16", len(grown))
	}
	if &grown[0] == &big[0] {
		t.Fatal("growScratch reused a too-small backing array")
	}
}

// TestBlurScratchReuse is the pooling correctness gate: a scratch whose buffers were grown by a prior (larger) draw and
// then filled with garbage must produce byte-identical blurRect output to a fresh scratch. growScratch reslices without
// zeroing, so this proves every buffer element that is read is written first (progress §14.1) — a future partial-write
// regression that relied on the old make()'s zeroing would fail here.
func TestBlurScratchReuse(t *testing.T) {
	rect := geom.RectLTRB(12, 8, 96, 60)
	const sigma = 4

	fresh := &blurScratch{}
	var want raster.Mask
	wantMargin, ok := blurRect(sigma, rect, BlurNormal, computeBoundsAndRenderImage, fresh, &want)
	if !ok {
		t.Fatal("fresh blurRect failed")
	}
	// blurRect's filled image aliases scratch.image; copy it before the reuse run clobbers it.
	wantImage := bytes.Clone(want.Image)
	wantBounds := want.Bounds
	wantRowBytes := want.RowBytes

	reuse := &blurScratch{}
	// A prior, larger draw grows every buffer past what the target rect needs.
	var prime raster.Mask
	if _, ok = blurRect(sigma, geom.RectLTRB(0, 0, 200, 160), BlurNormal, computeBoundsAndRenderImage, reuse, &prime); !ok {
		t.Fatal("priming blurRect failed")
	}
	poison(reuse.image)
	poison(reuse.profile)
	poison(reuse.hScan)
	poison(reuse.vScan)

	var got raster.Mask
	gotMargin, ok := blurRect(sigma, rect, BlurNormal, computeBoundsAndRenderImage, reuse, &got)
	if !ok {
		t.Fatal("reuse blurRect failed")
	}
	if got.Bounds != wantBounds || got.RowBytes != wantRowBytes || gotMargin != wantMargin {
		t.Fatalf("reuse metadata mismatch: bounds %+v/%+v rowBytes %d/%d margin %+v/%+v",
			got.Bounds, wantBounds, got.RowBytes, wantRowBytes, gotMargin, wantMargin)
	}
	if !bytes.Equal(got.Image, wantImage) {
		t.Fatal("reuse produced different pixels than a fresh scratch")
	}
}

func poison(b []uint8) {
	for i := range b {
		b[i] = 0xAA
	}
}
