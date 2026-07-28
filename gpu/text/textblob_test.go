// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package text

import (
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/patheffect"
	"github.com/richardwilkes/canvas/textblob"
)

func TestBlobKeyCaching(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "key", geom.Pt(3, 4))
	m := geom.IdentityMatrix()

	// Blob-backed fills cache.
	canCache, key := MakeBlobKey(list, fillPaint(), &m, nil, noSDFTControl())
	if !canCache {
		t.Fatal("blob-backed fill must cache")
	}
	if !key.hasSomeDirectSubRuns {
		t.Fatal("16px identity draw must have direct subruns")
	}

	// The same parameters produce an equal key; a different blob does not.
	_, key2 := MakeBlobKey(list, fillPaint(), &m, nil, noSDFTControl())
	if !key.equal(&key2) {
		t.Fatal("identical draws must produce equal keys")
	}
	other := runListForText(t, f, "key", geom.Pt(3, 4))
	_, key3 := MakeBlobKey(other, fillPaint(), &m, nil, noSDFTControl())
	if key.equal(&key3) {
		t.Fatal("distinct blobs must produce unequal keys")
	}

	// An integer translation of the position matrix matches under the fractional-offset rule; a fractional translation
	// does not.
	mInt := m
	mInt.PostTranslate(7, -2)
	_, keyInt := MakeBlobKey(list, fillPaint(), &mInt, nil, noSDFTControl())
	if !key.equal(&keyInt) {
		t.Fatal("integer translate must match the key")
	}
	mFrac := m
	mFrac.PostTranslate(0.25, 0)
	_, keyFrac := MakeBlobKey(list, fillPaint(), &mFrac, nil, noSDFTControl())
	if key.equal(&keyFrac) {
		t.Fatal("fractional translate must not match the key")
	}

	// Path effects disable caching; blur mask filters keep it (with the rec in the key).
	pePaint := canvas.NewPaint()
	pePaint.PathEffect = patheffect.MakeCorner(2)
	if can, _ := MakeBlobKey(list, pePaint, &m, nil, noSDFTControl()); can {
		t.Fatal("path effects must disable caching")
	}
	blurPaint := canvas.NewPaint()
	blurPaint.MaskFilter = maskfilter.NewBlur(maskfilter.BlurNormal, 2, true)
	can, blurKey := MakeBlobKey(list, blurPaint, &m, nil, noSDFTControl())
	if !can || !blurKey.hasBlur || blurKey.blurSigma != 2 {
		t.Fatalf("blur key: can=%v hasBlur=%v sigma=%g", can, blurKey.hasBlur, blurKey.blurSigma)
	}
	if key.equal(&blurKey) {
		t.Fatal("blur must change the key")
	}

	// Non-blob lists never cache.
	builder := textblob.NewGlyphRunBuilder()
	direct := builder.TextToGlyphRunList(f, nil, []byte("t"), font.TextEncodingUTF8,
		geom.Pt(0, 0))
	if can, _ = MakeBlobKey(direct, fillPaint(), &m, nil, noSDFTControl()); can {
		t.Fatal("blob-less lists must not cache")
	}
}

func TestCoordinatorReuse(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "reuse", geom.Pt(5, 10))
	m := geom.IdentityMatrix()

	c := NewTextBlobRedrawCoordinator()
	blob1 := c.findOrCreateBlob(&m, list, fillPaint(), nil, noSDFTControl())
	if blob1 == nil || blob1.SubRunContainer().IsEmpty() {
		t.Fatal("blob creation failed")
	}
	if c.UsedBytes() == 0 {
		t.Fatal("cached blob must be accounted")
	}

	// The same draw reuses the cached blob; an integer translate does too.
	if blob2 := c.findOrCreateBlob(&m, list, fillPaint(), nil, noSDFTControl()); blob2 != blob1 {
		t.Fatal("identical draw must reuse the blob")
	}
	mInt := m
	mInt.PostTranslate(3, 9)
	if blob3 := c.findOrCreateBlob(&mInt, list, fillPaint(), nil, noSDFTControl()); blob3 != blob1 {
		t.Fatal("integer translate must reuse the blob")
	}

	// A scale change regenerates (and replaces) the cached blob.
	var scaled geom.Matrix
	scaled.SetScale(2, 2)
	blob4 := c.findOrCreateBlob(&scaled, list, fillPaint(), nil, noSDFTControl())
	if blob4 == blob1 {
		t.Fatal("scale change must regenerate the blob")
	}
	if blob5 := c.findOrCreateBlob(&scaled, list, fillPaint(), nil, noSDFTControl()); blob5 != blob4 {
		t.Fatal("regenerated blob must be cached")
	}

	c.FreeAll()
	if c.UsedBytes() != 0 {
		t.Fatalf("freeAll left %d bytes", c.UsedBytes())
	}
	if blob6 := c.findOrCreateBlob(&scaled, list, fillPaint(), nil, noSDFTControl()); blob6 == blob4 {
		t.Fatal("freeAll must drop cached blobs")
	}
}

// TestCoordinatorRemoveClearsVacatedSlot verifies that evicting one blob from a per-ID bucket that still holds another
// does not leave the evicted (or shifted-down) *Blob pinned in the backing array past the bucket's new length: the
// stored-back array must hold no Blob beyond len, so an evicted blob's subruns cannot stay reachable until the bucket
// empties.
func TestCoordinatorRemoveClearsVacatedSlot(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	list := runListForText(t, f, "evict", geom.Pt(2, 6))
	m := geom.IdentityMatrix()
	var scaled geom.Matrix
	scaled.SetScale(2, 2)

	c := NewTextBlobRedrawCoordinator()
	// Two different keys (identity and 2x) share the glyph run list's blob ID, so both land in the same bucket.
	blob1 := c.findOrCreateBlob(&m, list, fillPaint(), nil, noSDFTControl())
	blob2 := c.findOrCreateBlob(&scaled, list, fillPaint(), nil, noSDFTControl())
	if blob1 == nil || blob2 == nil || blob1 == blob2 {
		t.Fatal("expected two distinct cached blobs")
	}
	id := blob1.key.uniqueID
	if got := len(c.blobIDCache[id]); got != 2 {
		t.Fatalf("bucket len = %d, want 2", got)
	}

	c.remove(blob1)
	entry := c.blobIDCache[id]
	if len(entry) != 1 || entry[0] != blob2 {
		t.Fatalf("bucket = %v, want just the surviving blob", entry)
	}
	for i, b := range entry[:cap(entry)][len(entry):] {
		if b != nil {
			t.Fatalf("stored-back bucket pins a *Blob %d slot(s) past len", i+1)
		}
	}
}

func TestCoordinatorBudgetPurge(t *testing.T) {
	f := loadTestFont(t, "Roboto-Regular.ttf", 16)
	m := geom.IdentityMatrix()

	c := NewTextBlobRedrawCoordinator()
	c.sizeBudget = 1500 // a few small blobs

	lists := make([]*textblob.GlyphRunList, 8)
	blobs := make([]*Blob, 8)
	for i := range lists {
		lists[i] = runListForText(t, f, "purge", geom.Pt(float32(i), 0))
		blobs[i] = c.findOrCreateBlob(&m, lists[i], fillPaint(), nil, noSDFTControl())
	}
	if c.UsedBytes() > c.sizeBudget {
		t.Fatalf("over budget after purge: %d > %d", c.UsedBytes(), c.sizeBudget)
	}
	// The most recent blob survives.
	if got := c.findOrCreateBlob(&m, lists[7], fillPaint(), nil, noSDFTControl()); got != blobs[7] {
		t.Fatal("most recent blob must survive the purge")
	}
	// The oldest blob was evicted and regenerates.
	if got := c.findOrCreateBlob(&m, lists[0], fillPaint(), nil, noSDFTControl()); got == blobs[0] {
		t.Fatal("oldest blob must have been evicted")
	}
}
