// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// newTightenCapsResourceProvider builds a caps-only ResourceProvider whose Caps report no WritePixelsRowBytesSupport,
// forcing prepareLevels down its tight-copy lane.
func newTightenCapsResourceProvider(t *testing.T) *ResourceProvider {
	t.Helper()
	ctxInfo := appleM4MaxDriver().newContextInfo(t, nil)
	ctxInfo.Caps.WritePixelsRowBytesSupport = false
	return &ResourceProvider{gpu: &Gpu{ctx: ctxInfo}}
}

// TestPrepareLevelsShortPixelsRejected verifies that a MipLevel whose Pixels buffer is shorter than its declared
// RowBytes stride requires is treated as an unsupported write instead of panicking with an out-of-range slice in the
// tight-copy loop.
func TestPrepareLevelsShortPixelsRejected(t *testing.T) {
	rp := newTightenCapsResourceProvider(t)
	const (
		colorType = gpu.ColorTypeBGRA8888
		format    = FormatRGBA8
	)
	baseSize := geom.ISize{Width: 4, Height: 4}
	minRB := int(baseSize.Width) * colorType.BytesPerPixel() // 16
	actualRB := minRB + 16                                   // padded stride of 32
	// The last row starts at actualRB*(height-1) and spans minRB bytes, so a fully valid buffer needs
	// actualRB*(height-1)+minRB bytes. Provide one byte short of that.
	required := actualRB*(int(baseSize.Height)-1) + minRB
	texels := []gpu.MipLevel{{Pixels: make([]byte, required-1), RowBytes: actualRB}}

	// Must not panic; must report the write as unsupported.
	_, out, ok := rp.prepareLevels(format, colorType, baseSize, texels, 1)
	if ok {
		t.Fatalf("prepareLevels accepted a too-short Pixels buffer (len=%d, required=%d)", required-1, required)
	}
	if out != nil {
		t.Errorf("prepareLevels returned levels on failure: %v", out)
	}
}

// TestPrepareLevelsBaseOnlyMipmapped verifies that supplying only the base level (len(texels) == 1) while requesting a
// mipmapped upload (mipLevelCount > 1) does not index-panic on the absent upper levels; the missing levels are simply
// left empty for later generation.
func TestPrepareLevelsBaseOnlyMipmapped(t *testing.T) {
	rp := newTightenCapsResourceProvider(t)
	const (
		colorType = gpu.ColorTypeBGRA8888
		format    = FormatRGBA8
	)
	baseSize := geom.ISize{Width: 4, Height: 4}
	minRB := int(baseSize.Width) * colorType.BytesPerPixel() // 16
	// A single, base-level-only slice with a tight stride, but ask for a full mip chain (4x4 -> 3 levels).
	texels := []gpu.MipLevel{{Pixels: make([]byte, minRB*int(baseSize.Height)), RowBytes: minRB}}

	// Must not panic reading texels[1]/texels[2]; must succeed with only the base level populated.
	_, out, ok := rp.prepareLevels(format, colorType, baseSize, texels, 3)
	if !ok {
		t.Fatal("prepareLevels rejected a valid base-only mipmapped request")
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 levels, got %d", len(out))
	}
	if out[0].Pixels == nil {
		t.Error("base level Pixels was not populated")
	}
	for i := 1; i < len(out); i++ {
		if out[i].Pixels != nil {
			t.Errorf("upper level %d unexpectedly populated: %v", i, out[i])
		}
	}
}

// TestPrepareLevelsTightCopy verifies the tight-copy lane still produces correctly tightened level data when the Pixels
// buffer fully covers its declared stride.
func TestPrepareLevelsTightCopy(t *testing.T) {
	rp := newTightenCapsResourceProvider(t)
	const (
		colorType = gpu.ColorTypeBGRA8888
		format    = FormatRGBA8
	)
	baseSize := geom.ISize{Width: 4, Height: 4}
	minRB := int(baseSize.Width) * colorType.BytesPerPixel() // 16
	actualRB := minRB + 16                                   // padded stride of 32
	required := actualRB*(int(baseSize.Height)-1) + minRB
	src := make([]byte, required)
	// Fill each row's live region with a recognizable per-row byte; leave the padding zeroed.
	for y := 0; y < int(baseSize.Height); y++ {
		for x := 0; x < minRB; x++ {
			src[y*actualRB+x] = byte(y*minRB + x + 1)
		}
	}
	texels := []gpu.MipLevel{{Pixels: src, RowBytes: actualRB}}

	_, out, ok := rp.prepareLevels(format, colorType, baseSize, texels, 1)
	if !ok {
		t.Fatal("prepareLevels rejected a valid padded buffer")
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 level, got %d", len(out))
	}
	if out[0].RowBytes != minRB {
		t.Errorf("tightened RowBytes = %d, want %d", out[0].RowBytes, minRB)
	}
	if len(out[0].Pixels) != minRB*int(baseSize.Height) {
		t.Fatalf("tightened Pixels len = %d, want %d", len(out[0].Pixels), minRB*int(baseSize.Height))
	}
	for y := 0; y < int(baseSize.Height); y++ {
		for x := 0; x < minRB; x++ {
			want := byte(y*minRB + x + 1)
			if got := out[0].Pixels[y*minRB+x]; got != want {
				t.Fatalf("tight[%d][%d] = %d, want %d", y, x, got, want)
			}
		}
	}
}

// TestWritePixelsBackendFailureFailsCreation pins the recoverable-failure contract of the scratch-texture upload lane:
// a backend upload failure unrefs the texture and fails the creation, exactly like the prepareLevels failure path,
// rather than aborting the process.
func TestWritePixelsBackendFailureFailsCreation(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.Destroy()
	rp := dc.ResourceProvider()
	texture := rp.CreateTexture(geom.ISize{Width: 4, Height: 4}, FormatRGBA8, gpu.TextureType2D,
		gpu.RenderableNo, 1, gpu.MipmappedNo, gpu.BudgetedYes, "write-pixels-failure")
	if texture == nil {
		t.Fatal("CreateTexture failed")
	}
	if dc.ResourceCache().ResourceCount() != 1 {
		t.Fatalf("resources = %d, want 1", dc.ResourceCache().ResourceCount())
	}

	// An 8x8 base level into a 4x4 texture: prepareLevels accepts it (the stride is tight), then the backend rejects a
	// single-level write that is not contained in the surface.
	baseSize := geom.ISize{Width: 8, Height: 8}
	rowBytes := int(baseSize.Width) * gpu.ColorTypeRGBA8888.BytesPerPixel()
	texels := []gpu.MipLevel{{Pixels: make([]byte, rowBytes*int(baseSize.Height)), RowBytes: rowBytes}}
	if got := rp.writePixels(texture, gpu.ColorTypeRGBA8888, baseSize, texels, 1); got != nil {
		t.Fatal("writePixels must return nil when the backend upload fails")
	}

	// The texture's ref was dropped, so it purges like any other unreferenced scratch texture.
	dc.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
	if n := dc.ResourceCache().ResourceCount(); n != 0 {
		t.Fatalf("resources = %d after purge, want 0 (the failed texture's ref leaked)", n)
	}
}

// TestPatternedIndexBuffersAreBorrowed pins the ownership contract of findOrCreatePatternedIndexBuffer: it hands back a
// borrowed pointer, keeping exactly one ref (the creation ref) for the context's lifetime, exactly like
// RefNonAAQuadIndexBuffer. The cache-hit path goes through FindAndRefUniqueResource, which takes a ref; none of the ops
// that ask for these shared buffers ever unrefs, so a ref surviving the lookup accumulates once per prepared op and
// pins the buffer permanently non-purgeable with an unbounded count.
func TestPatternedIndexBuffersAreBorrowed(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.Destroy()
	rp := dc.ResourceProvider()

	for _, tc := range []struct {
		get  func() *Buffer
		name string
	}{
		{func() *Buffer { return getRRectIndexBuffer(rrectGeomFill, rp) }, "fill-rrect"},
		{func() *Buffer { return getRRectIndexBuffer(rrectGeomStroke, rp) }, "stroke-rrect"},
		{func() *Buffer { return getStrokeRectIndexBuffer(rp, true) }, "miter-stroke-rect"},
		{func() *Buffer { return getStrokeRectIndexBuffer(rp, false) }, "bevel-stroke-rect"},
		{func() *Buffer { return getQuadsIndexBuffer(rp) }, "aa-hairline-quads"},
		{func() *Buffer { return getLinesIndexBuffer(rp) }, "aa-hairline-lines"},
	} {
		first := tc.get()
		if first == nil {
			t.Fatalf("%s: index buffer creation failed", tc.name)
		}
		if !first.Unique() {
			t.Errorf("%s: fresh buffer is not uniquely held", tc.name)
		}
		// Every subsequent lookup is a cache hit; the shared buffer's ref count must not move.
		for i := range 4 {
			again := tc.get()
			if again != first {
				t.Fatalf("%s: lookup %d returned a different buffer", tc.name, i)
			}
			if !first.Unique() {
				t.Fatalf("%s: lookup %d leaked a ref (the buffer is no longer uniquely held)",
					tc.name, i)
			}
		}
	}
}

// mipTexels builds a full mip chain of solid pixel data for a square power-of-two texture.
func mipTexels(t *testing.T, side int32, colorType gpu.ColorType) []gpu.MipLevel {
	t.Helper()
	levels := make([]gpu.MipLevel, 0, gpu.ComputeMipLevelCount(side, side)+1)
	bpp := colorType.BytesPerPixel()
	for w, h := side, side; ; w, h = max(w/2, 1), max(h/2, 1) {
		levels = append(levels, gpu.MipLevel{
			Pixels:   make([]byte, int(w)*int(h)*bpp),
			RowBytes: int(w) * bpp,
		})
		if w == 1 && h == 1 {
			break
		}
	}
	return levels
}

// TestCreateTextureWithDataFullChainStaysClean: a caller-supplied mip chain must leave the texture's mipmaps clean.
// Reporting them dirty makes the next sample regenerate the chain with glGenerateMipmap, silently discarding the levels
// the caller uploaded. CreateTexture compensates for this locally; ResourceProvider.writePixels — the lane
// CreateTextureWithData takes with MippedYes — does not, so the contract has to hold in WritePixels itself.
func TestCreateTextureWithDataFullChainStaysClean(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.Destroy()
	rp := dc.ResourceProvider()
	const (
		side      = int32(8)
		colorType = gpu.ColorTypeRGBA8888
	)

	// Seed a purgeable mipmapped scratch texture so the CreateTextureWithData calls below take the scratch-reuse lane
	// (ResourceProvider.writePixels), not CreateTexture's own lane — CreateTexture compensates for the mip status
	// locally, so only the scratch lane exercises the contract in WritePixels itself.
	seedScratch := func() {
		t.Helper()
		seed := rp.CreateTexture(geom.ISize{Width: side, Height: side}, FormatRGBA8,
			gpu.TextureType2D, gpu.RenderableNo, 1, gpu.MipmappedYes, gpu.BudgetedYes, "scratch-seed")
		if seed == nil {
			t.Fatal("scratch seed creation failed")
		}
		seed.AsTexture().MarkMipmapsClean() // a recycled texture carries whatever status its last user left
		seed.Unref()                        // returns it to the cache as purgeable scratch
	}

	seedScratch()
	full := rp.CreateTextureWithData(geom.ISize{Width: side, Height: side}, FormatRGBA8,
		gpu.TextureType2D, colorType, gpu.RenderableNo, 1, gpu.BudgetedYes, gpu.MipmappedYes,
		mipTexels(t, side, colorType), "full-chain")
	if full == nil {
		t.Fatal("CreateTextureWithData with a full mip chain failed")
	}
	defer full.Unref()
	if full.AsTexture().MipmapsAreDirty() {
		t.Error("a caller-supplied full mip chain must leave the mipmaps clean, not scheduled " +
			"for regeneration")
	}

	// The base-only case is the opposite contract: the upper levels really do need generating.
	seedScratch()
	baseOnly := rp.CreateTextureWithData(geom.ISize{Width: side, Height: side}, FormatRGBA8,
		gpu.TextureType2D, colorType, gpu.RenderableNo, 1, gpu.BudgetedYes, gpu.MipmappedYes,
		mipTexels(t, side, colorType)[:1], "base-only")
	if baseOnly == nil {
		t.Fatal("CreateTextureWithData with base-level-only data failed")
	}
	defer baseOnly.Unref()
	if !baseOnly.AsTexture().MipmapsAreDirty() {
		t.Error("a base-level-only upload must leave the mipmaps dirty")
	}
}
