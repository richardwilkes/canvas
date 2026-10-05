// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context tests for the lifetime of thread-safe-cache view hits. Once a cached view's proxy has been flushed it is
// instantiated and its lazy callback is gone, so the only things keeping its texture alive are the cache entry and the
// recording ref the cache hands out with every find or add. These tests record a cache hit, then drop the cache's
// uniquely held entries before the hit is flushed (an explicit purge mid-recording, and a budget purge from inside the
// flush's own resource allocation), and require the hit to render exactly like the first draw. Skips when no GL context
// is available.

package gl_test

import (
	"bytes"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/maskfilter"
)

// tscLiveLane is one mask-filter lane whose output view lives in the thread-safe cache.
type tscLiveLane struct {
	shape func() gl.StyledShape
	name  string
	lane  string
	style maskfilter.BlurStyle
	sigma float32
}

var tscLiveLanes = []tscLiveLane{
	{
		// A Normal-blurred circle takes the analytic circle lane, which caches a lazy-upload profile texture.
		name: "analytic-circle",
		lane: gl.MaskFilterLaneDirectName,
		shape: func() gl.StyledShape {
			circle := geom.MakeRRect(geom.RectLTRB(40, 40, 120, 120), 40, 40)
			return gl.MakeStyledShapeRRect(circle, gl.SimpleFillStyle(), gl.DoSimplifyYes)
		},
		style: maskfilter.BlurNormal,
		sigma: 6,
	},
	{
		// A small Solid-blurred rect takes the SW lane, which caches a lazy-upload filtered mask.
		name: "sw-mask",
		lane: gl.MaskFilterLaneSWName,
		shape: func() gl.StyledShape {
			return gl.MakeStyledShapeRect(geom.RectLTRB(70, 70, 90, 90), gl.SimpleFillStyle(), gl.DoSimplifyYes)
		},
		style: maskfilter.BlurSolid,
		sigma: 3,
	},
	{
		// A large Solid-blurred rect takes the HW lane, which caches a GPU-rendered (not lazy) filtered mask.
		name: "hw-mask",
		lane: gl.MaskFilterLaneHWName,
		shape: func() gl.StyledShape {
			return gl.MakeStyledShapeRect(geom.RectLTRB(30, 30, 130, 130), gl.SimpleFillStyle(), gl.DoSimplifyYes)
		},
		style: maskfilter.BlurSolid,
		sigma: 8,
	},
}

const tscLiveSize = 160

// recordBlurredShape records (without flushing) a premul-white draw of the lane's shape through its blur mask filter
// into a freshly cleared destination, returning the destination and the lane the draw took.
func (l *tscLiveLane) record(t *testing.T, dc *gl.DirectContext) (sdc *gl.SurfaceDrawContext, lane string) {
	t.Helper()
	sdc = maskFilterDest(t, dc, tscLiveSize)
	paint := gl.NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	mf := maskfilter.NewBlur(l.style, l.sigma, true)
	if mf == nil {
		t.Fatal("NewBlur returned nil")
	}
	shape := l.shape()
	vm := geom.IdentityMatrix()
	return sdc, gl.DrawShapeWithMaskFilterLaneForTest(sdc, nil, paint, &vm, mf, &shape)
}

// drawFirst renders the lane once (a cache miss that adds the entry) and flushes it via readback, which instantiates
// the cached proxy and releases any lazy callback it had. The destination is returned unreleased, so its texture stays
// in use and a later destination of the same size needs a freshly allocated one.
func (l *tscLiveLane) drawFirst(t *testing.T, dc *gl.DirectContext) (first []byte, sdc *gl.SurfaceDrawContext) {
	t.Helper()
	cache := dc.ThreadSafeCache()
	adds := cache.Stats().Adds
	var lane string
	sdc, lane = l.record(t, dc)
	if lane != l.lane {
		t.Fatalf("first draw took lane %q, want %q", lane, l.lane)
	}
	if cache.Stats().Adds != adds+1 {
		t.Fatalf("first draw made %d cache adds, want 1", cache.Stats().Adds-adds)
	}
	first = readSDC(t, sdc)
	if c := chanAt(first, tscLiveSize, tscLiveSize/2, tscLiveSize/2); c != 255 {
		t.Fatalf("first draw center coverage = %d, want 255", c)
	}
	return first, sdc
}

// recordHit records the lane again, requiring a cache hit.
func (l *tscLiveLane) recordHit(t *testing.T, dc *gl.DirectContext) *gl.SurfaceDrawContext {
	t.Helper()
	cache := dc.ThreadSafeCache()
	hits := cache.Stats().Hits
	sdc, lane := l.record(t, dc)
	if lane != l.lane {
		t.Fatalf("second draw took lane %q, want %q", lane, l.lane)
	}
	if cache.Stats().Hits <= hits {
		t.Fatal("second draw did not hit the thread-safe cache")
	}
	return sdc
}

// TestLiveThreadSafeCacheHitSurvivesPurge purges every unlocked resource between recording a cache hit and flushing it.
// The purge drops all uniquely held cache entries; a hit recorded in the pending flush must not be one of them, or its
// instantiated proxy loses its texture and the draw comes out blank.
func TestLiveThreadSafeCacheHitSurvivesPurge(t *testing.T) {
	for i := range tscLiveLanes {
		l := &tscLiveLanes[i]
		t.Run(l.name, func(t *testing.T) {
			_, dc := newLiveDirectContext(t)
			first, firstSDC := l.drawFirst(t, dc)
			defer firstSDC.Release()

			sdc := l.recordHit(t, dc)
			dc.PurgeUnlockedResources(gpu.PurgeAllResources)
			second := readSDC(t, sdc)
			sdc.Release()
			if !bytes.Equal(first, second) {
				t.Fatalf("hit purged mid-recording rendered differently: center %d, want %d",
					chanAt(second, tscLiveSize, tscLiveSize/2, tscLiveSize/2),
					chanAt(first, tscLiveSize, tscLiveSize/2, tscLiveSize/2))
			}

			// The flush released the recording refs, so the entry is uniquely held again and an idle purge reclaims it.
			before := dc.ThreadSafeCache().NumEntries()
			dc.PurgeUnlockedResources(gpu.PurgeAllResources)
			if after := dc.ThreadSafeCache().NumEntries(); after >= before {
				t.Fatalf("entries after an idle purge = %d, want fewer than %d", after, before)
			}
		})
	}
}

// TestLiveThreadSafeCacheHitSurvivesFlushBudgetPurge records a cache hit, then shrinks the budget to just above the
// current usage so that the flush's own allocation of the destination texture pushes the cache over budget. The
// resulting budget purge runs from inside the resource allocator's assignment pass, after the hit's already
// instantiated proxy was planned without a register, and must not drop the hit's entry.
func TestLiveThreadSafeCacheHitSurvivesFlushBudgetPurge(t *testing.T) {
	for i := range tscLiveLanes {
		l := &tscLiveLanes[i]
		t.Run(l.name, func(t *testing.T) {
			_, dc := newLiveDirectContext(t)
			first, firstSDC := l.drawFirst(t, dc)
			defer firstSDC.Release()

			sdc := l.recordHit(t, dc)
			rc := dc.ResourceCache()
			limit := rc.MaxResourceBytes()
			rc.SetLimit(rc.BudgetedResourceBytes() + 1)
			second := readSDC(t, sdc)
			rc.SetLimit(limit)
			sdc.Release()
			if !bytes.Equal(first, second) {
				t.Fatalf("hit purged mid-flush rendered differently: center %d, want %d",
					chanAt(second, tscLiveSize, tscLiveSize/2, tscLiveSize/2),
					chanAt(first, tscLiveSize, tscLiveSize/2, tscLiveSize/2))
			}
		})
	}
}
