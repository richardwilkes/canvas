// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the blur engine's context ownership: every intermediate SurfaceDrawContext the blur lanes
// create (the X pass's result, the rescale ladder's steps, the rescaled source, the reduced-sigma blur feeding the
// re-expand, and the mask-filter lane's generated mask) must be released once the draws reading its texture have been
// recorded. A surviving context ref keeps its proxy reffed, which keeps the backing texture out of both the resource
// cache's purgeable pool and the resource allocator's recycling test, so GPU memory would grow with every blurred draw.
// These live in package gl because the assertions read the drawing manager's task DAG and raw proxy ref counts.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/shaders"
)

// dagTargetRefs maps every proxy targeted by a task in the drawing manager's DAG to the number of tasks targeting it.
// Each such task holds exactly one ref on its target (RenderTaskBase.addTarget), and those refs are what keep a
// recorded blur's textures alive until flush, so this count is also the ref count each proxy should carry once every
// context that produced it has been released.
func dagTargetRefs(dm *DrawingManager) map[*SurfaceProxy]int32 {
	refs := make(map[*SurfaceProxy]int32)
	for _, task := range dm.dag {
		base := task.taskBase()
		for i := 0; i < base.NumTargets(); i++ {
			refs[base.Target(i)]++
		}
	}
	return refs
}

// checkNoExtraProxyRefs asserts that every proxy the DAG targets is held only by its tasks, allowing one extra ref for
// each proxy in 'owned' (contexts/views the caller still owns).
func checkNoExtraProxyRefs(t *testing.T, dm *DrawingManager, owned ...*SurfaceProxy) map[*SurfaceProxy]int32 {
	t.Helper()
	refs := dagTargetRefs(dm)
	for proxy, tasks := range refs {
		want := tasks
		for _, o := range owned {
			if proxy == o {
				want++
			}
		}
		if got := proxy.RefCnt(); got != want {
			t.Errorf("proxy %q ref count = %d, want %d: a context that produced it was never released",
				proxy.Label(), got, want)
		}
	}
	return refs
}

// TestGaussianBlurReleasesIntermediateContexts pins that the blur engine leaves no intermediate context alive: after a
// blur is recorded, every proxy in the task DAG carries only its tasks' refs, apart from the one result context the
// caller owns and releases itself.
func TestGaussianBlurReleasesIntermediateContexts(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode shaders.TileMode
		// sigma selects the lane: a tiny one takes the single-pass 2D kernel, a small one the separable X-then-Y
		// passes, and a large one the downscale→blur→re-expand ladder.
		sigma float32
		// minTargets is the smallest number of distinct render targets the lane must produce, so a lane that stopped
		// creating intermediates (and would therefore pass vacuously) fails instead.
		minTargets int
	}{
		{name: "2d", mode: shaders.TileDecal, sigma: 0.6, minTargets: 1},
		{name: "two-pass", mode: shaders.TileDecal, sigma: 1.5, minTargets: 2},
		{name: "rescale-ladder-decal", mode: shaders.TileDecal, sigma: 24, minTargets: 4},
		{name: "rescale-ladder-clamp", mode: shaders.TileClamp, sigma: 24, minTargets: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dc := newFakeDirectContext(t)
			defer dc.ReleaseResourcesAndAbandonContext()
			const size = 64
			src := makeDeferredProxy(t, dc, geom.ISize{Width: size, Height: size}, gpu.RenderableNo,
				gpu.BackingFitExact)
			defer src.Unref()

			bounds := geom.IRectWH(size, size)
			result := GaussianBlur(dc, MakeSurfaceProxyViewDefault(src), gpu.ColorTypeRGBA8888,
				gpu.AlphaTypePremul, bounds, bounds, tc.sigma, tc.sigma, tc.mode,
				gpu.BackingFitApprox)
			if result == nil {
				t.Fatal("GaussianBlur returned nil")
			}
			resultProxy := result.AsSurfaceProxy()

			refs := checkNoExtraProxyRefs(t, dc.DrawingManager(), resultProxy)
			if len(refs) < tc.minTargets {
				t.Fatalf("the blur recorded draws into %d render targets, want at least %d: this lane no longer builds the intermediates under test",
					len(refs), tc.minTargets)
			}
			if _, ok := refs[resultProxy]; !ok {
				t.Fatal("the returned context's proxy is not the target of any recorded task")
			}

			// The source is only sampled, never targeted, so the blur must not have reffed it either.
			if got := src.RefCnt(); got != 1 {
				t.Errorf("blur source proxy ref count = %d, want the test's own 1", got)
			}

			result.Release()
			if got := resultProxy.RefCnt(); got != refs[resultProxy] {
				t.Errorf("result proxy ref count after Release = %d, want %d", got, refs[resultProxy])
			}
		})
	}
}

// TestFilterBlurAlgorithmReleasesItsBlurContext pins the image-filter lane's half of the same contract: the special
// image the blur algorithm hands back only borrows the blurred texture (borrowTextureImage takes no ref), so the blur
// context has to be released once the blur has been recorded.
func TestFilterBlurAlgorithmReleasesItsBlurContext(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.ReleaseResourcesAndAbandonContext()
	const size = 64
	dst := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: size, Height: size},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes, "filter-dest")
	if dst == nil {
		t.Fatal("MakeSurfaceDrawContext returned nil")
	}
	defer dst.Release()

	src := makeDeferredProxy(t, dc, geom.ISize{Width: size, Height: size}, gpu.RenderableNo,
		gpu.BackingFitExact)
	defer src.Unref()
	// A texture backing on this context is sampled in place, so the blur takes the no-upload lane.
	special := filtercore.NewSpecialImageDrawable(geom.IRectWH(size, size),
		borrowTextureImage(dc, MakeSurfaceProxyViewDefault(src), gpu.ColorTypeRGBA8888,
			geom.ISize{Width: size, Height: size}))
	if special == nil || special.DrawableBacking() == nil {
		t.Fatal("the test source must be texture-backed")
	}

	bounds := geom.IRectWH(size, size)
	algorithm := &blurAlgorithm{dev: NewDevice(dst)}
	if out := algorithm.Blur(geom.Size{Width: 2, Height: 2}, special, bounds, shaders.TileDecal,
		bounds); out == nil {
		t.Fatal("Blur returned nil")
	}

	checkNoExtraProxyRefs(t, dc.DrawingManager(), dst.AsSurfaceProxy())
	if got := src.RefCnt(); got != 1 {
		t.Errorf("blur source proxy ref count = %d, want the test's own 1", got)
	}
}

// TestMaskFilterHWLaneReleasesItsContexts pins the same invariant for the GPU mask-filter lane, whose generated A8 mask
// (createMaskGPU) and blurred result (filterMask) are both intermediates the draw does not return. The rotated matrix
// keeps computeKeyAndClipBounds from caching the filtered mask, so the thread-safe cache's deliberate long-lived ref
// does not mask a leaked context ref here.
func TestMaskFilterHWLaneReleasesItsContexts(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.ReleaseResourcesAndAbandonContext()
	const size = 512
	dst := MakeSurfaceDrawContext(dc, gpu.ColorTypeRGBA8888, geom.ISize{Width: size, Height: size},
		gpu.BackingFitExact, 1, gpu.MipmappedNo, gpu.OriginTopLeft, gpu.BudgetedYes,
		"mask-filter-dest")
	if dst == nil {
		t.Fatal("MakeSurfaceDrawContext returned nil")
	}
	defer dst.Release()
	dst.Clear([4]float32{})

	// A diamond is rejected by every analytic blur profile, so the draw falls through to the mask lanes; at well over
	// 64px it takes the GPU one.
	p := path.New()
	p.MoveTo(256, 80)
	p.LineTo(432, 256)
	p.LineTo(256, 432)
	p.LineTo(80, 256)
	p.Close()
	shape := MakeStyledShapePath(p, SimpleFillStyle(), DoSimplifyYes)

	paint := NewPaint()
	paint.SetColor4f(colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1})
	mf := maskfilter.NewBlur(maskfilter.BlurNormal, 8, true)
	if mf == nil {
		t.Fatal("NewBlur returned nil")
	}
	vm := geom.RotateDegMatrix(20)
	if lane := drawShapeWithMaskFilter(dst, nil, paint, &vm, mf, &shape); lane != maskFilterLaneHW {
		t.Fatalf("mask-filter lane = %v, want the GPU (hw) lane", lane)
	}

	checkNoExtraProxyRefs(t, dc.DrawingManager(), dst.AsSurfaceProxy())
}
