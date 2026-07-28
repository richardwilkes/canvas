// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for a handful of invariants that are easy to regress because nothing else in the suite observes them
// directly: chopConic's untouched output slots, the CPU buffer cache's reference balance, ShaderSourceStrings' empty
// source list, the compile-failure cleanup path, SurfaceProxyView accessors on a proxy-less view, the round rect inner
// bounds safety margin, the vertex chunk builder's behavior after a failed pool allocation, and the explicit-proxy
// flush short-circuit's gating.

package gl

import (
	"testing"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// TestChopConicLeavesUnwrittenSlotsAlone pins that chopConic only writes the slots the returned count covers. A conic
// whose max curvature is outside (0, 1) is never split, so the second and later slots must come back exactly as the
// caller left them rather than as zero-valued conics.
func TestChopConicLeavesUnwrittenSlotsAlone(t *testing.T) {
	sentinel := geom.MakeConic(geom.Pt(-1, -1), geom.Pt(-2, -2), geom.Pt(-3, -3), 7)
	dst := [4]geom.Conic{sentinel, sentinel, sentinel, sentinel}

	// A straight, symmetric conic has no interior max curvature, so it comes back whole.
	src := []geom.Point{geom.Pt(0, 0), geom.Pt(10, 0), geom.Pt(20, 0)}
	if cnt := chopConic(src, &dst, 0.5); cnt != 1 {
		t.Fatalf("conic count = %d, want 1", cnt)
	}
	for i := 1; i < len(dst); i++ {
		if dst[i] != sentinel {
			t.Errorf("dst[%d] = %v, want the untouched sentinel %v", i, dst[i], sentinel)
		}
	}

	// Conics that split, but where at least one of the two subsections does not split again, so the reported count is
	// short of the four slots. Everything past the count must still hold the sentinel.
	for _, tc := range []struct {
		pts     []geom.Point
		weight  float32
		wantCnt int
	}{
		{pts: []geom.Point{geom.Pt(0, 0), geom.Pt(3, 3), geom.Pt(2, 7)}, weight: 0.2, wantCnt: 2},
		{pts: []geom.Point{geom.Pt(0, 0), geom.Pt(3, 9), geom.Pt(6, 7)}, weight: 0.2, wantCnt: 3},
	} {
		dst = [4]geom.Conic{sentinel, sentinel, sentinel, sentinel}
		if cnt := chopConic(tc.pts, &dst, tc.weight); cnt != tc.wantCnt {
			t.Fatalf("conic count for %v = %d, want %d", tc.pts, cnt, tc.wantCnt)
		}
		for i := tc.wantCnt; i < len(dst); i++ {
			if dst[i] != sentinel {
				t.Errorf("%v: dst[%d] past the reported count = %v, want the untouched sentinel %v",
					tc.pts, i, dst[i], sentinel)
			}
		}
	}
}

// TestCpuBufferCacheRefBalance pins that a buffer handed out by the cache carries exactly the refs it should: an
// uncached buffer holds only the caller's, so the caller's single Unref releases it, while a cached one additionally
// holds the cache's.
func TestCpuBufferCacheRefBalance(t *testing.T) {
	c := MakeCpuBufferCache(2)

	// Non-default sizes are never cached, so the buffer belongs solely to the caller.
	b := c.MakeBuffer(DefaultBufferSize/2, false)
	if !b.Unique() {
		t.Error("an uncached buffer must be unique when handed out")
	}
	b.Unref() // must not panic; without the balance this would leave a phantom ref instead

	// The same holds once the cache is saturated.
	held := []*CpuBuffer{
		c.MakeBuffer(DefaultBufferSize, false),
		c.MakeBuffer(DefaultBufferSize, false),
	}
	for i, h := range held {
		if h.Unique() {
			t.Errorf("cached buffer %d must hold the cache's ref as well as the caller's", i)
		}
	}
	overflow := c.MakeBuffer(DefaultBufferSize, false)
	if !overflow.Unique() {
		t.Error("a buffer handed out past the cache's capacity must be unique")
	}
	overflow.Unref()

	// A returned cached buffer is unique again and is the one the cache hands back next.
	held[0].Unref()
	if !held[0].Unique() {
		t.Error("a released cached buffer must be unique again")
	}
	if got := c.MakeBuffer(DefaultBufferSize, false); got != held[0] {
		t.Error("the cache must reuse the released buffer")
	}
	c.ReleaseAll()
}

// TestCpuBufferCacheZeroesFreshBuffers pins that a caller asking for initialized memory gets it on the uncached path
// too, where the zeroing comes from the fresh allocation rather than an explicit clear.
func TestCpuBufferCacheZeroesFreshBuffers(t *testing.T) {
	c := MakeCpuBufferCache(0)
	b := c.MakeBuffer(64, true)
	for i, v := range b.Data() {
		if v != 0 {
			t.Fatalf("byte %d = %d, want 0", i, v)
		}
	}
}

// shaderSourceCall records the arguments of the most recent glShaderSource call made through fakeShaderProcs.
var shaderSourceCall struct {
	count  int32
	str    uintptr
	length uintptr
}

// fakeShaderProcs holds the shader-pipeline callbacks, created once because purego callbacks are never freed.
var fakeShaderProcs struct {
	createShader  uintptr
	shaderSource  uintptr
	compileShader uintptr
	getShaderiv   uintptr
	attachShader  uintptr
	deleteShader  uintptr
	deleteProgram uintptr
	ready         bool
}

// fakeCompileStatus is what the fake glGetShaderiv reports for GL_COMPILE_STATUS.
var fakeCompileStatus int32

// installFakeShaderProcs points the shader-compilation entry points at recording callbacks that report
// fakeCompileStatus as the compile result.
func installFakeShaderProcs(g *Gpu) {
	if !fakeShaderProcs.ready {
		fakeShaderProcs.createShader = purego.NewCallback(func(_ uintptr) uintptr { return 11 })
		fakeShaderProcs.shaderSource = purego.NewCallback(
			func(_, count, str, length uintptr) uintptr {
				recCounts["glShaderSource"]++
				shaderSourceCall.count = int32(count)
				shaderSourceCall.str = str
				shaderSourceCall.length = length
				return 0
			},
		)
		fakeShaderProcs.compileShader = purego.NewCallback(func(_ uintptr) uintptr { return 0 })
		fakeShaderProcs.getShaderiv = purego.NewCallback(func(_, pname, params uintptr) uintptr {
			v := int32(0)
			if pname == COMPILE_STATUS {
				v = fakeCompileStatus
			}
			*(*int32)(ptrFromUintptr(params)) = v
			return 0
		})
		fakeShaderProcs.attachShader = purego.NewCallback(func(_, _ uintptr) uintptr { return 0 })
		fakeShaderProcs.deleteShader = purego.NewCallback(func(_ uintptr) uintptr {
			recCounts["glDeleteShader"]++
			return 0
		})
		fakeShaderProcs.deleteProgram = purego.NewCallback(func(_ uintptr) uintptr {
			recCounts["glDeleteProgram"]++
			return 0
		})
		fakeShaderProcs.ready = true
	}
	f := g.fns()
	f.createShader = fakeShaderProcs.createShader
	f.shaderSource = fakeShaderProcs.shaderSource
	f.compileShader = fakeShaderProcs.compileShader
	f.getShaderiv = fakeShaderProcs.getShaderiv
	f.attachShader = fakeShaderProcs.attachShader
	f.deleteShader = fakeShaderProcs.deleteShader
	f.deleteProgram = fakeShaderProcs.deleteProgram
}

// TestShaderSourceStringsEmpty pins that an empty source list is passed through as the legal
// glShaderSource(shader, 0, NULL, NULL) rather than indexing off the end of the (empty) pointer slices.
func TestShaderSourceStringsEmpty(t *testing.T) {
	g := newRecordingGpu(t)
	installFakeShaderProcs(g)

	shaderSourceCall.count = -1
	g.fns().ShaderSourceStrings(7, nil)
	if shaderSourceCall.count != 0 {
		t.Errorf("count = %d, want 0", shaderSourceCall.count)
	}
	if shaderSourceCall.str != 0 || shaderSourceCall.length != 0 {
		t.Errorf("string/length pointers = %#x/%#x, want both NULL", shaderSourceCall.str,
			shaderSourceCall.length)
	}

	// A non-empty list still passes real pointers and lengths.
	g.fns().ShaderSourceStrings(7, []string{"void main() {}"})
	if shaderSourceCall.count != 1 {
		t.Errorf("count = %d, want 1", shaderSourceCall.count)
	}
	if shaderSourceCall.str == 0 || shaderSourceCall.length == 0 {
		t.Error("a non-empty source list must pass non-NULL string and length pointers")
	}
}

// TestCompileFailureDeletesProgramAndShaders pins that a compile failure tears down every GL object created so far
// before it panics, so a recovered panic leaks neither the program nor the shaders already attached to it.
func TestCompileFailureDeletesProgramAndShaders(t *testing.T) {
	for _, tc := range []struct {
		name            string
		alreadyAttached []uint32
		wantShaders     int
	}{
		{name: "first shader", wantShaders: 1},
		{name: "second shader", alreadyAttached: []uint32{21}, wantShaders: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := newRecordingGpu(t)
			installFakeShaderProcs(g)
			fakeCompileStatus = 0

			pb := &ProgramBuilder{gpu: g}
			shadersToDelete := append([]uint32(nil), tc.alreadyAttached...)
			func() {
				defer func() {
					if recover() == nil {
						t.Error("a failed compile must panic")
					}
				}()
				pb.compileAndAttachShader(31, FRAGMENT_SHADER, []byte("bad source"),
					&shadersToDelete)
			}()

			if got := counts("glDeleteShader"); got != tc.wantShaders {
				t.Errorf("glDeleteShader calls = %d, want %d", got, tc.wantShaders)
			}
			if got := counts("glDeleteProgram"); got != 1 {
				t.Errorf("glDeleteProgram calls = %d, want 1", got)
			}
		})
	}
}

// TestSurfaceProxyViewMipmappedWithoutProxy pins that the zero-value and Reset views report MipmappedNo, as the doc
// promises, instead of dereferencing a nil proxy.
func TestSurfaceProxyViewMipmappedWithoutProxy(t *testing.T) {
	var v SurfaceProxyView
	if got := v.Mipmapped(); got != gpu.MipmappedNo {
		t.Errorf("zero-value Mipmapped = %v, want MipmappedNo", got)
	}
	if v.AsTextureProxy() != nil || v.AsRenderTargetProxy() != nil {
		t.Error("a proxy-less view must report no texture or render target proxy")
	}
}

// TestRRectInnerBoundsStayInsideCorners pins the safety margin on the corner-inscribed inner bounds: every corner of
// the returned rect must satisfy the round rect's own containment test. Each case below inscribes exactly on the
// corner ellipse without the margin, where float32 error pushes the corner marginally outside and the containment the
// clip stack relies on fails.
func TestRRectInnerBoundsStayInsideCorners(t *testing.T) {
	for _, size := range []struct{ w, r float32 }{
		{w: 16, r: 0.12},
		{w: 16, r: 0.16},
		{w: 32, r: 0.24},
		{w: 64, r: 0.48},
		{w: 100, r: 0.5},
		{w: 100, r: 1},
		{w: 128, r: 1.28},
		{w: 200, r: 2},
		{w: 256, r: 1.92},
		{w: 320, r: 2.4},
	} {
		rr := geom.MakeRRect(geom.RectLTRB(0, 0, size.w, size.w), size.r, size.r)
		inner := rrectInnerBounds(rr)
		if inner.IsEmpty() {
			t.Fatalf("inner bounds for %v square r=%v are empty", size.w, size.r)
		}
		if !rrectContainsRect(rr, inner) {
			t.Errorf("inner bounds %v for %v square r=%v are not contained by the round rect", inner,
				size.w, size.r)
		}
	}
}

// TestVertexChunkBuilderCloseAfterFailedAlloc pins that close() does not rewrite the last chunk's count once a pool
// allocation has failed: that chunk was finalized by allocChunk and its geometry is already in the GPU buffer, so
// resetting the count to zero would silently drop it.
func TestVertexChunkBuilderCloseAfterFailedAlloc(t *testing.T) {
	chunks := make([]vertexChunk, 0, 1)
	b := newVertexChunkBuilder(nil, &chunks, 4, 8)

	// Stand in for a successfully written chunk that allocChunk has already finalized, followed by a failed
	// allocation: no chunk is open, so close() must be a no-op (and must not touch the nil target).
	chunks = append(chunks, vertexChunk{count: 12, base: 0})
	b.currChunkVertexCount = 0
	b.currChunkVertexCapacity = 0
	b.currChunkOpen = false
	b.close()
	if chunks[0].count != 12 {
		t.Errorf("chunk count after a failed alloc = %d, want the finalized 12", chunks[0].count)
	}
}

// TestFlushShortCircuitCallsSubmittedProc pins the explicit-proxy short-circuit gate: a flush of proxies no task
// touches skips the work and reports success to the submitted proc, while a finished proc — which has to observe a
// real submission — forces the full flush instead.
func TestFlushShortCircuitCallsSubmittedProc(t *testing.T) {
	dc := newFakeDirectContext(t)
	defer dc.ReleaseResourcesAndAbandonContext()
	proxy := makeDeferredProxy(t, dc, geom.ISize{Width: 16, Height: 16}, gpu.RenderableNo,
		gpu.BackingFitExact)
	defer proxy.Unref()

	submitted := 0
	submittedOK := false
	flushed := dc.DrawingManager().Flush([]*SurfaceProxy{proxy}, FlushInfo{
		SubmittedProc: func(success bool) {
			submitted++
			submittedOK = success
		},
	})
	if flushed {
		t.Error("a flush of untouched proxies must report no work done")
	}
	if submitted != 1 || !submittedOK {
		t.Errorf("submitted proc calls = %d (success=%v), want 1 (success=true)", submitted,
			submittedOK)
	}

	// A finished proc has to observe a real submission, so the same call cannot short-circuit; both procs are handed to
	// the submit/finish machinery instead of being answered inline.
	submitted = 0
	if !dc.DrawingManager().Flush([]*SurfaceProxy{proxy}, FlushInfo{
		SubmittedProc: func(bool) { submitted++ },
		FinishedProc:  func() {},
	}) {
		t.Fatal("a flush carrying a finished proc must not short-circuit")
	}
	if submitted != 0 {
		t.Errorf("submitted proc calls before submit = %d, want 0", submitted)
	}
	dc.DrawingManager().SubmitToGpu(false)
	if submitted != 1 {
		t.Errorf("submitted proc calls after submit = %d, want 1", submitted)
	}
}
