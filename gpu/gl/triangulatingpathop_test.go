// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for triangulatingPathOp's vertex-data lifetime: the ref createNonAAMesh takes on the ThreadSafeCache's
// vertex data must be dropped by the terminal chain delete even when the prepared op is never executed, or the entry
// could never become uniquely held and its GPU buffer would stay pinned for the context's lifetime.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/gpu"
)

// cachedTessVertsForTest mirrors createNonAAMesh's ref protocol: the op holds one ref on the vertex data and the cache
// holds another, so the entry is not uniquely held while the op is alive. Returns the op's ref.
func cachedTessVertsForTest(tb testing.TB, cache *ThreadSafeCache) *VertexData {
	tb.Helper()
	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, tessPathKeyDomain, 1, "Path")
	builder.Slice()[0] = 0xABCD
	builder.Finish()
	key.SetCustomData(tessInfoEncode(tessInfo{numVertices: 3, isLinear: true, tolerance: 0.25}))

	verts := MakeVertexDataFromBytes(make([]byte, 24), 3, 8)
	verts.Ref() // the copy handed to the cache
	tmpV, _ := cache.AddVertsWithData(&key, verts, tessIsNewerBetter)
	tmpV.Unref()
	if verts.refCnt != 2 {
		tb.Fatalf("refCnt = %d after caching, want 2 (op + cache)", verts.refCnt)
	}
	return verts
}

// A prepared-but-never-executed op still releases its vertex-data ref when its chain is deleted, so the cache entry
// becomes purgeable. OpsTask.OnExecute can bail after the prepare phase (a failed stencil attach, or a target that lost
// its instantiation between the prepare and execute passes), leaving deleteOps as the only release point.
func TestTriangulatingPathOpDeleteReleasesVertexDataWithoutExecute(t *testing.T) {
	cache := NewThreadSafeCache()
	verts := cachedTessVertsForTest(t, cache)

	o := &triangulatingPathOp{}
	o.InitOp(triangulatingPathOpClassID, o)
	o.vertexData = verts

	chain := makeOpChain(o, EmptyProcessorSetAnalysis(), nil, nil)
	cache.DropUniqueRefs(nil)
	if cache.NumEntries() != 1 {
		t.Fatalf("entries = %d while the op holds its ref, want 1", cache.NumEntries())
	}

	chain.deleteOps()
	if o.vertexData != nil {
		t.Fatal("deleteOps left the op holding its vertex data")
	}
	if verts.refCnt != 1 {
		t.Fatalf("refCnt = %d after deleteOps, want 1 (the cache's ref only)", verts.refCnt)
	}
	cache.DropUniqueRefs(nil)
	if cache.NumEntries() != 0 {
		t.Fatalf("entries = %d after the op died, want 0", cache.NumEntries())
	}
}

// The release is idempotent: an op whose OnExecute already dropped the ref must not drop a second one when the chain is
// deleted afterward.
func TestTriangulatingPathOpRecycleAfterExecuteIsIdempotent(t *testing.T) {
	cache := NewThreadSafeCache()
	verts := cachedTessVertsForTest(t, cache)

	o := &triangulatingPathOp{}
	o.InitOp(triangulatingPathOpClassID, o)
	o.vertexData = verts

	// Stand in for the release OnExecute's deferred call performs once the draw has been submitted.
	o.releaseVertexData()
	if verts.refCnt != 1 {
		t.Fatalf("refCnt = %d after the execute-side release, want 1", verts.refCnt)
	}

	var op Op = o
	op.recycle()
	if verts.refCnt != 1 {
		t.Fatalf("refCnt = %d after recycling an already-released op, want 1", verts.refCnt)
	}
	if cache.NumEntries() != 1 {
		t.Fatalf("entries = %d, want the cache's own entry to survive", cache.NumEntries())
	}
}
