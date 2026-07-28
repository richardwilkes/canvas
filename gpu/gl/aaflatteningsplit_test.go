// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the AA-flattening op's split lane: a single path whose tessellation needs more vertices than a uint16 index
// can address must still come out as correct triangles, spread over as many draws as it takes.

package gl

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/stroke"
)

// oversizedTessellation synthesizes a tessellation result with n vertices — more than a uint16 index can address —
// triangulated as a strip so consecutive triangles share vertices and runs of them straddle the chunk boundaries. Only
// the result accessors the split lane uses (numPts/numIndices/index/point/coverage) need to be consistent, which is why
// the state is filled in directly instead of tessellating a path: the tessellator's near-colinear simplification keeps a
// convex polygon's surviving point count far below this at coordinates float32 can resolve.
func oversizedTessellation(n int) *aaConvexTessellator {
	tess := newAAConvexTessellator(stroke.StyleFill, -1, stroke.JoinBevel, 0)
	for i := 0; i < n; i++ {
		tess.pts = append(tess.pts, geom.Point{X: float32(i), Y: float32(i % 977)})
		tess.coverages = append(tess.coverages, float32(i%3)/2)
		tess.movable = append(tess.movable, false)
		tess.curveState = append(tess.curveState, aaTessCurveSharp)
	}
	for i := 0; i+2 < n; i++ {
		tess.indices = append(tess.indices, i, i+1, i+2)
	}
	return tess
}

// TestAAFlatteningSplitChunksCoverAllTriangles verifies that the split lane reproduces the whole tessellation: every
// triangle appears exactly once, in order, with the original vertex positions and coverages, and every emitted index
// addresses a vertex inside its own chunk (the wraparound the single-draw lane produced past 65535 vertices).
func TestAAFlatteningSplitChunksCoverAllTriangles(t *testing.T) {
	tess := oversizedTessellation(aaFlatteningMaxVertices + 5000)
	if tess.numPts() <= aaFlatteningMaxVertices {
		t.Fatalf("tessellation has %d vertices; the split lane needs more than %d", tess.numPts(),
			aaFlatteningMaxVertices)
	}

	// The layout createLinesOnlyGP declares for a non-wide color without local coords: position, 4 color bytes, coverage.
	const stride = 2*4 + 4 + 4
	color := colorcore.PMColor4f{R: 1, A: 1}
	b := aaFlatteningBatch{vertexStride: stride}

	numTris := tess.numIndices() / 3
	nextTri := 0
	chunks := 0
	for nextTri < numTris {
		tris := aaFlatteningBuildSplitChunk(&b, tess, nextTri, nil, color, false)
		if tris <= 0 {
			t.Fatalf("chunk %d staged no triangles at triangle %d", chunks, nextTri)
		}
		if b.vertexCount > aaFlatteningMaxVertices {
			t.Fatalf("chunk %d staged %d vertices, past the %d a uint16 index can address", chunks,
				b.vertexCount, aaFlatteningMaxVertices)
		}
		if b.indexCount != 3*tris {
			t.Fatalf("chunk %d staged %d indices for %d triangles", chunks, b.indexCount, tris)
		}
		for i := 0; i < b.indexCount; i++ {
			idx := int(b.indices[i])
			if idx >= b.vertexCount {
				t.Fatalf("chunk %d index %d = %d, past its %d vertices", chunks, i, idx,
					b.vertexCount)
			}
			// The vertex the chunk's index reaches must be the one the tessellation named.
			src := tess.index(3*nextTri + i)
			off := idx * stride
			gotX := math.Float32frombits(binary.LittleEndian.Uint32(b.vertices[off:]))
			gotY := math.Float32frombits(binary.LittleEndian.Uint32(b.vertices[off+4:]))
			gotCoverage := math.Float32frombits(binary.LittleEndian.Uint32(b.vertices[off+12:]))
			want := tess.point(src)
			if gotX != want.X || gotY != want.Y || gotCoverage != tess.coverage(src) {
				t.Fatalf("chunk %d index %d: vertex (%g,%g,cov %g), want (%g,%g,cov %g)", chunks, i,
					gotX, gotY, gotCoverage, want.X, want.Y, tess.coverage(src))
			}
		}
		// Emptied as recordBatch does, so the next chunk stages into a clean batch.
		b.vertexCount = 0
		b.indexCount = 0
		nextTri += tris
		chunks++
	}
	if chunks < 2 {
		t.Fatalf("chunks = %d; a tessellation this large must split into several draws", chunks)
	}
	if nextTri != numTris {
		t.Fatalf("staged %d of %d triangles", nextTri, numTris)
	}
}

// TestAAFlatteningSplitDedupesWithinChunk checks the remapping actually shares vertices within a chunk rather than
// writing three per triangle: the tessellation's triangles are a fan/strip, so a chunk holds far fewer vertices than
// indices.
func TestAAFlatteningSplitDedupesWithinChunk(t *testing.T) {
	tess := newAAConvexTessellator(stroke.StyleFill, -1, stroke.JoinBevel, 0)
	identity := geom.IdentityMatrix()
	if !tess.tessellate(&identity, squarePath(10, 10, 50, 50)) {
		t.Fatal("tessellate failed on a square")
	}
	b := aaFlatteningBatch{vertexStride: 16}
	tris := aaFlatteningBuildSplitChunk(&b, tess, 0, nil, colorcore.PMColor4f{A: 1}, false)
	if tris != tess.numIndices()/3 {
		t.Fatalf("staged %d of %d triangles in one chunk", tris, tess.numIndices()/3)
	}
	if b.vertexCount != tess.numPts() {
		t.Fatalf("chunk vertices = %d, want the tessellation's %d (each referenced vertex written once)",
			b.vertexCount, tess.numPts())
	}
}
