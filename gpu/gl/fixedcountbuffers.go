// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The static vertex and index buffers used as the template for instanced fixed-count tessellation draws of curves,
// wedges, and strokes, plus the heuristics for pre-allocating instance attribute space and converting accumulated
// tolerances into draw vertex counts. The fallback vertex buffer for strokes without vertex-ID support (and its
// edge-count clamp) is trimmed: it is only needed without gl_VertexID support, and desktop core profiles always have
// it. The buffer initializers are byte-slice generators cached under static unique keys, keyed the same way
// ResourceProvider.FindOrMakeStaticBuffer expects.

package gl

import (
	"encoding/binary"
	"math"
	"math/bits"
	"sync"

	"github.com/richardwilkes/canvas/gpu"
)

// Fixed-count curve buffer geometry: the vertex buffer holds (resolveLevel, idx) float pairs for
// kMaxParametricSegments+1 vertices; the index buffer connects them with a middle-out topology.
const (
	fixedCountCurvesVertexCount = tessMaxParametricSegments + 1
	fixedCountCurvesVertexSize  = fixedCountCurvesVertexCount * 8
	fixedCountCurvesIndexSize   = (tessMaxParametricSegments - 1) * 3 * 2
	fixedCountWedgesVertexCount = fixedCountCurvesVertexCount + 1 // +1 fan vertex
	fixedCountWedgesVertexSize  = fixedCountWedgesVertexCount * 8
	fixedCountWedgesIndexSize   = tessMaxParametricSegments * 3 * 2 // +1 fan triangle
)

// fixedCountCurvesPreallocCount is a heuristic for reserving instance attribute space before using a patchWriter.
// Over-allocate enough curves for 1 in 4 to chop: every chop introduces 2 new patches (another curve patch and a
// triangle patch that glues the two chops together), i.e. + 2 * ((count + 3) / 4) == (count + 3) / 2.
func fixedCountCurvesPreallocCount(totalCombinedPathVerbCnt int) int {
	return totalCombinedPathVerbCnt + (totalCombinedPathVerbCnt+3)/2
}

// fixedCountCurvesVertexCountForTolerances converts the accumulated worst-case tolerances into an index count passed
// into an instanced, indexed draw that uses the fixed-count curve static buffers.
func fixedCountCurvesVertexCountForTolerances(tolerances *linearTolerances) int {
	// We should already have chopped curves to make sure none needed a higher resolveLevel than tessMaxResolveLevel.
	resolveLevel := min(tolerances.requiredResolveLevel(), tessMaxResolveLevel)
	return numCurveTrianglesAtResolveLevel(resolveLevel) * 3
}

// fixedCountWedgesPreallocCount over-allocates enough wedges for 1 in 4 to chop, i.e. ceil(maxWedges * 5/4).
func fixedCountWedgesPreallocCount(totalCombinedPathVerbCnt int) int {
	return (totalCombinedPathVerbCnt*5 + 3) / 4
}

// fixedCountWedgesVertexCountForTolerances counts 3 vertices per curve triangle, plus 3 more for the wedge fan
// triangle.
func fixedCountWedgesVertexCountForTolerances(tolerances *linearTolerances) int {
	resolveLevel := min(tolerances.requiredResolveLevel(), tessMaxResolveLevel)
	return (numCurveTrianglesAtResolveLevel(resolveLevel) + 1) * 3
}

// fixedCountStrokesMaxEdges caps the edge count so the resulting vertices stay indexable by a signed short. We just
// have to draw the line somewhere and this seems reasonable enough. (There are two vertices per edge, so 2^14 edges
// make 2^15 vertices.)
const fixedCountStrokesMaxEdges = (1 << 14) - 1

// fixedCountStrokesPreallocCount over-allocates enough patches for each stroke to chop once, and for 8 extra caps.
// Since we have to chop at inflections, points of 180-degree rotation, and anywhere a stroke requires too many
// parametric segments, many strokes will end up getting chopped.
func fixedCountStrokesPreallocCount(totalCombinedStrokeVerbCnt int) int {
	return totalCombinedStrokeVerbCnt*2 + 8 /* caps */
}

// fixedCountStrokesVertexCountForTolerances counts two vertices per edge. (Does not account for a fallback path for no
// vertex-ID support — that lane is trimmed along with the fallback vertex buffer.)
func fixedCountStrokesVertexCountForTolerances(tolerances *linearTolerances) int {
	return min(tolerances.requiredStrokeEdges(), fixedCountStrokesMaxEdges) * 2
}

// skPrevLog2 returns floor(log2(value)) for value >= 1.
func skPrevLog2(value int) int {
	if value < 1 {
		panic("skPrevLog2 requires a positive value")
	}
	return bits.Len(uint(value)) - 1
}

// writeFixedCountCurveVertexData lays out the vertices in "middle-out" order:
//
//	T= 0/1, 1/1,              ; resolveLevel=0
//	   1/2,                   ; resolveLevel=1  (0/2 and 2/2 are already in resolveLevel 0)
//	   1/4, 3/4,              ; resolveLevel=2  (2/4 is already in resolveLevel 1)
//	   1/8, 3/8, 5/8, 7/8,    ; resolveLevel=3  (2/8 and 6/8 are already in resolveLevel 2)
//	   ...                    ; resolveLevel=...
func writeFixedCountCurveVertexData(buf []byte) {
	vertexCount := len(buf) / 8
	if vertexCount <= 3 {
		panic("fixed-count curve vertex buffer too small")
	}
	off := 0
	put := func(resolveLevel, idx float32) {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(resolveLevel))
		binary.LittleEndian.PutUint32(buf[off+4:], math.Float32bits(idx))
		off += 8
	}
	// Resolve level 0 is just the beginning and ending vertices.
	put(0, 0)
	put(0, 1)
	// Resolve levels 1..maxResolveLevel.
	maxResolveLevel := skPrevLog2(vertexCount - 1)
	if (1<<maxResolveLevel)+1 != vertexCount {
		panic("fixed-count curve vertex buffer size is not 2^n + 1 vertices")
	}
	for resolveLevel := 1; resolveLevel <= maxResolveLevel; resolveLevel++ {
		numSegmentsInResolveLevel := 1 << resolveLevel
		// Write out the odd vertices in this resolveLevel. The even vertices were already written out in previous
		// resolveLevels and will be indexed from there.
		for i := 1; i < numSegmentsInResolveLevel; i += 2 {
			put(float32(resolveLevel), float32(i))
		}
	}
	if off != len(buf) {
		panic("fixed-count curve vertex buffer size mismatch")
	}
}

// writeFixedCountCurveIndexData connects the vertices with a middle-out triangulation. Resolve level 1 is a single
// triangle at T=[0, 1/2, 1]; each subsequent resolve level's triangles share the edges of a neighbor triangle from the
// previous level.
func writeFixedCountCurveIndexData(buf []byte, baseIndex uint16) {
	triangleCount := len(buf) / (2 * 3)
	if triangleCount < 1 {
		panic("fixed-count curve index buffer too small")
	}
	indexData := make([][3]uint16, 0, triangleCount)

	// Resolve level 1 is just a single triangle at T=[0, 1/2, 1].
	indexData = append(indexData, [3]uint16{baseIndex, baseIndex + 2, baseIndex + 1})
	neighborInLastResolveLevel := 0

	// Resolve levels 2..maxResolveLevel.
	maxResolveLevel := skPrevLog2(triangleCount + 1)
	nextIndex := baseIndex + 3
	if numCurveTrianglesAtResolveLevel(maxResolveLevel) != triangleCount {
		panic("fixed-count curve index buffer size mismatch")
	}
	for resolveLevel := 2; resolveLevel <= maxResolveLevel; resolveLevel++ {
		numOuterTrianglesInResolveLevel := 1 << (resolveLevel - 1)
		numTrianglePairsInResolveLevel := numOuterTrianglesInResolveLevel >> 1
		for i := 0; i < numTrianglePairsInResolveLevel; i++ {
			neighbor := indexData[neighborInLastResolveLevel]
			// First triangle shares the left edge of the neighbor.
			indexData = append(indexData, [3]uint16{neighbor[0], nextIndex, neighbor[1]})
			nextIndex++
			// Second triangle shares the right edge of the neighbor.
			indexData = append(indexData, [3]uint16{neighbor[1], nextIndex, neighbor[2]})
			nextIndex++
			neighborInLastResolveLevel++
		}
	}
	if len(indexData) != triangleCount || int(nextIndex) != int(baseIndex)+triangleCount+2 {
		panic("fixed-count curve index topology bookkeeping error")
	}
	off := 0
	for _, tri := range indexData {
		for _, idx := range tri {
			binary.LittleEndian.PutUint16(buf[off:], idx)
			off += 2
		}
	}
}

// writeFixedCountWedgeVertexData writes the fan point first (a negative resolve level indicates the fan point), then
// the same layout as curves.
func writeFixedCountWedgeVertexData(buf []byte) {
	binary.LittleEndian.PutUint32(buf[0:], math.Float32bits(-1))
	binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(-1))
	writeFixedCountCurveVertexData(buf[8:])
}

// writeFixedCountWedgeIndexData writes the fan triangle first, then the same topology as curves with a baseIndex of 1.
func writeFixedCountWedgeIndexData(buf []byte) {
	binary.LittleEndian.PutUint16(buf[0:], 0)
	binary.LittleEndian.PutUint16(buf[2:], 1)
	binary.LittleEndian.PutUint16(buf[4:], 2)
	writeFixedCountCurveIndexData(buf[6:], 1)
}

// The static buffer contents and their unique keys, generated once.
var (
	fixedCountBuffersOnce      sync.Once
	fixedCountCurveVertexKey   gpu.UniqueKey
	fixedCountCurveIndexKey    gpu.UniqueKey
	fixedCountWedgeVertexKey   gpu.UniqueKey
	fixedCountWedgeIndexKey    gpu.UniqueKey
	fixedCountCurveVertexBytes []byte
	fixedCountCurveIndexBytes  []byte
	fixedCountWedgeVertexBytes []byte
	fixedCountWedgeIndexBytes  []byte
)

func fixedCountStaticBuffers() {
	fixedCountBuffersOnce.Do(func() {
		makeKey := func(key *gpu.UniqueKey, label string) {
			builder := gpu.UniqueKeyBuilder(key, gpu.GenerateUniqueKeyDomain(), 1, label)
			builder.Slice()[0] = 1
			builder.Finish()
		}
		makeKey(&fixedCountCurveVertexKey, "FixedCountCurveVertexBuffer")
		makeKey(&fixedCountCurveIndexKey, "FixedCountCurveIndexBuffer")
		makeKey(&fixedCountWedgeVertexKey, "FixedCountWedgesVertexBuffer")
		makeKey(&fixedCountWedgeIndexKey, "FixedCountWedgesIndexBuffer")

		fixedCountCurveVertexBytes = make([]byte, fixedCountCurvesVertexSize)
		writeFixedCountCurveVertexData(fixedCountCurveVertexBytes)
		fixedCountCurveIndexBytes = make([]byte, fixedCountCurvesIndexSize)
		writeFixedCountCurveIndexData(fixedCountCurveIndexBytes, 0)
		fixedCountWedgeVertexBytes = make([]byte, fixedCountWedgesVertexSize)
		writeFixedCountWedgeVertexData(fixedCountWedgeVertexBytes)
		fixedCountWedgeIndexBytes = make([]byte, fixedCountWedgesIndexSize)
		writeFixedCountWedgeIndexData(fixedCountWedgeIndexBytes)
	})
}
