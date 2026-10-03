// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the geometry processors that the batchable geometry ops (oval-family, stroked-rect, fillRect and
// fillRRect) build during flush, continuing the ProgramInfo/Pipeline pooling in programinfopool.go. The factories
// (MakeDefaultGeoProc, makeCircleGeometryProcessor, makeEllipseGeometryProcessor, makeDIEllipseGeometryProcessor,
// MakeQuadPerEdgeAAProcessor/MakeTexturedQuadProcessor, makeFillRRectProcessor) borrow from per-concrete-type free
// lists built on the op free-list machinery (oppool.go), so a steady-state frame allocates no GP storage.
//
// Lifetime safety: a geometry processor is dead once the op that owns it (via op.programInfo.geomProc) finishes
// OnExecute.
//   - It is never shared or cached: the only reference to a factory's result is the owning op's ProgramInfo. The GP is
//     read to build the program descriptor key (AddToKey / gpAttributeKey), to compute vertex strides and bind textures
//     (gpBase), and to upload uniforms (GPProgramImpl.SetData) — all synchronous within the owning op's
//     OnPrepare/OnExecute. The program cache keys on the descriptor bytes and returns a cached *Program; it never
//     retains the GP, and the GP's ProgramImpl receives the GP as a SetData parameter on every draw rather than storing
//     it (see the Impls in ovalgp.go / defaultgeoproc.go and glprogram.go's SetData call).
//   - It is recycled only via recycleProgramInfo (programinfopool.go), which the eight batchable geometry ops call from
//     recycle() at the four provably-dead points opstask.go establishes. recycleProgramInfo returns early for a
//     merged-away op (nil programInfo — createProgramInfo runs at flush, strictly after all recording-time merging), so
//     only the surviving op that actually executed reaches recycleGeomProc, carrying the unique GP it built.
//   - A plain full-zero recycle is complete; no capacity-preserving reset is needed. Every slice header and pointer a
//     pooled GP owns points into that same GP's own inline storage: GPBase.vertexAttributes.attributes into the inline
//     attrs array (setVertexAttributesWithImplicitOffsets(gp.attrs[:])), and, for the two GPs that have them,
//     GPBase.instanceAttributes.attributes into the fillRRect GP's instanceAttrs array, GPBase.textureSamplers into the
//     quad-per-edge GP's samplerStorage array, and the fillRRect GP's colorAttrib at an element of its own
//     instanceAttrs. So `*o = zero` drops no external heap backing, and the factory re-establishes every one of them on
//     the next borrow, so a reused shell is byte-identical to a fresh &T{}.
//
// Ops that never recycle (the path renderers, atlas text, the tessellation renderers, textureOp) also build these GP
// types and borrow from the same pools; their ProgramInfos are not routed through recycleProgramInfo, so their GPs are
// left to the GC. Recording/flush is single-threaded per context, so the sync.Pool underneath is only ever touched from
// one goroutine per frame.

package gl

// The per-concrete-type free lists. Every slice header and pointer in each of these six shapes aims at its own inline
// storage, so a plain full-zero recycle resets it completely (see the file comment's lifetime-safety bullets).
var (
	// The GPBase-plus-inline-attrs shapes.
	defaultGeoProcPool opPool[defaultGeoProc]
	circleGPPool       opPool[circleGeometryProcessor]
	ellipseGPPool      opPool[ellipseGeometryProcessor]
	diEllipseGPPool    opPool[diEllipseGeometryProcessor]
	// The fillRect/fillRRect GPs are full-zero-safe because the quad-per-edge GP's textured sampler lives in an inline
	// array (samplerStorage) and the fillRRect GP's colorAttrib only ever points into its own instanceAttrs array.
	qpaaGPPool      opPool[quadPerEdgeAAGeometryProcessor]
	fillRRectGPPool opPool[fillRRectProcessor]
)

// recycleGeomProc returns a dead geometry processor to its per-type free list. It is called only from
// recycleProgramInfo, only on the GP of a ProgramInfo whose owning (pooled) op is dead, so the GP is provably dead and
// uniquely owned (see the file comment). A GP of any other type, or a nil GP, matches no case and is ignored.
func recycleGeomProc(gp GeometryProcessor) {
	switch g := gp.(type) {
	case *defaultGeoProc:
		defaultGeoProcPool.recycle(g)
	case *circleGeometryProcessor:
		circleGPPool.recycle(g)
	case *ellipseGeometryProcessor:
		ellipseGPPool.recycle(g)
	case *diEllipseGeometryProcessor:
		diEllipseGPPool.recycle(g)
	case *quadPerEdgeAAGeometryProcessor:
		qpaaGPPool.recycle(g)
	case *fillRRectProcessor:
		fillRRectGPPool.recycle(g)
	}
}
