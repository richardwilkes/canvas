// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the geometry processors that the batchable oval-family and stroked-rect ops build during flush (the GPU
// op-stream-arena series, continuing S112's ProgramInfo/Pipeline pooling). The factories heap-allocate a fresh &T{} per
// executing op (MakeDefaultGeoProc, makeCircleGeometryProcessor, makeEllipseGeometryProcessor,
// makeDIEllipseGeometryProcessor); S109's op-struct recycle then dropped that GP every frame, so after S112 pooled the
// ProgramInfo/Pipeline pair the GP was the last un-pooled allocation left in the createProgramInfo path — the largest
// in-our-code allocation tier the GPU-frame benchmarks measure (MakeDefaultGeoProc + makeCircleGeometryProcessor ≈ 12
// allocs/frame in the panels frame). This pools each GP with per-concrete-type free lists, reusing the op free-list
// machinery (oppool.go), so a steady-state frame allocates no GP storage.
//
// Lifetime safety (the S112 argument, applied to the GP): a geometry processor is dead once the op that owns it (via
// op.programInfo.geomProc) finishes OnExecute.
//   - It is never shared or cached: each factory returns a fresh instance, and the only reference to
//     it is the owning op's ProgramInfo. The GP is read to build the program descriptor key
//     (AddToKey / gpAttributeKey), to compute vertex strides and bind textures (gpBase), and to
//     upload uniforms (GPProgramImpl.SetData) — all synchronous within the owning op's
//     OnPrepare/OnExecute. The program cache keys on the descriptor bytes and returns a cached
//     *Program; it never retains the GP, and the GP's ProgramImpl receives the GP as a SetData
//     parameter on every draw rather than storing it (see the Impls in ovalgp.go / defaultgeoproc.go
//     and glprogram.go's SetData call). So nothing observes the GP once the op has executed.
//   - It is recycled only via recycleProgramInfo (programinfopool.go), which the eight batchable geometry ops call from
//     recycle() at the four provably-dead points opstask.go establishes (S109). recycleProgramInfo returns early for a
//     merged-away op (nil programInfo — createProgramInfo runs at flush, strictly after all recording-time merging), so
//     only the surviving op that actually executed reaches recycleGeomProc, carrying the unique GP it built. This is
//     exactly the S112 ownership story for the ProgramInfo/Pipeline pair.
//   - A plain full-zero recycle is complete — no capacity-preserving reset (S111) is needed. The only slice a pooled GP
//     owns is its GPBase.vertexAttributes.attributes header, which points into the GP's *own* inline attrs array
//     (setVertexAttributesWithImplicitOffsets(gp.attrs[:])); `*o = zero` drops no external heap backing, and the
//     factory re-establishes the header on the next borrow, so a reused shell is byte-identical to a fresh &T{} (the
//     guarantee opPool already gives the op structs, S109). No GP field pins a heap object that a preserved backing
//     would need to carry.
//
// Non-pooled ops (the path renderers, atlas text, the tessellation renderers, textureOp) also build these GP types and
// borrow from the same pools, but they never recycle (their ProgramInfos are GC'd, not routed through
// recycleProgramInfo), so their GPs are simply GC'd — no aliasing, exactly as for the ProgramInfo/Pipeline pools
// (S112). The fillRect (quad-per-edge) and fillRRect (instance attribute) geometry processors joined the pools once the
// quad-per-edge GP's textured sampler slice moved into an inline array, making both full-zero-safe (see the pool var
// comment). Recording/flush is single-threaded per context, so the sync.Pool underneath is only ever touched from the
// one goroutine per frame.

package gl

// The per-concrete-type free lists for the batchable oval-family and stroked-rect geometry processors. Each of these
// four is the GPBase-plus-inline-attrs POD shape that a plain full-zero recycle resets completely (see the file
// comment); the differently-shaped fillRect/fillRRect GPs are deliberately excluded.
var (
	defaultGeoProcPool opPool[defaultGeoProc]
	circleGPPool       opPool[circleGeometryProcessor]
	ellipseGPPool      opPool[ellipseGeometryProcessor]
	diEllipseGPPool    opPool[diEllipseGeometryProcessor]
	// The fillRect/fillRRect GPs joined in B.3 once their shapes became full-zero-safe: the quad-per-edge GP's textured
	// sampler moved into an inline array (samplerStorage), and the fillRRect GP's colorAttrib only ever points into its
	// own instanceAttrs array. Their factories rebuild every field on borrow, so a reused shell is byte-identical to a
	// fresh one.
	qpaaGPPool      opPool[quadPerEdgeAAGeometryProcessor]
	fillRRectGPPool opPool[fillRRectProcessor]
)

// recycleGeomProc returns a dead geometry processor to its per-type free list. It is called only from
// recycleProgramInfo, only on the GP of a ProgramInfo whose owning (pooled) op is dead, so the GP is provably dead and
// uniquely owned (see the file comment). GP types other than the four pooled here — the fillRect/fillRRect GPs, and any
// GP built only by non-recycling renderers — match no case and are left for the GC, exactly as before this pooling
// existed. Nil-safe: a nil GP (which a non-nil ProgramInfo never carries, but which is cheap to tolerate) matches no
// case.
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
