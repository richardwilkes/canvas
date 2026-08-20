// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the transient ProgramInfo/Pipeline pair that each executing op builds during flush, the GPU counterpart
// of the op-struct free lists in oppool.go. Building one *ProgramInfo and one *Pipeline per executing op (NewPipeline
// + NewProgramInfo) and then dropping that pair every frame made the GPU-frame benchmarks rank it as the next-largest
// in-repo allocation tier after the FFI variadic (~8.4% of the panels frame). This pools both types with per-type free
// lists, reusing the op free-list machinery (oppool.go), so a steady-state frame allocates no program-info/pipeline
// storage.
//
// Lifetime safety: a ProgramInfo/Pipeline pair is dead once the op that owns it
// (op.programInfo) finishes OnExecute. The program cache keys on the descriptor bytes and returns a cached *Program —
// it never retains the *ProgramInfo or *Pipeline (FlushGLState → FindOrCreateProgram / Program.UpdateUniforms consume
// them synchronously within the op's OnExecute; see gpudraw.go / glprogram.go). ProgramInfo copies its target view's
// format/origin by value and owns no slices. Pipeline owns the fragmentProcessors slice (the FPs detached from the
// paint's ProcessorSet), which recyclePipeline clears — dropping every FP reference — before returning the shell, so
// the pool pins no processors. Each pooled op owns a unique ProgramInfo that owns a unique Pipeline (createProgramInfo
// builds both fresh, guarded by programInfo == nil), so there is no aliasing: recycling op.programInfo recycles exactly
// one pair, referenced nowhere else.
//
// Recycle timing follows the same schedule as the op-struct recycle (oppool.go): the eight batchable geometry ops call
// recycleProgramInfo(o.programInfo) from their recycle() method, immediately before the op shell itself is recycled, at
// the four provably-dead points in opstask.go. A merged-away op has a nil programInfo (createProgramInfo runs at flush,
// strictly after all recording-time merging), so recycleProgramInfo is a no-op for it; only the surviving op that
// actually executed carries a non-nil pair to recycle. Recording/flush is single-threaded per context, exactly as for
// the op pool, so the sync.Pool underneath is only ever touched from the one goroutine per frame.

package gl

// pipelinePool and programInfoPool are the per-type free lists for the executing op's Pipeline/ProgramInfo pair.
var (
	pipelinePool    opPool[Pipeline]
	programInfoPool opPool[ProgramInfo]
)

// borrowPipeline returns a Pipeline shell for a NewPipelineHardClip build. A reused shell is fully zeroed except for a
// preserved, cleared, length-0 fragmentProcessors backing (see recyclePipeline); a fresh shell is all-zero. Callers
// must set every field they use (NewPipelineHardClip does) and must not struct-literal-assign *p (that would drop the
// preserved backing) — only that fragmentProcessors has length 0.
func borrowPipeline() *Pipeline { return pipelinePool.borrow() }

// recyclePipeline returns a dead pipeline to the pool with a capacity-preserving reset (recycleKeepingBacking,
// oppool.go): its fragmentProcessors heap backing — when it grew past one element — is kept, reset to length 0 and
// cleared so no fragment processor is pinned, while every other field is zeroed. Called only via recycleProgramInfo,
// only on a pipeline whose owning op is dead. The element type is the FragmentProcessor interface (pointer-carrying),
// so the clear() in recycleKeepingBacking is load-bearing here, not just defensive.
func recyclePipeline(p *Pipeline) {
	pipelinePool.recycleKeepingBacking(p,
		func(p *Pipeline) []FragmentProcessor { return p.fragmentProcessors },
		func(p *Pipeline, s []FragmentProcessor) { p.fragmentProcessors = s })
}

// borrowProgramInfo returns a zeroed ProgramInfo shell for a NewProgramInfo build (a reused shell was fully zeroed at
// recycle; NewProgramInfo overwrites every field regardless).
func borrowProgramInfo() *ProgramInfo { return programInfoPool.borrow() }

// recycleProgramInfo returns a dead programInfo, its owned pipeline, and its geometry processor to their pools.
// Nil-safe: a merged-away op (nil programInfo) recycles nothing. Must be called only on a programInfo whose owning op
// is dead (see the file comment); after it returns, the programInfo, its pipeline, and its geometry processor are
// invalid. The GP recycle (recycleGeomProc, gppool.go) is safe for the same reason as the pipeline: a non-nil
// programInfo belongs to one of the eight batchable ops, which owns its GP uniquely (createProgramInfo builds it
// fresh), and the GP is dead once the op has executed.
func recycleProgramInfo(pi *ProgramInfo) {
	if pi == nil {
		return
	}
	if pi.pipeline != nil {
		recyclePipeline(pi.pipeline)
	}
	recycleGeomProc(pi.geomProc)
	programInfoPool.recycle(pi)
}
