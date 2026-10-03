// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the transient ProgramInfo/Pipeline pair that each executing op builds during flush (NewPipeline +
// NewProgramInfo), reusing the op free-list machinery (oppool.go) so a steady-state frame allocates no
// program-info/pipeline storage.
//
// Lifetime safety: a ProgramInfo/Pipeline pair is dead once the op that owns it (op.programInfo) finishes OnExecute.
// The program cache keys on the descriptor bytes and never retains the *ProgramInfo or *Pipeline (FlushGLState →
// FindOrCreateProgram / Program.UpdateUniforms consume them synchronously within the op's OnExecute; see gpudraw.go /
// glprogram.go). ProgramInfo copies its target view's format/origin by value and owns no slices. Pipeline owns the
// fragmentProcessors slice, which recyclePipeline clears before returning the shell, so the pool pins no processors.
// Each pooled op owns a unique ProgramInfo that owns a unique Pipeline (createProgramInfo builds both fresh, guarded by
// programInfo == nil), so recycling op.programInfo recycles exactly one pair, referenced nowhere else.
//
// Recycle timing follows the op-struct recycle (oppool.go): the eight batchable geometry ops call recycleProgramInfo
// from their recycle() method, immediately before the op shell itself is recycled, at the four provably-dead points in
// opstask.go. A merged-away op has a nil programInfo (createProgramInfo runs at flush, strictly after all
// recording-time merging), so recycleProgramInfo is a no-op for it. Recording/flush is single-threaded per context, so
// the sync.Pool underneath is only ever touched from the one goroutine per frame.

package gl

// pipelinePool and programInfoPool are the per-type free lists for the executing op's Pipeline/ProgramInfo pair.
var (
	pipelinePool    opPool[Pipeline]
	programInfoPool opPool[ProgramInfo]
)

// borrowPipeline returns a Pipeline shell for a NewPipelineHardClip build. A reused shell is fully zeroed except for a
// preserved, cleared, length-0 fragmentProcessors backing (see recyclePipeline); a fresh shell is all-zero. Callers
// must not struct-literal-assign *p, which would drop the preserved backing.
func borrowPipeline() *Pipeline { return pipelinePool.borrow() }

// recyclePipeline returns a dead pipeline to the pool with a capacity-preserving reset (recycleKeepingBacking,
// oppool.go): a fragmentProcessors backing that grew past one element is kept, reset to length 0 and cleared, while
// every other field is zeroed. The clear() is load-bearing here: the element type is the pointer-carrying
// FragmentProcessor interface, so without it the pool would pin fragment processors. Called only via
// recycleProgramInfo, only on a pipeline whose owning op is dead.
func recyclePipeline(p *Pipeline) {
	pipelinePool.recycleKeepingBacking(p,
		func(p *Pipeline) []FragmentProcessor { return p.fragmentProcessors },
		func(p *Pipeline, s []FragmentProcessor) { p.fragmentProcessors = s })
}

// borrowProgramInfo returns a zeroed ProgramInfo shell for a NewProgramInfo build (a reused shell was fully zeroed at
// recycle; NewProgramInfo overwrites every field regardless).
func borrowProgramInfo() *ProgramInfo { return programInfoPool.borrow() }

// recycleProgramInfo returns a dead programInfo, its owned pipeline, and its geometry processor to their pools, after
// which all three are invalid. Nil-safe: a merged-away op (nil programInfo) recycles nothing. Must be called only on a
// programInfo whose owning op is dead (see the file comment). The GP recycle (recycleGeomProc, gppool.go) is safe for
// the same reason as the pipeline: a non-nil programInfo belongs to one of the eight batchable ops, which owns its GP
// uniquely (createProgramInfo builds it fresh), and the GP is dead once the op has executed.
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
