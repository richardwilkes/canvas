// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the ProgramInfo/Pipeline free lists (programinfopool.go). Like the op pool tests (oppool_test.go)
// these need no GL context: they drive the borrow/recycle protocol directly and assert its safety invariants —
// recycleProgramInfo fully zeroes the programInfo and resets its owned pipeline, and recyclePipeline drops every
// retained fragment-processor reference while preserving a grown backing. The render-level cross-frame guard (that a
// reused ProgramInfo/ Pipeline shell never leaks state into a later frame) is TestOpPoolCrossFrameStability in
// oppool_live_test.go, which now exercises the pair automatically because the batchable ops recycle their programInfo.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// TestRecycleProgramInfoNilSafe verifies recycleProgramInfo tolerates a nil programInfo — the merged-away-op case,
// where createProgramInfo never ran (it runs at flush, strictly after all recording-time merging), so o.programInfo is
// nil when the op is recycled at a merge pop.
func TestRecycleProgramInfoNilSafe(_ *testing.T) {
	recycleProgramInfo(nil) // must not panic
}

// TestRecyclePipelineClearsFPsAndPreservesBacking verifies recyclePipeline (a) drops every retained fragment-processor
// reference so the pool pins no FPs — load-bearing because the element type is the pointer-carrying FragmentProcessor
// interface, not the POD the op instance slices hold — while (b) preserving the grown backing (same array, capacity
// kept, length 0) for reuse, and (c) zeroing every other field. This is the pipeline's use of the capacity-preserving
// reset that programinfopool.go depends on.
func TestRecyclePipelineClearsFPsAndPreservesBacking(t *testing.T) {
	p := borrowPipeline()
	// Grow the FP backing past one element so it lives on the heap (as NewPipeline does for a paint with color +
	// coverage FPs).
	p.fragmentProcessors = append(p.fragmentProcessors,
		MakeColorFP(colorcore.PMColor4f{R: 1, A: 1}),
		MakeColorFP(colorcore.PMColor4f{G: 1, A: 1}))
	grownCap := cap(p.fragmentProcessors)
	if grownCap <= 1 {
		t.Fatalf("expected the FP backing to grow onto the heap; cap=%d", grownCap)
	}
	backing0 := &p.fragmentProcessors[:grownCap][0]
	// Dirty a representative spread of the other fields.
	p.numColorProcessors = 1
	p.flags = pipelineScissorTestEnabled | pipelineHasStencilClip
	p.xferProcessor = MakeNoCoverageXP(raster.BlendSrcOver)

	recyclePipeline(p)

	if p.numColorProcessors != 0 || p.flags != 0 || p.xferProcessor != nil {
		t.Fatalf("recyclePipeline left stale non-backing state: numColor=%d flags=%#x xp!=nil=%v",
			p.numColorProcessors, p.flags, p.xferProcessor != nil)
	}
	if len(p.fragmentProcessors) != 0 {
		t.Fatalf("preserved backing length not reset to 0: %d", len(p.fragmentProcessors))
	}
	if cap(p.fragmentProcessors) != grownCap {
		t.Fatalf("preserved backing capacity not kept: got %d want %d",
			cap(p.fragmentProcessors), grownCap)
	}
	if &p.fragmentProcessors[:grownCap][0] != backing0 {
		t.Fatal("backing array was reallocated, not preserved")
	}
	for i, fp := range p.fragmentProcessors[:grownCap] {
		if fp != nil {
			t.Fatalf("preserved backing still pins a fragment processor at index %d", i)
		}
	}
}

// TestRecycleProgramInfoRecyclesPair verifies recycleProgramInfo zeroes both the programInfo and its owned pipeline in
// one call. It builds the pair by hand (a GL-free test cannot call NewProgramInfo, which requires a render-target
// proxy) and inspects both through the retained pointers immediately after recycle, so the check does not depend on
// which shells a later borrow returns.
func TestRecycleProgramInfoRecyclesPair(t *testing.T) {
	pipeline := borrowPipeline()
	pipeline.flags = pipelineScissorTestEnabled
	pipeline.xferProcessor = MakeNoCoverageXP(raster.BlendSrcOver)
	pipeline.fragmentProcessors = append(pipeline.fragmentProcessors,
		MakeColorFP(colorcore.PMColor4f{B: 1, A: 1}), MakeColorFP(colorcore.PMColor4f{A: 1}))
	pi := borrowProgramInfo()
	pi.pipeline = pipeline
	pi.numSamples = 4
	pi.targetsNumSamples = 4
	pi.needsStencil = true
	pi.primitiveType = gpu.PrimitiveTypeTriangles

	recycleProgramInfo(pi)

	if pi.pipeline != nil || pi.numSamples != 0 || pi.targetsNumSamples != 0 || pi.needsStencil ||
		pi.primitiveType != gpu.PrimitiveType(0) {
		t.Fatalf("programInfo not fully zeroed: pipeline!=nil=%v numSamples=%d needsStencil=%v prim=%d",
			pi.pipeline != nil, pi.numSamples, pi.needsStencil, pi.primitiveType)
	}
	if pipeline.flags != 0 || pipeline.xferProcessor != nil || len(pipeline.fragmentProcessors) != 0 {
		t.Fatalf("owned pipeline not reset: flags=%#x xp!=nil=%v fps=%d", pipeline.flags,
			pipeline.xferProcessor != nil, len(pipeline.fragmentProcessors))
	}
	for i, fp := range pipeline.fragmentProcessors[:cap(pipeline.fragmentProcessors)] {
		if fp != nil {
			t.Fatalf("recycled pipeline still pins a fragment processor at index %d", i)
		}
	}
}

// TestBorrowProgramInfoAndPipelineAreClean verifies both borrow entry points hand back usable, clean shells so
// NewProgramInfo / NewPipelineHardClip (which set their fields explicitly) behave identically on a reused shell as on a
// fresh allocation. A borrowed pipeline in particular must present a length-0 fragmentProcessors slice regardless of
// whether a preserved backing survived.
func TestBorrowProgramInfoAndPipelineAreClean(t *testing.T) {
	// Prime the pools with recycled shells so borrow takes the reuse path.
	pipeline := borrowPipeline()
	pipeline.flags = pipelineHasStencilClip
	pipeline.fragmentProcessors = append(pipeline.fragmentProcessors,
		MakeColorFP(colorcore.PMColor4f{A: 1}), MakeColorFP(colorcore.PMColor4f{R: 1, A: 1}))
	pi := borrowProgramInfo()
	pi.needsStencil = true
	pi.pipeline = pipeline
	recycleProgramInfo(pi)

	if got := borrowPipeline(); got.flags != 0 || got.xferProcessor != nil ||
		got.numColorProcessors != 0 || len(got.fragmentProcessors) != 0 {
		t.Fatalf("borrowPipeline returned an unclean shell: flags=%#x xp!=nil=%v numColor=%d fps=%d",
			got.flags, got.xferProcessor != nil, got.numColorProcessors, len(got.fragmentProcessors))
	}
	if got := borrowProgramInfo(); got.pipeline != nil || got.needsStencil || got.numSamples != 0 {
		t.Fatalf("borrowProgramInfo returned an unclean shell: pipeline!=nil=%v needsStencil=%v",
			got.pipeline != nil, got.needsStencil)
	}
}
