// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Fake-driver tests for the OpsRenderPass bind/draw call sequences (B.1): the deferred vertex- and instance-buffer
// binding in BindBuffers and the two DrawIndexed lanes. The recorder counts the GL calls that survive the attrib-state
// shadow, so these tests pin the exact per-draw glVertexAttribPointer traffic — one pass per attribute per indexed draw
// without baseVertexBaseInstance support, and none at all (after the eager offset-0 pass) with it.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/gpu"
)

// newDrawRecordingPass builds a recording Gpu over the given caps profile plus a minimal bound program (two float2
// vertex attributes, one float4 instance attribute) and a render pass that is ready to draw. The render target stays
// nil: hwWriteToColor is forced to triNo so didDrawTo skips the surface bookkeeping while still counting the draw.
func newDrawRecordingPass(t *testing.T, d *capsFakeDriver) (recGpu *Gpu, pass *OpsRenderPass, vertBuf, idxBuf, instBuf *Buffer) {
	t.Helper()
	g := newRecordingGpuWithDriver(t, d)
	vb := g.CreateBuffer(1024, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
	ib := g.CreateBuffer(1024, gpu.BufferTypeIndex, gpu.AccessPatternStatic)
	instb := g.CreateBuffer(1024, gpu.BufferTypeVertex, gpu.AccessPatternDynamic)
	if vb == nil || ib == nil || instb == nil {
		t.Fatal("CreateBuffer failed")
	}
	prog := &Program{
		gpu:       g,
		programID: 1,
		attributes: []ProgramAttribute{
			{CPUType: gpu.VertexAttribTypeFloat2, GPUType: GLSLTypeFloat2, Offset: 0, Location: 0},
			{CPUType: gpu.VertexAttribTypeFloat2, GPUType: GLSLTypeFloat2, Offset: 8, Location: 1},
			{CPUType: gpu.VertexAttribTypeFloat4, GPUType: GLSLTypeFloat4, Offset: 0, Location: 2},
		},
		vertexAttributeCnt:   2,
		instanceAttributeCnt: 1,
		vertexStride:         16,
		instanceStride:       16,
	}
	g.hwProgram = prog
	g.hwProgramID = 1
	g.hwWriteToColor = triNo
	pass = &OpsRenderPass{
		gpu:                g,
		primitiveType:      gpu.PrimitiveTypeTriangles,
		drawPipelineStatus: drawPipelineOK,
	}
	recCounts = map[string]int{}
	return g, pass, vb, ib, instb
}

// TestDrawIndexedDeferredVertexBind covers the !baseVertexBaseInstanceSupport lane (the M4 Max profile): BindBuffers
// must not touch the attrib pointers for an indexed mesh — each DrawIndexed performs the single glVertexAttribPointer
// pass at its baseVertex offset, and the attrib-state shadow elides the pass entirely when the offset repeats.
func TestDrawIndexedDeferredVertexBind(t *testing.T) {
	g, pass, vb, ib, _ := newDrawRecordingPass(t, appleM4MaxDriver())
	if g.glCaps().BaseVertexBaseInstanceSupport() {
		t.Fatal("profile unexpectedly supports baseVertexBaseInstance")
	}

	pass.BindBuffers(ib, nil, vb)
	if got := counts("glVertexAttribPointer"); got != 0 {
		t.Errorf("BindBuffers set %d attrib pointers; the indexed-draw bind must be deferred", got)
	}

	pass.DrawIndexed(6, 0, 0, 3, 5)
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("first draw set %d attrib pointers, want 2 (one per vertex attribute)", got)
	}
	if got := counts("glDrawRangeElements"); got != 1 {
		t.Errorf("glDrawRangeElements called %d times, want 1", got)
	}

	// Same baseVertex: the shadow elides the whole pass.
	pass.DrawIndexed(6, 6, 0, 3, 5)
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("repeat-offset draw grew attrib pointers to %d, want still 2", got)
	}

	// New baseVertex: exactly one more pass, not two (the pre-B.1 code paid an eager offset-0 pass in BindBuffers plus
	// the rebind here).
	pass.DrawIndexed(6, 0, 0, 3, 9)
	if got := counts("glVertexAttribPointer"); got != 4 {
		t.Errorf("new-offset draw set %d attrib pointers total, want 4", got)
	}
	if got := g.NumGLDraws(); got != 3 {
		t.Errorf("NumGLDraws = %d, want 3", got)
	}
}

// TestDrawIndexedBaseVertexFastLane covers the baseVertexBaseInstanceSupport lane (the Mesa llvmpipe profile):
// BindBuffers binds the vertex buffer once at offset 0, a nonzero baseVertex rides as a single-instance
// glDrawElementsInstancedBaseVertexBaseInstance with no attrib-pointer rebind, and the draw still counts toward
// NumGLDraws.
func TestDrawIndexedBaseVertexFastLane(t *testing.T) {
	g, pass, vb, ib, _ := newDrawRecordingPass(t, mesaLlvmpipeDriver())
	if !g.glCaps().BaseVertexBaseInstanceSupport() {
		t.Fatal("profile unexpectedly lacks baseVertexBaseInstance")
	}

	pass.BindBuffers(ib, nil, vb)
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("BindBuffers set %d attrib pointers, want the eager offset-0 pass of 2", got)
	}

	pass.DrawIndexed(6, 0, 0, 3, 5)
	if got := counts("glDrawElementsInstancedBaseVertexBaseInstance"); got != 1 {
		t.Errorf("base-vertex draw used the fast lane %d times, want 1", got)
	}
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("base-vertex draw rebound attrib pointers (%d total), want none beyond bind", got)
	}

	// baseVertex 0 falls through to the plain indexed draw.
	pass.DrawIndexed(6, 6, 0, 3, 0)
	if got := counts("glDrawRangeElements"); got != 1 {
		t.Errorf("glDrawRangeElements called %d times, want 1", got)
	}
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("zero-base draw rebound attrib pointers (%d total), want none beyond bind", got)
	}
	if got := g.NumGLDraws(); got != 2 {
		t.Errorf("NumGLDraws = %d, want 2 (the fast lane must still count)", got)
	}
}

// TestDrawIndexedInstancedSingleBindPass covers the instanced-indexed draw without baseVertexBaseInstance: BindBuffers
// defers both buffers, and the draw performs exactly one attrib-pointer pass for the vertex and instance attributes
// together.
func TestDrawIndexedInstancedSingleBindPass(t *testing.T) {
	_, pass, vb, ib, instb := newDrawRecordingPass(t, appleM4MaxDriver())

	pass.BindBuffers(ib, instb, vb)
	if got := counts("glVertexAttribPointer"); got != 0 {
		t.Errorf("BindBuffers set %d attrib pointers; both binds must be deferred", got)
	}

	pass.DrawIndexedInstanced(6, 0, 4, 2, 1)
	if got := counts("glVertexAttribPointer"); got != 3 {
		t.Errorf("draw set %d attrib pointers, want 3 (two vertex + one instance)", got)
	}
	if got := counts("glDrawElementsInstanced"); got != 1 {
		t.Errorf("glDrawElementsInstanced called %d times, want 1", got)
	}
}

// TestBindBuffersTypedNilIndexBuffer pins the AnyBuffer typed-nil contract: ops hand their *Buffer index field straight
// to BindBuffers, so a non-indexed mesh arrives as a non-nil interface holding a nil *Buffer, which must still take the
// eager (non-indexed) vertex bind so a following Draw has its attrib pointers in place.
func TestBindBuffersTypedNilIndexBuffer(t *testing.T) {
	_, pass, vb, _, _ := newDrawRecordingPass(t, appleM4MaxDriver())

	var nilIndex *Buffer
	pass.BindBuffers(nilIndex, nil, vb)
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("BindBuffers set %d attrib pointers, want the eager non-indexed pass of 2", got)
	}

	pass.Draw(4, 7)
	if got := counts("glDrawArrays"); got != 1 {
		t.Errorf("glDrawArrays called %d times, want 1", got)
	}
	if got := counts("glVertexAttribPointer"); got != 2 {
		t.Errorf("Draw rebound attrib pointers (%d total), want none beyond bind", got)
	}
}
