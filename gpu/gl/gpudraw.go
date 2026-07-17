// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GPU draw-path state flushing: FlushGLState (program lookup, blend, stencil, scissor test, conservative raster,
// wireframe, render-target bind), flushProgram, prepareToDraw, and didDrawTo — plus the OpsRenderPass draw surface
// (BindPipeline/SetScissorRect/BindBuffers/Draw/DrawIndexed) that the draw ops replay into. Trim: window rectangles
// stay disabled (EXT_window_rectangles is absent from the desktop trim's target drivers).

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// ProgramCacheStats exposes the program cache metrics.
func (g *Gpu) ProgramCacheStats() PipelineStats { return g.programCache.Stats() }

// ProgramCacheCount exposes the number of live cached programs.
func (g *Gpu) ProgramCacheCount() int { return g.programCache.Count() }

// CurrentProgram returns the currently bound GL program, refreshing dirty context state first.
func (g *Gpu) CurrentProgram() *Program {
	g.handleDirtyContext()
	return g.hwProgram
}

// flushProgram binds program for use with UseProgram, skipping the GL call if it is already bound.
func (g *Gpu) flushProgram(program *Program) {
	if program == nil {
		g.hwProgram = nil
		g.hwProgramID = 0
		return
	}
	if (program == g.hwProgram) != (g.hwProgramID == program.ProgramID()) {
		panic("program shadow out of sync")
	}
	if program == g.hwProgram {
		return
	}
	id := program.ProgramID()
	if id == 0 {
		panic("program has no GL object")
	}
	g.fns().UseProgram(id)
	g.hwProgram = program
	g.hwProgramID = id
}

// flushProgramID binds a raw GL program object by ID, skipping the GL call if it is already bound (used by the
// copy/mipmap programs later).
func (g *Gpu) flushProgramID(id uint32) {
	if id == 0 {
		panic("program id required")
	}
	if g.hwProgramID == id {
		if g.hwProgram != nil {
			panic("program shadow out of sync")
		}
		return
	}
	g.hwProgram = nil
	g.fns().UseProgram(id)
	g.hwProgramID = id
}

// FlushGLState resolves the program for the draw and flushes all non-vertex GL state.
func (g *Gpu) FlushGLState(renderTarget *RenderTarget, useMultisampleFBO bool, programInfo *ProgramInfo) bool {
	g.handleDirtyContext()

	program := g.programCache.FindOrCreateProgram(programInfo)
	if program == nil {
		return false
	}

	g.flushProgram(program)

	// Swizzle the blend to match what the shader will output.
	g.flushBlendAndColorWrite(xpGetBlendInfo(programInfo.Pipeline().XferProcessor()),
		programInfo.Pipeline().WriteSwizzle())

	g.hwProgram.UpdateUniforms(renderTarget, programInfo)

	var stencil StencilSettings
	stencil.SetDisabled()
	if programInfo.IsStencilEnabled() {
		if renderTarget.StencilAttachment(useMultisampleFBO) == nil {
			panic("stencil-enabled draw without a stencil attachment")
		}
		stencil.Reset(programInfo.UserStencilSettings(),
			programInfo.Pipeline().HasStencilClip(),
			renderTarget.NumStencilBits(useMultisampleFBO))
	}
	g.flushStencil(&stencil, programInfo.Origin())
	g.flushScissorTest(programInfo.Pipeline().IsScissorTestEnabled())
	g.disableWindowRectangles()
	g.flushConservativeRasterState(programInfo.Pipeline().UsesConservativeRaster())
	g.flushWireframeState(programInfo.Pipeline().IsWireframe())

	// This must come after textures are flushed because a texture may need to be msaa-resolved (which will modify bound
	// FBO state).
	g.flushRenderTarget(renderTarget, useMultisampleFBO)

	return true
}

// prepareToDraw converts the primitive type to the GL enum.
func (g *Gpu) prepareToDraw(primitiveType gpu.PrimitiveType) uint32 {
	g.handleDirtyContext()
	switch primitiveType {
	case gpu.PrimitiveTypeTriangles:
		return TRIANGLES
	case gpu.PrimitiveTypeTriangleStrip:
		return TRIANGLE_STRIP
	case gpu.PrimitiveTypePoints:
		return POINTS
	case gpu.PrimitiveTypeLines:
		return LINES
	case gpu.PrimitiveTypeLineStrip:
		return LINE_STRIP
	}
	panic("invalid primitive type")
}

// NumGLDraws returns the number of GL geometry draw calls issued since the last ResetGLDrawStats. Together with the
// number of ops recorded for the frame it is the batching-effectiveness metric of batching effectiveness (ops recorded
// vs GL draws issued): the closer draws-issued is to draws-recorded-worth of primitives, the worse the batching. Read
// it only on the GL context thread.
func (g *Gpu) NumGLDraws() int { return g.numGLDraws }

// ResetGLDrawStats zeroes the GL-draw-call counter. Callers take a per-frame reading by resetting before recording and
// reading after flush.
func (g *Gpu) ResetGLDrawStats() { g.numGLDraws = 0 }

// didDrawTo records post-draw bookkeeping: bumps the GL-draw-call counter and, if the color buffer was written, marks
// the render target's surface as written.
func (g *Gpu) didDrawTo(rt *RenderTarget) {
	g.numGLDraws++
	if g.hwWriteToColor == triUnknown {
		panic("color write state must be known after a draw")
	}
	if g.hwWriteToColor == triYes {
		g.didWriteToSurface(rt.Surface())
	}
}

//////////////////////////////////////////////////////////////////////////////
// OpsRenderPass draw surface: the pipeline-bind/scissor/buffer-bind/draw sequence an op replays into.

// BindPipeline flushes the pipeline's transfer barrier (if any) and the GL state needed before issuing draws.
func (p *OpsRenderPass) BindPipeline(programInfo *ProgramInfo, drawBounds geom.Rect) bool {
	_ = drawBounds
	if barrierType := programInfo.Pipeline().XferBarrierType(p.gpu.Caps()); barrierType !=
		XferBarrierNone {
		p.gpu.xferBarrier(p.renderTarget, barrierType)
	}
	p.primitiveType = programInfo.PrimitiveType()
	if !p.gpu.FlushGLState(p.renderTarget, p.useMultisampleFBO, programInfo) {
		p.drawPipelineStatus = drawPipelineFailedToBind
		return false
	}
	p.drawPipelineStatus = drawPipelineOK
	return true
}

// BindTextures binds the geometry processor's texture (if any), the dst-copy texture, and the pipeline's texture-effect
// textures to their units — the common single-geometry-processor-texture form.
func (p *OpsRenderPass) BindTextures(geomProc GeometryProcessor, geomProcTexture *SurfaceProxy, pipeline *Pipeline) {
	var geomProcTextures []*SurfaceProxy
	if geomProcTexture != nil {
		geomProcTextures = []*SurfaceProxy{geomProcTexture}
	}
	p.BindTexturesMulti(geomProc, geomProcTextures, pipeline)
}

// BindTexturesMulti is the general form taking a full slice of geometry-processor textures, used by geometry processors
// with more than one sampler (the multi-page atlas text geometry processor).
func (p *OpsRenderPass) BindTexturesMulti(geomProc GeometryProcessor, geomProcTextures []*SurfaceProxy, pipeline *Pipeline) {
	if p.drawPipelineStatus != drawPipelineOK {
		return
	}
	program := p.gpu.CurrentProgram()
	if program == nil {
		panic("bindTextures without a bound pipeline")
	}
	program.BindTextures(geomProc, geomProcTextures, pipeline)
}

// SetScissorRect sets the GL scissor rectangle for subsequent draws.
func (p *OpsRenderPass) SetScissorRect(scissor geom.IRect) {
	p.gpu.flushScissorRect(scissor, p.renderTarget.Surface().Height(), p.origin)
}

// BindBuffers binds the index, instance, and vertex buffers, in that order, for subsequent draws.
func (p *OpsRenderPass) BindBuffers(indexBuffer, instanceBuffer, vertexBuffer AnyBuffer) {
	if p.drawPipelineStatus != drawPipelineOK {
		return
	}
	program := p.gpu.CurrentProgram()
	if program == nil {
		panic("bindBuffers without a bound pipeline")
	}

	numAttribs := program.NumVertexAttributes() + program.NumInstanceAttributes()
	var glIndexBuffer *Buffer
	if indexBuffer != nil {
		if indexBuffer.IsCpuBuffer() {
			// CPU index buffers bind through the null vertex array's client-side pointer lane, which the desktop-core
			// trim never takes (core profiles require buffer objects).
			panic("CPU index buffers are unreachable on core-profile GL")
		}
		glIndexBuffer = indexBuffer.(*Buffer)
	}
	p.attribArrayState = p.gpu.BindInternalVertexArray(glIndexBuffer, numAttribs,
		p.primitiveRestart)

	// Bind the instance buffer up front only when the draws won't rebind it per-baseInstance anyway, and defer the
	// vertex buffer whenever an indexed draw without baseVertexBaseInstance support will have to bake its baseVertex
	// into the attrib pointers — the draw then performs the single glVertexAttribPointer pass per attribute instead of
	// an eager offset-0 pass here plus a rebind at the real offset (B.1: that double pass was ~half the panel frame's
	// VertexAttribPointer traffic). A driver workaround for broken base-vertex support in non-indexed draws is outside
	// the desktop trim, so non-indexed draws always bind eagerly and pass baseVertex to glDrawArrays directly.
	if instanceBuffer != nil && p.gpu.glCaps().BaseVertexBaseInstanceSupport() {
		p.bindInstanceBuffer(instanceBuffer, 0)
	}
	// glIndexBuffer (not the interface value) is the "is this an indexed bind" signal: ops hand in their *Buffer field
	// directly, so a non-indexed mesh arrives as a typed-nil AnyBuffer.
	if vertexBuffer != nil &&
		(glIndexBuffer == nil || p.gpu.glCaps().BaseVertexBaseInstanceSupport()) {
		p.bindVertexBuffer(vertexBuffer, 0)
	}
	p.activeVertexBuffer = vertexBuffer
	p.activeIndexBuffer = indexBuffer
	p.activeInstanceBuffer = instanceBuffer
}

// bindInstanceBuffer binds the instance buffer's vertex attributes at the given base-instance offset.
func (p *OpsRenderPass) bindInstanceBuffer(instanceBuffer AnyBuffer, baseInstance int) {
	program := p.gpu.CurrentProgram()
	if instanceStride := program.InstanceStride(); instanceStride != 0 {
		if instanceBuffer == nil {
			panic("instance attributes without an instance buffer")
		}
		if instanceBuffer.IsCpuBuffer() {
			panic("CPU instance buffers are unreachable on core-profile GL")
		}
		glBuffer := instanceBuffer.(*Buffer)
		bufferOffset := uintptr(baseInstance) * uintptr(instanceStride)
		for i := 0; i < program.NumInstanceAttributes(); i++ {
			attrib := program.InstanceAttribute(i)
			p.attribArrayState.Set(p.gpu, attrib.Location, glBuffer, attrib.CPUType,
				attrib.GPUType.IsFloatType(), int32(instanceStride),
				bufferOffset+attrib.Offset, 1 /* divisor */)
		}
	}
}

// bindVertexBuffer binds the vertex buffer's vertex attributes at the given base-vertex offset.
func (p *OpsRenderPass) bindVertexBuffer(vertexBuffer AnyBuffer, baseVertex int) {
	program := p.gpu.CurrentProgram()
	if vertexStride := program.VertexStride(); vertexStride != 0 {
		if vertexBuffer == nil {
			panic("vertex attributes without a vertex buffer")
		}
		if vertexBuffer.IsCpuBuffer() {
			panic("CPU vertex buffers are unreachable on core-profile GL")
		}
		glBuffer := vertexBuffer.(*Buffer)
		bufferOffset := uintptr(baseVertex) * uintptr(vertexStride)
		for i := 0; i < program.NumVertexAttributes(); i++ {
			attrib := program.VertexAttribute(i)
			p.attribArrayState.Set(p.gpu, attrib.Location, glBuffer, attrib.CPUType,
				attrib.GPUType.IsFloatType(), int32(vertexStride),
				bufferOffset+attrib.Offset, 0 /* divisor */)
		}
	}
}

// Draw issues a non-indexed glDrawArrays call for the currently bound pipeline.
func (p *OpsRenderPass) Draw(vertexCount, baseVertex int) {
	if p.drawPipelineStatus != drawPipelineOK {
		return
	}
	glPrimType := p.gpu.prepareToDraw(p.primitiveType)
	p.gpu.fns().DrawArrays(glPrimType, int32(baseVertex), int32(vertexCount))
	p.gpu.didDrawTo(p.renderTarget)
}

// DrawIndexed issues an indexed draw call. With baseVertexBaseInstance support the vertex buffer stays bound at offset
// 0 (BindBuffers) and a nonzero baseVertex rides as the draw's base-vertex argument (a single-instance
// glDrawElementsInstancedBaseVertexBaseInstance); without it the vertex buffer was deferred in BindBuffers and is bound
// here, once, at the baseVertex offset. Both paths run didDrawTo, so the GL-draw count and mipmap dirtying never depend
// on caps.
func (p *OpsRenderPass) DrawIndexed(indexCount, baseIndex int, minIndexValue, maxIndexValue uint16, baseVertex int) {
	if p.drawPipelineStatus != drawPipelineOK {
		return
	}
	glPrimType := p.gpu.prepareToDraw(p.primitiveType)
	indexOffset := uintptr(baseIndex) * 2
	if p.gpu.glCaps().BaseVertexBaseInstanceSupport() {
		if baseVertex != 0 {
			p.gpu.fns().DrawElementsInstancedBaseVertexBaseInstance(glPrimType, int32(indexCount),
				UNSIGNED_SHORT, indexOffset, 1, int32(baseVertex), 0)
			p.gpu.didDrawTo(p.renderTarget)
			return
		}
	} else {
		p.bindVertexBuffer(p.activeVertexBuffer, baseVertex)
	}
	if p.gpu.glCaps().DrawRangeElementsSupport() {
		p.gpu.fns().DrawRangeElements(glPrimType, uint32(minIndexValue), uint32(maxIndexValue),
			int32(indexCount), UNSIGNED_SHORT, indexOffset)
	} else {
		p.gpu.fns().DrawElements(glPrimType, int32(indexCount), UNSIGNED_SHORT, indexOffset)
	}
	p.gpu.didDrawTo(p.renderTarget)
}

// DrawIndexPattern issues repeated DrawIndexed calls over a patterned index buffer, advancing the base vertex between
// chunks.
func (p *OpsRenderPass) DrawIndexPattern(patternIndexCount, patternRepeatCount, maxPatternRepetitionsInIndexBuffer, patternVertexCount, baseVertex int) {
	for patternRepeatCount > 0 {
		repeatCount := min(patternRepeatCount, maxPatternRepetitionsInIndexBuffer)
		p.DrawIndexed(repeatCount*patternIndexCount, 0, 0,
			uint16(repeatCount*patternVertexCount-1), baseVertex)
		patternRepeatCount -= repeatCount
		baseVertex += repeatCount * patternVertexCount
	}
}

// DrawInstanced issues an instanced glDrawArrays call (chunking for a maximum-instances-per-draw driver limit and a
// workaround for broken base-vertex support are driver bugs outside the desktop trim).
func (p *OpsRenderPass) DrawInstanced(instanceCount, baseInstance, vertexCount, baseVertex int) {
	if p.drawPipelineStatus != drawPipelineOK {
		return
	}
	glPrimType := p.gpu.prepareToDraw(p.primitiveType)
	if p.gpu.glCaps().BaseVertexBaseInstanceSupport() {
		p.gpu.fns().DrawArraysInstancedBaseInstance(glPrimType, int32(baseVertex),
			int32(vertexCount), int32(instanceCount), uint32(baseInstance))
	} else {
		p.bindInstanceBuffer(p.activeInstanceBuffer, baseInstance)
		p.gpu.fns().DrawArraysInstanced(glPrimType, int32(baseVertex), int32(vertexCount),
			int32(instanceCount))
	}
	p.gpu.didDrawTo(p.renderTarget)
}

// DrawIndexedInstanced issues an indexed, instanced draw call.
func (p *OpsRenderPass) DrawIndexedInstanced(indexCount, baseIndex, instanceCount, baseInstance, baseVertex int) {
	if p.drawPipelineStatus != drawPipelineOK {
		return
	}
	glPrimType := p.gpu.prepareToDraw(p.primitiveType)
	indexOffset := uintptr(baseIndex) * 2
	if p.gpu.glCaps().BaseVertexBaseInstanceSupport() {
		p.gpu.fns().DrawElementsInstancedBaseVertexBaseInstance(glPrimType, int32(indexCount),
			UNSIGNED_SHORT, indexOffset, int32(instanceCount), int32(baseVertex),
			uint32(baseInstance))
	} else {
		p.bindInstanceBuffer(p.activeInstanceBuffer, baseInstance)
		p.bindVertexBuffer(p.activeVertexBuffer, baseVertex)
		p.gpu.fns().DrawElementsInstanced(glPrimType, int32(indexCount), UNSIGNED_SHORT,
			indexOffset, int32(instanceCount))
	}
	p.gpu.didDrawTo(p.renderTarget)
}
