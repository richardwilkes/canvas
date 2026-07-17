// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Program is the linked program object with its attribute layout, uniform-upload manager, and render-target state
// (RT-adjust vector). The generated GLSL sources are retained for the GLSL parity shader parity harness dumps.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// ProgramAttribute is a vertex or instance attribute resolved to a GL attribute location.
type ProgramAttribute struct {
	CPUType  gpu.VertexAttribType
	GPUType  GLSLType
	Offset   uintptr
	Location int
}

// ProgramDataManager uploads uniform values to a linked GL program by handle.
type ProgramDataManager struct {
	gpu       *Gpu
	locations []int32
}

func newProgramDataManager(g *Gpu, uniforms []UniformInfo) *ProgramDataManager {
	pdm := &ProgramDataManager{gpu: g, locations: make([]int32, len(uniforms))}
	for i := range uniforms {
		pdm.locations[i] = uniforms[i].Location
	}
	return pdm
}

// setSamplerUniforms assigns texture units to sampler uniforms one time up front.
func (m *ProgramDataManager) setSamplerUniforms(g *Gpu, samplers []UniformInfo, startUnit int) {
	for i := range samplers {
		if samplers[i].Location != unusedUniformLocation {
			g.fns().Uniform1i(samplers[i].Location, int32(i+startUnit))
		}
	}
}

func (m *ProgramDataManager) location(u UniformHandle) int32 { return m.locations[u.toIndex()] }

// Set1i uploads a single int32 uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set1i(u UniformHandle, i int32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform1i(loc, i)
	}
}

// Set1f uploads a single float32 uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set1f(u UniformHandle, v0 float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform1f(loc, v0)
	}
}

// Set1fv uploads a float32 array uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set1fv(u UniformHandle, arrayCount int32, v []float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform1fv(loc, arrayCount, &v[0])
	}
}

// Set2f uploads a vec2 uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set2f(u UniformHandle, v0, v1 float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform2f(loc, v0, v1)
	}
}

// Set2fv uploads a vec2 array uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set2fv(u UniformHandle, arrayCount int32, v []float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform2fv(loc, arrayCount, &v[0])
	}
}

// Set3f uploads a vec3 uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set3f(u UniformHandle, v0, v1, v2 float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform3f(loc, v0, v1, v2)
	}
}

// Set3fv uploads a vec3 array uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set3fv(u UniformHandle, arrayCount int32, v []float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform3fv(loc, arrayCount, &v[0])
	}
}

// Set4f uploads a vec4 uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set4f(u UniformHandle, v0, v1, v2, v3 float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform4f(loc, v0, v1, v2, v3)
	}
}

// Set4fv uploads a vec4 array uniform, skipping unused uniforms.
func (m *ProgramDataManager) Set4fv(u UniformHandle, arrayCount int32, v []float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().Uniform4fv(loc, arrayCount, &v[0])
	}
}

// SetMatrix3f uploads a mat3 uniform (column-major input), skipping unused uniforms.
func (m *ProgramDataManager) SetMatrix3f(u UniformHandle, matrix *[9]float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().UniformMatrix3fv(loc, 1, false, &matrix[0])
	}
}

// SetMatrix4f uploads a mat4 uniform (column-major input), skipping unused uniforms.
func (m *ProgramDataManager) SetMatrix4f(u UniformHandle, matrix *[16]float32) {
	if loc := m.location(u); loc != unusedUniformLocation {
		m.gpu.fns().UniformMatrix4fv(loc, 1, false, &matrix[0])
	}
}

// SetMatrix uploads a row-major 3x3 geom.Matrix as the column-major mat3 GL expects.
func (m *ProgramDataManager) SetMatrix(u UniformHandle, matrix *geom.Matrix) {
	mt := [9]float32{
		matrix.Get(0), matrix.Get(3), matrix.Get(6),
		matrix.Get(1), matrix.Get(4), matrix.Get(7),
		matrix.Get(2), matrix.Get(5), matrix.Get(8),
	}
	m.SetMatrix3f(u, &mt)
}

// glRenderTargetState caches the render target size/origin last used to set the RT-adjust uniform, so
// setRenderTargetState can skip redundant uploads.
type glRenderTargetState struct {
	size    geom.ISize
	origin  gpu.SurfaceOrigin
	isValid bool
}

// Program is a linked GL program together with the processor implementations that emitted it.
type Program struct {
	// The installed effects.
	gpImpl             GPProgramImpl
	xpImpl             XPProgramImpl
	gpu                *Gpu
	programDataManager *ProgramDataManager
	fragmentSource     string
	// The generated GLSL, kept for the parity harness.
	vertexSource          string
	fpImpls               []FPProgramImpl
	attributes            []ProgramAttribute
	vertexAttributeCnt    int
	instanceAttributeCnt  int
	vertexStride          int
	instanceStride        int
	numTextureSamplers    int
	renderTargetState     glRenderTargetState
	builtinUniformHandles BuiltinUniformHandles
	programID             uint32
}

// newGLProgram wraps a linked GL program, binding it and assigning texture units to sampler uniforms one time up front.
func newGLProgram(g *Gpu, builtinUniforms BuiltinUniformHandles, programID uint32, uniforms, textureSamplers []UniformInfo, gpImpl GPProgramImpl, xpImpl XPProgramImpl, fpImpls []FPProgramImpl, attributes []ProgramAttribute, vertexAttributeCnt, instanceAttributeCnt, vertexStride, instanceStride int, vertexSource, fragmentSource string) *Program {
	p := &Program{
		gpu:                   g,
		programDataManager:    newProgramDataManager(g, uniforms),
		builtinUniformHandles: builtinUniforms,
		programID:             programID,
		gpImpl:                gpImpl,
		xpImpl:                xpImpl,
		fpImpls:               fpImpls,
		attributes:            attributes,
		vertexAttributeCnt:    vertexAttributeCnt,
		instanceAttributeCnt:  instanceAttributeCnt,
		vertexStride:          vertexStride,
		instanceStride:        instanceStride,
		numTextureSamplers:    len(textureSamplers),
		vertexSource:          vertexSource,
		fragmentSource:        fragmentSource,
	}
	g.flushProgram(p)
	p.programDataManager.setSamplerUniforms(g, textureSamplers, 0)
	return p
}

// ProgramID returns the GL program object name.
func (p *Program) ProgramID() uint32 { return p.programID }

// VertexSource returns the generated vertex GLSL (parity-harness hook).
func (p *Program) VertexSource() string { return p.vertexSource }

// FragmentSource returns the generated fragment GLSL (parity-harness hook).
func (p *Program) FragmentSource() string { return p.fragmentSource }

// Abandon marks the program object as no longer owned by this Program, without deleting it (for use when the
// underlying GL context has already been lost).
func (p *Program) Abandon() { p.programID = 0 }

// destroy deletes the underlying GL program object.
func (p *Program) destroy(g *Gpu) {
	if p.programID != 0 {
		g.fns().DeleteProgram(p.programID)
		p.programID = 0
	}
}

// VertexStride returns the byte stride between vertices.
func (p *Program) VertexStride() int { return p.vertexStride }

// InstanceStride returns the byte stride between instances.
func (p *Program) InstanceStride() int { return p.instanceStride }

// NumVertexAttributes returns the number of vertex attributes bound to this program.
func (p *Program) NumVertexAttributes() int { return p.vertexAttributeCnt }

// VertexAttribute returns the i'th vertex attribute.
func (p *Program) VertexAttribute(i int) *ProgramAttribute { return &p.attributes[i] }

// NumInstanceAttributes returns the number of instance attributes bound to this program.
func (p *Program) NumInstanceAttributes() int { return p.instanceAttributeCnt }

// InstanceAttribute returns the i'th instance attribute.
func (p *Program) InstanceAttribute(i int) *ProgramAttribute {
	return &p.attributes[i+p.vertexAttributeCnt]
}

// UpdateUniforms uploads uniforms and calls each processor impl's SetData. The caller must ensure the program is bound.
func (p *Program) UpdateUniforms(renderTarget *RenderTarget, programInfo *ProgramInfo) {
	p.setRenderTargetState(renderTarget, programInfo.Origin())

	// We set the uniforms for installed processors in a generic way; textures must be bound in the same order the
	// uniforms were set in: GP, FPs, then XP.
	p.gpImpl.SetData(p.programDataManager, p.gpu.Caps().ShaderCaps, programInfo.GeomProc())

	for i := 0; i < programInfo.Pipeline().NumFragmentProcessors(); i++ {
		fp := programInfo.Pipeline().FragmentProcessor(i)
		fpVisitWithImpls(fp, p.fpImpls[i], func(fp FragmentProcessor, impl FPProgramImpl) {
			fpImplSetData(impl, p.programDataManager, fp)
		})
	}

	programInfo.Pipeline().SetDstTextureUniforms(p.programDataManager, &p.builtinUniformHandles)
	p.xpImpl.onSetData(p.programDataManager, programInfo.Pipeline().XferProcessor())
}

// BindTextures binds the program's texture samplers in a fixed order: the geometry processor's textures first, then the
// dst texture, then the pipeline's texture effects in visit order (matching the sampler emission order in the program
// builder). geomProcTextures may be nil when the GP has no samplers, and must have one entry per GP sampler otherwise
// (the textured QuadPerEdgeAA and lattice forms sample one texture; the atlas text form samples up to four pages).
func (p *Program) BindTextures(geomProc GeometryProcessor, geomProcTextures []*SurfaceProxy, pipeline *Pipeline) {
	nextTexSamplerIdx := 0
	for i := 0; i < geomProc.gpBase().NumTextureSamplers(); i++ {
		if i >= len(geomProcTextures) || geomProcTextures[i] == nil {
			panic("GP with texture samplers requires a texture per sampler")
		}
		p.gpu.BindTexture(nextTexSamplerIdx,
			geomProc.gpBase().TextureSampler(i).SamplerState(), geomProcTextures[i].PeekTexture())
		nextTexSamplerIdx++
	}
	if dstTexture := pipeline.PeekDstTexture(); dstTexture != nil {
		p.gpu.BindTexture(nextTexSamplerIdx, gpu.SamplerState{}, dstTexture)
		nextTexSamplerIdx++
	}
	pipeline.VisitTextureEffects(func(te *TextureEffect) {
		p.gpu.BindTexture(nextTexSamplerIdx, te.SamplerState(), te.Texture())
		nextTexSamplerIdx++
	})
	if nextTexSamplerIdx != p.numTextureSamplers {
		panic("texture sampler count mismatch")
	}
}

// setRenderTargetState sets the RT-adjust vector (and RT-flip when present) whenever the render target size or origin
// changed.
func (p *Program) setRenderTargetState(rt *RenderTarget, origin gpu.SurfaceOrigin) {
	dimensions := rt.Surface().Dimensions()
	if p.renderTargetState.isValid && p.renderTargetState.origin == origin &&
		p.renderTargetState.size == dimensions {
		return
	}
	p.renderTargetState.size = dimensions
	p.renderTargetState.origin = origin
	p.renderTargetState.isValid = true

	// The client marks a swap buffer as bottom-left-origin when making a surface because GL's framebuffer space has (0,
	// 0) at the bottom left. Device coords have (0, 0) at the top left, so a flip is required when the origin is
	// bottom-left.
	flip := origin == gpu.OriginBottomLeft
	v := rtAdjustVector(dimensions, flip)
	p.programDataManager.Set4fv(p.builtinUniformHandles.RTAdjustmentUni, 1, v[:])
	if p.builtinUniformHandles.RTFlipUni.IsValid() {
		d := rtFlipVector(dimensions.Height, flip)
		p.programDataManager.Set2fv(p.builtinUniformHandles.RTFlipUni, 1, d[:])
	}
}

// rtAdjustVector computes the vec4 that rescales/translates the clip-space _position (an unnormalized device-space
// position, y-down) into normalized device coordinates gl_Position expects (y-up), flipping y when the render target's
// origin is bottom-left.
func rtAdjustVector(rtDims geom.ISize, flipY bool) [4]float32 {
	var result [4]float32
	result[0] = 2 / float32(rtDims.Width)
	result[2] = 2 / float32(rtDims.Height)
	result[1] = -1
	result[3] = -1
	if flipY {
		result[2] = -result[2]
		result[3] = -result[3]
	}
	return result
}

// rtFlipVector computes the vec2 uniform used to flip fragment-coordinate y (and dFdy sign) when the render target's
// origin is bottom-left.
func rtFlipVector(rtHeight int32, flipY bool) [2]float32 {
	if flipY {
		return [2]float32{float32(rtHeight), -1}
	}
	return [2]float32{0, 1}
}
