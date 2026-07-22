// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU state-tracking core, flattened into one Gpu type since GL is the only backend: the HW state shadow (texture
// unit bindings, buffer bind points, vertex array state, scissor/viewport, blend, color write, clear color,
// framebuffer-sRGB, bound render target), the dirty-context machinery, and framebuffer bind/delete tracking. Texture
// creation/uploads, readbacks, and sync objects live in the sibling gpu*.go files.
//
// Scope notes: program flushing works in terms of raw program IDs — the program type and the program cache live in
// glprogram.go and programcache.go; stencil shadowing here is the enable/disable tri-state plus invalidation, with the
// full stencil-settings type in stencilsettings.go; window rectangles support only the disable/invalidate paths, since
// EXT_window_rectangles is outside the desktop trim and exclusion is never enabled. Dropped with the desktop trim: the
// flush-on-framebuffer-change and unbind-attachments-on-bound-render-fbo-delete workarounds, the
// force-update-scissor-state-when-binding-fbo0 and never-disable-color-writes workarounds, the
// must-reset-blend-func-between-dual-source-and-disable workaround, the ARM advanced-blend disable dance, and the
// bind-texture0-when-changing-texture-fbo-multisample-count workaround (only reachable through a render-to-texture FBO
// reconfiguration path that desktop GL never takes).

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// BackendState bits identify the facets of GL state that may be dirtied by external GL use and reset by ResetContext.
const (
	RenderTargetBackendState   uint32 = 1 << 0
	TextureBindingBackendState uint32 = 1 << 1
	// ViewBackendState stands for scissor and viewport.
	ViewBackendState       uint32 = 1 << 2
	BlendBackendState      uint32 = 1 << 3
	MSAAEnableBackendState uint32 = 1 << 4
	VertexBackendState     uint32 = 1 << 5
	StencilBackendState    uint32 = 1 << 6
	PixelStoreBackendState uint32 = 1 << 7
	ProgramBackendState    uint32 = 1 << 8
	MiscBackendState       uint32 = 1 << 10
	AllBackendState        uint32 = 0xffff
)

// triState is a three-valued yes/no/unknown flag for tracking hardware state that may need to be queried or re-synced.
type triState int8

const (
	triNo triState = iota
	triYes
	triUnknown
)

// nativeRect is a rect in GL's bottom-up viewport/scissor coordinates.
type nativeRect struct {
	x, y, width, height int32
	valid               bool
}

// nativeRectRelativeTo converts a top-down device-space rect into GL's bottom-up native coordinates, flipping the Y
// axis relative to a render target of height rtHeight when its origin is bottom-left.
func nativeRectRelativeTo(origin gpu.SurfaceOrigin, rtHeight int32, devRect geom.IRect) nativeRect {
	y := devRect.Top
	if origin == gpu.OriginBottomLeft {
		y = rtHeight - devRect.Bottom
	}
	return nativeRect{x: devRect.Left, y: y, width: devRect.Width(), height: devRect.Height(), valid: true}
}

// hwBufferState shadows the GL binding state of one buffer bind point, so redundant rebinds of the same buffer can be
// skipped.
type hwBufferState struct {
	glTarget             uint32
	boundBufferUniqueID  gpu.ResourceUniqueID
	bufferZeroKnownBound bool
}

func (s *hwBufferState) invalidate() {
	s.boundBufferUniqueID = 0
	s.bufferZeroKnownBound = false
}

// textureUnitBindings shadows one texture unit's bindings, tracking the 2D and RECTANGLE targets (the EXTERNAL target
// is dropped with the trim).
type textureUnitBindings struct {
	boundResourceID [2]gpu.ResourceUniqueID
	boundIDValid    [2]bool
	hasBeenModified [2]bool
}

func targetBindingIndex(target uint32) int {
	switch target {
	case TEXTURE_2D:
		return 0
	case TEXTURE_RECTANGLE:
		return 1
	default:
		panic("unexpected GL texture target")
	}
}

func (t *textureUnitBindings) boundID(target uint32) (gpu.ResourceUniqueID, bool) {
	i := targetBindingIndex(target)
	return t.boundResourceID[i], t.boundIDValid[i]
}

func (t *textureUnitBindings) setBoundID(target uint32, id gpu.ResourceUniqueID) {
	i := targetBindingIndex(target)
	t.boundResourceID[i] = id
	t.boundIDValid[i] = true
	t.hasBeenModified[i] = true
}

func (t *textureUnitBindings) invalidateForScratchUse(target uint32) {
	// Setting an invalid (zero) id still counts as "known" so the next real bind compares against it; this is the same
	// bookkeeping setBoundID performs, using a default UniqueID.
	i := targetBindingIndex(target)
	t.boundResourceID[i] = 0
	t.boundIDValid[i] = true
	t.hasBeenModified[i] = true
}

func (t *textureUnitBindings) targetHasBeenModified(target uint32) bool {
	return t.hasBeenModified[targetBindingIndex(target)]
}

func (t *textureUnitBindings) invalidateAllTargets(markUnmodified bool) {
	for i := range t.boundIDValid {
		t.boundIDValid[i] = false
		if markUnmodified {
			t.hasBeenModified[i] = false
		}
	}
}

// hwBlendState shadows the GL blend equation/coefficients/enable state.
type hwBlendState struct {
	equation        gpu.BlendEquation
	srcCoeff        gpu.BlendCoeff
	dstCoeff        gpu.BlendCoeff
	constColor      [4]float32
	constColorValid bool
	enabled         triState
}

func (s *hwBlendState) invalidate() {
	s.equation = gpu.BlendEquationIllegal
	s.srcCoeff = gpu.BlendCoeffIllegal
	s.dstCoeff = gpu.BlendCoeffIllegal
	s.constColorValid = false
	s.enabled = triUnknown
}

// hwVertexArrayState shadows the currently bound vertex array object and its attribute state.
type hwVertexArrayState struct {
	defaultVertexArrayAttribState *AttribArrayState
	// coreProfileVertexArray is the placeholder VAO used for internal draws on core profiles (where VAO zero does not
	// exist).
	coreProfileVertexArray    *VertexArray
	boundVertexArrayID        uint32
	boundVertexArrayIDIsValid bool
}

func (v *hwVertexArrayState) invalidate() {
	v.boundVertexArrayIDIsValid = false
	if v.defaultVertexArrayAttribState != nil {
		v.defaultVertexArrayAttribState.Invalidate()
	}
	if v.coreProfileVertexArray != nil {
		v.coreProfileVertexArray.InvalidateCachedState()
	}
}

func (v *hwVertexArrayState) notifyVertexArrayDelete(id uint32) {
	if v.boundVertexArrayIDIsValid && v.boundVertexArrayID == id {
		// Deleting the bound VAO implicitly binds 0.
		v.boundVertexArrayID = 0
	}
}

func (v *hwVertexArrayState) setVertexArrayID(g *Gpu, arrayID uint32) {
	if !g.glCaps().VertexArrayObjectSupport() {
		if arrayID != 0 {
			panic("VAO used without support")
		}
		return
	}
	if !v.boundVertexArrayIDIsValid || arrayID != v.boundVertexArrayID {
		g.fns().BindVertexArray(arrayID)
		v.boundVertexArrayIDIsValid = true
		v.boundVertexArrayID = arrayID
	}
}

// bindInternalVertexArray binds (creating if needed) the vertex array used for internal draws, returning its attribute
// state.
func (v *hwVertexArrayState) bindInternalVertexArray(g *Gpu, indexBuffer *Buffer) *AttribArrayState {
	if g.glCaps().IsCoreProfile() {
		if v.coreProfileVertexArray == nil {
			var arrayID uint32
			g.fns().GenVertexArrays(1, &arrayID)
			v.coreProfileVertexArray = NewVertexArray(arrayID, g.Caps().MaxVertexAttributes)
		}
		if indexBuffer != nil {
			return v.coreProfileVertexArray.BindWithIndexBuffer(g, indexBuffer)
		}
		return v.coreProfileVertexArray.Bind(g)
	}
	if indexBuffer != nil {
		// bindBuffer implicitly binds VAO 0 when binding an index buffer.
		g.bindBuffer(gpu.BufferTypeIndex, indexBuffer)
	} else {
		v.setVertexArrayID(g, 0)
	}
	attrCount := g.Caps().MaxVertexAttributes
	if v.defaultVertexArrayAttribState == nil {
		v.defaultVertexArrayAttribState = NewAttribArrayState(attrCount)
	} else if v.defaultVertexArrayAttribState.Count() != attrCount {
		v.defaultVertexArrayAttribState.Resize(attrCount)
	}
	return v.defaultVertexArrayAttribState
}

// Gpu is the GL backend object: it owns the resource cache and the GL state shadow. All methods must be called on the
// thread that owns the GL context.
type Gpu struct {
	hwVertexArrayState hwVertexArrayState
	// programCache is the in-memory LRU of linked programs.
	programCache *ProgramCache
	cache        *gpu.ResourceCache
	// cachedOpsRenderPass is the single reused render pass.
	cachedOpsRenderPass                *OpsRenderPass
	samplerObjectCache                 *samplerObjectCache
	hwProgram                          *Program
	mipmapProgramArrayBuffer           *Buffer
	ctx                                *ContextInfo
	finishCallbacks                    []finishCallback
	hwTextureUnitBindings              []textureUnitBindings
	resetTimestampForTextureParameters TextureResetTimestamp
	// GL state shadow.
	hwActiveTextureUnitIdx int
	// numGLDraws counts the GL geometry draw calls issued (glDrawArrays/glDrawElements and their instanced/range
	// variants) since the last ResetGLDrawStats. It is the "GL draws issued" side of the batching metric (batching
	// effectiveness = ops recorded vs GL draws issued). It is touched only from didDrawTo on the GL context thread, so
	// it needs no synchronization.
	numGLDraws    int
	hwBufferState [gpu.BufferTypeCount]hwBufferState
	// The manual mipmap-regeneration programs, built lazily outside the program cache, plus their shared unit-quad
	// vertex buffer. See mipmapprogram.go.
	mipmapPrograms    [4]mipmapProgram
	hwStencilSettings StencilSettings
	hwBlendState      hwBlendState
	hwScissorRect     nativeRect
	hwViewport        nativeRect
	hwClearColor      [4]float32
	tempDstFBOID      uint32
	tempSrcFBOID      uint32
	// resetBits are the dirty-context bits consumed by handleDirtyContext.
	resetBits                   uint32
	hwProgramID                 uint32
	boundDrawFramebuffer        uint32
	hwStencilOrigin             gpu.SurfaceOrigin
	stencilClearFBOID           uint32
	hwBoundRenderTargetUniqueID gpu.ResourceUniqueID
	hwBoundFramebufferIsMSAA    bool
	hwSRGBFramebuffer           triState
	hwScissorEnabled            triState
	hwWriteToColor              triState
	// hwWindowRectsDisabledKnown stands in for the full HWWindowRectsState: since exclusion is never enabled, it
	// records only whether window rectangles are known to be disabled.
	hwWindowRectsDisabledKnown  bool
	hwConservativeRasterEnabled triState
	hwStencilTestEnabled        triState
	oomed                       bool
	abandoned                   bool
	needsGLFlush                bool
	hwWireframeEnabled          triState
}

// NewGpu builds a Gpu driving the given GL context: ctx must be a validated context (NewContextInfo).
func NewGpu(ctx *ContextInfo) *Gpu {
	if ctx == nil || ctx.Caps == nil {
		return nil
	}
	g := &Gpu{
		ctx:   ctx,
		cache: gpu.NewResourceCache(),
	}
	g.programCache = newProgramCache(g, programCacheMaxEntries)
	// Clear errors so we don't get confused whether we caused an error, and toss out any pre-existing OOM hanging
	// around before we got started.
	g.clearErrorsAndCheckForOOM()
	g.oomed = false

	g.hwTextureUnitBindings = make([]textureUnitBindings, g.numTextureUnits())

	g.hwBufferState[gpu.BufferTypeVertex].glTarget = ARRAY_BUFFER
	g.hwBufferState[gpu.BufferTypeIndex].glTarget = ELEMENT_ARRAY_BUFFER
	g.hwBufferState[gpu.BufferTypeDrawIndirect].glTarget = DRAW_INDIRECT_BUFFER
	g.hwBufferState[gpu.BufferTypeXferCpuToGpu].glTarget = PIXEL_UNPACK_BUFFER
	g.hwBufferState[gpu.BufferTypeXferGpuToCpu].glTarget = PIXEL_PACK_BUFFER
	for i := range g.hwBufferState {
		g.hwBufferState[i].invalidate()
	}

	if g.glCaps().UseSamplerObjects() {
		g.samplerObjectCache = newSamplerObjectCache(g)
	}

	g.MarkContextDirty(AllBackendState)
	return g
}

// ContextInfo returns the GL context description this Gpu drives.
func (g *Gpu) ContextInfo() *ContextInfo { return g.ctx }

// Caps returns the backend-independent caps.
func (g *Gpu) Caps() *gpu.Caps { return &g.ctx.Caps.Caps }

// GLCaps returns the GL caps.
func (g *Gpu) GLCaps() *Caps { return g.ctx.Caps }

func (g *Gpu) glCaps() *Caps { return g.ctx.Caps }

func (g *Gpu) fns() *Functions { return &g.ctx.Interface.Functions }

// Functions returns the GL proc table.
func (g *Gpu) Functions() *Functions { return g.fns() }

// ResourceCache returns the cache that owns this Gpu's resources.
func (g *Gpu) ResourceCache() *gpu.ResourceCache { return g.cache }

func (g *Gpu) numTextureUnits() int { return g.Caps().ShaderCaps.MaxFragmentSamplers }

// Abandoned reports whether the context backing this Gpu has been abandoned.
func (g *Gpu) Abandoned() bool { return g.abandoned }

// AbandonContext drops backend objects without issuing GL calls (the context is assumed invalid).
func (g *Gpu) AbandonContext() {
	if g.abandoned {
		return
	}
	g.abandoned = true
	// Disconnect every cached GL object without touching the (assumed-dead) context.
	g.hwProgram = nil
	g.hwProgramID = 0
	g.programCache.Abandon()
	g.tempSrcFBOID = 0
	g.tempDstFBOID = 0
	g.stencilClearFBOID = 0
	if g.mipmapProgramArrayBuffer != nil {
		g.mipmapProgramArrayBuffer.Unref()
		g.mipmapProgramArrayBuffer = nil
	}
	for i := range g.mipmapPrograms {
		g.mipmapPrograms[i].program = 0
	}
	if g.samplerObjectCache != nil {
		g.samplerObjectCache.abandon()
	}
	g.callAllFinishedCallbacks(false)
	g.cache.AbandonAll()
}

// ReleaseResourcesAndAbandonContext is like AbandonContext, but backend objects are freed first (the context must be
// current).
func (g *Gpu) ReleaseResourcesAndAbandonContext() {
	if g.abandoned {
		return
	}
	g.abandoned = true
	// Disconnect every cached GL object, freeing each one via a real GL call first.
	if g.hwProgramID != 0 {
		g.fns().UseProgram(0)
		g.hwProgramID = 0
	}
	g.hwProgram = nil
	g.programCache.Reset()
	if g.tempSrcFBOID != 0 {
		g.DeleteFramebuffer(g.tempSrcFBOID)
		g.tempSrcFBOID = 0
	}
	if g.tempDstFBOID != 0 {
		g.DeleteFramebuffer(g.tempDstFBOID)
		g.tempDstFBOID = 0
	}
	if g.stencilClearFBOID != 0 {
		g.DeleteFramebuffer(g.stencilClearFBOID)
		g.stencilClearFBOID = 0
	}
	if g.mipmapProgramArrayBuffer != nil {
		g.mipmapProgramArrayBuffer.Unref()
		g.mipmapProgramArrayBuffer = nil
	}
	for i := range g.mipmapPrograms {
		if g.mipmapPrograms[i].program != 0 {
			g.fns().DeleteProgram(g.mipmapPrograms[i].program)
			g.mipmapPrograms[i].program = 0
		}
	}
	if g.samplerObjectCache != nil {
		g.samplerObjectCache.release()
	}
	g.callAllFinishedCallbacks(true)
	g.cache.ReleaseAll()
}

// MarkContextDirty records that state was modified externally and must be re-emitted before the next GL use.
func (g *Gpu) MarkContextDirty(state uint32) { g.resetBits |= state }

// handleDirtyContext re-emits any GL state marked dirty since the last reset.
func (g *Gpu) handleDirtyContext() {
	if g.resetBits != 0 {
		g.resetContext()
	}
}

// ResetContext is equivalent to MarkContextDirty(state) — the actual reset happens lazily before the next GL work.
func (g *Gpu) ResetContext(state uint32) { g.MarkContextDirty(state) }

// resetContext re-emits the pieces of GL state marked dirty in g.resetBits, then clears them.
func (g *Gpu) resetContext() {
	resetBits := g.resetBits
	g.resetBits = 0
	f := g.fns()

	if resetBits&MiscBackendState != 0 {
		// We don't use the zb at all.
		f.Disable(DEPTH_TEST)
		f.DepthMask(false)

		// We don't use face culling.
		f.Disable(CULL_FACE)
		// We do use separate stencil. Our algorithms don't care which face is front vs. back so just set this to the
		// default for self-consistency.
		f.FrontFace(CCW)

		g.hwBufferState[gpu.BufferTypeXferCpuToGpu].invalidate()
		g.hwBufferState[gpu.BufferTypeXferGpuToCpu].invalidate()

		// Desktop-only state that we never change.
		if !g.glCaps().IsCoreProfile() {
			f.Disable(POINT_SMOOTH)
			f.Disable(LINE_SMOOTH)
			f.Disable(POLYGON_SMOOTH)
			f.Disable(POLYGON_STIPPLE)
			f.Disable(COLOR_LOGIC_OP)
			f.Disable(INDEX_LOGIC_OP)
		}
		// The Windows NVIDIA driver has GL_ARB_imaging in the extension string when using a core profile, which seems
		// like a bug since the core spec removes any mention of it.
		if g.glCaps().ImagingSupport() && !g.glCaps().IsCoreProfile() {
			f.Disable(COLOR_TABLE)
		}
		f.Disable(POLYGON_OFFSET_FILL)
		g.hwWireframeEnabled = triUnknown

		// Since ES doesn't support glPointSize at all we always use the VS to set the point size.
		f.Enable(VERTEX_PROGRAM_POINT_SIZE)

		g.hwWriteToColor = triUnknown
		// We only ever use lines in hairline mode.
		f.LineWidth(1)
		f.Disable(DITHER)

		nan := float32(math.NaN())
		g.hwClearColor = [4]float32{nan, nan, nan, nan}
	}

	if resetBits&MSAAEnableBackendState != 0 {
		if g.glCaps().ClientCanDisableMultisample() {
			// Restore GL_MULTISAMPLE to its initial state. It being enabled has no effect on draws to non-MSAA targets.
			f.Enable(MULTISAMPLE)
		}
		g.hwConservativeRasterEnabled = triUnknown
	}

	g.hwActiveTextureUnitIdx = -1 // invalid

	if resetBits&TextureBindingBackendState != 0 {
		for i := range g.hwTextureUnitBindings {
			g.hwTextureUnitBindings[i].invalidateAllTargets(false)
		}
		if g.samplerObjectCache != nil {
			g.samplerObjectCache.invalidateBindings()
		}
	}

	if resetBits&BlendBackendState != 0 {
		g.hwBlendState.invalidate()
	}

	if resetBits&ViewBackendState != 0 {
		g.hwScissorEnabled = triUnknown
		g.hwScissorRect = nativeRect{}
		g.hwWindowRectsDisabledKnown = false
		g.hwViewport = nativeRect{}
	}

	if resetBits&StencilBackendState != 0 {
		g.hwStencilSettings.Invalidate()
		g.hwStencilTestEnabled = triUnknown
	}

	if resetBits&VertexBackendState != 0 {
		g.hwVertexArrayState.invalidate()
		g.hwBufferState[gpu.BufferTypeVertex].invalidate()
		g.hwBufferState[gpu.BufferTypeIndex].invalidate()
		g.hwBufferState[gpu.BufferTypeDrawIndirect].invalidate()
	}

	if resetBits&RenderTargetBackendState != 0 {
		g.hwBoundRenderTargetUniqueID = 0
		g.hwSRGBFramebuffer = triUnknown
		g.boundDrawFramebuffer = 0
	}

	// We assume these values.
	if resetBits&PixelStoreBackendState != 0 {
		if g.Caps().WritePixelsRowBytesSupport || g.Caps().TransferPixelsToRowBytesSupport {
			f.PixelStorei(UNPACK_ROW_LENGTH, 0)
		}
		if g.Caps().ReadPixelsRowBytesSupport {
			f.PixelStorei(PACK_ROW_LENGTH, 0)
		}
	}

	if resetBits&ProgramBackendState != 0 {
		g.hwProgramID = 0
		g.hwProgram = nil
	}
	g.resetTimestampForTextureParameters++
}

// ResetTextureBindings clears the tracked GL texture bindings, unbinding any texture on a unit whose binding this Gpu
// has modified.
func (g *Gpu) ResetTextureBindings() {
	targets := [...]uint32{TEXTURE_2D, TEXTURE_RECTANGLE}
	for i := range g.hwTextureUnitBindings {
		g.setTextureUnit(i)
		for _, target := range targets {
			if g.hwTextureUnitBindings[i].targetHasBeenModified(target) {
				g.fns().BindTexture(target, 0)
			}
		}
		g.hwTextureUnitBindings[i].invalidateAllTargets(true)
	}
}

// setTextureUnit makes the given texture unit active, skipping the GL call if it already is.
func (g *Gpu) setTextureUnit(unit int) {
	if unit < 0 || unit >= g.numTextureUnits() {
		panic("texture unit out of range")
	}
	if unit != g.hwActiveTextureUnitIdx {
		g.fns().ActiveTexture(TEXTURE0 + uint32(unit))
		g.hwActiveTextureUnitIdx = unit
	}
}

// bindTextureToScratchUnit binds a texture on the last unit for operations outside the normal draw flow.
func (g *Gpu) bindTextureToScratchUnit(target, textureID uint32) {
	// The last unit is the least likely to be used by a program.
	lastUnitIdx := g.numTextureUnits() - 1
	if lastUnitIdx != g.hwActiveTextureUnitIdx {
		g.fns().ActiveTexture(TEXTURE0 + uint32(lastUnitIdx))
		g.hwActiveTextureUnitIdx = lastUnitIdx
	}
	// Clear out the tracking so that if a program uses this unit it will rebind the correct texture.
	g.hwTextureUnitBindings[lastUnitIdx].invalidateForScratchUse(target)
	g.fns().BindTexture(target, textureID)
}

// BindTexture binds the texture on the given unit and flushes its sampler parameters, skipping redundant GL calls via
// the parameter shadow.
func (g *Gpu) BindTexture(unitIdx int, samplerState gpu.SamplerState, texture *Texture) {
	if texture == nil {
		panic("nil texture")
	}
	if !g.Caps().NPOTTextureTileSupport {
		if samplerState.IsRepeatedX() || samplerState.IsRepeatedY() {
			s := texture.Surface().Dimensions()
			if s.Width&(s.Width-1) != 0 || s.Height&(s.Height-1) != 0 {
				panic("NPOT repeat without support")
			}
		}
	}

	textureID := texture.Surface().UniqueID()
	target := texture.Target()
	if id, ok := g.hwTextureUnitBindings[unitIdx].boundID(target); !ok || id != textureID {
		g.setTextureUnit(unitIdx)
		g.fns().BindTexture(target, texture.TextureID())
		g.hwTextureUnitBindings[unitIdx].setBoundID(target, textureID)
	}

	if samplerState.Mipmapped() == gpu.MipmappedYes {
		if !g.Caps().MipmapSupport || texture.Mipmapped() == gpu.MipmappedNo {
			samplerState = gpu.MakeSamplerState(samplerState.WrapModeX, samplerState.WrapModeY,
				samplerState.Filter, gpu.MipmapModeNone)
		} else if texture.MipmapsAreDirty() {
			panic("sampling dirty mipmaps")
		}
	}

	setAll := texture.Parameters().ResetTimestamp() < g.resetTimestampForTextureParameters
	var samplerStateToRecord *SamplerOverriddenState
	var newSamplerState SamplerOverriddenState
	if g.glCaps().UseSamplerObjects() {
		g.samplerObjectCache.bindSampler(unitIdx, samplerState)
		// mustSetAnyTexParameterToEnableMipmapping is a mobile workaround dropped with the trim.
	} else {
		if g.samplerObjectCache != nil {
			g.samplerObjectCache.unbindSampler(unitIdx)
		}
		oldSamplerState := texture.Parameters().SamplerOverriddenState()
		samplerStateToRecord = &newSamplerState

		newSamplerState.MinFilter = filterToGLMinFilter(samplerState.Filter, samplerState.MipmapMode)
		newSamplerState.MagFilter = filterToGLMagFilter(samplerState.Filter)
		newSamplerState.WrapS = wrapModeToGLWrap(samplerState.WrapModeX, g.Caps())
		newSamplerState.WrapT = wrapModeToGLWrap(samplerState.WrapModeY, g.Caps())
		newSamplerState.MaxAniso = min(float32(samplerState.MaxAniso),
			g.glCaps().MaxTextureMaxAnisotropy())
		// These are the OpenGL default values.
		newSamplerState.MinLOD = -1000
		newSamplerState.MaxLOD = 1000

		if setAll || newSamplerState.MagFilter != oldSamplerState.MagFilter {
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteri(target, TEXTURE_MAG_FILTER, int32(newSamplerState.MagFilter))
		}
		if setAll || newSamplerState.MinFilter != oldSamplerState.MinFilter {
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteri(target, TEXTURE_MIN_FILTER, int32(newSamplerState.MinFilter))
		}
		if g.glCaps().MipmapLodControlSupport() {
			if setAll || newSamplerState.MinLOD != oldSamplerState.MinLOD {
				g.setTextureUnit(unitIdx)
				g.fns().TexParameterf(target, TEXTURE_MIN_LOD, newSamplerState.MinLOD)
			}
			if setAll || newSamplerState.MaxLOD != oldSamplerState.MaxLOD {
				g.setTextureUnit(unitIdx)
				g.fns().TexParameterf(target, TEXTURE_MAX_LOD, newSamplerState.MaxLOD)
			}
		}
		if setAll || newSamplerState.WrapS != oldSamplerState.WrapS {
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteri(target, TEXTURE_WRAP_S, int32(newSamplerState.WrapS))
		}
		if setAll || newSamplerState.WrapT != oldSamplerState.WrapT {
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteri(target, TEXTURE_WRAP_T, int32(newSamplerState.WrapT))
		}
		if g.Caps().ClampToBorderSupport {
			// Make sure the border color is transparent black (the default).
			if setAll || oldSamplerState.BorderColorInvalid {
				g.setTextureUnit(unitIdx)
				transparentBlack := [4]float32{0, 0, 0, 0}
				g.fns().TexParameterfv(target, TEXTURE_BORDER_COLOR, &transparentBlack[0])
			}
		}
		if g.Caps().AnisoSupport {
			if setAll || oldSamplerState.MaxAniso != newSamplerState.MaxAniso {
				g.fns().TexParameterf(target, TEXTURE_MAX_ANISOTROPY, newSamplerState.MaxAniso)
			}
		}
	}

	newNonsamplerState := NonsamplerState{
		BaseMipmapLevel: 0,
		MaxMipmapLevel:  int32(texture.MaxMipmapLevel()),
		SwizzleIsRGBA:   true,
	}
	oldNonsamplerState := texture.Parameters().NonsamplerState()
	if g.glCaps().TextureSwizzleSupport() {
		if setAll || !oldNonsamplerState.SwizzleIsRGBA {
			rgba := [4]int32{RED, GREEN, BLUE, ALPHA}
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteriv(target, TEXTURE_SWIZZLE_RGBA, &rgba[0])
		}
	}
	// These are not supported in ES2 contexts.
	if g.glCaps().MipmapLevelControlSupport() {
		if newNonsamplerState.BaseMipmapLevel != oldNonsamplerState.BaseMipmapLevel {
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteri(target, TEXTURE_BASE_LEVEL, newNonsamplerState.BaseMipmapLevel)
		}
		if newNonsamplerState.MaxMipmapLevel != oldNonsamplerState.MaxMipmapLevel {
			g.setTextureUnit(unitIdx)
			g.fns().TexParameteri(target, TEXTURE_MAX_LEVEL, newNonsamplerState.MaxMipmapLevel)
		}
	}
	texture.Parameters().Set(samplerStateToRecord, newNonsamplerState,
		g.resetTimestampForTextureParameters)
}

func filterToGLMagFilter(filter gpu.FilterMode) uint32 {
	switch filter {
	case gpu.FilterModeNearest:
		return NEAREST
	case gpu.FilterModeLinear:
		return LINEAR
	}
	panic("unknown filter")
}

func filterToGLMinFilter(filter gpu.FilterMode, mm gpu.MipmapMode) uint32 {
	switch mm {
	case gpu.MipmapModeNone:
		return filterToGLMagFilter(filter)
	case gpu.MipmapModeNearest:
		if filter == gpu.FilterModeNearest {
			return NEAREST_MIPMAP_NEAREST
		}
		return LINEAR_MIPMAP_NEAREST
	case gpu.MipmapModeLinear:
		if filter == gpu.FilterModeNearest {
			return NEAREST_MIPMAP_LINEAR
		}
		return LINEAR_MIPMAP_LINEAR
	}
	panic("unknown mipmap mode")
}

func wrapModeToGLWrap(wrapMode gpu.WrapMode, caps *gpu.Caps) uint32 {
	switch wrapMode {
	case gpu.WrapModeClamp:
		return CLAMP_TO_EDGE
	case gpu.WrapModeRepeat:
		return REPEAT
	case gpu.WrapModeMirrorRepeat:
		return MIRRORED_REPEAT
	case gpu.WrapModeClampToBorder:
		if !caps.ClampToBorderSupport {
			panic("clamp-to-border without support")
		}
		return CLAMP_TO_BORDER
	}
	panic("unknown wrap mode")
}

// BindVertexArray performs a tracked VAO binding, skipping the GL call if it is already bound.
func (g *Gpu) BindVertexArray(id uint32) {
	g.hwVertexArrayState.setVertexArrayID(g, id)
}

// NotifyVertexArrayDelete tells the vertex array state shadow that a VAO was deleted, so a stale binding is never
// compared as still current.
func (g *Gpu) NotifyVertexArrayDelete(id uint32) {
	g.hwVertexArrayState.notifyVertexArrayDelete(id)
}

// BindInternalVertexArray binds the vertex array used for internal draws, enables numAttribs arrays, and flushes
// primitive restart state.
func (g *Gpu) BindInternalVertexArray(indexBuffer *Buffer, numAttribs int, primitiveRestart bool) *AttribArrayState {
	attribState := g.hwVertexArrayState.bindInternalVertexArray(g, indexBuffer)
	attribState.EnableVertexArrays(g, numAttribs, primitiveRestart)
	return attribState
}

// bindBuffer binds a buffer to the GL target corresponding to its type, tracking state to skip redundant binds, and
// returns the GL target bound. Binding an index buffer implicitly binds VAO 0.
func (g *Gpu) bindBuffer(bufferType gpu.BufferType, buffer *Buffer) uint32 {
	g.handleDirtyContext()

	// Index buffer state is tied to the vertex array.
	if bufferType == gpu.BufferTypeIndex {
		g.BindVertexArray(0)
	}

	state := &g.hwBufferState[bufferType]
	if buffer.UniqueID() != state.boundBufferUniqueID {
		g.fns().BindBuffer(state.glTarget, buffer.BufferID())
		state.bufferZeroKnownBound = false
		state.boundBufferUniqueID = buffer.UniqueID()
	}
	return state.glTarget
}

// BindBuffer is the exported form of bindBuffer.
func (g *Gpu) BindBuffer(bufferType gpu.BufferType, buffer *Buffer) uint32 {
	return g.bindBuffer(bufferType, buffer)
}

// unbindXferBuffer unbinds a transfer buffer target before it might interfere with a call that expects client memory:
// before TexSubImage-style calls, unbind the PIXEL_UNPACK_BUFFER; before ReadPixels into CPU memory, unbind the
// PIXEL_PACK_BUFFER.
func (g *Gpu) unbindXferBuffer(bufferType gpu.BufferType) {
	if g.glCaps().TransferBufferType() != TransferBufferARBPBO {
		return
	}
	if bufferType != gpu.BufferTypeXferCpuToGpu && bufferType != gpu.BufferTypeXferGpuToCpu {
		panic("not a transfer buffer type")
	}
	state := &g.hwBufferState[bufferType]
	if !state.bufferZeroKnownBound {
		g.fns().BindBuffer(state.glTarget, 0)
		state.boundBufferUniqueID = 0
		state.bufferZeroKnownBound = true
	}
}

// BindFramebuffer performs a tracked framebuffer bind; use instead of a raw glBindFramebuffer call.
func (g *Gpu) BindFramebuffer(target, fboID uint32) {
	g.fns().BindFramebuffer(target, fboID)
	if target == FRAMEBUFFER || target == DRAW_FRAMEBUFFER {
		g.boundDrawFramebuffer = fboID
	}
}

// DeleteFramebuffer deletes the framebuffer and updates the tracked binding if it was current.
func (g *Gpu) DeleteFramebuffer(fboID uint32) {
	// We're relying on the GL state shadowing being correct below, so handle a dirty context.
	g.handleDirtyContext()
	g.fns().DeleteFramebuffers(1, &fboID)
	// Deleting the currently bound framebuffer rebinds to 0.
	if fboID == g.boundDrawFramebuffer {
		g.boundDrawFramebuffer = 0
	}
}

// InvalidateBoundRenderTarget forgets the tracked bound render target, forcing the next draw to rebind it.
func (g *Gpu) InvalidateBoundRenderTarget() { g.hwBoundRenderTargetUniqueID = 0 }

// flushScissorTest enables or disables the scissor test, skipping the GL call if it is already in the requested state.
func (g *Gpu) flushScissorTest(enabled bool) {
	if enabled {
		if g.hwScissorEnabled != triYes {
			g.fns().Enable(SCISSOR_TEST)
			g.hwScissorEnabled = triYes
		}
	} else if g.hwScissorEnabled != triNo {
		g.fns().Disable(SCISSOR_TEST)
		g.hwScissorEnabled = triNo
	}
}

// flushScissorRect sets the scissor rect, converting to native (bottom-up) coordinates and skipping the GL call if it
// matches the currently set rect.
func (g *Gpu) flushScissorRect(scissor geom.IRect, rtHeight int32, origin gpu.SurfaceOrigin) {
	if g.hwScissorEnabled != triYes {
		panic("scissor test not enabled")
	}
	native := nativeRectRelativeTo(origin, rtHeight, scissor)
	if g.hwScissorRect != native {
		g.fns().Scissor(native.x, native.y, native.width, native.height)
		g.hwScissorRect = native
	}
}

// flushScissor enables or disables the scissor test and, when enabled, sets its rect.
func (g *Gpu) flushScissor(state *gpu.ScissorState, rtHeight int32, origin gpu.SurfaceOrigin) {
	g.flushScissorTest(state.Enabled())
	if state.Enabled() {
		g.flushScissorRect(state.Rect(), rtHeight, origin)
	}
}

// flushViewport sets the GL viewport, converting to native (bottom-up) coordinates and skipping the GL call if it
// matches the currently set viewport.
func (g *Gpu) flushViewport(viewport geom.IRect, rtHeight int32, origin gpu.SurfaceOrigin) {
	native := nativeRectRelativeTo(origin, rtHeight, viewport)
	if g.hwViewport != native {
		g.fns().Viewport(native.x, native.y, native.width, native.height)
		g.hwViewport = native
	}
}

// disableWindowRectangles turns off window-rectangle exclusion. Only the disabled state is ever shadowed, since
// exclusion is never enabled.
func (g *Gpu) disableWindowRectangles() {
	if g.Caps().MaxWindowRectanglesCap == 0 || g.hwWindowRectsDisabledKnown {
		return
	}
	g.fns().WindowRectangles(EXCLUSIVE, 0, 0)
	g.hwWindowRectsDisabledKnown = true
}

// flushColorWrite enables or disables writing to the color buffer, skipping the GL call if it is already in the
// requested state.
func (g *Gpu) flushColorWrite(writeColor bool) {
	if !writeColor {
		if g.hwWriteToColor != triNo {
			g.fns().ColorMask(false, false, false, false)
			g.hwWriteToColor = triNo
		}
	} else if g.hwWriteToColor != triYes {
		g.fns().ColorMask(true, true, true, true)
		g.hwWriteToColor = triYes
	}
}

// flushClearColor sets the GL clear color, skipping the GL call if it matches the currently set color (a mobile-only
// clear-to-boundary-values workaround is dropped with the desktop trim).
func (g *Gpu) flushClearColor(color [4]float32) {
	if color != g.hwClearColor {
		g.fns().ClearColor(color[0], color[1], color[2], color[3])
		g.hwClearColor = color
	}
}

// flushFramebufferSRGB enables or disables FRAMEBUFFER_SRGB, skipping the GL call if it is already in the requested
// state.
func (g *Gpu) flushFramebufferSRGB(enable bool) {
	if enable && g.hwSRGBFramebuffer != triYes {
		g.fns().Enable(FRAMEBUFFER_SRGB)
		g.hwSRGBFramebuffer = triYes
	} else if !enable && g.hwSRGBFramebuffer != triNo {
		g.fns().Disable(FRAMEBUFFER_SRGB)
		g.hwSRGBFramebuffer = triNo
	}
}

// disableStencil turns off the stencil test, skipping the GL call if it is already disabled.
func (g *Gpu) disableStencil() {
	if g.hwStencilTestEnabled != triNo {
		g.fns().Disable(STENCIL_TEST)
		g.hwStencilTestEnabled = triNo
		g.hwStencilSettings.Invalidate()
	}
}

// stencilTestToGL translates a StencilTest to its GL comparison function enum.
func stencilTestToGL(test StencilTest) uint32 {
	switch test {
	case StencilTestAlways:
		return ALWAYS
	case StencilTestNever:
		return NEVER
	case StencilTestGreater:
		return GREATER
	case StencilTestGEqual:
		return GEQUAL
	case StencilTestLess:
		return LESS
	case StencilTestLEqual:
		return LEQUAL
	case StencilTestEqual:
		return EQUAL
	case StencilTestNotEqual:
		return NOTEQUAL
	}
	panic("invalid stencil test")
}

// stencilOpToGL translates a StencilOp to its GL stencil-op enum.
func stencilOpToGL(op StencilOp) uint32 {
	switch op {
	case StencilOpKeep:
		return KEEP
	case StencilOpZero:
		return ZERO
	case StencilOpReplace:
		return REPLACE
	case StencilOpInvert:
		return INVERT
	case StencilOpIncWrap:
		return INCR_WRAP
	case StencilOpDecWrap:
		return DECR_WRAP
	case StencilOpIncClamp:
		return INCR
	case StencilOpDecClamp:
		return DECR
	}
	panic("invalid stencil op")
}

// setGLStencil applies face's stencil test/ops/mask to the given GL face(s) (front, back, or both).
func (g *Gpu) setGLStencil(face *StencilFace, glFace uint32) {
	glFunc := stencilTestToGL(face.Test)
	glFailOp := stencilOpToGL(face.FailOp)
	glPassOp := stencilOpToGL(face.PassOp)

	if glFace == FRONT_AND_BACK {
		// We call the combined func just in case separate stencil is not supported.
		g.fns().StencilFunc(glFunc, int32(face.Ref), uint32(face.TestMask))
		g.fns().StencilMask(uint32(face.WriteMask))
		g.fns().StencilOp(glFailOp, KEEP, glPassOp)
	} else {
		g.fns().StencilFuncSeparate(glFace, glFunc, int32(face.Ref), uint32(face.TestMask))
		g.fns().StencilMaskSeparate(glFace, uint32(face.WriteMask))
		g.fns().StencilOpSeparate(glFace, glFailOp, KEEP, glPassOp)
	}
}

// flushStencil applies stencilSettings to GL state, skipping redundant calls when the settings (and, for two-sided
// stencil, the surface origin) already match what is bound.
func (g *Gpu) flushStencil(stencilSettings *StencilSettings, origin gpu.SurfaceOrigin) {
	if stencilSettings.IsDisabled() {
		g.disableStencil()
	} else if !g.hwStencilSettings.Equal(stencilSettings) ||
		(stencilSettings.IsTwoSided() && g.hwStencilOrigin != origin) {
		if g.hwStencilTestEnabled != triYes {
			g.fns().Enable(STENCIL_TEST)
			g.hwStencilTestEnabled = triYes
		}
		if !stencilSettings.IsTwoSided() {
			g.setGLStencil(stencilSettings.SingleSidedFace(), FRONT_AND_BACK)
		} else {
			g.setGLStencil(stencilSettings.PostOriginCWFace(origin), FRONT)
			g.setGLStencil(stencilSettings.PostOriginCCWFace(origin), BACK)
		}
		g.hwStencilSettings = *stencilSettings
		g.hwStencilOrigin = origin
	}
}

// flushConservativeRasterState enables or disables conservative rasterization (when supported), skipping the GL call if
// it is already in the requested state.
func (g *Gpu) flushConservativeRasterState(enabled bool) {
	if g.Caps().ConservativeRasterSupport {
		if enabled {
			if g.hwConservativeRasterEnabled != triYes {
				g.fns().Enable(CONSERVATIVE_RASTERIZATION)
				g.hwConservativeRasterEnabled = triYes
			}
		} else if g.hwConservativeRasterEnabled != triNo {
			g.fns().Disable(CONSERVATIVE_RASTERIZATION)
			g.hwConservativeRasterEnabled = triNo
		}
	}
}

// flushWireframeState toggles wireframe polygon mode (when supported), skipping the GL call if it is already in the
// requested state.
func (g *Gpu) flushWireframeState(enabled bool) {
	if g.Caps().WireframeSupport {
		if enabled {
			if g.hwWireframeEnabled != triYes {
				g.fns().PolygonMode(FRONT_AND_BACK, LINE)
				g.hwWireframeEnabled = triYes
			}
		} else if g.hwWireframeEnabled != triNo {
			g.fns().PolygonMode(FRONT_AND_BACK, FILL)
			g.hwWireframeEnabled = triNo
		}
	}
}

// gXfermodeCoeff2Blend, indexed by gpu.BlendCoeff.
var gXfermodeCoeff2Blend = [...]uint32{
	ZERO, ONE,
	SRC_COLOR, ONE_MINUS_SRC_COLOR,
	DST_COLOR, ONE_MINUS_DST_COLOR,
	SRC_ALPHA, ONE_MINUS_SRC_ALPHA,
	DST_ALPHA, ONE_MINUS_DST_ALPHA,
	CONSTANT_COLOR, ONE_MINUS_CONSTANT_COLOR,
	SRC1_COLOR, ONE_MINUS_SRC1_COLOR,
	SRC1_ALPHA, ONE_MINUS_SRC1_ALPHA,
	ZERO, // Illegal... needs to map to something.
}

// gXfermodeEquation2Blend, indexed by gpu.BlendEquation.
var gXfermodeEquation2Blend = [...]uint32{
	FUNC_ADD, FUNC_SUBTRACT, FUNC_REVERSE_SUBTRACT,
	SCREEN, OVERLAY, DARKEN, LIGHTEN,
	COLORDODGE, COLORBURN, HARDLIGHT, SOFTLIGHT,
	DIFFERENCE, EXCLUSION, MULTIPLY,
	HSL_HUE, HSL_SATURATION, HSL_COLOR, HSL_LUMINOSITY,
	FUNC_ADD, // Illegal.
}

// flushBlendAndColorWrite applies blendInfo's equation, coefficients, and constant color to GL blend state, skipping
// redundant calls and disabling blending outright when it reduces to a no-op. The write swizzle is applied to the blend
// constant when one is used (Swizzle.ApplyTo).
func (g *Gpu) flushBlendAndColorWrite(blendInfo gpu.BlendInfo, swizzle gpu.Swizzle) {
	// The neverDisableColorWrites workaround was mobile-only and is dropped with the trim.
	equation := blendInfo.Equation
	srcCoeff := blendInfo.SrcBlend
	dstCoeff := blendInfo.DstBlend

	// Any optimization to disable blending should have already been applied and tweaked the equation to "add" or
	// "subtract", and the coeffs to (1, 0).
	blendOff := gpu.BlendShouldDisable(equation, srcCoeff, dstCoeff) || !blendInfo.WritesColor

	if blendOff {
		if g.hwBlendState.enabled != triNo {
			g.fns().Disable(BLEND)
			g.hwBlendState.enabled = triNo
		}
	} else {
		if g.hwBlendState.enabled != triYes {
			g.fns().Enable(BLEND)
			g.hwBlendState.enabled = triYes
		}

		if g.hwBlendState.equation != equation {
			g.fns().BlendEquation(gXfermodeEquation2Blend[equation])
			g.hwBlendState.equation = equation
		}

		if equation.IsAdvanced() {
			if g.Caps().AdvancedBlendEquationSupport() {
				// Advanced equations have no other blend state.
				g.flushColorWrite(blendInfo.WritesColor)
				return
			}
			panic("advanced blend without support")
		}

		if g.hwBlendState.srcCoeff != srcCoeff || g.hwBlendState.dstCoeff != dstCoeff {
			g.fns().BlendFunc(gXfermodeCoeff2Blend[srcCoeff], gXfermodeCoeff2Blend[dstCoeff])
			g.hwBlendState.srcCoeff = srcCoeff
			g.hwBlendState.dstCoeff = dstCoeff
		}

		if srcCoeff.RefsConstant() || dstCoeff.RefsConstant() {
			blendConst := swizzle.ApplyTo(blendInfo.BlendConstant)
			if !g.hwBlendState.constColorValid || g.hwBlendState.constColor != blendConst {
				g.fns().BlendColor(blendConst[0], blendConst[1], blendConst[2], blendConst[3])
				g.hwBlendState.constColor = blendConst
				g.hwBlendState.constColorValid = true
			}
		}
	}

	g.flushColorWrite(blendInfo.WritesColor)
}

// flushRenderTarget binds the target's FBO (skipping redundant binds) and sets the viewport to its full extent.
func (g *Gpu) flushRenderTarget(target *RenderTarget, useMultisampleFBO bool) {
	rtID := target.Surface().UniqueID()
	if g.hwBoundRenderTargetUniqueID != rtID ||
		g.hwBoundFramebufferIsMSAA != useMultisampleFBO ||
		target.MustRebind(useMultisampleFBO) {
		target.Bind(useMultisampleFBO)
		g.hwBoundRenderTargetUniqueID = rtID
		g.hwBoundFramebufferIsMSAA = useMultisampleFBO
		dims := target.Surface().Dimensions()
		// The origin is irrelevant for a full-extent viewport.
		g.flushViewport(geom.IRectSize(dims), dims.Height, gpu.OriginTopLeft)
	}
	if g.glCaps().SRGBWriteControl() {
		g.flushFramebufferSRGB(g.glCaps().IsFormatSRGB(target.GLFormat()))
	}
	if g.glCaps().ShouldQueryImplementationReadSupport(target.GLFormat()) {
		var format, ttype int32
		g.fns().GetIntegerv(IMPLEMENTATION_COLOR_READ_FORMAT, &format)
		g.fns().GetIntegerv(IMPLEMENTATION_COLOR_READ_TYPE, &ttype)
		g.glCaps().DidQueryImplementationReadSupport(target.GLFormat(), uint32(format), uint32(ttype))
	}
}

// Clear clears the render target's color buffer within the scissor.
func (g *Gpu) Clear(scissor *gpu.ScissorState, color [4]float32, target *RenderTarget, useMultisampleFBO bool, origin gpu.SurfaceOrigin) {
	if target == nil {
		panic("nil render target")
	}
	if g.Caps().PerformColorClearsAsDraws ||
		(scissor.Enabled() && g.Caps().PerformPartialClearsAsDraws()) {
		panic("clear should have been converted to a draw")
	}
	g.handleDirtyContext()

	g.flushRenderTarget(target, useMultisampleFBO)
	g.flushScissor(scissor, target.Surface().Height(), origin)
	g.disableWindowRectangles()
	g.flushColorWrite(true)
	g.flushClearColor(color)
	g.fns().Clear(COLOR_BUFFER_BIT)
	g.didWriteToSurface(target.Surface())
}

// didWriteToSurface records that the surface was written to: writing to a texture dirties its mipmaps.
func (g *Gpu) didWriteToSurface(surface *Surface) {
	if surface.ReadOnly() {
		panic("wrote to read-only surface")
	}
	if tex := surface.AsTexture(); tex != nil {
		tex.MarkMipmapsDirty()
	}
}

// FlushProgram binds the given raw program ID if it is not already current, clearing the linked-program shadow so it
// stays consistent with the bound ID (the internal flushProgram, gpudraw.go, is the linked-*Program form).
func (g *Gpu) FlushProgram(id uint32) {
	if id == 0 {
		panic("flushing program 0")
	}
	g.flushProgramID(id)
}

// renderbufferStorageMSAA allocates multisampled renderbuffer storage (desktop: only the MSFBOStandard lane is
// reachable).
func (g *Gpu) renderbufferStorageMSAA(sampleCount int, format uint32, width, height int32) bool {
	if g.glCaps().MSFBOType() == MSFBONone {
		panic("MSAA renderbuffer without support")
	}
	return g.allocCall(func(f *Functions) {
		f.RenderbufferStorageMultisample(RENDERBUFFER, int32(sampleCount), format, width, height)
	}) == NO_ERROR
}

// ResolveRenderFBOs resolves a multisampled render target's samples into its resolve buffer (desktop: the
// BlitFramebuffer lane; a mobile-only ES resolve extension is unreachable).
func (g *Gpu) ResolveRenderFBOs(rt *RenderTarget, resolveRect geom.IRect, dir ResolveDirection, invalidateReadBufferAfterBlit bool) {
	g.handleDirtyContext()
	rt.BindForResolve(dir)

	caps := g.glCaps()

	// Make sure we go through flushRenderTarget() since we've modified the bound DRAW FBO ID.
	g.hwBoundRenderTargetUniqueID = 0

	if caps.FramebufferResolvesMustBeFullSize() &&
		resolveRect != geom.IRectSize(rt.Surface().Dimensions()) {
		panic("resolve must be full size")
	}
	l := resolveRect.Left
	b := resolveRect.Top
	r := resolveRect.Right
	t := resolveRect.Bottom

	// BlitFramebuffer respects the scissor, so disable it.
	g.flushScissorTest(false)
	g.disableWindowRectangles()
	g.fns().BlitFramebuffer(l, b, r, t, l, b, r, t, COLOR_BUFFER_BIT, NEAREST)

	if caps.InvalidateFBType() != InvalidateFBNone && invalidateReadBufferAfterBlit {
		// Invalidate the read FBO attachment after the blit in hopes that this allows the driver to perform tiling
		// optimizations.
		readBufferIsMSAA := dir == ResolveMSAAToSingle
		colorDiscardAttachment := uint32(COLOR_ATTACHMENT0)
		if rt.IsFBO0(readBufferIsMSAA) {
			colorDiscardAttachment = COLOR
		}
		if caps.InvalidateFBType() == InvalidateFBInvalidate {
			g.fns().InvalidateFramebuffer(READ_FRAMEBUFFER, 1, &colorDiscardAttachment)
		} else {
			// glDiscardFramebuffer only accepts GL_FRAMEBUFFER.
			rt.Bind(readBufferIsMSAA)
			g.fns().DiscardFramebuffer(FRAMEBUFFER, 1, &colorDiscardAttachment)
		}
	}
}
