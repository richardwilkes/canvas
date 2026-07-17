// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU capability base state: since this codebase targets exactly one backend, this package holds the field storage
// plus the helpers that need no backend dispatch, and gpu/gl's Caps embeds it and supplies everything backend-specific
// (including the option-override sequence applied during initialization).
//
// Trimmed to desktop needs: protected content, AHardwareBuffer images, push constants/DMSAA-resolve hooks for backends
// not present here, the test-utils device name, and debug JSON dumps are dropped; fields settable only from
// ES/ANGLE/WebGL/mobile driver paths are omitted for the same reason.

package gpu

// BlendEquationSupport describes the capabilities of the fixed-function blend unit.
type BlendEquationSupport int32

// BlendEquationSupport values.
const (
	// BasicBlendEquationSupport: can select the operator that combines src and dst terms.
	BasicBlendEquationSupport BlendEquationSupport = iota
	// AdvancedBlendEquationSupport: additional fixed-function support for specific SVG/PDF blend modes. Requires blend
	// barriers.
	AdvancedBlendEquationSupport
	// AdvancedCoherentBlendEquationSupport: advanced blend equation support that does not require blend barriers and
	// permits overlap.
	AdvancedCoherentBlendEquationSupport
)

// MapFlags describe whether GPU->CPU memory mapping for GPU resources allows partial or full mappings.
const (
	// NoMapFlags: cannot map the resource.
	NoMapFlags uint32 = 0
	// CanMapFlag: the resource can be mapped. Must be set for any of the other flags to have meaning.
	CanMapFlag uint32 = 1 << 0
	// SubsetMapFlag: the resource can be partially mapped.
	SubsetMapFlag uint32 = 1 << 1
	// AsyncReadMapFlag: maps for reading are asynchronous w.r.t. the render pass.
	AsyncReadMapFlag uint32 = 1 << 2
)

// MaxWindowRectangles is the most window rectangles the GPU backend will use.
const MaxWindowRectangles = 8

// SurfaceReadPixelsSupport describes how (or whether) a surface supports reading back pixels.
type SurfaceReadPixelsSupport int32

// SurfaceReadPixelsSupport values.
const (
	// SurfaceReadPixelsSupported: readPixels is supported by the surface.
	SurfaceReadPixelsSupported SurfaceReadPixelsSupport = iota
	// SurfaceReadPixelsCopyToTexture2D: not directly readable, but the surface can be drawn or copied to a 2D texture
	// and then that texture will be readable.
	SurfaceReadPixelsCopyToTexture2D
	// SurfaceReadPixelsUnsupported: not supported.
	SurfaceReadPixelsUnsupported
)

// SupportedWrite describes the color type the caller must coax data into in order to use GPU writePixels, and the
// transfer-buffer offset alignment when writing via transfer.
type SupportedWrite struct {
	ColorType                        ColorType
	OffsetAlignmentForTransferBuffer int
}

// SupportedRead describes a legal color type for GPU readPixels given a request, and the transfer-buffer offset
// alignment when reading via transfer.
type SupportedRead struct {
	ColorType                        ColorType
	OffsetAlignmentForTransferBuffer int
}

// TextureType identifies the shape of a GPU texture. External (OES) textures are excluded since this backend has no
// such concept, so only none/2D/rectangle remain.
type TextureType int32

// TextureType values.
const (
	TextureTypeNone TextureType = iota
	TextureType2D
	TextureTypeRectangle
)

// Caps holds the backend-neutral GPU capability field storage. Field defaults come from MakeCaps; the Go zero value is
// not meaningful. Backend-dependent queries live on gpu/gl.Caps, which embeds this.
type Caps struct {
	ShaderCaps                          *ShaderCaps
	BufferMapThreshold                  int
	MaxRenderTargetSize                 int
	MaxPreferredRenderTargetSize        int
	MaxVertexAttributes                 int
	MaxTextureSize                      int
	MaxWindowRectanglesCap              int
	InternalMultisampleCountCap         int
	TransferBufferRowBytesAlignment     int
	TransferFromBufferToBufferAlignment int
	BufferUpdateDataPreserveAlignment   int
	BlendEquationSupport                BlendEquationSupport
	AdvBlendEqDisableFlags              uint32
	MapBufferFlags                      uint32
	DriverBugWorkarounds                DriverBugWorkarounds
	NPOTTextureTileSupport              bool
	MipmapSupport                       bool
	AnisoSupport                        bool
	ReuseScratchTextures                bool
	ReuseScratchBuffers                 bool
	GpuTracingSupport                   bool
	OversizedStencilSupport             bool
	TextureBarrierSupport               bool
	SampleLocationsSupport              bool
	DrawInstancedSupport                bool
	// NativeDrawIndirectSupport: is there hardware support for indirect draws? Indirect draws are always supported at a
	// higher level as long as they can be polyfilled with instanced calls.
	NativeDrawIndirectSupport    bool
	UseClientSideIndirectBuffers bool
	ConservativeRasterSupport    bool
	WireframeSupport             bool
	// MSAAResolvesAutomatically means we never have to resolve MSAA (render-to-texture extensions); never true on
	// desktop GL but kept because shared caps-consuming logic checks it.
	MSAAResolvesAutomatically       bool
	PreferDiscardableMSAAAttachment bool
	UsePrimitiveRestart             bool
	PreferClientSideDynamicBuffers  bool
	// PreferFullscreenClears: on tilers, an initial fullscreen clear lets the hardware initialize each tile with a
	// constant value rather than loading each pixel from memory.
	PreferFullscreenClears               bool
	TwoSidedStencilRefsAndMasksMustMatch bool
	MustClearUploadedBufferData          bool
	BuffersAreInitiallyZero              bool
	ShouldInitializeTextures             bool
	HalfFloatVertexAttributeSupport      bool
	ClampToBorderSupport                 bool
	PerformPartialClearsAsDrawsCap       bool
	PerformColorClearsAsDraws            bool
	AvoidLargeIndexBufferDraws           bool
	PerformStencilClearsAsDraws          bool
	TransferFromBufferToTextureSupport   bool
	TransferFromSurfaceToBufferSupport   bool
	TransferFromBufferToBufferSupport    bool
	WritePixelsRowBytesSupport           bool
	TransferPixelsToRowBytesSupport      bool
	ReadPixelsRowBytesSupport            bool
	// ShouldCollapseSrcOverToSrcWhenAble: on some GPUs it is a performance win to disable blending instead of doing
	// src-over with a src alpha equal to 1.
	ShouldCollapseSrcOverToSrcWhenAble bool
	MustSyncGpuDuringAbandon           bool
	// Driver workarounds.
	DisableTessellationPathRendererCap bool
	AvoidStencilBuffers                bool
	AvoidWritePixelsFastPath           bool
	NativeDrawIndexedIndirectIsBroken  bool
	AvoidReorderingRenderTasks         bool
	AvoidDithering                     bool
	AvoidLineDraws                     bool
	// DisablePerspectiveSDFText disables signed-distance-field text under perspective transforms; a workaround for
	// specific mobile GPU drivers, which the desktop driver detection never reports, so this stays false. SubRunControl
	// construction still consults it faithfully.
	DisablePerspectiveSDFText        bool
	PreferVRAMUseOverFlushes         bool
	SemaphoreSupport                 bool
	BackendSemaphoreSupport          bool
	FinishedProcAsyncCallbackSupport bool
	// CrossContextTextureSupport requires fence sync support in GL.
	CrossContextTextureSupport                       bool
	DynamicStateArrayGeometryProcessorTextureSupport bool
}

// MakeCaps returns a Caps with the standard field defaults.
func MakeCaps(options *ContextOptions) Caps {
	return Caps{
		ShaderCaps:                          NewShaderCaps(),
		DriverBugWorkarounds:                options.DriverBugWorkarounds,
		ReuseScratchTextures:                true,
		ReuseScratchBuffers:                 true,
		MustSyncGpuDuringAbandon:            true,
		PreferVRAMUseOverFlushes:            true,
		ClampToBorderSupport:                true,
		MapBufferFlags:                      NoMapFlags,
		BufferMapThreshold:                  options.BufferMapThreshold,
		MaxRenderTargetSize:                 1,
		MaxPreferredRenderTargetSize:        1,
		MaxTextureSize:                      1,
		TransferBufferRowBytesAlignment:     1,
		TransferFromBufferToBufferAlignment: 1,
		BufferUpdateDataPreserveAlignment:   1,
	}
}

// AdvancedBlendEquationSupport reports whether the hardware supports any advanced blend equation.
func (c *Caps) AdvancedBlendEquationSupport() bool {
	return c.BlendEquationSupport >= AdvancedBlendEquationSupport
}

// AdvancedCoherentBlendEquationSupport reports whether advanced blend equations are supported without requiring blend
// barriers.
func (c *Caps) AdvancedCoherentBlendEquationSupport() bool {
	return c.BlendEquationSupport == AdvancedCoherentBlendEquationSupport
}

// IsAdvancedBlendEquationDisabled reports whether the given advanced blend equation has been disabled for this device.
func (c *Caps) IsAdvancedBlendEquationDisabled(equation BlendEquation) bool {
	return c.AdvBlendEqDisableFlags&(1<<uint32(equation)) != 0
}

// PerformPartialClearsAsDraws reports whether partial clears must be performed as draws: always true when full color
// clears are performed as draws.
func (c *Caps) PerformPartialClearsAsDraws() bool {
	return c.PerformColorClearsAsDraws || c.PerformPartialClearsAsDrawsCap
}

// DiscardStencilValuesAfterRenderPass reports whether stencil contents may be discarded after a render pass. Always
// false: discarding stencil values after a render pass has proven unreliable on enough drivers that it is never
// attempted.
func (c *Caps) DiscardStencilValuesAfterRenderPass() bool { return false }

// ReducedShaderMode is a shortcut for ShaderCaps.ReducedShaderMode.
func (c *Caps) ReducedShaderMode() bool { return c.ShaderCaps.ReducedShaderMode }
