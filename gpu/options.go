// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The subset of GPU context options and driver-bug workarounds the caps code reads. The public surface never exposes
// context options, so these exist for internal defaults and tests; fields with no surviving consumer in this codebase
// are omitted.

package gpu

import (
	"math"
	"runtime"
)

// Enable is a tri-state on/off/default switch. The Go zero value is EnableDefault.
type Enable int8

// Enable values.
const (
	EnableDefault Enable = iota
	EnableNo
	EnableYes
)

// ShaderCacheStrategy selects what form of compiled-shader caching to use.
type ShaderCacheStrategy int8

// ShaderCacheStrategy values. This codebase has no shader source caching layer, so only the backend-binary distinction
// matters: anything below BackendBinary disables program-binary caching.
const (
	ShaderCacheStrategyBackendSource ShaderCacheStrategy = iota
	ShaderCacheStrategyBackendBinary
)

// DriverBugWorkarounds holds the driver-bug-workaround bools consulted by surviving code paths. All default to false;
// kept to preserve the shape of the checks even though nothing currently sets them.
type DriverBugWorkarounds struct {
	AddAndTrueToLoopCondition              bool
	DisableBlendEquationAdvanced           bool
	DisableDiscardFramebuffer              bool
	DisableDualSourceBlendingSupport       bool
	DisableTextureStorage                  bool
	EmulateAbsIntFunction                  bool
	GLClearBroken                          bool
	MaxFragmentUniformVectors32            bool
	MaxMSAASampleCount4                    bool
	PackParametersWorkaroundWithPackBuffer bool
	RemovePowWithConstantExponent          bool
	RewriteDoWhileLoops                    bool
	UnfoldShortCircuitAsTernaryOperation   bool
}

// ApplyOverrides merges workarounds into d: each true in workarounds turns the corresponding field on (never off).
func (d *DriverBugWorkarounds) ApplyOverrides(workarounds *DriverBugWorkarounds) {
	d.AddAndTrueToLoopCondition = d.AddAndTrueToLoopCondition || workarounds.AddAndTrueToLoopCondition
	d.DisableBlendEquationAdvanced = d.DisableBlendEquationAdvanced || workarounds.DisableBlendEquationAdvanced
	d.DisableDiscardFramebuffer = d.DisableDiscardFramebuffer || workarounds.DisableDiscardFramebuffer
	d.DisableDualSourceBlendingSupport = d.DisableDualSourceBlendingSupport ||
		workarounds.DisableDualSourceBlendingSupport
	d.DisableTextureStorage = d.DisableTextureStorage || workarounds.DisableTextureStorage
	d.EmulateAbsIntFunction = d.EmulateAbsIntFunction || workarounds.EmulateAbsIntFunction
	d.GLClearBroken = d.GLClearBroken || workarounds.GLClearBroken
	d.MaxFragmentUniformVectors32 = d.MaxFragmentUniformVectors32 || workarounds.MaxFragmentUniformVectors32
	d.MaxMSAASampleCount4 = d.MaxMSAASampleCount4 || workarounds.MaxMSAASampleCount4
	d.PackParametersWorkaroundWithPackBuffer = d.PackParametersWorkaroundWithPackBuffer ||
		workarounds.PackParametersWorkaroundWithPackBuffer
	d.RemovePowWithConstantExponent = d.RemovePowWithConstantExponent ||
		workarounds.RemovePowWithConstantExponent
	d.RewriteDoWhileLoops = d.RewriteDoWhileLoops || workarounds.RewriteDoWhileLoops
	d.UnfoldShortCircuitAsTernaryOperation = d.UnfoldShortCircuitAsTernaryOperation ||
		workarounds.UnfoldShortCircuitAsTernaryOperation
}

// ContextOptions holds the read subset of GPU context configuration. Construct with DefaultContextOptions; the Go zero
// value does not carry the intended defaults.
type ContextOptions struct {
	// BufferMapThreshold: buffers smaller than this are updated with updateData rather than mapped. Negative means the
	// implementation default.
	BufferMapThreshold int
	// MaxTextureSizeOverride caps the maximum texture size below the driver's limit.
	MaxTextureSizeOverride int
	// InternalMultisampleCount is the sample count the GPU backend uses for internal draws needing MSAA (also the DMSAA
	// sample count).
	InternalMultisampleCount int
	// GlyphCacheTextureMaximumBytes is the maximum size of one atlas texture for the glyph cache (multitexturing may
	// multiply this).
	GlyphCacheTextureMaximumBytes uint64
	// MinDistanceFieldFontSize: below this threshold size in device space distance-field fonts won't be used — they
	// don't support hinting, which matters more at smaller sizes.
	MinDistanceFieldFontSize float32
	// GlyphsAsPathsFontSize: above this threshold size in device space glyphs are drawn as individual paths. The
	// default is per-OS.
	GlyphsAsPathsFontSize float32
	// PathRenderers is the set of GPU path renderers the path-renderer chain may construct. Defaults to
	// PathRenderersDefault.
	PathRenderers        PathRenderers
	DriverBugWorkarounds DriverBugWorkarounds
	// AllowMultipleGlyphCacheTextures: whether the glyph atlases may multitexture (EnableDefault means yes when the
	// shader caps can represent the page index in the texture coordinates).
	AllowMultipleGlyphCacheTextures     Enable
	SkipGLErrorChecks                   Enable
	UseDrawInsteadOfClear               Enable
	AvoidStencilBuffers                 bool
	DisableDriverCorrectnessWorkarounds bool
	DoManualMipmapping                  bool
	SuppressMipmapSupport               bool
	ReducedShaderVariations             bool
	AllowMSAAOnNewIntel                 bool
	DisableTessellationPathRenderer     bool
	// SupportBilerpFromGlyphAtlas pads glyph atlas entries so they can be sampled with bilerp.
	SupportBilerpFromGlyphAtlas bool
	ShaderCacheStrategy         ShaderCacheStrategy
	// AllowPathMaskCaching: whether distance-field and software path masks may be cached and reused across draws.
	// Defaults to true.
	AllowPathMaskCaching bool
}

// PathRenderers is a bitfield of options for GPU path rendering, used for testing and faster rendering in the
// presence of driver bugs.
type PathRenderers uint32

// PathRenderers values.
const (
	// PathRenderersNone always uses software masks and/or DefaultPathRenderer.
	PathRenderersNone          PathRenderers = 0
	PathRenderersDashLine      PathRenderers = 1 << 0
	PathRenderersAtlas         PathRenderers = 1 << 1
	PathRenderersTessellation  PathRenderers = 1 << 2
	PathRenderersAAHairline    PathRenderers = 1 << 4
	PathRenderersAAConvex      PathRenderers = 1 << 5
	PathRenderersAALinearizing PathRenderers = 1 << 6
	PathRenderersSmall         PathRenderers = 1 << 7
	PathRenderersTriangulating PathRenderers = 1 << 8

	// PathRenderersDefault enables all path renderers (bit 1<<3 is reserved and unused, kept for key/value parity
	// with the bit layout above).
	PathRenderersDefault PathRenderers = 1<<9 - 1
)

// defaultGlyphsAsPathsFontSize picks the per-platform GlyphsAsPathsFontSize default.
func defaultGlyphsAsPathsFontSize() float32 {
	switch runtime.GOOS {
	case "android":
		return 384
	case "darwin": // macOS only; iOS takes the default.
		return 256
	default:
		return 324
	}
}

// DefaultContextOptions returns a ContextOptions with the standard field defaults.
func DefaultContextOptions() *ContextOptions {
	return &ContextOptions{
		BufferMapThreshold:            -1,
		MaxTextureSizeOverride:        math.MaxInt32,
		InternalMultisampleCount:      4,
		GlyphCacheTextureMaximumBytes: 2048 * 1024 * 4,
		MinDistanceFieldFontSize:      18,
		GlyphsAsPathsFontSize:         defaultGlyphsAsPathsFontSize(),
		ShaderCacheStrategy:           ShaderCacheStrategyBackendBinary,
		PathRenderers:                 PathRenderersDefault,
		AllowPathMaskCaching:          true,
	}
}
