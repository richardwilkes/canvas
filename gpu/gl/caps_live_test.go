// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Live-context caps tests: initialize the desktop-trimmed Caps against a real GL context (gltest, the way unison will
// provide one) and sanity-check the derived capabilities. TestCapsLiveDump prints the caps block that gets appended to
// the gldumps/ file for the running platform, for regression comparison against future driver updates.

package gl_test

import (
	"testing"

	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/gl"
)

func newLiveContextInfo(tb testing.TB) *gl.ContextInfo {
	tb.Helper()
	env := newGLEnv(tb)
	ctxInfo := gl.NewContextInfo(env.intf, nil)
	if ctxInfo == nil {
		tb.Fatal("NewContextInfo returned nil on a live context")
	}
	return ctxInfo
}

func TestCapsLiveContext(t *testing.T) {
	ctxInfo := newLiveContextInfo(t)
	caps := ctxInfo.Caps

	// gltest creates a core-profile context (4.1 preferred, 3.2 fallback).
	if !caps.IsCoreProfile() {
		t.Error("expected a core profile context")
	}
	if ctxInfo.GLSLGeneration < gpu.GLSL150 {
		t.Errorf("GLSL generation = %d, want >= GLSL150 for a core context", ctxInfo.GLSLGeneration)
	}

	// The supported color-type matrix essentials must hold on any desktop core context.
	if !caps.IsFormatTexturable(gl.FormatRGBA8, gpu.TextureType2D) {
		t.Error("RGBA8 must be texturable")
	}
	if !caps.IsFormatAsColorTypeRenderable(gpu.ColorTypeRGBA8888, gl.FormatRGBA8, 1) {
		t.Error("RGBA8/RGBA_8888 must be renderable")
	}
	if got := caps.GetFormatFromColorType(gpu.ColorTypeRGBA8888); got != gl.FormatRGBA8 {
		t.Errorf("RGBA_8888 format = %v", got)
	}
	if got := caps.GetFormatFromColorType(gpu.ColorTypeBGRA8888); got != gl.FormatRGBA8 {
		t.Errorf("BGRA_8888 format = %v (desktop BGRA rides RGBA8)", got)
	}
	if got := caps.GetFormatFromColorType(gpu.ColorTypeAlpha8); got != gl.FormatR8 {
		t.Errorf("Alpha_8 format = %v, want R8 on a core profile", got)
	}
	// Core profiles (>= 3.2) always support MSAA FBOs and GLsync unless a workaround disabled them; stencil must offer
	// at least S8 and D24S8.
	stencils := caps.StencilFormats()
	hasS8, hasD24S8 := false, false
	for _, f := range stencils {
		switch f {
		case gl.FormatSTENCILINDEX8:
			hasS8 = true
		case gl.FormatDEPTH24STENCIL8:
			hasD24S8 = true
		}
	}
	if !hasS8 || !hasD24S8 {
		t.Errorf("stencil formats = %v, want S8 and D24S8", stencils)
	}
	if !caps.FenceSyncSupport() || caps.FenceType() != gl.FenceSyncObject {
		t.Error("core contexts (>= 3.2) must have GLsync support")
	}
	if !caps.VertexArrayObjectSupport() {
		t.Error("core contexts must have VAO support")
	}
	// The version decl the caps derived must match what the spike derives from the raw version, so generated programs
	// will compile on this context.
	if got := caps.ShaderCaps.VersionDeclString; got == "<no version>" || got == "" {
		t.Errorf("version decl = %q", got)
	}
	// The wrapped-FBO stencil-bits contract (gr_backendrendertarget_new_gl) needs sample counts: requesting 1 sample on
	// RGBA8 must succeed.
	if got := caps.GetRenderTargetSampleCount(1, gl.FormatRGBA8); got != 1 {
		t.Errorf("RGBA8 single-sample render = %d, want 1", got)
	}
}

func TestCapsLiveDump(t *testing.T) {
	ctxInfo := newLiveContextInfo(t)
	caps := ctxInfo.Caps

	t.Logf("CAPS core_profile: %v", caps.IsCoreProfile())
	t.Logf("CAPS glsl_generation: %d", ctxInfo.GLSLGeneration)
	t.Logf("CAPS version_decl: %q", caps.ShaderCaps.VersionDeclString)
	t.Logf("CAPS vendor/renderer/driver: %d/%d/%d", ctxInfo.Vendor(), ctxInfo.Renderer(),
		ctxInfo.Driver())
	t.Logf("CAPS driver_version: 0x%012x", uint64(ctxInfo.DriverVersion()))
	t.Logf("CAPS msfbo_type: %d", caps.MSFBOType())
	t.Logf("CAPS stencil_formats: %v", caps.StencilFormats())
	t.Logf("CAPS fence_type: %d timer_query: %d multi_draw: %d", caps.FenceType(),
		caps.TimerQueryType(), caps.MultiDrawType())
	t.Logf("CAPS map_buffer: %d transfer_buffer: %d invalidate_fb: %d invalidate_buffer: %d",
		caps.MapBufferType(), caps.TransferBufferType(), caps.InvalidateFBType(),
		caps.InvalidateBufferType())
	t.Logf("CAPS max_frag_uniform_vectors: %d max_vertex_attribs: %d max_frag_samplers: %d",
		caps.MaxFragmentUniformVectors(), caps.MaxVertexAttributes, caps.ShaderCaps.MaxFragmentSamplers)
	t.Logf("CAPS dual_source_blending: %v texture_barrier: %v texture_swizzle: %v sampler_objects: %v",
		caps.ShaderCaps.DualSourceBlendingSupport, caps.TextureBarrierSupport,
		caps.TextureSwizzleSupport(), caps.SamplerObjectSupport())
	t.Logf("CAPS program_binary: %v blend_equation_support: %d aniso: %v/%g",
		caps.ProgramBinarySupport(), caps.BlendEquationSupport, caps.AnisoSupport,
		caps.MaxTextureMaxAnisotropy())
	for _, entry := range []struct {
		name   string
		format gl.Format
	}{
		{name: "RGBA8", format: gl.FormatRGBA8},
		{name: "R8", format: gl.FormatR8},
		{name: "ALPHA8", format: gl.FormatALPHA8},
		{name: "LUMINANCE8", format: gl.FormatLUMINANCE8},
		{name: "BGRA8", format: gl.FormatBGRA8},
		{name: "RGB565", format: gl.FormatRGB565},
		{name: "RGBA16F", format: gl.FormatRGBA16F},
		{name: "R16F", format: gl.FormatR16F},
		{name: "RGB8", format: gl.FormatRGB8},
	} {
		t.Logf("CAPS format %s: texturable=%v tex_storage=%v max_samples=%d", entry.name,
			caps.IsFormatTexturableAny(entry.format), caps.FormatSupportsTexStorage(entry.format),
			caps.MaxRenderTargetSampleCount(entry.format))
	}
}
