// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Differential-style tests for the desktop-trimmed GL Caps: full caps initialization is driven through
// purego.NewCallback fake drivers (the same FFI boundary a real driver uses) for two realistic desktop profiles — the
// Apple M4 Max GL 4.1 core context recorded in gldumps and a Mesa llvmpipe GL 4.5 core context (the planned Linux CI
// environment) — plus an Intel-on-Mac profile that exercises the desktop driver workarounds.

package gl

import (
	"strings"
	"sync"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/richardwilkes/canvas/gpu"
)

// capsFakeDriver answers the queries GL Caps init makes. Pointers arrive as uintptr and are written through
// ptrFromUintptr, exercising the same conversion the production wrappers use.
type capsFakeDriver struct {
	integers             map[uint32]int32
	version              string
	vendor               string
	renderer             string
	glslVersion          string
	extensions           []string
	programBinaryFormats []int32
	sampleCounts         []int32 // what GetInternalformativ reports for SAMPLES, GL (descending) order
	maxAniso             float32
	precisionBits        int32
}

// The shared caps callbacks answer for whichever driver getProc installed here last. purego callbacks are never freed
// and the process has a hard cap on them, so minting fresh ones per getProc call exhausted the cap under go test
// -count>1; instead the callbacks are created once and dispatch to the current driver. Tests are not parallel and only
// one fake driver is ever live at a time (the same discipline recCounts et al. rely on).
var (
	capsFakeCurrent   *capsFakeDriver
	capsFakeProcsOnce sync.Once
	capsFakeProcs     map[string]uintptr
	capsFakeDummy     uintptr
)

func (d *capsFakeDriver) getProc() GetProc {
	capsFakeProcsOnce.Do(func() {
		getString := purego.NewCallback(func(name uintptr) uintptr {
			drv := capsFakeCurrent
			switch uint32(name) {
			case VERSION:
				return cstr(drv.version)
			case RENDERER:
				return cstr(drv.renderer)
			case VENDOR:
				return cstr(drv.vendor)
			case SHADING_LANGUAGE_VERSION:
				return cstr(drv.glslVersion)
			}
			return 0
		})
		getStringi := purego.NewCallback(func(name, index uintptr) uintptr {
			drv := capsFakeCurrent
			if uint32(name) == EXTENSIONS && int(index) < len(drv.extensions) {
				return cstr(drv.extensions[int(index)])
			}
			return 0
		})
		getIntegerv := purego.NewCallback(func(pname, params uintptr) uintptr {
			drv := capsFakeCurrent
			switch uint32(pname) {
			case NUM_EXTENSIONS:
				*(*int32)(ptrFromUintptr(params)) = int32(len(drv.extensions))
			case NUM_PROGRAM_BINARY_FORMATS:
				*(*int32)(ptrFromUintptr(params)) = int32(len(drv.programBinaryFormats))
			case PROGRAM_BINARY_FORMATS:
				for i, v := range drv.programBinaryFormats {
					*(*int32)(ptrFromUintptr(params + uintptr(i)*4)) = v
				}
			default:
				if v, ok := drv.integers[uint32(pname)]; ok {
					*(*int32)(ptrFromUintptr(params)) = v
				}
			}
			return 0
		})
		getFloatv := purego.NewCallback(func(pname, params uintptr) uintptr {
			if uint32(pname) == MAX_TEXTURE_MAX_ANISOTROPY {
				*(*float32)(ptrFromUintptr(params)) = capsFakeCurrent.maxAniso
			}
			return 0
		})
		getInternalformativ := purego.NewCallback(func(_, _, pname, bufSize, params uintptr) uintptr {
			drv := capsFakeCurrent
			switch uint32(pname) {
			case NUM_SAMPLE_COUNTS:
				*(*int32)(ptrFromUintptr(params)) = int32(len(drv.sampleCounts))
			case SAMPLES:
				n := int(bufSize)
				if n > len(drv.sampleCounts) {
					n = len(drv.sampleCounts)
				}
				for i := 0; i < n; i++ {
					*(*int32)(ptrFromUintptr(params + uintptr(i)*4)) = drv.sampleCounts[i]
				}
			}
			return 0
		})
		getShaderPrecisionFormat := purego.NewCallback(func(_, _, rangePtr, precisionPtr uintptr) uintptr {
			*(*int32)(ptrFromUintptr(rangePtr)) = 127
			*(*int32)(ptrFromUintptr(rangePtr + 4)) = 127
			*(*int32)(ptrFromUintptr(precisionPtr)) = capsFakeCurrent.precisionBits
			return 0
		})
		capsFakeProcs = map[string]uintptr{
			"glGetString":                getString,
			"glGetStringi":               getStringi,
			"glGetIntegerv":              getIntegerv,
			"glGetFloatv":                getFloatv,
			"glGetInternalformativ":      getInternalformativ,
			"glGetShaderPrecisionFormat": getShaderPrecisionFormat,
		}
		// Procs the caps code never calls just need to be resolvable; a no-op callback tolerates being invoked with any
		// argument registers if something unexpected does call it.
		capsFakeDummy = purego.NewCallback(func() uintptr { return 0 })
	})
	capsFakeCurrent = d
	return func(name string) uintptr {
		if p, ok := capsFakeProcs[name]; ok {
			return p
		}
		if strings.HasPrefix(name, "egl") {
			return 0
		}
		return capsFakeDummy
	}
}

func (d *capsFakeDriver) newContextInfo(t *testing.T, options *gpu.ContextOptions) *ContextInfo {
	t.Helper()
	intf := MakeAssembledInterface(d.getProc())
	if intf == nil {
		t.Fatal("MakeAssembledInterface returned nil for the fake driver")
	}
	ctxInfo := NewContextInfo(intf, options)
	if ctxInfo == nil {
		t.Fatal("NewContextInfo returned nil for the fake driver")
	}
	return ctxInfo
}

// appleM4MaxDriver models the context recorded in gldumps/darwin-arm64-apple-m4-max.txt.
func appleM4MaxDriver() *capsFakeDriver {
	return &capsFakeDriver{
		version:     "4.1 Metal - 90.5",
		vendor:      "Apple",
		renderer:    "Apple M4 Max",
		glslVersion: "4.10",
		extensions: []string{
			"GL_APPLE_client_storage", "GL_APPLE_container_object_shareable", "GL_APPLE_flush_render",
			"GL_APPLE_rgb_422", "GL_APPLE_row_bytes", "GL_APPLE_texture_range",
			"GL_ARB_ES2_compatibility", "GL_ARB_blend_func_extended", "GL_ARB_draw_buffers_blend",
			"GL_ARB_draw_indirect", "GL_ARB_explicit_attrib_location", "GL_ARB_gpu_shader5",
			"GL_ARB_gpu_shader_fp64", "GL_ARB_instanced_arrays", "GL_ARB_internalformat_query",
			"GL_ARB_occlusion_query2", "GL_ARB_sample_shading", "GL_ARB_sampler_objects",
			"GL_ARB_separate_shader_objects", "GL_ARB_shader_bit_encoding", "GL_ARB_shader_subroutine",
			"GL_ARB_shading_language_include", "GL_ARB_tessellation_shader",
			"GL_ARB_texture_buffer_object_rgb32", "GL_ARB_texture_cube_map_array",
			"GL_ARB_texture_gather", "GL_ARB_texture_query_lod", "GL_ARB_texture_rgb10_a2ui",
			"GL_ARB_texture_storage", "GL_ARB_texture_swizzle", "GL_ARB_timer_query",
			"GL_ARB_transform_feedback2", "GL_ARB_transform_feedback3", "GL_ARB_vertex_attrib_64bit",
			"GL_ARB_vertex_type_2_10_10_10_rev", "GL_ARB_viewport_array", "GL_EXT_debug_label",
			"GL_EXT_debug_marker", "GL_EXT_framebuffer_multisample_blit_scaled",
			"GL_EXT_texture_compression_s3tc", "GL_EXT_texture_filter_anisotropic",
			"GL_EXT_texture_sRGB_decode", "GL_NV_texture_barrier",
		},
		integers: map[uint32]int32{
			MAX_FRAGMENT_UNIFORM_COMPONENTS: 4096,
			CONTEXT_PROFILE_MASK:            CONTEXT_CORE_PROFILE_BIT,
			MAX_VERTEX_ATTRIBS:              16,
			MAX_TEXTURE_IMAGE_UNITS:         16,
			MAX_TEXTURE_SIZE:                16384,
			MAX_RENDERBUFFER_SIZE:           16384,
			MAX_SAMPLES:                     4,
		},
		sampleCounts:  []int32{4, 2}, // GL returns descending order
		maxAniso:      16,
		precisionBits: 23,
	}
}

func TestCapsAppleM4MaxProfile(t *testing.T) {
	ctxInfo := appleM4MaxDriver().newContextInfo(t, nil)
	caps := ctxInfo.Caps

	// Driver identity.
	if got := ctxInfo.Vendor(); got != VendorApple {
		t.Errorf("vendor = %d, want Apple", got)
	}
	if got := ctxInfo.Renderer(); got != RendererApple {
		t.Errorf("renderer = %d, want Apple", got)
	}
	if got := ctxInfo.Driver(); got != DriverApple {
		t.Errorf("driver = %d, want Apple", got)
	}
	if got, want := ctxInfo.DriverVersion(), DriverVer(90, 0, 0); got != want {
		t.Errorf("driver version = %x, want %x", got, want)
	}
	if got := ctxInfo.GLSLGeneration; got != gpu.GLSL400 {
		t.Errorf("GLSL generation = %d, want GLSL400", got)
	}

	// Context basics.
	if !caps.IsCoreProfile() {
		t.Error("expected core profile")
	}
	if got, want := caps.ShaderCaps.VersionDeclString, "#version 400\n"; got != want {
		t.Errorf("version decl = %q, want %q", got, want)
	}
	if got := caps.MaxFragmentUniformVectors(); got != 1024 {
		t.Errorf("max fragment uniform vectors = %d, want 1024", got)
	}
	if got := caps.MSFBOType(); got != MSFBOStandard {
		t.Errorf("MSFBOType = %d, want standard", got)
	}
	if got := caps.StencilFormats(); len(got) != 3 || got[0] != FormatSTENCILINDEX8 ||
		got[1] != FormatSTENCILINDEX16 || got[2] != FormatDEPTH24STENCIL8 {
		t.Errorf("stencil formats = %v", got)
	}
	if !caps.FenceSyncSupport() || caps.FenceType() != FenceSyncObject {
		t.Error("expected GLsync fence support")
	}
	if got := caps.TimerQueryType(); got != TimerQueryRegular {
		t.Errorf("timer query type = %d, want regular", got)
	}
	if !caps.TextureSwizzleSupport() {
		t.Error("expected texture swizzle support")
	}
	if !caps.SamplerObjectSupport() || !caps.UseSamplerObjects() {
		t.Error("expected sampler object support")
	}
	// GL 4.1 offers program binaries but this driver reports zero binary formats.
	if caps.ProgramBinarySupport() {
		t.Error("program binary support should be off with zero formats")
	}
	if !caps.ShaderCaps.DualSourceBlendingSupport {
		t.Error("expected dual source blending (GL_ARB_blend_func_extended)")
	}
	if !caps.TextureBarrierSupport {
		t.Error("expected texture barrier support (GL_NV_texture_barrier)")
	}
	if caps.BlendEquationSupport != gpu.BasicBlendEquationSupport {
		t.Error("expected basic blend equation support (no advanced-blend extensions)")
	}
	if !caps.AnisoSupport || caps.MaxTextureMaxAnisotropy() != 16 {
		t.Errorf("aniso support/%v max %v", caps.AnisoSupport, caps.MaxTextureMaxAnisotropy())
	}
	if !caps.ShaderCaps.FloatIs32Bits || !caps.ShaderCaps.HalfIs32Bits {
		t.Error("expected fp32 precision report (127/127/23)")
	}
	if caps.MaxTextureSize != 16384 || caps.MaxRenderTargetSize != 16384 {
		t.Errorf("max sizes = %d/%d", caps.MaxTextureSize, caps.MaxRenderTargetSize)
	}
	if !caps.GpuTracingSupport {
		t.Error("expected gpu tracing (GL_EXT_debug_marker)")
	}
	if caps.DebugSupport() {
		t.Error("GL 4.1 has no KHR_debug-style support")
	}
	// GL_ARB_draw_indirect and GL_ARB_instanced_arrays are present but base-instance is not, so native indirect draws
	// stay off and client-side indirect buffers are required.
	if caps.BaseVertexBaseInstanceSupport() || caps.NativeDrawIndirectSupport {
		t.Error("expected no base-instance/native-indirect support on GL 4.1 without ARB_base_instance")
	}
	if !caps.UseClientSideIndirectBuffers {
		t.Error("expected client-side indirect buffers")
	}
	if !caps.DrawInstancedSupport {
		t.Error("expected instanced draw support")
	}
	if caps.SkipErrorChecks() {
		t.Error("skip error checks should default off")
	}

	// Color type → format mapping for the matrix.
	ctFormats := map[gpu.ColorType]Format{
		gpu.ColorTypeRGBA8888:       FormatRGBA8,
		gpu.ColorTypeBGRA8888:       FormatRGBA8,
		gpu.ColorTypeAlpha8:         FormatR8,
		gpu.ColorTypeGray8:          FormatR8,
		gpu.ColorTypeR8:             FormatR8,
		gpu.ColorTypeBGR565:         FormatRGB565, // 4.1 < 4.2 but GL_ARB_ES2_compatibility is present
		gpu.ColorTypeRGBAF16:        FormatRGBA16F,
		gpu.ColorTypeRGBAF16Clamped: FormatRGBA16F,
		gpu.ColorTypeRGBF16F16F16x:  FormatRGBA16F,
		gpu.ColorTypeAlphaF16:       FormatR16F,
		gpu.ColorTypeRGB888x:        FormatRGB8,
	}
	for ct, want := range ctFormats {
		if got := caps.GetFormatFromColorType(ct); got != want {
			t.Errorf("format for color type %d = %v, want %v", ct, got, want)
		}
	}

	// Texturability / renderability / sample counts.
	if !caps.IsFormatTexturable(FormatRGBA8, gpu.TextureType2D) {
		t.Error("RGBA8 must be texturable")
	}
	if got := caps.GetRenderTargetSampleCount(1, FormatRGBA8); got != 1 {
		t.Errorf("RGBA8 sample count for 1 = %d", got)
	}
	if got := caps.GetRenderTargetSampleCount(3, FormatRGBA8); got != 4 {
		t.Errorf("RGBA8 sample count for 3 = %d, want 4", got)
	}
	if got := caps.GetRenderTargetSampleCount(8, FormatRGBA8); got != 0 {
		t.Errorf("RGBA8 sample count for 8 = %d, want 0", got)
	}
	if got := caps.MaxRenderTargetSampleCount(FormatRGBA8); got != 4 {
		t.Errorf("RGBA8 max sample count = %d, want 4", got)
	}
	// RGB8 renders but never with MSAA on desktop GL.
	if got := caps.MaxRenderTargetSampleCount(FormatRGB8); got != 1 {
		t.Errorf("RGB8 max sample count = %d, want 1", got)
	}
	// Core profile: legacy ALPHA8/LUMINANCE8 are gone; kAlpha_8/kGray_8 ride R8.
	if caps.IsFormatTexturableAny(FormatALPHA8) {
		t.Error("ALPHA8 must not be texturable on a core profile")
	}
	if caps.IsFormatTexturableAny(FormatLUMINANCE8) {
		t.Error("LUMINANCE8 must not be texturable on a core profile")
	}
	// BGRA8 as a format never gains desktop capabilities; the BGRA color type rides RGBA8.
	if caps.IsFormatTexturableAny(FormatBGRA8) {
		t.Error("BGRA8 format must not be texturable on desktop GL")
	}
	// Formats outside the ported matrix report unsupported.
	for _, f := range []Format{
		FormatRG8, FormatRGB10A2, FormatRGBA4, FormatSRGB8ALPHA8, FormatR16,
		FormatRG16, FormatRGBA16, FormatRG16F, FormatLUMINANCE16F, FormatRGBX8,
	} {
		if caps.IsFormatTexturableAny(f) || caps.MaxRenderTargetSampleCount(f) != 0 {
			t.Errorf("format %v should be unsupported under the trim", f)
		}
	}

	// Upload/readback plumbing.
	if got := caps.TexImageOrStorageInternalFormat(FormatRGBA8); got != RGBA8 {
		t.Errorf("RGBA8 internal format = 0x%04X", got)
	}
	if !caps.FormatSupportsTexStorage(FormatRGBA8) {
		t.Error("RGBA8 should use tex storage (GL_ARB_texture_storage)")
	}
	if caps.FormatSupportsTexStorage(FormatRGB565) {
		t.Error("RGB565 must not use tex storage on desktop GL")
	}
	extFmt, extType := caps.TexSubImageExternalFormatAndType(FormatRGBA8, gpu.ColorTypeBGRA8888,
		gpu.ColorTypeBGRA8888)
	if extFmt != BGRA || extType != UNSIGNED_BYTE {
		t.Errorf("BGRA upload format/type = 0x%04X/0x%04X", extFmt, extType)
	}
	readFmt, readType := caps.ReadPixelsFormat(FormatRGBA8, gpu.ColorTypeRGBA8888,
		gpu.ColorTypeBGRA8888)
	if readFmt != BGRA || readType != UNSIGNED_BYTE {
		t.Errorf("BGRA read format/type = 0x%04X/0x%04X", readFmt, readType)
	}
	write := caps.SupportedWritePixelsColorType(gpu.ColorTypeBGRA8888, FormatRGBA8,
		gpu.ColorTypeBGRA8888)
	if write.ColorType != gpu.ColorTypeBGRA8888 || write.OffsetAlignmentForTransferBuffer != 1 {
		t.Errorf("BGRA write = %+v", write)
	}
	read := caps.SupportedReadPixelsColorType(gpu.ColorTypeRGBA8888, FormatRGBA8,
		gpu.ColorTypeBGRA8888)
	if read.ColorType != gpu.ColorTypeBGRA8888 || read.OffsetAlignmentForTransferBuffer != 4 {
		t.Errorf("BGRA read = %+v", read)
	}

	// Swizzles.
	if got := caps.ReadSwizzle(FormatR8, gpu.ColorTypeAlpha8); got.String() != "000r" {
		t.Errorf("R8/Alpha8 read swizzle = %q", got)
	}
	if got := caps.WriteSwizzle(FormatR8, gpu.ColorTypeAlpha8); got.String() != "a000" {
		t.Errorf("R8/Alpha8 write swizzle = %q", got)
	}
	if got := caps.ReadSwizzle(FormatRGBA8, gpu.ColorTypeRGB888x); got.String() != "rgb1" {
		t.Errorf("RGBA8/RGB888x read swizzle = %q", got)
	}
	if got := caps.ReadSwizzle(FormatRGBA8, gpu.ColorTypeRGBA8888); got != gpu.SwizzleRGBA {
		t.Errorf("RGBA8/RGBA8888 read swizzle = %q", got)
	}

	// Default formats and fallbacks.
	if got := caps.GetDefaultBackendFormat(gpu.ColorTypeGray8, true); got != FormatUnknown {
		t.Errorf("Gray8 renderable default = %v, want unknown", got)
	}
	if got := caps.GetDefaultBackendFormat(gpu.ColorTypeGray8, false); got != FormatR8 {
		t.Errorf("Gray8 default = %v, want R8", got)
	}
	ct, format := caps.GetFallbackColorTypeAndFormat(gpu.ColorTypeGray8, 1)
	if ct != gpu.ColorTypeRGB888x || format != FormatRGB8 {
		t.Errorf("Gray8 fallback = %d/%v, want RGB888x/RGB8", ct, format)
	}

	// Copy primitives.
	tex2D := gpu.TextureType2D
	if !caps.CanCopyTexSubImage(FormatRGBA8, false, &tex2D, FormatRGBA8, false, &tex2D) {
		t.Error("RGBA8→RGBA8 CopyTexSubImage should be allowed")
	}
	if caps.CanCopyTexSubImage(FormatRGBA8, true, &tex2D, FormatRGBA8, false, &tex2D) {
		t.Error("CopyTexSubImage with MSAA renderbuffer dst must be rejected")
	}
	if caps.CanCopyTexSubImage(FormatRGB565, false, &tex2D, FormatRGBA8, false, &tex2D) {
		t.Error("CopyTexSubImage across external types must be rejected")
	}

	// Stencil format index bookkeeping.
	if caps.HasStencilFormatBeenDeterminedForFormat(FormatRGBA8) {
		t.Error("stencil format should start undetermined")
	}
	caps.SetStencilFormatIndexForFormat(FormatRGBA8, 2)
	if !caps.HasStencilFormatBeenDeterminedForFormat(FormatRGBA8) ||
		caps.StencilFormatIndexForFormat(FormatRGBA8) != 2 {
		t.Error("stencil format index not recorded")
	}
	caps.SetStencilFormatIndexForFormat(FormatRGB565, -1)
	if !caps.HasStencilFormatBeenDeterminedForFormat(FormatRGB565) ||
		caps.StencilFormatIndexForFormat(FormatRGB565) >= 0 {
		t.Error("unsupported stencil format not recorded")
	}
}

// mesaLlvmpipeDriver models the planned Linux CI environment: Mesa llvmpipe, GL 4.5 core.
func mesaLlvmpipeDriver() *capsFakeDriver {
	return &capsFakeDriver{
		version:     "4.5 (Core Profile) Mesa 24.0.5",
		vendor:      "Mesa/X.org",
		renderer:    "llvmpipe (LLVM 15.0.7, 256 bits)",
		glslVersion: "4.50",
		extensions:  []string{"GL_ARB_texture_storage"},
		integers: map[uint32]int32{
			MAX_FRAGMENT_UNIFORM_COMPONENTS: 16384,
			CONTEXT_PROFILE_MASK:            CONTEXT_CORE_PROFILE_BIT,
			MAX_VERTEX_ATTRIBS:              32,
			MAX_TEXTURE_IMAGE_UNITS:         32,
			MAX_TEXTURE_SIZE:                16384,
			MAX_RENDERBUFFER_SIZE:           16384,
			MAX_SAMPLES:                     8,
		},
		programBinaryFormats: []int32{0x1234},
		sampleCounts:         []int32{8, 4, 2},
		maxAniso:             0,
		precisionBits:        23,
	}
}

func TestCapsMesaLlvmpipeProfile(t *testing.T) {
	ctxInfo := mesaLlvmpipeDriver().newContextInfo(t, nil)
	caps := ctxInfo.Caps

	if got := ctxInfo.Driver(); got != DriverMesa {
		t.Errorf("driver = %d, want Mesa", got)
	}
	if got, want := ctxInfo.DriverVersion(), DriverVer(24, 0, 0); got != want {
		t.Errorf("driver version = %x, want %x", got, want)
	}
	if got := ctxInfo.Renderer(); got != RendererGalliumLLVM {
		t.Errorf("renderer = %d, want GalliumLLVM", got)
	}
	if got := ctxInfo.Vendor(); got != VendorOther {
		t.Errorf("vendor = %d, want Other", got)
	}
	if got := ctxInfo.GLSLGeneration; got != gpu.GLSL420 {
		t.Errorf("GLSL generation = %d, want GLSL420", got)
	}
	if got, want := caps.ShaderCaps.VersionDeclString, "#version 420\n"; got != want {
		t.Errorf("version decl = %q, want %q", got, want)
	}
	if !caps.DebugSupport() {
		t.Error("GL 4.5 has debug support")
	}
	if !caps.ClearTextureSupport() {
		t.Error("GL 4.5 (>= 4.4) has clear texture support")
	}
	if !caps.TextureBarrierSupport {
		t.Error("GL 4.5 has texture barrier support")
	}
	if !caps.BaseVertexBaseInstanceSupport() || !caps.NativeDrawIndirectSupport ||
		caps.MultiDrawType() != MultiDrawIndirect {
		t.Error("GL 4.5 (>= 4.3) supports native multi-draw-indirect")
	}
	if caps.UseClientSideIndirectBuffers {
		t.Error("native indirect support should not require client-side indirect buffers")
	}
	if got := caps.InvalidateFBType(); got != InvalidateFBInvalidate {
		t.Errorf("invalidate FB type = %d, want invalidate", got)
	}
	if got := caps.InvalidateBufferType(); got != InvalidateBufferNullData {
		t.Errorf("invalidate buffer type = %d, want null-data (extension string not advertised)", got)
	}
	if caps.AnisoSupport {
		t.Error("no aniso support without the extension below GL 4.6")
	}
	if !caps.ProgramBinarySupport() || !caps.ProgramBinaryFormatIsValid(0x1234) ||
		caps.ProgramBinaryFormatIsValid(0x9999) {
		t.Error("program binary format bookkeeping wrong")
	}
	// GL 4.5 >= 4.2: RGB565 is supported without ARB_ES2_compatibility.
	if got := caps.GetFormatFromColorType(gpu.ColorTypeBGR565); got != FormatRGB565 {
		t.Errorf("BGR565 format = %v, want RGB565", got)
	}
	// Sample counts through the internalformat-query path: [8,4,2] descending → [1,2,4,8] ascending.
	if got := caps.GetRenderTargetSampleCount(2, FormatRGBA8); got != 2 {
		t.Errorf("RGBA8 sample count for 2 = %d", got)
	}
	if got := caps.GetRenderTargetSampleCount(5, FormatRGBA8); got != 8 {
		t.Errorf("RGBA8 sample count for 5 = %d, want 8", got)
	}
	if got := caps.MaxRenderTargetSampleCount(FormatRGBA8); got != 8 {
		t.Errorf("RGBA8 max sample count = %d, want 8", got)
	}
}

// intelMacDriver models an Intel Iris mac context, exercising the desktop Intel workarounds.
func intelMacDriver() *capsFakeDriver {
	return &capsFakeDriver{
		version:     "4.1 INTEL-14.7.28",
		vendor:      "Intel Inc.",
		renderer:    "Intel(R) Iris(TM) Plus Graphics 640",
		glslVersion: "4.10",
		// GL_ARB_ES2_compatibility is core in 4.1 and advertised by every real 4.1 driver; this codebase resolves
		// glGetShaderPrecisionFormat only when it (or GL 4.3) is present.
		extensions: []string{
			"GL_ARB_texture_storage", "GL_KHR_blend_equation_advanced",
			"GL_ARB_ES2_compatibility",
		},
		integers: map[uint32]int32{
			MAX_FRAGMENT_UNIFORM_COMPONENTS: 4096,
			CONTEXT_PROFILE_MASK:            CONTEXT_CORE_PROFILE_BIT,
			MAX_VERTEX_ATTRIBS:              16,
			MAX_TEXTURE_IMAGE_UNITS:         16,
			MAX_TEXTURE_SIZE:                16384,
			MAX_RENDERBUFFER_SIZE:           16384,
			MAX_SAMPLES:                     8,
		},
		sampleCounts:  []int32{8, 4, 2},
		maxAniso:      16,
		precisionBits: 23,
	}
}

func TestCapsIntelWorkarounds(t *testing.T) {
	ctxInfo := intelMacDriver().newContextInfo(t, nil)
	caps := ctxInfo.Caps

	if got := ctxInfo.Vendor(); got != VendorIntel {
		t.Errorf("vendor = %d, want Intel", got)
	}
	if got := ctxInfo.Renderer(); got != RendererIntelKabyLake {
		t.Errorf("renderer = %d, want KabyLake", got)
	}
	if got := ctxInfo.Driver(); got != DriverIntel {
		t.Errorf("driver = %d, want Intel", got)
	}
	if got, want := ctxInfo.DriverVersion(), DriverVer(14, 7, 28); got != want {
		t.Errorf("driver version = %x, want %x", got, want)
	}
	// Pre-IceLake Intel: MSAA is fully disabled.
	if caps.MSFBOType() != MSFBONone {
		t.Error("expected MSAA disabled on pre-Gen11 Intel")
	}
	if got := caps.MaxRenderTargetSampleCount(FormatRGBA8); got != 1 {
		t.Errorf("RGBA8 max samples with Intel workaround = %d, want 1", got)
	}
	if !caps.DoManualMipmapping() {
		t.Error("expected manual mipmapping on Intel")
	}
	if !caps.ShaderCaps.MustForceNegatedAtanParamToFloat ||
		!caps.ShaderCaps.MustDoOpBetweenFloorAndAbs {
		t.Error("expected Intel shader workarounds")
	}
	// The KHR advanced-blend extension is advertised but the Intel driver disable wins.
	if caps.BlendEquationSupport != gpu.BasicBlendEquationSupport ||
		caps.ShaderCaps.AdvBlendEqInteraction != gpu.NotSupportedAdvBlendEqInteraction {
		t.Error("expected advanced blend disabled on the Intel driver")
	}
	if !caps.DisableTessellationPathRendererCap {
		t.Error("expected tessellation path renderer disabled on Intel")
	}
	if got := caps.RegenerateMipmapType(); got != RegenerateMipmapBasePlusMaxLevel {
		t.Errorf("regenerate mipmap type = %d, want base-plus-max-level", got)
	}

	// With workarounds disabled, none of the above fire.
	options := gpu.DefaultContextOptions()
	options.DisableDriverCorrectnessWorkarounds = true
	caps2 := intelMacDriver().newContextInfo(t, options).Caps
	if caps2.MSFBOType() != MSFBOStandard || caps2.DoManualMipmapping() ||
		caps2.ShaderCaps.MustForceNegatedAtanParamToFloat {
		t.Error("workarounds applied despite DisableDriverCorrectnessWorkarounds")
	}
	if caps2.BlendEquationSupport != gpu.AdvancedBlendEquationSupport {
		t.Error("expected advanced blend support with workarounds disabled")
	}
}

func TestCapsOptionsOverrides(t *testing.T) {
	options := gpu.DefaultContextOptions()
	options.AvoidStencilBuffers = true
	options.MaxTextureSizeOverride = 4096
	options.UseDrawInsteadOfClear = gpu.EnableYes
	options.SkipGLErrorChecks = gpu.EnableYes
	options.ShaderCacheStrategy = gpu.ShaderCacheStrategyBackendSource
	ctxInfo := mesaLlvmpipeDriver().newContextInfo(t, options)
	caps := ctxInfo.Caps

	// Avoiding stencil buffers also disables MSAA and leaves no stencil formats.
	if caps.MSFBOType() != MSFBONone || len(caps.StencilFormats()) != 0 {
		t.Error("AvoidStencilBuffers must disable MSAA and stencil formats")
	}
	if !caps.AvoidStencilBuffers {
		t.Error("AvoidStencilBuffers not recorded")
	}
	if caps.MaxTextureSize != 4096 || caps.MaxRenderTargetSize != 4096 {
		t.Errorf("size override not applied: %d/%d", caps.MaxTextureSize, caps.MaxRenderTargetSize)
	}
	if !caps.PerformColorClearsAsDraws || !caps.PerformStencilClearsAsDraws {
		t.Error("UseDrawInsteadOfClear not applied")
	}
	if !caps.SkipErrorChecks() {
		t.Error("SkipGLErrorChecks not applied")
	}
	if caps.ProgramBinarySupport() {
		t.Error("non-binary shader cache strategy must disable program binaries")
	}
}
