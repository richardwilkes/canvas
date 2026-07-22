// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The manual draw-based mipmap regeneration lane. On drivers where glGenerateMipmap is buggy for sRGB textures
// (doManualMipmapping: Vendor ATI/Intel anywhere, NVIDIA on Mac — see caps.go), each mip level is rendered from the
// previous one with a tiny dedicated program: a fullscreen quad sampling the prior level through TEXTURE_BASE_LEVEL
// clamping, box-filtering 1, 2, or 4 taps depending on the parity of the source dimensions. The four programs
// (even/even, odd-w, odd-h, odd/odd) are built lazily outside the program cache and emit GLSL directly.

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// mipmapProgram is one lazily-built mip-downsample program (see textureSizeToMipmapProgramIdx).
type mipmapProgram struct {
	program              uint32
	textureUniform       int32
	texCoordXformUniform int32
}

// textureSizeToMipmapProgramIdx selects which of the four mip-downsample programs to use based on the parity of the
// source dimensions.
func textureSizeToMipmapProgramIdx(width, height int32) int {
	idx := 0
	if width > 1 && width&1 != 0 {
		idx |= 0x2
	}
	if height > 1 && height&1 != 0 {
		idx |= 0x1
	}
	return idx
}

// compileMipmapShader compiles and attaches one shader of a mipmap program. Unlike the program-cache lane's panic on
// failure, the mipmap lane returns failure so the caller can fall back. Returns the shader id, or 0 on failure.
func (g *Gpu) compileMipmapShader(programID, shaderType uint32, source string) uint32 {
	shaderID := g.fns().CreateShader(shaderType)
	if shaderID == 0 {
		return 0
	}
	src := make([]byte, len(source)+1)
	copy(src, source)
	srcPtr := &src[0]
	length := int32(len(source))
	g.fns().ShaderSource(shaderID, 1, uintptr(unsafe.Pointer(&srcPtr)), &length)
	g.programCache.stats.ShaderCompilations++
	g.fns().CompileShader(shaderID)
	var compiled int32
	g.fns().GetShaderiv(shaderID, COMPILE_STATUS, &compiled)
	if compiled == 0 {
		g.fns().DeleteShader(shaderID)
		return 0
	}
	g.fns().AttachShader(programID, shaderID)
	return shaderID
}

// createMipmapProgram compiles and links the mip-downsample program for the given parity index.
func (g *Gpu) createMipmapProgram(progIdx int) bool {
	oddWidth := progIdx&0x2 != 0
	oddHeight := progIdx&0x1 != 0
	numTaps := 1
	if oddWidth {
		numTaps *= 2
	}
	if oddHeight {
		numTaps *= 2
	}

	if g.mipmapPrograms[progIdx].program != 0 {
		panic("mipmap program already created")
	}
	programID := g.fns().CreateProgram()
	if programID == 0 {
		return false
	}

	shaderCaps := g.Caps().ShaderCaps
	version := shaderCaps.VersionDeclString
	// noperspective is core in desktop GLSL 1.30+ (the extension string is only needed on ES, which is trimmed here);
	// the interpolation qualifier is emitted only when the caps say it's usable.
	varyingQual := ""
	if shaderCaps.NoPerspectiveInterpolationSupport {
		varyingQual = "noperspective "
	}

	vs := version
	vs += "in vec2 a_vertex;\n"
	vs += "uniform vec4 u_texCoordXform;\n"
	for i := 0; i < numTaps; i++ {
		vs += varyingQual + "out vec2 v_texCoord" + string(rune('0'+i)) + ";\n"
	}
	vs += "void main() {\n" +
		"    gl_Position = vec4(a_vertex * 2.0 - 1.0, 0.0, 1.0);\n"
	switch {
	case oddWidth && oddHeight:
		vs += "    v_texCoord0 = a_vertex.xy * u_texCoordXform.yw;\n" +
			"    v_texCoord1 = a_vertex.xy * u_texCoordXform.yw + vec2(u_texCoordXform.x, 0.0);\n" +
			"    v_texCoord2 = a_vertex.xy * u_texCoordXform.yw + vec2(0.0, u_texCoordXform.z);\n" +
			"    v_texCoord3 = a_vertex.xy * u_texCoordXform.yw + u_texCoordXform.xz;\n"
	case oddWidth:
		vs += "    v_texCoord0 = a_vertex.xy * vec2(u_texCoordXform.y, 1.0);\n" +
			"    v_texCoord1 = a_vertex.xy * vec2(u_texCoordXform.y, 1.0) + vec2(u_texCoordXform.x, 0.0);\n"
	case oddHeight:
		vs += "    v_texCoord0 = a_vertex.xy * vec2(1.0, u_texCoordXform.w);\n" +
			"    v_texCoord1 = a_vertex.xy * vec2(1.0, u_texCoordXform.w) + vec2(0.0, u_texCoordXform.z);\n"
	default:
		vs += "    v_texCoord0 = a_vertex.xy;\n"
	}
	vs += "}\n"

	fs := version
	for i := 0; i < numTaps; i++ {
		fs += varyingQual + "in vec2 v_texCoord" + string(rune('0'+i)) + ";\n"
	}
	fs += "uniform sampler2D u_texture;\n" +
		"out vec4 o_FragColor;\n" +
		"void main() {\n"
	switch {
	case oddWidth && oddHeight:
		fs += "    o_FragColor = (texture(u_texture, v_texCoord0) +\n" +
			"                   texture(u_texture, v_texCoord1) +\n" +
			"                   texture(u_texture, v_texCoord2) +\n" +
			"                   texture(u_texture, v_texCoord3)) * 0.25;\n"
	case oddWidth || oddHeight:
		fs += "    o_FragColor = (texture(u_texture, v_texCoord0) +\n" +
			"                   texture(u_texture, v_texCoord1)) * 0.5;\n"
	default:
		fs += "    o_FragColor = texture(u_texture, v_texCoord0);\n"
	}
	fs += "}\n"

	vshader := g.compileMipmapShader(programID, VERTEX_SHADER, vs)
	if vshader == 0 {
		g.fns().DeleteProgram(programID)
		return false
	}
	fshader := g.compileMipmapShader(programID, FRAGMENT_SHADER, fs)
	if fshader == 0 {
		g.fns().DeleteShader(vshader)
		g.fns().DeleteProgram(programID)
		return false
	}

	// Attribute locations must be bound before linking to take effect on the linked program.
	g.fns().BindAttribLocationString(programID, 0, "a_vertex")

	g.fns().LinkProgram(programID)
	var linked int32
	g.fns().GetProgramiv(programID, LINK_STATUS, &linked)
	if linked == 0 {
		g.fns().DeleteShader(vshader)
		g.fns().DeleteShader(fshader)
		g.fns().DeleteProgram(programID)
		return false
	}

	g.mipmapPrograms[progIdx].program = programID
	g.mipmapPrograms[progIdx].textureUniform = g.fns().GetUniformLocationString(programID, "u_texture")
	g.mipmapPrograms[progIdx].texCoordXformUniform = g.fns().GetUniformLocationString(programID, "u_texCoordXform")

	g.fns().DeleteShader(vshader)
	g.fns().DeleteShader(fshader)
	return true
}

// regenerateMipmapLevelsByDraws performs the manual (doManualMipmapping) mipmap regeneration: iterative downsampling
// draws, each mip level rendered from the previous through a temporary FBO. The caller has already validated the
// texture (2D target, dirty mips, renderable format) and handled the dirty context.
func (g *Gpu) regenerateMipmapLevelsByDraws(texture *Texture) bool {
	dims := texture.Surface().Dimensions()
	width := dims.Width
	height := dims.Height
	levelCount := int32(texture.MaxMipmapLevel()) + 1

	// Create (if necessary), then bind the temporary FBO.
	if g.tempDstFBOID == 0 {
		g.fns().GenFramebuffers(1, &g.tempDstFBOID)
	}
	g.BindFramebuffer(FRAMEBUFFER, g.tempDstFBOID)
	g.InvalidateBoundRenderTarget()

	// Bind the texture, to get things configured for filtering. The base level changes per blit below. The mipmap
	// program does not do any swizzling.
	g.setTextureUnit(0)
	g.BindTexture(0, gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp,
		gpu.FilterModeLinear, gpu.MipmapModeNone), texture)

	// Vertex data: a static unit quad, shared by all four programs.
	if g.mipmapProgramArrayBuffer == nil {
		vdata := []float32{
			0, 0,
			0, 1,
			1, 0,
			1, 1,
		}
		size := uint64(len(vdata) * 4)
		buffer := MakeBuffer(g, size, gpu.BufferTypeVertex, gpu.AccessPatternStatic)
		if buffer != nil {
			bytes := unsafe.Slice((*byte)(unsafe.Pointer(&vdata[0])), size)
			if !buffer.UpdateData(bytes, 0, false) {
				buffer.Unref()
				buffer = nil
			}
		}
		g.mipmapProgramArrayBuffer = buffer
	}
	if g.mipmapProgramArrayBuffer == nil {
		return false
	}

	g.hwVertexArrayState.setVertexArrayID(g, 0)
	attribs := g.hwVertexArrayState.bindInternalVertexArray(g, nil)
	attribs.EnableVertexArrays(g, 1, false)
	attribs.Set(g, 0, g.mipmapProgramArrayBuffer, gpu.VertexAttribTypeFloat2, true, 8, 0, 0)

	// Set "simple" state once.
	g.flushBlendAndColorWrite(gpu.MakeBlendInfo(), gpu.SwizzleRGBA)
	g.flushScissorTest(false)
	g.disableWindowRectangles()
	g.disableStencil()

	// Do all the blits.
	for level := int32(1); level < levelCount; level++ {
		// Get and bind the program for this particular downsample (filter shape can vary).
		progIdx := textureSizeToMipmapProgramIdx(width, height)
		if g.mipmapPrograms[progIdx].program == 0 {
			if !g.createMipmapProgram(progIdx) {
				// Invalidate all params to cover the base level change in a previous iteration.
				texture.TextureParamsModified()
				return false
			}
		}
		g.flushProgramID(g.mipmapPrograms[progIdx].program)

		// The texcoord uniform is (1/w, (w-1)/w, 1/h, (h-1)/h).
		invWidth := 1 / float32(width)
		invHeight := 1 / float32(height)
		g.fns().Uniform4f(g.mipmapPrograms[progIdx].texCoordXformUniform,
			invWidth, float32(width-1)*invWidth, invHeight, float32(height-1)*invHeight)
		g.fns().Uniform1i(g.mipmapPrograms[progIdx].textureUniform, 0)

		// Set the base level so that we only sample from the previous mip.
		g.fns().TexParameteri(TEXTURE_2D, TEXTURE_BASE_LEVEL, level-1)
		// Setting the max level is technically unnecessary and can affect validation for the framebuffer. However, by
		// making it clear that a rendering feedback loop is not occurring, we avoid hitting a slow path on some
		// drivers.
		if g.glCaps().RegenerateMipmapType() == RegenerateMipmapBasePlusMaxLevel {
			g.fns().TexParameteri(TEXTURE_2D, TEXTURE_MAX_LEVEL, level-1)
		}

		g.fns().FramebufferTexture2D(FRAMEBUFFER, COLOR_ATTACHMENT0, TEXTURE_2D,
			texture.TextureID(), level)

		width = max(1, width/2)
		height = max(1, height/2)
		g.flushViewport(geom.IRectLTRB(0, 0, width, height), height, gpu.OriginTopLeft)

		g.fns().DrawArrays(TRIANGLE_STRIP, 0, 4)
	}

	// Unbind.
	g.fns().FramebufferTexture2D(FRAMEBUFFER, COLOR_ATTACHMENT0, TEXTURE_2D, 0, 0)

	// We modified the base level (and possibly max level) params: record the shadow so the next BindTexture restores
	// them. We drew the 2nd to last level into the last level.
	nonsamplerState := *texture.Parameters().NonsamplerState()
	nonsamplerState.BaseMipmapLevel = levelCount - 2
	if g.glCaps().RegenerateMipmapType() == RegenerateMipmapBasePlusMaxLevel {
		nonsamplerState.MaxMipmapLevel = levelCount - 2
	}
	texture.Parameters().Set(nil, nonsamplerState, g.resetTimestampForTextureParameters)

	return true
}
