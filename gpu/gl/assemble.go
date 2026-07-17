// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Assembles a GL function-pointer interface by resolving each entry point through a GetProc callback, gating
// desktop-GL-only entry points by version/extension support. Per the desktop trim, only the desktop-GL assembler is
// implemented; an ES or WebGL context yields nil where the assembler would otherwise have dispatched to an
// ES/WebGL-specific assembler.

package gl

// Interface is the resolved GL proc table plus the standard and extension set it was assembled for. All GL calls go
// through it; the function pointers are only valid while the GL context they were resolved from is current.
type Interface struct {
	Extensions Extensions
	Functions  Functions
	Standard   Standard
}

// HasExtension reports whether ext is present in this interface's extension set.
func (i *Interface) HasExtension(ext string) bool { return i.Extensions.Has(ext) }

// MakeAssembledInterface sniffs the standard from GL_VERSION and dispatches to the standard-specific assembler. A GL
// context must be current on this thread.
func MakeAssembledInterface(get GetProc) *Interface {
	getString := get("glGetString")
	if getString == 0 {
		return nil
	}
	verStr := goString(rawCall(getString, VERSION))
	if verStr == "" {
		return nil
	}
	switch GetStandardInUseFromString(verStr) {
	case StandardGL:
		return MakeAssembledGLInterface(get)
	case StandardGLES, StandardWebGL:
		// Desktop trim: the ES and WebGL assemblers are not ported.
		return nil
	default:
		return nil
	}
}

// eglQueryAndDisplay resolves eglQueryString and the current EGL display so EGL extensions can be folded into the
// extension set (relevant only for EGL-hosted desktop GL).
func eglQueryAndDisplay(get GetProc) (queryString, display uintptr) {
	queryString = get("eglQueryString")
	display = EGLNoDisplay
	if queryString != 0 {
		getCurrentDisplay := get("eglGetCurrentDisplay")
		if getCurrentDisplay != 0 {
			display = rawCall(getCurrentDisplay)
		} else {
			queryString = 0
		}
	}
	return queryString, display
}

// MakeAssembledGLInterface assembles and version/extension-gates the proc table for a desktop GL context that is
// current on this thread.
func MakeAssembledGLInterface(get GetProc) *Interface {
	getString := get("glGetString")
	getStringi := get("glGetStringi")
	getIntegerv := get("glGetIntegerv")

	// GetStringi may be zero depending on the GL version.
	if getString == 0 || getIntegerv == 0 {
		return nil
	}

	glVer := GetVersionFromString(goString(rawCall(getString, VERSION)))
	if glVer < Ver(2, 0) || glVer == InvalidVersion {
		// This is our minimum for non-ES GL.
		return nil
	}

	queryString, display := eglQueryAndDisplay(get)
	var extensions Extensions
	if !extensions.Init(StandardGL, getString, getStringi, getIntegerv, queryString, display) {
		return nil
	}

	intf := &Interface{}
	f := &intf.Functions

	f.activeTexture = get("glActiveTexture")
	f.attachShader = get("glAttachShader")
	f.bindAttribLocation = get("glBindAttribLocation")
	f.bindBuffer = get("glBindBuffer")
	f.bindTexture = get("glBindTexture")
	f.blendColor = get("glBlendColor")
	f.blendEquation = get("glBlendEquation")
	f.blendFunc = get("glBlendFunc")
	f.bufferData = get("glBufferData")
	f.bufferSubData = get("glBufferSubData")
	f.clear = get("glClear")
	f.clearColor = get("glClearColor")
	f.clearStencil = get("glClearStencil")
	f.colorMask = get("glColorMask")
	f.compileShader = get("glCompileShader")
	f.compressedTexImage2D = get("glCompressedTexImage2D")
	f.compressedTexSubImage2D = get("glCompressedTexSubImage2D")
	f.copyTexSubImage2D = get("glCopyTexSubImage2D")
	f.createProgram = get("glCreateProgram")
	f.createShader = get("glCreateShader")
	f.cullFace = get("glCullFace")
	f.deleteBuffers = get("glDeleteBuffers")
	f.deleteProgram = get("glDeleteProgram")
	f.deleteShader = get("glDeleteShader")
	f.deleteTextures = get("glDeleteTextures")
	f.depthMask = get("glDepthMask")
	f.disable = get("glDisable")
	f.disableVertexAttribArray = get("glDisableVertexAttribArray")
	f.drawArrays = get("glDrawArrays")
	f.drawElements = get("glDrawElements")
	f.enable = get("glEnable")
	f.enableVertexAttribArray = get("glEnableVertexAttribArray")
	f.finish = get("glFinish")
	f.flush = get("glFlush")
	f.frontFace = get("glFrontFace")
	f.genBuffers = get("glGenBuffers")
	f.genTextures = get("glGenTextures")
	f.getBufferParameteriv = get("glGetBufferParameteriv")
	f.getError = get("glGetError")
	f.getFloatv = get("glGetFloatv")
	f.getIntegerv = get("glGetIntegerv")
	f.getProgramInfoLog = get("glGetProgramInfoLog")
	f.getProgramiv = get("glGetProgramiv")
	f.getShaderInfoLog = get("glGetShaderInfoLog")
	f.getShaderiv = get("glGetShaderiv")
	f.getString = get("glGetString")
	f.getUniformLocation = get("glGetUniformLocation")
	f.isTexture = get("glIsTexture")
	f.lineWidth = get("glLineWidth")
	f.linkProgram = get("glLinkProgram")
	f.pixelStorei = get("glPixelStorei")
	f.readPixels = get("glReadPixels")
	f.scissor = get("glScissor")
	f.shaderSource = get("glShaderSource")
	f.stencilFunc = get("glStencilFunc")
	f.stencilFuncSeparate = get("glStencilFuncSeparate")
	f.stencilMask = get("glStencilMask")
	f.stencilMaskSeparate = get("glStencilMaskSeparate")
	f.stencilOp = get("glStencilOp")
	f.stencilOpSeparate = get("glStencilOpSeparate")
	f.texImage2D = get("glTexImage2D")
	f.texParameterf = get("glTexParameterf")
	f.texParameterfv = get("glTexParameterfv")
	f.texParameteri = get("glTexParameteri")
	f.texParameteriv = get("glTexParameteriv")
	f.texSubImage2D = get("glTexSubImage2D")
	f.uniform1f = get("glUniform1f")
	f.uniform1fv = get("glUniform1fv")
	f.uniform1i = get("glUniform1i")
	f.uniform1iv = get("glUniform1iv")
	f.uniform2f = get("glUniform2f")
	f.uniform2fv = get("glUniform2fv")
	f.uniform2i = get("glUniform2i")
	f.uniform2iv = get("glUniform2iv")
	f.uniform3f = get("glUniform3f")
	f.uniform3fv = get("glUniform3fv")
	f.uniform3i = get("glUniform3i")
	f.uniform3iv = get("glUniform3iv")
	f.uniform4f = get("glUniform4f")
	f.uniform4fv = get("glUniform4fv")
	f.uniform4i = get("glUniform4i")
	f.uniform4iv = get("glUniform4iv")
	f.uniformMatrix2fv = get("glUniformMatrix2fv")
	f.uniformMatrix3fv = get("glUniformMatrix3fv")
	f.uniformMatrix4fv = get("glUniformMatrix4fv")
	f.useProgram = get("glUseProgram")
	f.vertexAttrib1f = get("glVertexAttrib1f")
	f.vertexAttrib2fv = get("glVertexAttrib2fv")
	f.vertexAttrib3fv = get("glVertexAttrib3fv")
	f.vertexAttrib4fv = get("glVertexAttrib4fv")
	f.vertexAttribPointer = get("glVertexAttribPointer")
	f.viewport = get("glViewport")

	f.drawBuffer = get("glDrawBuffer")
	f.polygonMode = get("glPolygonMode")

	if glVer >= Ver(3, 0) {
		f.getStringi = get("glGetStringi")
	}

	if glVer >= Ver(4, 2) {
		f.memoryBarrier = get("glMemoryBarrier")
	}

	switch {
	case glVer >= Ver(3, 0):
		f.bindVertexArray = get("glBindVertexArray")
		f.deleteVertexArrays = get("glDeleteVertexArrays")
		f.genVertexArrays = get("glGenVertexArrays")
	case extensions.Has("GL_ARB_vertex_array_object"):
		f.bindVertexArray = get("glBindVertexArray")
		f.deleteVertexArrays = get("glDeleteVertexArrays")
		f.genVertexArrays = get("glGenVertexArrays")
	case extensions.Has("GL_APPLE_vertex_array_object"):
		f.bindVertexArray = get("glBindVertexArrayAPPLE")
		f.deleteVertexArrays = get("glDeleteVertexArraysAPPLE")
		f.genVertexArrays = get("glGenVertexArraysAPPLE")
	}

	if glVer >= Ver(4, 0) || extensions.Has("GL_ARB_tessellation_shader") {
		f.patchParameteri = get("glPatchParameteri")
	}

	if glVer >= Ver(3, 0) {
		f.bindFragDataLocation = get("glBindFragDataLocation")
	}

	if glVer >= Ver(3, 3) || extensions.Has("GL_ARB_blend_func_extended") {
		f.bindFragDataLocationIndexed = get("glBindFragDataLocationIndexed")
	}

	switch {
	case extensions.Has("GL_KHR_blend_equation_advanced"):
		f.blendBarrier = get("glBlendBarrierKHR")
	case extensions.Has("GL_NV_blend_equation_advanced"):
		f.blendBarrier = get("glBlendBarrierNV")
	}

	if glVer >= Ver(4, 4) || extensions.Has("GL_ARB_clear_texture") {
		f.clearTexImage = get("glClearTexImage")
		f.clearTexSubImage = get("glClearTexSubImage")
	}

	switch {
	case glVer >= Ver(3, 1):
		f.drawArraysInstanced = get("glDrawArraysInstanced")
		f.drawElementsInstanced = get("glDrawElementsInstanced")
	case extensions.Has("GL_ARB_draw_instanced"):
		f.drawArraysInstanced = get("glDrawArraysInstanced")
		f.drawElementsInstanced = get("glDrawElementsInstanced")
	case extensions.Has("GL_EXT_draw_instanced"):
		f.drawArraysInstanced = get("glDrawArraysInstancedEXT")
		f.drawElementsInstanced = get("glDrawElementsInstancedEXT")
	}

	if glVer >= Ver(4, 2) || extensions.Has("GL_ARB_base_instance") {
		f.drawArraysInstancedBaseInstance = get("glDrawArraysInstancedBaseInstance")
		f.drawElementsInstancedBaseVertexBaseInstance = get("glDrawElementsInstancedBaseVertexBaseInstance")
	}

	f.drawBuffers = get("glDrawBuffers")
	f.readBuffer = get("glReadBuffer")

	if glVer >= Ver(4, 0) || extensions.Has("GL_ARB_draw_indirect") {
		f.drawArraysIndirect = get("glDrawArraysIndirect")
		f.drawElementsIndirect = get("glDrawElementsIndirect")
	}

	f.drawRangeElements = get("glDrawRangeElements")

	if glVer >= Ver(3, 2) || extensions.Has("GL_ARB_texture_multisample") {
		f.getMultisamplefv = get("glGetMultisamplefv")
	}

	f.getTexLevelParameteriv = get("glGetTexLevelParameteriv")

	if glVer >= Ver(4, 3) || extensions.Has("GL_ARB_multi_draw_indirect") {
		f.multiDrawArraysIndirect = get("glMultiDrawArraysIndirect")
		f.multiDrawElementsIndirect = get("glMultiDrawElementsIndirect")
	}

	if glVer >= Ver(3, 1) {
		f.texBuffer = get("glTexBuffer")
	}

	if glVer >= Ver(4, 3) {
		f.texBufferRange = get("glTexBufferRange")
	}

	switch {
	case glVer >= Ver(4, 2):
		f.texStorage2D = get("glTexStorage2D")
	case extensions.Has("GL_ARB_texture_storage"):
		f.texStorage2D = get("glTexStorage2D")
	case extensions.Has("GL_EXT_texture_storage"):
		f.texStorage2D = get("glTexStorage2DEXT")
	}

	switch {
	case glVer >= Ver(4, 5):
		f.textureBarrier = get("glTextureBarrier")
	case extensions.Has("GL_ARB_texture_barrier"):
		f.textureBarrier = get("glTextureBarrier")
	case extensions.Has("GL_NV_texture_barrier"):
		f.textureBarrier = get("glTextureBarrierNV")
	}

	if glVer >= Ver(3, 2) || extensions.Has("GL_ARB_instanced_arrays") {
		f.vertexAttribDivisor = get("glVertexAttribDivisor")
	}

	if glVer >= Ver(3, 0) {
		f.vertexAttribIPointer = get("glVertexAttribIPointer")
	}

	switch {
	case glVer >= Ver(3, 0):
		f.bindFramebuffer = get("glBindFramebuffer")
		f.bindRenderbuffer = get("glBindRenderbuffer")
		f.checkFramebufferStatus = get("glCheckFramebufferStatus")
		f.deleteFramebuffers = get("glDeleteFramebuffers")
		f.deleteRenderbuffers = get("glDeleteRenderbuffers")
		f.framebufferRenderbuffer = get("glFramebufferRenderbuffer")
		f.framebufferTexture2D = get("glFramebufferTexture2D")
		f.genFramebuffers = get("glGenFramebuffers")
		f.genRenderbuffers = get("glGenRenderbuffers")
		f.generateMipmap = get("glGenerateMipmap")
		f.getFramebufferAttachmentParameteriv = get("glGetFramebufferAttachmentParameteriv")
		f.getRenderbufferParameteriv = get("glGetRenderbufferParameteriv")
		f.renderbufferStorage = get("glRenderbufferStorage")
	case extensions.Has("GL_ARB_framebuffer_object"):
		f.bindFramebuffer = get("glBindFramebuffer")
		f.bindRenderbuffer = get("glBindRenderbuffer")
		f.checkFramebufferStatus = get("glCheckFramebufferStatus")
		f.deleteFramebuffers = get("glDeleteFramebuffers")
		f.deleteRenderbuffers = get("glDeleteRenderbuffers")
		f.framebufferRenderbuffer = get("glFramebufferRenderbuffer")
		f.framebufferTexture2D = get("glFramebufferTexture2D")
		f.genFramebuffers = get("glGenFramebuffers")
		f.genRenderbuffers = get("glGenRenderbuffers")
		f.generateMipmap = get("glGenerateMipmap")
		f.getFramebufferAttachmentParameteriv = get("glGetFramebufferAttachmentParameteriv")
		f.getRenderbufferParameteriv = get("glGetRenderbufferParameteriv")
		f.renderbufferStorage = get("glRenderbufferStorage")
	case extensions.Has("GL_EXT_framebuffer_object"):
		f.bindFramebuffer = get("glBindFramebufferEXT")
		f.bindRenderbuffer = get("glBindRenderbufferEXT")
		f.checkFramebufferStatus = get("glCheckFramebufferStatusEXT")
		f.deleteFramebuffers = get("glDeleteFramebuffersEXT")
		f.deleteRenderbuffers = get("glDeleteRenderbuffersEXT")
		f.framebufferRenderbuffer = get("glFramebufferRenderbufferEXT")
		f.framebufferTexture2D = get("glFramebufferTexture2DEXT")
		f.genFramebuffers = get("glGenFramebuffersEXT")
		f.genRenderbuffers = get("glGenRenderbuffersEXT")
		f.generateMipmap = get("glGenerateMipmapEXT")
		f.getFramebufferAttachmentParameteriv = get("glGetFramebufferAttachmentParameterivEXT")
		f.getRenderbufferParameteriv = get("glGetRenderbufferParameterivEXT")
		f.renderbufferStorage = get("glRenderbufferStorageEXT")
	}

	switch {
	case glVer >= Ver(3, 0):
		f.blitFramebuffer = get("glBlitFramebuffer")
	case extensions.Has("GL_ARB_framebuffer_object"):
		f.blitFramebuffer = get("glBlitFramebuffer")
	case extensions.Has("GL_EXT_framebuffer_blit"):
		f.blitFramebuffer = get("glBlitFramebufferEXT")
	}

	switch {
	case glVer >= Ver(3, 0):
		f.renderbufferStorageMultisample = get("glRenderbufferStorageMultisample")
	case extensions.Has("GL_ARB_framebuffer_object"):
		f.renderbufferStorageMultisample = get("glRenderbufferStorageMultisample")
	case extensions.Has("GL_EXT_framebuffer_multisample"):
		f.renderbufferStorageMultisample = get("glRenderbufferStorageMultisampleEXT")
	}

	f.mapBuffer = get("glMapBuffer")

	f.unmapBuffer = get("glUnmapBuffer")

	if glVer >= Ver(3, 0) || extensions.Has("GL_ARB_map_buffer_range") {
		f.flushMappedBufferRange = get("glFlushMappedBufferRange")
		f.mapBufferRange = get("glMapBufferRange")
	}

	if extensions.Has("GL_EXT_debug_marker") {
		f.insertEventMarker = get("glInsertEventMarkerEXT")
		f.popGroupMarker = get("glPopGroupMarkerEXT")
		f.pushGroupMarker = get("glPushGroupMarkerEXT")
	}

	if glVer >= Ver(3, 1) || extensions.Has("GL_ARB_copy_buffer") {
		f.copyBufferSubData = get("glCopyBufferSubData")
	}

	if glVer >= Ver(4, 3) || extensions.Has("GL_KHR_debug") {
		f.debugMessageCallback = get("glDebugMessageCallback")
		f.debugMessageControl = get("glDebugMessageControl")
		f.debugMessageInsert = get("glDebugMessageInsert")
		f.getDebugMessageLog = get("glGetDebugMessageLog")
		f.objectLabel = get("glObjectLabel")
		f.popDebugGroup = get("glPopDebugGroup")
		f.pushDebugGroup = get("glPushDebugGroup")
	}

	if extensions.Has("GL_EXT_window_rectangles") {
		f.windowRectangles = get("glWindowRectanglesEXT")
	}

	if glVer >= Ver(3, 2) || extensions.Has("GL_ARB_sync") {
		f.clientWaitSync = get("glClientWaitSync")
		f.deleteSync = get("glDeleteSync")
		f.fenceSync = get("glFenceSync")
		f.isSync = get("glIsSync")
		f.waitSync = get("glWaitSync")
	}

	if glVer >= Ver(4, 2) || extensions.Has("GL_ARB_internalformat_query") {
		f.getInternalformativ = get("glGetInternalformativ")
	}

	if glVer >= Ver(4, 1) {
		f.getProgramBinary = get("glGetProgramBinary")
		f.programBinary = get("glProgramBinary")
	}

	if glVer >= Ver(4, 1) {
		f.programParameteri = get("glProgramParameteri")
	}

	if glVer >= Ver(3, 2) || extensions.Has("GL_ARB_sampler_objects") {
		f.bindSampler = get("glBindSampler")
		f.deleteSamplers = get("glDeleteSamplers")
		f.genSamplers = get("glGenSamplers")
		f.samplerParameterf = get("glSamplerParameterf")
		f.samplerParameteri = get("glSamplerParameteri")
		f.samplerParameteriv = get("glSamplerParameteriv")
	}

	f.beginQuery = get("glBeginQuery")
	f.deleteQueries = get("glDeleteQueries")
	f.endQuery = get("glEndQuery")
	f.genQueries = get("glGenQueries")
	f.getQueryObjectuiv = get("glGetQueryObjectuiv")
	f.getQueryiv = get("glGetQueryiv")

	if glVer >= Ver(3, 3) || extensions.Has("GL_ARB_timer_query") {
		f.queryCounter = get("glQueryCounter")
	}

	switch {
	case glVer >= Ver(3, 3):
		f.getQueryObjecti64v = get("glGetQueryObjecti64v")
		f.getQueryObjectui64v = get("glGetQueryObjectui64v")
	case extensions.Has("GL_ARB_timer_query"):
		f.getQueryObjecti64v = get("glGetQueryObjecti64v")
		f.getQueryObjectui64v = get("glGetQueryObjectui64v")
	case extensions.Has("GL_EXT_timer_query"):
		f.getQueryObjecti64v = get("glGetQueryObjecti64vEXT")
		f.getQueryObjectui64v = get("glGetQueryObjectui64vEXT")
	}

	if glVer >= Ver(4, 3) || extensions.Has("GL_ARB_invalidate_subdata") {
		f.invalidateBufferData = get("glInvalidateBufferData")
		f.invalidateBufferSubData = get("glInvalidateBufferSubData")
		f.invalidateTexImage = get("glInvalidateTexImage")
		f.invalidateTexSubImage = get("glInvalidateTexSubImage")
	}

	if glVer >= Ver(4, 3) || extensions.Has("GL_ARB_invalidate_subdata") {
		f.invalidateFramebuffer = get("glInvalidateFramebuffer")
		f.invalidateSubFramebuffer = get("glInvalidateSubFramebuffer")
	}

	if glVer >= Ver(4, 3) || extensions.Has("GL_ARB_ES2_compatibility") {
		f.getShaderPrecisionFormat = get("glGetShaderPrecisionFormat")
	}

	if extensions.Has("GL_NV_fence") {
		f.deleteFences = get("glDeleteFencesNV")
		f.finishFence = get("glFinishFenceNV")
		f.genFences = get("glGenFencesNV")
		f.setFence = get("glSetFenceNV")
		f.testFence = get("glTestFenceNV")
	}

	f.initRegistered()
	intf.Standard = StandardGL
	intf.Extensions = extensions

	return intf
}
