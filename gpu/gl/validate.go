// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Interface validation for desktop GL only: since this package targets only StandardGL, requirement groups that would
// apply solely to ES/WebGL (discard-framebuffer, mapping sub-images, multisampled- render-to-texture, multisample
// resolve variants, bind-uniform-location, and multi-draw extensions specific to those backends) are omitted entirely.

package gl

// Validate checks that every function pointer required by the interface's version and advertised extensions was
// resolved. A GL context must be current, since the version is re-queried through GetString.
func (i *Interface) Validate() bool {
	if i.Standard != StandardGL {
		// This package never assembles GLES/WebGL interfaces, and StandardNone always fails here.
		return false
	}
	if !i.Extensions.IsInitialized() {
		return false
	}
	f := &i.Functions
	if f.getString == 0 {
		return false
	}
	glVer := GetVersionFromString(goString(rawCall(f.getString, VERSION)))
	if glVer == InvalidVersion {
		return false
	}
	ext := &i.Extensions

	// The unconditional block.
	if f.activeTexture == 0 ||
		f.attachShader == 0 ||
		f.bindAttribLocation == 0 ||
		f.bindBuffer == 0 ||
		f.bindTexture == 0 ||
		f.blendColor == 0 ||
		f.blendEquation == 0 ||
		f.blendFunc == 0 ||
		f.bufferData == 0 ||
		f.bufferSubData == 0 ||
		f.clear == 0 ||
		f.clearColor == 0 ||
		f.clearStencil == 0 ||
		f.colorMask == 0 ||
		f.compileShader == 0 ||
		f.compressedTexImage2D == 0 ||
		f.compressedTexSubImage2D == 0 ||
		f.copyTexSubImage2D == 0 ||
		f.createProgram == 0 ||
		f.createShader == 0 ||
		f.cullFace == 0 ||
		f.deleteBuffers == 0 ||
		f.deleteProgram == 0 ||
		f.deleteShader == 0 ||
		f.deleteTextures == 0 ||
		f.depthMask == 0 ||
		f.disable == 0 ||
		f.disableVertexAttribArray == 0 ||
		f.drawArrays == 0 ||
		f.drawElements == 0 ||
		f.enable == 0 ||
		f.enableVertexAttribArray == 0 ||
		f.finish == 0 ||
		f.flush == 0 ||
		f.frontFace == 0 ||
		f.genBuffers == 0 ||
		f.genTextures == 0 ||
		f.getBufferParameteriv == 0 ||
		f.getError == 0 ||
		f.getFloatv == 0 ||
		f.getIntegerv == 0 ||
		f.getProgramInfoLog == 0 ||
		f.getProgramiv == 0 ||
		f.getShaderInfoLog == 0 ||
		f.getShaderiv == 0 ||
		f.getString == 0 ||
		f.getUniformLocation == 0 ||
		f.isTexture == 0 ||
		f.lineWidth == 0 ||
		f.linkProgram == 0 ||
		f.pixelStorei == 0 ||
		f.readPixels == 0 ||
		f.scissor == 0 ||
		f.shaderSource == 0 ||
		f.stencilFunc == 0 ||
		f.stencilFuncSeparate == 0 ||
		f.stencilMask == 0 ||
		f.stencilMaskSeparate == 0 ||
		f.stencilOp == 0 ||
		f.stencilOpSeparate == 0 ||
		f.texImage2D == 0 ||
		f.texParameterf == 0 ||
		f.texParameterfv == 0 ||
		f.texParameteri == 0 ||
		f.texParameteriv == 0 ||
		f.texSubImage2D == 0 ||
		f.uniform1f == 0 ||
		f.uniform1fv == 0 ||
		f.uniform1i == 0 ||
		f.uniform1iv == 0 ||
		f.uniform2f == 0 ||
		f.uniform2fv == 0 ||
		f.uniform2i == 0 ||
		f.uniform2iv == 0 ||
		f.uniform3f == 0 ||
		f.uniform3fv == 0 ||
		f.uniform3i == 0 ||
		f.uniform3iv == 0 ||
		f.uniform4f == 0 ||
		f.uniform4fv == 0 ||
		f.uniform4i == 0 ||
		f.uniform4iv == 0 ||
		f.uniformMatrix2fv == 0 ||
		f.uniformMatrix3fv == 0 ||
		f.uniformMatrix4fv == 0 ||
		f.useProgram == 0 ||
		f.vertexAttrib1f == 0 ||
		f.vertexAttrib2fv == 0 ||
		f.vertexAttrib3fv == 0 ||
		f.vertexAttrib4fv == 0 ||
		f.vertexAttribPointer == 0 ||
		f.viewport == 0 {
		return false
	}

	if f.drawBuffer == 0 || f.polygonMode == 0 {
		return false
	}

	if glVer >= Ver(3, 0) {
		if f.getStringi == 0 {
			return false
		}
	}

	// GL 4.2+: all functions were marked optional or test_only.

	if glVer >= Ver(3, 0) || ext.Has("GL_ARB_vertex_array_object") || ext.Has("GL_APPLE_vertex_array_object") {
		if f.bindVertexArray == 0 || f.deleteVertexArrays == 0 || f.genVertexArrays == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 0) || ext.Has("GL_ARB_tessellation_shader") {
		if f.patchParameteri == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 0) {
		if f.bindFragDataLocation == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 3) || ext.Has("GL_ARB_blend_func_extended") {
		if f.bindFragDataLocationIndexed == 0 {
			return false
		}
	}

	if ext.Has("GL_KHR_blend_equation_advanced") || ext.Has("GL_NV_blend_equation_advanced") {
		if f.blendBarrier == 0 {
			return false
		}
	}

	// GL 4.4+ / GL_ARB_clear_texture: all functions were marked optional or test_only.

	if glVer >= Ver(3, 1) || ext.Has("GL_ARB_draw_instanced") || ext.Has("GL_EXT_draw_instanced") {
		if f.drawArraysInstanced == 0 || f.drawElementsInstanced == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 2) || ext.Has("GL_ARB_base_instance") {
		if f.drawArraysInstancedBaseInstance == 0 || f.drawElementsInstancedBaseVertexBaseInstance == 0 {
			return false
		}
	}

	if f.drawBuffers == 0 || f.readBuffer == 0 {
		return false
	}

	if glVer >= Ver(4, 0) || ext.Has("GL_ARB_draw_indirect") {
		if f.drawArraysIndirect == 0 || f.drawElementsIndirect == 0 {
			return false
		}
	}

	if f.drawRangeElements == 0 {
		return false
	}

	if glVer >= Ver(3, 2) || ext.Has("GL_ARB_texture_multisample") {
		if f.getMultisamplefv == 0 {
			return false
		}
	}

	if f.getTexLevelParameteriv == 0 {
		return false
	}

	if glVer >= Ver(4, 3) || ext.Has("GL_ARB_multi_draw_indirect") {
		if f.multiDrawArraysIndirect == 0 || f.multiDrawElementsIndirect == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 1) {
		if f.texBuffer == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 3) {
		if f.texBufferRange == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 2) || ext.Has("GL_ARB_texture_storage") || ext.Has("GL_EXT_texture_storage") {
		if f.texStorage2D == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 5) || ext.Has("GL_ARB_texture_barrier") || ext.Has("GL_NV_texture_barrier") {
		if f.textureBarrier == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 2) || ext.Has("GL_ARB_instanced_arrays") {
		if f.vertexAttribDivisor == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 0) {
		if f.vertexAttribIPointer == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 0) || ext.Has("GL_ARB_framebuffer_object") || ext.Has("GL_EXT_framebuffer_object") {
		if f.bindFramebuffer == 0 ||
			f.bindRenderbuffer == 0 ||
			f.checkFramebufferStatus == 0 ||
			f.deleteFramebuffers == 0 ||
			f.deleteRenderbuffers == 0 ||
			f.framebufferRenderbuffer == 0 ||
			f.framebufferTexture2D == 0 ||
			f.genFramebuffers == 0 ||
			f.genRenderbuffers == 0 ||
			f.generateMipmap == 0 ||
			f.getFramebufferAttachmentParameteriv == 0 ||
			f.getRenderbufferParameteriv == 0 ||
			f.renderbufferStorage == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 0) || ext.Has("GL_ARB_framebuffer_object") || ext.Has("GL_EXT_framebuffer_blit") {
		if f.blitFramebuffer == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 0) || ext.Has("GL_ARB_framebuffer_object") || ext.Has("GL_EXT_framebuffer_multisample") {
		if f.renderbufferStorageMultisample == 0 {
			return false
		}
	}

	if f.mapBuffer == 0 {
		return false
	}

	if f.unmapBuffer == 0 {
		return false
	}

	if glVer >= Ver(3, 0) || ext.Has("GL_ARB_map_buffer_range") {
		if f.flushMappedBufferRange == 0 || f.mapBufferRange == 0 {
			return false
		}
	}

	if ext.Has("GL_EXT_debug_marker") {
		if f.insertEventMarker == 0 || f.popGroupMarker == 0 || f.pushGroupMarker == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 1) || ext.Has("GL_ARB_copy_buffer") {
		if f.copyBufferSubData == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 3) || ext.Has("GL_KHR_debug") {
		if f.debugMessageCallback == 0 ||
			f.debugMessageControl == 0 ||
			f.debugMessageInsert == 0 ||
			f.getDebugMessageLog == 0 ||
			f.objectLabel == 0 ||
			f.popDebugGroup == 0 ||
			f.pushDebugGroup == 0 {
			return false
		}
	}

	if ext.Has("GL_EXT_window_rectangles") {
		if f.windowRectangles == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 2) || ext.Has("GL_ARB_sync") {
		if f.clientWaitSync == 0 ||
			f.deleteSync == 0 ||
			f.fenceSync == 0 ||
			f.isSync == 0 ||
			f.waitSync == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 2) || ext.Has("GL_ARB_internalformat_query") {
		if f.getInternalformativ == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 1) {
		if f.getProgramBinary == 0 || f.programBinary == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 1) {
		if f.programParameteri == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 2) || ext.Has("GL_ARB_sampler_objects") {
		if f.bindSampler == 0 ||
			f.deleteSamplers == 0 ||
			f.genSamplers == 0 ||
			f.samplerParameterf == 0 ||
			f.samplerParameteri == 0 ||
			f.samplerParameteriv == 0 {
			return false
		}
	}

	if f.beginQuery == 0 ||
		f.deleteQueries == 0 ||
		f.endQuery == 0 ||
		f.genQueries == 0 ||
		f.getQueryObjectuiv == 0 ||
		f.getQueryiv == 0 {
		return false
	}

	if glVer >= Ver(3, 3) || ext.Has("GL_ARB_timer_query") {
		if f.queryCounter == 0 {
			return false
		}
	}

	if glVer >= Ver(3, 3) || ext.Has("GL_ARB_timer_query") || ext.Has("GL_EXT_timer_query") {
		if f.getQueryObjecti64v == 0 || f.getQueryObjectui64v == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 3) || ext.Has("GL_ARB_invalidate_subdata") {
		if f.invalidateBufferData == 0 ||
			f.invalidateBufferSubData == 0 ||
			f.invalidateTexImage == 0 ||
			f.invalidateTexSubImage == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 3) || ext.Has("GL_ARB_invalidate_subdata") {
		if f.invalidateFramebuffer == 0 || f.invalidateSubFramebuffer == 0 {
			return false
		}
	}

	if glVer >= Ver(4, 3) || ext.Has("GL_ARB_ES2_compatibility") {
		if f.getShaderPrecisionFormat == 0 {
			return false
		}
	}

	if ext.Has("GL_NV_fence") {
		if f.deleteFences == 0 ||
			f.finishFence == 0 ||
			f.genFences == 0 ||
			f.setFence == 0 ||
			f.testFence == 0 {
			return false
		}
	}

	return true
}
