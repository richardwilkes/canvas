// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU's pixel readback and transfer-buffer paths: reading pixels back from a surface into CPU memory or a transfer
// buffer, uploading pixels from a transfer buffer into a texture, and copying between transfer buffers directly on the
// GPU.

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// readOrTransferPixelsFrom reads pixels from a surface, either directly into CPU memory or into a bound pack buffer.
// offsetOrPtr is either a CPU destination pointer (passed as uintptr(unsafe.Pointer(&dst[0]))) or a plain PBO byte
// offset when a pack buffer is bound. It is //go:uintptrescapes so a CPU destination handed in as a converted pointer
// stays alive and pinned across the nested glReadPixels call; a plain offset is a no-op for the directive.
//
//go:uintptrescapes
func (g *Gpu) readOrTransferPixelsFrom(surface *Surface, rect geom.IRect, surfaceColorType, dstColorType gpu.ColorType, offsetOrPtr uintptr, rowWidthInPixels int32) bool {
	format := surface.Format()
	renderTarget := surface.AsRenderTarget()
	if renderTarget == nil && !g.glCaps().IsFormatRenderable(format, 1) {
		return false
	}
	externalFormat, externalType := g.glCaps().ReadPixelsFormat(format, surfaceColorType,
		dstColorType)
	if externalFormat == 0 || externalType == 0 {
		return false
	}

	f := g.fns()
	if renderTarget != nil {
		// Always bind the single sample FBO since we can't read pixels from an MSAA framebuffer.
		const useMultisampleFBO = false
		if renderTarget.NumSamples() > 1 && renderTarget.IsFBO0(useMultisampleFBO) {
			return false
		}
		g.flushRenderTarget(renderTarget, useMultisampleFBO)
	} else {
		// Use a temporary FBO.
		g.bindSurfaceFBOForPixelOps(surface, 0, FRAMEBUFFER, tempFBOSrc)
		g.hwBoundRenderTargetUniqueID = 0
	}

	// Determine if GL can read using the passed rowBytes or if we need a scratch buffer.
	if rowWidthInPixels != rect.Width() {
		if !g.Caps().ReadPixelsRowBytesSupport {
			panic("row bytes without support")
		}
		f.PixelStorei(PACK_ROW_LENGTH, rowWidthInPixels)
	}
	f.PixelStorei(PACK_ALIGNMENT, 1)

	f.ReadPixels(rect.Left, rect.Top, rect.Width(), rect.Height(), externalFormat, externalType,
		offsetOrPtr)

	if rowWidthInPixels != rect.Width() {
		f.PixelStorei(PACK_ROW_LENGTH, 0)
	}

	if renderTarget == nil {
		g.unbindSurfaceFBOForPixelOps(surface, 0, FRAMEBUFFER)
	}
	return true
}

// ReadPixels reads a surface's pixels back into CPU memory. rowBytes expresses the destination stride; it must be a
// whole multiple of the pixel size.
func (g *Gpu) ReadPixels(surface *Surface, rect geom.IRect, surfaceColorType, dstColorType gpu.ColorType, buffer []byte, rowBytes int) bool {
	if surface.FramebufferOnly() {
		panic("read from framebuffer-only surface")
	}
	if !geom.IRectSize(surface.Dimensions()).ContainsRect(rect) {
		return false
	}
	bytesPerPixel := dstColorType.BytesPerPixel()
	minRowBytes := bytesPerPixel * int(rect.Width())
	if !g.Caps().ReadPixelsRowBytesSupport {
		if rowBytes != minRowBytes {
			return false
		}
	} else {
		if rowBytes < minRowBytes {
			return false
		}
		if rowBytes%bytesPerPixel != 0 {
			return false
		}
	}
	if len(buffer) < rowBytes*(int(rect.Height())-1)+minRowBytes {
		return false
	}
	g.handleDirtyContext()

	// GL_PACK_ROW_LENGTH is in terms of pixels not bytes.
	rowPixelWidth := rect.Width()
	if rowBytes != minRowBytes {
		rowPixelWidth = int32(rowBytes / bytesPerPixel)
	}
	g.unbindXferBuffer(gpu.BufferTypeXferGpuToCpu)
	return g.readOrTransferPixelsFrom(surface, rect, surfaceColorType, dstColorType,
		uintptr(unsafe.Pointer(&buffer[0])), rowPixelWidth)
}

// TransferPixelsTo uploads pixels from a transfer buffer into a texture.
func (g *Gpu) TransferPixelsTo(texture *Texture, rect geom.IRect, textureColorType, bufferColorType gpu.ColorType, transferBuffer *Buffer, offset, rowBytes uint64) bool {
	if transferBuffer.IntendedType() != gpu.BufferTypeXferCpuToGpu {
		panic("wrong transfer buffer type")
	}
	if texture.Surface().ReadOnly() {
		return false
	}
	if !geom.IRectSize(texture.Surface().Dimensions()).ContainsRect(rect) {
		return false
	}
	bpp := bufferColorType.BytesPerPixel()
	if g.Caps().WritePixelsRowBytesSupport {
		if rowBytes < uint64(bpp)*uint64(rect.Width()) || rowBytes%uint64(bpp) != 0 {
			return false
		}
	} else if rowBytes != uint64(bpp)*uint64(rect.Width()) {
		return false
	}
	g.handleDirtyContext()

	// Bind the destination texture and the source transfer buffer.
	g.bindTextureToScratchUnit(texture.Target(), texture.TextureID())

	if transferBuffer.IsMapped() {
		panic("transfer buffer is mapped")
	}
	g.bindBuffer(gpu.BufferTypeXferCpuToGpu, transferBuffer)

	trimRowBytes := uint64(rect.Width()) * uint64(bpp)
	if transferBuffer.Size() < offset+rowBytes*uint64(rect.Height()-1)+trimRowBytes {
		panic("transfer buffer too small")
	}

	f := g.fns()
	restoreGLRowLength := false
	if trimRowBytes != rowBytes {
		// We should have checked for this support already.
		if !g.Caps().TransferPixelsToRowBytesSupport {
			panic("row bytes without support")
		}
		f.PixelStorei(UNPACK_ROW_LENGTH, int32(rowBytes/uint64(bpp)))
		restoreGLRowLength = true
	}

	// External format and type come from the upload data.
	externalFormat, externalType := g.glCaps().TexSubImageExternalFormatAndType(
		texture.GLFormat(), textureColorType, bufferColorType,
	)
	if externalFormat == 0 || externalType == 0 {
		return false
	}

	f.PixelStorei(UNPACK_ALIGNMENT, 1)
	f.TexSubImage2D(texture.Target(), 0, rect.Left, rect.Top, rect.Width(), rect.Height(),
		externalFormat, externalType, uintptr(offset))

	if restoreGLRowLength {
		f.PixelStorei(UNPACK_ROW_LENGTH, 0)
	}
	g.didWriteToSurface(texture.Surface())
	return true
}

// TransferPixelsFrom reads a surface's pixels back into a transfer buffer.
func (g *Gpu) TransferPixelsFrom(surface *Surface, rect geom.IRect, surfaceColorType, bufferColorType gpu.ColorType, transferBuffer *Buffer, offset uint64) bool {
	if transferBuffer.IntendedType() != gpu.BufferTypeXferGpuToCpu {
		panic("wrong transfer buffer type")
	}
	if !geom.IRectSize(surface.Dimensions()).ContainsRect(rect) {
		return false
	}
	g.handleDirtyContext()

	// Bounds-check the destination transfer buffer.
	if transferBuffer.Size() < offset+
		uint64(rect.Width())*uint64(rect.Height())*uint64(bufferColorType.BytesPerPixel()) {
		panic("transfer buffer too small")
	}
	g.bindBuffer(gpu.BufferTypeXferGpuToCpu, transferBuffer)
	return g.readOrTransferPixelsFrom(surface, rect, surfaceColorType, bufferColorType,
		uintptr(offset), rect.Width())
}

// TransferFromBufferToBuffer copies a range of bytes directly between two GPU buffers.
func (g *Gpu) TransferFromBufferToBuffer(src *Buffer, srcOffset uint64, dst *Buffer, dstOffset, size uint64) bool {
	if src.IsMapped() || dst.IsMapped() {
		panic("transfer with mapped buffer")
	}
	if src.IntendedType() != gpu.BufferTypeXferCpuToGpu ||
		dst.IntendedType() == gpu.BufferTypeXferCpuToGpu {
		panic("wrong buffer types for transfer")
	}
	if srcOffset+size > src.Size() || dstOffset+size > dst.Size() {
		panic("transfer out of range")
	}
	g.handleDirtyContext()

	// If bindBuffer used something other than BufferType to indicate the binding target we could use the COPY_READ
	// and COPY_WRITE targets, but CopyBufferSubData works with all targets.
	g.bindBuffer(gpu.BufferTypeXferCpuToGpu, src)
	g.bindBuffer(gpu.BufferTypeXferGpuToCpu, dst)
	g.fns().CopyBufferSubData(PIXEL_UNPACK_BUFFER, PIXEL_PACK_BUFFER, int(srcOffset),
		int(dstOffset), int(size))
	return true
}
