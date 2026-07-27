// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Renderbuffer-backed stencil and MSAA color attachments. Under the desktop trim only the standard renderbuffer-storage
// lane exists (the ES/Apple/IMG multisample variants are unreachable), and memoryless attachments are always
// unsupported (a Vulkan/tiler concept).

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// AttachmentUsage is a bitmask of how an Attachment is used.
type AttachmentUsage uint8

// AttachmentUsage values.
const (
	AttachmentUsageStencil AttachmentUsage = 1 << 0
	AttachmentUsageColor   AttachmentUsage = 1 << 1
	AttachmentUsageTexture AttachmentUsage = 1 << 2
)

// Attachment is a single renderbuffer-backed GPU allocation with usage flags.
type Attachment struct {
	g *Gpu
	gpu.ResourceBase
	sampleCnt      int
	dims           geom.ISize
	format         Format
	renderbufferID uint32 // may be zero for external stencil buffers on wrapped FBOs
	usage          AttachmentUsage
}

// Dimensions returns the attachment's pixel dimensions.
func (a *Attachment) Dimensions() geom.ISize { return a.dims }

// SupportedUsages returns the usage flags this attachment was created with.
func (a *Attachment) SupportedUsages() AttachmentUsage { return a.usage }

// NumSamples returns the MSAA sample count (1 for non-MSAA).
func (a *Attachment) NumSamples() int { return a.sampleCnt }

// GLFormat returns the attachment's GL internal format.
func (a *Attachment) GLFormat() Format { return a.format }

// RenderbufferID returns the underlying GL renderbuffer object name.
func (a *Attachment) RenderbufferID() uint32 { return a.renderbufferID }

// ResourceType implements gpu.Resource.
func (a *Attachment) ResourceType() string {
	if a.usage == AttachmentUsageStencil {
		return "StencilAttachment"
	}
	return "Surface"
}

// OnGpuMemorySize implements gpu.Resource: only non-texture, non-memoryless attachments report their own size.
func (a *Attachment) OnGpuMemorySize() uint64 {
	if a.usage&AttachmentUsageTexture == 0 {
		return uint64(a.dims.Width) * uint64(a.dims.Height) *
			uint64(a.format.BytesPerBlock()) * uint64(a.sampleCnt)
	}
	return 0
}

// ComputeScratchKey implements gpu.Resource: stencil attachments are shared through unique keys instead, and
// texture-usage attachments are tracked by their owning texture, so only color attachments get scratch keys.
func (a *Attachment) ComputeScratchKey(key *gpu.ScratchKey) {
	if a.usage&AttachmentUsageStencil == 0 && a.usage&AttachmentUsageTexture == 0 {
		ComputeAttachmentScratchKey(a.g.glCaps(), a.format, a.dims, a.usage, a.sampleCnt, key)
	}
}

// OnRelease implements gpu.Resource.
func (a *Attachment) OnRelease() {
	if a.renderbufferID != 0 {
		id := a.renderbufferID
		a.g.fns().DeleteRenderbuffers(1, &id)
		a.renderbufferID = 0
	}
}

// OnAbandon implements gpu.Resource.
func (a *Attachment) OnAbandon() { a.renderbufferID = 0 }

var (
	attachmentScratchResourceType = gpu.GenerateScratchKeyResourceType()
	attachmentUniqueKeyDomain     = gpu.GenerateUniqueKeyDomain()
)

// attachmentKeyData builds the shared scratch/unique key layout (the protected and memoryless bits are always zero
// under the trim but stay in the layout).
func attachmentKeyData(caps *Caps, format Format, dims geom.ISize, usage AttachmentUsage, sampleCnt int, s []uint32) {
	if dims.IsEmpty() {
		panic("empty attachment dimensions")
	}
	formatKey := caps.ComputeFormatKey(format)
	s[0] = uint32(dims.Width)
	s[1] = uint32(dims.Height)
	s[2] = uint32(formatKey)
	s[3] = uint32(formatKey >> 32)
	s[4] = 0<<0 | 0<<1 | uint32(usage)<<2 | uint32(sampleCnt)<<10
}

// ComputeAttachmentScratchKey computes the scratch key for an attachment with the given dimensions, usage, and sample
// count.
func ComputeAttachmentScratchKey(caps *Caps, format Format, dims geom.ISize, usage AttachmentUsage, sampleCnt int, key *gpu.ScratchKey) {
	b := gpu.ScratchKeyBuilder(key, attachmentScratchResourceType, 5)
	attachmentKeyData(caps, format, dims, usage, sampleCnt, b.Slice())
	b.Finish()
}

// ComputeSharedAttachmentUniqueKey computes the key under which stencil attachments of the same dimensions, usage, and
// sample count are shared between render targets. The key carries no render-target identity, so a render pass can never
// assume anything about the contents an attachment reaches it with — see OpsTask.OnExecute's kUserBitsCleared case.
func ComputeSharedAttachmentUniqueKey(caps *Caps, format Format, dims geom.ISize, usage AttachmentUsage, sampleCnt int, key *gpu.UniqueKey) {
	b := gpu.UniqueKeyBuilder(key, attachmentUniqueKeyDomain, 5, "SharedAttachment")
	attachmentKeyData(caps, format, dims, usage, sampleCnt, b.Slice())
	b.Finish()
}

// NewStencilAttachment creates a renderbuffer-backed stencil attachment.
func NewStencilAttachment(g *Gpu, dims geom.ISize, sampleCnt int, format Format) *Attachment {
	var rbID uint32
	g.fns().GenRenderbuffers(1, &rbID)
	if rbID == 0 {
		return nil
	}
	g.fns().BindRenderbuffer(RENDERBUFFER, rbID)
	glFmt := format.ToEnum()
	// The "if" avoids calling the multisample version on a GL without an MSAA extension.
	if sampleCnt > 1 {
		if !g.renderbufferStorageMSAA(sampleCnt, glFmt, dims.Width, dims.Height) {
			g.fns().DeleteRenderbuffers(1, &rbID)
			return nil
		}
	} else if err := g.allocCall(func(f *Functions) {
		f.RenderbufferStorage(RENDERBUFFER, glFmt, dims.Width, dims.Height)
	}); err != NO_ERROR {
		g.fns().DeleteRenderbuffers(1, &rbID)
		return nil
	}
	return newAttachment(g, rbID, dims, AttachmentUsageStencil, sampleCnt, format,
		"GLAttachmentMakeStencil")
}

// NewMSAAAttachment creates a renderbuffer-backed MSAA color attachment.
func NewMSAAAttachment(g *Gpu, dims geom.ISize, sampleCnt int, format Format) *Attachment {
	var rbID uint32
	g.fns().GenRenderbuffers(1, &rbID)
	if rbID == 0 {
		return nil
	}
	g.fns().BindRenderbuffer(RENDERBUFFER, rbID)
	glFmt := g.glCaps().RenderbufferInternalFormat(format)
	if !g.renderbufferStorageMSAA(sampleCnt, glFmt, dims.Width, dims.Height) {
		g.fns().DeleteRenderbuffers(1, &rbID)
		return nil
	}
	return newAttachment(g, rbID, dims, AttachmentUsageColor, sampleCnt, format,
		"GLAttachmentMakeMSAA")
}

// makeWrappedRenderBufferAttachment wraps an externally-owned renderbuffer. Despite the name it registers as an owned,
// budgeted resource (the zero renderbuffer ID keeps it from deleting anything).
func makeWrappedRenderBufferAttachment(g *Gpu, renderbufferID uint32, dims geom.ISize, usage AttachmentUsage, sampleCnt int, format Format) *Attachment {
	return newAttachment(g, renderbufferID, dims, usage, sampleCnt, format,
		"MakeWrappedRenderBuffer")
}

func newAttachment(g *Gpu, renderbufferID uint32, dims geom.ISize, usage AttachmentUsage, sampleCnt int, format Format, label string) *Attachment {
	if usage != AttachmentUsageStencil && usage != AttachmentUsageColor {
		panic("unexpected attachment usage")
	}
	a := &Attachment{
		g:              g,
		dims:           dims,
		usage:          usage,
		sampleCnt:      sampleCnt,
		format:         format,
		renderbufferID: renderbufferID,
	}
	a.RegisterWithCache(g.ResourceCache(), a, gpu.BudgetedYes, label)
	return a
}
