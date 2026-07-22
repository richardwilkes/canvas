// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GL surface resource hierarchy: a backing GPU surface that may be a texture, a render target, or both, expressed
// as one Surface resource with optional Texture and RenderTarget facets, each holding a back-pointer to the Surface.
// All per-facet state and behavior lives on the facets, so a Surface with both facets behaves as one unit (one cache
// entry, one ref count, combined memory size, release order RT-then-texture).
//
// Trims: external textures and compressed formats are out of scope (desktop trim), and protected content is not
// implemented. The DMSAA dynamic MSAA attachment (EnsureDynamicMSAAAttachment) is live; only bindInternal's
// render-to-texture FBO reconfiguration stays out, because it requires the single-sample and multisample FBO ids to be
// equal, which only the ES render-to-texture extensions (msaaResolvesAutomatically) produce — never desktop GL. Release
// procs are not ported.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// UnresolvableFBOID is the sentinel FBO id meaning "no framebuffer / not multisampled."
const UnresolvableFBOID = 0

// Surface is a 2D GPU allocation that may be a texture, a render target, or both. It is a cache-managed resource.
type Surface struct {
	g   *Gpu
	tex *Texture
	rt  *RenderTarget
	gpu.ResourceBase
	dims  geom.ISize
	flags gpu.SurfaceFlags
}

// Dimensions returns the surface's dimensions.
func (s *Surface) Dimensions() geom.ISize { return s.dims }

// Width returns the surface's width.
func (s *Surface) Width() int32 { return s.dims.Width }

// Height returns the surface's height.
func (s *Surface) Height() int32 { return s.dims.Height }

// Format returns the surface's GL format.
func (s *Surface) Format() Format {
	if s.tex != nil {
		return s.tex.format
	}
	return s.rt.format
}

// AsTexture returns the texture facet; nil when the surface is not a texture.
func (s *Surface) AsTexture() *Texture { return s.tex }

// AsRenderTarget returns the render-target facet; nil when the surface is not a render target.
func (s *Surface) AsRenderTarget() *RenderTarget { return s.rt }

// Flags returns the surface's internal flags.
func (s *Surface) Flags() gpu.SurfaceFlags { return s.flags }

// ReadOnly reports whether the surface is read-only.
func (s *Surface) ReadOnly() bool { return s.flags&gpu.SurfaceFlagReadOnly != 0 }

// FramebufferOnly reports whether the surface can only be used as a framebuffer, not sampled.
func (s *Surface) FramebufferOnly() bool { return s.flags&gpu.SurfaceFlagFramebufferOnly != 0 }

// GLRTFBOIDIs0 reports whether this surface's render target wraps GL's default framebuffer (FBO 0).
func (s *Surface) GLRTFBOIDIs0() bool { return s.flags&gpu.SurfaceFlagGLRTFBOIDIs0 != 0 }

// RequiresManualMSAAResolve reports whether the surface needs an explicit MSAA resolve.
func (s *Surface) RequiresManualMSAAResolve() bool {
	return s.flags&gpu.SurfaceFlagRequiresManualMSAAResolve != 0
}

// SetRequiresManualMSAAResolve marks the surface as needing an explicit MSAA resolve.
func (s *Surface) SetRequiresManualMSAAResolve() {
	s.flags |= gpu.SurfaceFlagRequiresManualMSAAResolve
}

// SetFramebufferOnly marks the surface as usable only as a framebuffer, not sampled.
func (s *Surface) SetFramebufferOnly() { s.flags |= gpu.SurfaceFlagFramebufferOnly }

func (s *Surface) setReadOnly() { s.flags |= gpu.SurfaceFlagReadOnly }

// ResourceType implements gpu.Resource.
func (s *Surface) ResourceType() string { return "Surface" }

// OnGpuMemorySize implements gpu.Resource: a plain texture uses one color sample and the mip factor, a plain render
// target uses totalMemorySamplesPerPixel without mips, and a texture-render-target uses totalMemorySamplesPerPixel with
// the mip factor.
func (s *Surface) OnGpuMemorySize() uint64 {
	samples := 1
	mipmapped := gpu.MipmappedNo
	if s.rt != nil {
		samples = s.rt.totalMemorySamplesPerPixel
	}
	if s.tex != nil {
		mipmapped = s.tex.Mipmapped()
	}
	return ComputeSurfaceSize(s.Format(), s.dims, samples, mipmapped, false)
}

// ComputeScratchKey implements gpu.Resource: only textures are recycled as scratch.
func (s *Surface) ComputeScratchKey(key *gpu.ScratchKey) {
	if s.tex == nil {
		return
	}
	sampleCount := 1
	renderable := gpu.RenderableNo
	if s.rt != nil {
		sampleCount = s.rt.sampleCnt
		renderable = gpu.RenderableYes
	}
	ComputeTextureScratchKey(s.g.glCaps(), s.Format(), s.dims, renderable, sampleCount,
		s.tex.Mipmapped(), key)
}

// OnRelease implements gpu.Resource: the render target facet releases first, then the texture facet.
func (s *Surface) OnRelease() {
	if s.rt != nil {
		s.rt.release()
	}
	if s.tex != nil {
		s.tex.release()
	}
}

// OnAbandon implements gpu.Resource.
func (s *Surface) OnAbandon() {
	if s.rt != nil {
		s.rt.abandon()
	}
	if s.tex != nil {
		s.tex.abandon()
	}
}

// OnSetLabel implements gpu.Resource: sets a KHR_debug object label.
func (s *Surface) OnSetLabel() {
	if s.tex != nil && s.tex.id != 0 && s.Label() != "" && s.g.glCaps().DebugSupport() {
		s.g.fns().ObjectLabel(TEXTURE, s.tex.id, -1, cString("_Skia_"+s.Label()))
	}
}

// ComputeSurfaceSize estimates a surface's GPU memory footprint for the uncompressed desktop format set. binSize rounds
// the dimensions with the approx-size binning, matching how approx-fit proxies estimate the footprint of their eventual
// backing store.
func ComputeSurfaceSize(format Format, dims geom.ISize, colorSamplesPerPixel int, mipmapped gpu.Mipmapped, binSize bool) uint64 {
	if binSize {
		dims = gpu.GetApproxSize(dims)
	}
	colorSize := uint64(dims.Width) * uint64(dims.Height) * uint64(format.BytesPerBlock())
	finalSize := uint64(colorSamplesPerPixel) * colorSize
	if mipmapped == gpu.MipmappedYes {
		finalSize += colorSize / 3
	}
	return finalSize
}

var textureScratchResourceType = gpu.GenerateScratchKeyResourceType()

// ComputeTextureScratchKey derives the scratch-resource key for a recyclable texture matching the given description.
// The protected bit is always zero under the desktop trim but stays in the layout.
func ComputeTextureScratchKey(caps *Caps, format Format, dims geom.ISize, renderable gpu.Renderable, sampleCnt int, mipmapped gpu.Mipmapped, key *gpu.ScratchKey) {
	if dims.IsEmpty() || sampleCnt < 1 || (sampleCnt != 1 && renderable != gpu.RenderableYes) {
		panic("invalid texture scratch key inputs")
	}
	formatKey := caps.ComputeFormatKey(format)
	b := gpu.ScratchKeyBuilder(key, textureScratchResourceType, 5)
	s := b.Slice()
	s[0] = uint32(dims.Width)
	s[1] = uint32(dims.Height)
	s[2] = uint32(formatKey)
	s[3] = uint32(formatKey >> 32)
	var mip, rend uint32
	if mipmapped {
		mip = 1
	}
	if renderable {
		rend = 1
	}
	s[4] = mip<<0 | 0<<1 | rend<<2 | uint32(sampleCnt)<<3
	b.Finish()
}

// TextureDesc describes an already-created GL texture object to wrap in a Texture facet.
type TextureDesc struct {
	Size      geom.ISize
	Target    uint32 // GL texture target; only TEXTURE_2D and TEXTURE_RECTANGLE under the trim
	ID        uint32
	Format    Format
	Ownership gpu.Ownership
}

// Texture is the texture facet of a Surface.
type Texture struct {
	surf                *Surface
	params              *TextureParameters
	id                  uint32
	format              Format
	textureType         gpu.TextureType
	mipmapStatus        gpu.MipmapStatus
	maxMipmapLevel      int
	ownership           gpu.Ownership
	baseLevelBoundToFBO bool
}

// Surface returns the owning surface resource.
func (t *Texture) Surface() *Surface { return t.surf }

// TextureID returns the underlying GL texture object name.
func (t *Texture) TextureID() uint32 { return t.id }

// Target returns the GL texture target (TEXTURE_2D or TEXTURE_RECTANGLE).
func (t *Texture) Target() uint32 {
	switch t.textureType {
	case gpu.TextureType2D:
		return TEXTURE_2D
	case gpu.TextureTypeRectangle:
		return TEXTURE_RECTANGLE
	default:
		panic("unexpected texture type")
	}
}

// TextureTypeFromTarget maps a GL texture target to its TextureType (desktop targets only).
func TextureTypeFromTarget(target uint32) gpu.TextureType {
	switch target {
	case TEXTURE_2D:
		return gpu.TextureType2D
	case TEXTURE_RECTANGLE:
		return gpu.TextureTypeRectangle
	default:
		panic("unexpected texture target")
	}
}

// GLFormat returns the texture's GL format.
func (t *Texture) GLFormat() Format { return t.format }

// TextureType returns the texture's type.
func (t *Texture) TextureType() gpu.TextureType { return t.textureType }

// Parameters returns the texture's cached sampler parameters.
func (t *Texture) Parameters() *TextureParameters { return t.params }

// TextureParamsModified invalidates the cached sampler parameters, forcing them to be reapplied.
func (t *Texture) TextureParamsModified() { t.params.Invalidate() }

// Mipmapped reports whether the texture has mip levels allocated.
func (t *Texture) Mipmapped() gpu.Mipmapped {
	return t.mipmapStatus != gpu.MipmapStatusNotAllocated
}

// MipmapsAreDirty reports whether the texture's mip levels need regenerating.
func (t *Texture) MipmapsAreDirty() bool { return t.mipmapStatus != gpu.MipmapStatusValid }

// MarkMipmapsDirty flags the texture's mip levels as needing regeneration.
func (t *Texture) MarkMipmapsDirty() {
	if t.mipmapStatus == gpu.MipmapStatusValid {
		t.mipmapStatus = gpu.MipmapStatusDirty
	}
}

// MarkMipmapsClean flags the texture's mip levels as up to date.
func (t *Texture) MarkMipmapsClean() {
	if t.mipmapStatus == gpu.MipmapStatusNotAllocated {
		panic("mipmaps were never allocated")
	}
	t.mipmapStatus = gpu.MipmapStatusValid
}

// MipmapStatus returns the texture's current mip-level validity state.
func (t *Texture) MipmapStatus() gpu.MipmapStatus { return t.mipmapStatus }

// MaxMipmapLevel returns the highest mip level index the texture has allocated.
func (t *Texture) MaxMipmapLevel() int { return t.maxMipmapLevel }

// HasBaseLevelBeenBoundToFBO reports whether the texture's base level has ever been bound to a framebuffer for
// rendering.
func (t *Texture) HasBaseLevelBeenBoundToFBO() bool { return t.baseLevelBoundToFBO }

// BaseLevelWasBoundToFBO records that the texture's base level has been bound to a framebuffer.
func (t *Texture) BaseLevelWasBoundToFBO() { t.baseLevelBoundToFBO = true }

// release deletes the underlying GL texture object, unless it's borrowed.
func (t *Texture) release() {
	if t.id != 0 {
		if t.ownership != gpu.OwnershipBorrowed {
			id := t.id
			t.surf.g.fns().DeleteTextures(1, &id)
		}
		t.id = 0
	}
}

// abandon forgets the underlying GL texture object without deleting it (the context is gone).
func (t *Texture) abandon() { t.id = 0 }

func newTextureFacet(s *Surface, desc *TextureDesc, params *TextureParameters, mipmapStatus gpu.MipmapStatus) *Texture {
	if desc.ID == 0 || desc.Format == FormatUnknown {
		panic("invalid texture desc")
	}
	t := &Texture{
		surf:         s,
		params:       params,
		id:           desc.ID,
		format:       desc.Format,
		textureType:  TextureTypeFromTarget(desc.Target),
		mipmapStatus: mipmapStatus,
		ownership:    desc.Ownership,
	}
	if mipmapStatus != gpu.MipmapStatusNotAllocated {
		t.maxMipmapLevel = gpu.ComputeMipLevelCount(s.dims.Width, s.dims.Height)
	}
	return t
}

// NewTexture creates a texture-facet Surface and registers it with the cache as budgeted (or not).
func NewTexture(g *Gpu, budgeted gpu.Budgeted, desc *TextureDesc, mipmapStatus gpu.MipmapStatus, label string) *Surface {
	s := &Surface{g: g, dims: desc.Size}
	s.tex = newTextureFacet(s, desc, NewTextureParameters(), mipmapStatus)
	s.RegisterWithCache(g.ResourceCache(), s, budgeted, label)
	return s
}

// MakeWrappedTexture wraps an already-created GL texture object as a texture-facet Surface.
func MakeWrappedTexture(g *Gpu, mipmapStatus gpu.MipmapStatus, desc *TextureDesc, params *TextureParameters, cacheable gpu.WrapCacheable, ioType gpu.IOType, label string) *Surface {
	s := &Surface{g: g, dims: desc.Size}
	s.tex = newTextureFacet(s, desc, params, mipmapStatus)
	if ioType == gpu.IOTypeRead {
		s.setReadOnly()
	}
	s.RegisterWithCacheWrapped(g.ResourceCache(), s, cacheable, label)
	return s
}

// RenderTargetIDs describes the already-created GL framebuffer/renderbuffer objects to wrap in a RenderTarget facet.
type RenderTargetIDs struct {
	MultisampleFBOID           uint32
	SingleSampleFBOID          uint32
	MSColorRenderbufferID      uint32
	RTFBOOwnership             gpu.Ownership
	TotalMemorySamplesPerPixel int
}

// ResolveDirection selects which way an MSAA resolve blit runs.
type ResolveDirection bool

// ResolveDirection values.
const (
	// ResolveSingleToMSAA requires glCaps.canResolveSingleToMSAA().
	ResolveSingleToMSAA ResolveDirection = false
	// ResolveMSAAToSingle blits the multisampled FBO into the single sample one.
	ResolveMSAAToSingle ResolveDirection = true
)

// RenderTarget is the render-target facet of a Surface.
type RenderTarget struct {
	surf                  *Surface
	stencilAttachment     *Attachment
	msaaStencilAttachment *Attachment
	// dynamicMSAAAttachment is the discardable MSAA color renderbuffer a single-sample render target gains when DMSAA
	// promotes a render pass.
	dynamicMSAAAttachment *Attachment
	sampleCnt             int
	// totalMemorySamplesPerPixel is cached so VRAM footprint can still be reported after release/abandon zero the IDs.
	totalMemorySamplesPerPixel int
	multisampleFBOID           uint32
	singleSampleFBOID          uint32
	msColorRenderbufferID      uint32
	format                     Format
	needsStencilBind           [2]bool // indexed by useMultisampleFBO
	rtFBOOwnership             gpu.Ownership
}

// Surface returns the owning surface resource.
func (rt *RenderTarget) Surface() *Surface { return rt.surf }

// NumSamples returns the render target's sample count (1 if non-MSAA).
func (rt *RenderTarget) NumSamples() int { return rt.sampleCnt }

// GLFormat returns the render target's GL format.
func (rt *RenderTarget) GLFormat() Format { return rt.format }

// SingleSampleFBOID returns the resolved/single sample FBO id.
func (rt *RenderTarget) SingleSampleFBOID() uint32 { return rt.singleSampleFBOID }

// MultisampleFBOID returns the MSAA FBO id (UnresolvableFBOID when not multisampled).
func (rt *RenderTarget) MultisampleFBOID() uint32 { return rt.multisampleFBOID }

// IsFBO0 reports whether the given FBO slot is GL's default framebuffer (id 0).
func (rt *RenderTarget) IsFBO0(multisample bool) bool {
	if multisample {
		return rt.multisampleFBOID == 0
	}
	return rt.singleSampleFBOID == 0
}

// AlwaysClearStencil reports whether this render target must always clear stencil before use.
func (rt *RenderTarget) AlwaysClearStencil() bool { return rt.surf.GLRTFBOIDIs0() }

// StencilAttachment returns the stencil attachment for the given (MSAA or single-sample) slot.
func (rt *RenderTarget) StencilAttachment(useMSAASurface bool) *Attachment {
	if useMSAASurface {
		return rt.msaaStencilAttachment
	}
	return rt.stencilAttachment
}

// NumStencilBits returns the stencil bit depth of the given slot's attachment.
func (rt *RenderTarget) NumStencilBits(useMSAASurface bool) int {
	return rt.StencilAttachment(useMSAASurface).GLFormat().StencilBits()
}

// CanAttemptStencilAttachment reports whether a stencil buffer may be attached to this render target (a
// texture-render-target's FBO is never a wrapped FBO, so it always qualifies).
func (rt *RenderTarget) CanAttemptStencilAttachment(useMultisampleFBO bool) bool {
	if rt.surf.tex != nil {
		return true
	}
	// Only modify the FBO's attachments if we created the FBO. Public APIs do not currently allow for borrowed FBO
	// ownership, so we can safely assume that if an object is owned, this library created it.
	return rt.rtFBOOwnership == gpu.OwnershipOwned ||
		(rt.sampleCnt == 1 && useMultisampleFBO)
}

// AttachStencilAttachment attaches (or detaches, with nil) a stencil buffer to the given slot. The attachment ref moves
// to the render target.
func (rt *RenderTarget) AttachStencilAttachment(stencil *Attachment, useMSAASurface bool) {
	slot := &rt.stencilAttachment
	if useMSAASurface {
		slot = &rt.msaaStencilAttachment
	}
	if stencil == nil && *slot == nil {
		// No need to do any work since we currently don't have a stencil attachment and we're not actually adding one.
		return
	}
	// completeStencilAttachment: we defer attaching the new stencil buffer until the next time our framebuffer is
	// bound.
	if *slot != stencil {
		rt.needsStencilBind[boolIndex(useMSAASurface)] = true
	}
	if *slot != nil {
		(*slot).Unref()
	}
	*slot = stencil
}

// HasDynamicMSAAAttachment reports whether the render target has a dynamic MSAA attachment.
func (rt *RenderTarget) HasDynamicMSAAAttachment() bool { return rt.dynamicMSAAAttachment != nil }

// EnsureDynamicMSAAAttachment lazily creates the multisample FBO + discardable MSAA color renderbuffer a single-sample
// render target draws into under DMSAA. Resources here carry no back-pointer to a resource provider, so the caller
// passes it. The msaaResolvesAutomatically render-to-texture lane (multisample FBO == single-sample FBO) is unreachable
// on desktop GL and stays out.
func (rt *RenderTarget) EnsureDynamicMSAAAttachment(rp *ResourceProvider) bool {
	if rt.sampleCnt != 1 {
		panic("dynamic MSAA attachment on a multisampled render target")
	}
	if rt.multisampleFBOID != 0 {
		return true
	}
	if rt.dynamicMSAAAttachment != nil {
		panic("dynamic MSAA attachment without its FBO")
	}
	g := rt.surf.g
	caps := g.glCaps()

	internalSampleCount := caps.InternalMultisampleCount(rt.format)
	if internalSampleCount <= 1 {
		return false
	}

	g.fns().GenFramebuffers(1, &rt.multisampleFBOID)
	if rt.multisampleFBOID == 0 {
		return false
	}

	g.BindFramebuffer(FRAMEBUFFER, rt.multisampleFBOID)

	rt.dynamicMSAAAttachment = rp.GetDiscardableMSAAAttachment(rt.surf.dims, rt.format,
		internalSampleCount)
	if rt.dynamicMSAAAttachment == nil {
		// Drop the just-created FBO so a later retry doesn't short-circuit on the multisampleFBOID != 0 guard and
		// report success on a framebuffer that never got its color attachment.
		g.DeleteFramebuffer(rt.multisampleFBOID)
		rt.multisampleFBOID = 0
		return false
	}

	g.fns().FramebufferRenderbuffer(FRAMEBUFFER, COLOR_ATTACHMENT0, RENDERBUFFER,
		rt.dynamicMSAAAttachment.RenderbufferID())
	return true
}

// MustRebind reports whether the given FBO slot's stencil attachment needs to be (re)bound before use.
func (rt *RenderTarget) MustRebind(useMultisampleFBO bool) bool {
	return rt.needsStencilBind[boolIndex(useMultisampleFBO)]
}

// Bind binds the render target to FRAMEBUFFER for rendering.
func (rt *RenderTarget) Bind(useMultisampleFBO bool) {
	rt.bindInternal(FRAMEBUFFER, useMultisampleFBO)
}

// BindForPixelOps binds for copying, reading, or clearing pixel values, using the MSAA FBO only when there is a
// separate resolve texture.
func (rt *RenderTarget) BindForPixelOps(fboTarget uint32) {
	rt.bindInternal(fboTarget, rt.sampleCnt > 1)
}

// BindForResolve binds the multisampled and single sample FBOs to the two halves of a blit according to direction.
func (rt *RenderTarget) BindForResolve(dir ResolveDirection) {
	// If the multisample FBO is nonzero there is always something to resolve (even if the single sample buffer is FBO
	// 0). If it's zero, then there's nothing to resolve.
	if rt.multisampleFBOID == 0 {
		panic("nothing to resolve")
	}
	if dir == ResolveMSAAToSingle {
		rt.bindInternal(READ_FRAMEBUFFER, true)
		rt.bindInternal(DRAW_FRAMEBUFFER, false)
	} else {
		if !rt.surf.g.glCaps().CanResolveSingleToMSAA() {
			panic("cannot resolve single to MSAA")
		}
		rt.bindInternal(READ_FRAMEBUFFER, false)
		rt.bindInternal(DRAW_FRAMEBUFFER, true)
	}
}

// bindInternal binds the given FBO slot, minus the DMSAA render-to-texture FBO reconfiguration, which requires the
// single-sample and multisample FBO ids to be equal — only the ES render-to-texture extensions
// (msaaResolvesAutomatically) produce that, never desktop GL.
func (rt *RenderTarget) bindInternal(fboTarget uint32, useMultisampleFBO bool) {
	fboID := rt.singleSampleFBOID
	if useMultisampleFBO {
		fboID = rt.multisampleFBOID
	}
	g := rt.surf.g
	g.BindFramebuffer(fboTarget, fboID)

	// Make sure the stencil attachment is valid. Even though a color buffer op doesn't use stencil, our FBO still needs
	// to be "framebuffer complete".
	if rt.needsStencilBind[boolIndex(useMultisampleFBO)] {
		if stencil := rt.StencilAttachment(useMultisampleFBO); stencil != nil {
			g.fns().FramebufferRenderbuffer(fboTarget, STENCIL_ATTACHMENT, RENDERBUFFER,
				stencil.RenderbufferID())
			if stencil.GLFormat().IsPackedDepthStencil() {
				g.fns().FramebufferRenderbuffer(fboTarget, DEPTH_ATTACHMENT, RENDERBUFFER,
					stencil.RenderbufferID())
			} else {
				g.fns().FramebufferRenderbuffer(fboTarget, DEPTH_ATTACHMENT, RENDERBUFFER, 0)
			}
		} else {
			g.fns().FramebufferRenderbuffer(fboTarget, STENCIL_ATTACHMENT, RENDERBUFFER, 0)
			g.fns().FramebufferRenderbuffer(fboTarget, DEPTH_ATTACHMENT, RENDERBUFFER, 0)
		}
		rt.needsStencilBind[boolIndex(useMultisampleFBO)] = false
	}
}

// release deletes the render target's owned FBOs and renderbuffer. A dynamic MSAA FBO created on a *borrowed*
// (wrapped-FBO) render target is not deleted here — all FBO deletion is conditioned on ownership even though
// EnsureDynamicMSAAAttachment created that FBO itself; the name is reclaimed at GL context teardown. The attachment ref
// is released either way.
func (rt *RenderTarget) release() {
	g := rt.surf.g
	if rt.rtFBOOwnership != gpu.OwnershipBorrowed {
		if rt.singleSampleFBOID != 0 {
			g.DeleteFramebuffer(rt.singleSampleFBOID)
		}
		if rt.multisampleFBOID != 0 && rt.multisampleFBOID != rt.singleSampleFBOID {
			g.DeleteFramebuffer(rt.multisampleFBOID)
		}
		if rt.msColorRenderbufferID != 0 {
			id := rt.msColorRenderbufferID
			g.fns().DeleteRenderbuffers(1, &id)
		}
	}
	rt.multisampleFBOID = UnresolvableFBOID
	rt.singleSampleFBOID = UnresolvableFBOID
	rt.msColorRenderbufferID = 0
	rt.dropDynamicMSAAAttachment()
	rt.dropStencils(func(a *Attachment) { a.Unref() })
}

// abandon forgets the render target's FBOs and attachments without deleting them (the context is gone).
func (rt *RenderTarget) abandon() {
	rt.multisampleFBOID = UnresolvableFBOID
	rt.singleSampleFBOID = UnresolvableFBOID
	rt.msColorRenderbufferID = 0
	rt.dropDynamicMSAAAttachment()
	rt.dropStencils(func(a *Attachment) { a.Unref() })
}

func (rt *RenderTarget) dropDynamicMSAAAttachment() {
	if rt.dynamicMSAAAttachment != nil {
		rt.dynamicMSAAAttachment.Unref()
		rt.dynamicMSAAAttachment = nil
	}
}

func (rt *RenderTarget) dropStencils(drop func(*Attachment)) {
	if rt.stencilAttachment != nil {
		drop(rt.stencilAttachment)
		rt.stencilAttachment = nil
	}
	if rt.msaaStencilAttachment != nil {
		drop(rt.msaaStencilAttachment)
		rt.msaaStencilAttachment = nil
	}
}

func newRenderTargetFacet(s *Surface, format Format, sampleCount int, ids *RenderTargetIDs, stencil *Attachment) *RenderTarget {
	rt := &RenderTarget{
		surf:                       s,
		multisampleFBOID:           ids.MultisampleFBOID,
		singleSampleFBOID:          ids.SingleSampleFBOID,
		msColorRenderbufferID:      ids.MSColorRenderbufferID,
		format:                     format,
		sampleCnt:                  sampleCount,
		rtFBOOwnership:             ids.RTFBOOwnership,
		totalMemorySamplesPerPixel: ids.TotalMemorySamplesPerPixel,
	}
	if sampleCount > 1 {
		rt.msaaStencilAttachment = stencil
	} else {
		rt.stencilAttachment = stencil
	}
	if rt.multisampleFBOID|rt.singleSampleFBOID == 0 {
		s.flags |= gpu.SurfaceFlagGLRTFBOIDIs0
	}
	return rt
}

// stencilBitsToFormat returns a "fake" format whose stencil bit count matches a wrapped FBO's stencil buffer.
func stencilBitsToFormat(stencilBits int) Format {
	switch stencilBits {
	case 8:
		// The packed format at least does not underestimate the total size.
		return FormatDEPTH24STENCIL8
	case 16:
		return FormatSTENCILINDEX16
	default:
		panic("unexpected stencil bit count")
	}
}

// MakeWrappedRenderTarget wraps a caller-owned FBO (typically FBO 0) into a render target resource. Wrapped render
// targets are never budgeted.
func MakeWrappedRenderTarget(g *Gpu, dims geom.ISize, format Format, sampleCount int, ids *RenderTargetIDs, stencilBits int, label string) *Surface {
	var sb *Attachment
	if stencilBits > 0 {
		// We don't have the actual renderbuffer ID, so use 0 to ensure it is never deleted; only the stencil bit count
		// of the fake format matters.
		sb = makeWrappedRenderBufferAttachment(g, 0, dims, AttachmentUsageStencil, sampleCount,
			stencilBitsToFormat(stencilBits))
	}
	s := &Surface{g: g, dims: dims}
	s.rt = newRenderTargetFacet(s, format, sampleCount, ids, sb)
	s.RegisterWithCacheWrapped(g.ResourceCache(), s, gpu.WrapCacheableNo, label)
	return s
}

// NewTextureRenderTarget creates a Surface with both texture and render-target facets and registers it with the cache.
func NewTextureRenderTarget(g *Gpu, budgeted gpu.Budgeted, sampleCount int, texDesc *TextureDesc, ids *RenderTargetIDs, mipmapStatus gpu.MipmapStatus, label string) *Surface {
	s := &Surface{g: g, dims: texDesc.Size}
	s.tex = newTextureFacet(s, texDesc, NewTextureParameters(), mipmapStatus)
	s.rt = newRenderTargetFacet(s, texDesc.Format, sampleCount, ids, nil)
	s.RegisterWithCache(g.ResourceCache(), s, budgeted, label)
	return s
}

func boolIndex(b bool) int {
	if b {
		return 1
	}
	return 0
}
