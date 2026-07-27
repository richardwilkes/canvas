// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The surface proxy hierarchy: a proxy delays the acquisition of its backing surface until flush (or on-demand
// instantiation), which is what lets the resource allocator recycle backing stores across the render-task DAG. One
// SurfaceProxy carries optional TextureProxy and RenderTargetProxy facets rather than separate
// texture/render-target/texture-render-target subclasses; virtual dispatch
// (instantiate/createSurface/callbackDesc/onUninstantiatedGpuMemorySize) becomes facet checks instead.
//
// Trims: deferred display lists do not exist, so that provider plumbing is dropped (every proxy is created by the
// direct context's provider); protected content, Vulkan secondary command buffers, promise proxies, compressed
// textures, and the worker-thread upload hook for deferred proxies are not implemented.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// UseAllocator records whether the resource allocator should instantiate the proxy at flush (true) or the proxy is
// instantiated outside the allocator (false).
type UseAllocator bool

// UseAllocator values.
const (
	UseAllocatorNo  UseAllocator = false
	UseAllocatorYes UseAllocator = true
)

// ResolveFlags are the "resolutions" needed on a surface before its pixels can be accessed. If both are requested the
// MSAA resolve happens first.
type ResolveFlags uint32

// ResolveFlags values.
const (
	// ResolveFlagMSAA: blit and resolve an internal MSAA render buffer into the texture.
	ResolveFlagMSAA ResolveFlags = 1 << 0
	// ResolveFlagMipMaps: regenerate all mipmap levels.
	ResolveFlagMipMaps ResolveFlags = 1 << 1
)

// LazyInstantiationKeyMode describes the key relationship between a lazy proxy and the surface its callback returns.
type LazyInstantiationKeyMode int32

// LazyInstantiationKeyMode values.
const (
	// LazyKeyUnsynced: don't key the surface with the proxy's key; the callback may return a surface that already has
	// an unrelated unique key.
	LazyKeyUnsynced LazyInstantiationKeyMode = iota
	// LazyKeySynced: keep the surface's unique key in sync with the proxy's.
	LazyKeySynced
)

// LazySurfaceDesc describes the expected properties of the surface returned by a lazy instantiation callback.
// Dimensions are negative for a fully lazy proxy.
type LazySurfaceDesc struct {
	Label       string
	SampleCnt   int
	Dimensions  geom.ISize
	Format      Format
	TextureType gpu.TextureType
	Fit         gpu.BackingFit
	Renderable  gpu.Renderable
	Mipmapped   gpu.Mipmapped
	Budgeted    gpu.Budgeted
}

// LazyCallbackResult is the outcome of a lazy instantiation callback. Construct results with MakeLazyCallbackResult to
// get the usual defaults (KeyMode synced, callback released); the zero value is the "instantiation failed" result.
type LazyCallbackResult struct {
	// Surface is the backing surface; nil means instantiation failed.
	Surface *Surface
	KeyMode LazyInstantiationKeyMode
	// ReleaseCallback: dispose of the callback after it returns (only honored when Surface is non-nil; a nil Surface
	// always preserves the callback).
	ReleaseCallback bool
}

// MakeLazyCallbackResult builds a successful LazyCallbackResult with the usual defaults: key synced, callback released.
func MakeLazyCallbackResult(surface *Surface) LazyCallbackResult {
	return LazyCallbackResult{Surface: surface, KeyMode: LazyKeySynced, ReleaseCallback: true}
}

// LazyInstantiateCallback lazily creates (or reattaches to) the backing surface for a lazy proxy.
type LazyInstantiateCallback func(*ResourceProvider, *LazySurfaceDesc) LazyCallbackResult

const invalidProxyGpuMemorySize = ^uint64(0)

// SurfaceProxy stands in for a backing GPU surface before it's actually allocated, with the texture/render-target roles
// expressed as facets rather than subclasses. Wrapped proxies share the wrapped resource's unique ID; deferred/lazy
// proxies draw a fresh ID from the shared resource/proxy pool; the ID never changes across instantiation.
//
// Ref counting: Go has no destructors and the cache must stay on a deterministic free path, so the proxy ref count is
// explicit: constructors return a proxy with one ref owned by the caller, long-lived owners (render tasks, surface
// contexts) take refs, and the last Unref drops the target's cache ref so the backing resource can become purgeable.
// Transient holders (views, locals within a recording) do not ref; their validity is guaranteed by the owning
// task/context, since recording is single-threaded.
type SurfaceProxy struct {
	lazyCallback LazyInstantiateCallback
	// target is nil for deferred proxies until instantiation; for wrapped proxies it points to the wrapped resource.
	target *Surface
	rt     *RenderTargetProxy
	tex    *TextureProxy
	label  string
	// gpuMemorySize is lazily evaluated: when the proxy wraps a resource the resource is asked, otherwise the proxy
	// computes the answer itself.
	gpuMemorySize   uint64
	taskTargetCount int
	dims            geom.ISize
	// textureType is the backend format's texture type: TextureTypeNone for render-target-only proxies, otherwise 2D
	// (internal creations) or the wrapped texture's type.
	textureType                gpu.TextureType
	uniqueID                   gpu.ResourceUniqueID
	refCnt                     int32
	format                     Format
	surfaceFlags               gpu.SurfaceFlags
	useAllocator               UseAllocator
	ignoredByResourceAllocator bool
	budgeted                   gpu.Budgeted   // always yes for lazy-callback proxies; from the resource for wrapped
	fit                        gpu.BackingFit // always approx for lazy-callback proxies, exact for wrapped
}

// newDeferredSurfaceProxy builds a proxy with no backing surface yet, taking a new unique ID from the shared
// resource/proxy pool.
func newDeferredSurfaceProxy(format Format, textureType gpu.TextureType, dims geom.ISize, fit gpu.BackingFit, budgeted gpu.Budgeted, surfaceFlags gpu.SurfaceFlags, useAllocator UseAllocator, label string) *SurfaceProxy {
	if dims.Width <= 0 || dims.Height <= 0 {
		panic("deferred proxies require positive dimensions")
	}
	return &SurfaceProxy{
		refCnt:        1,
		surfaceFlags:  surfaceFlags,
		format:        format,
		textureType:   textureType,
		dims:          dims,
		fit:           fit,
		budgeted:      budgeted,
		useAllocator:  useAllocator,
		uniqueID:      gpu.CreateResourceUniqueID(),
		label:         label,
		gpuMemorySize: invalidProxyGpuMemorySize,
	}
}

// newLazySurfaceProxy builds a proxy that instantiates its backing surface on demand via callback.
func newLazySurfaceProxy(callback LazyInstantiateCallback, format Format, textureType gpu.TextureType, dims geom.ISize, fit gpu.BackingFit, budgeted gpu.Budgeted, surfaceFlags gpu.SurfaceFlags, useAllocator UseAllocator, label string) *SurfaceProxy {
	if callback == nil {
		panic("lazy proxies require a callback")
	}
	// A "fully" lazy proxy's dimensions are not known until instantiation, signaled by negative width and height (and
	// required to be approx fit); regular lazy proxies must be positive.
	fullyLazy := dims.Width < 0 && dims.Height < 0
	if !fullyLazy && (dims.Width <= 0 || dims.Height <= 0) {
		panic("lazy proxies require fully-negative or fully-positive dimensions")
	}
	if fullyLazy && fit != gpu.BackingFitApprox {
		panic("fully lazy proxies must be approx fit")
	}
	return &SurfaceProxy{
		refCnt:        1,
		surfaceFlags:  surfaceFlags,
		format:        format,
		textureType:   textureType,
		dims:          dims,
		fit:           fit,
		budgeted:      budgeted,
		useAllocator:  useAllocator,
		uniqueID:      gpu.CreateResourceUniqueID(),
		lazyCallback:  callback,
		label:         label,
		gpuMemorySize: invalidProxyGpuMemorySize,
	}
}

// newWrappedSurfaceProxy builds a proxy around an already-instantiated surface, sharing its unique ID and copying its
// properties. It still takes UseAllocator because an instantiated proxy can participate in allocation by having its
// backing resource recycled to other proxies.
func newWrappedSurfaceProxy(surface *Surface, fit gpu.BackingFit, useAllocator UseAllocator) *SurfaceProxy {
	textureType := gpu.TextureTypeNone
	if t := surface.AsTexture(); t != nil {
		textureType = t.TextureType()
	}
	return &SurfaceProxy{
		refCnt:        1,
		target:        surface,
		surfaceFlags:  surface.Flags(),
		format:        surface.Format(),
		textureType:   textureType,
		dims:          surface.Dimensions(),
		fit:           fit,
		budgeted:      gpu.Budgeted(surface.BudgetedType() == gpu.BudgetedTypeBudgeted),
		useAllocator:  useAllocator,
		uniqueID:      surface.UniqueID(), // converting from unique resource ID to a proxy ID
		label:         surface.Label(),
		gpuMemorySize: invalidProxyGpuMemorySize,
	}
}

// Ref takes a reference on the proxy.
func (p *SurfaceProxy) Ref() { p.refCnt++ }

// Unref drops a reference. The last unref clears an orphaned-but-still-valid unique key from the proxy provider
// (without invalidating the cached resource) and drops the target's cache ref so the backing resource can become
// purgeable.
func (p *SurfaceProxy) Unref() {
	p.refCnt--
	if p.refCnt > 0 {
		return
	}
	if p.refCnt < 0 {
		panic("SurfaceProxy over-unref")
	}
	// Zero out the target first so invalidation code doesn't try to use it, then send the invalid-key message to the
	// provider (which doesn't remove the cached resource).
	target := p.target
	p.target = nil
	if p.tex != nil && p.tex.uniqueKey.IsValid() && p.tex.proxyProvider != nil {
		p.tex.proxyProvider.ProcessInvalidUniqueKey(&p.tex.uniqueKey, p.tex,
			InvalidateGPUResourceNo)
	}
	// Drop the target's ref.
	if target != nil {
		target.Unref()
	}
}

// RefCnt exposes the current ref count for tests and debug assertions.
func (p *SurfaceProxy) RefCnt() int32 { return p.refCnt }

// IsLazy reports whether the proxy has yet to be instantiated and still has a pending callback.
func (p *SurfaceProxy) IsLazy() bool { return !p.IsInstantiated() && p.lazyCallback != nil }

// IsFullyLazy reports whether width and height are unknown until instantiation.
func (p *SurfaceProxy) IsFullyLazy() bool { return p.dims.Width < 0 }

// Dimensions returns the proxy's dimensions. Not valid on fully lazy proxies.
func (p *SurfaceProxy) Dimensions() geom.ISize {
	if p.IsFullyLazy() {
		panic("dimensions of a fully lazy proxy are unknown")
	}
	return p.dims
}

// Width returns the proxy's width.
func (p *SurfaceProxy) Width() int32 { return p.Dimensions().Width }

// Height returns the proxy's height.
func (p *SurfaceProxy) Height() int32 { return p.Dimensions().Height }

// BackingStoreDimensions returns the dimensions the backing surface has (or will have after instantiation).
func (p *SurfaceProxy) BackingStoreDimensions() geom.ISize {
	if p.IsFullyLazy() {
		panic("backing store dimensions of a fully lazy proxy are unknown")
	}
	if p.target != nil {
		return p.target.Dimensions()
	}
	if p.fit == gpu.BackingFitExact {
		return p.dims
	}
	return gpu.GetApproxSize(p.dims)
}

// BackingStoreBoundsIRect returns the backing store's dimensions as a rect at the origin.
func (p *SurfaceProxy) BackingStoreBoundsIRect() geom.IRect {
	return geom.IRectSize(p.BackingStoreDimensions())
}

// BoundsIRect returns the proxy's dimensions as a rect at the origin (the float form is built where needed).
func (p *SurfaceProxy) BoundsIRect() geom.IRect {
	return geom.IRectSize(p.Dimensions())
}

// IsFunctionallyExact is a perhaps faster check for Dimensions() == BackingStoreDimensions().
func (p *SurfaceProxy) IsFunctionallyExact() bool {
	if p.IsFullyLazy() {
		panic("isFunctionallyExact on a fully lazy proxy")
	}
	return p.fit == gpu.BackingFitExact || p.dims == gpu.GetApproxSize(p.dims)
}

// Format returns the proxy's GL format.
func (p *SurfaceProxy) Format() Format { return p.format }

// TextureType returns the texture type for this proxy's format.
func (p *SurfaceProxy) TextureType() gpu.TextureType { return p.textureType }

// UniqueID consistently identifies the proxy but must never be used to distinguish between resources and proxies.
func (p *SurfaceProxy) UniqueID() gpu.ResourceUniqueID { return p.uniqueID }

// UnderlyingUniqueID returns the target's unique ID if instantiated, else the proxy's own ID.
func (p *SurfaceProxy) UnderlyingUniqueID() gpu.ResourceUniqueID {
	if p.target != nil {
		return p.target.UniqueID()
	}
	return p.uniqueID
}

// IsInstantiated reports whether the proxy has an assigned backing surface.
func (p *SurfaceProxy) IsInstantiated() bool { return p.target != nil }

// IsUsedAsTaskTarget records that this proxy has become the target of a render task.
func (p *SurfaceProxy) IsUsedAsTaskTarget() { p.taskTargetCount++ }

// TaskTargetCount returns how many render tasks have targeted this proxy.
func (p *SurfaceProxy) TaskTargetCount() int { return p.taskTargetCount }

// PeekSurface returns the backing surface if instantiated, else nil.
func (p *SurfaceProxy) PeekSurface() *Surface { return p.target }

// PeekTexture returns the backing texture if instantiated and textured, else nil.
func (p *SurfaceProxy) PeekTexture() *Texture {
	if p.target == nil {
		return nil
	}
	return p.target.AsTexture()
}

// PeekRenderTarget returns the backing render target if instantiated and renderable, else nil.
func (p *SurfaceProxy) PeekRenderTarget() *RenderTarget {
	if p.target == nil {
		return nil
	}
	return p.target.AsRenderTarget()
}

// IsBudgeted reports whether the proxy counts against the resource cache budget.
func (p *SurfaceProxy) IsBudgeted() gpu.Budgeted { return p.budgeted }

// ReadOnly reports whether the proxy is read-only. Read-only proxies bypass interval tracking and assignment in the
// resource allocator.
func (p *SurfaceProxy) ReadOnly() bool { return p.surfaceFlags&gpu.SurfaceFlagReadOnly != 0 }

// FramebufferOnly reports whether the surface can only be used as a framebuffer, not sampled.
func (p *SurfaceProxy) FramebufferOnly() bool {
	return p.surfaceFlags&gpu.SurfaceFlagFramebufferOnly != 0
}

// RequiresManualMSAAResolve reports whether the surface is a multisampled render target that internally holds a
// non-MSAA texture to resolve into.
func (p *SurfaceProxy) RequiresManualMSAAResolve() bool {
	return p.surfaceFlags&gpu.SurfaceFlagRequiresManualMSAAResolve != 0
}

// SurfaceFlags returns the proxy's internal surface flags.
func (p *SurfaceProxy) SurfaceFlags() gpu.SurfaceFlags { return p.surfaceFlags }

// Label returns the proxy's debug label.
func (p *SurfaceProxy) Label() string { return p.label }

// UseAllocator reports whether the resource allocator should instantiate this proxy.
func (p *SurfaceProxy) UseAllocator() UseAllocator { return p.useAllocator }

// IgnoredByResourceAllocator reports whether the resource allocator should skip this proxy entirely.
func (p *SurfaceProxy) IgnoredByResourceAllocator() bool { return p.ignoredByResourceAllocator }

// SetIgnoredByResourceAllocator marks the proxy to be skipped by the resource allocator.
func (p *SurfaceProxy) SetIgnoredByResourceAllocator() { p.ignoredByResourceAllocator = true }

// GpuMemorySize returns the proxy's estimated GPU memory footprint (lazily computed, cached).
func (p *SurfaceProxy) GpuMemorySize() uint64 {
	if p.IsFullyLazy() {
		panic("gpuMemorySize of a fully lazy proxy is unknown")
	}
	if p.gpuMemorySize == invalidProxyGpuMemorySize {
		p.gpuMemorySize = p.onUninstantiatedGpuMemorySize()
	}
	return p.gpuMemorySize
}

// onUninstantiatedGpuMemorySize estimates the memory footprint from the proxy's facets (texture and/or render target)
// before it has an actual backing surface to ask.
func (p *SurfaceProxy) onUninstantiatedGpuMemorySize() uint64 {
	colorSamplesPerPixel := 1
	mipmapped := gpu.MipmappedNo
	if p.rt != nil {
		colorSamplesPerPixel = p.rt.NumSamples()
		if colorSamplesPerPixel > 1 {
			// Add one for the resolve buffer.
			colorSamplesPerPixel++
		}
	}
	if p.tex != nil {
		mipmapped = p.tex.ProxyMipmapped()
	}
	return ComputeSurfaceSize(p.format, p.dims, colorSamplesPerPixel, mipmapped, !p.isExact())
}

// isExact reports whether the proxy requires an exact-sized backing store.
func (p *SurfaceProxy) isExact() bool { return p.fit == gpu.BackingFitExact }

// AsTextureProxy returns the texture facet; nil when the proxy is not a texture.
func (p *SurfaceProxy) AsTextureProxy() *TextureProxy { return p.tex }

// AsRenderTargetProxy returns the render-target facet; nil when not a render target.
func (p *SurfaceProxy) AsRenderTargetProxy() *RenderTargetProxy { return p.rt }

// UniqueKey returns the proxy's unique key; only texture proxies ever have a valid one.
func (p *SurfaceProxy) UniqueKey() *gpu.UniqueKey {
	if p.tex != nil {
		return &p.tex.uniqueKey
	}
	var invalid gpu.UniqueKey
	return &invalid
}

// CanSkipResourceAllocator reports whether the resource allocator can skip this proxy: already-instantiated proxies
// whose backing surface cannot be recycled do not need to be considered.
func (p *SurfaceProxy) CanSkipResourceAllocator() bool {
	if p.useAllocator == UseAllocatorNo {
		// Usually an atlas or onFlush proxy.
		return true
	}
	peek := p.PeekSurface()
	if peek == nil {
		return false
	}
	// If this resource is already allocated and not recyclable then the resource allocator does not need to do anything
	// with it.
	return !peek.ScratchKey().IsValid()
}

// Deinstantiate drops the proxy's assigned backing surface, returning it to an uninstantiated state. The proxy owns one
// ref on that surface (see assign), so it is released here; without that the surface can never become purgeable.
func (p *SurfaceProxy) Deinstantiate() {
	if !p.IsInstantiated() {
		panic("deinstantiating an uninstantiated proxy")
	}
	target := p.target
	p.target = nil
	target.Unref()
}

// assign gives the proxy its backing surface.
func (p *SurfaceProxy) assign(surface *Surface) {
	if p.target != nil || surface == nil {
		panic("assign requires an uninstantiated proxy and a surface")
	}
	p.target = surface
}

// computeScratchKey derives the scratch-resource key a recyclable backing store for this proxy would use.
func (p *SurfaceProxy) computeScratchKey(caps *Caps, key *gpu.ScratchKey) {
	renderable := gpu.RenderableNo
	sampleCount := 1
	if p.rt != nil {
		renderable = gpu.RenderableYes
		sampleCount = p.rt.NumSamples()
	}
	mipmapped := gpu.MipmappedNo
	if p.tex != nil {
		mipmapped = p.tex.Mipmapped()
	}
	ComputeTextureScratchKey(caps, p.format, p.BackingStoreDimensions(), renderable, sampleCount,
		mipmapped, key)
}

// createSurfaceImpl allocates a fresh (or approx-fit recycled) backing surface for the proxy.
func (p *SurfaceProxy) createSurfaceImpl(provider *ResourceProvider, sampleCnt int, renderable gpu.Renderable, mipmapped gpu.Mipmapped) *Surface {
	if mipmapped == gpu.MipmappedYes && p.fit != gpu.BackingFitExact {
		panic("mipmapped surfaces must be exact fit")
	}
	if p.IsLazy() || p.target != nil {
		panic("createSurfaceImpl on a lazy or instantiated proxy")
	}
	if p.fit == gpu.BackingFitApprox {
		return provider.CreateApproxTexture(p.dims, p.format, p.textureType, renderable, sampleCnt,
			p.label)
	}
	return provider.CreateTexture(p.dims, p.format, p.textureType, renderable, sampleCnt,
		mipmapped, p.budgeted, p.label)
}

// createSurface allocates the right kind of backing surface (texture, render target, or both) based on the proxy's
// facets; used by the resource allocator (via ProxyProvider) when instantiating with a recycled or fresh backing store.
func (p *SurfaceProxy) createSurface(provider *ResourceProvider) *Surface {
	switch {
	case p.rt != nil && p.tex != nil:
		return p.createSurfaceImpl(provider, p.rt.NumSamples(), gpu.RenderableYes, p.tex.Mipmapped())
	case p.rt != nil:
		return p.createSurfaceImpl(provider, p.rt.NumSamples(), gpu.RenderableYes, gpu.MipmappedNo)
	default:
		return p.createSurfaceImpl(provider, 1, gpu.RenderableNo, p.tex.Mipmapped())
	}
}

// instantiateImpl acquires (or reuses) a backing surface and assigns it to the proxy.
func (p *SurfaceProxy) instantiateImpl(provider *ResourceProvider, sampleCnt int, renderable gpu.Renderable, mipmapped gpu.Mipmapped, uniqueKey *gpu.UniqueKey) bool {
	if p.IsLazy() {
		panic("instantiateImpl on a lazy proxy")
	}
	if p.target != nil {
		return true
	}
	surface := p.createSurfaceImpl(provider, sampleCnt, renderable, mipmapped)
	if surface == nil {
		return false
	}
	// If there was an invalidation pending for this key, we might have just processed it, causing the key (stored on
	// this proxy) to become invalid.
	if uniqueKey != nil && uniqueKey.IsValid() {
		provider.AssignUniqueKeyToResource(uniqueKey, surface)
	}
	p.assign(surface)
	return true
}

// Instantiate acquires the backing surface if necessary, dispatching on the proxy's facets. Returns false for lazy
// proxies (they instantiate through the lazy callback instead).
func (p *SurfaceProxy) Instantiate(provider *ResourceProvider) bool {
	if p.IsLazy() {
		return false
	}
	switch {
	case p.rt != nil && p.tex != nil:
		key := &p.tex.uniqueKey
		if !key.IsValid() {
			key = nil
		}
		return p.instantiateImpl(provider, p.rt.NumSamples(), gpu.RenderableYes, p.tex.Mipmapped(),
			key)
	case p.rt != nil:
		return p.instantiateImpl(provider, p.rt.NumSamples(), gpu.RenderableYes, gpu.MipmappedNo,
			nil)
	default:
		key := &p.tex.uniqueKey
		if !key.IsValid() {
			key = nil
		}
		return p.instantiateImpl(provider, 1, gpu.RenderableNo, p.tex.mipmapped, key)
	}
}

// callbackDesc describes the surface a lazy proxy's callback is expected to return, based on the proxy's facets.
func (p *SurfaceProxy) callbackDesc() LazySurfaceDesc {
	if p.rt != nil && p.tex == nil {
		// Only exactly-sized lazy render-target-only proxies are expected.
		if p.IsFullyLazy() || !p.IsFunctionallyExact() {
			panic("lazy render-target-only proxies must be exactly sized")
		}
		return LazySurfaceDesc{
			Dimensions:  p.dims,
			Fit:         gpu.BackingFitExact,
			Renderable:  gpu.RenderableYes,
			Mipmapped:   gpu.MipmappedNo,
			SampleCnt:   p.rt.NumSamples(),
			Format:      p.format,
			TextureType: gpu.TextureTypeNone,
			Budgeted:    p.budgeted,
			Label:       p.label,
		}
	}
	// The texture (and texture-render-target) case.
	var dims geom.ISize
	var fit gpu.BackingFit
	if p.IsFullyLazy() {
		fit = gpu.BackingFitApprox
		dims = geom.ISize{Width: -1, Height: -1}
	} else {
		if p.IsFunctionallyExact() {
			fit = gpu.BackingFitExact
		} else {
			fit = gpu.BackingFitApprox
		}
		dims = p.dims
	}
	renderable := gpu.RenderableNo
	sampleCnt := 1
	if p.rt != nil {
		renderable = gpu.RenderableYes
		sampleCnt = p.rt.NumSamples()
	}
	return LazySurfaceDesc{
		Dimensions:  dims,
		Fit:         fit,
		Renderable:  renderable,
		Mipmapped:   p.tex.Mipmapped(),
		SampleCnt:   sampleCnt,
		Format:      p.format,
		TextureType: p.textureType,
		Budgeted:    p.budgeted,
		Label:       p.label,
	}
}

// SetLazyDimensions lets the client specify a fully-lazy proxy's dimensions once they're decided, before instantiation.
func (p *SurfaceProxy) SetLazyDimensions(dims geom.ISize) {
	if !p.IsFullyLazy() {
		panic("setLazyDimensions on a non-fully-lazy proxy")
	}
	if dims.IsEmpty() {
		panic("setLazyDimensions requires non-empty dimensions")
	}
	p.dims = dims
}

// doLazyInstantiation instantiates a lazy proxy: reattaches to a cached surface by unique key if possible, otherwise
// runs the lazy callback.
func (p *SurfaceProxy) doLazyInstantiation(provider *ResourceProvider) bool {
	if !p.IsLazy() {
		panic("doLazyInstantiation on a non-lazy proxy")
	}
	var surface *Surface
	if key := p.UniqueKey(); key.IsValid() {
		// First try to reattach to a cached version if the proxy is uniquely keyed.
		if res := provider.FindByUniqueKey(key); res != nil {
			surface = res.(*Surface)
		}
	}
	syncKey := true
	releaseCallback := false
	if surface == nil {
		desc := p.callbackDesc()
		result := p.lazyCallback(provider, &desc)
		surface = result.Surface
		syncKey = result.KeyMode == LazyKeySynced
		releaseCallback = surface != nil && result.ReleaseCallback
	}
	if surface == nil {
		p.dims = geom.ISize{}
		return false
	}

	if p.IsFullyLazy() {
		// This was a fully lazy proxy: fill in the width & height. Partially lazy proxies preserve their original
		// dimensions since those indicate the content area.
		p.dims = surface.Dimensions()
	}
	if p.Width() > surface.Width() || p.Height() > surface.Height() {
		panic("lazy instantiation returned an undersized surface")
	}

	if p.tex != nil {
		p.tex.syncTargetKey = syncKey
		if syncKey {
			if key := &p.tex.uniqueKey; key.IsValid() {
				if !surface.UniqueKey().IsValid() {
					// If 'surface' is newly created, attach the unique key.
					provider.AssignUniqueKeyToResource(key, surface)
				} else if !surface.UniqueKey().Equal(key) {
					panic("reattached surface has a mismatched unique key")
				}
			} else if surface.UniqueKey().IsValid() {
				panic("unkeyed proxy synced to a keyed surface")
			}
		}
	}

	p.assign(surface)
	if releaseCallback {
		p.lazyCallback = nil
	}
	return true
}

// TextureProxy is the texture facet of a SurfaceProxy: the added state a texture-backed proxy needs beyond the base
// surface proxy.
type TextureProxy struct {
	proxy *SurfaceProxy
	// proxyProvider is only set while uniqueKey is valid.
	proxyProvider *ProxyProvider
	uniqueKey     gpu.UniqueKey
	// mipmapStatus tracks mipmap validity at the proxy level (distinct from the backing texture's status): it
	// determines when mip levels must be regenerated during execution of the task DAG.
	mipmapStatus gpu.MipmapStatus
	mipmapped    gpu.Mipmapped
	// syncTargetKey: should the target's unique key be kept in sync with ours.
	syncTargetKey bool
}

// Proxy returns the owning SurfaceProxy.
func (t *TextureProxy) Proxy() *SurfaceProxy { return t.proxy }

// Mipmapped returns the instantiated target's mip state if available (lazy instantiation may have given us a mipped
// target we can benefit from), else the creation-time value.
func (t *TextureProxy) Mipmapped() gpu.Mipmapped {
	if t.proxy.IsInstantiated() {
		return t.proxy.PeekTexture().Mipmapped()
	}
	return t.mipmapped
}

// ProxyMipmapped returns the creation-time mip state, regardless of instantiation.
func (t *TextureProxy) ProxyMipmapped() gpu.Mipmapped { return t.mipmapped }

// MipmapsAreDirty reports whether a mipmapped proxy's mip levels need regenerating.
func (t *TextureProxy) MipmapsAreDirty() bool {
	return t.mipmapped == gpu.MipmappedYes && t.mipmapStatus != gpu.MipmapStatusValid
}

// MarkMipmapsDirty flags a mipmapped proxy's mip levels as needing regeneration.
func (t *TextureProxy) MarkMipmapsDirty() {
	if t.mipmapped != gpu.MipmappedYes {
		panic("marking mipmaps dirty on a non-mipmapped proxy")
	}
	t.mipmapStatus = gpu.MipmapStatusDirty
}

// MarkMipmapsClean flags a mipmapped proxy's mip levels as up to date.
func (t *TextureProxy) MarkMipmapsClean() {
	if t.mipmapped != gpu.MipmappedYes {
		panic("marking mipmaps clean on a non-mipmapped proxy")
	}
	t.mipmapStatus = gpu.MipmapStatusValid
}

// TextureType returns the proxy's texture type.
func (t *TextureProxy) TextureType() gpu.TextureType { return t.proxy.textureType }

// HasRestrictedSampling reports whether the texture does not support mipmaps and only supports clamp wrap mode
// (rectangle textures, under this implementation's trim).
func (t *TextureProxy) HasRestrictedSampling() bool {
	return t.proxy.textureType == gpu.TextureTypeRectangle
}

// UniqueKey returns the texture proxy's unique key.
func (t *TextureProxy) UniqueKey() *gpu.UniqueKey { return &t.uniqueKey }

// setUniqueKey assigns the texture proxy's unique key, syncing it to the target's key if instantiated.
func (t *TextureProxy) setUniqueKey(provider *ProxyProvider, key *gpu.UniqueKey) {
	if !key.IsValid() {
		panic("setUniqueKey requires a valid key")
	}
	if t.uniqueKey.IsValid() {
		panic("proxies can only ever get one unique key")
	}
	if t.proxy.target != nil && t.syncTargetKey {
		if !t.proxy.target.UniqueKey().IsValid() {
			t.proxy.target.SetUniqueKey(key)
		}
		if !t.proxy.target.UniqueKey().Equal(key) {
			panic("target key out of sync with proxy key")
		}
	}
	t.uniqueKey = *key
	t.proxyProvider = provider
}

// clearUniqueKey clears the texture proxy's unique key.
func (t *TextureProxy) clearUniqueKey() {
	t.uniqueKey.Reset()
	t.proxyProvider = nil
}

// ProxiesAreCompatibleAsDynamicState reports whether the two proxies can be used as dynamic state together when
// flushing draws.
func ProxiesAreCompatibleAsDynamicState(first, second *SurfaceProxy) bool {
	return first.textureType == second.textureType && first.format == second.format
}

// Arenas matches the lifetime of a single frame — created on the SurfaceFillContext's render target proxy, ref'd by
// each OpsTask, and dropped after the first OpsTask's execution completes so new tasks get fresh arenas. The op/sub-run
// arena allocators themselves arrive with the op and sub-run recording layers, so for now this carries only the
// lifetime protocol.
type Arenas struct {
	flushed bool
}

// Flush marks the arenas flushed; using a flushed arena is a bug.
func (a *Arenas) Flush() { a.flushed = true }

// RenderTargetProxy is the render-target facet of a SurfaceProxy: the added state a renderable proxy needs beyond the
// base surface proxy. Beware: the proxy's uniqueID usually differs from the render target's.
type RenderTargetProxy struct {
	proxy         *SurfaceProxy
	arenas        *Arenas
	sampleCnt     int
	msaaDirtyRect geom.IRect
	needsStencil  bool
}

// Proxy returns the owning SurfaceProxy.
func (r *RenderTargetProxy) Proxy() *SurfaceProxy { return r.proxy }

// NumSamples returns the samples/pixel in the color buffer (one if non-MSAA).
func (r *RenderTargetProxy) NumSamples() int { return r.sampleCnt }

// SetNeedsStencil records that a draw to this proxy requires stencil.
func (r *RenderTargetProxy) SetNeedsStencil() { r.needsStencil = true }

// NeedsStencil reports whether some draw to this proxy has required stencil.
func (r *RenderTargetProxy) NeedsStencil() bool { return r.needsStencil }

// GLRTFBOIDIs0 reports whether this render target wraps GL's default framebuffer (FBO 0).
func (r *RenderTargetProxy) GLRTFBOIDIs0() bool {
	return r.proxy.surfaceFlags&gpu.SurfaceFlagGLRTFBOIDIs0 != 0
}

// MaxWindowRectangles returns how many window rectangles this render target supports (zero for FBO 0).
func (r *RenderTargetProxy) MaxWindowRectangles(caps *Caps) int {
	if r.GLRTFBOIDIs0() {
		return 0
	}
	return caps.MaxWindowRectanglesCap
}

// CanUseStencil reports whether this proxy has a stencil attachment already or one can be attached during flush.
// Wrapped render targets without stencil return false since their attachments cannot be modified.
func (r *RenderTargetProxy) CanUseStencil(caps *Caps) bool {
	if caps.AvoidStencilBuffers {
		return false
	}
	if !r.proxy.IsInstantiated() {
		if r.proxy.IsLazy() {
			// It's possible for wrapped GL render targets to not have stencil. We don't have an exact way of knowing
			// whether the target will be able to use stencil, so do the best we can: if a lazy GL proxy doesn't have a
			// texture it might be a wrapped target without stencil, so conservatively block stencil.
			return r.proxy.AsTextureProxy() != nil
		}
		// Otherwise the target will definitely not be wrapped, so a stencil can be attached freely on this internal
		// render target.
		return true
	}
	// Just ask the actual target if we can use stencil.
	rt := r.proxy.PeekRenderTarget()
	useMSAASurface := rt.NumSamples() > 1
	return rt.StencilAttachment(useMSAASurface) != nil ||
		rt.CanAttemptStencilAttachment(useMSAASurface)
}

// MarkMSAADirty records that the given region of the MSAA buffer needs resolving.
func (r *RenderTargetProxy) MarkMSAADirty(dirtyRect geom.IRect) {
	if !r.proxy.RequiresManualMSAAResolve() {
		panic("markMSAADirty without manual MSAA resolve")
	}
	if !r.proxy.BackingStoreBoundsIRect().ContainsRect(dirtyRect) {
		panic("MSAA dirty rect outside backing bounds")
	}
	r.msaaDirtyRect.Join(dirtyRect)
}

// MarkMSAAResolved clears the pending MSAA-resolve region.
func (r *RenderTargetProxy) MarkMSAAResolved() {
	if !r.proxy.RequiresManualMSAAResolve() {
		panic("markMSAAResolved without manual MSAA resolve")
	}
	r.msaaDirtyRect = geom.IRect{}
}

// IsMSAADirty reports whether any region of the MSAA buffer is pending resolve.
func (r *RenderTargetProxy) IsMSAADirty() bool {
	return r.proxy.RequiresManualMSAAResolve() && !r.msaaDirtyRect.IsEmpty()
}

// MSAADirtyRect returns the region of the MSAA buffer pending resolve.
func (r *RenderTargetProxy) MSAADirtyRect() geom.IRect { return r.msaaDirtyRect }

// RefsWrappedObjects reports whether the instantiated target wraps externally-owned GL objects.
func (r *RenderTargetProxy) RefsWrappedObjects() bool {
	if !r.proxy.IsInstantiated() {
		return false
	}
	return r.proxy.PeekSurface().RefsWrappedObjects()
}

// Arenas returns the render target's frame arenas, created and held with the first request; each OpsTask shares the
// same arenas until ClearArenas.
func (r *RenderTargetProxy) Arenas() *Arenas {
	if r.arenas == nil {
		r.arenas = &Arenas{}
	}
	return r.arenas
}

// ClearArenas flushes and drops the render target's frame arenas.
func (r *RenderTargetProxy) ClearArenas() {
	if r.arenas != nil {
		r.arenas.Flush()
	}
	r.arenas = nil
}

// addTextureFacet attaches the texture facet's state to a SurfaceProxy under construction.
func (p *SurfaceProxy) addTextureFacet(mipmapped gpu.Mipmapped, mipmapStatus gpu.MipmapStatus) *SurfaceProxy {
	if p.surfaceFlags&gpu.SurfaceFlagFramebufferOnly != 0 {
		panic("texture proxies cannot be framebuffer-only")
	}
	if (mipmapped == gpu.MipmappedNo) != (mipmapStatus == gpu.MipmapStatusNotAllocated) {
		panic("mipmapped/mipmapStatus mismatch")
	}
	p.tex = &TextureProxy{
		proxy:         p,
		mipmapped:     mipmapped,
		mipmapStatus:  mipmapStatus,
		syncTargetKey: true,
	}
	return p
}

// addRenderTargetFacet attaches the render-target facet's state to a SurfaceProxy under construction.
func (p *SurfaceProxy) addRenderTargetFacet(sampleCnt int) *SurfaceProxy {
	p.rt = &RenderTargetProxy{proxy: p, sampleCnt: sampleCnt}
	return p
}

// initTexRTSurfaceFlags runs for deferred and lazy texture-render-target proxies (wrapped ones copy their flags from
// the target instead): MSAA texture-render-targets always require manual resolve when resolves are not automatic. This
// is the only place the manual-resolve flag is set on a proxy — any other surface requiring it will be wrapped, and the
// wrapped constructor copies the target's flags.
func (p *SurfaceProxy) initTexRTSurfaceFlags(caps *Caps) {
	if p.rt.GLRTFBOIDIs0() {
		panic("FBO 0 should never be wrapped as a texture render target")
	}
	if p.rt.sampleCnt > 1 && !caps.MSAAResolvesAutomatically {
		p.surfaceFlags |= gpu.SurfaceFlagRequiresManualMSAAResolve
	}
}
