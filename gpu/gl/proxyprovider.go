// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The factory for surface proxies and the registry of uniquely keyed ones. Trims: the deferred display list provider
// split is gone (this provider always belongs to a direct context, so that plumbing and orphanAllUniqueKeys are
// dropped); bitmap-backed, compressed, promise, and wrapped-backend-texture proxies are not implemented. The
// uniquely-keyed proxy map holds the proxy alive, so an entry lives until its key is removed/invalidated or the map is
// cleared — entries can outlive external refs.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// TextureInfo describes texture properties for lazy render-target proxies that are also textureable (nil means a
// non-textureable render target).
type TextureInfo struct {
	Mipmapped   gpu.Mipmapped
	TextureType gpu.TextureType
}

// InvalidateGPUResource selects whether ProcessInvalidUniqueKey also invalidates the underlying GPU resource, not just
// the proxy's key.
type InvalidateGPUResource bool

// InvalidateGPUResource values.
const (
	InvalidateGPUResourceNo  InvalidateGPUResource = false
	InvalidateGPUResourceYes InvalidateGPUResource = true
)

// ProxyProvider is the factory for surface proxies and the registry of uniquely keyed ones.
type ProxyProvider struct {
	ctx *DirectContext
	// uniquelyKeyedProxies holds the texture proxies that have unique keys, keyed on the unique key's byte image.
	uniquelyKeyedProxies map[string]*SurfaceProxy
}

// NewProxyProvider builds a proxy provider for the given direct context.
func NewProxyProvider(ctx *DirectContext) *ProxyProvider {
	return &ProxyProvider{ctx: ctx, uniquelyKeyedProxies: make(map[string]*SurfaceProxy)}
}

// Caps returns the context's caps.
func (pp *ProxyProvider) Caps() *Caps { return pp.ctx.GLCaps() }

// IsAbandoned reports whether the owning context has been abandoned.
func (pp *ProxyProvider) IsAbandoned() bool { return pp.ctx.Abandoned() }

// resourceProvider returns the context's resource provider (always available: direct-only).
func (pp *ProxyProvider) resourceProvider() *ResourceProvider { return pp.ctx.ResourceProvider() }

// AssignUniqueKeyToProxy gives proxy a unique key and registers it in the provider's map. It is an error if the proxy
// already has a key.
func (pp *ProxyProvider) AssignUniqueKeyToProxy(key *gpu.UniqueKey, proxy *TextureProxy) bool {
	if !key.IsValid() {
		panic("assignUniqueKeyToProxy requires a valid key")
	}
	if pp.IsAbandoned() || proxy == nil {
		return false
	}
	if _, exists := pp.uniquelyKeyedProxies[key.MapKey()]; exists {
		panic("multiple proxies can't get the same key")
	}
	proxy.setUniqueKey(pp, key)
	pp.uniquelyKeyedProxies[key.MapKey()] = proxy.Proxy()
	return true
}

// AdoptUniqueKeyFromSurface sets the proxy's unique key to the surface's (which must be valid).
func (pp *ProxyProvider) AdoptUniqueKeyFromSurface(proxy *TextureProxy, surf *Surface) {
	key := surf.UniqueKey()
	if !key.IsValid() {
		panic("adoptUniqueKeyFromSurface requires a keyed surface")
	}
	if _, exists := pp.uniquelyKeyedProxies[key.MapKey()]; exists {
		panic("multiple proxies can't get the same key")
	}
	proxy.setUniqueKey(pp, key)
	pp.uniquelyKeyedProxies[key.MapKey()] = proxy.Proxy()
}

// RemoveUniqueKeyFromProxy clears proxy's unique key and invalidates the underlying resource.
func (pp *ProxyProvider) RemoveUniqueKeyFromProxy(proxy *TextureProxy) {
	if proxy == nil || !proxy.uniqueKey.IsValid() {
		panic("removeUniqueKeyFromProxy requires a keyed proxy")
	}
	if pp.IsAbandoned() {
		return
	}
	pp.ProcessInvalidUniqueKey(&proxy.uniqueKey, proxy, InvalidateGPUResourceYes)
}

// FindProxyByUniqueKey returns the registered proxy for key, or nil.
func (pp *ProxyProvider) FindProxyByUniqueKey(key *gpu.UniqueKey) *SurfaceProxy {
	if pp.IsAbandoned() {
		return nil
	}
	return pp.uniquelyKeyedProxies[key.MapKey()]
}

// FindOrCreateProxyByUniqueKey finds a proxy by unique key, or creates a new one wrapping the cached resource matching
// the key.
func (pp *ProxyProvider) FindOrCreateProxyByUniqueKey(key *gpu.UniqueKey, useAllocator UseAllocator) *SurfaceProxy {
	if pp.IsAbandoned() {
		return nil
	}
	if result := pp.FindProxyByUniqueKey(key); result != nil {
		return result
	}
	resource := pp.ctx.ResourceCache().FindAndRefUniqueResource(key)
	if resource == nil {
		return nil
	}
	texture, ok := resource.(*Surface)
	if !ok || texture.AsTexture() == nil {
		panic("uniquely keyed resource is not a texture")
	}
	result := pp.createWrapped(texture, useAllocator)
	if !result.UniqueKey().Equal(key) {
		panic("createWrapped failed to adopt the unique key")
	}
	return result
}

// createWrapped builds a proxy backed by an already-instantiated texture (adopting its unique key, if any).
func (pp *ProxyProvider) createWrapped(tex *Surface, useAllocator UseAllocator) *SurfaceProxy {
	proxy := newWrappedSurfaceProxy(tex, gpu.BackingFitExact, useAllocator)
	proxy.addTextureFacet(tex.AsTexture().Mipmapped(), tex.AsTexture().MipmapStatus())
	if rt := tex.AsRenderTarget(); rt != nil {
		proxy.addRenderTargetFacet(rt.NumSamples())
	}
	// Adopt the surface's unique key when present.
	if tex.UniqueKey().IsValid() {
		pp.AdoptUniqueKeyFromSurface(proxy.AsTextureProxy(), tex)
	}
	return proxy
}

// CreateProxy builds a deferred proxy without data.
func (pp *ProxyProvider) CreateProxy(format Format, dims geom.ISize, renderable gpu.Renderable, renderTargetSampleCnt int, mipmapped gpu.Mipmapped, fit gpu.BackingFit, budgeted gpu.Budgeted, label string, surfaceFlags gpu.SurfaceFlags, useAllocator UseAllocator) *SurfaceProxy {
	if pp.IsAbandoned() {
		return nil
	}
	caps := pp.Caps()

	if mipmapped == gpu.MipmappedYes {
		// The mip level count doesn't include the base level, so add 1.
		if gpu.ComputeMipLevelCount(dims.Width, dims.Height)+1 == 1 {
			mipmapped = gpu.MipmappedNo
		}
	}
	if !caps.ValidateSurfaceParams(dims, format, bool(renderable), renderTargetSampleCnt,
		bool(mipmapped), gpu.TextureType2D) {
		return nil
	}
	mipmapStatus := gpu.MipmapStatusNotAllocated
	if mipmapped == gpu.MipmappedYes {
		mipmapStatus = gpu.MipmapStatusDirty
	}
	if renderable == gpu.RenderableYes {
		renderTargetSampleCnt = caps.GetRenderTargetSampleCount(renderTargetSampleCnt, format)
		// Anything instantiated later from this deferred path will be both texturable and renderable.
		proxy := newDeferredSurfaceProxy(format, gpu.TextureType2D, dims, fit, budgeted,
			surfaceFlags, useAllocator, label)
		proxy.addTextureFacet(mipmapped, mipmapStatus)
		proxy.addRenderTargetFacet(renderTargetSampleCnt)
		proxy.initTexRTSurfaceFlags(caps)
		return proxy
	}
	proxy := newDeferredSurfaceProxy(format, gpu.TextureType2D, dims, fit, budgeted, surfaceFlags,
		useAllocator, label)
	proxy.addTextureFacet(mipmapped, mipmapStatus)
	return proxy
}

// CreateLazyProxy builds a texture proxy instantiated by a user-supplied callback during flush. Dimensions must either
// both be positive or both non-positive (unknown until instantiation). The texture will not be renderable.
func (pp *ProxyProvider) CreateLazyProxy(callback LazyInstantiateCallback, format Format, dims geom.ISize, mipmapped gpu.Mipmapped, mipmapStatus gpu.MipmapStatus, surfaceFlags gpu.SurfaceFlags, fit gpu.BackingFit, budgeted gpu.Budgeted, useAllocator UseAllocator, label string) *SurfaceProxy {
	if pp.IsAbandoned() {
		return nil
	}
	if dims.Width > int32(pp.Caps().MaxTextureSize) || dims.Height > int32(pp.Caps().MaxTextureSize) {
		return nil
	}
	if dims.Width <= 0 || dims.Height <= 0 {
		dims = geom.ISize{Width: -1, Height: -1}
	}
	proxy := newLazySurfaceProxy(callback, format, gpu.TextureType2D, dims, fit, budgeted,
		surfaceFlags, useAllocator, label)
	proxy.addTextureFacet(mipmapped, mipmapStatus)
	return proxy
}

// CreateLazyRenderTargetProxy builds a render-target proxy instantiated by a user-supplied callback during flush. A nil
// TextureInfo indicates a non-textureable render target.
func (pp *ProxyProvider) CreateLazyRenderTargetProxy(callback LazyInstantiateCallback, format Format, dims geom.ISize, sampleCnt int, surfaceFlags gpu.SurfaceFlags, textureInfo *TextureInfo, mipmapStatus gpu.MipmapStatus, fit gpu.BackingFit, budgeted gpu.Budgeted, useAllocator UseAllocator) *SurfaceProxy {
	if pp.IsAbandoned() {
		return nil
	}
	maxSize := int32(pp.Caps().MaxRenderTargetSize)
	if dims.Width > maxSize || dims.Height > maxSize {
		return nil
	}
	if dims.Width <= 0 || dims.Height <= 0 {
		dims = geom.ISize{Width: -1, Height: -1}
	}
	if textureInfo != nil {
		proxy := newLazySurfaceProxy(callback, format, textureInfo.TextureType, dims, fit,
			budgeted, surfaceFlags, useAllocator,
			"TextureRenderTarget_LazyRenderTargetProxy")
		mipmapped := textureInfo.Mipmapped
		texMipmapStatus := mipmapStatus
		if mipmapped == gpu.MipmappedNo {
			texMipmapStatus = gpu.MipmapStatusNotAllocated
		}
		proxy.addTextureFacet(mipmapped, texMipmapStatus)
		proxy.addRenderTargetFacet(sampleCnt)
		proxy.initTexRTSurfaceFlags(pp.Caps())
		return proxy
	}
	proxy := newLazySurfaceProxy(callback, format, gpu.TextureTypeNone, dims, fit, budgeted,
		surfaceFlags, useAllocator, "RenderTargetProxy_LazyRenderTargetProxy")
	proxy.addRenderTargetFacet(sampleCnt)
	return proxy
}

// MakeFullyLazyProxy builds a lazy proxy whose width and height are unspecified until instantiation.
func MakeFullyLazyProxy(callback LazyInstantiateCallback, format Format, renderable gpu.Renderable, renderTargetSampleCnt int, caps *Caps, useAllocator UseAllocator) *SurfaceProxy {
	if renderTargetSampleCnt != 1 && renderable != gpu.RenderableYes {
		panic("sample count requires a renderable proxy")
	}
	lazyDims := geom.ISize{Width: -1, Height: -1}
	if renderable == gpu.RenderableYes {
		proxy := newLazySurfaceProxy(callback, format, gpu.TextureType2D, lazyDims,
			gpu.BackingFitApprox, gpu.BudgetedYes, 0, useAllocator,
			"TextureRenderTarget_FullyLazyProxy")
		proxy.addTextureFacet(gpu.MipmappedNo, gpu.MipmapStatusNotAllocated)
		proxy.addRenderTargetFacet(renderTargetSampleCnt)
		proxy.initTexRTSurfaceFlags(caps)
		return proxy
	}
	proxy := newLazySurfaceProxy(callback, format, gpu.TextureType2D, lazyDims,
		gpu.BackingFitApprox, gpu.BudgetedYes, 0, useAllocator, "Texture_FullyLazyProxy")
	proxy.addTextureFacet(gpu.MipmappedNo, gpu.MipmapStatusNotAllocated)
	return proxy
}

// WrapBackendRenderTarget wraps a backend render target description (dimensions, format, sample count, stencil bits,
// FBO id) as an instantiated, non-textureable render target proxy that never participates in allocation.
func (pp *ProxyProvider) WrapBackendRenderTarget(dims geom.ISize, format Format, sampleCnt, stencilBits int, fboID uint32) *SurfaceProxy {
	if pp.IsAbandoned() {
		return nil
	}
	rt := pp.resourceProvider().WrapBackendRenderTarget(dims, format, sampleCnt, stencilBits, fboID)
	if rt == nil {
		return nil
	}
	if rt.AsTexture() != nil {
		panic("wrapped backend render target must not be textureable")
	}
	if rt.UniqueKey().IsValid() || rt.BudgetedType() == gpu.BudgetedTypeBudgeted {
		panic("wrapped backend render target must be unkeyed and unbudgeted")
	}
	proxy := newWrappedSurfaceProxy(rt, gpu.BackingFitExact, UseAllocatorNo)
	proxy.addRenderTargetFacet(rt.AsRenderTarget().NumSamples())
	return proxy
}

// ProcessInvalidUniqueKey ensures that, if a proxy with the supplied unique key exists, it is removed from the map and
// its key cleared. When invalidateGPUResource is true the key is also removed from any GPU resource holding it.
func (pp *ProxyProvider) ProcessInvalidUniqueKey(key *gpu.UniqueKey, proxy *TextureProxy, invalidateGPUResource InvalidateGPUResource) {
	if !key.IsValid() {
		panic("processInvalidUniqueKey requires a valid key")
	}
	if proxy == nil {
		if p := pp.uniquelyKeyedProxies[key.MapKey()]; p != nil {
			proxy = p.AsTextureProxy()
		}
	}
	if proxy != nil && !proxy.uniqueKey.Equal(key) {
		panic("proxy key mismatch")
	}

	// Locate the corresponding resource (if it needs invalidation) before clearing the proxy's key, because 'key' may
	// alias the proxy's key.
	var invalidResource gpu.Resource
	if invalidateGPUResource {
		invalidResource = pp.resourceProvider().FindByUniqueKey(key)
	}

	if proxy != nil {
		delete(pp.uniquelyKeyedProxies, key.MapKey())
		proxy.clearUniqueKey()
	}

	if invalidResource != nil {
		r := invalidResource.(interface {
			RemoveUniqueKey()
			Unref()
		})
		r.RemoveUniqueKey()
		r.Unref()
	}
}

// RemoveAllUniqueKeys clears every registered proxy's unique key (used only when abandoning and releasing all
// resources).
func (pp *ProxyProvider) RemoveAllUniqueKeys() {
	for _, proxy := range pp.uniquelyKeyedProxies {
		tex := proxy.AsTextureProxy()
		pp.ProcessInvalidUniqueKey(&tex.uniqueKey, tex, InvalidateGPUResourceNo)
	}
	clear(pp.uniquelyKeyedProxies)
}

// NumUniqueKeyProxies returns the number of registered uniquely-keyed proxies (test-only).
func (pp *ProxyProvider) NumUniqueKeyProxies() int { return len(pp.uniquelyKeyedProxies) }
