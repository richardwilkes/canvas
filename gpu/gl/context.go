// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU context hierarchy flattened to a single DirectContext. Under the trims there are no DDL recorders and only
// one backend, so what would otherwise be several layered classes exist here only as the union of their state: the
// context owns the Gpu, the resource cache/provider pair, the proxy provider, the drawing manager, the atlas manager,
// the text strike cache, and the text-blob redraw coordinator, and carries the lifecycle entry points that are publicly
// reachable. Trims: the thread-safe proxy (cross-thread image contexts), semaphore wait, and the client-mapped buffer
// manager (async reads) are not implemented; the delete-callback and executor options are not publicly reachable.

package gl

import (
	"time"

	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/text"
)

// DirectContext is the flattened top-level GPU context.
type DirectContext struct {
	gpu              *Gpu
	resourceProvider *ResourceProvider
	proxyProvider    *ProxyProvider
	drawingManager   *DrawingManager
	threadSafeCache  *ThreadSafeCache
	atlasManager     *AtlasManager
	textStrikeCache  *text.StrikeCache
	textBlobCache    *text.BlobRedrawCoordinator
	// gradientRampCache is a per-gradient-shader-style bitmap cache hung off the context (one per context, freed with
	// it); see gradientbitmapcache.go.
	gradientRampCache *gradientBitmapCache
	stats             ContextStats
}

// ContextStats holds the counters consumed by this codebase (debug/test-only; these two feed the SW-path-mask cache
// tests).
type ContextStats struct {
	NumPathMasksGenerated int
	NumPathMasksCacheHits int
}

// Stats returns the context's counters.
func (dc *DirectContext) Stats() *ContextStats { return &dc.stats }

// NumGLDraws returns the GL geometry draw calls issued since the last ResetGLDrawStats — the "GL draws issued" side of
// the batching metric. It delegates to the Gpu counter.
func (dc *DirectContext) NumGLDraws() int { return dc.gpu.NumGLDraws() }

// ResetGLDrawStats zeroes the GL-draw-call counter (see DirectContext.NumGLDraws).
func (dc *DirectContext) ResetGLDrawStats() { dc.gpu.ResetGLDrawStats() }

// MakeGLDirectContext builds the context info (validating the interface and deriving caps), the Gpu, and the context
// around it. A GL context must be current on this thread. Returns nil on failure.
func MakeGLDirectContext(iface *Interface, options *gpu.ContextOptions) *DirectContext {
	ctxInfo := NewContextInfo(iface, options)
	if ctxInfo == nil {
		return nil
	}
	g := NewGpu(ctxInfo)
	if g == nil {
		return nil
	}
	return NewDirectContext(g)
}

// NewDirectContext wraps an existing Gpu, finishing initialization (the parts not already performed by NewGpu, which
// owns caps construction and the resource cache).
func NewDirectContext(g *Gpu) *DirectContext {
	dc := &DirectContext{gpu: g}
	dc.resourceProvider = NewResourceProvider(g, g.ResourceCache())
	dc.proxyProvider = NewProxyProvider(dc)
	// The context owns the thread-safe cache and hands it to the resource cache.
	dc.threadSafeCache = NewThreadSafeCache()
	g.ResourceCache().SetThreadSafeCache(dc.threadSafeCache)
	// Reduced ops-task splitting is on unless the driver requires avoiding render-task reordering.
	reduceOpsTaskSplitting := !dc.Caps().AvoidReorderingRenderTasks
	dc.drawingManager = newDrawingManager(dc, reduceOpsTaskSplitting)

	// The atlas manager. Multitexturing is supported only if the shader range can represent the page index + texcoords
	// fully.
	options := g.ContextInfo().Options
	if options == nil {
		options = gpu.DefaultContextOptions()
	}
	allowMultitexturing := AllowMultitexturingYes
	if options.AllowMultipleGlyphCacheTextures == gpu.EnableNo ||
		(!dc.Caps().ShaderCaps.FloatIs32Bits && !dc.Caps().ShaderCaps.IntegerSupport) {
		allowMultitexturing = AllowMultitexturingNo
	}
	dc.atlasManager = NewAtlasManager(dc.proxyProvider, options.GlyphCacheTextureMaximumBytes,
		allowMultitexturing, options.SupportBilerpFromGlyphAtlas)
	dc.drawingManager.AddOnFlushCallbackObject(dc.atlasManager)

	// The text strike cache and the text-blob redraw coordinator.
	dc.textStrikeCache = text.NewStrikeCache()
	dc.textBlobCache = text.NewTextBlobRedrawCoordinator()

	// The gradient-LUT bitmap cache for the textured colorizer (per-context; see gradientbitmapcache.go).
	dc.gradientRampCache = newGradientBitmapCache(maxNumCachedGradientBitmaps, gradientTextureSize)
	return dc
}

// TextStrikeCache returns the text strike cache.
func (dc *DirectContext) TextStrikeCache() *text.StrikeCache { return dc.textStrikeCache }

// TextBlobCache returns the text-blob redraw coordinator.
func (dc *DirectContext) TextBlobCache() *text.BlobRedrawCoordinator {
	return dc.textBlobCache
}

// GetSubRunControl computes whether/where SDFT is used, from the shader caps (derivatives support), the
// perspective-SDFT workaround cap, and the context options' DF size thresholds.
func (dc *DirectContext) GetSubRunControl(useSDFTForSmallText bool) text.SubRunControl {
	options := dc.gpu.ContextInfo().Options
	if options == nil {
		options = gpu.DefaultContextOptions()
	}
	return text.NewSubRunControl(
		dc.Caps().ShaderCaps.SupportsDistanceFieldText(),
		useSDFTForSmallText,
		!dc.Caps().DisablePerspectiveSDFText,
		options.MinDistanceFieldFontSize,
		options.GlyphsAsPathsFontSize,
		false,
	)
}

// AtlasManager returns the glyph atlas manager. Only usable at flush time (via the OpFlushState).
func (dc *DirectContext) AtlasManager() *AtlasManager { return dc.atlasManager }

// Gpu returns the underlying Gpu.
func (dc *DirectContext) Gpu() *Gpu { return dc.gpu }

// Caps returns the backend-independent caps.
func (dc *DirectContext) Caps() *gpu.Caps { return dc.gpu.Caps() }

// GLCaps returns the GL caps (one backend under the trim).
func (dc *DirectContext) GLCaps() *Caps { return dc.gpu.GLCaps() }

// ResourceCache returns the GPU resource cache.
func (dc *DirectContext) ResourceCache() *gpu.ResourceCache { return dc.gpu.ResourceCache() }

// ResourceProvider returns the resource provider.
func (dc *DirectContext) ResourceProvider() *ResourceProvider { return dc.resourceProvider }

// ProxyProvider returns the surface proxy provider.
func (dc *DirectContext) ProxyProvider() *ProxyProvider { return dc.proxyProvider }

// DrawingManager returns the drawing manager.
func (dc *DirectContext) DrawingManager() *DrawingManager { return dc.drawingManager }

// ThreadSafeCache returns the thread-safe resource cache.
func (dc *DirectContext) ThreadSafeCache() *ThreadSafeCache { return dc.threadSafeCache }

// Abandoned reports whether the context has been abandoned (the device-lost probe is a Vulkan concept and is trimmed;
// GL reports lost contexts through OOM/error sweeps instead).
func (dc *DirectContext) Abandoned() bool { return dc.gpu.Abandoned() }

// OOMed reports (and clears) whether an out-of-memory condition was observed.
func (dc *DirectContext) OOMed() bool { return dc.gpu.CheckAndResetOOMed() }

// AbandonContext marks the context abandoned: the backing 3D API context is no longer valid, so backend objects are
// dropped without freeing them.
func (dc *DirectContext) AbandonContext() {
	if dc.Abandoned() {
		return
	}
	// Destroy the drawing manager.
	dc.drawingManager.destroy()

	// We need to make sure all work is finished on the gpu before we start releasing resources — but only if the
	// context is still usable (caps.mustSyncGpuDuringAbandon).
	if dc.Caps().MustSyncGpuDuringAbandon {
		dc.gpu.FinishOutstandingGpuWork()
	}

	dc.textStrikeCache.FreeAll()
	dc.resourceProvider.Abandon()
	// Gpu.AbandonContext disconnects the driver and abandons the cache.
	dc.gpu.AbandonContext()
	dc.atlasManager.FreeAll()
}

// ReleaseResourcesAndAbandonContext works like AbandonContext, but the context is still current, so backend objects are
// freed first.
func (dc *DirectContext) ReleaseResourcesAndAbandonContext() {
	if dc.Abandoned() {
		return
	}
	dc.drawingManager.destroy()

	// We need to make sure all work is finished on the gpu before we start releasing resources.
	dc.gpu.FinishOutstandingGpuWork()

	dc.resourceProvider.Abandon()
	dc.proxyProvider.RemoveAllUniqueKeys()
	// Gpu.ReleaseResourcesAndAbandonContext disconnects the driver and releases the cache.
	dc.gpu.ReleaseResourcesAndAbandonContext()
	dc.atlasManager.FreeAll()
}

// FreeGpuResources flushes then frees every GPU resource this context holds.
func (dc *DirectContext) FreeGpuResources() {
	if dc.Abandoned() {
		return
	}
	dc.FlushAndSubmit(false)
	dc.atlasManager.FreeAll()
	// TODO: the glyph cache doesn't hold any GPU resources so this call should not be needed here; some slack in the
	// text-blob implementation requires it.
	dc.textStrikeCache.FreeAll()
	dc.drawingManager.FreeGpuResources()
	dc.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}

// ResetContext marks that external GL state was modified; re-emit before the next use.
func (dc *DirectContext) ResetContext(state uint32) { dc.gpu.MarkContextDirty(state) }

// ResetGLTextureBindings clears the cached GL texture-binding shadow state.
func (dc *DirectContext) ResetGLTextureBindings() {
	if dc.Abandoned() {
		return
	}
	dc.gpu.ResetTextureBindings()
}

// Flush submits pending work for the whole context, invoking info's callbacks.
func (dc *DirectContext) Flush(info FlushInfo) bool {
	if dc.Abandoned() {
		if info.FinishedProc != nil {
			info.FinishedProc()
		}
		if info.SubmittedProc != nil {
			info.SubmittedProc(false)
		}
		return false
	}
	return dc.drawingManager.FlushSurfaces(nil, info)
}

// Submit submits queued GPU work; with syncCPU it blocks until all GPU work completes.
func (dc *DirectContext) Submit(syncCPU bool) bool {
	if dc.Abandoned() {
		return false
	}
	return dc.drawingManager.SubmitToGpu(syncCPU)
}

// FlushAndSubmit flushes then submits all pending work.
func (dc *DirectContext) FlushAndSubmit(syncCPU bool) {
	dc.Flush(FlushInfo{})
	dc.Submit(syncCPU)
}

// FlushSurfaces is the flush-and-submit path used by surface teardown; proxies lists the surface proxies whose
// MSAA/mipmap state must be resolved after the flush.
func (dc *DirectContext) FlushSurfaces(proxies []*SurfaceProxy, info FlushInfo) bool {
	if dc.Abandoned() {
		if info.FinishedProc != nil {
			info.FinishedProc()
		}
		if info.SubmittedProc != nil {
			info.SubmittedProc(false)
		}
		return false
	}
	return dc.drawingManager.FlushSurfaces(proxies, info)
}

// PurgeUnlockedResources purges unlocked resources per opts.
func (dc *DirectContext) PurgeUnlockedResources(opts gpu.PurgeResourceOptions) {
	if dc.Abandoned() {
		return
	}
	dc.ResourceCache().PurgeUnlockedResources(opts)
	dc.ResourceCache().PurgeAsNeeded()
}

// PerformDeferredCleanup purges resources not used since notUsedSince ago, per opts.
func (dc *DirectContext) PerformDeferredCleanup(notUsedSince time.Duration, opts gpu.PurgeResourceOptions) {
	if dc.Abandoned() {
		return
	}
	dc.gpu.CheckFinishedCallbacks()
	purgeTime := time.Now().Add(-notUsedSince)
	dc.ResourceCache().PurgeAsNeeded()
	dc.ResourceCache().PurgeResourcesNotUsedSince(purgeTime, opts)
}

// Destroy tears down the context, flushing and releasing all resources. The context must be current.
func (dc *DirectContext) Destroy() {
	if !dc.Abandoned() {
		dc.FlushAndSubmit(false)
		// Make sure all work is finished on the GPU before releasing resources.
		dc.gpu.FinishOutstandingGpuWork()
	}
	dc.drawingManager.destroy()
	dc.ResourceCache().ReleaseAll()
}
