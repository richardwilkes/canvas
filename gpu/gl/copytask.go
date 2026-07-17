// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The non-ops tasks of the render task DAG. Copy tasks copy pixels between proxies at flush; resolve tasks batch MSAA
// resolves and mipmap regenerations so they run back-to-back.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// CopyRenderTask copies pixels between surface proxies at flush time.
type CopyRenderTask struct {
	src *SurfaceProxy // ref held; dropped at destruction or when made skippable
	RenderTaskBase
	srcRect geom.IRect
	dstRect geom.IRect
	origin  gpu.SurfaceOrigin
	filter  gpu.FilterMode
}

// MakeCopyRenderTask creates a task that copies srcRect in src to dstRect in dst. The rects must be contained in their
// surfaces; they need not be the same size if the GPU supports scaling while copying. src and dst share a common
// origin.
func MakeCopyRenderTask(drawingMgr *DrawingManager, dst *SurfaceProxy, dstRect geom.IRect, src *SurfaceProxy, srcRect geom.IRect, filter gpu.FilterMode, origin gpu.SurfaceOrigin) *CopyRenderTask {
	if src == nil || dst == nil {
		panic("copy task requires src and dst")
	}
	if !geom.IRectSize(dst.Dimensions()).ContainsRect(dstRect) ||
		!geom.IRectSize(src.Dimensions()).ContainsRect(srcRect) {
		panic("copy rects must be contained in their surfaces (canCopySurface should hold)")
	}
	t := &CopyRenderTask{
		src:     src,
		srcRect: srcRect,
		dstRect: dstRect,
		filter:  filter,
		origin:  origin,
	}
	t.initRenderTask(t)
	src.Ref()
	t.addTarget(drawingMgr, dst)
	return t
}

// Name implements RenderTask.
func (t *CopyRenderTask) Name() string { return "Copy" }

func (t *CopyRenderTask) onDelete() {
	if t.src != nil {
		t.src.Unref()
		t.src = nil
	}
}

// OnMakeSkippable implements RenderTask, resetting the source.
func (t *CopyRenderTask) OnMakeSkippable() {
	if t.src != nil {
		t.src.Unref()
		t.src = nil
	}
}

// OnIsUsed implements RenderTask.
func (t *CopyRenderTask) OnIsUsed(proxy *SurfaceProxy) bool { return proxy == t.src }

// GatherProxyIntervals implements RenderTask.
func (t *CopyRenderTask) GatherProxyIntervals(alloc *ResourceAllocator) {
	if t.src == nil {
		alloc.IncOps()
		return
	}
	// This renderTask doesn't have "normal" ops; create a fake op# to capture the fact that we read the src and copy to
	// the target.
	alloc.AddInterval(t.src, alloc.CurOp(), alloc.CurOp(), ActualUseYes, AllowRecyclingYes)
	alloc.AddInterval(t.Target(0), alloc.CurOp(), alloc.CurOp(), ActualUseYes, AllowRecyclingYes)
	alloc.IncOps()
}

// OnMakeClosed implements RenderTask.
func (t *CopyRenderTask) OnMakeClosed(targetUpdateBounds *geom.IRect) ExpectedOutcome {
	if t.src == nil {
		panic("copy task should not be skippable before being closed")
	}
	*targetUpdateBounds = makeNativeIRect(t.origin, t.Target(0).Height(), t.dstRect)
	return ExpectedOutcomeTargetDirty
}

// OnExecute implements RenderTask.
func (t *CopyRenderTask) OnExecute(flushState *OpFlushState) bool {
	if t.src == nil {
		// Did nothing, just like we're supposed to.
		return true
	}
	dstProxy := t.Target(0)
	if !t.src.IsInstantiated() || !dstProxy.IsInstantiated() {
		return false
	}
	srcSurface := t.src.PeekSurface()
	dstSurface := dstProxy.PeekSurface()
	srcRect := makeNativeIRect(t.origin, srcSurface.Height(), t.srcRect)
	dstRect := makeNativeIRect(t.origin, dstSurface.Height(), t.dstRect)
	return flushState.Gpu().CopySurface(dstSurface, dstRect, srcSurface, srcRect, t.filter)
}

// resolveEntry is one target's pending MSAA-resolve/mipmap-regeneration work.
type resolveEntry struct {
	flags           ResolveFlags
	msaaResolveRect geom.IRect
}

// TextureResolveRenderTask batches MSAA resolves and mipmap regenerations across its targets.
type TextureResolveRenderTask struct {
	resolves []resolveEntry
	RenderTaskBase
}

// newTextureResolveRenderTask creates an empty resolve task.
func newTextureResolveRenderTask() *TextureResolveRenderTask {
	t := &TextureResolveRenderTask{}
	t.initRenderTask(t)
	return t
}

// Name implements RenderTask.
func (t *TextureResolveRenderTask) Name() string { return "TextureResolve" }

// AddProxy registers proxy as needing the given resolve flags, merging with any existing entry.
func (t *TextureResolveRenderTask) AddProxy(drawingMgr *DrawingManager, proxy *SurfaceProxy, flags ResolveFlags, caps *Caps) {
	newFlags := flags
	var resolve *resolveEntry
	newProxy := false

	// We might just need to update the flags for an existing dependency.
	found := -1
	for i := 0; i < t.NumTargets(); i++ {
		if t.Target(i) == proxy {
			found = i
			break
		}
	}
	if found >= 0 {
		resolve = &t.resolves[found]
		newFlags = ^resolve.flags & flags
		resolve.flags |= flags
	} else {
		// The last render task that operated on the proxy must be closed — that's where msaa and mipmaps were marked
		// dirty.
		if last := drawingMgr.GetLastRenderTask(proxy); last != nil && !last.taskBase().IsClosed() {
			panic("resolve target's previous task must be closed")
		}
		if flags == 0 {
			panic("resolve task needs resolve flags")
		}
		t.resolves = append(t.resolves, resolveEntry{flags: flags})
		resolve = &t.resolves[len(t.resolves)-1]
		newProxy = true
	}

	if newFlags&ResolveFlagMSAA != 0 {
		rtProxy := proxy.AsRenderTargetProxy()
		if rtProxy == nil || !rtProxy.IsMSAADirty() {
			panic("MSAA resolve of a clean or non-render-target proxy")
		}
		resolve.msaaResolveRect = rtProxy.MSAADirtyRect()
		rtProxy.MarkMSAAResolved()
	}

	if newFlags&ResolveFlagMipMaps != 0 {
		textureProxy := proxy.AsTextureProxy()
		if textureProxy == nil || textureProxy.Mipmapped() != gpu.MipmappedYes ||
			!textureProxy.MipmapsAreDirty() {
			panic("mipmap regen of a clean or non-mipmapped proxy")
		}
		textureProxy.MarkMipmapsClean()
	}

	// We must do this after updating the proxy state because of assertions that the proxy isn't dirty.
	if newProxy {
		// Add the proxy as a dependency: we will read the existing contents of this texture while generating mipmap
		// levels and/or resolving MSAA.
		t.AddDependency(drawingMgr, proxy, gpu.MipmappedNo, caps)
		t.addTarget(drawingMgr, proxy)
	}
}

// OnIsUsed implements RenderTask.
func (t *TextureResolveRenderTask) OnIsUsed(*SurfaceProxy) bool { return false }

// GatherProxyIntervals implements RenderTask: fake op#'s capture the resolve manipulations.
func (t *TextureResolveRenderTask) GatherProxyIntervals(alloc *ResourceAllocator) {
	fakeOp := alloc.CurOp()
	if len(t.resolves) != t.NumTargets() {
		panic("resolve entries out of sync with targets")
	}
	for i := 0; i < t.NumTargets(); i++ {
		alloc.AddInterval(t.Target(i), fakeOp, fakeOp, ActualUseYes, AllowRecyclingYes)
	}
	alloc.IncOps()
}

// OnMakeClosed implements RenderTask (the resolve task itself does not dirty its targets).
func (t *TextureResolveRenderTask) OnMakeClosed(*geom.IRect) ExpectedOutcome {
	return ExpectedOutcomeTargetUnchanged
}

// OnExecute implements RenderTask.
func (t *TextureResolveRenderTask) OnExecute(flushState *OpFlushState) bool {
	if len(t.resolves) != t.NumTargets() {
		panic("resolve entries out of sync with targets")
	}
	// Resolve all msaa back-to-back, before regenerating mipmaps.
	for i, resolve := range t.resolves {
		if resolve.flags&ResolveFlagMSAA != 0 {
			// peekRenderTarget may be nil if there was an instantiation error.
			if renderTarget := t.Target(i).PeekRenderTarget(); renderTarget != nil {
				flushState.Gpu().ResolveRenderTarget(renderTarget, resolve.msaaResolveRect)
			}
		}
	}
	// Regenerate all mipmaps back-to-back.
	for i, resolve := range t.resolves {
		if resolve.flags&ResolveFlagMipMaps != 0 {
			// peekTexture may be nil if there was an instantiation error.
			texture := t.Target(i).PeekTexture()
			if texture != nil && texture.MipmapsAreDirty() {
				flushState.Gpu().RegenerateMipmapLevels(texture)
				if texture.MipmapsAreDirty() {
					panic("mipmaps still dirty after regeneration")
				}
			}
		}
	}
	return true
}

// FlagsForProxy returns the resolve flags registered for proxy (test access).
func (t *TextureResolveRenderTask) FlagsForProxy(proxy *SurfaceProxy) ResolveFlags {
	for i := 0; i < t.NumTargets(); i++ {
		if t.Target(i) == proxy {
			return t.resolves[i].flags
		}
	}
	return 0
}

// WritePixelsTask records a pixel upload as a task so it stays ordered against draws targeting the same proxy.
type WritePixelsTask struct {
	levels []gpu.MipLevel
	RenderTaskBase
	rect         geom.IRect
	srcColorType gpu.ColorType
	dstColorType gpu.ColorType
}

// MakeWritePixelsTask creates a task uploading texels to dst. The pixel data in levels must remain valid until the task
// executes (callers flush when they don't own the storage).
func MakeWritePixelsTask(drawingMgr *DrawingManager, dst *SurfaceProxy, rect geom.IRect, srcColorType, dstColorType gpu.ColorType, texels []gpu.MipLevel) *WritePixelsTask {
	t := &WritePixelsTask{
		rect:         rect,
		srcColorType: srcColorType,
		dstColorType: dstColorType,
		levels:       append([]gpu.MipLevel(nil), texels...),
	}
	t.initRenderTask(t)
	t.addTarget(drawingMgr, dst)
	return t
}

// Name implements RenderTask.
func (t *WritePixelsTask) Name() string { return "WritePixels" }

// OnIsUsed implements RenderTask.
func (t *WritePixelsTask) OnIsUsed(*SurfaceProxy) bool { return false }

// GatherProxyIntervals implements RenderTask.
func (t *WritePixelsTask) GatherProxyIntervals(alloc *ResourceAllocator) {
	alloc.AddInterval(t.Target(0), alloc.CurOp(), alloc.CurOp(), ActualUseYes, AllowRecyclingYes)
	alloc.IncOps()
}

// OnMakeClosed implements RenderTask.
func (t *WritePixelsTask) OnMakeClosed(targetUpdateBounds *geom.IRect) ExpectedOutcome {
	*targetUpdateBounds = t.rect
	return ExpectedOutcomeTargetDirty
}

// OnExecute implements RenderTask.
func (t *WritePixelsTask) OnExecute(flushState *OpFlushState) bool {
	dstProxy := t.Target(0)
	if !dstProxy.IsInstantiated() {
		return false
	}
	return flushState.Gpu().WritePixels(dstProxy.PeekSurface(), t.rect, t.dstColorType,
		t.srcColorType, t.levels, false)
}
