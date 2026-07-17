// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The tracking object handed to render tasks while flushing — the gpu/provider back-pointers, the active render pass,
// per-op args, the sampled-proxy array hook, the vertex/index buffer pools, and the deferred-upload-target surface: the
// token tracker plus the ASAP/inline upload queues that the draw-op atlas schedules its texture writes through. Ops
// issue their GL draws directly in OnExecute instead of recording mesh-draw entries for a central replay, so the
// draw/upload token sequencing is exposed as explicit methods — an atlas-consuming op calls IssueDrawToken per draw
// while preparing, and at execution calls ProcessPendingInlineUploads before and IssueFlushToken after each of those
// draws, in the same order. ASAP uploads run in PreExecuteDraws, and executeRenderTasks checks the end-of-flush
// draw/flush token balance.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/text"
)

// OpArgs is the per-op information available while an op is being prepared and executed. surfaceView is held by value:
// a pointer field would make the prepare/execute loops' per-task dstView local escape to the heap when SetOpArgs copies
// the args into the long-lived OpFlushState. Readers get a pointer into the flush state's copy via WriteView, which
// nothing retains past the op call (ProgramInfo and the pipelines copy the fields they need).
type OpArgs struct {
	op                     Op
	appliedClip            *AppliedClip
	surfaceView            SurfaceProxyView
	dstProxyView           DstProxyView
	renderPassXferBarriers gpu.XferBarrierFlags
	colorLoadOp            gpu.LoadOp
	usesMSAASurface        bool
}

// Op returns the op these args belong to.
func (a *OpArgs) Op() Op { return a.op }

// WriteView returns the surface view the op writes into.
func (a *OpArgs) WriteView() *SurfaceProxyView { return &a.surfaceView }

// Proxy returns the write view's underlying proxy.
func (a *OpArgs) Proxy() *SurfaceProxy { return a.surfaceView.Proxy() }

// RTProxy returns the write view's proxy as a render-target proxy.
func (a *OpArgs) RTProxy() *RenderTargetProxy { return a.surfaceView.AsRenderTargetProxy() }

// UsesMSAASurface reports whether the op is drawing into an MSAA surface.
func (a *OpArgs) UsesMSAASurface() bool { return a.usesMSAASurface }

// AppliedClip returns the clip applied to this op, if any.
func (a *OpArgs) AppliedClip() *AppliedClip { return a.appliedClip }

// DstProxyView returns the destination-read proxy view, for shaders that sample the destination.
func (a *OpArgs) DstProxyView() *DstProxyView { return &a.dstProxyView }

// RenderPassBarriers returns the transfer barrier flags required around this op's render pass.
func (a *OpArgs) RenderPassBarriers() gpu.XferBarrierFlags { return a.renderPassXferBarriers }

// ColorLoadOp returns how the render pass's color attachment should be loaded for this op.
func (a *OpArgs) ColorLoadOp() gpu.LoadOp { return a.colorLoadOp }

// inlineUpload pairs a deferred texture upload with the draw token it must complete before.
type inlineUpload struct {
	upload            DeferredTextureUploadFn
	uploadBeforeToken gpu.Token
}

// OpFlushState is the tracking object handed to render tasks while flushing (the subset the task DAG and the
// deferred-upload machinery exercise).
type OpFlushState struct {
	vertexPool        *BufferAllocPool
	indexPool         *BufferAllocPool
	threadSafeCache   *ThreadSafeCache
	tokenTracker      *gpu.TokenTracker
	atlasManager      *AtlasManager
	strikeCache       *text.StrikeCache
	resourceProvider  *ResourceProvider
	sampledProxyArray *[]*SurfaceProxy
	gpu               *Gpu
	opsRenderPass     *OpsRenderPass
	asapUploads       []DeferredTextureUploadFn
	inlineUploads     []inlineUpload
	opArgs            OpArgs
	// currUpload is the index of the next inline upload to execute.
	currUpload int
	opArgsSet  bool
}

// NewOpFlushState creates an OpFlushState wired up to the given Gpu, resource provider, token tracker, buffer cache,
// thread-safe cache, atlas manager, and glyph strike cache.
func NewOpFlushState(g *Gpu, resourceProvider *ResourceProvider, tokenTracker *gpu.TokenTracker, cpuBufferCache *CpuBufferCache, threadSafeCache *ThreadSafeCache, atlasManager *AtlasManager, strikeCache *text.StrikeCache) *OpFlushState {
	return &OpFlushState{
		gpu:              g,
		resourceProvider: resourceProvider,
		threadSafeCache:  threadSafeCache,
		tokenTracker:     tokenTracker,
		atlasManager:     atlasManager,
		strikeCache:      strikeCache,
		vertexPool:       NewVertexBufferAllocPool(g, cpuBufferCache, resourceProvider),
		indexPool:        NewIndexBufferAllocPool(g, cpuBufferCache, resourceProvider),
	}
}

// ThreadSafeCache returns the cache shared across threads for resources like precomputed vertex data.
func (s *OpFlushState) ThreadSafeCache() *ThreadSafeCache { return s.threadSafeCache }

// AtlasManager returns the atlas manager used to pack and upload glyphs and other small textures.
func (s *OpFlushState) AtlasManager() *AtlasManager { return s.atlasManager }

// TextStrikeCache returns the glyph strike cache backing glyph-vector resolution during flush.
func (s *OpFlushState) TextStrikeCache() *text.StrikeCache { return s.strikeCache }

// Gpu returns the Gpu backend this flush state is executing against.
func (s *OpFlushState) Gpu() *Gpu { return s.gpu }

// ResourceProvider returns the resource provider used to create and find GPU resources.
func (s *OpFlushState) ResourceProvider() *ResourceProvider { return s.resourceProvider }

// Caps returns the GL capabilities in effect for this flush.
func (s *OpFlushState) Caps() *Caps { return s.gpu.GLCaps() }

// OpsRenderPass returns the currently active render pass, if any.
func (s *OpFlushState) OpsRenderPass() *OpsRenderPass { return s.opsRenderPass }

// SetOpsRenderPass sets the currently active render pass.
func (s *OpFlushState) SetOpsRenderPass(pass *OpsRenderPass) { s.opsRenderPass = pass }

// OpArgs returns the current per-op arguments.
func (s *OpFlushState) OpArgs() *OpArgs {
	if !s.opArgsSet {
		panic("no op args set")
	}
	return &s.opArgs
}

// SetOpArgs stores the OpArgs by value, copying the caller's value in rather than retaining its pointer, so the OpArgs
// the OpsTask prepare/execute loops build for each executing chain doesn't escape to the heap on every op. The pointer
// is only read here, so the caller's local stays on its stack.
func (s *OpFlushState) SetOpArgs(args *OpArgs) {
	if args == nil {
		s.opArgs = OpArgs{}
		s.opArgsSet = false
		return
	}
	s.opArgs = *args
	s.opArgsSet = true
}

// SetSampledProxyArray sets the array that ops append their sampled proxies to, for the drawing manager to track
// cross-task dependencies.
func (s *OpFlushState) SetSampledProxyArray(sampledProxies *[]*SurfaceProxy) {
	s.sampledProxyArray = sampledProxies
}

// SampledProxyArray returns the array that ops append their sampled proxies to.
func (s *OpFlushState) SampledProxyArray() *[]*SurfaceProxy { return s.sampledProxyArray }

// MakeVertexSpace allocates vertexCount vertices of vertexSize bytes from the vertex pool.
func (s *OpFlushState) MakeVertexSpace(vertexSize uint64, vertexCount int) (ptr []byte, buffer AnyBuffer, startVertex int) {
	return s.vertexPool.MakeVertexSpace(vertexSize, vertexCount)
}

// MakeIndexSpace allocates indexCount indices from the index pool.
func (s *OpFlushState) MakeIndexSpace(indexCount int) (ptr []byte, buffer AnyBuffer, startIndex int) {
	return s.indexPool.MakeIndexSpace(indexCount)
}

// MakeVertexSpaceAtLeast allocates at least minVertexCount vertices, falling back to fallbackVertexCount if the pool
// cannot satisfy the request from its current block.
func (s *OpFlushState) MakeVertexSpaceAtLeast(vertexSize uint64, minVertexCount, fallbackVertexCount int) (ptr []byte, buffer AnyBuffer, startVertex, actualVertexCount int) {
	return s.vertexPool.MakeVertexSpaceAtLeast(vertexSize, minVertexCount, fallbackVertexCount)
}

// MakeIndexSpaceAtLeast allocates at least minIndexCount indices, falling back to fallbackIndexCount if the pool cannot
// satisfy the request from its current block.
func (s *OpFlushState) MakeIndexSpaceAtLeast(minIndexCount, fallbackIndexCount int) (ptr []byte, buffer AnyBuffer, startIndex, actualIndexCount int) {
	return s.indexPool.MakeIndexSpaceAtLeast(minIndexCount, fallbackIndexCount)
}

// PutBackVertices returns unused trailing vertices to the vertex pool.
func (s *OpFlushState) PutBackVertices(vertices int, vertexStride uint64) {
	s.vertexPool.PutBack(vertexStride * uint64(vertices))
}

// PutBackIndices returns unused trailing indices to the index pool.
func (s *OpFlushState) PutBackIndices(indices int) { s.indexPool.PutBack(uint64(indices) * 2) }

// PreExecuteDraws unmaps the vertex/index pools so buffers are ready for drawing, performs all the ASAP uploads, and
// resets the inline-upload execution cursor.
func (s *OpFlushState) PreExecuteDraws() {
	s.vertexPool.Unmap()
	s.indexPool.Unmap()
	for _, upload := range s.asapUploads {
		s.DoUpload(upload, false)
	}
	s.currUpload = 0
}

// Reset clears the flush state for reuse on the next flush. It also fully clears the per-op scratch fields (opArgs,
// opsRenderPass, sampledProxyArray) that the prepare/execute loops set and clear again as they go: they are already
// nil/false here on the normal flow (the drawing manager panics if a render pass is still active), but zeroing them
// makes reuse of the persistent flush state (DrawingManager.flushState) safe by construction rather than by relying on
// that flow — no stale pointer can survive into the next flush (cf. the op-pool full-zero recycle).
func (s *OpFlushState) Reset() {
	if s.currUpload != len(s.inlineUploads) {
		panic("inline uploads were recorded but never executed")
	}
	s.vertexPool.Reset()
	s.indexPool.Reset()
	s.asapUploads = nil
	s.inlineUploads = nil
	s.currUpload = 0
	s.opArgs = OpArgs{}
	s.opArgsSet = false
	s.opsRenderPass = nil
	s.sampledProxyArray = nil
}

// TokenTracker returns the token tracker for this flush; callers may only query tokens, never issue them, through this
// accessor. Implements DeferredUploadTarget.
func (s *OpFlushState) TokenTracker() *gpu.TokenTracker { return s.tokenTracker }

// AddInlineUpload schedules upload to run before the next draw token is issued. Implements DeferredUploadTarget.
func (s *OpFlushState) AddInlineUpload(upload DeferredTextureUploadFn) gpu.Token {
	token := s.tokenTracker.NextDrawToken()
	s.inlineUploads = append(s.inlineUploads, inlineUpload{upload: upload, uploadBeforeToken: token})
	return token
}

// AddASAPUpload schedules upload to run as soon as possible, before any draws execute. Implements DeferredUploadTarget.
func (s *OpFlushState) AddASAPUpload(upload DeferredTextureUploadFn) gpu.Token {
	s.asapUploads = append(s.asapUploads, upload)
	return s.tokenTracker.NextFlushToken()
}

// IssueDrawToken issues a draw token: an op that reads the atlas issues one draw token per draw while preparing, and
// must pair it with an IssueFlushToken when that draw executes.
func (s *OpFlushState) IssueDrawToken() gpu.Token { return s.tokenTracker.IssueDrawToken() }

// ProcessPendingInlineUploads executes the scheduled inline uploads that must occur before the draw whose flush token
// is about to be issued. Must be called inside an active render pass, before the draw is issued.
func (s *OpFlushState) ProcessPendingInlineUploads() {
	drawToken := s.tokenTracker.NextFlushToken()
	for s.currUpload < len(s.inlineUploads) &&
		s.inlineUploads[s.currUpload].uploadBeforeToken == drawToken {
		s.opsRenderPass.InlineUpload(s, s.inlineUploads[s.currUpload].upload)
		s.currUpload++
	}
}

// IssueFlushToken issues a flush token, called after each draw that was recorded with IssueDrawToken executes.
func (s *OpFlushState) IssueFlushToken() { s.tokenTracker.IssueFlushToken() }

// DoUpload runs a deferred upload, handing it a writePixels function that pushes the texel data to the proxy's
// instantiated texture. The atlas formats upload as themselves on the desktop matrix, so this repacks row bytes only
// and fails on cross-type requirements (the same trim as SurfaceContext.WritePixels). shouldPrepareSurfaceForSampling
// is unused on this backend, kept for signature parity with other backends.
func (s *OpFlushState) DoUpload(upload DeferredTextureUploadFn, shouldPrepareSurfaceForSampling bool) {
	wp := func(dstProxy *SurfaceProxy, rect geom.IRect, colorType gpu.ColorType, data []byte,
		rowBytes int,
	) bool {
		dstSurface := dstProxy.PeekSurface()
		if dstSurface == nil {
			panic("upload to an uninstantiated proxy")
		}
		if !s.gpu.GLCaps().SurfaceSupportsWritePixels(dstSurface) {
			return false
		}
		supportedWrite := s.gpu.GLCaps().SupportedWritePixelsColorType(colorType,
			dstSurface.Format(), colorType)
		if supportedWrite.ColorType != colorType {
			// Cross-color-type conversion waits for the imagecore integration (unreachable for the atlas formats on the
			// desktop matrix).
			return false
		}
		tightRB := int(rect.Width()) * colorType.BytesPerPixel()
		if rowBytes < tightRB {
			panic("row bytes smaller than the tight row size")
		}
		if !s.gpu.GLCaps().WritePixelsRowBytesSupport && rowBytes != tightRB {
			tmp := make([]byte, int(rect.Height())*tightRB)
			for y := range int(rect.Height()) {
				copy(tmp[y*tightRB:(y+1)*tightRB], data[y*rowBytes:y*rowBytes+tightRB])
			}
			data = tmp
			rowBytes = tightRB
		}
		return s.gpu.WritePixels(dstSurface, rect, colorType, colorType,
			[]gpu.MipLevel{{Pixels: data, RowBytes: rowBytes}}, shouldPrepareSurfaceForSampling)
	}
	upload(wp)
}
