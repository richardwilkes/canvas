// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// TriangulatingPathRenderer linearizes and decomposes a path into triangles using a CPU triangulator, uploads them to a
// vertex buffer, and renders them with a single draw call; coverage AA uses a one-pixel alpha ramp. Non-AA/MSAA
// triangulations of keyed shapes are cached in the ThreadSafeCache as vertex data over a static GPU buffer. There is no
// invalidation listener for source-path changes: stale cached triangulations age out of the LRU instead (same rule as
// the SW path mask cache); keys are never reused, so correctness is unaffected.

package gl

import (
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/surface"
)

// aaTessellatorMaxVerbCount is the default maximum verb count for the coverage-AA triangulation path (see
// TriangulatingPathRenderer.SetMaxVerbCount).
const aaTessellatorMaxVerbCount = 10

// tessInfo is stored as the unique key's custom data: ancillary data not specifically required for the triangle data.
type tessInfo struct {
	numVertices int32
	isLinear    bool
	tolerance   float32
}

const tessInfoSize = 12

func tessInfoEncode(info tessInfo) []byte {
	data := make([]byte, tessInfoSize)
	binary.LittleEndian.PutUint32(data, uint32(info.numVertices))
	if info.isLinear {
		binary.LittleEndian.PutUint32(data[4:], 1)
	}
	binary.LittleEndian.PutUint32(data[8:], math.Float32bits(info.tolerance))
	return data
}

func tessInfoDecode(data []byte) tessInfo {
	return tessInfo{
		numVertices: int32(binary.LittleEndian.Uint32(data)),
		isLinear:    binary.LittleEndian.Uint32(data[4:]) != 0,
		tolerance:   math.Float32frombits(binary.LittleEndian.Uint32(data[8:])),
	}
}

// tessCacheMatch reports whether a cached triangulation's tolerance is fine enough for the requested tolerance tol.
func tessCacheMatch(data []byte, tol float32) bool {
	if data == nil {
		panic("cached triangulation without tess info")
	}
	info := tessInfoDecode(data)
	return info.isLinear || info.tolerance < 3.0*tol
}

// tessIsNewerBetter reports whether 'challenger' should replace 'incumbent' in the cache on a collision.
func tessIsNewerBetter(incumbent, challenger []byte) bool {
	i := tessInfoDecode(incumbent)
	c := tessInfoDecode(challenger)
	if i.isLinear || i.tolerance <= c.tolerance {
		return false // prefer the incumbent
	}
	return true
}

// staticVertexAllocator locks a static GPU vertex buffer (mapped when the caps allow, else via a CPU staging copy
// flushed with UpdateData) and detaches the result as thread-safe-cache vertex data.
type staticVertexAllocator struct {
	vertexData       *VertexData
	vertexBuffer     *Buffer
	resourceProvider *ResourceProvider
	vertices         []byte
	lockStride       uint64
	canMapVB         bool
}

func (a *staticVertexAllocator) lockWriter(stride uint64, eagerCount int) []byte {
	if a.lockStride != 0 || a.vertices != nil || a.vertexBuffer != nil || a.vertexData != nil {
		panic("lock while locked")
	}
	if stride == 0 || eagerCount == 0 {
		panic("invalid lock request")
	}
	size := uint64(eagerCount) * stride
	a.vertexBuffer = a.resourceProvider.CreateBuffer(size, gpu.BufferTypeVertex,
		gpu.AccessPatternStatic, ZeroInitNo)
	if a.vertexBuffer == nil {
		return nil
	}
	if a.canMapVB {
		if ptr := a.vertexBuffer.Map(); ptr != nil {
			a.vertices = unsafe.Slice((*byte)(ptr), size)
		}
	}
	if a.vertices == nil {
		a.vertices = make([]byte, size)
		a.canMapVB = false
	}
	a.lockStride = stride
	return a.vertices
}

func (a *staticVertexAllocator) unlock(actualCount int) {
	if a.lockStride == 0 || a.vertices == nil || a.vertexBuffer == nil || a.vertexData != nil {
		panic("unlock without lock")
	}
	if a.canMapVB {
		a.vertexBuffer.Unmap()
	} else {
		a.vertexBuffer.UpdateData(a.vertices[:uint64(actualCount)*a.lockStride], 0, false)
	}
	a.vertexData = MakeVertexDataFromBuffer(a.vertexBuffer, actualCount, a.lockStride)
	a.vertexBuffer = nil
	a.vertices = nil
	a.lockStride = 0
}

// release drops whatever the allocator is still holding, so an abandoned triangulation doesn't leave the static GPU
// vertex buffer it created pinned in the resource cache by its construction ref. Upstream leans on the allocator's
// destructor for this.
func (a *staticVertexAllocator) release() {
	if a.vertexData != nil {
		a.vertexData.Unref()
		a.vertexData = nil
	}
	if a.vertexBuffer != nil {
		a.vertexBuffer.Unref()
		a.vertexBuffer = nil
	}
	a.vertices = nil
	a.lockStride = 0
}

// detachVertexData returns the locked vertex data and clears the allocator's reference to it.
func (a *staticVertexAllocator) detachVertexData() *VertexData {
	if a.lockStride != 0 || a.vertices != nil || a.vertexBuffer != nil || a.vertexData == nil {
		panic("detach without unlock")
	}
	data := a.vertexData
	a.vertexData = nil
	return data
}

//////////////////////////////////////////////////////////////////////////////
// TriangulatingPathOp

var triangulatingPathOpClassID = GenOpClassID()

// tessPathKeyDomain is the unique-key domain used for triangulatingPathOp "Path" keys.
var tessPathKeyDomain = gpu.GenerateUniqueKeyDomain()

type triangulatingPathOp struct {
	drawOpNoClipToShape
	mesh        *simpleMesh
	programInfo *ProgramInfo
	vertexData  *VertexData
	helper      SimpleMeshDrawOpHelper
	OpBase
	shape         StyledShape
	viewMatrix    geom.Matrix
	color         colorcore.PMColor4f
	devClipBounds geom.IRect
	antiAlias     bool
}

// newTriangulatingPathOp builds a draw op that fills shape by triangulating it. The paint is consumed.
func newTriangulatingPathOp(paint *Paint, shape *StyledShape, viewMatrix *geom.Matrix, devClipBounds geom.IRect, aaType gpu.AAType, stencilSettings *UserStencilSettings) DrawOp {
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}

	o := &triangulatingPathOp{}
	o.InitOp(triangulatingPathOpClassID, o)
	o.helper.InitSimpleMeshDrawOpHelper(processors, aaType, stencilSettings,
		PipelineInputFlagNone)
	o.color = color
	o.shape = shape.clone()
	o.viewMatrix = *viewMatrix
	o.devClipBounds = devClipBounds
	o.antiAlias = aaType == gpu.AATypeCoverage

	devBounds, _ := viewMatrix.MapRect(shape.Bounds())
	if shape.InverseFilled() {
		// Because the clip bounds are used to add a contour for inverse fills, they must also include the path bounds.
		devBounds.Join(devClipBounds.ToRect())
	}
	o.SetBounds(devBounds, HasAABloat(o.antiAlias), IsHairline(false))
	return o
}

// Name implements Op.
func (o *triangulatingPathOp) Name() string { return "TriangulatingPathOp" }

// VisitProxies implements Op.
func (o *triangulatingPathOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *triangulatingPathOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *triangulatingPathOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *triangulatingPathOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	coverage := AnalysisCoverageNone
	if o.antiAlias {
		coverage = AnalysisCoverageSingleChannel
	}
	// This op uses uniform (not vertex) color, so doesn't need to track wide color.
	return o.helper.FinalizeProcessorsWithColor(caps, clip, clampType, coverage, &o.color, nil)
}

// createKey builds the unique key this op's triangulation is cached under.
func (o *triangulatingPathOp) createKey(key *gpu.UniqueKey) {
	inverseFill := o.shape.InverseFilled()

	const clipBoundsCnt = 4 // an IRect's four int32 fields
	shapeKeyDataCnt := o.shape.UnstyledKeySize()
	if shapeKeyDataCnt < 0 {
		panic("triangulation key requires a keyable shape")
	}
	builder := gpu.UniqueKeyBuilder(key, tessPathKeyDomain, shapeKeyDataCnt+clipBoundsCnt, "Path")
	data := builder.Slice()
	o.shape.WriteUnstyledKey(data[:shapeKeyDataCnt])
	// For inverse fills the tessellation depends on the clip bounds.
	if inverseFill {
		data[shapeKeyDataCnt] = uint32(o.devClipBounds.Left)
		data[shapeKeyDataCnt+1] = uint32(o.devClipBounds.Top)
		data[shapeKeyDataCnt+2] = uint32(o.devClipBounds.Right)
		data[shapeKeyDataCnt+3] = uint32(o.devClipBounds.Bottom)
	} else {
		data[shapeKeyDataCnt] = 0
		data[shapeKeyDataCnt+1] = 0
		data[shapeKeyDataCnt+2] = 0
		data[shapeKeyDataCnt+3] = 0
	}
	builder.Finish()
}

// triangulate triangulates the shape in its own coordinate space ('tol' has already been mapped back from device
// space).
func (o *triangulatingPathOp) triangulate(allocator eagerVertexAllocator, tol float32, isLinear *bool) int {
	clipBounds := o.devClipBounds.ToRect()

	vmi, ok := o.viewMatrix.Invert()
	if !ok {
		return 0
	}
	clipBounds, _ = vmi.MapRect(clipBounds)

	if o.shape.Style().Applies() {
		panic("style must have been applied before triangulation")
	}
	return triPathToTriangles(o.shape.AsPath(), tol, clipBounds, allocator, isLinear)
}

// createNonAAMesh builds (or retrieves from cache) the vertex mesh for a non-AA fill.
func (o *triangulatingPathOp) createNonAAMesh(target *OpFlushState) {
	if o.antiAlias {
		panic("createNonAAMesh with AA")
	}
	rp := target.ResourceProvider()
	threadSafeCache := target.ThreadSafeCache()

	var key gpu.UniqueKey
	o.createKey(&key)

	tol := scaleToleranceToSrc(pathUtilsDefaultTolerance, &o.viewMatrix, o.shape.Bounds())

	if o.vertexData == nil {
		cachedVerts, data := threadSafeCache.FindVertsWithData(&key)
		if cachedVerts != nil {
			if tessCacheMatch(data, tol) {
				o.vertexData = cachedVerts
			} else {
				cachedVerts.Unref()
			}
		}
	}

	if o.vertexData != nil {
		if o.vertexData.GpuBuffer() == nil {
			buffer := rp.CreateBufferWithData(o.vertexData.Vertices(), o.vertexData.Size(),
				gpu.BufferTypeVertex, gpu.AccessPatternStatic)
			if buffer == nil {
				return
			}
			o.vertexData.SetGpuBuffer(buffer)
		}
		o.mesh = tessCreateMesh(o.vertexData.GpuBuffer(), 0, o.vertexData.NumVertices())
		return
	}

	canMapVB := target.Caps().MapBufferFlags != gpu.NoMapFlags
	allocator := &staticVertexAllocator{resourceProvider: rp, canMapVB: canMapVB}

	var isLinear bool
	vertexCount := o.triangulate(allocator, tol, &isLinear)
	if vertexCount == 0 {
		// The ear clipper can lock vertex space and then emit nothing, which still leaves a static GPU buffer wrapped
		// in the allocator's vertex data; drop it rather than pinning it for the context's lifetime.
		allocator.release()
		return
	}

	o.vertexData = allocator.detachVertexData()

	key.SetCustomData(tessInfoEncode(tessInfo{
		numVertices: int32(vertexCount),
		isLinear:    isLinear, tolerance: tol,
	}))

	o.vertexData.Ref() // the sk_sp copy handed to the cache
	tmpV, _ := threadSafeCache.AddVertsWithData(&key, o.vertexData, tessIsNewerBetter)
	if tmpV != o.vertexData {
		if tmpV.GpuBuffer() != nil {
			panic("colliding cached triangulation unexpectedly has a gpu buffer")
		}
		// Although the different triangulation found in the cache is better, continue with the current one since it is
		// already on the gpu.
	}
	// There is no destruction hook to purge the entry when the source path dies, so it ages out of the LRU instead.
	tmpV.Unref()

	o.mesh = tessCreateMesh(o.vertexData.GpuBuffer(), 0, o.vertexData.NumVertices())
}

// createAAMesh builds the vertex mesh for a coverage-AA fill.
func (o *triangulatingPathOp) createAAMesh(target *OpFlushState) {
	if o.vertexData != nil || !o.antiAlias {
		panic("createAAMesh preconditions violated")
	}
	if o.shape.Style().Applies() {
		panic("style must have been applied before triangulation")
	}
	// AsPath's result is always freshly owned (it clones the backing path), so no second Clone is needed before the
	// in-place Transform (B.3).
	devPath := o.shape.AsPath()
	devPath.Transform(&o.viewMatrix)
	if devPath.IsEmpty() {
		return
	}
	clipBounds := o.devClipBounds.ToRect()
	tol := pathUtilsDefaultTolerance
	allocator := newEagerDynamicVertexAllocator(target)
	vertexCount := triPathToAATriangles(devPath, tol, clipBounds, allocator)
	if vertexCount == 0 {
		return
	}
	o.mesh = &simpleMesh{}
	o.mesh.vertexBuffer = allocator.buffer
	o.mesh.vertexCount = vertexCount
	o.mesh.baseVertex = allocator.firstVertex
	o.mesh.kind = simpleMeshNonIndexed
}

// tessCreateMesh builds a non-indexed simpleMesh over vb starting at firstVertex.
func tessCreateMesh(vb AnyBuffer, firstVertex, count int) *simpleMesh {
	mesh := &simpleMesh{}
	mesh.vertexBuffer = vb
	mesh.vertexCount = count
	mesh.baseVertex = firstVertex
	mesh.kind = simpleMeshNonIndexed
	return mesh
}

// createProgramInfo builds the program info needed to execute this op's draw.
func (o *triangulatingPathOp) createProgramInfo(state *OpFlushState) {
	localCoordsType := LocalCoordsTypeUnused
	if o.helper.UsesLocalCoords() {
		localCoordsType = LocalCoordsTypeUsePosition
	}
	var coverageType DefaultGeoProcCoverageType
	if o.antiAlias {
		if o.helper.CompatibleWithCoverageAsAlpha() {
			coverageType = CoverageTypeAttributeTweakAlpha
		} else {
			coverageType = CoverageTypeAttribute
		}
	} else {
		coverageType = CoverageTypeSolid
	}
	var gp GeometryProcessor
	if o.antiAlias {
		gp = MakeDefaultGeoProcForDeviceSpace(ColorTypePremulUniform, o.color, coverageType, 0xff,
			localCoordsType, nil, &o.viewMatrix)
	} else {
		gp = MakeDefaultGeoProc(ColorTypePremulUniform, o.color, coverageType, 0xff,
			localCoordsType, nil, &o.viewMatrix)
	}
	if gp == nil {
		return
	}
	expectedStride := 8
	if o.antiAlias {
		expectedStride += 4
	}
	if gp.gpBase().VertexStride() != expectedStride {
		panic("unexpected vertex stride")
	}
	args := state.OpArgs()
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op.
func (o *triangulatingPathOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
	}
	if o.antiAlias {
		o.createAAMesh(state)
	} else {
		o.createNonAAMesh(state)
	}
}

// releaseVertexData drops the ref taken on the cached vertex data in createNonAAMesh, if it is still held.
func (o *triangulatingPathOp) releaseVertexData() {
	if o.vertexData != nil {
		o.vertexData.Unref()
		o.vertexData = nil
	}
}

// recycle implements Op. This op has no free list, but the ref createNonAAMesh takes on the ThreadSafeCache's vertex
// data must be dropped even when the op is never executed — a prepared task can be abandoned before its execute loop
// (a failed AttachStencilAttachment, or a target that lost its instantiation between the prepare and execute passes),
// and without this the cache entry could never become uniquely held, so DropUniqueRefs could never purge it and the GPU
// vertex buffer would stay pinned for the context's lifetime. deleteOps calls this unconditionally, matching upstream's
// release from the op's destructor.
func (o *triangulatingPathOp) recycle() { o.releaseVertexData() }

// OnExecute implements Op. It explicitly releases the op's vertex-data ref once the draw has been submitted.
func (o *triangulatingPathOp) OnExecute(state *OpFlushState, chainBounds geom.Rect) {
	defer o.releaseVertexData()
	if o.programInfo == nil || o.mesh == nil {
		return
	}
	renderPass := state.OpsRenderPass()
	if !renderPass.BindPipeline(o.programInfo, chainBounds) {
		return
	}
	if o.programInfo.Pipeline().IsScissorTestEnabled() {
		renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
	}
	renderPass.BindTextures(o.programInfo.GeomProc(), nil, o.programInfo.Pipeline())
	o.mesh.draw(renderPass)
}

//////////////////////////////////////////////////////////////////////////////
// TriangulatingPathRenderer

// TriangulatingPathRenderer draws paths by triangulating them on the CPU.
type TriangulatingPathRenderer struct {
	maxVerbCount int
}

// NewTriangulatingPathRenderer returns a new TriangulatingPathRenderer.
func NewTriangulatingPathRenderer() *TriangulatingPathRenderer {
	return &TriangulatingPathRenderer{maxVerbCount: aaTessellatorMaxVerbCount}
}

// Name implements PathRenderer.
func (r *TriangulatingPathRenderer) Name() string { return "Triangulating" }

// SetMaxVerbCount overrides the maximum verb count for coverage-AA triangulation (testing only).
func (r *TriangulatingPathRenderer) SetMaxVerbCount(maxVerbCount int) {
	r.maxVerbCount = maxVerbCount
}

// OnGetStencilSupport implements PathRenderer: this renderer never uses the stencil buffer.
func (r *TriangulatingPathRenderer) OnGetStencilSupport(*StyledShape) StencilSupport {
	return StencilSupportNone
}

// OnCanDrawPath implements PathRenderer.
func (r *TriangulatingPathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	// Don't use this path renderer with dynamic MSAA. DMSAA tries to not rely on caching.
	if args.SurfaceProps.Flags&surface.DynamicMSAAFlag != 0 {
		return CanDrawPathNo
	}
	// This path renderer can draw fill styles, and can do screenspace antialiasing via a one-pixel coverage ramp. It
	// can do convex and concave paths, but the convex ones are left to simpler algorithms. Styled paths are passed on,
	// though they may come back around after the styling information is applied to the geometry to create a filled
	// path.
	if !args.Shape.Style().IsSimpleFill() || args.Shape.KnownToBeConvex() {
		return CanDrawPathNo
	}
	verbCount := args.Shape.VerbCount()
	// Don't use this path renderer if we exceed the max verb count.
	if verbCount > pathRendererMaxVerbs {
		return CanDrawPathNo
	}
	switch args.AAType {
	case gpu.AATypeNone, gpu.AATypeMSAA:
		// Prefer MSAA, if any antialiasing. In the non-analytic-AA case, paths without a key are skipped, since the
		// real advantage of this path renderer comes from caching the tessellated geometry.
		if !args.Shape.HasUnstyledKey() {
			return CanDrawPathNo
		}
	case gpu.AATypeCoverage:
		// Use analytic AA if we don't have MSAA. In this case there is no caching, so paths without keys are accepted.
		if verbCount > r.maxVerbCount {
			return CanDrawPathNo
		}
	}
	return CanDrawPathYes
}

// OnDrawPath implements PathRenderer.
func (r *TriangulatingPathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	op := newTriangulatingPathOp(args.Paint, args.Shape, args.ViewMatrix,
		args.ClipConservativeBounds, args.AAType, args.UserStencilSettings)
	args.SurfaceDrawContext.AddDrawOp(args.Clip, op)
	return true
}

// OnStencilPath implements PathRenderer (never reached: OnGetStencilSupport always reports none).
func (r *TriangulatingPathRenderer) OnStencilPath(*StencilPathArgs) {
	panic("TriangulatingPathRenderer cannot stencil paths")
}
