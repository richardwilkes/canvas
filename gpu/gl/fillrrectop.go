// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The instanced round-rect fill op — unison's most common styled primitive. Each instance carries the rrect's
// normalized-space matrix, radii, local coords, and color; the geometry is a static 40-vertex/90-index coverage mesh
// (inset octagon + linear AA ramps + corner arc pieces) shared through the resource provider's static-buffer cache.
// Adaptations: geom.RRect is uniform-radii only (the full extent of what's publicly reachable for rrects), so the
// per-corner radii attributes are splats and the nine-patch/complex lanes of canUseHWDerivativesWithCoverage are
// unreachable; the arena instance list becomes a slice; prePrepare is dropped with DDLs.

package gl

import (
	"encoding/binary"
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

var fillRRectOpClassID = GenOpClassID()

// fillRRectOpPool is the free list of fillRRectOp shells (see oppool.go).
var fillRRectOpPool opPool[fillRRectOp]

// FillRRectOp processor flags.
const (
	fillRRectFlagUseHWDerivatives uint32 = 1 << iota
	fillRRectFlagHasLocalCoords
	fillRRectFlagWideColor
	fillRRectFlagMSAAEnabled
	fillRRectFlagFakeNonAA
)

const numFillRRectProcessorFlags = 5

// FillRRectLocalCoords is either a local rect or a local matrix.
type FillRRectLocalCoords struct {
	matrix   geom.Matrix
	rect     geom.Rect
	isMatrix bool
}

// LocalCoordsFromRect returns FillRRectLocalCoords carrying a plain local rect.
func LocalCoordsFromRect(localRect geom.Rect) FillRRectLocalCoords {
	return FillRRectLocalCoords{rect: localRect}
}

// LocalCoordsFromMatrix returns FillRRectLocalCoords carrying a local matrix.
func LocalCoordsFromMatrix(localMatrix *geom.Matrix) FillRRectLocalCoords {
	return FillRRectLocalCoords{matrix: *localMatrix, isMatrix: true}
}

// skewsAreRelevant reports whether m's skew components are large enough to matter. Note: just checking m.RectStaysRect
// is not sufficient.
func skewsAreRelevant(m *geom.Matrix) bool {
	if m.HasPerspective() {
		panic("perspective is unreachable here")
	}
	skewX := m.Get(geom.MSkewX)
	skewY := m.Get(geom.MSkewY)
	if skewX == 0 && skewY == 0 {
		return false
	}
	const tol = 1.0 / (1 << 12) // scalar-nearly-zero tolerance
	absScaleX := abs32(m.Get(geom.MScaleX))
	absSkewX := abs32(skewX)
	absScaleY := abs32(m.Get(geom.MScaleY))
	absSkewY := abs32(skewY)
	// The maximum absolute column sum norm of the upper left 2x2.
	norm := max(absScaleX+absSkewY, absSkewX+absScaleY)
	return absSkewX > tol*norm || absSkewY > tol*norm
}

func abs32(v float32) float32 { return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)) }

// fillRRectInstance is one instance's per-draw data within a batched fillRRectOp.
type fillRRectInstance struct {
	viewMatrix  geom.Matrix
	rrect       geom.RRect
	localCoords FillRRectLocalCoords
	color       colorcore.PMColor4f
}

// fillRRectOp draws one or more instanced round-rect fills, batched together when possible.
type fillRRectOp struct {
	instanceBuffer AnyBuffer
	vertexBuffer   *Buffer
	indexBuffer    *Buffer
	programInfo    *ProgramInfo
	helper         SimpleMeshDrawOpHelper
	instances      []fillRRectInstance
	OpBase
	baseInstance int
	// instancesArr backs instances for the common single-instance case so a freshly constructed op does not
	// heap-allocate a separate backing array (see the aaStrokeRectOp.rectsArr note); combining more instances grows
	// past it into the heap as normal. FillRRectOp is unison's most common styled primitive, so this is the
	// highest-frequency of the instance ops.
	instancesArr   [1]fillRRectInstance
	processorFlags uint32
}

// NewFillRRectOp creates an op to fill rrect. The paint is consumed. Returns nil when the op cannot represent the draw
// (the caller falls back on path rendering).
func NewFillRRectOp(caps *Caps, paint *Paint, viewMatrix *geom.Matrix, rrect geom.RRect, localCoords FillRRectLocalCoords, aa gpu.AA) DrawOp {
	if !caps.DrawInstancedSupport {
		return nil
	}

	// We transform into a normalized -1..+1 space to draw the round rect. If the boundaries are too large, the math can
	// overflow. The caller can fall back on path rendering if this is the case.
	if max(rrect.Rect.Height(), rrect.Rect.Width()) >= 1e6 {
		return nil
	}

	var flags uint32
	// Perspective is unsupported.
	if viewMatrix.HasPerspective() {
		return nil
	}
	if canUseHWDerivativesWithCoverage(caps.ShaderCaps, viewMatrix, rrect) {
		// HW derivatives (more specifically, fwidth()) are consistently faster on all platforms in coverage mode. We
		// use them as long as the approximation will be accurate enough.
		flags |= fillRRectFlagUseHWDerivatives
	}
	if aa == gpu.AANo {
		flags |= fillRRectFlagFakeNonAA
	}

	// Only non-trivial paints allocate a processor set.
	color := paint.Color4f()
	var processors *ProcessorSet
	if !paint.IsTrivial() {
		processors = NewProcessorSetFromPaint(paint)
	}
	return newFillRRectOp(processors, color, viewMatrix, rrect, localCoords, flags)
}

// newFillRRectOp creates a fillRRectOp.
func newFillRRectOp(processors *ProcessorSet, paintColor colorcore.PMColor4f, viewMatrix *geom.Matrix, rrect geom.RRect, localCoords FillRRectLocalCoords, flags uint32) *fillRRectOp {
	o := fillRRectOpPool.borrow()
	o.instances = bootstrapInstances(o.instances, o.instancesArr[:0])
	o.InitOp(fillRRectOpClassID, o)
	aaType := gpu.AATypeCoverage // Use analytic AA even if the RT is MSAA.
	if flags&fillRRectFlagFakeNonAA != 0 {
		aaType = gpu.AATypeNone
	}
	o.helper.InitSimpleMeshDrawOpHelper(processors, aaType, nil, PipelineInputFlagNone)
	o.processorFlags = flags &^ (fillRRectFlagHasLocalCoords | fillRRectFlagWideColor |
		fillRRectFlagMSAAEnabled)
	o.instances = append(o.instances, fillRRectInstance{
		viewMatrix:  *viewMatrix,
		rrect:       rrect,
		localCoords: localCoords,
		color:       paintColor,
	})
	bounds, _ := viewMatrix.MapRect(rrect.Rect)
	o.SetBounds(bounds, HasAABloat(flags&fillRRectFlagFakeNonAA == 0), IsHairline(false))
	return o
}

// Name implements Op.
func (o *fillRRectOp) Name() string { return "FillRRectOp" }

// recycle implements Op: returns this dead op to its free list.
func (o *fillRRectOp) recycle() {
	recycleProgramInfo(o.programInfo)
	recycleKeepingBacking(&fillRRectOpPool, o,
		func(o *fillRRectOp) []fillRRectInstance { return o.instances },
		func(o *fillRRectOp, s []fillRRectInstance) { o.instances = s })
}

// NumInstances is test access to the batched instance count.
func (o *fillRRectOp) NumInstances() int { return len(o.instances) }

// ClipToShape implements DrawOp.
func (o *fillRRectOp) ClipToShape(clipOp raster.ClipOp, clipMatrix *geom.Matrix, shape *Shape, aa gpu.AA) ClipResult {
	if len(o.instances) != 1 {
		panic("clipToShape must be called before combining")
	}
	head := &o.instances[0]

	if (shape.IsRect() || shape.IsRRect()) && clipOp == raster.ClipIntersect &&
		(aa == gpu.AANo) == (o.processorFlags&fillRRectFlagFakeNonAA != 0) {
		// The clip shape is a round rect. Attempt to map it to a round rect in "viewMatrix" space.
		var clipRRect geom.RRect
		if clipMatrix.CheapEqual(&head.viewMatrix) {
			if shape.IsRect() {
				clipRRect = geom.MakeRRect(shape.Rect(), 0, 0)
			} else {
				clipRRect = shape.RRect()
			}
		} else {
			// Find a matrix that maps from "clipMatrix" space to "viewMatrix" space.
			if head.viewMatrix.HasPerspective() {
				panic("view matrix cannot have perspective")
			}
			if clipMatrix.HasPerspective() {
				return ClipResultFail
			}
			clipToView, ok := head.viewMatrix.Invert()
			if !ok {
				return ClipResultClippedOut
			}
			clipToView.PreConcat(clipMatrix)
			if clipToView.HasPerspective() {
				panic("clip-to-view cannot have perspective")
			}

			if skewsAreRelevant(&clipToView) {
				// A rect in "clipMatrix" space is not a rect in "viewMatrix" space.
				return ClipResultFail
			}
			clipToView.Set(geom.MSkewX, 0)
			clipToView.Set(geom.MSkewY, 0)
			if !clipToView.RectStaysRect() {
				panic("expected rect-stays-rect after zeroing skews")
			}

			if shape.IsRect() {
				mapped, _ := clipToView.MapRect(shape.Rect())
				clipRRect = geom.MakeRRect(mapped, 0, 0)
			} else {
				rr, ok2 := shape.RRect().Transform(&clipToView)
				if !ok2 {
					// Transforming the rrect failed. This shouldn't generally happen except in cases of fp32 overflow.
					return ClipResultFail
				}
				clipRRect = rr
			}
		}

		// Intersect our round rect with the clip shape.
		var isectRRect geom.RRect
		if head.rrect.Type <= geom.RRectRect && clipRRect.Type <= geom.RRectRect {
			isectRect := head.rrect.Rect
			if !isectRect.Intersect(clipRRect.Rect) {
				return ClipResultClippedOut
			}
			isectRRect = geom.MakeRRect(isectRect, 0, 0)
		} else {
			isectRRect = rrectConservativeIntersect(head.rrect, clipRRect)
			if isectRRect.IsEmpty() {
				// The round rects did not intersect at all or the intersection was too complicated to compute quickly.
				return ClipResultFail
			}
		}

		// Don't apply the clip geometrically if it becomes subpixel, since then the hairline rendering may outset
		// beyond the original clip.
		devISectBounds, _ := head.viewMatrix.MapRect(isectRRect.Rect)
		if devISectBounds.Width() < 1 || devISectBounds.Height() < 1 {
			return ClipResultFail
		}

		if !head.localCoords.isMatrix {
			// Update the local rect.
			rect := head.rrect.Rect
			local := head.localCoords.rect
			isect := isectRRect.Rect
			// rectToLocalSize = (local - local.zwxy) / (rect - rect.zwxy), lane for lane over (L, T, R, B).
			sizeL := (local.Left - local.Right) / (rect.Left - rect.Right)
			sizeT := (local.Top - local.Bottom) / (rect.Top - rect.Bottom)
			sizeR := (local.Right - local.Left) / (rect.Right - rect.Left)
			sizeB := (local.Bottom - local.Top) / (rect.Bottom - rect.Top)
			head.localCoords.rect = geom.Rect{
				Left:   (isect.Left-rect.Left)*sizeL + local.Left,
				Top:    (isect.Top-rect.Top)*sizeT + local.Top,
				Right:  (isect.Right-rect.Right)*sizeR + local.Right,
				Bottom: (isect.Bottom-rect.Bottom)*sizeB + local.Bottom,
			}
		}

		// Update the round rect.
		head.rrect = isectRRect
		return ClipResultClippedGeometrically
	}

	return ClipResultFail
}

// VisitProxies implements Op.
func (o *fillRRectOp) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if o.programInfo != nil {
		o.programInfo.VisitFPProxies(fn)
	} else {
		o.helper.VisitProxies(fn)
	}
}

// UsesMSAA implements DrawOp.
func (o *fillRRectOp) UsesMSAA() bool { return o.helper.UsesMSAA() }

// UsesStencil implements DrawOp.
func (o *fillRRectOp) UsesStencil() bool { return o.helper.UsesStencil() }

// Finalize implements DrawOp.
func (o *fillRRectOp) Finalize(caps *gpu.Caps, clip *AppliedClip, clampType gpu.ClampType) ProcessorAnalysis {
	if len(o.instances) != 1 {
		panic("finalize must run before combining")
	}
	var isWideColor bool
	analysis := o.helper.FinalizeProcessorsWithColor(caps, clip, clampType,
		AnalysisCoverageSingleChannel, &o.instances[0].color, &isWideColor)
	if isWideColor {
		o.processorFlags |= fillRRectFlagWideColor
	}
	if analysis.UsesLocalCoords() {
		o.processorFlags |= fillRRectFlagHasLocalCoords
	}
	return analysis
}

// OnCombineIfPossible implements Op.
func (o *fillRRectOp) OnCombineIfPossible(t Op) CombineResult {
	that, ok := t.(*fillRRectOp)
	if !ok {
		return CombineResultCannotCombine
	}
	if !o.helper.IsCompatible(&that.helper, false) ||
		o.processorFlags != that.processorFlags {
		return CombineResultCannotCombine
	}
	o.instances = append(o.instances, that.instances...)
	return CombineResultMerged
}

// canUseHWDerivativesWithCoverage reports whether shader derivatives can be used for coverage AA on this rrect.
// geom.RRect is uniform-radii, so only the empty/rect/oval/simple lanes are reachable.
func canUseHWDerivativesWithCoverage(shaderCaps *gpu.ShaderCaps, viewMatrix *geom.Matrix, rrect geom.RRect) bool {
	if !shaderCaps.ShaderDerivativeSupport {
		return false
	}
	sx := viewMatrix.Get(geom.MScaleX)
	kx := viewMatrix.Get(geom.MSkewX)
	ky := viewMatrix.Get(geom.MSkewY)
	sy := viewMatrix.Get(geom.MScaleY)
	devScaleX := float32(math.Sqrt(float64(sx*sx + ky*ky)))
	devScaleY := float32(math.Sqrt(float64(kx*kx + sy*sy)))
	switch rrect.Type {
	case geom.RRectEmpty, geom.RRectRect:
		return true
	default:
		return cornerUsableWithHWDerivatives(devScaleX*rrect.RadiusX, devScaleY*rrect.RadiusY)
	}
}

// cornerUsableWithHWDerivatives is the corner form of canUseHWDerivativesWithCoverage: will the corner look good if we
// use HW derivatives?
func cornerUsableWithHWDerivatives(devRadiusX, devRadiusY float32) bool {
	if devRadiusY < devRadiusX {
		devRadiusX, devRadiusY = devRadiusY, devRadiusX
	}
	minDevRadius := max(devRadiusX, 1) // Shader clamps radius at a minimum of 1.
	// Is the gradient smooth enough for this corner to look ok with hardware derivatives? The threshold was arrived at
	// subjectively on an NVIDIA chip.
	return minDevRadius*minDevRadius*5 > devRadiusY
}

//////////////////////////////////////////////////////////////////////////////
// Static geometry.

// This is the offset (when multiplied by radii) from the corners of a bounding box to the vertices of its inscribed
// octagon. We draw the outside portion of arcs with quarter-octagons rather than rectangles.
const octoOffset = float32(1 / (1 + math.Sqrt2/2))

// fillRRectVertexData is the static vertex table: {radiiSelector[4], corner[2], radiusOutset[2], aaBloatDirection[2],
// coverage, isLinearCoverage} per vertex.
var fillRRectVertexData = [40][12]float32{
	// Left inset edge.
	{0, 0, 0, 1, -1, +1, 0, -1, +1, 0, 1, 1},
	{1, 0, 0, 0, -1, -1, 0, +1, +1, 0, 1, 1},
	// Top inset edge.
	{1, 0, 0, 0, -1, -1, +1, 0, 0, +1, 1, 1},
	{0, 1, 0, 0, +1, -1, -1, 0, 0, +1, 1, 1},
	// Right inset edge.
	{0, 1, 0, 0, +1, -1, 0, +1, -1, 0, 1, 1},
	{0, 0, 1, 0, +1, +1, 0, -1, -1, 0, 1, 1},
	// Bottom inset edge.
	{0, 0, 1, 0, +1, +1, -1, 0, 0, -1, 1, 1},
	{0, 0, 0, 1, -1, +1, +1, 0, 0, -1, 1, 1},

	// Left outset edge.
	{0, 0, 0, 1, -1, +1, 0, -1, -1, 0, 0, 1},
	{1, 0, 0, 0, -1, -1, 0, +1, -1, 0, 0, 1},
	// Top outset edge.
	{1, 0, 0, 0, -1, -1, +1, 0, 0, -1, 0, 1},
	{0, 1, 0, 0, +1, -1, -1, 0, 0, -1, 0, 1},
	// Right outset edge.
	{0, 1, 0, 0, +1, -1, 0, +1, +1, 0, 0, 1},
	{0, 0, 1, 0, +1, +1, 0, -1, +1, 0, 0, 1},
	// Bottom outset edge.
	{0, 0, 1, 0, +1, +1, -1, 0, 0, +1, 0, 1},
	{0, 0, 0, 1, -1, +1, +1, 0, 0, +1, 0, 1},

	// Top-left corner.
	{1, 0, 0, 0, -1, -1, 0, +1, -1, 0, 0, 0},
	{1, 0, 0, 0, -1, -1, 0, +1, +1, 0, 1, 0},
	{1, 0, 0, 0, -1, -1, +1, 0, 0, +1, 1, 0},
	{1, 0, 0, 0, -1, -1, +1, 0, 0, -1, 0, 0},
	{1, 0, 0, 0, -1, -1, +octoOffset, 0, -1, -1, 0, 0},
	{1, 0, 0, 0, -1, -1, 0, +octoOffset, -1, -1, 0, 0},

	// Top-right corner.
	{0, 1, 0, 0, +1, -1, -1, 0, 0, -1, 0, 0},
	{0, 1, 0, 0, +1, -1, -1, 0, 0, +1, 1, 0},
	{0, 1, 0, 0, +1, -1, 0, +1, -1, 0, 1, 0},
	{0, 1, 0, 0, +1, -1, 0, +1, +1, 0, 0, 0},
	{0, 1, 0, 0, +1, -1, 0, +octoOffset, +1, -1, 0, 0},
	{0, 1, 0, 0, +1, -1, -octoOffset, 0, +1, -1, 0, 0},

	// Bottom-right corner.
	{0, 0, 1, 0, +1, +1, 0, -1, +1, 0, 0, 0},
	{0, 0, 1, 0, +1, +1, 0, -1, -1, 0, 1, 0},
	{0, 0, 1, 0, +1, +1, -1, 0, 0, -1, 1, 0},
	{0, 0, 1, 0, +1, +1, -1, 0, 0, +1, 0, 0},
	{0, 0, 1, 0, +1, +1, -octoOffset, 0, +1, +1, 0, 0},
	{0, 0, 1, 0, +1, +1, 0, -octoOffset, +1, +1, 0, 0},

	// Bottom-left corner.
	{0, 0, 0, 1, -1, +1, +1, 0, 0, +1, 0, 0},
	{0, 0, 0, 1, -1, +1, +1, 0, 0, -1, 1, 0},
	{0, 0, 0, 1, -1, +1, 0, -1, +1, 0, 1, 0},
	{0, 0, 0, 1, -1, +1, 0, -1, -1, 0, 0, 0},
	{0, 0, 0, 1, -1, +1, 0, -octoOffset, -1, +1, 0, 0},
	{0, 0, 0, 1, -1, +1, +octoOffset, 0, -1, +1, 0, 0},
}

// fillRRectIndexData is the static index table for fillRRectVertexData.
var fillRRectIndexData = []uint16{
	// Inset octagon (solid coverage).
	0, 1, 7,
	1, 2, 7,
	7, 2, 6,
	2, 3, 6,
	6, 3, 5,
	3, 4, 5,

	// AA borders (linear coverage).
	0, 1, 8, 1, 9, 8,
	2, 3, 10, 3, 11, 10,
	4, 5, 12, 5, 13, 12,
	6, 7, 14, 7, 15, 14,

	// Top-left arc.
	16, 17, 21,
	17, 21, 18,
	21, 18, 20,
	18, 20, 19,

	// Top-right arc.
	22, 23, 27,
	23, 27, 24,
	27, 24, 26,
	24, 26, 25,

	// Bottom-right arc.
	28, 29, 33,
	29, 33, 30,
	33, 30, 32,
	30, 32, 31,

	// Bottom-left arc.
	34, 35, 39,
	35, 39, 36,
	39, 36, 38,
	36, 38, 37,
}

var (
	fillRRectBufferKeyOnce sync.Once
	fillRRectVertexKey     gpu.UniqueKey
	fillRRectIndexKey      gpu.UniqueKey
	fillRRectVertexBytes   []byte
	fillRRectIndexBytes    []byte
)

func fillRRectStaticBuffers() {
	fillRRectBufferKeyOnce.Do(func() {
		builder := gpu.UniqueKeyBuilder(&fillRRectVertexKey, gpu.GenerateUniqueKeyDomain(), 1,
			"FillRRectVertexBuffer")
		builder.Slice()[0] = 1
		builder.Finish()
		builder = gpu.UniqueKeyBuilder(&fillRRectIndexKey, gpu.GenerateUniqueKeyDomain(), 1,
			"FillRRectIndexBuffer")
		builder.Slice()[0] = 1
		builder.Finish()

		fillRRectVertexBytes = make([]byte, len(fillRRectVertexData)*12*4)
		off := 0
		for i := range fillRRectVertexData {
			for _, v := range fillRRectVertexData[i] {
				binary.LittleEndian.PutUint32(fillRRectVertexBytes[off:], math.Float32bits(v))
				off += 4
			}
		}
		fillRRectIndexBytes = make([]byte, len(fillRRectIndexData)*2)
		for i, v := range fillRRectIndexData {
			binary.LittleEndian.PutUint16(fillRRectIndexBytes[i*2:], v)
		}
	})
}

// createProgramInfo builds the program info for this op's draw.
func (o *fillRRectOp) createProgramInfo(state *OpFlushState) {
	args := state.OpArgs()
	if args.UsesMSAASurface() {
		o.processorFlags |= fillRRectFlagMSAAEnabled
	}
	gp := makeFillRRectProcessor(o.processorFlags)
	o.programInfo = o.helper.CreateProgramInfoWithStencil(state.Caps(), args.WriteView(),
		args.UsesMSAASurface(), args.AppliedClip(), args.DstProxyView(), gp,
		gpu.PrimitiveTypeTriangles, args.RenderPassBarriers(), args.ColorLoadOp())
}

// OnPrepare implements Op.
func (o *fillRRectOp) OnPrepare(state *OpFlushState) {
	if o.programInfo == nil {
		o.createProgramInfo(state)
	}

	instanceStride := o.programInfo.GeomProc().gpBase().InstanceStride()
	idata, instanceBuffer, baseInstance := state.MakeVertexSpace(uint64(instanceStride),
		len(o.instances))
	if idata != nil {
		o.instanceBuffer = instanceBuffer
		o.baseInstance = baseInstance
		w := quadVertexWriter{buf: idata}
		for idx := range o.instances {
			i := &o.instances[idx]
			l := i.rrect.Rect.Left
			t := i.rrect.Rect.Top
			r := i.rrect.Rect.Right
			b := i.rrect.Rect.Bottom

			// Produce a matrix that draws the round rect from normalized [-1, -1, +1, +1] space.
			var m geom.Matrix
			// Unmap the normalized rect [-1, -1, +1, +1] back to [l, t, r, b].
			m.SetScaleTranslate((r-l)/2, (b-t)/2, (l+r)/2, (t+b)/2)
			// Map to device space.
			m.PostConcat(&i.viewMatrix)

			// Convert the radii to [-1, -1, +1, +1] space and write their attribs (uniform radii: all four corners
			// share the simple radii).
			radiusX := i.rrect.RadiusX * 2 / (r - l)
			radiusY := i.rrect.RadiusY * 2 / (b - t)
			for range 4 {
				w.putF32(radiusX)
			}
			for range 4 {
				w.putF32(radiusY)
			}

			w.putF32(m.Get(geom.MScaleX))
			w.putF32(m.Get(geom.MSkewX))
			w.putF32(m.Get(geom.MSkewY))
			w.putF32(m.Get(geom.MScaleY))
			w.putF32(m.Get(geom.MTransX))
			w.putF32(m.Get(geom.MTransY))

			if o.processorFlags&fillRRectFlagHasLocalCoords != 0 {
				if i.localCoords.isMatrix {
					bounds := i.rrect.Rect
					lm := &i.localCoords.matrix
					u := lm.MapVector(geom.Point{X: bounds.Right - bounds.Left, Y: 0})
					v := lm.MapVector(geom.Point{X: 0, Y: bounds.Bottom - bounds.Top})
					l0 := lm.MapPoint(geom.Point{X: bounds.Left, Y: bounds.Top})
					w.putF32(v.X) // localrotate
					w.putF32(u.Y)
					w.putF32(l0.X) // localrect
					w.putF32(l0.Y)
					w.putF32(l0.X + u.X)
					w.putF32(l0.Y + v.Y)
				} else {
					w.putF32(0) // localrotate
					w.putF32(0)
					w.putRect(i.localCoords.rect) // localrect
				}
			}

			w.putColor(i.color, o.processorFlags&fillRRectFlagWideColor != 0)
		}
		if w.offset != instanceStride*len(o.instances) {
			panic("instance data out of sync with the instance stride")
		}
	}

	fillRRectStaticBuffers()
	rp := state.ResourceProvider()
	o.indexBuffer = rp.FindOrMakeStaticBuffer(gpu.BufferTypeIndex,
		uint64(len(fillRRectIndexBytes)), fillRRectIndexBytes, &fillRRectIndexKey)
	o.vertexBuffer = rp.FindOrMakeStaticBuffer(gpu.BufferTypeVertex,
		uint64(len(fillRRectVertexBytes)), fillRRectVertexBytes, &fillRRectVertexKey)
}

// OnExecute implements Op.
func (o *fillRRectOp) OnExecute(state *OpFlushState, _chainBounds geom.Rect) {
	if o.instanceBuffer == nil || o.indexBuffer == nil || o.vertexBuffer == nil {
		return // Setup failed.
	}

	renderPass := state.OpsRenderPass()
	if !renderPass.BindPipeline(o.programInfo, o.Bounds()) {
		return
	}
	if o.programInfo.Pipeline().IsScissorTestEnabled() {
		renderPass.SetScissorRect(state.OpArgs().AppliedClip().ScissorState().Rect())
	}
	renderPass.BindTextures(o.programInfo.GeomProc(), nil, o.programInfo.Pipeline())
	renderPass.BindBuffers(o.indexBuffer, o.instanceBuffer, o.vertexBuffer)
	renderPass.DrawIndexedInstanced(len(fillRRectIndexData), 0, len(o.instances),
		o.baseInstance, 0)
}

//////////////////////////////////////////////////////////////////////////////
// Geometry processor.

// fillRRectProcessor is the geometry processor for fillRRectOp.
type fillRRectProcessor struct {
	colorAttrib   *Attribute
	instanceAttrs [6]Attribute
	vertexAttrs   [3]Attribute
	GPBase
	flags uint32
}

// makeFillRRectProcessor builds a fillRRectProcessor geometry processor.
func makeFillRRectProcessor(flags uint32) GeometryProcessor {
	gp := fillRRectGPPool.borrow()
	gp.flags = flags
	gp.initGP(FillRRectOpProcessorClassID)
	gp.vertexAttrs[0] = MakeAttribute("radii_selector", gpu.VertexAttribTypeFloat4,
		GLSLTypeFloat4)
	gp.vertexAttrs[1] = MakeAttribute("corner_and_radius_outsets", gpu.VertexAttribTypeFloat4,
		GLSLTypeFloat4)
	// Coverage only.
	gp.vertexAttrs[2] = MakeAttribute("aa_bloat_and_coverage", gpu.VertexAttribTypeFloat4,
		GLSLTypeFloat4)
	gp.setVertexAttributesWithImplicitOffsets(gp.vertexAttrs[:])

	n := 0
	gp.instanceAttrs[n] = MakeAttribute("radii_x", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	gp.instanceAttrs[n] = MakeAttribute("radii_y", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	gp.instanceAttrs[n] = MakeAttribute("skew", gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
	n++
	if flags&fillRRectFlagHasLocalCoords != 0 {
		gp.instanceAttrs[n] = MakeAttribute("translate_and_localrotate",
			gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
		n++
		gp.instanceAttrs[n] = MakeAttribute("localrect", gpu.VertexAttribTypeFloat4,
			GLSLTypeFloat4)
		n++
	} else {
		gp.instanceAttrs[n] = MakeAttribute("translate_and_localrotate",
			gpu.VertexAttribTypeFloat2, GLSLTypeFloat2)
		n++
	}
	gp.instanceAttrs[n] = MakeColorAttribute("color", flags&fillRRectFlagWideColor != 0)
	gp.colorAttrib = &gp.instanceAttrs[n]
	n++
	gp.setInstanceAttributesWithImplicitOffsets(gp.instanceAttrs[:n])
	return gp
}

func (g *fillRRectProcessor) Name() string { return "FillRRectOp::Processor" }

func (g *fillRRectProcessor) AddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBits(numFillRRectProcessorFlags, g.flags, "flags")
}

func (g *fillRRectProcessor) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &fillRRectProcessorImpl{}
}

// fillRRectProcessorImpl is the shader implementation for fillRRectProcessor.
type fillRRectProcessorImpl struct {
	GPImplBase
}

func (i *fillRRectProcessorImpl) SetData(*ProgramDataManager, *gpu.ShaderCaps, GeometryProcessor) {
}

func (i *fillRRectProcessorImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	v := args.VertBuilder
	f := args.FragBuilder
	proc := args.GeomProc.(*fillRRectProcessor)
	useHWDerivatives := proc.flags&fillRRectFlagUseHWDerivatives != 0

	varyings := args.VaryingHandler
	varyings.EmitAttributes(proc)
	f.CodeAppendf("vec4 %s;", args.OutputColor)
	varyings.AddPassThroughAttribute(proc.colorAttrib.AsShaderVar(), args.OutputColor,
		InterpolationCanBeFlat)

	// Emit the vertex shader. When MSAA is enabled, we need to make sure every sample gets lit up on pixels that have
	// fractional coverage. We do this by making the ramp wider.
	aaBloatMultiplier := 0 // No AA bloat.
	switch {
	case proc.flags&fillRRectFlagMSAAEnabled != 0:
		aaBloatMultiplier = 2 // Outset an entire pixel (2 radii).
	case proc.flags&fillRRectFlagFakeNonAA == 0:
		aaBloatMultiplier = 1 // Outset one half pixel (1 radius).
	}
	v.CodeAppendf("float aa_bloat_multiplier = float(%d);", aaBloatMultiplier)

	// Unpack vertex attribs.
	v.CodeAppend("vec2 corner = corner_and_radius_outsets.xy;")
	v.CodeAppend("vec2 radius_outset = corner_and_radius_outsets.zw;")
	v.CodeAppend("vec2 aa_bloat_direction = aa_bloat_and_coverage.xy;")
	v.CodeAppend("float is_linear_coverage = aa_bloat_and_coverage.w;")

	// Find the amount to bloat each edge for AA (in source space).
	v.CodeAppend("vec2 pixellength = inversesqrt(" +
		"vec2(dot(skew.xz, skew.xz), dot(skew.yw, skew.yw)));")
	v.CodeAppend("vec4 normalized_axis_dirs = skew * pixellength.xyxy;")
	v.CodeAppend("vec2 axiswidths = (abs(normalized_axis_dirs.xy) + " +
		"abs(normalized_axis_dirs.zw));")
	v.CodeAppend("vec2 aa_bloatradius = axiswidths * pixellength * .5;")

	// Identify our radii.
	v.CodeAppend("vec4 radii_and_neighbors = radii_selector" +
		"* mat4(radii_x, radii_y, radii_x.yxwz, radii_y.wzyx);")
	v.CodeAppend("vec2 radii = radii_and_neighbors.xy;")
	v.CodeAppend("vec2 neighbor_radii = radii_and_neighbors.zw;")

	v.CodeAppend("float coverage_multiplier = 1.0;")
	v.CodeAppend("if (any(greaterThan(aa_bloatradius, vec2(1.0)))) {")
	// The rrect is more narrow than a half-pixel AA coverage ramp. We can't draw as-is or else opposite AA borders will
	// overlap. Instead, fudge the size up to the width of a coverage ramp, and then reduce total coverage to make the
	// rect appear more thin.
	v.CodeAppend("    corner = max(abs(corner), aa_bloatradius) * sign(corner);")
	v.CodeAppend("    coverage_multiplier = 1.0 / (max(aa_bloatradius.x, 1.0) * " +
		"max(aa_bloatradius.y, 1.0));")
	// Set radii to zero to ensure we take the "linear coverage" codepath. (The "coverage" variable only has effect in
	// the linear codepath.)
	v.CodeAppend("    radii = vec2(0.0);")
	v.CodeAppend("}")

	// Unpack coverage.
	v.CodeAppend("float coverage = aa_bloat_and_coverage.z;")
	if proc.flags&fillRRectFlagMSAAEnabled != 0 {
		// MSAA has a wider ramp that goes from -.5 to 1.5 instead of 0 to 1.
		v.CodeAppendf("coverage = (coverage - .5) * aa_bloat_multiplier + .5;")
	}

	v.CodeAppend("if (any(lessThan(radii, aa_bloatradius * 1.5))) {")
	// The radii are very small. Demote this arc to a sharp 90 degree corner.
	v.CodeAppend("    radii = vec2(0.0);")
	// Convert to a standard picture frame for an AA rect instead of the round rect geometry.
	v.CodeAppend("    aa_bloat_direction = sign(corner);")
	v.CodeAppend("    if (coverage > .5) {") // Are we an inset edge?
	v.CodeAppend("        aa_bloat_direction = -aa_bloat_direction;")
	v.CodeAppend("    }")
	v.CodeAppend("    is_linear_coverage = 1.0;")
	v.CodeAppend("} else {")
	// Don't let radii get smaller than a coverage ramp plus an extra half pixel for MSAA. Always use the same amount so
	// we don't pop when switching between MSAA and coverage.
	v.CodeAppend("    radii = clamp(radii, pixellength * 1.5, vec2(2.0) - pixellength * 1.5);")
	v.CodeAppend("    neighbor_radii = clamp(neighbor_radii, pixellength * 1.5, " +
		"vec2(2.0) - pixellength * 1.5);")
	// Don't let neighboring radii get closer together than 1/16 pixel.
	v.CodeAppend("    vec2 spacing = 2.0 - radii - neighbor_radii;")
	v.CodeAppend("    vec2 extra_pad = max(pixellength * .0625 - spacing, vec2(0.0));")
	v.CodeAppend("    radii -= extra_pad * .5;")
	v.CodeAppend("}")

	// Find our vertex position, adjusted for radii and bloated for AA. Our rect is drawn in normalized [-1,-1,+1,+1]
	// space.
	v.CodeAppend("vec2 aa_outset = " +
		"aa_bloat_direction * aa_bloatradius * aa_bloat_multiplier;")
	v.CodeAppend("vec2 vertexpos = corner + radius_outset * radii + aa_outset;")

	v.CodeAppend("if (coverage > .5) {") // Are we an inset edge?
	// Don't allow the aa insets to overlap. i.e., Don't let them inset past the center (x=y=0). Since we don't allow
	// the rect to become thinner than 1px, this should only happen when using MSAA, where we inset by an entire pixel
	// instead of half.
	v.CodeAppend("    if (aa_bloat_direction.x != 0.0 && vertexpos.x * corner.x < 0.0) {")
	v.CodeAppend("        float backset = abs(vertexpos.x);")
	v.CodeAppend("        vertexpos.x = 0.0;")
	v.CodeAppend("        vertexpos.y += " +
		"backset * sign(corner.y) * pixellength.y/pixellength.x;")
	v.CodeAppend("        coverage = (coverage - .5) * abs(corner.x) / " +
		"(abs(corner.x) + backset) + .5;")
	v.CodeAppend("    }")
	v.CodeAppend("    if (aa_bloat_direction.y != 0.0 && vertexpos.y * corner.y < 0.0) {")
	v.CodeAppend("        float backset = abs(vertexpos.y);")
	v.CodeAppend("        vertexpos.y = 0.0;")
	v.CodeAppend("        vertexpos.x += " +
		"backset * sign(corner.x) * pixellength.x/pixellength.y;")
	v.CodeAppend("        coverage = (coverage - .5) * abs(corner.y) / " +
		"(abs(corner.y) + backset) + .5;")
	v.CodeAppend("    }")
	v.CodeAppend("}")

	// Transform to device space.
	v.CodeAppend("mat2 skewmatrix = mat2(skew.xy, skew.zw);")
	v.CodeAppend("vec2 devcoord = vertexpos * skewmatrix + translate_and_localrotate.xy;")
	gpArgs.PositionVar = NewShaderVar("devcoord", GLSLTypeFloat2)

	// Output local coordinates.
	if proc.flags&fillRRectFlagHasLocalCoords != 0 {
		// Do math in a way that preserves exact local coord boundaries when there is no local rotate and vertexpos is
		// on an exact shape boundary.
		v.CodeAppend("vec2 T = vertexpos * .5 + .5;")
		v.CodeAppend("vec2 localcoord = localrect.xy * (1.0 - T) + " +
			"localrect.zw * T + " +
			"translate_and_localrotate.zw * T.yx;")
		gpArgs.LocalCoordVar = NewShaderVar("localcoord", GLSLTypeFloat2)
	}

	// Setup interpolants for coverage.
	arcCoordType := GLSLTypeFloat4
	if useHWDerivatives {
		arcCoordType = GLSLTypeFloat2
	}
	arcCoord := NewVarying(arcCoordType)
	varyings.AddVarying("arccoord", &arcCoord)
	v.CodeAppend("if (0.0 != is_linear_coverage) {")
	// We are a non-corner piece: Set x=0 to indicate built-in coverage, and interpolate linear coverage across y.
	v.CodeAppendf("    %s.xy = vec2(0.0, coverage * coverage_multiplier);", arcCoord.VsOut())
	v.CodeAppend("} else {")
	// Find the normalized arc coordinates for our corner ellipse. (i.e., the coordinate system where x^2 + y^2 == 1).
	v.CodeAppend("    vec2 arccoord = 1.0 - abs(radius_outset) + aa_outset/radii * corner;")
	// We are a corner piece: Interpolate the arc coordinates for coverage. Emit x+1 to ensure no pixel in the arc has a
	// x value of 0 (since x=0 instructs the fragment shader to use linear coverage).
	v.CodeAppendf("    %s.xy = vec2(arccoord.x+1.0, arccoord.y);", arcCoord.VsOut())
	if !useHWDerivatives {
		// The gradient is order-1: Interpolate it across arccoord.zw.
		v.CodeAppendf("    mat2 derivatives = inverse(skewmatrix);")
		v.CodeAppendf("    %s.zw = derivatives * (arccoord/radii * 2.0);", arcCoord.VsOut())
	}
	v.CodeAppend("}")

	// Emit the fragment shader.
	f.CodeAppendf("float x_plus_1=%s.x, y=%s.y;", arcCoord.FsIn(), arcCoord.FsIn())
	f.CodeAppendf("float coverage;")
	f.CodeAppendf("if (0.0 == x_plus_1) {")
	f.CodeAppendf("    coverage = y;") // We are a non-arc pixel (linear coverage).
	f.CodeAppendf("} else {")
	f.CodeAppendf("    float fn = x_plus_1 * (x_plus_1 - 2.0);") // fn = (x+1)*(x-1) = x^2-1
	f.CodeAppendf("    fn = y*y + fn;")                          // fn = x^2 + y^2 - 1
	if useHWDerivatives {
		f.CodeAppendf("float fnwidth = fwidth(fn);")
	} else {
		// The gradient is interpolated across arccoord.zw.
		f.CodeAppendf("float gx=%s.z, gy=%s.w;", arcCoord.FsIn(), arcCoord.FsIn())
		f.CodeAppendf("float fnwidth = abs(gx) + abs(gy);")
	}
	f.CodeAppendf("    coverage = .5 - fn/fnwidth;")
	if proc.flags&fillRRectFlagMSAAEnabled != 0 {
		// MSAA uses ramps larger than 1px, so we need to clamp in both branches.
		f.CodeAppendf("}")
	}
	f.CodeAppendf("coverage = clamp(coverage, 0.0, 1.0);")
	if proc.flags&fillRRectFlagMSAAEnabled == 0 {
		// When not using MSAA, we only need to clamp in the "arc" branch.
		f.CodeAppendf("}")
	}
	if proc.flags&fillRRectFlagFakeNonAA != 0 {
		f.CodeAppendf("coverage = (coverage >= .5) ? 1.0 : 0.0;")
	}
	f.CodeAppendf("vec4 %s = vec4(coverage);", args.OutputCoverage)
}
