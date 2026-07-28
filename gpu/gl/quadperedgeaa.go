// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The vertex configuration (VertexSpec), CPU tessellation (inset/outset via TessellationHelper with per-vertex
// coverage), specialized geometry processor, and draw issuing for ops that render per-edge anti-aliased quads. Trims:
// the textured GP form (sampler + texture subset + saturate + color-space xform) belongs to TextureOp and arrives with
// the image work — the attribute/key layout here already reserves its slots so keys stay stable across that addition;
// specialized branch-elimination fast-path writers that would produce bytes identical to the generic writer are skipped
// in favor of only the generic writer (worth revisiting if per-frame recording cost becomes a concern).

package gl

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// QuadCoverageMode selects how per-vertex AA coverage is folded into the vertex data.
type QuadCoverageMode int32

// QuadCoverageMode values.
const (
	QuadCoverageModeNone QuadCoverageMode = iota
	QuadCoverageModeWithPosition
	QuadCoverageModeWithColor
)

// QuadColorType selects how per-vertex color is encoded.
type QuadColorType int32

// QuadColorType values.
const (
	QuadColorTypeNone QuadColorType = iota
	QuadColorTypeByte
	QuadColorTypeFloat
)

// IndexBufferOption selects the vertex/index layout for a batch of quads.
type IndexBufferOption int32

// IndexBufferOption values.
const (
	// IndexBufferOptionPictureFramed: geometrically AA'd -> 8 verts/quad + an index buffer.
	IndexBufferOptionPictureFramed IndexBufferOption = iota
	// IndexBufferOptionIndexedRects: non-AA'd but indexed -> 4 verts/quad + an index buffer.
	IndexBufferOptionIndexedRects
	// IndexBufferOptionTriStrips: non-AA'd -> 4 verts/quad but no index buffer.
	IndexBufferOptionTriStrips
)

// CalcIndexBufferOption picks the index-buffer option for a batch of aa quads.
func CalcIndexBufferOption(aa gpu.AAType, numQuads int) IndexBufferOption {
	switch {
	case aa == gpu.AATypeCoverage:
		return IndexBufferOptionPictureFramed
	case numQuads > 1:
		return IndexBufferOptionIndexedRects
	default:
		return IndexBufferOptionTriStrips
	}
}

// pmFitsInBytes reports whether a premultiplied color's channels all fit in [0,1] (alpha is assumed to already be in
// [0, 1]).
func pmFitsInBytes(c colorcore.PMColor4f) bool {
	return c.R >= 0 && c.R <= 1 && c.G >= 0 && c.G <= 1 && c.B >= 0 && c.B <= 1
}

// pmColorWhite is opaque white in premultiplied float form.
var pmColorWhite = colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}

// MinQuadColorType picks the narrowest color encoding for color: "no color" for white, byte for other in-range colors,
// float otherwise.
func MinQuadColorType(color colorcore.PMColor4f) QuadColorType {
	switch {
	case color == pmColorWhite:
		return QuadColorTypeNone
	case pmFitsInBytes(color):
		return QuadColorTypeByte
	default:
		return QuadColorTypeFloat
	}
}

// VertexSpec is the vertex configuration for an op that renders per-edge AA quads. The vertex order (when enabled) is
// device position, color, local position, subset, aa edge equations.
type VertexSpec struct {
	deviceQuadType                QuadType
	localQuadType                 QuadType
	indexBufferOption             IndexBufferOption
	colorType                     QuadColorType
	hasLocalCoords                bool
	hasSubset                     bool
	usesCoverageAA                bool
	compatibleWithCoverageAsAlpha bool
	// The geometry subset serves to clip off pixels touched by quads with sharp corners that would otherwise exceed the
	// miter limit for the AA-outset geometry.
	requiresGeometrySubset bool
}

// QuadSubset selects whether quads carry a texture subset rect.
type QuadSubset bool

// QuadSubset values.
const (
	QuadSubsetNo  QuadSubset = false
	QuadSubsetYes QuadSubset = true
)

// NewVertexSpec builds a VertexSpec for the given quad shapes and rendering options.
func NewVertexSpec(deviceQuadType QuadType, colorType QuadColorType, localQuadType QuadType, hasLocalCoords bool, subset QuadSubset, aa gpu.AAType, coverageAsAlpha bool, indexBufferOption IndexBufferOption) VertexSpec {
	return VertexSpec{
		deviceQuadType:                deviceQuadType,
		localQuadType:                 localQuadType,
		indexBufferOption:             indexBufferOption,
		colorType:                     colorType,
		hasLocalCoords:                hasLocalCoords,
		hasSubset:                     bool(subset),
		usesCoverageAA:                aa == gpu.AATypeCoverage,
		compatibleWithCoverageAsAlpha: coverageAsAlpha,
		requiresGeometrySubset: aa == gpu.AATypeCoverage &&
			deviceQuadType > QuadTypeRectilinear,
	}
}

// DeviceQuadType returns the device-space quad shape.
func (s *VertexSpec) DeviceQuadType() QuadType { return s.deviceQuadType }

// LocalQuadType returns the local-space quad shape.
func (s *VertexSpec) LocalQuadType() QuadType { return s.localQuadType }

// IndexBufferOption returns the spec's index-buffer layout choice.
func (s *VertexSpec) IndexBufferOption() IndexBufferOption { return s.indexBufferOption }

// HasLocalCoords reports whether the spec carries local coordinates.
func (s *VertexSpec) HasLocalCoords() bool { return s.hasLocalCoords }

// ColorType returns the spec's per-vertex color encoding.
func (s *VertexSpec) ColorType() QuadColorType { return s.colorType }

// HasVertexColors reports whether the spec carries per-vertex color.
func (s *VertexSpec) HasVertexColors() bool { return s.colorType != QuadColorTypeNone }

// HasSubset reports whether the spec carries a texture subset rect.
func (s *VertexSpec) HasSubset() bool { return s.hasSubset }

// UsesCoverageAA reports whether the spec uses geometric coverage AA.
func (s *VertexSpec) UsesCoverageAA() bool { return s.usesCoverageAA }

// CompatibleWithCoverageAsAlpha reports whether coverage can be folded in as a plain alpha multiplier.
func (s *VertexSpec) CompatibleWithCoverageAsAlpha() bool {
	return s.compatibleWithCoverageAsAlpha
}

// RequiresGeometrySubset reports whether the spec needs a clipping subset rect to bound AA outsetting on
// non-rectilinear quads.
func (s *VertexSpec) RequiresGeometrySubset() bool { return s.requiresGeometrySubset }

// DeviceDimensionality returns 2 or 3 depending on whether the device quad type needs perspective.
func (s *VertexSpec) DeviceDimensionality() int {
	if s.deviceQuadType == QuadTypePerspective {
		return 3
	}
	return 2
}

// LocalDimensionality returns 0 if hasLocalCoords is false, else 2 or 3 depending on the local quad type.
func (s *VertexSpec) LocalDimensionality() int {
	if !s.hasLocalCoords {
		return 0
	}
	if s.localQuadType == QuadTypePerspective {
		return 3
	}
	return 2
}

// VerticesPerQuad returns 8 for coverage-AA quads (inner + outer ring), else 4.
func (s *VertexSpec) VerticesPerQuad() int {
	if s.usesCoverageAA {
		return 8
	}
	return 4
}

// CoverageMode returns how (or whether) this spec folds AA coverage into the vertex data.
func (s *VertexSpec) CoverageMode() QuadCoverageMode {
	if s.UsesCoverageAA() {
		if s.CompatibleWithCoverageAsAlpha() && s.HasVertexColors() &&
			!s.RequiresGeometrySubset() {
			// Using a geometric subset acts as a second source of coverage and folding the original coverage into color
			// makes it impossible to apply the color's alpha to the geometric subset's coverage when the original shape
			// is clipped.
			return QuadCoverageModeWithColor
		}
		return QuadCoverageModeWithPosition
	}
	return QuadCoverageModeNone
}

// VertexSize returns the byte size of one vertex under this spec. This needs to stay in sync with
// quadPerEdgeAAGeometryProcessor.initializeAttrs.
func (s *VertexSpec) VertexSize() int {
	needsPerspective := s.DeviceDimensionality() == 3
	coverageMode := s.CoverageMode()

	count := 0
	if coverageMode == QuadCoverageModeWithPosition {
		if needsPerspective {
			count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat4)
		} else {
			count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat2) +
				gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat)
		}
	} else {
		if needsPerspective {
			count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat3)
		} else {
			count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat2)
		}
	}

	if s.RequiresGeometrySubset() {
		count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat4)
	}

	count += s.LocalDimensionality() * gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat)

	switch s.ColorType() {
	case QuadColorTypeByte:
		count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeUByte4Norm)
	case QuadColorTypeFloat:
		count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat4)
	case QuadColorTypeNone:
	}

	if s.HasSubset() {
		count += gpu.VertexAttribTypeSize(gpu.VertexAttribTypeFloat4)
	}

	return count
}

// NeedsIndexBuffer reports whether this spec's index-buffer option requires an index buffer.
func (s *VertexSpec) NeedsIndexBuffer() bool {
	return s.indexBufferOption != IndexBufferOptionTriStrips
}

// PrimitiveType returns the GL primitive type this spec draws with.
func (s *VertexSpec) PrimitiveType() gpu.PrimitiveType {
	if s.indexBufferOption == IndexBufferOptionTriStrips {
		return gpu.PrimitiveTypeTriangleStrip
	}
	return gpu.PrimitiveTypeTriangles
}

//////////////////////////////////////////////////////////////////////////////
// Tessellator

// quadVertexWriter incrementally packs vertex attribute values into a byte buffer for the tessellator.
type quadVertexWriter struct {
	buf    []byte
	offset int
}

func (w *quadVertexWriter) putF32(v float32) {
	binary.LittleEndian.PutUint32(w.buf[w.offset:], math.Float32bits(v))
	w.offset += 4
}

// putColor writes a color as either 4 unorm bytes (round-half-up truncate) or 4 floats.
func (w *quadVertexWriter) putColor(c colorcore.PMColor4f, wide bool) {
	if wide {
		w.putF32(c.R)
		w.putF32(c.G)
		w.putF32(c.B)
		w.putF32(c.A)
		return
	}
	toByte := func(v float32) byte {
		f := v*255 + 0.5
		if f < 0 {
			f = 0
		} else if f > 255 {
			f = 255
		}
		return byte(f)
	}
	w.buf[w.offset] = toByte(c.R)
	w.buf[w.offset+1] = toByte(c.G)
	w.buf[w.offset+2] = toByte(c.B)
	w.buf[w.offset+3] = toByte(c.A)
	w.offset += 4
}

func (w *quadVertexWriter) putRect(r geom.Rect) {
	w.putF32(r.Left)
	w.putF32(r.Top)
	w.putF32(r.Right)
	w.putF32(r.Bottom)
}

// QuadTessellator processes a series of device+local quads into a VBO as specified by a VertexSpec.
type QuadTessellator struct {
	writer   quadVertexWriter
	aaHelper TessellationHelper
	spec     VertexSpec
}

// NewQuadTessellator builds a tessellator that writes into vertices per spec.
func NewQuadTessellator(spec VertexSpec, vertices []byte) *QuadTessellator {
	return &QuadTessellator{spec: spec, writer: quadVertexWriter{buf: vertices}}
}

// writeQuad writes the 4 vertices in triangle strip order; the data per-vertex depends on the VertexSpec.
func (t *QuadTessellator) writeQuad(deviceQuad, localQuad *Quad, coverage [4]float32, color colorcore.PMColor4f, geomSubset, texSubset geom.Rect) {
	if t.spec.HasLocalCoords() && localQuad == nil {
		panic("local coords required but no local quad")
	}
	mode := t.spec.CoverageMode()
	for i := range 4 {
		// Save position: a float2/float3/float4 depending on the combination of perspective and coverage mode.
		t.writer.putF32(deviceQuad.X(i))
		t.writer.putF32(deviceQuad.Y(i))
		if t.spec.DeviceQuadType() == QuadTypePerspective {
			t.writer.putF32(deviceQuad.W(i))
		}
		if mode == QuadCoverageModeWithPosition {
			t.writer.putF32(coverage[i])
		}

		// Save color.
		if t.spec.HasVertexColors() {
			wide := t.spec.ColorType() == QuadColorTypeFloat
			c := color
			if mode == QuadCoverageModeWithColor {
				c = colorcore.PMColor4f{
					R: c.R * coverage[i], G: c.G * coverage[i],
					B: c.B * coverage[i], A: c.A * coverage[i],
				}
			}
			t.writer.putColor(c, wide)
		}

		// Save local position.
		if t.spec.HasLocalCoords() {
			t.writer.putF32(localQuad.X(i))
			t.writer.putF32(localQuad.Y(i))
			if t.spec.LocalQuadType() == QuadTypePerspective {
				t.writer.putF32(localQuad.W(i))
			}
		}

		// Save the geometry subset.
		if t.spec.RequiresGeometrySubset() {
			t.writer.putRect(geomSubset)
		}

		// Save the texture subset.
		if t.spec.HasSubset() {
			t.writer.putRect(texSubset)
		}
	}
}

// Append calculates (as needed) inset and outset geometry for anti-aliasing and appends all position and vertex
// attributes required by the VertexSpec. The insetting and outsetting may damage the provided quads (they are treated
// as scratch).
func (t *QuadTessellator) Append(deviceQuad, localQuad *Quad, color colorcore.PMColor4f, uvSubset geom.Rect, aaFlags gpu.QuadAAFlags) {
	if deviceQuad.QuadType() > t.spec.DeviceQuadType() {
		panic("device quad type exceeds the vertex spec")
	}
	if t.spec.HasLocalCoords() && localQuad == nil {
		panic("local coords required but no local quad")
	}

	fullCoverage := [4]float32{1, 1, 1, 1}
	zeroCoverage := [4]float32{}

	if t.spec.UsesCoverageAA() {
		if mode := t.spec.CoverageMode(); mode != QuadCoverageModeWithColor &&
			mode != QuadCoverageModeWithPosition {
			panic("coverage AA requires a coverage mode")
		}
		// Must calculate inner and outer quadrilaterals for the vertex coverage ramps, and possibly a geometry subset
		// if corners are not right angles.
		var geomSubset geom.Rect
		if t.spec.RequiresGeometrySubset() {
			// The GP code expects a 0.5 outset rect (coverage is computed as 0 at the values of the uniform). However,
			// quad edges that aren't supposed to be antialiased may lie close to the bounds, so in that case outset by
			// an additional 0.5 — a backup clipping mechanism for cases where quad outsetting of nearly parallel edges
			// produces long thin extrusions from the original geometry.
			outset := float32(1)
			if aaFlags == gpu.QuadAAFlagsAll {
				outset = 0.5
			}
			geomSubset = deviceQuad.Bounds().Outset(outset, outset)
		}

		if aaFlags == gpu.QuadAAFlagsNone {
			// Have to write the coverage AA vertex structure, but there's no math to be done for a non-aa quad batched
			// into a coverage AA op.
			t.writeQuad(deviceQuad, localQuad, fullCoverage, color, geomSubset, uvSubset)
			// Since we pass the same corners in, the outer vertex structure will have 0 area and the coverage
			// interpolation from 1 to 0 will not be visible.
			t.writeQuad(deviceQuad, localQuad, zeroCoverage, color, geomSubset, uvSubset)
			return
		}

		// Reset the tessellation helper to match the current geometry.
		t.aaHelper.Reset(deviceQuad, localQuad)

		// Edge inset/outset distance ordered LBTR, set to 0.5 for a half pixel if the AA flag is turned on, or 0.0 if
		// the edge is not anti-aliased.
		var edgeDistances [4]float32
		if aaFlags == gpu.QuadAAFlagsAll {
			edgeDistances = f4Splat(0.5)
		} else {
			flag := func(f gpu.QuadAAFlags) float32 {
				if aaFlags&f != 0 {
					return 0.5
				}
				return 0
			}
			edgeDistances = [4]float32{
				flag(gpu.QuadAAFlagLeft), flag(gpu.QuadAAFlagBottom),
				flag(gpu.QuadAAFlagTop), flag(gpu.QuadAAFlagRight),
			}
		}

		// Write inner vertices first.
		coverage := t.aaHelper.Inset(edgeDistances, deviceQuad, localQuad)
		t.writeQuad(deviceQuad, localQuad, coverage, color, geomSubset, uvSubset)

		// Then outer vertices, which use 0.f for their coverage. If the inset was degenerate to a line (had all
		// coverages < 1), tweak the outset distance so the outer frame's narrow axis reaches out to 2px, which gives
		// better animation under translation.
		hairline := aaFlags == gpu.QuadAAFlagsAll &&
			coverage[0] < 1 && coverage[1] < 1 && coverage[2] < 1 && coverage[3] < 1
		if hairline {
			length := t.aaHelper.GetEdgeLengths()
			// Using max guards against scaling a degenerate triangle edge of 0 len up to 2px. The shuffles are so that
			// edge 0's adjustment is based on the lengths of its connecting edges (1 and 2), and so forth.
			maxWH := f4MaxV(f4Shuffle(length, 1, 0, 3, 2), f4Shuffle(length, 2, 3, 0, 1))
			// wh + 2e' = 2, so e' = (2 - wh) / 2 => e' = e * (2 - wh). But if w or h > 1, then 2 - wh < 1 and
			// represents the non-narrow axis so clamp to 1.
			for i := range 4 {
				edgeDistances[i] *= max(1, 2-maxWH[i])
			}
		}
		t.aaHelper.Outset(edgeDistances, deviceQuad, localQuad)
		t.writeQuad(deviceQuad, localQuad, zeroCoverage, color, geomSubset, uvSubset)
		return
	}

	// No outsetting needed, just write a single quad with full coverage.
	if t.spec.CoverageMode() != QuadCoverageModeNone || t.spec.RequiresGeometrySubset() {
		panic("non-coverage-AA spec with coverage mode or geometry subset")
	}
	t.writeQuad(deviceQuad, localQuad, fullCoverage, color, geom.Rect{}, uvSubset)
}

// GetQuadIndexBuffer returns the correct index buffer for the index-buffer option, or nil for tri-strips.
func GetQuadIndexBuffer(state *OpFlushState, option IndexBufferOption) *Buffer {
	switch option {
	case IndexBufferOptionPictureFramed:
		return state.ResourceProvider().RefAAQuadIndexBuffer()
	case IndexBufferOptionIndexedRects:
		return state.ResourceProvider().RefNonAAQuadIndexBuffer()
	case IndexBufferOptionTriStrips:
		return nil
	default:
		return nil
	}
}

// QuadLimit returns the maximum number of quads allowed for the index-buffer option.
func QuadLimit(option IndexBufferOption) int {
	switch option {
	case IndexBufferOptionPictureFramed:
		return MaxNumAAQuads
	case IndexBufferOptionIndexedRects:
		return MaxNumNonAAQuads
	case IndexBufferOptionTriStrips:
		return math.MaxInt32 // Not limited by an index buffer.
	default:
		panic("unknown index buffer option")
	}
}

// IssueQuadDraw issues the draw call on the render pass as specified by the indexing method in the vertex spec. The
// caller has allocated, filled in, and bound a vertex buffer and acquired and bound the correct index buffer if needed.
func IssueQuadDraw(caps *Caps, renderPass *OpsRenderPass, spec *VertexSpec, runningQuadCount, quadsInDraw, absVertBufferOffset int) {
	if spec.IndexBufferOption() == IndexBufferOptionTriStrips {
		offset := absVertBufferOffset + runningQuadCount*NumVertsPerNonAAQuad
		renderPass.Draw(4, offset)
		return
	}

	var maxNumQuads, numIndicesPerQuad, numVertsPerQuad int
	if spec.IndexBufferOption() == IndexBufferOptionPictureFramed {
		// AA uses 8 vertices and 30 indices per quad, basically nested rectangles.
		maxNumQuads = MaxNumAAQuads
		numIndicesPerQuad = NumIndicesPerAAQuad
		numVertsPerQuad = NumVertsPerAAQuad
	} else {
		// Non-AA uses 4 vertices and 6 indices per quad.
		maxNumQuads = MaxNumNonAAQuads
		numIndicesPerQuad = NumIndicesPerNonAAQuad
		numVertsPerQuad = NumVertsPerNonAAQuad
	}

	if runningQuadCount+quadsInDraw > maxNumQuads {
		panic("draw exceeds the index buffer's quad limit")
	}

	if caps.AvoidLargeIndexBufferDraws {
		// When we need to avoid large index buffer draws we modify the base vertex of the draw, which, in GL, requires
		// rebinding all vertex attrib arrays, so a base index is generally preferred.
		offset := absVertBufferOffset + runningQuadCount*numVertsPerQuad
		renderPass.DrawIndexPattern(numIndicesPerQuad, quadsInDraw, maxNumQuads, numVertsPerQuad,
			offset)
	} else {
		baseIndex := runningQuadCount * numIndicesPerQuad
		numIndicesToDraw := quadsInDraw * numIndicesPerQuad
		minVertex := runningQuadCount * numVertsPerQuad
		maxVertex := (runningQuadCount+quadsInDraw)*numVertsPerQuad - 1 // Inclusive.
		renderPass.DrawIndexed(numIndicesToDraw, baseIndex, uint16(minVertex),
			uint16(maxVertex), absVertBufferOffset)
	}
}

//////////////////////////////////////////////////////////////////////////////
// Geometry processor

// quadPerEdgeAAGeometryProcessor is the geometry processor for per-edge AA quads (both the plain and the textured
// forms; the texture color-space xform is always identity in an sRGB-only pipeline and is dropped).
type quadPerEdgeAAGeometryProcessor struct {
	attrs [6]Attribute // position, coverage, color, localCoord, geomSubset, texSubset
	GPBase
	// samplerStorage inlines the textured form's single sampler so the GP owns no external heap backing — the
	// precondition for the plain full-zero pooling in gppool.go.
	samplerStorage [1]TextureSampler
	coverageMode   QuadCoverageMode
	// The position attribute may have coverage built into it, so float3 is an ambiguous type and may mean 2d with
	// coverage, or 3d with no coverage.
	needsPerspective bool
	// Textured-form state: whether saturate() is applied to the color (only relevant when created with a texture, since
	// TextureOp skips paint conversion which normally clamps).
	textured bool
	saturate bool
}

// Attribute slots in attrs, in the order handed to setVertexAttributesWithImplicitOffsets.
const (
	qpaaAttrPosition = iota
	qpaaAttrCoverage
	qpaaAttrColor
	qpaaAttrLocalCoord
	qpaaAttrGeomSubset
	qpaaAttrTexSubset
)

// MakeQuadPerEdgeAAProcessor builds the plain (non-textured) geometry processor for spec.
func MakeQuadPerEdgeAAProcessor(spec *VertexSpec) GeometryProcessor {
	if spec.HasSubset() {
		panic("the non-textured GP cannot have a texture subset")
	}
	gp := qpaaGPPool.borrow()
	gp.initGP(QuadPerEdgeAAGeometryProcessorClassID)
	gp.initializeAttrs(spec)
	return gp
}

// MakeTexturedQuadProcessor builds the textured geometry processor for spec (the color-space xform is identity in an
// sRGB-only pipeline and dropped).
func MakeTexturedQuadProcessor(spec *VertexSpec, textureType gpu.TextureType, samplerState gpu.SamplerState, swizzle gpu.Swizzle, saturate bool) GeometryProcessor {
	if !spec.HasLocalCoords() {
		panic("the textured GP requires local coords")
	}
	gp := qpaaGPPool.borrow()
	gp.textured = true
	gp.saturate = saturate
	gp.initGP(QuadPerEdgeAAGeometryProcessorClassID)
	gp.initializeAttrs(spec)
	gp.samplerStorage[0] = MakeGPTextureSampler(samplerState, textureType, swizzle)
	gp.setTextureSamplers(gp.samplerStorage[:])
	return gp
}

// initializeAttrs configures the GP's vertex attribute layout from spec. This needs to stay in sync with
// VertexSpec.VertexSize.
func (g *quadPerEdgeAAGeometryProcessor) initializeAttrs(spec *VertexSpec) {
	g.needsPerspective = spec.DeviceDimensionality() == 3
	g.coverageMode = spec.CoverageMode()

	if g.coverageMode == QuadCoverageModeWithPosition {
		if g.needsPerspective {
			g.attrs[qpaaAttrPosition] = MakeAttribute("positionWithCoverage",
				gpu.VertexAttribTypeFloat4, GLSLTypeFloat4)
		} else {
			g.attrs[qpaaAttrPosition] = MakeAttribute("position", gpu.VertexAttribTypeFloat2,
				GLSLTypeFloat2)
			g.attrs[qpaaAttrCoverage] = MakeAttribute("coverage", gpu.VertexAttribTypeFloat,
				GLSLTypeFloat)
		}
	} else {
		if g.needsPerspective {
			g.attrs[qpaaAttrPosition] = MakeAttribute("position", gpu.VertexAttribTypeFloat3,
				GLSLTypeFloat3)
		} else {
			g.attrs[qpaaAttrPosition] = MakeAttribute("position", gpu.VertexAttribTypeFloat2,
				GLSLTypeFloat2)
		}
	}

	// Need a geometry subset when the quads are AA and not rectilinear, since their AA outsetting can go beyond a half
	// pixel.
	if spec.RequiresGeometrySubset() {
		g.attrs[qpaaAttrGeomSubset] = MakeAttribute("geomSubset", gpu.VertexAttribTypeFloat4,
			GLSLTypeFloat4)
	}

	switch spec.LocalDimensionality() {
	case 3:
		g.attrs[qpaaAttrLocalCoord] = MakeAttribute("localCoord", gpu.VertexAttribTypeFloat3,
			GLSLTypeFloat3)
	case 2:
		g.attrs[qpaaAttrLocalCoord] = MakeAttribute("localCoord", gpu.VertexAttribTypeFloat2,
			GLSLTypeFloat2)
	}

	if spec.HasVertexColors() {
		g.attrs[qpaaAttrColor] = MakeColorAttribute("color",
			spec.ColorType() == QuadColorTypeFloat)
	}

	if spec.HasSubset() {
		g.attrs[qpaaAttrTexSubset] = MakeAttribute("texSubset", gpu.VertexAttribTypeFloat4,
			GLSLTypeFloat4)
	}

	g.setVertexAttributesWithImplicitOffsets(g.attrs[:])
}

func (g *quadPerEdgeAAGeometryProcessor) Name() string {
	return "QuadPerEdgeAAGeometryProcessor"
}

// AddToKey packs the GP's shape/mode/format bits into the program key.
func (g *quadPerEdgeAAGeometryProcessor) AddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	// Texturing and device dimensionality are single-bit flags.
	b.AddBool(g.attrs[qpaaAttrTexSubset].IsInitialized(), "subset")
	b.AddBool(g.textured, "textured")
	b.AddBool(g.needsPerspective, "perspective")
	b.AddBool(g.saturate, "saturate")

	b.AddBool(g.attrs[qpaaAttrLocalCoord].IsInitialized(), "hasLocalCoords")
	if g.attrs[qpaaAttrLocalCoord].IsInitialized() {
		// 2D (0) or 3D (1).
		var bit uint32
		if g.attrs[qpaaAttrLocalCoord].CPUType() == gpu.VertexAttribTypeFloat3 {
			bit = 1
		}
		b.AddBits(1, bit, "localCoordsType")
	}
	b.AddBool(g.attrs[qpaaAttrColor].IsInitialized(), "hasColor")
	if g.attrs[qpaaAttrColor].IsInitialized() {
		// Bytes (0) or floats (1).
		var bit uint32
		if g.attrs[qpaaAttrColor].CPUType() == gpu.VertexAttribTypeFloat4 {
			bit = 1
		}
		b.AddBits(1, bit, "colorType")
	}
	// Coverage mode: 00 for none, 01 for with-position, 10 for with-color, 11 for position+geomsubset.
	var coverageKey uint32
	if g.attrs[qpaaAttrGeomSubset].IsInitialized() &&
		g.coverageMode != QuadCoverageModeWithPosition {
		panic("geometry subset requires coverage-with-position")
	}
	if g.coverageMode != QuadCoverageModeNone {
		switch {
		case g.attrs[qpaaAttrGeomSubset].IsInitialized():
			coverageKey = 0x3
		case g.coverageMode == QuadCoverageModeWithPosition:
			coverageKey = 0x1
		default:
			coverageKey = 0x2
		}
	}
	b.AddBits(2, coverageKey, "coverageMode")

	// The texture color-space xform is a textured-GP concern (always identity here); keep the key slot reserved.
	b.Add32(0, "colorSpaceXform")
}

func (g *quadPerEdgeAAGeometryProcessor) MakeProgramImpl(*gpu.ShaderCaps) GPProgramImpl {
	return &quadPerEdgeAAGPImpl{}
}

// quadPerEdgeAAGPImpl is the shader-emitting counterpart of quadPerEdgeAAGeometryProcessor.
type quadPerEdgeAAGPImpl struct {
	GPImplBase
}

// SetData is a no-op: the only uniform state would be the texture color-space xform helper, which belongs to the
// textured form and is dropped.
func (i *quadPerEdgeAAGPImpl) SetData(*ProgramDataManager, *gpu.ShaderCaps, GeometryProcessor) {}

func (i *quadPerEdgeAAGPImpl) onEmitCode(args *GPEmitArgs, gpArgs *GPArgs) {
	gp := args.GeomProc.(*quadPerEdgeAAGeometryProcessor)
	vertBuilder := args.VertBuilder
	fragBuilder := args.FragBuilder
	varyingHandler := args.VaryingHandler

	varyingHandler.EmitAttributes(gp)

	if gp.coverageMode == QuadCoverageModeWithPosition {
		// Strip the last channel from the vertex attribute to remove coverage and get the actual position.
		if gp.needsPerspective {
			vertBuilder.CodeAppendf("vec3 position = %s.xyz;",
				gp.attrs[qpaaAttrPosition].Name())
			gpArgs.PositionVar = NewShaderVar("position", GLSLTypeFloat3)
		} else {
			vertBuilder.CodeAppendf("vec2 position = %s.xy;",
				gp.attrs[qpaaAttrPosition].Name())
			gpArgs.PositionVar = NewShaderVar("position", GLSLTypeFloat2)
		}
	} else {
		// No coverage to eliminate.
		gpArgs.PositionVar = gp.attrs[qpaaAttrPosition].AsShaderVar()
	}

	// This attribute will be uninitialized if earlier FP analysis determined no local coordinates are needed.
	gpArgs.LocalCoordVar = gp.attrs[qpaaAttrLocalCoord].AsShaderVar()

	// Solid color before any texturing gets modulated in.
	hasBlendDst := false
	if gp.attrs[qpaaAttrColor].IsInitialized() {
		if gp.coverageMode == QuadCoverageModeWithColor && gp.needsPerspective {
			panic("coverage-with-color is incompatible with perspective")
		}
		// The color cannot be flat if the varying coverage has been modulated into it.
		interpolation := InterpolationCanBeFlat
		if gp.coverageMode == QuadCoverageModeWithColor {
			interpolation = InterpolationInterpolated
		}
		fragBuilder.CodeAppendf("vec4 %s;", args.OutputColor)
		varyingHandler.AddPassThroughAttribute(gp.attrs[qpaaAttrColor].AsShaderVar(),
			args.OutputColor, interpolation)
		hasBlendDst = true
	} else {
		// Output color must be initialized to something.
		fragBuilder.CodeAppendf("vec4 %s = vec4(1.0);", args.OutputColor)
	}

	// If there is a texture, must also handle texture coordinates and reading from the texture in the fragment shader
	// before continuing to fragment processors.
	if gp.textured {
		// Texture coordinates clamped by the subset in the fragment shader; if the GP has a texture, it's guaranteed to
		// have local coordinates.
		fragBuilder.CodeAppend("vec2 texCoord;")
		if gp.attrs[qpaaAttrLocalCoord].CPUType() == gpu.VertexAttribTypeFloat3 {
			// Can't do a pass-through since we need to perform the perspective division.
			v := NewVarying(gp.attrs[qpaaAttrLocalCoord].GPUType())
			varyingHandler.AddVarying(gp.attrs[qpaaAttrLocalCoord].Name(), &v)
			vertBuilder.CodeAppendf("%s = %s;", v.VsOut(),
				gp.attrs[qpaaAttrLocalCoord].Name())
			fragBuilder.CodeAppendf("texCoord = %s.xy / %s.z;", v.FsIn(), v.FsIn())
		} else {
			varyingHandler.AddPassThroughAttribute(gp.attrs[qpaaAttrLocalCoord].AsShaderVar(),
				"texCoord", InterpolationInterpolated)
		}

		// Clamp the now-2D texCoord variable by the subset if it is provided.
		if gp.attrs[qpaaAttrTexSubset].IsInitialized() {
			fragBuilder.CodeAppend("vec4 subset;")
			varyingHandler.AddPassThroughAttribute(gp.attrs[qpaaAttrTexSubset].AsShaderVar(),
				"subset", InterpolationCanBeFlat)
			fragBuilder.CodeAppend("texCoord = clamp(texCoord, subset.xy, subset.zw);")
		}

		// Now modulate the starting output color by the texture lookup (modulate blend; the color-space xform is
		// identity).
		lookup := fragBuilder.AppendTextureLookup(args.UniformHandler, args.TexSamplers[0],
			"texCoord")
		expr := lookup
		if hasBlendDst {
			expr = "(" + lookup + " * " + args.OutputColor + ")"
		}
		if gp.saturate {
			expr = "clamp(" + expr + ", 0.0, 1.0)"
		}
		fragBuilder.CodeAppendf("%s = %s;", args.OutputColor, expr)
	} else if gp.saturate {
		// Saturate is only intended for use with a texture, to account for the fact that TextureOp skips paint
		// conversion, which normally handles this.
		panic("saturate requires the textured form")
	}

	// And lastly, output the coverage calculation code.
	if gp.coverageMode == QuadCoverageModeWithPosition {
		coverage := NewVarying(GLSLTypeFloat)
		varyingHandler.AddVarying("coverage", &coverage)
		if gp.needsPerspective {
			// Multiply by "W" in the vertex shader, then by 1/w (gl_FragCoord.w) in the fragment shader to get
			// screen-space linear coverage.
			vertBuilder.CodeAppendf("%s = %s.w * %s.z;", coverage.VsOut(),
				gp.attrs[qpaaAttrPosition].Name(), gp.attrs[qpaaAttrPosition].Name())
			fragBuilder.CodeAppendf("float coverage = %s * gl_FragCoord.w;", coverage.FsIn())
		} else {
			vertBuilder.CodeAppendf("%s = %s;", coverage.VsOut(),
				gp.attrs[qpaaAttrCoverage].Name())
			fragBuilder.CodeAppendf("float coverage = %s;", coverage.FsIn())
		}

		if gp.attrs[qpaaAttrGeomSubset].IsInitialized() {
			// Calculate the distance from gl_FragCoord to the 4 edges of the subset and clamp them to (0, 1). Use the
			// minimum of these and the original coverage. This only has to be done in the exterior triangles; the
			// interior of the quad geometry can never be clipped by the subset box.
			fragBuilder.CodeAppend("vec4 geoSubset;")
			varyingHandler.AddPassThroughAttribute(gp.attrs[qpaaAttrGeomSubset].AsShaderVar(),
				"geoSubset", InterpolationCanBeFlat)
			// saturate here means clamp to [0,1].
			fragBuilder.CodeAppendf(
				"vec4 dists4 = clamp(vec4(1.0, 1.0, -1.0, -1.0) * ((%s).xyxy - geoSubset), "+
					"0.0, 1.0);", fragBuilder.FragmentPosition(),
			)
			fragBuilder.CodeAppend("vec2 dists2 = dists4.xy + dists4.zw - 1.0;")
			fragBuilder.CodeAppend("coverage = min(coverage, dists2.x * dists2.y);")
		}

		fragBuilder.CodeAppendf("vec4 %s = vec4(coverage);", args.OutputCoverage)
	} else {
		// Set coverage to 1, since it's either non-AA or the coverage was already folded into the output color.
		if gp.attrs[qpaaAttrGeomSubset].IsInitialized() {
			panic("geometry subset requires coverage-with-position")
		}
		fragBuilder.CodeAppendf("const vec4 %s = vec4(1.0);", args.OutputCoverage)
	}
}
