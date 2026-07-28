// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The analytic coverage (clip) fragment processors: ClipEdgeType, the Rect / Circle / Ellipse / DeviceSpace /
// ModulateRGBA fragment processor factories, the oval effect, the convex-polygon effect, and the rrect effect with its
// circular/elliptical implementations. With no shader compiler, the Rect/Circle/Ellipse runtime effects are
// hand-written GLSL-emitting FPs with the specialized uniforms (edgeType, medPrecision) baked into the emitted source
// and keyed. geom.RRect trims: only uniform radii are representable, so the circular effect's partial corner-flag lanes
// and the elliptical effect's nine-patch lane are unreachable and trimmed; the rrect effect's complex/nine-patch
// analysis reduces to the simple/oval/rect lanes.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// ClipEdgeType selects a clip fragment processor's fill/AA behavior.
type ClipEdgeType int32

// ClipEdgeType values.
const (
	ClipEdgeFillBW ClipEdgeType = iota
	ClipEdgeFillAA
	ClipEdgeInverseFillBW
	ClipEdgeInverseFillAA
)

// IsFill reports whether t is one of the (non-inverse) fill variants.
func (t ClipEdgeType) IsFill() bool {
	return t == ClipEdgeFillAA || t == ClipEdgeFillBW
}

// IsInverseFill reports whether t is one of the inverse-fill variants.
func (t ClipEdgeType) IsInverseFill() bool {
	return t == ClipEdgeInverseFillAA || t == ClipEdgeInverseFillBW
}

// IsAA reports whether t is one of the antialiased variants.
func (t ClipEdgeType) IsAA() bool {
	return t == ClipEdgeFillAA || t == ClipEdgeInverseFillAA
}

// Invert returns the fill/inverse-fill-swapped counterpart of t.
func (t ClipEdgeType) Invert() ClipEdgeType {
	switch t {
	case ClipEdgeFillBW:
		return ClipEdgeInverseFillBW
	case ClipEdgeFillAA:
		return ClipEdgeInverseFillAA
	case ClipEdgeInverseFillBW:
		return ClipEdgeFillBW
	default:
		return ClipEdgeFillAA
	}
}

//////////////////////////////////////////////////////////////////////////////
// DeviceSpace: samples its child at the fragment coordinate instead of local coords.

type deviceSpaceFP struct {
	FPBase
}

// DeviceSpaceFP wraps fp so it samples its child at the device-space fragment coordinate instead of local coords.
func DeviceSpaceFP(fp FragmentProcessor) FragmentProcessor {
	if fp == nil {
		return nil
	}
	d := &deviceSpaceFP{}
	d.initFP(DeviceSpaceClassID, fpProcessorOptimizationFlags(fp))
	registerChildOf(d, fp, FragCoordSampleUsage())
	return d
}

func (f *deviceSpaceFP) Name() string { return "DeviceSpace" }

func (f *deviceSpaceFP) Clone() FragmentProcessor {
	return DeviceSpaceFP(f.ChildProcessor(0).Clone())
}

func (f *deviceSpaceFP) onAddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

func (f *deviceSpaceFP) onIsEqual(FragmentProcessor) bool { return true }

func (f *deviceSpaceFP) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	return fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
}

func (f *deviceSpaceFP) onMakeProgramImpl() FPProgramImpl { return &deviceSpaceFPImpl{} }

type deviceSpaceFPImpl struct {
	FPImplBase
}

func (i *deviceSpaceFPImpl) EmitCode(args *FPEmitArgs) {
	args.FragBuilder.CodeAppendf("return %s;", i.InvokeChildWithCoords(0, args.InputColor, args, fragCoordName))
}

//////////////////////////////////////////////////////////////////////////////
// ModulateRGBA.

// ModulateRGBAFP returns a fragment processor whose output is inputFP's output modulated by a constant color.
func ModulateRGBAFP(inputFP FragmentProcessor, color colorcore.PMColor4f) FragmentProcessor {
	return BlendFP(MakeColorFP(color), inputFP, raster.BlendModulate)
}

//////////////////////////////////////////////////////////////////////////////
// Rect ("Rect"): analytic device-space rect coverage.

type rectClipFP struct {
	FPBase
	edgeType ClipEdgeType
	rect     geom.Rect // original rect; the uniform outsets AA forms by 0.5
}

// RectClipFP builds a fragment processor that modulates inputFP by analytic coverage of a device-space rect.
func RectClipFP(inputFP FragmentProcessor, edgeType ClipEdgeType, rect geom.Rect) FragmentProcessor {
	if rect != rect.Sorted() {
		panic("rect must be sorted")
	}
	fp := &rectClipFP{edgeType: edgeType, rect: rect}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	return BlendFP(fp, inputFP, raster.BlendModulate)
}

func (f *rectClipFP) Name() string { return "Rect" }

func (f *rectClipFP) Clone() FragmentProcessor {
	clone := &rectClipFP{edgeType: f.edgeType, rect: f.rect}
	clone.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	return clone
}

func (f *rectClipFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPRectClip), "runtimeFPKind")
	b.Add32(uint32(f.edgeType), "edgeType")
}

func (f *rectClipFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*rectClipFP)
	return ok && f.edgeType == o.edgeType && f.rect == o.rect
}

func (f *rectClipFP) onMakeProgramImpl() FPProgramImpl { return &rectClipFPImpl{} }

type rectClipFPImpl struct {
	FPImplBase
	rectUni UniformHandle
}

func (i *rectClipFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*rectClipFP)
	rectName := ""
	i.rectUni, rectName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "rectUniform")
	fragCoord := args.FragBuilder.FragmentPosition()
	f := args.FragBuilder
	if fp.edgeType.IsAA() {
		// Compute coverage relative to left and right edges, add, then subtract 1 to account for double counting. And
		// similar for top/bottom.
		f.CodeAppendf("vec4 dists4 = clamp(vec4(1.0, 1.0, -1.0, -1.0) * (%s.xyxy - %s), 0.0, 1.0);",
			fragCoord, rectName)
		f.CodeAppend("vec2 dists2 = dists4.xy + dists4.zw - 1.0;")
		f.CodeAppend("float coverage = dists2.x * dists2.y;")
	} else {
		f.CodeAppendf("float coverage = all(greaterThan(vec4(%s, %s.zw), vec4(%s.xy, %s)))"+
			" ? 1.0 : 0.0;", fragCoord, rectName, rectName, fragCoord)
	}
	if fp.edgeType.IsInverseFill() {
		f.CodeAppend("coverage = 1.0 - coverage;")
	}
	f.CodeAppend("return vec4(coverage);")
}

func (i *rectClipFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*rectClipFP)
	// The AA math in the shader evaluates to 0 at the uploaded coordinates, so outset by 0.5 to interpolate from 0 at a
	// half pixel inset and 1 at a half pixel outset of the rect.
	r := f.rect
	if f.edgeType.IsAA() {
		r = r.Outset(0.5, 0.5)
	}
	pdman.Set4f(i.rectUni, r.Left, r.Top, r.Right, r.Bottom)
}

//////////////////////////////////////////////////////////////////////////////
// Circle ("Circle").

type circleClipFP struct {
	FPBase
	edgeType ClipEdgeType
	center   geom.Point
	radius   float32
}

// CircleClipFP builds a fragment processor for analytic circle coverage. Reports failure when the combination is
// unrepresentable.
func CircleClipFP(inputFP FragmentProcessor, edgeType ClipEdgeType, center geom.Point, radius float32) (FragmentProcessor, bool) {
	// A radius below half causes the implicit insetting done by this processor to become inverted. We could handle this
	// case by making the processor code more complicated.
	if radius < 0.5 && edgeType.IsInverseFill() {
		return inputFP, false
	}
	fp := &circleClipFP{edgeType: edgeType, center: center, radius: radius}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	return BlendFP(inputFP, fp, raster.BlendModulate), true
}

func (f *circleClipFP) Name() string { return "Circle" }

func (f *circleClipFP) Clone() FragmentProcessor {
	clone := &circleClipFP{edgeType: f.edgeType, center: f.center, radius: f.radius}
	clone.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	return clone
}

func (f *circleClipFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPCircleClip), "runtimeFPKind")
	b.Add32(uint32(f.edgeType), "edgeType")
}

func (f *circleClipFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*circleClipFP)
	return ok && f.edgeType == o.edgeType && f.center == o.center && f.radius == o.radius
}

func (f *circleClipFP) onMakeProgramImpl() FPProgramImpl { return &circleClipFPImpl{} }

type circleClipFPImpl struct {
	FPImplBase
	circleUni UniformHandle
}

func (i *circleClipFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*circleClipFP)
	circleName := ""
	// The circle uniform is (center.x, center.y, radius + 0.5, 1 / (radius + 0.5)) for regular fills and (..., radius -
	// 0.5, 1 / (radius - 0.5)) for inverse fills.
	i.circleUni, circleName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "circle")
	fragCoord := args.FragBuilder.FragmentPosition()
	f := args.FragBuilder
	// The distance-to-circle calculation is performed in a space normalized to the radius and then denormalized, to
	// mitigate overflow on devices without full float.
	if fp.edgeType.IsInverseFill() {
		f.CodeAppendf("float d = (length((%s.xy - %s) * %s.w) - 1.0) * %s.z;",
			circleName, fragCoord, circleName, circleName)
	} else {
		f.CodeAppendf("float d = (1.0 - length((%s.xy - %s) * %s.w)) * %s.z;",
			circleName, fragCoord, circleName, circleName)
	}
	if fp.edgeType.IsAA() {
		f.CodeAppend("return vec4(clamp(d, 0.0, 1.0));")
	} else {
		f.CodeAppend("return vec4(d > 0.5 ? 1.0 : 0.0);")
	}
}

func (i *circleClipFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*circleClipFP)
	effectiveRadius := f.radius
	if f.edgeType.IsInverseFill() {
		effectiveRadius -= 0.5
		// When the radius is 0.5, effectiveRadius is 0 which causes an inf * 0 in the shader.
		effectiveRadius = max(0.001, effectiveRadius)
	} else {
		effectiveRadius += 0.5
	}
	pdman.Set4f(i.circleUni, f.center.X, f.center.Y, effectiveRadius, 1/effectiveRadius)
}

//////////////////////////////////////////////////////////////////////////////
// Ellipse ("Ellipse").

type ellipseClipFP struct {
	FPBase
	edgeType     ClipEdgeType
	center       geom.Point
	radii        geom.Point
	medPrecision bool
}

// EllipseClipFP builds a fragment processor for analytic ellipse coverage.
func EllipseClipFP(inputFP FragmentProcessor, edgeType ClipEdgeType, center, radii geom.Point, caps *gpu.ShaderCaps) (FragmentProcessor, bool) {
	medPrecision := !caps.FloatIs32Bits

	// Small radii produce bad results on devices without full float.
	if medPrecision && (radii.X < 0.5 || radii.Y < 0.5) {
		return inputFP, false
	}
	// Very narrow ellipses produce bad results on devices without full float.
	if medPrecision && (radii.X > 255*radii.Y || radii.Y > 255*radii.X) {
		return inputFP, false
	}
	// Very large ellipses produce bad results on devices without full float.
	if medPrecision && (radii.X > 16384 || radii.Y > 16384) {
		return inputFP, false
	}
	fp := &ellipseClipFP{
		edgeType:     edgeType,
		center:       center,
		radii:        radii,
		medPrecision: medPrecision,
	}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	return BlendFP(inputFP, fp, raster.BlendModulate), true
}

func (f *ellipseClipFP) Name() string { return "Ellipse" }

func (f *ellipseClipFP) Clone() FragmentProcessor {
	clone := &ellipseClipFP{
		edgeType:     f.edgeType,
		center:       f.center,
		radii:        f.radii,
		medPrecision: f.medPrecision,
	}
	clone.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	return clone
}

func (f *ellipseClipFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPEllipseClip), "runtimeFPKind")
	b.Add32(uint32(f.edgeType), "edgeType")
	b.AddBool(f.medPrecision, "medPrecision")
}

func (f *ellipseClipFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*ellipseClipFP)
	return ok && f.edgeType == o.edgeType && f.center == o.center && f.radii == o.radii &&
		f.medPrecision == o.medPrecision
}

func (f *ellipseClipFP) onMakeProgramImpl() FPProgramImpl { return &ellipseClipFPImpl{} }

type ellipseClipFPImpl struct {
	FPImplBase
	ellipseUni UniformHandle
	scaleUni   UniformHandle
}

func (i *ellipseClipFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*ellipseClipFP)
	ellipseName := ""
	i.ellipseUni, ellipseName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "ellipse")
	scaleName := ""
	if fp.medPrecision {
		i.scaleUni, scaleName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeFloat2, "scale")
	}
	fragCoord := args.FragBuilder.FragmentPosition()
	f := args.FragBuilder
	// d is the offset to the ellipse center.
	f.CodeAppendf("vec2 d = %s - %s.xy;", fragCoord, ellipseName)
	// If we're on a device with a "real" mediump then we'll do the distance computation in a space that is normalized
	// by the larger radius. The scale uniform is (scale, 1/scale); the inverse squared radii uniform values are already
	// in this normalized space; the center is not.
	if fp.medPrecision {
		f.CodeAppendf("d *= %s.y;", scaleName)
	}
	f.CodeAppendf("vec2 Z = d * %s.zw;", ellipseName)
	// implicit is the evaluation of (x/rx)^2 + (y/ry)^2 - 1.
	f.CodeAppend("float implicit = dot(Z, d) - 1.0;")
	// grad_dot is the squared length of the gradient of the implicit.
	f.CodeAppend("float grad_dot = 4.0 * dot(Z, Z);")
	// Avoid calling inversesqrt on zero.
	if fp.medPrecision {
		f.CodeAppend("grad_dot = max(grad_dot, 6.1036e-5);")
	} else {
		f.CodeAppend("grad_dot = max(grad_dot, 1.1755e-38);")
	}
	f.CodeAppend("float approx_dist = implicit * inversesqrt(grad_dot);")
	if fp.medPrecision {
		f.CodeAppendf("approx_dist *= %s.x;", scaleName)
	}
	switch fp.edgeType {
	case ClipEdgeFillBW:
		f.CodeAppend("float alpha = approx_dist > 0.0 ? 0.0 : 1.0;")
	case ClipEdgeFillAA:
		f.CodeAppend("float alpha = clamp(0.5 - approx_dist, 0.0, 1.0);")
	case ClipEdgeInverseFillBW:
		f.CodeAppend("float alpha = approx_dist > 0.0 ? 1.0 : 0.0;")
	default: // ClipEdgeInverseFillAA
		f.CodeAppend("float alpha = clamp(0.5 + approx_dist, 0.0, 1.0);")
	}
	f.CodeAppend("return vec4(alpha);")
}

func (i *ellipseClipFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*ellipseClipFP)
	var invRXSqd, invRYSqd float32
	scale := geom.Point{X: 1, Y: 1}
	// If we're using a scale factor to work around precision issues, choose the larger radius as the scale factor. The
	// inv radii need to be pre-adjusted by the scale factor.
	if f.medPrecision {
		if f.radii.X > f.radii.Y {
			invRXSqd = 1
			invRYSqd = (f.radii.X * f.radii.X) / (f.radii.Y * f.radii.Y)
			scale = geom.Point{X: f.radii.X, Y: 1 / f.radii.X}
		} else {
			invRXSqd = (f.radii.Y * f.radii.Y) / (f.radii.X * f.radii.X)
			invRYSqd = 1
			scale = geom.Point{X: f.radii.Y, Y: 1 / f.radii.Y}
		}
	} else {
		invRXSqd = 1 / (f.radii.X * f.radii.X)
		invRYSqd = 1 / (f.radii.Y * f.radii.Y)
	}
	pdman.Set4f(i.ellipseUni, f.center.X, f.center.Y, invRXSqd, invRYSqd)
	if f.medPrecision {
		pdman.Set2f(i.scaleUni, scale.X, scale.Y)
	}
}

//////////////////////////////////////////////////////////////////////////////
// Oval effect.

// OvalEffectFP builds a circle or ellipse clip fragment processor for oval, depending on whether it's round.
func OvalEffectFP(inputFP FragmentProcessor, edgeType ClipEdgeType, oval geom.Rect, caps *gpu.ShaderCaps) (FragmentProcessor, bool) {
	w := oval.Width()
	h := oval.Height()
	if geom.ScalarNearlyEqual(w, h) {
		w /= 2
		return CircleClipFP(inputFP, edgeType,
			geom.Point{X: oval.Left + w, Y: oval.Top + w}, w)
	}
	w /= 2
	h /= 2
	return EllipseClipFP(inputFP, edgeType, geom.Point{X: oval.Left + w, Y: oval.Top + h},
		geom.Point{X: w, Y: h}, caps)
}

//////////////////////////////////////////////////////////////////////////////
// Convex polygon effect.

// convexPolyMaxEdges is the maximum number of half-plane edges a convexPolyFP can represent.
const convexPolyMaxEdges = 8

type convexPolyFP struct {
	FPBase
	edges     [3 * convexPolyMaxEdges]float32
	edgeType  ClipEdgeType
	edgeCount int
}

// ConvexPolyEffectFPFromEdges builds a convex-polygon clip fragment processor directly from edges: edges correspond to
// n interior half-plane equations (ax + by + c ≥ 0 for interior points).
func ConvexPolyEffectFPFromEdges(inputFP FragmentProcessor, edgeType ClipEdgeType, n int, edges []float32) (FragmentProcessor, bool) {
	if n <= 0 || n > convexPolyMaxEdges {
		return inputFP, false
	}
	fp := &convexPolyFP{edgeType: edgeType, edgeCount: n}
	copy(fp.edges[:3*n], edges[:3*n])
	// Outset the edges by 0.5 so that a pixel with center on an edge is 50% covered in the AA case and 100% covered in
	// the non-AA case.
	for i := 0; i < n; i++ {
		fp.edges[3*i+2] += 0.5
	}
	fp.initFP(ConvexPolyEffectClassID,
		fpProcessorOptimizationFlags(inputFP)&FPCompatibleWithCoverageAsAlpha)
	registerChildOf(fp, inputFP, PassThroughSampleUsage())
	return fp, true
}

// ConvexPolyEffectFP builds a convex-polygon clip fragment processor for analytic coverage of a convex,
// line-segment-only path in device space.
func ConvexPolyEffectFP(inputFP FragmentProcessor, edgeType ClipEdgeType, p *path.Path) (FragmentProcessor, bool) {
	if p.SegmentMasks() != path.SegmentLine || !p.IsConvex() {
		return inputFP, false
	}

	dir := p.ComputeFirstDirection()
	// The only way this should fail is if the clip is effectively an infinitely thin line. In that case nothing is
	// inside the clip.
	if dir == path.FirstDirectionUnknown {
		if edgeType.IsInverseFill() {
			return ModulateRGBAFP(inputFP, colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}), true
		}
		return ModulateRGBAFP(inputFP, colorcore.PMColor4f{}), true
	}

	var edges [3 * convexPolyMaxEdges]float32
	n := 0
	iter := path.NewIter(p, true)
	for {
		verb, pts, ok := iter.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove, path.VerbClose:
		case path.VerbLine:
			if n >= convexPolyMaxEdges {
				return inputFP, false
			}
			if pts[0] != pts[1] {
				v := pts[1].Sub(pts[0])
				v.Normalize()
				if dir == path.FirstDirectionCCW {
					edges[3*n] = v.Y
					edges[3*n+1] = -v.X
				} else {
					edges[3*n] = -v.Y
					edges[3*n+1] = v.X
				}
				edges[3*n+2] = -(edges[3*n]*pts[1].X + edges[3*n+1]*pts[1].Y)
				n++
			}
		default:
			// Non-linear segment, so not a polygon.
			return inputFP, false
		}
	}

	if p.IsInverseFillType() {
		edgeType = edgeType.Invert()
	}
	return ConvexPolyEffectFPFromEdges(inputFP, edgeType, n, edges[:])
}

func (f *convexPolyFP) Name() string { return "ConvexPoly" }

func (f *convexPolyFP) Clone() FragmentProcessor {
	clone := &convexPolyFP{edgeType: f.edgeType, edgeCount: f.edgeCount, edges: f.edges}
	clone.initFP(ConvexPolyEffectClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *convexPolyFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	// The edge type is packed into the low 3 bits, so the edge-type count must stay <= 8.
	b.Add32(uint32(f.edgeCount)<<3|uint32(f.edgeType), "edgesAndType")
}

func (f *convexPolyFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*convexPolyFP)
	if !ok || f.edgeType != o.edgeType || f.edgeCount != o.edgeCount {
		return false
	}
	for i := 0; i < 3*f.edgeCount; i++ {
		if f.edges[i] != o.edges[i] {
			return false
		}
	}
	return true
}

func (f *convexPolyFP) onMakeProgramImpl() FPProgramImpl { return &convexPolyFPImpl{} }

type convexPolyFPImpl struct {
	FPImplBase
	edgeUni UniformHandle
}

func (i *convexPolyFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*convexPolyFP)
	edgeArrayName := ""
	i.edgeUni, edgeArrayName = args.UniformHandler.AddUniformArray(args.FP, ShaderFlagFragment,
		GLSLTypeHalf3, "edgeArray", fp.edgeCount)
	fragCoord := args.FragBuilder.FragmentPosition()
	f := args.FragBuilder
	f.CodeAppend("float alpha = 1.0;")
	f.CodeAppend("float edge;")
	for j := 0; j < fp.edgeCount; j++ {
		f.CodeAppendf("edge = dot(vec3(%s[%d]), vec3(%s, 1.0));", edgeArrayName, j, fragCoord)
		if fp.edgeType.IsAA() {
			f.CodeAppend("alpha *= clamp(edge, 0.0, 1.0);")
		} else {
			f.CodeAppend("alpha *= step(0.5, edge);")
		}
	}
	if fp.edgeType.IsInverseFill() {
		f.CodeAppend("alpha = 1.0 - alpha;")
	}
	f.CodeAppendf("return %s * alpha;", i.InvokeChild(0, args))
}

func (i *convexPolyFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*convexPolyFP)
	pdman.Set3fv(i.edgeUni, int32(f.edgeCount), f.edges[:3*f.edgeCount])
}

//////////////////////////////////////////////////////////////////////////////
// RRect effect: circular + elliptical variants, plus the selection analysis.

// rrectRadiusMin is the minimum radius the effects handle: rrect radii >= 0.5.
const rrectRadiusMin = 0.5

// circularRRectAllCornerFlags is the "all corners rounded" flag combination — the only corner combination reachable
// through geom.RRect's uniform radii (partial-corner lanes trimmed).
const circularRRectAllCornerFlags = 0xF

type circularRRectFP struct {
	FPBase
	rrect    geom.RRect
	edgeType ClipEdgeType
}

// newCircularRRectFP builds a circular-rrect clip fragment processor for the all-corners-rounded case.
func newCircularRRectFP(inputFP FragmentProcessor, edgeType ClipEdgeType, rrect geom.RRect) (FragmentProcessor, bool) {
	if edgeType != ClipEdgeFillAA && edgeType != ClipEdgeInverseFillAA {
		return inputFP, false
	}
	fp := &circularRRectFP{rrect: rrect, edgeType: edgeType}
	fp.initFP(CircularRRectEffectClassID,
		fpProcessorOptimizationFlags(inputFP)&FPCompatibleWithCoverageAsAlpha)
	registerChildOf(fp, inputFP, PassThroughSampleUsage())
	return fp, true
}

func (f *circularRRectFP) Name() string { return "CircularRRect" }

func (f *circularRRectFP) Clone() FragmentProcessor {
	clone := &circularRRectFP{rrect: f.rrect, edgeType: f.edgeType}
	clone.initFP(CircularRRectEffectClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *circularRRectFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(circularRRectAllCornerFlags)<<3|uint32(f.edgeType), "cornerFlagsAndType")
}

func (f *circularRRectFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*circularRRectFP)
	// The corner flags are derived from the rrect, so no need to check them.
	return ok && f.edgeType == o.edgeType && rrectsEqual(f.rrect, o.rrect)
}

func (f *circularRRectFP) onMakeProgramImpl() FPProgramImpl { return &circularRRectFPImpl{} }

type circularRRectFPImpl struct {
	FPImplBase
	innerRectUni      UniformHandle
	radiusPlusHalfUni UniformHandle
}

func (i *circularRRectFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*circularRRectFP)
	// The inner rect is the rrect bounds inset by the radius. Its left, top, right, and bottom edges correspond to
	// components x, y, z, and w, respectively.
	rectName := ""
	i.innerRectUni, rectName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "innerRect")
	// x is (r + .5) and y is 1/(r + .5).
	radiusPlusHalfName := ""
	i.radiusPlusHalfUni, radiusPlusHalfName = args.UniformHandler.AddUniform(args.FP,
		ShaderFlagFragment, GLSLTypeHalf2, "radiusPlusHalf")

	// If we're on a device where float != fp32 then the length calculation could overflow.
	var clampedCircleDistance string
	if !args.ShaderCaps.FloatIs32Bits {
		clampedCircleDistance = "clamp(" + radiusPlusHalfName + ".x * (1.0 - length(dxy * " +
			radiusPlusHalfName + ".y)), 0.0, 1.0)"
	} else {
		clampedCircleDistance = "clamp(" + radiusPlusHalfName + ".x - length(dxy), 0.0, 1.0)"
	}

	fragCoord := args.FragBuilder.FragmentPosition()
	f := args.FragBuilder
	// At each quarter-circle corner we compute a vector that is the offset of the fragment position from the circle
	// center. The vector is pinned in x and y to be in the quarter-plane relevant to that corner. Fragments in the
	// interior of the rrect have a (0,0) vector at all four corners, so as long as the radius > 0.5 they produce alpha
	// 1. This is the kAll_CornerFlags form (the only one reachable with uniform radii).
	f.CodeAppendf("vec2 dxy0 = %s.xy - %s;", rectName, fragCoord)
	f.CodeAppendf("vec2 dxy1 = %s - %s.zw;", fragCoord, rectName)
	f.CodeAppend("vec2 dxy = max(max(dxy0, dxy1), 0.0);")
	f.CodeAppendf("float alpha = %s;", clampedCircleDistance)
	if fp.edgeType == ClipEdgeInverseFillAA {
		f.CodeAppend("alpha = 1.0 - alpha;")
	}
	f.CodeAppendf("return %s * alpha;", i.InvokeChild(0, args))
}

func (i *circularRRectFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*circularRRectFP)
	if !rrectIsSimpleCircular(f.rrect) && f.rrect.Type != geom.RRectOval {
		panic("circular rrect FP requires simple circular radii")
	}
	radius := rrectRadii(f.rrect).X
	if radius < rrectRadiusMin {
		panic("radius below the effect's minimum")
	}
	rect := f.rrect.Rect.Inset(radius, radius)
	pdman.Set4f(i.innerRectUni, rect.Left, rect.Top, rect.Right, rect.Bottom)
	radius += 0.5
	pdman.Set2f(i.radiusPlusHalfUni, radius, 1/radius)
}

type ellipticalRRectFP struct {
	FPBase
	rrect    geom.RRect
	edgeType ClipEdgeType
}

// newEllipticalRRectFP builds an elliptical-rrect clip fragment processor (simple type only; the nine-patch lane is
// unrepresentable with uniform radii).
func newEllipticalRRectFP(inputFP FragmentProcessor, edgeType ClipEdgeType, rrect geom.RRect) (FragmentProcessor, bool) {
	if edgeType != ClipEdgeFillAA && edgeType != ClipEdgeInverseFillAA {
		return inputFP, false
	}
	fp := &ellipticalRRectFP{rrect: rrect, edgeType: edgeType}
	fp.initFP(EllipticalRRectEffectClassID,
		fpProcessorOptimizationFlags(inputFP)&FPCompatibleWithCoverageAsAlpha)
	registerChildOf(fp, inputFP, PassThroughSampleUsage())
	return fp, true
}

// ellipticalEffectUsesScale reports whether the elliptical rrect shader needs the scale-factor workaround to avoid
// precision loss.
func ellipticalEffectUsesScale(caps *gpu.ShaderCaps, rrect geom.RRect) bool {
	// Keep shaders consistent across varying radii when in reduced shader mode.
	if caps.ReducedShaderMode {
		return true
	}
	// If we're on a device where float != fp32 then we'll do the distance computation in a space that is normalized by
	// the largest radius.
	if !caps.FloatIs32Bits {
		return true
	}
	// Additionally, even with fp32, large radii can underflow 1/radii^2 terms, leading to blurry coverage.
	r := rrectRadii(rrect)
	maxRadius := max(r.X, r.Y)
	return geom.ScalarNearlyZero(1 / (maxRadius * maxRadius))
}

func (f *ellipticalRRectFP) Name() string { return "EllipticalRRect" }

func (f *ellipticalRRectFP) Clone() FragmentProcessor {
	clone := &ellipticalRRectFP{rrect: f.rrect, edgeType: f.edgeType}
	clone.initFP(EllipticalRRectEffectClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *ellipticalRRectFP) onAddToKey(caps *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBits(2, uint32(f.edgeType), "edge_type")
	// The rrect type is keyed in 3 bits; only the simple type is reachable here, but it is keyed regardless to keep the
	// layout stable.
	b.AddBits(3, uint32(f.rrect.Type), "rrect_type")
	b.AddBool(ellipticalEffectUsesScale(caps, f.rrect), "scale_radii")
}

func (f *ellipticalRRectFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*ellipticalRRectFP)
	return ok && f.edgeType == o.edgeType && rrectsEqual(f.rrect, o.rrect)
}

func (f *ellipticalRRectFP) onMakeProgramImpl() FPProgramImpl {
	return &ellipticalRRectFPImpl{}
}

type ellipticalRRectFPImpl struct {
	FPImplBase
	innerRectUni   UniformHandle
	invRadiiSqdUni UniformHandle
	scaleUni       UniformHandle
	usesScale      bool
}

func (i *ellipticalRRectFPImpl) EmitCode(args *FPEmitArgs) {
	fp := args.FP.(*ellipticalRRectFP)
	// The inner rect is the rrect bounds inset by the x/y radii.
	rectName := ""
	i.innerRectUni, rectName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "innerRect")

	fragCoord := args.FragBuilder.FragmentPosition()
	f := args.FragBuilder
	// At each quarter-ellipse corner we compute a vector that is the offset of the fragment position from the ellipse
	// center, pinned in x and y to the quarter-plane relevant to that corner.
	f.CodeAppendf("vec2 dxy0 = %s.xy - %s;", rectName, fragCoord)
	f.CodeAppendf("vec2 dxy1 = %s - %s.zw;", fragCoord, rectName)

	scaleName := ""
	i.usesScale = ellipticalEffectUsesScale(args.ShaderCaps, fp.rrect)
	if i.usesScale {
		i.scaleUni, scaleName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
			GLSLTypeHalf2, "scale")
	}

	// Simple type: the radii uniform is (1/rx^2, 1/ry^2) — highp to prevent underflow.
	invRadiiXYSqdName := ""
	i.invRadiiSqdUni, invRadiiXYSqdName = args.UniformHandler.AddUniform(args.FP,
		ShaderFlagFragment, GLSLTypeFloat2, "invRadiiXY")
	f.CodeAppend("vec2 dxy = max(max(dxy0, dxy1), 0.0);")
	if i.usesScale {
		f.CodeAppendf("dxy *= %s.y;", scaleName)
	}
	// Z is the x/y offsets divided by the squared radii.
	f.CodeAppendf("vec2 Z = dxy * %s.xy;", invRadiiXYSqdName)
	// implicit is the evaluation of (x/a)^2 + (y/b)^2 - 1.
	f.CodeAppend("float implicit = dot(Z, dxy) - 1.0;")
	// grad_dot is the squared length of the gradient of the implicit.
	f.CodeAppend("float grad_dot = 4.0 * dot(Z, Z);")
	// Avoid calling inversesqrt on zero.
	f.CodeAppend("grad_dot = max(grad_dot, 1.0e-4);")
	f.CodeAppend("float approx_dist = implicit * inversesqrt(grad_dot);")
	if i.usesScale {
		f.CodeAppendf("approx_dist *= %s.x;", scaleName)
	}
	if fp.edgeType == ClipEdgeFillAA {
		f.CodeAppend("float alpha = clamp(0.5 - approx_dist, 0.0, 1.0);")
	} else {
		f.CodeAppend("float alpha = clamp(0.5 + approx_dist, 0.0, 1.0);")
	}
	f.CodeAppendf("return %s * alpha;", i.InvokeChild(0, args))
}

func (i *ellipticalRRectFPImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*ellipticalRRectFP)
	r := rrectRadii(f.rrect)
	if r.X < rrectRadiusMin || r.Y < rrectRadiusMin {
		panic("radii below the effect's minimum")
	}
	rect := f.rrect.Rect.Inset(r.X, r.Y)
	if i.usesScale {
		if r.X > r.Y {
			pdman.Set2f(i.invRadiiSqdUni, 1, (r.X*r.X)/(r.Y*r.Y))
			pdman.Set2f(i.scaleUni, r.X, 1/r.X)
		} else {
			pdman.Set2f(i.invRadiiSqdUni, (r.Y*r.Y)/(r.X*r.X), 1)
			pdman.Set2f(i.scaleUni, r.Y, 1/r.Y)
		}
	} else {
		pdman.Set2f(i.invRadiiSqdUni, 1/(r.X*r.X), 1/(r.Y*r.Y))
	}
	pdman.Set4f(i.innerRectUni, rect.Left, rect.Top, rect.Right, rect.Bottom)
}

// RRectEffectFP builds the appropriate clip fragment processor for rrect. With uniform radii, the complex/nine-patch
// corner analysis reduces to the rect/oval/simple lanes.
func RRectEffectFP(inputFP FragmentProcessor, edgeType ClipEdgeType, rrect geom.RRect, caps *gpu.ShaderCaps) (FragmentProcessor, bool) {
	switch rrect.Type {
	case geom.RRectRect:
		return RectClipFP(inputFP, edgeType, rrect.Rect), true
	case geom.RRectOval:
		return OvalEffectFP(inputFP, edgeType, rrect.Rect, caps)
	case geom.RRectSimple:
		if rrect.RadiusX < rrectRadiusMin || rrect.RadiusY < rrectRadiusMin {
			// The corners are extremely close to rectangular, so collapse the clip to a rectangular clip.
			return RectClipFP(inputFP, edgeType, rrect.Rect), true
		}
		if rrectIsSimpleCircular(rrect) {
			return newCircularRRectFP(inputFP, edgeType, rrect)
		}
		return newEllipticalRRectFP(inputFP, edgeType, rrect)
	default:
		return inputFP, false
	}
}
