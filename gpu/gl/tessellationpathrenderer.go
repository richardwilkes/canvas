// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// TessellationPathRenderer is the tie-in point for path rendering via the fixed-count tessellation ops. This renderer
// draws paths using a hybrid Red Book "stencil, then cover" method; curves get linearized by the tessellation shaders.
// It doesn't apply analytic AA, so it requires MSAA if AA is desired. Strokes and hairlines are drawn directly with
// StrokeTessellateOp.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// FillPathFlags is sent to the internal path filling ops to control how a path gets rendered. FillPathFlagWireframe is
// never set here: it would come from a debugging option this package doesn't expose.
type FillPathFlags uint8

// FillPathFlags bit values.
const (
	FillPathFlagsNone       FillPathFlags = 0
	FillPathFlagStencilOnly FillPathFlags = 1 << 0
	FillPathFlagWireframe   FillPathFlags = 1 << 1
)

// makeNonConvexFillOp chooses between an inner-triangulate op (large and/or simple enough paths triangulate the inner
// fan on the CPU — the fastest approach, stenciling only the curves and filling the majority of pixels in a single
// render pass) and a stencil-then-cover op. The paint is consumed.
func makeNonConvexFillOp(fillPathFlags FillPathFlags, aaType gpu.AAType, drawBounds geom.Rect, clipBounds geom.IRect, viewMatrix *geom.Matrix, p *path.Path, paint *Paint) DrawOp {
	if p.IsConvex() && !p.IsInverseFillType() {
		panic("non-convex fill op requires a non-convex or inverse-filled path")
	}
	numVerbs := p.CountVerbs()
	if numVerbs > 0 && !p.IsInverseFillType() {
		// Check if the path is large and/or simple enough that we can triangulate the inner fan on the CPU.
		clippedDrawBounds := clipBounds.ToRect()
		if clippedDrawBounds.Intersect(drawBounds) {
			gpuFragmentWork := clippedDrawBounds.Height() * clippedDrawBounds.Width()
			cpuTessellationWork := float32(numVerbs) * float32(skNextLog2(numVerbs)) // N log N
			const cpuWeight = 512
			const minNumPixelsToTriangulate = 256 * 256
			if cpuTessellationWork*cpuWeight+minNumPixelsToTriangulate < gpuFragmentWork {
				return newPathInnerTriangulateOp(viewMatrix, p, paint, aaType, fillPathFlags,
					drawBounds)
			}
		} // We should be clipped out when the clip is analyzed, so just use the default op.
	}
	return newPathStencilCoverOp(viewMatrix, p, paint, aaType, fillPathFlags, drawBounds)
}

// tessChopPathIfNecessary, when the path is extremely large, pre-chops its curves to keep the number of tessellation
// segments tractable (this also flattens curves that fall completely outside the viewport). choppedPath may be nil, in
// which case no chopping actually happens — the return value then reports whether chopping is possible.
func tessChopPathIfNecessary(viewMatrix *geom.Matrix, shape *StyledShape, clipConservativeBounds geom.IRect, rec *stroke.Rec, choppedPath **path.Path) bool {
	pathDevBounds, _ := viewMatrix.MapRect(shape.Bounds())
	n4 := wangsWorstCaseCubicP4(tessPrecision, pathDevBounds.Width(), pathDevBounds.Height())
	if n4 > tessMaxSegmentsPerCurveP4 && shape.SegmentMask() != path.SegmentLine {
		viewport := clipConservativeBounds.ToRect()
		if !shape.Style().IsSimpleFill() {
			// Outset the viewport to pad for the stroke width.
			var inflationRadius float32
			if rec.IsHairlineStyle() {
				// Rec.InflationRadius() doesn't handle hairlines robustly. Instead find the inflation of an
				// equivalent stroke in device space with a width of 1.
				inflationRadius = stroke.GetInflationRadius(rec.Join(), rec.Miter(), rec.Cap(), 1)
			} else {
				inflationRadius = rec.InflationRadius() * viewMatrix.MaxScale()
			}
			viewport = viewport.Outset(inflationRadius, inflationRadius)
		}
		if wangsWorstCaseCubic(tessPrecision, viewport.Width(),
			viewport.Height()) > tessMaxSegmentsPerCurve {
			return false
		}
		if choppedPath != nil {
			*choppedPath = preChopPathCurves(tessPrecision, *choppedPath, viewMatrix, viewport)
		}
	}
	return true
}

// TessellationPathRenderer draws paths via the fixed-count tessellation ops.
type TessellationPathRenderer struct{}

// NewTessellationPathRenderer returns the renderer (it is stateless).
func NewTessellationPathRenderer() *TessellationPathRenderer {
	return &TessellationPathRenderer{}
}

// TessellationPathRendererIsSupported reports whether the current caps allow this path renderer to be used.
func TessellationPathRendererIsSupported(caps *Caps) bool {
	return !caps.AvoidStencilBuffers &&
		caps.DrawInstancedSupport &&
		!caps.DisableTessellationPathRendererCap
}

// Name implements PathRenderer.
func (r *TessellationPathRenderer) Name() string { return "Tessellation" }

// OnGetStencilSupport implements PathRenderer.
func (r *TessellationPathRenderer) OnGetStencilSupport(shape *StyledShape) StencilSupport {
	if !shape.Style().IsSimpleFill() || shape.InverseFilled() {
		// Don't bother with stroke stenciling or inverse fills yet: clipping by a stroke isn't supported, and the
		// stenciling code already knows how to invert a fill.
		return StencilSupportNone
	}
	if shape.KnownToBeConvex() {
		return StencilSupportNoRestriction
	}
	return StencilSupportStencilOnly
}

// OnCanDrawPath implements PathRenderer.
func (r *TessellationPathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	shape := args.Shape
	if args.AAType == gpu.AATypeCoverage ||
		shape.Style().HasPathEffect() ||
		args.ViewMatrix.HasPerspective() ||
		shape.Style().Rec().Style() == stroke.StyleStrokeAndFill ||
		!args.Proxy.CanUseStencil(args.Caps) {
		return CanDrawPathNo
	}
	if !shape.Style().IsSimpleFill() {
		if shape.InverseFilled() {
			return CanDrawPathNo
		}
		if shape.Style().Rec().Width()*args.ViewMatrix.MaxScale() > 10000 {
			// crbug.com/1266446 -- Don't draw massively wide strokes with the tessellator. Since we outset the viewport
			// by stroke width for pre-chopping, astronomically wide strokes can result in an astronomical viewport
			// size, and therefore an exponential explosion of chops and memory usage. It is also simply inefficient to
			// tessellate these strokes due to the number of radial edges required. We're better off just converting
			// them to a path after a certain point.
			return CanDrawPathNo
		}
	}
	if args.HasUserStencilSettings {
		// Non-convex paths and strokes use the stencil buffer internally, so they can't support draws with stencil
		// settings.
		if !shape.Style().IsSimpleFill() || !shape.KnownToBeConvex() || shape.InverseFilled() {
			return CanDrawPathNo
		}
	}

	// By passing in nil for the chopped path, no chopping happens; rather this returns whether chopping is possible.
	if !tessChopPathIfNecessary(args.ViewMatrix, shape, args.ClipConservativeBounds,
		shape.Style().Rec(), nil) {
		return CanDrawPathNo
	}

	return CanDrawPathYes
}

// OnDrawPath implements PathRenderer.
func (r *TessellationPathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	sdc := args.SurfaceDrawContext

	p := args.Shape.AsPath()

	// OnDrawPath is only called when tessChopPathIfNecessary succeeded in OnCanDrawPath.
	if !tessChopPathIfNecessary(args.ViewMatrix, args.Shape, args.ClipConservativeBounds,
		args.Shape.Style().Rec(), &p) {
		panic("chopping succeeded in onCanDrawPath but failed in onDrawPath")
	}

	// Handle strokes first.
	if !args.Shape.Style().IsSimpleFill() {
		if p.IsInverseFillType() {
			panic("stroked shapes cannot be inverse filled (see onGetStencilSupport)")
		}
		if !args.UserStencilSettings.IsUnused() {
			panic("stroked shapes cannot carry user stencil settings")
		}
		rec := args.Shape.Style().Rec()
		if rec.Style() == stroke.StyleStrokeAndFill {
			panic("stroke-and-fill styles are rejected in onCanDrawPath")
		}
		op := newStrokeTessellateOp(args.AAType, args.ViewMatrix, p, rec, args.Paint)
		sdc.AddDrawOp(args.Clip, op)
		return true
	}

	// Handle empty paths.
	pathDevBounds, _ := args.ViewMatrix.MapRect(args.Shape.Bounds())
	if pathDevBounds.IsEmpty() {
		if p.IsInverseFillType() {
			sdc.DrawPaint(args.Clip, args.Paint, args.ViewMatrix)
		}
		return true
	}

	// Handle convex paths. Make sure to check the chopped path for convexity (it may have been pre-chopped, unlike the
	// shape).
	if p.IsConvex() && !p.IsInverseFillType() {
		op := newPathTessellateOp(args.AAType, args.UserStencilSettings, args.ViewMatrix, p,
			args.Paint, pathDevBounds)
		sdc.AddDrawOp(args.Clip, op)
		return true
	}

	if !args.UserStencilSettings.IsUnused() {
		panic("non-convex draws cannot carry user stencil settings (see onGetStencilSupport)")
	}
	drawBounds := pathDevBounds
	if p.IsInverseFillType() {
		drawBounds = sdc.AsSurfaceProxy().BackingStoreBoundsIRect().ToRect()
	}
	op := makeNonConvexFillOp(FillPathFlagsNone, args.AAType, drawBounds,
		args.ClipConservativeBounds, args.ViewMatrix, p, args.Paint)
	sdc.AddDrawOp(args.Clip, op)
	return true
}

// tessMarkStencil is the stencil settings used to mark stencil coverage before a "stencil, then cover" fill.
var tessMarkStencil = MakeUserStencilSettings(0x0001, UserStencilTestAlways, 0xffff,
	UserStencilOpReplace, UserStencilOpKeep, 0xffff)

// OnStencilPath implements PathRenderer.
func (r *TessellationPathRenderer) OnStencilPath(args *StencilPathArgs) {
	if !args.Shape.Style().IsSimpleFill() || args.Shape.InverseFilled() {
		panic("onStencilPath requires a simple non-inverse fill (see onGetStencilSupport)")
	}

	sdc := args.SurfaceDrawContext
	aaType := gpu.AATypeNone
	if args.DoStencilMSAA == gpu.AAYes {
		aaType = gpu.AATypeMSAA
	}

	pathDevBounds, _ := args.ViewMatrix.MapRect(args.Shape.Bounds())

	p := args.Shape.AsPath()

	// Pre-chop exactly as OnCanDrawPath/OnDrawPath do, via tessChopPathIfNecessary. That helper skips line-only paths
	// (no curves to chop) and, when the conservative clip viewport is too large for preChopPathCurves to handle, leaves
	// the path unchopped rather than crashing. A bare preChopPathCurves call here — lacking both guards — would panic
	// ("preChopPathCurves viewport is too large") on a large straight-line-only non-convex stencil/clip element, which
	// OnCanDrawPath happily accepts. Since OnStencilPath is always a simple fill, the helper's stroke-outset branch is
	// inert and its return value (whether chopping was possible) is irrelevant here.
	tessChopPathIfNecessary(args.ViewMatrix, args.Shape, args.ClipConservativeBounds,
		args.Shape.Style().Rec(), &p)

	// Make sure to check the chopped path for convexity (it may have been pre-chopped, unlike the shape).
	if p.IsConvex() {
		stencilPaint := NewPaint()
		stencilPaint.SetXPFactory(DisableColorXPFactory())
		op := newPathTessellateOp(aaType, &tessMarkStencil, args.ViewMatrix, p, stencilPaint,
			pathDevBounds)
		sdc.AddDrawOp(args.Clip, op)
		return
	}

	op := makeNonConvexFillOp(FillPathFlagStencilOnly, aaType, pathDevBounds,
		args.ClipConservativeBounds, args.ViewMatrix, p, NewPaint())
	sdc.AddDrawOp(args.Clip, op)
}
