// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The SurfaceDrawContext draw entry points: DrawRect (with stroke styles via StrokeRectOp), DrawRRect, DrawOval,
// DrawArc, DrawStrokedLine, DrawShape, drawSimpleShape (including the nested-rect StrokeRectOp lane), and
// drawShapeUsingPathRenderer over the path-renderer chain. Styles carry no path effect (they devolve to styled paths at
// the device). The DMSAA lanes (paths always trigger MSAA, the FillRRectOp preferences, and the StrokeRectOp bevel
// avoidance) are live.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/stroke"
)

// simpleFillStrokeRec returns the stroke rec for a simple (unstroked) fill style.
func simpleFillStrokeRec() stroke.Rec { return stroke.NewStrokeRec(stroke.InitStyleFill) }

// clipConservativeBounds returns clip's conservative bounds, or the full surface bounds if clip is nil.
func (sdc *SurfaceDrawContext) clipConservativeBounds(clip Clip) geom.IRect {
	if clip != nil {
		return clip.GetConservativeBounds()
	}
	return geom.IRectSize(sdc.Dimensions())
}

// DrawRect draws rect, filled or stroked according to style. style may be nil for a simple fill; path effects must have
// been devolved to a path by the device. The paint is consumed.
func (sdc *SurfaceDrawContext) DrawRect(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, rect geom.Rect, style *stroke.Rec) {
	if style == nil {
		fill := simpleFillStrokeRec()
		style = &fill
	}
	if sdc.ctx.Abandoned() {
		return
	}

	if style.Style() == stroke.StyleFill {
		// Fills the rect, using rect as its own local coordinates.
		sdc.FillRectToRect(clip, paint, aa, viewMatrix, rect, rect)
		return
	} else if (style.Style() == stroke.StyleStroke || style.Style() == stroke.StyleHairline) &&
		rect.Width() != 0 && rect.Height() != 0 {
		// Only use the StrokeRectOp for non-empty rectangles. Empty rectangles will be processed by the styled-shape
		// lane to handle stroke caps and dashing properly.
		//
		// skbug.com/40043303 — there is a double-blend issue with the bevel version of AAStrokeRectOp, and increasing
		// the AA bloat for MSAA makes it more pronounced, so the bevel version is not used with DMSAA.
		var aaType gpu.AAType
		if sdc.canUseDynamicMSAA && style.Join() == stroke.JoinMiter &&
			style.Miter() >= float32(math.Sqrt2) {
			aaType = gpu.AATypeCoverage
		} else {
			aaType = sdc.chooseAAType(aa)
		}
		op := MakeStrokeRectOp(sdc.Caps(), paint, aaType, viewMatrix, rect, style)
		// op may be nil if the stroke is not supported or if using coverage aa and the view matrix does not preserve
		// rectangles.
		if op != nil {
			sdc.AddDrawOp(clip, op)
			return
		}
	}
	styled := MakeStyledShapeRect(rect, MakeStyle(*style), DoSimplifyYes)
	sdc.drawShapeUsingPathRenderer(clip, paint, aa, viewMatrix, &styled, false)
}

// DrawAtlas draws the sprite arrays (xforms, texRect, colors) as a single non-AA DrawAtlasOp. The paint is consumed.
func (sdc *SurfaceDrawContext) DrawAtlas(clip Clip, paint *Paint, viewMatrix *geom.Matrix, xforms []geom.RSXform, texRect []geom.Rect, colors []colorcore.Color) {
	if sdc.ctx.Abandoned() {
		return
	}
	aaType := sdc.chooseAAType(gpu.AANo)
	op := NewDrawAtlasOp(paint, viewMatrix, aaType, xforms, texRect, colors)
	sdc.AddDrawOp(clip, op)
}

// DrawRRect draws rrect, filled or stroked according to style. The paint is consumed.
func (sdc *SurfaceDrawContext) DrawRRect(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, rrect geom.RRect, style *stroke.Rec) {
	if style == nil {
		fill := simpleFillStrokeRec()
		style = &fill
	}
	if sdc.ctx.Abandoned() {
		return
	}
	if style.Style() == stroke.StyleFill && rrect.IsEmpty() {
		return
	}

	aaType := sdc.chooseAAType(aa)

	var op DrawOp
	if aaType == gpu.AATypeCoverage && !sdc.canUseDynamicMSAA && rrect.Type == geom.RRectSimple &&
		rrect.RadiusX == rrect.RadiusY &&
		viewMatrix.RectStaysRect() && viewMatrix.IsSimilarity() {
		// In specific cases we use a dedicated circular round rect op to try and get better perf.
		op = MakeCircularRRectOp(paint, viewMatrix, rrect, style)
	}
	if op == nil && style.IsFillStyle() {
		op = NewFillRRectOp(sdc.Caps(), paint, viewMatrix, rrect,
			LocalCoordsFromRect(rrect.Rect), gpu.AA(aaType != gpu.AATypeNone))
	}
	if op == nil && (aaType == gpu.AATypeCoverage || sdc.canUseDynamicMSAA) {
		op = MakeRRectOp(sdc.Caps(), paint, viewMatrix, rrect, style)
	}
	if op != nil {
		sdc.AddDrawOp(clip, op)
		return
	}

	styled := MakeStyledShapeRRect(rrect, MakeStyle(*style), DoSimplifyYes)
	sdc.drawShapeUsingPathRenderer(clip, paint, aa, viewMatrix, &styled, false)
}

// DrawOval draws oval, filled or stroked according to style. The paint is consumed.
func (sdc *SurfaceDrawContext) DrawOval(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, oval geom.Rect, style *stroke.Rec) {
	if style == nil {
		fill := simpleFillStrokeRec()
		style = &fill
	}
	if sdc.ctx.Abandoned() {
		return
	}

	if oval.IsEmpty() {
		if style.Style() == stroke.StyleFill {
			return
		}
		sdc.DrawRect(clip, paint, aa, viewMatrix, oval, style)
		return
	}

	aaType := sdc.chooseAAType(aa)

	var op DrawOp
	if aaType == gpu.AATypeCoverage && !sdc.canUseDynamicMSAA &&
		oval.Width() > geom.ScalarNearlyZeroTol &&
		oval.Width() == oval.Height() && viewMatrix.IsSimilarity() {
		// In specific cases we use a dedicated circle op to try and get better perf.
		op = MakeCircleOp(paint, viewMatrix, oval, style)
	}
	if op == nil && style.IsFillStyle() {
		// FillRRectOp has special geometry and a fragment-shader branch to conditionally evaluate the arc equation.
		// This same special geometry and fragment branch also turn out to be a substantial optimization for drawing
		// ovals (namely, by not evaluating the arc equation inside the oval's inner diamond). Given these
		// optimizations, it's a clear win to draw ovals the exact same way we do round rects.
		op = NewFillRRectOp(sdc.Caps(), paint, viewMatrix,
			geom.MakeRRect(oval, oval.Width()/2, oval.Height()/2), LocalCoordsFromRect(oval),
			gpu.AA(aaType != gpu.AATypeNone))
	}
	if op == nil && (aaType == gpu.AATypeCoverage || sdc.canUseDynamicMSAA) {
		op = MakeOvalOp(sdc.Caps(), paint, viewMatrix, oval, style)
	}
	if op != nil {
		sdc.AddDrawOp(clip, op)
		return
	}

	// The oval's path representation starts at index 2 to match the historical winding used by callers that expect a
	// specific starting point.
	styled := MakeStyledShapeRRectWithWinding(
		geom.MakeRRect(oval, oval.Width()/2, oval.Height()/2), geom.DirectionCW, 2, false,
		MakeStyle(*style), DoSimplifyYes,
	)
	sdc.drawShapeUsingPathRenderer(clip, paint, aa, viewMatrix, &styled, false)
}

// DrawArc draws an arc of oval, filled or stroked according to style. Angles are in degrees. The paint is consumed.
func (sdc *SurfaceDrawContext) DrawArc(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, style *stroke.Rec) {
	if style == nil {
		fill := simpleFillStrokeRec()
		style = &fill
	}
	if sdc.ctx.Abandoned() {
		return
	}

	aaType := sdc.chooseAAType(aa)
	if aaType == gpu.AATypeCoverage {
		op := MakeArcOp(paint, viewMatrix, oval, startAngle, sweepAngle, useCenter, style)
		if op != nil {
			sdc.AddDrawOp(clip, op)
			return
		}
	}

	styled := MakeStyledShapeArc(oval, startAngle, sweepAngle, useCenter, MakeStyle(*style),
		DoSimplifyYes)
	sdc.drawShapeUsingPathRenderer(clip, paint, aa, viewMatrix, &styled, false)
}

// FillQuadWithEdgeAA fills a general quadrilateral (in the order top-left, top-right, bottom-right, bottom-left) with
// per-edge AA. localPoints may be nil to reuse points. The paint is consumed.
func (sdc *SurfaceDrawContext) FillQuadWithEdgeAA(clip Clip, paint *Paint, edgeAA gpu.QuadAAFlags, viewMatrix *geom.Matrix, points, localPoints *[4]geom.Point) {
	local := points
	if localPoints != nil {
		local = localPoints
	}
	identity := geom.IdentityMatrix()
	quad := DrawQuad{
		Device: MakeQuadFromPoints(points, viewMatrix),
		Local:  MakeQuadFromPoints(local, &identity), EdgeFlags: edgeAA,
	}
	sdc.DrawFilledQuad(clip, paint, &quad, nil)
}

// DrawStrokedLine draws a butt- or square-capped stroked line as an oriented quad, preferring FillRRectOp under DMSAA
// (or reduced shader mode). The paint is consumed.
func (sdc *SurfaceDrawContext) DrawStrokedLine(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, points [2]geom.Point, style *stroke.Rec) {
	if style.Style() != stroke.StyleStroke {
		panic("drawStrokedLine requires a stroke style")
	}
	if style.Width() <= 0 {
		panic("drawStrokedLine requires a positive width")
	}
	if style.Cap() == stroke.CapRound {
		panic("drawStrokedLine does not support round caps")
	}

	halfWidth := style.Width() / 2
	if halfWidth <= 0 {
		// Prevents underflow when the stroke width is epsilon > 0 (so technically not a hairline). At any reasonable
		// scale this line's coverage is negligible, so discarding the draw is visually equivalent.
		return
	}

	parallel := geom.Point{X: points[1].X - points[0].X, Y: points[1].Y - points[0].Y}
	if !parallel.Normalize() {
		parallel = geom.Point{X: 1, Y: 0}
	}
	parallel = geom.Point{X: parallel.X * halfWidth, Y: parallel.Y * halfWidth}

	ortho := geom.Point{X: parallel.Y, Y: -parallel.X}
	p0 := points[0]
	p1 := points[1]
	if style.Cap() == stroke.CapSquare {
		// Extra extension for square caps.
		p0 = geom.Point{X: p0.X - parallel.X, Y: p0.Y - parallel.Y}
		p1 = geom.Point{X: p1.X + parallel.X, Y: p1.Y + parallel.Y}
	}

	// If we are using dmsaa or reduced shader mode then attempt to draw with FillRRectOp.
	if sdc.Caps().DrawInstancedSupport &&
		(sdc.alwaysAntialias() || (sdc.Caps().ReducedShaderMode() && aa == gpu.AAYes)) {
		var localMatrix geom.Matrix
		localMatrix.SetAll(p1.X-p0.X, ortho.X, p0.X,
			p1.Y-p0.Y, ortho.Y, p0.Y,
			0, 0, 1)
		var combined geom.Matrix
		combined.SetConcat(viewMatrix, &localMatrix)
		if op := NewFillRRectOp(sdc.Caps(), paint, &combined,
			geom.MakeRRect(geom.Rect{Left: 0, Top: -1, Right: 1, Bottom: 1}, 0, 0),
			LocalCoordsFromMatrix(&localMatrix), gpu.AAYes); op != nil {
			sdc.AddDrawOp(clip, op)
			return
		}
	}

	// Order is TL, TR, BR, BL where arbitrarily "down" is p0 to p1 and "right" is positive.
	corners := [4]geom.Point{
		{X: p0.X - ortho.X, Y: p0.Y - ortho.Y},
		{X: p0.X + ortho.X, Y: p0.Y + ortho.Y},
		{X: p1.X + ortho.X, Y: p1.Y + ortho.Y},
		{X: p1.X - ortho.X, Y: p1.Y - ortho.Y},
	}

	edgeAA := gpu.QuadAAFlagsNone
	if aa == gpu.AAYes {
		edgeAA = gpu.QuadAAFlagsAll
	}
	sdc.FillQuadWithEdgeAA(clip, paint, edgeAA, viewMatrix, &corners, nil)
}

// drawSimpleShape attempts to draw shape with a dedicated op (stroked line, rect/oval/rrect, or nested-rect stroke),
// skipping general path rendering. Returns true when the shape was consumed.
func (sdc *SurfaceDrawContext) drawSimpleShape(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, shape *StyledShape) bool {
	// No path effects exist, so this whole body assumes none apply.
	aaType := sdc.chooseAAType(aa)
	// We can ignore the starting point and direction since there is no path effect.
	if linePts, inverted, isLine := shape.AsLine(); isLine && !inverted &&
		shape.Style().Rec().Style() == stroke.StyleStroke &&
		shape.Style().Rec().Cap() != stroke.CapRound {
		// The stroked line is an oriented rectangle, which looks the same or better (if perspective) compared to path
		// rendering. The exception is subpixel/hairline lines that are non-AA or MSAA, in which case the default path
		// renderer achieves higher quality.
		_, treatAsHairline := glDrawTreatAAStrokeAsHairline(
			shape.Style().Rec().Width(), viewMatrix,
		)
		if aaType == gpu.AATypeCoverage || !treatAsHairline {
			sdc.DrawStrokedLine(clip, paint, aa, viewMatrix, linePts,
				shape.Style().Rec())
			return true
		}
		return false
	}
	if rrect, inverted, isRRect := shape.AsRRect(); isRRect && !inverted {
		switch rrect.Type {
		case geom.RRectRect:
			sdc.DrawRect(clip, paint, aa, viewMatrix, rrect.Rect, shape.Style().Rec())
		case geom.RRectOval:
			sdc.DrawOval(clip, paint, aa, viewMatrix, rrect.Rect, shape.Style().Rec())
		default:
			sdc.DrawRRect(clip, paint, aa, viewMatrix, rrect, shape.Style().Rec())
		}
		return true
	}
	if aaType == gpu.AATypeCoverage && shape.Style().IsSimpleFill() &&
		viewMatrix.RectStaysRect() && !sdc.Caps().ReducedShaderMode() {
		// TODO: the rectStaysRect restriction could be lifted if we were willing to apply the matrix to all the points
		// individually rather than just to the rect.
		if rects, ok := shape.AsNestedRects(); ok {
			// Concave AA paths are expensive — try to avoid them for special cases.
			if op := MakeNestedStrokeRectOp(sdc.Caps(), paint, viewMatrix, rects); op != nil {
				sdc.AddDrawOp(clip, op)
				return true
			}
			// Fall through to let the path renderer handle subpixel nested rects with unequal stroke widths along X/Y.
		}
	}
	return false
}

// DrawShapeWithStyle is the entry the device's path draws funnel through: it builds a styled shape without
// simplification and forwards to DrawShape. The paint is consumed.
func (sdc *SurfaceDrawContext) DrawShapeWithStyle(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, shape *Shape, style *stroke.Rec) {
	if sdc.ctx.Abandoned() {
		return
	}
	st := SimpleFillStyle()
	if style != nil {
		st = MakeStyle(*style)
	}
	styled := makeStyledShapeFromShape(shape, st, DoSimplifyNo)
	sdc.DrawShape(clip, paint, aa, viewMatrix, &styled)
}

// DrawShape draws shape via path rendering. The paint is consumed and the shape may be mutated.
func (sdc *SurfaceDrawContext) DrawShape(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, shape *StyledShape) {
	if sdc.ctx.Abandoned() {
		return
	}
	if shape.IsEmpty() {
		if shape.InverseFilled() {
			sdc.DrawPaint(clip, paint, viewMatrix)
		}
		return
	}
	// If we get here in drawShape(), we definitely need to use path rendering.
	sdc.drawShapeUsingPathRenderer(clip, paint, aa, viewMatrix, shape, true)
}

// drawShapeUsingPathRenderer selects a renderer from the chain (simplifying and retrying the dedicated ops first),
// applying the style to the geometry if no renderer accepts the styled form, with the SW renderer as the last resort.
// The paint is consumed and the shape may be mutated.
func (sdc *SurfaceDrawContext) drawShapeUsingPathRenderer(clip Clip, paint *Paint, aa gpu.AA, viewMatrix *geom.Matrix, shape *StyledShape, attemptDrawSimple bool) {
	if sdc.ctx.Abandoned() {
		return
	}
	if !viewMatrix.IsFinite() || !shape.Bounds().IsFinite() {
		return
	}

	clipConservativeBounds := sdc.clipConservativeBounds(clip)

	// Always allow paths to trigger DMSAA.
	var aaType gpu.AAType
	if sdc.canUseDynamicMSAA {
		aaType = gpu.AATypeMSAA
	} else {
		aaType = sdc.chooseAAType(aa)
	}

	canDrawArgs := CanDrawPathArgs{
		Caps:                   sdc.Caps(),
		Proxy:                  sdc.AsRenderTargetProxy(),
		ClipConservativeBounds: clipConservativeBounds,
		ViewMatrix:             viewMatrix,
		Shape:                  shape,
		Paint:                  paint,
		SurfaceProps:           sdc.surfaceProps,
		AAType:                 aaType,
		HasUserStencilSettings: false,
	}

	var pr PathRenderer
	if !shape.Style().Rec().IsFillStyle() && !shape.IsEmpty() {
		// Give the tessellation path renderer a chance to claim this stroke before we simplify it (nil when the chain
		// carries no tessellation renderer).
		if tess := sdc.drawingManager().GetTessellationPathRenderer(); tess != nil &&
			tess.OnCanDrawPath(&canDrawArgs) == CanDrawPathYes {
			pr = tess
		}
	}

	if pr == nil {
		// The shape isn't a stroke that can be drawn directly. Simplify if possible.
		shape.simplify()

		if shape.IsEmpty() && !shape.InverseFilled() {
			return
		}

		if attemptDrawSimple || shape.Simplified() {
			// Usually we enter drawShapeUsingPathRenderer() because the shape+style was too complex for dedicated draw
			// ops. However, if the shape was simplified we ought to try again instead of going right to path rendering.
			if sdc.drawSimpleShape(clip, paint, aa, viewMatrix, shape) {
				return
			}
		}

		// Try a 1st time without applying any of the style to the geometry (and barring sw).
		pr = sdc.drawingManager().GetPathRenderer(&canDrawArgs, false, ChainDrawTypeColor, nil)
	}

	styleScale := MatrixToScaleFactor(viewMatrix)
	if styleScale == 0 {
		return
	}

	// A retry with just the path effect applied would go here; path effects devolve at the device, so that lane is
	// unreachable.
	if pr == nil && shape.Style().Applies() {
		*shape = shape.ApplyStyle(StyleApplyPathEffectAndStrokeRec, styleScale)
		if shape.IsEmpty() {
			return
		}
		pr = sdc.drawingManager().GetPathRenderer(&canDrawArgs, false, ChainDrawTypeColor, nil)
	}

	if pr == nil {
		// Fall back on the SW renderer as a last resort.
		if aaType.IsHW() {
			// No point in trying the SW renderer with MSAA.
			aaType = gpu.AATypeCoverage
			canDrawArgs.AAType = aaType
		}
		// This can only fail if (a) the style is not applied (checked above), or (b) the SW renderer's proxy provider
		// is nil, which never happens.
		pr = sdc.drawingManager().GetSoftwarePathRenderer()
		if pr.OnCanDrawPath(&canDrawArgs) == CanDrawPathNo {
			panic("the SW path renderer must accept the fallback draw")
		}
	}

	args := DrawPathArgs{
		Context:                sdc.ctx,
		Paint:                  paint,
		UserStencilSettings:    UnusedStencilSettings(),
		SurfaceDrawContext:     sdc,
		Clip:                   clip,
		ClipConservativeBounds: clipConservativeBounds,
		ViewMatrix:             viewMatrix,
		Shape:                  canDrawArgs.Shape,
		AAType:                 aaType,
	}
	PathRendererDrawPath(pr, &args)
}
