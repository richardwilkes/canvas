// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Lowers canvas draw calls onto the raster scan converters through the raster clip, choosing a blitter per paint.
// Paints carry solid colors or shaders (the raster-pipeline blitter lanes), color filters (folded into the shader or
// paint color), and mask filters (the filterPath/ nine-patch lanes); image filters are handled by the canvas layer
// machinery. Hairlines (zero-width strokes, and AA thin strokes promoted to hairlines) draw through the hairline scan
// converters; wider strokes and stroke-and-fill convert to fill geometry through the stroker
// (stroke.FillPathWithPaint).

package canvas

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stroke"
)

// PointMode selects how DrawPoints interprets its points: discrete points, independent line segments, or a connected
// polyline.
type PointMode uint8

// PointMode values.
const (
	PointModePoints PointMode = iota
	PointModeLines
	PointModePolygon
)

// draw is the per-draw lowering state: destination pixels, the CTM, and the raster clip.
type draw struct {
	dst *raster.Pixmap
	ctm *geom.Matrix
	rc  *raster.Clip
}

// tooBigForMath rejects bounds whose coordinates could overflow the scan converter's math (also returns true if the
// bounds contain NaN).
func tooBigForMath(bounds geom.Rect) bool {
	// This value is just a guess. smaller is safer, but we don't want to reject largish paths that we don't have to.
	const scaleDownToAllowForSmallMultiplies = 0.25
	const maxV = float32(3.402823466e+38) * scaleDownToAllowForSmallMultiplies

	// use ! expression so we return true if bounds contains NaN
	return !(bounds.Left >= -maxV && bounds.Top >= -maxV && bounds.Right <= maxV && bounds.Bottom <= maxV)
}

// fitsInFixed reports whether the rect fits in 16.16 fixed-point (conservative: |coord| <= 32767).
func fitsInFixed(r geom.Rect) bool {
	return geom.ScalarAbs(r.Left) <= 32767 && geom.ScalarAbs(r.Top) <= 32767 &&
		geom.ScalarAbs(r.Right) <= 32767 && geom.ScalarAbs(r.Bottom) <= 32767
}

// blendFastPath classifies how a paint's blend can be reduced before choosing a blitter.
type blendFastPath uint8

const (
	fastPathNormal      blendFastPath = iota // draw normally
	fastPathSrcOver                          // draw as if in srcover mode
	fastPathSkipDrawing                      // draw nothing
)

// checkFastPath reduces the blend mode when the paint's configuration permits: "solid" means alpha==255 with no color
// filter and no shader; the destination is never treated as opaque for our RGBA8888-premul device.
func checkFastPath(paint *Paint) blendFastPath {
	solid := paint.Color.A() == 0xFF && paint.ColorFilter == nil && paint.Shader == nil
	switch paint.BlendMode {
	case raster.BlendSrcOver:
		return fastPathSrcOver
	case raster.BlendSrc:
		if solid {
			return fastPathSrcOver
		}
		return fastPathNormal
	case raster.BlendDst:
		return fastPathSkipDrawing
	case raster.BlendDstIn:
		if solid {
			return fastPathSkipDrawing
		}
		return fastPathNormal
	default:
		return fastPathNormal
	}
}

// chooseBlitter picks the blitter for a paint: reduce the blend mode via checkFastPath, treat Clear as Src with
// transparent black (dropping the shader and color filter), remove the color filter by folding it into the shader or
// the paint color, veto dither unless it can matter on 8888 (it needs a mask filter or a non-constant shader), then
// pick the lane:
//
//   - a shader takes the raster-pipeline blitter (unless the legacy image lane below applies), with the constant
//     collapse and the srcover→src strength reduction;
//   - solid src-over without dither takes the solid blitter (SolidBlitter selects the black/opaque/general kernels
//     internally), converting a filtered float color by rounding it to 8 bits;
//   - a filtered (float) color or a dither-forced pipeline goes through the raster pipeline's constant collapse
//     (NewBlendBlitterPM4f keeps the float color — dither itself never applies to constant pipelines);
//   - everything else takes the general BlendBlitter.
func chooseBlitter(dst *raster.Pixmap, paint *Paint, ctm *geom.Matrix) raster.Blitter {
	mode := paint.BlendMode
	color := paint.Color
	shader := paint.Shader
	filter := paint.ColorFilter

	// We have the most fast-paths for SrcOver, so see if we can act like SrcOver.
	if mode != raster.BlendSrcOver {
		switch checkFastPath(paint) {
		case fastPathSrcOver:
			mode = raster.BlendSrcOver
		case fastPathSkipDrawing:
			return raster.NullBlitter{}
		default:
		}
	}

	// A Clear blend mode will ignore the entire color pipeline, as if Src mode with 0x00000000.
	if mode == raster.BlendClear {
		mode = raster.BlendSrc
		color = colorcore.Color(0)
		shader = nil
		filter = nil
	}

	// Remove the color filter by folding it away. When it folds into the paint color, the color becomes float-valued
	// (colorF4) — the pipeline lanes must keep those floats rather than rounding early.
	var colorF4 colorcore.Color4f
	haveFloatColor := false
	if filter != nil {
		if shader != nil {
			// The color-filter shader wrapper modulates the shader color by paint alpha before applying the filter, so
			// reset the paint to opaque.
			shader = shaders.NewColorFilterShader(shader, float32(color.A())*(1.0/255.0), filter)
			color = color.WithAlpha(0xFF)
		} else {
			// Fold into the paint color: premul, run the filter, pin alpha, unpremul.
			out, ok := shaders.FilterColor4f(filter, colorcore.Color4fFromColor(color).Premul())
			if !ok {
				out = colorcore.PMColor4f{}
			}
			if out.A < 0 {
				out.A = 0
			} else if out.A > 1 {
				out.A = 1
			}
			colorF4 = out.Unpremul()
			haveFloatColor = true
		}
	}

	// Dither on 8888 matters only with a mask filter or a non-constant shader; the paint flag is ignored otherwise.
	dither := paint.Dither && (paint.MaskFilter != nil || (shader != nil && !shader.IsConstant()))

	if shader != nil {
		// The legacy image lane: a src-over, non-dithered paint whose shader can produce a legacy context (only the
		// image shader can) draws through the fixed-point bitmap sampler instead of the raster pipeline. (Mask filters
		// do not veto this lane — only the unreachable 3D mask format would.) The drawAtlas lowering forces the
		// pipeline lane instead.
		if mode == raster.BlendSrcOver && !dither && !paint.forceRasterPipeline {
			if legacy := shaders.MakeLegacyImageContext(shader, *ctm, color.A()); legacy != nil {
				// Pass the context itself (it satisfies raster.SpanShader32) rather than a bound legacy.ShadeSpan32
				// method value, which would heap-allocate a closure each draw; recycleBlitter returns the context to
				// its pool via the blitter's Shade accessor.
				return raster.NewLegacyShaderBlitter(dst, legacy, legacy.OpaqueAlpha())
			}
		}

		// The raster-pipeline blitter: shader stages + paint-alpha scale; a nil pipeline means the shader cannot draw.
		pipeline := shaders.Compile(shader, *ctm, color)
		if pipeline == nil {
			return raster.NullBlitter{}
		}
		if shader.IsConstant() {
			// The constant collapse: run the pipeline once, then blend as a uniform color. Dither never applies to
			// constant pipelines. The pipeline is done here (the returned blitter does not hold it), so recycle it now.
			cf := pipeline.EvalConstant()
			shaders.RecyclePipeline(pipeline)
			if cf.A == 1 && mode == raster.BlendSrcOver {
				mode = raster.BlendSrc
			}
			return raster.NewBlendBlitterPM4f(dst, cf, mode)
		}
		if dither {
			pipeline.AppendDither(1.0 / 255.0)
		}
		if shader.IsOpaque() && color.A() == 0xFF && mode == raster.BlendSrcOver {
			mode = raster.BlendSrc
		}
		return raster.NewShaderBlitter(dst, pipeline, mode)
	}

	if haveFloatColor {
		if mode == raster.BlendSrcOver && !dither {
			// The legacy lane reads paint.getColor(): the float color rounded to 8 bits.
			return raster.NewSolidBlitter(dst, colorF4.ToColor())
		}
		// The pipeline lane keeps the float color: constant collapse (clamped) + srcover→src.
		pm := colorF4.Premul()
		pm.R = clampUnit(pm.R)
		pm.G = clampUnit(pm.G)
		pm.B = clampUnit(pm.B)
		pm.A = clampUnit(pm.A)
		if pm.A == 1 && mode == raster.BlendSrcOver {
			mode = raster.BlendSrc
		}
		return raster.NewBlendBlitterPM4f(dst, pm, mode)
	}

	if mode == raster.BlendSrcOver {
		if dither {
			// UseLegacyBlitter rejects dithering paints; the pipeline's constant collapse follows.
			pm := colorcore.Color4fFromColor(color).Premul()
			if pm.A == 1 {
				mode = raster.BlendSrc
			}
			return raster.NewBlendBlitterPM4f(dst, pm, mode)
		}
		return raster.NewSolidBlitter(dst, color)
	}
	return raster.NewBlendBlitter(dst, color, mode)
}

// recycleBlitter returns a pooled blitter (and its pooled shaders.Pipeline) built by chooseBlitter to its pool once the
// draw that used it is done. It is a no-op for the unpooled blitters (SolidBlitter, BlendBlitter, NullBlitter, …).
// Callers must invoke it only after the blitter is fully consumed by the synchronous fill — never on a blitter still in
// use or retained elsewhere. Missing a call is safe (the pool simply allocates a fresh instance next time); recycling
// too early is not.
func recycleBlitter(b raster.Blitter) {
	switch v := b.(type) {
	case *raster.SolidBlitter:
		raster.RecycleSolidBlitter(v)
	case *raster.ShaderBlitter:
		if p, ok := v.Shade().(*shaders.Pipeline); ok {
			shaders.RecyclePipeline(p)
		}
		raster.RecycleShaderBlitter(v)
	case *raster.LegacyShaderBlitter:
		if ctx, ok := v.Shade().(*shaders.BitmapProcContext); ok {
			shaders.RecycleLegacyImageContext(ctx)
		}
		raster.RecycleLegacyShaderBlitter(v)
	}
}

// clampUnit clamps v to [0, 1], as the constant collapse does before storing.
func clampUnit(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// computeConservativeLocalClipBounds returns the device clip bounds (outset 1 for AA/hairline slop) mapped back through
// the inverse CTM.
func (d *draw) computeConservativeLocalClipBounds(localBounds *geom.Rect) bool {
	if d.rc.IsEmpty() {
		return false
	}
	inverse, ok := d.ctm.Invert()
	if !ok {
		return false
	}
	devBounds := d.rc.Bounds().Inset(-1, -1)
	*localBounds, _ = inverse.MapRect(devBounds.ToRect())
	return true
}

// drawPaint fills the whole device through the raster clip.
func (d *draw) drawPaint(paint *Paint) {
	if d.rc.IsEmpty() {
		return
	}
	devRect := geom.IRectWH(d.dst.Width, d.dst.Height)
	blitter := chooseBlitter(d.dst, paint, d.ctm)

	// A full-device shaded fill is the CPU throughput bottleneck, same as the rect-fill lane: a gradient/image span
	// evaluates in scalar Go, one pixel at a time. Split it into horizontal row bands filled concurrently. drawPaint
	// fills an integer rect via FillIRectRasterClip, which is per-row independent for any clip (no AA-rect edge
	// coverage), so banding at integer scanlines is byte-identical to the serial fill — unlike the drawRectFull lane,
	// no simple-rect-clip gate is needed. Each band rebuilds its own blitter (shader blitters carry per-span scratch),
	// so the one chosen above is discarded when we take the parallel path.
	if isBandableShaderBlitter(blitter) {
		if bounds := raster.IRectFillBandBounds(devRect.Top, devRect.Bottom, 0); bounds != nil {
			recycleBlitter(blitter)
			fillIRectShadedParallel(d.dst, d.rc, devRect, paint, d.ctm, bounds)
			return
		}
	}

	raster.FillIRectRasterClip(devRect, d.rc, blitter)
	recycleBlitter(blitter)
}

///////////////////////////////////////////////////////////////////////////////
// rects

// rectType classifies how a rect draw lowers: full path, plain fill, stroked frame, or hairline.
type rectType uint8

const (
	rectTypePath rectType = iota
	rectTypeFill
	rectTypeStroke
	rectTypeHair
)

// computeStrokeSize returns the stroke width mapped through the (rect-stays-rect) matrix, per axis.
func computeStrokeSize(paint *Paint, matrix *geom.Matrix) geom.Point {
	size := matrix.MapVector(geom.Point{X: paint.StrokeWidth, Y: paint.StrokeWidth})
	return geom.Point{X: geom.ScalarAbs(size.X), Y: geom.ScalarAbs(size.Y)}
}

// easyRectJoin reports whether a mitered stroke of the rect keeps square corners (miter limit at least √2), returning
// the per-axis stroke size when it does.
func easyRectJoin(rect geom.Rect, paint *Paint, matrix *geom.Matrix) (geom.Point, bool) {
	if rect.IsEmpty() || paint.Join != JoinMiter || paint.MiterLimit < float32(math.Sqrt2) {
		return geom.Point{}, false
	}
	return computeStrokeSize(paint, matrix), true
}

// computeRectType classifies the rect draw, returning the per-axis stroke size for the stroked-frame case.
func computeRectType(rect geom.Rect, paint *Paint, matrix *geom.Matrix) (rectType, geom.Point) {
	width := paint.StrokeWidth
	zeroWidth := width == 0
	style := paint.Style

	if style == StyleStrokeAndFill && zeroWidth {
		style = StyleFill
	}

	if paint.PathEffect != nil || paint.MaskFilter != nil || !matrix.RectStaysRect() ||
		style == StyleStrokeAndFill {
		return rectTypePath, geom.Point{}
	}
	if style == StyleFill {
		return rectTypeFill, geom.Point{}
	}
	if zeroWidth {
		return rectTypeHair, geom.Point{}
	}
	if strokeSize, ok := easyRectJoin(rect, paint, matrix); ok {
		return rectTypeStroke, strokeSize
	}
	return rectTypePath, geom.Point{}
}

// drawRectAsPath lowers a rect draw to the path lane.
func drawRectAsPath(orig *draw, prePaintRect geom.Rect, paint *Paint, ctm *geom.Matrix) {
	dr := draw{dst: orig.dst, ctm: ctm, rc: orig.rc}
	// Borrow the lowering path from the shared pool so its verb/point storage is reused across draws; drawPath consumes
	// it synchronously (pathIsMutable), so recycling on return is safe.
	p := path.Borrow()
	defer path.Recycle(p)
	p.AddRect(prePaintRect, geom.DirectionCW)
	dr.drawPath(p, paint, nil, true)
}

// drawRect is drawRectFull without a paint matrix (the common form).
func (d *draw) drawRect(prePaintRect geom.Rect, paint *Paint) {
	d.drawRectFull(prePaintRect, paint, nil, nil, nil)
}

// drawRectFull is the general rect draw: paintMatrix pre-concatenates onto the CTM for shading and rect classification,
// while the geometry maps postPaintRect through the plain CTM (the image-draw lane passes the shader's source bounds as
// prePaintRect and the destination rect as postPaintRect). combined provides caller-owned storage for the
// CTM∘paintMatrix concatenation: its address escapes into computeRectType/chooseBlitter, so the image lane passes a
// pooled (heap-resident) matrix to avoid a per-draw heap allocation. It is used only when paintMatrix != nil; the plain
// drawRect caller passes nil for both.
func (d *draw) drawRectFull(prePaintRect geom.Rect, paint *Paint, paintMatrix *geom.Matrix, postPaintRect *geom.Rect, combined *geom.Matrix) {
	if d.rc.IsEmpty() {
		return
	}

	matrix := d.ctm
	if paintMatrix != nil {
		combined.SetConcat(d.ctm, paintMatrix)
		matrix = combined
	}

	rtype, strokeSize := computeRectType(prePaintRect, paint, matrix)

	if rtype == rectTypePath {
		drawRectAsPath(d, prePaintRect, paint, matrix)
		return
	}

	// skip the paintMatrix when transforming the rect by the CTM
	paintRect := prePaintRect
	if paintMatrix != nil {
		paintRect = *postPaintRect
	}
	var corners [2]geom.Point
	corners[0] = geom.Point{X: paintRect.Left, Y: paintRect.Top}
	corners[1] = geom.Point{X: paintRect.Right, Y: paintRect.Bottom}
	d.ctm.MapPointsInPlace(corners[:])
	devRect := geom.RectLTRB(corners[0].X, corners[0].Y, corners[1].X, corners[1].Y).Sorted()

	// look for the quick exit, before we build a blitter
	bbox := devRect
	if paint.Style != StyleFill {
		// extra space for hairlines
		if paint.StrokeWidth == 0 {
			bbox = bbox.Outset(1, 1)
		} else {
			// For rectTypeStroke, strokeSize is already computed.
			ssize := strokeSize
			if rtype != rectTypeStroke {
				ssize = computeStrokeSize(paint, matrix)
			}
			bbox = bbox.Outset(ssize.X/2, ssize.Y/2)
		}
	}
	if tooBigForMath(bbox) {
		return
	}

	if !fitsInFixed(bbox) && rtype != rectTypeHair {
		drawRectAsPath(d, prePaintRect, paint, matrix)
		return
	}

	ir := bbox.RoundOut()
	if d.rc.QuickReject(ir) {
		return
	}

	blitter := chooseBlitter(d.dst, paint, matrix)

	// Large shaded rect fills are the CPU throughput bottleneck: a gradient/image span evaluates in scalar Go, one
	// pixel at a time. Splitting the fill into horizontal row bands filled concurrently speeds it up byte-identically —
	// a rect fill's per-row coverage is row-independent, so banding at integer scanlines matches the serial fill
	// exactly. Each band rebuilds its own blitter (shader blitters carry per-span scratch), so the blitter chosen above
	// is discarded here. Gated to the fill lane, a per-span-scratch shader blitter, and a simple-rect clip (so the band
	// bounds are exact); RectFillBandBounds returns nil when the rect is too short to be worth the goroutine/rebuild
	// overhead.
	if rtype == rectTypeFill && isBandableShaderBlitter(blitter) && d.rc.IsRect() {
		if bounds := raster.RectFillBandBounds(devRect.Top, devRect.Bottom, 0); bounds != nil {
			recycleBlitter(blitter)
			fillRectShadedParallel(d.dst, d.rc, devRect, paint, matrix, paint.AntiAlias, bounds)
			return
		}
	}

	// Fill for both StyleFill and StyleStrokeAndFill: reaching here in the latter case means the stroke is a
	// hairline, which devolves to an effective fill.
	switch rtype {
	case rectTypeFill:
		if paint.AntiAlias {
			raster.AntiFillRect(devRect, d.rc, blitter)
		} else {
			raster.FillRectRasterClip(devRect, d.rc, blitter)
		}
	case rectTypeStroke:
		if paint.AntiAlias {
			raster.AntiFrameRect(devRect, strokeSize, d.rc, blitter)
		} else {
			raster.FrameRect(devRect, strokeSize, d.rc, blitter)
		}
	default: // rectTypeHair
		if paint.AntiAlias {
			raster.AntiHairRect(devRect, d.rc, blitter)
		} else {
			raster.HairRect(devRect, d.rc, blitter)
		}
	}
	recycleBlitter(blitter)
}

///////////////////////////////////////////////////////////////////////////////
// hairline promotion

// fastLen returns a cheap upper-bound approximation of the vector's length.
func fastLen(vec geom.Point) float32 {
	x := geom.ScalarAbs(vec.X)
	y := geom.ScalarAbs(vec.Y)
	if x < y {
		x, y = y, x
	}
	return x + y/2
}

// drawTreatAAStrokeAsHairline reports whether an AA stroke of the given width can draw as a coverage-modulated
// hairline, returning the coverage when it can.
func drawTreatAAStrokeAsHairline(strokeWidth float32, matrix *geom.Matrix) (float32, bool) {
	// We need to try to fake a thick-stroke with a modulated hairline.
	if matrix.HasPerspective() {
		return 0, false
	}

	src := [2]geom.Point{{X: strokeWidth, Y: 0}, {X: 0, Y: strokeWidth}}
	var dst [2]geom.Point
	matrix.MapVectors(dst[:], src[:])
	len0 := fastLen(dst[0])
	len1 := fastLen(dst[1])
	if len0 <= 1 && len1 <= 1 {
		return (len0 + len1) / 2, true
	}
	return 0, false
}

// drawTreatAsHairline reports whether the paint's stroke should draw as a hairline, returning the coverage to modulate
// by.
func drawTreatAsHairline(paint *Paint, matrix *geom.Matrix) (float32, bool) {
	if paint.Style != StyleStroke {
		return 0, false
	}
	if paint.StrokeWidth == 0 {
		return 1, true
	}
	if !paint.AntiAlias {
		return 0, false
	}
	return drawTreatAAStrokeAsHairline(paint.StrokeWidth, matrix)
}

// modifyPaintForHairlines returns a replacement paint when a thin stroke should draw as a (possibly alpha-modulated)
// hairline, else nil.
func modifyPaintForHairlines(origPaint *Paint, matrix *geom.Matrix) *Paint {
	coverage, ok := drawTreatAsHairline(origPaint, matrix)
	if !ok {
		return nil
	}
	if coverage == 1 {
		paint := *origPaint
		paint.StrokeWidth = 0
		return &paint
	}
	if raster.BlendSupportsCoverageAsAlpha(origPaint.BlendMode) {
		// this is the old technique, which we preserve for now so we don't change previous results (testing). the new
		// way seems fine, its just (a tiny bit) different.
		scale := int32(coverage * 256)
		newAlpha := uint8(int32(origPaint.Color.A()) * scale >> 8)
		paint := *origPaint
		paint.StrokeWidth = 0
		paint.Color = paint.Color.WithAlpha(newAlpha)
		return &paint
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// paths, ovals, rrects

// drawOval draws an oval via its path form (ovals always lower to paths).
func (d *draw) drawOval(oval geom.Rect, paint *Paint) {
	if d.rc.IsEmpty() {
		return
	}
	p := path.Borrow()
	defer path.Recycle(p)
	p.AddOval(oval, geom.DirectionCW)
	d.drawPath(p, paint, nil, true)
}

// drawRRect draws a rounded rect: paint configurations the mask-filter nine-patch fast path can handle try it first;
// everything else lowers to the path form.
func (d *draw) drawRRect(rrect geom.RRect, paint *Paint) {
	if d.rc.IsEmpty() {
		return
	}

	// These cases are in the same order as drawPath, which handles each of them.
	_, hairline := drawTreatAsHairline(paint, d.ctm)
	if !hairline && paint.PathEffect == nil && paint.Style == StyleFill && paint.MaskFilter != nil &&
		d.drawRRectNinePatch(rrect, paint) {
		return
	}

	// Now fall back to the default case of using a path. rrect is unison's most common styled primitive, so pooling its
	// lowering path (reused across draws) matters; drawPath consumes it synchronously.
	p := path.Borrow()
	defer path.Recycle(p)
	p.AddRRect(rrect, geom.DirectionCW)
	d.drawPath(p, paint, nil, true)
}

// drawRRectNinePatch tries the mask-filter nine-patch fast path, reporting whether it drew.
func (d *draw) drawRRectNinePatch(rrect geom.RRect, paint *Paint) bool {
	rr, ok := rrect.Transform(d.ctm)
	if !ok {
		return false
	}
	blitter := chooseBlitter(d.dst, paint, d.ctm)
	defer recycleBlitter(blitter)
	if rrect.Type == geom.RRectRect {
		devRect := rr.Rect
		return maskfilter.FilterRects(paint.MaskFilter, []geom.Rect{devRect}, d.ctm, d.rc, blitter) ==
			maskfilter.FilterTrue
	}
	// filterRRect() calls the blitter, so we're done when it succeeds.
	return maskfilter.FilterRRect(paint.MaskFilter, rr, d.ctm, d.rc, blitter)
}

// drawPath strokes or fills the path. prePathMatrix pre-concatenates onto the CTM (or transforms the path when the
// paint strokes); pathIsMutable permits transforming origSrc in place.
func (d *draw) drawPath(origSrc *path.Path, origPaint *Paint, prePathMatrix *geom.Matrix, pathIsMutable bool) {
	if d.rc.IsEmpty() {
		return
	}

	pathPtr := origSrc
	doFill := true
	// tmpPath is the per-draw scratch for the stroked/effected and device-space path. Borrow it from the shared pool so
	// its storage is reused across draws instead of allocated fresh each time; it is fully consumed by the synchronous
	// drawDevPath below, so recycling it on return is safe.
	tmpPath := path.Borrow()
	defer path.Recycle(tmpPath)
	matrix := d.ctm
	var matrixStorage geom.Matrix

	if prePathMatrix != nil {
		if origPaint.PathEffect != nil || origPaint.Style != StyleFill {
			result := pathPtr
			if !pathIsMutable {
				result = tmpPath
				pathIsMutable = true
			}
			pathPtr.TransformTo(prePathMatrix, result)
			pathPtr = result
		} else {
			matrixStorage = *d.ctm
			matrixStorage.PreConcat(prePathMatrix)
			matrix = &matrixStorage
		}
	}

	paint := origPaint
	if newPaint := modifyPaintForHairlines(origPaint, matrix); newPaint != nil {
		paint = newPaint
	}

	if paint.PathEffect != nil || paint.Style != StyleFill {
		var cullRect *geom.Rect
		var cullStorage geom.Rect
		if d.computeConservativeLocalClipBounds(&cullStorage) {
			cullRect = &cullStorage
		}
		spec := paint.strokeSpec()
		doFill = stroke.FillPathWithPaint(pathPtr, &spec, tmpPath, cullRect, d.ctm)
		pathPtr = tmpPath
	}

	// avoid possibly allocating a new path in transform if we can
	devPath := tmpPath
	if pathIsMutable {
		devPath = pathPtr
	}

	// transform the path into device space
	pathPtr.TransformTo(matrix, devPath)
	if !devPath.IsFinite() {
		return
	}

	d.drawDevPath(devPath, paint, doFill)
}

// drawDevPath scan-converts a device-space path through the chosen blitter.
func (d *draw) drawDevPath(devPath *path.Path, paint *Paint, doFill bool) {
	if tooBigForMath(devPath.Bounds()) {
		return
	}

	blitter := chooseBlitter(d.dst, paint, d.ctm)
	defer recycleBlitter(blitter)

	if paint.MaskFilter != nil &&
		maskfilter.FilterPath(paint.MaskFilter, devPath, d.ctm, d.rc, blitter, doFill) {
		return // filterPath() called the blitter, so we're done
	}

	if doFill {
		// Large shaded path fills band like the rect lanes: a gradient/image span evaluates per pixel, so a tall shaded
		// path (rrects and ovals lower to paths too) is worth splitting into row bands filled concurrently through
		// per-band blitters. Unlike the rect banding this is not byte-exact — AA seam pixels can differ by a coverage
		// step from the serial fill (see fillPathShadedParallel) — so it is gated to the lanes whose verification
		// already carries AA tolerances: a per-span-scratch shader blitter, a simple-rect clip (each band is then
		// exactly the region fill the serial lane lowers to), and a regular (non-inverse) fill. IRectFillBandBounds
		// returns nil when the clipped path spans too few rows to pay for the goroutine/blitter-rebuild overhead. The
		// drawAtlas per-sprite fills opt out (forceRasterPipeline): per-sprite quads are too small to amortize the band
		// dispatch, and the banding's per-draw scaffolding would break the atlas lane's zero-allocs-per-sprite
		// discipline.
		if isBandableShaderBlitter(blitter) && d.rc.IsRect() && !devPath.IsInverseFillType() &&
			!paint.forceRasterPipeline {
			clip := d.rc.Bounds()
			ir := devPath.Bounds().RoundOut()
			top := max(ir.Top, clip.Top)
			bottom := min(ir.Bottom, clip.Bottom)
			if bounds := raster.IRectFillBandBounds(top, bottom, 0); bounds != nil {
				fillPathShadedParallel(d.dst, devPath, paint, d.ctm, paint.AntiAlias, clip, bounds)
				return
			}
		}
		if paint.AntiAlias {
			raster.AntiFillPathRasterClip(devPath, d.rc, blitter)
		} else {
			raster.FillPathRasterClip(devPath, d.rc, blitter)
		}
		return
	}
	// hairline
	if paint.AntiAlias {
		switch paint.Cap {
		case CapButt:
			raster.AntiHairPath(devPath, d.rc, blitter)
		case CapSquare:
			raster.AntiHairSquarePath(devPath, d.rc, blitter)
		default:
			raster.AntiHairRoundPath(devPath, d.rc, blitter)
		}
		return
	}
	switch paint.Cap {
	case CapButt:
		raster.HairPath(devPath, d.rc, blitter)
	case CapSquare:
		raster.HairSquarePath(devPath, d.rc, blitter)
	default:
		raster.HairRoundPath(devPath, d.rc, blitter)
	}
}

///////////////////////////////////////////////////////////////////////////////
// sprites (layer restore composition)

// clipHandlesSprite reports whether the sprite fast path can be used: it needs a BW clip or an AA clip that fully
// contains the sprite (AA-clip blitters merge coverage runs, which sprite blitters cannot consume).
func clipHandlesSprite(clip *raster.Clip, x, y int32, pmap *raster.Pixmap) bool {
	return clip.IsBW() || clip.QuickContains(geom.IRectXYWH(x, y, pmap.Width, pmap.Height))
}

// drawSprite blits src positioned at the integer offset (x, y) using the paint's alpha and blend mode. When a color
// filter or a partial-AA clip prevents the sprite fast path, the source is wrapped in an integer-translated image
// shader and drawn through drawRect.
func (d *draw) drawSprite(src *raster.Pixmap, x, y int32, origPaint *Paint) {
	if d.rc.IsEmpty() || src.Width == 0 || src.Height == 0 {
		return
	}
	bounds := geom.IRectXYWH(x, y, src.Width, src.Height)
	if d.rc.QuickReject(bounds) {
		return
	}

	paint := *origPaint
	paint.Style = StyleFill

	if paint.ColorFilter == nil && clipHandlesSprite(d.rc, x, y, src) {
		// Layer sprites are never opaque (their alpha type is premul).
		blitter := raster.ChooseSprite(d.dst, src, x, y, paint.Color.A(), paint.BlendMode, false)
		raster.FillIRectRasterClip(bounds, d.rc, blitter)
		return
	}

	// The shader fallback: an integer-translated image shader over the sprite bounds, drawn with an identity CTM
	// through drawRect. Color filters and partial-AA clips reach this lane; the legacy image lane still picks up the
	// plain src-over cases. The wrapping image/shader are homed on the pooled scratch, consumed synchronously by
	// drawRect before recycle.
	scratch := acquireImageDrawScratch()
	defer releaseImageDrawScratch(scratch)
	matrix := &scratch.matrix
	matrix.SetTranslate(float32(x), float32(y))
	px := imagecore.PixelsFromPixmap(*src)
	paintWithShader := scratch.makePaintWithImage(&paint, px, shaders.SamplingOptions{}, matrix)
	identity := geom.IdentityMatrix()
	dr := draw{dst: d.dst, ctm: &identity, rc: d.rc}
	dr.drawRect(bounds.ToRect(), &paintWithShader)
}

///////////////////////////////////////////////////////////////////////////////
// points

// ptProcRec holds the state shared by the point-drawing fast paths.
type ptProcRec struct {
	paint   *Paint
	clip    *raster.Region
	rc      *raster.Clip
	wrapper raster.AAClipBlitterWrapper
	// computed values
	clipBounds geom.Rect
	radius     float32
	mode       PointMode
}

type ptProc func(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter)

// init reports whether the point fast paths apply, filling in the computed state when they do.
func (rec *ptProcRec) init(mode PointMode, paint *Paint, matrix *geom.Matrix, rc *raster.Clip) bool {
	if mode > PointModePolygon {
		return false
	}
	if paint.PathEffect != nil || paint.MaskFilter != nil {
		return false
	}
	width := paint.StrokeWidth
	radius := float32(-1) // sentinel value, a "valid" value must be > 0

	if width == 0 {
		radius = 0.5
	} else if paint.Cap != CapRound && matrix.IsScaleTranslate() && mode == PointModePoints {
		sx := matrix.Get(geom.MScaleX)
		sy := matrix.Get(geom.MScaleY)
		if geom.ScalarNearlyZero(sx - sy) {
			radius = width * geom.ScalarAbs(sx) / 2
		}
	}
	if radius > 0 {
		clipBounds := rc.Bounds().ToRect()
		// if we return true, the caller may assume that the constructed shapes can be represented in 16.16 fixed-point
		// (after clipping), so we preflight that here.
		if !fitsInFixed(clipBounds) {
			return false
		}
		rec.mode = mode
		rec.paint = paint
		rec.clip = nil
		rec.rc = rc
		rec.clipBounds = clipBounds
		rec.radius = radius
		return true
	}
	return false
}

// chooseProc picks the point proc for the paint/clip configuration; it may substitute an AA-clip-wrapped blitter.
// blitter is an in/out parameter — the AA-clip lane replaces the caller's blitter — so it must stay a pointer.
func (rec *ptProcRec) chooseProc(blitter *raster.Blitter) ptProc { //nolint:gocritic // see above
	if rec.rc.IsBW() {
		rec.clip = rec.rc.BWRgn()
	} else {
		rec.wrapper.Init(rec.rc, *blitter)
		rec.clip = rec.wrapper.Rgn()
		*blitter = rec.wrapper.Blitter()
	}

	if rec.paint.AntiAlias {
		if rec.paint.StrokeWidth == 0 {
			switch rec.mode {
			case PointModePoints:
				return aaSquareProc
			case PointModeLines:
				return aaLineHairProc
			default:
				return aaPolyHairProc
			}
		}
		if rec.paint.Cap != CapRound {
			return aaSquareProc
		}
		return nil
	}
	// BW
	if rec.radius <= 0.5 { // small radii and hairline
		switch rec.mode {
		case PointModePoints:
			return bwPtHairProc
		case PointModeLines:
			return bwLineHairProc
		default:
			return bwPolyHairProc
		}
	}
	return bwSquareProc
}

func bwPtHairProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	// Always blit through BlitH; a specialized direct-write fast path would produce identical pixels.
	for _, p := range devPts {
		x := geom.FloorToInt(p.X)
		y := geom.FloorToInt(p.Y)
		if rec.clip.ContainsPoint(x, y) {
			blitter.BlitH(x, y, 1)
		}
	}
}

func bwLineHairProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	for i := 0; i+1 < len(devPts); i += 2 {
		raster.HairLine(devPts[i:i+2], rec.rc, blitter)
	}
}

func bwPolyHairProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	raster.HairLine(devPts, rec.rc, blitter)
}

func aaLineHairProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	for i := 0; i+1 < len(devPts); i += 2 {
		raster.AntiHairLine(devPts[i:i+2], rec.rc, blitter)
	}
}

func aaPolyHairProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	raster.AntiHairLine(devPts, rec.rc, blitter)
}

// makeSquareRad returns the axis-aligned square of half-width radius centered at center.
func makeSquareRad(center geom.Point, radius float32) geom.Rect {
	return geom.RectLTRB(center.X-radius, center.Y-radius, center.X+radius, center.Y+radius)
}

func bwSquareProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	for _, p := range devPts {
		r := makeSquareRad(p, rec.radius)
		if r.Intersect(rec.clipBounds) {
			raster.FillXRect(raster.XRectFromRect(r), rec.rc, blitter)
		}
	}
}

func aaSquareProc(rec *ptProcRec, devPts []geom.Point, blitter raster.Blitter) {
	for _, p := range devPts {
		r := makeSquareRad(p, rec.radius)
		if r.Intersect(rec.clipBounds) {
			raster.AntiFillXRect(raster.XRectFromRect(r), rec.rc, blitter)
		}
	}
}

// maxDevPts is the per-batch device-point buffer size; must be even for lines/polygon to work.
const maxDevPts = 32

// drawPoints maps the points to device space in batches and dispatches to the fast point procs, falling back to
// drawDevicePoints when they do not apply.
func (d *draw) drawPoints(mode PointMode, pts []geom.Point, paint *Paint) {
	// if we're in lines mode, force count to be even
	if mode == PointModeLines {
		pts = pts[:len(pts)&^1]
	}

	// nothing to draw
	if len(pts) == 0 || d.rc.IsEmpty() {
		return
	}

	var rec ptProcRec
	if rec.init(mode, paint, d.ctm, d.rc) {
		// Can't easily get bounds of points so don't try.
		blitter := chooseBlitter(d.dst, paint, d.ctm)
		// Recycle the pooled blitter chooseBlitter built (not any wrapper chooseProc swaps in below): capture it before
		// chooseProc replaces *blitter, and recycle on return once the point loop is done consuming it.
		defer recycleBlitter(blitter)
		proc := rec.chooseProc(&blitter)
		if proc == nil {
			return
		}
		var devPts [maxDevPts]geom.Point
		// we have to back up subsequent passes if we're in polygon mode
		backup := 0
		if mode == PointModePolygon {
			backup = 1
		}

		count := len(pts)
		off := 0
		for count != 0 {
			n := count
			if n > maxDevPts {
				n = maxDevPts
			}
			d.ctm.MapPoints(devPts[:n], pts[off:off+n])
			for _, p := range devPts[:n] {
				if !float32Finite(p.X) || !float32Finite(p.Y) {
					return
				}
			}
			proc(&rec, devPts[:n], blitter)
			off += n - backup
			count -= n
			if count > 0 {
				count += backup
			}
		}
		return
	}
	d.drawDevicePoints(mode, pts, paint)
}

func float32Finite(x float32) bool {
	return math.Float32bits(x)&0x7F800000 != 0x7F800000
}

// drawDevicePoints draws points without the fast procs: round caps via circle paths, square caps via rects,
// lines/polygons via per-segment stroking, including the dashed-line asPoints acceleration for the lines mode.
func (d *draw) drawDevicePoints(mode PointMode, pts []geom.Point, paint *Paint) {
	// if we're in lines mode, force count to be even
	if mode == PointModeLines {
		pts = pts[:len(pts)&^1]
	}

	// nothing to draw
	if len(pts) == 0 || d.rc.IsEmpty() {
		return
	}

	for _, p := range pts {
		if !float32Finite(p.X) || !float32Finite(p.Y) {
			return
		}
	}

	switch mode {
	case PointModePoints:
		// temporarily mark the paint as filling.
		newPaint := *paint
		newPaint.Style = StyleFill

		width := newPaint.StrokeWidth
		radius := width / 2

		if newPaint.Cap == CapRound {
			circle := path.Borrow()
			circle.AddCircle(0, 0, radius, geom.DirectionCW)
			var preMatrix geom.Matrix
			for i := range pts {
				preMatrix.SetTranslate(pts[i].X, pts[i].Y)
				// pass true for the last point, since we can modify the path then
				isLast := i == len(pts)-1
				d.drawPath(circle, &newPaint, &preMatrix, isLast)
			}
			path.Recycle(circle)
		} else {
			for _, pt := range pts {
				r := geom.RectLTRB(pt.X-radius, pt.Y-radius, pt.X-radius+width, pt.Y-radius+width)
				d.drawRect(r, &newPaint)
			}
		}
	default: // lines and polygon
		if mode == PointModeLines && len(pts) == 2 && paint.PathEffect != nil &&
			d.drawDashedLineAsPoints(pts, paint) {
			// 'asPoints' managed to find some fast path
			return
		}
		// couldn't take fast path
		count := len(pts) - 1
		p := *paint
		p.Style = StyleStroke
		inc := 1
		if mode == PointModeLines {
			inc = 2
		}

		// One pooled path reused across segments (drawPath transforms it in place each iteration, so it is rewound at
		// the top of the next), instead of a fresh allocation per segment.
		line := path.Borrow()
		for i := 0; i < count; i += inc {
			line.Rewind()
			line.MoveToPt(pts[i])
			line.LineToPt(pts[i+1])
			d.drawPath(line, &p, nil, true)
		}
		path.Recycle(line)
	}
}

// drawDashedLineAsPoints handles a two-point line with a path effect — most likely a dashed line — by asking the effect
// for a point representation it can draw directly. Returns false when the effect found no fast path (the caller falls
// through to per-segment stroking).
func (d *draw) drawDashedLineAsPoints(pts []geom.Point, paint *Paint) bool {
	spec := paint.strokeSpec()
	strokeRec := stroke.NewStrokeRecFromPaint(&spec, 1)
	var pointData stroke.PointData

	line := &path.Path{}
	line.MoveToPt(pts[0])
	line.LineToPt(pts[1])

	cullRect := d.rc.Bounds().ToRect()

	if !paint.PathEffect.AsPoints(&pointData, line, &strokeRec, d.ctm, &cullRect) {
		return false
	}

	newP := *paint
	newP.PathEffect = nil
	newP.Style = StyleFill

	if !pointData.First.IsEmpty() {
		d.drawPath(&pointData.First, &newP, nil, true)
	}

	if !pointData.Last.IsEmpty() {
		d.drawPath(&pointData.Last, &newP, nil, true)
	}

	if pointData.Size.X == pointData.Size.Y {
		// The rest of the dashed line can just be drawn as points
		if pointData.Flags&stroke.CirclesPointFlag != 0 {
			newP.Cap = CapRound
		} else {
			newP.Cap = CapButt
		}
		d.drawDevicePoints(PointModePoints, pointData.Points, &newP)
	} else {
		// The rest of the dashed line must be drawn as rects
		for _, pt := range pointData.Points {
			r := geom.RectLTRB(pt.X-pointData.Size.X, pt.Y-pointData.Size.Y,
				pt.X+pointData.Size.X, pt.Y+pointData.Size.Y)
			d.drawRect(r, &newP)
		}
	}
	return true
}
