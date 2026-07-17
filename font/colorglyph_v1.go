// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The COLRv1 paint-graph interpreter: walks typesetting's parsed tables.PaintTable tree onto a minimal canvas built
// from the raster package's clip stack, blitters, and blend modes.
//
// Coordinate model. The paint graph carries a synthetic root transform (the face's scale plus matrix) wrapping the
// graph, and everything inside — glyph outlines loaded unscaled at upem, gradient geometry, transform centers, clip
// boxes — is in font units, converted to a y-down convention by negating y throughout. The walker works in this same
// "layer" space (font-unit x, negated font-unit y); the root transform is the scaler's single matrix pre-scaled by
// 1/upem (exactly mapDesignPoint as a matrix), and the mask-local translation (-left, -top, plus the sub-pixel phase)
// seeds the CTM.
//
// Deliberate limitations, all inherent to the interpreter's reachable surface:
//   - Variable (PaintVar*/VarColorLine/ClipBoxFormat2) values use their base values with no variation deltas applied:
//     no variation-axis surface is exposed, and at the default instance all deltas are zero, so base values are exact
//     for every reachable face.
//   - The radial negative-radius resolution lanes (truncateToStopInterpolating and the circle projection) are
//     unreachable and not implemented: COLR radii are UFWORD (unsigned) and negative radii can only arise from
//     variation deltas, which are never applied here.
//   - Gradient stop colors quantize to 8-bit at shader construction (the gradient shaders take byte colors); the
//     interpolation itself then runs in float either way.

package font

import (
	"math"
	"sort"

	"github.com/go-text/typesetting/font/opentype"
	"github.com/go-text/typesetting/font/opentype/tables"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// colrV1MaxDepth bounds paint-graph nesting. The OT spec (5.7.11.1.9) requires implementations to guard against
// unbounded recursion; HarfBuzz, for comparison, caps nesting at HB_MAX_NESTING_LEVEL (64). typesetting materializes
// the graph as a tree whose only re-entry is PaintColrGlyph (guarded by the visited set below), so the depth cap
// defends against pathologically deep — not cyclic — graphs.
const colrV1MaxDepth = 64

// colrF214 converts a 2.14 fixed-point table value (F2Dot14) to float.
func colrF214(v tables.Fixed214) float32 { return float32(v) / (1 << 14) }

// colrPt maps a font-unit point into the walker's working space: y is negated to get the y-down convention.
func colrPt(x, y float32) geom.Point { return geom.Pt(x, -y) }

// colrV1RootTransform is the synthetic root node wrapping the paint graph: font-unit (y-down) layer space → device
// space. It is mapDesignPoint as a matrix: the single matrix pre-scaled by 1/upem.
func (c *ScalerContext) colrV1RootTransform() geom.Matrix {
	m := c.single
	upem := float32(c.typeface.upem)
	m.PreScale(1/upem, 1/upem)
	return m
}

///////////////////////////////////////////////////////////////////////////////
// The minimal canvas

// colrV1Paint is the result of configuring a fill-format paint node: a solid color, or a gradient shader (with an
// opaque black paint color so paint alpha does not modulate it).
type colrV1Paint struct {
	shader shaders.Shader // nil for a solid color
	color  colorcore.Color
}

// colrV1SaveRec is one save-stack entry: the CTM to restore, and whether a compositing layer pops with it.
type colrV1SaveRec struct {
	ctm     geom.Matrix
	isLayer bool
}

// colrV1Canvas is the minimal canvas the walker needs: a CTM, an AA clip stack, and saveLayer compositing onto the
// glyph's ARGB32 mask through the raster package's blitters and blend modes.
type colrV1Canvas struct {
	devs  []*raster.Pixmap   // device stack: devs[0] is the glyph mask, one entry per active layer above
	modes []raster.BlendMode // restore blend mode per active layer (parallel to devs[1:])
	clips *raster.ClipStack
	saves []colrV1SaveRec
	ctm   geom.Matrix
	w, h  int32
}

func newColrV1Canvas(mask *raster.Pixmap, w, h int32, ctm geom.Matrix) *colrV1Canvas {
	return &colrV1Canvas{
		devs:  []*raster.Pixmap{mask},
		clips: raster.NewRasterClipStack(w, h),
		ctm:   ctm,
		w:     w,
		h:     h,
	}
}

// device is the pixmap draws currently target (the top of the device stack).
func (cv *colrV1Canvas) device() *raster.Pixmap { return cv.devs[len(cv.devs)-1] }

// saveCount returns the current save depth (relative form; only used with restoreToCount).
func (cv *colrV1Canvas) saveCount() int { return len(cv.saves) }

// save pushes the current CTM and clip state onto the save stack.
func (cv *colrV1Canvas) save() {
	cv.saves = append(cv.saves, colrV1SaveRec{ctm: cv.ctm})
	cv.clips.Save()
}

// saveLayer pushes a fresh transparent layer the size of the mask whose pixels composite onto the device below with
// mode at restore (the only paint field the walker uses).
func (cv *colrV1Canvas) saveLayer(mode raster.BlendMode) {
	cv.saves = append(cv.saves, colrV1SaveRec{ctm: cv.ctm, isLayer: true})
	cv.clips.Save()
	cv.devs = append(cv.devs, raster.NewPixmap(cv.w, cv.h))
	cv.modes = append(cv.modes, mode)
}

// restore pops the clip and CTM, and composites a popped layer onto the device below through the clip in effect at its
// saveLayer (draws inside the layer restore their clips first, so the current clip after the pop is exactly that clip).
func (cv *colrV1Canvas) restore() {
	if len(cv.saves) == 0 {
		return
	}
	rec := cv.saves[len(cv.saves)-1]
	cv.saves = cv.saves[:len(cv.saves)-1]
	cv.clips.Restore()
	if rec.isLayer {
		layer := cv.devs[len(cv.devs)-1]
		mode := cv.modes[len(cv.modes)-1]
		cv.devs = cv.devs[:len(cv.devs)-1]
		cv.modes = cv.modes[:len(cv.modes)-1]
		// The layer draws back as a full-coverage image over the clip: every clipped pixel blends, which matters for
		// modes that act where the source is transparent (Clear, SrcIn, DstIn, ...).
		blitter := raster.NewImageSpriteBlitter(cv.device(), layer, 0, 0, 0xFF, mode)
		raster.FillIRectRasterClip(geom.IRectWH(cv.w, cv.h), cv.clips.RC(), blitter)
	}
	cv.ctm = rec.ctm
}

// restoreToCount pops saves until the save stack depth reaches count.
func (cv *colrV1Canvas) restoreToCount(count int) {
	for len(cv.saves) > count {
		cv.restore()
	}
}

// concat pre-concatenates m onto the current CTM.
func (cv *colrV1Canvas) concat(m *geom.Matrix) { cv.ctm.PreConcat(m) }

// clipPath intersects the current clip with an anti-aliased fill of a layer-space path.
func (cv *colrV1Canvas) clipPath(p *path.Path) {
	cv.clips.ClipPath(&cv.ctm, p, raster.ClipIntersect, true)
}

// blitterFor builds the blitter a draw with paint targets: the shader pipeline compiled against the CTM (the gradient
// geometry is in layer space), or the solid src-over blitter.
func (cv *colrV1Canvas) blitterFor(paint *colrV1Paint) raster.Blitter {
	if paint.shader != nil {
		pipeline := shaders.Compile(paint.shader, cv.ctm, colorcore.RGB(0, 0, 0))
		if pipeline == nil {
			return nil
		}
		return raster.NewShaderBlitter(cv.device(), pipeline, raster.BlendSrcOver)
	}
	return raster.NewSolidBlitter(cv.device(), paint.color)
}

// drawPath fills a layer-space path with paint, anti-aliased.
func (cv *colrV1Canvas) drawPath(p *path.Path, paint *colrV1Paint) {
	dev := path.New()
	p.TransformTo(&cv.ctm, dev)
	if !dev.IsFinite() {
		return
	}
	blitter := cv.blitterFor(paint)
	if blitter == nil {
		return
	}
	raster.AntiFillPathRasterClip(dev, cv.clips.RC(), blitter)
}

// drawPaint fills the current clip with paint.
func (cv *colrV1Canvas) drawPaint(paint *colrV1Paint) {
	blitter := cv.blitterFor(paint)
	if blitter == nil {
		return
	}
	raster.FillIRectRasterClip(geom.IRectWH(cv.w, cv.h), cv.clips.RC(), blitter)
}

///////////////////////////////////////////////////////////////////////////////
// The walker

// colrV1Walker traverses one glyph's paint DAG, in draw mode (canvas != nil) or bounds mode (bounds != nil,
// accumulating PaintGlyph outline boxes under ctm).
type colrV1Walker struct {
	c       *ScalerContext
	t       *Typeface
	colr    *tables.COLR1
	canvas  *colrV1Canvas   // draw mode
	bounds  *geom.Rect      // bounds mode
	visited map[uint16]bool // base glyph IDs on the active PaintColrGlyph chain (5.7.11.1.9 cycle guard)
	depth   int
	ctm     geom.Matrix // bounds-mode CTM (draw mode uses canvas.ctm)
}

// renderCOLRv1 interprets the paint graph onto the glyph's (zeroed) ARGB32 mask. The CTM seed is the mask-local
// translation including the sub-pixel phase (translate(-left, -top) then translate(subX, subY)); startGlyph then
// applies the root transform and the optional ClipList clip.
func (c *ScalerContext) renderCOLRv1(g *Glyph) {
	colr := c.typeface.colrTable()
	if colr == nil {
		return
	}
	pm := g.Pixmap()
	w := &colrV1Walker{
		c:       c,
		t:       c.typeface,
		colr:    colr,
		canvas:  newColrV1Canvas(&pm, g.Width, g.Height, c.maskLocalTranslate(g)),
		visited: make(map[uint16]bool),
	}
	w.startGlyph(g.packedID.GlyphID(), true)
}

// colrV1Bounds returns the ClipList box when present (the font-unit corners map through mapDesignPoint), else the
// result of a measuring traversal of the paint graph. An empty rect (including a traversal failure — a cycle fails the
// bounds pass too) zaps the glyph.
func (c *ScalerContext) colrV1Bounds(gid uint16) geom.Rect {
	var bounds geom.Rect
	colr := c.typeface.colrTable()
	if colr == nil {
		return bounds
	}
	if xMin, yMin, xMax, yMax, ok := colrClipBox(colr, gid); ok {
		pts := [4]geom.Point{
			c.mapDesignPoint(xMin, yMin),
			c.mapDesignPoint(xMin, yMax),
			c.mapDesignPoint(xMax, yMax),
			c.mapDesignPoint(xMax, yMin),
		}
		return geom.RectLTRB(
			min(pts[0].X, pts[1].X, pts[2].X, pts[3].X),
			min(pts[0].Y, pts[1].Y, pts[2].Y, pts[3].Y),
			max(pts[0].X, pts[1].X, pts[2].X, pts[3].X),
			max(pts[0].Y, pts[1].Y, pts[2].Y, pts[3].Y),
		)
	}
	w := &colrV1Walker{c: c, t: c.typeface, colr: colr, bounds: &bounds, visited: make(map[uint16]bool)}
	w.ctm.SetIdentity()
	if !w.startGlyph(gid, true) {
		return geom.Rect{}
	}
	return bounds
}

// colrClipBox returns gid's ClipList box in font units. ClipBoxFormat2's variation deltas are not applied (base values;
// see the file comment).
func colrClipBox(colr *tables.COLR1, gid uint16) (xMin, yMin, xMax, yMax float32, ok bool) {
	box, ok := colr.ClipList.Search(gid)
	if !ok {
		return 0, 0, 0, 0, false
	}
	switch b := box.(type) {
	case tables.ClipBoxFormat1:
		return float32(b.XMin), float32(b.YMin), float32(b.XMax), float32(b.YMax), true
	case tables.ClipBoxFormat2:
		return float32(b.XMin), float32(b.YMin), float32(b.XMax), float32(b.YMax), true
	default:
		return 0, 0, 0, 0, false
	}
}

// startGlyph looks up gid's v1 paint, applies the root transform for the top-level glyph, clips to the ClipList box
// (draw mode only — bounds mode ignores it, the metrics lane having already preferred it), and traverses. The visited
// set spans the whole active chain: a PaintColrGlyph re-entering a base glyph already on the chain is the spec's cycle,
// which skips the subtree when drawing and fails the bounds pass.
func (w *colrV1Walker) startGlyph(gid uint16, includeRootTransform bool) bool {
	if w.visited[gid] {
		return w.canvas != nil
	}
	w.visited[gid] = true
	defer delete(w.visited, gid)

	paint, ok := w.t.faceColorPaint(opentype.GID(gid))
	if !ok {
		return false
	}
	if _, isV0 := paint.(tables.PaintColrLayersResolved); isV0 {
		return false
	}
	if includeRootTransform {
		root := w.c.colrV1RootTransform()
		if w.canvas != nil {
			w.canvas.concat(&root)
		} else {
			w.ctm.PreConcat(&root)
		}
	}
	if w.canvas != nil {
		if xMin, yMin, xMax, yMax, hasBox := colrClipBox(w.colr, gid); hasBox {
			// The clip box's polygon (bottom-left, top-left, top-right, bottom-right, negated y), clipped with AA, in
			// font-unit space under CTM·root.
			box := path.New()
			box.MoveToPt(colrPt(xMin, yMin))
			box.LineToPt(colrPt(xMin, yMax))
			box.LineToPt(colrPt(xMax, yMax))
			box.LineToPt(colrPt(xMax, yMin))
			box.Close()
			w.canvas.clipPath(box)
		}
	}
	return w.traverse(paint)
}

// traverse applies the depth guard, the per-node save/restore (canvas state or bounds-mode CTM), and dispatches to the
// draw- or bounds-mode handler for paint.
func (w *colrV1Walker) traverse(paint tables.PaintTable) bool {
	if w.depth >= colrV1MaxDepth {
		return false
	}
	w.depth++
	defer func() { w.depth-- }()

	if w.canvas != nil {
		count := w.canvas.saveCount()
		w.canvas.save()
		defer w.canvas.restoreToCount(count)
		return w.traverseDraw(paint)
	}
	saved := w.ctm
	defer func() { w.ctm = saved }()
	return w.traverseBounds(paint)
}

// traverseDraw dispatches paint to its draw-mode handler.
func (w *colrV1Walker) traverseDraw(paint tables.PaintTable) bool {
	switch p := paint.(type) {
	case tables.PaintColrLayers:
		children, err := w.colr.LayerList.Resolve(p)
		if err != nil {
			return false
		}
		for _, child := range children {
			if !w.traverse(child) {
				return false
			}
		}
		return true
	case tables.PaintGlyph:
		outline := w.glyphLayerPath(p.GlyphID)
		if outline == nil {
			return false
		}
		if colrIsFillPaint(p.Paint) {
			// The paint-graph leaf fast path: drawPath with the fill paint directly instead of clipPath + drawPaint.
			fill, ok := w.configurePaint(p.Paint)
			if !ok {
				return false
			}
			w.canvas.drawPath(outline, &fill)
			return true
		}
		w.canvas.clipPath(outline)
		return w.traverse(p.Paint)
	case tables.PaintColrGlyph:
		return w.startGlyph(p.GlyphID, false)
	case tables.PaintComposite:
		// COMPOSITE: an isolated group (saveLayer) for the backdrop, then a second layer whose restore blends the
		// source onto the backdrop with the composite mode. Both layers (and this node's save) pop through the caller's
		// deferred restoreToCount, in reverse (LIFO) order.
		w.canvas.saveLayer(raster.BlendSrcOver)
		if !w.traverse(p.BackdropPaint) {
			return false
		}
		w.canvas.saveLayer(colrBlendMode(p.CompositeMode))
		return w.traverse(p.SourcePaint)
	case tables.PaintSolid, tables.PaintVarSolid,
		tables.PaintLinearGradient, tables.PaintVarLinearGradient,
		tables.PaintRadialGradient, tables.PaintVarRadialGradient,
		tables.PaintSweepGradient, tables.PaintVarSweepGradient:
		// The fill lane: fill the current clip.
		fill, ok := w.configurePaint(paint)
		if !ok {
			return false
		}
		w.canvas.drawPaint(&fill)
		return true
	default:
		if m, child, ok := colrTransform(paint); ok {
			w.canvas.concat(&m)
			return w.traverse(child)
		}
		return false
	}
}

// traverseBounds dispatches paint to its bounds-mode handler, accumulating PaintGlyph outline bounds under ctm.
func (w *colrV1Walker) traverseBounds(paint tables.PaintTable) bool {
	switch p := paint.(type) {
	case tables.PaintColrLayers:
		children, err := w.colr.LayerList.Resolve(p)
		if err != nil {
			return false
		}
		for _, child := range children {
			if !w.traverse(child) {
				return false
			}
		}
		return true
	case tables.PaintGlyph:
		outline := w.glyphLayerPath(p.GlyphID)
		if outline == nil {
			return false
		}
		dev := path.New()
		outline.TransformTo(&w.ctm, dev)
		if b := dev.Bounds(); !b.IsEmpty() {
			w.bounds.Join(b)
		}
		return true
	case tables.PaintColrGlyph:
		return w.startGlyph(p.GlyphID, false)
	case tables.PaintComposite:
		return w.traverse(p.BackdropPaint) && w.traverse(p.SourcePaint)
	case tables.PaintSolid, tables.PaintVarSolid,
		tables.PaintLinearGradient, tables.PaintVarLinearGradient,
		tables.PaintRadialGradient, tables.PaintVarRadialGradient,
		tables.PaintSweepGradient, tables.PaintVarSweepGradient:
		return true
	default:
		if m, child, ok := colrTransform(paint); ok {
			w.ctm.PreConcat(&m)
			return w.traverse(child)
		}
		return false
	}
}

// glyphLayerPath returns gid's outline in layer space (font units, y negated), loading the glyph unscaled and
// untransformed at upem ppem. The raw-outline accessor bypasses the COLR-preferring lookup, since PaintGlyph outlines
// frequently reuse the base glyph's own gid.
func (w *colrV1Walker) glyphLayerPath(gid uint16) *path.Path {
	outline, ok := w.t.faceGlyphOutline(opentype.GID(gid))
	if !ok {
		return nil
	}
	return outlineToPath(outline, colrPt)
}

// colrIsFillPaint reports whether p is one of the four fill formats eligible for the drawPath fast path.
func colrIsFillPaint(p tables.PaintTable) bool {
	switch p.(type) {
	case tables.PaintSolid, tables.PaintVarSolid,
		tables.PaintLinearGradient, tables.PaintVarLinearGradient,
		tables.PaintRadialGradient, tables.PaintVarRadialGradient,
		tables.PaintSweepGradient, tables.PaintVarSweepGradient:
		return true
	default:
		return false
	}
}

///////////////////////////////////////////////////////////////////////////////
// Transforms

// colrTransform converts a transform-format paint into its layer-space matrix and child paint. Every conversion applies
// the same y-down conjugation: negated off-diagonal terms and y translations, negated rotation angles, negated y-skew,
// negated center y. Each table format (including the AroundCenter/Uniform variants) maps directly to the same matrix
// construction.
func colrTransform(paint tables.PaintTable) (m geom.Matrix, child tables.PaintTable, ok bool) {
	switch p := paint.(type) {
	case tables.PaintTransform:
		return colrAffine(p.Transform), p.Paint, true
	case tables.PaintVarTransform:
		return colrAffine(tables.Affine2x3{
			Xx: p.Transform.Xx, Yx: p.Transform.Yx, Xy: p.Transform.Xy,
			Yy: p.Transform.Yy, Dx: p.Transform.Dx, Dy: p.Transform.Dy,
		}), p.Paint, true
	case tables.PaintTranslate:
		m.SetTranslate(float32(p.Dx), -float32(p.Dy))
		return m, p.Paint, true
	case tables.PaintVarTranslate:
		m.SetTranslate(float32(p.Dx), -float32(p.Dy))
		return m, p.Paint, true
	case tables.PaintScale:
		m.SetScale(colrF214(p.ScaleX), colrF214(p.ScaleY))
		return m, p.Paint, true
	case tables.PaintVarScale:
		m.SetScale(colrF214(p.ScaleX), colrF214(p.ScaleY))
		return m, p.Paint, true
	case tables.PaintScaleAroundCenter:
		m.SetScalePivot(colrF214(p.ScaleX), colrF214(p.ScaleY), float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintVarScaleAroundCenter:
		m.SetScalePivot(colrF214(p.ScaleX), colrF214(p.ScaleY), float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintScaleUniform:
		m.SetScale(colrF214(p.Scale), colrF214(p.Scale))
		return m, p.Paint, true
	case tables.PaintVarScaleUniform:
		m.SetScale(colrF214(p.Scale), colrF214(p.Scale))
		return m, p.Paint, true
	case tables.PaintScaleUniformAroundCenter:
		m.SetScalePivot(colrF214(p.Scale), colrF214(p.Scale), float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintVarScaleUniformAroundCenter:
		m.SetScalePivot(colrF214(p.Scale), colrF214(p.Scale), float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintRotate:
		m.SetRotate(-colrF214(p.Angle) * 180)
		return m, p.Paint, true
	case tables.PaintVarRotate:
		m.SetRotate(-colrF214(p.Angle) * 180)
		return m, p.Paint, true
	case tables.PaintRotateAroundCenter:
		m.SetRotatePivot(-colrF214(p.Angle)*180, float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintVarRotateAroundCenter:
		m.SetRotatePivot(-colrF214(p.Angle)*180, float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintSkew:
		m.SetSkew(colrSkewTan(colrF214(p.XSkewAngle)), colrSkewTan(-colrF214(p.YSkewAngle)))
		return m, p.Paint, true
	case tables.PaintVarSkew:
		m.SetSkew(colrSkewTan(colrF214(p.XSkewAngle)), colrSkewTan(-colrF214(p.YSkewAngle)))
		return m, p.Paint, true
	case tables.PaintSkewAroundCenter:
		m.SetSkewPivot(colrSkewTan(colrF214(p.XSkewAngle)), colrSkewTan(-colrF214(p.YSkewAngle)),
			float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	case tables.PaintVarSkewAroundCenter:
		m.SetSkewPivot(colrSkewTan(colrF214(p.XSkewAngle)), colrSkewTan(-colrF214(p.YSkewAngle)),
			float32(p.CenterX), -float32(p.CenterY))
		return m, p.Paint, true
	default:
		return m, nil, false
	}
}

// colrAffine converts an Affine2x3 to the layer-space matrix, applying the y-down sign conjugation.
func colrAffine(a tables.Affine2x3) geom.Matrix {
	var m geom.Matrix
	m.SetAll(
		a.Xx, -a.Xy, a.Dx,
		-a.Yx, a.Yy, -a.Dy,
		0, 0, 1,
	)
	return m
}

// colrSkewTan converts a skew angle in F2Dot14 half-turns to a tangent, snapping near-zero results to zero to match the
// matrix builder's own rotation-snapping behavior.
func colrSkewTan(halfTurns float32) float32 {
	t := float32(math.Tan(float64(geom.DegreesToRadians(halfTurns * 180))))
	if geom.ScalarNearlyZero(t) {
		return 0
	}
	return t
}

// colrBlendMode maps a COLR composite mode to the raster package's blend mode (unknown values fall back to Dst, a no-op
// composite).
func colrBlendMode(mode tables.CompositeMode) raster.BlendMode {
	switch mode {
	case tables.CompositeClear:
		return raster.BlendClear
	case tables.CompositeSrc:
		return raster.BlendSrc
	case tables.CompositeDest:
		return raster.BlendDst
	case tables.CompositeSrcOver:
		return raster.BlendSrcOver
	case tables.CompositeDestOver:
		return raster.BlendDstOver
	case tables.CompositeSrcIn:
		return raster.BlendSrcIn
	case tables.CompositeDestIn:
		return raster.BlendDstIn
	case tables.CompositeSrcOut:
		return raster.BlendSrcOut
	case tables.CompositeDestOut:
		return raster.BlendDstOut
	case tables.CompositeSrcAtop:
		return raster.BlendSrcATop
	case tables.CompositeDestAtop:
		return raster.BlendDstATop
	case tables.CompositeXor:
		return raster.BlendXor
	case tables.CompositePlus:
		return raster.BlendPlus
	case tables.CompositeScreen:
		return raster.BlendScreen
	case tables.CompositeOverlay:
		return raster.BlendOverlay
	case tables.CompositeDarken:
		return raster.BlendDarken
	case tables.CompositeLighten:
		return raster.BlendLighten
	case tables.CompositeColorDodge:
		return raster.BlendColorDodge
	case tables.CompositeColorBurn:
		return raster.BlendColorBurn
	case tables.CompositeHardLight:
		return raster.BlendHardLight
	case tables.CompositeSoftLight:
		return raster.BlendSoftLight
	case tables.CompositeDifference:
		return raster.BlendDifference
	case tables.CompositeExclusion:
		return raster.BlendExclusion
	case tables.CompositeMultiply:
		return raster.BlendMultiply
	case tables.CompositeHslHue:
		return raster.BlendHue
	case tables.CompositeHslSaturation:
		return raster.BlendSaturation
	case tables.CompositeHslColor:
		return raster.BlendColor
	case tables.CompositeHslLuminosity:
		return raster.BlendLuminosity
	default:
		return raster.BlendDst
	}
}

// colrTileMode maps a color-line extend mode to the shader tile mode.
func colrTileMode(extend tables.Extend) shaders.TileMode {
	switch extend {
	case tables.ExtendRepeat:
		return shaders.TileRepeat
	case tables.ExtendReflect:
		return shaders.TileMirror
	default:
		return shaders.TileClamp
	}
}

///////////////////////////////////////////////////////////////////////////////
// Paint configuration

// colrTransparent is the transparent-color paint several degenerate gradient lanes produce; drawing it src-over is a
// no-op.
var colrTransparent = colrV1Paint{color: 0}

// fetchColorStops resolves palette/foreground colors, scales alpha, and stable-sorts by stop offset. Returns false for
// an empty color line or an out-of-range palette index (the layer is not drawn).
func (w *colrV1Walker) fetchColorStops(stops []tables.ColorStop) ([]float32, []colorcore.Color4f, bool) {
	if len(stops) == 0 {
		return nil, nil, false
	}
	type colorStop struct {
		pos   float32
		color colorcore.Color4f
	}
	sorted := make([]colorStop, len(stops))
	for i, s := range stops {
		col, ok := w.t.PaletteColor(s.PaletteIndex, w.c.rec.ForegroundColor)
		if !ok {
			return nil, nil, false
		}
		c4 := colorcore.Color4fFromColor(col)
		c4.A *= colrF214(s.Alpha)
		sorted[i] = colorStop{pos: colrF214(s.StopOffset), color: c4}
	}
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].pos < sorted[j].pos })
	pos := make([]float32, len(sorted))
	colors := make([]colorcore.Color4f, len(sorted))
	for i, s := range sorted {
		pos[i] = s.pos
		colors[i] = s.color
	}
	return pos, colors, true
}

// colrVarStops adapts a VarColorLine's stops (base values, no deltas — see the file comment).
func colrVarStops(v []tables.VarColorStop) []tables.ColorStop {
	out := make([]tables.ColorStop, len(v))
	for i, s := range v {
		out[i] = tables.ColorStop{StopOffset: s.StopOffset, PaletteIndex: s.PaletteIndex, Alpha: s.Alpha}
	}
	return out
}

// colrColors4b quantizes the working float colors to the byte colors the gradient shaders take.
func colrColors4b(colors []colorcore.Color4f) []colorcore.Color {
	out := make([]colorcore.Color, len(colors))
	for i, c := range colors {
		out[i] = c.ToColor()
	}
	return out
}

// configurePaint builds a colrV1Paint for the four fill formats (and their Var twins, on base values).
func (w *colrV1Walker) configurePaint(paint tables.PaintTable) (colrV1Paint, bool) {
	switch p := paint.(type) {
	case tables.PaintSolid:
		return w.solidPaint(p.PaletteIndex, p.Alpha)
	case tables.PaintVarSolid:
		return w.solidPaint(p.PaletteIndex, p.Alpha)
	case tables.PaintLinearGradient:
		stops, colors, ok := w.fetchColorStops(p.ColorLine.ColorStops)
		if !ok {
			return colrV1Paint{}, false
		}
		return w.linearPaint(p.X0, p.Y0, p.X1, p.Y1, p.X2, p.Y2, p.ColorLine.Extend, stops, colors)
	case tables.PaintVarLinearGradient:
		stops, colors, ok := w.fetchColorStops(colrVarStops(p.ColorLine.ColorStops))
		if !ok {
			return colrV1Paint{}, false
		}
		return w.linearPaint(p.X0, p.Y0, p.X1, p.Y1, p.X2, p.Y2, p.ColorLine.Extend, stops, colors)
	case tables.PaintRadialGradient:
		stops, colors, ok := w.fetchColorStops(p.ColorLine.ColorStops)
		if !ok {
			return colrV1Paint{}, false
		}
		return w.radialPaint(p.X0, p.Y0, p.Radius0, p.X1, p.Y1, p.Radius1, p.ColorLine.Extend, stops, colors)
	case tables.PaintVarRadialGradient:
		stops, colors, ok := w.fetchColorStops(colrVarStops(p.ColorLine.ColorStops))
		if !ok {
			return colrV1Paint{}, false
		}
		return w.radialPaint(p.X0, p.Y0, p.Radius0, p.X1, p.Y1, p.Radius1, p.ColorLine.Extend, stops, colors)
	case tables.PaintSweepGradient:
		stops, colors, ok := w.fetchColorStops(p.ColorLine.ColorStops)
		if !ok {
			return colrV1Paint{}, false
		}
		return w.sweepPaint(p.CenterX, p.CenterY, p.StartAngle, p.EndAngle, p.ColorLine.Extend, stops, colors)
	case tables.PaintVarSweepGradient:
		stops, colors, ok := w.fetchColorStops(colrVarStops(p.ColorLine.ColorStops))
		if !ok {
			return colrV1Paint{}, false
		}
		return w.sweepPaint(p.CenterX, p.CenterY, p.StartAngle, p.EndAngle, p.ColorLine.Extend, stops, colors)
	default:
		return colrV1Paint{}, false
	}
}

// solidPaint handles the solid-color paint format: the palette (or foreground) color with its alpha scaled.
func (w *colrV1Walker) solidPaint(paletteIndex uint16, alpha tables.Fixed214) (colrV1Paint, bool) {
	col, ok := w.t.PaletteColor(paletteIndex, w.c.rec.ForegroundColor)
	if !ok {
		return colrV1Paint{}, false
	}
	c4 := colorcore.Color4fFromColor(col)
	c4.A *= colrF214(alpha)
	return colrV1Paint{color: c4.ToColor()}, true
}

// linearPaint handles the linear-gradient paint format: project p1 onto the line perpendicular to p0p2 (the nanoemoji
// normal-form reduction), then rescale the gradient extent so the stops span [0, 1] (the repeat/reflect modes operate
// over that range in the shader).
func (w *colrV1Walker) linearPaint(x0, y0, x1, y1, x2, y2 int16, extend tables.Extend, stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
	if len(stops) == 1 {
		return colrV1Paint{color: colors[0].ToColor()}, true
	}
	p0 := colrPt(float32(x0), float32(y0))
	p1 := colrPt(float32(x1), float32(y1))
	p2 := colrPt(float32(x2), float32(y2))

	// Degenerate p0p1/p0p2, or parallel lines: just use the first color.
	if p1 == p0 || p2 == p0 || p1.Sub(p0).Cross(p2.Sub(p0)) == 0 {
		return colrV1Paint{color: colors[0].ToColor()}, true
	}

	// P3: the orthogonal projection of p0p1 onto a line perpendicular to p0p2 through p0.
	perp := p2.Sub(p0)
	perp = geom.Pt(perp.Y, -perp.X)
	p3 := p0.Add(colrVectorProjection(p1.Sub(p0), perp))

	tileMode := colrTileMode(extend)
	colorStopRange := stops[len(stops)-1] - stops[0]
	if colorStopRange == 0 {
		if tileMode != shaders.TileClamp {
			return colrTransparent, true
		}
		// Pad case: duplicate a fake stop at +1 so the projection below yields the equivalent first-color-before /
		// last-color-after gradient.
		stops = append(stops, stops[len(stops)-1]+1)
		colors = append(colors, colors[len(colors)-1])
		colorStopRange = 1
	}

	lineStart, lineEnd := p0, p3
	if colorStopRange != 1 || stops[0] != 0 {
		p0p3 := p3.Sub(p0)
		lineStart = p0.Add(p0p3.Scaled(stops[0]))
		lineEnd = p0.Add(p0p3.Scaled(stops[len(stops)-1]))
		scaleFactor := 1 / colorStopRange
		startOffset := stops[0]
		for i := range stops {
			stops[i] = (stops[i] - startOffset) * scaleFactor
		}
	}

	shader := shaders.NewLinearGradient(lineStart, lineEnd, colrColors4b(colors), stops, tileMode, nil)
	if shader == nil {
		return colrV1Paint{}, false
	}
	return colrV1Paint{shader: shader, color: colorcore.RGB(0, 0, 0)}, true
}

// radialPaint handles the radial-gradient paint format: rescale stops to [0, 1], interpolating new centers and radii so
// the two-point conical shader sees normalized stops. Negative-radius handling is unreachable here (radii are unsigned
// without variation deltas; see the file comment).
func (w *colrV1Walker) radialPaint(x0, y0 int16, r0 uint16, x1, y1 int16, r1 uint16, extend tables.Extend, stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
	if len(stops) == 1 {
		return colrV1Paint{color: colors[0].ToColor()}, true
	}
	start := colrPt(float32(x0), float32(y0))
	end := colrPt(float32(x1), float32(y1))
	startRadius := float32(r0)
	endRadius := float32(r1)

	tileMode := colrTileMode(extend)
	colorStopRange := stops[len(stops)-1] - stops[0]
	if colorStopRange == 0 {
		if tileMode != shaders.TileClamp {
			return colrTransparent, true
		}
		stops = append(stops, stops[len(stops)-1]+1)
		colors = append(colors, colors[len(colors)-1])
		colorStopRange = 1
	}
	if colorStopRange != 1 || stops[0] != 0 {
		startToEnd := end.Sub(start)
		radiusDiff := endRadius - startRadius
		scaleFactor := 1 / colorStopRange
		stopsStartOffset := stops[0]
		end = start.Add(startToEnd.Scaled(stops[len(stops)-1]))
		start = start.Add(startToEnd.Scaled(stopsStartOffset))
		endRadius = startRadius + radiusDiff*stops[len(stops)-1]
		startRadius += radiusDiff * stopsStartOffset
		for i := range stops {
			stops[i] = (stops[i] - stopsStartOffset) * scaleFactor
		}
	}

	shader := shaders.NewTwoPointConicalGradient(start, startRadius, end, endRadius,
		colrColors4b(colors), stops, tileMode, nil)
	if shader == nil {
		return colrV1Paint{}, false
	}
	return colrV1Paint{shader: shader, color: colorcore.RGB(0, 0, 0)}, true
}

// sweepPaint handles the sweep-gradient paint format per OpenType 1.9.1: angles are (value + 1) × 180°
// counter-clockwise from the positive x-axis; the color line is rescaled to [0, 1] with the angles adjusted inverse
// proportionally, then converted to the shader's clockwise convention (reversing when needed).
func (w *colrV1Walker) sweepPaint(centerX, centerY int16, start, end tables.Fixed214, extend tables.Extend, stops []float32, colors []colorcore.Color4f) (colrV1Paint, bool) {
	if len(stops) == 1 {
		return colrV1Paint{color: colors[0].ToColor()}, true
	}
	center := colrPt(float32(centerX), float32(centerY))
	startAngle := (colrF214(start) + 1) * 180
	endAngle := (colrF214(end) + 1) * 180

	sectorAngle := endAngle - startAngle
	tileMode := colrTileMode(extend)
	if sectorAngle == 0 && tileMode != shaders.TileClamp {
		// "If the ColorLine's extend mode is reflect or repeat and start and end angle are equal, nothing is drawn."
		return colrTransparent, true
	}

	startAngleScaled := startAngle + sectorAngle*stops[0]
	endAngleScaled := startAngle + sectorAngle*stops[len(stops)-1]

	colorStopRange := stops[len(stops)-1] - stops[0]
	if colorStopRange == 0 {
		if tileMode != shaders.TileClamp {
			return colrTransparent, true
		}
		stops = append(stops, stops[len(stops)-1]+1)
		colors = append(colors, colors[len(colors)-1])
		colorStopRange = 1
	}
	scaleFactor := 1 / colorStopRange
	startOffset := stops[0]
	for i := range stops {
		stops[i] = (stops[i] - startOffset) * scaleFactor
	}

	// Convert counter-clockwise angles/stops to the shader's clockwise convention, unless the gradient is already
	// reversed by startAngle > endAngle.
	startAngleScaled = 360 - startAngleScaled
	endAngleScaled = 360 - endAngleScaled
	if startAngleScaled >= endAngleScaled {
		startAngleScaled, endAngleScaled = endAngleScaled, startAngleScaled
		for i, j := 0, len(stops)-1; i < j; i, j = i+1, j-1 {
			stops[i], stops[j] = stops[j], stops[i]
			colors[i], colors[j] = colors[j], colors[i]
		}
		for i := range stops {
			stops[i] = 1 - stops[i]
		}
	}

	shader := shaders.NewSweepGradient(center, colrColors4b(colors), stops, tileMode,
		startAngleScaled, endAngleScaled, nil)
	if shader == nil {
		return colrV1Paint{}, false
	}
	return colrV1Paint{shader: shader, color: colorcore.RGB(0, 0, 0)}, true
}

// colrVectorProjection returns the projection of a onto b.
func colrVectorProjection(a, b geom.Point) geom.Point {
	length := b.Length()
	if length == 0 {
		return geom.Point{}
	}
	return b.Scaled(a.Dot(b) / (length * length))
}
