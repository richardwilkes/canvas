// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The measuring slice for the no-device strike: per-glyph advances and bounds and the font-wide metrics, computed from
// the sfnt tables (via go-text/typesetting) instead of a platform scaler. The rules that govern the observable
// quantization:
//
//   - makeCanonicalized decides whether the strike measures at the requested size or at the canonical path size
//     (canonicalTextSizeForPaths = 64) with a strike→source scale, dropping the paint in the canonical case.
//   - Advances are always linear (only hinting HintingNone/HintingSlight are honored): the design-space advance mapped
//     through the text matrix, the same values an unhinted linear-metrics lane would produce.
//   - Fill bounds come from the glyph extents (the outline control box) mapped through the text matrix and rounded out
//     (saturateBounds).
//   - Styled bounds (stroke and/or path effect) come from the styled outline path's bounds: for the no-device strike
//     the post 2x2 is identity, so the style applies directly in text-matrix space.
//   - Glyphs whose rounded bounds have a zero dimension are zeroed.
//
// Font-wide metrics follow a standard sfnt table recipe for scalable outline fonts; other platform text engines differ
// in documented ways.

package font

import (
	"math"

	"github.com/go-text/typesetting/font/opentype"

	tsfont "github.com/go-text/typesetting/font"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// canonicalTextSizeForPaths is the fixed strike size used when text is measured/drawn as paths.
const canonicalTextSizeForPaths = 64

// strike is the measuring state after canonicalization: a typeface at a text matrix, with the optional stroke style
// baked in.
type strike struct {
	pathEffect    stroke.PathEffect
	t             *Typeface
	size          float32
	scaleX        float32
	skewX         float32
	frameWidth    float32 // >= 0 when stroking; -1 for fill
	miterLimit    float32
	cap           stroke.Cap
	join          stroke.Join
	strokeAndFill bool
}

// makeCanonicalized returns the strike to measure with and the strike→source scale to apply to the results.
func makeCanonicalized(f *Font, paint *stroke.PaintSpec) (st strike, strikeToSourceScale float32) {
	st = strike{t: f.typeface, size: f.size, scaleX: f.scaleX, skewX: f.skewX, frameWidth: -1}
	strikeToSourceScale = 1
	if shouldDrawAsPath(paint, f) {
		// Measure at the canonical size and scale back; the paint is reset.
		strikeToSourceScale = f.size / canonicalTextSizeForPaths
		st.size = canonicalTextSizeForPaths
		return st, strikeToSourceScale
	}
	// The stroke style enters the strike only for non-fill styles with a non-negative width.
	if paint != nil && paint.Style != stroke.PaintStyleFill && paint.Width >= 0 {
		st.frameWidth = paint.Width
		st.miterLimit = paint.MiterLimit
		st.join = paint.Join
		st.cap = paint.Cap
		st.strokeAndFill = paint.Style == stroke.PaintStyleStrokeAndFill
	}
	if paint != nil {
		st.pathEffect = paint.PathEffect
	}
	return st, strikeToSourceScale
}

// shouldDrawAsPath reports whether the font/paint combination should be measured/drawn as a path rather than through
// the glyph cache, assuming the identity view matrix.
func shouldDrawAsPath(paint *stroke.PaintSpec, f *Font) bool {
	// Hairline glyphs are fast enough, so we don't need to cache them.
	if paint != nil && paint.Style == stroke.PaintStyleStroke && paint.Width == 0 {
		return true
	}
	// We have a self-imposed maximum, just to limit memory usage.
	const maxSizeSquared = 256 * 256
	// Text matrix: [size*scaleX, size*skewX; 0, size].
	scaleX := f.size * f.scaleX
	skewX := f.size * f.skewX
	return scaleX*scaleX > maxSizeSquared || skewX*skewX+f.size*f.size > maxSizeSquared
}

// nearlyZero is the tolerance used to treat a small scalar as effectively zero (1/4096).
const nearlyZero = 1.0 / 4096

// degenerate is the singular-matrix gate: when either scale of the rotation-free text matrix is nearly zero (or
// non-finite), glyph advances and bounds collapse to zero while the strike scale falls back to 1 (so font metrics come
// out at size 1).
func (st *strike) degenerate() bool {
	sx := float64(st.size) * float64(st.scaleX)
	sy := float64(st.size)
	kx := float64(st.size) * float64(st.skewX)
	finite := !math.IsNaN(sx) && !math.IsInf(sx, 0) && !math.IsNaN(sy) && !math.IsInf(sy, 0) &&
		!math.IsNaN(kx) && !math.IsInf(kx, 0)
	return !finite || math.Abs(sx) <= nearlyZero || math.Abs(sy) <= nearlyZero
}

// mapDesign maps a design-space point (font units, y up) through the text matrix into device space (y down).
func (st *strike) mapDesign(x, y float32) geom.Point {
	upem := float32(st.t.upem)
	px := x / upem
	py := -y / upem
	return geom.Pt(st.size*st.scaleX*px+st.size*st.skewX*py, st.size*py)
}

// glyphAdvance returns the x-advance of gid in strike space (the linear-metrics lane: design advance through the text
// matrix).
func (st *strike) glyphAdvance(gid uint16) float32 {
	t := st.t
	if t.face == nil || t.upem <= 0 || int(gid) >= t.nGlyphs || st.degenerate() {
		return 0
	}
	adv := t.faceHAdvance(opentype.GID(gid))
	return adv / float32(t.upem) * st.size * st.scaleX
}

// glyphBounds returns the integer-aligned glyph bounds in strike space (the strike's glyph rect): rounded out, zeroed
// when either dimension is empty.
func (st *strike) glyphBounds(gid uint16) geom.Rect {
	t := st.t
	if t.face == nil || t.upem <= 0 || int(gid) >= t.nGlyphs || st.degenerate() {
		return geom.Rect{}
	}
	if st.frameWidth >= 0 || st.pathEffect != nil {
		return st.styledGlyphBounds(gid)
	}
	ext, ok := t.faceGlyphExtents(opentype.GID(gid))
	if !ok || (ext.Width == 0 && ext.Height == 0) {
		return geom.Rect{}
	}
	// Map the extents quad through the text matrix and take the bounding box. Extents follow the HarfBuzz convention
	// (y-up): YBearing is the top, Width is positive, Height is negative, so the bottom edge is YBearing+Height and the
	// right edge XBearing+Width.
	p0 := st.mapDesign(ext.XBearing, ext.YBearing)
	p1 := st.mapDesign(ext.XBearing+ext.Width, ext.YBearing)
	p2 := st.mapDesign(ext.XBearing, ext.YBearing+ext.Height)
	p3 := st.mapDesign(ext.XBearing+ext.Width, ext.YBearing+ext.Height)
	r := geom.RectLTRB(
		min(p0.X, p1.X, p2.X, p3.X),
		min(p0.Y, p1.Y, p2.Y, p3.Y),
		max(p0.X, p1.X, p2.X, p3.X),
		max(p0.Y, p1.Y, p2.Y, p3.Y),
	)
	return saturateBounds(r)
}

// styledGlyphBounds computes bounds from the outline in text-matrix space, with any path effect and stroke applied,
// rounded out.
func (st *strike) styledGlyphBounds(gid uint16) geom.Rect {
	devPath := st.glyphPath(gid)
	if devPath == nil {
		return geom.Rect{}
	}
	rec := stroke.NewStrokeRec(stroke.InitStyleFill)
	if st.frameWidth >= 0 {
		rec.SetStrokeStyle(st.frameWidth, st.strokeAndFill)
		// Glyphs are always closed contours, so cap type is ignored.
		rec.SetStrokeParams(st.cap, st.join, st.miterLimit)
	}
	if st.pathEffect != nil {
		dst := path.New()
		identity := geom.IdentityMatrix()
		if st.pathEffect.FilterPath(dst, devPath, &rec, nil, &identity) {
			devPath = dst
		}
	}
	if rec.NeedToApply() {
		dst := path.New()
		if rec.ApplyToPath(dst, devPath) {
			devPath = dst
		}
	}
	return saturateBounds(devPath.Bounds())
}

// glyphPath converts the glyph outline to a path in strike (text-matrix) space. Contours are explicitly closed, as
// FT_Outline_Decompose's consumers do.
func (st *strike) glyphPath(gid uint16) *path.Path {
	return glyphOutlinePath(st.t, gid, st.mapDesign)
}

// glyphOutlinePath converts gid's raw 'glyf'/'CFF ' outline segments to a path through the given design-space point
// mapping. Contours are explicitly closed, as FT_Outline_Decompose's consumers do. Returns nil when the glyph has no
// outline data.
//
// The raw accessor is deliberate: go-text's GlyphData prefers a glyph's COLR/bitmap/SVG entry over its outline, and
// every caller here wants the outline specifically. The color lanes need it because a COLRv0 layer (or a COLRv1
// PaintGlyph) outline is frequently a glyph that itself carries a color entry — often the base glyph's own gid — which
// GlyphData would answer with instead, silently dropping the layer. The outline lane needs it because the lanes that
// set neverRequestPath cover only COLR and PNG strikes: an SVG glyph, an sbix 'jpg '/'tif ' graphic, or a B&W strike
// reaches the outline lane, and through GlyphData it would resolve no outline at all and render blank (go-text hands
// the spec-required fallback outline back in GlyphSVG.Outline/GlyphBitmap.Outline, so it is only the preference order
// that loses it).
func glyphOutlinePath(t *Typeface, gid uint16, mapPt func(x, y float32) geom.Point) *path.Path {
	outline, ok := t.faceGlyphOutline(opentype.GID(gid))
	if !ok {
		return nil
	}
	return outlineToPath(outline, mapPt)
}

// outlineToPath is glyphOutlinePath's segment walk over an already-fetched outline.
func outlineToPath(outline tsfont.GlyphOutline, mapPt func(x, y float32) geom.Point) *path.Path {
	p := path.New()
	open := false
	for _, seg := range outline.Segments {
		switch seg.Op {
		case opentype.SegmentOpMoveTo:
			if open {
				p.Close()
			}
			p.MoveToPt(mapPt(seg.Args[0].X, seg.Args[0].Y))
			open = true
		case opentype.SegmentOpLineTo:
			p.LineToPt(mapPt(seg.Args[0].X, seg.Args[0].Y))
		case opentype.SegmentOpQuadTo:
			p.QuadToPt(mapPt(seg.Args[0].X, seg.Args[0].Y), mapPt(seg.Args[1].X, seg.Args[1].Y))
		case opentype.SegmentOpCubeTo:
			p.CubicToPt(mapPt(seg.Args[0].X, seg.Args[0].Y), mapPt(seg.Args[1].X, seg.Args[1].Y),
				mapPt(seg.Args[2].X, seg.Args[2].Y))
		}
	}
	if open {
		p.Close()
	}
	return p
}

// saturateBounds rounds r out to integer bounds, zeroing it if either dimension is empty.
func saturateBounds(r geom.Rect) geom.Rect {
	if r.IsEmpty() {
		return geom.Rect{}
	}
	out := r.RoundOutRect()
	if out.Left >= out.Right || out.Top >= out.Bottom {
		return geom.Rect{}
	}
	return out
}

// fontMetrics computes the strike's font metrics for scalable outline fonts: hhea (or OS/2 typo metrics when
// fsSelection UseTypoMetrics is set) for ascent/descent/leading, the head bbox for top/bottom/xMin/xMax, post for
// underline, OS/2 for x-height/cap-height/average width/strikeout, synthesizing reasonable values for anything missing.
func (st *strike) fontMetrics() Metrics {
	var m Metrics
	t := st.t
	if t == nil || t.face == nil || t.upem <= 0 || t.hhea == nil {
		return m
	}
	upem := float32(t.upem)
	scaleY := st.size
	if st.degenerate() {
		// The scale falls back to 1 when the text matrix is singular; the metrics then come out at size 1 (observable
		// at size 0).
		scaleY = 1
	}

	// Use the OS/2 table as a source of reasonable defaults.
	xHeight := float32(0)
	avgCharWidth := float32(0)
	capHeight := float32(0)
	strikeoutThickness := float32(0)
	strikeoutPosition := float32(0)
	if t.os2 != nil {
		xHeight = float32(t.sxHght) / upem * scaleY
		avgCharWidth = float32(t.os2.XAvgCharWidth) / upem
		strikeoutThickness = float32(t.os2.YStrikeoutSize) / upem
		strikeoutPosition = -float32(t.os2.YStrikeoutPosition) / upem
		m.Flags |= MetricsFlagStrikeoutThicknessIsValid | MetricsFlagStrikeoutPositionIsValid
		capHeight = float32(t.sCapHgt) / upem * scaleY
	}

	var ascent, descent, leading float32
	const useTypoMetricsMask = 1 << 7
	if t.os2 != nil && t.os2.Version != 0xFFFF && t.os2.FsSelection&useTypoMetricsMask != 0 {
		ascent = -float32(t.os2.STypoAscender) / upem
		descent = -float32(t.os2.STypoDescender) / upem
		leading = float32(t.os2.STypoLineGap) / upem
	} else {
		ascent = -float32(t.hhea.Ascender) / upem
		descent = -float32(t.hhea.Descender) / upem
		leading = float32(t.hhea.LineGap) / upem
	}
	xmin := float32(t.head.XMin) / upem
	xmax := float32(t.head.XMax) / upem
	ymin := -float32(t.head.YMin) / upem
	ymax := -float32(t.head.YMax) / upem
	var underlineThickness, underlinePosition float32
	if t.post != nil {
		underlineThickness = float32(t.post.UnderlineThickness) / upem
		// Per the OpenType spec the post table's underlinePosition is the distance from the baseline to the *top* of
		// the underline stroke, y-up — exactly what Metrics.UnderlinePosition documents once flipped to y-down, so the
		// device-space value is simply its negation with no half-thickness adjustment. (FreeType's derived
		// underline_position is the stroke *center*, so it is not the value used here; CoreText's
		// CTFontGetUnderlinePosition reports the post value directly, as this does.)
		underlinePosition = -float32(t.post.UnderlinePosition) / upem
	}
	m.Flags |= MetricsFlagUnderlineThicknessIsValid | MetricsFlagUnderlinePositionIsValid

	// We may be able to synthesize x-height and cap-height from the outline.
	if xHeight == 0 {
		if top, ok := st.letterTop('x', scaleY); ok {
			xHeight = top
		}
	}
	if capHeight == 0 {
		if top, ok := st.letterTop('H', scaleY); ok {
			capHeight = top
		}
	}
	// Synthesize elements that were not provided by the OS/2 table or format-specific metrics.
	if xHeight == 0 {
		xHeight = -ascent * scaleY
	}
	if avgCharWidth == 0 {
		avgCharWidth = xmax - xmin
	}
	if capHeight == 0 {
		capHeight = -ascent * scaleY
	}
	// Disallow negative line spacing.
	if leading < 0 {
		leading = 0
	}

	m.Top = ymax * scaleY
	m.Ascent = ascent * scaleY
	m.Descent = descent * scaleY
	m.Bottom = ymin * scaleY
	m.Leading = leading * scaleY
	m.AvgCharWidth = avgCharWidth * scaleY
	m.XMin = xmin * scaleY
	m.XMax = xmax * scaleY
	m.MaxCharWidth = m.XMax - m.XMin
	m.XHeight = xHeight
	m.CapHeight = capHeight
	m.UnderlineThickness = underlineThickness * scaleY
	m.UnderlinePosition = underlinePosition * scaleY
	m.StrikeoutThickness = strikeoutThickness * scaleY
	m.StrikeoutPosition = strikeoutPosition * scaleY
	return m
}

// letterTop returns the device-space top (height above the baseline) of the outline of the glyph for ch at the strike
// scale, false if the glyph is missing. Used to synthesize x-height/cap-height when the OS/2 table doesn't provide
// them.
func (st *strike) letterTop(ch rune, scaleY float32) (float32, bool) {
	gid, ok := st.t.faceNominalGlyph(ch)
	if !ok || gid == 0 {
		return 0, false
	}
	ext, ok := st.t.faceGlyphExtents(gid)
	if !ok {
		return 0, false
	}
	return ext.YBearing / float32(st.t.upem) * scaleY, true
}
