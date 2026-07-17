// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file implements the content-stream operator emitters: the path constructors (m/l/c/y/re/h), the fill/stroke
// painting operators (f/B/S with the even-odd star), the transform operator (cm), the ExtGState/pattern selectors (gs,
// CS/cs/SCN/scn), and the PDF blend-mode name table. These build the page content stream the PDF device (device.go)
// generates. Object-serialization helpers (color/scalar formatting, string escaping) live in util.go and object.go.

package pdf

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stream"
)

// paintStyle selects which content-stream painting operator an emitter writes: fill, stroke, or both.
type paintStyle uint8

const (
	styleFill paintStyle = iota
	styleStroke
	styleStrokeAndFill
)

// writeScalar writes value's PDF decimal form.
func writeScalar(s stream.WStream, value float32) { s.Write(appendScalar(nil, value)) }

// blendModeName returns the PDF32000 §11.3.5 blend-mode name for mode, or "" for modes PDF cannot express as a blend
// mode (handled via form-XObject compositing in setUpContentEntry).
func blendModeName(mode raster.BlendMode) string {
	switch mode {
	case raster.BlendSrcOver:
		return "Normal"
	case raster.BlendXor:
		return "Normal" // unsupported mode
	case raster.BlendPlus:
		return "Normal" // unsupported mode
	case raster.BlendScreen:
		return "Screen"
	case raster.BlendOverlay:
		return "Overlay"
	case raster.BlendDarken:
		return "Darken"
	case raster.BlendLighten:
		return "Lighten"
	case raster.BlendColorDodge:
		return "ColorDodge"
	case raster.BlendColorBurn:
		return "ColorBurn"
	case raster.BlendHardLight:
		return "HardLight"
	case raster.BlendSoftLight:
		return "SoftLight"
	case raster.BlendDifference:
		return "Difference"
	case raster.BlendExclusion:
		return "Exclusion"
	case raster.BlendMultiply:
		return "Multiply"
	case raster.BlendHue:
		return "Hue"
	case raster.BlendSaturation:
		return "Saturation"
	case raster.BlendColor:
		return "Color"
	case raster.BlendLuminosity:
		return "Luminosity"
	default:
		return ""
	}
}

// matrixToArray returns m's affine coefficients as a PDF array, in the order PDF matrix operators expect.
func matrixToArray(m *geom.Matrix) *Array {
	a, ok := m.AsAffine()
	if !ok {
		a = geom.AffineIdentity()
	}
	return makeArrayScalars(a[0], a[1], a[2], a[3], a[4], a[5])
}

// writeColorComponentF writes value clamped to [0,1] with four significant digits.
func writeColorComponentF(s stream.WStream, value float32) {
	s.Write(appendColorComponentF(nil, value))
}

// inverseTransformBBox maps bbox by matrix's inverse in place, returning false when matrix is not invertible.
func inverseTransformBBox(matrix *geom.Matrix, bbox *geom.Rect) bool {
	inv, ok := matrix.Invert()
	if !ok {
		return false
	}
	mapped, _ := inv.MapRect(*bbox)
	*bbox = mapped
	return true
}

// populateTilingPatternDict fills pattern with the colored (PaintType 1), constant-spacing (TilingType 1)
// tiling-pattern (PatternType 1) entries over bbox, resources, and the (optional) placement matrix.
func populateTilingPatternDict(pattern *Dict, bbox geom.Rect, resources *Dict, matrix *geom.Matrix) {
	const tilingPatternType = 1
	const coloredTilingPatternPaintType = 1
	const constantSpacingTilingType = 1

	pattern.InsertName("Type", "Pattern")
	pattern.InsertInt("PatternType", tilingPatternType)
	pattern.InsertInt("PaintType", coloredTilingPatternPaintType)
	pattern.InsertInt("TilingType", constantSpacingTilingType)
	pattern.InsertObject("BBox", rectToArray(bbox))
	pattern.InsertScalar("XStep", bbox.Width())
	pattern.InsertScalar("YStep", bbox.Height())
	pattern.InsertObject("Resources", resources)
	if !matrix.IsIdentity() {
		pattern.InsertObject("Matrix", matrixToArray(matrix))
	}
}

// moveTo writes an "m" (moveto) content-stream operator.
func moveTo(x, y float32, content stream.WStream) {
	writeScalar(content, x)
	writeText(content, " ")
	writeScalar(content, y)
	writeText(content, " m\n")
}

// appendLine writes an "l" (lineto) content-stream operator.
func appendLine(x, y float32, content stream.WStream) {
	writeScalar(content, x)
	writeText(content, " ")
	writeScalar(content, y)
	writeText(content, " l\n")
}

// appendCubic writes a cubic Bezier curve operator, using the shorter "y" form (one control point equal to the
// endpoint) when possible and "c" otherwise.
func appendCubic(ctl1X, ctl1Y, ctl2X, ctl2Y, dstX, dstY float32, content stream.WStream) {
	cmd := "y\n"
	writeScalar(content, ctl1X)
	writeText(content, " ")
	writeScalar(content, ctl1Y)
	writeText(content, " ")
	if ctl2X != dstX || ctl2Y != dstY {
		cmd = "c\n"
		writeScalar(content, ctl2X)
		writeText(content, " ")
		writeScalar(content, ctl2Y)
		writeText(content, " ")
	}
	writeScalar(content, dstX)
	writeText(content, " ")
	writeScalar(content, dstY)
	writeText(content, " ")
	writeText(content, cmd)
}

// convertQuadToCubic returns the cubic Bezier control points equivalent to the quadratic Bezier src, since PDF's path
// operators have no quadratic curve primitive.
func convertQuadToCubic(src [3]geom.Point) [4]geom.Point {
	const scale = float32(2.0 / 3.0)
	s0, s1, s2 := src[0], src[1], src[2]
	return [4]geom.Point{
		s0,
		{X: s0.X + (s1.X-s0.X)*scale, Y: s0.Y + (s1.Y-s0.Y)*scale},
		{X: s2.X + (s1.X-s2.X)*scale, Y: s2.Y + (s1.Y-s2.Y)*scale},
		s2,
	}
}

// appendQuad writes a quadratic Bezier curve as an equivalent cubic curve operator.
func appendQuad(quad [3]geom.Point, content stream.WStream) {
	cubic := convertQuadToCubic(quad)
	appendCubic(cubic[1].X, cubic[1].Y, cubic[2].X, cubic[2].Y, cubic[3].X, cubic[3].Y, content)
}

// appendRectangle writes a "re" operator for rect, converting from a top-left-origin rect to PDF's bottom-left-origin
// convention.
func appendRectangle(rect geom.Rect, content stream.WStream) {
	bottom := rect.Bottom
	if rect.Top < bottom {
		bottom = rect.Top
	}
	writeScalar(content, rect.Left)
	writeText(content, " ")
	writeScalar(content, bottom)
	writeText(content, " ")
	writeScalar(content, rect.Width())
	writeText(content, " ")
	writeScalar(content, rect.Height())
	writeText(content, " re\n")
}

// closePath writes an "h" (closepath) content-stream operator.
func closePath(content stream.WStream) { writeText(content, "h\n") }

// allPointsEq reports whether every point in the span equals the first.
func allPointsEq(pts []geom.Point) bool {
	for i := 1; i < len(pts); i++ {
		if pts[i] != pts[0] {
			return false
		}
	}
	return true
}

// emitPath lowers p to PDF path-construction operators. Conics are flattened to quads at the given tolerance;
// degenerate all-equal segments are dropped when consumeDegenerates is set. Contours are buffered and flushed on close.
func emitPath(p *path.Path, style paintStyle, consumeDegenerates bool, content stream.WStream, tolerance float32) {
	if p.IsEmpty() && style == styleFill {
		appendRectangle(geom.Rect{}, content)
		return
	}
	// A closed rect (CW, or even-odd) collapses to a single "re" operator.
	if rect, isClosed, dir, ok := p.IsRectWithDirection(); ok && isClosed &&
		(dir == geom.DirectionCW || p.FillType() == path.FillEvenOdd) {
		appendRectangle(rect, content)
		return
	}

	currentSegment := stream.NewMemoryWStream()
	it := path.NewIter(p, false)
	for {
		verb, pts, ok := it.Next()
		if !ok {
			break
		}
		switch verb {
		case path.VerbMove:
			moveTo(pts[0].X, pts[0].Y, currentSegment)
		case path.VerbLine:
			if !consumeDegenerates || !allPointsEq(pts[:2]) {
				appendLine(pts[1].X, pts[1].Y, currentSegment)
			}
		case path.VerbQuad:
			if !consumeDegenerates || !allPointsEq(pts[:3]) {
				appendQuad([3]geom.Point{pts[0], pts[1], pts[2]}, currentSegment)
			}
		case path.VerbConic:
			if !consumeDegenerates || !allPointsEq(pts[:3]) {
				quads, count := path.ConicToQuads(pts[:3], it.ConicWeight(), tolerance)
				for i := 0; i < count; i++ {
					appendQuad([3]geom.Point{quads[i*2], quads[i*2+1], quads[i*2+2]}, currentSegment)
				}
			}
		case path.VerbCubic:
			if !consumeDegenerates || !allPointsEq(pts[:4]) {
				appendCubic(pts[1].X, pts[1].Y, pts[2].X, pts[2].Y, pts[3].X, pts[3].Y, currentSegment)
			}
		case path.VerbClose:
			closePath(currentSegment)
			content.Write(currentSegment.Bytes())
			currentSegment.Reset()
		}
	}
	if currentSegment.BytesWritten() > 0 {
		content.Write(currentSegment.Bytes())
	}
}

// paintPath writes the fill/stroke painting operator for style, appending the even-odd star when fill requires it
// (inverse fills are unreachable here — handled upstream).
func paintPath(style paintStyle, fill path.FillType, content stream.WStream) {
	switch style {
	case styleFill:
		writeText(content, "f")
	case styleStrokeAndFill:
		writeText(content, "B")
	case styleStroke:
		writeText(content, "S")
	}
	if style != styleStroke {
		if fill == path.FillEvenOdd {
			writeText(content, "*")
		}
	}
	writeText(content, "\n")
}

// strokePath writes the "S" stroke-painting operator.
func strokePath(content stream.WStream) { paintPath(styleStroke, path.FillWinding, content) }

// applyGraphicState writes a "gs" operator selecting the ExtGState resource at objectIndex.
func applyGraphicState(objectIndex int, content stream.WStream) {
	writeResourceName(content, resExtGState, objectIndex)
	writeText(content, " gs\n")
}

// applyPattern writes the CS/cs/SCN/scn operators that select the pattern resource at objectIndex as both the stroke
// and fill color space.
func applyPattern(objectIndex int, content stream.WStream) {
	writeText(content, "/Pattern CS/Pattern cs")
	writeResourceName(content, resPattern, objectIndex)
	writeText(content, " SCN")
	writeResourceName(content, resPattern, objectIndex)
	writeText(content, " scn\n")
}

// appendTransform writes a "cm" operator concatenating m into the current transformation matrix.
func appendTransform(m *geom.Matrix, content stream.WStream) {
	values, ok := m.AsAffine()
	if !ok {
		values = geom.AffineIdentity()
	}
	for _, v := range values {
		writeScalar(content, v)
		writeText(content, " ")
	}
	writeText(content, "cm\n")
}
