// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package tthint

const phantomPointCount = 4

// Outline is a hinted glyph outline.
type Outline struct {
	// Points are the outline's points in 26.6 pixels, y up, in the font's point order.
	Points []Point
	// OnCurve says, for each point, whether it is on the curve or a quadratic control point.
	OnCurve []bool
	// ContourEnds holds the index of the last point of each contour.
	ContourEnds []uint16
}

// scaler loads one glyph: it scales its points, composes its components and runs the glyph programs. The points of
// everything loaded so far sit at the front of scaled and flags; while a glyph is being worked on, its own points are
// followed there by its four phantom points.
type scaler struct {
	in         *Instance
	s          *scratch
	components int
	points     int
	contours   int
	phantom    [phantomPointCount]Point
}

// Outline loads and hints a glyph, appending to the slices of dst, which may be the zero value or one being reused. It
// reports false for a glyph the font does not have or describes wrongly. A glyph with no outline is not an error and
// leaves dst empty.
func (in *Instance) Outline(gid uint16, dst *Outline) bool {
	s, _ := in.pool.Get().(*scratch)
	defer in.pool.Put(s)
	sc := scaler{in: in, s: s}
	s.contours = s.contours[:0]
	if !sc.load(int(gid), 0) {
		return false
	}
	dst.Points = append(dst.Points[:0], s.scaled[:sc.points]...)
	// The outline is placed so that its origin, the first phantom point, is at zero.
	if shift := sc.phantom[0].X; shift != 0 {
		for i := range dst.Points {
			dst.Points[i].X -= shift
		}
	}
	dst.OnCurve = dst.OnCurve[:0]
	for _, flag := range s.flags[:sc.points] {
		dst.OnCurve = append(dst.OnCurve, flag&flagOnCurve != 0)
	}
	dst.ContourEnds = append(dst.ContourEnds[:0], s.contours[:sc.contours]...)
	return true
}

func (sc *scaler) scale(v int32) int32 { return mulFixed(v, sc.in.scale) }

func (sc *scaler) scalePoint(p Point) Point { return Point{X: sc.scale(p.X), Y: sc.scale(p.Y)} }

// reserve makes room in the accumulated buffers for n points.
func (sc *scaler) reserve(n int) {
	s := sc.s
	if n > len(s.scaled) {
		s.scaled = append(s.scaled, make([]Point, n-len(s.scaled))...)
		s.flags = append(s.flags, make([]uint8, n-len(s.flags))...)
	}
}

func grow(points []Point, n int) []Point {
	if n > cap(points) {
		return make([]Point, n)
	}
	return points[:n]
}

func (sc *scaler) load(gid, depth int) bool {
	if depth > maxRecursionDepth {
		return false
	}
	f := sc.in.font
	data, ok := f.glyphData(gid)
	if !ok {
		return false
	}
	var header glyphHeader
	var body []byte
	if len(data) != 0 {
		if header, body, ok = parseGlyphHeader(data); !ok {
			return false
		}
	}
	// The phantom points: the horizontal origin and advance, then the vertical ones, in font units for now.
	advance, lsb := f.hMetrics(gid)
	ascent, descent := int32(f.typoAscender), int32(f.typoDescender)
	sc.phantom[0] = Point{X: int32(header.xMin) - lsb}
	sc.phantom[1] = Point{X: sc.phantom[0].X + advance}
	sc.phantom[2] = Point{Y: ascent}
	sc.phantom[3] = Point{Y: ascent - (ascent - descent)}
	switch {
	case len(data) == 0:
		for i := range sc.phantom {
			sc.phantom[i] = sc.scalePoint(sc.phantom[i])
		}
		return true
	case header.contours >= 0:
		return sc.loadSimple(body, header.contours)
	default:
		return sc.loadComposite(body, depth)
	}
}

func (sc *scaler) loadSimple(body []byte, contourCount int) bool {
	s := sc.s
	glyph := &s.simple
	if !parseSimpleGlyph(body, contourCount, glyph) {
		return false
	}
	pointCount := len(glyph.points)
	pointsStart := sc.points
	pointsEnd := pointsStart + pointCount + phantomPointCount
	if pointsEnd > maxOutlinePoints {
		return false
	}
	sc.reserve(pointsEnd)
	scaled := s.scaled[pointsStart:pointsEnd]
	flags := s.flags[pointsStart:pointsEnd]
	s.unscaled = grow(s.unscaled, pointCount+phantomPointCount)
	unscaled := s.unscaled
	copy(unscaled, glyph.points)
	for i, on := range glyph.onCurve {
		flags[i] = 0
		if on {
			flags[i] = flagOnCurve
		}
	}
	contoursStart := len(s.contours)
	s.contours = append(s.contours, glyph.endPts...)
	contours := s.contours[contoursStart:]
	sc.points += pointCount
	sc.contours += len(contours)
	for i, phantom := range sc.phantom {
		unscaled[pointCount+i] = phantom
		flags[pointCount+i] = 0
	}
	for i, p := range unscaled {
		scaled[i] = sc.scalePoint(p)
	}
	copy(sc.phantom[:], scaled[pointCount:])
	if len(glyph.instructions) != 0 {
		s.original = grow(s.original, len(scaled))
		copy(s.original, scaled)
		for i := pointCount; i < len(scaled); i++ {
			scaled[i] = Point{X: round26(scaled[i].X), Y: round26(scaled[i].Y)}
		}
		if sc.in.hint(s, unscaled, s.original, scaled, flags, contours, glyph.instructions, false) {
			copy(sc.phantom[:], scaled[pointCount:])
		}
	} else if !sc.in.backwardCompatibility() {
		for i, p := range scaled[pointCount:] {
			sc.phantom[i] = Point{X: round26(p.X), Y: round26(p.Y)}
		}
	}
	if pointsStart != 0 {
		for i := range contours {
			contours[i] += uint16(pointsStart)
		}
	}
	return true
}

// componentScale widens a 2.14 transform term to 16.16.
func componentScale(v int16) int32 { return int32(v) * 4 }

// hypot16 is FreeType's cheap approximation of the length of a 16.16 vector.
func hypot16(a, b int32) int32 {
	a, b = abs32(a), abs32(b)
	if a > b {
		return a + ((3 * b) >> 3)
	}
	return b + ((3 * a) >> 3)
}

func (sc *scaler) loadComposite(body []byte, depth int) bool {
	s := sc.s
	components, instructions, ok := parseCompositeGlyph(body)
	if !ok {
		return false
	}
	pointBase := sc.points
	contourBase := sc.contours
	for i := range sc.phantom {
		sc.phantom[i] = sc.scalePoint(sc.phantom[i])
	}
	for i := range components {
		c := &components[i]
		sc.components++
		if sc.components > maxComponents {
			return false
		}
		phantom := sc.phantom
		startPoint := sc.points
		if !sc.load(c.glyph, depth+1) {
			return false
		}
		endPoint := sc.points
		if c.flags&compUseMyMetrics == 0 {
			sc.phantom = phantom
		}
		xx, yx := componentScale(c.xx), componentScale(c.yx)
		xy, yy := componentScale(c.xy), componentScale(c.yy)
		haveTransform := c.flags&compHaveTransformMask != 0
		if haveTransform {
			for j := startPoint; j < endPoint; j++ {
				p := s.scaled[j]
				s.scaled[j] = Point{
					X: mulFixed(p.X, xx) + mulFixed(p.Y, xy),
					Y: mulFixed(p.X, yx) + mulFixed(p.Y, yy),
				}
			}
		}
		var offset Point
		if c.flags&compArgsAreXYValues != 0 {
			x, y := c.arg1, c.arg2
			if haveTransform && c.flags&(compScaledOffset|compUnscaledOffset) == compScaledOffset {
				x = mulFixed(x, hypot16(xx, xy))
				y = mulFixed(y, hypot16(yy, yx))
			}
			offset = sc.scalePoint(Point{X: x, Y: y})
			if c.flags&compRoundXYToGrid != 0 {
				offset.Y = round26(offset.Y)
			}
		} else {
			// The component is placed by making one of its points coincide with one already in the composite.
			base := pointBase + int(c.arg1)
			comp := startPoint + int(c.arg2)
			if base >= len(s.scaled) || comp >= len(s.scaled) {
				return false
			}
			offset = Point{X: s.scaled[base].X - s.scaled[comp].X, Y: s.scaled[base].Y - s.scaled[comp].Y}
		}
		if offset.X != 0 || offset.Y != 0 {
			for j := startPoint; j < endPoint; j++ {
				s.scaled[j].X += offset.X
				s.scaled[j].Y += offset.Y
			}
		}
	}
	if len(instructions) == 0 {
		return true
	}
	// The composite's own program sees all of its components' points as one glyph, already scaled, so the "unscaled"
	// points it measures against are the scaled ones.
	endPoint := sc.points + phantomPointCount
	if endPoint > maxOutlinePoints {
		return false
	}
	sc.reserve(endPoint)
	scaled := s.scaled[pointBase:endPoint]
	flags := s.flags[pointBase:endPoint]
	phantomStart := len(scaled) - phantomPointCount
	for i, phantom := range sc.phantom {
		scaled[phantomStart+i] = phantom
		flags[phantomStart+i] = 0
	}
	s.unscaled = grow(s.unscaled, len(scaled))
	copy(s.unscaled, scaled)
	s.original = grow(s.original, len(scaled))
	copy(s.original, scaled)
	for i := phantomStart; i < len(scaled); i++ {
		scaled[i] = Point{X: round26(scaled[i].X), Y: round26(scaled[i].Y)}
	}
	for i := range flags {
		flags[i] &^= flagTouched
	}
	contours := s.contours[contourBase:sc.contours]
	if pointBase != 0 {
		for i := range contours {
			contours[i] -= uint16(pointBase)
		}
	}
	if sc.in.hint(s, s.unscaled, s.original, scaled, flags, contours, instructions, true) {
		copy(sc.phantom[:], scaled[phantomStart:])
	}
	if pointBase != 0 {
		for i := range contours {
			contours[i] += uint16(pointBase)
		}
	}
	return true
}
