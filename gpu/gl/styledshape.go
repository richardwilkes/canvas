// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// StyledShape pairs a geometric Shape with the Style it should be rendered with, style application (producing a new
// StyledShape whose geometry reflects the styling), and the unstyled/inherited key machinery the mask and geometry
// caches build on. No path effects reach the surface draw context, so the path-effect application lanes and the dash
// simplifications don't apply. There is no destruction hook for gen-ID change notifications, so stale mask-cache
// entries age out of the budgeted LRU instead. The arc shape type doesn't exist as its own geometry: MakeStyledShapeArc
// builds the equivalent arc path directly, and winding-sensitive simplification stays disabled for non-butt-cap arcs
// the same way it would for that path form.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// styledShapeMaxKeyFromDataVerbCnt is the verb-count threshold below which a path's key may be extracted directly from
// its data rather than relying on its generation ID.
const styledShapeMaxKeyFromDataVerbCnt = 10

// DoSimplify controls whether a StyledShape constructor simplifies the shape immediately.
type DoSimplify bool

// DoSimplify values.
const (
	DoSimplifyNo  DoSimplify = false
	DoSimplifyYes DoSimplify = true
)

// FillInversion informs MakeFilledStyledShape on how to modify the shape's fill rule when making a simple filled
// version of the shape.
type FillInversion uint8

// FillInversion values.
const (
	FillInversionPreserve FillInversion = iota
	FillInversionFlip
	FillInversionForceNoninverted
	FillInversionForceInverted
)

// StyledShape is a geometric shape paired with the style it should be rendered with. The zero value is an empty shape
// with a simple fill.
type StyledShape struct {
	inheritedKey []uint32
	shape        Shape
	style        Style
	// genID is the generation ID of the original path (the path may be modified or simplified away); 0 means "no key"
	// (volatile or inherited-key-failed).
	genID      uint32
	closed     bool
	simplified bool
}

// MakeStyledShapePath builds a StyledShape wrapping a path.
func MakeStyledShapePath(p *path.Path, style Style, doSimplify DoSimplify) StyledShape {
	s := StyledShape{style: style}
	s.shape.SetPath(p)
	if doSimplify {
		s.simplify()
	}
	return s
}

// MakeStyledShapeRect builds a StyledShape wrapping a rect.
func MakeStyledShapeRect(rect geom.Rect, style Style, doSimplify DoSimplify) StyledShape {
	s := StyledShape{style: style}
	s.shape.SetRect(rect)
	if doSimplify {
		s.simplify()
	}
	return s
}

// MakeStyledShapeRRect builds a StyledShape wrapping a round rect, using the legacy starting index (6, clockwise) a
// path builder would use when adding a round rect.
func MakeStyledShapeRRect(rrect geom.RRect, style Style, doSimplify DoSimplify) StyledShape {
	return MakeStyledShapeRRectWithWinding(rrect, geom.DirectionCW, 6, false, style, doSimplify)
}

// MakeStyledShapeRRectWithWinding builds a StyledShape wrapping a round rect with an explicit winding direction,
// starting point index, and inversion.
func MakeStyledShapeRRectWithWinding(rrect geom.RRect, dir geom.PathDirection, start uint8, inverted bool, style Style, doSimplify DoSimplify) StyledShape {
	s := StyledShape{style: style}
	s.shape.SetRRect(rrect)
	s.shape.setPathWindingParams(dir, start)
	s.shape.SetInverted(inverted)
	if doSimplify {
		s.simplify()
	}
	return s
}

// MakeStyledShapeArc builds a StyledShape for an arc. There is no dedicated arc geometry type, so the arc becomes the
// equivalent draw-arc path immediately. Winding-sensitive simplifications stay disabled for non-butt caps, since the
// path form always takes the plain path simplify branch, which never applies them.
func MakeStyledShapeArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, style Style, doSimplify DoSimplify) StyledShape {
	arcPath := path.CreateDrawArcPath(oval.Sorted(), startAngle, sweepAngle, useCenter,
		style.IsSimpleFill())
	// Arc paths stay non-volatile: they are small enough for the key-from-data lane, so identical arcs share cached
	// masks.
	return MakeStyledShapePath(arcPath, style, doSimplify)
}

// makeStyledShapeFromShape wraps an existing geometry Shape with a style; the surface draw context receives ready-made
// Shapes from the device. The shape is deep-copied.
func makeStyledShapeFromShape(shape *Shape, style Style, doSimplify DoSimplify) StyledShape {
	s := StyledShape{shape: *cloneShape(shape), style: style}
	if doSimplify {
		s.simplify()
	}
	return s
}

// MakeFilledStyledShape makes a filled shape from the pre-styled original shape and optionally modifies whether the
// fill is inverted or not.
func MakeFilledStyledShape(original *StyledShape, inversion FillInversion) StyledShape {
	var newIsInverted bool
	switch inversion {
	case FillInversionPreserve:
		newIsInverted = original.shape.Inverted()
	case FillInversionFlip:
		newIsInverted = !original.shape.Inverted()
	case FillInversionForceInverted:
		newIsInverted = true
	default:
		newIsInverted = false
	}
	if original.style.IsSimpleFill() && newIsInverted == original.shape.Inverted() {
		// By returning the original rather than falling through we can preserve any inherited style key. Otherwise, we
		// wipe it out below since the style change invalidates it.
		return original.clone()
	}
	var result StyledShape
	result.style = SimpleFillStyle()
	result.shape = *cloneShape(&original.shape)
	result.genID = original.genID
	result.shape.SetInverted(newIsInverted)

	if !original.style.IsSimpleFill() {
		// Going from a non-filled style to fill may allow additional simplifications (e.g. closing an open rect that
		// wasn't closed in the original shape because it had stroke style).
		result.simplify()
		// The above simplify() call only sets simplified to true if its geometry was changed, since it already sees its
		// style as a simple fill. Since the original style was not a simple fill, MakeFilled always simplifies.
		result.simplified = true
	}
	// We don't copy the inherited key since it can contain style information that we just stripped.
	return result
}

// cloneShape deep-copies a Shape (the path member is cloned so the two shapes cannot alias).
func cloneShape(s *Shape) *Shape {
	c := *s
	if c.typ == ShapeTypePath {
		c.path = s.path.Clone()
	}
	return &c
}

// clone deep-copies the styled shape.
func (s *StyledShape) clone() StyledShape {
	c := *s
	c.shape = *cloneShape(&s.shape)
	c.inheritedKey = append([]uint32(nil), s.inheritedKey...)
	return c
}

// Style returns the style this shape should be rendered with.
func (s *StyledShape) Style() *Style { return &s.style }

// Simplified reports whether the shape and/or style were modified into a simpler, equivalent pairing.
func (s *StyledShape) Simplified() bool { return s.simplified }

// ApplyStyle returns a shape that has applied the styling information from this shape's style to its geometry. scale is
// used when approximating the output geometry and typically is computed from the view matrix. (The path-effect-only
// apply mode is unreachable, since no path effect exists; both modes behave as applying both the path effect and stroke
// rec, minus the effect.)
func (s *StyledShape) ApplyStyle(apply StyleApply, scale float32) StyledShape {
	return makeStyledShapeFromParent(s, apply, scale)
}

// makeStyledShapeFromParent builds the styled shape that results from applying parent's style to its geometry. There is
// no path effect, so only the stroke-rec application applies.
func makeStyledShapeFromParent(parent *StyledShape, apply StyleApply, scale float32) StyledShape {
	if !parent.style.Applies() || apply == StyleApplyPathEffectOnly {
		// No path effect exists, so the path-effect-only mode has nothing to apply, and a non-applying style leaves the
		// geometry untouched.
		return parent.clone()
	}

	var result StyledShape
	srcForParentStyle := parent.AsPath()
	// SetPath (below) clones styled, so the transient styling output is pooled and returned.
	styled := path.Borrow()
	defer path.Recycle(styled)
	fillOrHairline, ok := parent.style.ApplyToPath(styled, srcForParentStyle, scale)
	if !ok {
		// A stroker failure leaves the freshly-reset empty path in place (drawing nothing) rather than panicking.
		fillOrHairline = stroke.InitStyleFill
	}
	result.shape.SetPath(styled)
	result.style.ResetToInitStyle(fillOrHairline)

	result.simplify()
	result.setInheritedKey(parent, apply, scale)
	return result
}

// IsRect reports whether the unstyled geometry is a rect.
func (s *StyledShape) IsRect() bool { return s.shape.IsRect() }

// Shape returns the underlying geometry, for callers (such as the surface draw context's simple-shape retry lane) that
// want the whole thing rather than a piecemeal accessor.
func (s *StyledShape) Shape() *Shape { return &s.shape }

// AsRRect returns the unstyled geometry as a round rect if possible.
func (s *StyledShape) AsRRect() (rrect geom.RRect, inverted, ok bool) {
	if !s.shape.IsRRect() && !s.shape.IsRect() {
		return geom.RRect{}, false, false
	}
	if s.shape.IsRect() {
		rrect = geom.MakeRRect(s.shape.Rect(), 0, 0)
	} else {
		rrect = s.shape.RRect()
	}
	return rrect, s.shape.Inverted(), true
}

// AsLine reports whether the unstyled shape is a straight line segment, returning its endpoints if so. An inverse
// filled line path is still considered a line.
func (s *StyledShape) AsLine() (pts [2]geom.Point, inverted, ok bool) {
	if !s.shape.IsLine() {
		return pts, false, false
	}
	return s.shape.Line(), s.shape.Inverted(), true
}

// AsNestedRects reports whether this shape can be drawn as a pair of filled nested rectangles.
func (s *StyledShape) AsNestedRects() (rects [2]geom.Rect, ok bool) {
	if !s.shape.IsPath() {
		return rects, false
	}
	if s.shape.Path().IsInverseFillType() {
		return rects, false
	}
	var dirs [2]geom.PathDirection
	if !s.shape.Path().IsNestedFillRectsWithDirections(&rects, &dirs) {
		return rects, false
	}
	if s.shape.Path().FillType() == path.FillWinding && dirs[0] == dirs[1] {
		// The two rects need to be wound opposite to each other.
		return rects, false
	}
	// Right now, nested rects where the margin is not the same width all around do not render correctly.
	outer := [4]float32{rects[0].Left, rects[0].Top, rects[0].Right, rects[0].Bottom}
	inner := [4]float32{rects[1].Left, rects[1].Top, rects[1].Right, rects[1].Bottom}
	allEq := true
	margin := geom.ScalarAbs(outer[0] - inner[0])
	allGoE1 := margin >= 1
	for i := 1; i < 4; i++ {
		temp := geom.ScalarAbs(outer[i] - inner[i])
		if temp < 1 {
			allGoE1 = false
		}
		if !geom.ScalarNearlyEqual(margin, temp) {
			allEq = false
		}
	}
	return rects, allEq || allGoE1
}

// AsPath returns the unstyled geometry as a path. (The simple-fill argument only affected the arc shape type, which
// doesn't exist as its own geometry here.)
func (s *StyledShape) AsPath() *path.Path {
	return s.shape.AsPath()
}

// VerbCount returns AsPath().CountVerbs() without materializing the clone AsPath would make when the geometry is
// already a path: the path renderers' OnCanDrawPath gates call this once per candidate renderer per draw, and only need
// the number.
func (s *StyledShape) VerbCount() int {
	if p := s.shape.PeekPath(); p != nil {
		return p.CountVerbs()
	}
	return s.shape.AsPath().CountVerbs()
}

// IsEmpty reports whether the geometry is empty. Note that applying the style could produce a non-empty shape; it also
// may have an inverse fill.
func (s *StyledShape) IsEmpty() bool { return s.shape.IsEmpty() }

// Bounds returns the bounds of the geometry without reflecting the shape's styling, ignoring the inverse fill nature.
func (s *StyledShape) Bounds() geom.Rect { return s.shape.Bounds() }

// StyledBounds returns the bounds of the geometry reflecting the shape's styling (ignoring inverse fill status).
func (s *StyledShape) StyledBounds() geom.Rect {
	if s.IsEmpty() {
		return geom.Rect{}
	}
	return s.style.AdjustBounds(s.Bounds())
}

// KnownToBeConvex reports whether this shape is known to be convex, before styling is applied.
func (s *StyledShape) KnownToBeConvex() bool {
	return s.shape.Convex(s.style.IsSimpleFill())
}

// KnownDirection reports whether the shape has a known winding direction.
func (s *StyledShape) KnownDirection() bool {
	return !s.shape.IsPath() ||
		s.shape.Path().ComputeFirstDirection() != path.FirstDirectionUnknown
}

// InverseFilled reports whether the pre-styled geometry is inverse filled.
func (s *StyledShape) InverseFilled() bool { return s.shape.Inverted() }

// MayBeInverseFilledAfterStyling reports whether the shape may be inverse filled after its style is applied. With no
// path effects this is exactly InverseFilled.
func (s *StyledShape) MayBeInverseFilledAfterStyling() bool { return s.InverseFilled() }

// KnownToBeClosed reports whether it is known that the unstyled geometry has no unclosed contours (so no caps when
// stroked).
func (s *StyledShape) KnownToBeClosed() bool { return s.shape.Closed() }

// SegmentMask returns the bitmask of segment types (line/quad/conic/cubic) present in the shape.
func (s *StyledShape) SegmentMask() uint8 { return s.shape.SegmentMask() }

// pathKeyFromDataSize returns the key length if the path is small enough to be keyed from its data, otherwise -1.
func pathKeyFromDataSize(p *path.Path) int {
	verbCnt := p.CountVerbs()
	if verbCnt > styledShapeMaxKeyFromDataVerbCnt {
		return -1
	}
	pointCnt := p.CountPoints()
	conicWeightCnt := 0
	it := path.NewRawIter(p)
	for {
		verb, _, _, ok := it.Next()
		if !ok {
			break
		}
		if verb == path.VerbConic {
			conicWeightCnt++
		}
	}
	// 1 is for the verb count. Each verb is a byte but we'll pad the verb data out to a uint32 length.
	return 1 + (verbCnt+3)/4 + 2*pointCnt + conicWeightCnt
}

// writePathKeyFromData writes the path data key into key.
func writePathKeyFromData(p *path.Path, key []uint32) {
	verbCnt := p.CountVerbs()
	i := 0
	key[i] = uint32(verbCnt)
	i++
	// Verbs, packed four to a uint32, padded with 0xDE (a value that will stand out when debugging).
	verbWords := (verbCnt + 3) / 4
	verbIdx := 0
	var weights []float32
	var pts []geom.Point
	it := path.NewRawIter(p)
	packVerb := func(v path.Verb) {
		key[i+verbIdx/4] |= uint32(v) << (8 * uint(verbIdx%4))
		verbIdx++
	}
	for w := 0; w < verbWords; w++ {
		key[i+w] = 0
	}
	for {
		verb, vpts, weight, ok := it.Next()
		if !ok {
			break
		}
		packVerb(verb)
		// Reconstruct the flat point-array layout: each verb contributes only its new points.
		switch verb {
		case path.VerbMove:
			pts = append(pts, vpts[0])
		case path.VerbLine:
			pts = append(pts, vpts[1])
		case path.VerbQuad:
			pts = append(pts, vpts[1], vpts[2])
		case path.VerbConic:
			pts = append(pts, vpts[1], vpts[2])
			weights = append(weights, weight)
		case path.VerbCubic:
			pts = append(pts, vpts[1], vpts[2], vpts[3])
		case path.VerbClose:
		}
	}
	// Pad the remaining verb bytes with 0xDE.
	for ; verbIdx < verbWords*4; verbIdx++ {
		key[i+verbIdx/4] |= 0xDE << (8 * uint(verbIdx%4))
	}
	i += verbWords
	for _, pt := range pts {
		key[i] = math.Float32bits(pt.X)
		key[i+1] = math.Float32bits(pt.Y)
		i += 2
	}
	for _, w := range weights {
		key[i] = math.Float32bits(w)
		i++
	}
	if i != pathKeyFromDataSize(p) {
		panic("path key size mismatch")
	}
}

// UnstyledKeySize returns the size of the key for the shape represented by this StyledShape (ignoring its styling). A
// negative value is returned if the shape has no key (it shouldn't be cached).
func (s *StyledShape) UnstyledKeySize() int {
	if len(s.inheritedKey) > 0 {
		return len(s.inheritedKey)
	}
	count := 1 // Every key has the state flags from the Shape.
	switch s.shape.Type() {
	case ShapeTypePoint:
		count += 2 // one geom.Point
	case ShapeTypeRect:
		count += 4 // one geom.Rect
	case ShapeTypeRRect:
		// The rect plus four radii pairs; the uniform-radii RRect representation expands to this same layout for keying
		// purposes.
		count += 12
	case ShapeTypeLine:
		count += 4 // two geom.Points
	case ShapeTypePath:
		if s.genID == 0 {
			return -1 // volatile, so won't be keyed
		}
		if dataKeySize := pathKeyFromDataSize(s.shape.Path()); dataKeySize >= 0 {
			count += dataKeySize
		} else {
			count++ // Just adds the gen ID.
		}
	default:
		// Else it's empty, which just needs the state flags for its key.
		if !s.shape.IsEmpty() {
			panic("unexpected shape type")
		}
	}
	return count
}

// HasUnstyledKey reports whether this shape can be assigned a cache key.
func (s *StyledShape) HasUnstyledKey() bool { return s.UnstyledKeySize() >= 0 }

// WriteUnstyledKey writes UnstyledKeySize values into key, which must be large enough.
func (s *StyledShape) WriteUnstyledKey(key []uint32) {
	if len(s.inheritedKey) > 0 {
		copy(key, s.inheritedKey)
		return
	}
	// Every key starts with the state from the Shape (this includes path fill type, and any tracked winding, start,
	// inversion, as well as the class of geometry).
	key[0] = s.shape.StateKey()
	i := 1
	switch s.shape.Type() {
	case ShapeTypePath:
		if s.genID == 0 {
			panic("writeUnstyledKey on un-keyable shape")
		}
		if dataKeySize := pathKeyFromDataSize(s.shape.Path()); dataKeySize >= 0 {
			writePathKeyFromData(s.shape.Path(), key[i:])
			return
		}
		key[i] = s.genID
	case ShapeTypePoint:
		pt := s.shape.Point()
		key[i] = math.Float32bits(pt.X)
		key[i+1] = math.Float32bits(pt.Y)
	case ShapeTypeRect:
		writeRectToKey(key[i:], s.shape.Rect())
	case ShapeTypeRRect:
		writeRRectToKey(key[i:], s.shape.RRect())
	case ShapeTypeLine:
		line := s.shape.Line()
		key[i] = math.Float32bits(line[0].X)
		key[i+1] = math.Float32bits(line[0].Y)
		key[i+2] = math.Float32bits(line[1].X)
		key[i+3] = math.Float32bits(line[1].Y)
	default:
		if !s.shape.IsEmpty() {
			panic("unexpected shape type")
		}
	}
}

// writeRectToKey writes a rect's four float bit patterns.
func writeRectToKey(key []uint32, r geom.Rect) {
	key[0] = math.Float32bits(r.Left)
	key[1] = math.Float32bits(r.Top)
	key[2] = math.Float32bits(r.Right)
	key[3] = math.Float32bits(r.Bottom)
}

// writeRRectToKey writes a round rect's key layout (rect + four radii pairs), expanded from the uniform-radii
// representation.
func writeRRectToKey(key []uint32, rr geom.RRect) {
	writeRectToKey(key, rr.Rect)
	radii := rrectRadii(rr)
	for c := 0; c < 4; c++ {
		key[4+2*c] = math.Float32bits(radii.X)
		key[5+2*c] = math.Float32bits(radii.Y)
	}
}

// setInheritedKey determines the key this shape should inherit from the input shape's geometry and style when applying
// the style to create a new shape.
func (s *StyledShape) setInheritedKey(parent *StyledShape, apply StyleApply, scale float32) {
	if len(s.inheritedKey) > 0 {
		panic("inherited key already set")
	}
	// If the output shape turns out to be simple, then we will just use its geometric key.
	if !s.shape.IsPath() {
		return
	}
	// The full key is structured as (geo,path_effect,stroke); with no path effects the parent's key is always its
	// geometry key.
	parentCnt := len(parent.inheritedKey)
	useParentGeoKey := parentCnt == 0
	if useParentGeoKey {
		parentCnt = parent.UnstyledKeySize()
		if parentCnt < 0 {
			// The parent's geometry has no key so we will have no key.
			s.genID = 0
			return
		}
	}
	var styleKeyFlags uint32
	if parent.KnownToBeClosed() {
		styleKeyFlags |= StyleKeyClosed
	}
	if _, _, isLine := parent.AsLine(); isLine {
		styleKeyFlags |= StyleKeyNoJoins
	}
	styleCnt := StyleKeySize(&parent.style, apply, styleKeyFlags)
	if styleCnt < 0 {
		// The style doesn't allow a key; set the path gen ID to 0 so that we fail when we try to get a key for the
		// shape.
		s.genID = 0
		return
	}
	s.inheritedKey = make([]uint32, parentCnt+styleCnt)
	if useParentGeoKey {
		// This will be the geo key.
		parent.WriteUnstyledKey(s.inheritedKey)
	} else {
		copy(s.inheritedKey, parent.inheritedKey)
	}
	// Now turn (geo) into (geo,stroke).
	WriteStyleKey(s.inheritedKey[parentCnt:], &parent.style, apply, scale, styleKeyFlags)
}

// simplify is like Shape.Simplify but also takes into account style and stroking, possibly applying the style
// explicitly to produce a new analytic shape with a simpler style.
func (s *StyledShape) simplify() {
	// Restore inverseness after simplification (dashing never applies, since no path effect exists).
	inverted := s.shape.Inverted()
	restoreInverseness := func() { s.shape.SetInverted(inverted) }

	var simplifyFlags uint32
	if s.style.IsSimpleFill() {
		simplifyFlags = ShapeSimplifyAll
	} else {
		// Everything but arcs with caps that might extend beyond the oval edge can ignore winding (the arc type is
		// trimmed — arc shapes are paths here and paths never take the destructive simplifications anyway).
		simplifyFlags = ShapeSimplifyIgnoreWinding | ShapeSimplifyMakeCanonical
	}

	// Remember if the original shape was closed; in the event we simplify to a point or line because of degenerate
	// geometry, we need to update joins and caps.
	oldType := s.shape.Type()
	s.closed = s.shape.Simplify(simplifyFlags)
	s.simplified = oldType != s.shape.Type()

	if s.shape.IsPath() {
		// The shape remains a path, so configure the gen ID and canonicalize the fill type if possible.
		if len(s.inheritedKey) > 0 || s.shape.Path().IsVolatile() {
			s.genID = 0
		} else {
			s.genID = s.shape.Path().GenerationID()
		}
		if s.style.Rec().Style() == stroke.StyleStroke ||
			s.style.Rec().Style() == stroke.StyleHairline ||
			s.shape.Path().IsConvex() {
			// Stroke styles don't differentiate between winding and even/odd. There is no distinction between even/odd
			// and non-zero winding count for convex paths.
			if s.shape.Path().IsInverseFillType() {
				s.shape.Path().SetFillType(path.FillInverseEvenOdd)
			} else {
				s.shape.Path().SetFillType(shapeDefaultFillType)
			}
		}
		restoreInverseness()
	} else {
		s.inheritedKey = nil
		restoreInverseness()
		// Further simplifications to the shape based on the style.
		s.simplifyStroke()
	}
}

// simplifyStroke handles shapes whose stroking can be trivially evaluated, forming a new geometry with just a fill.
func (s *StyledShape) simplifyStroke() {
	inverted := s.shape.Inverted()
	defer func() { s.shape.SetInverted(inverted) }()

	// For stroke+filled rects, a mitered shape becomes a larger rect and a rounded shape becomes a round rect.
	if s.shape.IsRect() && s.style.Rec().Style() == stroke.StyleStrokeAndFill {
		if s.style.Rec().Join() == stroke.JoinBevel ||
			(s.style.Rec().Join() == stroke.JoinMiter &&
				s.style.Rec().Miter() < float32(math.Sqrt2)) {
			// Bevel-stroked rect needs path rendering.
			return
		}
		r := s.style.Rec().Width() / 2
		// fShape.rect().outset(r, r) mutates the rect in place, keeping the winding params.
		s.shape.rect = s.shape.rect.Outset(r, r)
		if s.style.Rec().Join() == stroke.JoinRound {
			// There's no dashing to worry about if we got here, so it's okay that this resets winding parameters.
			s.shape.SetRRect(geom.MakeRRect(s.shape.rect, r, r))
		}
		s.style = SimpleFillStyle()
		s.simplified = true
		return
	}

	// Otherwise, if we're a point or a line, we might be able to explicitly apply some of the stroking. Any other
	// shape+style is too complicated to reduce. (The dash lanes are unreachable.)
	if (!s.shape.IsPoint() && !s.shape.IsLine()) || s.style.Rec().IsHairlineStyle() {
		return
	}

	// Tracks style simplifications, even if the geometry can't be further simplified.
	styleSimplified := false

	// At this point, we're a line or point with no path effects. Any fill portion of the style is empty, so a fill-only
	// style can be empty, and a stroke+fill becomes a stroke.
	if s.style.IsSimpleFill() {
		s.shape.Reset()
		s.simplified = true
		return
	}
	if s.style.Rec().Style() == stroke.StyleStrokeAndFill {
		// Stroke only.
		rec := *s.style.Rec()
		rec.SetStrokeStyle(s.style.Rec().Width(), false)
		s.style = MakeStyle(rec)
		styleSimplified = true
	}

	// A point or line that was formed by a degenerate closed shape needs its style updated to reflect the fact that it
	// doesn't actually produce caps.
	if s.closed {
		var capType stroke.Cap
		if s.shape.IsLine() && s.style.Rec().Join() == stroke.JoinRound {
			// As a closed shape, the line moves from a to b and back to a, producing a 180 degree turn. With round
			// joins, this would make a semi-circle at each end, which is visually identical to a round cap on the
			// reduced line geometry.
			capType = stroke.CapRound
		} else {
			// If this were a closed line, the 180 degree turn either is a miter join that exceeds the miter limit and
			// becomes a bevel, or a bevel join. In either case, the bevel shape of a 180 degree corner is equivalent to
			// a butt cap.
			capType = stroke.CapButt
		}
		if capType != s.style.Rec().Cap() || s.style.Rec().Join() != stroke.JoinMiter {
			rec := *s.style.Rec()
			rec.SetStrokeParams(capType, stroke.JoinMiter, s.style.Rec().Miter())
			s.style = MakeStyle(rec)
			styleSimplified = true
		}
	}

	if s.shape.IsPoint() {
		// The drawn geometry is entirely based on the cap style and stroke width. A butt-cap point doesn't draw
		// anything, a round cap is an oval and a square cap is a square.
		switch s.style.Rec().Cap() {
		case stroke.CapButt:
			s.shape.Reset()
		default:
			w := s.style.Rec().Width() / 2
			pt := s.shape.Point()
			r := geom.RectLTRB(pt.X, pt.Y, pt.X, pt.Y).Outset(w, w)
			if s.style.Rec().Cap() == stroke.CapRound {
				s.shape.SetRRect(geom.MakeRRect(r, r.Width()/2, r.Height()/2))
			} else {
				s.shape.SetRect(r)
			}
		}
	} else {
		// Stroked lines reduce to rectangles or round rects when they are axis-aligned.
		var r geom.Rect
		var outset geom.Point
		line := s.shape.Line()
		switch {
		case line[0].Y == line[1].Y:
			r.Left = min(line[0].X, line[1].X)
			r.Right = max(line[0].X, line[1].X)
			r.Top = line[0].Y
			r.Bottom = line[0].Y
			outset.Y = s.style.Rec().Width() / 2
			if s.style.Rec().Cap() == stroke.CapButt {
				outset.X = 0
			} else {
				outset.X = outset.Y
			}
		case line[0].X == line[1].X:
			r.Top = min(line[0].Y, line[1].Y)
			r.Bottom = max(line[0].Y, line[1].Y)
			r.Left = line[0].X
			r.Right = line[0].X
			outset.X = s.style.Rec().Width() / 2
			if s.style.Rec().Cap() == stroke.CapButt {
				outset.Y = 0
			} else {
				outset.Y = outset.X
			}
		default:
			// Geometrically can't apply the style and turn into a fill, but might still be simpler than before based
			// solely on changes to the style.
			s.simplified = s.simplified || styleSimplified
			return
		}
		r = r.Outset(outset.X, outset.Y)
		switch {
		case r.IsEmpty():
			s.shape.Reset()
		case s.style.Rec().Cap() == stroke.CapRound:
			s.shape.SetRRect(geom.MakeRRect(r, outset.X, outset.Y))
		default:
			s.shape.SetRect(r)
		}
	}
	// If we made it here, the stroke was fully applied to the new shape so we can become a fill.
	s.style = SimpleFillStyle()
	s.simplified = true
}
