// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU clip stack — elements with device-space bounds reasoning, save records with element invalidation/restoration,
// and the apply() strategy that converts the active elements into scissor state, analytic coverage FPs, stencil masks,
// or cached software masks. Trims: clip shaders have no entry point, so the SaveRecord shader lanes are dropped; window
// rectangles are dropped with the desktop trim (difference elements that would otherwise use window rects instead take
// the FP/mask lanes instead); the atlas-path-renderer clip lane is trimmed with that renderer; the threaded SW mask
// render reduces to a synchronous render on the recording goroutine; the path-genID containment fast path falls through
// to the conservative geometric test (the path has no generation ID).

package gl

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// ClipStackState summarizes the shape of the current clip stack.
type ClipStackState uint8

// ClipStackState values.
const (
	ClipStackEmpty ClipStackState = iota
	ClipStackWideOpen
	ClipStackDeviceRect
	ClipStackDeviceRRect
	ClipStackComplex
)

// ClipElement holds all data describing a geometric modification to the clip.
type ClipElement struct {
	Shape         Shape
	LocalToDevice geom.Matrix
	Op            raster.ClipOp
	AA            gpu.AA
}

// Reserved generation IDs: invalid, empty, and wide-open.
const (
	clipInvalidGenID  uint32 = 0
	clipEmptyGenID    uint32 = 1
	clipWideOpenGenID uint32 = 2
)

var nextClipGenID atomic.Uint32

// nextGenID returns the next clip generation ID; 0-2 are reserved for invalid, empty & wide-open.
func nextGenID() uint32 {
	for {
		if id := nextClipGenID.Add(1); id >= 3 {
			return id
		}
	}
}

// clipGeometry reports which of the two operands in (A op B) remain required when they are combined.
type clipGeometry int32

const (
	clipGeometryEmpty clipGeometry = iota
	clipGeometryAOnly
	clipGeometryBOnly
	clipGeometryBoth
)

// clipGeometryOperand is the shared surface getClipGeometry relies on. A and B can be a raw element, a save record, or
// a draw.
type clipGeometryOperand interface {
	geoOp() raster.ClipOp
	geoOuterBounds() geom.IRect
	// geoContains reports whether this operand's full-coverage (intersect) or zero-coverage (difference) region
	// contains the other operand, possibly just conservatively.
	geoContains(other clipGeometryOperand) bool
}

// getClipGeometry computes how operands a and b combine. NOTE: geom.IRect.Intersects returns false when two rectangles
// touch at an edge (the result is empty), which is the desired policy here.
func getClipGeometry(a, b clipGeometryOperand) clipGeometry {
	if a.geoOp() == raster.ClipIntersect {
		if b.geoOp() == raster.ClipIntersect {
			// Intersect (A) + Intersect (B)
			switch {
			case !a.geoOuterBounds().Intersects(b.geoOuterBounds()):
				// Regions with non-zero coverage are disjoint, so intersection = empty.
				return clipGeometryEmpty
			case b.geoContains(a):
				// B's full coverage region contains the entirety of A, so intersection = A.
				return clipGeometryAOnly
			case a.geoContains(b):
				// A's full coverage region contains the entirety of B, so intersection = B.
				return clipGeometryBOnly
			default:
				// The shapes intersect in some non-trivial manner.
				return clipGeometryBoth
			}
		}
		// Intersect (A) + Difference (B)
		switch {
		case !a.geoOuterBounds().Intersects(b.geoOuterBounds()):
			// A only intersects B's full coverage region, so intersection = A.
			return clipGeometryAOnly
		case b.geoContains(a):
			// B's zero coverage region completely contains A, so intersection = empty.
			return clipGeometryEmpty
		default:
			// Note that this op combination cannot produce clipGeometryBOnly.
			return clipGeometryBoth
		}
	}
	if b.geoOp() == raster.ClipIntersect {
		// Difference (A) + Intersect (B) — the mirror of Intersect(A) + Difference(B).
		switch {
		case !b.geoOuterBounds().Intersects(a.geoOuterBounds()):
			// B only intersects A's full coverage region, so intersection = B.
			return clipGeometryBOnly
		case a.geoContains(b):
			// A's zero coverage region completely contains B, so intersection = empty.
			return clipGeometryEmpty
		default:
			return clipGeometryBoth
		}
	}
	// Difference (A) + Difference (B)
	switch {
	case a.geoContains(b):
		// A's zero coverage region contains B, so B doesn't remove any extra coverage from their intersection.
		return clipGeometryAOnly
	case b.geoContains(a):
		// Mirror of the above case, intersection = B instead.
		return clipGeometryBOnly
	default:
		// It is not possible to produce clipGeometryEmpty for this op combination.
		return clipGeometryBoth
	}
}

// getClipGeometryVsDraw computes getClipGeometry with the draw as operand B, taking the draw by value so
// ClipStack.Apply never has to take its address. Boxing &draw into the clipGeometryOperand interface forces the draw
// onto the heap, and Go's escape analysis is flow-insensitive, so calling getClipGeometry(a, &draw) directly in Apply
// would heap-allocate the draw on every draw — including the common wide-open/empty clip early-outs (an unclipped
// canvas) that return before any geometry analysis. Confining the &-of-draw to this helper's own by-value copy keeps
// Apply's draw on the stack for those early-outs; the per-call copy here only happens on the complex-clip path that
// genuinely compares element geometry. It is marked noinline so the compiler cannot fold the &draw back into Apply's
// frame and re-escape Apply's draw.
//
//go:noinline
func getClipGeometryVsDraw(a clipGeometryOperand, draw clipStackDraw) clipGeometry {
	return getClipGeometry(a, &draw)
}

// shapeContainsRect reports a.contains(b) where a's local space is defined by aToDevice and b's by bToDevice. When the
// anti-aliasing policies differ (mixedAAMode), B is outset in device space first.
func shapeContainsRect(a *Shape, aToDevice, deviceToA *geom.Matrix, b geom.Rect, bToDevice *geom.Matrix, mixedAAMode bool) bool {
	if !a.Convex(true) {
		return false
	}

	if !mixedAAMode && *aToDevice == *bToDevice {
		// A and B are in the same coordinate space, so don't bother mapping.
		return a.ConservativeContainsRect(b)
	} else if bToDevice.IsIdentity() && aToDevice.RectStaysRect() {
		// Optimize the common case of draws (B, with identity matrix) and axis-aligned shapes, instead of checking the
		// four corners separately.
		bInA := b
		if mixedAAMode {
			bInA = bInA.Outset(0.5, 0.5)
		}
		bInA, _ = deviceToA.MapRect(bInA)
		return a.ConservativeContainsRect(bInA)
	}

	// Test each corner for contains; since a is convex, if all 4 corners of b's bounds are contained, then the entirety
	// of b is within a.
	deviceQuad := MakeQuadFromRect(b, bToDevice)
	if mixedAAMode {
		// Outset it so its edges are 1/2px out, giving a buffer against cases where a non-AA clip or draw would snap
		// outside an AA element.
		OutsetQuad([4]float32{0.5, 0.5, 0.5, 0.5}, &deviceQuad)
	}
	for _, w := range deviceQuad.W4f() {
		if w < quadW0PlaneDistance {
			// Something in B actually projects behind the W = 0 plane and would be clipped to infinity, so it's
			// extremely unlikely that A can contain B.
			return false
		}
	}
	for i := 0; i < 4; i++ {
		cornerInA := deviceToA.MapPoint(deviceQuad.Point(i))
		if !a.ConservativeContainsPoint(cornerInA) {
			return false
		}
	}
	return true
}

// subtractRects returns either A-B exactly, or (when not exactly representable and exact is required) the original A.
func subtractRects(a, b geom.IRect, exact bool) geom.IRect {
	diff, wasExact := rectSubtract(a, b)
	if wasExact || !exact {
		return diff
	}
	return a
}

// rectSubtract returns the largest rectangular subregion of a disjoint from b, reporting whether it is exactly a - b.
func rectSubtract(a, b geom.IRect) (geom.IRect, bool) {
	if a.IsEmpty() || b.IsEmpty() || !a.Intersects(b) {
		// Either already empty, or subtracting the empty rect, or no intersection: the answer is A in all cases.
		return a, true
	}

	// Four candidate subrectangles (left/right/top/bottom parts of A). Compute their relative areas; each shares A's
	// width or height, so dividing by the other dimension avoids overflow.
	aHeight := float32(a.Height())
	aWidth := float32(a.Width())
	var leftArea, rightArea, topArea, bottomArea float32
	positiveCount := 0
	if b.Left > a.Left {
		leftArea = float32(b.Left-a.Left) / aWidth
		positiveCount++
	}
	if a.Right > b.Right {
		rightArea = float32(a.Right-b.Right) / aWidth
		positiveCount++
	}
	if b.Top > a.Top {
		topArea = float32(b.Top-a.Top) / aHeight
		positiveCount++
	}
	if a.Bottom > b.Bottom {
		bottomArea = float32(a.Bottom-b.Bottom) / aHeight
		positiveCount++
	}

	if positiveCount == 0 {
		// B contains A.
		return geom.IRect{}, true
	}

	out := a
	switch {
	case leftArea > rightArea && leftArea > topArea && leftArea > bottomArea:
		out.Right = b.Left
	case rightArea > topArea && rightArea > bottomArea:
		out.Left = b.Right
	case topArea > bottomArea:
		out.Bottom = b.Top
	default:
		out.Top = b.Bottom
	}
	// If there is exactly one positive area, the disjoint shape is representable as a rectangle.
	return out, positiveCount == 1
}

// getClipEdgeType maps a clip op and AA setting to the corresponding ClipEdgeType.
func getClipEdgeType(op raster.ClipOp, aa gpu.AA) ClipEdgeType {
	if op == raster.ClipIntersect {
		if aa == gpu.AAYes {
			return ClipEdgeFillAA
		}
		return ClipEdgeFillBW
	}
	if aa == gpu.AAYes {
		return ClipEdgeInverseFillAA
	}
	return ClipEdgeInverseFillBW
}

// analyticClipFP represents the element as an analytic device-space coverage FP, if possible.
func analyticClipFP(e *ClipElement, caps *gpu.ShaderCaps, fp FragmentProcessor) (FragmentProcessor, bool) {
	// All analytic clip shape FPs need to be in device space.
	edgeType := getClipEdgeType(e.Op, e.AA)
	if e.LocalToDevice.IsIdentity() {
		if e.Shape.IsRect() {
			return RectClipFP(fp, edgeType, e.Shape.Rect()), true
		} else if e.Shape.IsRRect() {
			return RRectEffectFP(fp, edgeType, e.Shape.RRect(), caps)
		}
	}

	// A convex hull can be transformed into device space (this handles rect shapes with a non-identity transform).
	if e.Shape.SegmentMask() == path.SegmentLine && e.Shape.Convex(true) {
		devicePath := e.Shape.AsPath()
		devicePath.Transform(&e.LocalToDevice)
		return ConvexPolyEffectFP(fp, edgeType, devicePath)
	}

	return fp, false
}

// clipMaskOrigin is the coordinate origin used for generated clip mask surfaces.
const clipMaskOrigin = gpu.OriginTopLeft

// drawToSWMask renders one element into the helper with replace semantics. If the first element is an intersect, the
// mask is cleared to 0 and the element draws coverage 1 (subsequent intersects invert and draw 0 outside); a leading
// difference clears to 1 and always draws 0.
func drawToSWMask(helper *swMaskHelper, e *ClipElement, clearMask bool) {
	if clearMask {
		if e.Op == raster.ClipIntersect {
			helper.clear(0x00)
		} else {
			helper.clear(0xFF)
		}
	}

	var alpha uint8
	var invert bool
	if e.Op == raster.ClipIntersect {
		if clearMask {
			alpha = 0xFF
			invert = false
		} else {
			alpha = 0x00
			invert = true
		}
	} else {
		alpha = 0x00
		invert = false
	}

	if invert {
		// Draw the inverse-filled version of the shape with 0 coverage to erase everything outside the element. The
		// shape must be cloned rather than copied: a struct copy shares the path member with the live clip element, and
		// Shape.SetInverted toggles that path's fill type in place, which would leave the element itself inverted.
		inverted := cloneShape(&e.Shape)
		inverted.SetInverted(true)
		helper.drawShape(inverted, &e.LocalToDevice, e.AA, alpha)
	} else {
		helper.drawShape(&e.Shape, &e.LocalToDevice, e.AA, alpha)
	}
}

// renderSWMask, reduced to the synchronous lane (threaded render + deferred upload is trimmed), rasterizes the elements
// into an A8 mask and wraps it in a texture proxy view.
func renderSWMask(ctx *DirectContext, bounds geom.IRect, elements []*ClipElement) SurfaceProxyView {
	if len(elements) == 0 {
		panic("renderSWMask requires elements")
	}
	var helper swMaskHelper
	if !helper.init(bounds) {
		return SurfaceProxyView{}
	}
	for i, e := range elements {
		drawToSWMask(&helper, e, i == 0)
	}
	return helper.toTextureView(ctx)
}

// renderStencilMask rasterizes the elements into the stencil buffer and records the stencil clip on out.
func renderStencilMask(sdc *SurfaceDrawContext, genID uint32, bounds geom.IRect, elements []*ClipElement, out *AppliedClip) {
	helper := NewStencilMaskHelper(sdc)
	if helper.Init(bounds, genID, 0) {
		// This follows the same logic as in drawToSWMask.
		startInside := elements[0].Op == raster.ClipDifference
		helper.Clear(startInside)
		for i, e := range elements {
			var op raster.RegionOp
			if e.Op == raster.ClipIntersect {
				if i == 0 {
					op = raster.RegionReplace
				} else {
					op = raster.RegionIntersect
				}
			} else {
				op = raster.RegionDifference
			}
			helper.DrawShape(&e.Shape, &e.LocalToDevice, op, e.AA)
		}
		helper.Finish()
	}
	out.HardClip().AddStencilClip(genID)
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack draw query

// clipStackDraw is a draw query; it fits the same reasoning shape as an element with an implicit intersect op and empty
// inner bounds.
type clipStackDraw struct {
	originalBounds geom.Rect
	bounds         geom.IRect
	aa             gpu.AA
}

func makeClipStackDraw(drawBounds geom.Rect, aa gpu.AA) clipStackDraw {
	d := clipStackDraw{
		bounds: GetPixelIBounds(drawBounds, aa, ClipBoundsExterior),
		aa:     aa,
	}
	// Be slightly more forgiving on whether or not a draw is inside a clip element.
	d.originalBounds = drawBounds.Inset(clipBoundsTolerance, clipBoundsTolerance)
	if d.originalBounds.IsEmpty() {
		d.originalBounds = drawBounds
	}
	return d
}

func (d *clipStackDraw) geoOp() raster.ClipOp       { return raster.ClipIntersect }
func (d *clipStackDraw) geoOuterBounds() geom.IRect { return d.bounds }
func (d *clipStackDraw) geoContains(clipGeometryOperand) bool {
	// A draw does not have inner bounds, so it cannot contain anything.
	return false
}

func (d *clipStackDraw) applyDeviceBounds(deviceBounds geom.IRect) bool {
	return d.bounds.Intersect(deviceBounds)
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack raw element

// clipStackRawElement wraps the element data with containment and bounds testing, plus the invalidation bookkeeping.
type clipStackRawElement struct {
	ClipElement

	// Cached inverse of LocalToDevice for contains() optimization.
	deviceToLocal geom.Matrix

	// Device-space bounds, rounded in or out to pixel boundaries, accounting for AA and rasterization-snapping
	// uncertainty.
	innerBounds geom.IRect
	outerBounds geom.IRect

	// Elements are invalidated by save records as the record is updated with new elements that override old geometry.
	// An invalidated element stores the index of the first element of the save record that invalidated it, making it
	// easy to undo when the record is popped.
	invalidatedByIndex int
}

// newClipStackRawElement creates a raw element wrapping shape's clip contribution.
func newClipStackRawElement(localToDevice *geom.Matrix, shape *Shape, aa gpu.AA, op raster.ClipOp) clipStackRawElement {
	e := clipStackRawElement{
		ClipElement:        ClipElement{Shape: *shape, LocalToDevice: *localToDevice, Op: op, AA: aa},
		invalidatedByIndex: -1,
	}
	if inv, ok := localToDevice.Invert(); ok {
		e.deviceToLocal = inv
	} else {
		// If the transform can't be inverted, it means that two dimensions are collapsed to 0 or 1 dimensions, making
		// the device-space geometry effectively empty.
		e.Shape.Reset()
	}
	return e
}

func (e *clipStackRawElement) geoOp() raster.ClipOp       { return e.Op }
func (e *clipStackRawElement) geoOuterBounds() geom.IRect { return e.outerBounds }

func (e *clipStackRawElement) geoContains(other clipGeometryOperand) bool {
	switch o := other.(type) {
	case *clipStackDraw:
		return e.containsDraw(o)
	case *clipStackSaveRecord:
		return e.containsSave(o)
	case *clipStackRawElement:
		return e.containsElement(o)
	default:
		panic("unknown clip geometry operand")
	}
}

func (e *clipStackRawElement) isInvalid() bool { return e.invalidatedByIndex >= 0 }

func (e *clipStackRawElement) markInvalid(current *clipStackSaveRecord) {
	e.invalidatedByIndex = current.firstActiveElementIndex()
}

func (e *clipStackRawElement) restoreValid(current *clipStackSaveRecord) {
	if current.firstActiveElementIndex() < e.invalidatedByIndex {
		e.invalidatedByIndex = -1
	}
}

// containsDraw reports whether this element's full-coverage region contains d.
func (e *clipStackRawElement) containsDraw(d *clipStackDraw) bool {
	if e.innerBounds.ContainsRect(d.bounds) {
		return true
	}
	// If the draw is non-AA, use the already computed outer bounds so we don't need to use device-space outsetting
	// inside shapeContainsRect.
	queryBounds := d.originalBounds
	if d.aa != gpu.AAYes {
		queryBounds = d.bounds.ToRect()
	}
	identity := geom.IdentityMatrix()
	return shapeContainsRect(&e.Shape, &e.LocalToDevice, &e.deviceToLocal, queryBounds,
		&identity, false /* mixed-aa */)
}

// containsSave reports whether this element's full-coverage region contains s.
func (e *clipStackRawElement) containsSave(s *clipStackSaveRecord) bool {
	if e.innerBounds.ContainsRect(s.outerBounds) {
		return true
	}
	identity := geom.IdentityMatrix()
	return shapeContainsRect(&e.Shape, &e.LocalToDevice, &e.deviceToLocal,
		s.outerBounds.ToRect(), &identity, false /* mixed-aa */)
}

// containsElement reports whether this element's full-coverage region contains o.
func (e *clipStackRawElement) containsElement(o *clipStackRawElement) bool {
	// Similar to containsDraw, except that both the tester and testee have transforms.
	if e.innerBounds.ContainsRect(o.outerBounds) {
		return true
	}

	mixedAA := e.AA != o.AA
	if !mixedAA && e.LocalToDevice == o.LocalToDevice {
		// Test the shapes directly against each other with the rrect+rrect containment lane (a intersect b == a implies
		// b contains a). The path-genID / small-path-equality lane falls through to the conservative geometric test (no
		// path generation IDs).
		if e.Shape.IsRRect() && o.Shape.IsRRect() {
			return rrectsEqual(rrectConservativeIntersect(e.Shape.RRect(), o.Shape.RRect()),
				o.Shape.RRect())
		}
	}

	return shapeContainsRect(&e.Shape, &e.LocalToDevice, &e.deviceToLocal, o.Shape.Bounds(),
		&o.LocalToDevice, mixedAA)
}

// simplify reduces this element to its simplest representable form and recomputes its bounds.
func (e *clipStackRawElement) simplify(deviceBounds geom.IRect, forceAA bool) {
	// An inverted shape is equivalent to a non-inverted shape with the clip op toggled.
	if e.Shape.Inverted() {
		if e.Op == raster.ClipIntersect {
			e.Op = raster.ClipDifference
		} else {
			e.Op = raster.ClipIntersect
		}
		e.Shape.SetInverted(false)
	}

	// Then simplify the base shape; if it becomes empty, no need to update the bounds.
	e.Shape.Simplify(ShapeSimplifyAll)
	if e.Shape.Inverted() {
		panic("simplify must leave the shape non-inverted")
	}
	if e.Shape.IsEmpty() {
		return
	}
	// Lines and points should have been turned into empty since everything is filled.
	if e.Shape.IsPoint() || e.Shape.IsLine() {
		panic("point/line shapes should have simplified to empty")
	}

	outer, _ := e.LocalToDevice.MapRect(e.Shape.Bounds())
	if !outer.Intersect(deviceBounds.ToRect()) {
		// A non-empty shape is offscreen, so treat it as empty.
		e.Shape.Reset()
		return
	}

	// Except for axis-aligned clip rects, upgrade to AA when forced. Axis-aligned clip rects are skipped because a
	// non-AA axis-aligned rect can always be set as just a scissor test, avoiding an expensive stencil mask generation.
	if forceAA && (!e.Shape.IsRect() || !e.LocalToDevice.RectStaysRect()) {
		e.AA = gpu.AAYes
	}

	// Except for non-AA axis-aligned rects, the outer bounds is the rounded-out device-space mapped bounds of the
	// shape.
	e.outerBounds = GetPixelIBounds(outer, e.AA, ClipBoundsExterior)

	if e.LocalToDevice.RectStaysRect() {
		if e.Shape.IsRect() {
			// The actual geometry can be updated to the device-intersected bounds and we know the inner bounds.
			e.Shape.SetRect(outer)
			e.LocalToDevice = geom.IdentityMatrix()
			e.deviceToLocal = geom.IdentityMatrix()

			if e.AA == gpu.AANo && outer.Width() >= 1 && outer.Height() >= 1 {
				// NOTE: legacy behavior to avoid performance regressions. For non-AA axis-aligned clip rects, always
				// round so they can be scissor-only (avoiding uncertainty in how a GPU rounds an edge on fractional
				// coords).
				e.outerBounds = outer.Round()
				e.innerBounds = e.outerBounds
			} else {
				e.innerBounds = GetPixelIBounds(outer, e.AA, ClipBoundsInterior)
				if !e.innerBounds.IsEmpty() && !e.outerBounds.ContainsRect(e.innerBounds) {
					panic("inner bounds must be inside outer bounds")
				}
			}
		} else if e.Shape.IsRRect() {
			// Can't transform in place and must still check the transform result since some very ill-formed
			// scale+translate matrices can cause invalid rrect radii.
			if rr, ok := e.Shape.RRect().Transform(&e.LocalToDevice); ok {
				e.Shape.SetRRect(rr)
				e.LocalToDevice = geom.IdentityMatrix()
				e.deviceToLocal = geom.IdentityMatrix()

				inner := rrectInnerBounds(rr)
				e.innerBounds = GetPixelIBounds(inner, e.AA, ClipBoundsInterior)
				if !e.innerBounds.Intersect(deviceBounds) {
					e.innerBounds = geom.IRect{}
				}
			}
		}
	}

	if e.outerBounds.IsEmpty() {
		// This can happen if we have non-AA shapes smaller than a pixel that do not cover a pixel center; rasterization
		// would still result in an empty clip.
		e.Shape.Reset()
	}
}

// combine tries to reduce this element and 'other' to a single new element stored in e (returning true when e now
// represents the combination or the combination proved empty).
func (e *clipStackRawElement) combine(other *clipStackRawElement, current *clipStackSaveRecord) bool {
	// Only consider intersect+intersect. Difference and mixed op cases could be analyzed to simplify one of the shapes,
	// but that is rare and the math is much more complicated.
	if other.Op != raster.ClipIntersect || e.Op != raster.ClipIntersect {
		return false
	}

	// Only rect+rect or rrect+rrect combinations (with rect+rrect handled as a degenerate rrect+rrect) are supported.
	shapeUpdated := false
	switch {
	case e.Shape.IsRect() && other.Shape.IsRect():
		aaMatch := e.AA == other.AA
		if e.LocalToDevice.IsIdentity() && other.LocalToDevice.IsIdentity() && !aaMatch {
			switch {
			case IsPixelAligned(e.Shape.Rect()):
				// Our AA type doesn't really matter, take other's since its edges may not be pixel aligned, so after
				// intersection clip behavior should respect its type.
				e.AA = other.AA
			case !IsPixelAligned(other.Shape.Rect()):
				// Neither shape is pixel aligned and the AA types don't match, so can't combine.
				return false
			}
			// Either e.AA now actually matches, or other's AA doesn't matter. 'other' will be invalidated so its AA
			// can't be modified; e becomes the combination.
			aaMatch = true
		}

		if aaMatch && e.LocalToDevice == other.LocalToDevice {
			r := e.Shape.Rect()
			if !r.Intersect(other.Shape.Rect()) {
				// By floating point, it turns out the combination should be empty.
				e.Shape.Reset()
				e.markInvalid(current)
				return true
			}
			e.Shape.SetRect(r)
			shapeUpdated = true
		}

	case (e.Shape.IsRect() || e.Shape.IsRRect()) &&
		(other.Shape.IsRect() || other.Shape.IsRRect()):
		// No such pixel-aligned disregard for AA for round rects.
		if e.AA == other.AA && e.LocalToDevice == other.LocalToDevice {
			// Treat rrect+rect intersections as rrect+rrect.
			var a, b geom.RRect
			if e.Shape.IsRect() {
				a = geom.MakeRRect(e.Shape.Rect(), 0, 0)
			} else {
				a = e.Shape.RRect()
			}
			if other.Shape.IsRect() {
				b = geom.MakeRRect(other.Shape.Rect(), 0, 0)
			} else {
				b = other.Shape.RRect()
			}

			joined := rrectConservativeIntersect(a, b)
			if !joined.IsEmpty() {
				// Can reduce to a single element.
				if joined.Type == geom.RRectRect {
					// And with a simplified type.
					e.Shape.SetRect(joined.Rect)
				} else {
					e.Shape.SetRRect(joined)
				}
				shapeUpdated = true
			} else if !a.Rect.Intersects(b.Rect) {
				// Like the rect+rect combination, the intersection is actually empty.
				e.Shape.Reset()
				e.markInvalid(current)
				return true
			}
		}
	}

	if shapeUpdated {
		// This logic works under the assumption that both combined elements were intersect, so the full bounds
		// computations of simplify() are not needed.
		if !e.outerBounds.Intersect(other.outerBounds) {
			panic("combined intersect elements must have intersecting outer bounds")
		}
		if !e.innerBounds.Intersect(other.innerBounds) {
			e.innerBounds = geom.IRect{}
		}
		return true
	}
	return false
}

// updateForElement handles 'added', a new element added to the stack; the combination may invalidate either element or
// merge them into 'added'.
func (e *clipStackRawElement) updateForElement(added *clipStackRawElement, current *clipStackSaveRecord) {
	if e.isInvalid() {
		// Already doesn't do anything, so skip.
		return
	}

	// 'A' refers to this element, 'B' refers to 'added'.
	switch getClipGeometry(e, added) {
	case clipGeometryEmpty:
		// Mark both elements as invalid to signal that the clip is fully empty.
		e.markInvalid(current)
		added.markInvalid(current)
	case clipGeometryAOnly:
		// This element already clips more than 'added', so mark 'added' invalid to skip it.
		added.markInvalid(current)
	case clipGeometryBOnly:
		// 'added' clips more than this element, so mark this as invalid.
		e.markInvalid(current)
	case clipGeometryBoth:
		// The bounds checks think we need to keep both, but depending on the combination of ops and shape kinds we may
		// be able to do better.
		if added.combine(e, current) {
			// 'added' now fully represents the combination of the two elements.
			e.markInvalid(current)
		}
	}
}

// clipType classifies this element's contribution to the overall ClipStackState.
func (e *clipStackRawElement) clipType() ClipStackState {
	switch e.Shape.Type() {
	case ShapeTypeEmpty:
		return ClipStackEmpty
	case ShapeTypeRect:
		if e.Op == raster.ClipIntersect && e.LocalToDevice.IsIdentity() {
			return ClipStackDeviceRect
		}
		return ClipStackComplex
	case ShapeTypeRRect:
		if e.Op == raster.ClipIntersect && e.LocalToDevice.IsIdentity() {
			return ClipStackDeviceRRect
		}
		return ClipStackComplex
	case ShapeTypePath:
		return ClipStackComplex
	default:
		// Point/line types should never become raw elements.
		panic("unexpected shape type for a clip element")
	}
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack mask

// clipMaskKeyDomain is the unique-key domain generated for clip masks.
var clipMaskKeyDomain = gpu.GenerateUniqueKeyDomain()

// clipStackMask is an alpha mask with the rasterized coverage from elements in a draw query that could not be converted
// to analytic coverage FPs.
type clipStackMask struct {
	key gpu.UniqueKey
	// The gen ID of the save record and the query bounds uniquely define the set of elements that would go into a mask.
	bounds geom.IRect
	genID  uint32
}

// newClipStackMask creates a clipStackMask keyed by the save record's generation and drawBounds.
func newClipStackMask(current *clipStackSaveRecord, drawBounds geom.IRect) clipStackMask {
	m := clipStackMask{bounds: drawBounds, genID: current.genID()}
	if m.genID == clipInvalidGenID || m.genID == clipEmptyGenID ||
		m.genID == clipWideOpenGenID {
		panic("mask gen ID must be a real save record gen ID")
	}
	builder := gpu.UniqueKeyBuilder(&m.key, clipMaskKeyDomain, 5, "clip_mask")
	data := builder.Slice()
	data[0] = m.genID
	data[1] = uint32(drawBounds.Left)
	data[2] = uint32(drawBounds.Right)
	data[3] = uint32(drawBounds.Top)
	data[4] = uint32(drawBounds.Bottom)
	builder.Finish()
	return m
}

// appliesToDraw reports whether this mask can clip drawBounds: for the same save record, a larger mask has the same or
// more elements baked into it, so it can be reused to clip a smaller draw.
func (m *clipStackMask) appliesToDraw(current *clipStackSaveRecord, drawBounds geom.IRect) bool {
	return m.genID == current.genID() && m.bounds.ContainsRect(drawBounds)
}

// invalidate evicts this mask's cached GPU resource.
func (m *clipStackMask) invalidate(proxyProvider *ProxyProvider) {
	if !m.key.IsValid() {
		panic("mask should only be invalidated once")
	}
	proxyProvider.ProcessInvalidUniqueKey(&m.key, nil, InvalidateGPUResourceYes)
	m.key.Reset()
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack save record

// clipStackSaveRecord is a snapshot of the clip stack's aggregate bounds and state at a save point.
type clipStackSaveRecord struct {
	// Inner bounds is always contained in outer bounds, or empty. All bounds are contained in the device bounds. Inside
	// inner bounds is full coverage (stack op == intersect) or 0 coverage (difference); outside outer bounds is 0
	// coverage (intersect) or full (diff).
	innerBounds geom.IRect
	outerBounds geom.IRect

	startingMaskIndex    int
	startingElementIndex int
	oldestValidIndex     int

	deferredSaveCount int

	// stackOp is raster.ClipIntersect unless every valid element is raster.ClipDifference, which is significant
	// because then there is an implicit extra outer bounds at the device edges.
	stackOp raster.ClipOp
	state   ClipStackState
	genIDv  uint32
}

// newClipStackSaveRecord creates the initial, wide-open save record covering deviceBounds.
func newClipStackSaveRecord(deviceBounds geom.IRect) clipStackSaveRecord {
	return clipStackSaveRecord{
		innerBounds: deviceBounds,
		outerBounds: deviceBounds,
		stackOp:     raster.ClipIntersect,
		state:       ClipStackWideOpen,
		genIDv:      clipInvalidGenID,
	}
}

// duplicateSaveRecord creates a new save record copying prior's bounds/state, starting fresh element/mask indices.
func duplicateSaveRecord(prior *clipStackSaveRecord, startingMaskIndex, startingElementIndex int) clipStackSaveRecord {
	return clipStackSaveRecord{
		innerBounds:          prior.innerBounds,
		outerBounds:          prior.outerBounds,
		startingMaskIndex:    startingMaskIndex,
		startingElementIndex: startingElementIndex,
		oldestValidIndex:     prior.oldestValidIndex,
		stackOp:              prior.stackOp,
		state:                prior.state,
		genIDv:               clipInvalidGenID,
	}
}

func (s *clipStackSaveRecord) geoOp() raster.ClipOp       { return s.stackOp }
func (s *clipStackSaveRecord) geoOuterBounds() geom.IRect { return s.outerBounds }

func (s *clipStackSaveRecord) geoContains(other clipGeometryOperand) bool {
	// Both draw-bounds and raw-element containment reduce to the same inner-bounds test.
	return s.innerBounds.ContainsRect(other.geoOuterBounds())
}

func (s *clipStackSaveRecord) firstActiveElementIndex() int { return s.startingElementIndex }
func (s *clipStackSaveRecord) oldestElementIndex() int      { return s.oldestValidIndex }
func (s *clipStackSaveRecord) canBeUpdated() bool           { return s.deferredSaveCount == 0 }

// genID returns this save record's generation ID (empty/wide-open sentinels for those states).
func (s *clipStackSaveRecord) genID() uint32 {
	switch s.state {
	case ClipStackEmpty:
		return clipEmptyGenID
	case ClipStackWideOpen:
		return clipWideOpenGenID
	default:
		// May be clipInvalidGenID if the record hasn't had any elements added to it yet.
		return s.genIDv
	}
}

// recordState returns this save record's ClipStackState (the shader term is trimmed with clip shaders).
func (s *clipStackSaveRecord) recordState() ClipStackState { return s.state }

// removeElements truncates elements back to this save record's starting index.
func (s *clipStackSaveRecord) removeElements(elements *[]clipStackRawElement) {
	*elements = (*elements)[:s.startingElementIndex]
}

// restoreElements re-validates elements now that this record is the new top of the stack; elements from the removed
// save record have been destroyed already, so any still-present element can be re-validated.
func (s *clipStackSaveRecord) restoreElements(elements *[]clipStackRawElement) {
	for i := len(*elements) - 1; i >= s.oldestValidIndex; i-- {
		(*elements)[i].restoreValid(s)
	}
}

// invalidateMasks evicts every mask registered since this save record's starting index.
func (s *clipStackSaveRecord) invalidateMasks(proxyProvider *ProxyProvider, masks *[]clipStackMask) {
	for len(*masks) > s.startingMaskIndex {
		m := &(*masks)[len(*masks)-1]
		if proxyProvider == nil {
			panic("masks require a proxy provider")
		}
		m.invalidate(proxyProvider)
		*masks = (*masks)[:len(*masks)-1]
	}
}

// reset reinitializes this save record to a wide-open clip over bounds (for ReplaceClip).
func (s *clipStackSaveRecord) reset(bounds geom.IRect) {
	if !s.canBeUpdated() {
		panic("reset on a deferred save record")
	}
	s.oldestValidIndex = s.startingElementIndex
	s.outerBounds = bounds
	s.innerBounds = bounds
	s.stackOp = raster.ClipIntersect
	s.state = ClipStackWideOpen
}

// addElement returns true if the element was added to 'elements' or otherwise affected the save record (e.g. turned it
// empty).
func (s *clipStackSaveRecord) addElement(toAdd *clipStackRawElement, elements *[]clipStackRawElement) bool {
	// Validity check the element's state first.
	if !toAdd.Shape.IsEmpty() && toAdd.outerBounds.IsEmpty() {
		panic("non-empty shape must have outer bounds")
	}
	if !toAdd.innerBounds.IsEmpty() && !toAdd.outerBounds.ContainsRect(toAdd.innerBounds) {
		panic("inner bounds must be inside outer bounds")
	}
	if !s.canBeUpdated() {
		panic("cannot add an element with a deferred save")
	}

	if s.state == ClipStackEmpty {
		// The clip is already empty and can only shrink, so don't record this element.
		return false
	} else if toAdd.Shape.IsEmpty() {
		// An empty difference op should have been detected earlier, since it's a no-op.
		if toAdd.Op != raster.ClipIntersect {
			panic("empty difference elements are no-ops")
		}
		s.state = ClipStackEmpty
		return true
	}

	// 'A' refers to the existing stack's bounds and 'B' refers to the new element.
	switch getClipGeometry(s, toAdd) {
	case clipGeometryEmpty:
		// The combination results in an empty clip.
		s.state = ClipStackEmpty
		return true
	case clipGeometryAOnly:
		// The combination would not be any different than the existing clip.
		return false
	case clipGeometryBOnly:
		// The combination invalidates the entire existing stack and is replaced with just the new element.
		s.replaceWithElement(toAdd, elements)
		return true
	case clipGeometryBoth:
		// The new element combines in a complex manner; update the stack's bounds (below).
	}

	if s.state == ClipStackWideOpen {
		// When the stack was wide open and the clip effect was kBoth, the "complex" manner is simply to keep the
		// element and update the stack bounds to be the element's intersected with the device.
		s.replaceWithElement(toAdd, elements)
		return true
	}

	// Some form of actual clip element(s) to combine with.
	if s.stackOp == raster.ClipIntersect {
		if toAdd.Op == raster.ClipIntersect {
			// Intersect (stack) + Intersect (toAdd): paired intersections of outer and inner.
			if !s.outerBounds.Intersect(toAdd.outerBounds) {
				panic("kBoth combination must have intersecting outer bounds")
			}
			if !s.innerBounds.Intersect(toAdd.innerBounds) {
				// NOTE: this does the right thing if either rect is empty.
				s.innerBounds = geom.IRect{}
			}
		} else {
			// Intersect (stack) + Difference (toAdd): shrink the outer bounds if the difference op's inner bounds
			// completely cuts off an edge; shrink the inner bounds to completely exclude the op's outer bounds.
			s.outerBounds = subtractRects(s.outerBounds, toAdd.innerBounds, true /* exact */)
			s.innerBounds = subtractRects(s.innerBounds, toAdd.outerBounds, false /* exact */)
		}
	} else {
		if toAdd.Op == raster.ClipIntersect {
			// Difference (stack) + Intersect (toAdd): the mirror of the above.
			oldOuter := s.outerBounds
			s.outerBounds = subtractRects(toAdd.outerBounds, s.innerBounds, true /* exact */)
			s.innerBounds = subtractRects(toAdd.innerBounds, oldOuter, false /* exact */)
		} else {
			// Difference (stack) + Difference (toAdd): the updated outer bounds is the union, and the inner becomes the
			// largest of the two possible inner bounds.
			s.outerBounds.Join(toAdd.outerBounds)
			if int64(toAdd.innerBounds.Width())*int64(toAdd.innerBounds.Height()) >
				int64(s.innerBounds.Width())*int64(s.innerBounds.Height()) {
				s.innerBounds = toAdd.innerBounds
			}
		}
	}

	// We are keeping the new element and the stack's bounds have been updated; the cases where the stack bounds
	// resemble an empty or wide-open clip were caught above.
	if s.outerBounds.IsEmpty() ||
		(!s.innerBounds.IsEmpty() && !s.outerBounds.ContainsRect(s.innerBounds)) {
		panic("stack bounds invariant violated")
	}

	return s.appendElement(toAdd, elements)
}

// appendElement folds toAdd into the element list, invalidating any elements it supersedes.
func (s *clipStackSaveRecord) appendElement(toAdd *clipStackRawElement, elements *[]clipStackRawElement) bool {
	// Update past elements to account for the new element.
	count := len(*elements)

	// After the loop, elements between [max(youngestValid, startingIndex)+1, count-1] can be removed from the stack
	// (they are active elements that were invalidated by the newest element; since it's the active part of the stack,
	// no restore() can bring them back).
	youngestValid := s.startingElementIndex - 1
	// After the loop, elements between [0, oldestValid-1] are all invalid; oldestValid becomes the record's new
	// oldestValidIndex.
	oldestValid := count
	// After the loop, this is the earliest active element that was invalidated. It may be older in the stack than
	// youngestValid, so it cannot be popped off, but it can store the new element instead of allocating more.
	oldestActiveInvalid := -1

	for i := count - 1; i >= s.oldestValidIndex; i-- {
		existing := &(*elements)[i]
		// The minimum index of this save record produces the same restoration behavior as the actual index toAdd will
		// be saved to.
		existing.updateForElement(toAdd, s)

		switch {
		case toAdd.isInvalid():
			if existing.isInvalid() {
				// Both new and old invalid implies the entire clip becomes empty.
				s.state = ClipStackEmpty
				return true
			}
			// The new element doesn't change the clip beyond what the old element already does.
			return false
		case existing.isInvalid():
			// The new element cancels out the old element. The new element may have been modified to account for the
			// old element's geometry.
			if i >= s.startingElementIndex {
				// Still active, so the invalidated index could store the new element.
				oldestActiveInvalid = i
			}
		default:
			// Keep both new and old elements.
			oldestValid = i
			if i > youngestValid {
				youngestValid = i
			}
		}
	}

	// Update final state.
	if oldestValid < s.oldestValidIndex {
		panic("oldest valid index must not move backwards")
	}
	oldestActiveInvalidIndex := oldestActiveInvalid
	if oldestActiveInvalid < 0 {
		oldestActiveInvalidIndex = len(*elements)
	}
	s.oldestValidIndex = min(oldestValid, oldestActiveInvalidIndex)
	if oldestValid == len(*elements) {
		s.state = toAdd.clipType()
	} else {
		s.state = ClipStackComplex
	}
	if s.stackOp == raster.ClipDifference && toAdd.Op == raster.ClipIntersect {
		// The stack remains in difference mode only as long as all elements are difference.
		s.stackOp = raster.ClipIntersect
	}

	targetCount := youngestValid + 1
	if oldestActiveInvalid < 0 || oldestActiveInvalid >= targetCount {
		// toAdd will be stored right after youngestValid.
		targetCount++
		oldestActiveInvalid = -1
	}
	for len(*elements) > targetCount {
		if oldestActiveInvalid == len(*elements)-1 {
			panic("cannot delete the element that will be reused")
		}
		*elements = (*elements)[:len(*elements)-1]
	}
	switch {
	case oldestActiveInvalid >= 0:
		(*elements)[oldestActiveInvalid] = *toAdd
	case len(*elements) < targetCount:
		*elements = append(*elements, *toAdd)
	default:
		(*elements)[len(*elements)-1] = *toAdd
	}

	// Changing the gen ID prompts the ClipStack to invalidate masks for this record.
	s.genIDv = nextGenID()
	return true
}

// replaceWithElement sets the aggregate state of the record to match the element, and invalidates all older elements.
func (s *clipStackSaveRecord) replaceWithElement(toAdd *clipStackRawElement, elements *[]clipStackRawElement) {
	s.innerBounds = toAdd.innerBounds
	s.outerBounds = toAdd.outerBounds
	s.stackOp = toAdd.Op
	s.state = toAdd.clipType()

	// All prior active elements can be removed from the stack.
	targetCount := s.startingElementIndex + 1
	for len(*elements) > targetCount {
		*elements = (*elements)[:len(*elements)-1]
	}
	if len(*elements) < targetCount {
		*elements = append(*elements, *toAdd)
	} else {
		(*elements)[targetCount-1] = *toAdd
	}

	// This invalidates all older elements owned by save records lower in the clip stack.
	s.oldestValidIndex = s.startingElementIndex
	s.genIDv = nextGenID()
}

//////////////////////////////////////////////////////////////////////////////
// ClipStack

// clipMaxAnalyticFPs limits how many analytic coverage FPs a single draw's clip may accumulate before falling back to a
// mask; drawn from prior draw-call analysis showing the most complex clip used 5, with the limit set at the historical
// 4.
const clipMaxAnalyticFPs = 4

// ClipStack implements Clip as a stack of clip elements with save/restore semantics.
type ClipStack struct {
	proxyProvider *ProxyProvider
	elements      []clipStackRawElement
	saves         []clipStackSaveRecord // always has one wide-open record at the top
	// Masks are recorded during Apply() calls so they can be cached; they are not modifications of the actual clip
	// stack.
	masks        []clipStackMask
	deviceBounds geom.IRect
	// When there's MSAA, clip elements are applied using the stencil buffer; if a backend cannot disable MSAA per draw
	// then all elements are effectively AA'ed.
	forceAA bool
}

// NewClipStack creates a ClipStack covering deviceBounds. (Clip shaders have no entry point, so that lane is trimmed.)
func NewClipStack(deviceBounds geom.IRect, forceAA bool) *ClipStack {
	cs := &ClipStack{deviceBounds: deviceBounds, forceAA: forceAA}
	// Start with a wide-open save record.
	cs.saves = append(cs.saves, newClipStackSaveRecord(deviceBounds))
	return cs
}

// currentSaveRecord returns the topmost (active) save record.
func (cs *ClipStack) currentSaveRecord() *clipStackSaveRecord {
	return &cs.saves[len(cs.saves)-1]
}

// ClipState reports the current clip's overall shape.
func (cs *ClipStack) ClipState() ClipStackState { return cs.currentSaveRecord().recordState() }

// AnyElementAA reports whether any active element is anti-aliased.
func (cs *ClipStack) AnyElementAA() bool {
	current := cs.currentSaveRecord()
	for i := current.oldestElementIndex(); i < len(cs.elements); i++ {
		e := &cs.elements[i]
		if !e.isInvalid() && e.AA == gpu.AAYes {
			return true
		}
	}
	return false
}

// Save defers a save of the current clip state.
func (cs *ClipStack) Save() {
	cs.currentSaveRecord().deferredSaveCount++
}

// Restore undoes the most recent unmatched Save, popping the save record if one was created.
func (cs *ClipStack) Restore() {
	current := cs.currentSaveRecord()
	current.deferredSaveCount--
	if current.deferredSaveCount >= 0 {
		// This was just a deferred save being undone; the record stays.
		return
	}

	// Removing a save record deletes all elements >= its starting index and any masks that were rasterized for it.
	current.removeElements(&cs.elements)
	if cs.proxyProvider == nil && len(cs.masks) > 0 {
		panic("masks without a proxy provider")
	}
	if cs.proxyProvider != nil {
		current.invalidateMasks(cs.proxyProvider, &cs.masks)
	}
	cs.saves = cs.saves[:len(cs.saves)-1]
	// Restore any remaining elements that were only invalidated by the removed save record.
	cs.currentSaveRecord().restoreElements(&cs.elements)
}

// writableSaveRecord returns the current save record, properly updating deferred saves.
func (cs *ClipStack) writableSaveRecord(wasDeferred *bool) *clipStackSaveRecord {
	current := cs.currentSaveRecord()
	if current.canBeUpdated() {
		// Still open, so it can be modified directly.
		*wasDeferred = false
		return current
	}
	// Must undefer the save to get a new record.
	current.deferredSaveCount--
	*wasDeferred = true
	cs.saves = append(cs.saves, duplicateSaveRecord(current, len(cs.masks), len(cs.elements)))
	return cs.currentSaveRecord()
}

// ClipRect intersects (or subtracts, per op) rect, mapped by ctm, from the clip.
func (cs *ClipStack) ClipRect(ctm *geom.Matrix, rect geom.Rect, aa gpu.AA, op raster.ClipOp) {
	shape := MakeShapeRect(rect)
	element := newClipStackRawElement(ctm, &shape, aa, op)
	cs.clip(&element)
}

// ClipRRect intersects (or subtracts, per op) rrect, mapped by ctm, from the clip.
func (cs *ClipStack) ClipRRect(ctm *geom.Matrix, rrect geom.RRect, aa gpu.AA, op raster.ClipOp) {
	shape := MakeShapeRRect(rrect)
	element := newClipStackRawElement(ctm, &shape, aa, op)
	cs.clip(&element)
}

// ClipPath intersects (or subtracts, per op) p, mapped by ctm, from the clip.
func (cs *ClipStack) ClipPath(ctm *geom.Matrix, p *path.Path, aa gpu.AA, op raster.ClipOp) {
	shape := MakeShapePath(p)
	element := newClipStackRawElement(ctm, &shape, aa, op)
	cs.clip(&element)
}

// ReplaceClip discards the current save record's elements and resets the clip to rect.
func (cs *ClipStack) ReplaceClip(rect geom.IRect) {
	var wasDeferred bool
	save := cs.writableSaveRecord(&wasDeferred)

	if !wasDeferred {
		save.removeElements(&cs.elements)
		if cs.proxyProvider != nil {
			save.invalidateMasks(cs.proxyProvider, &cs.masks)
		}
	}

	save.reset(cs.deviceBounds)
	if rect != cs.deviceBounds {
		identity := geom.IdentityMatrix()
		cs.ClipRect(&identity, rect.ToRect(), gpu.AANo, raster.ClipIntersect)
	}
}

// clip simplifies and adds element to the current save record.
func (cs *ClipStack) clip(element *clipStackRawElement) {
	if cs.currentSaveRecord().recordState() == ClipStackEmpty {
		return
	}

	// Reduce the path to anything simpler, applying the transform if it's a scale+translate and ensuring the element's
	// bounds are clipped to the device (NOT the conservative clip bounds, since those reflect the net effect of all
	// elements while device-bounds clipping happens implicitly; during addElement some older elements may still be
	// invalidated).
	element.simplify(cs.deviceBounds, cs.forceAA)
	if element.Shape.Inverted() {
		panic("clip elements must not be inverted after simplify")
	}

	// An empty op means do nothing (for difference) or close the save record, so detect that early to avoid unnecessary
	// save record allocation.
	if element.Shape.IsEmpty() && element.Op == raster.ClipDifference {
		// Subtracting an empty shape has no effect on the clip.
		return
	}

	var wasDeferred bool
	save := cs.writableSaveRecord(&wasDeferred)
	elementCount := len(cs.elements)
	if !save.addElement(element, &cs.elements) {
		if wasDeferred {
			// We made a new save record but ended up not adding an element, so pop it off and restore the deferred
			// counter instead of keeping an empty record around.
			if elementCount != len(cs.elements) {
				panic("element count must be unchanged when addElement fails")
			}
			cs.saves = cs.saves[:len(cs.saves)-1]
			cs.currentSaveRecord().deferredSaveCount++
		}
	} else if cs.proxyProvider != nil && !wasDeferred {
		// We modified an active save record so any old masks it had can be invalidated.
		save.invalidateMasks(cs.proxyProvider, &cs.masks)
	}
}

// GetConservativeBounds implements Clip.
func (cs *ClipStack) GetConservativeBounds() geom.IRect {
	current := cs.currentSaveRecord()
	switch current.recordState() {
	case ClipStackEmpty:
		return geom.IRect{}
	case ClipStackWideOpen:
		return cs.deviceBounds
	default:
		if current.stackOp == raster.ClipDifference {
			// The outer/inner bounds represent what's cut out, so the full bounds remain the device bounds, minus any
			// fully clipped content that spans the device edge.
			return subtractRects(cs.deviceBounds, current.innerBounds, true /* exact */)
		}
		if !cs.deviceBounds.ContainsRect(current.outerBounds) {
			panic("outer bounds must be inside the device bounds")
		}
		return current.outerBounds
	}
}

// PreApply implements Clip.
func (cs *ClipStack) PreApply(bounds geom.Rect, aa gpu.AA) PreClipResult {
	if cs.forceAA {
		aa = gpu.AAYes
	}
	draw := makeClipStackDraw(bounds, aa)
	if !draw.applyDeviceBounds(cs.deviceBounds) {
		return MakePreClipResult(ClipEffectClippedOut)
	}

	current := cs.currentSaveRecord()
	// Early out if the clip is known to be full 0s or full 1s.
	switch current.recordState() {
	case ClipStackEmpty:
		return MakePreClipResult(ClipEffectClippedOut)
	case ClipStackWideOpen:
		return MakePreClipResult(ClipEffectUnclipped)
	default:
	}

	// Given argument order, 'A' == current clip, 'B' == draw. The by-value helper keeps this function's draw off the
	// heap for the common wide-open/empty early-outs above (see getClipGeometryVsDraw).
	switch getClipGeometryVsDraw(current, draw) {
	case clipGeometryEmpty:
		return MakePreClipResult(ClipEffectClippedOut)
	case clipGeometryBOnly:
		// Geometrically the draw is unclipped (the clip-shader term is trimmed).
		return MakePreClipResult(ClipEffectUnclipped)
	default:
		// clipGeometryBoth, plus clipGeometryAOnly: the latter shouldn't happen since the inner bounds of a draw are
		// unknown, but if it did, the draw covered the clip and gets the clipGeometryBoth handling below.
		if len(cs.elements) == 0 {
			panic("clipGeometryBoth requires elements")
		}
		back := &cs.elements[len(cs.elements)-1]
		switch current.recordState() {
		case ClipStackDeviceRect:
			if back.clipType() != ClipStackDeviceRect {
				panic("state and element type must agree")
			}
			return MakePreClipResultRect(back.Shape.Rect(), back.AA)
		case ClipStackDeviceRRect:
			if back.clipType() != ClipStackDeviceRRect {
				panic("state and element type must agree")
			}
			return MakePreClipResultRRect(back.Shape.RRect(), back.AA)
		default:
			// The clip stack has complex shapes or multiple elements; preApply is meant to be conservative and
			// efficient.
			return MakePreClipResult(ClipEffectClipped)
		}
	}
}

// Apply implements Clip.
func (cs *ClipStack) Apply(sdc *SurfaceDrawContext, op DrawOp, aa gpu.AAType, out *AppliedClip, bounds *geom.Rect) ClipEffect {
	ctx := sdc.ctx
	if cs.proxyProvider == nil {
		cs.proxyProvider = ctx.ProxyProvider()
	} else if cs.proxyProvider != ctx.ProxyProvider() {
		panic("clip stack used with multiple contexts")
	}
	caps := sdc.Caps()

	// Convert the bounds to a Draw and apply device bounds clipping, making the query as tight as possible.
	draw := makeClipStackDraw(*bounds, gpu.AA(cs.forceAA || aa != gpu.AATypeNone))
	if !draw.applyDeviceBounds(cs.deviceBounds) {
		return ClipEffectClippedOut
	}
	if !bounds.Intersect(cs.deviceBounds.ToRect()) {
		panic("draw bounds must intersect device bounds here")
	}

	current := cs.currentSaveRecord()
	// Early out if the clip is known to be full 0s or full 1s.
	switch current.recordState() {
	case ClipStackEmpty:
		return ClipEffectClippedOut
	case ClipStackWideOpen:
		return ClipEffectUnclipped
	default:
	}

	// The clip-shader conversion is trimmed with clipShader (it has no public entry point).
	var clipFP FragmentProcessor

	// A refers to the entire clip stack, B refers to the draw.
	switch getClipGeometryVsDraw(current, draw) {
	case clipGeometryEmpty:
		return ClipEffectClippedOut
	case clipGeometryBOnly:
		// Geometrically unclipped (no clip shader to add).
		return ClipEffectUnclipped
	default:
		// clipGeometryBoth, plus clipGeometryAOnly (shouldn't happen since draws don't report inner bounds; treated
		// as kBoth).
		if current.recordState() != ClipStackDeviceRect &&
			current.recordState() != ClipStackDeviceRRect &&
			current.recordState() != ClipStackComplex {
			panic("unexpected clip state for kBoth")
		}
	}

	// Determine a scissor from the draw and the overall stack bounds. Initially keep it as large as possible: if the
	// clip is applied solely with coverage FPs, a loose scissor increases the chance draws can batch. It is tightened
	// later if any form of mask is needed.
	var scissorBounds geom.IRect
	if current.stackOp == raster.ClipIntersect {
		scissorBounds = current.outerBounds
	} else {
		scissorBounds = subtractRects(draw.bounds, current.innerBounds, true /* exact */)
	}

	// Marked true once there is a coverage FP (complex clipping is occurring), or there is an element that wouldn't
	// affect the scissored draw bounds but does affect the regular draw bounds (the scissor then suffices for clipping
	// but definitely cannot be dropped).
	scissorIsNeeded := false
	opClippedInternally := false

	remainingAnalyticFPs := clipMaxAnalyticFPs

	// Elements not represented as an analytic FP or skipped are collected and later applied using the stencil buffer or
	// a cached SW mask. (Window rectangles and the atlas clip lane are both trimmed.)
	var elementsForMask []*ClipElement

	maskRequiresAA := false

	for i := len(cs.elements) - 1; i >= current.oldestElementIndex(); i-- {
		e := &cs.elements[i]
		if e.isInvalid() {
			continue
		}

		switch getClipGeometryVsDraw(e, draw) {
		case clipGeometryEmpty:
			// This can happen for difference op elements that have a larger innerBounds than can be preserved at the
			// next level.
			return ClipEffectClippedOut

		case clipGeometryBOnly:
			// The element does not affect the draw; no coverage FP or mask entry needed.

		default:
			// clipGeometryBoth, plus clipGeometryAOnly (shouldn't happen for draws; gets the same regular element
			// processing).
			// The element must apply coverage to the draw; enable the scissor to limit overdraw.
			scissorIsNeeded = true

			// First apply using HW methods (scissor). When the inner and outer bounds match, nothing else needs to be
			// done.
			fullyApplied := false

			// Check if the op knows how to apply this clip internally.
			if op != nil {
				if e.Shape.Inverted() {
					panic("clip elements must not be inverted")
				}
				result := op.ClipToShape(e.Op, &e.LocalToDevice, &e.Shape,
					gpu.AA(e.AA == gpu.AAYes || cs.forceAA))
				if result != ClipResultFail {
					if result == ClipResultClippedOut {
						return ClipEffectClippedOut
					}
					if result == ClipResultClippedGeometrically {
						// The op clipped its own geometry; tighten the draw bounds.
						bounds.Intersect(e.outerBounds.ToRect())
					}
					fullyApplied = true
					opClippedInternally = true
				}
			}

			if !fullyApplied {
				if e.Op == raster.ClipIntersect {
					// The second test allows clipped draws that are scissored by multiple elements to remain
					// scissor-only.
					fullyApplied = e.innerBounds == e.outerBounds ||
						e.innerBounds.ContainsRect(scissorBounds)
				}
				// The difference-op window-rectangle lane is trimmed with window rectangles.
			}

			if !fullyApplied && remainingAnalyticFPs > 0 {
				clipFP, fullyApplied = analyticClipFP(&e.ClipElement, caps.ShaderCaps,
					clipFP)
				// The atlas-path-renderer clip lane is trimmed with that renderer.
				if fullyApplied {
					remainingAnalyticFPs--
				}
			}

			if !fullyApplied {
				elementsForMask = append(elementsForMask, &e.ClipElement)
				maskRequiresAA = maskRequiresAA || e.AA == gpu.AAYes
			}
		}
	}

	if !scissorIsNeeded {
		// More detailed analysis of the element shapes determined no clip is needed.
		if len(elementsForMask) > 0 || clipFP != nil {
			panic("mask elements without a needed scissor")
		}
		// opClippedInternally cannot be set here: the only path that reaches op.ClipToShape sets scissorIsNeeded first.
		return ClipEffectUnclipped
	}

	// Fill out the applied clip, possibly with a tightened scissor.
	if current.stackOp == raster.ClipIntersect && len(elementsForMask) > 0 {
		if !scissorBounds.Intersect(draw.bounds) {
			panic("scissor bounds must intersect draw bounds")
		}
	}
	if !IsInsideClip(scissorBounds, *bounds, draw.aa) {
		out.HardClip().AddScissor(scissorBounds, bounds)
	}

	// Rasterize any remaining elements to the stencil or a SW mask; all elements are flattened into a single mask.
	if len(elementsForMask) > 0 {
		stencilUnavailable := !sdc.AsRenderTargetProxy().CanUseStencil(caps)

		hasSWMask := false
		if (sdc.NumSamples() <= 1 && !sdc.canUseDynamicMSAA && maskRequiresAA) || stencilUnavailable {
			// Must use a texture mask to represent the combined elements since the stencil cannot be used or cannot
			// handle smooth clips.
			clipFP, hasSWMask = cs.getSWMaskFP(ctx, current, scissorBounds, elementsForMask,
				clipFP)
		}

		if !hasSWMask {
			if stencilUnavailable {
				// Drop the draw when a stencil-requiring clip has no stencil buffer.
				return ClipEffectClippedOut
			}
			// Rasterize the remaining elements into the stencil buffer.
			renderStencilMask(sdc, current.genID(), scissorBounds, elementsForMask, out)
		}
	}

	if clipFP != nil {
		// This includes all analytic FPs and the SW mask FP.
		out.AddCoverageFP(clipFP)
	}

	if !out.DoesClip() && !opClippedInternally {
		panic("clipped effect must modify the draw state")
	}
	return ClipEffectClipped
}

// getSWMaskFP generates or finds a cached SW coverage mask and returns an FP that samples it (composed onto clipFP).
func (cs *ClipStack) getSWMaskFP(ctx *DirectContext, current *clipStackSaveRecord, bounds geom.IRect, elements []*ClipElement, clipFP FragmentProcessor) (FragmentProcessor, bool) {
	proxyProvider := ctx.ProxyProvider()
	var maskProxy SurfaceProxyView

	var maskBounds geom.IRect // may not be 'bounds' if a larger cached mask is reused
	// Check the existing masks from this save record for compatibility.
	for i := len(cs.masks) - 1; i >= 0; i-- {
		m := &cs.masks[i]
		if m.genID != current.genID() {
			break
		}
		if m.appliesToDraw(current, bounds) {
			if proxy := proxyProvider.FindOrCreateProxyByUniqueKey(&m.key,
				UseAllocatorYes); proxy != nil {
				swizzle := ctx.GLCaps().ReadSwizzle(proxy.Format(), gpu.ColorTypeAlpha8)
				maskProxy = MakeSurfaceProxyView(proxy, clipMaskOrigin, swizzle)
				maskBounds = m.bounds
				break
			}
		}
	}

	if maskProxy.Proxy() == nil {
		// No existing mask was found, so render a new one.
		maskProxy = renderSWMask(ctx, bounds, elements)
		if maskProxy.Proxy() == nil {
			// If we still don't have one, there's nothing we can do.
			return clipFP, false
		}

		// Register the mask for later invalidation.
		cs.masks = append(cs.masks, newClipStackMask(current, bounds))
		proxyProvider.AssignUniqueKeyToProxy(&cs.masks[len(cs.masks)-1].key,
			maskProxy.Proxy().AsTextureProxy())
		maskBounds = bounds
	}

	// Wrap the mask in an FP that samples it for coverage.
	samplerState := gpu.MakeSamplerState(gpu.WrapModeClampToBorder, gpu.WrapModeClampToBorder,
		gpu.FilterModeNearest, gpu.MipmapModeNone)
	// Map the device coords passed to the texture effect to the top-left corner of the mask, with the draw bounds
	// pre-mapped into the mask's space as well.
	m := geom.TranslateMatrix(float32(-maskBounds.Left), float32(-maskBounds.Top))
	subset := bounds.ToRect()
	subset.Left -= float32(maskBounds.Left)
	subset.Right -= float32(maskBounds.Left)
	subset.Top -= float32(maskBounds.Top)
	subset.Bottom -= float32(maskBounds.Top)
	// We scissor to bounds. The mask's texel centers are aligned to device-space pixel centers, hence this domain of
	// texture coordinates.
	domain := subset.Inset(0.5, 0.5)
	fp := MakeTextureEffectSubsetWithDomain(maskProxy, gpu.AlphaTypePremul, m, samplerState,
		subset, domain, ctx.GLCaps(), [4]float32{})
	fp = DeviceSpaceFP(fp)

	// Combine the coverage sampled from the texture effect with the previous coverage.
	fp = BlendFP(fp, clipFP, raster.BlendDstIn)
	return fp, true
}
