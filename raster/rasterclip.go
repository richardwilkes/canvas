// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Clip wraps a Region and an AAClip, so we have a single object that can represent either a BW or antialiased
// clip. Clip shaders arrive, if ever needed, with the shader phase.

package raster

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// Clip is a device-space clip that can be either a hard-edged Region or an antialiased AAClip. The zero value is
// NOT ready for use — construct via the NewRasterClip functions or SetEmpty/SetRect, which establish the empty cache
// state.
type Clip struct {
	bw   Region
	aa   AAClip
	isBW bool
	// these 2 are caches based on querying the right obj based on isBW
	isEmpty bool
	isRect  bool
}

// NewRasterClip returns an empty clip.
func NewRasterClip() *Clip {
	return &Clip{isBW: true, isEmpty: true}
}

// NewRasterClipRect returns a BW clip set to bounds.
func NewRasterClipRect(bounds geom.IRect) *Clip {
	rc := &Clip{isBW: true}
	rc.bw.SetRect(bounds)
	rc.isEmpty = rc.computeIsEmpty() // bounds might be empty, so compute
	rc.isRect = !rc.isEmpty
	return rc
}

// NewRasterClipRegion returns a BW clip set to rgn.
func NewRasterClipRegion(rgn *Region) *Clip {
	rc := &Clip{isBW: true}
	rc.bw.SetRegion(rgn)
	rc.isEmpty = rc.computeIsEmpty()
	rc.isRect = !rc.isEmpty
	return rc
}

// NewRasterClipPath returns a clip set to the path's coverage restricted to bounds, BW or AA per doAA.
func NewRasterClipPath(p *path.Path, bounds geom.IRect, doAA bool) *Clip {
	rc := &Clip{}
	if doAA {
		rc.isBW = false
		rc.aa.SetPath(p, bounds, true)
	} else {
		rc.isBW = true
		rc.bw.SetPath(p, NewRegionRect(bounds))
	}
	rc.isEmpty = rc.computeIsEmpty()
	rc.isRect = rc.computeIsRect()
	return rc
}

// Clone returns a copy sharing the underlying (immutable) run data.
func (rc *Clip) Clone() *Clip {
	clone := *rc
	return &clone
}

// IsBW reports whether the clip is hard-edged (a Region).
func (rc *Clip) IsBW() bool { return rc.isBW }

// IsAA reports whether the clip is antialiased (an AAClip).
func (rc *Clip) IsAA() bool { return !rc.isBW }

// BWRgn returns the underlying Region. Only valid when IsBW.
func (rc *Clip) BWRgn() *Region { return &rc.bw }

// AARgn returns the underlying AAClip. Only valid when IsAA.
func (rc *Clip) AARgn() *AAClip { return &rc.aa }

// IsEmpty reports whether the clip contains no pixels.
func (rc *Clip) IsEmpty() bool { return rc.isEmpty }

// IsRect reports whether the clip is a hard-edged rect.
func (rc *Clip) IsRect() bool { return rc.isRect }

// IsComplex reports whether the clip is more than a simple rect.
func (rc *Clip) IsComplex() bool {
	if rc.isBW {
		return rc.bw.IsComplex()
	}
	return !rc.aa.IsEmpty()
}

// Bounds returns the clip's bounding rect.
func (rc *Clip) Bounds() geom.IRect {
	if rc.isBW {
		return rc.bw.Bounds()
	}
	return rc.aa.Bounds()
}

func (rc *Clip) computeIsEmpty() bool {
	if rc.isBW {
		return rc.bw.IsEmpty()
	}
	return rc.aa.IsEmpty()
}

func (rc *Clip) computeIsRect() bool {
	if rc.isBW {
		return rc.bw.IsRect()
	}
	return rc.aa.IsRect()
}

// updateCacheAndReturnNonEmpty refreshes the isEmpty/isRect caches after a mutation, optionally collapsing an AA clip
// that turned out to be a hard-edged rect back to BW.
func (rc *Clip) updateCacheAndReturnNonEmpty(detectAARect bool) bool {
	rc.isEmpty = rc.computeIsEmpty()

	// detect that our computed AA is really just a (hard-edged) rect
	if detectAARect && !rc.isEmpty && !rc.isBW && rc.aa.IsRect() {
		rc.bw.SetRect(rc.aa.Bounds())
		rc.aa.SetEmpty() // don't need this anymore
		rc.isBW = true
	}

	rc.isRect = rc.computeIsRect()
	return !rc.isEmpty
}

// SetEmpty clears the clip to empty and returns false.
func (rc *Clip) SetEmpty() bool {
	rc.isBW = true
	rc.bw.SetEmpty()
	rc.aa.SetEmpty()
	rc.isEmpty = true
	rc.isRect = false
	return false
}

// SetRect sets the clip to a hard-edged rect.
func (rc *Clip) SetRect(rect geom.IRect) bool {
	rc.isBW = true
	rc.aa.SetEmpty()
	rc.isRect = rc.bw.SetRect(rect)
	rc.isEmpty = !rc.isRect
	return rc.isRect
}

// convertToAA converts a BW clip to its AA equivalent in place.
func (rc *Clip) convertToAA() {
	rc.aa.SetRegion(&rc.bw)
	rc.isBW = false
	rc.bw.SetEmpty()

	// since we are being explicitly asked to convert-to-aa, we pass false so we don't "optimize" ourselves back to BW.
	rc.updateCacheAndReturnNonEmpty(false)
}

// clipOpToRegionOp maps a ClipOp onto the equivalent RegionOp.
func clipOpToRegionOp(op ClipOp) RegionOp {
	if op == ClipDifference {
		return RegionDifference
	}
	return RegionIntersect
}

// OpIRect combines the clip with a hard-edged rect under op.
func (rc *Clip) OpIRect(rect geom.IRect, op ClipOp) bool {
	if rc.isBW {
		rc.bw.OpRect(rect, clipOpToRegionOp(op))
	} else {
		rc.aa.OpIRect(rect, op)
	}
	return rc.updateCacheAndReturnNonEmpty(true)
}

// OpRegion combines the clip with a Region under op.
func (rc *Clip) OpRegion(rgn *Region, op ClipOp) bool {
	if rc.isBW {
		rc.bw.Op(rgn, clipOpToRegionOp(op))
	} else {
		var tmp AAClip
		tmp.SetRegion(rgn)
		rc.aa.Op(&tmp, op)
	}
	return rc.updateCacheAndReturnNonEmpty(true)
}

// nearlyIntegral reports whether x is close enough to an integer to treat as one: antialiasing currently has a
// granularity of 1/4 of a pixel along each axis, so we can treat an axis coordinate as an integer if it differs from
// its nearest int by < half of that value (1/8 in this case).
func nearlyIntegral(x float32) bool {
	const domain = 1.0 / 4
	const halfDomain = domain / 2

	x += halfDomain
	return x-geom.FloorToScalar(x) < domain
}

// OpRect combines the clip with a (possibly fractional) rect, transformed by matrix, under op.
func (rc *Clip) OpRect(localRect geom.Rect, matrix *geom.Matrix, op ClipOp, doAA bool) bool {
	if !matrix.IsScaleTranslate() {
		return rc.OpPath(path.New().AddRect(localRect, geom.DirectionCW), matrix, op, doAA)
	}

	devRect, _ := matrix.MapRect(localRect)
	if rc.isBW && doAA {
		// check that the rect really needs aa, or is it close enough to integer boundaries that we can just treat it as
		// a BW rect?
		if nearlyIntegral(devRect.Left) && nearlyIntegral(devRect.Top) &&
			nearlyIntegral(devRect.Right) && nearlyIntegral(devRect.Bottom) {
			doAA = false
		}
	}

	if rc.isBW && !doAA {
		rc.bw.OpRect(devRect.Round(), clipOpToRegionOp(op))
	} else {
		if rc.isBW {
			rc.convertToAA()
		}
		rc.aa.OpRect(devRect, op, doAA)
	}
	return rc.updateCacheAndReturnNonEmpty(true)
}

// OpRRect combines the clip with a rounded rect, transformed by matrix, under op.
func (rc *Clip) OpRRect(rrect geom.RRect, matrix *geom.Matrix, op ClipOp, doAA bool) bool {
	return rc.OpPath(path.New().AddRRect(rrect, geom.DirectionCW), matrix, op, doAA)
}

// OpPath combines the clip with a path, transformed by matrix, under op.
func (rc *Clip) OpPath(p *path.Path, matrix *geom.Matrix, op ClipOp, doAA bool) bool {
	devPath := &path.Path{}
	p.TransformTo(matrix, devPath)

	// Since op is either intersect or difference, the clip is always shrinking; that means we can always use our
	// current bounds as the limiting factor for region/aaclip operations.
	if rc.IsRect() && op == ClipIntersect {
		// However, in the relatively common case of intersecting a new path with a rectangular clip, it's faster to
		// convert the path into a region/aa-mask in place than evaluate the actual intersection. See skbug.com/40043482
		if doAA && rc.isBW {
			rc.convertToAA()
		}
		if rc.isBW {
			rc.bw.SetPath(devPath, NewRegionRect(rc.Bounds()))
		} else {
			rc.aa.SetPath(devPath, rc.Bounds(), doAA)
		}
		return rc.updateCacheAndReturnNonEmpty(true)
	}
	return rc.opRasterClip(NewRasterClipPath(devPath, rc.Bounds(), doAA), op)
}

// opRasterClip combines the clip with another Clip under op.
func (rc *Clip) opRasterClip(clip *Clip, op ClipOp) bool {
	if rc.isBW && clip.isBW {
		rc.bw.Op(&clip.bw, clipOpToRegionOp(op))
	} else {
		var tmp AAClip
		var other *AAClip

		if rc.isBW {
			rc.convertToAA()
		}
		if clip.isBW {
			tmp.SetRegion(&clip.bw)
			other = &tmp
		} else {
			other = &clip.aa
		}
		rc.aa.Op(other, op)
	}
	return rc.updateCacheAndReturnNonEmpty(true)
}

// Translate sets dst to this clip offset by (dx, dy).
func (rc *Clip) Translate(dx, dy int32, dst *Clip) {
	if dst == nil {
		return
	}

	if rc.IsEmpty() {
		dst.SetEmpty()
		return
	}
	if dx|dy == 0 {
		*dst = *rc
		return
	}

	dst.isBW = rc.isBW
	if rc.isBW {
		dst.bw = rc.bw
		dst.bw.Translate(dx, dy)
		dst.aa.SetEmpty()
	} else {
		rc.aa.Translate(dx, dy, &dst.aa)
		dst.bw.SetEmpty()
	}
	dst.updateCacheAndReturnNonEmpty(true)
}

// QuickContains reports whether the clip is at full coverage over all of rect.
func (rc *Clip) QuickContains(rect geom.IRect) bool {
	if rc.isBW {
		return rc.bw.QuickContains(rect)
	}
	return rc.aa.QuickContains(rect)
}

// QuickReject reports whether rect is empty or definitely outside the clip.
func (rc *Clip) QuickReject(rect geom.IRect) bool {
	return !rc.Bounds().Intersects(rect)
}

///////////////////////////////////////////////////////////////////////////////

// AAClipBlitterWrapper encapsulates the logic of deciding if we need to change/wrap the blitter for aaclipping. If so,
// Rgn and Blitter return modified values; if not, they return the raw blitter and (bw) clip region.
type AAClipBlitterWrapper struct {
	blitter Blitter
	// what we return
	clipRgn   *Region
	bwRgn     Region
	aaBlitter AAClipBlitter
}

// Init sets up the wrapper for clip, wrapping blitter with an AAClipBlitter when clip is AA.
func (w *AAClipBlitterWrapper) Init(clip *Clip, blitter Blitter) {
	if clip.IsBW() {
		w.clipRgn = clip.BWRgn()
		w.blitter = blitter
	} else {
		aaclip := clip.AARgn()
		w.bwRgn.SetRect(aaclip.Bounds())
		w.aaBlitter.Init(blitter, aaclip)
		// now our return values
		w.clipRgn = &w.bwRgn
		w.blitter = &w.aaBlitter
	}
}

// InitAAClip sets up the wrapper directly from an AAClip, always wrapping blitter with an AAClipBlitter.
func (w *AAClipBlitterWrapper) InitAAClip(aaclip *AAClip, blitter Blitter) {
	w.bwRgn.SetRect(aaclip.Bounds())
	w.aaBlitter.Init(blitter, aaclip)
	w.clipRgn = &w.bwRgn
	w.blitter = &w.aaBlitter
}

// Bounds returns the effective clip region's bounds.
func (w *AAClipBlitterWrapper) Bounds() geom.IRect { return w.clipRgn.Bounds() }

// Rgn returns the effective (bw) clip region to apply.
func (w *AAClipBlitterWrapper) Rgn() *Region { return w.clipRgn }

// Blitter returns the effective blitter to draw through.
func (w *AAClipBlitterWrapper) Blitter() Blitter { return w.blitter }

///////////////////////////////////////////////////////////////////////////////

// rasterClipStackRec is one entry in the ClipStack.
type rasterClipStackRec struct {
	rc            Clip
	deferredCount int // 0 for a "normal" entry
}

// ClipStack is the save/restore stack of Clips a raster device maintains, with copy-on-write deferral of
// saves.
type ClipStack struct {
	stack      []rasterClipStackRec
	rootBounds geom.IRect
	disableAA  bool
}

// NewRasterClipStack returns a stack whose root clip covers the full width x height device.
func NewRasterClipStack(width, height int32) *ClipStack {
	s := &ClipStack{
		rootBounds: geom.IRectWH(width, height),
	}
	s.disableAA = PathRequiresTiling(s.rootBounds)
	s.stack = append(s.stack, rasterClipStackRec{rc: *NewRasterClipRect(s.rootBounds)})
	return s
}

// SetNewSize resizes the root clip to a new device size.
func (s *ClipStack) SetNewSize(w, h int32) {
	s.rootBounds = geom.IRectWH(w, h)
	s.stack[len(s.stack)-1].rc.SetRect(s.rootBounds)
}

// RC returns the current (top of stack) clip.
func (s *ClipStack) RC() *Clip {
	return &s.stack[len(s.stack)-1].rc
}

// Save defers pushing a new clip level until the next mutation.
func (s *ClipStack) Save() {
	s.stack[len(s.stack)-1].deferredCount++
}

// Restore pops a save level, discarding any clip pushed since the matching Save.
func (s *ClipStack) Restore() {
	top := &s.stack[len(s.stack)-1]
	top.deferredCount--
	if top.deferredCount < 0 {
		s.stack = s.stack[:len(s.stack)-1]
	}
}

// writableRC returns a clip safe to mutate in place, pushing a copy-on-write clone if a deferred save is pending.
func (s *ClipStack) writableRC() *Clip {
	top := &s.stack[len(s.stack)-1]
	if top.deferredCount > 0 {
		top.deferredCount--
		s.stack = append(s.stack, rasterClipStackRec{rc: top.rc})
	}
	return &s.stack[len(s.stack)-1].rc
}

// ClipRect combines the current clip with a rect, transformed by ctm, under op.
func (s *ClipStack) ClipRect(ctm *geom.Matrix, rect geom.Rect, op ClipOp, aa bool) {
	s.writableRC().OpRect(rect, ctm, op, s.finalAA(aa))
}

// ClipRRect combines the current clip with a rounded rect, transformed by ctm, under op.
func (s *ClipStack) ClipRRect(ctm *geom.Matrix, rrect geom.RRect, op ClipOp, aa bool) {
	s.writableRC().OpRRect(rrect, ctm, op, s.finalAA(aa))
}

// ClipPath combines the current clip with a path, transformed by ctm, under op.
func (s *ClipStack) ClipPath(ctm *geom.Matrix, p *path.Path, op ClipOp, aa bool) {
	s.writableRC().OpPath(p, ctm, op, s.finalAA(aa))
}

// ClipRegion combines the current clip with a Region under op.
func (s *ClipStack) ClipRegion(rgn *Region, op ClipOp) {
	s.writableRC().OpRegion(rgn, op)
}

// ReplaceClip replaces the current clip with rect, intersected with the root bounds.
func (s *ClipStack) ReplaceClip(rect geom.IRect) {
	devRect := rect
	if !devRect.Intersect(s.rootBounds) {
		s.writableRC().SetEmpty()
	} else {
		s.writableRC().SetRect(devRect)
	}
}

func (s *ClipStack) finalAA(aa bool) bool { return aa && !s.disableAA }
