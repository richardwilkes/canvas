// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// StencilClip is a hard clip carrying the currently-rendered stencil-clip contents plus a fixed clip, and
// StencilMaskHelper renders merged clip elements into the stencil buffer's clip bit using the user-to-clip stencil
// settings tables below. Window rectangles are not supported. ClipStack diverts AA elements to a SW mask whenever the
// target has no MSAA to resolve them with (a 1-sample target that is not running dynamic MSAA), so the drawPath lane is
// reached only from multisample and dynamic-MSAA targets; the inverse-fill table rows are present as data but only the
// non-inverted Replace/Intersect/Difference rows are reachable from the clip stack.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// StencilClip is a hard clip combining the currently-existing stencil buffer contents with a fixed clip.
type StencilClip struct {
	fixedClip      FixedClip
	stencilStackID uint32
}

// NewStencilClip creates a StencilClip over a render target of the given dimensions, referencing the given stencil-clip
// stack generation.
func NewStencilClip(rtDims geom.ISize, stencilStackID uint32) *StencilClip {
	return &StencilClip{fixedClip: *NewFixedClip(rtDims), stencilStackID: stencilStackID}
}

// FixedClip returns the underlying fixed (scissor) clip.
func (c *StencilClip) FixedClip() *FixedClip { return &c.fixedClip }

// StencilStackID returns the stencil-clip stack generation this clip references.
func (c *StencilClip) StencilStackID() uint32 { return c.stencilStackID }

// HasStencilClip reports whether this clip references a valid stencil-clip stack generation.
func (c *StencilClip) HasStencilClip() bool { return c.stencilStackID != invalidGenID }

// SetStencilClip sets the stencil-clip stack generation this clip references.
func (c *StencilClip) SetStencilClip(stencilStackID uint32) { c.stencilStackID = stencilStackID }

// GetConservativeBounds implements Clip.
func (c *StencilClip) GetConservativeBounds() geom.IRect {
	return c.fixedClip.GetConservativeBounds()
}

// Apply implements Clip through the hard-clip lane.
func (c *StencilClip) Apply(_ *SurfaceDrawContext, _ DrawOp, aa gpu.AAType, out *AppliedClip, bounds *geom.Rect) ClipEffect {
	return hardClipApply(c.applyHard, aa, out, bounds)
}

// applyHard applies the fixed clip and, if this clip references a stencil-clip stack generation, adds a stencil clip
// test to out as well.
func (c *StencilClip) applyHard(out *AppliedHardClip, bounds *geom.IRect) ClipEffect {
	effect := c.fixedClip.applyHard(out, bounds)
	if effect == ClipEffectClippedOut {
		// Stencil won't bring back coverage.
		return ClipEffectClippedOut
	}
	if c.HasStencilClip() {
		out.AddStencilClip(c.stencilStackID)
		effect = ClipEffectClipped
	}
	return effect
}

// PreApply implements Clip.
func (c *StencilClip) PreApply(drawBounds geom.Rect, aa gpu.AA) PreClipResult {
	if c.HasStencilClip() {
		return clipPreApplyDefault(c, drawBounds, aa)
	}
	return c.fixedClip.PreApply(drawBounds, aa)
}

//////////////////////////////////////////////////////////////////////////////
// Stencil rules for merging user stencil space into the clip.

// User-to-clip settings.
var (
	// Replace.
	userToClipReplace = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
		UserStencilOpSetClipAndReplaceUserBits, UserStencilOpZeroClipAndUserBits, 0xffff)
	invUserToClipReplace = MakeUserStencilSettings(0x0000, UserStencilTestEqual, 0xffff,
		UserStencilOpSetClipAndReplaceUserBits, UserStencilOpZeroClipAndUserBits, 0xffff)

	// Intersect: "0 < userBits" is equivalent to "0 != userBits".
	userToClipIsect = MakeUserStencilSettings(0x0000, UserStencilTestLessIfInClip, 0xffff,
		UserStencilOpSetClipAndReplaceUserBits, UserStencilOpZeroClipAndUserBits, 0xffff)

	// Difference.
	userToClipDiff = MakeUserStencilSettings(0x0000, UserStencilTestEqualIfInClip, 0xffff,
		UserStencilOpSetClipAndReplaceUserBits, UserStencilOpZeroClipAndUserBits, 0xffff)

	// Union.
	userToClipUnion = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
		UserStencilOpSetClipAndReplaceUserBits, UserStencilOpKeep, 0xffff)
	invUserToClipUnionPass0 = MakeUserStencilSettings(0x0000, UserStencilTestEqual, 0xffff,
		UserStencilOpSetClipBit, UserStencilOpKeep, 0x0000) // Does not zero user bits.

	// Xor.
	userToClipXorPass0 = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
		UserStencilOpInvertClipBit, UserStencilOpKeep, 0x0000) // Does not zero user bits.
	invUserToClipXorPass0 = MakeUserStencilSettings(0x0000, UserStencilTestEqual, 0xffff,
		UserStencilOpInvertClipBit, UserStencilOpKeep, 0x0000) // Does not zero user bits.

	// Reverse difference.
	userToClipRDiffPass0 = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
		UserStencilOpInvertClipBit, UserStencilOpZeroClipBit, 0x0000) // Does not zero user bits.
	invUserToClipRDiffPass0 = MakeUserStencilSettings(0x0000, UserStencilTestEqual, 0xffff,
		UserStencilOpInvertClipBit, UserStencilOpZeroClipBit, 0x0000) // Does not zero user bits.

	// Second pass to clear user bits (only needed sometimes).
	zeroUserBits = MakeUserStencilSettings(0x0000, UserStencilTestNotEqual, 0xffff,
		UserStencilOpZero, UserStencilOpKeep, 0xffff)
)

// userToClipTable holds, for each (fillInverted, region op) pair, the ordered list of stencil passes that merge the
// user stencil bits into the clip bit.
var userToClipTable = [2][6][]*UserStencilSettings{
	{ // Normal fill.
		{&userToClipDiff},                      // kDifference_Op.
		{&userToClipIsect},                     // kIntersect_Op.
		{&userToClipUnion},                     // kUnion_Op.
		{&userToClipXorPass0, &zeroUserBits},   // kXOR_Op.
		{&userToClipRDiffPass0, &zeroUserBits}, // kReverseDifference_Op.
		{&userToClipReplace},                   // kReplace_Op.
	},
	{ // Inverse fill.
		{&userToClipIsect},                        // ~diff (aka isect).
		{&userToClipDiff},                         // ~isect (aka diff).
		{&invUserToClipUnionPass0, &zeroUserBits}, // ~union.
		{&invUserToClipXorPass0, &zeroUserBits},   // ~xor.
		{&invUserToClipRDiffPass0, &zeroUserBits}, // ~reverse diff.
		{&invUserToClipReplace},                   // ~replace.
	},
}

// Direct-to-stencil settings: a clip element can render directly without first writing to the user bits when the fill
// is not inverse and the op only modifies samples covered by the element.
var (
	// replaceClip only works right after the stencil clip was cleared (clip mask creation does not allow midstream
	// replace ops).
	replaceClipSettings = MakeUserStencilSettings(0x0000, UserStencilTestAlways, 0xffff,
		UserStencilOpSetClipBit, UserStencilOpSetClipBit, 0x0000)
	unionClipSettings = MakeUserStencilSettings(0x0000, UserStencilTestAlwaysIfInClip, 0xffff,
		UserStencilOpKeep, UserStencilOpSetClipBit, 0x0000)
	xorClipSettings = MakeUserStencilSettings(0x0000, UserStencilTestAlways, 0xffff,
		UserStencilOpInvertClipBit, UserStencilOpInvertClipBit, 0x0000)
	diffClipSettings = MakeUserStencilSettings(0x0000, UserStencilTestAlwaysIfInClip, 0xffff,
		UserStencilOpZeroClipBit, UserStencilOpKeep, 0x0000)
)

// directDrawTable holds, for each region op that supports it, the stencil pass that lets an element render directly to
// the clip bit without first writing to the user bits.
var directDrawTable = [6][]*UserStencilSettings{
	{&diffClipSettings},    // kDifference_Op.
	nil,                    // kIntersect_Op.
	{&unionClipSettings},   // kUnion_Op.
	{&xorClipSettings},     // kXOR_Op.
	nil,                    // kReverseDifference_Op.
	{&replaceClipSettings}, // kReplace_Op.
}

// drawToStencilSettings fills the user stencil bits before applying the covering clip stencil passes when direct draws
// aren't possible.
var drawToStencilSettings = MakeUserStencilSettings(0x0000, UserStencilTestAlways, 0xffff,
	UserStencilOpIncMaybeClamp, UserStencilOpIncMaybeClamp, 0xffff)

// getStencilPasses returns the per-pass stencil settings to achieve the given fill+region-op effect on the clip bit.
// canRenderDirectToStencil stands in for the path renderer having no stencil-support restriction (always true for the
// rect lane). When drawDirectToClip comes back false, the caller must first draw the element into the user stencil bits
// and then cover the clip area with the returned passes.
func getStencilPasses(op raster.RegionOp, canRenderDirectToStencil, fillInverted bool) (passes []*UserStencilSettings, drawDirectToClip bool) {
	if canRenderDirectToStencil && !fillInverted {
		if directPass := directDrawTable[op]; directPass != nil {
			return directPass, true
		}
	}
	inv := 0
	if fillInverted {
		inv = 1
	}
	return userToClipTable[inv][op], false
}

//////////////////////////////////////////////////////////////////////////////
// StencilMaskHelper.

// StencilMaskHelper renders the merged elements of a clip query into the stencil buffer's clip bit.
type StencilMaskHelper struct {
	sdc    *SurfaceDrawContext
	clip   StencilClip
	numFPs int
}

// NewStencilMaskHelper creates a StencilMaskHelper targeting sdc's stencil buffer.
func NewStencilMaskHelper(sdc *SurfaceDrawContext) *StencilMaskHelper {
	return &StencilMaskHelper{
		sdc:  sdc,
		clip: StencilClip{fixedClip: *NewFixedClip(sdc.Dimensions()), stencilStackID: invalidGenID},
	}
}

// Init returns true if the mask must actually be rendered, configuring the scissor to bounds as a side effect.
func (h *StencilMaskHelper) Init(bounds geom.IRect, genID uint32, numFPs int) bool {
	if !h.sdc.MustRenderClip(genID, bounds, numFPs) {
		return false
	}
	h.clip.SetStencilClip(genID)
	// Bounds not intersecting the render target should have been caught much earlier in clip application.
	if !h.clip.fixedClip.SetScissor(bounds) {
		panic("stencil mask bounds do not intersect the render target")
	}
	h.numFPs = numFPs
	return true
}

// drawStencilRect issues a color-write-disabled stencil rect draw.
func (h *StencilMaskHelper) drawStencilRect(clip Clip, ss *UserStencilSettings, matrix *geom.Matrix, rect geom.Rect, aa gpu.AA) {
	paint := NewPaint()
	paint.SetXPFactory(DisableColorXPFactory())
	h.sdc.StencilRect(clip, ss, paint, aa, matrix, rect, nil)
}

// supportedAA reports the AA to actually use: MSAA is the only type of AA possible on a stencil buffer. A DMSAA surface
// reports one sample but promotes to an MSAA attachment for stencil draws, so it counts as MSAA-capable here.
func (h *StencilMaskHelper) supportedAA(gpu.AA) gpu.AA {
	return gpu.AA(h.sdc.NumSamples() > 1 || h.sdc.canUseDynamicMSAA)
}

// DrawRect merges rect into the clip mask under op.
func (h *StencilMaskHelper) DrawRect(rect geom.Rect, matrix *geom.Matrix, op raster.RegionOp, aa gpu.AA) {
	if rect.IsEmpty() {
		return
	}

	passes, drawDirectToClip := getStencilPasses(op, true /* no restriction */, false)
	aa = h.supportedAA(aa)

	if !drawDirectToClip {
		// Draw to the client stencil bits first.
		h.drawStencilRect(&h.clip.fixedClip, &drawToStencilSettings, matrix, rect, aa)
	}

	// Now modify the clip bit, either by rendering directly or by covering the bounding box of the clip.
	identity := geom.IdentityMatrix()
	for _, pass := range passes {
		if drawDirectToClip {
			h.drawStencilRect(&h.clip, pass, matrix, rect, aa)
		} else {
			h.drawStencilRect(&h.clip, pass, &identity,
				h.clip.fixedClip.ScissorRect().ToRect(), aa)
		}
	}
}

// stencilDrawPath issues a color-write-disabled path draw through the given renderer with the given user stencil
// settings.
func stencilDrawPath(sdc *SurfaceDrawContext, pr PathRenderer, clip Clip, bounds geom.IRect, ss *UserStencilSettings, matrix *geom.Matrix, shape *StyledShape, aa gpu.AA) {
	paint := NewPaint()
	paint.SetXPFactory(DisableColorXPFactory())

	// kMSAA is the only type of AA that's possible on a stencil buffer.
	pathAAType := gpu.AATypeNone
	if aa == gpu.AAYes {
		pathAAType = gpu.AATypeMSAA
	}

	args := DrawPathArgs{
		Context:                sdc.ctx,
		Paint:                  paint,
		UserStencilSettings:    ss,
		SurfaceDrawContext:     sdc,
		Clip:                   clip,
		ClipConservativeBounds: bounds,
		ViewMatrix:             matrix,
		Shape:                  shape,
		AAType:                 pathAAType,
	}
	PathRendererDrawPath(pr, &args)
}

// stencilStencilPath stencils shape through the given renderer without drawing any color.
func stencilStencilPath(sdc *SurfaceDrawContext, pr PathRenderer, clip *FixedClip, matrix *geom.Matrix, shape *StyledShape, aa gpu.AA) {
	args := StencilPathArgs{
		Context:                sdc.ctx,
		SurfaceDrawContext:     sdc,
		Clip:                   clip,
		ClipConservativeBounds: clip.ScissorRect(),
		ViewMatrix:             matrix,
		Shape:                  shape,
		DoStencilMSAA:          aa,
	}
	PathRendererStencilPath(pr, &args)
}

// DrawPath merges the path p into the clip mask under op. Like DrawRect, it either draws directly to the clip bit or
// first draws to the client bits and then applies a cover pass, depending on how the chosen path renderer uses the
// stencil buffer.
func (h *StencilMaskHelper) DrawPath(p *path.Path, matrix *geom.Matrix, op raster.RegionOp, aa gpu.AA) bool {
	if p.IsEmpty() {
		return true
	}
	aa = h.supportedAA(aa)

	pathAAType := gpu.AATypeNone
	if aa == gpu.AAYes {
		pathAAType = gpu.AATypeMSAA
	}

	// Make the path canonical with regards to fill type (inverse handled by stencil settings).
	fillInverted := p.IsInverseFillType()
	clipPath := p
	if fillInverted {
		clipPath = p.Clone()
		clipPath.ToggleInverseFillType()
	}

	shape := MakeStyledShapePath(clipPath, SimpleFillStyle(), DoSimplifyYes)
	if shape.InverseFilled() {
		panic("clip path must be non-inverse here")
	}

	canDrawArgs := CanDrawPathArgs{
		Caps:                   h.sdc.Caps(),
		Proxy:                  h.sdc.AsRenderTargetProxy(),
		ClipConservativeBounds: h.clip.fixedClip.ScissorRect(),
		ViewMatrix:             matrix,
		Shape:                  &shape,
		Paint:                  nil,
		SurfaceProps:           h.sdc.surfaceProps,
		AAType:                 pathAAType,
		HasUserStencilSettings: false,
	}

	// This determines whether the clip shape can be rendered into the stencil with arbitrary stencil settings.
	var stencilSupport StencilSupport
	pr := h.sdc.drawingManager().GetPathRenderer(&canDrawArgs, false, ChainDrawTypeStencil,
		&stencilSupport)
	if pr == nil {
		return false
	}

	passes, drawDirectToClip := getStencilPasses(op,
		stencilSupport == StencilSupportNoRestriction, fillInverted)

	// Write to the client bits if necessary.
	if !drawDirectToClip {
		if stencilSupport == StencilSupportNoRestriction {
			stencilDrawPath(h.sdc, pr, &h.clip.fixedClip, h.clip.fixedClip.ScissorRect(),
				&drawToStencilSettings, matrix, &shape, aa)
		} else {
			stencilStencilPath(h.sdc, pr, &h.clip.fixedClip, matrix, &shape, aa)
		}
	}

	// Now modify the clip bit, either by rendering directly or by covering the bounding box of the clip.
	identity := geom.IdentityMatrix()
	for _, pass := range passes {
		if drawDirectToClip {
			stencilDrawPath(h.sdc, pr, &h.clip, h.clip.fixedClip.ScissorRect(), pass, matrix,
				&shape, aa)
		} else {
			h.drawStencilRect(&h.clip, pass, &identity,
				h.clip.fixedClip.ScissorRect().ToRect(), aa)
		}
	}
	return true
}

// DrawShape merges shape into the clip mask under op, using the rect fast path when possible.
func (h *StencilMaskHelper) DrawShape(shape *Shape, matrix *geom.Matrix, op raster.RegionOp, aa gpu.AA) bool {
	if shape.IsRect() && !shape.Inverted() {
		h.DrawRect(shape.Rect(), matrix, op, aa)
		return true
	}
	return h.DrawPath(shape.AsPath(), matrix, op, aa)
}

// Clear clears the stencil clip bit over the current scissor rect, to insideStencil.
func (h *StencilMaskHelper) Clear(insideStencil bool) {
	h.sdc.ClearStencilClip(h.clip.fixedClip.ScissorRect(), insideStencil)
}

// Finish records the completed clip mask as the surface's last-applied stencil clip.
func (h *StencilMaskHelper) Finish() {
	h.sdc.SetLastClip(h.clip.StencilStackID(), h.clip.fixedClip.ScissorRect(), h.numFPs)
}
