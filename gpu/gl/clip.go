// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The abstract clip interface that fills out an AppliedClip, the pixel-bounds helpers all clipping decisions ride on,
// and the fixed-function clip implementation (scissor; window rectangles are trimmed — EXT_window_rectangles is absent
// from the desktop trim's target drivers). The full ClipStack lives in clipstack.go.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// ClipEffect describes how a clip affects a draw.
type ClipEffect int32

// ClipEffect values.
const (
	// ClipEffectClipped: the clip conservatively modifies the draw's coverage but doesn't eliminate the draw.
	ClipEffectClipped ClipEffect = iota
	// ClipEffectUnclipped: the clip definitely does not modify the draw's coverage and the draw can be performed
	// without clipping (beyond the automatic device bounds clip).
	ClipEffectUnclipped
	// ClipEffectClippedOut: the clip definitely eliminates all of the draw's coverage and the draw can be skipped.
	ClipEffectClippedOut
)

// PreClipResult is the outcome of a preliminary, conservative clip analysis.
type PreClipResult struct {
	RRect   geom.RRect // Ignore if IsRRect is false.
	Effect  ClipEffect
	AA      gpu.AA // Ignore if IsRRect is false.
	IsRRect bool
}

// MakePreClipResult returns a PreClipResult carrying only an effect (no rrect).
func MakePreClipResult(effect ClipEffect) PreClipResult {
	return PreClipResult{Effect: effect}
}

// MakePreClipResultRRect returns a clipped PreClipResult reporting the clip as rrect/aa.
func MakePreClipResultRRect(rrect geom.RRect, aa gpu.AA) PreClipResult {
	return PreClipResult{Effect: ClipEffectClipped, RRect: rrect, AA: aa, IsRRect: true}
}

// MakePreClipResultRect returns a clipped PreClipResult reporting the clip as a degenerate rrect.
func MakePreClipResultRect(rect geom.Rect, aa gpu.AA) PreClipResult {
	return MakePreClipResultRRect(geom.MakeRRect(rect, 0, 0), aa)
}

// Clip is an abstract base for applying a clip, constructing a clip mask if necessary and filling out an AppliedClip
// instructing the caller on how to set up the draw state.
type Clip interface {
	// GetConservativeBounds returns a conservative pixel bounds restricted to the given render target dimensions;
	// anything outside will be entirely clipped out.
	GetConservativeBounds() geom.IRect
	// Apply computes an AppliedClip from the clip. On input 'bounds' is a conservative bounds of the draw to be
	// clipped; if Clipped or Unclipped is returned it has been updated to be contained within the clip bounds. The op
	// parameter lets clip implementations offer the op a chance to clip its own geometry (as ClipStack does).
	Apply(sdc *SurfaceDrawContext, op DrawOp, aa gpu.AAType, out *AppliedClip,
		bounds *geom.Rect) ClipEffect
	// PreApply performs preliminary, conservative analysis on the draw bounds as if it were provided to Apply.
	PreApply(drawBounds geom.Rect, aa gpu.AA) PreClipResult
}

const (
	// clipBoundsTolerance is the maximum distance a draw may extend beyond a clip's boundary and still count as "on the
	// other side", leaving slack for floating point rounding error.
	clipBoundsTolerance = 1e-3
	// clipHalfPixelRoundingTolerance is slack around a half-pixel vertex coordinate where GPU rasterizer rounding is
	// not trusted.
	clipHalfPixelRoundingTolerance = 5e-2
)

// ClipBoundsType selects the rounding mode for GetPixelIBounds.
type ClipBoundsType int32

// ClipBoundsType values.
const (
	// ClipBoundsExterior: the tightest integer pixel bounding box containing all non-zero coverage from rasterizing the
	// analytic bounds with the given aa method.
	ClipBoundsExterior ClipBoundsType = iota
	// ClipBoundsInterior: the largest integer pixel bounding box such that every pixel is fully covered were the bounds
	// rendered as a solid fill.
	ClipBoundsInterior
)

// GetPixelIBounds converts the analytic bounds of a shape into integer pixel bounds, where the given aa type is used
// when the shape is rendered.
func GetPixelIBounds(bounds geom.Rect, aa gpu.AA, mode ClipBoundsType) geom.IRect {
	roundLow := func(v float32) int32 {
		v += clipBoundsTolerance
		if aa == gpu.AANo {
			return geom.RoundToInt(v - clipHalfPixelRoundingTolerance)
		}
		return geom.FloorToInt(v)
	}
	roundHigh := func(v float32) int32 {
		v -= clipBoundsTolerance
		if aa == gpu.AANo {
			return geom.RoundToInt(v + clipHalfPixelRoundingTolerance)
		}
		return geom.CeilToInt(v)
	}

	if bounds.IsEmpty() {
		return geom.IRect{}
	}

	if mode == ClipBoundsExterior {
		return geom.IRect{
			Left: roundLow(bounds.Left), Top: roundLow(bounds.Top),
			Right: roundHigh(bounds.Right), Bottom: roundHigh(bounds.Bottom),
		}
	}
	return geom.IRect{
		Left: roundHigh(bounds.Left), Top: roundHigh(bounds.Top),
		Right: roundLow(bounds.Right), Bottom: roundLow(bounds.Bottom),
	}
}

// IsInsideClip reports whether the given draw bounds count as entirely inside the clip. innerClipBounds is a
// device-space rect fully contained by the clip.
func IsInsideClip(innerClipBounds geom.IRect, drawBounds geom.Rect, aa gpu.AA) bool {
	return innerClipBounds.ContainsRect(GetPixelIBounds(drawBounds, aa, ClipBoundsExterior))
}

// IsOutsideClip reports whether the given draw bounds count as entirely outside the clip. outerClipBounds is a
// device-space rect that contains the clip.
func IsOutsideClip(outerClipBounds geom.IRect, drawBounds geom.Rect, aa gpu.AA) bool {
	return !outerClipBounds.Intersects(GetPixelIBounds(drawBounds, aa, ClipBoundsExterior))
}

// IsPixelAligned reports whether rect's edges all sit on integer pixel boundaries.
func IsPixelAligned(rect geom.Rect) bool {
	nearlyInt := func(v float32) bool {
		return math.Abs(math.Round(float64(v))-float64(v)) <= clipBoundsTolerance
	}
	return nearlyInt(rect.Left) && nearlyInt(rect.Top) && nearlyInt(rect.Right) &&
		nearlyInt(rect.Bottom)
}

// clipPreApplyDefault is the default PreApply implementation shared by simple Clip types.
func clipPreApplyDefault(c Clip, drawBounds geom.Rect, aa gpu.AA) PreClipResult {
	pixelBounds := GetPixelIBounds(drawBounds, aa, ClipBoundsExterior)
	if !pixelBounds.Intersects(c.GetConservativeBounds()) {
		return MakePreClipResult(ClipEffectClippedOut)
	}
	return MakePreClipResult(ClipEffectClipped)
}

// hardClipApply implements Apply for hard clips: hard clips never use coverage FPs and only touch the fixed-function
// state.
func hardClipApply(applyHard func(out *AppliedHardClip, bounds *geom.IRect) ClipEffect, aa gpu.AAType, out *AppliedClip, bounds *geom.Rect) ClipEffect {
	pixelBounds := GetPixelIBounds(*bounds, gpu.AA(aa != gpu.AATypeNone), ClipBoundsExterior)
	effect := applyHard(out.HardClip(), &pixelBounds)
	bounds.Intersect(pixelBounds.ToRect())
	return effect
}

// FixedClip is a hard clip implemented with a scissor (window rectangles trimmed).
type FixedClip struct {
	scissorState gpu.ScissorState
}

// NewFixedClip returns a FixedClip whose scissor covers rtDims (initially disabled).
func NewFixedClip(rtDims geom.ISize) *FixedClip {
	return &FixedClip{scissorState: gpu.MakeScissorState(rtDims)}
}

// NewFixedClipRect returns a FixedClip whose scissor is set to scissorRect within rtDims.
func NewFixedClipRect(rtDims geom.ISize, scissorRect geom.IRect) *FixedClip {
	c := NewFixedClip(rtDims)
	if !c.scissorState.Set(scissorRect) {
		panic("scissor rect does not intersect the render target")
	}
	return c
}

// ScissorState returns the scissor rectangle state.
func (c *FixedClip) ScissorState() *gpu.ScissorState { return &c.scissorState }

// ScissorEnabled reports whether the scissor test is enabled.
func (c *FixedClip) ScissorEnabled() bool { return c.scissorState.Enabled() }

// ScissorRect returns the scissor rect, or rt bounds if the scissor test is not enabled.
func (c *FixedClip) ScissorRect() geom.IRect { return c.scissorState.Rect() }

// DisableScissor disables the scissor test.
func (c *FixedClip) DisableScissor() { c.scissorState.SetDisabled() }

// SetScissor sets the scissor rectangle directly.
func (c *FixedClip) SetScissor(irect geom.IRect) bool { return c.scissorState.Set(irect) }

// Intersect intersects the scissor rectangle with irect.
func (c *FixedClip) Intersect(irect geom.IRect) bool { return c.scissorState.Intersect(irect) }

// GetConservativeBounds implements Clip.
func (c *FixedClip) GetConservativeBounds() geom.IRect { return c.scissorState.Rect() }

// PreApply implements Clip.
func (c *FixedClip) PreApply(drawBounds geom.Rect, aa gpu.AA) PreClipResult {
	pixelBounds := GetPixelIBounds(drawBounds, aa, ClipBoundsExterior)
	if !c.scissorState.Rect().Intersects(pixelBounds) {
		return MakePreClipResult(ClipEffectClippedOut)
	}

	if !c.scissorState.Enabled() || c.scissorState.Rect().ContainsRect(pixelBounds) {
		// Either no scissor or the scissor doesn't clip the draw.
		return MakePreClipResult(ClipEffectUnclipped)
	}
	// Report the scissor as a degenerate round rect.
	return MakePreClipResultRect(c.scissorState.Rect().ToRect(), gpu.AANo)
}

// Apply implements Clip.
func (c *FixedClip) Apply(_ *SurfaceDrawContext, _ DrawOp, aa gpu.AAType, out *AppliedClip, bounds *geom.Rect) ClipEffect {
	return hardClipApply(c.applyHard, aa, out, bounds)
}

// applyHard restricts out's scissor to the intersection of the fixed clip's scissor and bounds.
func (c *FixedClip) applyHard(out *AppliedHardClip, bounds *geom.IRect) ClipEffect {
	if !c.scissorState.Rect().Intersects(*bounds) {
		return ClipEffectClippedOut
	}

	effect := ClipEffectUnclipped
	if c.scissorState.Enabled() && !c.scissorState.Rect().ContainsRect(*bounds) {
		if !bounds.Intersect(c.scissorState.Rect()) {
			panic("bounds must intersect the scissor here")
		}
		out.SetScissor(*bounds)
		effect = ClipEffectClipped
	}
	return effect
}
