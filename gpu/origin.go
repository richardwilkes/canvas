// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SurfaceOrigin and ScissorState: the scissor rectangle plus the backing-store bounds of the render target it applies
// to. Whether the scissor test is "enabled" is derived from the rect not covering the render target — a failed
// intersect leaves an empty, still-enabled scissor.

package gpu

import "github.com/richardwilkes/canvas/geom"

// SurfaceOrigin identifies which corner of a surface is the origin of its device space. GL's device space has the
// origin at bottom-left; wrapped FBOs from the host embedder also use bottom-left.
type SurfaceOrigin int32

// SurfaceOrigin values.
const (
	OriginTopLeft SurfaceOrigin = iota
	OriginBottomLeft
)

// ScissorState tracks a scissor rectangle relative to a render target's backing-store dimensions. Use MakeScissorState
// so the render-target size is tracked.
type ScissorState struct {
	rtSize geom.ISize
	rect   geom.IRect
}

// MakeScissorState returns a ScissorState initially disabled, covering the full surface.
func MakeScissorState(dims geom.ISize) ScissorState {
	return ScissorState{rtSize: dims, rect: geom.IRectSize(dims)}
}

// SetDisabled resets the scissor rect to cover the whole render target (disabling the test).
func (s *ScissorState) SetDisabled() {
	s.rect = geom.IRectSize(s.rtSize)
}

// Set replaces the scissor rect with rect, first resetting to the full render target.
func (s *ScissorState) Set(rect geom.IRect) bool {
	s.SetDisabled()
	return s.Intersect(rect)
}

// Intersect intersects the running scissor with rect. On failure the scissor becomes empty (which still tests as
// enabled).
func (s *ScissorState) Intersect(rect geom.IRect) bool {
	if !s.rect.Intersect(rect) {
		s.rect = geom.IRect{}
		return false
	}
	return true
}

// RelaxTest discards the scissor test when the scissor was configured for the backing store dimensions, it's acceptable
// to draw outside the logical dimensions of the target, and doing so wouldn't modify the logical dimensions.
func (s *ScissorState) RelaxTest(logicalDimensions geom.ISize) bool {
	if s.rect.Left == 0 && s.rect.Top == 0 && s.rect.Right >= logicalDimensions.Width &&
		s.rect.Bottom >= logicalDimensions.Height {
		s.SetDisabled()
		return true
	}
	return false
}

// Equal reports whether two scissor states have the same render-target size and rect.
func (s *ScissorState) Equal(other *ScissorState) bool {
	return s.rtSize == other.rtSize && s.rect == other.rect
}

// Enabled reports whether the scissor test is enabled: the scissor is enabled when the rectangle does not cover the
// render target.
func (s *ScissorState) Enabled() bool {
	return s.rect.Left > 0 || s.rect.Top > 0 ||
		s.rect.Right < s.rtSize.Width || s.rect.Bottom < s.rtSize.Height
}

// Rect returns the current scissor rectangle: always equal to or contained in the render-target bounds, or empty if
// scissor rectangles were added that did not intersect the render target or a prior scissor.
func (s *ScissorState) Rect() geom.IRect { return s.rect }
