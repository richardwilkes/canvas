// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The set of modifications to the draw state that implement a clip — the fixed-function hard clip (scissor and a
// stencil-clip stack ID) and the coverage fragment processor. Window rectangles are not supported:
// EXT_window_rectangles is absent from this codebase's target drivers, so that state is unreachable; the stencil stack
// ID is carried for the stencil-clip lanes.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// invalidGenID marks a stencilStackID as having no stencil clip applied.
const invalidGenID uint32 = 0

// AppliedHardClip is produced by hard clips; only fixed-function state.
type AppliedHardClip struct {
	scissorState   gpu.ScissorState
	stencilStackID uint32
}

// MakeAppliedHardClip returns an AppliedHardClip whose scissor covers rtDims.
func MakeAppliedHardClip(rtDims geom.ISize) AppliedHardClip {
	return AppliedHardClip{
		scissorState:   gpu.MakeScissorState(rtDims),
		stencilStackID: invalidGenID,
	}
}

// MakeAppliedHardClipWithLogicalDims returns an AppliedHardClip whose scissor covers the backing store but is initially
// set to the logical dimensions.
func MakeAppliedHardClipWithLogicalDims(logicalRTDims, backingStoreDims geom.ISize) AppliedHardClip {
	c := AppliedHardClip{
		scissorState:   gpu.MakeScissorState(backingStoreDims),
		stencilStackID: invalidGenID,
	}
	c.scissorState.Set(geom.IRectSize(logicalRTDims))
	return c
}

// ScissorState returns the scissor rectangle state.
func (c *AppliedHardClip) ScissorState() *gpu.ScissorState { return &c.scissorState }

// StencilStackID returns the ID of the stencil-clip stack entry, if any.
func (c *AppliedHardClip) StencilStackID() uint32 { return c.stencilStackID }

// HasStencilClip reports whether a stencil clip has been applied.
func (c *AppliedHardClip) HasStencilClip() bool { return c.stencilStackID != invalidGenID }

// AddScissor intersects the applied clip with the provided rect, intersecting clippedDrawBounds with it as well.
// Returns false if the clip becomes empty or the draw no longer intersects the clip (the draw can be skipped).
func (c *AppliedHardClip) AddScissor(irect geom.IRect, clippedDrawBounds *geom.Rect) bool {
	return c.scissorState.Intersect(irect) && clippedDrawBounds.Intersect(irect.ToRect())
}

// SetScissor sets the scissor rectangle directly.
func (c *AppliedHardClip) SetScissor(irect geom.IRect) {
	c.scissorState.Set(irect)
}

// AddStencilClip records the stencil-clip stack entry to test against. Panics if one is already applied.
func (c *AppliedHardClip) AddStencilClip(stencilStackID uint32) {
	if c.stencilStackID != invalidGenID {
		panic("stencil clip already applied")
	}
	c.stencilStackID = stencilStackID
}

// DoesClip reports whether this hard clip restricts the draw at all.
func (c *AppliedHardClip) DoesClip() bool {
	return c.scissorState.Enabled() || c.HasStencilClip()
}

// Equal reports whether c and that have equivalent scissor and stencil-clip state.
func (c *AppliedHardClip) Equal(that *AppliedHardClip) bool {
	return c.scissorState.Equal(&that.scissorState) && c.stencilStackID == that.stencilStackID
}

// AppliedClip is the full set of modifications to the pipeline that implement a clip: the hard clip plus an optional
// coverage fragment processor.
type AppliedClip struct {
	coverageFP FragmentProcessor
	hardClip   AppliedHardClip
}

// MakeAppliedClip returns an AppliedClip whose hard clip covers rtDims.
func MakeAppliedClip(rtDims geom.ISize) AppliedClip {
	return AppliedClip{hardClip: MakeAppliedHardClip(rtDims)}
}

// MakeAppliedClipWithLogicalDims returns an AppliedClip whose hard clip covers the backing store but is initially set
// to the logical dimensions.
func MakeAppliedClipWithLogicalDims(logicalRTDims, backingStoreDims geom.ISize) AppliedClip {
	return AppliedClip{
		hardClip: MakeAppliedHardClipWithLogicalDims(logicalRTDims, backingStoreDims),
	}
}

// HardClip returns the fixed-function hard clip.
func (c *AppliedClip) HardClip() *AppliedHardClip { return &c.hardClip }

// ScissorState returns the hard clip's scissor rectangle state.
func (c *AppliedClip) ScissorState() *gpu.ScissorState { return c.hardClip.ScissorState() }

// StencilStackID returns the hard clip's stencil-clip stack entry ID.
func (c *AppliedClip) StencilStackID() uint32 { return c.hardClip.StencilStackID() }

// HasStencilClip reports whether the hard clip has a stencil clip applied.
func (c *AppliedClip) HasStencilClip() bool { return c.hardClip.HasStencilClip() }

// HasCoverageFragmentProcessor reports whether a coverage fragment processor has been added.
func (c *AppliedClip) HasCoverageFragmentProcessor() bool { return c.coverageFP != nil }

// CoverageFragmentProcessor returns the coverage fragment processor. Panics if none was added.
func (c *AppliedClip) CoverageFragmentProcessor() FragmentProcessor {
	if c.coverageFP == nil {
		panic("no coverage fragment processor")
	}
	return c.coverageFP
}

// DetachCoverageFragmentProcessor removes and returns the coverage fragment processor. Panics if none was added.
func (c *AppliedClip) DetachCoverageFragmentProcessor() FragmentProcessor {
	if c.coverageFP == nil {
		panic("no coverage fragment processor")
	}
	fp := c.coverageFP
	c.coverageFP = nil
	return fp
}

// AddCoverageFP adds a coverage fragment processor, composing it with any previously added one.
func (c *AppliedClip) AddCoverageFP(fp FragmentProcessor) {
	if c.coverageFP == nil {
		c.coverageFP = fp
	} else {
		// Compose this coverage FP with the previously-added coverage.
		c.coverageFP = ComposeFP(fp, c.coverageFP)
	}
}

// DoesClip reports whether this clip restricts the draw at all.
func (c *AppliedClip) DoesClip() bool {
	return c.hardClip.DoesClip() || c.coverageFP != nil
}

// Equal reports whether c and that have equivalent hard-clip and coverage-FP state.
func (c *AppliedClip) Equal(that *AppliedClip) bool {
	if !c.hardClip.Equal(&that.hardClip) ||
		c.HasCoverageFragmentProcessor() != that.HasCoverageFragmentProcessor() {
		return false
	}
	if c.coverageFP != nil && !fpIsEqual(c.coverageFP, that.coverageFP) {
		return false
	}
	return true
}

// VisitProxies calls fn for every proxy the coverage fragment processor references.
func (c *AppliedClip) VisitProxies(fn func(*SurfaceProxy, gpu.Mipmapped)) {
	if c.coverageFP != nil {
		fpVisitProxies(c.coverageFP, fn)
	}
}
