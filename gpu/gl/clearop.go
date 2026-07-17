// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ClearOp: color and stencil-clip clears recorded as ops when a load-op clear can't express them (scissored clears,
// clears after other ops).

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

var clearOpClassID = GenOpClassID()

// clearBuffer is a bitmask of which buffers a ClearOp clears.
type clearBuffer uint32

const (
	clearBufferColor       clearBuffer = 0b01
	clearBufferStencilClip clearBuffer = 0b10
)

// ClearOp clears the color and/or stencil-clip buffer within a scissor rect.
type ClearOp struct {
	OpBase
	scissor           gpu.ScissorState
	color             [4]float32
	stencilInsideMask bool
	buffer            clearBuffer
}

func newClearOp(buffer clearBuffer, scissor *gpu.ScissorState, color [4]float32, insideMask bool) *ClearOp {
	op := &ClearOp{
		scissor:           *scissor,
		color:             color,
		stencilInsideMask: insideMask,
		buffer:            buffer,
	}
	op.InitOp(clearOpClassID, op)
	op.SetBounds(scissor.Rect().ToRect(), HasAABloat(false), IsHairline(false))
	return op
}

// MakeColorClearOp creates a fullscreen or scissored color clear depending on the scissor state.
func MakeColorClearOp(scissor *gpu.ScissorState, color [4]float32) Op {
	return newClearOp(clearBufferColor, scissor, color, false)
}

// MakeStencilClipClearOp creates a fullscreen or scissored stencil-clip clear.
func MakeStencilClipClearOp(scissor *gpu.ScissorState, insideMask bool) Op {
	return newClearOp(clearBufferStencilClip, scissor, [4]float32{}, insideMask)
}

// Name implements Op.
func (o *ClearOp) Name() string { return "Clear" }

// Color returns the clear color (for tests).
func (o *ClearOp) Color() [4]float32 { return o.color }

// StencilInsideMask returns the stencil clear mode (for tests).
func (o *ClearOp) StencilInsideMask() bool { return o.stencilInsideMask }

// containsScissor reports whether a fully contains b.
func containsScissor(a, b *gpu.ScissorState) bool {
	return !a.Enabled() || (b.Enabled() && a.Rect().ContainsRect(b.Rect()))
}

// OnCombineIfPossible implements Op.
func (o *ClearOp) OnCombineIfPossible(t Op) CombineResult {
	other, ok := t.(*ClearOp)
	if !ok {
		panic("class ID matched a non-ClearOp")
	}
	if other.buffer == o.buffer {
		// This could be much more complicated. Currently we look at cases where the new clear contains the old clear,
		// or when the new clear is a subset of the old clear and they clear to the same value (color or stencil mask
		// depending on target).
		switch {
		case containsScissor(&other.scissor, &o.scissor):
			o.scissor = other.scissor
			o.color = other.color
			o.stencilInsideMask = other.stencilInsideMask
			return CombineResultMerged
		case other.color == o.color && other.stencilInsideMask == o.stencilInsideMask &&
			containsScissor(&o.scissor, &other.scissor):
			return CombineResultMerged
		}
	} else if other.scissor.Enabled() == o.scissor.Enabled() &&
		(!o.scissor.Enabled() || other.scissor.Rect() == o.scissor.Rect()) {
		// When the scissors are the exact same but the buffers are different, we can combine and clear both stencil and
		// color together in OnExecute.
		if other.buffer&clearBufferColor != 0 {
			o.color = other.color
		}
		if other.buffer&clearBufferStencilClip != 0 {
			o.stencilInsideMask = other.stencilInsideMask
		}
		o.buffer = clearBufferColor | clearBufferStencilClip
		return CombineResultMerged
	}
	return CombineResultCannotCombine
}

// OnPrepare implements Op (no-op for clears).
func (o *ClearOp) OnPrepare(*OpFlushState) {}

// OnExecute implements Op.
func (o *ClearOp) OnExecute(state *OpFlushState, _ geom.Rect) {
	if state.OpsRenderPass() == nil {
		panic("clear without a render pass")
	}
	if o.buffer&clearBufferColor != 0 {
		state.OpsRenderPass().Clear(&o.scissor, o.color)
	}
	if o.buffer&clearBufferStencilClip != 0 {
		state.OpsRenderPass().ClearStencilClip(&o.scissor, o.stencilInsideMask)
	}
}
