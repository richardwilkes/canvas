// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package shaders implements the shader model: the shader descriptors (color, blend, local-matrix, and the four
// gradient families) and their CPU evaluation. Descriptors are kept cleanly separated from evaluation — the GPU
// fragment processors consume the same descriptor data — and evaluation builds the raster-pipeline highp stages each
// shader appends, assembled into a Pipeline by Compile.
package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// TileMode selects how a shader repeats outside its primary domain.
type TileMode uint8

// TileMode values.
const (
	TileClamp TileMode = iota
	TileRepeat
	TileMirror
	TileDecal
)

// tileModeCount is the number of TileMode values.
const tileModeCount = 4

// Shader is the common contract every shader implements. Concrete shaders are immutable descriptors; evaluation happens
// through Compile.
type Shader interface {
	// IsOpaque reports whether the shader always produces fully-opaque output.
	IsOpaque() bool

	// IsConstant reports whether the shader always produces the same color regardless of position.
	IsConstant() bool

	// appendStages appends this shader's raster-pipeline stages. It returns false if the shader cannot draw (the empty
	// shader, a non-invertible total matrix). m is deliberately passed by value: a pointer handed through this
	// interface call would defeat escape analysis and heap-allocate a MatrixRec at every level of the shader tree on
	// every compile, so implementations suppress gocritic's hugeParam check instead.
	appendStages(p *Pipeline, m MatrixRec) bool
}

// MatrixRec tracks, for the raster-pipeline leg, the CTM and the local matrices accumulated by wrapper shaders between
// the root and the point where a shader consumes coordinates, plus whether the total matrix still describes the
// coordinates in the registers (a runtime-effect kernel sampling a child at computed coordinates invalidates it).
type MatrixRec struct {
	ctm          geom.Matrix
	pendingLocal geom.Matrix
	ctmApplied   bool
	totalInvalid bool
}

// newMatrixRec builds a MatrixRec seeded with ctm.
func newMatrixRec(ctm geom.Matrix) MatrixRec {
	return MatrixRec{ctm: ctm, pendingLocal: geom.IdentityMatrix()}
}

// concat returns a copy with an additional local matrix folded into pendingLocal: pendingLocal' = pendingLocal * lm.
func (m *MatrixRec) concat(lm *geom.Matrix) MatrixRec {
	out := *m
	var pending geom.Matrix
	pending.SetConcat(&m.pendingLocal, lm)
	out.pendingLocal = pending
	return out
}

// coordsSeeded reports whether the pipeline's registers already hold seeded coordinates.
func (m *MatrixRec) coordsSeeded() bool { return m.ctmApplied }

// markTotalMatrixInvalid returns a modified copy with the total matrix marked invalid: a shader kernel applies it to
// children sampled with explicit (non-passthrough) coordinates so their sampling optimizations cannot assume the
// register coordinates come from the tracked matrices.
func (m *MatrixRec) markTotalMatrixInvalid() MatrixRec {
	out := *m
	out.totalInvalid = true
	return out
}

// totalMatrixValid reports whether the tracked total matrix still describes the register coordinates.
func (m *MatrixRec) totalMatrixValid() bool { return !m.totalInvalid }

// apply appends seed_shader (if the coords are not yet seeded) and a matrix stage for postInv * (ctm *
// pendingLocal)^-1, returning the updated copy. Returns ok=false when the total matrix is not invertible.
func (m *MatrixRec) apply(p *Pipeline, postInv *geom.Matrix) (MatrixRec, bool) {
	total := m.pendingLocal
	if !m.ctmApplied {
		var t geom.Matrix
		t.SetConcat(&m.ctm, &total)
		total = t
	}
	inv, ok := total.Invert()
	if !ok {
		return *m, false
	}
	var tot geom.Matrix
	tot.SetConcat(postInv, &inv)
	if !m.ctmApplied {
		p.appendSeed()
	}
	p.appendMatrix(&tot)
	out := *m
	out.pendingLocal = geom.IdentityMatrix()
	out.ctmApplied = true
	return out, true
}

// totalInverse returns the inverse of ctm * pendingLocal. Callers must check totalMatrixValid first — a shader kernel
// sampling a child at explicit coordinates invalidates the tracked total.
func (m *MatrixRec) totalInverse() (geom.Matrix, bool) {
	total := m.pendingLocal
	if !m.ctmApplied {
		var t geom.Matrix
		t.SetConcat(&m.ctm, &total)
		total = t
	}
	return total.Invert()
}

// NewMatrixRec builds a MatrixRec seeded with ctm, for the GPU fragment-processor leg (gpu/gl's shader→FP conversion
// walks wrappers with the same pending-matrix accumulation the CPU leg uses).
func NewMatrixRec(ctm geom.Matrix) MatrixRec { return newMatrixRec(ctm) }

// Concat is the exported form of concat, for the GPU FP leg.
func (m *MatrixRec) Concat(lm geom.Matrix) MatrixRec { return m.concat(&lm) }

// ApplyForFragmentProcessor returns the inverted pending local matrix (with postInv post-applied) that a GPU
// matrix-effect wrapper applies to the FP's sample coords. The coordinates provided to the root shader's FP are already
// in local space, so the CTM is never inverted here — only the pending local matrix is. Returns ok=false when it is not
// invertible.
func (m *MatrixRec) ApplyForFragmentProcessor(postInv *geom.Matrix) (geom.Matrix, bool) {
	if m.ctmApplied {
		panic("CTM already applied (FPs always receive raw local coords)")
	}
	inv, ok := m.pendingLocal.Invert()
	if !ok {
		return geom.IdentityMatrix(), false
	}
	var out geom.Matrix
	out.SetConcat(postInv, &inv)
	return out, true
}

// Compile appends the shader's root stages against ctm, then scales by the paint alpha when it is not 1 (a
// scale_1_float stage). It returns nil when the shader cannot draw with the raster pipeline — the caller maps that to a
// null blitter.
func Compile(s Shader, ctm geom.Matrix, paintColor colorcore.Color) *Pipeline {
	p := borrowPipeline()
	p.paintColor = colorcore.Color4fFromColor(paintColor)
	if !s.appendStages(p, newMatrixRec(ctm)) {
		RecyclePipeline(p)
		return nil
	}
	// paint_color_to_dst is the exact /255 conversion for sRGB-only content.
	if alpha := float32(paintColor.A()) * (1.0 / 255.0); alpha != 1 {
		p.appendScale1Float(alpha)
	}
	return p
}
