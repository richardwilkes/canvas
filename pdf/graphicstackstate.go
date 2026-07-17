// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// graphicStackState is the small (depth <= 2) stack of q/Q graphics states the content generator maintains as it lowers
// draw ops. It tracks the current CTM, clip (by ClipStack gen-ID), fill/stroke color (or pattern), applied /ExtGState,
// and text horizontal scale, emitting only the operators needed to move from the current state to each draw's state
// (updateClip/updateMatrix/updateDrawingState). Not to be confused with the /ExtGState objects built in
// graphicstate.go.

package pdf

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/pathops"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stream"
)

// gsMaxStackDepth is the deepest the q/Q stack ever needs to go: one level for the matrix, one for the clip.
const gsMaxStackDepth = 2

// gsEntry is one level of the q/Q stack: the CTM, clip gen-ID, fill/stroke color or pattern, applied graphic state, and
// text horizontal scale in effect at that depth.
type gsEntry struct {
	matrix            geom.Matrix
	color             colorcore.Color4f
	textScaleX        float32
	clipStackGenID    uint32
	shaderIndex       int
	graphicStateIndex int
}

// nanColor4f is the sentinel entry color ({NaN,NaN,NaN,NaN}); any real color compares unequal to it, so the first
// updateDrawingState always emits a color.
var nanColor4f = colorcore.Color4f{
	R: float32(math.NaN()), G: float32(math.NaN()), B: float32(math.NaN()), A: float32(math.NaN()),
}

func defaultGSEntry() gsEntry {
	return gsEntry{
		matrix:            geom.IdentityMatrix(),
		clipStackGenID:    wideOpenGenID,
		color:             nanColor4f,
		textScaleX:        1, // Zero means "don't care".
		shaderIndex:       -1,
		graphicStateIndex: -1,
	}
}

// graphicStackState tracks the emitted q/Q nesting and per-level graphics state described above.
type graphicStackState struct {
	contentStream stream.WStream
	entries       [gsMaxStackDepth + 1]gsEntry
	stackDepth    int
}

// newGraphicStackState creates a graphicStackState that writes q/Q and state-transition operators to s.
func newGraphicStackState(s stream.WStream) *graphicStackState {
	g := &graphicStackState{contentStream: s}
	for i := range g.entries {
		g.entries[i] = defaultGSEntry()
	}
	return g
}

func (g *graphicStackState) currentEntry() *gsEntry { return &g.entries[g.stackDepth] }

func (g *graphicStackState) push() {
	writeText(g.contentStream, "q\n")
	g.stackDepth++
	g.entries[g.stackDepth] = g.entries[g.stackDepth-1]
}

func (g *graphicStackState) pop() {
	writeText(g.contentStream, "Q\n")
	g.entries[g.stackDepth] = defaultGSEntry()
	g.stackDepth--
}

// drainStack pops every remaining q level, restoring the content stream to depth 0.
func (g *graphicStackState) drainStack() {
	if g.contentStream != nil {
		for g.stackDepth > 0 {
			g.pop()
		}
	}
}

// updateMatrix emits whatever q/cm operators are needed so the content stream's CTM becomes matrix.
func (g *graphicStackState) updateMatrix(matrix *geom.Matrix) {
	if matrixEqual(matrix, &g.currentEntry().matrix) {
		return
	}
	if g.currentEntry().matrix.Type() != geom.TypeIdentity {
		g.pop()
	}
	if matrix.Type() == geom.TypeIdentity {
		return
	}
	g.push()
	appendTransform(matrix, g.contentStream)
	g.currentEntry().matrix = *matrix
}

// updateDrawingState emits whatever color/pattern, graphic-state, and text-scale operators are needed so the content
// stream's drawing state matches state.
func (g *graphicStackState) updateDrawingState(state *gsEntry) {
	// PDF treats a shader as a color, so we only set one or the other.
	if state.shaderIndex >= 0 {
		if state.shaderIndex != g.currentEntry().shaderIndex {
			applyPattern(state.shaderIndex, g.contentStream)
			g.currentEntry().shaderIndex = state.shaderIndex
		}
	} else if state.color != g.currentEntry().color || g.currentEntry().shaderIndex >= 0 {
		emitPDFColor(state.color, g.contentStream)
		writeText(g.contentStream, "RG ")
		emitPDFColor(state.color, g.contentStream)
		writeText(g.contentStream, "rg\n")
		g.currentEntry().color = state.color
		g.currentEntry().shaderIndex = -1
	}

	if state.graphicStateIndex != g.currentEntry().graphicStateIndex {
		applyGraphicState(state.graphicStateIndex, g.contentStream)
		g.currentEntry().graphicStateIndex = state.graphicStateIndex
	}

	if state.textScaleX != 0 {
		if state.textScaleX != g.currentEntry().textScaleX {
			pdfScale := state.textScaleX * 100
			writeScalar(g.contentStream, pdfScale)
			writeText(g.contentStream, " Tz\n")
			g.currentEntry().textScaleX = state.textScaleX
		}
	}
}

// updateClip emits whatever q/W operators are needed so the content stream's clip matches clipStack, intersected with
// bounds.
func (g *graphicStackState) updateClip(clipStack *ClipStack, bounds geom.IRect) {
	clipStackGenID := wideOpenGenID
	if clipStack != nil {
		clipStackGenID = clipStack.getTopmostGenID()
	}
	if clipStackGenID == g.currentEntry().clipStackGenID {
		return
	}
	for g.stackDepth > 0 {
		g.pop()
		if clipStackGenID == g.currentEntry().clipStackGenID {
			return
		}
	}
	if clipStackGenID != wideOpenGenID {
		g.push()
		g.currentEntry().clipStackGenID = clipStackGenID
		appendClip(clipStack, bounds, g.contentStream)
	}
}

// matrixEqual reports whether two matrices are element-wise equal.
func matrixEqual(a, b *geom.Matrix) bool {
	av, bv := a.As9(), b.As9()
	return av == bv
}

// emitPDFColor writes color's RGB components as space-separated PDF operands (alpha is handled elsewhere).
func emitPDFColor(color colorcore.Color4f, result stream.WStream) {
	result.Write(appendColorComponentF(nil, color.R))
	writeText(result, " ")
	result.Write(appendColorComponentF(nil, color.G))
	writeText(result, " ")
	result.Write(appendColorComponentF(nil, color.B))
	writeText(result, " ")
}

// ---- appendClip and helpers ---------------------------------------------------------------------------

// rectPath returns a clockwise-wound rect path.
func rectPath(r geom.Rect) *path.Path { return (&path.Path{}).AddRect(r, geom.DirectionCW) }

// rectIntersect returns the intersection of u and v, or an empty rect if either is empty or they don't overlap.
func rectIntersect(u, v geom.Rect) geom.Rect {
	if u.IsEmpty() || v.IsEmpty() {
		return geom.Rect{}
	}
	if u.Intersect(v) {
		return u
	}
	return geom.Rect{}
}

// isRectClip reports whether the whole clip stack reduces to an intersection of device-space rects, returning that
// equivalent rect when it does.
func isRectClip(cs *ClipStack, bounds geom.Rect) (geom.Rect, bool) {
	currentClip := bounds
	for _, element := range cs.elements() {
		var elementRect geom.Rect
		switch element.deviceSpaceType {
		case clipEmpty:
			// elementRect stays empty.
		case clipRect:
			elementRect = element.deviceSpaceRect
		default:
			return geom.Rect{}, false
		}
		switch {
		case element.isReplace:
			currentClip = rectIntersect(bounds, elementRect)
		case element.op == raster.ClipIntersect:
			currentClip = rectIntersect(currentClip, elementRect)
		default:
			return geom.Rect{}, false
		}
	}
	return currentClip, true
}

// isComplexClip reports whether the clip stack contains a replace op, which rules out the incremental intersect-only
// clip fast paths.
func isComplexClip(cs *ClipStack) bool {
	for _, element := range cs.elements() {
		if element.isReplace {
			return true
		}
	}
	return false
}

// clipHuge is the "large enough to not bother clipping against" bound used by applyClip.
var clipHuge = geom.Rect{Left: -30000, Top: -30000, Right: 30000, Bottom: 30000}

// applyClip calls fn with each clip stack element's path, in order, intersected with the shrinking running bounds; it
// assumes the stack is not complex (no replace ops).
func applyClip(cs *ClipStack, outerBounds geom.Rect, fn func(*path.Path)) {
	bounds := outerBounds
	for _, element := range cs.elements() {
		operand := element.asDeviceSpacePath()
		op := pathops.Intersect
		if element.op == raster.ClipDifference {
			op = pathops.Difference
		}
		if op == pathops.Difference || operand.IsInverseFillType() ||
			!clipHuge.ContainsRect(operand.Bounds()) {
			if result, ok := pathops.Op(rectPath(bounds), operand, op); ok {
				operand = result
			}
		}
		fn(operand)
		if !bounds.Intersect(operand.Bounds()) {
			return
		}
	}
}

// appendClipPath emits clipPath as a fill followed by a clip (W/W*) operator; inverse fills are unreachable here.
func appendClipPath(clipPath *path.Path, wStream stream.WStream) {
	emitPath(clipPath, styleFill, true, wStream, 0.25)
	if clipPath.FillType() == path.FillEvenOdd {
		writeText(wStream, "W* n\n")
	} else {
		writeText(wStream, "W n\n")
	}
}

// clipStackAsPath folds the whole clip stack down into a single path via successive path-op combines.
func clipStackAsPath(cs *ClipStack) *path.Path {
	p := &path.Path{}
	p.SetFillType(path.FillInverseEvenOdd)
	for _, element := range cs.elements() {
		var operand *path.Path
		if element.deviceSpaceType != clipEmpty {
			operand = element.asDeviceSpacePath()
		} else {
			operand = &path.Path{}
		}
		if element.isReplace {
			p = operand
		} else {
			op := pathops.Intersect
			if element.op == raster.ClipDifference {
				op = pathops.Difference
			}
			if result, ok := pathops.Op(p, operand, op); ok {
				p = result
			}
		}
	}
	return p
}

// appendClip emits the operators that set the content stream's clip to clipStack intersected with bounds, taking the
// cheapest applicable path: a plain rect, a path-op combine for a complex (replace-containing) stack, or an incremental
// intersect-only clip otherwise.
func appendClip(clipStack *ClipStack, bounds geom.IRect, wStream stream.WStream) {
	// Slightly outset for float accuracy / region-bitmap approximations.
	outsetBounds := bounds.Inset(-1, -1).ToRect()

	if clipStackRect, ok := isRectClip(clipStack, outsetBounds); ok {
		appendRectangle(clipStackRect, wStream)
		writeText(wStream, "W* n\n")
		return
	}

	if isComplexClip(clipStack) {
		if clipPath, ok := pathops.Op(clipStackAsPath(clipStack), rectPath(outsetBounds),
			pathops.Intersect); ok {
			appendClipPath(clipPath, wStream)
		}
		// If Op() fails (pathological input), emit no clip at all.
	} else {
		applyClip(clipStack, outsetBounds, func(p *path.Path) {
			appendClipPath(p, wStream)
		})
	}
}
