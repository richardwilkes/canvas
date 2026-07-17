// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The path boolean operation entry points: Op, Simplify, and Builder. The operator/fill-type algebra
// (opInverse/outInverse), the rect-intersect and empty-operand fast paths, Simplify's convex fast path with
// pathIsTrivial, and the builder's first-op padding rule live here. The boolean engine behind them is implemented in
// op.go/simplify.go: Op and Simplify call runOp/runSimplify, which build the opContour data model from the edge
// builder, bounds-sort it (sortContourList), intersect every segment pair (addIntersections), resolve coincidence
// (handleCoincidence), and trace the resolved graph into an output path (bridgeOp/bridgeWinding/bridgeXor +
// pathWriter). Results carry curves exactly, with no flattening or snapping.

package pathops

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// PathOp selects the boolean combination performed by Op.
type PathOp int32

const (
	// Difference subtracts the op path from the first path.
	Difference PathOp = iota
	// Intersect intersects the two paths.
	Intersect
	// Union unions (inclusive-or) the two paths.
	Union
	// XOR exclusive-ors the two paths.
	XOR
	// ReverseDifference subtracts the first path from the op path.
	ReverseDifference
)

// opInverse gives the equivalent non-inverse operator for each combination of operator and operand inverse fill types,
// indexed [op][oneIsInverse][twoIsInverse].
var opInverse = [ReverseDifference + 1][2][2]PathOp{
	{{Difference, Intersect}, {Union, ReverseDifference}},
	{{Intersect, Difference}, {ReverseDifference, Union}},
	{{Union, ReverseDifference}, {Difference, Intersect}},
	{{XOR, XOR}, {XOR, XOR}},
	{{ReverseDifference, Union}, {Intersect, Difference}},
}

// outInverse reports whether the result is inverse-filled, indexed with the already-remapped operator and the original
// operand inverse fill types.
var outInverse = [ReverseDifference + 1][2][2]bool{
	{{false, false}, {true, false}}, // diff
	{{false, false}, {false, true}}, // sect
	{{false, true}, {true, true}},   // union
	{{false, true}, {true, false}},  // xor
	{{false, true}, {false, false}}, // rev diff
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Op computes the boolean combination of two paths. On success the returned path describes the result region with an
// even-odd (or inverse even-odd) fill; on failure it returns (nil, false) and the caller's result path is left
// unmodified.
func Op(one, two *path.Path, op PathOp) (*path.Path, bool) {
	if op < Difference || op > ReverseDifference {
		return nil, false
	}
	oneInv, twoInv := b2i(one.IsInverseFillType()), b2i(two.IsInverseFillType())
	op = opInverse[op][oneInv][twoInv]
	fillType := path.FillEvenOdd
	if outInverse[op][oneInv][twoInv] {
		fillType = path.FillInverseEvenOdd
	}
	// Rect-intersect fast path. This runs on the remapped operator, so inverse-filled rect geometry can reach it too
	// (for example difference against an inverse rect remaps to intersect); IsRect ignores fill types.
	if op == Intersect {
		if rect1, ok1 := one.IsRect(); ok1 {
			if rect2, ok2 := two.IsRect(); ok2 {
				result := path.New()
				if rect1.Intersect(rect2) {
					result.AddRect(rect1, geom.DirectionCW)
				}
				result.SetFillType(fillType)
				return result, true
			}
		}
	}
	if one.IsEmpty() || two.IsEmpty() {
		work := path.New()
		switch op {
		case Intersect:
		case Union, XOR:
			if one.IsEmpty() {
				work = two.Clone()
			} else {
				work = one.Clone()
			}
		case Difference:
			if !one.IsEmpty() {
				work = one.Clone()
			}
		case ReverseDifference:
			if !two.IsEmpty() {
				work = two.Clone()
			}
		}
		if (fillType == path.FillInverseEvenOdd) != work.IsInverseFillType() {
			work.ToggleInverseFillType()
		}
		return Simplify(work)
	}
	// The edge builder rejects non-finite operands (unparseable), failing the whole operation.
	if !one.IsFinite() || !two.IsFinite() {
		return nil, false
	}
	minuend, subtrahend := one, two
	if op == ReverseDifference {
		minuend, subtrahend = subtrahend, minuend
		op = Difference
	}
	return runOp(minuend, subtrahend, op, fillType)
}

// Simplify rewrites the path as a set of non-overlapping contours describing the same region, with an even-odd (or
// inverse even-odd) fill.
func Simplify(p *path.Path) (*path.Path, bool) {
	// Returns even-odd regardless of the input rule, inverse-ness preserved.
	fillType := path.FillEvenOdd
	if p.IsInverseFillType() {
		fillType = path.FillInverseEvenOdd
	}
	if p.GetConvexity().IsConvex() {
		// If the path is trivially convex, simplify to empty, else copy.
		result := path.New()
		if !pathIsTrivial(p) {
			result = p.Clone()
		}
		result.SetFillType(fillType)
		return result, true
	}
	if !p.IsFinite() {
		return nil, false
	}
	return runSimplify(p, fillType)
}

// pathIsTrivial reports whether every contour's points are coincident or collinear, so that a convex path covers no
// area. Points within each verb are visited last-first, and consecutive edge vectors are compared via their cross
// product to detect any turn (a nonzero cross product means the contour isn't degenerate).
func pathIsTrivial(p *path.Path) bool {
	var prevPt, prevVec geom.Point
	addTrivialContourPoint := func(currPt geom.Point) bool {
		if currPt == prevPt {
			return true
		}
		currVec := geom.Point{X: currPt.X - prevPt.X, Y: currPt.Y - prevPt.Y}
		if prevVec.X*currVec.Y-prevVec.Y*currVec.X != 0 {
			return false
		}
		prevVec = currVec
		prevPt = currPt
		return true
	}
	it := path.NewIter(p, true)
	for {
		verb, pts, ok := it.Next()
		if !ok {
			return true
		}
		switch verb {
		case path.VerbMove:
			prevPt = pts[0]
			prevVec = geom.Point{}
		case path.VerbCubic:
			if !addTrivialContourPoint(pts[3]) || !addTrivialContourPoint(pts[2]) ||
				!addTrivialContourPoint(pts[1]) || !addTrivialContourPoint(pts[0]) {
				return false
			}
		case path.VerbConic, path.VerbQuad:
			if !addTrivialContourPoint(pts[2]) || !addTrivialContourPoint(pts[1]) ||
				!addTrivialContourPoint(pts[0]) {
				return false
			}
		case path.VerbLine:
			if !addTrivialContourPoint(pts[1]) || !addTrivialContourPoint(pts[0]) {
				return false
			}
		case path.VerbClose:
		}
	}
}

// Builder accumulates paths and operators, then resolves them all at once. Resolve resets the builder.
type Builder struct {
	paths []*path.Path
	ops   []PathOp
}

// Add appends a path and the operator to combine it with the accumulated result. The builder is empty before the first
// path is added, so the result of a single add with a non-union operator is (emptyPath op path).
func (b *Builder) Add(p *path.Path, op PathOp) {
	if len(b.ops) == 0 && op != Union {
		b.paths = append(b.paths, path.New())
		b.ops = append(b.ops, Union)
	}
	b.paths = append(b.paths, p.Clone())
	b.ops = append(b.ops, op)
}

func (b *Builder) reset() {
	b.paths, b.ops = nil, nil
}

// allUnion is Resolve's optimization gate: it reports whether every operator is a union over non-inverse paths, every
// convex operand has a determinable direction, and every non-convex operand's bounds are disjoint from all earlier
// operands. It stops scanning at the first failing operand rather than continuing to check the rest, since the caller
// only needs a yes/no answer; this is safe because winding orientation is normalized away downstream regardless — the
// pairwise Op path re-derives winding, and the all-union path runs each operand through Simplify (an
// orientation-independent even-odd computation).
func allUnion(paths []*path.Path, ops []PathOp) bool {
	firstDir := path.FirstDirectionUnknown
	for i, p := range paths {
		if ops[i] != Union || p.IsInverseFillType() {
			return false
		}
		if p.GetConvexity().IsConvex() {
			dir := p.ComputeFirstDirection()
			if dir == path.FirstDirectionUnknown {
				return false
			}
			if firstDir == path.FirstDirectionUnknown {
				firstDir = dir
			}
			continue
		}
		// If the path is not convex but its bounds do not intersect the others, simplify is enough.
		testBounds := p.Bounds()
		for inner := 0; inner < i; inner++ {
			if paths[inner].Bounds().Intersects(testBounds) {
				return false
			}
		}
	}
	return true
}

// Resolve computes the accumulated boolean combination of every path added via Add, then resets the builder. When the
// allUnion gate holds (every operator is a union over non-inverse paths), it takes the fast path: simplify each operand
// to even-odd form, convert it back to winding form via fixWinding so overlapping unions reinforce rather than cancel,
// accumulate, and simplify the sum once. Otherwise it folds the operands pairwise through Op, where a lone survivor
// passes through unmodified with its original fill type. An empty builder resolves to Simplify(empty), since allUnion
// is vacuously true and the accumulation loop is empty.
func (b *Builder) Resolve() (*path.Path, bool) {
	paths, ops := b.paths, b.ops
	b.reset()
	if !allUnion(paths, ops) {
		result := paths[0]
		for i := 1; i < len(paths); i++ {
			res, ok := Op(result, paths[i], ops[i])
			if !ok {
				return nil, false
			}
			result = res
		}
		return result, true
	}
	sum := path.New()
	for _, p := range paths {
		simplified, ok := Simplify(p)
		if !ok {
			return nil, false
		}
		if simplified.IsEmpty() {
			continue
		}
		wound, ok := fixWinding(simplified)
		if !ok {
			return nil, false
		}
		sum.AddPath(wound, path.AddPathAppend)
	}
	return Simplify(sum)
}
