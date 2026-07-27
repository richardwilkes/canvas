// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The lattice iterator for the divs form DrawImageNine produces. Rect types/colors are trimmed: the public surface has
// no drawImageLattice entry point, so lattices always come from DrawImageNine, which passes neither.

package canvas

import "github.com/richardwilkes/canvas/geom"

// latticeSpec is the lattice description DrawImageNine builds: div positions plus the total bounds.
type latticeSpec struct {
	xDivs  []int32
	yDivs  []int32
	bounds geom.IRect
}

// validDivs reports whether the divs are strictly increasing and inside [start, end]. The inclusive start is
// deliberate: a first div equal to the bounds edge marks a degenerate leading patch, which is exactly what
// latticeValid's zeroX/zeroY test and newLatticeIter's xIsScalable/yIsScalable test key off of, so rejecting it here
// would make those cases unreachable.
func validDivs(divs []int32, start, end int32) bool {
	prev := start - 1
	for _, d := range divs {
		if prev >= d || d > end {
			return false
		}
		prev = d
	}
	return true
}

// latticeValid reports whether the lattice produces at least one real patch inside the image bounds.
func latticeValid(width, height int32, lattice *latticeSpec) bool {
	totalBounds := geom.IRectWH(width, height)
	lb := lattice.bounds
	if !totalBounds.ContainsRect(lb) {
		return false
	}
	zeroX := len(lattice.xDivs) == 0 || (len(lattice.xDivs) == 1 && lb.Left == lattice.xDivs[0])
	zeroY := len(lattice.yDivs) == 0 || (len(lattice.yDivs) == 1 && lb.Top == lattice.yDivs[0])
	if zeroX && zeroY {
		return false
	}
	return validDivs(lattice.xDivs, lb.Left, lb.Right) && validDivs(lattice.yDivs, lb.Top, lb.Bottom)
}

// countScalablePixels returns how many pixels along the axis belong to scalable patches.
func countScalablePixels(divs []int32, firstIsScalable bool, start, end int32) int32 {
	if len(divs) == 0 {
		if firstIsScalable {
			return end - start
		}
		return 0
	}
	var count int32
	var i int
	if firstIsScalable {
		count = divs[0] - start
		i = 1
	}
	for ; i < len(divs); i += 2 {
		left := divs[i]
		right := end
		if i+1 < len(divs) {
			right = divs[i+1]
		}
		count += right - left
	}
	return count
}

// setLatticePoints computes the src/dst patch boundaries along one axis: scalable patches stretch to absorb the size
// difference, or (when the fixed patches alone exceed dst) the fixed patches shrink and the scalable ones collapse.
func setLatticePoints(dst []float32, src, divs []int32, srcFixed, srcScalable, srcStart, srcEnd int32, dstStart, dstEnd float32, isScalable bool) {
	dstLen := dstEnd - dstStart
	var scale float32
	if float32(srcFixed) <= dstLen {
		// The "normal" case: scale the scalable patches, leave the fixed ones alone.
		scale = (dstLen - float32(srcFixed)) / float32(srcScalable)
	} else {
		// Eliminate the scalable patches and scale the fixed ones.
		scale = dstLen / float32(srcFixed)
	}
	src[0] = srcStart
	dst[0] = dstStart
	for i, div := range divs {
		src[i+1] = div
		srcDelta := src[i+1] - src[i]
		var dstDelta float32
		if float32(srcFixed) <= dstLen {
			if isScalable {
				dstDelta = scale * float32(srcDelta)
			} else {
				dstDelta = float32(srcDelta)
			}
		} else if !isScalable {
			dstDelta = scale * float32(srcDelta)
		}
		dst[i+1] = dst[i] + dstDelta
		isScalable = !isScalable
	}
	src[len(divs)+1] = srcEnd
	dst[len(divs)+1] = dstEnd
}

// latticeIter iterates the lattice's patches in row-major order.
type latticeIter struct {
	srcX, srcY []int32
	dstX, dstY []float32
	currX      int
	currY      int
	numRects   int
}

// newLatticeIter builds the iterator mapping the lattice's patches onto dst.
func newLatticeIter(lattice *latticeSpec, dst geom.Rect) *latticeIter {
	xDivs := lattice.xDivs
	yDivs := lattice.yDivs
	src := lattice.bounds

	// If the first div lines up with the bounds edge, the first patch is degenerate and the first real patch is
	// scalable.
	xIsScalable := len(xDivs) > 0 && src.Left == xDivs[0]
	if xIsScalable {
		xDivs = xDivs[1:]
	}
	yIsScalable := len(yDivs) > 0 && src.Top == yDivs[0]
	if yIsScalable {
		yDivs = yDivs[1:]
	}

	xCountScalable := countScalablePixels(xDivs, xIsScalable, src.Left, src.Right)
	xCountFixed := src.Width() - xCountScalable
	yCountScalable := countScalablePixels(yDivs, yIsScalable, src.Top, src.Bottom)
	yCountFixed := src.Height() - yCountScalable

	it := &latticeIter{
		srcX: make([]int32, len(xDivs)+2),
		dstX: make([]float32, len(xDivs)+2),
		srcY: make([]int32, len(yDivs)+2),
		dstY: make([]float32, len(yDivs)+2),
	}
	setLatticePoints(it.dstX, it.srcX, xDivs, xCountFixed, xCountScalable,
		src.Left, src.Right, dst.Left, dst.Right, xIsScalable)
	setLatticePoints(it.dstY, it.srcY, yDivs, yCountFixed, yCountScalable,
		src.Top, src.Bottom, dst.Top, dst.Bottom, yIsScalable)
	it.numRects = (len(xDivs) + 1) * (len(yDivs) + 1)
	return it
}

// next yields the next patch's src and dst rects, returning false when done.
func (it *latticeIter) next(src *geom.IRect, dst *geom.Rect) bool {
	currRect := it.currX + it.currY*(len(it.srcX)-1)
	if currRect == it.numRects {
		return false
	}
	x := it.currX
	y := it.currY
	it.currX++
	if it.currX == len(it.srcX)-1 {
		it.currX = 0
		it.currY++
	}
	*src = geom.IRect{Left: it.srcX[x], Top: it.srcY[y], Right: it.srcX[x+1], Bottom: it.srcY[y+1]}
	*dst = geom.RectLTRB(it.dstX[x], it.dstY[y], it.dstX[x+1], it.dstY[y+1])
	return true
}
