// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Distance-field generation: 8-bit signed distance fields generated from A8 coverage masks via Danielsson's 8SSEDT,
// with the Gustavson (2011) edge-distance estimate seeding the edge texels. Only the A8 source lane is implemented —
// the SDF strike rec always rasterizes an A8 coverage mask first (the SDF rec is its own mask format, so no LCD16 mask
// ever reaches the generator).

package font

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
)

// Distance-field constants.
const (
	// DistanceFieldMagnitude is the max magnitude for the distance field; distance values are limited to the range
	// (-DistanceFieldMagnitude, DistanceFieldMagnitude].
	DistanceFieldMagnitude = 4
	// DistanceFieldPad is the padding around the original glyph that allows the maximum distance of
	// DistanceFieldMagnitude texels away from any edge.
	DistanceFieldPad = 4
	// DistanceFieldInset is the amount the rect we render with is inset from the distance field glyph size, to allow
	// for bilerp.
	DistanceFieldInset = 2
)

// Fragment-shader constants: the distance field is constructed as unsigned byte values with the zero value at 128, so
// the supported range of distances is [-4*127/128, 4]. The multiplier is the width of the range (4 * 255/128) and the
// zero threshold is 128/255.
const (
	DistanceFieldMultiplier = "7.96875"
	DistanceFieldThreshold  = "0.50196078431"
)

// dfData holds the per-texel working state for the distance transform.
type dfData struct {
	alpha      float32    // alpha value of source texel
	distSq     float32    // distance squared to nearest (so far) edge texel
	distVector geom.Point // distance vector to nearest (so far) edge texel
}

// Neighbor flags, one bit per compass direction around a texel.
const (
	leftNeighborFlag        = 0x01
	rightNeighborFlag       = 0x02
	topLeftNeighborFlag     = 0x04
	topNeighborFlag         = 0x08
	topRightNeighborFlag    = 0x10
	bottomLeftNeighborFlag  = 0x20
	bottomNeighborFlag      = 0x40
	bottomRightNeighborFlag = 0x80
	allNeighborFlags        = 0xff

	neighborFlagCount = 8
)

// foundEdge reports whether at is an edge texel: an "edge" is a place where we cross from >=128 to <128, or vice versa,
// or where we have two non-zero pixels that are <128. neighborFlags limits the directions tested to avoid indexing
// outside of the image.
func foundEdge(image []uint8, at, width, neighborFlags int) bool {
	// The order of these matches the neighbor flags above.
	offsets := [neighborFlagCount]int{-1, 1, -width - 1, -width, -width + 1, width - 1, width, width + 1}

	currVal := image[at]
	currCheck := currVal >> 7
	for i := range neighborFlagCount {
		var neighborVal uint8
		if (1<<i)&neighborFlags != 0 {
			neighborVal = image[at+offsets[i]]
		}
		neighborCheck := neighborVal >> 7
		// A sharp transition, or both <128 and >0.
		if currCheck != neighborCheck ||
			(currCheck == 0 && neighborCheck == 0 && currVal != 0 && neighborVal != 0) {
			return true
		}
	}
	return false
}

// initGlyphData copies image's alpha values into data and marks edge texels in edges, both laid out with the given pad
// on each side.
func initGlyphData(data []dfData, edges, image []uint8, dataWidth, imageWidth, imageHeight, pad int) {
	d := pad*dataWidth + pad
	e := pad*dataWidth + pad
	at := 0
	for j := 0; j < imageHeight; j++ {
		for i := 0; i < imageWidth; i++ {
			if image[at] == 255 {
				data[d].alpha = 1
			} else {
				data[d].alpha = float32(image[at]) * 0.00392156862 // 1/255
			}
			checkMask := allNeighborFlags
			if i == 0 {
				checkMask &^= leftNeighborFlag | topLeftNeighborFlag | bottomLeftNeighborFlag
			}
			if i == imageWidth-1 {
				checkMask &^= rightNeighborFlag | topRightNeighborFlag | bottomRightNeighborFlag
			}
			if j == 0 {
				checkMask &^= topLeftNeighborFlag | topNeighborFlag | topRightNeighborFlag
			}
			if j == imageHeight-1 {
				checkMask &^= bottomLeftNeighborFlag | bottomNeighborFlag | bottomRightNeighborFlag
			}
			if foundEdge(image, at, imageWidth, checkMask) {
				edges[e] = 255 // using 255 makes for convenient debug rendering
			}
			d++
			at++
			e++
		}
		d += 2 * pad
		e += 2 * pad
	}
}

// edgeDistance estimates the distance to an edge given an edge normal vector and a pixel's alpha value (Gustavson
// 2011); assumes direction has been pre-normalized.
func edgeDistance(direction geom.Point, alpha float32) float32 {
	dx := direction.X
	dy := direction.Y
	var distance float32
	if geom.ScalarNearlyZero(dx) || geom.ScalarNearlyZero(dy) {
		distance = 0.5 - alpha
	} else {
		// This is easier if we treat the direction as being in the first octant (other octants are symmetrical).
		dx = absF32(dx)
		dy = absF32(dy)
		if dx < dy {
			dx, dy = dy, dx
		}

		// a1 = 0.5*dy/dx is the smaller fractional area chopped off by the edge; to avoid the divide, we just consider
		// the numerator.
		a1num := 0.5 * dy

		// We now compute the approximate distance, depending where the alpha falls relative to the edge fractional
		// area.
		switch {
		case alpha*dx < a1num: // 0 <= alpha < a1
			distance = 0.5*(dx+dy) - sqrtF32(2*dx*dy*alpha)
		case alpha*dx < dx-a1num: // a1 <= alpha <= 1 - a1
			distance = (0.5 - alpha) * dx
		default: // 1 - a1 < alpha <= 1
			distance = -0.5*(dx+dy) + sqrtF32(2*dx*dy*(1-alpha))
		}
	}
	return distance
}

func absF32(v float32) float32  { return math.Float32frombits(math.Float32bits(v) &^ (1 << 31)) }
func sqrtF32(v float32) float32 { return float32(math.Sqrt(float64(v))) }

// setLengthFast scales p to the given length, computed via an ordinary sqrt round trip (the precision this needs is
// loose enough that the comparisons elsewhere in this file cannot tell the difference from a fast inverse-square-root
// approximation).
func setLengthFast(p geom.Point, length float32) geom.Point {
	mag2 := p.X*p.X + p.Y*p.Y
	if !(mag2 > 0) || math.IsInf(float64(mag2), 0) {
		return geom.Point{}
	}
	scale := length / sqrtF32(mag2)
	return geom.Pt(p.X*scale, p.Y*scale)
}

const sqrt2 = float32(math.Sqrt2)

// initDistances creates initial distance data, particularly at edges.
func initDistances(data []dfData, edges []uint8, width, height int) {
	// Skip one pixel border.
	curr := 0
	prev := -width
	next := width
	e := 0
	for j := 0; j < height; j++ {
		for i := 0; i < width; i++ {
			if edges[e] != 0 {
				// The gradient points from low to high: outside it points toward the edge, inside away from the edge
				// (+y is down here).
				var currGrad geom.Point
				currGrad.X = data[prev+1].alpha - data[prev-1].alpha +
					sqrt2*data[curr+1].alpha - sqrt2*data[curr-1].alpha +
					data[next+1].alpha - data[next-1].alpha
				currGrad.Y = data[next-1].alpha - data[prev-1].alpha +
					sqrt2*data[next].alpha - sqrt2*data[prev].alpha +
					data[next+1].alpha - data[prev+1].alpha
				currGrad = setLengthFast(currGrad, 1)

				// Init squared distance to edge and distance vector.
				dist := edgeDistance(currGrad, data[curr].alpha)
				data[curr].distVector = geom.Pt(currGrad.X*dist, currGrad.Y*dist)
				data[curr].distSq = dist * dist
			} else {
				// Init distance to "far away".
				data[curr].distSq = 2000000
				data[curr].distVector = geom.Pt(1000, 1000)
			}
			curr++
			prev++
			next++
			e++
		}
	}
}

// Danielsson's 8SSEDT passes (f1/f2/b1/b2).

// f1 is the first stage forward pass (forward in y, forward in x).
func f1(data []dfData, curr, width int) {
	// Upper left.
	check := &data[curr-width-1]
	distVec := check.distVector
	distSq := check.distSq - 2*(distVec.X+distVec.Y-1)
	if distSq < data[curr].distSq {
		distVec.X--
		distVec.Y--
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}

	// Up.
	check = &data[curr-width]
	distVec = check.distVector
	distSq = check.distSq - 2*distVec.Y + 1
	if distSq < data[curr].distSq {
		distVec.Y--
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}

	// Upper right.
	check = &data[curr-width+1]
	distVec = check.distVector
	distSq = check.distSq + 2*(distVec.X-distVec.Y+1)
	if distSq < data[curr].distSq {
		distVec.X++
		distVec.Y--
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}

	// Left.
	check = &data[curr-1]
	distVec = check.distVector
	distSq = check.distSq - 2*distVec.X + 1
	if distSq < data[curr].distSq {
		distVec.X--
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}
}

// f2 is the second stage forward pass (forward in y, backward in x).
func f2(data []dfData, curr int) {
	// Right.
	check := &data[curr+1]
	distVec := check.distVector
	distSq := check.distSq + 2*distVec.X + 1
	if distSq < data[curr].distSq {
		distVec.X++
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}
}

// b1 is the first stage backward pass (backward in y, forward in x).
func b1(data []dfData, curr int) {
	// Left.
	check := &data[curr-1]
	distVec := check.distVector
	distSq := check.distSq - 2*distVec.X + 1
	if distSq < data[curr].distSq {
		distVec.X--
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}
}

// b2 is the second stage backward pass (backward in y, backward in x).
func b2(data []dfData, curr, width int) {
	// Right.
	check := &data[curr+1]
	distVec := check.distVector
	distSq := check.distSq + 2*distVec.X + 1
	if distSq < data[curr].distSq {
		distVec.X++
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}

	// Bottom left.
	check = &data[curr+width-1]
	distVec = check.distVector
	distSq = check.distSq - 2*(distVec.X-distVec.Y-1)
	if distSq < data[curr].distSq {
		distVec.X--
		distVec.Y++
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}

	// Bottom.
	check = &data[curr+width]
	distVec = check.distVector
	distSq = check.distSq + 2*distVec.Y + 1
	if distSq < data[curr].distSq {
		distVec.Y++
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}

	// Bottom right.
	check = &data[curr+width+1]
	distVec = check.distVector
	distSq = check.distSq + 2*(distVec.X+distVec.Y+1)
	if distSq < data[curr].distSq {
		distVec.X++
		distVec.Y++
		data[curr].distSq = distSq
		data[curr].distVector = distVec
	}
}

// packDistanceFieldVal packs a signed distance into an unsigned byte, with the zero value at 128. Besides 128 there are
// 128 values in [0, 128) but only 127 in (128, 255], so the magnitude is scaled by 127/128 in the latter range to avoid
// overflow.
func packDistanceFieldVal(dist float32) uint8 {
	const magnitude = float32(DistanceFieldMagnitude)
	d := -dist
	if d < -magnitude {
		d = -magnitude
	} else if d > magnitude*127.0/128.0 {
		d = magnitude * 127.0 / 128.0
	}
	// Scale into the positive range for unsigned distance, then into byte range, rounding to place negative and
	// positive values as equally as possible around 128 (which represents zero).
	d += magnitude
	return uint8(int32(math.Floor(float64(d/(2*magnitude)*256.0 + 0.5))))
}

// generateDistanceFieldFromImage assumes a (1-texel) padded 8-bit image copy and a distance field allocated with
// DistanceFieldPad padding; width and height are the original dimensions of the image.
func generateDistanceFieldFromImage(distanceField, paddedImage []uint8, width, height int) {
	// We expand the temp data by one more on each side to simplify the scanning code — the border is always treated as
	// infinitely far away.
	pad := DistanceFieldPad + 1

	dataWidth := width + 2*pad
	dataHeight := height + 2*pad

	data := make([]dfData, dataWidth*dataHeight)
	edges := make([]uint8, dataWidth*dataHeight)

	// Copy the glyph into the distance-field storage.
	initGlyphData(data, edges, paddedImage, dataWidth, width+2, height+2, DistanceFieldPad)

	// Create the initial distance data, particularly at edges.
	initDistances(data, edges, dataWidth, dataHeight)

	// Perform the Euclidean distance transform to propagate distances.

	// Forwards in y.
	curr := dataWidth + 1 // skip the outer buffer
	e := dataWidth + 1
	for j := 1; j < dataHeight-1; j++ {
		// Forwards in x.
		for i := 1; i < dataWidth-1; i++ {
			// Don't need to calculate distance for edge pixels.
			if edges[e] == 0 {
				f1(data, curr, dataWidth)
			}
			curr++
			e++
		}

		// Backwards in x.
		curr-- // reset to end
		e--
		for i := 1; i < dataWidth-1; i++ {
			if edges[e] == 0 {
				f2(data, curr)
			}
			curr--
			e--
		}

		curr += dataWidth + 1
		e += dataWidth + 1
	}

	// Backwards in y. The start is the first interior texel of the last interior row — (dataHeight-2, 1) — mirroring
	// the forward pass's (1, 1). (Upstream Skia starts two texels earlier, at (dataHeight-3, dataWidth-1), which shifts
	// every row's window two columns left: the two rightmost columns of each row never get backward propagation and
	// column 0's b1 reads the previous row's last column. That asymmetry is invisible in a rendered glyph only because
	// DistanceFieldInset trims the affected padding, so it is fixed here rather than ported.)
	curr = dataWidth*(dataHeight-2) + 1 // skip the outer buffer
	e = dataWidth*(dataHeight-2) + 1
	for j := 1; j < dataHeight-1; j++ {
		// Forwards in x.
		for i := 1; i < dataWidth-1; i++ {
			if edges[e] == 0 {
				b1(data, curr)
			}
			curr++
			e++
		}

		// Backwards in x.
		curr-- // reset to end
		e--
		for i := 1; i < dataWidth-1; i++ {
			if edges[e] == 0 {
				b2(data, curr, dataWidth)
			}
			curr--
			e--
		}

		curr -= dataWidth - 1
		e -= dataWidth - 1
	}

	// Copy the results into the final distance-field data.
	curr = dataWidth + 1
	df := 0
	for j := 1; j < dataHeight-1; j++ {
		for i := 1; i < dataWidth-1; i++ {
			var dist float32
			if data[curr].alpha > 0.5 {
				dist = -sqrtF32(data[curr].distSq)
			} else {
				dist = sqrtF32(data[curr].distSq)
			}
			distanceField[df] = packDistanceFieldVal(dist)
			df++
			curr++
		}
		curr += 2
	}
}

// GenerateDistanceFieldFromA8Image generates, from 8-bit mask data of width x height (rowBytes stride), the associated
// distance field into distanceField, which must already be allocated at (width+2*DistanceFieldPad) x
// (height+2*DistanceFieldPad) bytes.
func GenerateDistanceFieldFromA8Image(distanceField, image []uint8, width, height, rowBytes int) {
	// Copy the source image into a 1-texel padded copy to ensure we catch edge transitions around the outside.
	paddedW := width + 2
	copyPtr := make([]uint8, paddedW*(height+2))
	for y := 0; y < height; y++ {
		copy(copyPtr[(y+1)*paddedW+1:(y+1)*paddedW+1+width], image[y*rowBytes:y*rowBytes+width])
	}
	generateDistanceFieldFromImage(distanceField, copyPtr, width, height)
}

// ComputeDistanceFieldSize returns the number of bytes needed for a distance field over a w x h glyph.
func ComputeDistanceFieldSize(w, h int) int {
	return (w + 2*DistanceFieldPad) * (h + 2*DistanceFieldPad)
}
