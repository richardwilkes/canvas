// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Mask blurring for A8 masks: sigmas below 2 run a direct discrete Gaussian (modified-Bessel factors, radius <= 4, 8.8
// fixed point with 0.16 factors — the 8-wide SIMD ganging implemented lane-for-lane), and larger sigmas run the
// three-pass sliding-box planGauss engine with its exact 32.32 fixed-point normalization.

package maskfilter

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
)

// maskBlurFilter holds the (clamped) sigmas for a mask blur. 135 is the largest sigma that will not overflow the
// box-sum accumulators.
type maskBlurFilter struct {
	sigmaW, sigmaH float64
}

func newMaskBlurFilter(sigmaW, sigmaH float64) maskBlurFilter {
	return maskBlurFilter{
		sigmaW: pinDouble(sigmaW, 0, 135),
		sigmaH: pinDouble(sigmaH, 0, 135),
	}
}

func pinDouble(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(v, hi))
}

// hasNoBlur reports whether both sigmas are small enough to treat as no blur at all (the 1/3 cutoff).
func (f maskBlurFilter) hasNoBlur() bool {
	const kNoWindowSigma = 1.0 / 3.0
	return f.sigmaW < kNoWindowSigma && f.sigmaH <= kNoWindowSigma
}

// prepareDestination sizes dst to src outset by the radii, allocating only when src has pixels. An overflow leaves dst
// empty with a zero row-byte count.
func prepareDestination(radiusX, radiusY int, src *raster.Mask) raster.Mask {
	var dst raster.Mask
	dstW := int64(src.Bounds.Width()) + 2*int64(radiusX)
	dstH := int64(src.Bounds.Height()) + 2*int64(radiusY)
	toAlloc := dstW * dstH
	if dstW < 0 || dstH < 0 || dstW > math.MaxInt32 || dstH > math.MaxInt32 || toAlloc > math.MaxInt32 {
		return dst
	}
	dst.Bounds = geom.IRectXYWH(src.Bounds.Left-int32(radiusX), src.Bounds.Top-int32(radiusY),
		int32(dstW), int32(dstH))
	dst.RowBytes = int32(dstW)
	if src.Image != nil {
		dst.Image = make([]uint8, toAlloc)
	}
	return dst
}

// blur runs the mask blur over src into dst (A8 only), returning the border (the blur radii in pixels).
func (f maskBlurFilter) blur(src, dst *raster.Mask) geom.IPoint {
	if f.sigmaW < 2.0 && f.sigmaH < 2.0 {
		return smallBlur(f.sigmaW, f.sigmaH, src, dst)
	}

	planW := newPlanGauss(f.sigmaW)
	planH := newPlanGauss(f.sigmaH)

	borderW := planW.border
	borderH := planH.border

	*dst = prepareDestination(borderW, borderH, src)
	if src.Image == nil {
		return geom.IPoint{X: int32(borderW), Y: int32(borderH)}
	}
	if dst.Image == nil {
		dst.Bounds = geom.IRect{}
		return geom.IPoint{}
	}

	srcW := int(src.Bounds.Width())
	srcH := int(src.Bounds.Height())
	dstW := int(dst.Bounds.Width())
	dstH := int(dst.Bounds.Height())

	bufferSize := planW.bufferSize()
	if s := planH.bufferSize(); s > bufferSize {
		bufferSize = s
	}
	buffer := make([]uint32, bufferSize)

	// Blur both directions.
	tmpW := srcH
	tmpH := dstW

	// Make sure not to overflow the multiply for the tmp buffer size.
	if tmpW > 0 && tmpH > math.MaxInt32/tmpW {
		return geom.IPoint{}
	}
	tmp := make([]uint8, tmpW*tmpH)

	// Blur horizontally, and transpose.
	scanW := planW.makeBlurScan(srcW, buffer)
	for y := 0; y < srcH; y++ {
		srcRow := src.Image[y*int(src.RowBytes) : y*int(src.RowBytes)+srcW]
		scanW.blur(srcRow, tmp, y, tmpW, y+tmpW*tmpH)
	}

	// Blur vertically (scan in memory order because of the transposition), and transpose back.
	scanH := planH.makeBlurScan(tmpW, buffer)
	for y := 0; y < tmpH; y++ {
		tmpRow := tmp[y*tmpW : y*tmpW+tmpW]
		scanH.blur(tmpRow, dst.Image, y, int(dst.RowBytes), y+int(dst.RowBytes)*dstH)
	}

	return geom.IPoint{X: int32(borderW), Y: int32(borderH)}
}

///////////////////////////////////////////////////////////////////////////////
// planGauss — the three-pass sliding-box engine for sigma >= 2

// planGauss holds the precomputed pass sizes, border, and normalization weight for a three-pass sliding-box blur at a
// given sigma.
type planGauss struct {
	weight        uint64
	border        int
	slidingWindow int
	pass0Size     int
	pass1Size     int
	pass2Size     int
}

func newPlanGauss(sigma float64) planGauss {
	possibleWindow := int(math.Floor(sigma*3*math.Sqrt(2*math.Pi)/4 + 0.5))
	window := possibleWindow
	if window < 1 {
		window = 1
	}

	var p planGauss
	p.pass0Size = window - 1
	p.pass1Size = window - 1
	if window&1 == 1 {
		p.pass2Size = window - 1
		p.border = 3 * ((window - 1) / 2)
	} else {
		p.pass2Size = window
		p.border = 3*(window/2) - 1
	}
	p.slidingWindow = 2*p.border + 1

	// If the window is odd then the divisor is window^3, otherwise window^3 + window^2.
	window2 := window * window
	window3 := window2 * window
	divisor := window3
	if window&1 != 1 {
		divisor = window3 + window2
	}
	p.weight = uint64(math.Round(1.0 / float64(divisor) * (1 << 32)))
	return p
}

func (p *planGauss) bufferSize() int { return p.pass0Size + p.pass1Size + p.pass2Size }

// gaussScan runs one planGauss pass over A8 sources: three chained ring-buffer box sums.
type gaussScan struct {
	buf           []uint32
	weight        uint64
	noChangeCount int
	b0            int // start indices of the three ring buffers in buf
	b1            int
	b2            int
	end           int // one past buffer2
}

func (p *planGauss) makeBlurScan(width int, buffer []uint32) gaussScan {
	noChangeCount := 0
	if p.slidingWindow > width {
		noChangeCount = p.slidingWindow - width
	}
	return gaussScan{
		weight:        p.weight,
		noChangeCount: noChangeCount,
		buf:           buffer,
		b0:            0,
		b1:            p.pass0Size,
		b2:            p.pass0Size + p.pass1Size,
		end:           p.pass0Size + p.pass1Size + p.pass2Size,
	}
}

// finalScale computes (weight * sum + 2^31) >> 32.
func (s *gaussScan) finalScale(sum uint32) uint8 {
	const half = uint64(1) << 31
	return uint8((s.weight*uint64(sum) + half) >> 32)
}

// blur runs one three-pass sliding-box scan for A8: src is one row of source alphas; dst[dstStart + k*dstStride] for k
// in [0, n) receives the blurred output, with dstEnd = dstStart + n*dstStride.
func (s *gaussScan) blur(src, dst []uint8, dstStart, dstStride, dstEnd int) {
	buffer0Cursor := s.b0
	buffer1Cursor := s.b1
	buffer2Cursor := s.b2

	for i := s.b0; i < s.end; i++ {
		s.buf[i] = 0
	}

	sum0 := uint32(0)
	sum1 := uint32(0)
	sum2 := uint32(0)

	di := dstStart

	step := func(leadingEdge uint32) {
		sum0 += leadingEdge
		sum1 += sum0
		sum2 += sum1

		dst[di] = s.finalScale(sum2)

		sum2 -= s.buf[buffer2Cursor]
		s.buf[buffer2Cursor] = sum1
		buffer2Cursor++
		if buffer2Cursor >= s.end {
			buffer2Cursor = s.b2
		}

		sum1 -= s.buf[buffer1Cursor]
		s.buf[buffer1Cursor] = sum0
		buffer1Cursor++
		if buffer1Cursor >= s.b2 {
			buffer1Cursor = s.b1
		}

		sum0 -= s.buf[buffer0Cursor]
		s.buf[buffer0Cursor] = leadingEdge
		buffer0Cursor++
		if buffer0Cursor >= s.b1 {
			buffer0Cursor = s.b0
		}

		di += dstStride
	}

	// Consume the source generating pixels.
	for _, a := range src {
		step(uint32(a))
	}

	// The leading edge is off the right side of the mask.
	for i := 0; i < s.noChangeCount; i++ {
		step(0)
	}

	// Starting from the right, fill in the rest of the buffer.
	for i := s.b0; i < s.end; i++ {
		s.buf[i] = 0
	}

	sum0, sum1, sum2 = 0, 0, 0

	dstCursor := dstEnd
	si := len(src)
	for dstCursor > di {
		dstCursor -= dstStride
		si--
		leadingEdge := uint32(src[si])
		sum0 += leadingEdge
		sum1 += sum0
		sum2 += sum1

		dst[dstCursor] = s.finalScale(sum2)

		sum2 -= s.buf[buffer2Cursor]
		s.buf[buffer2Cursor] = sum1
		buffer2Cursor++
		if buffer2Cursor >= s.end {
			buffer2Cursor = s.b2
		}

		sum1 -= s.buf[buffer1Cursor]
		s.buf[buffer1Cursor] = sum0
		buffer1Cursor++
		if buffer1Cursor >= s.b2 {
			buffer1Cursor = s.b1
		}

		sum0 -= s.buf[buffer0Cursor]
		s.buf[buffer0Cursor] = leadingEdge
		buffer0Cursor++
		if buffer0Cursor >= s.b1 {
			buffer0Cursor = s.b0
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// gaussFilter — the discrete Gaussian factors for the small-sigma path

// gaussArrayMax is the largest number of Gaussian factors gaussFilter can hold.
const gaussArrayMax = 6

// gaussFilter holds n factors of a normalized discrete Gaussian.
type gaussFilter struct {
	basis [gaussArrayMax]float64
	n     int
}

// newGaussFilter computes the discrete Gaussian factors (via modified-Bessel functions) for 0 <= sigma < 2.
func newGaussFilter(sigma float64) gaussFilter {
	var g gaussFilter
	variance := sigma * sigma

	// The value when we can stop expanding the filter. The spec implies 3% is acceptable, but we use 1%.
	const kGoodEnough = 1.0 / 100.0

	besselI0 := func(t float64) float64 {
		tSquaredOver4 := t * t / 4.0
		sum := 1.0
		factor := 1.0
		k := 1
		for factor > 1.0/1000000.0 {
			factor *= tSquaredOver4 / float64(k*k)
			sum += factor
			k++
		}
		return sum
	}
	besselI1 := func(t float64) float64 {
		tSquaredOver4 := t * t / 4.0
		sum := t / 2.0
		factor := sum
		k := 1
		for factor > 1.0/1000000.0 {
			factor *= tSquaredOver4 / float64(k*(k+1))
			sum += factor
			k++
		}
		return sum
	}

	// gauss(n; var) = besselI_n(var) / e^var (Lindeberg's discrete scale space).
	d := math.Exp(variance)
	var b [gaussArrayMax]float64
	b[0] = besselI0(variance)
	b[1] = besselI1(variance)
	g.basis[0] = b[0] / d
	g.basis[1] = b[1] / d

	n := 1
	// Recurrence from Numerical Recipes 3rd Edition, 6.5.16 p.282.
	for g.basis[n] > kGoodEnough {
		b[n+1] = -(2 * float64(n) / variance * b[n]) + b[n-1]
		g.basis[n+1] = b[n+1] / d
		n++
	}

	// Normalize the values of gauss to 1.0, carefully adding from smallest to largest.
	sum := 0.0
	for i := n - 1; i >= 1; i-- {
		sum += 2 * g.basis[i]
	}
	sum += g.basis[0]
	for i := 0; i < n; i++ {
		g.basis[i] /= sum
	}
	// The factors should sum to 1. Take any remaining slop and add it to basis[0].
	sum = 0.0
	for i := n - 1; i >= 1; i-- {
		sum += 2 * g.basis[i]
	}
	g.basis[0] = 1 - sum

	g.n = n
	return g
}

func (g *gaussFilter) radius() int { return g.n - 1 }

///////////////////////////////////////////////////////////////////////////////
// the small-sigma direct convolution (8-wide 8.8 fixed point, ported lane-for-lane)

// fp88 is an 8-wide 8.8 fixed-point vector.
type fp88 [8]uint16

const fpHalf = uint16(0x80)

func fp88Splat(v uint16) fp88 {
	return fp88{v, v, v, v, v, v, v, v}
}

// mulhi computes (a*b) >> 16 per lane (0.16 x 8.8 -> 8.8).
func mulhi(a, b fp88) fp88 {
	var r fp88
	for i := range r {
		r[i] = uint16((uint32(a[i]) * uint32(b[i])) >> 16)
	}
	return r
}

func fp88Add(a, b fp88) fp88 {
	var r fp88
	for i := range r {
		r[i] = a[i] + b[i]
	}
	return r
}

// addShifted adds v shifted right by offset lanes across the (d0, d8) register pair — the generic form of the
// per-radius shifted-lane accumulation blurXRadius needs.
func addShifted(d0, d8 *fp88, v fp88, offset int) {
	for i := 0; i < 8; i++ {
		j := i + offset
		switch {
		case j < 8:
			d0[j] += v[i]
		case j < 16:
			d8[j-8] += v[i]
		}
	}
}

// loadA8 converts up to 8 A8 bytes to 8.8 fixed point.
func loadA8(from []uint8, width int) fp88 {
	var r fp88
	if width > 8 {
		width = 8
	}
	for i := 0; i < width; i++ {
		r[i] = uint16(from[i]) << 8
	}
	return r
}

// store converts 8.8 fixed point back to bytes.
func store(to []uint8, v fp88, width int) {
	if width > 8 {
		width = 8
	}
	for i := 0; i < width; i++ {
		to[i] = uint8(v[i] >> 8)
	}
}

// blurXRadius handles radius 1 through 4: given 8 source values s and the gauss factors g, accumulate the destination
// registers d0 (current 8) and d8 (next 8).
func blurXRadius(radius int, s fp88, g *[5]fp88, d0, d8 *fp88) {
	var v [5]fp88
	for k := 0; k <= radius; k++ {
		v[k] = mulhi(s, g[k])
	}
	for j := 0; j <= 2*radius; j++ {
		k := radius - j
		if k < 0 {
			k = -k
		}
		addShifted(d0, d8, v[k], j)
	}
}

// blurRow runs the direct-convolution box blur across one row.
func blurRow(radius int, g *[5]fp88, src []uint8, srcW int, dst []uint8, dstW int) {
	// Clear the buffer to handle summing wider than source.
	d0 := fp88Splat(fpHalf)
	d8 := fp88Splat(fpHalf)

	// Go by multiples of 8 in src.
	x := 0
	si, di := 0, 0
	for ; x <= srcW-8; x += 8 {
		blurXRadius(radius, loadA8(src[si:], 8), g, &d0, &d8)
		store(dst[di:], d0, 8)
		d0 = d8
		d8 = fp88Splat(fpHalf)
		si += 8
		di += 8
	}

	blurRowTail(radius, g, src[si:], srcW-x, dst[di:], dstW-x, d0, d8)
}

// blurRowTail finishes a row the 8-wide main loop has taken as far as it can. src starts at the first unconsumed
// source byte and srcTail (0 through 7) of them are still to be read; dst starts at the first destination byte not yet
// written and dstRemaining of them are still owed; d0 and d8 are the accumulator pair the main loop left behind, d0
// covering the next 8 destination positions and d8 the 8 after those.
//
// This is a straight lift of what used to be blurRow's tail, kept separate because the simd kernel runs the same main
// loop in vector registers and then hands its accumulator pair back here rather than duplicating the edge phases.
func blurRowTail(radius int, g *[5]fp88, src []uint8, srcTail int, dst []uint8, dstRemaining int, d0, d8 fp88) {
	di := 0

	// There are src values left, but the remainder of src values is not a multiple of 8.
	if srcTail > 0 {
		blurXRadius(radius, loadA8(src, srcTail), g, &d0, &d8)
		dstTail := dstRemaining
		if dstTail > 8 {
			dstTail = 8
		}
		store(dst, d0, dstTail)
		d0 = d8
		di += dstTail
		dstRemaining -= dstTail
	}

	// There are dst mask values to complete.
	if dstRemaining > 0 {
		store(dst[di:], d0, dstRemaining)
	}
}

// directBlurXFn and directBlurYFn are the dispatch points for the two hot small-sigma drivers. The default build runs
// the portable 8-lane fp88 forms; a goexperiment.simd build on qualifying hardware repoints them at the archsimd
// kernels in maskblur_simd.go, which reproduce the same integer arithmetic lane for lane (locked by
// TestMaskBlurSIMDMatchesScalar). The cut is at the whole-rect drivers so the gauss factor splats are built once per
// rect and no per-row or per-column loop pays an indirect call.
var (
	directBlurXFn = directBlurX
	directBlurYFn = directBlurY
)

// directBlurX runs the direct-convolution box blur horizontally over a rect.
func directBlurX(radius int, gauss *[5]uint16, src []uint8, srcStride, srcW int, dst []uint8, dstStride, dstW, dstH int) {
	var g [5]fp88
	for i := range g {
		g[i] = fp88Splat(gauss[i])
	}
	si, di := 0, 0
	for y := 0; y < dstH; y++ {
		blurRow(radius, &g, src[si:], srcW, dst[di:], dstW)
		si += srcStride
		di += dstStride
	}
}

// blurYRadius handles radius 1 through 4: regs holds the 2*radius pipeline registers (d01, d12, …); the answer for this
// row pops out of regs[0].
func blurYRadius(radius int, s fp88, g *[5]fp88, regs *[8]fp88) fp88 {
	var v [5]fp88
	for k := 0; k <= radius; k++ {
		v[k] = mulhi(s, g[k])
	}
	answer := fp88Add(regs[0], v[radius])
	n2 := 2 * radius
	for n := 0; n < n2-1; n++ {
		k := radius - 1 - n
		if k < 0 {
			k = -k
		}
		regs[n] = fp88Add(regs[n+1], v[k])
	}
	regs[n2-1] = fp88Add(v[radius], fp88Splat(fpHalf))
	return answer
}

// blurColumn runs the direct-convolution box blur down one column, for A8.
func blurColumn(radius, width int, g *[5]fp88, src []uint8, srcRB, srcH int, dst []uint8, dstRB int) {
	var regs [8]fp88
	for i := range regs {
		regs[i] = fp88Splat(fpHalf)
	}

	si, di := 0, 0
	for y := 0; y < srcH; y++ {
		s := loadA8(src[si:], width)
		b := blurYRadius(radius, s, g, &regs)
		store(dst[di:], b, width)
		si += srcRB
		di += dstRB
	}

	// Flush the pipeline registers: radius pairs of trailing rows.
	for r := 0; r < radius; r++ {
		store(dst[di:], regs[2*r], width)
		di += dstRB
		store(dst[di:], regs[2*r+1], width)
		di += dstRB
	}
}

// directBlurY runs the direct-convolution box blur vertically over a rect, for A8 (strideOf8 = 8).
func directBlurY(radius int, gauss *[5]uint16, src []uint8, srcRB, srcW, srcH int, dst []uint8, dstRB int) {
	var g [5]fp88
	for i := range g {
		g[i] = fp88Splat(gauss[i])
	}
	x := 0
	si, di := 0, 0
	for ; x <= srcW-8; x += 8 {
		blurColumn(radius, 8, &g, src[si:], srcRB, srcH, dst[di:], dstRB)
		si += 8
		di += 8
	}
	if xTail := srcW - x; xTail > 0 {
		blurColumn(radius, xTail, &g, src[si:], srcRB, srcH, dst[di:], dstRB)
	}
}

// smallBlur runs the direct discrete-Gaussian convolution for 0.01 <= sigma < 2.
func smallBlur(sigmaX, sigmaY float64, src, dst *raster.Mask) geom.IPoint {
	filterX := newGaussFilter(sigmaX)
	filterY := newGaussFilter(sigmaY)

	radiusX := filterX.radius()
	radiusY := filterY.radius()

	prepareGauss := func(filter *gaussFilter, factors *[5]uint16) {
		for i := 0; i < filter.n; i++ {
			factors[i] = uint16(math.Round(filter.basis[i] * (1 << 16)))
		}
	}

	var gaussFactorsX, gaussFactorsY [5]uint16
	prepareGauss(&filterX, &gaussFactorsX)
	prepareGauss(&filterY, &gaussFactorsY)

	*dst = prepareDestination(radiusX, radiusY, src)
	if src.Image == nil {
		return geom.IPoint{X: int32(radiusX), Y: int32(radiusY)}
	}
	if dst.Image == nil {
		dst.Bounds = geom.IRect{}
		return geom.IPoint{}
	}

	srcW := int(src.Bounds.Width())
	srcH := int(src.Bounds.Height())
	dstW := int(dst.Bounds.Width())
	dstH := int(dst.Bounds.Height())

	// Blur vertically and copy to destination.
	directBlurYFn(radiusY, &gaussFactorsY, src.Image, int(src.RowBytes), srcW, srcH,
		dst.Image[radiusX:], int(dst.RowBytes))

	// Blur horizontally in place.
	directBlurXFn(radiusX, &gaussFactorsX, dst.Image[radiusX:], int(dst.RowBytes), srcW,
		dst.Image, int(dst.RowBytes), dstW, dstH)

	return geom.IPoint{X: int32(radiusX), Y: int32(radiusY)}
}
