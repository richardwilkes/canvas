// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The blur engine the image filters consume: the raster engine's RGBA8888 algorithm — a true 1D Gaussian pass for sigma
// < 2 (gaussianPass) and the three-pass box approximation for sigma in [2, 135] (threeBoxApproxPass), evaluated as
// separated X/Y passes writing into one shared buffer. A tent-filter pass would only serve legacy blurs that bypass
// FilterResult.rescale and is unreachable here; the A8 and shader-based algorithms wait for their consumers.

package filtercore

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

// IsEffectivelyIdentity reports whether sigma is small enough that the blur has no visible effect.
func IsEffectivelyIdentity(sigma float32) bool { return sigma <= 0.03 }

// SigmaToRadius returns the pixel radius a Gaussian blur of the given sigma needs.
func SigmaToRadius(sigma float32) int32 {
	if IsEffectivelyIdentity(sigma) {
		return 0
	}
	return geom.CeilToInt(3 * sigma)
}

// boxBlurWindow returns the SVG successive-box-blur window size for sigma.
func boxBlurWindow(sigma float32) int32 {
	possibleWindow := geom.FloorToInt(sigma*3*sqrtf32(2*math.Pi)/4 + 0.5)
	return max(1, possibleWindow)
}

// compute1DBlurKernel fills kernel with normalized Gaussian weights over [-radius, radius].
func compute1DBlurKernel(sigma float32, radius int32, kernel []float32) {
	twoSigmaSqrd := 2 * sigma * sigma
	sigmaDenom := float32(1)
	if radius > 0 {
		sigmaDenom = 1 / twoSigmaSqrd
	}
	width := 2*radius + 1
	sum := float32(0)
	for x := int32(0); x < width; x++ {
		term := float32(x - radius)
		w := float32(math.Exp(float64(-(term * term * sigmaDenom))))
		kernel[x] = w
		sum += w
	}
	scale := 1 / sum
	for i := int32(0); i < width; i++ {
		kernel[i] *= scale
	}
}

// BlurAlgorithm implements one blur strategy for a range of sigmas.
type BlurAlgorithm interface {
	// MaxSigma reports the largest sigma this algorithm handles directly; larger sigmas require the caller to downscale
	// the input first (FilterResult.rescale handles this).
	MaxSigma() float32
	// SupportsOnlyDecalTiling reports whether this algorithm can only apply decal tiling directly (other tile modes
	// must be applied via the crop machinery).
	SupportsOnlyDecalTiling() bool
	// Blur produces a blurred image filling dstBounds, or nil if it cannot (an unusable source: a failed readback of a
	// drawable backing, a failed upload). srcBounds restricts the input pixels and both rects are relative to src's
	// logical bounds; tileMode applies at srcBounds' boundary.
	Blur(sigma geom.Size, src *SpecialImage, srcBounds geom.IRect, tileMode shaders.TileMode,
		dstBounds geom.IRect) *SpecialImage
}

// BlurEngine selects the backend's BlurAlgorithm, with the color type pinned to N32.
type BlurEngine interface {
	FindAlgorithm() BlurAlgorithm
}

// rasterBlurEngine is the raster backend's blur engine, reduced to the reachable N32 algorithm.
type rasterBlurEngine struct {
	rgba8Algorithm raster8888BlurAlgorithm
}

var theRasterBlurEngine rasterBlurEngine

// RasterBlurEngine returns the raster backend's BlurEngine.
func RasterBlurEngine() BlurEngine { return &theRasterBlurEngine }

func (e *rasterBlurEngine) FindAlgorithm() BlurAlgorithm {
	return &e.rgba8Algorithm
}

///////////////////////////////////////////////////////////////////////////////
// pass plumbing

// The two hot per-scanline kernels — the Gaussian pass's windowed dot product and the three-box pass's prefix-sum
// pipeline — dispatch through package-level function variables so a build can substitute a vector form for the
// portable one: a goexperiment.simd build repoints them at the archsimd kernels in blurengine_simd.go where those are
// the faster lane. The portable bodies below stay the reference implementation and remain the default everywhere else.
type (
	gaussianSegmentFn func(p *gaussianPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32)
	threeBoxSegmentFn func(p *threeBoxApproxPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32)
)

var (
	gaussianBlurSegmentFn gaussianSegmentFn = gaussianBlurSegmentGeneric
	threeBoxBlurSegmentFn threeBoxSegmentFn = threeBoxBlurSegmentGeneric
)

// blurPass is a scanline processor with a border (the distance between the first dst pixel and the first src pixel it
// depends on).
type blurPass interface {
	startBlur()
	// blurSegment processes n pixels; nil src means transparent input, nil dst discards output. Strides are in words.
	blurSegment(n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32)
	border() int32
}

// runPass walks one scanline, feeding the pass leading zeroes, source pixels, and trailing zeroes as the spec requires.
// src/dst index the first pixel of the row/column being processed, with the given strides.
func runPass(p blurPass, srcLeft, srcRight, dstRight int32, src []uint32, srcOff, srcStride int32, dst []uint32, dstOff, dstStride int32) {
	p.startBlur()

	srcIdx := srcLeft - p.border()
	srcEnd := srcRight - p.border()
	dstIdx := int32(0)
	dstEnd := dstRight

	srcCursor := srcOff
	dstCursor := dstOff

	switch {
	case dstIdx < srcIdx:
		// The destination pixels are not affected by the src pixels: zero per the spec.
		commonEnd := min(srcIdx, dstEnd)
		for dstIdx < commonEnd {
			dst[dstCursor] = 0
			dstCursor += dstStride
			dstIdx++
		}
	case srcIdx < dstIdx:
		// The source edge precedes the destination edge: preload the blur with src values.
		if commonEnd := min(dstIdx, srcEnd); srcIdx < commonEnd {
			n := commonEnd - srcIdx
			p.blurSegment(n, src[srcCursor:], srcStride, nil, 0)
			srcIdx += n
			srcCursor += n * srcStride
		}
		if srcIdx < dstIdx {
			// The weird case where src runs out of pixels before dst has started.
			p.blurSegment(dstIdx-srcIdx, nil, 0, nil, 0)
		}
	}

	if commonEnd := min(dstEnd, srcEnd); dstIdx < commonEnd {
		// srcIdx and dstIdx are in sync: the normal 1:1 mode of operation.
		n := commonEnd - dstIdx
		p.blurSegment(n, src[srcCursor:], srcStride, dst[dstCursor:], dstStride)
		dstCursor += n * dstStride
		dstIdx += n
	}

	// Drain the remaining blur values into dst, assuming 0s for the leading edge.
	if dstIdx < dstEnd {
		n := dstEnd - dstIdx
		p.blurSegment(n, nil, 0, dst[dstCursor:], dstStride)
	}
}

// passMaker builds a fresh blurPass for one axis, along with the window size and sigma it was built for.
type passMaker struct {
	makePass func() blurPass
	window   int32
	sigma    float32
}

// evalBlurPasses runs an X pass from src into dst (inflated by the Y window), then an in-place Y pass.
func evalBlurPasses(makerX, makerY *passMaker, src *imagecore.Pixels, originalSrcBounds, originalDstBounds geom.IRect) *SpecialImage {
	srcBounds := originalSrcBounds
	dstBounds := originalDstBounds
	if makerX.window > 1 {
		// Inflate dst by the Y pass's radius so the X pass prepares the extra rows the Y pass reads; one buffer serves
		// both passes since the blur can write in place.
		dstBounds = geom.IRect{
			Left:   dstBounds.Left,
			Top:    dstBounds.Top - SigmaToRadius(makerY.sigma),
			Right:  dstBounds.Right,
			Bottom: dstBounds.Bottom + SigmaToRadius(makerY.sigma),
		}
	}

	dstOrigin := geom.IPoint{X: dstBounds.Left, Y: dstBounds.Top}
	dst := imagecore.NewPixels(imagecore.MakeN32Premul(dstBounds.Width(), dstBounds.Height()))
	if dst == nil {
		return nil
	}

	// Initialized assuming the Y-only case.
	loopStart := max(srcBounds.Left, dstBounds.Left)
	loopEnd := min(srcBounds.Right, dstBounds.Right)
	dstYOffset := int32(0)

	srcWords, srcRow := src.Words, src.RowElems
	dstWords, dstRow := dst.Words, dst.RowElems

	if makerX.window > 1 {
		// X-only blur from src into dst, including the extra rows for the Y pass.
		loopStart = max(srcBounds.Top, dstBounds.Top)
		loopEnd = min(srcBounds.Bottom, dstBounds.Bottom)

		srcOff := (loopStart - srcBounds.Top) * srcRow
		dstOff := (loopStart - dstBounds.Top) * dstRow

		pass := makerX.makePass()
		for y := loopStart; y < loopEnd; y++ {
			runPass(pass, srcBounds.Left-dstBounds.Left, srcBounds.Right-dstBounds.Left,
				dstBounds.Width(), srcWords, srcOff, 1, dstWords, dstOff, 1)
			srcOff += srcRow
			dstOff += dstRow
		}

		// Set up the Y pass to blur from the full dst into the non-outset portion of dst.
		srcWords, srcRow = dstWords, dstRow
		loopStart = originalDstBounds.Left
		loopEnd = originalDstBounds.Right
		dstYOffset = originalDstBounds.Top - dstBounds.Top

		srcBounds = dstBounds
		dstBounds = originalDstBounds
	}

	if makerY.window > 1 {
		srcOff := loopStart - srcBounds.Left
		dstOff := (loopStart - dstBounds.Left) + dstYOffset*dstRow

		pass := makerY.makePass()
		for x := loopStart; x < loopEnd; x++ {
			runPass(pass, srcBounds.Top-dstBounds.Top, srcBounds.Bottom-dstBounds.Top,
				dstBounds.Height(), srcWords, srcOff, srcRow, dstWords, dstOff, dstRow)
			srcOff++
			dstOff++
		}
	}

	subset := originalDstBounds.Offset(-dstOrigin.X, -dstOrigin.Y)
	return NewSpecialImage(subset, dst)
}

///////////////////////////////////////////////////////////////////////////////
// GaussianPass (true 1D kernel, sigma < 2)

type gaussianPass struct {
	kernel    []float32
	srcBuffer [][4]float32
	radius    int32
	window    int32
	base      int32
}

func makeGaussianPassMaker(sigma float32) *passMaker {
	const maxSigma = 2
	if sigma >= maxSigma {
		return nil
	}
	return &passMaker{
		window: 2*SigmaToRadius(sigma) + 1,
		sigma:  sigma,
		makePass: func() blurPass {
			radius := SigmaToRadius(sigma)
			window := 2*radius + 1
			p := &gaussianPass{
				radius:    radius,
				window:    window,
				kernel:    make([]float32, window),
				srcBuffer: make([][4]float32, window),
			}
			compute1DBlurKernel(sigma, radius, p.kernel)
			return p
		},
	}
}

func (p *gaussianPass) border() int32 { return p.radius }

func (p *gaussianPass) startBlur() {
	for i := range p.srcBuffer {
		p.srcBuffer[i] = [4]float32{}
	}
	p.base = 0
}

func (p *gaussianPass) blurSegment(n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	gaussianBlurSegmentFn(p, n, src, srcStride, dst, dstStride)
}

// gaussianBlurSegmentGeneric is the portable windowed dot product: per output sample the leading source pixel is
// unpacked into the circular buffer and the whole window is re-summed against the kernel, four float channels at a
// time. Both float sites here are plain Go expressions, so the compiler is free to contract them into fused
// multiply-adds and on arm64 it does; any vector twin has to reproduce the lowering of *this* build (see
// blurengine_simd_arm64.go's and blurengine_simd_amd64.go's exprMulAdd4).
func gaussianBlurSegmentGeneric(p *gaussianPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	base := p.base
	window := p.window
	srcCursor := int32(0)
	dstCursor := int32(0)
	for ; n > 0; n-- {
		var leadingEdge [4]float32
		if src != nil {
			w := src[srcCursor]
			const inv255 = 1.0 / 255.0
			leadingEdge = [4]float32{
				float32(w&0xff) * inv255,
				float32((w>>8)&0xff) * inv255,
				float32((w>>16)&0xff) * inv255,
				float32(w>>24) * inv255,
			}
			srcCursor += srcStride
		}
		p.srcBuffer[(base+window-1)%window] = leadingEdge

		if dst != nil {
			var sum [4]float32
			for i := int32(0); i < window; i++ {
				s := (i + base) % window
				k := p.kernel[i]
				sum[0] += p.srcBuffer[s][0] * k
				sum[1] += p.srcBuffer[s][1] * k
				sum[2] += p.srcBuffer[s][2] * k
				sum[3] += p.srcBuffer[s][3] * k
			}
			var out uint32
			for c := range 4 {
				v := sum[c]*255 + 0.5
				if v > 255 {
					v = 255
				} else if v < 0 {
					v = 0
				}
				out |= uint32(uint8(v)) << (8 * c)
			}
			dst[dstCursor] = out
			dstCursor += dstStride
		}
		base = (base + 1) % window
	}
	p.base = base
}

///////////////////////////////////////////////////////////////////////////////
// ThreeBoxApproxPass (successive box approximation, sigma < 135)

type threeBoxApproxPass struct {
	buffer0, buffer1, buffer2 [][4]uint32
	borderPx                  int32
	divisorFactor             uint32 // ScaledDividerU32
	half                      uint32

	// blur state
	sum0, sum1, sum2 [4]uint32
	cursor0          int32
	cursor1          int32
	cursor2          int32
}

func makeThreeBoxPassMaker(sigma float32) *passMaker {
	window := boxBlurWindow(sigma)
	if window >= 255 {
		return nil
	}
	return &passMaker{
		window: window,
		sigma:  sigma,
		makePass: func() blurPass {
			passSize := window - 1
			// If the window is odd just one buffer size is needed; even windows give the third pass one more element.
			size2 := passSize
			if window&1 == 0 {
				size2 = passSize + 1
			}
			// Odd windows: border = 3*((window-1)/2); even: 3*(window/2) - 1.
			var border int32
			if window&1 == 1 {
				border = 3 * ((window - 1) / 2)
			} else {
				border = 3*(window/2) - 1
			}
			// Odd windows divide by window^3; even by window^2 * (window+1).
			window2 := window * window
			window3 := window2 * window
			divisor := uint32(window3)
			if window&1 == 0 {
				divisor = uint32(window3 + window2)
			}
			return &threeBoxApproxPass{
				buffer0:       make([][4]uint32, passSize),
				buffer1:       make([][4]uint32, passSize),
				buffer2:       make([][4]uint32, size2),
				borderPx:      border,
				divisorFactor: uint32(math.Round((1.0 / float64(divisor)) * float64(uint64(1)<<32))),
				half:          (divisor + 1) >> 1,
			}
		},
	}
}

func (p *threeBoxApproxPass) border() int32 { return p.borderPx }

func (p *threeBoxApproxPass) startBlur() {
	p.sum0 = [4]uint32{}
	p.sum1 = [4]uint32{}
	p.sum2 = [4]uint32{p.half, p.half, p.half, p.half}
	for i := range p.buffer0 {
		p.buffer0[i] = [4]uint32{}
	}
	for i := range p.buffer1 {
		p.buffer1[i] = [4]uint32{}
	}
	for i := range p.buffer2 {
		p.buffer2[i] = [4]uint32{}
	}
	p.cursor0 = 0
	p.cursor1 = 0
	p.cursor2 = 0
}

func (p *threeBoxApproxPass) divide(v uint32) uint32 {
	return uint32((uint64(v) * uint64(p.divisorFactor)) >> 32)
}

func (p *threeBoxApproxPass) blurSegment(n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	threeBoxBlurSegmentFn(p, n, src, srcStride, dst, dstStride)
}

// threeBoxBlurSegmentGeneric is the portable prefix-sum pipeline: three chained running sums per channel, each reduced
// by the trailing edge its circular buffer remembers. The scan axis carries a true loop-carried dependency, so the
// only vectorizable axis is the four channels.
func threeBoxBlurSegmentGeneric(p *threeBoxApproxPass, n int32, src []uint32, srcStride int32, dst []uint32, dstStride int32) {
	sum0, sum1, sum2 := p.sum0, p.sum1, p.sum2
	c0, c1, c2 := p.cursor0, p.cursor1, p.cursor2
	srcCursor := int32(0)
	dstCursor := int32(0)
	for ; n > 0; n-- {
		var leadingEdge [4]uint32
		if src != nil {
			w := src[srcCursor]
			leadingEdge = [4]uint32{w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, w >> 24}
			srcCursor += srcStride
		}

		var blurred [4]uint32
		for c := range 4 {
			sum0[c] += leadingEdge[c]
			sum1[c] += sum0[c]
			sum2[c] += sum1[c]
			blurred[c] = p.divide(sum2[c])
		}
		// Reduce the sums by the trailing edges stored in the circular buffers, storing the current sums for the next
		// go-around *before* their own trailing-edge subtraction.
		trailing2 := p.buffer2[c2]
		p.buffer2[c2] = sum1
		c2++
		if c2 >= int32(len(p.buffer2)) {
			c2 = 0
		}
		trailing1 := p.buffer1[c1]
		p.buffer1[c1] = sum0
		c1++
		if c1 >= int32(len(p.buffer1)) {
			c1 = 0
		}
		trailing0 := p.buffer0[c0]
		p.buffer0[c0] = leadingEdge
		c0++
		if c0 >= int32(len(p.buffer0)) {
			c0 = 0
		}
		for c := range 4 {
			sum2[c] -= trailing2[c]
			sum1[c] -= trailing1[c]
			sum0[c] -= trailing0[c]
		}

		if dst != nil {
			dst[dstCursor] = uint32(uint8(blurred[0])) |
				uint32(uint8(blurred[1]))<<8 |
				uint32(uint8(blurred[2]))<<16 |
				uint32(uint8(blurred[3]))<<24
			dstCursor += dstStride
		}
	}
	p.sum0, p.sum1, p.sum2 = sum0, sum1, sum2
	p.cursor0, p.cursor1, p.cursor2 = c0, c1, c2
}

///////////////////////////////////////////////////////////////////////////////
// Raster8888BlurAlgorithm

type raster8888BlurAlgorithm struct{}

// MaxSigma is 135, so non-legacy blurs always use the gaussianPass/threeBoxApproxPass implementations (a tent-filter
// pass would only be needed for legacy blurs).
func (raster8888BlurAlgorithm) MaxSigma() float32 { return 135 }

// SupportsOnlyDecalTiling is true: tile modes are applied via the crop machinery and carried as FilterResult metadata.
func (raster8888BlurAlgorithm) SupportsOnlyDecalTiling() bool { return true }

func (raster8888BlurAlgorithm) Blur(sigma geom.Size, src *SpecialImage, srcBounds geom.IRect, tileMode shaders.TileMode, dstBounds geom.IRect) *SpecialImage {
	if tileMode != shaders.TileDecal {
		panic("raster blur supports only decal tiling")
	}
	makeMaker := func(sigma float32) *passMaker {
		if maker := makeGaussianPassMaker(sigma); maker != nil {
			return maker
		}
		if maker := makeThreeBoxPassMaker(sigma); maker != nil {
			return maker
		}
		panic("sigma out of range")
	}
	// The passes read pixel storage as 32-bit words, so an N32 backing is a hard requirement (FilterResult.rescale
	// re-renders anything else before the blur sees it), and a drawable backing whose readback failed resolves to nil
	// pixels. Both are the BlurAlgorithm.Blur contract's failure signal — a nil result — rather than a panic, matching
	// the GPU algorithm's failed-resolve/failed-upload lanes.
	px := src.subsetPixels()
	if px == nil || px.Info.ColorType != imagecore.ColorTypeN32 {
		return nil
	}
	makerX := makeMaker(sigma.Width)
	makerY := makeMaker(sigma.Height)
	return evalBlurPasses(makerX, makerY, px, srcBounds, dstBounds)
}
