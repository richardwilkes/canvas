// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// PerlinNoiseShader: the painting-data tables built with the SVG feTurbulence LCG (so seeded output is
// bit-reproducible) and the perlin-noise raster-pipeline stage, both computing the same per-lane float math the oracle
// verifies against — iround is round-to-nearest-even.

package shaders

import (
	"math"
	"sync/atomic"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// PerlinNoiseType selects between fractal noise and turbulence.
type PerlinNoiseType uint8

// PerlinNoiseType values.
const (
	PerlinFractalNoise PerlinNoiseType = iota
	PerlinTurbulence
)

const (
	perlinBlockSize   = 256
	perlinNoise       = 4096
	perlinRandMaximum = math.MaxInt32
	perlinMaxOctaves  = 255
)

// PerlinNoiseShader generates Perlin noise or turbulence; the painting data is built eagerly at construction.
type PerlinNoiseShader struct {
	numOctaves    int
	baseFreqX     float32
	baseFreqY     float32
	stitchDataInX int32
	stitchDataInY int32
	// uniqueID is a process-unique identifier used to key the GPU permutations/noise textures, one per painting-data
	// instance. Assigned eagerly at construction (the shader is immutable and only ever handled by pointer, so a plain
	// field suffices — no atomic-in-struct copy hazard).
	uniqueID        uint32
	noise           [4][perlinBlockSize][2]uint16
	latticeSelector [perlinBlockSize]uint8
	noiseType       PerlinNoiseType
	stitchTiles     bool
}

// nextPerlinUniqueID hands out the process-unique IDs used to key GPU noise textures.
var nextPerlinUniqueID atomic.Uint32

// NewFractalNoise builds a fractal-noise Perlin shader.
func NewFractalNoise(baseFreqX, baseFreqY float32, numOctaves int, seed float32, tileW, tileH int32) Shader {
	if !validPerlinInput(baseFreqX, baseFreqY, numOctaves, tileW, tileH, seed) {
		return nil
	}
	if numOctaves == 0 {
		// The entire shader collapses to [0,0,0,0] * 0.5 + 0.5.
		return NewColor4f(colorcore.Color4f{R: 0.5, G: 0.5, B: 0.5, A: 0.5})
	}
	return newPerlin(PerlinFractalNoise, baseFreqX, baseFreqY, numOctaves, seed, tileW, tileH)
}

// NewTurbulence builds a turbulence Perlin shader.
func NewTurbulence(baseFreqX, baseFreqY float32, numOctaves int, seed float32, tileW, tileH int32) Shader {
	if !validPerlinInput(baseFreqX, baseFreqY, numOctaves, tileW, tileH, seed) {
		return nil
	}
	if numOctaves == 0 {
		// Collapses to transparent black.
		return NewColor4f(colorcore.Color4f{})
	}
	return newPerlin(PerlinTurbulence, baseFreqX, baseFreqY, numOctaves, seed, tileW, tileH)
}

// validPerlinInput validates the parameters shared by NewFractalNoise and NewTurbulence.
func validPerlinInput(baseX, baseY float32, numOctaves int, tileW, tileH int32, seed float32) bool {
	if !(baseX >= 0 && baseY >= 0) {
		return false
	}
	if numOctaves < 0 || numOctaves > perlinMaxOctaves {
		return false
	}
	if tileW < 0 || tileH < 0 {
		return false
	}
	return !math.IsNaN(float64(seed)) && !math.IsInf(float64(seed), 0)
}

func newPerlin(nt PerlinNoiseType, fx, fy float32, octaves int, seed float32, tileW, tileH int32) *PerlinNoiseShader {
	s := &PerlinNoiseShader{
		noiseType:   nt,
		baseFreqX:   fx,
		baseFreqY:   fy,
		numOctaves:  octaves,
		stitchTiles: tileW != 0 && tileH != 0,
		uniqueID:    nextPerlinUniqueID.Add(1),
	}
	s.initPaintingData(seed, tileW, tileH)
	return s
}

// NoiseType returns the noise type (for the FP conversion).
func (s *PerlinNoiseShader) NoiseType() PerlinNoiseType { return s.noiseType }

// NumOctaves returns the octave count (FP conversion).
func (s *PerlinNoiseShader) NumOctaves() int { return s.numOctaves }

// StitchTiles reports whether tile-stitching is enabled (FP conversion).
func (s *PerlinNoiseShader) StitchTiles() bool { return s.stitchTiles }

// BaseFrequency returns the stitch-adjusted base frequency. It feeds the GPU fragment processor's baseFrequency uniform
// (FP conversion).
func (s *PerlinNoiseShader) BaseFrequency() (x, y float32) { return s.baseFreqX, s.baseFreqY }

// StitchData returns the stitch data (only meaningful when StitchTiles is true). It feeds the GPU fragment processor's
// stitchData uniform (FP conversion).
func (s *PerlinNoiseShader) StitchData() (w, h int32) { return s.stitchDataInX, s.stitchDataInY }

// UniqueID returns the process-unique identifier used to key the GPU noise textures.
func (s *PerlinNoiseShader) UniqueID() uint32 { return s.uniqueID }

// PermutationsBitmap returns the A8 256x1 lattice-selector table, one byte per lattice index. The fragment processor
// uploads it as the permutations texture, sampled RepeatX/Nearest.
func (s *PerlinNoiseShader) PermutationsBitmap() []byte {
	out := make([]byte, perlinBlockSize)
	copy(out, s.latticeSelector[:])
	return out
}

// NoiseBitmap returns the RGBA8888 256x4 gradient table: one row per channel, each texel the two 16-bit gradient
// components (noise[ch][i][0] then [1]) laid out low byte then high byte in native little-endian order. The fragment
// processor uploads it as the noise texture and recombines the byte pairs (high in G/A, low in R/B) back into ~16-bit
// gradient vectors.
func (s *PerlinNoiseShader) NoiseBitmap() []byte {
	out := make([]byte, 4*perlinBlockSize*4)
	idx := 0
	for channel := range 4 {
		for i := range perlinBlockSize {
			n0 := s.noise[channel][i][0]
			n1 := s.noise[channel][i][1]
			out[idx+0] = byte(n0)
			out[idx+1] = byte(n0 >> 8)
			out[idx+2] = byte(n1)
			out[idx+3] = byte(n1 >> 8)
			idx += 4
		}
	}
	return out
}

// IsOpaque implements Shader (perlin noise is never opaque).
func (s *PerlinNoiseShader) IsOpaque() bool { return false }

// IsConstant implements Shader.
func (s *PerlinNoiseShader) IsConstant() bool { return false }

// initPaintingData builds the LCG-seeded lattice/noise tables and applies the stitch frequency adjustment.
func (s *PerlinNoiseShader) initPaintingData(seed float32, tileW, tileH int32) {
	// See https://www.w3.org/TR/SVG11/filters.html#feTurbulenceElement
	const randAmplitude = 16807 // 7**5; primitive root of m
	const randQ = 127773        // m / a
	const randR = 2836          // m % a

	// According to the SVG spec, we must truncate (not round) the seed value.
	iseed := int32(seed)
	if iseed <= 0 {
		iseed = -(iseed % (perlinRandMaximum - 1)) + 1
	}
	if iseed > perlinRandMaximum-1 {
		iseed = perlinRandMaximum - 1
	}
	random := func() int32 {
		result := randAmplitude*(iseed%randQ) - randR*(iseed/randQ)
		if result <= 0 {
			result += perlinRandMaximum
		}
		iseed = result
		return result
	}

	for channel := range 4 {
		for i := range perlinBlockSize {
			s.latticeSelector[i] = uint8(i)
			s.noise[channel][i][0] = uint16(random() % (2 * perlinBlockSize))
			s.noise[channel][i][1] = uint16(random() % (2 * perlinBlockSize))
		}
	}
	for i := perlinBlockSize - 1; i > 0; i-- {
		k := s.latticeSelector[i]
		j := random() % perlinBlockSize
		s.latticeSelector[i] = s.latticeSelector[j]
		s.latticeSelector[j] = k
	}

	// Perform the permutations now.
	noise := s.noise
	for i := range perlinBlockSize {
		for channel := range 4 {
			for j := range 2 {
				s.noise[channel][i][j] = noise[channel][s.latticeSelector[i]][j]
			}
		}
	}

	// Compute normalized gradients from the permuted noise data: the magnitude math runs in double and scales the float
	// components by the double reciprocal.
	const halfMax16bits = 32767.5
	const invBlockSize = 1.0 / float32(perlinBlockSize)
	for channel := range 4 {
		for i := range perlinBlockSize {
			gx := float32(int32(s.noise[channel][i][0])-perlinBlockSize) * invBlockSize
			gy := float32(int32(s.noise[channel][i][1])-perlinBlockSize) * invBlockSize
			dmag := math.Sqrt(float64(gx)*float64(gx) + float64(gy)*float64(gy))
			dscale := 1.0 / dmag
			nx := float32(float64(gx) * dscale)
			ny := float32(float64(gy) * dscale)
			if math.IsInf(float64(nx), 0) || math.IsNaN(float64(nx)) ||
				math.IsInf(float64(ny), 0) || math.IsNaN(float64(ny)) {
				nx, ny = 0, 0
			}
			// Round to nearest via floor(x + 0.5).
			s.noise[channel][i][0] = uint16(int32(math.Floor(float64((nx+1)*halfMax16bits + 0.5))))
			s.noise[channel][i][1] = uint16(int32(math.Floor(float64((ny+1)*halfMax16bits + 0.5))))
		}
	}

	if s.stitchTiles {
		// When stitching tiled turbulence, adjust the frequencies so tile borders are continuous.
		tileWf := float32(tileW)
		tileHf := float32(tileH)
		if s.baseFreqX != 0 {
			lo := floorf(tileWf*s.baseFreqX) / tileWf
			hi := ceilf(tileWf*s.baseFreqX) / tileWf
			if ieeeFloatDivide(s.baseFreqX, lo) < hi/s.baseFreqX {
				s.baseFreqX = lo
			} else {
				s.baseFreqX = hi
			}
		}
		if s.baseFreqY != 0 {
			lo := floorf(tileHf*s.baseFreqY) / tileHf
			hi := ceilf(tileHf*s.baseFreqY) / tileHf
			if ieeeFloatDivide(s.baseFreqY, lo) < hi/s.baseFreqY {
				s.baseFreqY = lo
			} else {
				s.baseFreqY = hi
			}
		}
		// Stitch data from (tileW * freqX, tileH * freqY), with each extent capped at MaxInt32 - perlinNoise.
		w := int32(math.Floor(float64(tileWf*s.baseFreqX + 0.5)))
		h := int32(math.Floor(float64(tileHf*s.baseFreqY + 0.5)))
		if w > math.MaxInt32-perlinNoise {
			w = math.MaxInt32 - perlinNoise
		}
		if h > math.MaxInt32-perlinNoise {
			h = math.MaxInt32 - perlinNoise
		}
		s.stitchDataInX = w
		s.stitchDataInY = h
	}
}

func ceilf(v float32) float32 { return float32(math.Ceil(float64(v))) }

// ieeeFloatDivide is an ordinary IEEE division (NaN/inf allowed), named to mark intent at call sites.
func ieeeFloatDivide(a, b float32) float32 { return a / b }

// iroundEven rounds to the nearest integer, ties to even.
func iroundEven(v float32) int32 { return int32(math.RoundToEven(float64(v))) }

// computePerlinVector evaluates one gradient's contribution at the given fractional offset.
func computePerlinVector(sampleLo, sampleHi uint16, x, y float32) float32 {
	vecX := madf(float32(sampleLo), 2.0/65535.0, -1.0)
	vecY := madf(float32(sampleHi), 2.0/65535.0, -1.0)
	return madf(vecX, x, vecY*y)
}

// appendStages appends the perlin-noise sampling stage.
func (s *PerlinNoiseShader) appendStages(p *Pipeline, m MatrixRec) bool { //nolint:gocritic // see Shader.appendStages
	identity := geom.IdentityMatrix()
	if _, ok := m.apply(p, &identity); !ok {
		return false
	}
	// The shader itself is the stage context (its immutable painting-data tables), so the stage stays a static function
	// rather than a closure capturing s.
	p.appendCtx(perlinNoiseStage, s)
	return true
}

// perlinNoiseStage is the perlin_noise stage; the *PerlinNoiseShader is the stage context.
func perlinNoiseStage(z *lanes) {
	s := z.ctx.(*PerlinNoiseShader)
	for i := range z.n {
		z.r[i], z.g[i], z.b[i], z.a[i] = s.samplePixel(z.r[i], z.g[i])
	}
}

// samplePixel is the perlin_noise stage body for one lane.
func (s *PerlinNoiseShader) samplePixel(x, y float32) (r, g, b, a float32) {
	noiseVecX := (x + 0.5) * s.baseFreqX
	noiseVecY := (y + 0.5) * s.baseFreqY
	var out [4]float32
	stitchDataX := float32(s.stitchDataInX)
	stitchDataY := float32(s.stitchDataInY)
	ratio := float32(1)

	for octave := 0; octave < s.numOctaves; octave++ {
		floorValX := floorf(noiseVecX)
		floorValY := floorf(noiseVecY)
		ceilValX := floorValX + 1
		ceilValY := floorValY + 1
		fractValX := noiseVecX - floorValX
		fractValY := noiseVecY - floorValY

		if s.stitchTiles {
			// Wrap the coordinates to the stitch position.
			if floorValX >= stitchDataX {
				floorValX -= stitchDataX
			}
			if floorValY >= stitchDataY {
				floorValY -= stitchDataY
			}
			if ceilValX >= stitchDataX {
				ceilValX -= stitchDataX
			}
			if ceilValY >= stitchDataY {
				ceilValY -= stitchDataY
			}
		}

		latticeIdxX := float32(s.latticeSelector[uint32(iroundEven(floorValX))&0xFF])
		latticeIdxY := float32(s.latticeSelector[uint32(iroundEven(ceilValX))&0xFF])

		b00 := uint32(iroundEven(latticeIdxX+floorValY)) & 0xFF
		b10 := uint32(iroundEven(latticeIdxY+floorValY)) & 0xFF
		b01 := uint32(iroundEven(latticeIdxX+ceilValY)) & 0xFF
		b11 := uint32(iroundEven(latticeIdxY+ceilValY)) & 0xFF

		// Hermite interpolation of the fractional value.
		smoothX := fractValX * fractValX * (3 - 2*fractValX)
		smoothY := fractValY * fractValY * (3 - 2*fractValY)

		for channel := range 4 {
			sample00 := &s.noise[channel][b00]
			sample10 := &s.noise[channel][b10]
			sample01 := &s.noise[channel][b01]
			sample11 := &s.noise[channel][b11]

			u := computePerlinVector(sample00[0], sample00[1], fractValX, fractValY)
			v := computePerlinVector(sample10[0], sample10[1], fractValX-1, fractValY)
			A := lerpf(u, v, smoothX)

			u = computePerlinVector(sample01[0], sample01[1], fractValX, fractValY-1)
			v = computePerlinVector(sample11[0], sample11[1], fractValX-1, fractValY-1)
			B := lerpf(u, v, smoothX)

			color := lerpf(A, B, smoothY)
			if s.noiseType != PerlinFractalNoise {
				// For turbulence the result is abs(noise[-1,1]).
				color = absf(color)
			}
			out[channel] = madf(color, ratio, out[channel])
		}

		noiseVecX *= 2
		noiseVecY *= 2
		stitchDataX *= 2
		stitchDataY *= 2
		ratio *= 0.5
	}

	if s.noiseType == PerlinFractalNoise {
		// For fractal noise the result is noise[-1,1] * 0.5 + 0.5.
		for channel := range 4 {
			out[channel] = madf(out[channel], 0.5, 0.5)
		}
	}

	// Note: rgb premultiplies by the *unclamped* alpha, then alpha itself is clamped.
	rawA := out[3]
	return clamp01(out[0]) * rawA, clamp01(out[1]) * rawA, clamp01(out[2]) * rawA, clamp01(rawA)
}

// lerpf computes lerp(from, to, t) = mad(to-from, t, from).
func lerpf(from, to, t float32) float32 {
	return madf(to-from, t, from)
}
