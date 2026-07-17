// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// AlphaRuns: a sparse run-length-encoded scanline of alpha (coverage) values. Sparseness allows several paths to be
// composed independently into the same buffer. The Break/BreakAt helpers live in blitter.go for RectClipBlitter; the
// instance side lands here for the AA scan converters.

package raster

// AlphaRuns holds a run-length-encoded coverage scanline. Runs is the zero-terminated run-length array and Alpha the
// coverage values, coupled by index in the same order BlitAntiH consumes them.
type AlphaRuns struct {
	Runs  []int16
	Alpha []Alpha
}

// CatchOverflow returns 0-255 given 0-256.
func CatchOverflow(alpha int) Alpha {
	return Alpha(alpha - (alpha >> 8))
}

// Empty reports whether the scanline contains only a single run of alpha 0.
func (a *AlphaRuns) Empty() bool {
	return a.Alpha[0] == 0 && a.Runs[a.Runs[0]] == 0
}

// Reset reinitializes the scanline for a new scanline of the given width.
func (a *AlphaRuns) Reset(width int) {
	a.Runs[0] = int16(width)
	a.Runs[width] = 0
	a.Alpha[0] = 0
}

// Add inserts into the buffer a run starting at (x - offsetX):
//
//	if startAlpha > 0: one pixel with value += startAlpha (max 255)
//	if middleCount > 0: middleCount pixels with value += maxValue
//	if stopAlpha > 0: one pixel with value += stopAlpha
//
// Returns the offsetX value that should be passed on the next call, assuming we're on the same scanline; if the caller
// is switching scanlines, offsetX should be 0.
func (a *AlphaRuns) Add(x int, startAlpha uint8, middleCount int, stopAlpha, maxValue uint8, offsetX int) int {
	runs := offsetX  // index into a.Runs
	alpha := offsetX // index into a.Alpha
	lastAlpha := alpha
	x -= offsetX

	if startAlpha != 0 {
		AlphaRunsBreak(a.Runs[runs:], a.Alpha[alpha:], x, 1)
		// I should be able to just add alpha[x] + startAlpha. However, if the trailing edge of the previous span and
		// the leading edge of the current span round to the same super-sampled x value, I might overflow to 256 with
		// this add, hence the funny subtract (crud).
		tmp := uint32(a.Alpha[alpha+x]) + uint32(startAlpha)
		a.Alpha[alpha+x] = Alpha(tmp - (tmp >> 8))

		runs += x + 1
		alpha += x + 1
		x = 0
	}

	if middleCount != 0 {
		AlphaRunsBreak(a.Runs[runs:], a.Alpha[alpha:], x, middleCount)
		alpha += x
		runs += x
		x = 0
		for {
			a.Alpha[alpha] = CatchOverflow(int(a.Alpha[alpha]) + int(maxValue))
			n := int(a.Runs[runs])
			alpha += n
			runs += n
			middleCount -= n
			if middleCount <= 0 {
				break
			}
		}
		lastAlpha = alpha
	}

	if stopAlpha != 0 {
		AlphaRunsBreak(a.Runs[runs:], a.Alpha[alpha:], x, 1)
		alpha += x
		a.Alpha[alpha] += stopAlpha
		lastAlpha = alpha
	}

	return lastAlpha // new offsetX
}
