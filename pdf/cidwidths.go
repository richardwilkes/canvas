// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file builds the /W array of a CIDFontType2 font, giving the (base-1000) advance of every used glyph that differs
// from the default (/DW). Advances come from the typeface's design-space advances (font.Typeface.DesignAdvance) read
// from a strike sized at unitsPerEm, and are compacted into runs and ranges to keep the array small.

package pdf

import (
	"sort"

	"github.com/richardwilkes/canvas/font"
)

// widthFromFontUnits scales a design-unit advance to the base-1000 glyph space (from_font_units).
func widthFromFontUnits(scaled float32, emSize int) float32 {
	if emSize == 1000 {
		return scaled
	}
	return scaled * 1000 / float32(emSize)
}

// findModeOr0 returns the most common advance in a sorted slice, or 0 for an empty slice.
func findModeOr0(advances []float32) float32 {
	if len(advances) == 0 {
		return 0
	}
	currentAdvance := advances[0]
	currentModeAdvance := advances[0]
	currentCount := 1
	currentModeCount := 1
	for i := 1; i < len(advances); i++ {
		if advances[i] == currentAdvance {
			currentCount++
		} else {
			if currentCount > currentModeCount {
				currentModeAdvance = currentAdvance
				currentModeCount = currentCount
			}
			currentAdvance = advances[i]
			currentCount = 1
		}
	}
	if currentCount > currentModeCount {
		return currentAdvance
	}
	return currentModeAdvance
}

// makeCIDGlyphWidthsArray returns the /W array and the default advance to write as /DW. The array is nil (with a zero
// default) only when the typeface reports a non-positive units-per-em; when no glyphs are used it is non-nil but empty,
// so callers must check both nil and Size() before emitting it.
func makeCIDGlyphWidthsArray(tf *font.Typeface, subset *glyphUse) (w *Array, defaultAdvance int32) {
	emSize := tf.UnitsPerEm()
	if emSize <= 0 {
		return nil, 0
	}

	var glyphIDs []uint16
	subset.forEach(func(gid uint16) { glyphIDs = append(glyphIDs, gid) })

	result := NewArray()

	advances := make([]float32, len(glyphIDs))

	// Find the pdf integer mode (most common integer advance). Poppler enforces that /DW is an integer, so only integer
	// advances are considered.
	intAdvances := make([]float32, 0, len(glyphIDs))
	for _, gid := range glyphIDs {
		adv := widthFromFontUnits(tf.DesignAdvance(gid), emSize)
		if float32(int32(adv)) == adv {
			intAdvances = append(intAdvances, adv)
		}
	}
	sort.Slice(intAdvances, func(i, j int) bool { return intAdvances[i] < intAdvances[j] })
	modeAdvance := int32(findModeOr0(intAdvances))

	// Pre-convert to pdf advances.
	for i, gid := range glyphIDs {
		advances[i] = widthFromFontUnits(tf.DesignAdvance(gid), emSize)
	}

	for i := 0; i < len(glyphIDs); i++ {
		advance := advances[i]

		// a. Skipping don't cares or defaults is a win (trivial).
		if advance == float32(modeAdvance) {
			continue
		}

		// b. 2+ repeats create a run as long as possible, else start a range.
		{
			j := i + 1 // j is always one past the last known repeat
			for ; j < len(glyphIDs); j++ {
				if advance != advances[j] {
					break
				}
			}
			if j-i >= 2 {
				result.AppendInt(int32(glyphIDs[i]))
				result.AppendInt(int32(glyphIDs[j-1]))
				result.AppendScalar(advance)
				i = j - 1
				continue
			}
		}

		{
			result.AppendInt(int32(glyphIDs[i]))
			advanceArray := NewArray()
			advanceArray.AppendScalar(advance)
			j := i + 1 // j is always one past the last output
			for ; j < len(glyphIDs); j++ {
				advance = advances[j]

				// c. End the range if the default is seen.
				if advance == float32(modeAdvance) {
					break
				}

				dontCares := int(glyphIDs[j]) - int(glyphIDs[j-1]) - 1
				// d. End the range on 4+ don't cares.
				if dontCares >= 4 {
					break
				}

				var nextAdvance float32
				// e. End the range for 2+ repeats with 4+ don't cares.
				if j+1 < len(glyphIDs) {
					nextAdvance = advances[j+1]
					nextDontCares := int(glyphIDs[j+1]) - int(glyphIDs[j]) - 1
					if advance == nextAdvance && dontCares+nextDontCares >= 4 {
						break
					}
				}

				// f. End the range for 3+ repeats.
				if j+2 < len(glyphIDs) && advance == nextAdvance {
					nextAdvance = advances[j+2]
					if advance == nextAdvance {
						break
					}
				}

				for ; dontCares > 0; dontCares-- {
					advanceArray.AppendScalar(0)
				}
				advanceArray.AppendScalar(advance)
			}
			result.AppendObject(advanceArray)
			i = j - 1
		}
	}

	return result, modeAdvance
}
