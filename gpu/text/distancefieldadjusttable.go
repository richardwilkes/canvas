// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// DistanceFieldAdjustTable holds the per-luminance distance corrections the LCD distance-field geometry processor
// applies, approximating the mask gamma hack by manipulating the glyph geometry (the distance to the edge) instead of
// the mask coverage. Built from the same mask-gamma rows the scaler's LCD pre-blend uses (font.GammaLUTData). This
// codebase only targets sRGB (non-linearly blended) destinations, so there is only one table and no gamma-correct-table
// selector.

package text

import (
	"sync"

	"github.com/richardwilkes/canvas/font"
)

// distanceAdjustLumShift selects which table row a luminance value maps to.
const distanceAdjustLumShift = 5

// distanceFieldAAFactor matches the anti-aliasing factor baked into the distance-field shaders.
const distanceFieldAAFactor = 0.65

// DistanceFieldAdjustTable holds the built per-luminance-row distance adjustment table.
type DistanceFieldAdjustTable struct {
	table [8]float32
}

var (
	dfAdjustOnce  sync.Once
	dfAdjustTable DistanceFieldAdjustTable
)

// GetDistanceFieldAdjustTable returns the process-wide adjustment table, building it on first use.
func GetDistanceFieldAdjustTable() *DistanceFieldAdjustTable {
	dfAdjustOnce.Do(func() { dfAdjustTable.build() })
	return &dfAdjustTable
}

// GetAdjustment returns the distance adjustment for a given luminance value.
func (t *DistanceFieldAdjustTable) GetAdjustment(lum int) float32 {
	return t.table[lum>>distanceAdjustLumShift]
}

// build fills the table: for each mask-gamma row, find the mask value whose adjusted coverage is 0.5, invert the
// smoothstep to a t value, and convert that to a distance. Rows with no 0.5 crossing keep adjustment 0 (no valid data
// means no adjustment).
func (t *DistanceFieldAdjustTable) build() {
	data := font.GammaLUTData()
	for row := range t.table {
		rowData := &data[row]
		for col := 0; col < 255; col++ {
			if rowData[col] > 127 || rowData[col+1] < 128 {
				continue
			}
			// Compute the point where a mask value will give us a result of 0.5.
			interp := (127.5 - float32(rowData[col])) / float32(rowData[col+1]-rowData[col])
			borderAlpha := (float32(col) + interp) / 255.0

			// Compute the t value for that alpha: an approximate inverse for smoothstep().
			tVal := borderAlpha * (borderAlpha*(4.0*borderAlpha-6.0) + 5.0) / 3.0

			// Compute the distance which gives us that t value.
			t.table[row] = 2.0*distanceFieldAAFactor*tVal - distanceFieldAAFactor
			break
		}
	}
}
