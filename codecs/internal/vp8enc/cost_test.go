// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package vp8enc

import "testing"

// TestFixedCostTables pins the embedded cost tables against spot values from libwebp's sources (src/enc/cost_enc.c,
// src/dsp/cost.c), guarding against extraction drift.
func TestFixedCostTables(t *testing.T) {
	// Spot values from libwebp's VP8LevelFixedCosts table.
	levelSpots := map[int]uint16{
		0:    0,
		1:    256,
		2:    256,
		4:    256,
		5:    432,
		6:    618,
		7:    630,
		8:    731,
		9:    640, // libwebp's table omits the parity-bit cost here; embedded verbatim
		11:   828,
		18:   1294,
		19:   1042,
		66:   1905,
		67:   1727,
		68:   1733,
		2047: 7761,
	}
	for level, want := range levelSpots {
		if got := levelFixedCosts[level]; got != want {
			t.Errorf("levelFixedCosts[%d] = %d, want %d", level, got, want)
		}
	}

	if got, want := fixedCostsI16, [numPredModes]uint16{663, 919, 872, 919}; got != want {
		t.Errorf("fixedCostsI16 = %v, want %v", got, want)
	}
	if got, want := fixedCostsUV, [numPredModes]uint16{302, 984, 439, 642}; got != want {
		t.Errorf("fixedCostsUV = %v, want %v", got, want)
	}

	// Spot rows and entries from libwebp's VP8FixedCostsI4[top][left][mode].
	wantRow00 := [numBModes]uint16{40, 1151, 1723, 1874, 2103, 2019, 1628, 1777, 2226, 2137}
	if fixedCostsI4[0][0] != wantRow00 {
		t.Errorf("fixedCostsI4[0][0] = %v, want %v", fixedCostsI4[0][0], wantRow00)
	}
	wantRow01 := [numBModes]uint16{192, 469, 1296, 1308, 1849, 1794, 1781, 1703, 1713, 1522}
	if fixedCostsI4[0][1] != wantRow01 {
		t.Errorf("fixedCostsI4[0][1] = %v, want %v", fixedCostsI4[0][1], wantRow01)
	}
	if got := fixedCostsI4[1][0][3]; got != 1491 {
		t.Errorf("fixedCostsI4[1][0][3] = %d, want 1491", got)
	}
	if got := fixedCostsI4[9][9][9]; got != 501 {
		t.Errorf("fixedCostsI4[9][9][9] = %d, want 501", got)
	}
	if got := fixedCostsI4[5][5][6]; got != 2881 {
		t.Errorf("fixedCostsI4[5][5][6] = %d, want 2881", got)
	}
}

// TestQualityToQuantizer pins the quality → quantizer-index mapping against values computed from libwebp's
// QualityToCompression/VP8SetSegmentParams math.
func TestQualityToQuantizer(t *testing.T) {
	cases := []struct {
		quality float32
		want    int
	}{
		{quality: 0, want: 127},
		{quality: 10, want: 75},
		{quality: 50, want: 38},
		{quality: 75, want: 26},
		{quality: 90, want: 9},
		{quality: 100, want: 0},
	}
	for _, c := range cases {
		e := newEncoder(16, 16)
		e.setSegmentParams(c.quality)
		if e.baseQuant != c.want {
			t.Errorf("quality %v: quant = %d, want %d", c.quality, e.baseQuant, c.want)
		}
	}
}
