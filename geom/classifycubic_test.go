// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "testing"

// The cubic corpora and their expected classifications exercise every branch of ClassifyCubic.

var classifySerpentines = [][4]Point{
	{{X: 149.325, Y: 107.705}, {X: 149.325, Y: 103.783}, {X: 151.638, Y: 100.127}, {X: 156.263, Y: 96.736}},
	{{X: 225.694, Y: 223.15}, {X: 209.831, Y: 224.837}, {X: 195.994, Y: 230.237}, {X: 184.181, Y: 239.35}},
	{{X: 4.873, Y: 5.581}, {X: 5.083, Y: 5.2783}, {X: 5.182, Y: 4.8593}, {X: 5.177, Y: 4.3242}},
	{{X: 285.625, Y: 499.687}, {X: 411.625, Y: 808.188}, {X: 1064.62, Y: 135.688}, {X: 1042.63, Y: 585.187}},
}

var classifyLoops = [][4]Point{
	{{X: 635.625, Y: 614.687}, {X: 171.625, Y: 236.188}, {X: 1064.62, Y: 135.688}, {X: 516.625, Y: 570.187}},
	{{X: 653.050, Y: 725.049}, {X: 663.000, Y: 176.000}, {X: 1189.000, Y: 508.000}, {X: 288.050, Y: 564.950}},
	{{X: 631.050, Y: 478.049}, {X: 730.000, Y: 302.000}, {X: 870.000, Y: 350.000}, {X: 905.050, Y: 528.950}},
	{{X: 631.050, Y: 478.0499}, {X: 221.000, Y: 230.000}, {X: 1265.000, Y: 451.000}, {X: 905.050, Y: 528.950}},
}

var classifyLinear = [][4]Point{
	{{X: 0, Y: 0}, {X: 0, Y: 1}, {X: 0, Y: 2}, {X: 0, Y: 3}},   // 0-degree flat line
	{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 0}},   // 180-degree flat line
	{{X: 0, Y: 1}, {X: 0, Y: 0}, {X: 0, Y: 2}, {X: 0, Y: 3}},   // 180-degree flat line
	{{X: 0, Y: 1}, {X: 0, Y: 0}, {X: 0, Y: 3}, {X: 0, Y: 2}},   // 360-degree flat line
	{{X: 0, Y: 0}, {X: 2, Y: 0}, {X: 1, Y: 0}, {X: 64, Y: 0}},  // 360-degree flat line
	{{X: 1, Y: 0}, {X: 0, Y: 0}, {X: 3, Y: 0}, {X: -64, Y: 0}}, // 360-degree flat line
}

var classifyCusps = [][4]Point{
	{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 1, Y: 0}, {X: 0, Y: 1}},
	{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 0, Y: 1}, {X: 1, Y: 0}},
	{{X: 0, Y: 1}, {X: 1, Y: 0}, {X: 0, Y: 0}, {X: 1, Y: 1}},
	{{X: 0, Y: 1}, {X: 1, Y: 0}, {X: 1, Y: 1}, {X: 0, Y: 0}},
}

func TestClassifyCubic(t *testing.T) {
	check := func(name string, corpus [][4]Point, want CubicType) {
		for i, pts := range corpus {
			if got, _, _ := ClassifyCubic(pts); got != want {
				t.Errorf("%s[%d]: ClassifyCubic = %d, want %d", name, i, got, want)
			}
		}
	}
	check("serpentine", classifySerpentines, CubicSerpentine)
	check("loop", classifyLoops, CubicLoop)
	check("linear", classifyLinear, CubicLineOrPoint)
	check("cusp", classifyCusps, CubicLocalCusp)

	if got, _, _ := ClassifyCubic([4]Point{{X: 0, Y: 0}, {X: 6, Y: 2}, {X: 10, Y: 2}, {X: 12, Y: 0}}); got != CubicQuadratic {
		t.Errorf("exact quad: ClassifyCubic = %d, want %d", got, CubicQuadratic)
	}
	if got, _, _ := ClassifyCubic([4]Point{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 1}, {X: 1, Y: 1}}); got != CubicCuspAtInfinity {
		t.Errorf("exact cusp-at-infinity: ClassifyCubic = %d, want %d", got, CubicCuspAtInfinity)
	}
}
