// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gorender

import (
	gocanvas "github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/raster"
)

// RenderScenarioRaster renders one scenario through the port's CPU raster backend into RGBA8888-premul pixels (tightly
// packed, top-left origin) — directly comparable with the self-captured raster goldens in ../goldens/raster, which
// gate bit-exactly per platform. It needs no GL and is the raster gate's render path, plus a backend-independent check
// of the sceneCanvas adapter itself (so a Go-GPU mismatch can be localized to the GPU path rather than the replay).
// `oracle gen` and `oracle bless -lane raster` render through it.
func RenderScenarioRaster(sc scenario.Scenario) []byte {
	pm := raster.NewPixmap(int32(sc.Width), int32(sc.Height))
	c := gocanvas.NewForPixmap(pm)
	sc.Draw(sceneCanvas{c: c})
	return pm.RGBA8888Bytes()
}
