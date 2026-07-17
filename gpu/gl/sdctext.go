// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GPU text funnel: the text-blob redraw coordinator converts glyph-run lists to subrun containers (cached per
// blob), and the atlas delegate turns each atlas subrun into an AtlasTextOp.

package gl

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/text"
	"github.com/richardwilkes/canvas/textblob"
)

// DrawGlyphRunList converts a run of shaped glyphs into subrun containers and adds the resulting draw ops (typically
// one AtlasTextOp) to the surface's op list.
func (sdc *SurfaceDrawContext) DrawGlyphRunList(cv *canvas.Canvas, clip Clip, viewMatrix *geom.Matrix, glyphRunList *textblob.GlyphRunList, paint *canvas.Paint) {
	if sdc.ctx.Abandoned() {
		return
	}

	textBlobCache := sdc.ctx.TextBlobCache()

	atlasDelegate := func(subRun text.AtlasSubRun, drawOrigin geom.Point,
		runPaint *canvas.Paint,
	) {
		drawingClip, op := MakeAtlasTextOp(sdc, subRun, clip, viewMatrix, drawOrigin, runPaint)
		if op != nil {
			sdc.AddDrawOp(drawingClip, op)
		}
	}

	deviceProps := sdc.SurfaceProps().DeviceProps()
	if sdc.subRunControl == nil {
		// Built lazily from the recording context and the device-independent-fonts flag on first use, and cached
		// thereafter since the surface draw context's context and props are immutable after creation.
		control := sdc.ctx.GetSubRunControl(deviceProps.UseDeviceIndependentFonts)
		sdc.subRunControl = &control
	}
	textBlobCache.DrawGlyphRunList(cv, viewMatrix, glyphRunList, paint, &deviceProps,
		sdc.subRunControl, atlasDelegate)
}
