// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The drop-shadow image filters: built as a composition — blur, color-filter to the shadow color, offset, and (when not
// shadow-only) a merge with the original. Merge is visually equivalent to a src-over blend but draws each child
// independently, avoiding shader-based tile-edge evaluation over the union bounds.

package imagefilter

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// makeDropShadowGraph builds the blur/color-filter/offset/merge filter graph shared by DropShadow and DropShadowOnly.
func makeDropShadowGraph(dx, dy, sigmaX, sigmaY float32, color colorcore.Color, shadowOnly bool, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	filter := input
	filter = Blur(sigmaX, sigmaY, shaders.TileDecal, filter, nil)
	filter = ColorFilter(colorfilter.NewBlend(color, raster.BlendSrcIn), filter, nil)
	// Offset would take sampling options too, but linear filtering is needed to hide nearest-neighbor artifacts from
	// fractional offsets applied post-blur.
	var t geom.Matrix
	t.SetTranslate(dx, dy)
	filter = MatrixTransform(&t, shaders.SamplingOptions{Filter: shaders.FilterLinear}, filter)
	if !shadowOnly {
		filter = Merge([]filtercore.Filter{filter, input}, nil)
	}
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// DropShadow builds a filter drawing a blurred, colored, offset shadow behind the input.
func DropShadow(dx, dy, sigmaX, sigmaY float32, color colorcore.Color, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeDropShadowGraph(dx, dy, sigmaX, sigmaY, color, false, input, cropRect)
}

// DropShadowOnly builds a filter drawing only the blurred, colored, offset shadow, without the input.
func DropShadowOnly(dx, dy, sigmaX, sigmaY float32, color colorcore.Color, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeDropShadowGraph(dx, dy, sigmaX, sigmaY, color, true, input, cropRect)
}
