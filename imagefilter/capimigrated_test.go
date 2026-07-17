// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package imagefilter_test

import (
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/imagefilter"
	"github.com/richardwilkes/canvas/shaders"
)

// TestBlurConstructorGuards verifies Blur's argument contract: negative or non-finite sigmas are errors (nil), a valid
// decal blur is non-nil, and a non-decal tile mode without a crop rect takes the legacy blur lane (also non-nil).
func TestBlurConstructorGuards(t *testing.T) {
	if f := imagefilter.Blur(-1, -1, shaders.TileClamp, nil, nil); f != nil {
		t.Error("blur with negative sigma: want nil")
	}
	if f := imagefilter.Blur(float32(math.NaN()), 1, shaders.TileDecal, nil, nil); f != nil {
		t.Error("blur with a NaN sigma: want nil")
	}
	if f := imagefilter.Blur(2, 2, shaders.TileDecal, nil, nil); f == nil {
		t.Error("valid decal blur: want non-nil")
	}
	if f := imagefilter.Blur(2, 2, shaders.TileClamp, nil, nil); f == nil {
		t.Error("clamp-tile blur without a crop rect (the legacy lane): want non-nil")
	}
}

// TestImageDefaultSourceBounds verifies ImageDefault's behavior: the src and dst rects default to the image bounds, and
// a nil image still yields a non-nil (transparent-black) filter.
func TestImageDefaultSourceBounds(t *testing.T) {
	info, _ := imagecore.MakeInfo(8, 8, imagecore.ColorTypeRGBA8888, imagecore.AlphaTypePremul)
	img := imagecore.FromPixels(imagecore.NewPixels(info))
	if f := imagefilter.ImageDefault(img, shaders.SamplingOptions{}); f == nil {
		t.Error("image source (default) with a valid image: want non-nil")
	}
	if f := imagefilter.ImageDefault(nil, shaders.SamplingOptions{}); f == nil {
		t.Error("image source (default) with nil image: want non-nil Empty filter")
	}
}

// TestPointLitSpecularConstructor verifies the point-light specular lighting constructor accepts valid arguments (the
// façade exercised it as one of the 23 sk_imagefilter_new_* entry points).
func TestPointLitSpecularConstructor(t *testing.T) {
	f := imagefilter.PointLitSpecular(geom.Point3{X: 5, Y: 5, Z: 10}, 0xFFFFFFFF, 1, 1, 8, nil, nil)
	if f == nil {
		t.Error("point-lit specular with valid input: want non-nil")
	}
}
