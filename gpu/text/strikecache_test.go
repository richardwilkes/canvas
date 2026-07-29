// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package text

import (
	"testing"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// unhashablePathEffect is a value-receiver stroke.PathEffect over unhashable state (a slice field), the shape a caller
// outside the module can reach by accident. It never filters, so it changes no geometry.
type unhashablePathEffect struct{ intervals []float32 }

func (unhashablePathEffect) FilterPath(_, _ *path.Path, _ *stroke.Rec, _ *geom.Rect, _ *geom.Matrix) bool {
	return false
}

func (unhashablePathEffect) AsPoints(_ *stroke.PointData, _ *path.Path, _ *stroke.Rec, _ *geom.Matrix,
	_ *geom.Rect,
) bool {
	return false
}

func (unhashablePathEffect) ComputeFastBounds(_ *geom.Rect) bool { return false }

func TestGPUStrikeCacheUnhashableEffects(t *testing.T) {
	// This cache keys on the whole font.StrikeSpec, so it has the CPU cache's exposure to an effect Go refuses to hash:
	// the map key would panic on the first text draw that used one. Such a spec must skip the cache instead.
	f := loadTestFont(t, "Roboto-Regular.ttf", 24)
	identity := geom.IdentityMatrix()
	spec := font.MakeMaskSpec(f, &font.ScalerPaint{PathEffect: unhashablePathEffect{intervals: []float32{2, 2}}},
		&identity, nil)
	if spec.Keyable() {
		t.Fatal("the spec must not report itself keyable")
	}

	cache := NewStrikeCache()
	strike := cache.FindOrCreateStrike(&spec)
	if strike == nil {
		t.Fatal("no strike")
	}
	// The strike still hands out glyph records, they just do not accumulate anywhere the cache accounts for or purges.
	if g := strike.GetGlyph(font.PackGlyphID(f.UnicharToGlyph('A'))); g == nil {
		t.Fatal("no glyph record")
	}
	if got := cache.StrikeCount(); got != 0 {
		t.Errorf("cached strike count %d, want 0", got)
	}
	if got := cache.TotalMemoryUsed(); got != 0 {
		t.Errorf("cached bytes %d, want 0", got)
	}
	if again := cache.FindOrCreateStrike(&spec); again == strike {
		t.Error("an uncached spec must not resolve to the same strike twice")
	}

	// A spec with no effects still keys, and keying is still what the cache does for it.
	plain := font.MakeMaskSpec(f, nil, &identity, nil)
	if !plain.Keyable() {
		t.Fatal("a spec with no effects must be keyable")
	}
	first := cache.FindOrCreateStrike(&plain)
	if second := cache.FindOrCreateStrike(&plain); second != first {
		t.Error("a keyable spec must resolve to the same strike twice")
	}
	if got := cache.StrikeCount(); got != 1 {
		t.Errorf("cached strike count %d, want 1", got)
	}
}
