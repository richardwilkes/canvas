// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// noSamplerObjectAnisoDriver models a GL 3.2 core context that predates sampler objects (< 3.3, no
// GL_ARB_sampler_objects) yet advertises anisotropic filtering. It exercises BindTexture's texture-parameter path
// (g.samplerObjectCache stays nil), the only path that touches glTexParameterf(TEXTURE_MAX_ANISOTROPY).
func noSamplerObjectAnisoDriver() *capsFakeDriver {
	return &capsFakeDriver{
		version:     "3.2 Metal - 90.5",
		vendor:      "Apple",
		renderer:    "Apple M4 Max",
		glslVersion: "1.50",
		extensions: []string{
			"GL_ARB_ES2_compatibility", "GL_ARB_texture_storage", "GL_ARB_texture_swizzle",
			"GL_EXT_texture_filter_anisotropic",
		},
		integers: map[uint32]int32{
			MAX_FRAGMENT_UNIFORM_COMPONENTS: 4096,
			CONTEXT_PROFILE_MASK:            CONTEXT_CORE_PROFILE_BIT,
			MAX_VERTEX_ATTRIBS:              16,
			MAX_TEXTURE_IMAGE_UNITS:         16,
			MAX_TEXTURE_SIZE:                16384,
			MAX_RENDERBUFFER_SIZE:           16384,
			MAX_SAMPLES:                     4,
		},
		sampleCounts:  []int32{4, 2},
		maxAniso:      16,
		precisionBits: 23,
	}
}

// TestGpuBindTextureAnisoUnit guards the fix for the anisotropy branch of BindTexture's non-sampler-object path, which
// previously emitted glTexParameterf(TEXTURE_MAX_ANISOTROPY) without first calling setTextureUnit(unitIdx) as every
// sibling filter/wrap/LOD/swizzle branch does. When only MaxAniso changes, the earlier branches are skipped, so the
// anisotropy value would land on whatever texture was bound to the currently-active unit instead of unitIdx.
func TestGpuBindTextureAnisoUnit(t *testing.T) {
	g := newRecordingGpuWithDriver(t, noSamplerObjectAnisoDriver())
	if g.glCaps().UseSamplerObjects() {
		t.Fatal("expected the pre-3.3 profile to use texture parameters, not sampler objects")
	}
	if g.samplerObjectCache != nil {
		t.Fatal("sampler object cache should be nil without sampler objects")
	}
	if !g.Caps().AnisoSupport {
		t.Fatal("expected anisotropy support (GL_EXT_texture_filter_anisotropic)")
	}

	dims := geom.ISize{Width: 16, Height: 16}
	surf0 := g.CreateTexture(dims, FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "tex0")
	surf1 := g.CreateTexture(dims, FormatRGBA8, gpu.TextureType2D, gpu.RenderableNo, 1,
		gpu.BudgetedYes, 1, gpu.ColorTypeRGBA8888, gpu.ColorTypeRGBA8888, nil, "tex1")
	if surf0 == nil || surf1 == nil {
		t.Fatal("CreateTexture failed")
	}
	tex0 := surf0.AsTexture()
	tex1 := surf1.AsTexture()

	// Record the active texture unit at the moment each TEXTURE_MAX_ANISOTROPY parameter is applied. setTextureUnit
	// updates hwActiveTextureUnitIdx before glActiveTexture returns, so reading it here reflects the unit the GL call
	// actually targets.
	var anisoUnits []int
	g.ctx.Interface.Functions.fnTexParameterf = func(_, pname uint32, _ float32) {
		if pname == TEXTURE_MAX_ANISOTROPY {
			anisoUnits = append(anisoUnits, g.hwActiveTextureUnitIdx)
		}
	}

	// Fully bind tex0 on unit 0 (records MaxAniso=2), then bind tex1 on unit 1 so the active unit is left at 1.
	stateAniso2 := gpu.SamplerStateAniso(gpu.WrapModeClamp, gpu.WrapModeClamp, 2, gpu.MipmappedNo)
	g.BindTexture(0, stateAniso2, tex0)
	g.BindTexture(1, stateAniso2, tex1)
	if g.hwActiveTextureUnitIdx != 1 {
		t.Fatalf("active unit = %d, want 1 after binding tex1 on unit 1", g.hwActiveTextureUnitIdx)
	}

	// Re-bind tex0 on unit 0 changing only MaxAniso. tex0 is already tracked on unit 0, so the texture is not rebound
	// and every sibling branch (filter/wrap/LOD/border/swizzle) is a no-op; only the anisotropy branch fires. The
	// anisotropy parameter must be routed to unit 0, not the stale active unit 1.
	anisoUnits = nil
	stateAniso8 := gpu.SamplerStateAniso(gpu.WrapModeClamp, gpu.WrapModeClamp, 8, gpu.MipmappedNo)
	g.BindTexture(0, stateAniso8, tex0)

	if len(anisoUnits) != 1 {
		t.Fatalf("anisotropy parameter applied %d times, want exactly 1", len(anisoUnits))
	}
	if anisoUnits[0] != 0 {
		t.Errorf("anisotropy applied while unit %d was active, want unit 0 (bug: applied to the wrong texture's "+
			"sampler state)", anisoUnits[0])
	}

	surf0.Unref()
	surf1.Unref()
	g.ResourceCache().PurgeUnlockedResources(gpu.PurgeAllResources)
}
