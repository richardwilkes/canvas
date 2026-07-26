// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the geometry-processor free lists (gppool.go). Like the op-pool and ProgramInfo/Pipeline pool tests
// these need no GL context: they drive the borrow/recycle protocol directly and assert its safety invariant —
// recycleGeomProc routes each pooled concrete type to its pool and fully zeroes it (so a reused shell carries no stale
// attributes or config), and borrow hands back clean shells. The render-level cross-frame guard (that a reused GP shell
// never leaks state into a later frame) is TestOpPoolCrossFrameStability in oppool_live_test.go, which exercises the
// default/circle/ellipse GPs automatically because its churn scene draws stroked rects, rounded rects, circles, and
// ovals through the batchable ops that recycle their programInfo (and hence GP).

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// TestRecycleGeomProcZeroesPooledTypes builds each of the six pooled GP types with a distinctive, fully-populated
// config (including the optional attributes) and verifies recycleGeomProc zeroes the concrete struct — the config
// fields, the class id, the inline attribute arrays, and the vertex/instance attribute sets and texture samplers. This
// is the safety-critical invariant: a shell sitting in the pool must pin nothing and carry no stale attributes into
// its next use.
func TestRecycleGeomProcZeroesPooledTypes(t *testing.T) {
	m := geom.Matrix{}
	m.SetScaleTranslate(2, 3, 4, 5)

	t.Run("default", func(t *testing.T) {
		gp := MakeDefaultGeoProc(ColorTypePremulWideAttribute,
			colorcore.PMColor4f{R: 0.25, A: 1}, CoverageTypeAttribute, 0xAB,
			LocalCoordsTypeHasExplicit, &m, &m).(*defaultGeoProc)
		if !gp.attrs[0].IsInitialized() || gp.vertexAttributes.count == 0 {
			t.Fatal("factory did not populate the GP before recycle")
		}
		recycleGeomProc(gp)
		if gp.flags != 0 || gp.coverage != 0 || gp.localCoordsWillBeRead ||
			gp.color != (colorcore.PMColor4f{}) || gp.ClassID() != 0 {
			t.Fatalf("defaultGeoProc config not zeroed: flags=%#x cov=%d color=%v class=%d",
				gp.flags, gp.coverage, gp.color, gp.ClassID())
		}
		assertGPBaseZeroed(t, &gp.GPBase, gp.attrs[0].IsInitialized())
	})

	t.Run("circle", func(t *testing.T) {
		gp := makeCircleGeometryProcessor(true, true, true, true, true, true, &m).(*circleGeometryProcessor)
		if !gp.stroke || gp.vertexAttributes.count == 0 {
			t.Fatal("factory did not populate the circle GP before recycle")
		}
		recycleGeomProc(gp)
		if gp.stroke || gp.ClassID() != 0 {
			t.Fatalf("circleGeometryProcessor not zeroed: stroke=%v class=%d", gp.stroke, gp.ClassID())
		}
		assertGPBaseZeroed(t, &gp.GPBase, gp.attrs[0].IsInitialized())
	})

	t.Run("ellipse", func(t *testing.T) {
		gp := makeEllipseGeometryProcessor(true, true, true, &m).(*ellipseGeometryProcessor)
		if !gp.stroke || !gp.useScale || gp.vertexAttributes.count == 0 {
			t.Fatal("factory did not populate the ellipse GP before recycle")
		}
		recycleGeomProc(gp)
		if gp.stroke || gp.useScale || gp.ClassID() != 0 {
			t.Fatalf("ellipseGeometryProcessor not zeroed: stroke=%v useScale=%v class=%d",
				gp.stroke, gp.useScale, gp.ClassID())
		}
		assertGPBaseZeroed(t, &gp.GPBase, gp.attrs[0].IsInitialized())
	})

	t.Run("diEllipse", func(t *testing.T) {
		gp := makeDIEllipseGeometryProcessor(true, true, &m, diEllipseStyleFill).(*diEllipseGeometryProcessor)
		if gp.style != diEllipseStyleFill || !gp.useScale || gp.vertexAttributes.count == 0 {
			t.Fatal("factory did not populate the di-ellipse GP before recycle")
		}
		recycleGeomProc(gp)
		if gp.style != 0 || gp.useScale || gp.ClassID() != 0 {
			t.Fatalf("diEllipseGeometryProcessor not zeroed: style=%d useScale=%v class=%d",
				gp.style, gp.useScale, gp.ClassID())
		}
		assertGPBaseZeroed(t, &gp.GPBase, gp.attrs[0].IsInitialized())
	})

	t.Run("quadPerEdgeAA", func(t *testing.T) {
		gp := MakeTexturedQuadProcessor(&texturedQuadSpec, gpu.TextureType2D,
			gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeNearest,
				gpu.MipmapModeNone), gpu.SwizzleRGBA, true).(*quadPerEdgeAAGeometryProcessor)
		if !gp.textured || !gp.saturate || gp.coverageMode == QuadCoverageModeNone ||
			gp.NumTextureSamplers() != 1 || gp.vertexAttributes.count == 0 {
			t.Fatal("factory did not populate the quad-per-edge GP before recycle")
		}
		for i := range gp.attrs {
			if !gp.attrs[i].IsInitialized() {
				t.Fatalf("the fully-optioned spec left attribute %d unset, so the recycle is not "+
					"being tested against a fully-populated GP", i)
			}
		}
		recycleGeomProc(gp)
		for i := range gp.attrs {
			if gp.attrs[i].IsInitialized() {
				t.Errorf("inline attribute %d not zeroed by recycle", i)
			}
		}
		if gp.textured || gp.saturate || gp.needsPerspective ||
			gp.coverageMode != QuadCoverageModeNone || gp.ClassID() != 0 {
			t.Fatalf("quadPerEdgeAAGeometryProcessor config not zeroed: textured=%v saturate=%v persp=%v cov=%d class=%d",
				gp.textured, gp.saturate, gp.needsPerspective, gp.coverageMode, gp.ClassID())
		}
		if gp.samplerStorage[0].IsInitialized() {
			t.Error("inline sampler storage not zeroed by recycle")
		}
		assertGPBaseZeroed(t, &gp.GPBase, gp.attrs[0].IsInitialized())
	})

	t.Run("fillRRect", func(t *testing.T) {
		gp := makeFillRRectProcessor(fillRRectFlagUseHWDerivatives | fillRRectFlagHasLocalCoords |
			fillRRectFlagWideColor).(*fillRRectProcessor)
		if gp.flags == 0 || gp.colorAttrib == nil || gp.vertexAttributes.count == 0 ||
			gp.instanceAttributes.count == 0 {
			t.Fatal("factory did not populate the fillRRect GP before recycle")
		}
		recycleGeomProc(gp)
		if gp.flags != 0 || gp.colorAttrib != nil || gp.ClassID() != 0 {
			t.Fatalf("fillRRectProcessor not zeroed: flags=%#x colorAttrib!=nil=%v class=%d",
				gp.flags, gp.colorAttrib != nil, gp.ClassID())
		}
		if gp.instanceAttrs[0].IsInitialized() {
			t.Error("inline instance attribute array not zeroed by recycle")
		}
		assertGPBaseZeroed(t, &gp.GPBase, gp.vertexAttrs[0].IsInitialized())
	})
}

// texturedQuadSpec is a fully-optioned vertex spec for the quad-per-edge GP: a general (non-rectilinear) coverage-AA
// device quad, which forces the geometry subset, plus float vertex colors, local coords, and a texture subset — so
// every optional attribute slot is populated.
var texturedQuadSpec = NewVertexSpec(QuadTypeGeneral, QuadColorTypeFloat, QuadTypeAxisAligned, true,
	QuadSubsetYes, gpu.AATypeCoverage, false, IndexBufferOptionPictureFramed)

// assertGPBaseZeroed checks the shared GPBase state a recycle must clear: the attribute sets (count/stride/backing),
// the texture samplers, and the leading inline attribute. attr0Init is the concrete GP's leading inline attribute's
// IsInitialized() (the arrays are not on GPBase, so the caller reads it).
func assertGPBaseZeroed(t *testing.T, b *GPBase, attr0Init bool) {
	t.Helper()
	if attr0Init {
		t.Error("inline attribute array not zeroed by recycle")
	}
	if b.vertexAttributes.count != 0 || b.vertexAttributes.stride != 0 ||
		b.vertexAttributes.attributes != nil {
		t.Errorf("vertexAttributes not reset: count=%d stride=%d attrs!=nil=%v",
			b.vertexAttributes.count, b.vertexAttributes.stride, b.vertexAttributes.attributes != nil)
	}
	if b.instanceAttributes.count != 0 || b.instanceAttributes.stride != 0 ||
		b.instanceAttributes.attributes != nil {
		t.Errorf("instanceAttributes not reset: count=%d stride=%d attrs!=nil=%v",
			b.instanceAttributes.count, b.instanceAttributes.stride, b.instanceAttributes.attributes != nil)
	}
	if b.textureSamplers != nil {
		t.Errorf("textureSamplers not reset: %d retained", len(b.textureSamplers))
	}
}

// TestPooledGPsOwnOnlyInlineStorage pins the precondition that justifies pooling every one of these types with a plain
// full-zero recycle (gppool.go's lifetime-safety bullets): every slice header and pointer a pooled GP holds must aim at
// that same GP's own inline storage, so `*o = zero` drops no external heap backing. If a factory ever hands one of
// these a heap-allocated attribute slice or an out-of-struct sampler, recycle would leak it and the reused shell would
// no longer be byte-identical to a fresh one — so this fails loudly rather than silently regressing the audit.
func TestPooledGPsOwnOnlyInlineStorage(t *testing.T) {
	m := geom.Matrix{}
	m.SetIdentity()

	t.Run("inline attrs arrays", func(t *testing.T) {
		def := MakeDefaultGeoProc(ColorTypePremulWideAttribute, colorcore.PMColor4f{A: 1},
			CoverageTypeAttribute, 0xAB, LocalCoordsTypeHasExplicit, &m, &m).(*defaultGeoProc)
		assertBackedByInlineArray(t, "defaultGeoProc.vertexAttributes",
			def.vertexAttributes.attributes, def.attrs[:])
		circle := makeCircleGeometryProcessor(true, true, true, true, true, true,
			&m).(*circleGeometryProcessor)
		assertBackedByInlineArray(t, "circleGeometryProcessor.vertexAttributes",
			circle.vertexAttributes.attributes, circle.attrs[:])
		ellipse := makeEllipseGeometryProcessor(true, true, true, &m).(*ellipseGeometryProcessor)
		assertBackedByInlineArray(t, "ellipseGeometryProcessor.vertexAttributes",
			ellipse.vertexAttributes.attributes, ellipse.attrs[:])
		di := makeDIEllipseGeometryProcessor(true, true, &m,
			diEllipseStyleFill).(*diEllipseGeometryProcessor)
		assertBackedByInlineArray(t, "diEllipseGeometryProcessor.vertexAttributes",
			di.vertexAttributes.attributes, di.attrs[:])
	})

	t.Run("quad-per-edge sampler", func(t *testing.T) {
		gp := MakeTexturedQuadProcessor(&texturedQuadSpec, gpu.TextureType2D,
			gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeNearest,
				gpu.MipmapModeNone), gpu.SwizzleRGBA, false).(*quadPerEdgeAAGeometryProcessor)
		assertBackedByInlineArray(t, "quadPerEdgeAAGeometryProcessor.vertexAttributes",
			gp.vertexAttributes.attributes, gp.attrs[:])
		if len(gp.textureSamplers) != 1 || &gp.textureSamplers[0] != &gp.samplerStorage[0] {
			t.Error("textured quad GP's samplers are not its own inline samplerStorage")
		}
	})

	t.Run("fillRRect color attribute", func(t *testing.T) {
		gp := makeFillRRectProcessor(fillRRectFlagHasLocalCoords |
			fillRRectFlagWideColor).(*fillRRectProcessor)
		assertBackedByInlineArray(t, "fillRRectProcessor.vertexAttributes",
			gp.vertexAttributes.attributes, gp.vertexAttrs[:])
		assertBackedByInlineArray(t, "fillRRectProcessor.instanceAttributes",
			gp.instanceAttributes.attributes, gp.instanceAttrs[:])
		if gp.colorAttrib == nil {
			t.Fatal("fillRRect GP has no color attribute")
		}
		if !pointsIntoAttrs(gp.colorAttrib, gp.instanceAttrs[:]) {
			t.Error("fillRRect GP's colorAttrib does not point into its own instanceAttrs array")
		}
	})
}

// assertBackedByInlineArray verifies that set is backed by inline (the GP's own array) rather than by a separately
// allocated slice.
func assertBackedByInlineArray(t *testing.T, name string, set, inline []Attribute) {
	t.Helper()
	if len(set) == 0 {
		t.Errorf("%s is empty", name)
		return
	}
	if &set[0] != &inline[0] {
		t.Errorf("%s is not backed by the GP's own inline attribute array", name)
	}
}

// pointsIntoAttrs reports whether a is the address of one of attrs' elements.
func pointsIntoAttrs(a *Attribute, attrs []Attribute) bool {
	for i := range attrs {
		if a == &attrs[i] {
			return true
		}
	}
	return false
}

// TestBorrowGeomProcReusesCleanShell primes the circle-GP pool with a round-cap circle (which sets the optional
// clip-plane and round-cap attributes), recycles it, then builds a plain circle through the factory and confirms the
// result carries none of the previous shell's optional attributes — i.e. a reused shell is byte-identical to a fresh
// one. This is the reuse correctness point behind the whole pool: the factory's `*gp = T{}` overwrite plus the attrs
// re-population must leave no stale optional attribute from a prior, differently-configured use.
func TestBorrowGeomProcReusesCleanShell(t *testing.T) {
	m := geom.Matrix{}
	m.SetIdentity()
	// Prime and recycle a fully-optioned circle GP so a recycled shell is waiting in the pool.
	recycleGeomProc(makeCircleGeometryProcessor(true, true, true, true, true, false, &m))

	// A plain fill circle: only inPosition, inColor, inCircleEdge (attrs 0..2) are set.
	gp := makeCircleGeometryProcessor(false, false, false, false, false, false, &m).(*circleGeometryProcessor)
	if !gp.attrs[0].IsInitialized() || !gp.attrs[1].IsInitialized() || !gp.attrs[2].IsInitialized() {
		t.Fatal("plain circle GP is missing its required attributes")
	}
	for i := 3; i < len(gp.attrs); i++ {
		if gp.attrs[i].IsInitialized() {
			t.Fatalf("plain circle GP carries a stale optional attribute at index %d", i)
		}
	}
	if gp.stroke {
		t.Fatal("plain circle GP kept a stale stroke flag from the recycled shell")
	}
}

// TestBorrowQuadPerEdgeAAReusesCleanShell primes the quad-per-edge pool with a fully-optioned *textured* GP (sampler,
// texture subset, geometry subset, vertex colors, local coords, saturate) and then builds the minimal plain GP through
// the non-textured factory, which sets neither the textured/saturate flags nor the samplers and populates only the
// position attribute. Everything else must come back zero — this is the reuse half of the full-zero-safety argument for
// the newest pooled type: nothing the factory does not rewrite may survive from the prior use.
func TestBorrowQuadPerEdgeAAReusesCleanShell(t *testing.T) {
	recycleGeomProc(MakeTexturedQuadProcessor(&texturedQuadSpec, gpu.TextureType2D,
		gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeNearest,
			gpu.MipmapModeNone), gpu.SwizzleRGBA, true))

	plain := NewVertexSpec(QuadTypeAxisAligned, QuadColorTypeNone, QuadTypeAxisAligned, false,
		QuadSubsetNo, gpu.AATypeNone, true, IndexBufferOptionTriStrips)
	gp := MakeQuadPerEdgeAAProcessor(&plain).(*quadPerEdgeAAGeometryProcessor)
	if !gp.attrs[qpaaAttrPosition].IsInitialized() {
		t.Fatal("plain quad GP is missing its position attribute")
	}
	for i := range gp.attrs {
		if i != qpaaAttrPosition && gp.attrs[i].IsInitialized() {
			t.Fatalf("plain quad GP carries a stale optional attribute at index %d", i)
		}
	}
	if gp.textured || gp.saturate || gp.needsPerspective || gp.coverageMode != QuadCoverageModeNone {
		t.Fatalf("plain quad GP kept stale config: textured=%v saturate=%v persp=%v cov=%d",
			gp.textured, gp.saturate, gp.needsPerspective, gp.coverageMode)
	}
	if gp.NumTextureSamplers() != 0 || gp.samplerStorage[0].IsInitialized() {
		t.Fatalf("plain quad GP kept a stale sampler from the recycled shell (%d samplers)",
			gp.NumTextureSamplers())
	}
}

// TestBorrowFillRRectReusesCleanShell primes the fillRRect pool with a local-coords GP (which uses six instance
// attributes) and then builds one without local coords (which uses five), confirming the sixth slot and the colorAttrib
// pointer both come from the new build rather than the recycled shell. A stale colorAttrib would aim the shader's
// pass-through at the wrong attribute.
func TestBorrowFillRRectReusesCleanShell(t *testing.T) {
	recycleGeomProc(makeFillRRectProcessor(fillRRectFlagHasLocalCoords | fillRRectFlagWideColor))

	gp := makeFillRRectProcessor(0).(*fillRRectProcessor)
	if gp.flags != 0 {
		t.Fatalf("plain fillRRect GP kept stale flags %#x", gp.flags)
	}
	// radii_x, radii_y, skew, translate_and_localrotate, color — the localrect slot stays unused.
	const usedInstanceAttrs = 5
	if gp.instanceAttributes.count != usedInstanceAttrs {
		t.Fatalf("plain fillRRect GP has %d instance attributes, want %d",
			gp.instanceAttributes.count, usedInstanceAttrs)
	}
	if gp.instanceAttrs[usedInstanceAttrs].IsInitialized() {
		t.Fatal("plain fillRRect GP carries a stale instance attribute from the recycled shell")
	}
	if gp.colorAttrib != &gp.instanceAttrs[usedInstanceAttrs-1] {
		t.Fatal("plain fillRRect GP's colorAttrib does not point at its own color attribute slot")
	}
}

// TestRecycleGeomProcNilSafe verifies recycleGeomProc tolerates a nil GeometryProcessor. A non-nil ProgramInfo never
// carries a nil GP, but recycleProgramInfo calls recycleGeomProc unconditionally, so nil-tolerance keeps the contract
// cheap and total. A nil interface matches no type-switch case.
func TestRecycleGeomProcNilSafe(_ *testing.T) {
	recycleGeomProc(nil) // must not panic
}
