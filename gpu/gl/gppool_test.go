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
)

// TestRecycleGeomProcZeroesPooledTypes builds each of the four pooled GP types with a distinctive, fully-populated
// config (including the optional attributes) and verifies recycleGeomProc zeroes the concrete struct — the config
// fields, the class id, the inline attribute array, and the vertex attribute set. This is the safety-critical
// invariant: a shell sitting in the pool must pin nothing and carry no stale attributes into its next use.
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
}

// assertGPBaseZeroed checks the shared GPBase state a recycle must clear: the vertex attribute set
// (count/stride/backing) and the leading inline attribute. attr0Init is the concrete GP's attrs[0].IsInitialized() (the
// array is not on GPBase, so the caller reads it).
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

// TestRecycleGeomProcNilSafe verifies recycleGeomProc tolerates a nil GeometryProcessor. A non-nil ProgramInfo never
// carries a nil GP, but recycleProgramInfo calls recycleGeomProc unconditionally, so nil-tolerance keeps the contract
// cheap and total. A nil interface matches no type-switch case.
func TestRecycleGeomProcNilSafe(_ *testing.T) {
	recycleGeomProc(nil) // must not panic
}
