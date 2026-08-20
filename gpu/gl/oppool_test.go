// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Unit tests for the op free lists (oppool.go). These need no GL context: they exercise the borrow/recycle protocol
// directly and assert its core safety invariant — recycle fully zeroes the op — so an incomplete reset (the "corrupts a
// later frame" failure mode) is caught cheaply and under the race detector. The cross-frame render-level guard lives in
// oppool_live_test.go.

package gl

import (
	"reflect"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// TestOpPoolRecycleZeroes verifies that recycle fully zeroes the op before pooling it. It inspects the op through the
// retained pointer immediately after recycle: recycle assigns the zero value to *o synchronously before the Put, so the
// object the pointer still refers to is fully cleared, and the check does not depend on which object a later borrow
// happens to return.
func TestOpPoolRecycleZeroes(t *testing.T) {
	var p opPool[fillRRectOp]
	o := p.borrow()
	// Dirty a representative spread: a scalar, an inline-array-backed slice element, and the OpBase identity
	// established by InitOp.
	o.instances = append(o.instancesArr[:0], fillRRectInstance{color: colorcore.PMColor4f{A: 1}})
	o.processorFlags = 0x1F
	o.baseInstance = 7
	o.InitOp(fillRRectOpClassID, o)
	if o.ClassID() == 0 || o.self == nil {
		t.Fatal("InitOp did not set up the op")
	}

	p.recycle(o)

	if o.processorFlags != 0 || o.baseInstance != 0 || o.instances != nil ||
		o.ClassID() != 0 || o.self != nil || o.uniqueID != 0 || o.instancesArr[0].color.A != 0 {
		t.Fatalf("recycle left stale state: flags=%#x base=%d instances=%v classID=%d self!=nil=%v "+
			"uniqueID=%d arr0.A=%v", o.processorFlags, o.baseInstance, o.instances, o.ClassID(),
			o.self != nil, o.uniqueID, o.instancesArr[0].color.A)
	}
}

// TestOpPoolBorrowAfterRecycleIsZeroed verifies the reuse path hands back a clean op: after a dirty op is recycled, the
// next borrow returns a zero-initialized op (whether the recycled shell or a fresh allocation), so op constructors that
// relied on `&T{}` behave identically on a reused shell.
func TestOpPoolBorrowAfterRecycleIsZeroed(t *testing.T) {
	var p opPool[circularRRectOp]
	o := p.borrow()
	o.rrects = append(o.rrectsArr[:0], circularRRectGeom{})
	o.InitOp(circularRRectOpClassID, o)
	p.recycle(o)

	o2 := p.borrow()
	if o2.self != nil || o2.ClassID() != 0 || len(o2.rrects) != 0 || o2.uniqueID != 0 {
		t.Fatalf("borrow after recycle returned a non-zeroed op: classID=%d self!=nil=%v rrects=%d",
			o2.ClassID(), o2.self != nil, len(o2.rrects))
	}
}

// typeHasReferences reports whether t contains any field whose kind can hold a heap reference (pointer, slice, map,
// chan, interface, func, string, or unsafe.Pointer), recursing through structs and arrays.
func typeHasReferences(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.UnsafePointer, reflect.Slice, reflect.Map, reflect.Chan,
		reflect.Interface, reflect.Func, reflect.String:
		return true
	case reflect.Array:
		return typeHasReferences(t.Elem())
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if typeHasReferences(t.Field(i).Type) {
				return true
			}
		}
	}
	return false
}

// TestInstanceTypesArePointerFree guards a precondition of recycleKeepingBacking: the per-instance element types the
// batchable geometry ops store in their inline-array-backed slices are pure POD. The capacity-preserving reset
// preserves a grown backing across recycle/borrow (reset to length 0 and clear()ed); if any of these element types
// gained a reference-holding field, a preserved backing could pin stale heap objects beyond the live length. clear()
// already defends against that, but keeping the types POD keeps the design's reasoning (and the inline-array
// bootstrap) simple, so this test fails fast if the assumption is broken.
func TestInstanceTypesArePointerFree(t *testing.T) {
	types := []reflect.Type{
		reflect.TypeOf(fillRRectInstance{}),
		reflect.TypeOf(aaStrokeRectInfo{}),
		reflect.TypeOf(circleGeom{}),
		reflect.TypeOf(ellipseGeom{}),
		reflect.TypeOf(diEllipseGeom{}),
		reflect.TypeOf(circularRRectGeom{}),
		reflect.TypeOf(ellipticalRRectGeom{}),
	}
	for _, typ := range types {
		if typeHasReferences(typ) {
			t.Errorf("%s must be pointer-free (a batchable op instance type); a reference field would "+
				"let recycleKeepingBacking pin stale heap objects in a preserved backing", typ)
		}
	}
}

// aaStrokeRectGetSet returns the get/set backing accessors an aaStrokeRectOp passes to recycleKeepingBacking, so the
// tests exercise the exact production path.
func aaStrokeRectGetSet() (get func(*aaStrokeRectOp) []aaStrokeRectInfo, set func(*aaStrokeRectOp, []aaStrokeRectInfo)) {
	return func(o *aaStrokeRectOp) []aaStrokeRectInfo { return o.rects },
		func(o *aaStrokeRectOp, s []aaStrokeRectInfo) { o.rects = s }
}

// TestRecycleKeepingBackingPreservesGrownBacking verifies the capacity-preserving reset: once an op's instance slice
// has grown past its inline [1] backing into the heap, recycle keeps that heap backing — same array, capacity
// preserved, length 0, cleared — while still zeroing every other field, and the next constructor bootstrap reuses it
// instead of re-growing. This recovers the merge-time OnCombineIfPossible grow that a plain full-zero recycle would
// drop every frame.
func TestRecycleKeepingBackingPreservesGrownBacking(t *testing.T) {
	var p opPool[aaStrokeRectOp]
	o := p.borrow()
	o.rects = bootstrapInstances(o.rects, o.rectsArr[:0])
	// Grow past the inline [1] backing so the slice moves to the heap (as OnCombineIfPossible does).
	o.rects = append(o.rects, aaStrokeRectInfo{color: colorcore.PMColor4f{A: 1}},
		aaStrokeRectInfo{degenerate: true}, aaStrokeRectInfo{devHalfStrokeSize: geom.Point{X: 1}})
	grownCap := cap(o.rects)
	if grownCap <= 1 {
		t.Fatalf("expected the append to grow past the inline backing; cap=%d", grownCap)
	}
	backing0 := &o.rects[:grownCap][0] // address of the heap backing array's first element
	// Dirty other fields to prove the reset zeroes them.
	o.wideColor = true
	o.miterStroke = true
	o.InitOp(aaStrokeRectOpClassID, o)
	if o.self == nil || o.ClassID() == 0 {
		t.Fatal("InitOp did not set up the op")
	}

	get, set := aaStrokeRectGetSet()
	p.recycleKeepingBacking(o, get, set)

	// Every field except the preserved backing must be zeroed (the pool's safety property).
	if o.wideColor || o.miterStroke || o.self != nil || o.ClassID() != 0 || o.uniqueID != 0 {
		t.Fatalf("recycleKeepingBacking left stale non-backing state: wideColor=%v miterStroke=%v "+
			"self!=nil=%v classID=%d uniqueID=%d", o.wideColor, o.miterStroke, o.self != nil,
			o.ClassID(), o.uniqueID)
	}
	// The grown backing must survive: same array, capacity preserved, length 0, and cleared.
	if o.rects == nil {
		t.Fatal("grown backing was dropped")
	}
	if len(o.rects) != 0 {
		t.Fatalf("preserved backing length not reset to 0: %d", len(o.rects))
	}
	if cap(o.rects) != grownCap {
		t.Fatalf("preserved backing capacity not kept: got %d want %d", cap(o.rects), grownCap)
	}
	if &o.rects[:grownCap][0] != backing0 {
		t.Fatal("backing array was reallocated, not preserved")
	}
	for i, v := range o.rects[:grownCap] {
		if v != (aaStrokeRectInfo{}) {
			t.Fatalf("preserved backing not cleared at index %d: %+v", i, v)
		}
	}
	// A subsequent constructor bootstrap must reuse the preserved backing, not the inline array.
	got := bootstrapInstances(o.rects, o.rectsArr[:0])
	if cap(got) != grownCap || &got[:grownCap][0] != backing0 {
		t.Fatalf("bootstrapInstances did not reuse the preserved backing: cap=%d", cap(got))
	}
}

// TestRecycleKeepingBackingDropsInlineBacking verifies the common single-instance case keeps the clean full zero: an op
// whose instance slice never grew past the inline [1] backing recycles to a nil field (no heap backing to preserve),
// and the next bootstrap falls back to the inline array.
func TestRecycleKeepingBackingDropsInlineBacking(t *testing.T) {
	var p opPool[aaStrokeRectOp]
	o := p.borrow()
	o.rects = bootstrapInstances(o.rects, o.rectsArr[:0])
	o.rects = append(o.rects, aaStrokeRectInfo{color: colorcore.PMColor4f{A: 1}}) // stays inline (cap 1)
	if cap(o.rects) != 1 {
		t.Fatalf("expected the single-instance slice to stay inline (cap 1), got %d", cap(o.rects))
	}
	o.InitOp(aaStrokeRectOpClassID, o)

	get, set := aaStrokeRectGetSet()
	p.recycleKeepingBacking(o, get, set)

	if o.rects != nil {
		t.Fatalf("inline-backed op should recycle to a nil field (clean full zero); got len=%d cap=%d",
			len(o.rects), cap(o.rects))
	}
	if o.rectsArr[0].color.A != 0 || o.self != nil || o.ClassID() != 0 {
		t.Fatal("recycleKeepingBacking left stale state on the inline-backed op")
	}
	got := bootstrapInstances(o.rects, o.rectsArr[:0])
	if cap(got) != 1 || &got[:1][0] != &o.rectsArr[0] {
		t.Fatal("bootstrapInstances did not fall back to the inline array")
	}
}
