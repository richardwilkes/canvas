// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
)

func TestGenerationIDEmptyAndStability(t *testing.T) {
	var p Path
	if got := p.GenerationID(); got != 1 {
		t.Fatalf("empty path generation ID = %d, want 1 (kEmptyGenID)", got)
	}
	p.MoveTo(1, 2)
	id := p.GenerationID()
	if id < 2 {
		t.Fatalf("non-empty path generation ID = %d, want >= 2", id)
	}
	if again := p.GenerationID(); again != id {
		t.Fatalf("generation ID changed without mutation: %d then %d", id, again)
	}
}

func TestGenerationIDChangesOnMutation(t *testing.T) {
	p := New().MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10)
	id := p.GenerationID()
	p.LineTo(0, 10)
	if next := p.GenerationID(); next == id {
		t.Fatal("generation ID unchanged after LineTo")
	}
	id = p.GenerationID()
	p.SetLastPt(3, 4)
	if next := p.GenerationID(); next == id {
		t.Fatal("generation ID unchanged after SetLastPt")
	}
	id = p.GenerationID()
	m := geom.TranslateMatrix(1, 2)
	p.Transform(&m)
	if next := p.GenerationID(); next == id {
		t.Fatal("generation ID unchanged after in-place Transform")
	}
	p.Reset()
	if got := p.GenerationID(); got != 1 {
		t.Fatalf("reset path generation ID = %d, want 1", got)
	}
}

func TestGenerationIDIgnoresFillType(t *testing.T) {
	p := New().MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).Close()
	id := p.GenerationID()
	p.SetFillType(FillEvenOdd)
	if got := p.GenerationID(); got != id {
		t.Fatalf("SetFillType changed the generation ID: %d -> %d", id, got)
	}
	p.ToggleInverseFillType()
	if got := p.GenerationID(); got != id {
		t.Fatalf("ToggleInverseFillType changed the generation ID: %d -> %d", id, got)
	}
}

func TestGenerationIDCloneAndTransformTo(t *testing.T) {
	p := New().MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).Close()
	id := p.GenerationID()

	// A clone is the same geometry, so it keeps the ID.
	c := p.Clone()
	if got := c.GenerationID(); got != id {
		t.Fatalf("clone generation ID = %d, want %d", got, id)
	}
	// Mutating the clone must not affect the original.
	c.LineTo(0, 10)
	if got := c.GenerationID(); got == id {
		t.Fatal("mutated clone kept the original generation ID")
	}
	if got := p.GenerationID(); got != id {
		t.Fatalf("original generation ID changed: %d -> %d", got, id)
	}

	// TransformTo writes fresh geometry into dst.
	dst := &Path{}
	m := geom.TranslateMatrix(5, 5)
	p.TransformTo(&m, dst)
	if got := dst.GenerationID(); got == id {
		t.Fatal("transformed copy kept the source generation ID")
	}

	// Distinct identical paths get distinct IDs (identity, not content).
	q := New().MoveTo(0, 0).LineTo(10, 0).LineTo(10, 10).Close()
	if got := q.GenerationID(); got == id {
		t.Fatal("independently built path shares a generation ID")
	}
}

func TestSetVolatile(t *testing.T) {
	p := New().MoveTo(0, 0).LineTo(4, 4)
	if p.IsVolatile() {
		t.Fatal("paths default to non-volatile")
	}
	p.SetVolatile(true)
	if !p.IsVolatile() {
		t.Fatal("SetVolatile(true) not recorded")
	}
	if !p.Clone().IsVolatile() {
		t.Fatal("clone dropped the volatile flag")
	}
	dst := &Path{}
	m := geom.TranslateMatrix(1, 1)
	p.TransformTo(&m, dst)
	if !dst.IsVolatile() {
		t.Fatal("transform dropped the volatile flag")
	}
}
