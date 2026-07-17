// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package pdf

import (
	"reflect"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/stream"
)

func TestGlyphUseSetHasForEach(t *testing.T) {
	g := newGlyphUse(1, 10)
	for _, gid := range []uint16{0, 3, 7, 10} {
		g.set(gid)
	}
	for _, gid := range []uint16{0, 3, 7, 10} {
		if !g.has(gid) {
			t.Errorf("expected %d set", gid)
		}
	}
	for _, gid := range []uint16{1, 2, 4, 5, 6, 8, 9} {
		if g.has(gid) {
			t.Errorf("expected %d clear", gid)
		}
	}
	var got []uint16
	g.forEach(func(gid uint16) { got = append(got, gid) })
	if want := []uint16{0, 3, 7, 10}; !reflect.DeepEqual(got, want) {
		t.Errorf("forEach = %v, want %v", got, want)
	}
}

func TestGlyphUseFirstNonZeroOffset(t *testing.T) {
	// A subset whose firstNonZero is not 1 exercises the toCode offset lane.
	g := newGlyphUse(5, 9)
	for _, gid := range []uint16{0, 5, 8} {
		g.set(gid)
	}
	var got []uint16
	g.forEach(func(gid uint16) { got = append(got, gid) })
	if want := []uint16{0, 5, 8}; !reflect.DeepEqual(got, want) {
		t.Errorf("forEach = %v, want %v", got, want)
	}
}

func TestNextFontSubsetTag(t *testing.T) {
	var buf stream.MemoryWStream
	d := NewDocument(&buf, DefaultMetadata())
	for i, want := range []string{"AAAAAA+", "BAAAAA+", "CAAAAA+"} {
		if got := d.nextFontSubsetTag(); got != want {
			t.Errorf("tag %d = %q, want %q", i, got, want)
		}
	}
}

func TestFindModeOr0(t *testing.T) {
	cases := []struct {
		in   []float32
		want float32
	}{
		{in: nil, want: 0},
		{in: []float32{5}, want: 5},
		{in: []float32{1, 1, 2, 3, 3, 3}, want: 3},
		{in: []float32{1, 2, 3}, want: 1},
		{in: []float32{2, 2, 2, 5, 5}, want: 2},
	}
	for _, c := range cases {
		if got := findModeOr0(c.in); got != c.want {
			t.Errorf("findModeOr0(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestAppendCmapSections(t *testing.T) {
	// gids 3,4,5 map to consecutive code points (a bfrange); gid 8 is isolated (a bfchar).
	glyphToUnicode := []int32{0, 0, 0, 0x41, 0x42, 0x43, 0, 0, 0x58}
	sub := newGlyphUse(1, 8)
	for _, gid := range []uint16{3, 4, 5, 8} {
		sub.set(gid)
	}
	cmap := stream.NewMemoryWStream()
	appendCmapSections(glyphToUnicode, &sub, cmap, true, 1, 8)
	out := string(cmap.Bytes())

	// bfchar entries must precede bfrange entries.
	ci := strings.Index(out, "beginbfchar")
	ri := strings.Index(out, "beginbfrange")
	if ci < 0 || ri < 0 || ci > ri {
		t.Fatalf("expected bfchar before bfrange; got:\n%s", out)
	}
	mustContain(t, out, "<0008> <0058>")        // the isolated glyph
	mustContain(t, out, "<0003> <0005> <0041>") // the consecutive range
}

func TestMakeToUnicodeCmapWrapsSections(t *testing.T) {
	glyphToUnicode := []int32{0, 0x30, 0x31}
	sub := newGlyphUse(1, 2)
	sub.set(1)
	sub.set(2)
	out := string(makeToUnicodeCmap(glyphToUnicode, &sub, true, 1, 2))
	mustContain(t, out, "/CIDInit /ProcSet findresource begin")
	mustContain(t, out, "/CMapType 2 def")
	mustContain(t, out, "<0000> <FFFF>") // multibyte codespace range
	mustContain(t, out, "endcmap")
	// gids 1,2 → 0x30,0x31 consecutive → a bfrange.
	mustContain(t, out, "<0001> <0002> <0030>")
}
