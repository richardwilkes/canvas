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
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/stream"
)

func emitUnion(u union) string {
	s := stream.NewMemoryWStream()
	u.emit(s)
	return string(s.Bytes())
}

func TestUnionEmit(t *testing.T) {
	cases := []struct {
		name string
		want string
		u    union
	}{
		{name: "int", u: unionInt(42), want: "42"},
		{name: "int-neg", u: unionInt(-7), want: "-7"},
		{name: "bool-true", u: unionBool(true), want: "true"},
		{name: "bool-false", u: unionBool(false), want: "false"},
		{name: "scalar", u: unionScalar(1.5), want: "1.5"},
		{name: "scalar-zero", u: unionScalar(0), want: "0"},
		{name: "colorcomponent-0", u: unionColorComponent(0), want: "0"},
		{name: "colorcomponent-255", u: unionColorComponent(255), want: "1"},
		{name: "colorcomponent-128", u: unionColorComponent(128), want: ".502"},
		{name: "colorcomponentF-half", u: unionColorComponentF(0.5), want: ".5"},
		{name: "colorcomponentF-one", u: unionColorComponentF(1.0), want: "1"},
		{name: "colorcomponentF-zero", u: unionColorComponentF(0.0), want: "0"},
		{name: "name", u: unionName("Type"), want: "/Type"},
		{name: "name-escaped-plain", u: unionNameEscaped("Helvetica"), want: "/Helvetica"},
		{name: "name-escaped-special", u: unionNameEscaped("A B#C"), want: "/A#20B#23C"},
		{name: "ref", u: unionRef(IndirectReference{value: 5}), want: "5 0 R"},
	}
	for _, c := range cases {
		if got := emitUnion(c.u); got != c.want {
			t.Errorf("%s: emit = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestByteStringEmit(t *testing.T) {
	// Literal form is chosen when it is no longer than the hex form; else hex.
	cases := []struct {
		in   string
		want string
	}{
		{in: "", want: "()"},
		{in: "abc", want: "(abc)"},
		{in: "a(b)c", want: "(a\\(b\\)c)"},         // escape parens
		{in: "a\\b", want: "(a\\\\b)"},             // escape backslash
		{in: "\x01\x02", want: "<0102>"},           // non-printable -> hex is shorter
		{in: "\x0a", want: "<0A>"},                 // single control byte -> hex (3+len vs 2+2)
		{in: string([]byte{0xFF}), want: "<FF>"},   // high byte -> hex
		{in: "hello world", want: "(hello world)"}, // printable -> literal
	}
	for _, c := range cases {
		s := stream.NewMemoryWStream()
		writeByteString(s, c.in)
		if got := string(s.Bytes()); got != c.want {
			t.Errorf("writeByteString(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestTextStringEmit(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{in: "Hello", want: "(Hello)"},
		{in: "a(b)", want: "(a\\(b\\))"},
		// Non-ASCII UTF-8 -> UTF-16BE hex with BOM. "é" is U+00E9.
		{in: "é", want: "<FEFF00E9>"},
		{in: "Aé", want: "<FEFF004100E9>"},
		// Astral plane character -> surrogate pair. U+1F600 -> D83D DE00.
		{in: "\U0001F600", want: "<FEFFD83DDE00>"},
		// Invalid UTF-8 -> "<>".
		{in: string([]byte{0xFF, 0xFE}), want: "<>"},
	}
	for _, c := range cases {
		s := stream.NewMemoryWStream()
		writeTextString(s, c.in)
		if got := string(s.Bytes()); got != c.want {
			t.Errorf("writeTextString(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestArrayEmit(t *testing.T) {
	a := NewArray()
	a.AppendInt(1)
	a.AppendInt(2)
	a.AppendScalar(3.5)
	if got, want := string(emitToBytes(a)), "[1 2 3.5]"; got != want {
		t.Errorf("array = %q, want %q", got, want)
	}

	empty := NewArray()
	if got, want := string(emitToBytes(empty)), "[]"; got != want {
		t.Errorf("empty array = %q, want %q", got, want)
	}

	// rectToArray produces [left top right bottom].
	if got, want := string(emitToBytes(makeArrayScalars(0, 0, 612, 792))), "[0 0 612 792]"; got != want {
		t.Errorf("rect array = %q, want %q", got, want)
	}
}

func TestOptionalArrayEmit(t *testing.T) {
	single := NewOptionalArray()
	single.AppendInt(7)
	if got, want := string(emitToBytes(single)), "7"; got != want {
		t.Errorf("single optional = %q, want %q", got, want)
	}
	multi := NewOptionalArray()
	multi.AppendInt(7)
	multi.AppendInt(8)
	if got, want := string(emitToBytes(multi)), "[7 8]"; got != want {
		t.Errorf("multi optional = %q, want %q", got, want)
	}
	none := NewOptionalArray()
	if got, want := string(emitToBytes(none)), "[]"; got != want {
		t.Errorf("empty optional = %q, want %q", got, want)
	}
}

func TestDictEmit(t *testing.T) {
	d := NewTypedDict("Page")
	if got, want := string(emitToBytes(d)), "<</Type /Page>>"; got != want {
		t.Errorf("typed dict = %q, want %q", got, want)
	}

	d2 := NewTypedDict("Page")
	d2.InsertInt("Count", 3)
	d2.InsertRef("Parent", IndirectReference{value: 4})
	// Records are separated by '\n', in insertion order.
	want := "<</Type /Page\n/Count 3\n/Parent 4 0 R>>"
	if got := string(emitToBytes(d2)); got != want {
		t.Errorf("dict = %q, want %q", got, want)
	}

	// Nested objects.
	outer := NewDict()
	inner := NewArray()
	inner.AppendInt(1)
	inner.AppendInt(2)
	outer.InsertObject("Kids", inner)
	if got, wantKids := string(emitToBytes(outer)), "<</Kids [1 2]>>"; got != wantKids {
		t.Errorf("nested = %q, want %q", got, wantKids)
	}

	if NewDict().Size() != 0 {
		t.Error("empty dict size != 0")
	}
	if got, wantEmpty := string(emitToBytes(NewDict())), "<<>>"; got != wantEmpty {
		t.Errorf("empty dict = %q, want %q", got, wantEmpty)
	}
}

func TestNameEscaping(t *testing.T) {
	// Every byte outside '!'..'~' and the special set is escaped as #XX (uppercase hex).
	s := stream.NewMemoryWStream()
	writeNameEscaped(s, "Aa0 /%()<>[]{}#\t")
	got := string(s.Bytes())
	// space=20, /=2F, %=25, (=28, )=29, <=3C, >=3E, [=5B, ]=5D, {=7B, }=7D, #=23, tab=09
	want := "Aa0#20#2F#25#28#29#3C#3E#5B#5D#7B#7D#23#09"
	if got != want {
		t.Errorf("writeNameEscaped = %q, want %q", got, want)
	}
	if !strings.HasPrefix(got, "Aa0") {
		t.Error("printable prefix altered")
	}
}

func TestIndirectReferenceValidity(t *testing.T) {
	if (IndirectReference{}).IsValid() {
		t.Error("zero reference should be invalid")
	}
	if !(IndirectReference{value: 1}).IsValid() {
		t.Error("reference 1 should be valid")
	}
}
