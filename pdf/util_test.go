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
	"strconv"
	"testing"

	"github.com/richardwilkes/canvas/stream"
)

func TestColorToDecimal(t *testing.T) {
	cases := []struct {
		want string
		in   uint8
	}{
		{in: 0, want: "0"},
		{in: 255, want: "1"},
		{in: 128, want: ".502"},
		{in: 1, want: ".004"},
		{in: 64, want: ".251"},
		{in: 192, want: ".753"},
		{in: 254, want: ".996"},
	}
	for _, c := range cases {
		if got := string(appendColorComponent(nil, c.in)); got != c.want {
			t.Errorf("appendColorComponent(%d) = %q, want %q", c.in, got, c.want)
		}
	}
	// Every value must reproduce round(1000*value/255) trimmed of trailing zeros, and parse back within a permil of
	// value/255.
	for v := 0; v < 256; v++ {
		got := string(appendColorComponent(nil, uint8(v)))
		f, err := strconv.ParseFloat("0"+got, 64)
		if err != nil {
			t.Fatalf("value %d produced unparseable %q", v, got)
		}
		if diff := f - float64(v)/255.0; diff > 0.001 || diff < -0.001 {
			t.Errorf("value %d -> %q parses to %g, too far from %g", v, got, f, float64(v)/255.0)
		}
	}
}

func TestColorToDecimalF(t *testing.T) {
	cases := []struct {
		want string
		in   float32
	}{
		{in: 0, want: "0"},
		{in: 1, want: "1"},
		{in: 0.5, want: ".5"},
		{in: 0.25, want: ".25"},
		{in: -0.1, want: "0"},        // clamp low
		{in: 1.5, want: "1"},         // clamp high
		{in: 0.12345, want: ".1235"}, // 4 significant digits, rounded
		{in: 0.0001, want: ".0001"},
	}
	for _, c := range cases {
		if got := string(appendColorComponentF(nil, c.in)); got != c.want {
			t.Errorf("appendColorComponentF(%g) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestWriteBigDecAsText(t *testing.T) {
	cases := []struct {
		want      string
		v         int64
		minDigits int
	}{
		{v: 0, minDigits: 10, want: "0000000000"},
		{v: 123, minDigits: 10, want: "0000000123"},
		{v: 9999999999, minDigits: 10, want: "9999999999"},
		{v: 42, minDigits: 0, want: "42"},
		{v: 5, minDigits: 3, want: "005"},
		{v: 1234, minDigits: 2, want: "1234"}, // already longer than minDigits
	}
	for _, c := range cases {
		s := stream.NewMemoryWStream()
		writeBigDecAsText(s, c.v, c.minDigits)
		if got := string(s.Bytes()); got != c.want {
			t.Errorf("writeBigDecAsText(%d, %d) = %q, want %q", c.v, c.minDigits, got, c.want)
		}
	}
}

func TestWriteUTF16beHex(t *testing.T) {
	cases := []struct {
		want string
		in   rune
	}{
		{in: 'A', want: "0041"},
		{in: 0x00E9, want: "00E9"},      // é
		{in: 0x1F600, want: "D83DDE00"}, // grinning face -> surrogate pair
		{in: 0xFFFF, want: "FFFF"},
	}
	for _, c := range cases {
		s := stream.NewMemoryWStream()
		writeUTF16beHex(s, c.in)
		if got := string(s.Bytes()); got != c.want {
			t.Errorf("writeUTF16beHex(%U) = %q, want %q", c.in, got, c.want)
		}
	}
}
