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
	"bytes"
	"compress/zlib"
	"io"
	"math/rand"
	"testing"

	"github.com/richardwilkes/canvas/stream"
)

func inflate(t *testing.T, compressed []byte) []byte {
	t.Helper()
	r, err := zlib.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatalf("zlib.NewReader: %v", err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("inflate: %v", err)
	}
	return out
}

func TestDeflateRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	random := make([]byte, 50000)
	rng.Read(random)
	inputs := [][]byte{
		[]byte(""),
		[]byte("hello world"),
		bytes.Repeat([]byte("q 1 0 0 1 0 0 cm BT /F1 12 Tf (Hello) Tj ET Q\n"), 100),
		random,
	}

	for _, level := range []int{zlib.DefaultCompression, 1, 6, 9} {
		for i, in := range inputs {
			var out stream.MemoryWStream
			d := NewDeflateWStream(&out, level)
			if !d.Write(in) {
				t.Fatalf("level %d input %d: write failed", level, i)
			}
			if d.BytesWritten() != int64(len(in)) {
				t.Errorf("level %d input %d: BytesWritten = %d, want %d (uncompressed count)",
					level, i, d.BytesWritten(), len(in))
			}
			d.Finalize()
			if got := inflate(t, out.Bytes()); !bytes.Equal(got, in) {
				t.Errorf("level %d input %d: round trip mismatch (%d vs %d bytes)", level, i, len(got), len(in))
			}
		}
	}
}

func TestDeflateChunkedWrites(t *testing.T) {
	var out stream.MemoryWStream
	d := NewDeflateWStream(&out, zlib.DefaultCompression)
	full := []byte("The quick brown fox jumps over the lazy dog. ")
	for i := 0; i < 10; i++ {
		d.Write(full)
	}
	d.Finalize()
	want := bytes.Repeat(full, 10)
	if int(d.BytesWritten()) != len(want) {
		t.Errorf("BytesWritten = %d, want %d", d.BytesWritten(), len(want))
	}
	if got := inflate(t, out.Bytes()); !bytes.Equal(got, want) {
		t.Error("chunked round trip mismatch")
	}
}

func TestDeflateWriteAfterFinalizeFails(t *testing.T) {
	var out stream.MemoryWStream
	d := NewDeflateWStream(&out, zlib.DefaultCompression)
	d.Write([]byte("data"))
	d.Finalize()
	if d.Write([]byte("more")) {
		t.Error("write after finalize should fail")
	}
	d.Finalize() // idempotent, must not panic
}
