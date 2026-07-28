// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package stream

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestMemoryWStream(t *testing.T) {
	s := NewMemoryWStream()
	if !s.Write([]byte("hello ")) || !WriteText(s, "world") {
		t.Fatal("writes failed")
	}
	if s.BytesWritten() != 11 {
		t.Errorf("BytesWritten = %d", s.BytesWritten())
	}
	dst := make([]byte, 11)
	s.CopyTo(dst)
	if string(dst) != "hello world" {
		t.Errorf("CopyTo = %q", dst)
	}
	data := s.DetachAsData()
	if string(data) != "hello world" || s.BytesWritten() != 0 {
		t.Error("DetachAsData did not detach")
	}
	WriteU16LE(s, 0x0201)
	WriteU32LE(s, 0x06050403)
	if !bytes.Equal(s.Bytes(), []byte{1, 2, 3, 4, 5, 6}) {
		t.Errorf("LE writers produced %v", s.Bytes())
	}
}

func TestFileWStream(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.bin")
	s, ok := NewFileWStream(path)
	if !ok {
		t.Fatal("open failed")
	}
	if !s.Write([]byte("data")) || s.BytesWritten() != 4 {
		t.Fatal("write failed")
	}
	s.Flush()
	s.Close()
	content, err := os.ReadFile(path)
	if err != nil || string(content) != "data" {
		t.Errorf("file content = %q err=%v", content, err)
	}
	// Writes after close fail and latch.
	if s.Write([]byte("x")) {
		t.Error("write after close should fail")
	}
	// Opening an impossible path reports failure.
	if _, ok = NewFileWStream(filepath.Join(path, "sub", "impossible")); ok {
		t.Error("open of impossible path should fail")
	}
	// A stream that wrote and closed cleanly never latched.
	if s.Failed() {
		t.Error("a clean write/flush/close must not latch the error state")
	}
}

// TestFileWStreamLatchedErrorIsReadable pins that the latched error state is observable. A failure can surface only at
// Flush or Close — the OS may not report a write error (ENOSPC, a broken device) until the data is actually flushed —
// and Close leaves f nil, so every later Write returns false whether or not anything went wrong. Without Failed, a
// caller that wrote a whole file successfully has no way to tell a complete file from a truncated one. The underlying
// descriptor is closed out from under the stream here to make the syscalls fail on demand.
func TestFileWStreamLatchedErrorIsReadable(t *testing.T) {
	dir := t.TempDir()

	// A failure that first surfaces at Flush.
	s, ok := NewFileWStream(filepath.Join(dir, "flush.bin"))
	if !ok {
		t.Fatal("open failed")
	}
	if !s.Write([]byte("data")) || s.Failed() {
		t.Fatal("the write should have succeeded without latching")
	}
	if err := s.f.Close(); err != nil {
		t.Fatalf("closing the descriptor: %v", err)
	}
	s.Flush()
	if !s.Failed() {
		t.Error("a failed Flush must latch the error state")
	}
	s.Close()
	if !s.Failed() {
		t.Error("the latched state must survive Close")
	}

	// A failure that first surfaces at Close, with every Write having reported success.
	s2, ok := NewFileWStream(filepath.Join(dir, "close.bin"))
	if !ok {
		t.Fatal("open failed")
	}
	if !s2.Write([]byte("data")) {
		t.Fatal("write failed")
	}
	if err := s2.f.Close(); err != nil {
		t.Fatalf("closing the descriptor: %v", err)
	}
	if s2.Failed() {
		t.Fatal("nothing has failed through the stream yet")
	}
	s2.Close()
	if !s2.Failed() {
		t.Error("a failed Close must latch the error state, or a truncated file is undetectable")
	}
	if s2.Write([]byte("x")) {
		t.Error("write after close should fail")
	}
}
