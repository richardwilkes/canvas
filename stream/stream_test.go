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
}
