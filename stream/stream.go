// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package stream provides the write-stream types that are publicly reachable: a dynamic memory stream with read-back
// and a file stream. Writes report success as a bool, and a failed file stream latches into an error state rather than
// returning Go errors per call.
package stream

import (
	"bytes"
	"encoding/binary"
	"os"
)

// WStream is the core write-stream surface: append bytes and report how many have been written.
type WStream interface {
	// Write appends buf, returning false on failure.
	Write(buf []byte) bool
	// BytesWritten returns the total number of bytes successfully written so far.
	BytesWritten() int64
}

// MemoryWStream is a dynamic, growable in-memory write stream.
type MemoryWStream struct {
	buf bytes.Buffer
}

// NewMemoryWStream creates an empty dynamic memory stream.
func NewMemoryWStream() *MemoryWStream {
	return &MemoryWStream{}
}

// Write appends buf. It always succeeds (memory exhaustion panics, as is normal in Go).
func (s *MemoryWStream) Write(buf []byte) bool {
	s.buf.Write(buf)
	return true
}

// BytesWritten returns the total size written.
func (s *MemoryWStream) BytesWritten() int64 {
	return int64(s.buf.Len())
}

// Bytes returns the accumulated contents without copying; the result aliases the stream's storage and is only valid
// until the next write or reset.
func (s *MemoryWStream) Bytes() []byte {
	return s.buf.Bytes()
}

// CopyTo copies the contents into dst, which must be at least BytesWritten long.
func (s *MemoryWStream) CopyTo(dst []byte) {
	copy(dst, s.buf.Bytes())
}

// DetachAsData returns the contents and resets the stream.
func (s *MemoryWStream) DetachAsData() []byte {
	data := make([]byte, s.buf.Len())
	copy(data, s.buf.Bytes())
	s.buf.Reset()
	return data
}

// Reset discards the contents.
func (s *MemoryWStream) Reset() {
	s.buf.Reset()
}

// WriteToAndReset appends this stream's contents to the end of dst, then resets this stream. dst must not alias this
// stream.
func (s *MemoryWStream) WriteToAndReset(dst *MemoryWStream) {
	dst.buf.Write(s.buf.Bytes())
	s.buf.Reset()
}

// PrependToAndReset inserts this stream's contents before dst's existing contents, then resets this stream. dst must
// not alias this stream.
func (s *MemoryWStream) PrependToAndReset(dst *MemoryWStream) {
	if s.buf.Len() == 0 {
		return
	}
	combined := make([]byte, 0, s.buf.Len()+dst.buf.Len())
	combined = append(combined, s.buf.Bytes()...)
	combined = append(combined, dst.buf.Bytes()...)
	dst.buf.Reset()
	dst.buf.Write(combined)
	s.buf.Reset()
}

// FileWStream writes to a file, latching into an error state on failure.
type FileWStream struct {
	f       *os.File
	written int64
	err     bool
}

// NewFileWStream creates or truncates the named file. A nil result with false indicates the file could not be opened.
func NewFileWStream(path string) (*FileWStream, bool) {
	f, err := os.Create(path)
	if err != nil {
		return nil, false
	}
	return &FileWStream{f: f}, true
}

// Write appends buf, returning false if the stream is in the error state or the write fails.
func (s *FileWStream) Write(buf []byte) bool {
	if s.err || s.f == nil {
		return false
	}
	n, err := s.f.Write(buf)
	s.written += int64(n)
	if err != nil {
		s.err = true
		return false
	}
	return true
}

// BytesWritten returns the total bytes successfully handed to the OS.
func (s *FileWStream) BytesWritten() int64 {
	return s.written
}

// Flush syncs any buffered writes to the underlying file. A failed sync latches the error state.
func (s *FileWStream) Flush() {
	if s.f != nil {
		if err := s.f.Sync(); err != nil {
			s.err = true
		}
	}
}

// Close closes the underlying file. Further writes fail. A failed close latches the error state.
func (s *FileWStream) Close() {
	if s.f != nil {
		if err := s.f.Close(); err != nil {
			s.err = true
		}
		s.f = nil
	}
}

// Write helpers shared by the PDF backend and encoders.

// WriteU16LE writes a little-endian uint16.
func WriteU16LE(s WStream, v uint16) bool {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	return s.Write(b[:])
}

// WriteU32LE writes a little-endian uint32.
func WriteU32LE(s WStream, v uint32) bool {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return s.Write(b[:])
}

// WriteText writes a string's bytes.
func WriteText(s WStream, text string) bool {
	return s.Write([]byte(text))
}
