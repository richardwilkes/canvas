// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A write-stream wrapper that DEFLATE-compresses everything written to it into an underlying stream, producing the
// zlib-format data a PDF /FlateDecode filter expects (a 2-byte zlib header + adler32 trailer, windowBits 15 — Go's
// compress/zlib emits exactly that container). The compressed bytes themselves differ from zlib's C reference output (a
// different DEFLATE implementation), which is invisible through a FlateDecode reader; gzip output is not needed by the
// PDF backend and is omitted.

package pdf

import (
	"compress/zlib"

	"github.com/richardwilkes/canvas/stream"
)

// DeflateWStream DEFLATE-compresses everything written to it into an underlying stream.
type DeflateWStream struct {
	adapter    *wstreamAdapter
	zw         *zlib.Writer
	inputBytes int64 // total uncompressed bytes written
	done       bool
}

// wstreamAdapter adapts a stream.WStream to io.Writer so compress/zlib can drive it.
type wstreamAdapter struct {
	s stream.WStream
}

func (w *wstreamAdapter) Write(p []byte) (int, error) {
	if !w.s.Write(p) {
		return 0, errStreamWrite
	}
	return len(p), nil
}

// errStreamWrite reports a failed downstream write (a full/broken FileWStream). Memory streams never fail.
var errStreamWrite = errWrite{}

type errWrite struct{}

func (errWrite) Error() string { return "pdf: stream write failed" }

// NewDeflateWStream wraps out so that data written through the returned stream is DEFLATE-compressed into out.
// compressionLevel is 1 (best speed) through 9 (best compression); -1 selects the default level. A level of 0 must
// never be passed here — the caller compresses only when the level is not None.
func NewDeflateWStream(out stream.WStream, compressionLevel int) *DeflateWStream {
	a := &wstreamAdapter{s: out}
	// NewWriterLevel only errors for an out-of-range level; the PDF caller always passes a valid one.
	zw, err := zlib.NewWriterLevel(a, compressionLevel)
	if err != nil {
		zw = zlib.NewWriter(a)
	}
	return &DeflateWStream{adapter: a, zw: zw}
}

// Write compresses buf into the underlying stream. Writes after Finalize fail.
func (d *DeflateWStream) Write(buf []byte) bool {
	if d.done {
		return false
	}
	n, err := d.zw.Write(buf)
	d.inputBytes += int64(n)
	return err == nil
}

// BytesWritten returns the total uncompressed bytes written, not the compressed size. The compressed size is observed
// on the underlying stream.
func (d *DeflateWStream) BytesWritten() int64 { return d.inputBytes }

// Finalize writes the end of the compressed stream. Subsequent Write calls fail; repeated Finalize calls are no-ops.
func (d *DeflateWStream) Finalize() {
	if d.done {
		return
	}
	// A Close failure means a downstream write failed, which the underlying stream has already latched; there is
	// nothing further to record here.
	d.zw.Close() //nolint:errcheck // see above
	d.done = true
}
