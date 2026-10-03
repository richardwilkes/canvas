// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import "sync"

// pathPool retains Path storage across uses so the draw pipeline's transient working paths (stroker scratch,
// FillPathWithPaint's builder, per-draw device-space paths) do not allocate fresh backing arrays on every draw.
var pathPool = sync.Pool{New: func() any { return &Path{} }}

// Borrow returns an empty Path drawn from a process-wide pool that retains slice capacity across uses. The returned
// Path is in the same state as a freshly-constructed one (empty, winding fill). Hand it back with Recycle when done.
// Because its storage is reused, callers must not retain the Path — or any points/verbs slice obtained from it — past
// the matching Recycle. Ownership must stay unique: never Recycle a Path whose storage has been moved elsewhere via
// `*dst = *p`; copy with Set instead.
func Borrow() *Path {
	return pathPool.Get().(*Path)
}

// Recycle returns a Path obtained from Borrow to the shared pool, rewinding it first so its storage is retained for the
// next Borrow. After Recycle the caller must treat p (and any slice previously obtained from it) as invalid.
func Recycle(p *Path) {
	p.Rewind()
	// Rewind deliberately keeps the volatile flag (it describes the object, not its contents), but the pool hands the
	// object to an unrelated caller, so clear it here to honor Borrow's fresh-state contract.
	p.isVolatile = false
	pathPool.Put(p)
}

// Set replaces the receiver's contents with a copy of src (every field, including the generation ID and fill type),
// reusing the receiver's storage where possible. src keeps its own storage, so both paths may be pooled independently.
// src may equal the receiver (a no-op).
func (p *Path) Set(src *Path) {
	p.copyFrom(src)
}
