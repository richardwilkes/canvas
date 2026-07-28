// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Free list for the OpsTask []opChain backing. A fresh OpsTask is created for each surface every frame — the previous
// one is closed at flush and replaced (SurfaceFillContext. GetOpsTask → replaceOpsTask → NewOpsTask) — so without
// pooling every frame re-grows its opChains slice from nil, which the GPU-frame benchmarks measure as the single
// largest in-our-code allocation (the append in OpsTask.recordOp). The C++ OpsTask is arena-allocated and freed
// wholesale with the flush, so a steady-state C++ frame allocates none of this; the library recovers that by handing
// each dead task's grown backing to the next frame's task.
//
// The backing is wrapped in a pointer-sized box so the free list stores a *opChainsBox rather than a []opChain —
// putting a slice value into a sync.Pool boxes its 3-word header (a heap allocation per Put), which pooling a pointer
// avoids (matching oppool.go / paintpool.go, which pool *T).
//
// Lifetime safety: a box is recycled only from OpsTask.onDelete — the refCnt==0 death hook — where the task is provably
// dead (disowned, and its ops already recycled by EndFlush→deleteOps), so the backing is uniquely owned and cannot be
// referenced by any live task. recycle clears the backing (dropping any residual opChain contents — op pointers,
// applied-clip coverage FPs) before returning it, so the pool pins nothing while preserving the array's capacity for
// reuse.

package gl

import "sync"

// opChainsBox carries an OpsTask's []opChain backing between frames.
type opChainsBox struct {
	chains []opChain
}

var opChainsBoxPool = sync.Pool{New: func() any { return &opChainsBox{} }}

// borrowOpChainsBox returns a box whose backing is either a prior frame's grown array (reset to length 0) or, for a
// fresh box, nil (append grows it as usual). NewOpsTask starts its opChains from the box's backing so recordOp's append
// reuses the array instead of re-growing from nil.
func borrowOpChainsBox() *opChainsBox { return opChainsBoxPool.Get().(*opChainsBox) }

// recycleOpChainsBox clears the box's backing and returns it to the pool. It must be called only on a box whose OpsTask
// is dead (see the file comment). The full clear drops every reference the backing held so the pool retains only an
// empty (but capacity-preserving) array.
func recycleOpChainsBox(box *opChainsBox) {
	backing := box.chains[:cap(box.chains)]
	clear(backing)
	box.chains = backing[:0]
	opChainsBoxPool.Put(box)
}
