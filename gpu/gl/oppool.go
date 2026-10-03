// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Op-recording free lists: a per-op-type free list so a steady-state frame allocates essentially no op storage. A draw
// borrows its op shell from the pool instead of heap allocating a fresh one, and the ops-task machinery recycles each
// op back to its pool the instant it becomes dead (surviving ops once execution completes, merged-away ops at the merge
// points in the op chain's combine logic — see opstask.go).
//
// Lifetime safety (recycling a live op corrupts a frame):
//   - recycle zeroes the whole op (`var zero T; *o = zero`) before returning it to the pool. A complete zero, rather
//     than a hand-written field reset, makes stale state on reuse impossible and drops every reference the op held
//     (ProcessorSet, program info, GPU buffers, any heap-grown instance slice), so the pool retains only empty shells.
//     The one exception is recycleKeepingBacking (below), which preserves only the heap-grown instance-slice backing;
//     see that function for why it is safe.
//   - An op is recycled only when it is provably dead: a survivor is recycled at deleteOps, which the drawing manager
//     calls in EndFlush strictly after OnExecute; a merged-away op is recycled right where the batching machinery drops
//     it, after CombineIfPossible has copied its instances into the surviving op (append copies element values, so the
//     survivor never aliases the recycled op's storage). Nothing else in the library retains an op pointer past its
//     chain — the program/resource/thread-safe caches are keyed on descriptors and shapes, not ops.
//   - Because a merge and its recycle happen during recording/close, an op recycled here may be re-borrowed by a later
//     draw in the same frame; that is safe for the same reason (the recycled op is dead, its data already copied out).
//   - If a still-live op were ever recycled, the zero would corrupt that op's own draw in the current frame, which the
//     single-frame live render/compare suite catches; TestOpPoolCrossFrameStability adds a same-context A/B/A
//     pixel-equality check so a stale reused shell (were the reset ever incomplete) is caught too.
//
// The pool is a package-level sync.Pool per op type, matching paintpool.go: it is safe for the single recording
// goroutine per context and cooperates with the GC (entries may be dropped under pressure, harmlessly falling back to a
// fresh allocation). The Arenas type in proxy.go carries only the per-frame flush protocol and holds no arena storage.

package gl

import "sync"

// opPool is the free list for one concrete op type. Its zero value is ready to use.
type opPool[T any] struct {
	pool sync.Pool
}

// borrow returns a recycled shell when one is available and otherwise allocates a fresh one. The result is zero except
// for a slice backing preserved by recycleKeepingBacking.
func (p *opPool[T]) borrow() *T {
	if v := p.pool.Get(); v != nil {
		return v.(*T)
	}
	return new(T)
}

// recycle zeroes the op (dropping every reference it held) and returns it to the pool. It must be called only on an op
// that is dead — see the file comment for the invariant.
func (p *opPool[T]) recycle(o *T) {
	var zero T
	*o = zero
	p.pool.Put(o)
}

// recycleKeepingBacking returns dead value o to the pool after a full-zero reset that preserves its heap-grown slice
// backing, so a later user reuses the backing instead of re-growing it. Two callers: the batchable geometry ops (this
// file's op pools), whose instance slice starts in an inline [1] array and grows into the heap when OnCombineIfPossible
// merges instances in, and the pipeline pool (programinfopool.go), which preserves a Pipeline's fragmentProcessors
// backing the same way. getBacking reads the slice before the zero and setBacking re-installs it on the zeroed value;
// both are non-capturing, so this allocates nothing.
//
// Safety — this is the sole exception to the file comment's "full zero" rule:
//   - Only the slice backing survives, reset to length 0 and clear()ed, so every other field is zero exactly as with
//     plain recycle and no stale state can reach a reused shell.
//   - The backing carries no stale data: the live [:len] range is always overwritten before it is read (ops via
//     OnCombineIfPossible's append; a pipeline via NewPipeline's append), and clear() zeroes the reserved capacity
//     regardless. The op instance element types are pointer-free (TestInstanceTypesArePointerFree), so there a
//     preserved backing pins nothing and clear() is defensive; the pipeline's element type is the pointer-carrying
//     FragmentProcessor interface, so there clear() is load-bearing: it drops every retained FP reference.
//   - A slice that never grew past one element (cap ≤ 1) is dropped to nil, keeping the clean full zero for the common
//     single-instance op / trivial-paint pipeline; bootstrapInstances re-bootstraps ops from the inline array on the
//     next borrow, and NewPipeline re-grows the pipeline's from nil.
//
// It must be called only on a dead value, exactly like recycle.
func (p *opPool[T]) recycleKeepingBacking[E any](o *T, getBacking func(*T) []E, setBacking func(*T, []E)) {
	backing := getBacking(o)
	if cap(backing) > 1 {
		backing = backing[:cap(backing)]
		clear(backing)
		backing = backing[:0]
	} else {
		backing = nil
	}
	var zero T
	*o = zero
	setBacking(o, backing)
	p.pool.Put(o)
}

// bootstrapInstances returns the slice a batchable op's constructor should start its instance field from: a heap-grown
// backing preserved by a prior recycleKeepingBacking (reset to length 0) if one survived in the borrowed shell,
// otherwise the op's inline single-element backing array.
func bootstrapInstances[E any](preserved, inline []E) []E {
	if cap(preserved) > 1 {
		return preserved[:0]
	}
	return inline
}
