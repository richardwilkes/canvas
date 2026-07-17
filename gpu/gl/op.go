// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The base of all deferred GPU operations. Draw arguments are captured into ops and geometry is generated at flush
// time, which is what allows the batching machinery (merge and chain in OpsTask) to minimize draw calls. This takes the
// form of an Op interface plus an embedded OpBase carrying the chain links and bounds. "Deleting" an op is simply
// dropping the reference to it — the arena allocation strategy arrives with the op recording work, so combineIfPossible
// has no arena-allocator parameter for now. There is no separate "pre-prepare" pass; prepare happens once, right before
// execute.

package gl

import (
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// CombineResult reports the outcome of attempting to combine two ops.
type CombineResult int32

// CombineResult values.
const (
	// CombineResultMerged: the receiving op now represents its own work plus that of the passed op; the passed op
	// should be dropped without being flushed.
	CombineResultMerged CombineResult = iota
	// CombineResultMayChain: the caller may (but need not) chain the ops; the head op prepares and executes on behalf
	// of the whole chain.
	CombineResultMayChain
	// CombineResultCannotCombine: the ops cannot be combined.
	CombineResultCannotCombine
)

var (
	currOpUniqueID atomic.Uint32
	currOpClassID  atomic.Uint32
)

// GenOpClassID returns a fresh, process-unique class ID; called once per op type to obtain its class ID.
func GenOpClassID() uint32 { return currOpClassID.Add(1) }

func genOpID() uint32 { return currOpUniqueID.Add(1) }

// Op is the interface all deferred GPU operations implement.
type Op interface {
	// opBase gives the task machinery access to the embedded OpBase.
	opBase() *OpBase
	// Name returns a human-readable name for the op, used in logging and debugging.
	Name() string
	// VisitProxies calls fn for every SurfaceProxy the op reads from or writes to; implementations with no proxies
	// inherit OpBase's no-op.
	VisitProxies(func(*SurfaceProxy, gpu.Mipmapped))
	// OnCombineIfPossible attempts to combine this op with that, which must be of the same class.
	OnCombineIfPossible(that Op) CombineResult
	// OnPrepare performs resource creation/data transfer before execution.
	OnPrepare(*OpFlushState)
	// OnExecute issues the op's commands to the Gpu. If the op is chained, chainBounds is the union of the bounds of
	// all ops in the chain, else the op's bounds.
	OnExecute(state *OpFlushState, chainBounds geom.Rect)
	// recycle returns a dead op to its type's free list for reuse; ops without a pool inherit OpBase's no-op and are
	// left for the GC. Called by the OpsTask machinery the instant an op becomes dead — see oppool.go for the lifetime
	// invariant.
	recycle()
}

// Op bounds flags.
const (
	opBoundsFlagAABloat  uint8 = 0x1
	opBoundsFlagZeroArea uint8 = 0x2
)

// HasAABloat reports that the op produces geometry beyond its bounds so the fragment shader runs on partially covered
// pixels for non-MSAA antialiasing.
type HasAABloat bool

// IsHairline reports that the geometry is a hairline stroke (or device-space point).
type IsHairline bool

// OpBase carries the common op state and is embedded by every concrete op.
type OpBase struct {
	self        Op
	nextInChain Op
	prevInChain Op
	classID     uint32
	uniqueID    uint32
	boundsFlags uint8
	bounds      geom.Rect
}

func (b *OpBase) opBase() *OpBase { return b }

// InitOp initializes the embedded OpBase; every concrete op must call it with its class ID and itself.
func (b *OpBase) InitOp(classID uint32, self Op) {
	if classID == 0 {
		panic("op class ID must be non-zero")
	}
	b.classID = classID
	b.self = self
}

// ClassID returns the op's class ID, as assigned by GenOpClassID.
func (b *OpBase) ClassID() uint32 { return b.classID }

// OpUniqueID returns a unique ID for this op instance, lazily assigning one on first call.
func (b *OpBase) OpUniqueID() uint32 {
	if b.uniqueID == 0 {
		b.uniqueID = genOpID()
	}
	return b.uniqueID
}

// Bounds returns the op's bounds; it must contain all vertices in device space irrespective of clip.
func (b *OpBase) Bounds() geom.Rect { return b.bounds }

// SetClippedBounds replaces the op's bounds with a tighter, clip-intersected rect.
func (b *OpBase) SetClippedBounds(clippedBounds geom.Rect) {
	b.bounds = clippedBounds
	// The clipped bounds already incorporate any effect of the bounds flags.
	b.boundsFlags = 0
}

// HasAABloat reports whether the op's bounds were expanded for antialiasing.
func (b *OpBase) HasAABloat() bool { return b.boundsFlags&opBoundsFlagAABloat != 0 }

// HasZeroArea reports whether the op's bounds have zero area (e.g. a hairline or point).
func (b *OpBase) HasZeroArea() bool { return b.boundsFlags&opBoundsFlagZeroArea != 0 }

// SetBounds sets the op's bounds and records whether they include AA bloat and/or zero area.
func (b *OpBase) SetBounds(newBounds geom.Rect, aabloat HasAABloat, zeroArea IsHairline) {
	b.bounds = newBounds
	b.boundsFlags = 0
	if aabloat {
		b.boundsFlags |= opBoundsFlagAABloat
	}
	if zeroArea {
		b.boundsFlags |= opBoundsFlagZeroArea
	}
}

// VisitProxies is the default no-op for ops without proxies.
func (b *OpBase) VisitProxies(func(*SurfaceProxy, gpu.Mipmapped)) {}

// recycle is the default no-op for ops that are not pooled; such ops are simply dropped for the GC when their chain is
// deleted. Pooled ops (the batchable geometry ops) override this to return themselves to their free list.
func (b *OpBase) recycle() {}

// OnCombineIfPossible is the default: ops cannot combine.
func (b *OpBase) OnCombineIfPossible(Op) CombineResult { return CombineResultCannotCombine }

// joinBounds folds that's bounds and bounds flags into b's, as part of a successful merge.
func (b *OpBase) joinBounds(that *OpBase) {
	b.boundsFlags |= that.boundsFlags & (opBoundsFlagAABloat | opBoundsFlagZeroArea)
	b.bounds = joinPossiblyEmptyRect(b.bounds, that.bounds)
}

// CombineIfPossible attempts to combine this op with that, requiring them to share a class ID and delegating to
// OnCombineIfPossible; on a merge, the bounds are joined.
func (b *OpBase) CombineIfPossible(that Op) CombineResult {
	if b.self == that {
		panic("an op cannot combine with itself")
	}
	if b.classID != that.opBase().classID {
		return CombineResultCannotCombine
	}
	result := b.self.OnCombineIfPossible(that)
	if result == CombineResultMerged {
		b.joinBounds(that.opBase())
	}
	return result
}

// ChainConcat appends next to this op's chain: this op must be a tail, next a head of the same class.
func (b *OpBase) ChainConcat(next Op) {
	if next == nil || b.classID != next.opBase().classID ||
		!b.IsChainTail() || !next.opBase().IsChainHead() {
		panic("invalid chain concat")
	}
	b.nextInChain = next
	next.opBase().prevInChain = b.self
}

// IsChainHead reports whether this op is the first in its chain.
func (b *OpBase) IsChainHead() bool { return b.prevInChain == nil }

// IsChainTail reports whether this op is the last in its chain.
func (b *OpBase) IsChainTail() bool { return b.nextInChain == nil }

// NextInChain returns the next op in this op's chain, or nil if this is the tail.
func (b *OpBase) NextInChain() Op { return b.nextInChain }

// PrevInChain returns the previous op in this op's chain, or nil if this is the head.
func (b *OpBase) PrevInChain() Op { return b.prevInChain }

// CutChain cuts after this op, returning the previously-next op (nil if this was already a tail).
func (b *OpBase) CutChain() Op {
	if b.nextInChain != nil {
		next := b.nextInChain
		next.opBase().prevInChain = nil
		b.nextInChain = nil
		return next
	}
	return nil
}

// prepareOp / executeOp are small dispatch helpers used by the task machinery.
func prepareOp(op Op, state *OpFlushState) { op.OnPrepare(state) }

func executeOp(op Op, state *OpFlushState, chainBounds geom.Rect) {
	op.OnExecute(state, chainBounds)
}

// Rect helpers for bounds arithmetic used throughout the op pipeline.

// rectsOverlap reports whether a and b overlap. Invalid (NaN) rects report overlapping everything.
func rectsOverlap(a, b geom.Rect) bool {
	return a.Right > b.Left && a.Bottom > b.Top && b.Right > a.Left && b.Bottom > a.Top
}

// rectsTouchOrOverlap reports whether a and b overlap or share an edge.
func rectsTouchOrOverlap(a, b geom.Rect) bool {
	return a.Right >= b.Left && a.Bottom >= b.Top && b.Right >= a.Left && b.Bottom >= a.Top
}

// largestInvertedRect returns an inverted rect (Left/Top at +max, Right/Bottom at -max) suitable as the starting
// accumulator for a join-based bounds computation.
func largestInvertedRect() geom.Rect {
	const maxF32 = 3.402823466e+38
	return geom.Rect{Left: maxF32, Top: maxF32, Right: -maxF32, Bottom: -maxF32}
}

// joinPossiblyEmptyRect returns the unconditional min/max join of r and other, without special- casing empty rects.
func joinPossiblyEmptyRect(r, other geom.Rect) geom.Rect {
	return geom.Rect{
		Left:   min(r.Left, other.Left),
		Top:    min(r.Top, other.Top),
		Right:  max(r.Right, other.Right),
		Bottom: max(r.Bottom, other.Bottom),
	}
}
