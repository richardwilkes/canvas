// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the two per-draw temporaries SurfaceDrawContext.AddDrawOp must pass by address into the virtual
// Clip.Apply and DrawOp.Finalize calls: the conservative device-space draw bounds and the AppliedClip being filled in.
// Ideally both would be plain stack locals whose addresses are handed to the callees, which only read them or modify
// them in place and never retain them, costing no heap allocation. But in Go the addresses flow into interface-method
// calls (Clip.Apply, DrawOp.Finalize), which escape analysis cannot see through, so both locals were heap-allocated on
// every device draw — the single largest per-frame allocation tier the GPU-frame benchmarks measure (~2 allocs on every
// AddDrawOp call). Pooling recovers stack-allocated-like behavior.
//
// A sync.Pool (rather than a scratch field on the SurfaceDrawContext) is required because clip application re-enters
// AddDrawOp on the *same* context: a stencil or SW-mask clip renders its mask through sdc.StencilRect / sdc.AddDrawOp
// from inside Clip.Apply, so a shared field would be clobbered mid-draw while the outer call still holds live pointers
// into it. Each (possibly nested) call borrows its own scratch and hands it back on return, just like nested stack
// frames would get their own locals.
//
// Ownership invariant (the retention audit that makes recycling safe): neither temporary is reachable once AddDrawOp
// returns. bounds is only ever copied by value (into the op via SetClippedBounds, into clip.Apply/setupDstProxyView
// which read or shrink it in place). appliedClip is copied by value into OpsTask.AddDrawOp, which — only when it
// actually clips — makes its own retained copy in recordOp; no Op or RenderTask ever holds a pointer to this struct.
// Therefore the deferred recycle at the single return path is always safe: by the time it runs, every consumer that
// needed the data has copied it.

package gl

import (
	"sync"

	"github.com/richardwilkes/canvas/geom"
)

// drawOpScratch bundles AddDrawOp's two escaping per-draw temporaries so a single pool Get/Put covers both.
type drawOpScratch struct {
	appliedClip AppliedClip
	bounds      geom.Rect
}

var drawOpScratchPool = sync.Pool{New: func() any { return &drawOpScratch{} }}

// borrowDrawOpScratch draws a scratch from the pool. Its fields are overwritten in full by the caller before use
// (bounds from opBounds, appliedClip from MakeAppliedClipWithLogicalDims), so no reset is needed here.
func borrowDrawOpScratch() *drawOpScratch { return drawOpScratchPool.Get().(*drawOpScratch) }

// recycleDrawOpScratch returns a scratch to the pool, clearing it first so the pool pins neither the last draw's
// coverage fragment processor (a non-nil appliedClip.coverageFP transitively references textures) nor anything else.
// After recycleDrawOpScratch the caller must treat the scratch as invalid.
func recycleDrawOpScratch(s *drawOpScratch) {
	*s = drawOpScratch{}
	drawOpScratchPool.Put(s)
}
