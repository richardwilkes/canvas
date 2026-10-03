// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the two per-draw temporaries SurfaceDrawContext.AddDrawOp must pass by address into Clip.Apply and
// DrawOp.Finalize: the conservative device-space draw bounds and the AppliedClip being filled in. The callees only read
// them or modify them in place and never retain them, but escape analysis cannot see through interface-method calls, so
// as plain locals both were heap-allocated on every device draw — the single largest per-frame allocation tier the
// GPU-frame benchmarks measure (~2 allocs on every AddDrawOp call).
//
// A sync.Pool (rather than a scratch field on the SurfaceDrawContext) is required because clip application re-enters
// AddDrawOp on the *same* context: a stencil or SW-mask clip renders its mask through sdc.StencilRect / sdc.AddDrawOp
// from inside Clip.Apply, so a shared field would be clobbered mid-draw while the outer call still holds live pointers
// into it. Each (possibly nested) call borrows its own scratch and hands it back on return.
//
// Ownership invariant that makes recycling safe: neither temporary is reachable once AddDrawOp returns. bounds is only
// read or shrunk in place by Clip.Apply and copied by value into the op via SetClippedBounds. appliedClip is copied by
// value into OpsTask.AddDrawOp, which — only when it actually clips — makes its own retained copy in recordOp; no Op or
// RenderTask ever holds a pointer to this struct.

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

// recycleDrawOpScratch returns a scratch to the pool, clearing it first so the pool does not pin the last draw's
// coverage fragment processor (a non-nil appliedClip.coverageFP transitively references textures). The caller must
// treat the scratch as invalid afterward.
func recycleDrawOpScratch(s *drawOpScratch) {
	*s = drawOpScratch{}
	drawOpScratchPool.Put(s)
}
