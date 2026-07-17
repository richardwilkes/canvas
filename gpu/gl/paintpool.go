// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pooling for the transient *Paint (paint.go) produced when a user-facing paint is converted for a draw. Since the
// conversion must return a *Paint (it flows through the surface draw context into an op factory), the value escapes to
// the heap — ~one allocation per draw, the largest single per-frame allocation tier the GPU-frame benchmarks measure.
// Pooling recovers a stack-like allocation cost by reusing paint containers across draws instead of allocating a fresh
// one each time.
//
// Ownership invariant (the retention audit that makes recycling safe): a converted *Paint is dead once the top-level
// draw call that consumed it returns. Every op factory reads the paint color by value (paint.Color4f()) and, only for
// non-trivial paints, moves the fragment processors and XP factory out via NewProcessorSetFromPaint (which marks the
// paint moved-from); no Op or RenderTask ever retains a *Paint — the op holds the resulting *ProcessorSet instead. The
// only structs that hold a *Paint (CanDrawPathArgs / DrawPathArgs) are themselves transient and die with the draw call.
// Therefore recyclePaint is called at the draw-call sites (device.go) after the draw has been recorded, never earlier,
// and never on a paint whose contents are still reachable.

package gl

import (
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
)

// paintPool retains *Paint containers across draws. The stored FPs/XP are always moved out before a paint is recycled
// (see the ownership invariant above), so the pool never pins fragment processors.
var paintPool = sync.Pool{New: func() any { return &Paint{} }}

// borrowPaint returns a default paint (solid white, no XP factory, trivial) drawn from the pool — the pooled analog
// of NewPaint. Hand it back with recyclePaint once the draw that consumed it has been recorded. The returned paint is
// byte-identical in state to NewPaint()'s result.
func borrowPaint() *Paint {
	p := paintPool.Get().(*Paint)
	*p = Paint{color: colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}, initialized: true}
	return p
}

// recyclePaint returns a paint obtained from borrowPaint to the pool, clearing it first so the pool retains no
// reference to any fragment processor or XP factory (defense in depth: a non-trivial paint's FPs were already moved out
// by NewProcessorSetFromPaint). After recyclePaint the caller must treat the paint as invalid.
func recyclePaint(p *Paint) {
	*p = Paint{}
	paintPool.Put(p)
}
