// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression guard for the buffer-offset-as-pointer GC hazard (GA hardening). GL entry points whose trailing "pointer"
// argument is really a byte offset into a bound buffer object (glVertexAttribPointer, glDrawElements and friends) must
// pass that offset as a uintptr, never synthesized into an unsafe.Pointer. When it was an unsafe.Pointer, a nonzero
// offset (e.g. 0xc) occupied a pointer-typed stack slot across the purego.SyscallN call; if that goroutine's stack was
// copied mid-call (morestack growth, which aggressive GC makes recur by repeatedly shrinking the stack), the runtime's
// stack scanner rejected 0xc as an invalid pointer and aborted the process with "invalid pointer found on stack". Skips
// when no GL context is available.

package gl_test

import (
	"runtime"
	"runtime/debug"
	"testing"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/gpu/gl"
)

// TestOffsetDrawGCStackSafety hammers the offset-based indexed draw path (filled AA circles lower to circleOp, whose
// interleaved vertex-attribute fetch and glDrawElements both use nonzero buffer offsets) while a background goroutine
// forces GC cycles. The GC cycles keep shrinking the drawing goroutine's stack, so each frame's SyscallN re-triggers a
// growth-time stack copy that scans the wrapper frame — exactly the window in which the pre-fix code found the fake 0xc
// pointer and crashed. With offsets riding as uintptr, that scan sees a plain integer and the run completes.
func TestOffsetDrawGCStackSafety(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	const side = 128
	sdc := newLiveDrawSDC(t, dc, side, side)
	defer sdc.Release()
	c := canvas.New(gl.NewDevice(sdc))

	fill := canvas.NewPaint()
	fill.AntiAlias = true
	fill.Color = 0xFF3388CC

	// Warm the program/resource caches so the timed frames run the steady-state offset draw path rather than
	// first-frame shader compiles and buffer allocations.
	drawCircleGrid(c, fill, side)
	dc.FlushAndSubmit(false)
	if dc.NumGLDraws() == 0 {
		t.Fatal("warm-up frame issued no GL draws — the frame did not render")
	}

	// Maximal stack-move pressure: a minimal GC percent plus a background goroutine forcing full GC cycles. Each cycle
	// scans and can shrink the drawing goroutine's stack; the next frame's deep SyscallN then grows it again, copying
	// (and scanning) the wrapper frames that carry the offset.
	defer debug.SetGCPercent(debug.SetGCPercent(1))
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()

	for i := 0; i < 300; i++ {
		drawCircleGrid(c, fill, side)
		dc.FlushAndSubmit(false)
		runtime.Gosched() // give the GC hammer a chance to shrink the stack between frames
	}
	close(stop)
	<-done
}

// drawCircleGrid draws a grid of filled AA circles. Each circle lowers to an indexed circleOp whose
// glVertexAttribPointer (interleaved layout → nonzero attribute offsets) and glDrawElements (nonzero index offset when
// batched) pass the buffer offset that must ride as a uintptr.
func drawCircleGrid(c *canvas.Canvas, fill *canvas.Paint, side float32) {
	const n = 5
	step := side / float32(n)
	r := step * 0.35
	for gy := 0; gy < n; gy++ {
		for gx := 0; gx < n; gx++ {
			cx := (float32(gx) + 0.5) * step
			cy := (float32(gy) + 0.5) * step
			c.DrawCircle(cx, cy, r, fill)
		}
	}
}
