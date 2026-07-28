// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The analytic anti-aliased path scan converter, plus the AntiFillPath dispatcher. The algorithm walks
// non-equally-spaced scan lines (integer y values, segment endpoints, and approximate intersections) and computes
// per-pixel coverage analytically from the trapezoid each pair of consecutive scan lines cuts out of the path.

package raster

import (
	"cmp"
	"math"
	"slices"
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
)

// addAlpha adds delta into *alpha; the caller guarantees *alpha + delta <= 256.
func addAlpha(alpha *Alpha, delta Alpha) {
	*alpha = CatchOverflow(int(*alpha) + int(delta))
}

// safelyAddAlpha adds delta into *alpha, clamping at 0xFF.
func safelyAddAlpha(alpha *Alpha, delta Alpha) {
	v := int(*alpha) + int(delta)
	if v > 0xFF {
		v = 0xFF
	}
	*alpha = Alpha(v)
}

// additiveBlitter is implemented by the trapezoid-accumulation blitters: alphas are accumulated (added), not
// overwritten, so several trapezoids can compose into one scanline before it reaches the real blitter.
type additiveBlitter interface {
	// getRealBlitter returns the underlying device blitter, or the additive blitter itself when forceRealBlitter is
	// false and it can serve as its own "real blitter".
	getRealBlitter(forceRealBlitter bool) Blitter

	// blitAntiHRun adds len(antialias) alphas starting at x.
	blitAntiHRun(x, y int32, antialias []Alpha)

	// blitAntiH1 adds one alpha.
	blitAntiH1(x, y int32, alpha Alpha)

	// blitAntiHW adds a constant-alpha run of width pixels.
	blitAntiHW(x, y, width int32, alpha Alpha)

	// flushIfYChanged flushes the additive alpha cache if floor(y) != floor(nextY) (i.e., we'll start working on a new
	// pixel row). Only the RLE blitters need it.
	flushIfYChanged(y, nextY Fixed)

	// trapezoidScratch returns three distinct reusable row buffers of length n for blitAAATrapezoidRow, avoiding a
	// per-scanline allocation (see aaaScratch).
	trapezoidScratch(n int32) (alphas, tempAlphas []Alpha, runs []int16)
}

// aaaScratch holds the per-fill reusable row buffers used by blitAAATrapezoidRow. The analytic-AA walker calls that
// function once per scanline; allocating alphas/tempAlphas/runs fresh each time was the dominant source of per-fill
// heap traffic (pooled temporaries). The buffers live on the additive blitter, which is constructed once per fill — and
// once per band under FillPathParallel, so there is no cross-goroutine sharing — and grow to the widest trapezoid row
// seen.
type aaaScratch struct {
	alphas     []Alpha
	tempAlphas []Alpha
	runs       []int16
}

// trapezoidScratch returns three distinct scratch slices of length n (n = trapezoid width + 1), growing the backing
// arrays as needed. The buffers need not be zeroed: blitAAATrapezoidRow fully writes alphas[0:n] and runs[0:n] before
// use, and writes exactly the tempAlphas indices it later reads (computeAlpha{Above,Below}Line covers precisely the
// read range).
func (s *aaaScratch) trapezoidScratch(n int32) (alphas, tempAlphas []Alpha, runs []int16) {
	if int(n) > cap(s.alphas) {
		s.alphas = make([]Alpha, n)
		s.tempAlphas = make([]Alpha, n)
		s.runs = make([]int16, n)
	}
	return s.alphas[:n], s.tempAlphas[:n], s.runs[:n]
}

///////////////////////////////////////////////////////////////////////////////
// MaskAdditiveBlitter

// Mask blitter limits (so we don't try to do very wide things, where the RLE blitter would be faster).
const (
	maskBlitterMaxWidth   = 32
	maskBlitterMaxStorage = 1024
)

// maskRow is the AAA walkers' handle on the mask blitter's current row: img[x+off] is the accumulated alpha for device
// pixel x. A nil img means no mask is in use.
type maskRow struct {
	img []uint8
	off int
}

// maskAdditiveBlitter accumulates coverage directly into an A8 mask and blits the whole mask to the real blitter once
// at the end. It significantly accelerates small path filling. It also acts as its own "real blitter" (getRealBlitter
// returns itself) so the walkers' rect fast paths write into the mask with overwrite semantics.
type maskAdditiveBlitter struct {
	realBlitter Blitter
	aaaScratch
	// storage backs the mask image with 1 spare byte at either end, because the walkers can write one extra byte at
	// either end due to precision error (Image points 1 byte into storage).
	storage  []uint8
	mask     Mask
	rowOff   int
	clipRect geom.IRect
	rowY     int32
}

// maskBlitterCanHandleRect reports whether bounds is small enough for the mask blitter's fixed storage budget.
func maskBlitterCanHandleRect(bounds geom.IRect) bool {
	width := bounds.Width()
	if width > maskBlitterMaxWidth {
		return false
	}
	rb := int64(width+3) &^ 3
	// use 64 bits to detect overflow
	storage := rb * int64(bounds.Height())
	return storage <= maskBlitterMaxStorage
}

// The additive blitters are constructed once per fill (and once per band under FillPathParallel, so a sync.Pool rather
// than a shared instance keeps concurrent bands independent). They own the widest-row scratch buffers (aaaScratch) and
// their RLE/mask storage; pooling them retains that storage across draws so steady-state fills allocate nothing at the
// blitter layer. Each is fully consumed by the synchronous aaaFillPath before it is returned, so putting it back on
// function exit is safe.
var maskBlitterPool = sync.Pool{New: func() any { return new(maskAdditiveBlitter) }}

func getMaskAdditiveBlitter(realBlitter Blitter, ir, clipBounds geom.IRect) *maskAdditiveBlitter {
	m := maskBlitterPool.Get().(*maskAdditiveBlitter)
	m.init(realBlitter, ir, clipBounds)
	return m
}

func putMaskAdditiveBlitter(m *maskAdditiveBlitter) {
	m.realBlitter = nil // drop the external device-blitter reference; keep storage/scratch for reuse
	maskBlitterPool.Put(m)
}

// init sets up the blitter, reusing the receiver's storage across pooled fills.
func (m *maskAdditiveBlitter) init(realBlitter Blitter, ir, clipBounds geom.IRect) {
	m.realBlitter = realBlitter
	m.clipRect = ir
	m.rowY = ir.Top - 1
	m.rowOff = 0
	if !m.clipRect.Intersect(clipBounds) {
		m.clipRect = geom.IRect{}
	}
	rb := ir.Width()
	// The mask blitter accumulates coverage additively, so the storage must start zeroed. make() zeroes; a reused
	// backing array must be cleared explicitly over the bytes this fill will touch.
	need := int(rb)*int(ir.Height()) + 2
	if need > cap(m.storage) {
		m.storage = make([]uint8, need)
	} else {
		m.storage = m.storage[:need]
		clear(m.storage)
	}
	m.mask = Mask{
		Image:    m.storage[1:],
		Bounds:   ir,
		RowBytes: rb,
	}
}

// done blits the accumulated mask through the real blitter.
func (m *maskAdditiveBlitter) done() {
	if !m.clipRect.IsEmpty() {
		m.realBlitter.BlitMask(&m.mask, m.clipRect)
	}
}

// getRow returns the handle for row y, recomputing the row offset when y changes.
func (m *maskAdditiveBlitter) getRow(y int32) maskRow {
	if y != m.rowY {
		m.rowY = y
		m.rowOff = 1 + int(y-m.mask.Bounds.Top)*int(m.mask.RowBytes) - int(m.mask.Bounds.Left)
	}
	return maskRow{img: m.storage, off: m.rowOff}
}

func (m *maskAdditiveBlitter) getRealBlitter(forceRealBlitter bool) Blitter {
	// Most of the time, we still consider this mask blitter as the real blitter so we can accelerate blitRect and
	// others. But sometimes we want to return the absolute real blitter (e.g., when we fall back to the old code path).
	if forceRealBlitter {
		return m.realBlitter
	}
	return m
}

func (m *maskAdditiveBlitter) blitAntiHRun(_, _ int32, _ []Alpha) {
	panic("raster: MaskAdditiveBlitter.blitAntiHRun should never be called; add alphas to the mask directly")
}

func (m *maskAdditiveBlitter) blitAntiH1(x, y int32, alpha Alpha) {
	row := m.getRow(y)
	addAlpha(&row.img[row.off+int(x)], alpha)
}

func (m *maskAdditiveBlitter) blitAntiHW(x, y, width int32, alpha Alpha) {
	row := m.getRow(y)
	for i := int32(0); i < width; i++ {
		addAlpha(&row.img[row.off+int(x+i)], alpha)
	}
}

func (m *maskAdditiveBlitter) flushIfYChanged(_, _ Fixed) {}

// The Blitter half: these run when the walkers treat the mask blitter as the real blitter, and set (rather than add)
// alpha.

// BlitV implements Blitter.
func (m *maskAdditiveBlitter) BlitV(x, y, height int32, alpha Alpha) {
	if alpha == 0 {
		return
	}
	// The vertical walk steps i by RowBytes rather than calling getRow again, so the row cache is left describing row
	// y — which getRow already made true — and needs no fixup here.
	row := m.getRow(y)
	i := row.off + int(x)
	for h := int32(0); h < height; h++ {
		row.img[i] = alpha
		i += int(m.mask.RowBytes)
	}
}

// BlitRect implements Blitter.
func (m *maskAdditiveBlitter) BlitRect(x, y, width, height int32) {
	row := m.getRow(y)
	i := row.off + int(x)
	for h := int32(0); h < height; h++ {
		for w := 0; w < int(width); w++ {
			row.img[i+w] = 0xFF
		}
		i += int(m.mask.RowBytes)
	}
}

// BlitAntiRect implements Blitter.
func (m *maskAdditiveBlitter) BlitAntiRect(x, y, width, height int32, leftAlpha, rightAlpha Alpha) {
	m.BlitV(x, y, height, leftAlpha)
	m.BlitV(x+1+width, y, height, rightAlpha)
	m.BlitRect(x+1, y, width, height)
}

// BlitH implements Blitter (never called on the mask blitter; the walkers use the mask row instead).
func (m *maskAdditiveBlitter) BlitH(_, _, _ int32) {}

// BlitAntiH implements Blitter (never called on the mask blitter).
func (m *maskAdditiveBlitter) BlitAntiH(_, _ int32, _ []Alpha, _ []int16) {}

// BlitMask implements Blitter (never called on the mask blitter).
func (m *maskAdditiveBlitter) BlitMask(_ *Mask, _ geom.IRect) {}

// BlitAntiH2 implements Blitter (never called on the mask blitter).
func (m *maskAdditiveBlitter) BlitAntiH2(_, _ int32, _, _ Alpha) {}

// BlitAntiV2 implements Blitter (never called on the mask blitter).
func (m *maskAdditiveBlitter) BlitAntiV2(_, _ int32, _, _ Alpha) {}

///////////////////////////////////////////////////////////////////////////////
// RunBasedAdditiveBlitter

// runBasedAdditiveBlitter accumulates each pixel row into an AlphaRuns RLE and flushes it to the real blitter's
// BlitAntiH when the row changes.
type runBasedAdditiveBlitter struct {
	realBlitter Blitter
	aaaScratch
	runs    AlphaRuns
	offsetX int
	currY   int32 // current y coordinate
	width   int32 // widest row of region to be blitted
	left    int32 // leftmost x coordinate in any row
	top     int32 // initial y coordinate (top of bounds)
}

var runBasedBlitterPool = sync.Pool{New: func() any { return new(runBasedAdditiveBlitter) }}

func getRunBasedAdditiveBlitter(realBlitter Blitter, ir, clipBounds geom.IRect, isInverse bool) *runBasedAdditiveBlitter {
	r := runBasedBlitterPool.Get().(*runBasedAdditiveBlitter)
	r.init(realBlitter, ir, clipBounds, isInverse)
	return r
}

func putRunBasedAdditiveBlitter(r *runBasedAdditiveBlitter) {
	r.realBlitter = nil
	runBasedBlitterPool.Put(r)
}

// init sets up the blitter, reusing the receiver's runs storage across pooled fills. All this package's real blitters
// preserve one row at a time, so a single runs buffer suffices (no circular buffer needed). The runs buffer is
// re-established by advanceRuns (AlphaRuns.Reset), so a grown-but-not-zeroed backing array is fine — the sparse RLE
// structure is self-describing from index 0.
func (r *runBasedAdditiveBlitter) init(realBlitter Blitter, ir, clipBounds geom.IRect, isInverse bool) {
	var sectBounds geom.IRect
	if isInverse {
		// We use the clip bounds instead of the ir, since we may be asked to draw outside of the rect when we're an
		// inverse fill type.
		sectBounds = clipBounds
	} else {
		sectBounds = ir
		if !sectBounds.Intersect(clipBounds) {
			sectBounds = geom.IRect{}
		}
	}
	r.realBlitter = realBlitter
	r.left = sectBounds.Left
	r.width = sectBounds.Width()
	r.top = sectBounds.Top
	r.currY = sectBounds.Top - 1
	r.offsetX = 0
	if n := int(r.width) + 1; n > cap(r.runs.Runs) {
		r.runs.Runs = make([]int16, n)
	} else {
		r.runs.Runs = r.runs.Runs[:n]
	}
	if n := int(r.width) + 2; n > cap(r.runs.Alpha) {
		r.runs.Alpha = make([]Alpha, n)
	} else {
		r.runs.Alpha = r.runs.Alpha[:n]
	}
	r.advanceRuns()
}

func (r *runBasedAdditiveBlitter) getRealBlitter(_ bool) Blitter {
	return r.realBlitter
}

func (r *runBasedAdditiveBlitter) advanceRuns() {
	r.runs.Reset(int(r.width))
}

func (r *runBasedAdditiveBlitter) check(x, width int32) bool {
	return x >= 0 && x+width <= r.width
}

// snapAlpha blits 0xFF and 0 much faster than other values, so alphas close to them are snapped.
func snapAlpha(alpha Alpha) Alpha {
	switch {
	case alpha > 247:
		return 0xFF
	case alpha < 8:
		return 0
	default:
		return alpha
	}
}

func (r *runBasedAdditiveBlitter) flush() {
	if r.currY >= r.top {
		for x := 0; r.runs.Runs[x] != 0; x += int(r.runs.Runs[x]) {
			// It seems that blitting 255 or 0 is much faster than blitting 254 or 1.
			r.runs.Alpha[x] = snapAlpha(r.runs.Alpha[x])
		}
		if !r.runs.Empty() {
			r.realBlitter.BlitAntiH(r.left, r.currY, r.runs.Alpha, r.runs.Runs)
			r.advanceRuns()
			r.offsetX = 0
		}
		r.currY = r.top - 1
	}
}

func (r *runBasedAdditiveBlitter) checkY(y int32) {
	if y != r.currY {
		r.flush()
		r.currY = y
	}
}

func (r *runBasedAdditiveBlitter) blitAntiHRun(x, y int32, antialias []Alpha) {
	r.checkY(y)
	x -= r.left

	length := int32(len(antialias))
	if x < 0 {
		// Clamp before reslicing. A run lying entirely left of r.left by more than its own length makes -x exceed
		// len(antialias), so the reslice would panic; Skia does the same step as pointer arithmetic, where an
		// out-of-bounds pointer is harmless because the negative length is caught next.
		if length += x; length <= 0 {
			return
		}
		antialias = antialias[-x:]
		x = 0
	}
	length = min(length, r.width-x)
	if length <= 0 { // nothing sensible can be added
		return
	}

	if int(x) < r.offsetX {
		r.offsetX = 0
	}

	r.offsetX = r.runs.Add(int(x), 0, int(length), 0, 0, r.offsetX) // break the run
	for i := int32(0); i < length; i += int32(r.runs.Runs[x+i]) {
		for j := int32(1); j < int32(r.runs.Runs[x+i]); j++ {
			r.runs.Runs[x+i+j] = 1
			r.runs.Alpha[x+i+j] = r.runs.Alpha[x+i]
		}
		r.runs.Runs[x+i] = 1
	}
	for i := int32(0); i < length; i++ {
		addAlpha(&r.runs.Alpha[x+i], antialias[i])
	}
}

func (r *runBasedAdditiveBlitter) blitAntiH1(x, y int32, alpha Alpha) {
	r.checkY(y)
	x -= r.left

	if int(x) < r.offsetX {
		r.offsetX = 0
	}

	if r.check(x, 1) {
		r.offsetX = r.runs.Add(int(x), 0, 1, 0, alpha, r.offsetX)
	}
}

func (r *runBasedAdditiveBlitter) blitAntiHW(x, y, width int32, alpha Alpha) {
	r.checkY(y)
	x -= r.left

	if int(x) < r.offsetX {
		r.offsetX = 0
	}

	if r.check(x, width) {
		r.offsetX = r.runs.Add(int(x), 0, int(width), 0, alpha, r.offsetX)
	}
}

func (r *runBasedAdditiveBlitter) flushIfYChanged(y, nextY Fixed) {
	if FixedFloorToInt(y) != FixedFloorToInt(nextY) {
		r.flush()
	}
}

///////////////////////////////////////////////////////////////////////////////
// SafeRLEAdditiveBlitter

// safeRLEAdditiveBlitter exists specifically for concave path filling, where we can easily accumulate alpha greater
// than 0xFF; it clamps at a cost of performance.
type safeRLEAdditiveBlitter struct {
	runBasedAdditiveBlitter
}

var safeRLEBlitterPool = sync.Pool{New: func() any { return new(safeRLEAdditiveBlitter) }}

func getSafeRLEAdditiveBlitter(realBlitter Blitter, ir, clipBounds geom.IRect, isInverse bool) *safeRLEAdditiveBlitter {
	r := safeRLEBlitterPool.Get().(*safeRLEAdditiveBlitter)
	r.init(realBlitter, ir, clipBounds, isInverse)
	return r
}

func putSafeRLEAdditiveBlitter(r *safeRLEAdditiveBlitter) {
	r.realBlitter = nil
	safeRLEBlitterPool.Put(r)
}

func (r *safeRLEAdditiveBlitter) blitAntiHRun(x, y int32, antialias []Alpha) {
	r.checkY(y)
	x -= r.left

	length := int32(len(antialias))
	if x < 0 {
		// Clamp before reslicing, for the same reason as runBasedAdditiveBlitter.blitAntiHRun above.
		if length += x; length <= 0 {
			return
		}
		antialias = antialias[-x:]
		x = 0
	}
	length = min(length, r.width-x)
	if length <= 0 {
		return
	}

	if int(x) < r.offsetX {
		r.offsetX = 0
	}

	r.offsetX = r.runs.Add(int(x), 0, int(length), 0, 0, r.offsetX) // break the run
	for i := int32(0); i < length; i += int32(r.runs.Runs[x+i]) {
		for j := int32(1); j < int32(r.runs.Runs[x+i]); j++ {
			r.runs.Runs[x+i+j] = 1
			r.runs.Alpha[x+i+j] = r.runs.Alpha[x+i]
		}
		r.runs.Runs[x+i] = 1
	}
	for i := int32(0); i < length; i++ {
		safelyAddAlpha(&r.runs.Alpha[x+i], antialias[i])
	}
}

func (r *safeRLEAdditiveBlitter) blitAntiH1(x, y int32, alpha Alpha) {
	r.checkY(y)
	x -= r.left

	if int(x) < r.offsetX {
		r.offsetX = 0
	}

	if r.check(x, 1) {
		// break the run
		r.offsetX = r.runs.Add(int(x), 0, 1, 0, 0, r.offsetX)
		safelyAddAlpha(&r.runs.Alpha[x], alpha)
	}
}

func (r *safeRLEAdditiveBlitter) blitAntiHW(x, y, width int32, alpha Alpha) {
	r.checkY(y)
	x -= r.left

	if int(x) < r.offsetX {
		r.offsetX = 0
	}

	if r.check(x, width) {
		// break the run
		r.offsetX = r.runs.Add(int(x), 0, int(width), 0, 0, r.offsetX)
		for i := x; i < x+width; i += int32(r.runs.Runs[i]) {
			safelyAddAlpha(&r.runs.Alpha[i], alpha)
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Coverage math

// trapezoidToAlpha returns the alpha of a trapezoid whose height is 1.
func trapezoidToAlpha(l1, l2 Fixed) Alpha {
	area := (l1 + l2) / 2
	return Alpha(area >> 8)
}

// partialTriangleToAlpha returns the alpha of the right triangle (a, a*b), coarsely approximated.
func partialTriangleToAlpha(a, b Fixed) Alpha {
	area := (a >> 11) * (a >> 11) * (b >> 11)
	return Alpha((area >> 8) & 0xFF)
}

// getPartialAlphaFixed scales alpha by partialHeight (a Fixed fraction in [0, 1]).
func getPartialAlphaFixed(alpha Alpha, partialHeight Fixed) Alpha {
	return Alpha(FixedRoundToInt(Fixed(int32(alpha) * int32(partialHeight))))
}

// getPartialAlphaMul scales alpha by fullAlpha (both 0..255).
func getPartialAlphaMul(alpha, fullAlpha Alpha) Alpha {
	return Alpha((uint32(alpha) * uint32(fullAlpha)) >> 8)
}

// fixedToAlpha converts f to an Alpha; for Fixed close to FixedOne we can't just shift right (that would map FixedOne
// to 256, not 255); rarely a problem, so it's only used for blitting rectangles.
func fixedToAlpha(f Fixed) Alpha {
	return getPartialAlphaFixed(0xFF, f)
}

// approximateIntersection: suppose line (l1, y)-(r1, y+1) intersects with (l2, y)-(r2, y+1); approximate (very
// coarsely) the x coordinate of the intersection.
func approximateIntersection(l1, r1, l2, r2 Fixed) Fixed {
	if l1 > r1 {
		l1, r1 = r1, l1
	}
	if l2 > r2 {
		l2, r2 = r2, l2
	}
	return (max(l1, l2) + min(r1, r2)) / 2
}

// computeAlphaAboveLine computes the coverage above a line for a run of pixels: l < FixedOne always, and the first
// alpha to compute is alphas[0].
func computeAlphaAboveLine(alphas []Alpha, l, r, dY Fixed, fullAlpha Alpha) {
	upper := FixedCeilToInt(r)
	if upper == 0 {
		return
	}
	if upper == 1 {
		alphas[0] = getPartialAlphaMul(Alpha(((upper<<17)-int32(l)-int32(r))>>9), fullAlpha)
		return
	}
	first := FixedOne - l                            // horizontal edge length of the left-most triangle
	last := r - Fixed(upper-1)<<16                   // horizontal edge length of the right-most triangle
	firstH := FixedMul(first, dY)                    // vertical edge of the left-most triangle
	alphas[0] = Alpha(FixedMul(first, firstH) >> 9)  // triangle alpha
	alpha16 := sat32Add(int32(firstH), int32(dY>>1)) // rectangle plus triangle
	for i := int32(1); i < upper-1; i++ {
		alphas[i] = Alpha(alpha16 >> 8)
		alpha16 = sat32Add(alpha16, int32(dY))
	}
	alphas[upper-1] = fullAlpha - partialTriangleToAlpha(last, dY)
}

// computeAlphaBelowLine computes the coverage below a line for a run of pixels.
func computeAlphaBelowLine(alphas []Alpha, l, r, dY Fixed, fullAlpha Alpha) {
	upper := FixedCeilToInt(r)
	if upper == 0 {
		return
	}
	if upper == 1 {
		alphas[0] = getPartialAlphaMul(trapezoidToAlpha(l, r), fullAlpha)
		return
	}
	first := FixedOne - l
	last := r - Fixed(upper-1)<<16
	lastH := FixedMul(last, dY)                         // vertical edge of the right-most triangle
	alphas[upper-1] = Alpha(FixedMul(last, lastH) >> 9) // triangle alpha
	alpha16 := sat32Add(int32(lastH), int32(dY>>1))     // rectangle plus triangle
	for i := upper - 2; i > 0; i-- {
		alphas[i] = Alpha((alpha16 >> 8) & 0xFF)
		alpha16 = sat32Add(alpha16, int32(dY))
	}
	alphas[0] = fullAlpha - partialTriangleToAlpha(first, dY)
}

///////////////////////////////////////////////////////////////////////////////
// Trapezoid blitting

// blitSingleAlpha blits one pixel's alpha. If fullAlpha != 0xFF, alpha is scaled by fullAlpha.
func blitSingleAlpha(blitter additiveBlitter, y, x int32, alpha, fullAlpha Alpha, mrow maskRow, noRealBlitter bool) {
	if mrow.img != nil {
		if fullAlpha == 0xFF && !noRealBlitter { // noRealBlitter is needed for concave paths
			mrow.img[mrow.off+int(x)] = alpha
		} else {
			safelyAddAlpha(&mrow.img[mrow.off+int(x)], getPartialAlphaMul(alpha, fullAlpha))
		}
		return
	}
	if fullAlpha == 0xFF && !noRealBlitter {
		blitter.getRealBlitter(false).BlitV(x, y, 1, alpha)
	} else {
		blitter.blitAntiH1(x, y, getPartialAlphaMul(alpha, fullAlpha))
	}
}

// blitTwoAlphas blits two adjacent pixels' alphas.
func blitTwoAlphas(blitter additiveBlitter, y, x int32, a1, a2, fullAlpha Alpha, mrow maskRow, noRealBlitter bool) {
	if mrow.img != nil {
		safelyAddAlpha(&mrow.img[mrow.off+int(x)], a1)
		safelyAddAlpha(&mrow.img[mrow.off+int(x)+1], a2)
		return
	}
	if fullAlpha == 0xFF && !noRealBlitter {
		blitter.getRealBlitter(false).BlitAntiH2(x, y, a1, a2)
	} else {
		blitter.blitAntiH1(x, y, a1)
		blitter.blitAntiH1(x+1, y, a2)
	}
}

// blitFullAlpha blits a run of length pixels all at fullAlpha.
func blitFullAlpha(blitter additiveBlitter, y, x, length int32, fullAlpha Alpha, mrow maskRow, noRealBlitter bool) {
	if mrow.img != nil {
		for i := int32(0); i < length; i++ {
			safelyAddAlpha(&mrow.img[mrow.off+int(x+i)], fullAlpha)
		}
		return
	}
	if fullAlpha == 0xFF && !noRealBlitter {
		blitter.getRealBlitter(false).BlitH(x, y, length)
	} else {
		blitter.blitAntiHW(x, y, length, fullAlpha)
	}
}

// blitAAATrapezoidRow blits one scanline's worth of a trapezoid, computing per-pixel coverage from the top edge (ul,
// ur), bottom edge (ll, lr), and the left/right edges' abs(dy/dx) (lDY, rDY).
func blitAAATrapezoidRow(blitter additiveBlitter, y int32, ul, ur, ll, lr, lDY, rDY Fixed, fullAlpha Alpha, mrow maskRow, noRealBlitter bool) {
	left := FixedFloorToInt(ul)
	rite := FixedCeilToInt(lr)
	length := rite - left

	if length == 1 {
		alpha := trapezoidToAlpha(ur-ul, lr-ll)
		blitSingleAlpha(blitter, y, left, alpha, fullAlpha, mrow, noRealBlitter)
		return
	}

	alphas, tempAlphas, runs := blitter.trapezoidScratch(length + 1)

	for i := int32(0); i < length; i++ {
		runs[i] = 1
		alphas[i] = fullAlpha
	}
	runs[length] = 0
	alphas[length] = 0 // the init loop leaves [length] untouched; the original make() zeroed it

	uL := FixedFloorToInt(ul)
	lL := FixedCeilToInt(ll)
	if uL+2 == lL { // We only need to compute two triangles, accelerate this special case
		first := Fixed(uL)<<16 + FixedOne - ul
		second := ll - ul - first
		a1 := fullAlpha - partialTriangleToAlpha(first, lDY)
		a2 := partialTriangleToAlpha(second, lDY)
		if alphas[0] > a1 {
			alphas[0] -= a1
		} else {
			alphas[0] = 0
		}
		if alphas[1] > a2 {
			alphas[1] -= a2
		} else {
			alphas[1] = 0
		}
	} else {
		computeAlphaBelowLine(tempAlphas[uL-left:], ul-Fixed(uL)<<16, ll-Fixed(uL)<<16, lDY, fullAlpha)
		for i := uL; i < lL; i++ {
			if alphas[i-left] > tempAlphas[i-left] {
				alphas[i-left] -= tempAlphas[i-left]
			} else {
				alphas[i-left] = 0
			}
		}
	}

	uR := FixedFloorToInt(ur)
	lR := FixedCeilToInt(lr)
	if uR+2 == lR { // We only need to compute two triangles, accelerate this special case
		first := Fixed(uR)<<16 + FixedOne - ur
		second := lr - ur - first
		a1 := partialTriangleToAlpha(first, rDY)
		a2 := fullAlpha - partialTriangleToAlpha(second, rDY)
		if alphas[length-2] > a1 {
			alphas[length-2] -= a1
		} else {
			alphas[length-2] = 0
		}
		if alphas[length-1] > a2 {
			alphas[length-1] -= a2
		} else {
			alphas[length-1] = 0
		}
	} else {
		computeAlphaAboveLine(tempAlphas[uR-left:], ur-Fixed(uR)<<16, lr-Fixed(uR)<<16, rDY, fullAlpha)
		for i := uR; i < lR; i++ {
			if alphas[i-left] > tempAlphas[i-left] {
				alphas[i-left] -= tempAlphas[i-left]
			} else {
				alphas[i-left] = 0
			}
		}
	}

	if mrow.img != nil {
		for i := int32(0); i < length; i++ {
			safelyAddAlpha(&mrow.img[mrow.off+int(left+i)], alphas[i])
		}
	} else {
		if fullAlpha == 0xFF && !noRealBlitter {
			// Real blitter is faster than RunBasedAdditiveBlitter
			blitter.getRealBlitter(false).BlitAntiH(left, y, alphas, runs)
		} else {
			blitter.blitAntiHRun(left, y, alphas[:length])
		}
	}
}

// blitTrapezoidRow blits the trapezoid with top line (ul, ur) and bottom line (ll, lr), lDY and rDY the abs(dy/dx) of
// the left and right edges.
func blitTrapezoidRow(blitter additiveBlitter, y int32, ul, ur, ll, lr, lDY, rDY Fixed, fullAlpha Alpha, mrow maskRow, noRealBlitter bool) {
	if ul > ur {
		return
	}

	// Edge crosses. Approximate it. This should only happen due to precision limit, so the approximation could be very
	// coarse.
	if ll > lr {
		ll = approximateIntersection(ul, ll, ur, lr)
		lr = ll
	}

	if ul == ur && ll == lr {
		return // empty trapezoid
	}

	// We're going to use the left line ul-ll and the rite line ur-lr to exclude the area that's not covered by the
	// path. Swapping (ul, ll) or (ur, lr) won't affect that exclusion, so we'll do that for simplicity.
	if ul > ll {
		ul, ll = ll, ul
	}
	if ur > lr {
		ur, lr = lr, ur
	}

	joinLeft := FixedCeilToFixed(ll)
	joinRite := FixedFloorToFixed(ur)
	if joinLeft <= joinRite { // There's a rect from joinLeft to joinRite that we can blit
		if ul < joinLeft {
			length := FixedCeilToInt(joinLeft - ul)
			switch length {
			case 1:
				alpha := trapezoidToAlpha(joinLeft-ul, joinLeft-ll)
				blitSingleAlpha(blitter, y, int32(ul>>16), alpha, fullAlpha, mrow, noRealBlitter)
			case 2:
				first := joinLeft - FixedOne - ul
				second := ll - ul - first
				a1 := partialTriangleToAlpha(first, lDY)
				a2 := fullAlpha - partialTriangleToAlpha(second, lDY)
				blitTwoAlphas(blitter, y, int32(ul>>16), a1, a2, fullAlpha, mrow, noRealBlitter)
			default:
				blitAAATrapezoidRow(blitter, y, ul, joinLeft, ll, joinLeft, lDY, math.MaxInt32,
					fullAlpha, mrow, noRealBlitter)
			}
		}
		// The AA clip requires that we blit from left to right. Hence we must blit [ul, joinLeft] before blitting
		// [joinLeft, joinRite].
		if joinLeft < joinRite {
			blitFullAlpha(blitter, y, FixedFloorToInt(joinLeft), FixedFloorToInt(joinRite-joinLeft),
				fullAlpha, mrow, noRealBlitter)
		}
		if lr > joinRite {
			length := FixedCeilToInt(lr - joinRite)
			switch length {
			case 1:
				alpha := trapezoidToAlpha(ur-joinRite, lr-joinRite)
				blitSingleAlpha(blitter, y, int32(joinRite>>16), alpha, fullAlpha, mrow, noRealBlitter)
			case 2:
				first := joinRite + FixedOne - ur
				second := lr - ur - first
				a1 := fullAlpha - partialTriangleToAlpha(first, rDY)
				a2 := partialTriangleToAlpha(second, rDY)
				blitTwoAlphas(blitter, y, int32(joinRite>>16), a1, a2, fullAlpha, mrow, noRealBlitter)
			default:
				blitAAATrapezoidRow(blitter, y, joinRite, ur, joinRite, lr, math.MaxInt32, rDY,
					fullAlpha, mrow, noRealBlitter)
			}
		}
	} else {
		blitAAATrapezoidRow(blitter, y, ul, ur, ll, lr, lDY, rDY, fullAlpha, mrow, noRealBlitter)
	}
}

///////////////////////////////////////////////////////////////////////////////
// Edge list plumbing

func removeAnalyticEdge(edge *AnalyticEdge) {
	edge.Prev.Next = edge.Next
	edge.Next.Prev = edge.Prev
}

func insertAnalyticEdgeAfter(edge, afterMe *AnalyticEdge) {
	edge.Prev = afterMe
	edge.Next = afterMe.Next
	afterMe.Next.Prev = edge
	afterMe.Next = edge
}

func backwardInsertAnalyticEdgeBasedOnX(edge *AnalyticEdge) {
	x := edge.X
	prev := edge.Prev
	for prev.Prev != nil && prev.X > x {
		prev = prev.Prev
	}
	if prev.Next != edge {
		removeAnalyticEdge(edge)
		insertAnalyticEdgeAfter(edge, prev)
	}
}

func backwardInsertStartAnalytic(prev *AnalyticEdge, x Fixed) *AnalyticEdge {
	for prev.Prev != nil && prev.X > x {
		prev = prev.Prev
	}
	return prev
}

// sortAnalyticEdges sorts by (UpperY, X, DX) and links the list, returning first and last. slices.SortFunc avoids
// sort.Slice's reflect.Swapper heap allocation.
func sortAnalyticEdges(list []*AnalyticEdge) (first, last *AnalyticEdge) {
	slices.SortFunc(list, func(a, b *AnalyticEdge) int {
		if a.UpperY != b.UpperY {
			return cmp.Compare(a.UpperY, b.UpperY)
		}
		if a.X != b.X {
			return cmp.Compare(a.X, b.X)
		}
		return cmp.Compare(a.DX, b.DX)
	})
	for i := 1; i < len(list); i++ {
		list[i-1].Next = list[i]
		list[i].Prev = list[i-1]
	}
	return list[0], list[len(list)-1]
}

///////////////////////////////////////////////////////////////////////////////
// Convex walker

// isSmoothEnoughEdge reports whether an edge is smooth: Dx doesn't change much and Dy is large enough.
func isSmoothEnoughEdge(thisEdge, nextEdge *AnalyticEdge) bool {
	if thisEdge.CurveCount < 0 {
		cEdge := thisEdge.curve.(*AnalyticCubicEdge)
		ddshift := int(cEdge.CurveShift)
		return FixedAbs(cEdge.cDx)>>1 >= FixedAbs(cEdge.cDDx)>>ddshift &&
			FixedAbs(cEdge.cDy)>>1 >= FixedAbs(cEdge.cDDy)>>ddshift &&
			// current Dy is (cDy - (cDDy >> ddshift)) >> dshift
			(cEdge.cDy-(cEdge.cDDy>>ddshift))>>int(cEdge.cubicDShift) >= FixedOne
	} else if thisEdge.CurveCount > 0 {
		qEdge := thisEdge.curve.(*AnalyticQuadraticEdge)
		return FixedAbs(qEdge.qDx)>>1 >= FixedAbs(qEdge.qDDx) &&
			FixedAbs(qEdge.qDy)>>1 >= FixedAbs(qEdge.qDDy) &&
			// current Dy is (qDy - qDDy) >> shift
			(qEdge.qDy-qEdge.qDDy)>>int(qEdge.CurveShift) >= FixedOne
	}
	// DDx should be small and Dy should be large
	return FixedAbs(Fixed(sat32Sub(int32(nextEdge.DX), int32(thisEdge.DX)))) <= FixedOne &&
		nextEdge.LowerY-nextEdge.UpperY >= FixedOne
}

// isSmoothEnough checks if leftE and riteE are changing smoothly in terms of DX; if yes, we can later skip the
// fractional y and directly jump to integer y.
func isSmoothEnough(leftE, riteE, currE *AnalyticEdge, stopY int32) bool {
	if currE.UpperY >= Fixed(stopY)<<16 {
		return false // We're at the end so we won't skip anything
	}
	if leftE.LowerY+FixedOne < riteE.LowerY {
		return isSmoothEnoughEdge(leftE, currE) // Only leftE is changing
	} else if leftE.LowerY > riteE.LowerY+FixedOne {
		return isSmoothEnoughEdge(riteE, currE) // Only riteE is changing
	}

	// Now both edges are changing, find the second next edge
	nextCurrE := currE.Next
	if nextCurrE.UpperY >= Fixed(stopY)<<16 { // Check if we're at the end
		return false
	}
	// Ensure that currE is the next left edge and nextCurrE is the next right edge. Swap if not.
	if nextCurrE.UpperX < currE.UpperX {
		currE, nextCurrE = nextCurrE, currE
	}
	return isSmoothEnoughEdge(leftE, currE) && isSmoothEnoughEdge(riteE, nextCurrE)
}

// aaaWalkConvexEdges is the two-edge (left/right) analytic-AA walker for convex fills. maskB is non-nil when the mask
// additive blitter is in use.
func aaaWalkConvexEdges(prevHead *AnalyticEdge, blitter additiveBlitter, maskB *maskAdditiveBlitter, stopY int32, leftBound, riteBound Fixed) {
	leftE := prevHead.Next
	riteE := leftE.Next
	currE := riteE.Next

	y := max(leftE.UpperY, riteE.UpperY)

walk:
	for {
		// We have to check LowerY first because some edges might be alone (e.g., there's only a left edge but no right
		// edge in a given y scan line) due to the precision limit.
		for leftE.LowerY <= y { // Due to smooth jump, we may pass multiple short edges
			if !leftE.update() {
				if FixedFloorToInt(currE.UpperY) >= stopY {
					break walk
				}
				leftE = currE
				currE = currE.Next
			}
		}
		for riteE.LowerY <= y { // Due to smooth jump, we may pass multiple short edges
			if !riteE.update() {
				if FixedFloorToInt(currE.UpperY) >= stopY {
					break walk
				}
				riteE = currE
				currE = currE.Next
			}
		}

		// check our bottom clip
		if FixedFloorToInt(y) >= stopY {
			break
		}

		leftE.goY(y)
		riteE.goY(y)

		if leftE.X > riteE.X || (leftE.X == riteE.X && leftE.DX > riteE.DX) {
			leftE, riteE = riteE, leftE
		}

		localBotFixed := min(leftE.LowerY, riteE.LowerY)
		if isSmoothEnough(leftE, riteE, currE, stopY) {
			localBotFixed = FixedCeilToFixed(localBotFixed)
		}
		localBotFixed = min(localBotFixed, Fixed(stopY)<<16)

		left := max(leftBound, leftE.X)
		dLeft := leftE.DX
		rite := min(riteBound, riteE.X)
		dRite := riteE.DX
		if dLeft|dRite == 0 {
			fullLeft := FixedCeilToInt(left)
			fullRite := FixedFloorToInt(rite)
			partialLeft := Fixed(fullLeft)<<16 - left
			partialRite := rite - Fixed(fullRite)<<16
			fullTop := FixedCeilToInt(y)
			fullBot := FixedFloorToInt(localBotFixed)
			partialTop := Fixed(fullTop)<<16 - y
			partialBot := localBotFixed - Fixed(fullBot)<<16
			if fullTop > fullBot { // The rectangle is within one pixel height...
				partialTop -= FixedOne - partialBot
				partialBot = 0
			}

			if fullRite >= fullLeft {
				if partialTop > 0 { // blit first partial row
					if partialLeft > 0 {
						blitter.blitAntiH1(fullLeft-1, fullTop-1,
							fixedToAlpha(FixedMul(partialTop, partialLeft)))
					}
					blitter.blitAntiHW(fullLeft, fullTop-1, fullRite-fullLeft, fixedToAlpha(partialTop))
					if partialRite > 0 {
						blitter.blitAntiH1(fullRite, fullTop-1,
							fixedToAlpha(FixedMul(partialTop, partialRite)))
					}
					blitter.flushIfYChanged(y, y+partialTop)
				}

				// Blit all full-height rows from fullTop to fullBot.
				if fullBot > fullTop &&
					// An empty rect must not reach the blitter, so check the non-emptiness here (bug chromium:662800)
					(fullRite > fullLeft || fixedToAlpha(partialLeft) > 0 ||
						fixedToAlpha(partialRite) > 0) {
					blitter.getRealBlitter(false).BlitAntiRect(fullLeft-1, fullTop,
						fullRite-fullLeft, fullBot-fullTop,
						fixedToAlpha(partialLeft), fixedToAlpha(partialRite))
				}

				if partialBot > 0 { // blit last partial row
					if partialLeft > 0 {
						blitter.blitAntiH1(fullLeft-1, fullBot,
							fixedToAlpha(FixedMul(partialBot, partialLeft)))
					}
					blitter.blitAntiHW(fullLeft, fullBot, fullRite-fullLeft, fixedToAlpha(partialBot))
					if partialRite > 0 {
						blitter.blitAntiH1(fullRite, fullBot,
							fixedToAlpha(FixedMul(partialBot, partialRite)))
					}
				}
			} else {
				// Normal conditions, this means left and rite are within the same pixel, but if both left and rite were
				// < leftBounds or > rightBounds, both edges are clipped and we should not do any blitting (particularly
				// since the negative width saturates to full alpha).
				width := rite - left
				if width > 0 {
					if partialTop > 0 {
						blitter.blitAntiHW(fullLeft-1, fullTop-1, 1,
							fixedToAlpha(FixedMul(partialTop, width)))
						blitter.flushIfYChanged(y, y+partialTop)
					}
					if fullBot > fullTop {
						blitter.getRealBlitter(false).BlitV(fullLeft-1, fullTop, fullBot-fullTop,
							fixedToAlpha(width))
					}
					if partialBot > 0 {
						blitter.blitAntiHW(fullLeft-1, fullBot, 1,
							fixedToAlpha(FixedMul(partialBot, width)))
					}
				}
			}

			y = localBotFixed
		} else {
			// The following constants are used to snap X. We snap X mainly for speedup (no tiny triangle) and to avoid
			// edge cases caused by precision errors.
			const kSnapDigit = FixedOne >> 4
			const kSnapHalf = kSnapDigit >> 1
			const kSnapMask = -1 ^ (kSnapDigit - 1)
			left += kSnapHalf
			rite += kSnapHalf // For fast rounding

			// Number of blit_trapezoid_row calls we'll have
			count := FixedCeilToInt(localBotFixed) - FixedFloorToInt(y)

			// If we're using the mask blitter, we advance the mask row in this function to save some "if" condition
			// checks.
			var mrow maskRow
			if maskB != nil {
				mrow = maskB.getRow(int32(y >> 16))
			}

			// Instead of writing one loop that handles both partial-row blit_trapezoid_row and full-row trapezoid_row
			// together, we use the following 3-stage flow to handle partial-row blit and full-row blit separately. It
			// will save us much time on changing y, left, and rite.
			if count > 1 {
				if y&^Fixed(0xFFFF) != y { // There's a partial-row on the top
					count--
					nextY := FixedCeilToFixed(y + 1)
					dY := nextY - y
					nextLeft := left + FixedMul(dLeft, dY)
					nextRite := rite + FixedMul(dRite, dY)
					blitTrapezoidRow(blitter, int32(y>>16), left&kSnapMask, rite&kSnapMask,
						nextLeft&kSnapMask, nextRite&kSnapMask, leftE.DY, riteE.DY,
						getPartialAlphaFixed(0xFF, dY), mrow, false)
					blitter.flushIfYChanged(y, nextY)
					left = nextLeft
					rite = nextRite
					y = nextY
				}

				for count > 1 { // Full rows in the middle
					count--
					if maskB != nil {
						mrow = maskB.getRow(int32(y >> 16))
					}
					nextY := y + FixedOne
					nextLeft := left + dLeft
					nextRite := rite + dRite
					blitTrapezoidRow(blitter, int32(y>>16), left&kSnapMask, rite&kSnapMask,
						nextLeft&kSnapMask, nextRite&kSnapMask, leftE.DY, riteE.DY,
						0xFF, mrow, false)
					blitter.flushIfYChanged(y, nextY)
					left = nextLeft
					rite = nextRite
					y = nextY
				}
			}

			if maskB != nil {
				mrow = maskB.getRow(int32(y >> 16))
			}

			dY := localBotFixed - y // partial-row on the bottom
			// Smooth jumping to integer y may make the last nextLeft/nextRite out of bound. Take them back into the
			// bound here. Note that we subtract kSnapHalf later, so we have to add it to leftBound/riteBound.
			nextLeft := max(left+FixedMul(dLeft, dY), leftBound+kSnapHalf)
			nextRite := min(rite+FixedMul(dRite, dY), riteBound+kSnapHalf)
			blitTrapezoidRow(blitter, int32(y>>16), left&kSnapMask, rite&kSnapMask,
				nextLeft&kSnapMask, nextRite&kSnapMask, leftE.DY, riteE.DY,
				getPartialAlphaFixed(0xFF, dY), mrow, false)
			blitter.flushIfYChanged(y, localBotFixed)
			left = nextLeft
			rite = nextRite
			y = localBotFixed
			left -= kSnapHalf
			rite -= kSnapHalf
		}

		leftE.X = left
		riteE.X = rite
		leftE.Y = y
		riteE.Y = y
	}
}

///////////////////////////////////////////////////////////////////////////////
// General (winding/even-odd) walker

func updateNextNextY(y, nextY Fixed, nextNextY *Fixed) {
	if y > nextY && y < *nextNextY {
		*nextNextY = y
	}
}

func checkIntersection(edge *AnalyticEdge, nextY Fixed, nextNextY *Fixed) {
	if edge.Prev.Prev != nil && edge.Prev.X+edge.Prev.DX > edge.X+edge.DX {
		*nextNextY = nextY + (FixedOne >> analyticAccuracy)
	}
}

func checkIntersectionFwd(edge *AnalyticEdge, nextY Fixed, nextNextY *Fixed) {
	if edge.Next.Next != nil && edge.X+edge.DX > edge.Next.X+edge.Next.DX {
		*nextNextY = nextY + (FixedOne >> analyticAccuracy)
	}
}

// insertNewAnalyticEdges inserts all edges starting at or before y into x-sorted position.
func insertNewAnalyticEdges(newEdge *AnalyticEdge, y Fixed, nextNextY *Fixed) {
	if newEdge.UpperY > y {
		updateNextNextY(newEdge.UpperY, y, nextNextY)
		return
	}
	prev := newEdge.Prev
	if prev.X <= newEdge.X {
		for newEdge.UpperY <= y {
			checkIntersection(newEdge, y, nextNextY)
			updateNextNextY(newEdge.LowerY, y, nextNextY)
			newEdge = newEdge.Next
		}
		updateNextNextY(newEdge.UpperY, y, nextNextY)
		return
	}
	// find first x pos to insert
	start := backwardInsertStartAnalytic(prev, newEdge.X)
	// insert the lot, fixing up the links as we go
	for {
		next := newEdge.Next
		inserted := false
		for {
			if start.Next == newEdge {
				inserted = true
				break
			}
			after := start.Next
			if after.X >= newEdge.X {
				break
			}
			start = after
		}
		if !inserted {
			removeAnalyticEdge(newEdge)
			insertAnalyticEdgeAfter(newEdge, start)
		}
		checkIntersection(newEdge, y, nextNextY)
		checkIntersectionFwd(newEdge, y, nextNextY)
		updateNextNextY(newEdge.LowerY, y, nextNextY)
		start = newEdge
		newEdge = next
		if newEdge.UpperY > y {
			break
		}
	}
	updateNextNextY(newEdge.UpperY, y, nextNextY)
}

// edgesTooClose reports whether prev.X and next.X are too close in the current pixel row.
func edgesTooClose(prev, next *AnalyticEdge, lowerY Fixed) bool {
	// When next.DX == 0, prev.X >= next.X - abs(next.DX) would be false even if prev.X and next.X are close and within
	// one pixel (e.g., prev.X == 0.1, next.X == 0.9). Adding SLACK = 1 guarantees it to be true if the two edges are
	// within one pixel.
	const slack = FixedOne

	// Note that even if the following test failed, the edges might still be very close to each other at some point
	// within the current pixel row because of prev.DX and next.DX (handling that would sacrifice performance; the
	// current quality is deemed good enough).
	return next != nil && prev != nil && next.UpperY < lowerY &&
		prev.X+slack >= next.X-FixedAbs(next.DX)
}

// edgesTooCloseRite is edgesTooClose's counterpart for the case where the previous rite edge was removed because its
// LowerY <= nextY.
func edgesTooCloseRite(prevRite int32, ul, ll Fixed) bool {
	return prevRite > FixedFloorToInt(ul) || prevRite > FixedFloorToInt(ll)
}

// aaaWalkEdges is the general (winding/even-odd) analytic-AA walker.
func aaaWalkEdges(prevHead, nextTail *AnalyticEdge, fillType path.FillType, blitter additiveBlitter, maskB *maskAdditiveBlitter, startY, stopY int32, leftClip, rightClip Fixed, forceRLE, skipIntersect bool) {
	prevHead.X = leftClip
	prevHead.UpperX = leftClip
	nextTail.X = rightClip
	nextTail.UpperX = rightClip
	y := max(prevHead.Next.UpperY, Fixed(startY)<<16)
	nextNextY := Fixed(math.MaxInt32)

	{
		edge := prevHead.Next
		for ; edge.UpperY <= y; edge = edge.Next {
			edge.goY(y)
			updateNextNextY(edge.LowerY, y, &nextNextY)
		}
		updateNextNextY(edge.UpperY, y, &nextNextY)
	}

	windingMask := -1
	if fillType.IsEvenOdd() {
		windingMask = 1
	}
	isInverse := fillType.IsInverse()

	if isInverse && Fixed(startY)<<16 != y {
		width := FixedFloorToInt(rightClip - leftClip)
		if FixedFloorToInt(y) != startY {
			blitter.getRealBlitter(false).BlitRect(FixedFloorToInt(leftClip), startY, width,
				FixedFloorToInt(y)-startY)
			startY = FixedFloorToInt(y)
		}
		var mrow maskRow
		if maskB != nil {
			mrow = maskB.getRow(startY)
		}
		blitFullAlpha(blitter, startY, FixedFloorToInt(leftClip), width,
			fixedToAlpha(y-Fixed(startY)<<16), mrow, false)
	}

	for {
		w := 0
		inInterval := isInverse
		prevX := prevHead.X
		nextY := min(nextNextY, FixedCeilToFixed(y+1))
		currE := prevHead.Next
		leftE := prevHead
		left := leftClip
		leftDY := Fixed(0)
		prevRite := FixedFloorToInt(leftClip)

		nextNextY = math.MaxInt32

		yShift := 0
		if (nextY-y)&(FixedOne>>2) != 0 {
			yShift = 2
			nextY = y + (FixedOne >> 2)
		} else if (nextY-y)&(FixedOne>>1) != 0 {
			yShift = 1
		}

		fullAlpha := fixedToAlpha(nextY - y)

		// If we're using the mask blitter, we advance the mask row in this function to save some "if" condition checks.
		var mrow maskRow
		if maskB != nil {
			mrow = maskB.getRow(FixedFloorToInt(y))
		}

		// Even if nextY - y == Fixed1, we can still break the left-to-right order requirement of the clip mask: |\|
		// (two trapezoids with overlapping middle wedges)
		noRealBlitter := forceRLE

		for currE.UpperY <= y {
			w += int(currE.Winding)
			prevInInterval := inInterval
			inInterval = (w&windingMask == 0) == isInverse

			isLeft := inInterval && !prevInInterval
			isRite := !inInterval && prevInInterval

			if isRite {
				rite := currE.X
				currE.goYShift(nextY, yShift)
				nextLeft := max(leftClip, leftE.X)
				rite = min(rightClip, rite)
				nextRite := min(rightClip, currE.X)
				blitTrapezoidRow(blitter, int32(y>>16), left, rite, nextLeft, nextRite,
					leftDY, currE.DY, fullAlpha, mrow,
					noRealBlitter || (fullAlpha == 0xFF &&
						(edgesTooCloseRite(prevRite, left, leftE.X) ||
							edgesTooClose(currE, currE.Next, nextY))))
				prevRite = FixedCeilToInt(max(rite, currE.X))
			} else {
				if isLeft {
					left = max(currE.X, leftClip)
					leftDY = currE.DY
					leftE = currE
				}
				currE.goYShift(nextY, yShift)
			}

			next := currE.Next

			for currE.LowerY <= nextY {
				if currE.CurveCount == 0 {
					break
				}
				c := currE.curve
				c.keepContinuous()
				if !c.updateCurve() {
					break
				}
			}

			if currE.LowerY <= nextY {
				removeAnalyticEdge(currE)
			} else {
				updateNextNextY(currE.LowerY, nextY, &nextNextY)
				newX := currE.X
				if newX < prevX {
					// ripple currE backwards until it is x-sorted
					backwardInsertAnalyticEdgeBasedOnX(currE)
				} else {
					prevX = newX
				}
				if !skipIntersect {
					checkIntersection(currE, nextY, &nextNextY)
				}
			}

			currE = next
		}

		// was our right-edge culled away?
		if inInterval {
			blitTrapezoidRow(blitter, int32(y>>16), left, rightClip, max(leftClip, leftE.X), rightClip,
				leftDY, 0, fullAlpha, mrow,
				noRealBlitter || (fullAlpha == 0xFF && edgesTooClose(leftE.Prev, leftE, nextY)))
		}

		if forceRLE {
			blitter.flushIfYChanged(y, nextY)
		}

		y = nextY
		if y >= Fixed(stopY)<<16 {
			break
		}

		// now currE points to the first edge with an UpperY larger than the previous y
		insertNewAnalyticEdges(currE, y, &nextNextY)
	}
}

///////////////////////////////////////////////////////////////////////////////
// Fill entry points

// aaaFillPath builds the analytic edge list for p and walks it. forceRLE implies the AA clip machinery is calling us.
func aaaFillPath(p *path.Path, clipRect geom.IRect, blitter additiveBlitter, maskB *maskAdditiveBlitter, startY, stopY int32, pathContainedInClip, forceRLE bool) {
	builder := getAnalyticEdgeBuilder()
	defer putAnalyticEdgeBuilder(builder)
	var clipArg *geom.IRect
	if !pathContainedInClip {
		clipArg = &clipRect
	}
	count := builder.buildEdges(p, clipArg)
	list := builder.list

	rect := clipRect
	if count == 0 {
		if p.IsInverseFillType() {
			// Since we are in inverse-fill, our caller has already drawn above our top (startY) and will draw below our
			// bottom (stopY). Thus we need to restrict our drawing to the intersection of the clip and those two
			// limits.
			if rect.Top < startY {
				rect.Top = startY
			}
			if rect.Bottom > stopY {
				rect.Bottom = stopY
			}
			if !rect.IsEmpty() {
				blitter.getRealBlitter(false).BlitRect(rect.Left, rect.Top, rect.Width(), rect.Height())
			}
		}
		return
	}

	// headEdge/tailEdge are the boundary sentinels; they live on the pooled builder (zeroed by reset) so the real edges
	// pointing back at them do not force per-fill heap allocations.
	headEdge, tailEdge := &builder.head, &builder.tail
	edge, last := sortAnalyticEdges(list)

	headEdge.Prev = nil
	headEdge.Next = edge
	headEdge.UpperY = math.MinInt32
	headEdge.LowerY = math.MinInt32
	headEdge.X = math.MinInt32
	headEdge.DX = 0
	headEdge.DY = math.MaxInt32
	headEdge.UpperX = math.MinInt32
	edge.Prev = headEdge

	tailEdge.Prev = last
	tailEdge.Next = nil
	tailEdge.UpperY = math.MaxInt32
	tailEdge.LowerY = math.MaxInt32
	tailEdge.X = math.MaxInt32
	tailEdge.DX = 0
	tailEdge.DY = math.MaxInt32
	tailEdge.UpperX = math.MaxInt32
	last.Next = tailEdge

	// now edge is the head of the sorted linked list

	if !pathContainedInClip && startY < clipRect.Top {
		startY = clipRect.Top
	}
	if !pathContainedInClip && stopY > clipRect.Bottom {
		stopY = clipRect.Bottom
	}

	leftBound := Fixed(rect.Left) << 16
	rightBound := Fixed(rect.Right) << 16
	if maskB != nil {
		// If we're using the mask, then we have to limit the bound within the path bounds; otherwise, the edge drift
		// may access an invalid address inside the mask.
		ir := p.Bounds().RoundOut()
		leftBound = max(leftBound, Fixed(ir.Left)<<16)
		rightBound = min(rightBound, Fixed(ir.Right)<<16)
	}

	if !p.IsInverseFillType() && p.IsConvex() && count >= 2 {
		aaaWalkConvexEdges(headEdge, blitter, maskB, stopY, leftBound, rightBound)
	} else {
		// We skip intersection computation if there are many points, which probably already give us enough fractional
		// scan lines.
		skipIntersect := p.CountPoints() > int(stopY-startY)*2

		aaaWalkEdges(headEdge, tailEdge, p.FillType(), blitter, maskB, startY, stopY,
			leftBound, rightBound, forceRLE, skipIntersect)
	}
}

// tryBlitFatAntiRect checks if the path is a rect and fat enough after clipping; if so, blits it.
func tryBlitFatAntiRect(blitter Blitter, p *path.Path, clip geom.IRect) bool {
	rect, isRect := p.IsRect()
	if !isRect {
		return false
	}
	if !rect.Intersect(clip.ToRect()) {
		return true // The intersection is empty. Hence consider it done.
	}
	bounds := rect.RoundOut()
	if bounds.Width() < 3 {
		return false // not fat
	}
	blitFatAntiRect(blitter, rect)
	return true
}

// AAAFillPath fills the path with analytic antialiasing into blitter. ir is the path's conservative integer bounds;
// clipBounds the clip's bounds. forceRLE is set by the AA clip machinery.
func AAAFillPath(p *path.Path, blitter Blitter, ir, clipBounds geom.IRect, forceRLE bool) {
	containedInClip := clipBounds.ContainsRect(ir)
	isInverse := p.IsInverseFillType()

	// The mask blitter (which accumulates alphas into a small A8 mask and blits it once at the end) is faster than the
	// RLE blitter when the blit region is small enough. When isInverse is true, the blit region is no longer the
	// rectangle ir, so we won't use the mask blitter. When the path is a simple fat rect, blitFatAntiRect avoids the
	// mask-and-blit overhead entirely.
	switch {
	case maskBlitterCanHandleRect(ir) && !isInverse && !forceRLE:
		// blitFatAntiRect is slower than the normal AAA flow without MaskAdditiveBlitter, hence only tryBlitFatAntiRect
		// when MaskAdditiveBlitter would have been used.
		if !tryBlitFatAntiRect(blitter, p, clipBounds) {
			additive := getMaskAdditiveBlitter(blitter, ir, clipBounds)
			aaaFillPath(p, clipBounds, additive, additive, ir.Top, ir.Bottom, containedInClip, forceRLE)
			additive.done()
			putMaskAdditiveBlitter(additive)
		}
	case !isInverse && p.IsConvex():
		// If the filling area is convex, the simpler aaa_walk_convex_edges won't generate alphas above 255, so the
		// basic RLE blitter suffices.
		additive := getRunBasedAdditiveBlitter(blitter, ir, clipBounds, isInverse)
		aaaFillPath(p, clipBounds, additive, nil, ir.Top, ir.Bottom, containedInClip, forceRLE)
		additive.flush()
		putRunBasedAdditiveBlitter(additive)
	default:
		// The filling area might not be convex, so the more involved aaa_walk_edges runs and alphas have to be clamped
		// down to 255; SafeRLEAdditiveBlitter does that at a performance cost.
		additive := getSafeRLEAdditiveBlitter(blitter, ir, clipBounds, isInverse)
		aaaFillPath(p, clipBounds, additive, nil, ir.Top, ir.Bottom, containedInClip, forceRLE)
		additive.flush()
		putSafeRLEAdditiveBlitter(additive)
	}
}

///////////////////////////////////////////////////////////////////////////////
// AntiFillPath dispatch

// superSampleShift is the legacy supersample shift; AntiFillPath is always analytic here, and the shift only survives
// in the coordinate-overflow guard below (which falls back to non-AA filling).
const superSampleShift = 2

// safeRoundOut rounds src outward: RoundOut pins huge floats to max/min int; then intersects with a smaller huge rect
// so the rect will not be considered empty for being too large (e.g. a rect spanning the full int32 range is considered
// empty because its width exceeds signed 32 bits).
func safeRoundOut(src geom.Rect) geom.IRect {
	dst := src.RoundOut()
	const limit = math.MaxInt32 >> superSampleShift
	if !dst.Intersect(geom.IRectLTRB(-limit, -limit, limit, limit)) {
		dst = geom.IRect{}
	}
	return dst
}

func overflowsShortShift(value int32, shift int) int32 {
	s := 16 + shift
	return (value << s >> s) - value
}

// rectOverflowsShortShift reports whether any of the coordinates of this rectangle would not fit in a short when
// left-shifted by shift.
func rectOverflowsShortShift(rect geom.IRect, shift int) bool {
	// Since we expect these to succeed, we bit-or together for a tiny extra bit of speed.
	return overflowsShortShift(rect.Left, shift)|overflowsShortShift(rect.Right, shift)|
		overflowsShortShift(rect.Top, shift)|overflowsShortShift(rect.Bottom, shift) != 0
}

// AntiFillPath fills the path, with analytic antialiasing, into blitter, restricted to a rectangular clip (origClip).
func AntiFillPath(p *path.Path, origClip geom.IRect, blitter Blitter) {
	antiFillPathRegion(p, NewRegionRect(origClip), blitter, false)
}

// AntiFillPathRegion fills the path, with analytic antialiasing, into blitter, restricted to a region clip.
func AntiFillPathRegion(p *path.Path, origClip *Region, blitter Blitter) {
	antiFillPathRegion(p, origClip, blitter, false)
}

func antiFillPathRegion(p *path.Path, origClip *Region, blitter Blitter, forceRLE bool) {
	if origClip.IsEmpty() {
		return
	}

	isInverse := p.IsInverseFillType()
	ir := safeRoundOut(p.Bounds())
	if ir.IsEmpty() {
		if isInverse {
			blitRegion(blitter, origClip)
		}
		return
	}

	// If the intersection of the path bounds and the clip bounds will overflow 32767 when << by the supersample shift,
	// we can't represent coordinates in the fixed-point scan converter, so draw without antialiasing.
	var clippedIR geom.IRect
	if isInverse {
		// If the path is an inverse fill, it's going to fill the entire clip, and we care whether the entire clip
		// exceeds our limits.
		clippedIR = origClip.Bounds()
	} else {
		clippedIR = ir
		if !clippedIR.Intersect(origClip.Bounds()) {
			return
		}
	}
	if rectOverflowsShortShift(clippedIR, superSampleShift) {
		FillPathRegion(p, origClip, blitter)
		return
	}

	// Our antialiasing can't handle a clip larger than 32767, so we restrict the clip to that limit here (the runs[]
	// indexing uses int16_t).
	clipRgn := origClip
	const kMaxClipCoord = 32767
	if b := origClip.Bounds(); b.Right > kMaxClipCoord || b.Bottom > kMaxClipCoord {
		tmp := &Region{}
		tmp.OpRegion(origClip, NewRegionRect(geom.IRectLTRB(0, 0, kMaxClipCoord, kMaxClipCoord)),
			RegionIntersect)
		clipRgn = tmp
	}
	// for here down, use clipRgn, not origClip

	clipper := getScanClipper(blitter, clipRgn, ir, false, false)
	defer putScanClipper(clipper)
	if clipper.blitter == nil { // clipped out
		if isInverse {
			blitRegion(blitter, clipRgn)
		}
		return
	}

	// now use the (possibly wrapped) blitter
	blitter = clipper.blitter

	if isInverse {
		blitAboveClip(blitter, ir, clipRgn)
	}

	AAAFillPath(p, blitter, ir, clipRgn.Bounds(), forceRLE)

	if isInverse {
		blitBelowClip(blitter, ir, clipRgn)
	}
}
