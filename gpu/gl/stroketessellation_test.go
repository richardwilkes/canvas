// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for the stroke tessellation stack: StrokeIterator's verb/cap conversion, the stroke PatchWriter's join
// deferral and encodings, FixedCountStrokes' heuristics, StrokesHaveEqualParams, and StrokeTessellateOp's record-time
// merging (dynamic stroke/color promotion) on the fake driver.

package gl

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// strokeRec builds a plain (non-stroke-and-fill) stroke rec via the styledshape_test helper.
func strokeRec(width float32, capType stroke.Cap, join stroke.Join, miter float32) stroke.Rec {
	return strokeRecWith(width, capType, join, miter, false)
}

// collectStrokeIter drains the iterator, returning the sequence of "current" verbs and their first points.
func collectStrokeIter(it *strokeIterator) (verbs []strokeIterVerb, firstPts []geom.Point) {
	for it.next() {
		verbs = append(verbs, it.verb())
		if pts := it.pts(); len(pts) > 0 {
			firstPts = append(firstPts, pts[0])
		} else {
			firstPts = append(firstPts, geom.Point{})
		}
	}
	return verbs, firstPts
}

func TestStrokeIterVerbClassification(t *testing.T) {
	// Geometric verbs (including the circle verb) sort below the helper verbs.
	for _, v := range []strokeIterVerb{
		strokeIterVerbLine, strokeIterVerbQuad,
		strokeIterVerbConic, strokeIterVerbCubic, strokeIterVerbCircle,
	} {
		if !strokeIterVerbIsGeometric(v) {
			t.Fatalf("verb %d must be geometric", v)
		}
	}
	for _, v := range []strokeIterVerb{
		strokeIterVerbMoveWithinContour,
		strokeIterVerbContourFinished,
	} {
		if strokeIterVerbIsGeometric(v) {
			t.Fatalf("verb %d must not be geometric", v)
		}
	}
	// The verb constants are pinned to specific values: line=1 .. cubic=4, circle=5, moveWithinContour=6,
	// contourFinished=7.
	if strokeIterVerbLine != 1 || strokeIterVerbCubic != 4 || strokeIterVerbCircle != 5 ||
		strokeIterVerbMoveWithinContour != 6 || strokeIterVerbContourFinished != 7 {
		t.Fatal("strokeIterVerb values changed unexpectedly")
	}
}

func TestStrokeIteratorClosedContour(t *testing.T) {
	identity := geom.IdentityMatrix()
	rec := strokeRec(4, stroke.CapButt, stroke.JoinMiter, 4)

	p := &path.Path{}
	p.MoveTo(10, 10).LineTo(50, 10).LineTo(50, 50).Close()

	it := newStrokeIterator(p, &rec, &identity)
	verbs, firstPts := collectStrokeIter(it)

	// The first verb is deferred and repeated at the end as "current"; the close injects a line back to the start.
	want := []strokeIterVerb{
		strokeIterVerbLine, strokeIterVerbLine, strokeIterVerbLine,
		strokeIterVerbContourFinished,
	}
	if len(verbs) != len(want) {
		t.Fatalf("verbs = %v, want %v", verbs, want)
	}
	for i := range want {
		if verbs[i] != want[i] {
			t.Fatalf("verb %d = %v, want %v", i, verbs[i], want[i])
		}
	}
	// Current strokes in order: line2 (50,10 -> 50,50), close line (50,50 -> 10,10), repeated line1 (10,10 -> 50,10).
	if firstPts[0] != (geom.Point{X: 50, Y: 10}) ||
		firstPts[1] != (geom.Point{X: 50, Y: 50}) ||
		firstPts[2] != (geom.Point{X: 10, Y: 10}) {
		t.Fatalf("first points = %v", firstPts[:3])
	}
}

func TestStrokeIteratorAlreadyClosedContour(t *testing.T) {
	// When the contour already ends at its start point, the close must not inject a line.
	identity := geom.IdentityMatrix()
	rec := strokeRec(4, stroke.CapButt, stroke.JoinMiter, 4)

	p := &path.Path{}
	p.MoveTo(10, 10).LineTo(50, 10).LineTo(10, 10).Close()

	it := newStrokeIterator(p, &rec, &identity)
	verbs, _ := collectStrokeIter(it)
	want := []strokeIterVerb{
		strokeIterVerbLine, strokeIterVerbLine,
		strokeIterVerbContourFinished,
	}
	if len(verbs) != len(want) {
		t.Fatalf("verbs = %v, want %v", verbs, want)
	}
}

func TestStrokeIteratorOpenContourCaps(t *testing.T) {
	identity := geom.IdentityMatrix()
	mkPath := func() *path.Path {
		p := &path.Path{}
		p.MoveTo(10, 10).LineTo(50, 10).QuadTo(70, 10, 70, 30)
		return p
	}

	// Butt caps: a kMoveWithinContour barrier, then the repeated first verb.
	rec := strokeRec(4, stroke.CapButt, stroke.JoinBevel, 4)
	verbs, _ := collectStrokeIter(newStrokeIterator(mkPath(), &rec, &identity))
	want := []strokeIterVerb{
		strokeIterVerbQuad, strokeIterVerbMoveWithinContour,
		strokeIterVerbLine, strokeIterVerbContourFinished,
	}
	for i := range want {
		if verbs[i] != want[i] {
			t.Fatalf("butt verbs = %v, want %v", verbs, want)
		}
	}

	// Round caps: two kCircle verbs — the contour's end point then its start point.
	rec = strokeRec(4, stroke.CapRound, stroke.JoinBevel, 4)
	it := newStrokeIterator(mkPath(), &rec, &identity)
	verbs, firstPts := collectStrokeIter(it)
	want = []strokeIterVerb{
		strokeIterVerbQuad, strokeIterVerbCircle, strokeIterVerbCircle,
		strokeIterVerbLine, strokeIterVerbContourFinished,
	}
	for i := range want {
		if verbs[i] != want[i] {
			t.Fatalf("round verbs = %v, want %v", verbs, want)
		}
	}
	if firstPts[1] != (geom.Point{X: 70, Y: 30}) || firstPts[2] != (geom.Point{X: 10, Y: 10}) {
		t.Fatalf("round cap circle points = %v, %v", firstPts[1], firstPts[2])
	}

	// Square caps: cap lines extend by half the stroke width along the end tangents.
	rec = strokeRec(4, stroke.CapSquare, stroke.JoinBevel, 4)
	it = newStrokeIterator(mkPath(), &rec, &identity)
	type rec2 struct {
		pts  []geom.Point
		verb strokeIterVerb
	}
	var seq []rec2
	for it.next() {
		seq = append(seq, rec2{verb: it.verb(), pts: it.pts()})
	}
	wantVerbs := []strokeIterVerb{
		strokeIterVerbQuad, strokeIterVerbLine,
		strokeIterVerbMoveWithinContour, strokeIterVerbLine, strokeIterVerbLine,
		strokeIterVerbContourFinished,
	}
	for i := range wantVerbs {
		if seq[i].verb != wantVerbs[i] {
			t.Fatalf("square verbs mismatch at %d: %v, want %v", i, seq[i].verb, wantVerbs[i])
		}
	}
	// The ending cap extends from the quad's endpoint (70,30) along its end tangent (0,1) by half the stroke width (2).
	endCap := seq[1].pts
	if endCap[0] != (geom.Point{X: 70, Y: 30}) || endCap[1] != (geom.Point{X: 70, Y: 32}) {
		t.Fatalf("ending square cap = %v", endCap)
	}
	// The beginning cap is set back from the contour's start (10,10) along -tangent (-1,0) by 2.
	beginCap := seq[3].pts
	if beginCap[0] != (geom.Point{X: 8, Y: 10}) || beginCap[1] != (geom.Point{X: 10, Y: 10}) {
		t.Fatalf("beginning square cap = %v", beginCap)
	}
}

func TestStrokeIteratorZeroLengthContours(t *testing.T) {
	identity := geom.IdentityMatrix()
	mkPath := func() *path.Path {
		p := &path.Path{}
		p.MoveTo(5, 5).LineTo(5, 5) // degenerate: zero-length subpath
		return p
	}

	// Butt caps: nothing is generated.
	rec := strokeRec(4, stroke.CapButt, stroke.JoinBevel, 4)
	verbs, _ := collectStrokeIter(newStrokeIterator(mkPath(), &rec, &identity))
	if len(verbs) != 0 {
		t.Fatalf("butt zero-length verbs = %v, want none", verbs)
	}

	// Round caps: a circle at the point.
	rec = strokeRec(4, stroke.CapRound, stroke.JoinBevel, 4)
	verbs, firstPts := collectStrokeIter(newStrokeIterator(mkPath(), &rec, &identity))
	if len(verbs) != 2 || verbs[0] != strokeIterVerbCircle ||
		verbs[1] != strokeIterVerbContourFinished {
		t.Fatalf("round zero-length verbs = %v", verbs)
	}
	if firstPts[0] != (geom.Point{X: 5, Y: 5}) {
		t.Fatalf("round zero-length circle at %v", firstPts[0])
	}

	// Square caps: a stroke-width square expressed as a line through the point. The cap line is first enqueued as the
	// "prev" join context; the currents are the move barrier, then the repeated cap line.
	rec = strokeRec(4, stroke.CapSquare, stroke.JoinBevel, 4)
	it := newStrokeIterator(mkPath(), &rec, &identity)
	if !it.next() {
		t.Fatal("square zero-length must produce geometry")
	}
	if it.prevVerb() != strokeIterVerbLine || it.verb() != strokeIterVerbMoveWithinContour {
		t.Fatalf("square zero-length pair = (%v, %v), want (line, moveWithinContour)",
			it.prevVerb(), it.verb())
	}
	if !it.next() {
		t.Fatal("square zero-length must repeat the cap line")
	}
	if it.verb() != strokeIterVerbLine {
		t.Fatalf("square zero-length verb = %v, want line", it.verb())
	}
	pts := it.pts()
	if pts[0] != (geom.Point{X: 3, Y: 5}) || pts[1] != (geom.Point{X: 7, Y: 5}) {
		t.Fatalf("square zero-length line = %v", pts)
	}
}

func TestStrokeIteratorDegenerateVerbsSkipped(t *testing.T) {
	identity := geom.IdentityMatrix()
	rec := strokeRec(4, stroke.CapButt, stroke.JoinBevel, 4)

	p := &path.Path{}
	p.MoveTo(10, 10)
	p.LineTo(10, 10)                  // degenerate line
	p.CubicTo(10, 10, 10, 10, 10, 10) // degenerate cubic
	p.LineTo(50, 10)                  // real line
	p.QuadTo(50, 10, 50, 10)          // degenerate quad (all points colocated at (50,10))
	p.CubicTo(60, 10, 60, 20, 60, 20) // NOT degenerate (only the tail is colocated)
	it := newStrokeIterator(p, &rec, &identity)
	verbs, _ := collectStrokeIter(it)
	want := []strokeIterVerb{
		strokeIterVerbCubic, strokeIterVerbMoveWithinContour,
		strokeIterVerbLine, strokeIterVerbContourFinished,
	}
	if len(verbs) != len(want) {
		t.Fatalf("verbs = %v, want %v", verbs, want)
	}
	for i := range want {
		if verbs[i] != want[i] {
			t.Fatalf("verb %d = %v, want %v", i, verbs[i], want[i])
		}
	}
}

func strokeWriterAttribs() PatchAttribs { return PatchAttribJoinControlPoint }

func TestStrokePatchWriterDeferredJoin(t *testing.T) {
	// A closed triangle contour A->B, B->C, C->A: the first patch is deferred and flushed with the join control point
	// defined by the final patch's outgoing tangent.
	alloc := &testPatchAllocator{stride: patchStride(strokeWriterAttribs())}
	w := newStrokePatchWriter(strokeWriterAttribs(), alloc)

	a := geom.Point{X: 10, Y: 10}
	b := geom.Point{X: 50, Y: 10}
	c := geom.Point{X: 30, Y: 40}
	w.writeLine(a, b) // deferred
	w.writeLine(b, c)
	w.writeLine(c, a)
	w.writeDeferredStrokePatch()

	if len(alloc.patches) != 3 {
		t.Fatalf("patches = %d, want 3", len(alloc.patches))
	}
	// ReplicateLineEndPoints: lines are encoded {p, p, q, q}.
	first := alloc.patches[0]
	if patchPointAt(first, 0) != b || patchPointAt(first, 1) != b ||
		patchPointAt(first, 2) != c || patchPointAt(first, 3) != c {
		t.Fatalf("line patch control points = %v %v %v %v", patchPointAt(first, 0),
			patchPointAt(first, 1), patchPointAt(first, 2), patchPointAt(first, 3))
	}
	// Patch order: B->C (join A), C->A (join B), then the deferred A->B flushed with join C.
	if got := patchPointAt(alloc.patches[0], 4); got != a {
		t.Fatalf("patch 0 join = %v, want %v", got, a)
	}
	if got := patchPointAt(alloc.patches[1], 4); got != b {
		t.Fatalf("patch 1 join = %v, want %v", got, b)
	}
	deferred := alloc.patches[2]
	if patchPointAt(deferred, 0) != a || patchPointAt(deferred, 3) != b {
		t.Fatalf("deferred patch spans %v..%v, want %v..%v", patchPointAt(deferred, 0),
			patchPointAt(deferred, 3), a, b)
	}
	if got := patchPointAt(deferred, 4); got != c {
		t.Fatalf("deferred join = %v, want %v", got, c)
	}
}

func TestStrokePatchWriterManualJoinDisablesDeferral(t *testing.T) {
	// Once updateJoinControlPointAttrib supplies a join (open contours), nothing defers.
	alloc := &testPatchAllocator{stride: patchStride(strokeWriterAttribs())}
	w := newStrokePatchWriter(strokeWriterAttribs(), alloc)

	a := geom.Point{X: 10, Y: 10}
	b := geom.Point{X: 50, Y: 10}
	w.updateJoinControlPointAttrib(a)
	w.writeLine(a, b)
	if len(alloc.patches) != 1 {
		t.Fatalf("patches = %d, want 1 (no deferral)", len(alloc.patches))
	}
	if got := patchPointAt(alloc.patches[0], 4); got != a {
		t.Fatalf("join = %v, want %v", got, a)
	}
	// Flushing produces nothing further.
	w.writeDeferredStrokePatch()
	if len(alloc.patches) != 1 {
		t.Fatalf("patches after flush = %d, want 1", len(alloc.patches))
	}
}

func TestStrokePatchWriterCircle(t *testing.T) {
	// Circles encode as 4 identical control points with their own location as the join, and never defer.
	alloc := &testPatchAllocator{stride: patchStride(strokeWriterAttribs())}
	w := newStrokePatchWriter(strokeWriterAttribs(), alloc)

	p := geom.Point{X: 7, Y: 9}
	w.writeCircle(p)
	if len(alloc.patches) != 1 {
		t.Fatalf("patches = %d, want 1", len(alloc.patches))
	}
	patch := alloc.patches[0]
	for i := range 4 {
		if patchPointAt(patch, i) != p {
			t.Fatalf("circle control point %d = %v, want %v", i, patchPointAt(patch, i), p)
		}
	}
	if got := patchPointAt(patch, 4); got != p {
		t.Fatalf("circle join = %v, want %v", got, p)
	}
}

func TestStrokePatchWriterAttribLayout(t *testing.T) {
	// join + strokeParams + byte color + explicit curve type, in PatchAttribs order.
	attribs := PatchAttribJoinControlPoint | PatchAttribStrokeParams | PatchAttribColor |
		PatchAttribExplicitCurveType
	alloc := &testPatchAllocator{stride: patchStride(attribs)}
	w := newStrokePatchWriter(attribs, alloc)

	rec := strokeRec(6, stroke.CapButt, stroke.JoinMiter, 4)
	w.updateStrokeParamsAttrib(makeStrokeParams(&rec))
	w.updateColorAttrib(colorcore.PMColor4f{R: 1, G: 0.5, B: 0, A: 1})
	w.updateJoinControlPointAttrib(geom.Point{X: 1, Y: 2})
	w.writeLine(geom.Point{X: 0, Y: 0}, geom.Point{X: 10, Y: 0})

	if len(alloc.patches) != 1 {
		t.Fatalf("patches = %d, want 1", len(alloc.patches))
	}
	patch := alloc.patches[0]
	if len(patch) != 32+8+8+4+4 {
		t.Fatalf("stride = %d", len(patch))
	}
	if got := patchPointAt(patch, 4); got != (geom.Point{X: 1, Y: 2}) {
		t.Fatalf("join = %v", got)
	}
	radius := math.Float32frombits(binary.LittleEndian.Uint32(patch[40:]))
	joinType := math.Float32frombits(binary.LittleEndian.Uint32(patch[44:]))
	if radius != 3 || joinType != 4 {
		t.Fatalf("stroke params = (%v, %v), want (3, 4)", radius, joinType)
	}
	if patch[48] != 255 || patch[49] != 128 || patch[50] != 0 || patch[51] != 255 {
		t.Fatalf("byte color = %v", patch[48:52])
	}
	curveType := math.Float32frombits(binary.LittleEndian.Uint32(patch[52:]))
	if curveType != tessCubicCurveType {
		t.Fatalf("curve type = %v", curveType)
	}
}

// TestTessJoinEncodingAndEdges covers the join encoding and the per-lane fixed join-edge counts: a negative miter limit
// must not encode as a round join, and the static-join lane's count must match the constant its shader compiles in
// (tessNumFixedEdgesInJoinType), including for a miter join whose limit is 0 — which encodes as a bevel, so the
// dynamic lane's count is one edge lower and would size the draw short of what the static shader consumes.
func TestTessJoinEncodingAndEdges(t *testing.T) {
	round := strokeRec(4, stroke.CapButt, stroke.JoinRound, 4)
	bevel := strokeRec(4, stroke.CapButt, stroke.JoinBevel, 4)
	miter := strokeRec(4, stroke.CapButt, stroke.JoinMiter, 4)
	miter0 := strokeRec(4, stroke.CapButt, stroke.JoinMiter, 0)
	miterNeg := strokeRec(4, stroke.CapButt, stroke.JoinMiter, -3)
	for _, c := range []struct {
		rec  *stroke.Rec
		name string
		want float32
	}{
		{name: "round", rec: &round, want: -1},
		{name: "bevel", rec: &bevel, want: 0},
		{name: "miter", rec: &miter, want: 4},
		{name: "miter limit 0", rec: &miter0, want: 0},
		{name: "negative miter limit", rec: &miterNeg, want: 0},
	} {
		if got := tessGetJoinType(c.rec); got != c.want {
			t.Errorf("%s: tessGetJoinType = %v, want %v", c.name, got, c.want)
		}
	}

	// The static lane counts the miter's extra edge from the join enum whatever the limit is; the dynamic lane can only
	// read the encoding, which collapses a 0 limit onto bevel.
	if got := tessNumFixedEdgesInJoinType(stroke.JoinMiter); got != 4 {
		t.Errorf("static miter join edges = %d, want 4", got)
	}
	if got := tessNumFixedEdgesInJoin(makeStrokeParams(&miter0)); got != 3 {
		t.Errorf("dynamic miter-limit-0 join edges = %d, want 3", got)
	}

	// The uniform/static lane must size the draw for the enum's count, so the final (exact-p3) edge of the instance is
	// still emitted for a miter join with a 0 limit.
	writer := &patchWriter{maxScale: 1}
	writer.tolerances = newLinearTolerances()
	writer.updateUniformStrokeParams(makeStrokeParams(&miter0), miter0.Join())
	staticEdges := writer.tolerances.requiredStrokeEdges()
	dynamic := &patchWriter{maxScale: 1, attribs: PatchAttribStrokeParams}
	dynamic.tolerances = newLinearTolerances()
	dynamic.updateStrokeParamsAttrib(makeStrokeParams(&miter0))
	if got := staticEdges - dynamic.tolerances.requiredStrokeEdges(); got != 1 {
		t.Errorf("static lane allots %d extra join edge(s) over the dynamic lane, want 1", got)
	}
	if got := writer.tolerances.edgesInJoins; got != tessNumFixedEdgesInJoinType(stroke.JoinMiter) {
		t.Errorf("static lane join edges = %d, want %d", got,
			tessNumFixedEdgesInJoinType(stroke.JoinMiter))
	}
}

func TestStrokesHaveEqualParams(t *testing.T) {
	a := strokeRec(4, stroke.CapButt, stroke.JoinMiter, 4)
	b := strokeRec(4, stroke.CapRound, stroke.JoinMiter, 4)
	if !strokesHaveEqualParams(&a, &b) {
		t.Fatal("caps must not affect stroke param equality")
	}
	b = strokeRec(4, stroke.CapButt, stroke.JoinMiter, 5)
	if strokesHaveEqualParams(&a, &b) {
		t.Fatal("miter limits differ for miter joins")
	}
	a = strokeRec(4, stroke.CapButt, stroke.JoinBevel, 4)
	b = strokeRec(4, stroke.CapButt, stroke.JoinBevel, 5)
	if !strokesHaveEqualParams(&a, &b) {
		t.Fatal("miter limits are ignored for non-miter joins")
	}
	b = strokeRec(5, stroke.CapButt, stroke.JoinBevel, 4)
	if strokesHaveEqualParams(&a, &b) {
		t.Fatal("widths differ")
	}
}

func TestFixedCountStrokes(t *testing.T) {
	if got := fixedCountStrokesPreallocCount(10); got != 28 {
		t.Fatalf("prealloc count = %d, want 28", got)
	}
	// Vertex count is 2x the required stroke edges, clamped to kMaxEdges.
	var tol linearTolerances
	rec := strokeRec(10, stroke.CapButt, stroke.JoinRound, 4)
	params := makeStrokeParams(&rec)
	tol.setStroke(params, tessNumFixedEdgesInJoin(params), 1)
	tol.setParametricSegments(16 * 16 * 16 * 16)
	edges := tol.requiredStrokeEdges()
	if got := fixedCountStrokesVertexCountForTolerances(&tol); got != edges*2 {
		t.Fatalf("vertex count = %d, want %d", got, edges*2)
	}
	tol.numRadialSegmentsPerRadian = 1e9
	if got := fixedCountStrokesVertexCountForTolerances(&tol); got != fixedCountStrokesMaxEdges*2 {
		t.Fatalf("clamped vertex count = %d, want %d", got, fixedCountStrokesMaxEdges*2)
	}
}

// tessOpenStrokePath returns a volatile open path with a curve for stroke-op tests.
func tessOpenStrokePath(dx, dy float32) *path.Path {
	p := &path.Path{}
	p.MoveTo(10+dx, 30+dy)
	p.LineTo(40+dx, 30+dy)
	p.QuadTo(60+dx, 30+dy, 60+dx, 60+dy)
	p.SetVolatile(true)
	return p
}

func drawStroked(t *testing.T, sdc *SurfaceDrawContext, p *path.Path, rec stroke.Rec, paint *Paint) {
	t.Helper()
	identity := geom.IdentityMatrix()
	shape := MakeStyledShapePath(p, MakeStyle(rec), DoSimplifyNo)
	sdc.DrawShape(nil, paint, gpu.AANo, &identity, &shape)
}

func TestStrokeTessellateOpMerging(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()

	// Two identical strokes with different opaque colors merge, promoting color to a dynamic attrib (opaque src-over is
	// unaffected by dst, so no stencil).
	rec := strokeRec(4, stroke.CapButt, stroke.JoinBevel, 4)
	drawStroked(t, sdc, tessOpenStrokePath(0, 0), rec, solidPaint(1, 0, 0, 1))
	drawStroked(t, sdc, tessOpenStrokePath(50, 20), rec, solidPaint(0, 0, 1, 1))

	task := sdc.GetOpsTask()
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("op chains = %d, want 1 (merged)", n)
	}
	op, ok := task.GetChainHead(0).(*strokeTessellateOp)
	if !ok {
		t.Fatalf("chain head is %T, want *strokeTessellateOp", task.GetChainHead(0))
	}
	if op.patchAttribs&PatchAttribColor == 0 {
		t.Fatal("merged op with differing colors must move color into patch attribs")
	}
	if op.patchAttribs&PatchAttribStrokeParams != 0 {
		t.Fatal("equal strokes must not enable dynamic stroke params")
	}
	if op.pathStrokeList.next == nil {
		t.Fatal("merged op must carry both draws")
	}

	// A different stroke width merges by promoting dynamic stroke params (verb count is small).
	wide := strokeRec(9, stroke.CapButt, stroke.JoinBevel, 4)
	drawStroked(t, sdc, tessOpenStrokePath(20, 60), wide, solidPaint(0, 1, 0, 1))
	if n := task.NumOpChains(); n != 1 {
		t.Fatalf("op chains after width change = %d, want 1", n)
	}
	if op.patchAttribs&PatchAttribStrokeParams == 0 {
		t.Fatal("differing widths must enable dynamic stroke params")
	}
	if op.totalCombinedVerbCnt != 3*3 {
		t.Fatalf("combined verb count = %d, want 9", op.totalCombinedVerbCnt)
	}

	// Hairlines cannot merge with non-hairlines.
	hairline := stroke.NewStrokeRec(stroke.InitStyleHairline)
	drawStroked(t, sdc, tessOpenStrokePath(5, 80), hairline, solidPaint(0, 0, 0, 1))
	if n := task.NumOpChains(); n != 2 {
		t.Fatalf("op chains after hairline = %d, want 2", n)
	}

	// A translucent stroke needs the stencil pass (affected by dst): it cannot merge with anything and reports
	// UsesStencil.
	drawStroked(t, sdc, tessOpenStrokePath(30, 40), rec, solidPaint(1, 0, 0, 0.5))
	if n := task.NumOpChains(); n != 3 {
		t.Fatalf("op chains after translucent stroke = %d, want 3", n)
	}
	last, ok := task.GetChainHead(2).(*strokeTessellateOp)
	if !ok {
		t.Fatalf("chain head 2 is %T, want *strokeTessellateOp", task.GetChainHead(2))
	}
	if !last.needsStencil || !last.UsesStencil() {
		t.Fatal("translucent strokes must take the stencil-then-fill lane")
	}
}

func TestStrokeTessellationRendererSelection(t *testing.T) {
	dc := newShaderRecordingContext(t)
	sdc := newDrawTestSDC(t, dc, 128, 128)
	defer sdc.Release()

	identity := geom.IdentityMatrix()
	bounds := geom.IRectSize(sdc.Dimensions())
	makeArgs := func(shape *StyledShape, aaType gpu.AAType) *CanDrawPathArgs {
		return &CanDrawPathArgs{
			Caps:                   sdc.Caps(),
			Proxy:                  sdc.AsRenderTargetProxy(),
			ClipConservativeBounds: bounds,
			ViewMatrix:             &identity,
			Shape:                  shape,
			AAType:                 aaType,
		}
	}

	star := &path.Path{}
	star.MoveTo(50, 5).LineTo(70, 70).LineTo(10, 30).LineTo(90, 30).LineTo(30, 70).Close()

	// Non-AA strokes select the tessellation renderer.
	rec := strokeRec(5, stroke.CapRound, stroke.JoinRound, 4)
	shape := MakeStyledShapePath(star, MakeStyle(rec), DoSimplifyNo)
	pr := dc.DrawingManager().GetPathRenderer(makeArgs(&shape, gpu.AATypeNone), false,
		ChainDrawTypeColor, nil)
	if pr == nil || pr.Name() != "Tessellation" {
		t.Fatalf("non-AA stroke selected %v, want Tessellation", pr)
	}

	// Coverage AA strokes are rejected (no analytic AA in this renderer).
	if r := NewTessellationPathRenderer(); r.OnCanDrawPath(
		makeArgs(&shape, gpu.AATypeCoverage),
	) != CanDrawPathNo {
		t.Fatal("coverage-AA strokes must be rejected")
	}

	// Massively wide strokes (post-scale) are rejected (crbug.com/1266446).
	wide := strokeRec(20000, stroke.CapButt, stroke.JoinBevel, 4)
	wideShape := MakeStyledShapePath(star, MakeStyle(wide), DoSimplifyNo)
	if r := NewTessellationPathRenderer(); r.OnCanDrawPath(
		makeArgs(&wideShape, gpu.AATypeNone),
	) != CanDrawPathNo {
		t.Fatal("massively wide strokes must be rejected")
	}

	// Stroke-and-fill styles are rejected.
	sf := stroke.NewStrokeRec(stroke.InitStyleFill)
	sf.SetStrokeStyle(5, true)
	sfShape := MakeStyledShapePath(star, MakeStyle(sf), DoSimplifyNo)
	if r := NewTessellationPathRenderer(); r.OnCanDrawPath(
		makeArgs(&sfShape, gpu.AATypeNone),
	) != CanDrawPathNo {
		t.Fatal("stroke-and-fill styles must be rejected")
	}
}
