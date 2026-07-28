// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The replay half of the reference ops for the op groups that still have a capture counterpart
// (ref_ops_capture_test.go, built under `-tags skiaoracle`). The tag keeps the two definitions from colliding: an
// ordinary `go test` compiles this file and reads frozen values with no C toolchain, while a capture run compiles the
// other and forwards to live Skia.
//
// Op groups whose capture code has been retired live in untagged files instead (see ref_ops_pathops_test.go).
//go:build !skiaoracle

package probe

import (
	"strconv"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
)

// The frozen reference ops. Each returns Skia's value for the given inputs, captured on darwin/arm64; see
// ref_test.go.
//
// A lookup failure is fatal rather than a skip: a missing key means the probe corpus drifted away from what was
// captured, and there is no oracle left to answer for the new input. Silently skipping would let coverage evaporate
// unnoticed.

// --- matrix ---

func refMatrixSetConcat(a, b *geom.Matrix) geom.Matrix {
	return mustMatrix(refGet("MatrixSetConcat", hashKey(encMatrix(a)+"|"+encMatrix(b))))
}

func refMatrixInvert(m *geom.Matrix) (geom.Matrix, bool) {
	inv, ok := split2(refGet("MatrixInvert", hashKey(encMatrix(m))))
	return mustMatrix(inv), mustBool(ok)
}

func refMatrixMapPoints(m *geom.Matrix, pts []geom.Point) []geom.Point {
	return mustPoints(refGet("MatrixMapPoints", hashKey(encMatrix(m)+"|"+encPoints(pts))))
}

func refMatrixMapRect(m *geom.Matrix, src geom.Rect) (geom.Rect, bool) {
	r, ok := split2(refGet("MatrixMapRect", hashKey(encMatrix(m)+"|"+encRect(src))))
	return mustRect(r), mustBool(ok)
}

// --- rect ---

func refRectIntersect(a, b geom.Rect) (geom.Rect, bool) {
	r, ok := split2(refGet("RectIntersect", hashKey(encRect(a)+"|"+encRect(b))))
	return mustRect(r), mustBool(ok)
}

func refRectJoin(a, b geom.Rect) geom.Rect {
	return mustRect(refGet("RectJoin", hashKey(encRect(a)+"|"+encRect(b))))
}

func refRectSetBounds(pts []geom.Point) (geom.Rect, bool) {
	r, ok := split2(refGet("RectSetBounds", hashKey(encPoints(pts))))
	return mustRect(r), mustBool(ok)
}

func refRectRound(r geom.Rect) geom.IRect {
	return mustIRect(refGet("RectRound", hashKey(encRect(r))))
}

func refRectRoundOut(r geom.Rect) geom.IRect {
	return mustIRect(refGet("RectRoundOut", hashKey(encRect(r))))
}

func refRectRoundIn(r geom.Rect) geom.IRect {
	return mustIRect(refGet("RectRoundIn", hashKey(encRect(r))))
}

func refScalarRoundToInt(x float32) int32 {
	return int32(mustInt(refGet("ScalarRoundToInt", hashKey(f32(x)))))
}

// --- geometry (Skia's Geometry chop/conic) ---

func refChopQuadAt(src [3]geom.Point, t float32) [5]geom.Point {
	v := mustPoints(refGet("ChopQuadAt", hashKey(encPoints(src[:])+"|"+f32(t))))
	if len(v) != 5 {
		refFatal(errCount("ChopQuadAt", len(v), 5))
	}
	return [5]geom.Point(v)
}

func refChopCubicAt(src [4]geom.Point, t float32) [7]geom.Point {
	v := mustPoints(refGet("ChopCubicAt", hashKey(encPoints(src[:])+"|"+f32(t))))
	if len(v) != 7 {
		refFatal(errCount("ChopCubicAt", len(v), 7))
	}
	return [7]geom.Point(v)
}

func refConicComputeQuadPOW2(pts [3]geom.Point, w, tol float32) int {
	return mustInt(refGet("ConicComputeQuadPOW2", hashKey(encPoints(pts[:])+"|"+encFloats(w, tol))))
}

func refConicChopIntoQuadsPOW2(pts [3]geom.Point, w float32, pow2 int) (int, []geom.Point) {
	n, ptsEnc := split2(refGet("ConicChopIntoQuadsPOW2",
		hashKey(encPoints(pts[:])+"|"+f32(w)+"|"+strconv.Itoa(pow2))))
	return mustInt(n), mustPoints(ptsEnc)
}

func refConicBuildUnitArc(u, v geom.Point, dir int, m *geom.Matrix) (pts [][3]geom.Point, weights []float32) {
	mEnc := "nil"
	if m != nil {
		mEnc = encMatrix(m)
	}
	ptsEnc, wEnc := split2(refGet("ConicBuildUnitArc",
		hashKey(encPoint(u)+"|"+encPoint(v)+"|"+strconv.Itoa(dir)+"|"+mEnc)))
	flat := mustPoints(ptsEnc)
	if len(flat)%3 != 0 {
		refFatal(errCount("ConicBuildUnitArc", len(flat), 3))
	}
	pts = make([][3]geom.Point, len(flat)/3)
	for i := range pts {
		pts[i] = [3]geom.Point(flat[i*3 : i*3+3])
	}
	ws, err := decFloats(wEnc, -1)
	if err != nil {
		refFatal(err)
	}
	return pts, ws
}

// --- path structure/bounds/containment/transform ---
//
// Grouped by probe rather than by C call: each of these reads several facts off one built path, and one fixture entry
// per path is both smaller and easier to read than one per accessor. Containment in particular arrives as a whole grid
// — 147 queries per path — packed into a single bool run.

func refPathStructure(s *scenario.PathSpec) pathFacts {
	f := splitN("PathStructure", refGet("PathStructure", hashKey(encSpec(s))), 5)
	facts := pathFacts{
		isEmpty:  mustBool(f[0]),
		points:   mustPoints(f[1]),
		lastOK:   mustBool(f[3]),
		fillType: mustInt(f[4]),
	}
	if facts.lastOK {
		pts := mustPoints(f[2])
		if len(pts) != 1 {
			refFatal(errCount("PathStructure lastPt", len(pts), 1))
		}
		facts.lastPt = pts[0]
	}
	return facts
}

func refPathBounds(s *scenario.PathSpec) (bounds, tight geom.Rect) {
	f := splitN("PathBounds", refGet("PathBounds", hashKey(encSpec(s))), 2)
	return mustRect(f[0]), mustRect(f[1])
}

func refPathContains(s *scenario.PathSpec, pts []geom.Point) []bool {
	return decBools(refGet("PathContains", hashKey(encSpec(s)+"|"+encPoints(pts))))
}

func refPathTransform(s *scenario.PathSpec, m *geom.Matrix) []geom.Point {
	return mustPoints(refGet("PathTransform", hashKey(encSpec(s)+"|"+encMatrix(m))))
}

// --- path measure ---

func refPathMeasure(s *scenario.PathSpec, forceClosed bool, resScale float32) []measureContour {
	v, err := decMeasure(refGet("PathMeasure",
		hashKey(encSpec(s)+"|"+encBool(forceClosed)+"|"+f32(resScale))))
	if err != nil {
		refFatal(err)
	}
	return v
}

// --- stroker ---

func refStrokeFillPath(s *scenario.PathSpec, sp *strokePaint) strokeResult {
	f := splitN("StrokeFillPath", refGet("StrokeFillPath", hashKey(encSpec(s)+"|"+encStrokePaint(sp))), 3)
	return strokeResult{isFill: mustBool(f[0]), fillType: mustInt(f[1]), points: mustPoints(f[2])}
}

// --- analytic AA edge walkers ---

func refAnalyticQuadSegments(pts [3]geom.Point) [][7]int32 {
	return mustSegments(refGet("AnalyticQuadSegments", hashKey(encPoints(pts[:]))))
}

func refAnalyticCubicSegments(pts [4]geom.Point) [][7]int32 {
	return mustSegments(refGet("AnalyticCubicSegments", hashKey(encPoints(pts[:]))))
}
