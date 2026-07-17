// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The Skia's PathMeasure probe.
//
// TestPathMeasureProbe drives the contour package against Skia's PathMeasure through the shim side door (the public
// surface never exposes measurement standalone): contour lengths, closedness, pos/tan samples, matrices, and segment
// extraction, iterated in lock-step across contours. The oracle's whole walk is replayed from the frozen reference
// fixtures (see ref_test.go), so this no longer needs the C library — which is also why it no longer carries //go:build
// !windows: that tag existed because mingw cannot resolve Skia's MSVC-mangled C++ exports, and replay needs no linker.
//
// The path-effect probes that shared this file (fill-path, canvas rendering, dashed asPoints) drove the C library live
// and were removed with the rest of the render probes.

package probe

import (
	"fmt"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/contour"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/path"
)

// measureCorpus: geometry chosen to hit each measurement lane — lines, polylines, closed contours, rects, all curve
// types (subdivision), shapes, multiple contours, and degenerate contours.
func measureCorpus() []*scenario.PathSpec {
	sp := scenario.NewPathSpec
	return []*scenario.PathSpec{
		sp().MoveTo(10, 10).LineTo(90, 70),
		sp().MoveTo(10, 80).LineTo(40, 20).LineTo(70, 80).LineTo(100, 20),
		sp().MoveTo(20, 20).LineTo(90, 25).LineTo(60, 80).Close(),
		sp().AddRect(geom.RectLTRB(20, 20, 100, 70), geom.DirectionCW),
		sp().MoveTo(10, 70).QuadTo(60, 10, 110, 70),
		sp().MoveTo(10, 40).QuadTo(60, 40.001, 110, 40),
		sp().MoveTo(10, 70).ConicTo(60, 10, 110, 70, 0.70710678),
		sp().MoveTo(10, 70).ConicTo(60, 10, 110, 70, 2.5),
		sp().MoveTo(10, 40).CubicTo(40, 10, 70, 70, 100, 40),
		sp().MoveTo(20, 60).CubicTo(110, 10, 10, 10, 100, 60),
		sp().AddCircle(60, 45, 30, geom.DirectionCW),
		sp().AddOval(geom.RectLTRB(15, 25, 105, 65), geom.DirectionCCW),
		sp().AddRoundRect(geom.RectLTRB(20, 20, 100, 70), 12, 8, geom.DirectionCW),
		// multiple contours incl. an empty one in the middle
		sp().MoveTo(10, 10).LineTo(40, 30).MoveTo(200, 200).MoveTo(60, 10).LineTo(90, 30).LineTo(75, 50).Close(),
		// degenerate contours
		sp().MoveTo(50, 40).Close(),
		sp().MoveTo(50, 40).LineTo(50, 40),
	}
}

// measureFractions and measureSpans are the positions this probe samples along each contour. They are package-level
// because the reference capture must sample exactly the same ones for the frozen walk to line up with what the probe
// reads back (see ref_test.go).
var (
	measureFractions = []float32{-0.25, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, 1.25}
	measureSpans     = [][2]float32{{0.2, 0.8}, {0, 1}, {0.37, 0.37}}
)

func TestPathMeasureProbe(t *testing.T) {
	corpus := measureCorpus()
	fractions := measureFractions

	// Distance accumulation sums hundreds of float segment lengths whose dx*dx+dy*dy terms the C compiler may contract;
	// positions/tangents add interpolation and normalize (sqrt) on top. The corpus coordinate magnitude is ~200; scale
	// the agree() band by the contour length for lengths and by the coordinate magnitude for points.
	lenAgree := func(g, c float32) bool {
		return f32Eq(g, c) || agree(g, c, 8*math.Abs(float64(c))+1)
	}
	ptAgree := func(g, c geom.Point) bool {
		return (f32Eq(g.X, c.X) || agree(g.X, c.X, 2048)) && (f32Eq(g.Y, c.Y) || agree(g.Y, c.Y, 2048))
	}

	for si, spec := range corpus {
		gp := spec.BuildGo()
		for _, forceClosed := range []bool{false, true} {
			for _, resScale := range []float32{1, 0.3, 4.7} {
				what := fmt.Sprintf("corpus[%d] forceClosed=%v res=%v", si, forceClosed, resScale)
				cContours := refPathMeasure(spec, forceClosed, resScale)
				gm := contour.NewPathMeasure(gp, forceClosed, resScale)

				for contourIdx := 0; ; contourIdx++ {
					cw := fmt.Sprintf("%s contour %d", what, contourIdx)
					if contourIdx >= len(cContours) {
						t.Errorf("%s: port has more contours than the oracle (%d)", cw, len(cContours))
						break
					}
					cc := &cContours[contourIdx]
					cLen := cc.length
					gLen := gm.Length()
					if !lenAgree(gLen, cLen) {
						t.Errorf("%s: length go %v c %v", cw, gLen, cLen)
					}
					if gm.IsClosed() != cc.isClosed {
						t.Errorf("%s: isClosed go %v c %v", cw, gm.IsClosed(), cc.isClosed)
					}

					if cLen > 0 {
						for fi, f := range fractions {
							d := cLen * f
							s := cc.posTan[fi]
							cPos, cTan, cOK := s.pos, s.tan, s.ok
							gPos, gTan, gOK := gm.GetPosTan(d, true, true)
							if cOK != gOK {
								t.Errorf("%s: posTan(%v) ok go %v c %v", cw, d, gOK, cOK)
								continue
							}
							if cOK && (!ptAgree(gPos, cPos) || !ptAgree(gTan, cTan)) {
								t.Errorf("%s: posTan(%v) go %v/%v c %v/%v", cw, d, gPos, gTan, cPos, cTan)
							}
						}

						var gMat geom.Matrix
						cMat, cOK := cc.matrix, cc.matrixOK
						gOK := gm.GetMatrix(cLen/3, &gMat, contour.GetPosAndTanMatrixFlag)
						if cOK != gOK {
							t.Errorf("%s: getMatrix ok go %v c %v", cw, gOK, cOK)
						} else if cOK {
							c9 := cMat.As9()
							g9 := gMat.As9()
							for i := range c9 {
								if !f32Eq(g9[i], c9[i]) && !agree(g9[i], c9[i], 2048) {
									t.Errorf("%s: matrix[%d] go %v c %v", cw, i, g9[i], c9[i])
									break
								}
							}
						}

						// segment extraction: interior span, full span, and a zero-length span into a non-empty dst
						// (the dash zero-length on-interval rule)
						for spi, span := range measureSpans {
							gDst := &path.Path{}
							gDst.MoveTo(-5, -5)
							cSeg := cc.segments[spi]
							segCOK := cSeg.ok
							segGOK := gm.GetSegment(cLen*span[0], cLen*span[1], gDst, true)
							if segCOK != segGOK {
								t.Errorf("%s: getSegment(%v) ok go %v c %v", cw, span, segGOK, segCOK)
							} else if segCOK {
								cPts := cSeg.points
								if gDst.CountPoints() != len(cPts) {
									t.Errorf("%s: getSegment(%v) points go %d c %d",
										cw, span, gDst.CountPoints(), len(cPts))
								} else {
									gPts := make([]geom.Point, gDst.CountPoints())
									gDst.Points(gPts)
									for i := range gPts {
										if !ptAgree(gPts[i], cPts[i]) {
											t.Errorf("%s: getSegment(%v) point %d go %v c %v",
												cw, span, i, gPts[i], cPts[i])
											break
										}
									}
								}
							}
						}
					}

					// The frozen walk's length is the oracle's contour count, so running off its end is the same
					// disagreement NextContour() used to report.
					cNext := contourIdx+1 < len(cContours)
					gNext := gm.NextContour()
					if cNext != gNext {
						t.Errorf("%s: nextContour go %v c %v", cw, gNext, cNext)
						break
					}
					if !cNext {
						break
					}
				}
			}
		}
	}
	fmt.Println("path measure probe done")
}
