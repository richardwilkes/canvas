// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Hermetic tests for the triangulator: a corpus of paths that historically crashed, hung, or asserted in triangulators
// of this kind (these pass by not panicking or hanging), plus differential rasterization checks the emitted triangles,
// rasterized by point-in-triangle sampling (with barycentric coverage interpolation for the AA mesh), must reproduce
// the CPU rasterizer's fill of the same path within an edge-pixel budget.

package gl

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
)

// triangulateForTest runs the non-AA triangulator and decodes the emitted positions.
func triangulateForTest(tb testing.TB, p *path.Path, tol float32, clip geom.Rect) (pts []geom.Point, isLinear bool) {
	tb.Helper()
	var alloc cpuVertexAllocator
	count := triPathToTriangles(p, tol, clip, &alloc, &isLinear)
	data := alloc.detachVertexData()
	if count*8 != len(data) {
		tb.Fatalf("vertex count %d does not match %d bytes", count, len(data))
	}
	pts = make([]geom.Point, count)
	for i := range pts {
		pts[i].X = math.Float32frombits(binary.LittleEndian.Uint32(data[i*8:]))
		pts[i].Y = math.Float32frombits(binary.LittleEndian.Uint32(data[i*8+4:]))
	}
	return pts, isLinear
}

// aaVert is one decoded AA-mesh vertex.
type aaVert struct {
	pos      geom.Point
	coverage float32
}

// triangulateAAForTest runs the AA triangulator and decodes positions + coverage.
func triangulateAAForTest(tb testing.TB, p *path.Path, tol float32, clip geom.Rect) []aaVert {
	tb.Helper()
	var alloc cpuVertexAllocator
	count := triPathToAATriangles(p, tol, clip, &alloc)
	data := alloc.detachVertexData()
	if count*12 != len(data) {
		tb.Fatalf("vertex count %d does not match %d bytes", count, len(data))
	}
	verts := make([]aaVert, count)
	for i := range verts {
		verts[i].pos.X = math.Float32frombits(binary.LittleEndian.Uint32(data[i*12:]))
		verts[i].pos.Y = math.Float32frombits(binary.LittleEndian.Uint32(data[i*12+4:]))
		verts[i].coverage = math.Float32frombits(binary.LittleEndian.Uint32(data[i*12+8:]))
	}
	return verts
}

// rasterizeTrianglesCoverage rasterizes triangles over a w×h grid by pixel-center sampling with barycentric coverage
// interpolation, keeping the max coverage per pixel (the triangles partition the region, so max == the mesh's value
// everywhere it is defined).
func rasterizeTrianglesCoverage(pos []geom.Point, cov []float32, w, h int) []float32 {
	buf := make([]float32, w*h)
	for i := 0; i+2 < len(pos); i += 3 {
		p0, p1, p2 := pos[i], pos[i+1], pos[i+2]
		area := (float64(p1.X)-float64(p0.X))*(float64(p2.Y)-float64(p0.Y)) -
			(float64(p1.Y)-float64(p0.Y))*(float64(p2.X)-float64(p0.X))
		if area == 0 {
			continue
		}
		minX := int(math.Floor(math.Min(float64(p0.X), math.Min(float64(p1.X), float64(p2.X)))))
		maxX := int(math.Ceil(math.Max(float64(p0.X), math.Max(float64(p1.X), float64(p2.X)))))
		minY := int(math.Floor(math.Min(float64(p0.Y), math.Min(float64(p1.Y), float64(p2.Y)))))
		maxY := int(math.Ceil(math.Max(float64(p0.Y), math.Max(float64(p1.Y), float64(p2.Y)))))
		minX = max(minX, 0)
		minY = max(minY, 0)
		maxX = min(maxX, w-1)
		maxY = min(maxY, h-1)
		for y := minY; y <= maxY; y++ {
			py := float64(y) + 0.5
			for x := minX; x <= maxX; x++ {
				px := float64(x) + 0.5
				w0 := (float64(p1.X)-px)*(float64(p2.Y)-py) - (float64(p1.Y)-py)*(float64(p2.X)-px)
				w1 := (float64(p2.X)-px)*(float64(p0.Y)-py) - (float64(p2.Y)-py)*(float64(p0.X)-px)
				w2 := (float64(p0.X)-px)*(float64(p1.Y)-py) - (float64(p0.Y)-py)*(float64(p1.X)-px)
				if area > 0 {
					if w0 < 0 || w1 < 0 || w2 < 0 {
						continue
					}
				} else if w0 > 0 || w1 > 0 || w2 > 0 {
					continue
				}
				c := float32(1)
				if cov != nil {
					c = float32((w0*float64(cov[i]) + w1*float64(cov[i+1]) +
						w2*float64(cov[i+2])) / area)
				}
				if c > buf[y*w+x] {
					buf[y*w+x] = c
				}
			}
		}
	}
	return buf
}

// diffTrianglesVsRaster triangulates p and compares the triangle coverage against the CPU rasterizer's non-AA fill over
// bounds, budgeting edge-pixel disagreement.
func diffTrianglesVsRaster(t *testing.T, p *path.Path, bounds geom.IRect, budget int, tag string) {
	t.Helper()
	w := int(bounds.Width())
	h := int(bounds.Height())
	identity := geom.IdentityMatrix()
	tol := scaleToleranceToSrc(pathUtilsDefaultTolerance, &identity, p.Bounds())
	pts, _ := triangulateForTest(t, p, tol, bounds.ToRect())
	got := rasterizeTrianglesCoverage(pts, nil, w, h)
	mask := raster.FillPathToMask(p, bounds, false)
	mismatches := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gpuIn := got[y*w+x] > 0.5
			cpuIn := mask.Image[y*int(mask.RowBytes)+x] != 0
			if gpuIn != cpuIn {
				mismatches++
			}
		}
	}
	if mismatches > budget {
		t.Fatalf("%s: %d pixels disagree with the CPU fill (budget %d)", tag, mismatches, budget)
	}
	t.Logf("%s: %d mismatching pixels (budget %d)", tag, mismatches, budget)
}

func testStarPath() *path.Path {
	p := &path.Path{}
	p.MoveTo(50.3, 10.2)
	p.LineTo(70.1, 90.4)
	p.LineTo(10.2, 40.3)
	p.LineTo(90.4, 40.3)
	p.LineTo(30.1, 90.2)
	p.Close()
	return p
}

func TestTriangulatorFillDifferential(t *testing.T) {
	bounds := geom.IRectWH(100, 100)

	star := testStarPath()
	for _, ft := range []path.FillType{
		path.FillWinding, path.FillEvenOdd,
		path.FillInverseWinding, path.FillInverseEvenOdd,
	} {
		p := star.Clone()
		p.SetFillType(ft)
		diffTrianglesVsRaster(t, p, bounds, 3*(100+100), fmt.Sprintf("star fill %d", ft))
	}

	// Overlapping rects: winding vs even-odd of the overlap region.
	rects := &path.Path{}
	rects.AddRect(geom.RectXYWH(10, 10, 50, 50), geom.DirectionCW)
	rects.AddRect(geom.RectXYWH(30, 30, 50, 50), geom.DirectionCW)
	diffTrianglesVsRaster(t, rects, bounds, 3*(100+100), "rects winding")
	eo := rects.Clone()
	eo.SetFillType(path.FillEvenOdd)
	diffTrianglesVsRaster(t, eo, bounds, 3*(100+100), "rects even-odd")

	// Curved geometry: a circle and a cubic blob (conics and cubics linearize).
	circle := &path.Path{}
	circle.AddCircle(50, 50, 35, geom.DirectionCW)
	diffTrianglesVsRaster(t, circle, bounds, 3*(100+100), "circle")

	blob := &path.Path{}
	blob.MoveTo(20, 50)
	blob.CubicTo(10, 10, 90, 15, 80, 50)
	blob.CubicTo(75, 85, 25, 80, 20, 50)
	blob.Close()
	diffTrianglesVsRaster(t, blob, bounds, 3*(100+100), "cubic blob")
}

func TestTriangulatorIsLinear(t *testing.T) {
	bounds := geom.RectWH(100, 100)
	var isLinear bool
	var alloc cpuVertexAllocator
	triPathToTriangles(testStarPath(), 0.25, bounds, &alloc, &isLinear)
	alloc.detachVertexData()
	if !isLinear {
		t.Fatal("a polygonal path must report isLinear")
	}
	circle := &path.Path{}
	circle.AddCircle(50, 50, 35, geom.DirectionCW)
	var alloc2 cpuVertexAllocator
	triPathToTriangles(circle, 0.25, bounds, &alloc2, &isLinear)
	alloc2.detachVertexData()
	if isLinear {
		t.Fatal("a conic path must not report isLinear")
	}
}

func TestAATriangulatorCoverage(t *testing.T) {
	const w, h = 100, 100
	bounds := geom.IRectWH(w, h)

	for _, tc := range []struct {
		p    *path.Path
		name string
	}{
		{name: "star", p: testStarPath()},
		{name: "circle", p: (&path.Path{}).AddCircle(50, 50, 35, geom.DirectionCW)},
	} {
		verts := triangulateAAForTest(t, tc.p, pathUtilsDefaultTolerance, bounds.ToRect())
		if len(verts) == 0 {
			t.Fatalf("%s: AA triangulation produced no vertices", tc.name)
		}
		pos := make([]geom.Point, len(verts))
		cov := make([]float32, len(verts))
		for i, v := range verts {
			if v.coverage < 0 || v.coverage > 1 {
				t.Fatalf("%s: coverage %g out of range", tc.name, v.coverage)
			}
			pos[i] = v.pos
			cov[i] = v.coverage
		}
		got := rasterizeTrianglesCoverage(pos, cov, w, h)
		mask := raster.FillPathToMask(tc.p, bounds, true)

		// Compare against the CPU analytic-AA fill: interior pixels (≥ 2px inside per the CPU mask neighborhood) must
		// be fully covered, far-exterior pixels empty, and the mean absolute delta over the whole grid small (the
		// one-pixel ramp differs from analytic coverage by design).
		var sumDelta float64
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				cpu := float64(mask.Image[y*int(mask.RowBytes)+x]) / 255
				gpu := float64(got[y*w+x])
				sumDelta += math.Abs(cpu - gpu)
				fullNeighborhood := true
				emptyNeighborhood := true
				for dy := -2; dy <= 2 && (fullNeighborhood || emptyNeighborhood); dy++ {
					for dx := -2; dx <= 2; dx++ {
						nx, ny := x+dx, y+dy
						if nx < 0 || ny < 0 || nx >= w || ny >= h {
							fullNeighborhood = false
							continue
						}
						v := mask.Image[ny*int(mask.RowBytes)+nx]
						if v != 255 {
							fullNeighborhood = false
						}
						if v != 0 {
							emptyNeighborhood = false
						}
					}
				}
				if fullNeighborhood && gpu < 0.99 {
					t.Fatalf("%s: interior pixel (%d,%d) coverage %g, want 1", tc.name, x, y, gpu)
				}
				if emptyNeighborhood && gpu > 0.01 {
					t.Fatalf("%s: exterior pixel (%d,%d) coverage %g, want 0", tc.name, x, y, gpu)
				}
			}
		}
		mean := sumDelta / float64(w*h)
		if mean > 0.02 {
			t.Fatalf("%s: mean AA coverage delta %.4f vs CPU (cap 0.02)", tc.name, mean)
		}
		t.Logf("%s: mean AA coverage delta %.4f", tc.name, mean)
	}
}

// TestTriangulatorCrashCorpus is a corpus of adversarial paths (adopted from a well-known triangulator test suite; a
// large ~90-line bit-exact quad path case is omitted). These pass by not panicking, hanging, or emitting a mismatched
// vertex count; the corpus paths intentionally contain NaN/infinity/huge coordinates and degenerate topologies.
func TestTriangulatorCrashCorpus(t *testing.T) {
	clip := geom.RectWH(800, 800)
	for i, build := range triangulatorCrashCorpus {
		p := &path.Path{}
		build(p)
		var alloc cpuVertexAllocator
		var isLinear bool
		count := triPathToTriangles(p, 0.25, clip, &alloc, &isLinear)
		data := alloc.detachVertexData()
		if count*8 != len(data) {
			t.Fatalf("case %d: count %d vs %d bytes", i, count, len(data))
		}
		var aaAlloc cpuVertexAllocator
		aaCount := triPathToAATriangles(p, 0.25, clip, &aaAlloc)
		aaData := aaAlloc.detachVertexData()
		if aaCount*12 != len(aaData) {
			t.Fatalf("case %d (AA): count %d vs %d bytes", i, aaCount, len(aaData))
		}
	}
}

var triangulatorCrashCorpus = []func(*path.Path){
	// Tests active edges made inactive by splitting; also an active edge list forced into an invalid ordering by
	// splitting.
	func(p *path.Path) {
		p.MoveTo(229.127044677734375, 67.34100341796875)
		p.LineTo(187.8097381591796875, -6.7729740142822265625)
		p.LineTo(171.411407470703125, 50.94266510009765625)
		p.LineTo(245.5253753662109375, 9.6253643035888671875)
		p.MoveTo(208.4683990478515625, 30.284009933471679688)
		p.LineTo(171.411407470703125, 50.94266510009765625)
		p.LineTo(187.8097381591796875, -6.7729740142822265625)
	},
	// Intersections which fall exactly on the current vertex, requiring a restart of the intersection checking.
	func(p *path.Path) {
		p.MoveTo(314.483551025390625, 486.246002197265625)
		p.LineTo(385.41949462890625, 532.8087158203125)
		p.LineTo(373.232879638671875, 474.05938720703125)
		p.LineTo(326.670166015625, 544.995361328125)
		p.MoveTo(349.951507568359375, 509.52734375)
		p.LineTo(373.232879638671875, 474.05938720703125)
		p.LineTo(385.41949462890625, 532.8087158203125)
	},
	// Tests active edges which are removed by splitting.
	func(p *path.Path) {
		p.MoveTo(343.107391357421875, 613.62176513671875)
		p.LineTo(426.632415771484375, 628.5740966796875)
		p.LineTo(392.3460693359375, 579.33544921875)
		p.LineTo(377.39373779296875, 662.86041259765625)
		p.MoveTo(384.869873046875, 621.097900390625)
		p.LineTo(392.3460693359375, 579.33544921875)
		p.LineTo(426.632415771484375, 628.5740966796875)
	},
	// Collinear edges merged in set_top(); also an intersection between the left and right enclosing edges which falls
	// above the current vertex.
	func(p *path.Path) {
		p.MoveTo(545.95751953125, 791.69854736328125)
		p.LineTo(612.05816650390625, 738.494140625)
		p.LineTo(552.4056396484375, 732.0209960937)
		p.LineTo(605.61004638671875, 798.12160034179688)
		p.MoveTo(579.00787353515625, 765.0578125)
		p.LineTo(552.4056396484375, 732.0209960937)
		p.LineTo(612.05816650390625, 738.494140625)
	},
	// Tests active edges which are made inactive by set_top().
	func(p *path.Path) {
		p.MoveTo(819.2725830078125, 751.77447509765625)
		p.LineTo(820.70904541015625, 666.933837890625)
		p.LineTo(777.57049560546875, 708.63592529296875)
		p.LineTo(862.4111328125, 710.0723876953125)
		p.MoveTo(819.99078369140625, 709.3541259765625)
		p.LineTo(777.57049560546875, 708.63592529296875)
		p.LineTo(820.70904541015625, 666.933837890625)
	},
	func(p *path.Path) {
		p.MoveTo(823.33209228515625, 749.052734375)
		p.LineTo(823.494873046875, 664.20013427734375)
		p.LineTo(780.9871826171875, 706.5450439453125)
		p.LineTo(865.8397216796875, 706.70782470703125)
		p.MoveTo(823.4134521484375, 706.6263427734375)
		p.LineTo(780.9871826171875, 706.5450439453125)
		p.LineTo(823.494873046875, 664.20013427734375)
	},
	func(p *path.Path) {
		p.MoveTo(954.862548828125, 562.8349609375)
		p.LineTo(899.32818603515625, 498.679443359375)
		p.LineTo(895.017578125, 558.52435302734375)
		p.LineTo(959.17315673828125, 502.990081787109375)
		p.MoveTo(927.0953369140625, 530.7572021484375)
		p.LineTo(895.017578125, 558.52435302734375)
		p.LineTo(899.32818603515625, 498.679443359375)
	},
	func(p *path.Path) {
		p.MoveTo(958.5330810546875, 547.35516357421875)
		p.LineTo(899.93109130859375, 485.989013671875)
		p.LineTo(898.54901123046875, 545.97308349609375)
		p.LineTo(959.9151611328125, 487.37109375)
		p.MoveTo(929.2320556640625, 516.67205810546875)
		p.LineTo(898.54901123046875, 545.97308349609375)
		p.LineTo(899.93109130859375, 485.989013671875)
	},
	func(p *path.Path) {
		p.MoveTo(389.8609619140625, 369.326873779296875)
		p.LineTo(470.6290283203125, 395.33697509765625)
		p.LineTo(443.250030517578125, 341.9478759765625)
		p.LineTo(417.239959716796875, 422.7159423828125)
		p.MoveTo(430.244964599609375, 382.3319091796875)
		p.LineTo(443.250030517578125, 341.9478759765625)
		p.LineTo(470.6290283203125, 395.33697509765625)
	},
	func(p *path.Path) {
		p.MoveTo(20, 20)
		p.LineTo(50, 80)
		p.LineTo(20, 80)
		p.MoveTo(80, 50)
		p.LineTo(50, 50)
		p.LineTo(20, 50)
	},
	func(p *path.Path) {
		p.MoveTo(257.19439697265625, 320.876617431640625)
		p.LineTo(190.113037109375, 320.58978271484375)
		p.LineTo(203.64404296875, 293.8145751953125)
		p.MoveTo(203.357177734375, 360.896026611328125)
		p.LineTo(216.88824462890625, 334.120819091796875)
		p.LineTo(230.41925048828125, 307.345611572265625)
	},
	// A degenerate segments case, where both upper and lower segments of a split edge must remain active.
	func(p *path.Path) {
		p.MoveTo(231.9331207275390625, 306.2012939453125)
		p.LineTo(191.4859161376953125, 306.04547119140625)
		p.LineTo(231.0659332275390625, 300.2642822265625)
		p.MoveTo(189.946807861328125, 302.072265625)
		p.LineTo(179.79705810546875, 294.859771728515625)
		p.LineTo(191.0016021728515625, 296.165679931640625)
		p.MoveTo(150.8942108154296875, 304.900146484375)
		p.LineTo(179.708892822265625, 297.849029541015625)
		p.LineTo(190.4742279052734375, 299.11895751953125)
	},
	// Handle the case where edge.dist(edge.fTop) != 0.0.
	func(p *path.Path) {
		p.MoveTo(0.0, 400.0)
		p.LineTo(138.0, 202.0)
		p.LineTo(0.0, 202.0)
		p.MoveTo(12.62693023681640625, 250.57464599609375)
		p.LineTo(8.13896942138671875, 254.556884765625)
		p.LineTo(-18.15641021728515625, 220.40203857421875)
		p.LineTo(-15.986493110656738281, 219.6513519287109375)
		p.MoveTo(36.931194305419921875, 282.485504150390625)
		p.LineTo(15.617521286010742188, 261.2901611328125)
		p.LineTo(10.3829498291015625, 252.565765380859375)
		p.LineTo(-16.165292739868164062, 222.646026611328125)
	},
	// A degenerate segments case which exercises inactive edges being made active by splitting.
	func(p *path.Path) {
		p.MoveTo(690.62127685546875, 509.25555419921875)
		p.LineTo(99.336181640625, 511.71405029296875)
		p.LineTo(708.362548828125, 512.4349365234375)
		p.LineTo(729.9940185546875, 516.3114013671875)
		p.LineTo(738.708984375, 518.76995849609375)
		p.LineTo(678.3463134765625, 510.0819091796875)
		p.LineTo(681.21795654296875, 504.81378173828125)
		p.MoveTo(758.52764892578125, 521.55963134765625)
		p.LineTo(719.1549072265625, 514.50372314453125)
		p.LineTo(689.59063720703125, 512.0628662109375)
		p.LineTo(679.78216552734375, 507.447845458984375)
	},
	// Tests vertices which become "orphaned" (no connected edges) after simplification.
	func(p *path.Path) {
		p.MoveTo(217.326019287109375, 166.4752960205078125)
		p.LineTo(226.279266357421875, 170.929473876953125)
		p.LineTo(234.3973388671875, 177.0623626708984375)
		p.LineTo(262.0921630859375, 188.746124267578125)
		p.MoveTo(196.23638916015625, 174.0722198486328125)
		p.LineTo(416.15277099609375, 180.138214111328125)
		p.LineTo(192.651947021484375, 304.0228271484375)
	},
	func(p *path.Path) {
		p.MoveTo(0.0, 0.0)
		p.LineTo(10000.0, 0.0)
		p.LineTo(0.0, -1.0)
		p.LineTo(10000.0, 0.000001)
		p.LineTo(0.0, -30.0)
	},
	// Reduction of Nebraska-StateSeal.svg: floating point error causes the same edge to be added to more than one poly
	// on the same side.
	func(p *path.Path) {
		p.MoveTo(170.8199920654296875, 491.86700439453125)
		p.LineTo(173.7649993896484375, 489.7340087890625)
		p.LineTo(174.1450958251953125, 498.545989990234375)
		p.LineTo(171.998992919921875, 500.88201904296875)
		p.MoveTo(168.2922515869140625, 498.66265869140625)
		p.LineTo(169.8589935302734375, 497.94500732421875)
		p.LineTo(172, 500.88299560546875)
		p.MoveTo(169.555267333984375, 490.70111083984375)
		p.LineTo(173.7649993896484375, 489.7340087890625)
		p.LineTo(170.82000732421875, 491.86700439453125)
	},
	// A shape with a vertex collinear to the right hand edge (messes up find_enclosing_edges).
	func(p *path.Path) {
		p.MoveTo(80, 20)
		p.LineTo(80, 60)
		p.LineTo(20, 60)
		p.MoveTo(80, 50)
		p.LineTo(80, 80)
		p.LineTo(20, 80)
	},
	// An edge that becomes collinear with *two* of its adjacent neighbor edges after splitting (reduction from
	// horner_ulp.svg).
	func(p *path.Path) {
		p.MoveTo(351.99298095703125, 348.23046875)
		p.LineTo(351.91876220703125, 347.33984375)
		p.LineTo(351.91876220703125, 346.1953125)
		p.LineTo(351.90313720703125, 347.734375)
		p.LineTo(351.90313720703125, 346.1328125)
		p.LineTo(351.87579345703125, 347.93359375)
		p.LineTo(351.87579345703125, 345.484375)
		p.LineTo(351.86407470703125, 347.7890625)
		p.LineTo(351.86407470703125, 346.2109375)
		p.LineTo(351.84844970703125, 347.63763427734375)
		p.LineTo(351.84454345703125, 344.19232177734375)
		p.LineTo(351.78204345703125, 346.9483642578125)
		p.LineTo(351.758636474609375, 347.18310546875)
		p.LineTo(351.75469970703125, 346.75)
		p.LineTo(351.75469970703125, 345.46875)
		p.LineTo(352.5546875, 345.46875)
		p.LineTo(352.55078125, 347.01953125)
		p.LineTo(351.75079345703125, 347.02313232421875)
		p.LineTo(351.74688720703125, 346.15203857421875)
		p.LineTo(351.74688720703125, 347.646148681640625)
		p.LineTo(352.5390625, 346.94140625)
		p.LineTo(351.73907470703125, 346.94268798828125)
		p.LineTo(351.73516845703125, 344.48565673828125)
		p.LineTo(352.484375, 346.73828125)
		p.LineTo(351.68438720703125, 346.7401123046875)
		p.LineTo(352.4765625, 346.546875)
		p.LineTo(351.67657470703125, 346.54937744140625)
		p.LineTo(352.47265625, 346.75390625)
		p.LineTo(351.67266845703125, 346.756622314453125)
		p.LineTo(351.66876220703125, 345.612091064453125)
	},
	// A path which contains out-of-range colinear intersections.
	func(p *path.Path) {
		p.MoveTo(0, 63.39080047607421875)
		p.LineTo(-0.70804601907730102539, 63.14350128173828125)
		p.LineTo(-7.8608899287380243391e-17, 64.14080047607421875)
		p.MoveTo(0, 64.14080047607421875)
		p.LineTo(44.285900115966796875, 64.14080047607421875)
		p.LineTo(0, 62.64080047607421875)
		p.MoveTo(21.434900283813476562, -0.24732701480388641357)
		p.LineTo(-0.70804601907730102539, 63.14350128173828125)
		p.LineTo(0.70804601907730102539, 63.6381988525390625)
	},
	// A path which results in infs and nans when conics are converted to quads.
	func(p *path.Path) {
		p.MoveTo(-2.20883e+37, -1.02892e+37)
		p.ConicTo(-2.00958e+38, -9.36107e+37, -1.7887e+38, -8.33215e+37, 0.707107)
		p.ConicTo(-1.56782e+38, -7.30323e+37, 2.20883e+37, 1.02892e+37, 0.707107)
		p.ConicTo(2.00958e+38, 9.36107e+37, 1.7887e+38, 8.33215e+37, 0.707107)
		p.ConicTo(1.56782e+38, 7.30323e+37, -2.20883e+37, -1.02892e+37, 0.707107)
	},
	// A quad which generates a huge number of points (>2B) when uniformly linearized. This should not hang or OOM.
	func(p *path.Path) {
		p.MoveTo(10, 0)
		p.LineTo(0, 0)
		p.QuadTo(10, 0, 0, 8315084722602508288)
	},
	// A path which hung during simplification: it produces an edge which is to the left of its own endpoints.
	func(p *path.Path) {
		p.MoveTo(0.75001740455627441406, 23.051967620849609375)
		p.LineTo(5.8471612930297851562, 22.731662750244140625)
		p.LineTo(10.749670028686523438, 22.253145217895507812)
		p.LineTo(13.115868568420410156, 22.180681228637695312)
		p.LineTo(15.418928146362304688, 22.340015411376953125)
		p.LineTo(17.654022216796875, 22.82159423828125)
		p.LineTo(19.81632232666015625, 23.715869903564453125)
		p.LineTo(40, 0)
		p.LineTo(5.5635203441547955577e-15, 0)
		p.LineTo(5.5635203441547955577e-15, 47)
		p.LineTo(-1.4210854715202003717e-14, 21.713298797607421875)
		p.LineTo(0.75001740455627441406, 21.694292068481445312)
		p.LineTo(0.75001740455627441406, 23.051967620849609375)
	},
	// Reduction from a historical regression that crashed splitting a zombie edge.
	func(p *path.Path) {
		p.MoveTo(0, 1.0927740941146660348e+24)
		p.LineTo(2.9333931225865729333e+32, 16476101)
		p.LineTo(1.0927731573659435417e+24, 1.0927740941146660348e+24)
		p.LineTo(1.0927740941146660348e+24, 3.7616281094287041715e-37)
		p.LineTo(1.0927740941146660348e+24, 1.0927740941146660348e+24)
		p.LineTo(1.3061803026169399536e-33, 1.0927740941146660348e+24)
		p.LineTo(4.7195362919941370727e-16, -8.4247545146051822591e+32)
	},
	// From a historical regression: crashed trying to merge a zombie edge.
	func(p *path.Path) {
		p.MoveTo(316.000579833984375, -4338355948977389568)
		p.LineTo(1.5069369808623501312e+20, 75180972320904708096.0)
		p.LineTo(1.5069369808623501312e+20, 75180972320904708096.0)
		p.LineTo(771.21014404296875, -4338355948977389568.0)
		p.LineTo(316.000579833984375, -4338355948977389568.0)
		p.MoveTo(354.208984375, -4338355948977389568.0)
		p.LineTo(773.00177001953125, -4338355948977389568.0)
		p.LineTo(1.5069369808623501312e+20, 75180972320904708096.0)
		p.LineTo(1.5069369808623501312e+20, 75180972320904708096.0)
		p.LineTo(354.208984375, -4338355948977389568.0)
	},
	// From a historical regression: hung repeatedly splitting alternate vertices.
	func(p *path.Path) {
		p.MoveTo(10, -1e+20)
		p.LineTo(11, 25000)
		p.LineTo(10, 25000)
		p.LineTo(11, 25010)
	},
	// Reduction from circular_arcs_stroke_and_fill_round GM which repeatedly splits on the opposite edge.
	func(p *path.Path) {
		p.MoveTo(16.25, 26.495191574096679688)
		p.LineTo(32.420825958251953125, 37.377376556396484375)
		p.LineTo(25.176382064819335938, 39.31851959228515625)
		p.MoveTo(20, 20)
		p.LineTo(28.847436904907226562, 37.940830230712890625)
		p.LineTo(25.17638397216796875, 39.31851959228515625)
	},
	// Reduction from a historical regression where an intersection is found below the bottom of both intersected edges.
	func(p *path.Path) {
		p.MoveTo(-2791476679359332352, 2608107002026524672)
		p.LineTo(0, 11.95427703857421875)
		p.LineTo(-2781824066779086848, 2599088532777598976)
		p.LineTo(-7772.6875, 7274)
	},
	// Reduction from a historical regression: a missed intersection causing bad ordering in the active edge list.
	func(p *path.Path) {
		p.MoveTo(-1.0662557646016024569e+23, 9.9621425197286319718e+22)
		p.LineTo(-121806400, 113805032)
		p.LineTo(-120098872, 112209680)
		p.LineTo(6.2832999862817380468e-36, 2.9885697364807128906)
	},
	// Reduction from a historical regression: collinear last vertex.
	func(p *path.Path) {
		p.MoveTo(2072553216, 0)
		p.LineTo(2072553216, 1)
		p.LineTo(2072553472, -13.5)
		p.LineTo(2072553216, 0)
		p.LineTo(2072553472, -6.5)
	},
	// Another reduction from a historical regression: two sequential collinear edges.
	func(p *path.Path) {
		p.MoveTo(2072553216, 0)
		p.LineTo(2072553216, 1)
		p.LineTo(2072553472, -13)
		p.LineTo(2072553216, 0)
		p.LineTo(2072553472, -6)
		p.LineTo(2072553472, -13)
	},
	// Reduction from a historical regression: three collinear edges discovered during the sanitize-contours pass,
	// before the vertices have been found coincident.
	func(p *path.Path) {
		p.MoveTo(32572426382475264, -3053391034974208)
		p.LineTo(521289856, -48865776)
		p.LineTo(130322464, -12215873)
		p.MoveTo(32572426382475264, -3053391034974208)
		p.LineTo(521289856, -48865776)
		p.LineTo(130322464, -12215873)
		p.MoveTo(32572426382475264, -3053391034974208)
		p.LineTo(32114477642022912, -3010462031544320)
		p.LineTo(32111784697528320, -3010209702215680)
	},
	// create_path_17: a simple concave path.
	func(p *path.Path) {
		p.MoveTo(20, 20)
		p.LineTo(80, 20)
		p.LineTo(30, 30)
		p.LineTo(20, 80)
	},
	// create_path_20: an intersection above the first vertex in the mesh.
	func(p *path.Path) {
		p.MoveTo(2822128.5, 235.026336669921875)
		p.LineTo(2819349.25, 235.3623504638671875)
		p.LineTo(-340558688, 23.83478546142578125)
		p.LineTo(-340558752, 25.510419845581054688)
		p.LineTo(-340558720, 27.18605804443359375)
	},
	// create_path_21: an intersection whose result is NaN (rounded-to-inf endpoint).
	func(p *path.Path) {
		p.MoveTo(1.7889142061167663539e+38, 39338463358011572224.0)
		p.LineTo(1647.4193115234375, -522.603515625)
		p.LineTo(1677.74560546875, -529.0028076171875)
		p.LineTo(1678.29541015625, -528.7847900390625)
		p.LineTo(1637.5167236328125, -519.79266357421875)
		p.LineTo(1647.4193115234375, -522.603515625)
	},
	// create_path_25/26: an edge collapse event which also collapses a neighbor / causes an edge to become collinear,
	// requiring its event to be removed.
	func(p *path.Path) {
		p.MoveTo(43.44110107421875, 148.15106201171875)
		p.LineTo(44.64471435546875, 148.16748046875)
		p.LineTo(46.35009765625, 147.403076171875)
		p.LineTo(46.45404052734375, 148.34906005859375)
		p.LineTo(45.0400390625, 148.54205322265625)
		p.LineTo(44.624053955078125, 148.9810791015625)
		p.LineTo(44.59405517578125, 149.16107177734375)
		p.LineTo(44.877044677734375, 149.62005615234375)
		p.LineTo(144.373016357421875, 68.8070068359375)
	},
	// create_path_27: results in non-finite points when stroked and beveled for AA.
	func(p *path.Path) {
		p.MoveTo(8.5027233009104409507e+37, 1.7503381025241130639e+37)
		p.LineTo(7.0923661737711584874e+37, 1.4600074517285415699e+37)
		p.LineTo(7.0848733446033294691e+37, 1.4584649744781838604e+37)
		p.LineTo(-2.0473916115129349496e+37, -4.2146796450364162012e+36)
		p.LineTo(2.0473912312177548811e+37, 4.2146815465123165435e+36)
	},
	// create_path_28: AA stroking produces intersection failures on bevelling.
	func(p *path.Path) {
		p.MoveTo(-7.5952312625177475154e+21, -2.6819185100266674911e+24)
		p.LineTo(1260.3787841796875, 1727.7947998046875)
		p.LineTo(1260.5567626953125, 1728.0386962890625)
		p.LineTo(1.1482511310557754163e+21, 4.054538502765980051e+23)
		p.LineTo(-7.5952312625177475154e+21, -2.6819185100266674911e+24)
	},
	// create_path_31: vertices which become infinite on AA stroking.
	func(p *path.Path) {
		p.MoveTo(2.0257809259190991347e+36, -1244080640)
		p.ConicTo(2.0257809259190991347e+36, -1244080640,
			2.0257809259190991347e+36, 0.10976474732160568237, 0.70710676908493041992)
		p.LineTo(-10036566016, -1954718402215936)
		p.ConicTo(-1.1375507718551896064e+20, -1954721086570496,
			10036566016, -1954721086570496, 0.70710676908493041992)
	},
	// create_path_38: reduction from a historical regression.
	func(p *path.Path) {
		p.MoveTo(14.400531768798828125, 17.711114883422851562)
		p.LineTo(14.621990203857421875, 171563104293879808)
		p.LineTo(14.027951240539550781, 872585759381520384)
		p.LineTo(14.0216827392578125, 872665817571917824)
		p.LineTo(7.699314117431640625, -3417320793833472)
		p.MoveTo(11.606547355651855469, 17.40966796875)
		p.LineTo(7642114886926860288, 21.08358001708984375)
		p.LineTo(11.606547355651855469, 21.08358001708984375)
	},
	// create_path_41: a "missing" intersection requiring the active edge list to go out-of-order.
	func(p *path.Path) {
		p.MoveTo(72154931603311689728.0, 330.95965576171875)
		p.LineTo(24053266013925408768.0, 78.11376953125)
		p.LineTo(1.2031099003292404941e+20, 387.168731689453125)
		p.LineTo(68859835992355373056.0, 346.55047607421875)
		p.LineTo(76451708695451009024.0, 337.780029296875)
		p.MoveTo(-20815817797613387776.0, 18065700622522384384.0)
		p.LineTo(-72144121204987396096.0, 142.855804443359375)
		p.LineTo(72144121204987396096.0, 325.184783935546875)
		p.LineTo(1.2347242901040791552e+20, 18065700622522384384.0)
	},
	// create_path_43: edges collinear from one side but not the other.
	func(p *path.Path) {
		p.MoveTo(307316821852160, -28808363114496)
		p.LineTo(307165222928384, -28794154909696)
		p.LineTo(307013691113472, -28779948802048)
		p.LineTo(306862159298560, -28765744791552)
		p.LineTo(306870313025536, -28766508154880)
		p.LineTo(307049695019008, -28783327313920)
		p.LineTo(307408660332544, -28816974020608)
	},
	// create_path_44: reduction from a historical regression.
	func(p *path.Path) {
		p.MoveTo(114.4606170654296875, 186.443878173828125)
		p.LineTo(91.5394744873046875, 185.4189453125)
		p.LineTo(306.45538330078125, 3203.986083984375)
		p.MoveTo(16276206965409972224.0, 815.59393310546875)
		p.LineTo(-3.541605062372533207e+20, 487.7236328125)
		p.LineTo(-3.541605062372533207e+20, 168.204071044921875)
		p.LineTo(16276206965409972224.0, 496.07427978515625)
		p.MoveTo(-3.541605062372533207e+20, 167.00958251953125)
		p.LineTo(-3.541605062372533207e+20, 488.32086181640625)
		p.LineTo(16276206965409972224.0, 816.78839111328125)
		p.LineTo(16276206965409972224.0, 495.47705078125)
	},
	// create_path_45: reduction from a historical regression.
	func(p *path.Path) {
		p.MoveTo(706471854080, 379003666432)
		p.LineTo(706503180288, 379020443648)
		p.LineTo(706595717120, 379070087168)
		p.LineTo(706626060288, 379086372864)
		p.LineTo(706656141312, 379102527488)
		p.LineTo(706774171648, 379165835264)
		p.LineTo(706803073024, 379181334528)
		p.LineTo(706831712256, 379196702720)
		p.LineTo(706860154880, 379211939840)
		p.LineTo(706888335360, 379227078656)
		p.LineTo(706916253696, 379242053632)
		p.LineTo(706956820480, 379263811584)
		p.LineTo(706929098752, 379248934912)
		p.LineTo(706901114880, 379233927168)
		p.LineTo(706872934400, 379218821120)
		p.LineTo(706844491776, 379203551232)
		p.LineTo(706815787008, 379188183040)
		p.LineTo(706786885632, 379172651008)
		p.LineTo(706757722112, 379156987904)
		p.LineTo(706728296448, 379141226496)
		p.LineTo(706698608640, 379125301248)
		p.LineTo(706668724224, 379109244928)
		p.LineTo(706638577664, 379093090304)
		p.LineTo(706608168960, 379076771840)
		p.LineTo(706484174848, 379010252800)
	},
	// create_path_46: inf generated by intersections causes NaN in subsequent intersections.
	func(p *path.Path) {
		p.MoveTo(1.0321827899075254821e+37, -5.1199920965387697886e+37)
		p.LineTo(-1.0321827899075254821e+37, 5.1199920965387697886e+37)
		p.LineTo(-1.0425214946728668754e+37, 4.5731834042267216669e+37)
		p.MoveTo(-9.5077331762291841872e+36, 8.1304868292377430302e+37)
		p.LineTo(9.5077331762291841872e+36, -8.1304868292377430302e+37)
		p.LineTo(1.0795449417808426232e+37, 1.2246856113744539311e+37)
		p.MoveTo(-165.8018341064453125, -44.859375)
		p.LineTo(-9.558702871563160835e+36, -7.9814405281448285475e+37)
		p.LineTo(-9.4147814283168490381e+36, -8.3935116522790983488e+37)
	},
}

// TestTriangulatorFuzzDifferential is the differential fuzz slice: seeded random polygons across all fill types,
// compared against the CPU rasterizer.
func TestTriangulatorFuzzDifferential(t *testing.T) {
	rng := newTestLCG(0x5EED)
	bounds := geom.IRectWH(96, 96)
	fillTypes := []path.FillType{
		path.FillWinding, path.FillEvenOdd, path.FillInverseWinding,
		path.FillInverseEvenOdd,
	}
	for trial := 0; trial < 60; trial++ {
		p := &path.Path{}
		n := 3 + int(rng.next()%9)
		for i := 0; i < n; i++ {
			x := float32(rng.next()%9200)/100 + 2
			y := float32(rng.next()%9200)/100 + 2
			if i == 0 {
				p.MoveTo(x, y)
			} else {
				p.LineTo(x, y)
			}
		}
		p.Close()
		p.SetFillType(fillTypes[trial%len(fillTypes)])
		diffTrianglesVsRaster(t, p, bounds, 4*(96+96), "fuzz trial")
	}
}

// testLCG is a tiny deterministic generator for the fuzz corpus.
type testLCG struct{ state uint64 }

func newTestLCG(seed uint64) *testLCG { return &testLCG{state: seed} }

func (g *testLCG) next() uint64 {
	g.state = g.state*6364136223846793005 + 1442695040888963407
	return g.state >> 33
}
