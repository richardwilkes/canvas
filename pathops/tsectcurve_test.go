// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The end-to-end drivers for the tSect conic/cubic curve-vs-curve solver, covering conic/quad, conic/conic, cubic/quad,
// cubic/conic, and cubic/cubic pairs. Every reported intersection must be a valid touch point (the two curves' points
// at the reported T pair coincide within approximatelyEqual, and equal the recorded intersection point); coincident
// corpora must report at least two intersections. Points are doubles; literals that were originally float32 are widened
// through float32 via f32d to reproduce that precision exactly. cu4/cp3/dq/f32d live in the sibling _test.go files.

package pathops

import (
	"fmt"
	"testing"
)

func sprintfIdx(name string, i int) string     { return fmt.Sprintf("%s[%d]", name, i) }
func sprintfPair(name string, a, b int) string { return fmt.Sprintf("%s[%d,%d]", name, a, b) }

// assertValidTouch is the per-intersection validity check shared by all the drivers above: xy1≈xy2, and (when checkPt)
// both ≈ the recorded intersection point.
func assertValidTouch(t *testing.T, in *intersections, c1, c2 interface{ ptAtT(float64) dPoint }, checkPt bool, ctx string) {
	t.Helper()
	for pt := 0; pt < in.usedCount(); pt++ {
		xy1 := c1.ptAtT(in.tVal(0, pt))
		xy2 := c2.ptAtT(in.tVal(1, pt))
		if !xy1.approximatelyEqual(xy2) {
			t.Fatalf("%s pt %d: t1=%g (%g,%g) t2=%g (%g,%g)", ctx, pt,
				in.tVal(0, pt), xy1.x, xy1.y, in.tVal(1, pt), xy2.x, xy2.y)
		}
		if checkPt {
			iPt := in.pt(pt)
			if !xy1.approximatelyEqual(iPt) || !xy2.approximatelyEqual(iPt) {
				t.Fatalf("%s pt %d: iPt=(%g,%g) xy1=(%g,%g) xy2=(%g,%g)", ctx, pt,
					iPt.x, iPt.y, xy1.x, xy1.y, xy2.x, xy2.y)
			}
		}
	}
}

// ---- conic / quad ----

type conicQuadCase struct {
	conic  [3]dPoint
	weight float32
	quad   dQuad
}

var conicQuadTests = []conicQuadCase{
	{
		conic: cp3(0, -1.8135968446731567, 0, -1.0033817291259766, -0.0073835160583257675, 0), weight: 2.26585215e+11,
		quad: dq(0, -1.0000113248825073, -2.4824290449032560e-05, -1.0000115633010864, -0.0073835160583257675, 0),
	},
	{
		conic: cp3(494.348663, 224.583771, 494.365143, 224.633194, 494.376404, 224.684067), weight: 0.998645842,
		quad: dq(494.30481, 224.474213, 494.334961, 224.538284, 494.355774, 224.605927),
	},
	{
		conic: cp3(494.348663, 224.583771, 494.365143, 224.633194, 494.376404, 224.684067), weight: 0.998645842,
		quad: dq(f32d(494.355774), f32d(224.605927), f32d(494.363708), f32d(224.631714), f32d(494.370148), f32d(224.657471)),
	},
}

func TestPathOpsConicQuadIntersection(t *testing.T) {
	for index, tc := range conicQuadTests {
		conic := mkConic(tc.conic, tc.weight)
		in := newIntersections()
		in.intersectConicQuad(conic, tc.quad)
		assertValidTouch(t, in, conic, tc.quad, false, sprintfIdx("conicQuad", index))
	}
}

// ---- conic / conic ----

type conicCase struct {
	pts    [3]dPoint
	weight float32
}

var conicIntersectTestSet = []conicCase{
	{pts: cp3(306.588013, -227.983994, 212.464996, -262.242004, 95.5512009, 58.9763985), weight: 0.707107008},
	{pts: cp3(377.218994, -141.981003, 40.578701, -201.339996, 23.1854992, -102.697998), weight: 0.707107008},
	{pts: cp3(5.1114602088928223, 628.77813720703125, 10.834027290344238, 988.964111328125,
		163.40835571289062, 988.964111328125), weight: 0.72944212},
	{pts: cp3(163.40835571289062, 988.964111328125, 5, 988.964111328125, 5, 614.7423095703125), weight: 0.707106769},
	{pts: cp3(11.17222976684570312, -8.103978157043457031, 22.91432571411132812, -10.37866020202636719,
		23.7764129638671875, -7.725424289703369141), weight: 1.00862849},
	{pts: cp3(-1.545085430145263672, -4.755282402038574219, 22.23132705688476562, -12.48070907592773438,
		23.7764129638671875, -7.725427150726318359), weight: 0.707106769},
	{pts: cp3(-4, 1, -4, 5, 0, 5), weight: 0.707106769},
	{pts: cp3(-3, 4, -3, 1, 0, 1), weight: 0.707106769},
	{pts: cp3(0, 0, 0, 1, 1, 1), weight: 0.5},
	{pts: cp3(1, 0, 0, 0, 0, 1), weight: 0.5},
}

func conicOneOff(t *testing.T, outer, inner int) {
	c1 := mkConic(conicIntersectTestSet[outer].pts, conicIntersectTestSet[outer].weight)
	c2 := mkConic(conicIntersectTestSet[inner].pts, conicIntersectTestSet[inner].weight)
	in := newIntersections()
	in.intersectConicConic(c1, c2)
	assertValidTouch(t, in, c1, c2, true, sprintfPair("conicConic", outer, inner))
}

func TestPathOpsConicIntersectionOneOff(t *testing.T) { conicOneOff(t, 0, 1) }

func TestPathOpsConicIntersection(t *testing.T) {
	for outer := 0; outer < len(conicIntersectTestSet)-1; outer++ {
		for inner := outer + 1; inner < len(conicIntersectTestSet); inner++ {
			conicOneOff(t, outer, inner)
		}
	}
}

// ---- cubic / quad ----

type quadCubicCase struct {
	cubic [4]dPoint
	quad  dQuad
}

var quadCubicTests = []quadCubicCase{
	{
		cubic: cu4(945.08099365234375, 747.1619873046875, 982.5679931640625, 747.1619873046875,
			1013.6290283203125, 719.656005859375, 1019.1910400390625, 683.72601318359375),
		quad: dq(945, 747, 976.0660400390625, 747, 998.03302001953125, 725.03302001953125),
	},
	{
		cubic: cu4(778, 14089, 778, 14091.208984375, 776.20916748046875, 14093, 774, 14093),
		quad:  dq(778, 14089, 777.99957275390625, 14090.65625, 776.82843017578125, 14091.828125),
	},
	{
		cubic: cu4(1020.08099, 672.161987, 1020.08002, 630.73999, 986.502014, 597.161987, 945.080994, 597.161987),
		quad:  dq(1020, 672, 1020, 640.93396, 998.03302, 618.96698),
	},
	{
		cubic: cu4(778, 14089, 778, 14091.208984375, 776.20916748046875, 14093, 774, 14093),
		quad:  dq(778, 14089, 777.99957275390625, 14090.65625, 776.82843017578125, 14091.828125),
	},
	{
		cubic: cu4(1110, 817, f32d(1110.55225), 817, 1111, f32d(817.447693), 1111, 818),
		quad:  dq(f32d(1110.70715), f32d(817.292908), f32d(1110.41406), f32d(817.000122), 1110, 817),
	},
	{
		cubic: cu4(1110, 817, f32d(1110.55225), 817, 1111, f32d(817.447693), 1111, 818),
		quad:  dq(1111, 818, f32d(1110.99988), f32d(817.585876), f32d(1110.70715), f32d(817.292908)),
	},
	{
		cubic: cu4(55, 207, 52.238574981689453, 207, 50, 204.76142883300781, 50, 202),
		quad:  dq(55, 207, 52.929431915283203, 206.99949645996094, 51.464466094970703, 205.53553771972656),
	},
	{
		cubic: cu4(49, 47, 49, 74.614250183105469, 26.614250183105469, 97, -1, 97),
		quad: dq(-8.659739592076221e-015, 96.991401672363281, 20.065492630004883, 96.645187377929688,
			34.355339050292969, 82.355339050292969),
	},
	{
		cubic: cu4(10, 234, 10, 229.58172607421875, 13.581720352172852, 226, 18, 226),
		quad:  dq(18, 226, 14.686291694641113, 226, 12.342399597167969, 228.3424072265625),
	},
	{
		cubic: cu4(10, 234, 10, 229.58172607421875, 13.581720352172852, 226, 18, 226),
		quad:  dq(12.342399597167969, 228.3424072265625, 10, 230.68629455566406, 10, 234),
	},
}

func TestPathOpsCubicQuadIntersection(t *testing.T) {
	for index, tc := range quadCubicTests {
		cubic := mkCubic(tc.cubic)
		in := newIntersections()
		in.intersectCubicQuad(cubic, tc.quad)
		assertValidTouch(t, in, cubic, tc.quad, false, sprintfIdx("cubicQuad", index))
	}
}

// ---- cubic / conic ----

type cubicConicCase struct {
	cubic  [4]dPoint
	conic  [3]dPoint
	weight float32
}

var cubicConicTests = []cubicConicCase{
	{
		cubic: cu4(188.60000610351562, 2041.5999755859375, 188.60000610351562, 2065.39990234375,
			208, 2084.800048828125, 231.80000305175781, 2084.800048828125),
		conic: cp3(231.80000305175781, 2084.800048828125, 188.60000610351562, 2084.800048828125,
			188.60000610351562, 2041.5999755859375), weight: 0.707107008,
	},
	{
		cubic: cu4(231.80000305175781, 2084.800048828125, 255.60000610351562, 2084.800048828125,
			275, 2065.39990234375, 275, 2041.5999755859375),
		conic:  cp3(275, 2041.5999755859375, 275, 2084.800048828125, 231.80000305175781, 2084.800048828125),
		weight: 0.707107008,
	},
}

func TestPathOpsCubicConicIntersection(t *testing.T) {
	for index, tc := range cubicConicTests {
		cubic := mkCubic(tc.cubic)
		conic := mkConic(tc.conic, tc.weight)
		in := newIntersections()
		in.intersectCubicConic(cubic, conic)
		assertValidTouch(t, in, cubic, conic, false, sprintfIdx("cubicConic", index))
	}
}

// ---- cubic / cubic ----

var cubicIntersectTestSet = [][4]dPoint{
	cu4(0, 0, 0, 1, 1, 1, 1, 0),
	cu4(1, 0, 0, 0, 0, 1, 1, 1),
	cu4(0, 1, 4, 5, 1, 0, 5, 3),
	cu4(0, 1, 3, 5, 1, 0, 5, 4),
	cu4(0, 1, 1, 6, 1, 0, 1, 0),
	cu4(0, 1, 0, 1, 1, 0, 6, 1),
	cu4(0, 1, 3, 4, 1, 0, 5, 1),
	cu4(0, 1, 1, 5, 1, 0, 4, 3),
	cu4(0, 1, 1, 2, 1, 0, 6, 1),
	cu4(0, 1, 1, 6, 1, 0, 2, 1),
	cu4(0, 1, 0, 5, 1, 0, 4, 0),
	cu4(0, 1, 0, 4, 1, 0, 5, 0),
	cu4(0, 1, 3, 4, 1, 0, 3, 0),
	cu4(0, 1, 0, 3, 1, 0, 4, 3),
	cu4(0, 0, 1, 2, 3, 4, 4, 4),
	cu4(0, 0, 1, 2, 3, 4, 4, 4),
	cu4(4, 4, 3, 4, 1, 2, 0, 0),
	cu4(0, 1, 2, 3, 1, 0, 1, 0),
	cu4(0, 1, 0, 1, 1, 0, 3, 2),
	cu4(0, 2, 0, 1, 1, 0, 1, 0),
	cu4(0, 1, 0, 1, 2, 0, 1, 0),
	cu4(0, 1, 0, 2, 1, 0, 1, 0),
	cu4(0, 1, 0, 1, 1, 0, 2, 0),
	cu4(0, 1, 1, 6, 1, 0, 2, 0),
	cu4(0, 1, 0, 2, 1, 0, 6, 1),
	cu4(0, 1, 5, 6, 1, 0, 1, 0),
	cu4(0, 1, 0, 1, 1, 0, 6, 5),
	cu4(95.837747722788592, 45.025976907939643, 16.564570095652982, 0.72959763963222402,
		63.209855865319199, 68.047528419665767, 57.640240647662544, 59.524565264361243),
	cu4(51.593891741518817, 38.53849970667553, 62.34752929878772, 74.924924725166022,
		74.810149322641152, 34.17966562983564, 29.368398119401373, 94.66719277886078),
	cu4(39.765160968417838, 33.060396198677083, 5.1922921581157908, 66.854301452103215,
		31.619281802149157, 25.269248720849514, 81.541621071073038, 70.025341524754353),
	cu4(46.078911165743556, 48.259962651999651, 20.24450549867214, 49.403916182650214,
		0.26325131778756683, 24.46489805563581, 15.915006546264051, 83.515023059917155),
	cu4(65.454505973241524, 93.881892270353575, 45.867360264932437, 92.723972719499827,
		2.1464054482739447, 74.636369140183717, 33.774068594804994, 40.770872887582925),
	cu4(72.963387832494163, 95.659300729473728, 11.809496633619768, 82.209921247423594,
		13.456139067865974, 57.329313623406605, 36.060621606214262, 70.867335643091849),
	cu4(32.484981432782945, 75.082940782924624, 42.467313093350882, 48.131159948246157,
		3.5963115764764657, 43.208665839959245, 79.442476890721579, 89.709102357602262),
	cu4(18.98573861410177, 93.308887208490106, 40.405250173250792, 91.039661826118675,
		8.0467721950480584, 42.100282172719147, 40.883324221187891, 26.030185504830527),
	cu4(7.5374809128872498, 82.441702896003477, 22.444346930107265, 22.138854312775123,
		66.76091829629658, 50.753805856571446, 78.193478508942519, 97.7932997968948),
	cu4(97.700573130371311, 53.53260215070685, 87.72443481149358, 84.575876772671876,
		19.215031396232092, 47.032676472809484, 11.989686410869325, 10.659507480757082),
	cu4(26.192053931854691, 9.8504326817814416, 10.174241480498686, 98.476562741434464,
		21.177712558385782, 33.814968789841501, 75.329030899018534, 55.02231980442177),
	cu4(56.222082700683771, 24.54395039218662, 95.589995289030483, 81.050822735322086,
		28.180450866082897, 28.837706255185282, 60.128952916771617, 87.311672180570511),
	cu4(42.449716172390481, 52.379709366885805, 27.896043159019225, 48.797373636065686,
		92.770268299044233, 89.899302036454571, 12.102066544863426, 99.43241951960718),
	cu4(45.77532924980639, 45.958701495993274, 37.458701356062065, 68.393691335056758,
		37.569326692060258, 27.673713456687381, 60.674866037757539, 62.47349659096146),
	cu4(67.426548091427676, 37.993772624988935, 23.483695892376684, 90.476863174921306,
		35.597065061143162, 79.872482633158796, 75.38634169631932, 18.244890038969412),
	cu4(61.336508189019057, 82.693132843213675, 44.639380902349664, 54.074825790745592,
		16.815615499771951, 20.049704667203923, 41.866884958868326, 56.735503699973002),
	cu4(18.1312339, 31.6473732, 95.5711034, 63.5350219, 92.3283165, 62.0158945, 18.5656052, 32.1268808),
	cu4(97.402018, 35.7169972, 33.1127443, 25.8935163, 1.13970027, 54.9424981, 56.4860195, 60.529264),
}

func cubicOneOff(t *testing.T, cubic1, cubic2 [4]dPoint, coin bool, ctx string) {
	c1 := mkCubic(cubic1)
	c2 := mkCubic(cubic2)
	in := newIntersections()
	in.intersectCubicCubic(c1, c2)
	if coin && in.usedCount() < 2 {
		t.Fatalf("%s: coincident used=%d, want >= 2", ctx, in.usedCount())
	}
	assertValidTouch(t, in, c1, c2, true, ctx)
}

var cubicCoinSet = [][4]dPoint{
	cu4(72.350448608398438, 27.966041564941406, 72.58441162109375, 27.861515045166016,
		72.818222045898437, 27.756658554077148, 73.394996643066406, 27.49799919128418),
	cu4(73.394996643066406, 27.49799919128418, 72.818222045898437, 27.756658554077148,
		72.58441162109375, 27.861515045166016, 72.350448608398438, 27.966041564941406),
	cu4(297.04998779296875, 43.928997039794922, 297.04998779296875, 43.928997039794922,
		300.69699096679688, 45.391998291015625, 306.92498779296875, 43.08599853515625),
	cu4(297.04998779296875, 43.928997039794922, 297.04998779296875, 43.928997039794922,
		300.69699096679688, 45.391998291015625, 306.92498779296875, 43.08599853515625),
	cu4(2, 3, 0, 4, 3, 2, 5, 3),
	cu4(2, 3, 0, 4, 3, 2, 5, 3),
	cu4(317, 711, 322.52285766601562, 711, 327, 715.4771728515625, 327, 721),
	cu4(324.07107543945312, 713.928955078125, 324.4051513671875, 714.26300048828125,
		324.71566772460937, 714.62060546875, 325, 714.9990234375),
}

func TestPathOpsCubicIntersectionOneOff(t *testing.T) {
	cubicOneOff(t, cubicIntersectTestSet[0], cubicIntersectTestSet[1], false, "cubicOneOff[0,1]")
}

func TestPathOpsCubicCoinOneOff(t *testing.T) {
	cubicOneOff(t, cubicCoinSet[0], cubicCoinSet[1], true, "cubicCoin[0,1]")
}

func TestPathOpsCubicIntersection(t *testing.T) {
	// oneOffTests: every ordered pair (outer < inner) in the corpus.
	for outer := 0; outer < len(cubicIntersectTestSet)-1; outer++ {
		for inner := outer + 1; inner < len(cubicIntersectTestSet); inner++ {
			cubicOneOff(t, cubicIntersectTestSet[outer], cubicIntersectTestSet[inner], false,
				sprintfPair("cubicOneOff", outer, inner))
		}
	}
	// cubicIntersectionCoinTest: consecutive pairs, each expected coincident (>= 2 intersections).
	for i := 0; i+1 < len(cubicCoinSet); i += 2 {
		cubicOneOff(t, cubicCoinSet[i], cubicCoinSet[i+1], true, sprintfPair("cubicCoin", i, i+1))
	}
}
