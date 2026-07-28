// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The path-ops reference values, replay-only and untagged.
//
// Unlike the other op groups there is no capture counterpart left, so recapturing these would mean restoring the
// capture code — and the SVG emit/parse pair it read result paths through — from git history. The fixtures are checked
// in; that is the supported state.

package probe

import (
	"strconv"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/path"
)

func mustPath(s string) *path.Path {
	p, err := decPath(s)
	if err != nil {
		refFatal(err)
	}
	return p
}

// --- path ops ---
//
// These return the oracle's result already materialized as a canvas path. Live, the probe had to ferry Skia's result
// across through the SVG emit/parse pair — the C API exposes no path iterator, so that was the only readable form. The
// fixture stores the geometry directly, so the bridge (and with it the last reason for the SVG parser to exist) is
// gone.

func refPathOpsOp(a, b *scenario.PathSpec, op int) (*path.Path, bool) {
	res, ok := split2(refGet("PathOpsOp", hashKey(encSpec(a)+"|"+encSpec(b)+"|"+strconv.Itoa(op))))
	if !mustBool(ok) {
		return nil, false
	}
	return mustPath(res), true
}

func refPathOpsSimplify(s *scenario.PathSpec) (*path.Path, bool) {
	res, ok := split2(refGet("PathOpsSimplify", hashKey(encSpec(s))))
	if !mustBool(ok) {
		return nil, false
	}
	return mustPath(res), true
}

// refPathOpsBuilder replays a Skia OpBuilder script: a sequence of (path, op) adds resolved as one.
func refPathOpsBuilder(adds []builderAdd) (*path.Path, bool) {
	res, ok := split2(refGet("PathOpsBuilder", hashKey(encBuilderScript(adds))))
	if !mustBool(ok) {
		return nil, false
	}
	return mustPath(res), true
}

// The bit-exact path-ops lanes freeze the oracle's raw point stream and fill type rather than a rebuilt path. The
// rebuilt form would not do: capture reads a C path through SVG emit/parse, which re-approximates conics with last-ULP
// float noise — fine for the area-sampled comparisons, fatal for a probe whose whole point is bit-exactness. Points
// come straight off the C path instead, no encoding in between.

func decExact(s string) ([]geom.Point, int, bool) {
	ptsEnc, rest := split2(s)
	fillEnc, okEnc := split2(rest)
	if !mustBool(okEnc) {
		return nil, 0, false
	}
	return mustPoints(ptsEnc), mustInt(fillEnc), true
}

func refPathOpsOpExact(a, b *scenario.PathSpec, op int, aFill, bFill path.FillType) ([]geom.Point, int, bool) {
	return decExact(refGet("PathOpsOpExact", hashKey(encSpec(a)+"|"+encSpec(b)+"|"+strconv.Itoa(op)+"|"+
		strconv.Itoa(int(aFill))+"|"+strconv.Itoa(int(bFill)))))
}

func refPathOpsSimplifyExact(s *scenario.PathSpec, fill path.FillType) ([]geom.Point, int, bool) {
	return decExact(refGet("PathOpsSimplifyExact", hashKey(encSpec(s)+"|"+strconv.Itoa(int(fill)))))
}

func refPathOpsBuilderUnionExact(s *scenario.PathSpec) ([]geom.Point, int, bool) {
	return decExact(refGet("PathOpsBuilderUnionExact", hashKey(encSpec(s))))
}
