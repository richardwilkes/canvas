// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Reference-value record/replay for the numeric probes.
//
// The probes were originally live differentials: each one called into the C Skia library (via skiac) for the reference
// answer and compared the Go port against it. The C library is gone, so the answers it gave are frozen here instead —
// captured from Skia one last time on darwin/arm64 and checked in under testdata/ref. The values are still *Skia's*, so
// the probes remain correctness references rather than mere pins on the port's current behavior, and their agree()
// tolerances keep their original meaning.
//
// What this cannot do is answer for input that was never captured. A probe that invents new input (the SVG parse fuzz
// is the notable one) can only replay the corpus frozen here; growing it needs a Skia build from git history. Corpus
// changes therefore fail loudly (missing key) rather than silently skipping.
//
// This file is the replay half and ships unconditionally. The capture half (ref_ops_capture_test.go, build tag
// `skiaoracle`) is what talked to skiac; it was removed with the rest of the C bindings and survives only in git
// history.
package probe

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/richardwilkes/canvas/geom"
)

// refDir is where the frozen fixtures live, relative to the probe package directory (go test runs with the package
// directory as the working directory).
const refDir = "testdata/ref"

// fixture is one op's frozen reference values, keyed by a hash of the call's inputs. Keying by input rather than by
// call order means a probe can reorder or subset its corpus without invalidating the fixture; only a genuine change of
// input produces a miss.
type fixture struct {
	Entries map[string]string `json:"entries"`
	Op      string            `json:"op"`
}

var (
	refMu    sync.Mutex
	refCache = map[string]*fixture{}
)

// loadFixture reads and caches the fixture for op.
func loadFixture(op string) (*fixture, error) {
	refMu.Lock()
	defer refMu.Unlock()
	if f, ok := refCache[op]; ok {
		return f, nil
	}
	buf, err := os.ReadFile(filepath.Join(refDir, op+".json"))
	if err != nil {
		return nil, fmt.Errorf("ref: no frozen fixture for op %q: %w", op, err)
	}
	f := &fixture{}
	if err = json.Unmarshal(buf, f); err != nil {
		return nil, fmt.Errorf("ref: fixture %q is malformed: %w", op, err)
	}
	refCache[op] = f
	return f, nil
}

// refLookup returns the frozen reference value for op applied to the inputs encoded in key.
func refLookup(op, key string) (string, error) {
	f, err := loadFixture(op)
	if err != nil {
		return "", err
	}
	v, ok := f.Entries[key]
	if !ok {
		return "", fmt.Errorf("ref: op %q has no frozen value for input %s — the probe corpus changed, but "+
			"the C oracle that would answer for new input is gone; either revert the "+
			"corpus change or rebuild Skia from git history to recapture", op, key)
	}
	return v, nil
}

// hashKey reduces an encoded input to a short stable key.
func hashKey(encoded string) string {
	sum := sha256.Sum256([]byte(encoded))
	return hex.EncodeToString(sum[:8])
}

// f32 encodes a float32 exactly and reversibly. The hex-float form ('x') round-trips bit-for-bit — which the probes
// require, since they assert bit-exactness on the double-computed lanes — and unlike JSON numbers it represents the Inf
// and NaN the corpora deliberately contain (degenerate conic weights, sliver arcs).
func f32(v float32) string {
	f := float64(v)
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "+Inf"
	case math.IsInf(f, -1):
		return "-Inf"
	}
	return strconv.FormatFloat(f, 'x', -1, 32)
}

// parseF32 inverts f32.
func parseF32(s string) (float32, error) {
	switch s {
	case "NaN":
		return float32(math.NaN()), nil
	case "+Inf":
		return float32(math.Inf(1)), nil
	case "-Inf":
		return float32(math.Inf(-1)), nil
	}
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return 0, fmt.Errorf("ref: bad float %q: %w", s, err)
	}
	return float32(v), nil
}

// refFatal reports a fixture miss. Probes call these helpers outside a *testing.T (corpus construction), so this panics
// rather than threading a t through every call; the message names the cause and the remedy.
func refFatal(err error) {
	panic(err.Error())
}

// errCount reports a fixture entry whose length does not match the op's fixed arity — a malformed or hand-edited
// fixture, since capture could not have produced one.
func errCount(op string, got, want int) error {
	return fmt.Errorf("ref: op %q fixture has %d points, want a multiple of %d", op, got, want)
}

func refGet(op, key string) string {
	v, err := refLookup(op, key)
	if err != nil {
		refFatal(err)
	}
	return v
}

// split2 divides a multi-value fixture entry on the "|" separator.
func split2(s string) (first, rest string) {
	i := strings.IndexByte(s, '|')
	if i < 0 {
		return s, ""
	}
	return s[:i], s[i+1:]
}

// splitN divides a fixture entry into exactly n "|"-separated fields, failing loudly on a malformed one rather than
// letting a short read silently decode as a zero value.
func splitN(op, s string, n int) []string {
	parts := strings.Split(s, "|")
	if len(parts) != n {
		refFatal(fmt.Errorf("ref: op %q fixture entry has %d fields, want %d: %q", op, len(parts), n, s))
	}
	return parts
}

// encBools packs a bool run compactly — these come in per-path batches (containment grids), which would be unreadable
// and needlessly large one entry per query.
//
//nolint:unused // the capture half is gone (see the package comment); kept as the spec of the fixture encoding
func encBools(vs []bool) string {
	var b strings.Builder
	b.Grow(len(vs))
	for _, v := range vs {
		if v {
			b.WriteByte('1')
		} else {
			b.WriteByte('0')
		}
	}
	return b.String()
}

func decBools(s string) []bool {
	out := make([]bool, len(s))
	for i := range s {
		switch s[i] {
		case '1':
			out[i] = true
		case '0':
		default:
			refFatal(fmt.Errorf("ref: bad bool run %q", s))
		}
	}
	return out
}

func mustMatrix(s string) geom.Matrix {
	m, err := decMatrix(s)
	if err != nil {
		refFatal(err)
	}
	return m
}

func mustPoints(s string) []geom.Point {
	p, err := decPoints(s)
	if err != nil {
		refFatal(err)
	}
	return p
}

func mustRect(s string) geom.Rect {
	r, err := decRect(s)
	if err != nil {
		refFatal(err)
	}
	return r
}

func mustIRect(s string) geom.IRect {
	r, err := decIRect(s)
	if err != nil {
		refFatal(err)
	}
	return r
}

func mustBool(s string) bool {
	b, err := decBool(s)
	if err != nil {
		refFatal(err)
	}
	return b
}

func mustInt(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		refFatal(err)
	}
	return v
}

func mustSegments(s string) [][7]int32 {
	v, err := decSegments(s)
	if err != nil {
		refFatal(err)
	}
	return v
}

// pathFacts is what TestPathStructureProbe reads off the oracle's path.
type pathFacts struct {
	points   []geom.Point
	fillType int
	lastPt   geom.Point
	isEmpty  bool
	lastOK   bool
}

// strokeResult is the oracle's Skia's Paint::getFillPath outcome: whether the paint reduced to a plain fill, and the
// resulting geometry.
type strokeResult struct {
	points   []geom.Point
	fillType int
	isFill   bool
}

// The Skia's PathMeasure record. Measuring is a stateful walk — construct, read the contour, step to the next — so the
// fixture freezes the whole walk for one (path, forceClosed, resScale) as an ordered list of contours, rather than
// trying to key each individual query. The fractions and spans sampled are fixed constants in the probe, so both halves
// index them identically.
type measureContour struct {
	posTan   []posTanSample   // one per probe fraction; empty when the contour has no length
	segments []measureSegment // one per probe span; empty when the contour has no length
	matrix   geom.Matrix      // at length/3
	length   float32
	isClosed bool
	matrixOK bool
}

type posTanSample struct {
	ok       bool
	pos, tan geom.Point
}

type measureSegment struct {
	points []geom.Point
	ok     bool
}
