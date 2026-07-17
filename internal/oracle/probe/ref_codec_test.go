// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package probe

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/path"
)

// Encoders for the value types the frozen fixtures carry. Every float goes through f32/parseF32, so the round trip is
// bit-exact and survives Inf/NaN; the surrounding structure is space-separated so a fixture stays greppable and
// reviewable in a diff.

func encFloats(vs ...float32) string {
	parts := make([]string, len(vs))
	for i, v := range vs {
		parts[i] = f32(v)
	}
	return strings.Join(parts, " ")
}

// decFloats decodes a space-separated float list. n is the expected count, or -1 for variable length. An empty string
// is a legitimate empty list whenever the count is not pinned to a positive arity: a path op can succeed and produce no
// points at all (intersecting disjoint rects, say).
func decFloats(s string, n int) ([]float32, error) {
	if s == "" {
		if n <= 0 {
			return nil, nil
		}
		return nil, fmt.Errorf("ref: empty float list, want %d", n)
	}
	parts := strings.Split(s, " ")
	if n >= 0 && len(parts) != n {
		return nil, fmt.Errorf("ref: float list has %d entries, want %d", len(parts), n)
	}
	out := make([]float32, len(parts))
	for i, p := range parts {
		v, err := parseF32(p)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func encMatrix(m *geom.Matrix) string {
	v := m.As9()
	return encFloats(v[:]...)
}

func decMatrix(s string) (geom.Matrix, error) {
	vs, err := decFloats(s, 9)
	if err != nil {
		return geom.Matrix{}, err
	}
	return geom.MatrixFrom9([9]float32(vs)), nil
}

func encPoint(p geom.Point) string { return encFloats(p.X, p.Y) }

func encPoints(pts []geom.Point) string {
	parts := make([]string, 0, len(pts)*2)
	for _, p := range pts {
		parts = append(parts, f32(p.X), f32(p.Y))
	}
	return strings.Join(parts, " ")
}

func decPoints(s string) ([]geom.Point, error) {
	vs, err := decFloats(s, -1)
	if err != nil {
		return nil, err
	}
	if len(vs)%2 != 0 {
		return nil, fmt.Errorf("ref: point list has odd length %d", len(vs))
	}
	out := make([]geom.Point, len(vs)/2)
	for i := range out {
		out[i] = geom.Point{X: vs[i*2], Y: vs[i*2+1]}
	}
	return out, nil
}

func encRect(r geom.Rect) string { return encFloats(r.Left, r.Top, r.Right, r.Bottom) }

func decRect(s string) (geom.Rect, error) {
	vs, err := decFloats(s, 4)
	if err != nil {
		return geom.Rect{}, err
	}
	return geom.Rect{Left: vs[0], Top: vs[1], Right: vs[2], Bottom: vs[3]}, nil
}

//nolint:unused // the capture half is gone (see the package comment); the enc side is kept as the spec of the fixture encoding
func encIRect(r geom.IRect) string {
	return fmt.Sprintf("%d %d %d %d", r.Left, r.Top, r.Right, r.Bottom)
}

func decIRect(s string) (geom.IRect, error) {
	var r geom.IRect
	if _, err := fmt.Sscanf(s, "%d %d %d %d", &r.Left, &r.Top, &r.Right, &r.Bottom); err != nil {
		return geom.IRect{}, fmt.Errorf("ref: bad IRect %q: %w", s, err)
	}
	return r, nil
}

func encBool(b bool) string { return strconv.FormatBool(b) }

func decBool(s string) (bool, error) {
	v, err := strconv.ParseBool(s)
	if err != nil {
		return false, fmt.Errorf("ref: bad bool %q: %w", s, err)
	}
	return v, nil
}

// encPath encodes a path's geometry: fill type, then one record per verb. This is what the path-ops fixtures freeze —
// the oracle's *result geometry*, not a rasterization of it. Freezing pixels would bind the fixture to today's
// rasterizer, so a later rasterizer change would surface as a false divergence in a path-ops test; freezing geometry
// lets replay rasterize with whatever is current and keeps the comparison honest.
//
// Verb records carry only the points the verb contributes (the iterator's pts[0] repeats the current point), so the
// encoding is exactly what the builder calls need to reconstruct the path.
//
//nolint:unused // kept as the spec of the fixture encoding (see encIRect)
func encPath(p *path.Path) string {
	var b strings.Builder
	fmt.Fprintf(&b, "FT=%d", int(p.FillType()))
	it := path.NewIter(p, false)
	for {
		verb, pts, ok := it.Next()
		if !ok {
			break
		}
		b.WriteByte(';')
		switch verb {
		case path.VerbMove:
			fmt.Fprintf(&b, "M %s", encPoint(pts[0]))
		case path.VerbLine:
			fmt.Fprintf(&b, "L %s", encPoint(pts[1]))
		case path.VerbQuad:
			fmt.Fprintf(&b, "Q %s %s", encPoint(pts[1]), encPoint(pts[2]))
		case path.VerbConic:
			fmt.Fprintf(&b, "K %s %s %s", encPoint(pts[1]), encPoint(pts[2]), f32(it.ConicWeight()))
		case path.VerbCubic:
			fmt.Fprintf(&b, "C %s %s %s", encPoint(pts[1]), encPoint(pts[2]), encPoint(pts[3]))
		case path.VerbClose:
			b.WriteByte('Z')
		}
	}
	return b.String()
}

// decPath inverts encPath, rebuilding through the ordinary builder calls.
func decPath(s string) (*path.Path, error) {
	recs := strings.Split(s, ";")
	if len(recs) == 0 || !strings.HasPrefix(recs[0], "FT=") {
		return nil, fmt.Errorf("ref: path encoding lacks a fill type: %q", s)
	}
	ft, err := strconv.Atoi(strings.TrimPrefix(recs[0], "FT="))
	if err != nil {
		return nil, fmt.Errorf("ref: bad fill type in %q: %w", recs[0], err)
	}
	p := path.New()
	for _, rec := range recs[1:] {
		if rec == "" {
			continue
		}
		kind, rest, _ := strings.Cut(rec, " ")
		var v []float32
		if kind != "Z" {
			if v, err = decFloats(rest, -1); err != nil {
				return nil, err
			}
		}
		switch kind {
		case "M":
			if len(v) != 2 {
				return nil, fmt.Errorf("ref: M needs 2 values, got %d", len(v))
			}
			p.MoveTo(v[0], v[1])
		case "L":
			if len(v) != 2 {
				return nil, fmt.Errorf("ref: L needs 2 values, got %d", len(v))
			}
			p.LineTo(v[0], v[1])
		case "Q":
			if len(v) != 4 {
				return nil, fmt.Errorf("ref: Q needs 4 values, got %d", len(v))
			}
			p.QuadTo(v[0], v[1], v[2], v[3])
		case "K":
			if len(v) != 5 {
				return nil, fmt.Errorf("ref: K needs 5 values, got %d", len(v))
			}
			p.ConicTo(v[0], v[1], v[2], v[3], v[4])
		case "C":
			if len(v) != 6 {
				return nil, fmt.Errorf("ref: C needs 6 values, got %d", len(v))
			}
			p.CubicTo(v[0], v[1], v[2], v[3], v[4], v[5])
		case "Z":
			p.Close()
		default:
			return nil, fmt.Errorf("ref: unknown path record %q", kind)
		}
	}
	p.SetFillType(path.FillType(ft))
	return p, nil
}

// encSpec canonically encodes a PathSpec for use as a fixture key. Keyed on the spec rather than on the built path so
// the key survives a change in how the port builds paths — a build change should fail the probe that checks it, not
// silently miss every path-ops fixture.
func encSpec(s *scenario.PathSpec) string {
	if s == nil {
		return "nil"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "F%d", int(s.Fill))
	for i := range s.Ops {
		op := &s.Ops[i]
		fmt.Fprintf(&b, ";%d,%s,%d,%d,%d,%t", int(op.Kind), encFloats(op.F[:]...), int(op.Dir), int(op.Size),
			int(op.Mode), op.Bool)
		fmt.Fprintf(&b, ",%s", encRect(op.Rect))
		if op.Pts != nil {
			fmt.Fprintf(&b, ",[%s]", encPoints(op.Pts))
		}
		if op.M != nil {
			fmt.Fprintf(&b, ",{%s}", encMatrix(op.M))
		}
		if op.Sub != nil {
			fmt.Fprintf(&b, ",(%s)", encSpec(op.Sub))
		}
	}
	return b.String()
}

// encMeasure encodes an Skia's PathMeasure walk. The nesting gets one separator per level — "@" contours, "|" fields,
// ";" list items, "," item fields — above encPoints' own " ", so every level stays unambiguous and the fixture is still
// greppable.
//
//nolint:unused // kept as the spec of the fixture encoding (see encIRect)
func encMeasure(cs []measureContour) string {
	out := make([]string, len(cs))
	for i, c := range cs {
		samples := make([]string, len(c.posTan))
		for j, s := range c.posTan {
			samples[j] = strings.Join([]string{
				encBool(s.ok), f32(s.pos.X), f32(s.pos.Y), f32(s.tan.X),
				f32(s.tan.Y),
			}, ",")
		}
		segs := make([]string, len(c.segments))
		for j, s := range c.segments {
			segs[j] = encBool(s.ok) + "," + encPoints(s.points)
		}
		out[i] = strings.Join([]string{
			f32(c.length), encBool(c.isClosed), strings.Join(samples, ";"), encMatrix(&c.matrix),
			encBool(c.matrixOK), strings.Join(segs, ";"),
		}, "|")
	}
	return strings.Join(out, "@")
}

func decMeasure(s string) ([]measureContour, error) {
	if s == "" {
		return nil, nil
	}
	// splitList treats "" as the empty list; strings.Split would yield one empty element.
	splitList := func(v, sep string) []string {
		if v == "" {
			return nil
		}
		return strings.Split(v, sep)
	}
	var out []measureContour
	for _, rec := range strings.Split(s, "@") {
		f := strings.Split(rec, "|")
		if len(f) != 6 {
			return nil, fmt.Errorf("ref: measure contour has %d fields, want 6", len(f))
		}
		var c measureContour
		var err error
		if c.length, err = parseF32(f[0]); err != nil {
			return nil, err
		}
		if c.isClosed, err = decBool(f[1]); err != nil {
			return nil, err
		}
		for _, sr := range splitList(f[2], ";") {
			sf := strings.Split(sr, ",")
			if len(sf) != 5 {
				return nil, fmt.Errorf("ref: posTan sample has %d fields, want 5", len(sf))
			}
			var ps posTanSample
			if ps.ok, err = decBool(sf[0]); err != nil {
				return nil, err
			}
			vals := make([]float32, 4)
			for k := range vals {
				if vals[k], err = parseF32(sf[k+1]); err != nil {
					return nil, err
				}
			}
			ps.pos = geom.Point{X: vals[0], Y: vals[1]}
			ps.tan = geom.Point{X: vals[2], Y: vals[3]}
			c.posTan = append(c.posTan, ps)
		}
		if c.matrix, err = decMatrix(f[3]); err != nil {
			return nil, err
		}
		if c.matrixOK, err = decBool(f[4]); err != nil {
			return nil, err
		}
		for _, sr := range splitList(f[5], ";") {
			ok, pts, _ := strings.Cut(sr, ",")
			var ms measureSegment
			if ms.ok, err = decBool(ok); err != nil {
				return nil, err
			}
			if ms.points, err = decPoints(pts); err != nil {
				return nil, err
			}
			c.segments = append(c.segments, ms)
		}
		out = append(out, c)
	}
	return out, nil
}

// encStrokePaint encodes a stroke configuration for use as a fixture key.
func encStrokePaint(sp *strokePaint) string {
	return fmt.Sprintf("%s,%s,%d,%d,%t,%s", f32(sp.width), f32(sp.miter), sp.cap, sp.join, sp.fill,
		f32(sp.resScale))
}

// builderAdd is one (path, op) step of an Skia's OpBuilder script. It lives here rather than in the probe that drives
// it so both the replay and capture halves can key on the same encoding.
type builderAdd struct {
	spec *scenario.PathSpec
	op   int
}

func encBuilderScript(adds []builderAdd) string {
	parts := make([]string, len(adds))
	for i, a := range adds {
		parts[i] = strconv.Itoa(a.op) + "(" + encSpec(a.spec) + ")"
	}
	return strings.Join(parts, "|")
}

// encSegments encodes the analytic edge-walker output: a list of 7-tuple int32 segments.
//
//nolint:unused // kept as the spec of the fixture encoding (see encIRect)
func encSegments(segs [][7]int32) string {
	rows := make([]string, len(segs))
	for i, s := range segs {
		parts := make([]string, 7)
		for j, v := range s {
			parts[j] = strconv.FormatInt(int64(v), 10)
		}
		rows[i] = strings.Join(parts, ",")
	}
	return strings.Join(rows, ";")
}

func decSegments(s string) ([][7]int32, error) {
	if s == "" {
		return nil, nil
	}
	rows := strings.Split(s, ";")
	out := make([][7]int32, len(rows))
	for i, row := range rows {
		parts := strings.Split(row, ",")
		if len(parts) != 7 {
			return nil, fmt.Errorf("ref: segment %d has %d fields, want 7", i, len(parts))
		}
		for j, p := range parts {
			v, err := strconv.ParseInt(p, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("ref: bad segment field %q: %w", p, err)
			}
			out[i][j] = int32(v)
		}
	}
	return out, nil
}
