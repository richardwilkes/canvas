// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The path effects that are publicly reachable: dash, corner, discrete, trim, 1D path stamp, 2D line/path lattice, and
// the sum/compose pair effects. Every effect implements stroke.PathEffect; this file holds the pair effects and the
// shared defaults.

package patheffect

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/stroke"
)

// noAsPoints supplies the default AsPoints (false); only the dash effect overrides it.
type noAsPoints struct{}

// AsPoints is the default: this effect has no point representation.
func (noAsPoints) AsPoints(*stroke.PointData, *path.Path, *stroke.Rec, *geom.Matrix, *geom.Rect) bool {
	return false
}

///////////////////////////////////////////////////////////////////////////////
// compose path effect

// composeEffect applies inner first, then outer.
type composeEffect struct {
	noAsPoints
	outer, inner stroke.PathEffect
}

// MakeCompose creates an effect that applies inner first, then outer (outer(inner(path))). If either is nil, the other
// is returned.
func MakeCompose(outer, inner stroke.PathEffect) stroke.PathEffect {
	if outer == nil {
		return inner
	}
	if inner == nil {
		return outer
	}
	return &composeEffect{outer: outer, inner: inner}
}

func (c *composeEffect) FilterPath(dst, src *path.Path, rec *stroke.Rec, cullRect *geom.Rect, ctm *geom.Matrix) bool {
	ptr := src
	if c.inner.FilterPath(dst, src, rec, cullRect, ctm) {
		// Detach the built path into tmp and re-filter from there.
		ptr = dst.Clone()
		dst.Reset()
	}
	return c.outer.FilterPath(dst, ptr, rec, cullRect, ctm)
}

func (c *composeEffect) ComputeFastBounds(bounds *geom.Rect) bool {
	// inner is computed first, automatically updating bounds before computing outer.
	return c.inner.ComputeFastBounds(bounds) && c.outer.ComputeFastBounds(bounds)
}

///////////////////////////////////////////////////////////////////////////////
// sum path effect

// sumEffect applies two effects independently and combines their output paths.
type sumEffect struct {
	noAsPoints
	first, second stroke.PathEffect
}

// MakeSum creates an effect that applies two effects in sequence (first(path) + second(path)). If either is nil, the
// other is returned.
func MakeSum(first, second stroke.PathEffect) stroke.PathEffect {
	if first == nil {
		return second
	}
	if second == nil {
		return first
	}
	return &sumEffect{first: first, second: second}
}

func (s *sumEffect) FilterPath(dst, src *path.Path, rec *stroke.Rec, cullRect *geom.Rect, ctm *geom.Matrix) bool {
	// always call both, even if the first one succeeds
	filteredFirst := s.first.FilterPath(dst, src, rec, cullRect, ctm)
	filteredSecond := s.second.FilterPath(dst, src, rec, cullRect, ctm)
	return filteredFirst || filteredSecond
}

func (s *sumEffect) ComputeFastBounds(bounds *geom.Rect) bool {
	// Unlike Compose(), first modifies the path first for Sum
	return s.first.ComputeFastBounds(bounds) && s.second.ComputeFastBounds(bounds)
}
