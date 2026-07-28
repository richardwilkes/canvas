// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package imagefilter provides the image filter implementations, built on the filtercore machinery: each filter
// implements filtercore.Filter with consistent bounds propagation and FilterResult composition, and the factory
// functions handle graph-building concerns like crop-rect wrapping, filter collapses, and compositions such as
// DropShadow. The set covers crop/tile/empty, blur, matrix-transform/offset, color-filter, compose, merge, the
// drop-shadow compositions, the shader-evaluating filters (morphology, displacement, lighting, magnifier, matrix
// convolution, arithmetic) over the runtime-effect kernels in shaders, and the image-source leaf.
package imagefilter

import (
	"math"

	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// isValidRect reports whether r is finite and sorted (empty rects are allowed).
func isValidRect(r geom.Rect) bool {
	return r.IsFinite() && r.Left <= r.Right && r.Top <= r.Bottom
}

func absf(v float32) float32 { return float32(math.Abs(float64(v))) }

func sqrtf(v float32) float32 { return float32(math.Sqrt(float64(v))) }

// filterDefaults supplies the default implementations of filtercore.Filter's optional methods; filters embed it and
// override what they need.
type filterDefaults struct {
	base filtercore.FilterBase
}

// Base implements filtercore.Filter.
func (f *filterDefaults) Base() *filtercore.FilterBase { return &f.base }

// OnAffectsTransparentBlack implements filtercore.Filter with the default (false).
func (f *filterDefaults) OnAffectsTransparentBlack() bool { return false }

// IgnoreInputsAffectsTransparentBlack implements filtercore.Filter with the default (false).
func (f *filterDefaults) IgnoreInputsAffectsTransparentBlack() bool { return false }

// OnCTMCapability implements filtercore.Filter with the default (filtercore.MatrixCapabilityScaleTranslate).
func (f *filterDefaults) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityScaleTranslate
}

// OnIsColorFilterNode implements filtercore.Filter with the default (not a color filter node).
func (f *filterDefaults) OnIsColorFilterNode() (shaders.ColorFilter, bool) { return nil, false }
