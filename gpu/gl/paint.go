// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Paint is the input to GPU draws: a color, an XP factory for blending, and at most one color and one coverage fragment
// processor.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
)

// Paint describes how a draw is colored and blended. Its zero value is not directly usable; construct one with
// NewPaint, which yields solid white, src-over, trivial.
type Paint struct {
	xpFactory                 XPFactory
	colorFragmentProcessor    FragmentProcessor
	coverageFragmentProcessor FragmentProcessor
	color                     colorcore.PMColor4f
	notTrivial                bool
	initialized               bool
}

// NewPaint returns a default paint (solid white, no XP factory, trivial).
func NewPaint() *Paint {
	return &Paint{color: colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}, initialized: true}
}

// ClonePaint returns a deep copy of that, cloning its fragment processors.
func ClonePaint(that *Paint) *Paint {
	p := &Paint{
		xpFactory:   that.xpFactory,
		color:       that.color,
		notTrivial:  that.notTrivial,
		initialized: true,
	}
	if that.colorFragmentProcessor != nil {
		p.colorFragmentProcessor = that.colorFragmentProcessor.Clone()
	}
	if that.coverageFragmentProcessor != nil {
		p.coverageFragmentProcessor = that.coverageFragmentProcessor.Clone()
	}
	return p
}

// SetColor4f sets the initial color of the drawn primitive; it defaults to solid white.
func (p *Paint) SetColor4f(color colorcore.PMColor4f) { p.color = color }

// Color4f returns the paint's current color.
func (p *Paint) Color4f() colorcore.PMColor4f { return p.color }

// SetXPFactory sets the transfer-mode (blending) factory used for the draw.
func (p *Paint) SetXPFactory(factory XPFactory) {
	p.xpFactory = factory
	if factory != nil {
		p.notTrivial = true
	}
}

// SetPorterDuffXPFactory sets the XP factory to the Porter-Duff blend mode's implementation.
func (p *Paint) SetPorterDuffXPFactory(mode raster.BlendMode) {
	p.SetXPFactory(PorterDuffXPFactory(mode))
}

// SetCoverageSetOpXPFactory sets the XP factory to a coverage set-operation implementation.
func (p *Paint) SetCoverageSetOpXPFactory(regionOp raster.RegionOp, invertCoverage bool) {
	p.SetXPFactory(CoverageSetOpXPFactory(regionOp, invertCoverage))
}

// XPFactory returns the paint's XP factory, or nil for the default src-over blend.
func (p *Paint) XPFactory() XPFactory { return p.xpFactory }

// SetColorFragmentProcessor sets the paint's color fragment processor. Panics if fp is nil or one is already set.
func (p *Paint) SetColorFragmentProcessor(fp FragmentProcessor) {
	if fp == nil {
		panic("color fragment processor must not be nil")
	}
	if p.colorFragmentProcessor != nil {
		panic("color fragment processor already set")
	}
	p.colorFragmentProcessor = fp
	p.notTrivial = true
}

// SetCoverageFragmentProcessor sets the paint's coverage fragment processor. Panics if fp is nil or one is already set.
func (p *Paint) SetCoverageFragmentProcessor(fp FragmentProcessor) {
	if fp == nil {
		panic("coverage fragment processor must not be nil")
	}
	if p.coverageFragmentProcessor != nil {
		panic("coverage fragment processor already set")
	}
	p.coverageFragmentProcessor = fp
	p.notTrivial = true
}

// HasColorFragmentProcessor reports whether a color fragment processor is set.
func (p *Paint) HasColorFragmentProcessor() bool { return p.colorFragmentProcessor != nil }

// HasCoverageFragmentProcessor reports whether a coverage fragment processor is set.
func (p *Paint) HasCoverageFragmentProcessor() bool { return p.coverageFragmentProcessor != nil }

// NumTotalFragmentProcessors returns the count of set fragment processors (0, 1, or 2).
func (p *Paint) NumTotalFragmentProcessors() int {
	n := 0
	if p.HasColorFragmentProcessor() {
		n++
	}
	if p.HasCoverageFragmentProcessor() {
		n++
	}
	return n
}

// ColorFragmentProcessor returns the paint's color fragment processor, or nil if unset.
func (p *Paint) ColorFragmentProcessor() FragmentProcessor { return p.colorFragmentProcessor }

// CoverageFragmentProcessor returns the paint's coverage fragment processor, or nil if unset.
func (p *Paint) CoverageFragmentProcessor() FragmentProcessor {
	return p.coverageFragmentProcessor
}

// UsesLocalCoords reports whether either fragment processor samples coordinates; the sample coords for the top-level
// FPs are implicitly the geometry processor's local coords.
func (p *Paint) UsesLocalCoords() bool {
	return (p.colorFragmentProcessor != nil && p.colorFragmentProcessor.fpBase().UsesSampleCoords()) ||
		(p.coverageFragmentProcessor != nil &&
			p.coverageFragmentProcessor.fpBase().UsesSampleCoords())
}

// IsTrivial reports whether the paint is src-over with no fragment processors.
func (p *Paint) IsTrivial() bool { return !p.notTrivial }

// IsConstantBlendedColor looks for the common cases where the output color after blending is constant.
func (p *Paint) IsConstantBlendedColor() (colorcore.PMColor4f, bool) {
	kSrc := PorterDuffXPFactory(raster.BlendSrc)
	kClear := PorterDuffXPFactory(raster.BlendClear)
	if p.xpFactory == kClear {
		return colorcore.PMColor4f{}, true
	}
	if p.HasColorFragmentProcessor() {
		return colorcore.PMColor4f{}, false
	}
	if p.xpFactory == kSrc || (p.xpFactory == nil && pmIsOpaque(p.color)) {
		return p.color, true
	}
	return colorcore.PMColor4f{}, false
}
