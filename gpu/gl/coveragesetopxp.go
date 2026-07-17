// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// An xfer processor that blends the src coverage with the dst using a region set operator, optionally inverting the
// coverage first. It is used to render coverage masks with CSG semantics — the mask-filter draw lane drives it (Replace
// to burn a shape into a fresh A8 mask, and Intersect/Union/Difference to composite the non-normal blur styles).

package gl

import (
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// coverageSetOpXP is a transfer processor that blends src coverage with dst via a region set op.
type coverageSetOpXP struct {
	XPBase
	regionOp       raster.RegionOp
	invertCoverage bool
}

func newCoverageSetOpXP(regionOp raster.RegionOp, invertCoverage bool) *coverageSetOpXP {
	xp := &coverageSetOpXP{regionOp: regionOp, invertCoverage: invertCoverage}
	xp.initXP(CoverageSetOpXPClassID)
	return xp
}

func (x *coverageSetOpXP) Name() string { return "Coverage Set Op" }

func (x *coverageSetOpXP) MakeProgramImpl() XPProgramImpl { return &coverageSetOpXPImpl{} }

func (x *coverageSetOpXP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.AddBool(x.invertCoverage, "invert coverage")
}

func (x *coverageSetOpXP) onGetBlendInfo() gpu.BlendInfo {
	blendInfo := gpu.MakeBlendInfo()
	switch x.regionOp {
	case raster.RegionReplace:
		blendInfo.SrcBlend = gpu.BlendCoeffOne
		blendInfo.DstBlend = gpu.BlendCoeffZero
	case raster.RegionIntersect:
		blendInfo.SrcBlend = gpu.BlendCoeffDC
		blendInfo.DstBlend = gpu.BlendCoeffZero
	case raster.RegionUnion:
		blendInfo.SrcBlend = gpu.BlendCoeffOne
		blendInfo.DstBlend = gpu.BlendCoeffISC
	case raster.RegionXor:
		blendInfo.SrcBlend = gpu.BlendCoeffIDC
		blendInfo.DstBlend = gpu.BlendCoeffISC
	case raster.RegionDifference:
		blendInfo.SrcBlend = gpu.BlendCoeffZero
		blendInfo.DstBlend = gpu.BlendCoeffISC
	case raster.RegionReverseDifference:
		blendInfo.SrcBlend = gpu.BlendCoeffIDC
		blendInfo.DstBlend = gpu.BlendCoeffZero
	}
	blendInfo.BlendConstant = [4]float32{}
	return blendInfo
}

func (x *coverageSetOpXP) onIsEqual(other XferProcessor) bool {
	o, ok := other.(*coverageSetOpXP)
	return ok && x.regionOp == o.regionOp && x.invertCoverage == o.invertCoverage
}

type coverageSetOpXPImpl struct {
	XPImplBase
}

func (i *coverageSetOpXPImpl) emitOutputsForBlendState(args *XPEmitArgs) {
	xp := args.XP.(*coverageSetOpXP)
	if xp.invertCoverage {
		args.FragBuilder.CodeAppendf("%s = 1.0 - %s;", args.OutputPrimary, args.InputCoverage)
	} else {
		args.FragBuilder.CodeAppendf("%s = %s;", args.OutputPrimary, args.InputCoverage)
	}
}

//////////////////////////////////////////////////////////////////////////////

// coverageSetOpXPFactory is a stateless immutable singleton compared by identity.
type coverageSetOpXPFactory struct {
	regionOp       raster.RegionOp
	invertCoverage bool
}

// The static factory singletons, keyed [regionOp][invertCoverage].
var coverageSetOpXPFactories = func() [6][2]*coverageSetOpXPFactory {
	var fs [6][2]*coverageSetOpXPFactory
	for op := range fs {
		fs[op][0] = &coverageSetOpXPFactory{regionOp: raster.RegionOp(op), invertCoverage: false}
		fs[op][1] = &coverageSetOpXPFactory{regionOp: raster.RegionOp(op), invertCoverage: true}
	}
	return fs
}()

// CoverageSetOpXPFactory returns the shared factory singleton for regionOp/invertCoverage.
func CoverageSetOpXPFactory(regionOp raster.RegionOp, invertCoverage bool) XPFactory {
	if int(regionOp) >= len(coverageSetOpXPFactories) {
		panic("unknown region op")
	}
	if invertCoverage {
		return coverageSetOpXPFactories[regionOp][1]
	}
	return coverageSetOpXPFactories[regionOp][0]
}

func (f *coverageSetOpXPFactory) makeXferProcessor(_ ProcessorAnalysisColor, _ AnalysisCoverage, _ *gpu.Caps, _ gpu.ClampType) XferProcessor {
	return newCoverageSetOpXP(f.regionOp, f.invertCoverage)
}

func (f *coverageSetOpXPFactory) analysisProperties(_ ProcessorAnalysisColor, _ AnalysisCoverage, _ *gpu.Caps, _ gpu.ClampType) uint32 {
	props := XPAnalysisIgnoresInputColor
	if f.regionOp == raster.RegionReplace {
		// Only Replace is claimed here. Union/Difference are also unaffected by the dst value when the coverage and
		// color are all opaque, but proving that needs the opacity plumbed through the factory, so they are left out.
		// The omission costs a missed optimization on those ops and nothing else: claiming too little is the safe
		// direction, since the flag only ever licenses the backend to skip reading dst.
		props |= XPAnalysisUnaffectedByDstValue
	}
	return props
}
