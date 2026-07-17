// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The GLSL form of the decal filter — the GPU twin of shaders/filterdecal.go, which documents the same formula for the
// CPU pipeline. It wraps a child shader with a one-pixel AA falloff ramp at the edges of decalBounds so that sampling
// outside the bounds fades to transparent instead of clamping or repeating. This is used whenever decal tiling must
// apply at layer-space resolution; with it, every shader form the GPU-native filter lanes emit converts to an FP.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

type filterDecalFP struct {
	FPBase
	decalBounds geom.Rect
}

// NewFilterDecalFP builds an FP that samples child at the pass-through coords, weighted by the half-pixel decal ramp
// over decalBounds.
func NewFilterDecalFP(decalBounds geom.Rect, child FragmentProcessor) FragmentProcessor {
	fp := &filterDecalFP{decalBounds: decalBounds}
	fp.initFP(GLSLFPClassID, FPCompatibleWithCoverageAsAlpha)
	fp.setUsesSampleCoordsDirectly()
	registerChildOf(fp, child, ExplicitSampleUsage())
	return fp
}

func (f *filterDecalFP) Name() string { return "FilterDecal" }

func (f *filterDecalFP) Clone() FragmentProcessor {
	return NewFilterDecalFP(f.decalBounds, f.ChildProcessor(0).Clone())
}

func (f *filterDecalFP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	b.Add32(uint32(runtimeFPFilterDecal), "runtimeFPKind")
}

func (f *filterDecalFP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*filterDecalFP)
	return ok && f.decalBounds == o.decalBounds
}

func (f *filterDecalFP) onMakeProgramImpl() FPProgramImpl { return &filterDecalImpl{} }

type filterDecalImpl struct {
	FPImplBase
	decalBoundsUni UniformHandle
}

func (i *filterDecalImpl) EmitCode(args *FPEmitArgs) {
	fb := args.FragBuilder
	var boundsName string
	i.decalBoundsUni, boundsName = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat4, "decalBounds")

	// d = saturate((decalBounds - coord.xyxy) * (-1,-1,1,1) + 0.5); the ramp is the product of the four edge distances,
	// giving a one-pixel AA falloff at the decal boundary.
	coord := args.SampleCoord
	fb.CodeAppendf("vec4 d = clamp((%s - %s.xyxy) * vec4(-1.0, -1.0, 1.0, 1.0) + 0.5, 0.0, 1.0);",
		boundsName, coord)
	fb.CodeAppendf("return (d.x * d.y * d.z * d.w) * %s;",
		i.InvokeChildWithCoords(0, "", args, coord))
}

func (i *filterDecalImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	f := fp.(*filterDecalFP)
	pdman.Set4f(i.decalBoundsUni, f.decalBounds.Left, f.decalBounds.Top, f.decalBounds.Right,
		f.decalBounds.Bottom)
}
