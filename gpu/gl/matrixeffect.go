// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// matrixEffect samples its child with a uniform matrix applied to the incoming coordinates. This is the FP that carries
// local-matrix transforms into the shader pipeline; the geometry processor lifts chains of these to vertex-shader
// varyings where possible (see collectTransforms).

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

type matrixEffect struct {
	FPBase
	matrix geom.Matrix
}

// MatrixEffectFP returns an FP that applies matrix to child's sample coordinates, folding into an existing matrixEffect
// child instead of nesting when the perspective requirements are compatible.
func MatrixEffectFP(matrix geom.Matrix, child FragmentProcessor) FragmentProcessor {
	if child.ClassID() == MatrixEffectClassID {
		me := child.(*matrixEffect)
		// registerChild's sample usage records whether the matrix used has perspective or not, so we can't add
		// perspective to 'me' if it doesn't already have it.
		if me.matrix.HasPerspective() || !matrix.HasPerspective() {
			me.matrix.PreConcat(&matrix)
			return child
		}
	}
	fp := &matrixEffect{matrix: matrix}
	fp.initFP(MatrixEffectClassID, fpProcessorOptimizationFlags(child))
	registerChildOf(fp, child, UniformMatrixSampleUsage(matrix.HasPerspective()))
	return fp
}

func (f *matrixEffect) Name() string { return "MatrixEffect" }

func (f *matrixEffect) Clone() FragmentProcessor {
	clone := &matrixEffect{matrix: f.matrix}
	clone.initFP(MatrixEffectClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	return clone
}

func (f *matrixEffect) onAddToKey(*gpu.ShaderCaps, *gpu.KeyBuilder) {}

func (f *matrixEffect) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*matrixEffect)
	return ok && f.matrix == o.matrix
}

func (f *matrixEffect) constantOutputForConstantInput(input colorcore.PMColor4f) colorcore.PMColor4f {
	return fpConstantOutputForConstantInput(f.ChildProcessor(0), input)
}

func (f *matrixEffect) onMakeProgramImpl() FPProgramImpl { return &matrixEffectImpl{} }

type matrixEffectImpl struct {
	FPImplBase
	matrixVar UniformHandle
}

func (i *matrixEffectImpl) EmitCode(args *FPEmitArgs) {
	i.matrixVar, _ = args.UniformHandler.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeFloat3x3, MatrixUniformName())
	args.FragBuilder.CodeAppendf("return %s;\n", i.InvokeChildWithMatrix(0, args))
}

func (i *matrixEffectImpl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	mtx := fp.(*matrixEffect)
	if te, ok := mtx.ChildProcessor(0).(*TextureEffect); ok {
		// The texture effect's y-flip/normalization concat happens here because the texture size is not always known
		// when the effect is made (fully-lazy proxies).
		m := te.coordAdjustmentMatrix()
		m.PreConcat(&mtx.matrix)
		pdman.SetMatrix(i.matrixVar, &m)
	} else {
		pdman.SetMatrix(i.matrixVar, &mtx.matrix)
	}
}
