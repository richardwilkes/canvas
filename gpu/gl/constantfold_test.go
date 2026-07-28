// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests that the CPU constant-folding twins of the color-matrix and color-xform processors reproduce the arithmetic
// their own emitted GLSL performs. A folded draw (ProcessorSet.Finalize collapsing a constant color FP chain) must
// render exactly what the unfolded shader would have.

package gl

import (
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// shaderUnpremul is what both processors' emitted GLSL computes: rgb / max(a, 0.0001).
func shaderUnpremul(v, a float32) float32 { return v / max32(a, 1e-4) }

// TestConstantFoldUnpremulMatchesShader covers the guard the shader uses instead of an exact division: for a premul
// input with 0 < a < 1e-4 an exact 1/a explodes (and truncates to a wildly different color), while the shader's
// max(a, 0.0001) floor keeps the result small.
func TestConstantFoldUnpremulMatchesShader(t *testing.T) {
	identity := &[20]float32{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}
	for _, in := range []colorcore.PMColor4f{
		{R: 1e-5, G: 1e-5, B: 1e-5, A: 1e-5}, // the divergent case: a below the shader's floor
		{R: 0.25, G: 0.5, B: 0.125, A: 0.5},  // an ordinary premul color
		{R: 0, G: 0, B: 0, A: 0},             // fully transparent: both must stay at zero
		{R: 1, G: 1, B: 1, A: 1},             // opaque
	} {
		matrixFP := ColorMatrixFP(nil, identity, true /* unpremulInput */, false, false)
		got := matrixFP.(*colorMatrixFP).constantOutputForConstantInput(in)
		want := colorcore.PMColor4f{
			R: shaderUnpremul(in.R, in.A),
			G: shaderUnpremul(in.G, in.A),
			B: shaderUnpremul(in.B, in.A),
			A: in.A,
		}
		if got != want {
			t.Errorf("colorMatrixFP fold of %+v = %+v, want the shader's %+v", in, got, want)
		}

		xformFP := ColorXformFP(nil, XformStepUnpremul)
		got = xformFP.(*colorXformFP).constantOutputForConstantInput(in)
		if got != want {
			t.Errorf("colorXformFP fold of %+v = %+v, want the shader's %+v", in, got, want)
		}
	}
}

// TestConstantFoldUnpremulEmitsGuardedDivide pins the shader side of the pairing: both processors emit the guarded
// divide the CPU twins above mirror, so a change to one is caught here rather than showing up as a color difference
// between a folded and an unfolded draw.
func TestConstantFoldUnpremulEmitsGuardedDivide(t *testing.T) {
	identity := &[20]float32{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}
	for _, c := range []struct {
		makeFP func() FragmentProcessor
		name   string
	}{
		{name: "colorMatrix", makeFP: func() FragmentProcessor {
			return ColorMatrixFP(newTestCoordsFP(), identity, true, false, false)
		}},
		{name: "colorXform", makeFP: func() FragmentProcessor {
			return ColorXformFP(newTestCoordsFP(), XformStepUnpremul)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			dc := newShaderRecordingContext(t)
			proxy := makeDeferredProxy(t, dc, geom.ISize{Width: 32, Height: 32},
				gpu.RenderableYes, gpu.BackingFitExact)
			defer proxy.Unref()
			view := MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, gpu.SwizzleRGBA)
			paint := NewPaint()
			paint.SetPorterDuffXPFactory(raster.BlendSrcOver)
			paint.SetColorFragmentProcessor(c.makeFP())
			ps := NewProcessorSetFromPaint(paint)
			clip := MakeAppliedClip(geom.ISize{Width: 32, Height: 32})
			ps.Finalize(AnalysisColorUnknown(), AnalysisCoverageNone, &clip,
				UnusedStencilSettings(), dc.Caps(), gpu.ClampTypeAuto)
			pipeline := NewPipeline(&PipelineInitArgs{
				Caps:         dc.GLCaps(),
				WriteSwizzle: gpu.SwizzleRGBA,
			}, ps, &clip)
			identityMatrix := geom.IdentityMatrix()
			gp := MakeDefaultGeoProc(ColorTypePremulUniform,
				colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}, CoverageTypeSolid, 0xff,
				LocalCoordsTypeUsePosition, nil, &identityMatrix)
			info := NewProgramInfo(dc.GLCaps(), &view, false, pipeline, UnusedStencilSettings(), gp,
				gpu.PrimitiveTypeTriangles, 0, gpu.LoadOpLoad)
			program := dc.Gpu().programCache.FindOrCreateProgram(info)
			if program == nil {
				t.Fatal("program creation failed")
			}
			if !strings.Contains(program.FragmentSource(), "max(color.a, 0.0001)") {
				t.Errorf("emitted unpremul is not the guarded divide the CPU fold mirrors:\n%s",
					program.FragmentSource())
			}
		})
	}
}
