// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests for the default geometry processor's coverage lanes: the emitter and the uniform uploader must agree about when
// the fragment Coverage uniform exists, since a declared-but-never-written uniform reads as zero coverage (an invisible
// draw).

package gl

import (
	"slices"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// makeCoverageProgramInfo builds a minimal src-over program info around a defaultGeoProc with the given coverage
// configuration.
func makeCoverageProgramInfo(t *testing.T, dc *DirectContext, view *SurfaceProxyView, coverageType DefaultGeoProcCoverageType, coverage uint8) *ProgramInfo {
	t.Helper()
	paint := NewPaint()
	paint.SetPorterDuffXPFactory(raster.BlendSrcOver)
	ps := NewProcessorSetFromPaint(paint)
	clip := MakeAppliedClip(geom.ISize{Width: 32, Height: 32})
	white := colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}
	ps.Finalize(AnalysisColorConstant(white), AnalysisCoverageSingleChannel, &clip,
		UnusedStencilSettings(), dc.Caps(), gpu.ClampTypeAuto)
	pipeline := NewPipeline(&PipelineInitArgs{Caps: dc.GLCaps(), WriteSwizzle: gpu.SwizzleRGBA}, ps,
		&clip)
	identity := geom.IdentityMatrix()
	gp := MakeDefaultGeoProc(ColorTypePremulAttribute, white, coverageType, coverage,
		LocalCoordsTypeUnused, nil, &identity)
	return NewProgramInfo(dc.GLCaps(), view, false, pipeline, UnusedStencilSettings(), gp,
		gpu.PrimitiveTypeTriangles, 0, gpu.LoadOpLoad)
}

// TestDefaultGeoProcCoverageUniformUploaded pins the emitter/uploader agreement for every coverage configuration that
// declares the fragment Coverage uniform. The tweak-alpha case is the interesting one: it has a vertex coverage
// attribute (folded into the color in the vertex shader) yet still reads the uniform for the extra non-opaque coverage,
// so keying the upload on "no vertex coverage" alone left it unwritten.
func TestDefaultGeoProcCoverageUniformUploaded(t *testing.T) {
	for _, c := range []struct {
		name         string
		coverageType DefaultGeoProcCoverageType
		coverage     uint8
		wantUniform  bool
	}{
		{name: "uniform", coverageType: CoverageTypeUniform, coverage: 0x80, wantUniform: true},
		{name: "tweak alpha partial", coverageType: CoverageTypeAttributeTweakAlpha, coverage: 0x80, wantUniform: true},
		{name: "tweak alpha opaque", coverageType: CoverageTypeAttributeTweakAlpha, coverage: 0xff},
		{name: "vertex passthrough", coverageType: CoverageTypeAttribute, coverage: 0x80},
	} {
		t.Run(c.name, func(t *testing.T) {
			dc := newShaderRecordingContext(t)
			proxy := makeDeferredProxy(t, dc, geom.ISize{Width: 32, Height: 32},
				gpu.RenderableYes, gpu.BackingFitExact)
			defer proxy.Unref()
			if !proxy.Instantiate(dc.ResourceProvider()) {
				t.Fatal("render target proxy instantiation failed")
			}
			rt := proxy.PeekRenderTarget()
			if rt == nil {
				t.Fatal("instantiated renderable proxy has no render target")
			}
			view := MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, gpu.SwizzleRGBA)
			info := makeCoverageProgramInfo(t, dc, &view, c.coverageType, c.coverage)
			program := dc.Gpu().programCache.FindOrCreateProgram(info)
			if program == nil {
				t.Fatal("program creation failed")
			}

			declared := strings.Contains(program.FragmentSource(), "uCoverage")
			if declared != c.wantUniform {
				t.Fatalf("fragment shader declares uCoverage = %v, want %v:\n%s", declared,
					c.wantUniform, program.FragmentSource())
			}

			// Capture the 1f uploads so the declared uniform is proven written with the coverage value.
			var uploaded []float32
			f := &dc.Gpu().ctx.Interface.Functions
			prev := f.fnUniform1f
			f.fnUniform1f = func(_ int32, v float32) { uploaded = append(uploaded, v) }
			t.Cleanup(func() { f.fnUniform1f = prev })
			program.UpdateUniforms(rt, info)

			want := float32(c.coverage) / 255
			if got := slices.Contains(uploaded, want); got != c.wantUniform {
				t.Errorf("coverage %v uploaded = %v, want %v (uploads: %v)", want, got,
					c.wantUniform, uploaded)
			}
		})
	}
}
