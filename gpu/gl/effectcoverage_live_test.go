// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Guards against the "silent no-op" class of GPU correctness gap: when MakePaint cannot convert a paint's shader or
// color filter to a fragment processor it returns ok=false, and the GPU device then draws *nothing* — no error, no
// fallback, just a blank result. Two real instances shipped and were caught only later: image shaders and perlin-noise
// shaders, both because MakeShaderFP's default case returned nil for a shader a caller can actually construct.
//
// TestGPUEffectCoverageNoSilentNoOp asserts every paint-attachable shader and color-filter family in the public
// shaders/colorfilter packages (the surface unison constructs effects through) converts (MakePaint returns ok=true).
// TestEffectCoverageIsExhaustive then parses those packages' sources and
// fails if a new exported shader/color-filter constructor is added without either a coverage case or an explicit
// filter-internal exclusion here — so a future family cannot silently regress the same way. Skips the live-draw
// assertions when no GL context is available; the exhaustiveness guard needs no GL.

package gl_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu/gl"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// shaderCase is one paint-attachable shader family and the shaders-package constructor it stands in for.
type shaderCase struct {
	shader shaders.Shader
	coreFn string // the shaders constructor this case represents (see TestEffectCoverageIsExhaustive)
	name   string
}

// colorFilterCase is one paint-attachable color-filter family and the colorfilter-package constructor it stands in for.
type colorFilterCase struct {
	filter shaders.ColorFilter
	coreFn string
	name   string
}

// shaderCoverageCases builds a representative, non-empty instance of every paint-attachable shader family the shaders
// package exports. Constructing these needs no GL context (NewImage just wraps the image; the upload happens later
// during conversion). Each coreFn must name a real exported shaders constructor returning Shader (asserted by
// TestEffectCoverageIsExhaustive).
func shaderCoverageCases(img *imagecore.Image) []shaderCase {
	grad := []colorcore.Color{colorcore.Red, colorcore.Blue}
	base := shaders.NewColor(colorcore.White)
	cf := colorfilter.NewBlend(colorcore.Green, raster.BlendSrcOver)
	nearest := shaders.SamplingOptions{}
	return []shaderCase{
		{coreFn: "shaders.NewColor", name: "color", shader: shaders.NewColor(colorcore.Red)},
		{
			coreFn: "shaders.NewBlend", name: "blend",
			shader: shaders.NewBlend(raster.BlendSrcOver, base, shaders.NewColor(colorcore.Blue)),
		},
		{
			coreFn: "shaders.NewLinearGradient", name: "linear-gradient",
			shader: shaders.NewLinearGradient(geom.Point{}, geom.Point{X: 16}, grad, nil, shaders.TileClamp, nil),
		},
		{
			coreFn: "shaders.NewRadialGradient", name: "radial-gradient",
			shader: shaders.NewRadialGradient(geom.Point{X: 8, Y: 8}, 8, grad, nil, shaders.TileClamp, nil),
		},
		{
			coreFn: "shaders.NewSweepGradient", name: "sweep-gradient",
			shader: shaders.NewSweepGradient(geom.Point{X: 8, Y: 8}, grad, nil, shaders.TileClamp, 0, 360, nil),
		},
		{
			coreFn: "shaders.NewTwoPointConicalGradient", name: "conical-gradient",
			shader: shaders.NewTwoPointConicalGradient(geom.Point{X: 3, Y: 3}, 1, geom.Point{X: 8, Y: 8}, 5,
				grad, nil, shaders.TileClamp, nil),
		},
		{
			coreFn: "shaders.NewFractalNoise", name: "perlin-fractal",
			shader: shaders.NewFractalNoise(0.1, 0.1, 2, 0, 0, 0),
		},
		{
			coreFn: "shaders.NewTurbulence", name: "perlin-turbulence",
			shader: shaders.NewTurbulence(0.1, 0.1, 2, 0, 0, 0),
		},
		{
			coreFn: "shaders.NewImage", name: "image",
			shader: shaders.NewImage(img, shaders.TileClamp, shaders.TileClamp, nearest, nil),
		},
		{
			coreFn: "shaders.NewImageDrawable", name: "image-drawable",
			shader: shaders.NewImageDrawable(img, shaders.TileClamp, shaders.TileClamp, nearest, nil),
		},
		{coreFn: "shaders.NewWithColorFilter", name: "color-filter-shader", shader: shaders.NewWithColorFilter(base, cf)},
		{
			coreFn: "shaders.NewWithLocalMatrix", name: "local-matrix",
			shader: shaders.NewWithLocalMatrix(base, geom.ScaleMatrix(2, 2)),
		},
	}
}

// colorFilterCoverageCases builds a representative instance of every paint-attachable color-filter family the
// colorfilter package exports.
func colorFilterCoverageCases() []colorFilterCase {
	identity := [20]float32{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}
	return []colorFilterCase{
		{coreFn: "colorfilter.NewMatrix", name: "color-matrix", filter: colorfilter.NewMatrix(&identity)},
		{coreFn: "colorfilter.NewBlend", name: "blend-mode", filter: colorfilter.NewBlend(colorcore.Green, raster.BlendSrcOver)},
		{coreFn: "colorfilter.NewLighting", name: "lighting", filter: colorfilter.NewLighting(colorcore.White, colorcore.Black)},
		{coreFn: "colorfilter.NewLuma", name: "luma", filter: colorfilter.NewLuma()},
		{
			coreFn: "colorfilter.NewHighContrast", name: "high-contrast",
			filter: colorfilter.NewHighContrast(colorfilter.HighContrastConfig{InvertStyle: colorfilter.InvertLightness}),
		},
		{
			coreFn: "colorfilter.NewCompose", name: "compose",
			filter: colorfilter.NewCompose(colorfilter.NewBlend(colorcore.Green, raster.BlendSrcOver),
				colorfilter.NewMatrix(&identity)),
		},
	}
}

func TestGPUEffectCoverageNoSilentNoOp(t *testing.T) {
	_, dc := newLiveDirectContext(t)
	img := rgbaTestImage(4, 4)
	ctm := geom.IdentityMatrix()
	white := colorcore.Color4f{R: 1, G: 1, B: 1, A: 1}

	convert := func(t *testing.T, tag string, pp *gl.PaintParams) {
		t.Helper()
		sdc := newLiveSDC(t, dc, 16)
		defer sdc.Release()
		gpuPaint, ok := gl.MakePaint(sdc, pp, ctm)
		if !ok {
			t.Fatalf("%s: MakePaint returned ok=false — the paint would be a silent no-op on "+
				"the GPU device (a MakeShaderFP/MakeColorFilterFP conversion returned nil for a "+
				"publicly constructible effect)", tag)
		}
		if gpuPaint == nil {
			t.Fatalf("%s: MakePaint returned a nil paint with ok=true", tag)
		}
	}

	for _, tc := range shaderCoverageCases(img) {
		t.Run("shader/"+tc.name, func(t *testing.T) {
			if tc.shader == nil {
				t.Fatalf("%s (%s): shader construction returned nil", tc.name, tc.coreFn)
			}
			convert(t, "shader "+tc.name, &gl.PaintParams{
				Color: white, BlendMode: raster.BlendSrcOver, Shader: tc.shader,
			})
		})
	}

	// Color filters gate the draw through two distinct paths: with no shader the filter folds into the paint color
	// (FilterColor4f), and with a shader present it converts to an FP (MakeColorFilterFP). Both return ok=false on
	// failure, so both are silent-no-op surfaces.
	for _, tc := range colorFilterCoverageCases() {
		t.Run("colorfilter/"+tc.name, func(t *testing.T) {
			if tc.filter == nil {
				t.Fatalf("%s (%s): color-filter construction returned nil", tc.name, tc.coreFn)
			}
			convert(t, "colorfilter "+tc.name+" (constant-color fold)", &gl.PaintParams{
				Color: white, BlendMode: raster.BlendSrcOver, ColorFilter: tc.filter,
			})
			convert(t, "colorfilter "+tc.name+" (over shader FP)", &gl.PaintParams{
				Color: white, BlendMode: raster.BlendSrcOver,
				Shader: shaders.NewColor(colorcore.Red), ColorFilter: tc.filter,
			})
		})
	}
}

// filterInternalConstructors are exported shaders-package constructors that are NOT paint-attachable public effect
// families: they are the image-filter shader-evaluation kernels (and helpers) that only the image-filter engine
// instantiates, wired into GPU rendering through the image-filter backend rather than through MakePaint. They are
// consciously excluded from the silent-no-op guard; adding a new *paint-attachable* constructor still trips
// TestEffectCoverageIsExhaustive.
var filterInternalConstructors = map[string]bool{
	"shaders.Empty":                true, // renders nothing by contract
	"shaders.NewColor4f":           true, // the float form of the covered NewColor family
	"shaders.NewColorFilterShader": true, // internal form of NewWithColorFilter (alpha-modulating)
	"shaders.NewFilterDecal":       true, // image-filter kernel
	"shaders.NewLinearMorphology":  true, // image-filter kernel
	"shaders.NewSparseMorphology":  true, // image-filter kernel
	"shaders.NewDisplacement":      true, // image-filter kernel
	"shaders.NewNormal":            true, // image-filter kernel
	"shaders.NewMagnifier":         true, // image-filter kernel
	"shaders.NewMatrixConv":        true, // image-filter kernel
	"shaders.NewLighting":          true, // image-filter kernel
	"shaders.NewArithmeticBlend":   true, // image-filter kernel
}

// TestEffectCoverageIsExhaustive parses the shaders and colorfilter package sources and asserts that every exported
// constructor returning a Shader or ColorFilter has either a coverage case above or an explicit filter-internal
// exclusion: adding a new paint-attachable effect constructor without wiring a GPU conversion case here fails this
// test, forcing the silent-no-op question to be answered.
func TestEffectCoverageIsExhaustive(t *testing.T) {
	covered := map[string]bool{}
	for _, tc := range shaderCoverageCases(rgbaTestImage(4, 4)) {
		covered[tc.coreFn] = true
	}
	for _, tc := range colorFilterCoverageCases() {
		covered[tc.coreFn] = true
	}

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")

	// isEffectResult reports whether a constructor's single result type marks it as an effect constructor in the given
	// package: the shaders package declares `Shader` (or a concrete `*ColorShader`), the colorfilter package declares
	// `shaders.ColorFilter`.
	isEffectResult := func(pkg string, expr ast.Expr) bool {
		switch typ := expr.(type) {
		case *ast.Ident:
			return (pkg == "shaders" && typ.Name == "Shader") ||
				(pkg == "colorfilter" && typ.Name == "ColorFilter")
		case *ast.StarExpr:
			ident, identOK := typ.X.(*ast.Ident)
			return identOK && pkg == "shaders" && strings.HasSuffix(ident.Name, "Shader")
		case *ast.SelectorExpr:
			return pkg == "colorfilter" && typ.Sel.Name == "ColorFilter"
		}
		return false
	}

	constructors := map[string]bool{}
	fset := token.NewFileSet()
	for _, pkg := range []string{"shaders", "colorfilter"} {
		dir := filepath.Join(root, pkg)
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("reading %s: %v", dir, err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			path := filepath.Join(dir, e.Name())
			parsed, parseErr := parser.ParseFile(fset, path, nil, 0)
			if parseErr != nil {
				t.Fatalf("parsing %s: %v", path, parseErr)
			}
			for _, decl := range parsed.Decls {
				fn, fnOK := decl.(*ast.FuncDecl)
				if !fnOK || fn.Recv != nil || !fn.Name.IsExported() {
					continue // methods and unexported helpers are not effect constructors
				}
				if fn.Type.Results == nil || len(fn.Type.Results.List) != 1 {
					continue
				}
				if !isEffectResult(pkg, fn.Type.Results.List[0].Type) {
					continue
				}
				constructors[pkg+"."+fn.Name.Name] = true
			}
		}
	}
	if len(constructors) == 0 {
		t.Fatal("scanned shaders/colorfilter but found no constructors — scan is broken")
	}

	var missing, stale []string
	for name := range constructors {
		if !covered[name] && !filterInternalConstructors[name] {
			missing = append(missing, name)
		}
	}
	for name := range covered {
		if !constructors[name] {
			stale = append(stale, name)
		}
	}
	sort.Strings(missing)
	sort.Strings(stale)
	if len(missing) > 0 {
		t.Errorf("shader/color-filter constructors with NO GPU-conversion coverage case (add one to "+
			"shaderCoverageCases/colorFilterCoverageCases, or a documented filterInternalConstructors "+
			"exclusion, or they can silently no-op on the GPU device): %v", missing)
	}
	if len(stale) > 0 {
		t.Errorf("coverage cases naming constructors that no longer exist (remove or rename the "+
			"coreFn): %v", stale)
	}
}
