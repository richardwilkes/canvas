// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package gorender renders the scenario corpus through the canvas library (the canvas package over a raster or GL
// device), so the harness can compare its output against the checked-in goldens — the library's own output,
// self-captured per platform by `oracle bless` (see ../goldens/README.md). It materializes each scenario into the
// library's own canvas/paint/path/shader types. This package is cgo-free — it imports only the shipped Go module — so
// the golden gates drive the library exactly as unison does.
//
// It is the harness's only renderer; every gate in this package reads its reference side from ../goldens rather than
// rendering it.
package gorender

import (
	"math"
	"sync"

	gocanvas "github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/colorfilter"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/imagefilter"
	"github.com/richardwilkes/canvas/internal/oracle/scenario"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/patheffect"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stroke"
)

// sceneCanvas adapts a *canvas.Canvas to scenario.Canvas. The scenario enum ordinals and the library's canvas/paint
// enums mirror the same underlying values (see scenario.go), so the mode/style/cap/join/tile/clip-op casts below are
// one-to-one.
type sceneCanvas struct {
	c *gocanvas.Canvas
}

// buildPaint materializes a scenario.Paint into a *canvas.Paint (plus its effects/filters, if any). Nothing needs
// freeing; the result is GC-managed.
func buildPaint(sp scenario.Paint) *gocanvas.Paint {
	p := gocanvas.NewPaint()
	p.Color = sp.Color
	p.Style = gocanvas.Style(sp.Style)
	p.BlendMode = raster.BlendMode(sp.Blend)
	p.StrokeWidth = sp.StrokeWidth
	p.MiterLimit = sp.StrokeMiter
	p.Cap = gocanvas.StrokeCap(sp.Cap)
	p.Join = gocanvas.StrokeJoin(sp.Join)
	p.AntiAlias = sp.AA
	p.Dither = sp.Dither
	if sp.Shader != nil {
		p.Shader = buildShader(sp.Shader)
	}
	if sp.PathEffect != nil {
		p.PathEffect = buildPathEffect(sp.PathEffect)
	}
	if sp.ColorFilter != nil {
		p.ColorFilter = buildColorFilter(sp.ColorFilter)
	}
	if sp.MaskFilter != nil {
		p.MaskFilter = maskfilter.NewBlur(maskfilter.BlurStyle(sp.MaskFilter.Style),
			sp.MaskFilter.Sigma, sp.MaskFilter.RespectCTM)
	}
	if sp.ImageFilter != nil {
		p.ImageFilter = buildImageFilter(sp.ImageFilter)
	}
	return p
}

// buildShader materializes a scenario.ShaderSpec into a canvas shader (recursively for blend shaders).
func buildShader(s *scenario.ShaderSpec) shaders.Shader {
	switch s.Kind {
	case scenario.ShaderLinearGradient:
		return shaders.NewLinearGradient(s.Pts[0], s.Pts[1], s.Colors, s.Pos, shaders.TileMode(s.Tile), s.Local)
	case scenario.ShaderRadialGradient:
		return shaders.NewRadialGradient(s.Center, s.Radius, s.Colors, s.Pos, shaders.TileMode(s.Tile), s.Local)
	case scenario.ShaderSweepGradient:
		start, end := s.StartAngle, s.EndAngle
		if start == 0 && end == 0 {
			end = 360
		}
		return shaders.NewSweepGradient(s.Center, s.Colors, s.Pos, shaders.TileMode(s.Tile), start, end, s.Local)
	case scenario.ShaderTwoPointConicalGradient:
		return shaders.NewTwoPointConicalGradient(s.Pts[0], s.Radius, s.Pts[1], s.Radius2, s.Colors,
			s.Pos, shaders.TileMode(s.Tile), s.Local)
	case scenario.ShaderColor:
		return shaders.NewColor(s.Color)
	case scenario.ShaderBlend:
		return shaders.NewBlend(raster.BlendMode(s.Blend), buildShader(s.Dst), buildShader(s.Src))
	}
	return nil
}

// buildPathEffect materializes a scenario.PathEffectSpec into a canvas path effect (recursively for sum/compose).
func buildPathEffect(s *scenario.PathEffectSpec) stroke.PathEffect {
	switch s.Kind {
	case scenario.PathEffectDash:
		return patheffect.MakeDash(s.Intervals, s.Phase)
	case scenario.PathEffectCorner:
		return patheffect.MakeCorner(s.Radius)
	case scenario.PathEffectDiscrete:
		return patheffect.MakeDiscrete(s.SegLength, s.Deviation, s.Seed)
	case scenario.PathEffectTrim:
		return patheffect.MakeTrim(s.Start, s.Stop, patheffect.TrimMode(s.Trim))
	case scenario.PathEffect1DPath:
		return patheffect.MakePath1D(s.Path.BuildGo(), s.Advance, s.Phase, patheffect.Path1DStyle(s.Style1D))
	case scenario.PathEffect2DLine:
		return patheffect.MakeLine2D(s.Width, s.Matrix)
	case scenario.PathEffect2DPath:
		return patheffect.MakePath2D(s.Matrix, s.Path.BuildGo())
	case scenario.PathEffectSum:
		return patheffect.MakeSum(buildPathEffect(s.A), buildPathEffect(s.B))
	case scenario.PathEffectCompose:
		return patheffect.MakeCompose(buildPathEffect(s.A), buildPathEffect(s.B))
	}
	return nil
}

// buildColorFilter materializes a scenario.ColorFilterSpec into a canvas color filter (recursively for compose).
func buildColorFilter(s *scenario.ColorFilterSpec) shaders.ColorFilter {
	switch s.Kind {
	case scenario.ColorFilterMatrix:
		return colorfilter.NewMatrix(s.Matrix)
	case scenario.ColorFilterBlend:
		return colorfilter.NewBlend(s.Color, raster.BlendMode(s.Blend))
	case scenario.ColorFilterLighting:
		return colorfilter.NewLighting(s.Mul, s.Add)
	case scenario.ColorFilterLuma:
		return colorfilter.NewLuma()
	case scenario.ColorFilterHighContrast:
		return colorfilter.NewHighContrast(colorfilter.HighContrastConfig{
			Grayscale:   s.Grayscale,
			InvertStyle: colorfilter.HighContrastInvertStyle(s.InvertStyle),
			Contrast:    s.Contrast,
		})
	case scenario.ColorFilterCompose:
		return colorfilter.NewCompose(buildColorFilter(s.Outer), buildColorFilter(s.Inner))
	}
	return nil
}

func specPoint3(p scenario.Point3) geom.Point3 { return geom.Point3{X: p.X, Y: p.Y, Z: p.Z} }

// buildImageFilter materializes a scenario.ImageFilterSpec DAG into a canvas image filter. A nil spec yields nil (the
// filtered source), matching the public surface's NULL-input convention.
func buildImageFilter(s *scenario.ImageFilterSpec) filtercore.Filter {
	if s == nil {
		return nil
	}
	sampling := shaders.SamplingOptions{}
	if s.Linear {
		sampling.Filter = shaders.FilterLinear
	}
	switch s.Kind {
	case scenario.FilterBlur:
		return imagefilter.Blur(s.SigmaX, s.SigmaY, shaders.TileMode(s.Tile), buildImageFilter(s.Input), s.Crop)
	case scenario.FilterDropShadow:
		return imagefilter.DropShadow(s.Dx, s.Dy, s.SigmaX, s.SigmaY, s.Color, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterDropShadowOnly:
		return imagefilter.DropShadowOnly(s.Dx, s.Dy, s.SigmaX, s.SigmaY, s.Color,
			buildImageFilter(s.Input), s.Crop)
	case scenario.FilterOffset:
		return imagefilter.Offset(s.Dx, s.Dy, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterColorFilter:
		return imagefilter.ColorFilter(buildColorFilter(s.ColorFilter), buildImageFilter(s.Input), s.Crop)
	case scenario.FilterCompose:
		return imagefilter.Compose(buildImageFilter(s.A), buildImageFilter(s.B))
	case scenario.FilterMerge:
		filters := make([]filtercore.Filter, len(s.Inputs))
		for i, sub := range s.Inputs {
			filters[i] = buildImageFilter(sub)
		}
		return imagefilter.Merge(filters, s.Crop)
	case scenario.FilterArithmetic:
		return imagefilter.Arithmetic(s.K[0], s.K[1], s.K[2], s.K[3], s.EnforcePM,
			buildImageFilter(s.A), buildImageFilter(s.B), s.Crop)
	case scenario.FilterDilate:
		return imagefilter.Dilate(float32(s.RadiusX), float32(s.RadiusY), buildImageFilter(s.Input), s.Crop)
	case scenario.FilterErode:
		return imagefilter.Erode(float32(s.RadiusX), float32(s.RadiusY), buildImageFilter(s.Input), s.Crop)
	case scenario.FilterDisplacement:
		return imagefilter.DisplacementMap(imagefilter.ColorChannel(s.XChannel),
			imagefilter.ColorChannel(s.YChannel), s.Scale, buildImageFilter(s.A), buildImageFilter(s.B), s.Crop)
	case scenario.FilterConvolution:
		return imagefilter.MatrixConvolution(geom.ISize{Width: s.KernelW, Height: s.KernelH}, s.Kernel,
			s.Gain, s.Bias, geom.IPoint{X: s.OffsetX, Y: s.OffsetY}, shaders.TileMode(s.Tile),
			s.ConvolveAlpha, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterMatrixTransform:
		return imagefilter.MatrixTransform(s.Matrix, sampling, buildImageFilter(s.Input))
	case scenario.FilterMagnifier:
		return imagefilter.Magnifier(s.Lens, s.Zoom, s.Inset, sampling, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterDistantLitDiffuse:
		return imagefilter.DistantLitDiffuse(specPoint3(s.Direction), s.LightColor, s.SurfaceScale, s.KD,
			buildImageFilter(s.Input), s.Crop)
	case scenario.FilterPointLitDiffuse:
		return imagefilter.PointLitDiffuse(specPoint3(s.Location), s.LightColor, s.SurfaceScale, s.KD,
			buildImageFilter(s.Input), s.Crop)
	case scenario.FilterSpotLitDiffuse:
		return imagefilter.SpotLitDiffuse(specPoint3(s.Location), specPoint3(s.Target), s.SpecExp,
			s.CutoffAngle, s.LightColor, s.SurfaceScale, s.KD, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterDistantLitSpecular:
		return imagefilter.DistantLitSpecular(specPoint3(s.Direction), s.LightColor, s.SurfaceScale,
			s.KS, s.Shininess, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterPointLitSpecular:
		return imagefilter.PointLitSpecular(specPoint3(s.Location), s.LightColor, s.SurfaceScale,
			s.KS, s.Shininess, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterSpotLitSpecular:
		return imagefilter.SpotLitSpecular(specPoint3(s.Location), specPoint3(s.Target), s.SpecExp,
			s.CutoffAngle, s.LightColor, s.SurfaceScale, s.KS, s.Shininess, buildImageFilter(s.Input), s.Crop)
	case scenario.FilterTile:
		return imagefilter.Tile(s.SrcRect, s.DstRect, buildImageFilter(s.Input))
	}
	return nil
}

func (s sceneCanvas) Clear(c colorcore.Color) { s.c.Clear(c) }

func (s sceneCanvas) DrawColor(c colorcore.Color, mode scenario.BlendMode) {
	s.c.DrawColor(c, raster.BlendMode(mode))
}

func (s sceneCanvas) DrawPaint(sp scenario.Paint) { s.c.DrawPaint(buildPaint(sp)) }

func (s sceneCanvas) DrawRect(r geom.Rect, sp scenario.Paint) { s.c.DrawRect(r, buildPaint(sp)) }

func (s sceneCanvas) DrawOval(r geom.Rect, sp scenario.Paint) { s.c.DrawOval(r, buildPaint(sp)) }

func (s sceneCanvas) DrawCircle(cx, cy, radius float32, sp scenario.Paint) {
	s.c.DrawCircle(cx, cy, radius, buildPaint(sp))
}

func (s sceneCanvas) DrawRoundRect(r geom.Rect, rx, ry float32, sp scenario.Paint) {
	s.c.DrawRoundRect(r, rx, ry, buildPaint(sp))
}

func (s sceneCanvas) DrawArc(oval geom.Rect, startAngle, sweepAngle float32, useCenter bool, sp scenario.Paint) {
	s.c.DrawArc(oval, startAngle, sweepAngle, useCenter, buildPaint(sp))
}

func (s sceneCanvas) DrawLine(x0, y0, x1, y1 float32, sp scenario.Paint) {
	s.c.DrawLine(x0, y0, x1, y1, buildPaint(sp))
}

func (s sceneCanvas) DrawPoints(mode scenario.PointMode, pts []geom.Point, sp scenario.Paint) {
	s.c.DrawPoints(gocanvas.PointMode(mode), pts, buildPaint(sp))
}

func (s sceneCanvas) DrawPath(spec *scenario.PathSpec, sp scenario.Paint) {
	s.c.DrawPath(spec.BuildGo(), buildPaint(sp))
}

// corpusTypefaces lazily loads the corpus text fixtures (scenario.FontDataFor) once per FontID.
var (
	corpusTypefaceMu sync.Mutex
	corpusTypefaces  = map[scenario.FontID]*font.Typeface{}
)

func corpusTypeface(id scenario.FontID) *font.Typeface {
	corpusTypefaceMu.Lock()
	defer corpusTypefaceMu.Unlock()
	tf, ok := corpusTypefaces[id]
	if !ok {
		var err error
		if tf, err = font.NewTypefaceFromData(scenario.FontDataFor(id), 0); err != nil {
			panic(err)
		}
		corpusTypefaces[id] = tf
	}
	return tf
}

func (s sceneCanvas) DrawSimpleText(text string, x, y, size float32, sp scenario.Paint) {
	s.DrawSimpleTextFont(text, x, y, size, scenario.FontRoboto, sp)
}

func (s sceneCanvas) DrawSimpleTextFont(text string, x, y, size float32, id scenario.FontID, sp scenario.Paint) {
	f := font.NewFont(corpusTypeface(id), size, 1, 0)
	s.c.DrawSimpleText([]byte(text), font.TextEncodingUTF8, x, y, f, buildPaint(sp))
}

// atlasImages caches the imagecore materialization of a scenario atlas by its pixel-slice identity (the corpus shares
// one AtlasPixels slice), so repeated DrawAtlas calls — and repeated scenario renders — reuse one image (one GPU
// texture upload per context, keyed by the image's UniqueID).
var (
	atlasImageMu sync.Mutex
	atlasImages  = map[*byte]*imagecore.Image{}
)

func atlasImage(w, h int32, rgba []byte) *imagecore.Image {
	atlasImageMu.Lock()
	defer atlasImageMu.Unlock()
	key := &rgba[0]
	if img, ok := atlasImages[key]; ok {
		return img
	}
	pm := raster.NewPixmap(w, h)
	for y := int32(0); y < h; y++ {
		for x := int32(0); x < w; x++ {
			i := (int(y)*int(w) + int(x)) * 4
			pm.Pix[y*pm.RowPixels+x] = uint32(rgba[i]) | uint32(rgba[i+1])<<8 |
				uint32(rgba[i+2])<<16 | uint32(rgba[i+3])<<24
		}
	}
	img := imagecore.Copy(pm)
	atlasImages[key] = img
	return img
}

func (s sceneCanvas) DrawAtlas(atlasW, atlasH int32, atlasRGBA []byte, xforms []geom.RSXform, tex []geom.Rect, colors []colorcore.Color, mode scenario.BlendMode, linear bool, cull *geom.Rect, sp scenario.Paint) {
	sampling := shaders.SamplingOptions{}
	if linear {
		sampling.Filter = shaders.FilterLinear
	}
	s.c.DrawAtlas(atlasImage(atlasW, atlasH, atlasRGBA), xforms, tex, colors,
		raster.BlendMode(mode), sampling, cull, buildPaint(sp))
}

func (s sceneCanvas) ClipRect(r geom.Rect, op scenario.ClipOp, aa bool) {
	s.c.ClipRect(r, raster.ClipOp(op), aa)
}

func (s sceneCanvas) ClipPath(spec *scenario.PathSpec, op scenario.ClipOp, aa bool) {
	s.c.ClipPath(spec.BuildGo(), raster.ClipOp(op), aa)
}

func (s sceneCanvas) Translate(dx, dy float32) { s.c.Translate(dx, dy) }
func (s sceneCanvas) Scale(sx, sy float32)     { s.c.Scale(sx, sy) }

// RotateRadians maps to the library's degree-based Rotate, converting with radians * 180/pi.
func (s sceneCanvas) RotateRadians(rad float32) {
	s.c.Rotate(float32(float64(rad) * (180.0 / math.Pi)))
}

func (s sceneCanvas) Skew(sx, sy float32)   { s.c.Skew(sx, sy) }
func (s sceneCanvas) Concat(m *geom.Matrix) { s.c.Concat(m) }
func (s sceneCanvas) Save() int             { return s.c.Save() }
func (s sceneCanvas) Restore()              { s.c.Restore() }

func (s sceneCanvas) SaveLayerAlpha(bounds *geom.Rect, alpha uint8) int {
	return s.c.SaveLayerAlpha(bounds, alpha)
}

func (s sceneCanvas) SaveLayer(bounds *geom.Rect, sp *scenario.Paint) int {
	if sp == nil {
		return s.c.SaveLayer(bounds, nil)
	}
	return s.c.SaveLayer(bounds, buildPaint(*sp))
}
