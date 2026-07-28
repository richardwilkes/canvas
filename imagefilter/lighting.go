// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The lighting image filter nodes: the six distant/point/spot x diffuse/specular constructors, built on the normal-map
// and lighting shader kernels. The emboss material has no reachable entry point and is trimmed. The 3D light
// coordinates follow this convention: X/Y transform as parameter-space points/vectors and Z scales by the average of
// the X and Y scale factors.

package imagefilter

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// light holds the parameters of one of the three light types (distant, point, spot).
type light struct {
	lightType   float32 // shaders.LightDistant/LightPoint/LightSpot
	color       colorcore.Color
	locationXY  geom.Point // spot and point lights only (parameter space)
	locationZ   float32
	directionXY geom.Point // spot and distant lights only (parameter space)
	directionZ  float32
	// Spot only (unchanged by the layer matrix).
	falloffExponent float32
	cosCutoffAngle  float32
}

func pointLight(color colorcore.Color, location geom.Point3) light {
	return light{
		lightType:  shaders.LightPoint,
		color:      color,
		locationXY: geom.Point{X: location.X, Y: location.Y},
		locationZ:  location.Z,
	}
}

func distantLight(color colorcore.Color, direction geom.Point3) light {
	return light{
		lightType:   shaders.LightDistant,
		color:       color,
		directionXY: geom.Point{X: direction.X, Y: direction.Y},
		directionZ:  direction.Z,
	}
}

func spotLight(color colorcore.Color, location, direction geom.Point3, falloffExponent, cosCutoffAngle float32) light {
	return light{
		lightType:       shaders.LightSpot,
		color:           color,
		locationXY:      geom.Point{X: location.X, Y: location.Y},
		locationZ:       location.Z,
		directionXY:     geom.Point{X: direction.X, Y: direction.Y},
		directionZ:      direction.Z,
		falloffExponent: falloffExponent,
		cosCutoffAngle:  cosCutoffAngle,
	}
}

// material holds the surface parameters shared by the diffuse and specular lighting equations (emboss trimmed).
type material struct {
	specular     bool
	surfaceDepth float32 // the scale applied to alpha before computing surface normals
	k            float32 // reflectance coefficient
	shininess    float32 // specular only
}

type lightingFilter struct {
	filterDefaults
	light    light
	material material
}

// makeLighting validates l and mat and builds the lighting filter, applying cropRect to both the input and the output
// when given.
func makeLighting(l light, mat material, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	// According to the SVG spec, ks and kd can be any non-negative number.
	if !geom.IsFinite(mat.k, mat.shininess, mat.surfaceDepth) || mat.k < 0 {
		return nil
	}
	// Ensure light values are finite and the cosine is between -1 and 1.
	if !geom.IsFinite(l.locationXY.X, l.locationXY.Y, l.directionXY.X, l.directionXY.Y,
		l.falloffExponent, l.cosCutoffAngle, l.locationZ, l.directionZ) ||
		l.cosCutoffAngle < -1 || l.cosCutoffAngle > 1 {
		return nil
	}

	// A crop rect clamps both the input (to better match the SVG's normal boundary condition spec) and the output
	// (because otherwise it has infinite bounds).
	filter := input
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	f := &lightingFilter{light: l, material: mat}
	f.base = filtercore.NewFilterBase(filter)
	filter = f
	if cropRect != nil {
		filter = Crop(*cropRect, shaders.TileDecal, filter)
	}
	return filter
}

// DistantLitDiffuse builds a diffuse lighting filter using a distant (directional) light.
func DistantLitDiffuse(direction geom.Point3, lightColor colorcore.Color, surfaceScale, kd float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeLighting(distantLight(lightColor, direction),
		material{surfaceDepth: surfaceScale, k: kd}, input, cropRect)
}

// PointLitDiffuse builds a diffuse lighting filter using a point light at location.
func PointLitDiffuse(location geom.Point3, lightColor colorcore.Color, surfaceScale, kd float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeLighting(pointLight(lightColor, location),
		material{surfaceDepth: surfaceScale, k: kd}, input, cropRect)
}

// SpotLitDiffuse builds a diffuse lighting filter using a spot light aimed from location at target.
func SpotLitDiffuse(location, target geom.Point3, falloffExponent, cutoffAngle float32, lightColor colorcore.Color, surfaceScale, kd float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	dir := geom.Point3{X: target.X - location.X, Y: target.Y - location.Y, Z: target.Z - location.Z}
	cosCutoffAngle := cosDegrees(cutoffAngle)
	return makeLighting(spotLight(lightColor, location, dir, falloffExponent, cosCutoffAngle),
		material{surfaceDepth: surfaceScale, k: kd}, input, cropRect)
}

// DistantLitSpecular builds a specular lighting filter using a distant (directional) light.
func DistantLitSpecular(direction geom.Point3, lightColor colorcore.Color, surfaceScale, ks, shininess float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeLighting(distantLight(lightColor, direction),
		material{specular: true, surfaceDepth: surfaceScale, k: ks, shininess: shininess},
		input, cropRect)
}

// PointLitSpecular builds a specular lighting filter using a point light at location.
func PointLitSpecular(location geom.Point3, lightColor colorcore.Color, surfaceScale, ks, shininess float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	return makeLighting(pointLight(lightColor, location),
		material{specular: true, surfaceDepth: surfaceScale, k: ks, shininess: shininess},
		input, cropRect)
}

// SpotLitSpecular builds a specular lighting filter using a spot light aimed from location at target.
func SpotLitSpecular(location, target geom.Point3, falloffExponent, cutoffAngle float32, lightColor colorcore.Color, surfaceScale, ks, shininess float32, input filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	dir := geom.Point3{X: target.X - location.X, Y: target.Y - location.Y, Z: target.Z - location.Z}
	cosCutoffAngle := cosDegrees(cutoffAngle)
	return makeLighting(spotLight(lightColor, location, dir, falloffExponent, cosCutoffAngle),
		material{specular: true, surfaceDepth: surfaceScale, k: ks, shininess: shininess},
		input, cropRect)
}

// cosDegrees returns the cosine of degrees, converting from degrees to radians first.
func cosDegrees(degrees float32) float32 {
	return float32(math.Cos(float64(degrees * (math.Pi / 180))))
}

// OnAffectsTransparentBlack always reports true: the lighting equation is defined everywhere, even where the input is
// transparent.
func (f *lightingFilter) OnAffectsTransparentBlack() bool { return true }

// GPUEvaluable implements filtercore.GPUEvaluable: the normal + lighting kernels convert to FPs
// (gpu/gl/filterkernelfp.go).
func (f *lightingFilter) GPUEvaluable() bool { return true }

// requiredInput outsets desiredOutput by 1px of padding so the visible normal map can do a regular Sobel kernel eval.
func (f *lightingFilter) requiredInput(desiredOutput geom.IRect) geom.IRect {
	return desiredOutput.Inset(-1, -1)
}

// mapZToLayer maps a parameter-space Z value into layer space by scaling it with the average of the X and Y scale
// factors (so uniform scaling transforms scale the Z axis uniformly too).
func mapZToLayer(mapping *filtercore.Mapping, z float32) float32 {
	z2d := mapping.ParamToLayerVector(geom.Point{X: z, Y: z})
	return (z2d.X + z2d.Y) * 0.5
}

// OnFilterImage builds the normal map from the child's output and evaluates the lighting equation over it.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *lightingFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	// Map lighting and material parameters into layer space.
	surfaceDepth := mapZToLayer(ctx.Mapping(), f.material.surfaceDepth)
	lightLocationXY := ctx.Mapping().ParamToLayerPoint(f.light.locationXY)
	lightLocationZ := mapZToLayer(ctx.Mapping(), f.light.locationZ)
	lightDirXY := ctx.Mapping().ParamToLayerVector(f.light.directionXY)
	lightDirZ := mapZToLayer(ctx.Mapping(), f.light.directionZ)

	// The normal map is determined by a 3x3 kernel, so request a 1px outset of what the lighting equation fills. If the
	// required input is incomplete, boundaries are handled two ways: clamp tiling at the desired output where the
	// actual child edge matches it (approximating the SVG spec's modified Sobel kernels), else the pipeline's default
	// decal tiling.
	requiredInput := f.requiredInput(ctx.DesiredOutput())
	inputCtx := ctx.WithNewDesiredOutput(requiredInput)
	childOutput := f.base.ChildOutput(0, &inputCtx)

	clampRect := requiredInput // effectively no clamping of normals
	if !childOutput.LayerBounds().ContainsRect(requiredInput) {
		// Adjust clampRect edges to desiredOutput if the actual child output matched the lighting output size (typical
		// SVG case); otherwise leave coordinates alone to use decal tiling automatically for pixels outside the child
		// image but inside the desired output.
		edgeClamp := func(actualEdgeValue, requestedEdgeValue, outputEdge int32) int32 {
			if actualEdgeValue == outputEdge {
				return outputEdge
			}
			return requestedEdgeValue
		}
		inputRect := childOutput.LayerBounds()
		clampTo := ctx.DesiredOutput()
		clampRect = geom.IRect{
			Left:   edgeClamp(inputRect.Left, requiredInput.Left, clampTo.Left),
			Top:    edgeClamp(inputRect.Top, requiredInput.Top, clampTo.Top),
			Right:  edgeClamp(inputRect.Right, requiredInput.Right, clampTo.Right),
			Bottom: edgeClamp(inputRect.Bottom, requiredInput.Bottom, clampTo.Bottom),
		}
	}

	builder := filtercore.NewBuilder(&ctx)
	builder.AddSampled(&childOutput, &clampRect, filtercore.ShaderFlagSampledRepeatedly, filtercore.DefaultSampling)
	lgt := f.light
	mat := f.material
	return builder.Eval(func(inputs []shaders.Shader) shaders.Shader {
		// The normal shader's edge bounds are clampRect inset by half a pixel on each side.
		normals := shaders.NewNormal(inputs[0], clampRect.ToRect().Inset(0.5, 0.5), -surfaceDepth)

		// Pack the lighting shader's uniforms. Pre-normalize the light direction; (0,0,0) for point lights stays zero
		// (the uniform is unused).
		dir := [3]float32{lightDirXY.X, lightDirXY.Y, lightDirZ}
		invDirLen := sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2])
		if invDirLen != 0 {
			invDirLen = 1 / invDirLen
		}
		// The light color is intentionally not color-managed; K is applied up front.
		colorScale := mat.k / 255
		return shaders.NewLighting(normals, surfaceDepth, mat.shininess, mat.specular,
			lgt.lightType,
			[3]float32{lightLocationXY.X, lightLocationXY.Y, lightLocationZ},
			lgt.falloffExponent,
			[3]float32{invDirLen * dir[0], invDirLen * dir[1], invDirLen * dir[2]},
			lgt.cosCutoffAngle,
			[3]float32{
				float32(lgt.color.R()) * colorScale,
				float32(lgt.color.G()) * colorScale,
				float32(lgt.color.B()) * colorScale,
			})
	}, nil)
}

// OnInputLayerBounds requires the child's output outset by 1px for the normal-map kernel.
func (f *lightingFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	requiredInput := f.requiredInput(desiredOutput)
	return f.base.InputLayerBounds(0, mapping, requiredInput, contentBounds)
}

// OnOutputLayerBounds always returns nil (unbounded): the lighting equation is defined on the entire plane, even with a
// bounded normal map.
func (f *lightingFilter) OnOutputLayerBounds(_ *filtercore.Mapping, _ *geom.IRect) *geom.IRect {
	return nil
}

// OnComputeFastBounds always returns an unbounded rect (see OnOutputLayerBounds).
func (f *lightingFilter) OnComputeFastBounds(_ geom.Rect) geom.Rect {
	return filtercore.MakeILarge().ToRect()
}
