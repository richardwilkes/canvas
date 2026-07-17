// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The gradient introspection surface: a read-back the PDF backend uses to reconstruct a gradient's stops and geometry.
// This is kept as a small introspection surface separate from the raster-pipeline evaluation stages, matching the
// descriptor/evaluation split the rest of the package uses.

package shaders

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// GradientType identifies which gradient family a shader implements.
type GradientType int

// GradientType values.
const (
	GradientNone GradientType = iota
	GradientColor
	GradientLinear
	GradientRadial
	GradientSweep
	GradientConical
)

// GradientInfo holds the fields reachable through AsGradient. Colors are the stop-fixed unpremultiplied colors and
// ColorOffsets their unit positions; Point/Radius are type-specific (see each AsGradient). GradientFlags is always 0
// (interpolation is always in unpremultiplied sRGB).
type GradientInfo struct {
	Colors        []colorcore.Color4f
	ColorOffsets  []float32
	Point         [2]geom.Point
	Radius        [2]float32
	TileMode      TileMode
	GradientFlags uint32
}

// Gradient is the subset of Shader that reports itself as a gradient. The four gradient families implement it.
type Gradient interface {
	Shader
	// AsGradient returns the gradient's type and stop/geometry read-back.
	AsGradient() (GradientType, GradientInfo)
}

// AsGradientShader reports whether s (possibly behind a local-matrix wrapper) is a gradient, returning the underlying
// Gradient. The PDF backend treats a local-matrix-wrapped gradient as a gradient whose shader transform is the
// wrapper's matrix.
func AsGradientShader(s Shader) (Gradient, bool) {
	if lm, ok := s.(*LocalMatrixShader); ok {
		s = lm.Wrapped()
	}
	g, ok := s.(Gradient)
	return g, ok
}

// getPos returns the explicit offset, or the uniform i/(n-1) when the stops are evenly spaced (initGradientBase folds
// uniform positions to nil).
func (g *gradientBase) getPos(i int) float32 {
	if g.positions != nil {
		return g.positions[i]
	}
	return float32(i) / float32(len(g.colors)-1)
}

// commonGradientInfo returns the colors, offsets, and tile mode shared by every gradient family. GradientFlags stays 0
// (unpremultiplied interpolation).
func (g *gradientBase) commonGradientInfo() GradientInfo {
	n := len(g.colors)
	info := GradientInfo{
		Colors:       append([]colorcore.Color4f(nil), g.colors...),
		ColorOffsets: make([]float32, n),
		TileMode:     g.tileMode,
	}
	for i := 0; i < n; i++ {
		info.ColorOffsets[i] = g.getPos(i)
	}
	return info
}

// AsGradient implements Gradient: Point[0]/[1] are the endpoints.
func (g *LinearGradient) AsGradient() (GradientType, GradientInfo) {
	info := g.commonGradientInfo()
	info.Point[0] = g.start
	info.Point[1] = g.end
	return GradientLinear, info
}

// AsGradient implements Gradient: Point[0]=center, Radius[0]=radius.
func (g *RadialGradient) AsGradient() (GradientType, GradientInfo) {
	info := g.commonGradientInfo()
	info.Point[0] = g.center
	info.Radius[0] = g.radius
	return GradientRadial, info
}

// AsGradient implements Gradient: Point[0]=center, Point[1]={tScale, tBias}.
func (g *SweepGradient) AsGradient() (GradientType, GradientInfo) {
	info := g.commonGradientInfo()
	info.Point[0] = g.center
	info.Point[1] = geom.Point{X: g.tScale, Y: g.tBias}
	return GradientSweep, info
}

// AsGradient implements Gradient: Point[0/1]=centers, Radius[0/1]=radii.
func (g *ConicalGradient) AsGradient() (GradientType, GradientInfo) {
	info := g.commonGradientInfo()
	info.Point[0] = g.center1
	info.Point[1] = g.center2
	info.Radius[0] = g.radius1
	info.Radius[1] = g.radius2
	return GradientConical, info
}
