// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The shader-to-PDF-pattern dispatcher: a paint shader becomes a PDF pattern object. Gradients route to
// gradientshader.go; image shaders and the generic fallback route to imageshader.go.

package pdf

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

// getShaderLocalMatrix returns the local matrix of a local-matrix shader, or identity for any other shader.
func getShaderLocalMatrix(s shaders.Shader) geom.Matrix {
	if lm, ok := s.(*shaders.LocalMatrixShader); ok {
		return lm.LocalMatrix()
	}
	return geom.IdentityMatrix()
}

// adjustColor returns the color to pair with shader when building its PDF pattern: an alpha-only image shader keeps the
// paint color (the image supplies coverage); every other shader preserves only the paint's alpha.
func adjustColor(shader shaders.Shader, paintColor colorcore.Color4f) colorcore.Color4f {
	if img, _, _, _, ok := shaders.AsImage(shader); ok {
		if img.IsAlphaOnly() {
			return paintColor
		}
	}
	return colorcore.Color4f{R: 0, G: 0, B: 0, A: paintColor.A}
}

// makeShader turns shader into a PDF pattern object, or returns an invalid reference when it cannot be represented (an
// empty surface bounds, or a non-invertible transform). paintColor carries the paint's color at full alpha (the paint
// alpha is applied via the graphic state).
func makeShader(doc *Document, shader shaders.Shader, canvasTransform geom.Matrix, surfaceBBox geom.IRect, paintColor colorcore.Color4f) IndirectReference {
	if g, ok := shaders.AsGradientShader(shader); ok {
		shaderTransform := getShaderLocalMatrix(shader)
		return makeGradientShader(doc, g, shaderTransform, canvasTransform, surfaceBBox)
	}
	if surfaceBBox.IsEmpty() {
		return IndirectReference{}
	}
	paintColor = adjustColor(shader, paintColor)
	if img, tmx, tmy, shaderTransform, ok := shaders.AsImage(shader); ok {
		var finalMatrix geom.Matrix
		finalMatrix.SetConcat(&canvasTransform, &shaderTransform)
		key := imageShaderKeyString(finalMatrix, surfaceBBox, newKeyedImage(img).key(), tmx, tmy, paintColor)
		if ref, cached := doc.imageShaderMap[key]; cached {
			return ref
		}
		pdfShader := makeImageShader(doc, finalMatrix, tmx, tmy, surfaceBBox.ToRect(), img, paintColor)
		doc.imageShaderMap[key] = pdfShader
		return pdfShader
	}
	// Don't bother to de-dup the fallback shader.
	return makeFallbackShader(doc, shader, canvasTransform, surfaceBBox, paintColor)
}
