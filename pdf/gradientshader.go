// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A gradient shader becomes a PDF pattern. Clamp/decal non-perspective gradients emit as native PDF axial (type 2) /
// radial (type 3) shadings with a Type 2/3 stitching color function; repeat/mirror, perspective, and sweep gradients
// emit as a Function-based (type 1) shading whose Type 4 PostScript function maps device coordinates to t and then to a
// color. Gradients with alpha are wrapped in a tiling pattern that paints the color shading under a luminosity soft
// mask carrying the alpha. Identical gradients share one pattern object (Document.gradientPatternMap).
//
// GradientFlags is always 0 in the public surface (the legacy default interpolates in unpremultiplied sRGB), so the
// premultiplied-interpolation lanes (isPremul, unpremulCode, the four-component color functions) are implemented in
// full but never taken.

package pdf

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/stream"
)

// gradientKey holds everything that determines the emitted pattern for a gradient.
type gradientKey struct {
	info            shaders.GradientInfo
	typ             shaders.GradientType
	canvasTransform geom.Matrix
	shaderTransform geom.Matrix
	bbox            geom.IRect
}

// gradientInterpolateInPremulFlag marks a gradient whose stop colors should be interpolated in premultiplied space
// rather than the default unpremultiplied sRGB.
const gradientInterpolateInPremulFlag = 1

// isPremul reports whether info's gradient interpolates stop colors in premultiplied space.
func isPremul(info *shaders.GradientInfo) bool {
	return info.GradientFlags&gradientInterpolateInPremulFlag != 0
}

// ---- small color helpers ----------------------------------------------------------------------------

func color4fComponent(c colorcore.Color4f, i int) float32 {
	switch i {
	case 0:
		return c.R
	case 1:
		return c.G
	case 2:
		return c.B
	default:
		return c.A
	}
}

func color4fMakeOpaque(c colorcore.Color4f) colorcore.Color4f {
	c.A = 1
	return c
}

func color4fEqual(a, b colorcore.Color4f) bool { return a == b }

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// gradientKeyString serializes a key into a collision-free byte string suitable for use as a map key in
// Document.gradientPatternMap.
func gradientKeyString(k *gradientKey) string {
	var b []byte
	putU32 := func(v uint32) { b = append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }
	putF := func(v float32) { putU32(math.Float32bits(v)) }
	putU32(uint32(k.typ))
	putU32(uint32(len(k.info.Colors)))
	for _, c := range k.info.Colors {
		putF(c.R)
		putF(c.G)
		putF(c.B)
		putF(c.A)
	}
	for _, o := range k.info.ColorOffsets {
		putF(o)
	}
	for _, p := range k.info.Point {
		putF(p.X)
		putF(p.Y)
	}
	putF(k.info.Radius[0])
	putF(k.info.Radius[1])
	putU32(uint32(k.info.TileMode))
	putU32(k.info.GradientFlags)
	for _, v := range k.canvasTransform.As9() {
		putF(v)
	}
	for _, v := range k.shaderTransform.As9() {
		putF(v)
	}
	putU32(uint32(k.bbox.Left))
	putU32(uint32(k.bbox.Top))
	putU32(uint32(k.bbox.Right))
	putU32(uint32(k.bbox.Bottom))
	return string(b)
}

// unitToPointsMatrix returns the matrix mapping the unit segment (0,0)-(1,0) onto pts[0]-pts[1].
func unitToPointsMatrix(pts [2]geom.Point) geom.Matrix {
	vec := pts[1].Sub(pts[0])
	mag := vec.Length()
	inv := float32(0)
	if mag != 0 {
		inv = geom.ScalarInvert(mag)
	}
	vec.X *= inv
	vec.Y *= inv
	var m geom.Matrix
	m.SetSinCos(vec.Y, vec.X)
	m.PreScale(mag, mag)
	m.PostTranslate(pts[0].X, pts[0].Y)
	return m
}

// interpolateColorCode emits PostScript that interpolates between prevColor and curColor over the range, assuming
// t-startOffset is on the stack.
func interpolateColorCode(rng float32, numComponents int, prevColor, curColor colorcore.Color4f, result stream.WStream) {
	var multiplier [4]float32
	for i := 0; i < numComponents; i++ {
		multiplier[i] = (color4fComponent(curColor, i) - color4fComponent(prevColor, i)) / rng
	}
	var dupInput [4]bool
	dupInput[numComponents-1] = false
	for i := numComponents - 2; i >= 0; i-- {
		dupInput[i] = dupInput[i+1] || multiplier[i+1] != 0
	}
	if !dupInput[0] && multiplier[0] == 0 {
		writeText(result, "pop ")
	}
	for i := 0; i < numComponents; i++ {
		if dupInput[i] && multiplier[i] != 0 {
			writeText(result, "dup ")
		}
		if multiplier[i] == 0 {
			writeColorComponentF(result, color4fComponent(prevColor, i))
			writeText(result, " ")
		} else {
			if multiplier[i] != 1 {
				writeScalar(result, multiplier[i])
				writeText(result, " mul ")
			}
			if color4fComponent(prevColor, i) != 0 {
				writeColorComponentF(result, color4fComponent(prevColor, i))
				writeText(result, " add ")
			}
		}
		if dupInput[i] {
			writeText(result, "exch ")
		}
	}
}

// unpremulCode emits PostScript implementing { r g b a } -> a==0 ? {0 0 0} : { r/a g/a b/a }.
func unpremulCode(function stream.WStream) {
	function.Write([]byte("dup abs 0.00001 lt" +
		"{ pop pop pop pop 0 0 0 }" +
		"{" +
		" dup" + // r g b a a
		" 3 1 roll" + // r g a b a
		" div" + // r g a b/a
		" 4 1 roll" + // b/a r g a
		" dup" + // b/a r g a a
		" 3 1 roll" + // b/a r a g a
		" div" + // b/a r a g/a
		" 4 1 roll" + // g/a b/a r a
		" div" + // g/a b/a r/a
		" 3 1 roll" + // r/a g/a b/a
		"} ifelse\n"))
}

// writeGradientRanges emits PostScript that binary-searches over the gradient's stop ranges, running an interpolation
// for whichever range t falls in.
func writeGradientRanges(info *shaders.GradientInfo, rangeEnds []int, numComponents int, top, first bool, result stream.WStream) {
	rangeEndIndex := rangeEnds[len(rangeEnds)-1]
	rangeEnd := info.ColorOffsets[rangeEndIndex]

	switch {
	case top:
		result.Write([]byte("dup dup 0 gt exch "))
		writeScalar(result, rangeEnd)
		result.Write([]byte(" le and {\n"))
	case first:
		result.Write([]byte("dup "))
		writeScalar(result, rangeEnd)
		result.Write([]byte(" le {\n"))
	default:
		result.Write([]byte("{\n"))
	}

	if len(rangeEnds) == 1 {
		rangeBeginIndex := rangeEndIndex - 1
		rangeBegin := info.ColorOffsets[rangeBeginIndex]
		writeScalar(result, rangeBegin)
		result.Write([]byte(" sub ")) // consume t, put t - startOffset on the stack.
		interpolateColorCode(rangeEnd-rangeBegin, numComponents, info.Colors[rangeBeginIndex],
			info.Colors[rangeEndIndex], result)
		result.Write([]byte("\n"))
	} else {
		loCount := len(rangeEnds) / 2
		writeGradientRanges(info, rangeEnds[:loCount], numComponents, false, true, result)
		writeGradientRanges(info, rangeEnds[loCount:], numComponents, false, false, result)
	}

	switch {
	case top:
		result.Write([]byte("0} if\n"))
	case first:
		result.Write([]byte("}")) // The else (hi) side will come next.
	default:
		result.Write([]byte("} ifelse\n"))
	}
}

// gradientFunctionCode emits the Type 4 PostScript function body mapping t to a clamped gradient color.
func gradientFunctionCode(info *shaders.GradientInfo, result stream.WStream) {
	premul := isPremul(info)
	numComponents := 3
	if premul {
		numComponents = 4
	}

	// The initial range has no previous and contains a solid color; t <= 0 handled here.
	result.Write([]byte("dup 0 le {pop "))
	writeColorComponentF(result, info.Colors[0].R)
	result.Write([]byte(" "))
	writeColorComponentF(result, info.Colors[0].G)
	result.Write([]byte(" "))
	writeColorComponentF(result, info.Colors[0].B)
	if numComponents == 4 {
		result.Write([]byte(" "))
		writeColorComponentF(result, info.Colors[0].A)
	}
	result.Write([]byte(" 0} if\n"))

	// Optimize out ranges which don't make any visual difference.
	colorCount := len(info.Colors)
	rangeEnds := make([]int, 0, colorCount)
	for i := 1; i < colorCount; i++ {
		eqIgnoringAlpha := func(a, b colorcore.Color4f) bool {
			if premul {
				return color4fEqual(a, b)
			}
			return color4fEqual(color4fMakeOpaque(a), color4fMakeOpaque(b))
		}
		constantColorBothSides := eqIgnoringAlpha(info.Colors[i-1], info.Colors[i]) &&
			i != colorCount-1 &&
			eqIgnoringAlpha(info.Colors[i], info.Colors[i+1])
		degenerateRange := info.ColorOffsets[i-1] == info.ColorOffsets[i]
		if !degenerateRange && !constantColorBothSides {
			rangeEnds = append(rangeEnds, i)
		}
	}

	if len(rangeEnds) > 0 {
		writeGradientRanges(info, rangeEnds, numComponents, true, true, result)
	}

	// Clamp the final color.
	result.Write([]byte("0 gt {"))
	writeColorComponentF(result, info.Colors[colorCount-1].R)
	result.Write([]byte(" "))
	writeColorComponentF(result, info.Colors[colorCount-1].G)
	result.Write([]byte(" "))
	writeColorComponentF(result, info.Colors[colorCount-1].B)
	if numComponents == 4 {
		result.Write([]byte(" "))
		writeColorComponentF(result, info.Colors[colorCount-1].A)
	}
	result.Write([]byte("} if\n"))

	if premul {
		unpremulCode(result)
	}
}

// createInterpolationFunction builds a Type 2 exponential interpolation function between two colors.
func createInterpolationFunction(color1, color2 colorcore.Color4f) *Dict {
	retval := NewDict()

	c0 := NewArray()
	c0.AppendColorComponentF(color1.R)
	c0.AppendColorComponentF(color1.G)
	c0.AppendColorComponentF(color1.B)
	retval.InsertObject("C0", c0)

	c1 := NewArray()
	c1.AppendColorComponentF(color2.R)
	c1.AppendColorComponentF(color2.G)
	c1.AppendColorComponentF(color2.B)
	retval.InsertObject("C1", c1)

	retval.InsertObject("Domain", makeArrayInts(0, 1))
	retval.InsertInt("FunctionType", 2)
	retval.InsertScalar("N", 1.0)
	return retval
}

// gradientStitchCode builds a Type 3 stitching function over per-stop Type 2 interpolations, collapsed to a single Type
// 2 function when only two stops remain.
func gradientStitchCode(info *shaders.GradientInfo) *Dict {
	retval := NewDict()

	colorCount := len(info.Colors)
	colors := append([]colorcore.Color4f(nil), info.Colors...)
	colorOffsets := append([]float32(nil), info.ColorOffsets...)

	i := 1
	for i < colorCount-1 {
		// Ensure stops are in order.
		if colorOffsets[i-1] > colorOffsets[i] {
			colorOffsets[i] = colorOffsets[i-1]
		}
		// Remove points that are between two coincident points.
		if colorOffsets[i-1] == colorOffsets[i] && colorOffsets[i] == colorOffsets[i+1] {
			colorCount--
			colors = append(colors[:i], colors[i+1:]...)
			colorOffsets = append(colorOffsets[:i], colorOffsets[i+1:]...)
		} else {
			i++
		}
	}
	// Find coincident points and slightly move them over.
	for i = 1; i < colorCount-1; i++ {
		if colorOffsets[i-1] == colorOffsets[i] {
			colorOffsets[i] += 0.00001
		}
	}
	// Check if last two stops coincide.
	if colorOffsets[i-1] == colorOffsets[i] {
		colorOffsets[i-1] -= 0.00001
	}

	// No need for a stitch function if there are only two stops.
	if colorCount == 2 {
		return createInterpolationFunction(colors[0], colors[1])
	}

	encode := NewArray()
	bounds := NewArray()
	functions := NewArray()

	retval.InsertObject("Domain", makeArrayInts(0, 1))
	retval.InsertInt("FunctionType", 3)

	for idx := 1; idx < colorCount; idx++ {
		if idx > 1 {
			bounds.AppendScalar(colorOffsets[idx-1])
		}
		encode.AppendScalar(0)
		encode.AppendScalar(1.0)
		functions.AppendObject(createInterpolationFunction(colors[idx-1], colors[idx]))
	}

	retval.InsertObject("Encode", encode)
	retval.InsertObject("Bounds", bounds)
	retval.InsertObject("Functions", functions)
	return retval
}

// tileModeCode emits PostScript that maps t into [0,1) for Repeat or Mirror tile modes.
func tileModeCode(mode shaders.TileMode, result stream.WStream) {
	if mode == shaders.TileRepeat {
		result.Write([]byte("dup truncate sub\n"))    // Get the fractional part.
		result.Write([]byte("dup 0 le {1 add} if\n")) // Map (-1,0) => (0,1).
		return
	}
	if mode == shaders.TileMirror {
		// Map t mod 2 into [0, 1, 1, 0]; the `0 gt` (not `1 eq`) works around a Preview 11.0 bug.
		result.Write([]byte("abs " + // +t
			"dup " + // +t.s +t.s
			"truncate " + // +t.s +t
			"dup " + // +t.s +t +t
			"cvi " + // +t.s +t +T
			"2 mod " + // +t.s +t (+T mod 2)
			"0 gt " + // +t.s +t true|false
			"3 1 roll " + // true|false +t.s +t
			"sub " + // true|false 0.s
			"exch " + // 0.s true|false
			"{1 exch sub} if\n")) // 1 - 0.s|0.s
	}
}

// applyPerspectiveToCoordinates emits PostScript that applies the inverse perspective to the top two (x, y) stack
// elements.
func applyPerspectiveToCoordinates(inversePerspectiveMatrix geom.Matrix, code stream.WStream) {
	if !inversePerspectiveMatrix.HasPerspective() {
		return
	}
	p0 := inversePerspectiveMatrix.Get(geom.MPersp0)
	p1 := inversePerspectiveMatrix.Get(geom.MPersp1)
	p2 := inversePerspectiveMatrix.Get(geom.MPersp2)

	code.Write([]byte(" dup ")) // x y y
	writeScalar(code, p1)       // x y y p1
	code.Write([]byte(" mul " + // x y y*p1
		" 2 index ")) // x y y*p1 x
	writeScalar(code, p0)       // x y y p1 x p0
	code.Write([]byte(" mul ")) // x y y*p1 x*p0
	writeScalar(code, p2)       // x y y p1 x*p0 p2
	code.Write([]byte(" add " + // x y y*p1 x*p0+p2
		"add " + // x y y*p1+x*p0+p2
		"3 1 roll " + // y*p1+x*p0+p2 x y
		"2 index " + // z x y y*p1+x*p0+p2
		"div " + // y*p1+x*p0+p2 x y/(...)
		"3 1 roll " + // y/(...) y*p1+x*p0+p2 x
		"exch " + // y/(...) x y*p1+x*p0+p2
		"div " + // y/(...) x/(...)
		"exch\n")) // x/(...) y/(...)
}

// linearCode emits the Type 4 PostScript function body for a linear gradient.
func linearCode(info *shaders.GradientInfo, perspectiveRemover geom.Matrix, function stream.WStream) {
	function.Write([]byte("{"))
	applyPerspectiveToCoordinates(perspectiveRemover, function)
	function.Write([]byte("pop\n")) // Just ditch the y value.
	tileModeCode(info.TileMode, function)
	gradientFunctionCode(info, function)
	function.Write([]byte("}"))
}

// radialCode emits the Type 4 PostScript function body for a radial gradient.
func radialCode(info *shaders.GradientInfo, perspectiveRemover geom.Matrix, function stream.WStream) {
	function.Write([]byte("{"))
	applyPerspectiveToCoordinates(perspectiveRemover, function)
	// Find the distance from the origin.
	function.Write([]byte("dup " + // x y y
		"mul " + // x y^2
		"exch " + // y^2 x
		"dup " + // y^2 x x
		"mul " + // y^2 x^2
		"add " + // y^2+x^2
		"sqrt\n")) // sqrt(y^2+x^2)
	tileModeCode(info.TileMode, function)
	gradientFunctionCode(info, function)
	function.Write([]byte("}"))
}

// twoPointConicalCode emits the Type 4 PostScript function body for the canvas-spec conical gradient.
func twoPointConicalCode(info *shaders.GradientInfo, perspectiveRemover geom.Matrix, function stream.WStream) {
	dx := info.Point[1].X - info.Point[0].X
	dy := info.Point[1].Y - info.Point[0].Y
	r0 := info.Radius[0]
	dr := info.Radius[1] - info.Radius[0]
	a := dx*dx + dy*dy - dr*dr

	function.Write([]byte("{"))
	applyPerspectiveToCoordinates(perspectiveRemover, function)
	function.Write([]byte("2 copy "))

	// b = -2 * (y * dy + x * dx + r0 * dr); leave b and b^2.
	writeScalar(function, dy)
	function.Write([]byte(" mul exch "))
	writeScalar(function, dx)
	function.Write([]byte(" mul add "))
	writeScalar(function, r0*dr)
	function.Write([]byte(" add -2 mul dup dup mul\n"))

	// c = x^2 + y^2 - radius0^2
	function.Write([]byte("4 2 roll dup mul exch dup mul add "))
	writeScalar(function, r0*r0)
	function.Write([]byte(" sub dup 4 1 roll\n"))

	// Stack: c, b, b^2, c
	if a == 0 {
		// t = -c/b
		function.Write([]byte("pop pop div neg dup "))
		writeScalar(function, dr)
		function.Write([]byte(" mul "))
		writeScalar(function, r0)
		function.Write([]byte(" add\n"))
		// If r(t) < 0, then it's outside the cone.
		function.Write([]byte("0 lt {pop false} {true} ifelse\n"))
	} else {
		// Quadratic case: largest root t for which radius(t) > 0.
		writeScalar(function, a*4)
		function.Write([]byte(" mul sub dup\n"))
		function.Write([]byte("0 ge {\n"))
		function.Write([]byte("sqrt exch dup 0 lt {exch -1 mul} if"))
		function.Write([]byte(" add -0.5 mul dup\n"))
		writeScalar(function, a)
		function.Write([]byte(" div\n"))
		function.Write([]byte("3 1 roll div\n"))
		function.Write([]byte("2 copy gt {exch} if\n"))
		function.Write([]byte("dup "))
		writeScalar(function, dr)
		function.Write([]byte(" mul "))
		writeScalar(function, r0)
		function.Write([]byte(" add\n"))
		function.Write([]byte(" 0 gt {exch pop true}\n"))
		function.Write([]byte("{pop dup\n"))
		writeScalar(function, dr)
		function.Write([]byte(" mul "))
		writeScalar(function, r0)
		function.Write([]byte(" add\n"))
		function.Write([]byte("0 le {pop false} {true} ifelse\n"))
		function.Write([]byte("} ifelse\n"))
		function.Write([]byte("} {pop pop pop false} ifelse\n"))
	}

	// If the pixel is in the cone, compute a color; otherwise write black.
	function.Write([]byte("{"))
	tileModeCode(info.TileMode, function)
	gradientFunctionCode(info, function)
	function.Write([]byte("} {0 0 0} ifelse }"))
}

// sweepCode emits the Type 4 PostScript function body for a sweep (angular) gradient. perspectiveRemover is unused:
// sweep gradients have no perspective-dependent coordinate mapping.
func sweepCode(info *shaders.GradientInfo, function stream.WStream) {
	function.Write([]byte("{exch atan 360 div\n"))
	bias := info.Point[1].Y
	if bias != 0 {
		writeScalar(function, bias)
		function.Write([]byte(" add\n"))
	}
	scale := info.Point[1].X
	if scale != 1 {
		writeScalar(function, scale)
		function.Write([]byte(" mul\n"))
	}
	tileModeCode(info.TileMode, function)
	gradientFunctionCode(info, function)
	function.Write([]byte("}"))
}

// fixUpRadius nudges just-touching circles apart slightly so the PDF rendering matches the raster rendering at the
// boundary.
func fixUpRadius(p1 geom.Point, r1 float32, p2 geom.Point, r2 float32) (newR1, newR2 float32) {
	distance := p1.Sub(p2).Length()
	subtractRadii := absf(r1 - r2)
	if absf(distance-subtractRadii) < 0.002 {
		if r1 > r2 {
			r1 += 0.002
		} else {
			r2 += 0.002
		}
	}
	return r1, r2
}

// splitPerspective finds affine and (the inverse of) perspective such that in = affine * persp.
func splitPerspective(in geom.Matrix) (affine, perspectiveInverse geom.Matrix, ok bool) {
	p2 := in.Get(geom.MPersp2)
	if geom.ScalarNearlyZero(p2) {
		return affine, perspectiveInverse, false
	}
	sx := in.Get(geom.MScaleX)
	kx := in.Get(geom.MSkewX)
	tx := in.Get(geom.MTransX)
	ky := in.Get(geom.MSkewY)
	sy := in.Get(geom.MScaleY)
	ty := in.Get(geom.MTransY)
	p0 := in.Get(geom.MPersp0)
	p1 := in.Get(geom.MPersp1)

	perspectiveInverse.SetAll(1, 0, 0, 0, 1, 0, -p0/p2, -p1/p2, 1/p2)
	affine.SetAll(sx-p0*tx/p2, kx-p1*tx/p2, tx/p2, ky-p0*ty/p2, sy-p1*ty/p2, ty/p2, 0, 0, 1)
	return affine, perspectiveInverse, true
}

// makePSFunction emits a Type 4 PostScript calculator function object wrapping psCode.
func makePSFunction(doc *Document, psCode []byte, domain, rangeObj *Array) IndirectReference {
	dict := NewDict()
	dict.InsertInt("FunctionType", 4)
	dict.InsertObject("Domain", domain)
	dict.InsertObject("Range", rangeObj)
	return doc.StreamOut(dict, psCode, true)
}

// pdfShadingType enumerates the PDF shading dictionary types this package can emit.
const (
	shadingFunction = 1
	shadingAxial    = 2
	shadingRadial   = 3
)

// makeFunctionShader emits the PDF pattern object for a gradient, choosing between a native axial/radial shading (with
// a stitched color function) and a Function-based shading whose PostScript function computes t and maps it to a color,
// depending on the gradient's tile mode, perspective, and interpolation space.
func makeFunctionShader(doc *Document, state *gradientKey) IndirectReference {
	info := &state.info
	finalMatrix := state.canvasTransform
	finalMatrix.PreConcat(&state.shaderTransform)

	doStitchFunctions := (state.typ == shaders.GradientLinear || state.typ == shaders.GradientRadial ||
		state.typ == shaders.GradientConical) &&
		(info.TileMode == shaders.TileClamp || info.TileMode == shaders.TileDecal) &&
		!finalMatrix.HasPerspective() && !isPremul(info)

	var shadingType int
	pdfShader := NewDict()
	if doStitchFunctions {
		pdfShader.InsertObject("Function", gradientStitchCode(info))

		if info.TileMode == shaders.TileClamp {
			extend := NewArray()
			extend.AppendBool(true)
			extend.AppendBool(true)
			pdfShader.InsertObject("Extend", extend)
		}

		var coords *Array
		switch state.typ {
		case shaders.GradientLinear:
			shadingType = shadingAxial
			pt1 := info.Point[0]
			pt2 := info.Point[1]
			coords = makeArrayScalars(pt1.X, pt1.Y, pt2.X, pt2.Y)
		case shaders.GradientRadial:
			shadingType = shadingRadial
			pt1 := info.Point[0]
			coords = makeArrayScalars(pt1.X, pt1.Y, 0, pt1.X, pt1.Y, info.Radius[0])
		case shaders.GradientConical:
			shadingType = shadingRadial
			r1 := info.Radius[0]
			r2 := info.Radius[1]
			pt1 := info.Point[0]
			pt2 := info.Point[1]
			r1, r2 = fixUpRadius(pt1, r1, pt2, r2)
			coords = makeArrayScalars(pt1.X, pt1.Y, r1, pt2.X, pt2.Y, r2)
		default:
			return IndirectReference{}
		}
		pdfShader.InsertObject("Coords", coords)
	} else {
		shadingType = shadingFunction

		// Transform the coordinate space for the type of gradient.
		var transformPoints [2]geom.Point
		transformPoints[0] = info.Point[0]
		transformPoints[1] = info.Point[1]
		switch state.typ {
		case shaders.GradientLinear:
		case shaders.GradientRadial:
			transformPoints[1] = transformPoints[0]
			transformPoints[1].X += info.Radius[0]
		case shaders.GradientConical:
			transformPoints[1] = transformPoints[0]
			transformPoints[1].X++
		case shaders.GradientSweep:
			transformPoints[1] = transformPoints[0]
			transformPoints[1].X++
		default:
			return IndirectReference{}
		}

		// Move scaling/translation (and rotation for linear) into the matrix so the gradient runs on the unit segment.
		mapperMatrix := unitToPointsMatrix(transformPoints)
		finalMatrix.PreConcat(&mapperMatrix)

		// Remove only the perspective into perspectiveInverseOnly so the shader handles it efficiently.
		perspectiveInverseOnly := geom.IdentityMatrix()
		if finalMatrix.HasPerspective() {
			affine, persp, ok := splitPerspective(finalMatrix)
			if !ok {
				return IndirectReference{}
			}
			finalMatrix = affine
			perspectiveInverseOnly = persp
		}

		bbox := state.bbox.ToRect()
		if !inverseTransformBBox(&finalMatrix, &bbox) {
			return IndirectReference{}
		}

		functionCode := stream.NewMemoryWStream()
		switch state.typ {
		case shaders.GradientLinear:
			linearCode(info, perspectiveInverseOnly, functionCode)
		case shaders.GradientRadial:
			radialCode(info, perspectiveInverseOnly, functionCode)
		case shaders.GradientConical:
			inverseMapperMatrix, ok := mapperMatrix.Invert()
			if !ok {
				return IndirectReference{}
			}
			infoCopy := *info
			inverseMapperMatrix.MapPointsInPlace(infoCopy.Point[:])
			infoCopy.Radius[0] = inverseMapperMatrix.MapRadius(info.Radius[0])
			infoCopy.Radius[1] = inverseMapperMatrix.MapRadius(info.Radius[1])
			twoPointConicalCode(&infoCopy, perspectiveInverseOnly, functionCode)
		case shaders.GradientSweep:
			sweepCode(info, functionCode)
		}

		pdfShader.InsertObject("Domain", makeArrayScalars(bbox.Left, bbox.Right, bbox.Top, bbox.Bottom))
		domain := makeArrayScalars(bbox.Left, bbox.Right, bbox.Top, bbox.Bottom)
		rangeObject := makeArrayInts(0, 1, 0, 1, 0, 1)
		pdfShader.InsertRef("Function", makePSFunction(doc, functionCode.Bytes(), domain, rangeObject))
	}

	pdfShader.InsertInt("ShadingType", int32(shadingType))
	pdfShader.InsertName("ColorSpace", "DeviceRGB")

	pdfFunctionShader := NewTypedDict("Pattern")
	pdfFunctionShader.InsertInt("PatternType", 2)
	pdfFunctionShader.InsertObject("Matrix", matrixToArray(&finalMatrix))
	pdfFunctionShader.InsertObject("Shading", pdfShader)
	return doc.Emit(pdfFunctionShader)
}

// getGradientResourceDict builds the resource dictionary listing the pattern and graphic state a gradient's content
// stream references.
func getGradientResourceDict(functionShader, gState IndirectReference) *Dict {
	var patternShaders, graphicStates []IndirectReference
	if functionShader.IsValid() {
		patternShaders = append(patternShaders, functionShader)
	}
	if gState.IsValid() {
		graphicStates = append(graphicStates, gState)
	}
	return makeResourceDict(graphicStates, patternShaders, nil, nil)
}

// createPatternFillContent builds a content stream filling bounds with pattern patternIndex, optionally applying
// graphic state gsIndex first.
func createPatternFillContent(gsIndex, patternIndex int, bounds geom.Rect) []byte {
	content := stream.NewMemoryWStream()
	if gsIndex >= 0 {
		applyGraphicState(gsIndex, content)
	}
	applyPattern(patternIndex, content)
	appendRectangle(bounds, content)
	paintPath(styleFill, path.FillEvenOdd, content)
	return content.Bytes()
}

// gradientHasAlpha reports whether any stop color in the gradient key has alpha less than 1.
func gradientHasAlpha(key *gradientKey) bool {
	for _, c := range key.info.Colors {
		if c.A != 1 {
			return true
		}
	}
	return false
}

// cloneKey returns a deep copy of k whose color/offset edits don't affect the source key.
func cloneKey(k *gradientKey) gradientKey {
	clone := *k
	clone.info.Colors = append([]colorcore.Color4f(nil), k.info.Colors...)
	clone.info.ColorOffsets = append([]float32(nil), k.info.ColorOffsets...)
	return clone
}

// createSMaskGraphicState builds a luminosity soft mask carrying the gradient's alpha as a grayscale color shading.
func createSMaskGraphicState(doc *Document, state *gradientKey) IndirectReference {
	luminosityState := cloneKey(state)
	for i := range luminosityState.info.Colors {
		alpha := luminosityState.info.Colors[i].A
		luminosityState.info.Colors[i] = colorcore.Color4f{R: alpha, G: alpha, B: alpha, A: 1}
	}
	luminosityState.info.GradientFlags &^= gradientInterpolateInPremulFlag

	luminosityShader := findPDFShader(doc, &luminosityState, false)
	resources := getGradientResourceDict(luminosityShader, IndirectReference{})
	bbox := state.bbox.ToRect()
	identity := geom.IdentityMatrix()
	alphaMask := makeFormXObject(doc, createPatternFillContent(-1, int(luminosityShader.value), bbox),
		rectToArray(bbox), resources, &identity, "DeviceRGB")
	return getSMaskGraphicState(alphaMask, false, sMaskLuminosity, doc)
}

// makeAlphaFunctionShader builds a tiling pattern that paints the opaque color shading under the alpha soft mask.
func makeAlphaFunctionShader(doc *Document, state *gradientKey) IndirectReference {
	opaqueState := cloneKey(state)
	keepAlpha := isPremul(&opaqueState.info)
	if !keepAlpha {
		for i := range opaqueState.info.Colors {
			opaqueState.info.Colors[i].A = 1
		}
	}
	bbox := state.bbox.ToRect()
	colorShader := findPDFShader(doc, &opaqueState, false)
	if !colorShader.IsValid() {
		return IndirectReference{}
	}
	// Resource dict: alpha graphic state as the graphic state, color shading as the pattern.
	alphaGsRef := createSMaskGraphicState(doc, state)
	resourceDict := getGradientResourceDict(colorShader, alphaGsRef)
	colorStream := createPatternFillContent(int(alphaGsRef.value), int(colorShader.value), bbox)
	alphaFunctionShader := NewDict()
	identity := geom.IdentityMatrix()
	populateTilingPatternDict(alphaFunctionShader, bbox, resourceDict, &identity)
	return doc.StreamOut(alphaFunctionShader, colorStream, true)
}

// makeKey builds the gradientKey for g under the given transforms and clip bounding box, premultiplying stop colors
// when the gradient requests premultiplied interpolation.
func makeKey(g shaders.Gradient, shaderTransform, canvasTransform geom.Matrix, bbox geom.IRect) gradientKey {
	typ, info := g.AsGradient()
	key := gradientKey{
		typ:             typ,
		info:            info,
		canvasTransform: canvasTransform,
		shaderTransform: shaderTransform,
		bbox:            bbox,
	}
	if isPremul(&key.info) {
		// Premultiply colors; drop the flag if nothing changed. Not reachable through the public surface (flags=0).
		changedByPremul := false
		for i := range key.info.Colors {
			c := key.info.Colors[i]
			if c.A != 1 {
				changedByPremul = true
			}
			key.info.Colors[i] = colorcore.Color4f{R: c.R * c.A, G: c.G * c.A, B: c.B * c.A, A: c.A}
		}
		if !changedByPremul {
			key.info.GradientFlags &^= gradientInterpolateInPremulFlag
		}
	}
	return key
}

// findPDFShader returns the pattern object for key, building it (as an alpha or plain function shader per
// makeAlphaShader) and caching it in Document.gradientPatternMap on first use.
func findPDFShader(doc *Document, key *gradientKey, makeAlphaShader bool) IndirectReference {
	ks := gradientKeyString(key)
	if ref, ok := doc.gradientPatternMap[ks]; ok {
		return ref
	}
	var ref IndirectReference
	if makeAlphaShader {
		ref = makeAlphaFunctionShader(doc, key)
	} else {
		ref = makeFunctionShader(doc, key)
	}
	doc.gradientPatternMap[ks] = ref
	return ref
}

// makeGradientShader emits (or reuses) the PDF pattern object for gradient g under the given transforms and clip
// bounding box.
func makeGradientShader(doc *Document, g shaders.Gradient, shaderTransform, canvasTransform geom.Matrix, bbox geom.IRect) IndirectReference {
	key := makeKey(g, shaderTransform, canvasTransform, bbox)
	makeAlphaShader := gradientHasAlpha(&key)
	return findPDFShader(doc, &key, makeAlphaShader)
}
