// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package filtercore provides the shared layer-space machinery every image filter evaluates in — the Mapping that
// decomposes the CTM into parameter-to-layer and layer-to-device transforms, the epsilon-tolerant layer-space bounds
// algebra, the lazy FilterResult with its crop/transform/color-filter combining rules, the evaluation Context with its
// Backend, and the raster blur engine. Coordinate spaces (parameter/device/layer) are tracked via names and comments
// rather than distinct types. All matrices are 3x3 geom.Matrix.
package filtercore

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/shaders"
)

// roundEpsilon covers float imprecision where infinite precision would produce integers (crbug.com/1313579).
const roundEpsilon = 1e-3

// RoundOut rounds r outward to integers, but with a tolerance.
func RoundOut(r geom.Rect) geom.IRect {
	return r.Inset(roundEpsilon, roundEpsilon).RoundOut()
}

// RoundIn rounds r inward to integers, but with a tolerance.
func RoundIn(r geom.Rect) geom.IRect {
	return r.Outset(roundEpsilon, roundEpsilon).RoundIn()
}

// saturate2Int converts x to int32, saturating to the int32 range and mapping NaN to 0.
func saturate2Int(x float64) int32 {
	switch {
	case x > float64(math.MaxInt32):
		return math.MaxInt32
	case x < float64(math.MinInt32):
		return math.MinInt32
	case math.IsNaN(x):
		return 0 // matches the saturating-cast convention used throughout this package
	default:
		return int32(x)
	}
}

// mapIRect maps r through m, preserving 1px precision for scale-translate matrices by computing in doubles, else
// falling back to RoundOut of the float rect mapping.
func mapIRect(m *geom.Matrix, r geom.IRect) geom.IRect {
	if r.IsEmpty() {
		return geom.IRect{}
	}
	if m.IsScaleTranslate() {
		sx := float64(m.Get(geom.MScaleX))
		sy := float64(m.Get(geom.MScaleY))
		tx := float64(m.Get(geom.MTransX))
		ty := float64(m.Get(geom.MTransY))
		l := sx*float64(r.Left) + tx
		rt := sx*float64(r.Right) + tx
		t := sy*float64(r.Top) + ty
		b := sy*float64(r.Bottom) + ty
		return geom.IRect{
			Left:   saturate2Int(math.Floor(math.Min(l, rt) + roundEpsilon)),
			Top:    saturate2Int(math.Floor(math.Min(t, b) + roundEpsilon)),
			Right:  saturate2Int(math.Ceil(math.Max(l, rt) - roundEpsilon)),
			Bottom: saturate2Int(math.Ceil(math.Max(t, b) - roundEpsilon)),
		}
	}
	return RoundOut(mapRect(m, r.ToRect()))
}

// mapRect maps r through m: empty maps to empty; perspective routes through path.MapRect (see the perspective note on
// geom.Matrix.MapRect).
func mapRect(m *geom.Matrix, r geom.Rect) geom.Rect {
	if r.IsEmpty() {
		return geom.Rect{}
	}
	mapped, _ := path.MapRect(m, r)
	return mapped
}

// mapPoint maps point p through m.
func mapPoint(m *geom.Matrix, p geom.Point) geom.Point {
	return m.MapPoint(p)
}

// mapVector maps vector v through m (translation-invariant).
func mapVector(m *geom.Matrix, v geom.Point) geom.Point {
	return m.MapVector(v)
}

// mapSize maps size s through m: sizes are non-positioned axis lengths.
func mapSize(m *geom.Matrix, s geom.Size) geom.Size {
	if m.IsScaleTranslate() {
		// Equivalent to mapping the two basis vectors and calculating their lengths.
		sizes := m.MapVector(geom.Point{X: s.Width, Y: s.Height})
		return geom.Size{Width: absf32(sizes.X), Height: absf32(sizes.Y)}
	}
	xAxis := m.MapVector(geom.Point{X: s.Width, Y: 0})
	yAxis := m.MapVector(geom.Point{X: 0, Y: s.Height})
	return geom.Size{Width: xAxis.Length(), Height: yAxis.Length()}
}

// invertOrIdentity returns the inverse of m, or the identity matrix when m is not invertible. geom.Matrix's zero value
// is the all-zeros matrix, which maps every coordinate to (0,0), so it must never stand in for a failed inversion.
func invertOrIdentity(m *geom.Matrix) geom.Matrix {
	if inv, ok := m.Invert(); ok {
		return inv
	}
	return geom.IdentityMatrix()
}

// mapMatrix re-expresses a transform m (operating in space C1) in the space C2 that 'matrix' maps C1 into: matrix * m *
// matrix^-1.
func mapMatrix(transform, matrix *geom.Matrix) geom.Matrix {
	inv := invertOrIdentity(matrix)
	inv.PostConcat(transform)
	inv.PostConcat(matrix)
	return inv
}

// sizeCeil rounds s up to integers, with rounding tolerance matched to RoundOut.
func sizeCeil(s geom.Size) geom.ISize {
	return geom.ISize{
		Width:  geom.CeilToInt(s.Width - roundEpsilon),
		Height: geom.CeilToInt(s.Height - roundEpsilon),
	}
}

// SizeCeil is the exported form of sizeCeil for filter implementations.
func SizeCeil(s geom.Size) geom.ISize { return sizeCeil(s) }

// RelevantSubset returns the smallest subset of r needed to fill dstRect under the given tile mode.
func RelevantSubset(r, dstRect geom.IRect, tileMode shaders.TileMode) geom.IRect {
	fitted := r
	if tileMode == shaders.TileDecal || tileMode == shaders.TileClamp {
		// For both decal/clamp, we only care about the region that is in dstRect, unless we are clamping and have to
		// preserve edge pixels when there's no overlap.
		if !fitted.Intersect(dstRect) {
			if tileMode == shaders.TileDecal {
				// The dstRect would be filled with transparent black.
				fitted = geom.IRect{}
			} else {
				// We just need the closest row/column/corner of this rect to dstRect.
				fitted = closestDisjointEdge(r, dstRect)
			}
		}
	} // else assume the entire source is needed for periodic tile modes
	return fitted
}

// closestDisjointEdge returns the closest row/column/corner of src to dst when the two rects don't overlap.
func closestDisjointEdge(src, dst geom.IRect) geom.IRect {
	if src.IsEmpty() || dst.IsEmpty() {
		return geom.IRect{}
	}
	l, r := src.Left, src.Right
	switch {
	case r <= dst.Left:
		l = r - 1 // right column of src
	case l >= dst.Right:
		r = l + 1 // left column of src
	default:
		l = pin32(l, dst.Left, dst.Right)
		r = pin32(r, dst.Left, dst.Right)
	}
	t, b := src.Top, src.Bottom
	switch {
	case b <= dst.Top:
		t = b - 1 // bottom row of src
	case t >= dst.Bottom:
		b = t + 1 // top row of src
	default:
		t = pin32(t, dst.Top, dst.Bottom)
		b = pin32(b, dst.Top, dst.Bottom)
	}
	return geom.IRect{Left: l, Top: t, Right: r, Bottom: b}
}

// inverseMapRect inverse-maps r through m; empty inverse-maps to empty "successfully".
func inverseMapRect(m *geom.Matrix, r geom.Rect) (geom.Rect, bool) {
	if r.IsEmpty() {
		return geom.Rect{}, true
	}
	inv, ok := m.Invert()
	if !ok {
		return geom.Rect{}, false
	}
	return mapRect(&inv, r), true
}

// inverseMapIRect inverse-maps r through m, with specialized 1px-preserving double math for scale-translate matrices.
func inverseMapIRect(m *geom.Matrix, r geom.IRect) (geom.IRect, bool) {
	if r.IsEmpty() {
		return geom.IRect{}, true
	}
	if m.IsScaleTranslate() {
		sx := float64(m.Get(geom.MScaleX))
		sy := float64(m.Get(geom.MScaleY))
		if sx == 0 || sy == 0 {
			return geom.IRect{}, false
		}
		tx := float64(m.Get(geom.MTransX))
		ty := float64(m.Get(geom.MTransY))
		l := (float64(r.Left) - tx) / sx
		rt := (float64(r.Right) - tx) / sx
		t := (float64(r.Top) - ty) / sy
		b := (float64(r.Bottom) - ty) / sy
		return geom.IRect{
			Left:   saturate2Int(math.Floor(math.Min(l, rt) + roundEpsilon)),
			Top:    saturate2Int(math.Floor(math.Min(t, b) + roundEpsilon)),
			Right:  saturate2Int(math.Ceil(math.Max(l, rt) - roundEpsilon)),
			Bottom: saturate2Int(math.Ceil(math.Max(t, b) - roundEpsilon)),
		}, true
	}
	mapped, ok := inverseMapRect(m, r.ToRect())
	if !ok {
		return geom.IRect{}, false
	}
	return RoundOut(mapped), true
}

// MapIRect is the exported form of mapIRect for filter implementations.
func MapIRect(m *geom.Matrix, r geom.IRect) geom.IRect { return mapIRect(m, r) }

// MapLayerRect is the exported form of mapRect for filter implementations.
func MapLayerRect(m *geom.Matrix, r geom.Rect) geom.Rect { return mapRect(m, r) }

// InverseMapIRect is the exported form of inverseMapIRect for filter implementations.
func InverseMapIRect(m *geom.Matrix, r geom.IRect) (geom.IRect, bool) {
	return inverseMapIRect(m, r)
}

// rectToRectOrIdentity returns the matrix mapping from onto to, or the identity matrix if either rect is empty.
func rectToRectOrIdentity(from, to geom.Rect) geom.Matrix {
	if from.IsEmpty() || to.IsEmpty() {
		return geom.IdentityMatrix()
	}
	var m geom.Matrix
	sx := to.Width() / from.Width()
	sy := to.Height() / from.Height()
	m.SetScaleTranslate(sx, sy, to.Left-from.Left*sx, to.Top-from.Top*sy)
	return m
}

///////////////////////////////////////////////////////////////////////////////
// integer-translation analysis

// areAxesNearlyIntegerAligned reports whether m is (nearly) an axis-aligned integer translation, per axis. When both
// axes are aligned and out is non-nil, *out receives the integer translation.
func areAxesNearlyIntegerAligned(m *geom.Matrix, out *geom.IPoint) (xAxis, yAxis bool) {
	invW := ieeeFloatDivide(1, m.Get(geom.MPersp2))
	tx := roundToScalar(m.Get(geom.MTransX) * invW)
	ty := roundToScalar(m.Get(geom.MTransY) * invW)
	// expected = [1 0 tx] after normalizing perspective (divide by m[2,2])
	//            [0 1 ty]
	//            [0 0  1]
	affine := nearlyEqual(m.Get(geom.MPersp0)*invW, 0, roundEpsilon) &&
		nearlyEqual(m.Get(geom.MPersp1)*invW, 0, roundEpsilon)
	if !affine {
		return false, false
	}
	xAxis = nearlyEqual(1, m.Get(geom.MScaleX)*invW, roundEpsilon) &&
		nearlyEqual(0, m.Get(geom.MSkewX)*invW, roundEpsilon) &&
		nearlyEqual(tx, m.Get(geom.MTransX)*invW, roundEpsilon)
	yAxis = nearlyEqual(0, m.Get(geom.MSkewY)*invW, roundEpsilon) &&
		nearlyEqual(1, m.Get(geom.MScaleY)*invW, roundEpsilon) &&
		nearlyEqual(ty, m.Get(geom.MTransY)*invW, roundEpsilon)
	if out != nil && xAxis && yAxis {
		*out = geom.IPoint{X: int32(tx), Y: int32(ty)}
	}
	return xAxis, yAxis
}

// isNearlyIntegerTranslation reports whether m is (nearly) an integer translation on both axes.
func isNearlyIntegerTranslation(m *geom.Matrix, out *geom.IPoint) bool {
	xAxis, yAxis := areAxesNearlyIntegerAligned(m, out)
	return xAxis && yAxis
}

// decomposeTransform splits transform into postScaling * scaling. Either output may be nil.
func decomposeTransform(transform *geom.Matrix, representativePoint geom.Point, postScaling, scaling *geom.Matrix) {
	if scale, remaining, ok := decomposeScale(transform); ok {
		if postScaling != nil {
			*postScaling = remaining
		}
		scaling.SetScale(scale.Width, scale.Height)
		return
	}
	// Perspective, which has a non-uniform scaling effect on the filter. Pick a single scale factor that best matches
	// where the filter will be evaluated.
	approxScale := differentialAreaScale(transform, representativePoint)
	if geom.IsFinite(approxScale) && !nearlyZero(approxScale) {
		// Take the sqrt to go from an area scale factor to a scaling per X and Y.
		approxScale = sqrtf32(approxScale)
	} else {
		// The point was behind the W = 0 plane, so don't factor out any scale.
		approxScale = 1
	}
	if postScaling != nil {
		*postScaling = *transform
		invScale := 1 / approxScale
		var s geom.Matrix
		s.SetScale(invScale, invScale)
		postScaling.PreConcat(&s)
	}
	scaling.SetScale(approxScale, approxScale)
}

// periodicAxisTransform reports whether a periodic tiling of crop covers output with a single (possibly mirrored)
// instance, returning the transform that replaces the tiling when it does.
func periodicAxisTransform(tileMode shaders.TileMode, crop, output geom.IRect) (geom.Matrix, bool) {
	if tileMode == shaders.TileClamp || tileMode == shaders.TileDecal {
		return geom.Matrix{}, false // not periodic
	}

	// Lift crop dimensions into doubles so combining with output cannot overflow 32 bits.
	cropL := float64(crop.Left)
	cropT := float64(crop.Top)
	cropWidth := float64(crop.Right) - cropL
	cropHeight := float64(crop.Bottom) - cropT

	// Calculate normalized periodic coordinates of 'output' relative to the 'crop' being tiled.
	periodL := math.Floor((float64(output.Left) - cropL) / cropWidth)
	periodT := math.Floor((float64(output.Top) - cropT) / cropHeight)
	periodR := math.Ceil((float64(output.Right) - cropL) / cropWidth)
	periodB := math.Ceil((float64(output.Bottom) - cropT) / cropHeight)

	if periodR-periodL <= 1.0 && periodB-periodT <= 1.0 {
		// The tiling pattern won't be visible, so we can draw the image without tiling and an adjusted transform,
		// computed in doubles to be exact.
		sx := float32(1)
		sy := float32(1)
		tx := -cropL
		ty := -cropT
		if tileMode == shaders.TileMirror {
			// Flip image when in odd periods on each axis (the periods hold integer values).
			if math.Mod(periodL, 2) > float64(geom.ScalarNearlyZeroTol) {
				sx = -1
				tx = cropWidth - tx
			}
			if math.Mod(periodT, 2) > float64(geom.ScalarNearlyZeroTol) {
				sy = -1
				ty = cropHeight - ty
			}
		}
		// Translate by periods and make relative to crop's top left again.
		tx += periodL*cropWidth + cropL
		ty += periodT*cropHeight + cropT

		// A float matrix would lose the pixel precision required to represent this periodic tiling, so don't apply the
		// optimization in that case.
		if float64(saturate2Int(tx)) != float64(float32(tx)) ||
			float64(saturate2Int(ty)) != float64(float32(ty)) {
			return geom.Matrix{}, false
		}
		var periodicTransform geom.Matrix
		periodicTransform.SetScaleTranslate(sx, sy, float32(tx), float32(ty))
		return periodicTransform, true
	}
	// Both low and high edges of the crop would be visible in 'output', or a mirrored boundary is visible: keep the
	// periodic tiling.
	return geom.Matrix{}, false
}

///////////////////////////////////////////////////////////////////////////////
// Mapping

// MatrixCapability describes how complex a layer matrix a filter (or filter DAG) can handle.
type MatrixCapability int32

// MatrixCapability values.
const (
	MatrixCapabilityTranslate MatrixCapability = iota
	MatrixCapabilityScaleTranslate
	MatrixCapabilityComplex
)

// Mapping is the decomposition of the total CTM into the parameter-to-layer matrix and the layer-to-device matrix, with
// the cached inverse of the latter.
type Mapping struct {
	layerToDevMatrix   geom.Matrix
	paramToLayerMatrix geom.Matrix
	devToLayerMatrix   geom.Matrix
}

// MappingOf builds a Mapping where device and layer space are the same coordinate space.
func MappingOf(paramToLayer geom.Matrix) Mapping {
	return Mapping{
		layerToDevMatrix:   geom.IdentityMatrix(),
		paramToLayerMatrix: paramToLayer,
		devToLayerMatrix:   geom.IdentityMatrix(),
	}
}

// NewMapping builds a Mapping from the explicit decomposition, with devToLayer assumed to already hold layerToDev's
// inverse.
func NewMapping(layerToDev, devToLayer, paramToLayer geom.Matrix) Mapping {
	return Mapping{
		layerToDevMatrix:   layerToDev,
		paramToLayerMatrix: paramToLayer,
		devToLayerMatrix:   devToLayer,
	}
}

// DecomposeCTM decomposes ctm into the mapping's layer/device matrices, given the filter DAG's matrix capability and a
// representative point (used when perspective forces an approximation).
func (m *Mapping) DecomposeCTM(ctm *geom.Matrix, capability MatrixCapability, representativePt geom.Point) bool {
	var remainder, layer geom.Matrix
	switch {
	case capability == MatrixCapabilityTranslate:
		// Apply the entire CTM post-filtering.
		remainder = *ctm
		layer = geom.IdentityMatrix()
	case ctm.IsScaleTranslate() || capability == MatrixCapabilityComplex:
		// Either layer space can be anything (MatrixCapabilityComplex), or it can be scale+translate and the ctm is.
		// In both cases the layer space can be equivalent to device space.
		remainder = geom.IdentityMatrix()
		layer = *ctm
	default:
		// This case implies some amount of sampling post-filtering, either due to skew or rotation in the original
		// matrix. As such, keep the layer matrix as simple as possible.
		decomposeTransform(ctm, representativePt, nil, &layer)
		remainder = *ctm
		var invScale geom.Matrix
		invScale.SetScale(1/layer.Get(geom.MScaleX), 1/layer.Get(geom.MScaleY))
		remainder.PreConcat(&invScale)
	}
	invRemainder, ok := remainder.Invert()
	if !ok {
		// Decomposing an invertible matrix can produce a non-invertible remainder under float arithmetic; such matrices
		// are so ill-conditioned that failing to make a layer is okay.
		return false
	}
	m.paramToLayerMatrix = layer
	m.layerToDevMatrix = remainder
	m.devToLayerMatrix = invRemainder
	return true
}

// AdjustLayerSpace post-concatenates 'layer' onto the parameter-to-layer transform and pre-concatenates its inverse
// with layer-to-device, leaving the parameter and device coordinate systems unchanged.
func (m *Mapping) AdjustLayerSpace(layer *geom.Matrix) bool {
	invLayer, ok := layer.Invert()
	if !ok {
		return false
	}
	m.paramToLayerMatrix.PostConcat(layer)
	m.devToLayerMatrix.PostConcat(layer)
	m.layerToDevMatrix.PreConcat(&invLayer)
	return true
}

// ApplyOrigin adjusts the mapping so that origin in the current layer space maps to (0,0) in the adjusted layer space.
func (m *Mapping) ApplyOrigin(origin geom.IPoint) {
	var t geom.Matrix
	t.SetTranslate(float32(-origin.X), float32(-origin.Y))
	if !m.AdjustLayerSpace(&t) {
		panic("applyOrigin: translation must be invertible")
	}
}

// LayerToDevice returns the layer-to-device matrix.
func (m *Mapping) LayerToDevice() geom.Matrix { return m.layerToDevMatrix }

// DeviceToLayer returns the device-to-layer matrix.
func (m *Mapping) DeviceToLayer() geom.Matrix { return m.devToLayerMatrix }

// LayerMatrix returns the parameter-to-layer matrix.
func (m *Mapping) LayerMatrix() geom.Matrix { return m.paramToLayerMatrix }

// TotalMatrix returns the composed parameter-to-device matrix.
func (m *Mapping) TotalMatrix() geom.Matrix {
	var t geom.Matrix
	t.SetConcat(&m.layerToDevMatrix, &m.paramToLayerMatrix)
	return t
}

// ParamToLayerRect maps a parameter-space rect into layer space.
func (m *Mapping) ParamToLayerRect(r geom.Rect) geom.Rect {
	return mapRect(&m.paramToLayerMatrix, r)
}

// ParamToLayerPoint maps a parameter-space point into layer space.
func (m *Mapping) ParamToLayerPoint(p geom.Point) geom.Point {
	return mapPoint(&m.paramToLayerMatrix, p)
}

// ParamToLayerVector maps a parameter-space vector into layer space.
func (m *Mapping) ParamToLayerVector(v geom.Point) geom.Point {
	return mapVector(&m.paramToLayerMatrix, v)
}

// ParamToLayerSize maps a parameter-space size into layer space.
func (m *Mapping) ParamToLayerSize(s geom.Size) geom.Size {
	return mapSize(&m.paramToLayerMatrix, s)
}

// ParamToLayerMatrix re-expresses parameter-space transform t in layer space.
func (m *Mapping) ParamToLayerMatrix(t *geom.Matrix) geom.Matrix {
	return mapMatrix(t, &m.paramToLayerMatrix)
}

// DeviceToLayerIRect maps a device-space rect into layer space. Inverse mapping back to layer space derives the 3x3
// inverse of the flattened layer-to-device matrix; a non-invertible flattened matrix maps to empty.
func (m *Mapping) DeviceToLayerIRect(r geom.IRect) geom.IRect {
	devToLayer, ok := m.layerToDevMatrix.Invert()
	if !ok {
		return geom.IRect{}
	}
	return mapIRect(&devToLayer, r)
}

// LayerToDeviceIRect maps a layer-space rect into device space.
func (m *Mapping) LayerToDeviceIRect(r geom.IRect) geom.IRect {
	return mapIRect(&m.layerToDevMatrix, r)
}

// LayerToDeviceRect maps a layer-space rect into device space.
func (m *Mapping) LayerToDeviceRect(r geom.Rect) geom.Rect {
	return mapRect(&m.layerToDevMatrix, r)
}
