// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

import "math"

// Matrix element indices: row-major 3x3 with the translation in the third column.
const (
	MScaleX = 0
	MSkewX  = 1
	MTransX = 2
	MSkewY  = 3
	MScaleY = 4
	MTransY = 5
	MPersp0 = 6
	MPersp1 = 7
	MPersp2 = 8
)

// TypeMask bits describing what kind of transform a Matrix performs.
const (
	TypeIdentity    = 0
	TypeTranslate   = 0x01
	TypeScale       = 0x02
	TypeAffine      = 0x04
	TypePerspective = 0x08
)

const (
	maskRectStaysRect        = 0x10
	maskOnlyPerspectiveValid = 0x40
	maskUnknown              = 0x80
	maskORable               = TypeTranslate | TypeScale | TypeAffine | TypePerspective
	maskAllTypes             = maskORable
)

// scalar1Int is the bit pattern of float32(1), used by the type-mask computation.
var scalar1Int = int32(math.Float32bits(1))

// Matrix is a 3x3 float32 matrix with a lazily-computed type mask that routes operations to specialized code paths (the
// specialized paths matter for float-level parity across the pipeline, not just speed).
//
// The zero value is the all-zero (degenerate) matrix, NOT identity — use IdentityMatrix or the setters.
type Matrix struct {
	mat      [9]float32
	typeMask uint8
}

// IdentityMatrix returns the identity matrix.
func IdentityMatrix() Matrix {
	var m Matrix
	m.SetIdentity()
	return m
}

// TranslateMatrix returns a matrix that translates by (dx, dy).
func TranslateMatrix(dx, dy float32) Matrix {
	var m Matrix
	m.SetTranslate(dx, dy)
	return m
}

// ScaleMatrix returns a matrix that scales by (sx, sy).
func ScaleMatrix(sx, sy float32) Matrix {
	var m Matrix
	m.SetScale(sx, sy)
	return m
}

// RotateDegMatrix returns a matrix that rotates by the given number of degrees.
func RotateDegMatrix(degrees float32) Matrix {
	var m Matrix
	m.SetRotate(degrees)
	return m
}

// MatrixFrom9 builds a Matrix from the nine values in row-major order (see the M* index constants).
func MatrixFrom9(v [9]float32) Matrix {
	return Matrix{mat: v, typeMask: maskUnknown}
}

// As9 returns the nine values in row-major order.
func (m *Matrix) As9() [9]float32 {
	return m.mat
}

// Get returns the element at the given M* index.
func (m *Matrix) Get(index int) float32 {
	return m.mat[index]
}

// Set sets the element at the given M* index and marks the type mask dirty.
func (m *Matrix) Set(index int, value float32) {
	m.mat[index] = value
	m.setTypeMask(maskUnknown)
}

// SetAll sets all nine matrix elements directly.
func (m *Matrix) SetAll(scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2 float32) {
	m.mat = [9]float32{scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2}
	m.setTypeMask(maskUnknown)
}

///////////////////////////////////////////////////////////////////////////////
// Type mask machinery

func (m *Matrix) setTypeMask(mask uint8) {
	m.typeMask = mask
}

// orTypeMask goes through storedMask so that a zero-valued Matrix stays "unknown" rather than caching the OR'd bits as
// a clean mask: PreScale is reachable without a prior Set*/Type() call, and its fixups must not claim the all-zero
// matrix is a plain scale.
func (m *Matrix) orTypeMask(mask uint8) {
	m.typeMask = m.storedMask() | mask
}

func (m *Matrix) clearTypeMask(mask uint8) {
	m.typeMask = m.storedMask() &^ mask
}

// storedMask maps the Go zero value (0) to "unknown". A raw zero mask is never stored through the normal code paths
// (identity always carries the rect-stays-rect bit), so this only affects zero-valued Matrix variables, which then
// honestly recompute as the degenerate all-zero matrix.
func (m *Matrix) storedMask() uint8 {
	if m.typeMask == 0 {
		return maskUnknown
	}
	return m.typeMask
}

// Type returns the matrix's type mask, computing and caching it when dirty.
func (m *Matrix) Type() uint8 {
	mask := m.storedMask()
	if mask&maskUnknown != 0 {
		mask = m.computeTypeMask()
		m.typeMask = mask
	}
	return mask & 0xF
}

// perspectiveTypeOnly is cheaper than Type when only the perspective bit is needed.
func (m *Matrix) perspectiveTypeOnly() uint8 {
	mask := m.storedMask()
	if mask&maskUnknown != 0 && mask&maskOnlyPerspectiveValid == 0 {
		mask = m.computeTypeMask()
		m.typeMask = mask
	}
	return mask & 0xF
}

// HasPerspective reports whether the matrix has a non-affine (perspective) component.
func (m *Matrix) HasPerspective() bool {
	return m.perspectiveTypeOnly()&TypePerspective != 0
}

// IsIdentity reports whether the matrix is the identity matrix.
func (m *Matrix) IsIdentity() bool {
	return m.Type() == TypeIdentity
}

// IsScaleTranslate reports whether the matrix is at most a scale plus translate.
func (m *Matrix) IsScaleTranslate() bool {
	return m.Type() <= TypeScale|TypeTranslate
}

// IsTranslate reports whether the matrix is at most a translate.
func (m *Matrix) IsTranslate() bool {
	return m.Type() <= TypeTranslate
}

// isTriviallyIdentity is true only if the cached mask already proves identity; it never recomputes.
func (m *Matrix) isTriviallyIdentity() bool {
	mask := m.storedMask()
	if mask&maskUnknown != 0 {
		return false
	}
	return mask&0xF == 0
}

// RectStaysRect reports whether mapping an axis-aligned rect through the matrix produces another axis-aligned rect.
func (m *Matrix) RectStaysRect() bool {
	mask := m.storedMask()
	if mask&maskUnknown != 0 {
		mask = m.computeTypeMask()
		m.typeMask = mask
	}
	return mask&maskRectStaysRect != 0
}

// IsFinite reports whether all nine elements are finite.
func (m *Matrix) IsFinite() bool {
	return IsFinite(m.mat[0], m.mat[1], m.mat[2], m.mat[3], m.mat[4], m.mat[5], m.mat[6], m.mat[7], m.mat[8])
}

// isDegenerate2x2 reports whether the upper-left 2x2 has a near-zero determinant. The explicit float32 conversions
// force both products to round, forbidding the compiler from fusing a multiply into the subtract (Go fuses on arm64);
// the result is compared against a near-zero tolerance, so fusion would flip the classification per platform. Pinned
// unfused for the same reason as Point.Cross.
func isDegenerate2x2(scaleX, skewX, skewY, scaleY float32) bool {
	perpDot := float32(scaleX*scaleY) - float32(skewX*skewY)
	return ScalarNearlyZeroTolerance(perpDot, ScalarNearlyZeroTol*ScalarNearlyZeroTol)
}

// PreservesRightAngles reports, using the default tolerance, whether the matrix maps a rect to another rect or to a
// rotated rect (the upper 2x2 basis vectors are orthogonal and non-degenerate).
func (m *Matrix) PreservesRightAngles() bool {
	mask := m.Type()
	if mask <= TypeTranslate {
		// Identity and/or translate. Scale-only matrices (TypeScale is 0x02, so they exceed TypeTranslate)
		// deliberately fall through to the degeneracy test below: ScaleMatrix(0, 5) must report false. Do not widen
		// this guard to TypeScale|TypeTranslate.
		return true
	}
	if mask&TypePerspective != 0 {
		return false
	}

	mx := m.mat[MScaleX]
	my := m.mat[MScaleY]
	sx := m.mat[MSkewX]
	sy := m.mat[MSkewY]

	if isDegenerate2x2(mx, sx, sy, my) {
		return false
	}

	// The upper 2x2 is scale + rotation/reflection if the basis vectors are orthogonal. Pinned unfused like
	// isDegenerate2x2: the dot product is compared against a near-zero tolerance.
	return ScalarNearlyZeroTolerance(float32(mx*sx)+float32(sy*my), ScalarNearlyZeroTol*ScalarNearlyZeroTol)
}

// CheapEqual reports element-wise equality of the nine values (Go's struct == also compares the lazily-computed
// type-mask cache, which differs between otherwise identical matrices, so it must not be used for matrix equality).
func (m *Matrix) CheapEqual(other *Matrix) bool {
	return m.mat == other.mat
}

// IsSimilarity reports, using the default tolerance, whether the matrix is a similarity transform (uniform scale with
// rotation/reflection, plus translation).
func (m *Matrix) IsSimilarity() bool {
	mask := m.Type()
	if mask <= TypeTranslate {
		// Identity or translate matrix.
		return true
	}
	if mask&TypePerspective != 0 {
		return false
	}

	mx := m.mat[MScaleX]
	my := m.mat[MScaleY]
	// If no skew, can just compare scale factors.
	if mask&TypeAffine == 0 {
		return !ScalarNearlyZero(mx) && ScalarNearlyEqual(ScalarAbs(mx), ScalarAbs(my))
	}
	sx := m.mat[MSkewX]
	sy := m.mat[MSkewY]

	if isDegenerate2x2(mx, sx, sy, my) {
		return false
	}

	// The upper 2x2 is rotation/reflection + uniform scale if the basis vectors are 90 degree rotations of each other.
	return (ScalarNearlyEqual(mx, my) && ScalarNearlyEqual(sx, -sy)) ||
		(ScalarNearlyEqual(mx, -my) && ScalarNearlyEqual(sx, sy))
}

// computeTypeMask derives the type mask from the matrix elements, using two's-complement integer comparisons that make
// -0.0 compare equal to 0.
func (m *Matrix) computeTypeMask() uint8 {
	var mask uint8
	if m.mat[MPersp0] != 0 || m.mat[MPersp1] != 0 || m.mat[MPersp2] != 1 {
		return maskORable
	}
	if m.mat[MTransX] != 0 || m.mat[MTransY] != 0 {
		mask |= TypeTranslate
	}
	m00 := scalarAs2sCompliment(m.mat[MScaleX])
	m01 := scalarAs2sCompliment(m.mat[MSkewX])
	m10 := scalarAs2sCompliment(m.mat[MSkewY])
	m11 := scalarAs2sCompliment(m.mat[MScaleY])
	if m01 != 0 || m10 != 0 {
		// Skew components may be scale-inducing; testing for pure rotation is expensive, so both bits are
		// conservatively set, which also keeps masks equal between a matrix and its inverse.
		mask |= TypeAffine | TypeScale
		// rect-stays-rect in the affine case: primary diagonal all zeros, secondary diagonal all non-zero.
		if m00 == 0 && m11 == 0 && m01 != 0 && m10 != 0 {
			mask |= maskRectStaysRect
		}
	} else {
		if m00^scalar1Int != 0 || m11^scalar1Int != 0 {
			mask |= TypeScale
		}
		// Secondary diagonal is known zero; rect-stays-rect iff the primary diagonal is all non-zero.
		if m00 != 0 && m11 != 0 {
			mask |= maskRectStaysRect
		}
	}
	return mask
}

func (m *Matrix) updateTranslateMask() {
	if m.mat[MTransX] != 0 || m.mat[MTransY] != 0 {
		m.typeMask |= TypeTranslate
	} else {
		m.typeMask &^= TypeTranslate
	}
}

///////////////////////////////////////////////////////////////////////////////
// Setters

// SetIdentity resets the matrix to the identity matrix.
func (m *Matrix) SetIdentity() {
	m.mat = [9]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	m.setTypeMask(TypeIdentity | maskRectStaysRect)
}

// SetTranslate resets the matrix to a pure translation by (dx, dy).
func (m *Matrix) SetTranslate(dx, dy float32) {
	m.mat = [9]float32{1, 0, dx, 0, 1, dy, 0, 0, 1}
	if dx != 0 || dy != 0 {
		m.setTypeMask(TypeTranslate | maskRectStaysRect)
	} else {
		m.setTypeMask(TypeIdentity | maskRectStaysRect)
	}
}

// SetScale resets the matrix to a pure scale by (sx, sy).
func (m *Matrix) SetScale(sx, sy float32) {
	m.mat = [9]float32{sx, 0, 0, 0, sy, 0, 0, 0, 1}
	var mask uint8
	if sx == 1 && sy == 1 {
		mask = TypeIdentity
	} else {
		mask = TypeScale
	}
	if sx != 0 && sy != 0 {
		mask |= maskRectStaysRect
	}
	m.setTypeMask(mask)
}

// SetScalePivot resets the matrix to a scale by (sx, sy) about the pivot point (px, py).
func (m *Matrix) SetScalePivot(sx, sy, px, py float32) {
	if sx == 1 && sy == 1 {
		m.SetIdentity()
	} else {
		m.SetScaleTranslate(sx, sy, px-sx*px, py-sy*py)
	}
}

// SetScaleTranslate resets the matrix to a scale by (sx, sy) followed by a translate by (tx, ty).
func (m *Matrix) SetScaleTranslate(sx, sy, tx, ty float32) {
	m.mat = [9]float32{sx, 0, tx, 0, sy, ty, 0, 0, 1}
	var mask uint8
	if sx != 1 || sy != 1 {
		mask |= TypeScale
	}
	if tx != 0 || ty != 0 {
		mask |= TypeTranslate
	}
	if sx != 0 && sy != 0 {
		mask |= maskRectStaysRect
	}
	m.setTypeMask(mask)
}

// SetSinCos resets the matrix to a rotation given by its sine and cosine.
func (m *Matrix) SetSinCos(sinV, cosV float32) {
	m.mat = [9]float32{cosV, -sinV, 0, sinV, cosV, 0, 0, 0, 1}
	m.setTypeMask(maskUnknown | maskOnlyPerspectiveValid)
}

// SetSinCosPivot resets the matrix to a rotation given by its sine and cosine, about the pivot point (px, py).
func (m *Matrix) SetSinCosPivot(sinV, cosV, px, py float32) {
	oneMinusCosV := 1 - cosV
	m.mat = [9]float32{
		cosV, -sinV, sinV*py + oneMinusCosV*px,
		sinV, cosV, -sinV*px + oneMinusCosV*py,
		0, 0, 1,
	}
	m.setTypeMask(maskUnknown | maskOnlyPerspectiveValid)
}

// SetRotate resets the matrix to a rotation by degrees, snapping near-zero sin/cos to exact zero so right-angle
// rotations are exact.
func (m *Matrix) SetRotate(degrees float32) {
	rad := DegreesToRadians(degrees)
	m.SetSinCos(ScalarSinSnapToZero(rad), ScalarCosSnapToZero(rad))
}

// SetRotatePivot resets the matrix to a rotation by degrees about the pivot point (px, py).
func (m *Matrix) SetRotatePivot(degrees, px, py float32) {
	rad := DegreesToRadians(degrees)
	m.SetSinCosPivot(ScalarSinSnapToZero(rad), ScalarCosSnapToZero(rad), px, py)
}

// SetSkew resets the matrix to a skew by (kx, ky).
func (m *Matrix) SetSkew(kx, ky float32) {
	m.mat = [9]float32{1, kx, 0, ky, 1, 0, 0, 0, 1}
	m.setTypeMask(maskUnknown | maskOnlyPerspectiveValid)
}

// SetSkewPivot resets the matrix to a skew by (kx, ky) about the pivot point (px, py).
func (m *Matrix) SetSkewPivot(kx, ky, px, py float32) {
	m.mat = [9]float32{1, kx, -kx * py, ky, 1, -ky * px, 0, 0, 1}
	m.setTypeMask(maskUnknown | maskOnlyPerspectiveValid)
}

///////////////////////////////////////////////////////////////////////////////
// Concatenation

// muladdmul computes a*b + c*d in double precision, truncated to float once.
func muladdmul(a, b, c, d float32) float32 {
	return DoubleToScalar(float64(a)*float64(b) + float64(c)*float64(d))
}

func rowcol3(row *[9]float32, rowOffset int, col *[9]float32, colOffset int) float32 {
	return row[rowOffset]*col[colOffset] + row[rowOffset+1]*col[colOffset+3] + row[rowOffset+2]*col[colOffset+6]
}

func onlyScaleAndTranslate(mask uint8) bool {
	return mask&(TypeAffine|TypePerspective) == 0
}

// SetConcat sets m = a * b.
func (m *Matrix) SetConcat(a, b *Matrix) {
	aType := a.Type()
	bType := b.Type()
	switch {
	case a.isTriviallyIdentity():
		*m = *b
	case b.isTriviallyIdentity():
		*m = *a
	case onlyScaleAndTranslate(aType | bType):
		m.SetScaleTranslate(a.mat[MScaleX]*b.mat[MScaleX],
			a.mat[MScaleY]*b.mat[MScaleY],
			a.mat[MScaleX]*b.mat[MTransX]+a.mat[MTransX],
			a.mat[MScaleY]*b.mat[MTransY]+a.mat[MTransY])
	default:
		var tmp Matrix
		if (aType|bType)&TypePerspective != 0 {
			tmp.mat[MScaleX] = rowcol3(&a.mat, 0, &b.mat, 0)
			tmp.mat[MSkewX] = rowcol3(&a.mat, 0, &b.mat, 1)
			tmp.mat[MTransX] = rowcol3(&a.mat, 0, &b.mat, 2)
			tmp.mat[MSkewY] = rowcol3(&a.mat, 3, &b.mat, 0)
			tmp.mat[MScaleY] = rowcol3(&a.mat, 3, &b.mat, 1)
			tmp.mat[MTransY] = rowcol3(&a.mat, 3, &b.mat, 2)
			tmp.mat[MPersp0] = rowcol3(&a.mat, 6, &b.mat, 0)
			tmp.mat[MPersp1] = rowcol3(&a.mat, 6, &b.mat, 1)
			tmp.mat[MPersp2] = rowcol3(&a.mat, 6, &b.mat, 2)
			tmp.setTypeMask(maskUnknown)
		} else {
			tmp.mat[MScaleX] = muladdmul(a.mat[MScaleX], b.mat[MScaleX], a.mat[MSkewX], b.mat[MSkewY])
			tmp.mat[MSkewX] = muladdmul(a.mat[MScaleX], b.mat[MSkewX], a.mat[MSkewX], b.mat[MScaleY])
			tmp.mat[MTransX] = muladdmul(a.mat[MScaleX], b.mat[MTransX], a.mat[MSkewX], b.mat[MTransY]) + a.mat[MTransX]
			tmp.mat[MSkewY] = muladdmul(a.mat[MSkewY], b.mat[MScaleX], a.mat[MScaleY], b.mat[MSkewY])
			tmp.mat[MScaleY] = muladdmul(a.mat[MSkewY], b.mat[MSkewX], a.mat[MScaleY], b.mat[MScaleY])
			tmp.mat[MTransY] = muladdmul(a.mat[MSkewY], b.mat[MTransX], a.mat[MScaleY], b.mat[MTransY]) + a.mat[MTransY]
			tmp.mat[MPersp0] = 0
			tmp.mat[MPersp1] = 0
			tmp.mat[MPersp2] = 1
			tmp.setTypeMask(maskUnknown | maskOnlyPerspectiveValid)
		}
		*m = tmp
	}
}

// PreConcat sets m = m * other.
func (m *Matrix) PreConcat(other *Matrix) {
	if !other.IsIdentity() {
		m.SetConcat(m, other)
	}
}

// PostConcat sets m = other * m.
func (m *Matrix) PostConcat(other *Matrix) {
	if !other.IsIdentity() {
		m.SetConcat(other, m)
	}
}

// PreTranslate applies a translate by (dx, dy) before the matrix's existing transform.
func (m *Matrix) PreTranslate(dx, dy float32) {
	mask := m.Type()
	switch {
	case mask <= TypeTranslate:
		m.mat[MTransX] += dx
		m.mat[MTransY] += dy
	case mask&TypePerspective != 0:
		var t Matrix
		t.SetTranslate(dx, dy)
		m.PreConcat(&t)
		return
	default:
		m.mat[MTransX] += m.mat[MScaleX]*dx + m.mat[MSkewX]*dy
		m.mat[MTransY] += m.mat[MSkewY]*dx + m.mat[MScaleY]*dy
	}
	m.updateTranslateMask()
}

// PostTranslate applies a translate by (dx, dy) after the matrix's existing transform.
func (m *Matrix) PostTranslate(dx, dy float32) {
	if m.HasPerspective() {
		var t Matrix
		t.SetTranslate(dx, dy)
		m.PostConcat(&t)
	} else {
		m.mat[MTransX] += dx
		m.mat[MTransY] += dy
		m.updateTranslateMask()
	}
}

// PreScale applies a scale by (sx, sy) before the matrix's existing transform, using a blind-multiply fast path with
// the necessary mask fixups.
func (m *Matrix) PreScale(sx, sy float32) {
	if sx == 1 && sy == 1 {
		return
	}
	m.mat[MScaleX] *= sx
	m.mat[MSkewY] *= sx
	m.mat[MPersp0] *= sx
	m.mat[MSkewX] *= sy
	m.mat[MScaleY] *= sy
	m.mat[MPersp1] *= sy
	if m.mat[MScaleX] == 1 && m.mat[MScaleY] == 1 && m.storedMask()&(TypePerspective|TypeAffine) == 0 {
		m.clearTypeMask(TypeScale)
	} else {
		m.orTypeMask(TypeScale)
		// A zero scale factor collapses the rect-stays-rect property.
		if sx == 0 || sy == 0 {
			m.clearTypeMask(maskRectStaysRect)
		}
	}
}

// DecomposeScale splits the matrix into an anisotropic scale (the column lengths) and the remaining
// rotation/skew/translation. Returns false for a perspective, non-finite, or degenerate (near-zero column) matrix.
// Either out-param may be nil.
func (m *Matrix) DecomposeScale(scale *Size, remaining *Matrix) bool {
	if m.HasPerspective() {
		return false
	}
	sx := Point{X: m.mat[MScaleX], Y: m.mat[MSkewY]}.Length()
	sy := Point{X: m.mat[MSkewX], Y: m.mat[MScaleY]}.Length()
	if !IsFinite(sx, sy) || ScalarNearlyZero(sx) || ScalarNearlyZero(sy) {
		return false
	}
	if scale != nil {
		*scale = Size{Width: sx, Height: sy}
	}
	if remaining != nil {
		*remaining = *m
		remaining.PreScale(1/sx, 1/sy)
	}
	return true
}

// PreScalePivot applies a scale by (sx, sy) about the pivot point (px, py) before the matrix's existing transform.
func (m *Matrix) PreScalePivot(sx, sy, px, py float32) {
	if sx == 1 && sy == 1 {
		return
	}
	var s Matrix
	s.SetScalePivot(sx, sy, px, py)
	m.PreConcat(&s)
}

// PostScale applies a scale by (sx, sy) after the matrix's existing transform.
func (m *Matrix) PostScale(sx, sy float32) {
	if sx == 1 && sy == 1 {
		return
	}
	var s Matrix
	s.SetScale(sx, sy)
	m.PostConcat(&s)
}

// PreRotate applies a rotation by degrees before the matrix's existing transform.
func (m *Matrix) PreRotate(degrees float32) {
	var r Matrix
	r.SetRotate(degrees)
	m.PreConcat(&r)
}

// PostRotate applies a rotation by degrees after the matrix's existing transform.
func (m *Matrix) PostRotate(degrees float32) {
	var r Matrix
	r.SetRotate(degrees)
	m.PostConcat(&r)
}

///////////////////////////////////////////////////////////////////////////////
// Inversion

func dcross(a, b, c, d float64) float64 {
	return a*b - c*d
}

func scrossDscale(a, b, c, d float32, scale float64) float32 {
	return DoubleToScalar(float64(a*b-c*d) * scale)
}

func dcrossDscale(a, b, c, d, scale float64) float32 {
	return DoubleToScalar(dcross(a, b, c, d) * scale)
}

func (m *Matrix) determinant(isPerspective bool) float64 {
	if isPerspective {
		return float64(m.mat[MScaleX])*dcross(float64(m.mat[MScaleY]), float64(m.mat[MPersp2]),
			float64(m.mat[MTransY]), float64(m.mat[MPersp1])) +
			float64(m.mat[MSkewX])*dcross(float64(m.mat[MTransY]), float64(m.mat[MPersp0]),
				float64(m.mat[MSkewY]), float64(m.mat[MPersp2])) +
			float64(m.mat[MTransX])*dcross(float64(m.mat[MSkewY]), float64(m.mat[MPersp1]),
				float64(m.mat[MScaleY]), float64(m.mat[MPersp0]))
	}
	return dcross(float64(m.mat[MScaleX]), float64(m.mat[MScaleY]),
		float64(m.mat[MSkewX]), float64(m.mat[MSkewY]))
}

func (m *Matrix) invDeterminant(isPerspective bool) float64 {
	det := m.determinant(isPerspective)
	// The determinant scales with the cube of the elements, so compare against the cube of the nearly-zero constant.
	if ScalarNearlyZeroTolerance(DoubleToScalar(det), ScalarNearlyZeroTol*ScalarNearlyZeroTol*ScalarNearlyZeroTol) {
		return 0
	}
	return 1.0 / det
}

// computeInv computes the matrix inverse given the source elements and precomputed inverse determinant.
func computeInv(dst, src *[9]float32, invDet float64, isPersp bool) {
	if isPersp {
		dst[MScaleX] = scrossDscale(src[MScaleY], src[MPersp2], src[MTransY], src[MPersp1], invDet)
		dst[MSkewX] = scrossDscale(src[MTransX], src[MPersp1], src[MSkewX], src[MPersp2], invDet)
		dst[MTransX] = scrossDscale(src[MSkewX], src[MTransY], src[MTransX], src[MScaleY], invDet)
		dst[MSkewY] = scrossDscale(src[MTransY], src[MPersp0], src[MSkewY], src[MPersp2], invDet)
		dst[MScaleY] = scrossDscale(src[MScaleX], src[MPersp2], src[MTransX], src[MPersp0], invDet)
		dst[MTransY] = scrossDscale(src[MTransX], src[MSkewY], src[MScaleX], src[MTransY], invDet)
		dst[MPersp0] = scrossDscale(src[MSkewY], src[MPersp1], src[MScaleY], src[MPersp0], invDet)
		dst[MPersp1] = scrossDscale(src[MSkewX], src[MPersp0], src[MScaleX], src[MPersp1], invDet)
		dst[MPersp2] = scrossDscale(src[MScaleX], src[MScaleY], src[MSkewX], src[MSkewY], invDet)
	} else {
		dst[MScaleX] = DoubleToScalar(float64(src[MScaleY]) * invDet)
		dst[MSkewX] = DoubleToScalar(float64(-src[MSkewX]) * invDet)
		dst[MTransX] = dcrossDscale(float64(src[MSkewX]), float64(src[MTransY]),
			float64(src[MScaleY]), float64(src[MTransX]), invDet)
		dst[MSkewY] = DoubleToScalar(float64(-src[MSkewY]) * invDet)
		dst[MScaleY] = DoubleToScalar(float64(src[MScaleX]) * invDet)
		dst[MTransY] = dcrossDscale(float64(src[MSkewY]), float64(src[MTransX]),
			float64(src[MScaleX]), float64(src[MTransY]), invDet)
		dst[MPersp0] = 0
		dst[MPersp1] = 0
		dst[MPersp2] = 1
	}
}

// Invert returns the inverse and true, or the zero Matrix and false when the matrix is not invertible (degenerate, or
// overflow during inversion).
func (m *Matrix) Invert() (Matrix, bool) {
	mask := m.Type()
	if mask == TypeIdentity {
		return *m, true
	}
	if mask&^(TypeScale|TypeTranslate) == 0 {
		if mask&TypeScale != 0 {
			invSX := IEEEFloatDivide(1, m.mat[MScaleX])
			invSY := IEEEFloatDivide(1, m.mat[MScaleY])
			// Denormalized (non-zero) scale factors overflow when inverted.
			if !IsFinite(invSX, invSY) {
				return Matrix{}, false
			}
			invTX := -m.mat[MTransX] * invSX
			invTY := -m.mat[MTransY] * invSY
			if !IsFinite(invTX, invTY) {
				return Matrix{}, false
			}
			inv := Matrix{mat: [9]float32{invSX, 0, invTX, 0, invSY, invTY, 0, 0, 1}}
			inv.setTypeMask(mask | maskRectStaysRect)
			return inv, true
		}
		if !IsFinite(m.mat[MTransX], m.mat[MTransY]) {
			return Matrix{}, false
		}
		return TranslateMatrix(-m.mat[MTransX], -m.mat[MTransY]), true
	}
	isPersp := mask&TypePerspective != 0
	invDet := m.invDeterminant(isPersp)
	if invDet == 0 {
		return Matrix{}, false
	}
	var inv Matrix
	computeInv(&inv.mat, &m.mat, invDet, isPersp)
	if !inv.IsFinite() {
		return Matrix{}, false
	}
	inv.setTypeMask(m.typeMask)
	return inv, true
}

///////////////////////////////////////////////////////////////////////////////
// Mapping

// MapXY maps a single point through the matrix, dispatching on the type mask to specialized per-class code paths so the
// float operations are consistent for a given class of matrix.
func (m *Matrix) MapXY(x, y float32) Point {
	switch m.Type() & maskAllTypes {
	case TypeIdentity:
		return Point{X: x, Y: y}
	case TypeTranslate:
		return Point{X: x + m.mat[MTransX], Y: y + m.mat[MTransY]}
	case TypeScale, TypeScale | TypeTranslate:
		return Point{X: x*m.mat[MScaleX] + m.mat[MTransX], Y: y*m.mat[MScaleY] + m.mat[MTransY]}
	default:
		if m.Type()&TypePerspective != 0 {
			return m.mapPointPerspective(x, y)
		}
		// Affine: (src*scale + swz*skew) + trans, evaluated in this left-associative operation order for float
		// consistency.
		return Point{
			X: (x*m.mat[MScaleX] + y*m.mat[MSkewX]) + m.mat[MTransX],
			Y: (y*m.mat[MScaleY] + x*m.mat[MSkewY]) + m.mat[MTransY],
		}
	}
}

// mapPointPerspective maps a single point through a perspective matrix.
func (m *Matrix) mapPointPerspective(x, y float32) Point {
	px := x*m.mat[MScaleX] + y*m.mat[MSkewX] + m.mat[MTransX]
	py := x*m.mat[MSkewY] + y*m.mat[MScaleY] + m.mat[MTransY]
	z := x*m.mat[MPersp0] + y*m.mat[MPersp1] + m.mat[MPersp2]
	if z != 0 {
		z = 1 / z
	}
	return Point{X: px * z, Y: py * z}
}

// MapPoint maps a single point.
func (m *Matrix) MapPoint(p Point) Point {
	return m.MapXY(p.X, p.Y)
}

// MapPoints maps src through the matrix into dst. dst and src should be the same length; if they differ, only
// min(len(dst), len(src)) points are mapped.
func (m *Matrix) MapPoints(dst, src []Point) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n == 0 {
		return
	}
	switch m.Type() & maskAllTypes {
	case TypeIdentity:
		copy(dst[:n], src[:n])
	case TypeTranslate:
		tx := m.mat[MTransX]
		ty := m.mat[MTransY]
		for i := 0; i < n; i++ {
			dst[i] = Point{X: src[i].X + tx, Y: src[i].Y + ty}
		}
	case TypeScale, TypeScale | TypeTranslate:
		sx := m.mat[MScaleX]
		sy := m.mat[MScaleY]
		tx := m.mat[MTransX]
		ty := m.mat[MTransY]
		for i := 0; i < n; i++ {
			dst[i] = Point{X: src[i].X*sx + tx, Y: src[i].Y*sy + ty}
		}
	default:
		if m.Type()&TypePerspective != 0 {
			for i := 0; i < n; i++ {
				dst[i] = m.mapPointPerspective(src[i].X, src[i].Y)
			}
			return
		}
		sx := m.mat[MScaleX]
		sy := m.mat[MScaleY]
		kx := m.mat[MSkewX]
		ky := m.mat[MSkewY]
		tx := m.mat[MTransX]
		ty := m.mat[MTransY]
		for i := 0; i < n; i++ {
			x := src[i].X
			y := src[i].Y
			dst[i] = Point{X: (x*sx + y*kx) + tx, Y: (y*sy + x*ky) + ty}
		}
	}
}

// MapPointsInPlace maps pts through m in place.
func (m *Matrix) MapPointsInPlace(pts []Point) {
	m.MapPoints(pts, pts)
}

// MapVectors maps src as vectors (ignoring translation). For perspective matrices, each vector maps as mapPoint(v) -
// mapPoint(0,0).
func (m *Matrix) MapVectors(dst, src []Point) {
	if m.HasPerspective() {
		origin := m.mapPointPerspective(0, 0)
		n := len(src)
		if len(dst) < n {
			n = len(dst)
		}
		for i := n - 1; i >= 0; i-- {
			p := m.mapPointPerspective(src[i].X, src[i].Y)
			dst[i] = Point{X: p.X - origin.X, Y: p.Y - origin.Y}
		}
		return
	}
	tmp := *m
	tmp.mat[MTransX] = 0
	tmp.mat[MTransY] = 0
	tmp.clearTypeMask(TypeTranslate)
	tmp.MapPoints(dst, src)
}

// MapVector maps a single vector through m, ignoring translation.
func (m *Matrix) MapVector(v Point) Point {
	var dst [1]Point
	src := [1]Point{v}
	m.MapVectors(dst[:], src[:])
	return dst[0]
}

// MapRadius maps a circle of the given radius through the matrix and returns the geometric mean of the mapped axes'
// lengths.
func (m *Matrix) MapRadius(radius float32) float32 {
	src := [2]Point{{X: radius, Y: 0}, {X: 0, Y: radius}}
	var vec [2]Point
	m.MapVectors(vec[:], src[:])
	return ScalarSqrt(vec[0].Length() * vec[1].Length())
}

// AffineIdentity returns the six-element affine identity, in [scaleX, skewY, skewX, scaleY, transX, transY] order (this
// is the PDF "a b c d e f cm" order).
func AffineIdentity() [6]float32 { return [6]float32{1, 0, 0, 1, 0, 0} }

// AsAffine returns the six affine coefficients (scaleX, skewY, skewX, scaleY, transX, transY) and true when the matrix
// has no perspective, else the zero array and false.
func (m *Matrix) AsAffine() ([6]float32, bool) {
	if m.HasPerspective() {
		return [6]float32{}, false
	}
	return [6]float32{
		m.mat[MScaleX], m.mat[MSkewY], m.mat[MSkewX], m.mat[MScaleY], m.mat[MTransX], m.mat[MTransY],
	}, true
}

// MapRect returns the bounds of src mapped through m. The boolean result reports whether the mapped rect corresponds
// exactly to the image of src, i.e. whether the matrix is rect-stays-rect (identity, translate, scale, and the 90/270
// degree rotations and axis flips built from them). It is false for perspective matrices and for affine matrices that
// are not rect-stays-rect, where the returned rect is only the bounding box of the mapped quad.
func (m *Matrix) MapRect(src Rect) (Rect, bool) {
	if m.Type() <= TypeTranslate {
		tx := m.mat[MTransX]
		ty := m.mat[MTransY]
		r := Rect{Left: src.Left + tx, Top: src.Top + ty, Right: src.Right + tx, Bottom: src.Bottom + ty}
		return r.Sorted(), true
	}
	if m.IsScaleTranslate() {
		sx := m.mat[MScaleX]
		sy := m.mat[MScaleY]
		tx := m.mat[MTransX]
		ty := m.mat[MTransY]
		r := Rect{
			Left:   src.Left*sx + tx,
			Top:    src.Top*sy + ty,
			Right:  src.Right*sx + tx,
			Bottom: src.Bottom*sy + ty,
		}
		return r.Sorted(), true
	}
	quad := src.ToQuad()
	m.MapPointsInPlace(quad[:])
	var dst Rect
	if m.HasPerspective() {
		// A full perspective rect mapping needs to clip the path against the w = 1/16384 half-plane, which this corner
		// mapping cannot reproduce: the clip machinery lives above geom in the package graph. Use path.MapRect (which
		// implements the full perspective-aware mapping and delegates here for non-perspective matrices) whenever the
		// matrix may have perspective. Results agree for fully-visible rects and differ when src has a corner at or
		// below the w plane.
		dst.SetBounds(quad[:])
		return dst, false
	}
	// The affine general path uses setBoundsNoCheck: non-finite corners poison the result to all-NaN.
	dst.SetBoundsNoCheck(quad[:])
	return dst, m.RectStaysRect()
}

// MaxScale returns the maximum singular value of the upper 2x2, or -1 for perspective matrices and for matrices whose
// 2x2 entries make the singular value non-finite (the caller decides what scale to substitute in those cases).
func (m *Matrix) MaxScale() float32 {
	typeMask := m.Type()
	if typeMask&TypePerspective != 0 {
		return -1
	}
	if typeMask == TypeIdentity {
		return 1
	}
	if typeMask&TypeAffine == 0 {
		result := max(ScalarAbs(m.mat[MScaleX]), ScalarAbs(m.mat[MScaleY]))
		if !IsFinite(result) {
			return -1
		}
		return result
	}
	// Ignore the translation part of the matrix, just look at the 2x2 portion: compute the singular values and take the
	// largest. [a b; b c] = A^T*A. The dot products and the discriminant are pinned unfused (explicit float32
	// conversions round each product) like Point.Dot: bSqd is compared against a near-zero tolerance to pick the
	// orthogonal branch, and the result feeds stroke widths and size bucketing, so fusion would diverge per platform.
	a := float32(m.mat[MScaleX]*m.mat[MScaleX]) + float32(m.mat[MSkewY]*m.mat[MSkewY])
	b := float32(m.mat[MScaleX]*m.mat[MSkewX]) + float32(m.mat[MScaleY]*m.mat[MSkewY])
	c := float32(m.mat[MSkewX]*m.mat[MSkewX]) + float32(m.mat[MScaleY]*m.mat[MScaleY])
	// The eigenvalues of A^T*A are the squared singular values of A; solve the characteristic equation l^2 - (a+c)l +
	// (ac - b^2) via the quadratic formula.
	bSqd := b * b
	var result float32
	if bSqd <= ScalarNearlyZeroTol*ScalarNearlyZeroTol {
		// If the upper-left 2x2 is orthogonal, save some math.
		result = max(a, c)
	} else {
		aminusc := a - c
		// Both halvings are wrapped in float32 conversions to keep this branch's pinning uniform with the dot products
		// above: each is a multiply feeding the final add, which arm64 would otherwise contract into a single FMADDS.
		// The contraction is provably unobservable here — halving is exact unless the result underflows to subnormal,
		// and the bSqd guard keeps both operands far above that (bSqd > (1/4096)^2 gives sqrt(disc) >= 4.8e-4, and
		// |b| <= (a+c)/2 by Cauchy-Schwarz gives a+c > 4.9e-4) — so this is consistency, not a bug fix. Pinned anyway
		// so the branch cannot start diverging if the guard or the discriminant is ever reworked.
		apluscdiv2 := float32(0.5 * (a + c))
		x := float32(0.5 * ScalarSqrt(float32(aminusc*aminusc)+float32(4*bSqd)))
		result = apluscdiv2 + x
	}
	if !IsFinite(result) {
		return -1
	}
	// Float inaccuracy in the quadratic can push a nearly-zero eigenvalue negative; cap it rather than taking the square
	// root of a negative and returning NaN.
	if result < 0 {
		result = 0
	}
	return ScalarSqrt(result)
}

// ComputeResScaleForStroking returns the larger of the two column lengths of the upper 2x2, or 1 when degenerate or
// non-finite. Perspective is not handled specially.
func (m *Matrix) ComputeResScaleForStroking() float32 {
	sx := Point{X: m.mat[MScaleX], Y: m.mat[MSkewY]}.Length()
	sy := Point{X: m.mat[MSkewX], Y: m.mat[MScaleY]}.Length()
	if IsFinite(sx, sy) {
		scale := max(sx, sy)
		if scale > 0 {
			return scale
		}
	}
	return 1
}
