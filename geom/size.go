// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package geom

// ISize is a pair of int32 width/height dimensions.
type ISize struct {
	Width  int32
	Height int32
}

// IsEmpty reports true if either dimension is not positive.
func (s ISize) IsEmpty() bool {
	return s.Width <= 0 || s.Height <= 0
}

// IsZero reports whether both dimensions are zero.
func (s ISize) IsZero() bool {
	return s.Width == 0 && s.Height == 0
}

// Area returns Width * Height (only meaningful for non-empty sizes).
func (s ISize) Area() int64 {
	return int64(s.Width) * int64(s.Height)
}

// Size is a pair of float32 width/height dimensions.
type Size struct {
	Width  float32
	Height float32
}

// IsEmpty reports true if either dimension is not positive (NaN dimensions count as empty).
func (s Size) IsEmpty() bool {
	return !(s.Width > 0 && s.Height > 0)
}

// IsZero reports whether both dimensions are zero.
func (s Size) IsZero() bool {
	return s.Width == 0 && s.Height == 0
}

// ToRound rounds both dimensions to the nearest integer.
func (s Size) ToRound() ISize {
	return ISize{Width: RoundToInt(s.Width), Height: RoundToInt(s.Height)}
}

// ToCeil rounds both dimensions up to the nearest integer.
func (s Size) ToCeil() ISize {
	return ISize{Width: CeilToInt(s.Width), Height: CeilToInt(s.Height)}
}

// ToFloor rounds both dimensions down to the nearest integer.
func (s Size) ToFloor() ISize {
	return ISize{Width: FloorToInt(s.Width), Height: FloorToInt(s.Height)}
}
