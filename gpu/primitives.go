// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// PrimitiveType and ClampType, needed by the shader pipeline: the program info carries the primitive type into the
// program key, and ClampType gates blend analysis for the "plus" mode.

package gpu

// PrimitiveType identifies the GPU primitive topology a draw uses. The numbering matches Skia's GrPrimitiveType so key
// values stay comparable.
type PrimitiveType uint8

// PrimitiveType values.
const (
	PrimitiveTypeTriangles PrimitiveType = iota
	PrimitiveTypeTriangleStrip
	PrimitiveTypePoints
	PrimitiveTypeLines     // 1 pix wide only
	PrimitiveTypeLineStrip // 1 pix wide only
)

// ClampType describes how the pixel values of the render target are clamped: ClampTypeAuto for normalized formats,
// which clamp to [0,1] automatically; ClampTypeManual for float formats that need shader clamping; ClampTypeNone for
// unclamped float formats. Every draw passes ClampTypeAuto, but the blend analysis consults the type, so it is kept.
type ClampType int32

// ClampType values.
const (
	ClampTypeAuto ClampType = iota
	ClampTypeManual
	ClampTypeNone
)
