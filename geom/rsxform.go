// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// RSXform is the compressed rotation+scale matrix Canvas.DrawAtlas positions its sprites with.

package geom

// RSXform is a compressed form of a rotation+scale matrix,
//
//	[ SCos    -SSin    Tx ]
//	[ SSin     SCos    Ty ]
//	[    0        0     1 ]
type RSXform struct {
	SCos float32
	SSin float32
	Tx   float32
	Ty   float32
}

// MakeRSXform builds an RSXform from its four compressed components.
func MakeRSXform(scos, ssin, tx, ty float32) RSXform {
	return RSXform{SCos: scos, SSin: ssin, Tx: tx, Ty: ty}
}

// RSXformFromRadians builds a new xform from the scale, rotation (in radians), final tx,ty location, and anchor point
// ax,ay within the src quad. Note: the anchor point is not normalized (e.g. 0...1) but is in pixels of the src image.
func RSXformFromRadians(scale, radians, tx, ty, ax, ay float32) RSXform {
	s := ScalarSin(radians) * scale
	c := ScalarCos(radians) * scale
	return MakeRSXform(c, s, tx+-c*ax+s*ay, ty+-s*ax-c*ay)
}

// RectStaysRect reports whether the xform maps an axis-aligned rect to another axis-aligned rect.
func (x RSXform) RectStaysRect() bool {
	return x.SCos == 0 || x.SSin == 0
}

// ToQuad returns the four corners of the transformed [0,0,width,height] rect, in the order (0,0), (w,0), (w,h), (0,h).
func (x RSXform) ToQuad(width, height float32, quad *[4]Point) {
	m00 := x.SCos
	m01 := -x.SSin
	m02 := x.Tx
	m10 := -m01
	m11 := m00
	m12 := x.Ty

	quad[0] = Point{X: m02, Y: m12}
	quad[1] = Point{X: m00*width + m02, Y: m10*width + m12}
	quad[2] = Point{X: m00*width + m01*height + m02, Y: m10*width + m11*height + m12}
	quad[3] = Point{X: m01*height + m02, Y: m11*height + m12}
}

// ToTriStrip returns the four transformed corners in triangle-strip order (0,0), (0,h), (w,0), (w,h).
func (x RSXform) ToTriStrip(width, height float32, strip *[4]Point) {
	m00 := x.SCos
	m01 := -x.SSin
	m02 := x.Tx
	m10 := -m01
	m11 := m00
	m12 := x.Ty

	strip[0] = Point{X: m02, Y: m12}
	strip[1] = Point{X: m01*height + m02, Y: m11*height + m12}
	strip[2] = Point{X: m00*width + m02, Y: m10*width + m12}
	strip[3] = Point{X: m00*width + m01*height + m02, Y: m10*width + m11*height + m12}
}

// Matrix returns the 3x3 matrix this xform compresses.
func (x RSXform) Matrix() Matrix {
	var m Matrix
	m.SetAll(x.SCos, -x.SSin, x.Tx, x.SSin, x.SCos, x.Ty, 0, 0, 1)
	return m
}
