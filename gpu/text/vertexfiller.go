// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// VertexFiller writes the per-glyph quad vertex data (position + atlas UVs, optionally packed color) that the text
// rendering shaders consume. It assumes that all points, glyph atlas entries, and bounds are created with respect to
// the creation matrix, so mapping any point, mask or bounds through the creation matrix results in the proper device
// position. To draw with an arbitrary position matrix, a viewDifference = [positionMatrix] * [creationMatrix]^-1 maps
// everything into place. There is no serialization support here — that only serves a remote glyph cache and text
// "slugs", neither of which is reachable in this codebase.

package text

import (
	"encoding/binary"
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/path"
)

// FillerType selects whether a VertexFiller can use the direct (untransformed) fast path.
type FillerType int32

// FillerType values.
const (
	IsDirect FillerType = iota
	IsTransformed
)

// Vertex strides: device position floats, then a packed color for the mask formats, then uint16 atlas coords.
const (
	mask2DVertexStride = 8 + 4 + 4  // {float2 pos; packed color; ushort2 uv}
	argb2DVertexStride = 8 + 4      // {float2 pos; ushort2 uv}
	mask3DVertexStride = 12 + 4 + 4 // {float3 pos; packed color; ushort2 uv}
	argb3DVertexStride = 12 + 4     // {float3 pos; ushort2 uv}
)

// VertexFiller writes glyph quad vertex data for a text subrun.
type VertexFiller struct {
	leftTop        []geom.Point
	creationMatrix geom.Matrix
	creationBounds geom.Rect
	maskType       gpu.MaskFormat
	canDrawDirect  bool
}

// MakeVertexFiller returns a VertexFiller for the given glyph positions, copying them into its own storage.
func MakeVertexFiller(maskType gpu.MaskFormat, creationMatrix *geom.Matrix, creationBounds geom.Rect, positions []geom.Point, fillerType FillerType) VertexFiller {
	leftTop := make([]geom.Point, len(positions))
	copy(leftTop, positions)
	return VertexFiller{
		leftTop:        leftTop,
		creationMatrix: *creationMatrix,
		creationBounds: creationBounds,
		maskType:       maskType,
		canDrawDirect:  fillerType == IsDirect,
	}
}

// MaskType returns the atlas mask format this filler writes vertices for.
func (f *VertexFiller) MaskType() gpu.MaskFormat { return f.maskType }

// IsLCD reports whether this filler targets the LCD (subpixel) mask format.
func (f *VertexFiller) IsLCD() bool { return f.maskType == gpu.MaskFormatA565 }

// Count returns the number of glyphs this filler holds.
func (f *VertexFiller) Count() int { return len(f.leftTop) }

// VertexStride returns the per-vertex byte size for the given transform.
func (f *VertexFiller) VertexStride(matrix *geom.Matrix) int {
	if f.maskType != gpu.MaskFormatARGB {
		// For formats MaskFormatA565 and MaskFormatA8.
		if matrix.HasPerspective() {
			return mask3DVertexStride
		}
		return mask2DVertexStride
	}
	if matrix.HasPerspective() {
		return argb3DVertexStride
	}
	return argb2DVertexStride
}

// scalarIsInt reports whether v has no fractional part.
func scalarIsInt(v float32) bool { return v == float32(math.Floor(float64(v))) }

// canUseDirect checks for an integer translation with the same 2x2 matrix and no perspective; returns the translation.
func canUseDirect(creationMatrix, positionMatrix *geom.Matrix) (bool, geom.Point) {
	positionOrigin := positionMatrix.MapXY(0, 0)
	creationOrigin := creationMatrix.MapXY(0, 0)
	translation := geom.Pt(positionOrigin.X-creationOrigin.X, positionOrigin.Y-creationOrigin.Y)
	ok := creationMatrix.Get(geom.MScaleX) == positionMatrix.Get(geom.MScaleX) &&
		creationMatrix.Get(geom.MScaleY) == positionMatrix.Get(geom.MScaleY) &&
		creationMatrix.Get(geom.MSkewX) == positionMatrix.Get(geom.MSkewX) &&
		creationMatrix.Get(geom.MSkewY) == positionMatrix.Get(geom.MSkewY) &&
		!positionMatrix.HasPerspective() && !creationMatrix.HasPerspective() &&
		scalarIsInt(translation.X) && scalarIsInt(translation.Y)
	return ok, translation
}

// viewDifference returns the matrix mapping from creation space to positionMatrix's device space.
func (f *VertexFiller) viewDifference(positionMatrix *geom.Matrix) geom.Matrix {
	if inverse, ok := f.creationMatrix.Invert(); ok {
		var result geom.Matrix
		result.SetConcat(positionMatrix, &inverse)
		return result
	}
	return geom.IdentityMatrix()
}

// DeviceRectAndCheckTransform returns whether the positionMatrix represents an integer translation of the creation
// matrix, plus the device bounding box of all the glyphs (empty when something went singular).
func (f *VertexFiller) DeviceRectAndCheckTransform(positionMatrix *geom.Matrix) (bool, geom.Rect) {
	if f.canDrawDirect {
		if directDrawCompatible, offset := canUseDirect(&f.creationMatrix, positionMatrix); directDrawCompatible {
			return true, f.creationBounds.Offset(offset.X, offset.Y)
		}
	}
	if _, ok := f.creationMatrix.Invert(); ok {
		viewDifference := f.viewDifference(positionMatrix)
		mapped, _ := path.MapRect(&viewDifference, f.creationBounds)
		return false, mapped
	}
	// The creation matrix is singular. Do nothing.
	return false, geom.Rect{}
}

// packColor packs a premultiplied float color into bytes (clamped, x*255 + 0.5 truncation), in r, g, b, a memory order.
func packColor(c colorcore.PMColor4f) uint32 {
	toByte := func(v float32) uint32 {
		f := v*255 + 0.5
		if f < 0 {
			f = 0
		} else if f > 255 {
			f = 255
		}
		return uint32(f)
	}
	return toByte(c.R) | toByte(c.G)<<8 | toByte(c.B)<<16 | toByte(c.A)<<24
}

// vertexWriter writes the packed vertex forms.
type vertexWriter struct {
	buf    []byte
	offset int
}

func (w *vertexWriter) putF32(v float32) {
	binary.LittleEndian.PutUint32(w.buf[w.offset:], math.Float32bits(v))
	w.offset += 4
}

func (w *vertexWriter) putU32(v uint32) {
	binary.LittleEndian.PutUint32(w.buf[w.offset:], v)
	w.offset += 4
}

func (w *vertexWriter) putU16(v uint16) {
	binary.LittleEndian.PutUint16(w.buf[w.offset:], v)
	w.offset += 2
}

// putVertex writes one vertex: position (2 or 3 floats), optional packed color, atlas coords.
func (w *vertexWriter) putVertex(x, y, z float32, hasZ bool, color uint32, hasColor bool, u, v uint16) {
	w.putF32(x)
	w.putF32(y)
	if hasZ {
		w.putF32(z)
	}
	if hasColor {
		w.putU32(color)
	}
	w.putU16(u)
	w.putU16(v)
}

// FillVertexData writes count quads' worth of vertices (4 per glyph) for glyphs[offset:offset+count] into dst.
func (f *VertexFiller) FillVertexData(dst []byte, offset, count int, glyphs []*Glyph, color colorcore.PMColor4f, positionMatrix *geom.Matrix, clip geom.IRect) {
	packed := packColor(color)
	hasColor := f.maskType != gpu.MaskFormatARGB
	w := vertexWriter{buf: dst}

	// Handle direct mask drawing specifically.
	if f.canDrawDirect {
		if noTransformNeeded, originOffset := canUseDirect(&f.creationMatrix, positionMatrix); noTransformNeeded {
			var clipPtr *geom.IRect
			if !clip.IsEmpty() {
				clipPtr = &clip
			}
			f.fillDirect(&w, offset, count, glyphs, packed, hasColor, originOffset, clipPtr)
			return
		}
	}

	// Handle the general transformed case.
	viewDifference := f.viewDifference(positionMatrix)
	if !positionMatrix.HasPerspective() {
		f.fill2D(&w, offset, count, glyphs, packed, hasColor, &viewDifference)
	} else {
		f.fill3D(&w, offset, count, glyphs, packed, hasColor, &viewDifference)
	}
}

// fillDirect merges fillDirectNoClipping and fillDirectClipped: device positions are integer translations of the stored
// leftTop values; the optional clip adjusts both the device rect and the atlas coordinates.
func (f *VertexFiller) fillDirect(w *vertexWriter, offset, count int, glyphs []*Glyph, color uint32, hasColor bool, originOffset geom.Point, clip *geom.IRect) {
	for i := offset; i < offset+count; i++ {
		glyph := glyphs[i]
		uvs := glyph.AtlasLocator.UVs()
		al, at, ar, ab := uvs[0], uvs[1], uvs[2], uvs[3]
		gw := ar - al
		gh := ab - at
		l := f.leftTop[i].X + originOffset.X
		t := f.leftTop[i].Y + originOffset.Y
		dl, dt, dr, db := l, t, l+float32(gw), t+float32(gh)
		if clip != nil {
			devIRect := geom.IRectLTRB(int32(l), int32(t), int32(l)+int32(gw), int32(t)+int32(gh))
			if !clip.ContainsRect(devIRect) {
				clipped := devIRect
				if clipped.Intersect(*clip) {
					al = uint16(int32(al) + clipped.Left - devIRect.Left)
					at = uint16(int32(at) + clipped.Top - devIRect.Top)
					ar = uint16(int32(ar) + clipped.Right - devIRect.Right)
					ab = uint16(int32(ab) + clipped.Bottom - devIRect.Bottom)
					dl, dt = float32(clipped.Left), float32(clipped.Top)
					dr, db = float32(clipped.Right), float32(clipped.Bottom)
				} else {
					dl, dt, dr, db = 0, 0, 0, 0
					al, at, ar, ab = 0, 0, 0, 0
				}
			} else {
				dl, dt = float32(devIRect.Left), float32(devIRect.Top)
				dr, db = float32(devIRect.Right), float32(devIRect.Bottom)
			}
		}
		w.putVertex(dl, dt, 0, false, color, hasColor, al, at) // L,T
		w.putVertex(dl, db, 0, false, color, hasColor, al, ab) // L,B
		w.putVertex(dr, dt, 0, false, color, hasColor, ar, at) // R,T
		w.putVertex(dr, db, 0, false, color, hasColor, ar, ab) // R,B
	}
}

// fill2D maps the four corners of each glyph quad through the view difference.
func (f *VertexFiller) fill2D(w *vertexWriter, offset, count int, glyphs []*Glyph, color uint32, hasColor bool, viewDifference *geom.Matrix) {
	for i := offset; i < offset+count; i++ {
		glyph := glyphs[i]
		l := f.leftTop[i].X
		t := f.leftTop[i].Y
		r := l + float32(glyph.AtlasLocator.Width())
		b := t + float32(glyph.AtlasLocator.Height())
		lt := viewDifference.MapXY(l, t)
		lb := viewDifference.MapXY(l, b)
		rt := viewDifference.MapXY(r, t)
		rb := viewDifference.MapXY(r, b)
		uvs := glyph.AtlasLocator.UVs()
		w.putVertex(lt.X, lt.Y, 0, false, color, hasColor, uvs[0], uvs[1]) // L,T
		w.putVertex(lb.X, lb.Y, 0, false, color, hasColor, uvs[0], uvs[3]) // L,B
		w.putVertex(rt.X, rt.Y, 0, false, color, hasColor, uvs[2], uvs[1]) // R,T
		w.putVertex(rb.X, rb.Y, 0, false, color, hasColor, uvs[2], uvs[3]) // R,B
	}
}

// mapHomogeneous computes (x, y) -> M * (x, y, 1) without the perspective divide.
func mapHomogeneous(m *geom.Matrix, x, y float32) (hx, hy, hw float32) {
	hx = m.Get(geom.MScaleX)*x + m.Get(geom.MSkewX)*y + m.Get(geom.MTransX)
	hy = m.Get(geom.MSkewY)*x + m.Get(geom.MScaleY)*y + m.Get(geom.MTransY)
	hw = m.Get(geom.MPersp0)*x + m.Get(geom.MPersp1)*y + m.Get(geom.MPersp2)
	return hx, hy, hw
}

// fill3D computes homogeneous corners for perspective draws.
func (f *VertexFiller) fill3D(w *vertexWriter, offset, count int, glyphs []*Glyph, color uint32, hasColor bool, viewDifference *geom.Matrix) {
	for i := offset; i < offset+count; i++ {
		glyph := glyphs[i]
		l := f.leftTop[i].X
		t := f.leftTop[i].Y
		r := l + float32(glyph.AtlasLocator.Width())
		b := t + float32(glyph.AtlasLocator.Height())
		ltx, lty, ltw := mapHomogeneous(viewDifference, l, t)
		lbx, lby, lbw := mapHomogeneous(viewDifference, l, b)
		rtx, rty, rtw := mapHomogeneous(viewDifference, r, t)
		rbx, rby, rbw := mapHomogeneous(viewDifference, r, b)
		uvs := glyph.AtlasLocator.UVs()
		w.putVertex(ltx, lty, ltw, true, color, hasColor, uvs[0], uvs[1]) // L,T
		w.putVertex(lbx, lby, lbw, true, color, hasColor, uvs[0], uvs[3]) // L,B
		w.putVertex(rtx, rty, rtw, true, color, hasColor, uvs[2], uvs[1]) // R,T
		w.putVertex(rbx, rby, rbw, true, color, hasColor, uvs[2], uvs[3]) // R,B
	}
}
