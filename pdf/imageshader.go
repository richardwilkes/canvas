// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The non-gradient shader lanes: makeImageShader (a tiled raster pattern rendered into a pattern-cell PDF device, with
// mirror/clamp/decal edge synthesis) and makeFallbackShader (rasterize an arbitrary shader into an N32 surface, then
// image-shader it). Gradients route to gradientshader.go instead. Alpha-only source images draw tinted by the paint
// color the same way any other geometry does: the pattern-cell draw routes through internalDrawImageRect's alpha-only
// luminosity-SMask lane, so the A8 coverage renders correctly.

package pdf

import (
	"math"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
	"github.com/richardwilkes/canvas/surface"
)

// isTiled reports whether m wraps the pattern cell: true for mirror/repeat, false for clamp/decal.
func isTiled(m shaders.TileMode) bool {
	return m == shaders.TileMirror || m == shaders.TileRepeat
}

// scaleTranslate builds a matrix that scales by (sx, sy) then translates by (tx, ty).
func scaleTranslate(sx, sy, tx, ty float32) geom.Matrix {
	var m geom.Matrix
	m.SetScaleTranslate(sx, sy, tx, ty)
	return m
}

// drawImageIntoCell draws image at (0,0) with a solid paint carrying paintColor (default nearest sampling, src-over).
// For a color image the paint RGB is ignored; for an alpha-only image it tints the coverage (internalDrawImageRect's
// alpha-only luminosity-SMask lane).
func drawImageIntoCell(c *canvas.Canvas, img *imagecore.Image, paintColor colorcore.Color4f) {
	paint := canvas.NewPaint()
	paint.Color = paintColor.ToColor()
	c.DrawImage(img, 0, 0, shaders.SamplingOptions{}, paint)
}

// drawImageMatrix concats matrix onto the canvas, draws image at (0,0), then restores.
func drawImageMatrix(c *canvas.Canvas, img *imagecore.Image, matrix geom.Matrix, paintColor colorcore.Color4f) {
	if img == nil {
		return
	}
	c.Save()
	c.Concat(&matrix)
	drawImageIntoCell(c, img, paintColor)
	c.Restore()
}

// patternBitmap is the image read back once as unpremultiplied RGBA8888 for clamp corner/edge synthesis; the edge
// strips draw as fresh images, which the device premultiplies exactly as drawing the source image would.
type patternBitmap struct {
	rgba []byte // row-major, w*h*4 bytes
	w    int32
	h    int32
}

// toPatternBitmap reads the whole image into unpremul RGBA, falling back to an all-transparent buffer when the read
// fails.
func toPatternBitmap(img *imagecore.Image) patternBitmap {
	w, h := img.Width(), img.Height()
	bm := patternBitmap{w: w, h: h, rgba: make([]byte, int(w)*int(h)*4)}
	info := imagecore.ImageInfo{
		Width:     w,
		Height:    h,
		ColorType: imagecore.ColorTypeRGBA8888,
		AlphaType: imagecore.AlphaTypeUnpremul,
	}
	_ = img.ReadPixels(info, bm.rgba, int(w)*4, 0, 0, imagecore.CachingAllow)
	return bm
}

// color4f returns the unpremultiplied color at (x, y) in the stored bitmap.
func (bm patternBitmap) color4f(x, y int32) colorcore.Color4f {
	i := (int(y)*int(bm.w) + int(x)) * 4
	return colorcore.Color4f{
		R: float32(bm.rgba[i]) / 255,
		G: float32(bm.rgba[i+1]) / 255,
		B: float32(bm.rgba[i+2]) / 255,
		A: float32(bm.rgba[i+3]) / 255,
	}
}

// subsetImage returns a fresh image over the sub-rect [l,t)-[r,b) of bm.
func (bm patternBitmap) subsetImage(l, t, r, b int32) *imagecore.Image {
	sw, sh := r-l, b-t
	if sw <= 0 || sh <= 0 {
		return nil
	}
	dstRow := int(sw) * 4
	data := make([]byte, dstRow*int(sh))
	for y := int32(0); y < sh; y++ {
		srcOff := (int(t+y)*int(bm.w) + int(l)) * 4
		copy(data[int(y)*dstRow:int(y)*dstRow+dstRow], bm.rgba[srcOff:srcOff+dstRow])
	}
	info := imagecore.ImageInfo{
		Width:     sw,
		Height:    sh,
		ColorType: imagecore.ColorTypeRGBA8888,
		AlphaType: imagecore.AlphaTypeUnpremul,
	}
	return imagecore.NewRasterData(info, data, dstRow)
}

// fillColorFromBitmap fills a rect with the (alpha-scaled) color sampled at bitmap pixel (x, y), skipping empty rects.
func fillColorFromBitmap(c *canvas.Canvas, left, top, right, bottom float32, bm patternBitmap, x, y int32, alpha float32) {
	rect := geom.RectLTRB(left, top, right, bottom)
	if rect.IsEmpty() {
		return
	}
	color := bm.color4f(x, y)
	paint := canvas.NewPaint()
	paint.Color = colorcore.Color4f{R: color.R, G: color.G, B: color.B, A: alpha * color.A}.ToColor()
	c.DrawRect(rect, paint)
}

// makeImageShader draws the image (plus mirror/clamp edge synthesis) into a pattern-cell PDF device sized to the
// shader-space clip, then wraps the recorded content as a PatternType 1 tiling pattern positioned by finalMatrix.
func makeImageShader(doc *Document, finalMatrix geom.Matrix, tileModesX, tileModesY shaders.TileMode, bBox geom.Rect, img *imagecore.Image, paintColor colorcore.Color4f) IndirectReference {
	// Map clip bounds to shader space so the pattern device is large enough for fake clamping.
	deviceBounds := bBox
	if !inverseTransformBBox(&finalMatrix, &deviceBounds) {
		return IndirectReference{}
	}

	bitmapBounds := geom.RectWH(float32(img.Width()), float32(img.Height()))

	// For tiling modes, extend the bounds to include the bitmap so it is not clipped out.
	if isTiled(tileModesX) || isTiled(tileModesY) {
		deviceBounds.Join(bitmapBounds)
	}

	patternDeviceSize := geom.ISize{
		Width:  geom.CeilToInt(deviceBounds.Width()),
		Height: geom.CeilToInt(deviceBounds.Height()),
	}
	patternDevice := newDevice(patternDeviceSize, doc, geom.IdentityMatrix())
	c := canvas.New(patternDevice)

	patternBBox := geom.RectWH(float32(img.Width()), float32(img.Height()))
	width := patternBBox.Width()
	height := patternBBox.Height()

	// Translate so the bitmap origin is at (0, 0); undo the translation in the final matrix.
	c.Translate(-deviceBounds.Left, -deviceBounds.Top)
	patternBBox = patternBBox.Offset(-deviceBounds.Left, -deviceBounds.Top)
	finalMatrix.PreTranslate(deviceBounds.Left, deviceBounds.Top)

	drawImageIntoCell(c, img, paintColor)

	// Tiling is implied. First handle mirroring.
	if tileModesX == shaders.TileMirror {
		drawImageMatrix(c, img, scaleTranslate(-1, 1, 2*width, 0), paintColor)
		patternBBox.Right += width
	}
	if tileModesY == shaders.TileMirror {
		drawImageMatrix(c, img, scaleTranslate(1, -1, 0, 2*height), paintColor)
		patternBBox.Bottom += height
	}
	if tileModesX == shaders.TileMirror && tileModesY == shaders.TileMirror {
		drawImageMatrix(c, img, scaleTranslate(-1, -1, 2*width, 2*height), paintColor)
	}

	// Then handle clamping, which expands the pattern canvas to cover the surface bbox.
	var bm patternBitmap
	if tileModesX == shaders.TileClamp || tileModesY == shaders.TileClamp {
		bm = toPatternBitmap(img)
	}

	// With both axes clamped, start by filling the four corners with the corner colors.
	if tileModesX == shaders.TileClamp && tileModesY == shaders.TileClamp {
		fillColorFromBitmap(c, deviceBounds.Left, deviceBounds.Top, 0, 0, bm, 0, 0, paintColor.A)
		fillColorFromBitmap(c, width, deviceBounds.Top, deviceBounds.Right, 0, bm, bm.w-1, 0, paintColor.A)
		fillColorFromBitmap(c, width, height, deviceBounds.Right, deviceBounds.Bottom, bm, bm.w-1, bm.h-1,
			paintColor.A)
		fillColorFromBitmap(c, deviceBounds.Left, height, 0, deviceBounds.Bottom, bm, 0, bm.h-1, paintColor.A)
	}

	// Then expand the left, right, top, then bottom edges by stretching the outermost pixel strip.
	if tileModesX == shaders.TileClamp {
		if deviceBounds.Left < 0 {
			left := bm.subsetImage(0, 0, 1, bm.h)
			leftMatrix := scaleTranslate(-deviceBounds.Left, 1, deviceBounds.Left, 0)
			drawImageMatrix(c, left, leftMatrix, paintColor)
			if tileModesY == shaders.TileMirror {
				leftMatrix.PostScale(1, -1)
				leftMatrix.PostTranslate(0, 2*height)
				drawImageMatrix(c, left, leftMatrix, paintColor)
			}
			patternBBox.Left = 0
		}
		if deviceBounds.Right > width {
			right := bm.subsetImage(bm.w-1, 0, bm.w, bm.h)
			rightMatrix := scaleTranslate(deviceBounds.Right-width, 1, width, 0)
			drawImageMatrix(c, right, rightMatrix, paintColor)
			if tileModesY == shaders.TileMirror {
				rightMatrix.PostScale(1, -1)
				rightMatrix.PostTranslate(0, 2*height)
				drawImageMatrix(c, right, rightMatrix, paintColor)
			}
			patternBBox.Right = deviceBounds.Width()
		}
	}
	if tileModesX == shaders.TileDecal {
		if deviceBounds.Left < 0 {
			patternBBox.Left = 0
		}
		if deviceBounds.Right > width {
			patternBBox.Right = deviceBounds.Width()
		}
	}

	if tileModesY == shaders.TileClamp {
		if deviceBounds.Top < 0 {
			top := bm.subsetImage(0, 0, bm.w, 1)
			topMatrix := scaleTranslate(1, -deviceBounds.Top, 0, deviceBounds.Top)
			drawImageMatrix(c, top, topMatrix, paintColor)
			if tileModesX == shaders.TileMirror {
				topMatrix.PostScale(-1, 1)
				topMatrix.PostTranslate(2*width, 0)
				drawImageMatrix(c, top, topMatrix, paintColor)
			}
			patternBBox.Top = 0
		}
		if deviceBounds.Bottom > height {
			bottom := bm.subsetImage(0, bm.h-1, bm.w, bm.h)
			bottomMatrix := scaleTranslate(1, deviceBounds.Bottom-height, 0, height)
			drawImageMatrix(c, bottom, bottomMatrix, paintColor)
			if tileModesX == shaders.TileMirror {
				bottomMatrix.PostScale(-1, 1)
				bottomMatrix.PostTranslate(2*width, 0)
				drawImageMatrix(c, bottom, bottomMatrix, paintColor)
			}
			patternBBox.Bottom = deviceBounds.Height()
		}
	}
	if tileModesY == shaders.TileDecal {
		if deviceBounds.Top < 0 {
			patternBBox.Top = 0
		}
		if deviceBounds.Bottom > height {
			patternBBox.Bottom = deviceBounds.Height()
		}
	}

	imageShader := patternDevice.contentBytes()
	resourceDict := patternDevice.makeResourceDict()
	dict := NewDict()
	populateTilingPatternDict(dict, patternBBox, resourceDict, &finalMatrix)
	return doc.StreamOut(dict, imageShader, true)
}

// pinInt clamps x to the range [lo, hi].
func pinInt(x, lo, hi int32) int32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

// makeFallbackShader rasterizes the whole shader into an N32 surface (clamped to ~1M pixels), snapshots it, and
// image-shaders the result with clamp/clamp.
func makeFallbackShader(doc *Document, shader shaders.Shader, canvasTransform geom.Matrix, surfaceBBox geom.IRect, paintColor colorcore.Color4f) IndirectReference {
	// surfaceBBox is device space; map it to shader space for the adjustments make_image_shader expects.
	shaderRect := surfaceBBox.ToRect()
	if !inverseTransformBBox(&canvasTransform, &shaderRect) {
		return IndirectReference{}
	}
	const maxBitmapArea = 1024 * 1024
	bitmapArea := float32(surfaceBBox.Width()) * float32(surfaceBBox.Height())
	rasterScale := float32(1)
	if bitmapArea > float32(maxBitmapArea) {
		rasterScale *= geom.ScalarSqrt(float32(maxBitmapArea) / bitmapArea)
	}
	sizeW := pinInt(geom.CeilToInt(rasterScale*float32(surfaceBBox.Width())), 1, maxBitmapArea)
	sizeH := pinInt(geom.CeilToInt(rasterScale*float32(surfaceBBox.Height())), 1, maxBitmapArea)
	scaleW := float32(sizeW) / shaderRect.Width()
	scaleH := float32(sizeH) / shaderRect.Height()

	surf := surface.NewRasterN32Premul(sizeW, sizeH, nil)
	if surf == nil {
		return IndirectReference{}
	}
	c := surf.Canvas()
	c.Clear(colorcore.Color(0)) // Transparent black.

	p := canvas.NewPaint()
	p.Color = paintColor.ToColor()
	p.Shader = shader

	c.Scale(scaleW, scaleH)
	c.Translate(-shaderRect.Left, -shaderRect.Top)
	c.DrawPaint(p)

	var shaderTransform geom.Matrix
	shaderTransform.SetTranslate(shaderRect.Left, shaderRect.Top)
	shaderTransform.PreScale(1/scaleW, 1/scaleH)

	image := surf.MakeImageSnapshot()
	if image == nil {
		return IndirectReference{}
	}
	var finalMatrix geom.Matrix
	finalMatrix.SetConcat(&canvasTransform, &shaderTransform)
	return makeImageShader(doc, finalMatrix, shaders.TileClamp, shaders.TileClamp, surfaceBBox.ToRect(),
		image, paintColor)
}

// imageShaderKeyString serializes the image-shader parameters into a collision-free byte string suitable for use as a
// map key in Document.imageShaderMap.
func imageShaderKeyString(finalMatrix geom.Matrix, surfaceBBox geom.IRect, k bitmapKey, tmx, tmy shaders.TileMode, paintColor colorcore.Color4f) string {
	var b []byte
	putU32 := func(v uint32) { b = append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }
	putF := func(v float32) { putU32(math.Float32bits(v)) }
	for _, v := range finalMatrix.As9() {
		putF(v)
	}
	putU32(uint32(surfaceBBox.Left))
	putU32(uint32(surfaceBBox.Top))
	putU32(uint32(surfaceBBox.Right))
	putU32(uint32(surfaceBBox.Bottom))
	putU32(uint32(k.subset.Left))
	putU32(uint32(k.subset.Top))
	putU32(uint32(k.subset.Right))
	putU32(uint32(k.subset.Bottom))
	putU32(k.id)
	putU32(uint32(tmx))
	putU32(uint32(tmy))
	putF(paintColor.R)
	putF(paintColor.G)
	putF(paintColor.B)
	putF(paintColor.A)
	return string(b)
}
