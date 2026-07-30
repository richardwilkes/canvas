// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The color-glyph image lanes:
//
//   - COLRv0: each layer's outline is filled in device space with its CPAL palette color (or the foreground color for
//     palette index 0xFFFF) into the ARGB32 mask through the raster package's solid src-over blitter.
//   - Bitmap strikes (the sbix/CBDT PNG lane): the strike PNG is decoded, premultiplied, and drawn scaled into the mask
//     with a linear-filtered image shader over the strike-to-device transform. sbix 'jpg '/'tif ' graphic types, B&W
//     (EBDT/CBDT imageFormat 2/5) strikes, and SVG-table glyphs have no lane here and fall through to the outline lane,
//     which fills whatever outline the glyph carries (glyphOutlinePath loads it raw for exactly that reason, since
//     go-text's GlyphData would answer with the strike instead). The sbix and SVG specifications require such a glyph to
//     carry a fallback outline; EBDT/CBDT does not, so a bitmap-only face of that kind has nothing to draw here and
//     generateMetrics measures it as empty rather than as the strike it cannot decode.
//
// The mask buffer is premultiplied RGBA in the device word layout (raster.Pixmap), this library's N32.

package font

import (
	"bytes"
	"image"
	"image/draw"
	"image/png"

	tsfont "github.com/go-text/typesetting/font"
	"github.com/go-text/typesetting/font/opentype"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/path"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// maskLocalTranslate returns the device→mask-local translation for g, including the sub-pixel offset when the context
// is sub-pixel.
func (c *ScalerContext) maskLocalTranslate(g *Glyph) geom.Matrix {
	dx := float32(-g.Left)
	dy := float32(-g.Top)
	if c.rec.isSubpixel() {
		dx += g.packedID.SubXOffset()
		dy += g.packedID.SubYOffset()
	}
	var m geom.Matrix
	m.SetTranslate(dx, dy)
	return m
}

// renderCOLRv0 fills each layer's device-space outline with its palette color into the glyph's (zeroed) ARGB32 mask.
// Layers with an out-of-range palette index or no outline are skipped.
func (c *ScalerContext) renderCOLRv0(g *Glyph) {
	layers, ok := c.typeface.faceColorV0Layers(opentype.GID(g.packedID.GlyphID()))
	if !ok {
		return
	}
	pm := g.Pixmap()
	clip := raster.NewRasterClipRect(geom.IRectWH(g.Width, g.Height))
	translate := c.maskLocalTranslate(g)
	for _, layer := range layers {
		color, colorOK := c.typeface.PaletteColor(layer.PaletteIndex, c.rec.ForegroundColor)
		if !colorOK {
			continue
		}
		devPath := glyphOutlinePath(c.typeface, layer.GlyphID, c.mapDesignPoint)
		if devPath == nil {
			continue
		}
		local := path.New()
		devPath.TransformTo(&translate, local)
		if !local.IsFinite() {
			continue
		}
		// Anti-aliased fill with a solid color takes the solid src-over N32 blitter.
		raster.AntiFillPathRasterClip(local, clip, raster.NewSolidBlitter(&pm, color))
	}
}

// renderBitmap decodes the strike PNG, then draws it into the mask through the strike-to-device transform with a
// linear-filtered clamp/clamp image shader (a non-AA fill). The transform composes strike-pixel→font-unit placement
// (typesetting's extents), the single matrix, the sub-pixel offset, and the mask-local translation.
func (c *ScalerContext) renderBitmap(g *Glyph) {
	bm, ext, ok := c.typeface.faceBitmapGlyph(opentype.GID(g.packedID.GlyphID()), c.strikePpem)
	if !ok || bm.Format != tsfont.PNG {
		return
	}
	img := decodePremulPNG(bm.Data, bm.Width, bm.Height)
	if img == nil {
		return
	}
	w := img.Info.Width
	h := img.Info.Height
	if w <= 0 || h <= 0 {
		return
	}

	// Image pixel (u, v) → font units: (XBearing + u*Width/w, YBearing + v*Height/h) (Height is negative: rows advance
	// downward), then font units → device through mapDesignPoint's (x/upem, -y/upem) into the single matrix.
	upem := float32(c.typeface.upem)
	var m geom.Matrix
	m.SetAll(
		ext.Width/float32(w)/upem, 0, ext.XBearing/upem,
		0, -ext.Height/float32(h)/upem, -ext.YBearing/upem,
		0, 0, 1,
	)
	m.PostConcat(&c.single)
	if c.rec.isSubpixel() {
		// Bitmap glyphs come from effectively non-scalable faces, where the resample always applies the sub-pixel
		// phase.
		m.PostTranslate(g.packedID.SubXOffset(), g.packedID.SubYOffset())
	}
	m.PostTranslate(float32(-g.Left), float32(-g.Top))

	shader := shaders.NewImage(img.Image, shaders.TileClamp, shaders.TileClamp,
		shaders.SamplingOptions{Filter: shaders.FilterLinear}, nil)
	if shader == nil {
		return
	}
	pipeline := shaders.Compile(shader, m, colorcore.RGB(0, 0, 0))
	if pipeline == nil {
		return
	}
	pm := g.Pixmap()
	blitter := raster.NewShaderBlitter(&pm, pipeline, raster.BlendSrcOver)

	// The image rect through the same transform, filled without AA.
	quad := path.New()
	quad.MoveToPt(m.MapXY(0, 0))
	quad.LineToPt(m.MapXY(float32(w), 0))
	quad.LineToPt(m.MapXY(float32(w), float32(h)))
	quad.LineToPt(m.MapXY(0, float32(h)))
	quad.Close()
	clip := raster.NewRasterClipRect(geom.IRectWH(g.Width, g.Height))
	raster.FillPathRasterClip(quad, clip, blitter)
}

// decodedImage pairs the imagecore image with its info for the bitmap lane.
type decodedImage struct {
	Image *imagecore.Image
	Info  imagecore.ImageInfo
}

// maxStrikePixels bounds the pixel count a strike PNG may decode to. The per-side ceiling alone is not enough on the
// sbix path: there the "strike metrics" the header is checked against are themselves png.DecodeConfig's answer on these
// very bytes (go-text derives GlyphBitmap.Width/Height that way, having no independent metrics to read), so the equality
// check compares the header with itself and only this cap stands between a crafted IHDR and its allocation. 4 Mpixel is
// roughly 16 MB per intermediate buffer and many times the largest strike any shipping color font carries (Apple Color
// Emoji tops out at 160 px square, Noto Color Emoji at 136), while the 8191x8191 an IHDR one step below the per-side
// ceiling can declare costs about 768 MB across the NRGBA image, the premultiplied buffer, and the imagecore copy.
const maxStrikePixels = 1 << 22

// decodePremulPNG decodes PNG bytes into an N32-premul imagecore image, premultiplying with round-to-nearest rounding.
//
// strikeW/strikeH are the strike's own declared dimensions, and they gate the decode: png.Decode allocates the full
// image from the IHDR width/height before it reads a single IDAT byte, so a few-hundred-byte strike claiming
// 65535x65535 would otherwise force a multi-gigabyte allocation out of untrusted font data. The header is therefore
// read with DecodeConfig (which allocates nothing) and must agree with the strike metrics, both must stay under the
// glyph-mask ceiling — a strike bigger than the largest mask that can be allocated has nothing to contribute anyway —
// and the area must stay under maxStrikePixels, which is the only one of the three the sbix lane's self-comparison
// leaves with any work to do. FreeType's Load_SBit_Png makes the same equality check against known metrics (CBDT/EBDT)
// and bounds the dimensions it takes from the header (sbix).
func decodePremulPNG(data []byte, strikeW, strikeH int) *decodedImage {
	if strikeW <= 0 || strikeH <= 0 || strikeW >= maxGlyphWidth || strikeH >= maxGlyphHeight {
		return nil
	}
	if strikeW*strikeH > maxStrikePixels {
		return nil
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width != strikeW || cfg.Height != strikeH {
		return nil
	}
	src, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil
	}
	nrgba, ok := src.(*image.NRGBA)
	if !ok || nrgba.Rect.Min != (image.Point{}) {
		converted := image.NewNRGBA(image.Rect(0, 0, w, h))
		draw.Draw(converted, converted.Bounds(), src, b.Min, draw.Src)
		nrgba = converted
	}
	buf := make([]byte, w*h*4)
	for y := 0; y < h; y++ {
		row := nrgba.Pix[y*nrgba.Stride : y*nrgba.Stride+w*4]
		out := buf[y*w*4 : (y+1)*w*4]
		for x := 0; x < w*4; x += 4 {
			a := row[x+3]
			out[x] = colorcore.MulDiv255Round(row[x], a)
			out[x+1] = colorcore.MulDiv255Round(row[x+1], a)
			out[x+2] = colorcore.MulDiv255Round(row[x+2], a)
			out[x+3] = a
		}
	}
	info := imagecore.MakeN32Premul(int32(w), int32(h))
	img := imagecore.NewRasterData(info, buf, w*4)
	if img == nil {
		return nil
	}
	return &decodedImage{Image: img, Info: info}
}
