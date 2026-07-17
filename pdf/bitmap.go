// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file serializes an image as a PDF Image XObject, plus the keyed-image de-dup helpers that avoid re-serializing
// the same (sub)image within a document. It implements the Flate/Uncompressed lanes (doDeflatedImage plus the
// DeviceGray SMask via doDeflatedAlpha) that the default configuration uses (encodingQuality 101 → lossless → Flate).
// The DCT/JPEG lanes (passthrough of already-encoded JPEG data, and re-encoding opaque images for encodingQuality ≤
// 100) are deferred: they only trigger when the caller explicitly requests lossy encoding, and the Flate lane produces
// a correct — if larger — image either way. ICC color-profile embedding is dropped, since all content is sRGB: images
// use DeviceRGB/ DeviceGray directly.

package pdf

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/stream"
)

// ---- keyed image de-dup ----------------------------------------------------------------------------

// bitmapKey is an image's subset rect (in the original image's pixel space) plus its unique ID. It de-dups repeated
// draws of the same (sub)image within a document (the pdfBitmapMap key).
type bitmapKey struct {
	subset geom.IRect
	id     uint32
}

// keyedImage is an image paired with the bitmapKey that de-dups its subsets. The logical image is the region sub-rect
// of img (in img's own pixel coordinates); the logical bounds are (0,0,region.width,region.height).
type keyedImage struct {
	img    *imagecore.Image
	region geom.IRect
	id     uint32
}

// newKeyedImage wraps the whole image, keyed on its bounds and unique ID.
func newKeyedImage(img *imagecore.Image) keyedImage {
	if img == nil {
		return keyedImage{}
	}
	return keyedImage{img: img, region: geom.IRectWH(img.Width(), img.Height()), id: img.UniqueID()}
}

func (k keyedImage) valid() bool { return k.img != nil && !k.region.IsEmpty() }

// bounds returns the logical image bounds, origin at (0,0).
func (k keyedImage) bounds() geom.IRect { return geom.IRectWH(k.region.Width(), k.region.Height()) }

// image returns the logical (possibly subsetted) image with origin (0,0). When the region already covers the whole
// underlying image it is returned as-is; otherwise the region is extracted into a fresh raster image in the source
// color type (the alpha-only / color-filter / perspective draw lanes rasterize this image directly).
func (k keyedImage) image() *imagecore.Image {
	if k.region == geom.IRectWH(k.img.Width(), k.img.Height()) {
		return k.img
	}
	info := imagecore.ImageInfo{
		Width:     k.region.Width(),
		Height:    k.region.Height(),
		ColorType: k.img.ColorType(),
		AlphaType: k.img.AlphaType(),
	}
	rowBytes := info.MinRowBytes()
	data := make([]byte, rowBytes*int(info.Height))
	if !k.img.ReadPixels(info, data, rowBytes, k.region.Left, k.region.Top, imagecore.CachingAllow) {
		return nil
	}
	return imagecore.NewRasterData(info, data, rowBytes)
}

func (k keyedImage) isOpaque() bool    { return k.img.IsOpaque() }
func (k keyedImage) isAlphaOnly() bool { return k.img.IsAlphaOnly() }
func (k keyedImage) key() bitmapKey    { return bitmapKey{subset: k.region, id: k.id} }

// subset narrows k to sub, which is in the logical image's coordinates (origin 0,0). The returned keyed image's region
// is that sub-rect intersected with the current region, offset by the current region's top-left so the key still refers
// to the original image's pixel space.
func (k keyedImage) subset(sub geom.IRect) keyedImage {
	s := sub
	if !s.Intersect(k.bounds()) {
		return keyedImage{}
	}
	return keyedImage{img: k.img, region: s.Offset(k.region.Left, k.region.Top), id: k.id}
}

// ---- image serialization ---------------------------------------------------------------------------

// pdfStreamFormat identifies which PDF stream filter an image was encoded with.
type pdfStreamFormat uint8

const (
	pdfFormatDCT pdfStreamFormat = iota // deferred
	pdfFormatFlate
	pdfFormatUncompressed
)

// pdfPixels holds an image read into one of the three PDF-serializable layouts: Alpha8 / Gray8 (one byte per pixel) or
// RGBA8888-unpremul (four bytes per pixel).
type pdfPixels struct {
	data      []byte
	width     int32
	height    int32
	colorType imagecore.ColorType
}

// serializeImage reads the image's pixels and emits the color Image XObject plus (for non-opaque images) a DeviceGray
// SMask. Only the Flate/Uncompressed lane is implemented.
func serializeImage(k keyedImage, doc *Document, ref IndirectReference) {
	pm := readImagePixels(k)
	doDeflatedImage(pm, doc, computeIsOpaque(pm), ref)
}

// readImagePixels reads the keyed image's region into the layout doDeflatedImage consumes (Alpha8 / Gray8 /
// RGBA8888-unpremul), erasing to opaque black when the read fails.
func readImagePixels(k keyedImage) pdfPixels {
	w, h := k.region.Width(), k.region.Height()
	var ct imagecore.ColorType
	var at imagecore.AlphaType
	var bpp int
	switch {
	case k.img.IsAlphaOnly():
		ct, at, bpp = imagecore.ColorTypeAlpha8, imagecore.AlphaTypePremul, 1
	case k.img.ColorType() == imagecore.ColorTypeGray8:
		ct, at, bpp = imagecore.ColorTypeGray8, imagecore.AlphaTypeOpaque, 1
	default:
		// The color channels are always read unpremultiplied; doDeflatedImage recomputes opacity by scanning the read
		// alpha rather than trusting the source image's opaque flag.
		ct, at, bpp = imagecore.ColorTypeRGBA8888, imagecore.AlphaTypeUnpremul, 4
	}
	info := imagecore.ImageInfo{Width: w, Height: h, ColorType: ct, AlphaType: at}
	rowBytes := int(w) * bpp
	data := make([]byte, rowBytes*int(h))
	if !k.img.ReadPixels(info, data, rowBytes, k.region.Left, k.region.Top, imagecore.CachingAllow) {
		eraseBlack(data, ct)
	}
	return pdfPixels{width: w, height: h, colorType: ct, data: data}
}

// eraseBlack fills data with opaque black, the read-failure fallback.
func eraseBlack(data []byte, ct imagecore.ColorType) {
	switch ct {
	case imagecore.ColorTypeAlpha8:
		for i := range data {
			data[i] = 0xFF
		}
	case imagecore.ColorTypeGray8:
		// Black is gray 0; data is already zero.
	default:
		for i := 0; i+3 < len(data); i += 4 {
			data[i+3] = 0xFF // opaque
		}
	}
}

// computeIsOpaque scans pm's alpha data and reports whether every pixel is fully opaque.
func computeIsOpaque(pm pdfPixels) bool {
	switch pm.colorType {
	case imagecore.ColorTypeGray8:
		return true // opaque alpha type
	case imagecore.ColorTypeAlpha8:
		for _, a := range pm.data {
			if a != 0xFF {
				return false
			}
		}
		return true
	default:
		n := int(pm.width) * int(pm.height)
		for i := 0; i < n; i++ {
			if pm.data[i*4+3] != 0xFF {
				return false
			}
		}
		return true
	}
}

// doDeflatedImage emits the color plane as an Image XObject (DeviceGray for Alpha8/Gray8, DeviceRGB for color) plus,
// when the image is not opaque, a DeviceGray SMask.
func doDeflatedImage(pm pdfPixels, doc *Document, isOpaque bool, ref IndirectReference) {
	colorSpace := "DeviceGray"
	var raw []byte
	switch pm.colorType {
	case imagecore.ColorTypeAlpha8:
		// The color of an alpha-only image is all black (its alpha rides the SMask).
		raw = make([]byte, int(pm.width)*int(pm.height))
	case imagecore.ColorTypeGray8:
		raw = pm.data
	default:
		colorSpace = "DeviceRGB"
		raw = buildRGBPlane(pm)
	}
	content, format := deflateForPDF(doc, raw)
	var sMask IndirectReference
	if !isOpaque {
		sMask = doc.ReserveRef()
	}
	emitImageStream(doc, ref, content, geom.ISize{Width: pm.width, Height: pm.height}, colorSpace, sMask, format)
	if !isOpaque {
		doDeflatedAlpha(pm, doc, sMask)
	}
}

// doDeflatedAlpha emits the image's alpha channel as a single-component DeviceGray Image XObject used as the color
// image's soft mask.
func doDeflatedAlpha(pm pdfPixels, doc *Document, ref IndirectReference) {
	var raw []byte
	if pm.colorType == imagecore.ColorTypeAlpha8 {
		raw = pm.data
	} else {
		n := int(pm.width) * int(pm.height)
		raw = make([]byte, n)
		for i := 0; i < n; i++ {
			raw[i] = pm.data[i*4+3]
		}
	}
	content, format := deflateForPDF(doc, raw)
	emitImageStream(doc, ref, content, geom.ISize{Width: pm.width, Height: pm.height}, "DeviceGray",
		IndirectReference{}, format)
}

// buildRGBPlane emits the R,G,B bytes of the color plane, replacing fully-transparent pixels with their neighbor
// average (neighborAvgColor) so a soft-mask resample of a sharp transparency edge does not bleed black into visible
// pixels.
func buildRGBPlane(pm pdfPixels) []byte {
	w, h := int(pm.width), int(pm.height)
	out := make([]byte, 0, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 4
			r, g, b := pm.data[i], pm.data[i+1], pm.data[i+2]
			if pm.data[i+3] == 0 { // fully transparent
				r, g, b = neighborAvgColor(pm, x, y)
			}
			out = append(out, r, g, b)
		}
	}
	return out
}

// neighborAvgColor averages the RGB of the 3x3 neighborhood, counting only pixels whose color is not fully transparent
// black. Returns black when none qualify.
func neighborAvgColor(pm pdfPixels, xOrig, yOrig int) (avgR, avgG, avgB byte) {
	w, h := int(pm.width), int(pm.height)
	var rs, gs, bs, n uint32
	ymin := max(0, yOrig-1)
	ymax := min(yOrig+1, h-1)
	xmin := max(0, xOrig-1)
	xmax := min(xOrig+1, w-1)
	for y := ymin; y <= ymax; y++ {
		for x := xmin; x <= xmax; x++ {
			i := (y*w + x) * 4
			r, g, b, a := pm.data[i], pm.data[i+1], pm.data[i+2], pm.data[i+3]
			if r != 0 || g != 0 || b != 0 || a != 0 {
				rs += uint32(r)
				gs += uint32(g)
				bs += uint32(b)
				n++
			}
		}
	}
	if n > 0 {
		return byte(rs / n), byte(gs / n), byte(bs / n)
	}
	return 0, 0, 0
}

// deflateForPDF compresses raw with FlateDecode unless the document disables compression. Both doDeflatedImage and
// doDeflatedAlpha compress image data unconditionally — there is no minimum-savings heuristic, as there is for
// text/content streams — because image data essentially always compresses.
func deflateForPDF(doc *Document, raw []byte) ([]byte, pdfStreamFormat) {
	if doc.metadata.CompressionLevel == CompressionNone {
		return raw, pdfFormatUncompressed
	}
	var buf stream.MemoryWStream
	dw := NewDeflateWStream(&buf, int(doc.metadata.CompressionLevel))
	dw.Write(raw)
	dw.Finalize()
	return append([]byte(nil), buf.Bytes()...), pdfFormatFlate
}

// emitImageStream builds the Image XObject dictionary and emits its stream. ICC color spaces are dropped, since all
// content is sRGB: colorSpace is always DeviceRGB or DeviceGray.
func emitImageStream(doc *Document, ref IndirectReference, content []byte, size geom.ISize, colorSpace string, sMask IndirectReference, format pdfStreamFormat) {
	if !ref.IsValid() {
		return
	}
	d := NewTypedDict("XObject")
	d.InsertName("Subtype", "Image")
	d.InsertInt("Width", size.Width)
	d.InsertInt("Height", size.Height)
	d.InsertName("ColorSpace", colorSpace)
	if sMask.IsValid() {
		d.InsertRef("SMask", sMask)
	}
	d.InsertInt("BitsPerComponent", 8)
	switch format {
	case pdfFormatDCT:
		d.InsertName("Filter", "DCTDecode")
		d.InsertInt("ColorTransform", 0)
	case pdfFormatFlate:
		d.InsertName("Filter", "FlateDecode")
	case pdfFormatUncompressed:
	}
	d.InsertInt("Length", int32(len(content)))
	doc.emitStream(d, content, ref)
}

// serializeImageCached serializes a keyed image at most once per document; repeated draws reuse the cached indirect
// reference.
func (d *Document) serializeImageCached(k keyedImage) IndirectReference {
	key := k.key()
	if ref, ok := d.pdfBitmapMap[key]; ok {
		return ref
	}
	ref := d.ReserveRef()
	serializeImage(k, d, ref)
	d.pdfBitmapMap[key] = ref
	return ref
}
