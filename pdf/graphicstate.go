// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The /ExtGState objects that carry a paint's alpha, blend mode, and (for strokes) line cap/join/width/ miter. Objects
// are canonicalized on the document (fillGSMap / strokeGSMap) so identical paints share one indirect object. The
// soft-mask graphic states (getSMaskGraphicState + the invert function) are used by the gradient alpha (luminosity
// SMask) path; the mask-filter draw lane reuses them in a later slice.

package pdf

import (
	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/stream"
)

// sMaskMode selects whether a soft mask's /S entry is Alpha or Luminosity.
type sMaskMode uint8

const (
	sMaskAlpha sMaskMode = iota
	sMaskLuminosity
)

// fillGSKey is the canonicalization key for a fill /ExtGState.
type fillGSKey struct {
	alpha float32
	blend uint8
}

// strokeGSKey is the canonicalization key for a stroke /ExtGState.
type strokeGSKey struct {
	width float32
	miter float32
	alpha float32
	cap   uint8
	join  uint8
	blend uint8
}

// pdfBlendMode returns the raster.BlendMode index to use for the /BM entry: unsupported modes (and Xor/Plus, which PDF
// has no equivalent for) collapse to SrcOver.
func pdfBlendMode(mode raster.BlendMode) uint8 {
	if blendModeName(mode) == "" || mode == raster.BlendXor || mode == raster.BlendPlus {
		mode = raster.BlendSrcOver
	}
	return uint8(mode)
}

// toStrokeCap maps a stroke cap to its PDF /LC value (Butt->0, Round->1, Square->2).
func toStrokeCap(c canvas.StrokeCap) int32 {
	switch c {
	case canvas.CapRound:
		return 1
	case canvas.CapSquare:
		return 2
	default:
		return 0
	}
}

// toStrokeJoin maps a stroke join to its PDF /LJ value (Miter->0, Round->1, Bevel->2).
func toStrokeJoin(j canvas.StrokeJoin) int32 {
	switch j {
	case canvas.JoinRound:
		return 1
	case canvas.JoinBevel:
		return 2
	default:
		return 0
	}
}

// getGraphicStateForPaint returns the (canonicalized, shared) /ExtGState indirect reference carrying p's alpha, blend
// mode, and stroke properties.
func getGraphicStateForPaint(doc *Document, p *canvas.Paint) IndirectReference {
	mode := p.BlendMode
	alpha := colorcore.Color4fFromColor(p.Color).A

	if p.Style == canvas.StyleFill {
		key := fillGSKey{alpha: alpha, blend: pdfBlendMode(mode)}
		if ref, ok := doc.fillGSMap[key]; ok {
			return ref
		}
		state := NewDict()
		state.InsertColorComponentF("ca", key.alpha)
		state.InsertName("BM", blendModeName(raster.BlendMode(key.blend)))
		ref := doc.Emit(state)
		doc.fillGSMap[key] = ref
		return ref
	}

	key := strokeGSKey{
		width: p.StrokeWidth,
		miter: p.MiterLimit,
		alpha: alpha,
		cap:   uint8(p.Cap),
		join:  uint8(p.Join),
		blend: pdfBlendMode(mode),
	}
	if ref, ok := doc.strokeGSMap[key]; ok {
		return ref
	}
	state := NewDict()
	state.InsertColorComponentF("CA", key.alpha)
	state.InsertColorComponentF("ca", key.alpha)
	state.InsertInt("LC", toStrokeCap(canvas.StrokeCap(key.cap)))
	state.InsertInt("LJ", toStrokeJoin(canvas.StrokeJoin(key.join)))
	state.InsertScalar("LW", key.width)
	state.InsertScalar("ML", key.miter)
	state.InsertBool("SA", true) // SA = Auto stroke adjustment.
	state.InsertName("BM", blendModeName(raster.BlendMode(key.blend)))
	ref := doc.Emit(state)
	doc.strokeGSMap[key] = ref
	return ref
}

// makeInvertFunction builds a Type 4 PostScript function "{1 exch sub}" that inverts a single input (used as an SMask
// transfer function). Acrobat crashes on a Type 0 function and kpdf on a Type 2, so Type 4 is used.
func makeInvertFunction(doc *Document) IndirectReference {
	dict := NewDict()
	dict.InsertInt("FunctionType", 4)
	dict.InsertObject("Domain", makeArrayInts(0, 1))
	dict.InsertObject("Range", makeArrayInts(0, 1))
	content := stream.NewMemoryWStream()
	writeText(content, "{1 exch sub}")
	return doc.StreamOut(dict, content.Bytes(), true)
}

// getSMaskGraphicState builds an /ExtGState whose /SMask references the form-XObject sMask, in the given mask mode,
// optionally inverted through the document's shared invert function. The mask is not canonicalized (reuse is unlikely
// enough not to be worth it).
func getSMaskGraphicState(sMask IndirectReference, invert bool, mode sMaskMode, doc *Document) IndirectReference {
	sMaskDict := NewTypedDict("Mask")
	switch mode {
	case sMaskAlpha:
		sMaskDict.InsertName("S", "Alpha")
	case sMaskLuminosity:
		sMaskDict.InsertName("S", "Luminosity")
	}
	sMaskDict.InsertRef("G", sMask)
	if invert {
		if !doc.invertFunction.IsValid() {
			doc.invertFunction = makeInvertFunction(doc)
		}
		sMaskDict.InsertRef("TR", doc.invertFunction)
	}
	result := NewTypedDict("ExtGState")
	result.InsertObject("SMask", sMaskDict)
	return doc.Emit(result)
}
