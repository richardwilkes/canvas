// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The program key: a uint32 word array built from the geometry processor (class id, subclass key, attribute key,
// sampler keys), the dst-texture key, the FP tree keys, the transfer-processor key, write swizzle, and the
// primitive-type bit. The GL backend appends nothing extra, so the initial key length is the whole key.

package gl

import (
	"github.com/richardwilkes/canvas/gpu"
)

// classIDBits is the number of bits allotted to a class id within the key.
const classIDBits = 8

// samplerOrImageTypeKeyBits is the number of bits allotted to a sampler/image type within the key.
const samplerOrImageTypeKeyBits = 4

// ProgramDesc is a uint32 key array identifying a program's configuration. Its byte image doubles as the program cache
// map key.
type ProgramDesc struct {
	key []uint32
}

// IsValid reports whether the descriptor holds a built key.
func (d *ProgramDesc) IsValid() bool { return len(d.key) > 0 }

// Key returns the raw key words.
func (d *ProgramDesc) Key() []uint32 { return d.key }

// AppendKeyBytes appends the key's little-endian byte image to dst and returns the extended slice. The byte image is
// the program cache's map key; appending into reused scratch (rather than a fresh buffer) lets the per-draw cache
// lookup avoid allocating on a hit, paired with the compiler's m[string(bytes)] no-allocation map access (see
// ProgramCache.findOrCreateProgramImpl).
func (d *ProgramDesc) AppendKeyBytes(dst []byte) []byte {
	for _, w := range d.key {
		dst = append(dst, byte(w), byte(w>>8), byte(w>>16), byte(w>>24))
	}
	return dst
}

// MapKey returns the key bytes as a string usable as a Go map key.
func (d *ProgramDesc) MapKey() string {
	return string(d.AppendKeyBytes(make([]byte, 0, len(d.key)*4)))
}

// textureTypeKey returns the small integer tag for a texture type used within a sampler key. Value 1 is reserved for
// external textures (unreachable with the desktop trim), so rectangle textures use 2.
func textureTypeKey(t gpu.TextureType) uint32 {
	switch t {
	case gpu.TextureType2D:
		return 0
	case gpu.TextureTypeRectangle:
		return 2
	default:
		panic("unexpected texture type")
	}
}

// samplerKey packs a sampler's texture type and swizzle into one key word.
func samplerKey(textureType gpu.TextureType, swizzle gpu.Swizzle) uint32 {
	samplerTypeKey := textureTypeKey(textureType)
	return samplerTypeKey | uint32(swizzle)<<samplerOrImageTypeKeyBits
}

// addGeomProcSamplerKeys appends the geometry processor's texture-sampler keys.
func addGeomProcSamplerKeys(b *gpu.KeyBuilder, geomProc GeometryProcessor, caps *Caps) {
	numTextureSamplers := geomProc.gpBase().NumTextureSamplers()
	b.Add32(uint32(numTextureSamplers), "ppNumSamplers")
	for i := 0; i < numTextureSamplers; i++ {
		sampler := geomProc.gpBase().TextureSampler(i)
		b.Add32(samplerKey(sampler.TextureType(), sampler.Swizzle()), "samplerKey")
		// caps.addExtraSamplerKey is only non-empty for external textures/decal-emulation workarounds that are
		// unreachable with the desktop trim.
		_ = caps
	}
}

// genGeomProcKey appends the geometry processor's contribution to the program key: class id, subclass key, attribute
// key, and sampler keys.
func genGeomProcKey(geomProc GeometryProcessor, caps *Caps, b *gpu.KeyBuilder) {
	b.AppendComment(geomProc.Name())
	b.AddBits(classIDBits, uint32(geomProc.ClassID()), "geomProcClassID")

	geomProc.AddToKey(caps.ShaderCaps, b)
	gpAttributeKey(geomProc, b)

	addGeomProcSamplerKeys(b, geomProc, caps)
}

// genXPKey appends the transfer processor's contribution to the program key.
func genXPKey(xp XferProcessor, caps *Caps, b *gpu.KeyBuilder) {
	b.AppendComment(xp.Name())
	b.AddBits(classIDBits, uint32(xp.ClassID()), "xpClassID")
	xpAddToKey(xp, caps.ShaderCaps, b)
}

// genFPKey appends a fragment processor's contribution to the program key, then recurses into its child processors (a
// nil child is folded in as a sentinel "Null" class id).
func genFPKey(fp FragmentProcessor, caps *Caps, b *gpu.KeyBuilder) {
	b.AppendComment(fp.Name())
	b.AddBits(classIDBits, uint32(fp.ClassID()), "fpClassID")
	b.AddBits(coordTransformKeyBits, ComputeCoordTransformsKey(fp), "fpTransforms")

	if te, ok := fp.(*TextureEffect); ok {
		b.Add32(samplerKey(te.view.Proxy().TextureType(), te.view.Swizzle()), "fpSamplerKey")
		// caps.addExtraSamplerKey is only non-empty for external textures/decal emulation, both outside the desktop
		// trim.
	}

	fp.onAddToKey(caps.ShaderCaps, b)
	b.Add32(uint32(fp.fpBase().NumChildProcessors()), "fpNumChildren")

	for i := 0; i < fp.fpBase().NumChildProcessors(); i++ {
		if child := fp.fpBase().ChildProcessor(i); child != nil {
			genFPKey(child, caps, b)
		} else {
			// Fold in a sentinel value as the "class ID" for any null children.
			b.AppendComment("Null")
			b.AddBits(classIDBits, uint32(NullClassID), "fpClassID")
		}
	}
}

// genDstTextureKey appends the destination-texture contribution to the program key, when the pipeline reads back a copy
// of the destination.
func genDstTextureKey(pipeline *Pipeline, b *gpu.KeyBuilder) {
	dstView := pipeline.DstProxyView()
	if dstView.IsValid() {
		b.Add32(samplerKey(dstView.Proxy().TextureType(), dstView.Swizzle()), "dstSamplerKey")
		b.AddBool(dstView.Origin() == gpu.OriginTopLeft, "surfaceOrigin")
		b.AddBool(pipeline.UsesDstInputAttachment(), "usesDstInputAttachment")
	}
}

// genProgramKey builds the full program key by appending the geometry processor, dst-texture, fragment-processor, and
// transfer-processor contributions in order.
func genProgramKey(b *gpu.KeyBuilder, programInfo *ProgramInfo, caps *Caps) {
	genGeomProcKey(programInfo.GeomProc(), caps, b)

	pipeline := programInfo.Pipeline()

	genDstTextureKey(pipeline, b)

	b.AddBits(2, uint32(pipeline.NumFragmentProcessors()), "numFPs")
	b.AddBits(1, uint32(pipeline.NumColorFragmentProcessors()), "numColorFPs")
	for i := 0; i < pipeline.NumFragmentProcessors(); i++ {
		genFPKey(pipeline.FragmentProcessor(i), caps, b)
	}

	genXPKey(pipeline.XferProcessor(), caps, b)

	b.AddBits(16, uint32(pipeline.WriteSwizzle()), "writeSwizzle")
	b.AddBool(pipeline.SnapVerticesToPixelCenters(), "snapVertices")
	// The base descriptor only stores whether or not the primitiveType is PrimitiveTypePoints.
	b.AddBool(programInfo.PrimitiveType() == gpu.PrimitiveTypePoints, "isPoints")

	// Put a clean break between the "common" data and any backend data appended later.
	b.Flush()
}

// BuildProgramDesc builds the program descriptor for the GL backend, which appends nothing beyond the generic key.
func BuildProgramDesc(desc *ProgramDesc, programInfo *ProgramInfo, caps *Caps) {
	var b gpu.KeyBuilder
	buildProgramDescReusing(desc, &b, programInfo, caps)
}

// buildProgramDescReusing is BuildProgramDesc over a caller-owned builder. The per-draw cache path
// (ProgramCache.FindOrCreateProgram) passes a builder it keeps across draws, so building a key allocates nothing (the
// builder is heap-resident as a cache field, sidestepping the escape the add methods' interface calls force on a fresh
// builder).
func buildProgramDescReusing(desc *ProgramDesc, b *gpu.KeyBuilder, programInfo *ProgramInfo, caps *Caps) {
	desc.key = desc.key[:0]
	b.Reset(&desc.key)
	genProgramKey(b, programInfo, caps)
}

// DescribeProgram returns the human-readable field-by-field dump of the key. Nothing in the module calls it; it exists
// for debugging a program key by hand, which is why every key field carries a name.
func DescribeProgram(programInfo *ProgramInfo, caps *Caps) string {
	var key []uint32
	b := gpu.NewStringKeyBuilder(&key)
	genProgramKey(&b.KeyBuilder, programInfo, caps)
	b.Flush()
	return b.Description()
}
