// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// gradientBitmapCache implements the 256×1 LUT fallback for gradients whose fixed stops exceed the looping colorizer's
// 128-color limit. The ramp is baked by evaluating the existing CPU gradient stages (shaders.EvalGradientRamp), so the
// GPU LUT and a CPU render of the same gradient agree by construction. Design notes: the cache hangs off each
// DirectContext (one per context, freed with it) instead of being a single cache shared across contexts, and the cache
// key reduces to colors + interior positions + color type (no interpolation/dst-color-space plumbing — sRGB only — and
// alphaType reduces to the end-premul convention: the LUT bakes the final premultiplication directly whenever any stop
// is translucent, which is derivable from the colors already in the key).

package gl

import (
	"math"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

// Cache geometry: each entry costs 1K or 2K of RAM — a 256×1 bitmap at 32bpp or 64bpp.
const (
	maxNumCachedGradientBitmaps = 32
	gradientTextureSize         = 256
)

// gradientRampKeyDomain keys the baked ramp textures in the per-context proxy cache; the counter gives each baked ramp
// a fresh id (a re-baked ramp gets a fresh id, so an evicted entry's texture is never aliased).
var (
	gradientRampKeyDomain = gpu.GenerateUniqueKeyDomain()
	gradientRampNextID    atomic.Uint32
)

// gradientBitmapEntry is one baked ramp: the LUT pixel bytes plus the unique key its texture proxy is cached under.
type gradientBitmapEntry struct {
	proxyKey  gpu.UniqueKey
	key       []uint32
	pixels    []byte
	rowBytes  int
	colorType gpu.ColorType
}

// gradientBitmapCache is an LRU (cap 32) of baked gradient ramps, searched linearly. The mutex is cheap, and it keeps
// the per-context placement safe even if a context's conversion paths are ever driven from more than one goroutine.
type gradientBitmapCache struct {
	entries    []*gradientBitmapEntry // most recently used first
	maxEntries int
	resolution int
	mu         sync.Mutex
}

func newGradientBitmapCache(maxEntries, resolution int) *gradientBitmapCache {
	return &gradientBitmapCache{maxEntries: maxEntries, resolution: resolution}
}

// buildGradientRampKey builds getGradient's key layout — count + colors + interior positions (pos[0] == 0 and
// pos[count-1] == 1 are guaranteed by the stop fixing) — with the sRGB-only reduction: alphaType, interpolation, and
// dstColorSpace have no equivalents here (alphaType is implied by the colors' opacity under the end-premul convention,
// since colorsAreOpaque is redundant with the colors).
func buildGradientRampKey(colors []colorcore.Color4f, positions []float32, colorType gpu.ColorType) []uint32 {
	count := len(colors)
	key := make([]uint32, 0, 1+4*count+(count-2)+1)
	key = append(key, uint32(count))
	for _, c := range colors {
		key = append(key, math.Float32bits(c.R), math.Float32bits(c.G), math.Float32bits(c.B),
			math.Float32bits(c.A))
	}
	for i := 1; i < count-1; i++ {
		key = append(key, math.Float32bits(positions[i]))
	}
	key = append(key, uint32(colorType))
	return key
}

// getGradient returns the cached ramp for this gradient configuration, baking (and inserting, evicting the LRU tail at
// capacity) on a miss. premul is the end-premul spelling of the alphaType parameter (pass !allOpaque); it is derivable
// from colors, so it rides outside the key.
func (c *gradientBitmapCache) getGradient(colors []colorcore.Color4f, positions []float32, premul bool, colorType gpu.ColorType) *gradientBitmapEntry {
	key := buildGradientRampKey(colors, positions, colorType)
	c.mu.Lock()
	defer c.mu.Unlock()
	if entry := c.find(key); entry != nil {
		return entry
	}
	entry := &gradientBitmapEntry{
		key:       key,
		colorType: colorType,
		rowBytes:  c.resolution * colorType.BytesPerPixel(),
	}
	entry.pixels = fillGradientRamp(colors, positions, premul, colorType, c.resolution)
	builder := gpu.UniqueKeyBuilder(&entry.proxyKey, gradientRampKeyDomain, 1, "GradientRamp")
	builder.Slice()[0] = gradientRampNextID.Add(1)
	builder.Finish()
	c.add(entry)
	return entry
}

// find performs a linear search; a hit moves the entry to the head so it is purged last.
func (c *gradientBitmapCache) find(key []uint32) *gradientBitmapEntry {
	for i, entry := range c.entries {
		if slices.Equal(entry.key, key) {
			copy(c.entries[1:], c.entries[:i])
			c.entries[0] = entry
			return entry
		}
	}
	return nil
}

// add evicts the tail at capacity, then attaches entry to the head.
func (c *gradientBitmapCache) add(entry *gradientBitmapEntry) {
	if len(c.entries) == c.maxEntries {
		c.entries = c.entries[:len(c.entries)-1]
	}
	c.entries = append(c.entries, nil)
	copy(c.entries[1:], c.entries)
	c.entries[0] = entry
}

// fillGradientRamp evaluates the CPU gradient stages across the ramp (shaders.EvalGradientRamp — seed + 1/width scale +
// fill stages + the end premul) and stores to the LUT color type: 8888 through round-to-nearest-even unorm conversion,
// F16 as raw halves (no clamping).
func fillGradientRamp(colors []colorcore.Color4f, positions []float32, premul bool, colorType gpu.ColorType, resolution int) []byte {
	ramp := make([]colorcore.PMColor4f, resolution)
	shaders.EvalGradientRamp(colors, positions, premul, ramp)
	out := make([]byte, resolution*colorType.BytesPerPixel())
	switch colorType {
	case gpu.ColorTypeRGBA8888:
		for i, c := range ramp {
			o := i * 4
			out[o] = rampToUnormByte(c.R)
			out[o+1] = rampToUnormByte(c.G)
			out[o+2] = rampToUnormByte(c.B)
			out[o+3] = rampToUnormByte(c.A)
		}
	case gpu.ColorTypeRGBAF16:
		for i, c := range ramp {
			o := i * 8
			putHalfLE(out[o:], c.R)
			putHalfLE(out[o+2:], c.G)
			putHalfLE(out[o+4:], c.B)
			putHalfLE(out[o+6:], c.A)
		}
	default:
		panic("unsupported gradient LUT color type")
	}
	return out
}

// rampToUnormByte converts a float channel value to a unorm byte: round(min(max(0, v*255), 255)) with
// round-to-nearest-even, NaN to 0.
func rampToUnormByte(v float32) byte {
	f := v * 255
	if !(f > 0) { // also catches NaN
		return 0
	}
	if f > 255 {
		f = 255
	}
	return byte(math.RoundToEven(float64(f)))
}

// putHalfLE stores one channel as a little-endian IEEE binary16.
func putHalfLE(dst []byte, v float32) {
	h := imagecore.FloatToHalf(v)
	dst[0] = byte(h)
	dst[1] = byte(h >> 8)
}

// colorTypeIsWiderThan8 reports whether any channel of ct carries more than 8 bits.
func colorTypeIsWiderThan8(ct gpu.ColorType) bool {
	switch ct {
	case gpu.ColorTypeRGBA1010102, gpu.ColorTypeBGRA1010102, gpu.ColorTypeRGB101010x,
		gpu.ColorTypeRGBA10x6, gpu.ColorTypeAlphaF16, gpu.ColorTypeRGBAF16, gpu.ColorTypeRGBF16F16F16x,
		gpu.ColorTypeRGBAF16Clamped, gpu.ColorTypeRGBAF32, gpu.ColorTypeAlpha16, gpu.ColorTypeRG1616,
		gpu.ColorTypeRGF16, gpu.ColorTypeRGBA16161616, gpu.ColorTypeAlphaF32xxx, gpu.ColorTypeR16,
		gpu.ColorTypeRF16, gpu.ColorTypeGrayF16:
		return true
	default:
		return false
	}
}

// makeTexturedColorizer bakes the gradient into a 256×1 LUT through the per-context cache, uploads it via the
// pixels-proxy lane, and samples it with clamp wrap + linear filtering through a texture effect whose local matrix
// scales by (width, 1). The LUT is 8888, or F16 when the destination is wider than 8 bits per channel and an F16
// backend format exists. The LUT directly encodes the interpolated-to-dst result — including the final
// premultiplication when any stop is translucent — so the caller must not wrap the returned FP in the ColorXformFP
// premul step the analytic colorizers get. Returns nil only when texture creation fails, which skips the draw.
func makeTexturedColorizer(colors []colorcore.Color4f, positions []float32, allOpaque bool, args *FPArgs) FragmentProcessor {
	colorType := gpu.ColorTypeRGBA8888
	if colorTypeIsWiderThan8(args.DstColorType) &&
		args.Caps.GetDefaultBackendFormat(gpu.ColorTypeRGBAF16, false /* renderable */) != 0 {
		colorType = gpu.ColorTypeRGBAF16
	}
	entry := args.Ctx.gradientRampCache.getGradient(colors, positions, !allOpaque, colorType)
	view := FindOrCreatePixelsProxyView(args.Ctx, &entry.proxyKey,
		geom.ISize{Width: gradientTextureSize, Height: 1}, colorType, entry.pixels, entry.rowBytes,
		"MakeTexturedColorizer")
	if !view.IsValid() {
		return nil
	}
	m := geom.ScaleMatrix(gradientTextureSize, 1)
	// The texture is always labeled unpremul (under the legacy-sRGB interpolation), never opaque, so sampler-opt-flag
	// modulation contributes no PreservesOpaqueInput either way; premul here is the end-premul spelling of the same
	// label.
	return MakeTextureEffectSimple(view, gpu.AlphaTypePremul, m, gpu.FilterModeLinear, gpu.MipmapModeNone)
}
