// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// This file backs the bitmap→texture upload lane the FP catalog (dither LUT, gradient texture colorizer) needs: small
// uniquely-keyed textures created directly from CPU pixels. The pixels ride a lazy proxy whose callback creates the
// texture with initial data at flush, so instantiation is deferred until the proxy is actually needed. The image lanes
// will generalize this over imagecore bitmaps (mipmaps, color-type fallback).

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// FindOrCreatePixelsProxyView returns a uniquely-keyed, top-left-origin texture proxy view over the given pixels,
// creating (and keying) it if the context has not seen the key before. The pixels slice is retained until instantiation
// and must not be modified. Returns an invalid view when the format is unsupported or creation fails.
func FindOrCreatePixelsProxyView(ctx *DirectContext, key *gpu.UniqueKey, dims geom.ISize, colorType gpu.ColorType, pixels []byte, rowBytes int, label string) SurfaceProxyView {
	caps := ctx.GLCaps()
	format := caps.GetDefaultBackendFormat(colorType, false /* renderable */)
	if format == 0 {
		return SurfaceProxyView{}
	}
	swizzle := caps.ReadSwizzle(format, colorType)

	proxyProvider := ctx.ProxyProvider()
	if proxy := proxyProvider.FindOrCreateProxyByUniqueKey(key, UseAllocatorYes); proxy != nil {
		return MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, swizzle)
	}

	// A lazy proxy whose callback creates the texture with the pixel data, so instantiation is deferred until flush.
	data := pixels
	rb := rowBytes
	ct := colorType
	proxy := proxyProvider.CreateLazyProxy(
		func(rp *ResourceProvider, desc *LazySurfaceDesc) LazyCallbackResult {
			tex := rp.CreateTextureWithData(desc.Dimensions, desc.Format, desc.TextureType, ct,
				gpu.RenderableNo, 1, desc.Budgeted, gpu.MipmappedNo,
				[]gpu.MipLevel{{Pixels: data, RowBytes: rb}}, desc.Label)
			return MakeLazyCallbackResult(tex)
		},
		format, dims, gpu.MipmappedNo, gpu.MipmapStatusNotAllocated, gpu.SurfaceFlags(0),
		gpu.BackingFitExact, gpu.BudgetedYes, UseAllocatorYes, label,
	)
	if proxy == nil {
		return SurfaceProxyView{}
	}
	proxyProvider.AssignUniqueKeyToProxy(key, proxy.AsTextureProxy())
	return MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, swizzle)
}
