// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A surface proxy view pairs a surface proxy with the origin and swizzle with which it is viewed. Views are value
// types; an empty view has a nil proxy.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// SurfaceProxyView pairs a surface proxy with the origin and swizzle with which it is read.
type SurfaceProxyView struct {
	proxy   *SurfaceProxy
	origin  gpu.SurfaceOrigin
	swizzle gpu.Swizzle
}

// MakeSurfaceProxyView creates a view with an explicit origin and swizzle.
func MakeSurfaceProxyView(proxy *SurfaceProxy, origin gpu.SurfaceOrigin, swizzle gpu.Swizzle) SurfaceProxyView {
	return SurfaceProxyView{proxy: proxy, origin: origin, swizzle: swizzle}
}

// MakeSurfaceProxyViewDefault is the entry point used when origin and swizzle don't matter.
func MakeSurfaceProxyViewDefault(proxy *SurfaceProxy) SurfaceProxyView {
	return SurfaceProxyView{proxy: proxy, origin: gpu.OriginTopLeft, swizzle: gpu.SwizzleRGBA}
}

// IsValid reports whether the view has a non-nil proxy.
func (v *SurfaceProxyView) IsValid() bool { return v.proxy != nil }

// Equal reports whether two views reference the same proxy with the same origin and swizzle.
func (v *SurfaceProxyView) Equal(other *SurfaceProxyView) bool {
	return v.proxy == other.proxy && v.origin == other.origin && v.swizzle == other.swizzle
}

// Width returns the width of the underlying proxy.
func (v *SurfaceProxyView) Width() int32 { return v.proxy.Width() }

// Height returns the height of the underlying proxy.
func (v *SurfaceProxyView) Height() int32 { return v.proxy.Height() }

// Dimensions returns the size of the underlying proxy.
func (v *SurfaceProxyView) Dimensions() geom.ISize { return v.proxy.Dimensions() }

// Mipmapped reports the mipmap state of the underlying texture proxy, or MipmappedNo if the proxy isn't a texture.
func (v *SurfaceProxyView) Mipmapped() gpu.Mipmapped {
	if t := v.proxy.AsTextureProxy(); t != nil {
		return t.Mipmapped()
	}
	return gpu.MipmappedNo
}

// Proxy returns the underlying surface proxy.
func (v *SurfaceProxyView) Proxy() *SurfaceProxy { return v.proxy }

// AsTextureProxy returns the underlying proxy as a texture proxy, or nil if it isn't one.
func (v *SurfaceProxyView) AsTextureProxy() *TextureProxy {
	if v.proxy == nil {
		return nil
	}
	return v.proxy.AsTextureProxy()
}

// AsRenderTargetProxy returns the underlying proxy as a render target proxy, or nil if it isn't one.
func (v *SurfaceProxyView) AsRenderTargetProxy() *RenderTargetProxy {
	if v.proxy == nil {
		return nil
	}
	return v.proxy.AsRenderTargetProxy()
}

// Origin returns the view's surface origin.
func (v *SurfaceProxyView) Origin() gpu.SurfaceOrigin { return v.origin }

// Swizzle returns the view's swizzle.
func (v *SurfaceProxyView) Swizzle() gpu.Swizzle { return v.swizzle }

// ConcatSwizzle composes swizzle onto the view's existing swizzle.
func (v *SurfaceProxyView) ConcatSwizzle(swizzle gpu.Swizzle) {
	v.swizzle = gpu.ConcatSwizzles(v.swizzle, swizzle)
}

// MakeSwizzle returns a copy of the view with swizzle composed onto its existing swizzle.
func (v *SurfaceProxyView) MakeSwizzle(swizzle gpu.Swizzle) SurfaceProxyView {
	return SurfaceProxyView{
		proxy:   v.proxy,
		origin:  v.origin,
		swizzle: gpu.ConcatSwizzles(v.swizzle, swizzle),
	}
}

// Reset clears the view back to its zero value.
func (v *SurfaceProxyView) Reset() { *v = SurfaceProxyView{} }
