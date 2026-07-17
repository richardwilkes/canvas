// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The draw-support types the OpsTask machinery stores and compares: DstProxyView. AppliedClip lives in appliedclip.go,
// the processor-set analysis in processorset.go, and the user stencil settings in stencilsettings.go.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// DstProxyView is the proxy view (and offset) a draw samples its destination from when dst reads are required.
type DstProxyView struct {
	proxyView      SurfaceProxyView
	offset         geom.IPoint
	dstSampleFlags gpu.DstSampleFlags
}

// Equal reports whether d and that describe the same destination sample source.
func (d *DstProxyView) Equal(that *DstProxyView) bool {
	return d.proxyView.Equal(&that.proxyView) && d.offset == that.offset &&
		d.dstSampleFlags == that.dstSampleFlags
}

// Proxy returns the destination proxy, if any.
func (d *DstProxyView) Proxy() *SurfaceProxy { return d.proxyView.Proxy() }

// ProxyView returns the destination proxy view.
func (d *DstProxyView) ProxyView() *SurfaceProxyView { return &d.proxyView }

// Offset returns the offset within the destination proxy.
func (d *DstProxyView) Offset() geom.IPoint { return d.offset }

// SetOffset sets the offset within the destination proxy.
func (d *DstProxyView) SetOffset(offset geom.IPoint) { d.offset = offset }

// SetProxyView sets the destination proxy view, clearing the offset if it is invalid.
func (d *DstProxyView) SetProxyView(view SurfaceProxyView) {
	d.proxyView = view
	if d.proxyView.Proxy() == nil {
		d.offset = geom.IPoint{}
	}
}

// DstSampleFlags returns how the destination is sampled.
func (d *DstProxyView) DstSampleFlags() gpu.DstSampleFlags { return d.dstSampleFlags }

// SetDstSampleFlags sets how the destination is sampled.
func (d *DstProxyView) SetDstSampleFlags(flags gpu.DstSampleFlags) { d.dstSampleFlags = flags }
