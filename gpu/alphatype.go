// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// AlphaType describes how pixel alpha relates to the color channels. The GPU layer needs it for sampler optimization
// flags and image/surface plumbing; the CPU raster package operates purely on premul buffers and never branches on it.

package gpu

// AlphaType describes how a pixel's alpha channel relates to its color channels.
type AlphaType int32

// AlphaType values.
const (
	// AlphaTypeUnknown: alpha handling has not been determined.
	AlphaTypeUnknown AlphaType = iota
	// AlphaTypeOpaque: all pixels are opaque.
	AlphaTypeOpaque
	// AlphaTypePremul: color components are premultiplied by alpha.
	AlphaTypePremul
	// AlphaTypeUnpremul: color components are not premultiplied by alpha.
	AlphaTypeUnpremul
)
