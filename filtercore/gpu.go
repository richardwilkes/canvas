// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The staging query: which filter DAGs the GPU backend may evaluate GPU-natively. A DAG is GPU-evaluable only when
// every node's evaluation builds shaders with fragment-processor conversions in the catalog (gpu/gl's MakeShaderFP).
// Nodes advertise that via GPUEvaluable; the GPU backend routes everything else through the readback->CPU->upload
// fallback with its visibility counter.

package filtercore

// GPUEvaluable is implemented by filters whose OnFilterImage builds only shader forms with GPU fragment-processor
// conversions (plus the filtercore-internal image/color/color-filter/local-matrix/ filter-decal shaders, which all
// convert). A node without the method is treated as CPU-only.
type GPUEvaluable interface {
	// GPUEvaluable reports whether this node (ignoring its inputs) evaluates GPU-natively.
	GPUEvaluable() bool
}

// CanEvaluateOnGPU reports whether the whole DAG rooted at f can evaluate on the GPU backend: every node advertises
// GPUEvaluable and every input recursively qualifies (nil inputs are the dynamic source image, which the GPU backend
// snaps without readback). A nil filter is trivially GPU-evaluable (the layer restore draws the source through
// FilterResult with no filter nodes).
func CanEvaluateOnGPU(f Filter) bool {
	if f == nil {
		return true
	}
	ge, ok := f.(GPUEvaluable)
	if !ok || !ge.GPUEvaluable() {
		return false
	}
	for _, in := range f.Base().inputs {
		if !CanEvaluateOnGPU(in) {
			return false
		}
	}
	return true
}
