// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// ProgramInfo is the bundle handed to program building and GL state flushing for one draw — target
// format/origin/samples, pipeline, geometry processor, stencil settings, and primitive type.

package gl

import (
	"github.com/richardwilkes/canvas/gpu"
)

// ProgramInfo bundles everything program building and GL state flushing need for one draw.
type ProgramInfo struct {
	geomProc            GeometryProcessor
	pipeline            *Pipeline
	userStencilSettings *UserStencilSettings
	numSamples          int
	targetsNumSamples   int
	backendFormat       Format
	origin              gpu.SurfaceOrigin
	renderPassBarriers  gpu.XferBarrierFlags
	colorLoadOp         gpu.LoadOp
	primitiveType       gpu.PrimitiveType
	needsStencil        bool
}

// NewProgramInfo builds a ProgramInfo for a draw targeting targetView.
func NewProgramInfo(caps *Caps, targetView *SurfaceProxyView, usesMSAASurface bool, pipeline *Pipeline, userStencilSettings *UserStencilSettings, geomProc GeometryProcessor, primitiveType gpu.PrimitiveType, renderPassXferBarriers gpu.XferBarrierFlags, colorLoadOp gpu.LoadOp) *ProgramInfo {
	rtProxy := targetView.AsRenderTargetProxy()
	if rtProxy == nil {
		panic("program info requires a render target proxy")
	}
	// Borrow the shell from the free list (programinfopool.go) and overwrite every field — the struct-literal
	// assignment covers all but numSamples, set just below.
	info := borrowProgramInfo()
	*info = ProgramInfo{
		pipeline:            pipeline,
		userStencilSettings: userStencilSettings,
		geomProc:            geomProc,
		backendFormat:       targetView.Proxy().Format(),
		origin:              targetView.Origin(),
		primitiveType:       primitiveType,
		renderPassBarriers:  renderPassXferBarriers,
		colorLoadOp:         colorLoadOp,
		targetsNumSamples:   rtProxy.NumSamples(),
		needsStencil:        rtProxy.NeedsStencil(),
	}
	if info.targetsNumSamples <= 0 {
		panic("target must have a positive sample count")
	}
	info.numSamples = info.targetsNumSamples
	if info.numSamples == 1 && usesMSAASurface {
		info.numSamples = caps.InternalMultisampleCount(info.backendFormat)
	}
	return info
}

// NumSamples returns the sample count actually used for this draw (may exceed TargetsNumSamples when the target is
// upgraded to MSAA).
func (i *ProgramInfo) NumSamples() int { return i.numSamples }

// NeedsStencil reports whether the render target needs a stencil buffer.
func (i *ProgramInfo) NeedsStencil() bool { return i.needsStencil }

// IsStencilEnabled reports whether this draw uses the stencil buffer, either through explicit user stencil settings or
// a pipeline-applied stencil clip.
func (i *ProgramInfo) IsStencilEnabled() bool {
	return i.userStencilSettings != UnusedStencilSettings() || i.pipeline.HasStencilClip()
}

// UserStencilSettings returns the explicit stencil settings for this draw.
func (i *ProgramInfo) UserStencilSettings() *UserStencilSettings { return i.userStencilSettings }

// BackendFormat returns the format of the destination render target proxy.
func (i *ProgramInfo) BackendFormat() Format { return i.backendFormat }

// Origin returns the destination render target's surface origin.
func (i *ProgramInfo) Origin() gpu.SurfaceOrigin { return i.origin }

// Pipeline returns the draw's pipeline.
func (i *ProgramInfo) Pipeline() *Pipeline { return i.pipeline }

// GeomProc returns the draw's geometry processor.
func (i *ProgramInfo) GeomProc() GeometryProcessor { return i.geomProc }

// PrimitiveType returns the primitive type this draw issues.
func (i *ProgramInfo) PrimitiveType() gpu.PrimitiveType { return i.primitiveType }

// TargetsNumSamples returns the render target's own sample count, before any MSAA upgrade.
func (i *ProgramInfo) TargetsNumSamples() int { return i.targetsNumSamples }

// RenderPassBarriers returns the transfer barriers required for this render pass.
func (i *ProgramInfo) RenderPassBarriers() gpu.XferBarrierFlags { return i.renderPassBarriers }

// ColorLoadOp returns how the render pass should load the destination's existing color.
func (i *ProgramInfo) ColorLoadOp() gpu.LoadOp { return i.colorLoadOp }

// VisitFPProxies calls f for every proxy referenced by the pipeline's fragment processors.
func (i *ProgramInfo) VisitFPProxies(f func(*SurfaceProxy, gpu.Mipmapped)) {
	i.pipeline.VisitProxies(f)
}
