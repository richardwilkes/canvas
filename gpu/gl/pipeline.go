// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Pipeline is the immutable object holding everything needed to build a shader program and set API state for a draw —
// the fragment-processor list (color, paint coverage, clip coverage), the transfer processor, the dst proxy view for
// dst reads, flags, and the write swizzle. Window rectangles are not yet supported (pending a dedicated clip-stack
// type); the dst-input-attachment lane is Vulkan-only and reduces to false on GL.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// Pipeline input flags.
const (
	// PipelineInputFlagNone is the zero value: no special input flags.
	PipelineInputFlagNone uint8 = 0
	// PipelineConservativeRaster requests conservative rasterization.
	PipelineConservativeRaster uint8 = 1 << 1
	// PipelineWireframe requests wireframe rendering.
	PipelineWireframe uint8 = 1 << 2
	// PipelineSnapVerticesToPixelCenters snaps vertices to pixel centers (must be last: the private continuation flags
	// below are shifted up from it).
	PipelineSnapVerticesToPixelCenters uint8 = 1 << 3

	// The private continuation flags, packed into the same flags byte above the public input flags.
	pipelineHasStencilClip     uint8 = PipelineSnapVerticesToPixelCenters << 1
	pipelineScissorTestEnabled uint8 = PipelineSnapVerticesToPixelCenters << 2
)

// PipelineInitArgs bundles the arguments needed to construct a Pipeline.
type PipelineInitArgs struct {
	Caps         *Caps
	DstProxyView DstProxyView
	WriteSwizzle gpu.Swizzle
	InputFlags   uint8
}

// Pipeline is the immutable draw-time state for one draw: it can contain up to three fragment processors — color, paint
// coverage, and clip coverage — plus the transfer processor, dst proxy, flags, and write swizzle.
type Pipeline struct {
	xferProcessor      XferProcessor // nil means the shared simple src-over XP
	fragmentProcessors []FragmentProcessor
	dstProxy           DstProxyView
	// numColorProcessors is also the index in fragmentProcessors where coverage FPs begin.
	numColorProcessors int
	writeSwizzle       gpu.Swizzle
	flags              uint8
}

// NewPipelineFromBlendMode builds a simple pipeline with default settings and no processors, deriving its transfer
// processor from blend. The blend mode must be a coefficient mode.
func NewPipelineFromBlendMode(scissorEnabled bool, blend raster.BlendMode, writeSwizzle gpu.Swizzle, inputFlags uint8) *Pipeline {
	return NewPipelineFromXP(scissorEnabled, MakeNoCoverageXP(blend), writeSwizzle, inputFlags)
}

// NewPipelineFromXP builds a pipeline directly from an explicit transfer processor.
func NewPipelineFromXP(scissorEnabled bool, xp XferProcessor, writeSwizzle gpu.Swizzle, inputFlags uint8) *Pipeline {
	p := &Pipeline{
		xferProcessor: xp,
		writeSwizzle:  writeSwizzle,
		flags:         inputFlags,
	}
	if scissorEnabled {
		p.flags |= pipelineScissorTestEnabled
	}
	return p
}

// NewPipelineHardClip builds a pipeline from InitArgs, an explicit transfer processor, and an optional hard clip.
func NewPipelineHardClip(args *PipelineInitArgs, xp XferProcessor, hardClip *AppliedClip) *Pipeline {
	// Borrow the shell from the free list (programinfopool.go). The borrowed shell is zero except for a preserved
	// length-0 fragmentProcessors backing, so set each field explicitly rather than struct-literal-assigning *p, which
	// would drop that backing. numColorProcessors stays 0 (NewPipeline sets it when it appends a color FP).
	p := borrowPipeline()
	p.dstProxy = args.DstProxyView
	p.xferProcessor = xp
	p.writeSwizzle = args.WriteSwizzle
	p.flags = args.InputFlags
	if hardClip != nil {
		if hardClip.HasStencilClip() {
			p.flags |= pipelineHasStencilClip
		}
		if hardClip.ScissorState().Enabled() {
			p.flags |= pipelineScissorTestEnabled
		}
	}
	// If we have any special dst sample flags we better also have a dst proxy.
	if p.DstSampleFlags() != 0 && !p.dstProxy.ProxyView().IsValid() {
		panic("dst sample flags without a dst proxy")
	}
	return p
}

// NewPipeline builds a pipeline from InitArgs, a finalized processor set, and an optional applied clip, moving the
// processor set's fragment processors (and any clip coverage FP) into the pipeline.
func NewPipeline(args *PipelineInitArgs, processors *ProcessorSet, appliedClip *AppliedClip) *Pipeline {
	if !processors.IsFinalized() {
		panic("processor set must be finalized")
	}
	p := NewPipelineHardClip(args, processors.XferProcessor(), appliedClip)

	// Copy the FPs from the processor set to the pipeline.
	if processors.HasColorFragmentProcessor() {
		p.numColorProcessors = 1
		p.fragmentProcessors = append(p.fragmentProcessors,
			processors.detachColorFragmentProcessor())
	}
	if processors.HasCoverageFragmentProcessor() {
		p.fragmentProcessors = append(p.fragmentProcessors,
			processors.detachCoverageFragmentProcessor())
	}
	if appliedClip != nil && appliedClip.HasCoverageFragmentProcessor() {
		p.fragmentProcessors = append(p.fragmentProcessors,
			appliedClip.DetachCoverageFragmentProcessor())
	}
	return p
}

// NumFragmentProcessors returns the total number of fragment processors (color + coverage) attached to this pipeline.
func (p *Pipeline) NumFragmentProcessors() int { return len(p.fragmentProcessors) }

// NumColorFragmentProcessors returns the number of leading fragment processors that contribute color (as opposed to
// coverage).
func (p *Pipeline) NumColorFragmentProcessors() int { return p.numColorProcessors }

// IsColorFragmentProcessor reports whether the fragment processor at idx contributes color.
func (p *Pipeline) IsColorFragmentProcessor(idx int) bool { return idx < p.numColorProcessors }

// IsCoverageFragmentProcessor reports whether the fragment processor at idx contributes coverage.
func (p *Pipeline) IsCoverageFragmentProcessor(idx int) bool { return idx >= p.numColorProcessors }

// FragmentProcessor returns the fragment processor at idx.
func (p *Pipeline) FragmentProcessor(idx int) FragmentProcessor {
	return p.fragmentProcessors[idx]
}

// UsesLocalCoords reports whether any fragment processor samples using local coordinates; the sample coords for the
// top-level FPs are implicitly the geometry processor's local coords.
func (p *Pipeline) UsesLocalCoords() bool {
	for _, fp := range p.fragmentProcessors {
		if fp.fpBase().UsesSampleCoords() {
			return true
		}
	}
	return false
}

// XferProcessor returns the pipeline's transfer processor; a nil member means the common src-over case.
func (p *Pipeline) XferProcessor() XferProcessor {
	if p.xferProcessor != nil {
		return p.xferProcessor
	}
	return SimpleSrcOverXP()
}

// UsesDstTexture reports whether the pipeline reads the destination through a texture (as opposed to an input
// attachment).
func (p *Pipeline) UsesDstTexture() bool {
	return p.dstProxy.ProxyView().IsValid() && !p.UsesDstInputAttachment()
}

// UsesDstInputAttachment reports whether the pipeline reads the destination through an input attachment. The
// input-attachment lane is Vulkan-only and was dropped from gpu.DstSampleFlags, so this is constant false on GL.
func (p *Pipeline) UsesDstInputAttachment() bool { return false }

// DstProxyView returns the destination proxy view. If the transfer processor does not read the dst then the view's
// proxy is nil.
func (p *Pipeline) DstProxyView() *SurfaceProxyView { return p.dstProxy.ProxyView() }

// DstTextureOffset returns the offset of the destination-read texture within its proxy.
func (p *Pipeline) DstTextureOffset() geom.IPoint { return p.dstProxy.Offset() }

// DstSampleFlags returns the destination-sampling flags recorded on the dst proxy view.
func (p *Pipeline) DstSampleFlags() gpu.DstSampleFlags { return p.dstProxy.DstSampleFlags() }

// PeekDstTexture returns the texture used to access the dst color, if any.
func (p *Pipeline) PeekDstTexture() *Texture {
	if !p.UsesDstTexture() {
		return nil
	}
	if dstProxy := p.DstProxyView().AsTextureProxy(); dstProxy != nil {
		return dstProxy.Proxy().PeekTexture()
	}
	return nil
}

// IsScissorTestEnabled reports whether the scissor test is enabled for this draw.
func (p *Pipeline) IsScissorTestEnabled() bool { return p.flags&pipelineScissorTestEnabled != 0 }

// UsesConservativeRaster reports whether conservative rasterization is requested.
func (p *Pipeline) UsesConservativeRaster() bool { return p.flags&PipelineConservativeRaster != 0 }

// IsWireframe reports whether wireframe rendering is requested.
func (p *Pipeline) IsWireframe() bool { return p.flags&PipelineWireframe != 0 }

// SnapVerticesToPixelCenters reports whether vertices should be snapped to pixel centers.
func (p *Pipeline) SnapVerticesToPixelCenters() bool {
	return p.flags&PipelineSnapVerticesToPixelCenters != 0
}

// HasStencilClip reports whether the applied clip contributes a stencil clip.
func (p *Pipeline) HasStencilClip() bool { return p.flags&pipelineHasStencilClip != 0 }

// XferBarrierType reports what barrier, if any, must be issued before this draw given caps.
func (p *Pipeline) XferBarrierType(caps *gpu.Caps) XferBarrierType {
	if p.DstSampleFlags()&gpu.DstSampleFlagRequiresTextureBarrier != 0 {
		return XferBarrierTexture
	}
	return p.XferProcessor().XferBarrierType(caps)
}

// WriteSwizzle returns the swizzle applied to color values written to the destination.
func (p *Pipeline) WriteSwizzle() gpu.Swizzle { return p.writeSwizzle }

// VisitTextureEffects calls f for every texture effect among the pipeline's fragment processors.
func (p *Pipeline) VisitTextureEffects(f func(*TextureEffect)) {
	for _, fp := range p.fragmentProcessors {
		fpVisitTextureEffects(fp, f)
	}
}

// VisitProxies calls f for every proxy referenced by the pipeline's fragment processors (including any clip coverage
// FPs) and, if present, the dst-read proxy.
func (p *Pipeline) VisitProxies(f func(*SurfaceProxy, gpu.Mipmapped)) {
	for _, fp := range p.fragmentProcessors {
		fpVisitProxies(fp, f)
	}
	if p.UsesDstTexture() {
		f(p.DstProxyView().Proxy(), gpu.MipmappedNo)
	}
}

// Upstream's GrPipeline::genKey (pipeline flags plus the resolved blend state) is not ported: only the Vulkan, Metal
// and D3D caps call it, to key their pipeline-state objects. The GL program key covers the same ground through
// genXPKey and the snap-vertices/write-swizzle bits genProgramKey adds directly.

// SetDstTextureUniforms sets the uniforms a shader needs to sample the destination-read texture, if the program
// declared one.
func (p *Pipeline) SetDstTextureUniforms(pdman *ProgramDataManager, builtin *BuiltinUniformHandles) {
	dstTexture := p.PeekDstTexture()
	if dstTexture == nil {
		if builtin.DstTextureCoordsUni.IsValid() {
			panic("dst-texture uniform without a dst texture")
		}
		return
	}
	if builtin.DstTextureCoordsUni.IsValid() {
		scaleX := float32(1)
		scaleY := float32(1)
		if dstTexture.TextureType() == gpu.TextureTypeRectangle {
			// For rectangle textures the scaleX component stores the height in case the coords must be flipped for a
			// bottom-left origin.
			scaleX = float32(dstTexture.Surface().Height())
		} else {
			scaleX /= float32(dstTexture.Surface().Width())
			scaleY /= float32(dstTexture.Surface().Height())
		}
		pdman.Set4f(builtin.DstTextureCoordsUni,
			float32(p.DstTextureOffset().X),
			float32(p.DstTextureOffset().Y),
			scaleX, scaleY)
	}
}
