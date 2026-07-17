// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The PathRenderer interface and PathRendererChain: the base model for drawing paths into an OpsTask and the ordered
// chain that selects the best renderer for a given shape/style/AA combination. Chain membership, in construction order:
// AAConvexPathRenderer → AAHairLinePathRenderer → AALinearizingConvexPathRenderer → TriangulatingPathRenderer →
// TessellationPathRenderer → DefaultPathRenderer (always appended), plus the SoftwarePathRenderer fallback held by the
// drawing manager. DashLinePathRenderer's slot is unreachable while styles carry no path effect, and the Atlas/Small
// members are trimmed (the atlas member is an optional DMSAA atlasing optimization, and the small-path member needs the
// SmallPathAtlasMgr/shape-cache machinery; its distance-field geometry processor exists but has no consumer until that
// renderer lands).

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/surface"
)

// pathRendererMaxVerbs guards TriangulatingPathRenderer and DefaultPathRenderer from excessively complex paths.
const pathRendererMaxVerbs = 1 << 14

// StencilSupport is the level of support a path renderer has for stenciling paths.
type StencilSupport uint8

// StencilSupport values, in ascending order of capability.
const (
	// StencilSupportNone: this path renderer cannot be used to stencil the path.
	StencilSupportNone StencilSupport = iota
	// StencilSupportStencilOnly: the renderer cannot apply arbitrary stencil rules nor shade and stencil
	// simultaneously, but it supports StencilPath (no color writes, non-zero stencil values written for pixels covered
	// by the path).
	StencilSupportStencilOnly
	// StencilSupportNoRestriction: the caller passes a paint and calls DrawPath; the path is rendered exactly as the
	// draw state indicates, including simultaneous color and stenciling with arbitrary rules.
	StencilSupportNoRestriction
)

// CanDrawPath reports how well a path renderer can draw a given path.
type CanDrawPath uint8

// CanDrawPath values, in ascending order of confidence.
const (
	CanDrawPathNo CanDrawPath = iota
	// CanDrawPathAsBackup: this renderer is better than the SW fallback if no others can draw the path.
	CanDrawPathAsBackup
	CanDrawPathYes
)

// CanDrawPathArgs bundles the inputs OnCanDrawPath needs to decide whether a renderer can handle a shape. SurfaceProps'
// chain-side consumer is TriangulatingPathRenderer's DMSAA rejection (an atlas-path renderer would be another consumer,
// but that renderer stays trimmed here).
type CanDrawPathArgs struct {
	Caps                   *Caps
	Proxy                  *RenderTargetProxy
	ViewMatrix             *geom.Matrix
	Shape                  *StyledShape
	Paint                  *Paint
	ClipConservativeBounds geom.IRect
	SurfaceProps           surface.Props
	AAType                 gpu.AAType
	// HasUserStencilSettings is only used by TessellationPathRenderer.
	HasUserStencilSettings bool
}

// DrawPathArgs bundles the inputs OnDrawPath needs to draw a path (a gamma-correct flag is trimmed: all rendering stays
// in non-linear sRGB, and no in-scope renderer consumes it).
type DrawPathArgs struct {
	Clip                   Clip
	Context                *DirectContext
	Paint                  *Paint
	UserStencilSettings    *UserStencilSettings
	SurfaceDrawContext     *SurfaceDrawContext
	ViewMatrix             *geom.Matrix
	Shape                  *StyledShape
	ClipConservativeBounds geom.IRect
	AAType                 gpu.AAType
}

// StencilPathArgs bundles the inputs OnStencilPath needs. AAType cannot be AATypeCoverage.
type StencilPathArgs struct {
	Clip                   Clip
	Context                *DirectContext
	SurfaceDrawContext     *SurfaceDrawContext
	ViewMatrix             *geom.Matrix
	Shape                  *StyledShape
	ClipConservativeBounds geom.IRect
	DoStencilMSAA          gpu.AA
}

// PathRenderer is the interface every path-drawing strategy implements. The exported wrappers (PathRendererDrawPath,
// PathRendererStencilPath, PathRendererGetStencilSupport) provide the public entry points and their precondition checks
// around the interface methods below.
type PathRenderer interface {
	// Name identifies the renderer for logging/diagnostics.
	Name() string
	// OnGetStencilSupport reports the level of stenciling this renderer supports. The shape must be simple fill styled
	// and non-inverse filled.
	OnGetStencilSupport(shape *StyledShape) StencilSupport
	// OnCanDrawPath reports how well this renderer can render the given path. CanDrawPathNo/CanDrawPathAsBackup let the
	// caller keep searching for a better renderer.
	OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath
	// OnDrawPath draws the path. If OnGetStencilSupport would return StencilSupportNoRestriction then the
	// implementation must respect the stencil settings.
	OnDrawPath(args *DrawPathArgs) bool
	// OnStencilPath draws the path to the stencil buffer (writable bits already zeroed; pixels inside the path get
	// non-zero values). Must be overridden iff OnGetStencilSupport ever returns StencilSupportStencilOnly;
	// StencilSupportNoRestriction renderers may use PathRendererDefaultStencilPath.
	OnStencilPath(args *StencilPathArgs)
}

// PathRendererGetStencilSupport calls OnGetStencilSupport after checking its precondition: the path's fill must not be
// an inverse type; the path will always be filled and not stroked.
func PathRendererGetStencilSupport(pr PathRenderer, shape *StyledShape) StencilSupport {
	if !shape.Style().IsSimpleFill() || shape.InverseFilled() {
		panic("getStencilSupport requires a simple-fill, non-inverse shape")
	}
	return pr.OnGetStencilSupport(shape)
}

// PathRendererDrawPath draws the path into the draw target, checking that any user stencil settings are only used where
// the renderer supports them.
func PathRendererDrawPath(pr PathRenderer, args *DrawPathArgs) bool {
	if !args.UserStencilSettings.IsUnused() {
		if !args.Shape.Style().IsSimpleFill() ||
			PathRendererGetStencilSupport(pr, args.Shape) != StencilSupportNoRestriction {
			panic("user stencil settings require a simple fill and kNoRestriction support")
		}
	}
	return pr.OnDrawPath(args)
}

// PathRendererStencilPath calls OnStencilPath after checking the renderer supports stenciling.
func PathRendererStencilPath(pr PathRenderer, args *StencilPathArgs) {
	if PathRendererGetStencilSupport(pr, args.Shape) == StencilSupportNone {
		panic("stencilPath on a renderer without stencil support")
	}
	pr.OnStencilPath(args)
}

// pathRendererIncrementStencil is the stencil settings PathRendererDefaultStencilPath uses: always pass the stencil
// test and replace the stencil value on a pass.
var pathRendererIncrementStencil = func() *UserStencilSettings {
	s := MakeUserStencilSettings(0xffff, UserStencilTestAlways, 0xffff, UserStencilOpReplace,
		UserStencilOpReplace, 0xffff)
	return &s
}()

// PathRendererDefaultStencilPath is the default OnStencilPath implementation: renderers with
// StencilSupportNoRestriction support stencil by drawing with a no-color paint and replace-on-pass stencil settings.
func PathRendererDefaultStencilPath(pr PathRenderer, args *StencilPathArgs) {
	paint := NewPaint()
	aaType := gpu.AATypeNone
	if args.DoStencilMSAA == gpu.AAYes {
		aaType = gpu.AATypeMSAA
	}
	drawArgs := DrawPathArgs{
		Context:                args.Context,
		Paint:                  paint,
		UserStencilSettings:    pathRendererIncrementStencil,
		SurfaceDrawContext:     args.SurfaceDrawContext,
		Clip:                   nil,
		ClipConservativeBounds: args.ClipConservativeBounds,
		ViewMatrix:             args.ViewMatrix,
		Shape:                  args.Shape,
		AAType:                 aaType,
	}
	pr.OnDrawPath(&drawArgs)
}

// getPathDevBounds computes a path's device-space bounds: inverse filled paths get bounds set by devSize; non-inverse
// path bounds are not necessarily clipped to devSize.
func getPathDevBounds(p interface {
	IsInverseFillType() bool
	Bounds() geom.Rect
}, devSize geom.ISize, matrix *geom.Matrix,
) geom.Rect {
	if p.IsInverseFillType() {
		return geom.RectWH(float32(devSize.Width), float32(devSize.Height))
	}
	bounds, _ := matrix.MapRect(p.Bounds())
	return bounds
}

//////////////////////////////////////////////////////////////////////////////
// PathRendererChain

// PathRendererChainOptions configures which renderers NewPathRendererChain includes and whether path mask caching is
// allowed.
type PathRendererChainOptions struct {
	AllowPathMaskCaching bool
	PathRenderers        gpu.PathRenderers
}

// ChainDrawType describes what kind of draw GetPathRenderer is selecting a renderer for.
type ChainDrawType uint8

// ChainDrawType values, in ascending order of stencil requirement.
const (
	ChainDrawTypeColor ChainDrawType = iota
	ChainDrawTypeStencil
	ChainDrawTypeStencilAndColor
)

// PathRendererChain keeps the renderers in a priority order and selects the best one for a given request.
type PathRendererChain struct {
	// tessellationPathRenderer is a direct pointer for GetTessellationPathRenderer, since the drawing manager queries
	// it through a narrower interface than PathRenderer.
	tessellationPathRenderer *TessellationPathRenderer
	chain                    []PathRenderer
}

// NewPathRendererChain builds the chain in priority order. The DashLinePathRenderer's option gate is preserved so the
// construction order stays consistent, but the member itself is unreachable while styles carry no path effect. Atlas
// (an optional DMSAA atlasing optimization) and Small (needs the SmallPathAtlasMgr/shape-cache machinery) are trimmed.
func NewPathRendererChain(ctx *DirectContext, options PathRendererChainOptions) *PathRendererChain {
	caps := ctx.GLCaps()
	c := &PathRendererChain{}
	// if options.PathRenderers&gpu.PathRenderersDashLine != 0 — unreachable
	if options.PathRenderers&gpu.PathRenderersAAConvex != 0 {
		c.chain = append(c.chain, NewAAConvexPathRenderer())
	}
	if options.PathRenderers&gpu.PathRenderersAAHairline != 0 {
		c.chain = append(c.chain, NewAAHairLinePathRenderer())
	}
	if options.PathRenderers&gpu.PathRenderersAALinearizing != 0 {
		c.chain = append(c.chain, NewAALinearizingConvexPathRenderer())
	}
	// PathRenderersAtlas: trimmed (an optional DMSAA atlasing optimization). PathRenderersSmall: trimmed (the
	// small-path atlas renderer).
	if options.PathRenderers&gpu.PathRenderersTriangulating != 0 {
		c.chain = append(c.chain, NewTriangulatingPathRenderer())
	}
	if options.PathRenderers&gpu.PathRenderersTessellation != 0 {
		if TessellationPathRendererIsSupported(caps) {
			tess := NewTessellationPathRenderer()
			c.tessellationPathRenderer = tess
			c.chain = append(c.chain, tess)
		}
	}

	// We always include the default path renderer (as well as SW), so we can draw any path.
	c.chain = append(c.chain, NewDefaultPathRenderer())
	return c
}

// GetTessellationPathRenderer returns the chain's TessellationPathRenderer, or nil if it is not in the chain.
func (c *PathRendererChain) GetTessellationPathRenderer() *TessellationPathRenderer {
	return c.tessellationPathRenderer
}

// GetPathRenderer returns a renderer that can draw the shape, or nil. If drawType requires stenciling, stencilSupport
// (when non-nil) receives the selected renderer's support level.
func (c *PathRendererChain) GetPathRenderer(args *CanDrawPathArgs, drawType ChainDrawType, stencilSupport *StencilSupport) PathRenderer {
	minStencilSupport := StencilSupportNone
	switch drawType {
	case ChainDrawTypeStencil:
		minStencilSupport = StencilSupportStencilOnly
	case ChainDrawTypeStencilAndColor:
		minStencilSupport = StencilSupportNoRestriction
	default:
	}
	if minStencilSupport != StencilSupportNone {
		// We don't support (and shouldn't need) stenciling of non-fill paths.
		if !args.Shape.Style().IsSimpleFill() {
			return nil
		}
	}

	var best PathRenderer
	for _, pr := range c.chain {
		support := StencilSupportNone
		if minStencilSupport != StencilSupportNone {
			support = PathRendererGetStencilSupport(pr, args.Shape)
			if support < minStencilSupport {
				continue
			}
		}
		canDrawPath := pr.OnCanDrawPath(args)
		if canDrawPath == CanDrawPathNo {
			continue
		}
		if canDrawPath == CanDrawPathAsBackup && best != nil {
			continue
		}
		if stencilSupport != nil {
			*stencilSupport = support
		}
		best = pr
		if canDrawPath == CanDrawPathYes {
			break
		}
	}
	return best
}
