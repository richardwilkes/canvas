// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The last-resort path renderer: rasterizes the shape into an A8 coverage mask on the CPU (over the rasterizer),
// uploads it, and draws it as a coverage-FP rect, caching masks by unique key when the matrix preserves axis alignment.
// Trims: the threaded mask-render lane reduces to a synchronous lane (same output), and there is no invalidation
// listener on the original path — Go has no destruction hook, so stale cached masks age out of the budgeted
// resource-cache LRU instead.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// SoftwarePathRenderer is the last-resort CPU-rasterization fallback path renderer.
type SoftwarePathRenderer struct {
	proxyProvider *ProxyProvider
	allowCaching  bool
}

// NewSoftwarePathRenderer builds a software path renderer.
func NewSoftwarePathRenderer(proxyProvider *ProxyProvider, allowCaching bool) *SoftwarePathRenderer {
	return &SoftwarePathRenderer{proxyProvider: proxyProvider, allowCaching: allowCaching}
}

// Name implements PathRenderer.
func (r *SoftwarePathRenderer) Name() string { return "SW" }

// OnGetStencilSupport implements PathRenderer (the base-class default, matching OnStencilPath's panic and upstream's own
// override: this renderer cannot stencil, and reporting anything else would make PathRendererStencilPath — which only
// rejects StencilSupportNone — reach that panic).
func (r *SoftwarePathRenderer) OnGetStencilSupport(*StyledShape) StencilSupport {
	return StencilSupportNone
}

// OnCanDrawPath implements PathRenderer: pass on any style that applies — the caller will apply the style if a suitable
// renderer is not found and try again with the new shape.
func (r *SoftwarePathRenderer) OnCanDrawPath(args *CanDrawPathArgs) CanDrawPath {
	if !args.Shape.Style().Applies() && r.proxyProvider != nil &&
		(args.AAType == gpu.AATypeCoverage || args.AAType == gpu.AATypeNone) {
		// This is the fallback renderer for when a path is too complicated for the GPU ones.
		return CanDrawPathAsBackup
	}
	return CanDrawPathNo
}

// OnStencilPath implements PathRenderer (never reached: the chain does not hold the SW renderer, and the drawing
// manager only offers it for color draws).
func (r *SoftwarePathRenderer) OnStencilPath(*StencilPathArgs) {
	panic("SoftwarePathRenderer cannot stencil paths")
}

// getUnclippedShapeDevBounds computes the shape's device-space bounds under matrix, clipped to the representable int32
// range.
func getUnclippedShapeDevBounds(shape *StyledShape, matrix *geom.Matrix) (geom.IRect, bool) {
	shapeBounds := shape.StyledBounds()
	if shapeBounds.IsEmpty() {
		return geom.IRect{}, false
	}
	shapeDevBounds, _ := matrix.MapRect(shapeBounds)
	// Even though these are "unclipped" bounds we still clip to the int32_t range: this is the largest int32 that is
	// representable exactly as a float (INT32_MIN is exactly representable).
	const maxInt = 2147483520
	if !shapeDevBounds.Intersect(geom.RectLTRB(math.MinInt32, math.MinInt32, maxInt, maxInt)) {
		return geom.IRect{}, false
	}
	// Make sure that the resulting rect can have representable width and height.
	if geom.RoundToInt(shapeDevBounds.Width()) > maxInt ||
		geom.RoundToInt(shapeDevBounds.Height()) > maxInt {
		return geom.IRect{}, false
	}
	return shapeDevBounds.RoundOutRect().Round(), true
}

// GetShapeAndClipBounds computes the shape bounds, the clip bounds, and their intersection (returns false when there is
// no intersection).
func GetShapeAndClipBounds(sdc *SurfaceDrawContext, clip Clip, shape *StyledShape, matrix *geom.Matrix) (unclippedDevShapeBounds, clippedDevShapeBounds, devClipBounds geom.IRect, ok bool) {
	// Compute the bounds as the intersection of rt size, clip, and path.
	devClipBounds = sdc.clipConservativeBounds(clip)

	unclippedDevShapeBounds, boundsOK := getUnclippedShapeDevBounds(shape, matrix)
	if !boundsOK {
		return geom.IRect{}, geom.IRect{}, devClipBounds, false
	}
	clippedDevShapeBounds = devClipBounds
	if !clippedDevShapeBounds.Intersect(unclippedDevShapeBounds) {
		return unclippedDevShapeBounds, geom.IRect{}, devClipBounds, false
	}
	return unclippedDevShapeBounds, clippedDevShapeBounds, devClipBounds, true
}

// swDrawNonAARect draws rect with no antialiasing. The paint is consumed.
func swDrawNonAARect(sdc *SurfaceDrawContext, paint *Paint, userStencilSettings *UserStencilSettings, clip Clip, viewMatrix *geom.Matrix, rect geom.Rect, localMatrix *geom.Matrix) {
	sdc.StencilRect(clip, userStencilSettings, paint, gpu.AANo, viewMatrix, rect, localMatrix)
}

// swDrawAroundInvPath fills the clip area around the (device-space) path bounds. The paint is consumed.
func swDrawAroundInvPath(sdc *SurfaceDrawContext, paint *Paint, userStencilSettings *UserStencilSettings, clip Clip, viewMatrix *geom.Matrix, devClipBounds, devPathBounds geom.IRect) {
	invert, ok := viewMatrix.Invert()
	if !ok {
		return
	}
	identity := geom.IdentityMatrix()
	if devClipBounds.Top < devPathBounds.Top {
		rect := geom.RectLTRB(float32(devClipBounds.Left), float32(devClipBounds.Top),
			float32(devClipBounds.Right), float32(devPathBounds.Top))
		swDrawNonAARect(sdc, ClonePaint(paint), userStencilSettings, clip, &identity, rect,
			&invert)
	}
	if devClipBounds.Left < devPathBounds.Left {
		rect := geom.RectLTRB(float32(devClipBounds.Left), float32(devPathBounds.Top),
			float32(devPathBounds.Left), float32(devPathBounds.Bottom))
		swDrawNonAARect(sdc, ClonePaint(paint), userStencilSettings, clip, &identity, rect,
			&invert)
	}
	if devClipBounds.Right > devPathBounds.Right {
		rect := geom.RectLTRB(float32(devPathBounds.Right), float32(devPathBounds.Top),
			float32(devClipBounds.Right), float32(devPathBounds.Bottom))
		swDrawNonAARect(sdc, ClonePaint(paint), userStencilSettings, clip, &identity, rect,
			&invert)
	}
	if devClipBounds.Bottom > devPathBounds.Bottom {
		rect := geom.RectLTRB(float32(devClipBounds.Left), float32(devPathBounds.Bottom),
			float32(devClipBounds.Right), float32(devClipBounds.Bottom))
		swDrawNonAARect(sdc, paint, userStencilSettings, clip, &identity, rect, &invert)
	}
}

// swDrawToTargetWithShapeMask draws a device-space rect textured by the mask, with local coordinates that undo the view
// matrix so paint-stage local coords stay in source space. The paint is consumed.
func swDrawToTargetWithShapeMask(view SurfaceProxyView, sdc *SurfaceDrawContext, paint *Paint, userStencilSettings *UserStencilSettings, clip Clip, viewMatrix *geom.Matrix, textureOriginInDeviceSpace geom.IPoint, deviceSpaceRectToDraw geom.IRect) {
	invert, ok := viewMatrix.Invert()
	if !ok {
		return
	}

	view.ConcatSwizzle(gpu.MakeSwizzle("aaaa"))

	dstRect := deviceSpaceRectToDraw.ToRect()

	// We use device coords to compute the texture coordinates: translate so that the top-left of the device bounds maps
	// to 0,0.
	maskMatrix := geom.TranslateMatrix(float32(-textureOriginInDeviceSpace.X),
		float32(-textureOriginInDeviceSpace.Y))
	maskMatrix.PreConcat(viewMatrix)

	sampler := gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp,
		gpu.FilterModeNearest, gpu.MipmapModeNone)
	paint.SetCoverageFragmentProcessor(MakeTextureEffect(view, gpu.AlphaTypePremul, maskMatrix,
		sampler, sdc.Caps(), [4]float32{}))
	identity := geom.IdentityMatrix()
	swDrawNonAARect(sdc, paint, userStencilSettings, clip, &identity, dstRect, &invert)
}

// swPathMaskKeyDomain is the unique-key domain for "SW Path Mask" cache keys.
var swPathMaskKeyDomain = gpu.GenerateUniqueKeyDomain()

// OnDrawPath implements PathRenderer. Returns true on success (which includes "nothing visible to draw").
func (r *SoftwarePathRenderer) OnDrawPath(args *DrawPathArgs) bool {
	if r.proxyProvider == nil {
		return false
	}
	if args.Shape.Style().Applies() {
		panic("the SW renderer requires styles to have been applied")
	}
	// We really need to know if the shape will be inverse filled or not; if the path is hairline, ignore inverse fill.
	_, isHairline := isStrokeHairlineOrEquivalent(args.Shape.Style(), args.ViewMatrix)
	inverseFilled := args.Shape.InverseFilled() && !isHairline

	// To prevent overloading the cache with entries during animations we limit the cache of masks to cases where the
	// matrix preserves axis alignment.
	useCache := r.allowCaching && !inverseFilled && args.ViewMatrix.RectStaysRect() &&
		args.Shape.HasUnstyledKey() && args.AAType == gpu.AATypeCoverage

	unclippedDevShapeBounds, clippedDevShapeBounds, devClipBounds, boundsOK := GetShapeAndClipBounds(args.SurfaceDrawContext, args.Clip, args.Shape, args.ViewMatrix)
	if !boundsOK {
		if inverseFilled {
			swDrawAroundInvPath(args.SurfaceDrawContext, args.Paint, args.UserStencilSettings,
				args.Clip, args.ViewMatrix, devClipBounds, unclippedDevShapeBounds)
		}
		return true
	}

	boundsForMask := clippedDevShapeBounds
	if useCache {
		// Use the cache only if >50% of the path is visible.
		unclippedWidth := int64(unclippedDevShapeBounds.Width())
		unclippedHeight := int64(unclippedDevShapeBounds.Height())
		unclippedArea := unclippedWidth * unclippedHeight
		clippedArea := int64(clippedDevShapeBounds.Width()) *
			int64(clippedDevShapeBounds.Height())
		maxTextureSize := int64(args.SurfaceDrawContext.Caps().MaxTextureSize)
		if unclippedArea > 2*clippedArea || unclippedWidth > maxTextureSize ||
			unclippedHeight > maxTextureSize {
			useCache = false
		} else {
			boundsForMask = unclippedDevShapeBounds
		}
	}

	var maskKey gpu.UniqueKey
	if useCache {
		// We require the upper left 2x2 of the matrix to match exactly for a cache hit.
		sx := args.ViewMatrix.Get(geom.MScaleX)
		sy := args.ViewMatrix.Get(geom.MScaleY)
		kx := args.ViewMatrix.Get(geom.MSkewX)
		ky := args.ViewMatrix.Get(geom.MSkewY)
		builder := gpu.UniqueKeyBuilder(&maskKey, swPathMaskKeyDomain,
			7+args.Shape.UnstyledKeySize(), "SW Path Mask")
		data := builder.Slice()
		data[0] = uint32(boundsForMask.Width())
		data[1] = uint32(boundsForMask.Height())
		// Allow 8 bits each in x and y of subpixel positioning (the fractional translation converted to 16.16 fixed
		// point, masked to the high subpixel byte).
		tx := args.ViewMatrix.Get(geom.MTransX)
		ty := args.ViewMatrix.Get(geom.MTransY)
		fracX := int32((tx-geom.FloorToScalar(tx))*65536) & 0x0000FF00
		fracY := int32((ty-geom.FloorToScalar(ty))*65536) & 0x0000FF00
		data[2] = math.Float32bits(sx)
		data[3] = math.Float32bits(sy)
		data[4] = math.Float32bits(kx)
		data[5] = math.Float32bits(ky)
		// Distinguish between hairline and filled paths. For hairlines, we also need to include the cap (SW grows
		// hairlines by 0.5 pixel with round and square caps). Note that stroke-and-fill of hairlines is turned into
		// pure fill upstream, so this covers all cases we might see.
		var styleBits uint32
		if args.Shape.Style().IsSimpleHairline() {
			styleBits = uint32(args.Shape.Style().Rec().Cap())<<1 | 1
		}
		data[6] = uint32(fracX) | uint32(fracY)>>8 | styleBits<<16
		args.Shape.WriteUnstyledKey(data[7:])
		builder.Finish()
	}

	var view SurfaceProxyView
	if useCache {
		if proxy := r.proxyProvider.FindOrCreateProxyByUniqueKey(&maskKey,
			UseAllocatorYes); proxy != nil {
			caps := args.SurfaceDrawContext.Caps()
			swizzle := caps.ReadSwizzle(proxy.Format(), gpu.ColorTypeAlpha8)
			view = MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, swizzle)
			args.Context.Stats().NumPathMasksCacheHits++
		}
	}
	if view.Proxy() == nil {
		aa := gpu.AA(args.AAType == gpu.AATypeCoverage)

		// The threaded deferred-upload lane is trimmed to the synchronous render.
		var helper swMaskHelper
		if !helper.init(boundsForMask) {
			return false
		}
		helper.drawStyledShape(args.Shape, args.ViewMatrix, aa, 0xFF)
		view = helper.toTextureView(args.Context)
		if view.Proxy() == nil {
			return false
		}
		if useCache {
			if view.Origin() != gpu.OriginTopLeft {
				panic("cached mask views must be top-left")
			}
			// There is no invalidation listener on the original path to purge the mask when it dies; the entry ages out
			// of the budgeted LRU instead.
			r.proxyProvider.AssignUniqueKeyToProxy(&maskKey, view.AsTextureProxy())
		}
		args.Context.Stats().NumPathMasksGenerated++
	}
	if view.Proxy() == nil {
		panic("expected a mask view")
	}
	if inverseFilled {
		swDrawAroundInvPath(args.SurfaceDrawContext, ClonePaint(args.Paint),
			args.UserStencilSettings, args.Clip, args.ViewMatrix, devClipBounds,
			unclippedDevShapeBounds)
	}
	swDrawToTargetWithShapeMask(view, args.SurfaceDrawContext, args.Paint,
		args.UserStencilSettings, args.Clip, args.ViewMatrix,
		geom.IPoint{X: boundsForMask.Left, Y: boundsForMask.Top}, boundsForMask)
	return true
}
