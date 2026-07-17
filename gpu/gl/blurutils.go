// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GaussianBlur is the GPU Gaussian-blur engine that mask filters (DrawShapeWithMaskFilter) and image filters drive. It
// blurs a texture-backed source into a new SurfaceDrawContext, choosing a single non-separable 2D pass for tiny radii,
// a two-pass separable X-then-Y linear convolution up to gpu.MaxLinearBlurSigma, and a downscale→blur→re-expand ladder
// above it. The linear-sampling passes fold pairs of Gaussian taps into bilinear samples (LinearBlur1DFP) so a radius-N
// blur costs N+1 texture reads.
//
// Two simplifications, neither changing output for the reachable (decal/clamp) tile modes:
//   - convolveGaussian always takes the single-draw shader-tiling branch; a left/mid/right/top/ bottom split is a
//     draw-count optimization that would produce identical pixels because the texture effect's wrap mode + domain
//     already enforce the tile mode.
//   - rescaleIntoRepeatedLinear implements only the rescale configuration GaussianBlur uses (repeated bilinear
//     halving/doubling over a texturable source); cubic and linear-gamma rescale lanes are unreachable in an sRGB-only
//     pipeline.

package gl

import (
	"math"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
	"github.com/richardwilkes/canvas/shaders"
)

// blurDirection is the axis a 1D convolution pass runs along.
type blurDirection int

const (
	blurDirX blurDirection = iota
	blurDirY
)

// tileModeToWrapMode maps a shader tile mode to the corresponding GL sampler wrap mode.
func tileModeToWrapMode(mode shaders.TileMode) gpu.WrapMode {
	switch mode {
	case shaders.TileClamp:
		return gpu.WrapModeClamp
	case shaders.TileRepeat:
		return gpu.WrapModeRepeat
	case shaders.TileMirror:
		return gpu.WrapModeMirrorRepeat
	default: // shaders.TileDecal
		return gpu.WrapModeClampToBorder
	}
}

// srcBlendFPPaint builds the Paint that fillWithFP-style draws use: the FP as the color processor with a src blend, so
// its output replaces the destination.
func srcBlendFPPaint(fp FragmentProcessor) *Paint {
	p := NewPaint()
	p.SetColorFragmentProcessor(fp)
	p.SetPorterDuffXPFactory(raster.BlendSrc)
	return p
}

// makeBlurTextureEffect builds a subset-domain texture effect reduced to the desktop (non-reduced-shader-mode) domain
// branch: the domain is the dst rect inset by half a pixel (samples land at pixel centers) and outset by the blur
// radii. The effect is matrix-wrapped (identity user matrix) so its coords are normalized before the sampler read; the
// blur FPs sample it explicitly at the kernel offsets.
func makeBlurTextureEffect(caps *Caps, view SurfaceProxyView, alphaType gpu.AlphaType, sampler gpu.SamplerState, srcSubset, srcRelativeDstRect geom.IRect, radii geom.ISize) FragmentProcessor {
	domain := srcRelativeDstRect.ToRect().Inset(0.5, 0.5).Outset(float32(radii.Width),
		float32(radii.Height))
	return MakeTextureEffectSubsetWithDomain(view, alphaType, geom.IdentityMatrix(), sampler,
		srcSubset.ToRect(), domain, caps, [4]float32{})
}

// convolveGaussian1D draws dstRect into sfc evaluating a 1D Gaussian over srcView. srcRect is dstRect offset by
// dstToSrcOffset; mode/bounds apply to the src coords.
func convolveGaussian1D(sfc *SurfaceFillContext, srcView SurfaceProxyView, srcSubset geom.IRect, dstToSrcOffset geom.IPoint, dstRect geom.IRect, srcAlphaType gpu.AlphaType, direction blurDirection, radius int32, sigma float32, mode shaders.TileMode) {
	srcRect := dstRect.Offset(dstToSrcOffset.X, dstToSrcOffset.Y)
	offsetsAndKernel := gpu.Compute1DBlurLinearKernel(sigma, radius)

	// The child of the 1D linear blur effect must be linearly sampled.
	sampler := gpu.MakeSamplerState(tileModeToWrapMode(mode), tileModeToWrapMode(mode),
		gpu.FilterModeLinear, gpu.MipmapModeNone)

	radii := geom.ISize{}
	dir := geom.Point{X: 1}
	if direction == blurDirX {
		radii.Width = radius
	} else {
		radii.Height = radius
		dir = geom.Point{Y: 1}
	}

	child := makeBlurTextureEffect(sfc.Caps(), srcView, srcAlphaType, sampler, srcSubset, srcRect,
		radii)
	conv := NewLinearBlur1DFP(radius, &offsetsAndKernel, dir, child)
	sfc.FillRectToRectWithFP(srcRect.ToRect(), dstRect, conv)
}

// convolveGaussian2D is a single non-separable pass for small radii.
func convolveGaussian2D(ctx *DirectContext, srcView SurfaceProxyView, srcColorType gpu.ColorType, srcBounds, dstBounds geom.IRect, radiusX, radiusY int32, sigmaX, sigmaY float32, mode shaders.TileMode, fit gpu.BackingFit) *SurfaceDrawContext {
	sdc := MakeSurfaceDrawContext(ctx, srcColorType, dstBounds.Size(), fit, 1, gpu.MipmappedNo,
		srcView.Origin(), gpu.BudgetedYes, "SurfaceDrawContext_ConvolveGaussian2d")
	if sdc == nil {
		return nil
	}

	var kernel [gpu.MaxBlurSamples]float32
	var offsets [gpu.MaxBlurSamples * 2]float32
	gpu.Compute2DBlurKernel(sigmaX, sigmaY, radiusX, radiusY, kernel[:])
	gpu.Compute2DBlurOffsets(radiusX, radiusY, offsets[:])

	sampler := gpu.MakeSamplerState(tileModeToWrapMode(mode), tileModeToWrapMode(mode),
		gpu.FilterModeNearest, gpu.MipmapModeNone)
	child := makeBlurTextureEffect(sdc.Caps(), srcView, gpu.AlphaTypePremul, sampler, srcBounds,
		dstBounds, geom.ISize{Width: radiusX, Height: radiusY})
	conv := NewBlur2DFP(radiusX, radiusY, &kernel, &offsets, child)

	// 'dstBounds' is in 'srcView' proxy space; draw its size and use it directly as the local rect.
	identity := geom.IdentityMatrix()
	sdc.FillRectToRect(nil, srcBlendFPPaint(conv), gpu.AANo, &identity,
		geom.RectWH(float32(dstBounds.Width()), float32(dstBounds.Height())), dstBounds.ToRect())
	return sdc
}

// convolveGaussian takes the single-draw shader-tiling branch (see the file comment): it captures the 'dstBounds'
// portion of an infinite blur of srcBounds in a new SDC with dstBounds' top-left at {0, 0}.
func convolveGaussian(ctx *DirectContext, srcView SurfaceProxyView, srcColorType gpu.ColorType, srcAlphaType gpu.AlphaType, srcBounds, dstBounds geom.IRect, direction blurDirection, radius int32, sigma float32, mode shaders.TileMode, fit gpu.BackingFit) *SurfaceDrawContext {
	dstSDC := MakeSurfaceDrawContext(ctx, srcColorType, dstBounds.Size(), fit, 1, gpu.MipmappedNo,
		srcView.Origin(), gpu.BudgetedYes, "SurfaceDrawContext_ConvolveGaussian")
	if dstSDC == nil {
		return nil
	}
	rtcToSrcOffset := geom.IPoint{X: dstBounds.Left, Y: dstBounds.Top}
	convolveGaussian1D(&dstSDC.SurfaceFillContext, srcView, srcBounds, rtcToSrcOffset,
		geom.IRectSize(dstBounds.Size()), srcAlphaType, direction, radius, sigma, mode)
	return dstSDC
}

// twoPassGaussian runs an X pass (inflated for the Y pass and clipped in a tile-mode-dependent way) followed by an
// in-place Y pass.
func twoPassGaussian(ctx *DirectContext, srcView SurfaceProxyView, srcColorType gpu.ColorType, srcAlphaType gpu.AlphaType, srcBounds, dstBounds geom.IRect, sigmaX, sigmaY float32, radiusX, radiusY int32, mode shaders.TileMode, fit gpu.BackingFit) *SurfaceDrawContext {
	var dstSDC *SurfaceDrawContext
	if radiusX > 0 {
		xFit := fit
		if radiusY > 0 {
			xFit = gpu.BackingFitApprox
		}
		// Expand dstBounds vertically to produce the rows the Y pass reads, then clip tile-mode dependently. If there's
		// no Y pass we must keep the original dstBounds for the right size.
		xPassDstBounds := dstBounds
		if radiusY != 0 {
			xPassDstBounds = xPassDstBounds.Inset(0, -radiusY)
			if mode == shaders.TileRepeat || mode == shaders.TileMirror {
				srcH := srcBounds.Height()
				srcTop := srcBounds.Top
				if mode == shaders.TileMirror {
					srcTop -= srcH
					srcH *= 2
				}
				floatH := float32(srcH)
				n := geom.FloorToInt(float32(xPassDstBounds.Top-srcTop) / floatH)
				topClip := srcTop + n*srcH
				n = geom.CeilToInt(float32(xPassDstBounds.Bottom-srcBounds.Bottom) / floatH)
				bottomClip := srcBounds.Bottom + n*srcH
				xPassDstBounds.Top = max(xPassDstBounds.Top, topClip)
				xPassDstBounds.Bottom = min(xPassDstBounds.Bottom, bottomClip)
			} else {
				switch {
				case xPassDstBounds.Bottom <= srcBounds.Top:
					if mode == shaders.TileDecal {
						return nil
					}
					xPassDstBounds.Top = srcBounds.Top
					xPassDstBounds.Bottom = xPassDstBounds.Top + 1
				case xPassDstBounds.Top >= srcBounds.Bottom:
					if mode == shaders.TileDecal {
						return nil
					}
					xPassDstBounds.Bottom = srcBounds.Bottom
					xPassDstBounds.Top = xPassDstBounds.Bottom - 1
				default:
					xPassDstBounds.Top = max(xPassDstBounds.Top, srcBounds.Top)
					xPassDstBounds.Bottom = min(xPassDstBounds.Bottom, srcBounds.Bottom)
				}
				leftSrcEdge := srcBounds.Left - radiusX
				rightSrcEdge := srcBounds.Right + radiusX
				if mode == shaders.TileClamp {
					// In clamp the column just outside src has the same value as just inside.
					leftSrcEdge++
					rightSrcEdge--
				}
				if xPassDstBounds.Right <= leftSrcEdge {
					if mode == shaders.TileDecal {
						return nil
					}
					xPassDstBounds.Left = xPassDstBounds.Right - 1
				} else {
					xPassDstBounds.Left = max(xPassDstBounds.Left, leftSrcEdge)
				}
				if xPassDstBounds.Left >= rightSrcEdge {
					if mode == shaders.TileDecal {
						return nil
					}
					xPassDstBounds.Right = xPassDstBounds.Left + 1
				} else {
					xPassDstBounds.Right = min(xPassDstBounds.Right, rightSrcEdge)
				}
			}
		}
		dstSDC = convolveGaussian(ctx, srcView, srcColorType, srcAlphaType, srcBounds,
			xPassDstBounds, blurDirX, radiusX, sigmaX, mode, xFit)
		if dstSDC == nil {
			return nil
		}
		srcView = dstSDC.ReadSurfaceView()
		dstBounds = geom.IRectSize(dstBounds.Size()).Offset(dstBounds.Left-xPassDstBounds.Left,
			dstBounds.Top-xPassDstBounds.Top)
		srcBounds = geom.IRectSize(xPassDstBounds.Size())
	}

	if radiusY == 0 {
		return dstSDC
	}
	return convolveGaussian(ctx, srcView, srcColorType, srcAlphaType, srcBounds, dstBounds,
		blurDirY, radiusY, sigmaY, mode, fit)
}

// rescaleIntoRepeatedLinear implements the rescale configuration GaussianBlur drives (over a texturable source):
// repeated bilinear halving/doubling passes until srcRect reaches dstRect's size.
func rescaleIntoRepeatedLinear(ctx *DirectContext, srcView SurfaceProxyView, colorType gpu.ColorType, alphaType gpu.AlphaType, dst *SurfaceDrawContext, dstRect, srcRect geom.IRect) bool {
	if !geom.IRectSize(dst.Dimensions()).ContainsRect(dstRect) {
		return false
	}
	finalSize := dstRect.Size()
	// A size-preserving rescale degenerates to a single nearest-filtered copy.
	nearest := finalSize == srcRect.Size()

	texView := srcView
	for {
		nextDims := finalSize
		if !nearest {
			switch {
			case srcRect.Width() > finalSize.Width:
				nextDims.Width = max((srcRect.Width()+1)/2, finalSize.Width)
			case srcRect.Width() < finalSize.Width:
				nextDims.Width = min(srcRect.Width()*2, finalSize.Width)
			}
			switch {
			case srcRect.Height() > finalSize.Height:
				nextDims.Height = max((srcRect.Height()+1)/2, finalSize.Height)
			case srcRect.Height() < finalSize.Height:
				nextDims.Height = min(srcRect.Height()*2, finalSize.Height)
			}
		}

		var stepDst *SurfaceDrawContext
		var stepDstRect geom.IRect
		if nextDims == finalSize {
			stepDst = dst
			stepDstRect = dstRect
		} else {
			stepDst = MakeSurfaceDrawContext(ctx, colorType, nextDims, gpu.BackingFitApprox, 1,
				gpu.MipmappedNo, dst.Origin(), gpu.BudgetedYes, "RescaledSurfaceDrawContext")
			if stepDst == nil {
				return false
			}
			stepDstRect = geom.IRectSize(stepDst.Dimensions())
		}

		filter := gpu.FilterModeLinear
		if nearest {
			filter = gpu.FilterModeNearest
		}
		srcRectF := srcRect.ToRect()
		fp := MakeTextureEffectSubsetWithDomain(texView, alphaType, geom.IdentityMatrix(),
			gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, filter, gpu.MipmapModeNone),
			srcRectF, srcRectF, stepDst.Caps(), [4]float32{})
		stepDst.FillRectToRectWithFP(srcRectF, stepDstRect, fp)

		texView = stepDst.ReadSurfaceView()
		srcRect = geom.IRectSize(nextDims)
		if srcRect.Size() == finalSize {
			return true
		}
	}
}

// reexpand bilinearly expands the intermediate blurred src to fill dstSize.
func reexpand(ctx *DirectContext, src *SurfaceDrawContext, srcBounds geom.Rect, dstSize geom.ISize, fit gpu.BackingFit) *SurfaceDrawContext {
	srcView := src.ReadSurfaceView()
	if srcView.AsTextureProxy() == nil {
		return nil
	}
	srcColorType := src.ColorType()

	dstSDC := MakeSurfaceDrawContext(ctx, srcColorType, dstSize, fit, 1, gpu.MipmappedNo,
		srcView.Origin(), gpu.BudgetedYes, "SurfaceDrawContext_Reexpand")
	if dstSDC == nil {
		return nil
	}
	fp := MakeTextureEffectSubsetWithDomain(srcView, gpu.AlphaTypePremul, geom.IdentityMatrix(),
		gpu.MakeSamplerState(gpu.WrapModeClamp, gpu.WrapModeClamp, gpu.FilterModeLinear,
			gpu.MipmapModeNone), srcBounds, srcBounds, dstSDC.Caps(), [4]float32{})
	identity := geom.IdentityMatrix()
	dstSDC.FillRectToRect(nil, srcBlendFPPaint(fp), gpu.AANo, &identity,
		geom.RectWH(float32(dstSize.Width), float32(dstSize.Height)), srcBounds)
	return dstSDC
}

// GaussianBlur blurs the dstBounds portion of an infinite blur of srcBounds within srcView, returning a new
// SurfaceDrawContext with dstBounds' top-left at {0, 0}. Returns nil for an unblurrable request or an allocation
// failure.
func GaussianBlur(ctx *DirectContext, srcView SurfaceProxyView, srcColorType gpu.ColorType, srcAlphaType gpu.AlphaType, dstBounds, srcBounds geom.IRect, sigmaX, sigmaY float32, mode shaders.TileMode, fit gpu.BackingFit) *SurfaceDrawContext {
	if srcView.AsTextureProxy() == nil {
		return nil
	}
	maxRTSize := int32(ctx.GLCaps().MaxRenderTargetSize)
	if dstBounds.Width() > maxRTSize || dstBounds.Height() > maxRTSize {
		return nil
	}

	radiusX := gpu.BlurSigmaRadius(sigmaX)
	radiusY := gpu.BlurSigmaRadius(sigmaY)
	// Reduce srcBounds to detect zero-sigma cases and to shrink rescale work for large sigmas.
	if mode == shaders.TileClamp || mode == shaders.TileDecal {
		reach := dstBounds.Inset(-radiusX, -radiusY)
		intersection := reach
		if !intersection.Intersect(srcBounds) {
			if mode == shaders.TileDecal {
				return nil
			}
			switch {
			case reach.Left >= srcBounds.Right:
				srcBounds.Left = srcBounds.Right - 1
			case reach.Right <= srcBounds.Left:
				srcBounds.Right = srcBounds.Left + 1
			}
			switch {
			case reach.Top >= srcBounds.Bottom:
				srcBounds.Top = srcBounds.Bottom - 1
			case reach.Bottom <= srcBounds.Top:
				srcBounds.Bottom = srcBounds.Top + 1
			}
		} else {
			srcBounds = intersection
		}
	}

	if mode != shaders.TileDecal {
		// For non-decal modes a one-pixel src column/row is a single repeated color; blurring a normalized kernel over
		// it yields the same color, so no blur is needed in that axis.
		if srcBounds.Width() == 1 {
			sigmaX, radiusX = 0, 0
		}
		if srcBounds.Height() == 1 {
			sigmaY, radiusY = 0, 0
		}
	}

	// No blurring needed in either direction: just apply the tile mode with a single draw.
	if radiusX == 0 && radiusY == 0 {
		result := MakeSurfaceDrawContext(ctx, srcColorType, dstBounds.Size(), fit, 1,
			gpu.MipmappedNo, srcView.Origin(), gpu.BudgetedYes, "SurfaceDrawContext_GaussianBlur")
		if result == nil {
			return nil
		}
		sampler := gpu.MakeSamplerState(tileModeToWrapMode(mode), tileModeToWrapMode(mode),
			gpu.FilterModeNearest, gpu.MipmapModeNone)
		fp := MakeTextureEffectSubsetWithDomain(srcView, srcAlphaType, geom.IdentityMatrix(),
			sampler, srcBounds.ToRect(), dstBounds.ToRect(), result.Caps(), [4]float32{})
		result.FillRectToRectWithFP(dstBounds.ToRect(), geom.IRectSize(dstBounds.Size()), fp)
		return result
	}

	// Sigmas within the linear-blur limit are handled directly (2D for tiny kernels, else two 1D passes). The 2D limit
	// is always below the linear limit.
	const kMaxSigma = float32(gpu.MaxLinearBlurSigma)
	if sigmaX <= kMaxSigma && sigmaY <= kMaxSigma {
		// For really small blurs a single non-separable kernel beats two passes.
		kernelSize := gpu.BlurKernelWidth(radiusX) * gpu.BlurKernelWidth(radiusY)
		if radiusX > 0 && radiusY > 0 && kernelSize <= gpu.MaxBlurSamples {
			return convolveGaussian2D(ctx, srcView, srcColorType, srcBounds, dstBounds, radiusX,
				radiusY, sigmaX, sigmaY, mode, fit)
		}
		// Degenerates into a single X or Y pass when only one radius is non-zero.
		return twoPassGaussian(ctx, srcView, srcColorType, srcAlphaType, srcBounds, dstBounds,
			sigmaX, sigmaY, radiusX, radiusY, mode, fit)
	}

	// Large sigma: downscale the source, blur it at the reduced sigma, then re-expand.
	scaleX := float32(1)
	if sigmaX > kMaxSigma {
		scaleX = kMaxSigma / sigmaX
	}
	scaleY := float32(1)
	if sigmaY > kMaxSigma {
		scaleY = kMaxSigma / sigmaY
	}
	// Round down so the recomputed sigmas stay below kMaxSigma (clamp to 1 for a non-empty texture).
	rescaledW := max(int32(math.Floor(float64(float32(srcBounds.Width())*scaleX))), 1)
	rescaledH := max(int32(math.Floor(float64(float32(srcBounds.Height())*scaleY))), 1)
	// Recompute the scale factors from the integerized size, then the reduced sigmas.
	scaleX = float32(rescaledW) / float32(srcBounds.Width())
	scaleY = float32(rescaledH) / float32(srcBounds.Height())
	sigmaX *= scaleX
	sigmaY *= scaleY

	padX := int32(0)
	if mode == shaders.TileClamp || (mode == shaders.TileDecal && sigmaX > kMaxSigma) {
		padX = 1
	}
	padY := int32(0)
	if mode == shaders.TileClamp || (mode == shaders.TileDecal && sigmaY > kMaxSigma) {
		padY = 1
	}

	rescaledSDC := MakeSurfaceDrawContext(ctx, srcColorType,
		geom.ISize{Width: rescaledW + 2*padX, Height: rescaledH + 2*padY}, gpu.BackingFitApprox, 1,
		gpu.MipmappedNo, srcView.Origin(), gpu.BudgetedYes, "RescaledSurfaceDrawContext")
	if rescaledSDC == nil {
		return nil
	}
	if (padX != 0 || padY != 0) && mode == shaders.TileDecal {
		rescaledSDC.Clear([4]float32{})
	}
	if !rescaleIntoRepeatedLinear(ctx, srcView, srcColorType, srcAlphaType, rescaledSDC,
		geom.IRectXYWH(padX, padY, rescaledW, rescaledH), srcBounds) {
		return nil
	}
	if mode == shaders.TileClamp {
		blurRescaleClampEdges(rescaledSDC, srcView, srcAlphaType, srcBounds,
			geom.ISize{Width: rescaledW, Height: rescaledH})
	}
	srcView = rescaledSDC.ReadSurfaceView()

	// Compute the dst bounds in the scaled-down space (origin moved to src's top-left).
	scaledDstBounds := dstBounds.Offset(-srcBounds.Left, -srcBounds.Top).ToRect()
	scaledDstBounds.Left *= scaleX
	scaledDstBounds.Top *= scaleY
	scaledDstBounds.Right *= scaleX
	scaledDstBounds.Bottom *= scaleY
	scaledDstBounds = scaledDstBounds.Offset(float32(padX), float32(padY))
	scaledDstBoundsI := scaledDstBounds.RoundOut()

	scaledSrcBounds := geom.IRectSize(srcView.Dimensions())
	sdc := GaussianBlur(ctx, srcView, srcColorType, srcAlphaType, scaledDstBoundsI, scaledSrcBounds,
		sigmaX, sigmaY, mode, fit)
	if sdc == nil {
		return nil
	}
	// Select the fractional dst bounds from the integer-rounded blurred result during re-expand.
	scaledDstBounds = scaledDstBounds.Offset(float32(-scaledDstBoundsI.Left),
		float32(-scaledDstBoundsI.Top))
	return reexpand(ctx, sdc, scaledDstBounds, dstBounds.Size(), fit)
}

// blurRescaleClampEdges applies the clamp-mode edge/corner padding after the interior rescale: rather than multi-pass
// rescale single rows/columns, it does one bilinear draw per edge and a nearest draw per corner from the original
// source.
func blurRescaleClampEdges(rescaledSDC *SurfaceDrawContext, srcView SurfaceProxyView, srcAlphaType gpu.AlphaType, srcBounds geom.IRect, rescaledSize geom.ISize) {
	dw := rescaledSize.Width
	dh := rescaledSize.Height
	identity := geom.IdentityMatrix()
	cheapDownscale := func(dstRect, srcRect geom.IRect) {
		rescaledSDC.DrawTexture(nil, srcView, srcAlphaType, gpu.FilterModeLinear, gpu.MipmapModeNone,
			raster.BlendSrc, colorcore.PMColor4f{R: 1, G: 1, B: 1, A: 1}, srcRect.ToRect(),
			dstRect.ToRect(), gpu.QuadAAFlagsNone, SrcRectConstraintFast, &identity)
	}
	// Rows/columns of the original source scaled into the dst padding.
	sLCol := srcBounds.Left
	sTRow := srcBounds.Top
	sRCol := srcBounds.Right - 1
	sBRow := srcBounds.Bottom - 1
	sx := srcBounds.Left
	sy := srcBounds.Top
	sw := srcBounds.Width()
	sh := srcBounds.Height()

	// Edges.
	cheapDownscale(geom.IRectXYWH(0, 1, 1, dh), geom.IRectXYWH(sLCol, sy, 1, sh))
	cheapDownscale(geom.IRectXYWH(1, 0, dw, 1), geom.IRectXYWH(sx, sTRow, sw, 1))
	cheapDownscale(geom.IRectXYWH(dw+1, 1, 1, dh), geom.IRectXYWH(sRCol, sy, 1, sh))
	cheapDownscale(geom.IRectXYWH(1, dh+1, dw, 1), geom.IRectXYWH(sx, sBRow, sw, 1))
	// Corners.
	cheapDownscale(geom.IRectXYWH(0, 0, 1, 1), geom.IRectXYWH(sLCol, sTRow, 1, 1))
	cheapDownscale(geom.IRectXYWH(dw+1, 0, 1, 1), geom.IRectXYWH(sRCol, sTRow, 1, 1))
	cheapDownscale(geom.IRectXYWH(dw+1, dh+1, 1, 1), geom.IRectXYWH(sRCol, sBRow, 1, 1))
	cheapDownscale(geom.IRectXYWH(0, dh+1, 1, 1), geom.IRectXYWH(sLCol, sBRow, 1, 1))
}
