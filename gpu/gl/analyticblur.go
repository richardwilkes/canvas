// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Factories for the analytic rect/circle/rrect blur fragment processors: MakeCircleBlur (circle blur), MakeRectBlur
// (rect blur), and MakeRRectBlur (simple-circular rrect blur). Each builds the profile/integral/mask A8 table
// (gpu.Create*), uploads it as a texture effect, and wraps the analytic runtime FP (analyticblurfp.go). The
// profile/integral/mask textures are memoized in the DirectContext's thread-safe cache (threadsafecache.go), keyed by
// the sigma-to-radius ratio for the circle profile, table width for the rect integral, and the sigma+per-corner-radii
// key for the rrect mask, so a repeated blur reuses the upload. The rrect mask is generated on the CPU, designed to be
// interchangeable with a direct-context GPU fill-in this codebase does not implement.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/raster"
)

// kProfileTextureWidth is the circle-blur profile texture width.
const kProfileTextureWidth = 512

// Profile/mask textures are keyed in the thread-safe cache under one domain apiece.
var (
	circleBlurProfileDomain = gpu.GenerateUniqueKeyDomain()
	rectBlurMaskDomain      = gpu.GenerateUniqueKeyDomain()
	roundRectBlurMaskDomain = gpu.GenerateUniqueKeyDomain()
)

// createProfileEffect builds (or finds in the thread-safe cache) the 1-D circular-blur profile texture and returns a
// texture effect that samples it, along with the solid and texture radii the shader needs.
func createProfileEffect(ctx *DirectContext, circle geom.Rect, sigma float32) (fp FragmentProcessor, solidRadius, textureRadius float32, ok bool) {
	circleR := circle.Width() / 2
	if !geom.IsFinite(circleR) || circleR < geom.ScalarNearlyZeroTol {
		return nil, 0, 0, false
	}

	threadSafeCache := ctx.ThreadSafeCache()

	// Profile textures are keyed by the ratio of sigma to circle radius (binned in fixed point). The binning changes
	// the sigma actually used, so it is replicated regardless of the cache outcome.
	sigmaToCircleRRatio := sigma / circleR
	sigmaToCircleRRatio = min(sigmaToCircleRRatio, 8)
	var sigmaToCircleRRatioFixed int32
	const kHalfPlaneThreshold = 0.1
	useHalfPlaneApprox := false
	if sigmaToCircleRRatio <= kHalfPlaneThreshold {
		useHalfPlaneApprox = true
		sigmaToCircleRRatioFixed = 0
		solidRadius = circleR - 3*sigma
		textureRadius = 6 * sigma
	} else {
		// Convert to 16.16 fixed point and shave off the low 8 bits to reduce unique entries.
		sigmaToCircleRRatioFixed = int32(sigmaToCircleRRatio*float32(1<<16)) &^ 0xff
		sigmaToCircleRRatio = float32(sigmaToCircleRRatioFixed) * (1.0 / float32(1<<16))
		sigma = circleR * sigmaToCircleRRatio
		solidRadius = 0
		textureRadius = circleR + 3*sigma
	}

	// This would be kProfileTextureWidth/textureRadius if the profile coord weren't already computed in a coord space
	// pre-scaled by 1/textureRadius (done to avoid overflow in length()).
	texM := geom.ScaleMatrix(kProfileTextureWidth, 1)

	var key gpu.UniqueKey
	kb := gpu.UniqueKeyBuilder(&key, circleBlurProfileDomain, 1, "1-D Circular Blur")
	kb.Slice()[0] = uint32(sigmaToCircleRRatioFixed)
	kb.Finish()

	if profileView := threadSafeCache.Find(&key); profileView.IsValid() {
		fp = MakeTextureEffectSimple(profileView, gpu.AlphaTypePremul, texM, gpu.FilterModeNearest,
			gpu.MipmapModeNone)
		return fp, solidRadius, textureRadius, true
	}

	var profile []byte
	if useHalfPlaneApprox {
		profile = gpu.CreateHalfPlaneProfile(kProfileTextureWidth)
	} else {
		// Rescale params to the size of the texture we're creating.
		scale := kProfileTextureWidth / textureRadius
		profile = gpu.CreateCircleProfile(sigma*scale, circleR*scale, kProfileTextureWidth)
	}
	if len(profile) == 0 {
		return nil, 0, 0, false
	}

	profileView := a8DataToTextureView(ctx, profile, kProfileTextureWidth,
		geom.ISize{Width: kProfileTextureWidth, Height: 1}, "1-D Circular Blur")
	if profileView.Proxy() == nil {
		return nil, 0, 0, false
	}
	profileView = threadSafeCache.Add(&key, profileView)

	fp = MakeTextureEffectSimple(profileView, gpu.AlphaTypePremul, texM, gpu.FilterModeNearest,
		gpu.MipmapModeNone)
	return fp, solidRadius, textureRadius, true
}

// MakeCircleBlur builds an FP that analytically blurs the given device-space circle. Returns nil if the blur is a no-op
// or the profile could not be built.
func MakeCircleBlur(ctx *DirectContext, circle geom.Rect, sigma float32) FragmentProcessor {
	if gpu.BlurIsEffectivelyIdentity(sigma) {
		return nil
	}
	profile, solidRadius, textureRadius, ok := createProfileEffect(ctx, circle, sigma)
	if !ok {
		return nil
	}
	circleData := [4]float32{circle.CenterX(), circle.CenterY(), solidRadius, 1 / textureRadius}
	circleBlurFP := NewCircleBlurFP(circleData, profile)
	// Modulate blur with the input color.
	return BlendFP(circleBlurFP, nil, raster.BlendModulate)
}

// makeRectIntegralFP builds (or finds in the thread-safe cache) the normal-distribution integral table and returns a
// linear-sampled texture effect over it.
func makeRectIntegralFP(ctx *DirectContext, sixSigma float32) FragmentProcessor {
	threadSafeCache := ctx.ThreadSafeCache()
	width := gpu.ComputeIntegralTableWidth(sixSigma)

	var key gpu.UniqueKey
	kb := gpu.UniqueKeyBuilder(&key, rectBlurMaskDomain, 1, "Rect Blur Mask")
	kb.Slice()[0] = uint32(width)
	kb.Finish()

	m := geom.ScaleMatrix(float32(width)/sixSigma, 1)

	if view := threadSafeCache.Find(&key); view.IsValid() {
		return MakeTextureEffectSimple(view, gpu.AlphaTypePremul, m, gpu.FilterModeLinear,
			gpu.MipmapModeNone)
	}

	table := gpu.CreateIntegralTable(width)
	if len(table) == 0 {
		return nil
	}
	view := a8DataToTextureView(ctx, table, width, geom.ISize{Width: int32(width), Height: 1},
		"Rect Blur Mask")
	if view.Proxy() == nil {
		return nil
	}
	view = threadSafeCache.Add(&key, view)

	return MakeTextureEffectSimple(view, gpu.AlphaTypePremul, m, gpu.FilterModeLinear,
		gpu.MipmapModeNone)
}

// MakeRectBlur builds an FP that analytically blurs the given rect. devRect, when non-nil, is the rect already mapped
// to device space. Returns nil if the blur is a no-op or the integral table could not be built.
func MakeRectBlur(ctx *DirectContext, caps *gpu.ShaderCaps, srcRect geom.Rect, devRect *geom.Rect, viewMatrix *geom.Matrix, transformedSigma float32) FragmentProcessor {
	if gpu.BlurIsEffectivelyIdentity(transformedSigma) {
		return nil
	}

	var invM geom.Matrix
	invM.SetIdentity()
	var rect geom.Rect
	switch {
	case devRect != nil:
		rect = *devRect
	case viewMatrix.RectStaysRect():
		// Everything can be done in device space when the src rect projects to a device-space rect.
		rect, _ = viewMatrix.MapRect(srcRect)
	default:
		// The view matrix may scale, perhaps anisotropically. Factor out the scaling to a space that is pure
		// rotation/translation from device space (and scale from src space): pre-scale the src rect into that space and
		// apply the inverse rotation/translation to the frag coord.
		var scale geom.Size
		var m geom.Matrix
		if !viewMatrix.DecomposeScale(&scale, &m) {
			return nil
		}
		inv, ok := m.Invert()
		if !ok {
			return nil
		}
		invM = inv
		rect = geom.Rect{
			Left:   srcRect.Left * scale.Width,
			Top:    srcRect.Top * scale.Height,
			Right:  srcRect.Right * scale.Width,
			Bottom: srcRect.Bottom * scale.Height,
		}
	}

	if !caps.FloatIs32Bits {
		// Promoting the Gaussian-space math to full float needs 32-bit float support; without it, fail for large rect
		// coords (unreachable on this codebase's desktop core-profile target).
		const lim = 16000.0
		if abs32(rect.Left) > lim || abs32(rect.Top) > lim || abs32(rect.Right) > lim ||
			abs32(rect.Bottom) > lim {
			return nil
		}
	}

	sixSigma := 6 * transformedSigma
	integral := makeRectIntegralFP(ctx, sixSigma)
	if integral == nil {
		return nil
	}

	// Inset the rect so that the inset edge corresponds to t=0 in the integral texture.
	threeSigma := sixSigma / 2
	insetRect := geom.Rect{
		Left:   rect.Left + threeSigma,
		Top:    rect.Top + threeSigma,
		Right:  rect.Right - threeSigma,
		Bottom: rect.Bottom - threeSigma,
	}
	// The fast variant (rect at least 6 sigma wide in both axes) looks up only the nearest edges.
	isFast := insetRect.IsSorted()

	fp := NewRectBlurFP(insetRect, isFast, integral)
	// Modulate blur with the input color.
	fp = BlendFP(fp, nil, raster.BlendModulate)
	if !invM.IsIdentity() {
		fp = MatrixEffectFP(invM, fp)
	}
	return DeviceSpaceFP(fp)
}

// makeBlurredRRectKey builds the cache key for a blurred-rrect nine-patch mask from the ceil'd per-corner radii; since
// this codebase's RRect only supports uniform radii, every corner shares one (X, Y) pair.
func makeBlurredRRectKey(key *gpu.UniqueKey, rrectToDraw geom.RRect, xformedSigma float32) {
	kb := gpu.UniqueKeyBuilder(key, roundRectBlurMaskDomain, 9, "RoundRect Blur Mask")
	s := kb.Slice()
	s[0] = uint32(geom.CeilToInt(xformedSigma - 1.0/6.0))
	rx := uint32(geom.CeilToInt(rrectToDraw.RadiusX))
	ry := uint32(geom.CeilToInt(rrectToDraw.RadiusY))
	for i := 0; i < 4; i++ {
		s[1+i*2] = rx
		s[2+i*2] = ry
	}
	kb.Finish()
}

// makeRRectBlurMaskFP looks the blurred-rrect nine-patch mask up in the thread-safe cache, and on a miss builds it on
// the CPU (gpu.CreateRRectBlurMask), uploads it, and adds the resulting view. Returns a texture effect that samples it,
// scaled by the mask dimensions so the shader's normalized coords address it, with the mask sampled by nearest
// filtering. A GPU fill-in lane (as opposed to CPU generation) is not implemented (see threadsafecache.go).
func makeRRectBlurMaskFP(ctx *DirectContext, rrectToDraw geom.RRect, dimensions geom.ISize, xformedSigma float32) FragmentProcessor {
	var key gpu.UniqueKey
	makeBlurredRRectKey(&key, rrectToDraw, xformedSigma)
	threadSafeCache := ctx.ThreadSafeCache()

	m := geom.ScaleMatrix(float32(dimensions.Width), float32(dimensions.Height))

	if view := threadSafeCache.Find(&key); view.IsValid() {
		return MakeTextureEffectSimple(view, gpu.AlphaTypePremul, m, gpu.FilterModeNearest,
			gpu.MipmapModeNone)
	}

	mask := gpu.CreateRRectBlurMask(rrectToDraw, dimensions, xformedSigma)
	if len(mask) == 0 {
		return nil
	}
	view := a8DataToTextureView(ctx, mask, int(dimensions.Width), dimensions, "Blurred RRect Mask")
	if view.Proxy() == nil {
		return nil
	}
	view = threadSafeCache.Add(&key, view)

	return MakeTextureEffectSimple(view, gpu.AlphaTypePremul, m, gpu.FilterModeNearest,
		gpu.MipmapModeNone)
}

// MakeRRectBlur builds an FP that analytically blurs a simple-circular rrect via a nine-patch mask. srcRRect and
// devRRect are the source- and device-space rrects (devRRect is guaranteed by the caller to be neither a rect nor a
// circle). Returns nil when the rrect is not simple-circular, the blur is a no-op, or the rrect is not nine-patchable.
func MakeRRectBlur(ctx *DirectContext, sigma, xformedSigma float32, srcRRect, devRRect geom.RRect) FragmentProcessor {
	// Require devRRect to be simple-circular: this could in principle be loosened, but for the uniform-radii subset
	// this codebase supports, a simple type with near-equal radii suffices.
	if devRRect.Type != geom.RRectSimple ||
		!geom.ScalarNearlyEqual(devRRect.RadiusX, devRRect.RadiusY) {
		return nil
	}
	if gpu.BlurIsEffectivelyIdentity(xformedSigma) {
		return nil
	}

	// Make sure we can nine-patch this rrect (the blur has to be small enough relative to both the corner radius and
	// the rrect's width and height).
	rrectToDraw, dimensions, ninePatchable := gpu.ComputeBlurredRRectParams(srcRRect, devRRect, sigma,
		xformedSigma)
	if !ninePatchable {
		return nil
	}

	maskFP := makeRRectBlurMaskFP(ctx, rrectToDraw, dimensions, xformedSigma)
	if maskFP == nil {
		return nil
	}

	cornerRadius := devRRect.RadiusX
	blurRadius := 3 * geom.CeilToScalar(xformedSigma-1.0/6.0)
	proxyRect := devRRect.Rect.Outset(blurRadius, blurRadius)

	rrectBlurFP := NewRRectBlurFP(cornerRadius, proxyRect, blurRadius, maskFP)
	// Modulate blur with the input color.
	return BlendFP(rrectBlurFP, nil, raster.BlendModulate)
}
