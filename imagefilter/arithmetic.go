// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The arithmetic blend image filter node: combines a background and foreground filter with the four-coefficient
// equation k1*fg*bg + k2*fg + k3*bg + k4, optionally enforcing premultiplied output. The node carries the arithmetic
// coefficients directly, and nearly-a-blend-mode coefficient sets (Src/Dst/Clear) collapse to the corresponding
// short-circuit filter.

package imagefilter

import (
	"github.com/richardwilkes/canvas/filtercore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/shaders"
)

type arithmeticFilter struct {
	filterDefaults
	k             [4]float32
	enforcePremul bool
}

// Input image filter indices.
const (
	arithmeticBackground = 0
	arithmeticForeground = 1
)

// Arithmetic builds an image filter that blends background and foreground with the four arithmetic coefficients k1..k4,
// optionally cropped to cropRect.
func Arithmetic(k1, k2, k3, k4 float32, enforcePMColor bool, background, foreground filtercore.Filter, cropRect *geom.Rect) filtercore.Filter {
	if !geom.IsFinite(k1, k2, k3, k4) {
		// Non-finite coefficients are an error, not an opportunity to optimize toward src-over.
		return nil
	}

	cropped := func(filter filtercore.Filter) filtercore.Filter {
		if cropRect != nil {
			filter = Crop(*cropRect, shaders.TileDecal, filter)
		}
		return filter
	}

	// Coefficients that are nearly one of the Src/Dst/Clear blend modes collapse directly to the corresponding filter
	// instead of running the general arithmetic equation.
	nearly := func(v, target float32) bool {
		return absf(v-target) <= geom.ScalarNearlyZeroTol
	}
	switch {
	case nearly(k1, 0) && nearly(k2, 1) && nearly(k3, 0) && nearly(k4, 0): // kSrc
		return cropped(foreground)
	case nearly(k1, 0) && nearly(k2, 0) && nearly(k3, 1) && nearly(k4, 0): // kDst
		return cropped(background)
	case nearly(k1, 0) && nearly(k2, 0) && nearly(k3, 0) && nearly(k4, 0): // kClear
		return Empty()
	}

	f := &arithmeticFilter{k: [4]float32{k1, k2, k3, k4}, enforcePremul: enforcePMColor}
	f.base = filtercore.NewFilterBase(background, foreground)
	return cropped(f)
}

// OnCTMCapability reports that arithmetic blending is independent of the layer matrix.
func (f *arithmeticFilter) OnCTMCapability() filtercore.MatrixCapability {
	return filtercore.MatrixCapabilityComplex
}

// OnAffectsTransparentBlack reports that the arithmetic equation affects transparent black when k4 != 0 (the constant
// term floods the output).
func (f *arithmeticFilter) OnAffectsTransparentBlack() bool { return f.k[3] != 0 }

// GPUEvaluable implements filtercore.GPUEvaluable: the arithmetic blend kernel converts to an FP
// (gpu/gl/filterkernelfp.go).
func (f *arithmeticFilter) GPUEvaluable() bool { return true }

// makeBlendShader builds the shader implementing the arithmetic equation over bg and fg. A nil input shader signifies
// transparent black; unlike a blend-mode filter, there is no single-input passthrough shortcut here, since the general
// equation always needs both operands.
func (f *arithmeticFilter) makeBlendShader(bg, fg shaders.Shader) shaders.Shader {
	if bg == nil || fg == nil {
		if f.k[3] == 0 && bg == nil && fg == nil {
			// Both inputs are transparent black and the blend leaves that untouched.
			return nil
		}
		if bg == nil {
			bg = shaders.NewColor(0)
		}
		if fg == nil {
			fg = shaders.NewColor(0)
		}
	}
	return shaders.NewArithmeticBlend(f.k, f.enforcePremul, bg, fg)
}

// OnFilterImage evaluates both children over the maximum possible output (derived from the content bounds) so that
// coefficients which restrict the output see the full extent of each input.
//
//nolint:gocritic // hugeParam: ctx is by value per the interface; see filtercore.Filter.OnFilterImage
func (f *arithmeticFilter) OnFilterImage(ctx filtercore.Context) filtercore.FilterResult {
	srcBounds := ctx.Source().LayerBounds()
	requiredInput := f.OnOutputLayerBounds(ctx.Mapping(), &srcBounds)
	if requiredInput != nil {
		if !requiredInput.Intersect(ctx.DesiredOutput()) {
			return filtercore.FilterResult{}
		}
	} else {
		out := ctx.DesiredOutput()
		requiredInput = &out
	}

	inputCtx := ctx.WithNewDesiredOutput(*requiredInput)
	builder := filtercore.NewBuilder(&ctx)
	background := f.base.ChildOutput(arithmeticBackground, &inputCtx)
	builder.Add(&background)
	foreground := f.base.ChildOutput(arithmeticForeground, &inputCtx)
	builder.Add(&foreground)
	return builder.Eval(func(inputs []shaders.Shader) shaders.Shader {
		return f.makeBlendShader(inputs[arithmeticBackground], inputs[arithmeticForeground])
	}, requiredInput)
}

// OnInputLayerBounds reports the union of the background's and foreground's required input bounds, restricted to the
// portion of the desired output the arithmetic equation can actually produce.
func (f *arithmeticFilter) OnInputLayerBounds(mapping *filtercore.Mapping, desiredOutput geom.IRect, contentBounds *geom.IRect) geom.IRect {
	var requiredInput geom.IRect
	var maxOutput *geom.IRect
	if contentBounds != nil {
		maxOutput = f.OnOutputLayerBounds(mapping, contentBounds)
	}
	if maxOutput != nil {
		requiredInput = *maxOutput
		if !requiredInput.Intersect(desiredOutput) {
			// The blend will discard everything, so don't bother recursing.
			return geom.IRect{}
		}
	} else {
		// The content and/or child output are unbounded, so the intersection with the desired output is simply the
		// desired output.
		requiredInput = desiredOutput
	}

	// The union of both FG and BG required inputs ensures both have all necessary pixels.
	bgInput := f.base.InputLayerBounds(arithmeticBackground, mapping, requiredInput, contentBounds)
	fgInput := f.base.InputLayerBounds(arithmeticForeground, mapping, requiredInput, contentBounds)
	bgInput.Join(fgInput)
	return bgInput
}

// OnOutputLayerBounds reports the arithmetic equation's output extent: k4 != 0 floods everywhere; otherwise
// transparency outside the foreground requires k3 == 0 and outside the background requires k2 == 0, and the output is
// the intersection, one side, or the union of the child bounds accordingly.
func (f *arithmeticFilter) OnOutputLayerBounds(mapping *filtercore.Mapping, contentBounds *geom.IRect) *geom.IRect {
	if f.k[3] != 0 {
		// The arithmetic equation produces non-transparent black everywhere.
		return nil
	}
	transparentOutsideFG := f.k[2] == 0
	transparentOutsideBG := f.k[1] == 0

	foregroundBounds := f.base.OutputLayerBounds(arithmeticForeground, mapping, contentBounds)
	backgroundBounds := f.base.OutputLayerBounds(arithmeticBackground, mapping, contentBounds)
	if transparentOutsideFG {
		if transparentOutsideBG {
			// Output is the intersection of both.
			switch {
			case foregroundBounds == nil:
				foregroundBounds = backgroundBounds
			case backgroundBounds != nil && !foregroundBounds.Intersect(*backgroundBounds):
				empty := geom.IRect{}
				return &empty
			}
		}
		return foregroundBounds
	}
	if !transparentOutsideBG {
		// Output is the union of both (blend-induced infinite bounds were detected earlier).
		if foregroundBounds != nil && backgroundBounds != nil {
			backgroundBounds.Join(*foregroundBounds)
		} else {
			backgroundBounds = nil
		}
	}
	return backgroundBounds
}

// OnComputeFastBounds applies the same coefficient-driven extent logic as OnOutputLayerBounds, but over the
// conservative (unmapped) bounds convention used by the fast-bounds pass.
func (f *arithmeticFilter) OnComputeFastBounds(bounds geom.Rect) geom.Rect {
	if f.k[3] != 0 {
		return filtercore.MakeILarge().ToRect()
	}
	transparentOutsideFG := f.k[2] == 0
	transparentOutsideBG := f.k[1] == 0

	foregroundBounds := bounds
	if in := f.base.Input(arithmeticForeground); in != nil {
		foregroundBounds = in.OnComputeFastBounds(bounds)
	}
	backgroundBounds := bounds
	if in := f.base.Input(arithmeticBackground); in != nil {
		backgroundBounds = in.OnComputeFastBounds(bounds)
	}
	if transparentOutsideFG {
		if transparentOutsideBG {
			if !foregroundBounds.Intersect(backgroundBounds) {
				return geom.Rect{}
			}
		}
		return foregroundBounds
	}
	if !transparentOutsideBG {
		backgroundBounds.Join(foregroundBounds)
	}
	return backgroundBounds
}
