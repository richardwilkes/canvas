// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The SVG feTurbulence fractal-noise / turbulence shader as a GPU fragment processor. The two painting-data tables ride
// texture-effect child FPs — an A8 256x1 permutations texture and an RGBA8 256x4 noise texture (channel per row, each
// texel two 16-bit gradient components split high/low across the RGBA bytes) — sampled RepeatX/Nearest; the octave loop
// and the noise helper function are emitted as desktop GLSL directly. No perlin-noise rounding fix (a workaround for an
// old mobile-GPU quirk that doesn't apply here). Before this lane a paint carrying a perlin-noise shader failed FP
// conversion and the GPU device drew nothing.

package gl

import (
	"fmt"
	"strings"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/shaders"
)

// perlinBlockSizePx is the width, in pixels, of both data textures.
const perlinBlockSizePx = 256

// perlinProxyKeyDomain keys the cached permutations/noise texture proxies (per-shader, per-table).
var perlinProxyKeyDomain = gpu.GenerateUniqueKeyDomain()

// makePerlinNoiseFP uploads the two painting-data tables to cached textures, builds the perlin-noise fragment processor
// over them, and wraps it in the local-to-device matrix effect. Returns nil on a failed upload or a singular total
// matrix.
func makePerlinNoiseFP(s *shaders.PerlinNoiseShader, args *FPArgs, mRec *shaders.MatrixRec) FragmentProcessor {
	// numOctaves != 0 is guaranteed: NewFractalNoise/NewTurbulence collapse the octave-0 case to a color shader, so a
	// *PerlinNoiseShader always has at least one octave.
	permView := perlinTextureView(args.Ctx, s, 0,
		geom.ISize{Width: perlinBlockSizePx, Height: 1}, gpu.ColorTypeAlpha8,
		s.PermutationsBitmap(), perlinBlockSizePx, "PerlinNoiseShader_Permutations")
	noiseView := perlinTextureView(args.Ctx, s, 1,
		geom.ISize{Width: perlinBlockSizePx, Height: 4}, gpu.ColorTypeRGBA8888,
		s.NoiseBitmap(), perlinBlockSizePx*4, "PerlinNoiseShader_Noise")
	if permView.Proxy() == nil || noiseView.Proxy() == nil {
		return nil
	}
	fp := newPerlinNoise2FP(s, permView, noiseView, args.Caps)
	identity := geom.IdentityMatrix()
	total, ok := mRec.ApplyForFragmentProcessor(&identity)
	if !ok {
		return nil
	}
	return MatrixEffectFP(total, fp)
}

// perlinTextureView returns the uniquely-keyed proxy view for one of the shader's data tables (which=0 permutations,
// which=1 noise), caching per-bitmap so repeated draws reuse the same uploaded texture.
func perlinTextureView(ctx *DirectContext, s *shaders.PerlinNoiseShader, which uint32, dims geom.ISize, colorType gpu.ColorType, pixels []byte, rowBytes int, label string) SurfaceProxyView {
	var key gpu.UniqueKey
	builder := gpu.UniqueKeyBuilder(&key, perlinProxyKeyDomain, 2, label)
	builder.Slice()[0] = s.UniqueID()
	builder.Slice()[1] = which
	builder.Finish()
	return FindOrCreatePixelsProxyView(ctx, &key, dims, colorType, pixels, rowBytes, label)
}

// perlinNoise2FP is the fragment processor for perlin noise: the descriptive fields plus two texture-effect children
// (permutations at index 0, noise at index 1) sampled explicitly.
type perlinNoise2FP struct {
	FPBase
	numOctaves  int
	baseFreqX   float32
	baseFreqY   float32
	stitchW     int32
	stitchH     int32
	noiseType   shaders.PerlinNoiseType
	stitchTiles bool
}

// newPerlinNoise2FP builds the two RepeatX/Nearest texture effects (identity sampling matrix) and registers them as
// explicitly-sampled children.
func newPerlinNoise2FP(s *shaders.PerlinNoiseShader, permView, noiseView SurfaceProxyView, caps *Caps) FragmentProcessor {
	sampler := gpu.MakeSamplerState(gpu.WrapModeRepeat, gpu.WrapModeClamp, gpu.FilterModeNearest,
		gpu.MipmapModeNone)
	var border [4]float32
	permFP := MakeTextureEffect(permView, gpu.AlphaTypePremul, geom.IdentityMatrix(), sampler, caps,
		border)
	noiseFP := MakeTextureEffect(noiseView, gpu.AlphaTypePremul, geom.IdentityMatrix(), sampler,
		caps, border)
	fx, fy := s.BaseFrequency()
	sw, sh := s.StitchData()
	fp := &perlinNoise2FP{
		noiseType:   s.NoiseType(),
		numOctaves:  s.NumOctaves(),
		stitchTiles: s.StitchTiles(),
		baseFreqX:   fx,
		baseFreqY:   fy,
		stitchW:     sw,
		stitchH:     sh,
	}
	fp.initFP(PerlinNoise2EffectClassID, 0) // kNone_OptimizationFlags
	registerChildOf(fp, permFP, ExplicitSampleUsage())
	registerChildOf(fp, noiseFP, ExplicitSampleUsage())
	fp.setUsesSampleCoordsDirectly()
	return fp
}

func (f *perlinNoise2FP) Name() string { return "PerlinNoise" }

func (f *perlinNoise2FP) Clone() FragmentProcessor {
	clone := &perlinNoise2FP{
		noiseType:   f.noiseType,
		numOctaves:  f.numOctaves,
		stitchTiles: f.stitchTiles,
		baseFreqX:   f.baseFreqX,
		baseFreqY:   f.baseFreqY,
		stitchW:     f.stitchW,
		stitchH:     f.stitchH,
	}
	clone.initFP(PerlinNoise2EffectClassID, f.optimizationFlags())
	cloneAndRegisterAllChildProcessors(clone, f)
	clone.setUsesSampleCoordsDirectly()
	return clone
}

// onAddToKey packs the shader key: numOctaves in the high bits, the type in the low two bits, and the stitch flag in
// the third bit.
func (f *perlinNoise2FP) onAddToKey(_ *gpu.ShaderCaps, b *gpu.KeyBuilder) {
	key := uint32(f.numOctaves)
	key <<= 3 // make room for the next 3 bits
	switch f.noiseType {
	case shaders.PerlinFractalNoise:
		key |= 0x1
	case shaders.PerlinTurbulence:
		key |= 0x2
	}
	if f.stitchTiles {
		key |= 0x4
	}
	b.Add32(key, "perlinKey")
}

func (f *perlinNoise2FP) onIsEqual(other FragmentProcessor) bool {
	o, ok := other.(*perlinNoise2FP)
	return ok && f.noiseType == o.noiseType && f.baseFreqX == o.baseFreqX &&
		f.baseFreqY == o.baseFreqY && f.numOctaves == o.numOctaves &&
		f.stitchTiles == o.stitchTiles && f.stitchW == o.stitchW && f.stitchH == o.stitchH
}

func (f *perlinNoise2FP) onMakeProgramImpl() FPProgramImpl { return &perlinNoise2Impl{} }

type perlinNoise2Impl struct {
	FPImplBase
	baseFrequencyUni UniformHandle
	stitchDataUni    UniformHandle
}

// emitHelper emits the per-channel noise function (sampling the two data textures via the child FPs) and returns its
// mangled name, using desktop GLSL float/vec2/vec4 types.
func (i *perlinNoise2Impl) emitHelper(args *FPEmitArgs) string {
	pne := args.FP.(*perlinNoise2FP)
	fb := args.FragBuilder

	var noiseCode strings.Builder
	noiseCode.WriteString("vec4 floorVal;" +
		"floorVal.xy = floor(noiseVec);" +
		"floorVal.zw = floorVal.xy + vec2(1.0);" +
		"vec2 fractVal = fract(noiseVec);" +
		// Hermite interpolation: t^2*(3 - 2*t).
		"vec2 noiseSmooth = smoothstep(0.0, 1.0, fractVal);")

	// Adjust frequencies if we're stitching tiles.
	if pne.stitchTiles {
		noiseCode.WriteString("floorVal -= step(stitchData.xyxy, floorVal) * stitchData.xyxy;")
	}

	// Sample the permutations texture. vec4(1.0) is passed explicitly as the input color because the helper function
	// cannot see the FP's own input color.
	sampleX := i.InvokeChildWithCoords(0, "vec4(1.0)", args, "vec2(floorVal.x + 0.5, 0.5)")
	sampleY := i.InvokeChildWithCoords(0, "vec4(1.0)", args, "vec2(floorVal.z + 0.5, 0.5)")
	fmt.Fprintf(&noiseCode, "vec2 latticeIdx = vec2(%s.a, %s.a);", sampleX, sampleY)

	// Get (x,y) coordinates with the permuted x.
	noiseCode.WriteString("vec4 bcoords = 256.0*latticeIdx.xyxy + floorVal.yyww;")

	// Convert the two 16-bit integers packed into the RGBA8 texel into a [-1,1] vector and dot it with the provided
	// vector. Reused 4x. inc8bit = 1.0 / 256.0.
	const inc8bit = "0.00390625"
	dotLattice := fmt.Sprintf("dot((lattice.ga + lattice.rb*%s)*2.0 - vec2(1.0), fractVal)", inc8bit)

	// Sample the noise texture at the four lattice corners.
	sampleA := i.InvokeChildWithCoords(1, "vec4(1.0)", args, "vec2(bcoords.x, chanCoord)")
	sampleB := i.InvokeChildWithCoords(1, "vec4(1.0)", args, "vec2(bcoords.y, chanCoord)")
	sampleC := i.InvokeChildWithCoords(1, "vec4(1.0)", args, "vec2(bcoords.w, chanCoord)")
	sampleD := i.InvokeChildWithCoords(1, "vec4(1.0)", args, "vec2(bcoords.z, chanCoord)")

	// Compute u, at offset (0,0).
	fmt.Fprintf(&noiseCode, "vec4 lattice = %s;", sampleA)
	fmt.Fprintf(&noiseCode, "float u = %s;", dotLattice)
	// Compute v, at offset (-1,0).
	noiseCode.WriteString("fractVal.x -= 1.0;")
	fmt.Fprintf(&noiseCode, "lattice = %s;", sampleB)
	fmt.Fprintf(&noiseCode, "float v = %s;", dotLattice)
	// Compute 'a' as a linear interpolation of 'u' and 'v'.
	noiseCode.WriteString("float a = mix(u, v, noiseSmooth.x);")
	// Compute v, at offset (-1,-1).
	noiseCode.WriteString("fractVal.y -= 1.0;")
	fmt.Fprintf(&noiseCode, "lattice = %s;", sampleC)
	fmt.Fprintf(&noiseCode, "v = %s;", dotLattice)
	// Compute u, at offset (0,-1).
	noiseCode.WriteString("fractVal.x += 1.0;")
	fmt.Fprintf(&noiseCode, "lattice = %s;", sampleD)
	fmt.Fprintf(&noiseCode, "u = %s;", dotLattice)
	// Compute 'b' as a linear interpolation of 'u' and 'v'.
	noiseCode.WriteString("float b = mix(u, v, noiseSmooth.x);")
	// Compute the noise as a linear interpolation of 'a' and 'b'.
	noiseCode.WriteString("return mix(a, b, noiseSmooth.y);")

	noiseFuncName := fb.GetMangledFunctionName("noiseFuncName")
	funcArgs := []ShaderVar{
		NewShaderVar("chanCoord", GLSLTypeHalf),
		NewShaderVar("noiseVec", GLSLTypeHalf2),
	}
	if pne.stitchTiles {
		funcArgs = append(funcArgs, NewShaderVar("stitchData", GLSLTypeHalf2))
	}
	fb.EmitFunction(GLSLTypeHalf, noiseFuncName, funcArgs, noiseCode.String())
	return noiseFuncName
}

// EmitCode emits the octave-accumulation loop over the noise helper. Perlin noise adds 0.5 to the local coordinate; the
// CPU raster stage does the same, so both agree per pixel.
func (i *perlinNoise2Impl) EmitCode(args *FPEmitArgs) {
	noiseFuncName := i.emitHelper(args)

	pne := args.FP.(*perlinNoise2FP)
	fb := args.FragBuilder
	uh := args.UniformHandler

	var baseFrequencyName string
	i.baseFrequencyUni, baseFrequencyName = uh.AddUniform(args.FP, ShaderFlagFragment,
		GLSLTypeHalf2, "baseFrequency")

	var stitchDataName string
	if pne.stitchTiles {
		i.stitchDataUni, stitchDataName = uh.AddUniform(args.FP, ShaderFlagFragment, GLSLTypeHalf2,
			"stitchData")
	}

	fb.CodeAppendf("vec2 noiseVec = vec2((%s + 0.5) * %s);", args.SampleCoord, baseFrequencyName)
	fb.CodeAppend("vec4 color = vec4(0.0);")
	if pne.stitchTiles {
		fb.CodeAppendf("vec2 stitchData = %s;", stitchDataName)
	}
	fb.CodeAppend("float ratio = 1.0;")

	// Loop over all octaves.
	fb.CodeAppendf("for (int octave = 0; octave < %d; ++octave) {", pne.numOctaves)
	fb.CodeAppend("color += ")
	if pne.noiseType != shaders.PerlinFractalNoise {
		fb.CodeAppend("abs(")
	}
	// There are 4 lines (channels); put the y coordinate at the center of each.
	if pne.stitchTiles {
		fb.CodeAppendf("vec4(%[1]s(0.5, noiseVec, stitchData), %[1]s(1.5, noiseVec, stitchData),"+
			"%[1]s(2.5, noiseVec, stitchData), %[1]s(3.5, noiseVec, stitchData))", noiseFuncName)
	} else {
		fb.CodeAppendf("vec4(%[1]s(0.5, noiseVec), %[1]s(1.5, noiseVec),"+
			"%[1]s(2.5, noiseVec), %[1]s(3.5, noiseVec))", noiseFuncName)
	}
	if pne.noiseType != shaders.PerlinFractalNoise {
		fb.CodeAppend(")") // end of "abs("
	}
	fb.CodeAppend(" * ratio;")

	fb.CodeAppend("noiseVec *= vec2(2.0);ratio *= 0.5;")
	if pne.stitchTiles {
		fb.CodeAppend("stitchData *= vec2(2.0);")
	}
	fb.CodeAppend("}") // end of the octave loop

	if pne.noiseType == shaders.PerlinFractalNoise {
		// Fractal noise maps turbulence [-1,1] into [0,1] via (x + 1) / 2.
		fb.CodeAppend("color = color * vec4(0.5) + vec4(0.5);")
	}
	// Clamp to [0,1].
	fb.CodeAppend("color = clamp(color, 0.0, 1.0);")
	// Premultiply the result.
	fb.CodeAppend("return vec4(color.rgb * color.aaa, color.a);")
}

func (i *perlinNoise2Impl) onSetData(pdman *ProgramDataManager, fp FragmentProcessor) {
	pne := fp.(*perlinNoise2FP)
	pdman.Set2f(i.baseFrequencyUni, pne.baseFreqX, pne.baseFreqY)
	if pne.stitchTiles {
		pdman.Set2f(i.stitchDataUni, float32(pne.stitchW), float32(pne.stitchH))
	}
}
