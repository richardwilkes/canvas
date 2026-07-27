// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The span evaluation engine: the highp raster-pipeline stages the shaders assemble, run over chunks of up to 16 pixels
// with consistent per-lane math throughout. A Pipeline instance carries per-stage scratch (decal masks, blend-shader
// buffers), so it must not be shared across goroutines.

package shaders

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
)

// stride is the pipeline's chunk width. Every stage is lane-wise (the dither and seed stages use absolute pixel
// coordinates), so the chunk width never changes results.
const stride = 16

// pm4f is shorthand for the premultiplied float color the pipeline traffics in.
type pm4f = colorcore.PMColor4f

// lanes is the highp register file: src registers r,g,b,a and dst registers dr,dg,db,da, plus the chunk's device
// position.
type lanes struct {
	// ctx points at the current stage's context (nil for context-free stages). ShadeSpan sets it before invoking each
	// stage, pairing a function with its data — so a parameterized stage reads its data from reused pipeline storage
	// instead of a captured (heap-allocated) closure.
	ctx any
	n   int
	r   [stride]float32
	g   [stride]float32
	b   [stride]float32
	a   [stride]float32
	dr  [stride]float32
	dg  [stride]float32
	db  [stride]float32
	da  [stride]float32
	dx  int32
	dy  int32
}

type stageFn func(z *lanes)

// stage pairs a stage function with its context. A context-free stage carries a nil ctx and ignores z.ctx; ShadeSpan
// points z.ctx at ctx before it invokes fn.
type stage struct {
	fn  stageFn
	ctx any
}

// Pipeline is a compiled color pipeline: the shader stages plus whatever the blitter assembly appends (paint alpha,
// dither). paintColor is the unpremultiplied paint color the alpha-only image lane tints with, set by Compile before
// stages are appended.
type Pipeline struct {
	// gatherCtxs/samplerCtxs/decalCtxs hold the image-shader sampling contexts (one gatherCtx per image level, one
	// samplerCtx per bilinear/bicubic image, one decalTileCtx per decal-tiled image; a BlendShader can inline two image
	// children), retained across pooled compiles so a repeated image draw rebuilds no sampling scratch. The gatherCtx
	// carries a *imagecore.Pixels — RecyclePipeline drops it so an idle pooled pipeline does not pin source pixels.
	gatherCtxs []*gatherCtx
	// laneMasks holds one laneMask per decal/conical masking stage this pipeline compiled (a decal tile mode allocates
	// one; each conical strip/degenerate lane allocates one; a BlendShader can inline two masked gradients), retained
	// across pooled compiles so a repeated masked draw rebuilds no mask storage. nextLaneMask hands out a distinct one
	// per stage, matching the matrixCtxs/gradCtxs discipline. The mask is pure scratch — each ShadeSpan chunk fully
	// writes it before its consumer reads — so reuse needs no zeroing.
	laneMasks  []*laneMask
	normalCtxs []*normalCtx
	// gradCtxs holds one gradientCtx per gradient fill stage this pipeline compiled (a BlendShader can inline two
	// gradient children, so a single shared ctx would clobber; nextGradCtx hands out a distinct one). They are retained
	// across pooled compiles (their factor/bias/ts slices are reused), so a steady-state gradient draw allocates no
	// fill-stage storage after warm-up.
	gradCtxs    []*gradientCtx
	matConvCtxs []*matrixConvCtx
	// matrixCtxs holds one matrixCtx per matrix stage this pipeline compiled (the base gradient/shader matrix, wrapper
	// local matrices, and each blended child add their own), retained across pooled compiles so a repeated shaded draw
	// rebuilds no matrix-stage storage. nextMatrixCtx hands out a distinct one per stage, matching the gradCtxs
	// discipline.
	matrixCtxs   []*matrixCtx
	lightingCtxs []*lightingCtx
	// postStages collects the deferred tail stages a concrete gradient records (conical's apply_vector_mask, which must
	// run after the fill); appendBaseStages splices them onto stages after the fill and rewinds. Retained for reuse.
	// Threading them through a reused field (rather than a local slice whose address is passed to the gradStages
	// callback) keeps the slice header off the heap.
	postStages  []stage
	samplerCtxs []*samplerCtx
	arithCtxs   []*arithmeticCtx
	stages      []stage
	// blendCtxs holds one blendCtx per blend stage this pipeline compiled — a BlendShader and the blend-mode color
	// filter each append one — retained across pooled compiles.
	blendCtxs []*blendCtx
	// colorFuncCtxs holds one colorFuncCtx per AppendColorFunc stage (the runtime-effect color filters), retained
	// across pooled compiles.
	colorFuncCtxs []*colorFuncCtx
	dispCtxs      []*displacementCtx
	// constColorCtxs holds one color per AppendConstantColor stage (a ColorShader per sprite in the DrawAtlas color
	// lane; a blend tree can inline several), retained across pooled compiles so a repeated constant-color draw appends
	// a static stage instead of heap-allocating a capturing closure per compile. Values only — no external references
	// to drop on recycle.
	constColorCtxs []*colorcore.PMColor4f
	// The image-filter runtime-kernel contexts (filterkernels.go): one per morphology/displacement/
	// normal/lighting/matrix-convolution/arithmetic shader this pipeline compiled (a shader tree can nest kernels, so
	// each nextXxxCtx hands out a distinct one), retained across pooled compiles so a repeated filter draw rebuilds no
	// per-kernel scratch. Each holds only values + inline register-file scratch (no external references), so
	// RecyclePipeline needs only counter resets.
	morphCtxs []*morphologyCtx
	decalCtxs []*decalTileCtx
	// blendShaderCtxs holds one blendShaderCtx per BlendShader this pipeline compiled (a nested blend shader inlines
	// another, so a single shared ctx would clobber; nextBlendShaderCtx hands out a distinct one), retained across
	// pooled compiles so a repeated blend draw rebuilds no store/load scratch.
	blendShaderCtxs []*blendShaderCtx
	// z is the reusable highp register file. Like the decal/blend scratch above, it makes a Pipeline single-goroutine;
	// holding it here keeps ShadeSpan from heap-allocating a ~half-KB lanes struct on every scanline.
	z               lanes
	dispCtxN        int
	laneMaskN       int
	samplerCtxN     int
	blendShaderCtxN int
	decalCtxN       int
	constColorCtxN  int
	gatherCtxN      int
	morphCtxN       int
	colorFuncCtxN   int
	normalCtxN      int
	blendCtxN       int
	matrixCtxN      int
	lightingCtxN    int
	gradCtxN        int
	matConvCtxN     int
	arithCtxN       int
	paintColor      colorcore.Color4f
}

func (p *Pipeline) append(fn stageFn) { p.stages = append(p.stages, stage{fn: fn}) }

// appendCtx appends a parameterized stage: ShadeSpan points z.ctx at ctx before calling fn, so fn is a static
// (non-capturing) function rather than a per-draw heap-allocated closure.
func (p *Pipeline) appendCtx(fn stageFn, ctx any) {
	p.stages = append(p.stages, stage{fn: fn, ctx: ctx})
}

// appendPostCtx records a deferred, parameterized tail stage: like appendCtx (fn reads z.ctx) but deferred to
// postStages (see Pipeline.postStages).
func (p *Pipeline) appendPostCtx(fn stageFn, ctx any) {
	p.postStages = append(p.postStages, stage{fn: fn, ctx: ctx})
}

// nextLaneMask hands a decal/conical masking stage a distinct laneMask, retained on the pipeline so its storage is
// reused across pooled compiles (a shader tree can append several masking stages; a single shared mask would clobber).
// The mask is written in full before it is read within each ShadeSpan chunk, so reuse needs no zeroing.
func (p *Pipeline) nextLaneMask() *laneMask {
	if p.laneMaskN == len(p.laneMasks) {
		p.laneMasks = append(p.laneMasks, new(laneMask))
	}
	m := p.laneMasks[p.laneMaskN]
	p.laneMaskN++
	return m
}

// pipelinePool retains Pipeline storage — the ~half-KB lanes register file, the stages slice backing array, and the
// per-gradient fill-stage scratch — across draws so that a repeated shaded draw does not rebuild that storage every
// time. Compile borrows; the caller hands the compiled pipeline back with RecyclePipeline once the blitter that
// consumed it is done. Because a ShaderBlitter (and its pipeline) is fully consumed by the synchronous fill before the
// draw returns, recycling on draw exit is safe; sync.Pool keeps nested/re-entrant draws independent.
var pipelinePool = sync.Pool{New: func() any { return &Pipeline{} }}

// borrowPipeline returns a reset Pipeline drawn from the shared pool.
func borrowPipeline() *Pipeline {
	return pipelinePool.Get().(*Pipeline)
}

// RecyclePipeline returns a Pipeline obtained from Compile to the shared pool. It drops the compiled stage closures
// (they capture per-draw data) and every reference to per-draw external data the stages or their contexts held, so an
// idle pooled pipeline pins nothing, but it retains the register-file, stages, and gradient scratch storage for reuse.
// After RecyclePipeline the caller must treat p as invalid.
func RecyclePipeline(p *Pipeline) {
	if p == nil {
		return
	}
	clear(p.stages)
	p.stages = p.stages[:0]
	clear(p.postStages)
	p.postStages = p.postStages[:0]
	// The register file's ctx still points at the last-executed stage's context, which for a stage parameterized with
	// external data (a *PerlinNoiseShader and its ~8KB tables, a color filter's *[20]float32 matrix, a *ConicalGradient)
	// would outlive the draw. ShadeSpan rewrites it per stage, so dropping it costs nothing.
	p.z.ctx = nil
	p.paintColor = colorcore.Color4f{}
	p.gradCtxN = 0
	p.matrixCtxN = 0
	p.laneMaskN = 0
	p.blendCtxN = 0
	// Drop the retained color-transform closures: a runtime-effect color filter's fn captures that filter's per-draw
	// uniforms, so keeping it would pin them (the ctx storage itself is retained for reuse).
	for _, c := range p.colorFuncCtxs {
		c.fn = nil
	}
	p.colorFuncCtxN = 0
	p.blendShaderCtxN = 0
	p.constColorCtxN = 0
	// Drop the source-pixel references the image-shader gather contexts hold so an idle pooled pipeline does not pin
	// image pixels (the ctx storage itself is retained for reuse). The sampler/decal contexts hold no external
	// references.
	for _, g := range p.gatherCtxs {
		g.px = nil
	}
	p.gatherCtxN = 0
	p.samplerCtxN = 0
	p.decalCtxN = 0
	p.morphCtxN = 0
	p.dispCtxN = 0
	p.normalCtxN = 0
	p.lightingCtxN = 0
	p.matConvCtxN = 0
	p.arithCtxN = 0
	pipelinePool.Put(p)
}

// ShadeSpan evaluates the pipeline for the pixel run [x, x+len(dst)) on row y, writing premultiplied float colors. It
// runs the compiled pipeline over one row: chunks of stride pixels with dx advancing, identical per-lane math
// throughout.
func (p *Pipeline) ShadeSpan(x, y int32, dst []colorcore.PMColor4f) {
	z := &p.z
	for off := 0; off < len(dst); off += stride {
		n := len(dst) - off
		if n > stride {
			n = stride
		}
		z.dx = x + int32(off)
		z.dy = y
		z.n = n
		for _, st := range p.stages {
			z.ctx = st.ctx
			st.fn(z)
		}
		for i := range n {
			dst[off+i] = colorcore.PMColor4f{R: z.r[i], G: z.g[i], B: z.b[i], A: z.a[i]}
		}
	}
}

// EvalConstant runs the color pipeline once at (0,0) with the 8888 normalization clamp and returns the constant color —
// the fast path for a shader known to produce the same color everywhere.
func (p *Pipeline) EvalConstant() colorcore.PMColor4f {
	var out [1]colorcore.PMColor4f
	p.ShadeSpan(0, 0, out[:])
	return colorcore.PMColor4f{
		R: clamp01(out[0].R),
		G: clamp01(out[0].G),
		B: clamp01(out[0].B),
		A: clamp01(out[0].A),
	}
}

///////////////////////////////////////////////////////////////////////////////
// lane math

// madf computes mad(f,m,a): fused, matching the raster package's fmaf decision.
func madf(f, m, a float32) float32 {
	return float32(math.FMA(float64(f), float64(m), float64(a)))
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// clamp01 computes min(max(v, 0), 1).
func clamp01(v float32) float32 {
	return minf(maxf(v, 0), 1)
}

func floorf(v float32) float32 { return float32(math.Floor(float64(v))) }

func sqrtf(v float32) float32 { return float32(math.Sqrt(float64(v))) }

func absf(v float32) float32 { return float32(math.Abs(float64(v))) }

///////////////////////////////////////////////////////////////////////////////
// coordinate stages

// iota05 is the per-lane x-offset table the seed stage adds to the chunk's base x coordinate.
var iota05 = [stride]float32{
	0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
	8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
}

// appendSeed appends the seed_shader stage, which seeds (r,g) with the chunk's device coordinates.
func (p *Pipeline) appendSeed() {
	p.append(seedStageFn)
}

// seedStage is the portable seed_shader stage; on arm64 seedStageFn points at the NEON form instead (stage_arm64.go),
// which writes the same lanes with the same per-lane rounding.
func seedStage(z *lanes) {
	fx := float32(uint32(z.dx))
	fy := float32(uint32(z.dy)) + 0.5
	for i := range z.n {
		z.r[i] = fx + iota05[i]
		z.g[i] = fy
		z.b[i] = 1 // w=1 for matrix multiplies by the device coords
		z.a[i] = 0
	}
}

// matrixCtx holds a matrix stage's 9 coefficients in geom.Matrix index order (As9). Each of the four matrix stages
// reads only the slots it needs, held in reused pipeline storage so the stage function need not capture.
type matrixCtx struct {
	m [9]float32
}

// nextMatrixCtx hands appendMatrix a distinct matrixCtx for one matrix stage, retained on the pipeline so its
// coefficient storage is reused across pooled compiles (a shader tree can append several matrix stages; a single shared
// ctx would clobber). It loads the coefficients from m.
func (p *Pipeline) nextMatrixCtx(m *geom.Matrix) *matrixCtx {
	if p.matrixCtxN == len(p.matrixCtxs) {
		p.matrixCtxs = append(p.matrixCtxs, &matrixCtx{})
	}
	c := p.matrixCtxs[p.matrixCtxN]
	p.matrixCtxN++
	c.m = m.As9()
	return c
}

// appendMatrix picks the matrix stage by type mask. The translate/scale-translate/affine stages dispatch through
// per-arch fn variables (NEON on arm64, the portable forms elsewhere); perspective stays portable.
func (p *Pipeline) appendMatrix(m *geom.Matrix) {
	mt := m.Type()
	switch {
	case mt == geom.TypeIdentity:
	case mt == geom.TypeTranslate:
		p.appendCtx(matrixTranslateStageFn, p.nextMatrixCtx(m))
	case mt|(geom.TypeScale|geom.TypeTranslate) == geom.TypeScale|geom.TypeTranslate:
		p.appendCtx(matrixScaleTranslateStageFn, p.nextMatrixCtx(m))
	case !m.HasPerspective():
		p.appendCtx(matrixAffineStageFn, p.nextMatrixCtx(m))
	default:
		p.appendCtx(matrixPerspStage, p.nextMatrixCtx(m))
	}
}

func matrixTranslateStage(z *lanes) {
	c := z.ctx.(*matrixCtx)
	tx, ty := c.m[geom.MTransX], c.m[geom.MTransY]
	for i := range z.n {
		z.r[i] += tx
		z.g[i] += ty
	}
}

func matrixScaleTranslateStage(z *lanes) {
	c := z.ctx.(*matrixCtx)
	sx, sy := c.m[geom.MScaleX], c.m[geom.MScaleY]
	tx, ty := c.m[geom.MTransX], c.m[geom.MTransY]
	for i := range z.n {
		z.r[i] = madf(z.r[i], sx, tx)
		z.g[i] = madf(z.g[i], sy, ty)
	}
}

func matrixAffineStage(z *lanes) {
	s := z.ctx.(*matrixCtx).m
	for i := range z.n {
		r := madf(z.r[i], s[0], madf(z.g[i], s[1], s[2]))
		g := madf(z.r[i], s[3], madf(z.g[i], s[4], s[5]))
		z.r[i] = r
		z.g[i] = g
	}
}

func matrixPerspStage(z *lanes) {
	s := z.ctx.(*matrixCtx).m
	for i := range z.n {
		r := madf(z.r[i], s[0], madf(z.g[i], s[1], s[2]))
		g := madf(z.r[i], s[3], madf(z.g[i], s[4], s[5]))
		zz := madf(z.r[i], s[6], madf(z.g[i], s[7], s[8]))
		// rcp_precise
		z.r[i] = r / zz
		z.g[i] = g / zz
	}
}

///////////////////////////////////////////////////////////////////////////////
// color stages

// AppendConstantColor appends a stage that splats the given premultiplied color into every lane.
func (p *Pipeline) AppendConstantColor(c colorcore.PMColor4f) {
	ctx := p.nextConstColorCtx()
	*ctx = c
	p.appendCtx(constantColorStage, ctx)
}

// constantColorStage is AppendConstantColor's static stage function (the color rides the pooled ctx, not a per-compile
// closure).
func constantColorStage(z *lanes) {
	c := z.ctx.(*colorcore.PMColor4f)
	for i := range z.n {
		z.r[i] = c.R
		z.g[i] = c.G
		z.b[i] = c.B
		z.a[i] = c.A
	}
}

// nextConstColorCtx hands an AppendConstantColor stage a distinct pooled color slot, matching the matrixCtxs/gradCtxs
// discipline.
func (p *Pipeline) nextConstColorCtx() *colorcore.PMColor4f {
	if p.constColorCtxN == len(p.constColorCtxs) {
		p.constColorCtxs = append(p.constColorCtxs, new(colorcore.PMColor4f))
	}
	c := p.constColorCtxs[p.constColorCtxN]
	p.constColorCtxN++
	return c
}

// AppendPremul appends the premul stage: multiplies rgb by alpha.
func (p *Pipeline) AppendPremul() {
	p.append(func(z *lanes) {
		for i := range z.n {
			z.r[i] *= z.a[i]
			z.g[i] *= z.a[i]
			z.b[i] *= z.a[i]
		}
	})
}

// appendScale1Float appends the scale_1_float stage: multiplies all four channels by a constant.
func (p *Pipeline) appendScale1Float(c float32) {
	p.append(func(z *lanes) {
		for i := range z.n {
			z.r[i] *= c
			z.g[i] *= c
			z.b[i] *= c
			z.a[i] *= c
		}
	})
}

// AppendDither appends an 8x8 ordered-dither stage, used when the paint dithers a non-constant pipeline; rate is 1/255
// for 8888 destinations.
func (p *Pipeline) AppendDither(rate float32) {
	p.append(func(z *lanes) {
		yb := uint32(z.dy)
		for i := range z.n {
			x := uint32(z.dx) + uint32(i)
			y := yb ^ x
			// If X=abc and Y=def, we make fcebda.
			m := (y&1)<<5 | (x&1)<<4 | (y&2)<<2 | (x&2)<<1 | (y&4)>>1 | (x&4)>>2
			dither := madf(float32(m), 2.0/128.0, -63.0/128.0)
			r := madf(dither, rate, z.r[i])
			g := madf(dither, rate, z.g[i])
			b := madf(dither, rate, z.b[i])
			z.r[i] = maxf(0, minf(r, z.a[i]))
			z.g[i] = maxf(0, minf(g, z.a[i]))
			z.b[i] = maxf(0, minf(b, z.a[i]))
		}
	})
}

///////////////////////////////////////////////////////////////////////////////
// tile stages (the 1-domain forms gradients use)

// appendClampX1 appends the clamp_x_1 stage: clamps the r register to [0, 1].
func (p *Pipeline) appendClampX1() {
	p.append(clampX1StageFn)
}

// clampX1Stage is the portable clamp_x_1 stage; clampX1StageFn selects the NEON form on arm64.
func clampX1Stage(z *lanes) {
	for i := range z.n {
		z.r[i] = clamp01(z.r[i])
	}
}

// appendRepeatX1 appends the repeat_x_1 stage: wraps the r register into [0, 1).
func (p *Pipeline) appendRepeatX1() {
	p.append(func(z *lanes) {
		for i := range z.n {
			z.r[i] = clamp01(z.r[i] - floorf(z.r[i]))
		}
	})
}

// appendMirrorX1 appends the mirror_x_1 stage: reflects the r register into [0, 1].
func (p *Pipeline) appendMirrorX1() {
	p.append(func(z *lanes) {
		for i := range z.n {
			v := z.r[i] - 1
			z.r[i] = clamp01(absf(v - 2*floorf(v*0.5) - 1))
		}
	})
}

// laneMask is the per-lane 32-bit mask storage shared between a masking stage and its apply_vector_mask /
// check_decal_mask consumer.
type laneMask [stride]uint32

// appendDecalX appends the gradient's decal_x stage: limit_x is 1+ulp and inclusiveEdge_x is 0 (value-initialized). The
// mask is the stage context.
func (p *Pipeline) appendDecalX(mask *laneMask) {
	p.appendCtx(decalXStage, mask)
}

func decalXStage(z *lanes) {
	mask := z.ctx.(*laneMask)
	limit := math.Float32frombits(math.Float32bits(1) + 1)
	for i := range z.n {
		r := z.r[i]
		if (0 < r && r < limit) || r == 0 {
			mask[i] = 0xFFFFFFFF
		} else {
			mask[i] = 0
		}
	}
}

// maskApplyStage applies a lane mask to all four color channels (the same operation check_decal_mask performs on the
// decal ctx's mask). The laneMask is the stage context.
func maskApplyStage(z *lanes) {
	mask := z.ctx.(*laneMask)
	for i := range z.n {
		z.r[i] = math.Float32frombits(math.Float32bits(z.r[i]) & mask[i])
		z.g[i] = math.Float32frombits(math.Float32bits(z.g[i]) & mask[i])
		z.b[i] = math.Float32frombits(math.Float32bits(z.b[i]) & mask[i])
		z.a[i] = math.Float32frombits(math.Float32bits(z.a[i]) & mask[i])
	}
}

// appendApplyVectorMask appends maskApplyStage.
func (p *Pipeline) appendApplyVectorMask(mask *laneMask) {
	p.appendCtx(maskApplyStage, mask)
}

///////////////////////////////////////////////////////////////////////////////
// gradient coordinate stages

// appendXYToRadius appends the xy_to_radius stage: r = sqrt(r^2 + g^2).
func (p *Pipeline) appendXYToRadius() {
	p.append(func(z *lanes) {
		for i := range z.n {
			z.r[i] = sqrtf(z.r[i]*z.r[i] + z.g[i]*z.g[i])
		}
	})
}

// appendXYToUnitAngle appends the xy_to_unit_angle stage: a 7th degree polynomial approximation of atan, folded into
// [0, 1) turns.
func (p *Pipeline) appendXYToUnitAngle() {
	p.append(func(z *lanes) {
		for i := range z.n {
			x := z.r[i]
			y := z.g[i]
			xabs := absf(x)
			yabs := absf(y)
			slope := minf(xabs, yabs) / maxf(xabs, yabs)
			s := slope * slope
			phi := slope *
				(0.15912117063999176025390625 + s*
					(-5.185396969318389892578125e-2+s*
						(2.476101927459239959716796875e-2+s*
							(-7.0547382347285747528076171875e-3))))
			if xabs < yabs {
				phi = 1.0/4.0 - phi
			}
			if x < 0 {
				phi = 1.0/2.0 - phi
			}
			if y < 0 {
				phi = 1.0 - phi
			}
			if phi != phi { // Check for NaN.
				phi = 0
			}
			z.r[i] = phi
		}
	})
}

// appendMask2PtConicalNaN appends the mask_2pt_conical_nan stage: masks out NaN t values (the conical gradient's strip
// lane). The laneMask is the stage context.
func (p *Pipeline) appendMask2PtConicalNaN(mask *laneMask) {
	p.appendCtx(mask2PtConicalNaNStage, mask)
}

func mask2PtConicalNaNStage(z *lanes) {
	mask := z.ctx.(*laneMask)
	for i := range z.n {
		t := z.r[i]
		if t != t { // NaN
			z.r[i] = 0
			mask[i] = 0
		} else {
			mask[i] = 0xFFFFFFFF
		}
	}
}

// appendMask2PtConicalDegenerates appends the mask_2pt_conical_degenerates stage: masks out non-positive or NaN t
// values (the conical gradient's focal lanes). The laneMask is the stage context.
func (p *Pipeline) appendMask2PtConicalDegenerates(mask *laneMask) {
	p.appendCtx(mask2PtConicalDegeneratesStage, mask)
}

func mask2PtConicalDegeneratesStage(z *lanes) {
	mask := z.ctx.(*laneMask)
	for i := range z.n {
		t := z.r[i]
		if t <= 0 || t != t {
			z.r[i] = 0
			mask[i] = 0
		} else {
			mask[i] = 0xFFFFFFFF
		}
	}
}
