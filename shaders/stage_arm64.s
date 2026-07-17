// Copyright 2026 Richard Wilkes
//
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
//
// NEON forms of the hot SkRasterPipeline span stages (per-arch assembly for the top span kernels,
// adopted only after profiling showed they dominate). Each kernel is bit-identical to its portable Go
// stage:
//
//   - madf is float32(math.FMA(float64(f), float64(m), float64(a))) — a double-precision FMA rounded
//     to single. That is NOT the same as a single-precision FMLA (the 48-bit product makes the 53-bit
//     double rounding non-innocuous; ~1-ULP divergences exist), so every madf site is emulated
//     exactly: FCVTL widens the singles (exact), VFMLA.D2 performs the fused double FMA (one rounding
//     to 53 bits, same as math.FMA), and FCVTN rounds to single — the identical two-rounding sequence
//     the scalar code performs, lanewise.
//   - FMAXNM/FMINNM return the non-NaN operand, matching maxf/minf's "comparison false -> second
//     operand" behavior for the quiet NaNs arithmetic can produce (NaN clamps to 0), and +0 over -0.
//   - FCVTZS truncates toward zero like Go's int32(float32) on arm64, and NaN converts to 0.
//   - FADD/FMUL are the same IEEE single ops the scalar code performs, in the same order.
//
// NaN payload bits may differ from the scalar stages when several FMA operands are NaN (hardware
// operand precedence); payloads are unobservable downstream — every consumer (clamp01, toUnorm, the
// blend math) tests NaN-ness, never payload — and TestStageNEONMatchesScalar checks values bit-exactly
// while treating NaNs as equivalent.
//
// The kernels process all 16 lanes of the register file regardless of z.n — lanes at or beyond z.n
// are scratch no consumer reads (ShadeSpan copies out only [0, n)), and the evenly-spaced gradient
// kernel clamps its stop indices into bounds so a garbage tail lane can never read outside the stop
// array. Vector registers stay within V0-V7/V16-V31 (V8-V15 have callee-saved low halves).
//
// The Go assembler lacks mnemonics for several vector float/integer ops; those are WORD-encoded with
// the intended instruction spelled in the comment, and TestStageNEONMatchesScalar exercises every
// path against the scalar stages.

#include "textflag.h"

// func seed16(r, gn, b, a *[16]float32, fx, fy float32, iotas *[16]float32)
// r = fx + iota (FADD, matching the scalar fx + iota05[i]); g = fy; b = 1; a = 0.
TEXT ·seed16(SB), NOSPLIT, $0-48
	MOVD  r+0(FP), R0
	MOVD  gn+8(FP), R1
	MOVD  b+16(FP), R2
	MOVD  a+24(FP), R3
	FMOVS fx+32(FP), F0
	VDUP  V0.S[0], V0.S4
	FMOVS fy+36(FP), F1
	VDUP  V1.S[0], V1.S4
	MOVD  iotas+40(FP), R4
	VLD1.P 32(R4), [V2.S4, V3.S4]
	VLD1  (R4), [V4.S4, V5.S4]
	FMOVS $(1.0), F6
	VDUP  V6.S[0], V6.S4
	VEOR  V7.B16, V7.B16, V7.B16
	WORD $0x4E22D410 // FADD V16.4S, V0.4S, V2.4S
	WORD $0x4E23D411 // FADD V17.4S, V0.4S, V3.4S
	WORD $0x4E24D412 // FADD V18.4S, V0.4S, V4.4S
	WORD $0x4E25D413 // FADD V19.4S, V0.4S, V5.4S
	VST1  [V16.S4, V17.S4, V18.S4, V19.S4], (R0)
	VST1.P [V1.S4], 16(R1)
	VST1.P [V1.S4], 16(R1)
	VST1.P [V1.S4], 16(R1)
	VST1  [V1.S4], (R1)
	VST1.P [V6.S4], 16(R2)
	VST1.P [V6.S4], 16(R2)
	VST1.P [V6.S4], 16(R2)
	VST1  [V6.S4], (R2)
	VST1.P [V7.S4], 16(R3)
	VST1.P [V7.S4], 16(R3)
	VST1.P [V7.S4], 16(R3)
	VST1  [V7.S4], (R3)
	RET

// func clampX116(r *[16]float32)
// r = min(max(r, 0), 1) with FMAXNM/FMINNM (quiet NaN -> 0, matching clamp01's minf/maxf).
TEXT ·clampX116(SB), NOSPLIT, $0-8
	MOVD r+0(FP), R0
	VEOR V1.B16, V1.B16, V1.B16
	FMOVS $(1.0), F2
	VDUP V2.S[0], V2.S4
	VLD1 (R0), [V16.S4, V17.S4, V18.S4, V19.S4]
	WORD $0x4E21C610 // FMAXNM V16.4S, V16.4S, V1.4S
	WORD $0x4E21C631 // FMAXNM V17.4S, V17.4S, V1.4S
	WORD $0x4E21C652 // FMAXNM V18.4S, V18.4S, V1.4S
	WORD $0x4E21C673 // FMAXNM V19.4S, V19.4S, V1.4S
	WORD $0x4EA2C610 // FMINNM V16.4S, V16.4S, V2.4S
	WORD $0x4EA2C631 // FMINNM V17.4S, V17.4S, V2.4S
	WORD $0x4EA2C652 // FMINNM V18.4S, V18.4S, V2.4S
	WORD $0x4EA2C673 // FMINNM V19.4S, V19.4S, V2.4S
	VST1 [V16.S4, V17.S4, V18.S4, V19.S4], (R0)
	RET

// func matrixTranslate16(r, gn *[16]float32, tx, ty float32)
// r += tx; g += ty (plain FADD, matching the scalar stage).
TEXT ·matrixTranslate16(SB), NOSPLIT, $0-24
	MOVD  r+0(FP), R0
	MOVD  gn+8(FP), R1
	FMOVS tx+16(FP), F0
	VDUP  V0.S[0], V0.S4
	FMOVS ty+20(FP), F1
	VDUP  V1.S[0], V1.S4
	VLD1 (R0), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1 (R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	WORD $0x4E20D610 // FADD V16.4S, V16.4S, V0.4S
	WORD $0x4E20D631 // FADD V17.4S, V17.4S, V0.4S
	WORD $0x4E20D652 // FADD V18.4S, V18.4S, V0.4S
	WORD $0x4E20D673 // FADD V19.4S, V19.4S, V0.4S
	WORD $0x4E21D694 // FADD V20.4S, V20.4S, V1.4S
	WORD $0x4E21D6B5 // FADD V21.4S, V21.4S, V1.4S
	WORD $0x4E21D6D6 // FADD V22.4S, V22.4S, V1.4S
	WORD $0x4E21D6F7 // FADD V23.4S, V23.4S, V1.4S
	VST1 [V16.S4, V17.S4, V18.S4, V19.S4], (R0)
	VST1 [V20.S4, V21.S4, V22.S4, V23.S4], (R1)
	RET

// func matrixScaleTranslate16(r, gn *[16]float32, sx, sy, tx, ty float32)
// r = madf(r, sx, tx); g = madf(g, sy, ty), each madf emulated exactly as
// FCVTN(FMLA.2D(FCVTL(v), s64, t64)) — widen, one fused double rounding, round to single.
TEXT ·matrixScaleTranslate16(SB), NOSPLIT, $0-32
	MOVD  r+0(FP), R0
	MOVD  gn+8(FP), R1
	FMOVS sx+16(FP), F0
	VDUP  V0.S[0], V0.S4
	WORD $0x0E617800 // FCVTL V0.2D, V0.2S (sx as doubles)
	FMOVS sy+20(FP), F1
	VDUP  V1.S[0], V1.S4
	WORD $0x0E617821 // FCVTL V1.2D, V1.2S
	FMOVS tx+24(FP), F2
	VDUP  V2.S[0], V2.S4
	WORD $0x0E617842 // FCVTL V2.2D, V2.2S
	FMOVS ty+28(FP), F3
	VDUP  V3.S[0], V3.S4
	WORD $0x0E617863 // FCVTL V3.2D, V3.2S
	VLD1 (R0), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1 (R1), [V20.S4, V21.S4, V22.S4, V23.S4]

	// r quads: widen, acc = tx64, acc += r64*sx64, narrow.
	WORD $0x0E617A1C // FCVTL V28.2D, V16.2S
	WORD $0x4E617A1D // FCVTL2 V29.2D, V16.4S
	VMOV V2.B16, V30.B16
	VMOV V2.B16, V31.B16
	VFMLA V0.D2, V28.D2, V30.D2
	VFMLA V0.D2, V29.D2, V31.D2
	WORD $0x0E616BD8 // FCVTN V24.2S, V30.2D
	WORD $0x4E616BF8 // FCVTN2 V24.4S, V31.2D
	WORD $0x0E617A3C // FCVTL V28.2D, V17.2S
	WORD $0x4E617A3D // FCVTL2 V29.2D, V17.4S
	VMOV V2.B16, V30.B16
	VMOV V2.B16, V31.B16
	VFMLA V0.D2, V28.D2, V30.D2
	VFMLA V0.D2, V29.D2, V31.D2
	WORD $0x0E616BD9 // FCVTN V25.2S, V30.2D
	WORD $0x4E616BF9 // FCVTN2 V25.4S, V31.2D
	WORD $0x0E617A5C // FCVTL V28.2D, V18.2S
	WORD $0x4E617A5D // FCVTL2 V29.2D, V18.4S
	VMOV V2.B16, V30.B16
	VMOV V2.B16, V31.B16
	VFMLA V0.D2, V28.D2, V30.D2
	VFMLA V0.D2, V29.D2, V31.D2
	WORD $0x0E616BDA // FCVTN V26.2S, V30.2D
	WORD $0x4E616BFA // FCVTN2 V26.4S, V31.2D
	WORD $0x0E617A7C // FCVTL V28.2D, V19.2S
	WORD $0x4E617A7D // FCVTL2 V29.2D, V19.4S
	VMOV V2.B16, V30.B16
	VMOV V2.B16, V31.B16
	VFMLA V0.D2, V28.D2, V30.D2
	VFMLA V0.D2, V29.D2, V31.D2
	WORD $0x0E616BDB // FCVTN V27.2S, V30.2D
	WORD $0x4E616BFB // FCVTN2 V27.4S, V31.2D
	VST1 [V24.S4, V25.S4, V26.S4, V27.S4], (R0)

	// g quads: acc = ty64, acc += g64*sy64.
	WORD $0x0E617A9C // FCVTL V28.2D, V20.2S
	WORD $0x4E617A9D // FCVTL2 V29.2D, V20.4S
	VMOV V3.B16, V30.B16
	VMOV V3.B16, V31.B16
	VFMLA V1.D2, V28.D2, V30.D2
	VFMLA V1.D2, V29.D2, V31.D2
	WORD $0x0E616BD8 // FCVTN V24.2S, V30.2D
	WORD $0x4E616BF8 // FCVTN2 V24.4S, V31.2D
	WORD $0x0E617ABC // FCVTL V28.2D, V21.2S
	WORD $0x4E617ABD // FCVTL2 V29.2D, V21.4S
	VMOV V3.B16, V30.B16
	VMOV V3.B16, V31.B16
	VFMLA V1.D2, V28.D2, V30.D2
	VFMLA V1.D2, V29.D2, V31.D2
	WORD $0x0E616BD9 // FCVTN V25.2S, V30.2D
	WORD $0x4E616BF9 // FCVTN2 V25.4S, V31.2D
	WORD $0x0E617ADC // FCVTL V28.2D, V22.2S
	WORD $0x4E617ADD // FCVTL2 V29.2D, V22.4S
	VMOV V3.B16, V30.B16
	VMOV V3.B16, V31.B16
	VFMLA V1.D2, V28.D2, V30.D2
	VFMLA V1.D2, V29.D2, V31.D2
	WORD $0x0E616BDA // FCVTN V26.2S, V30.2D
	WORD $0x4E616BFA // FCVTN2 V26.4S, V31.2D
	WORD $0x0E617AFC // FCVTL V28.2D, V23.2S
	WORD $0x4E617AFD // FCVTL2 V29.2D, V23.4S
	VMOV V3.B16, V30.B16
	VMOV V3.B16, V31.B16
	VFMLA V1.D2, V28.D2, V30.D2
	VFMLA V1.D2, V29.D2, V31.D2
	WORD $0x0E616BDB // FCVTN V27.2S, V30.2D
	WORD $0x4E616BFB // FCVTN2 V27.4S, V31.2D
	VST1 [V24.S4, V25.S4, V26.S4, V27.S4], (R1)
	RET

// func matrixAffine16(r, gn *[16]float32, m *[9]float32)
// r' = madf(r, s0, madf(g, s1, s2)); g' = madf(r, s3, madf(g, s4, s5)) — the inner madf narrows to
// single before the outer one re-widens it, exactly like the scalar nesting.
TEXT ·matrixAffine16(SB), NOSPLIT, $0-24
	MOVD r+0(FP), R0
	MOVD gn+8(FP), R1
	MOVD m+16(FP), R2
	VLD1 (R2), [V0.S4, V1.S4]
	VDUP V0.S[0], V2.S4
	WORD $0x0E617842 // FCVTL V2.2D, V2.2S (s0)
	VDUP V0.S[1], V3.S4
	WORD $0x0E617863 // FCVTL V3.2D, V3.2S (s1)
	VDUP V0.S[2], V4.S4
	WORD $0x0E617884 // FCVTL V4.2D, V4.2S (s2)
	VDUP V0.S[3], V5.S4
	WORD $0x0E6178A5 // FCVTL V5.2D, V5.2S (s3)
	VDUP V1.S[0], V6.S4
	WORD $0x0E6178C6 // FCVTL V6.2D, V6.2S (s4)
	VDUP V1.S[1], V7.S4
	WORD $0x0E6178E7 // FCVTL V7.2D, V7.2S (s5)
	VLD1 (R0), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1 (R1), [V20.S4, V21.S4, V22.S4, V23.S4]
	MOVD $0, R5

affquad:
	// r64 (V24,V25), g64 (V26,V27) for this quad.
	WORD $0x0E617A18 // FCVTL V24.2D, V16.2S
	WORD $0x4E617A19 // FCVTL2 V25.2D, V16.4S
	WORD $0x0E617A9A // FCVTL V26.2D, V20.2S
	WORD $0x4E617A9B // FCVTL2 V27.2D, V20.4S
	// inner_r = madf(g, s1, s2): acc = s2, += g64*s1; narrow to single, re-widen.
	VMOV V4.B16, V28.B16
	VMOV V4.B16, V29.B16
	VFMLA V3.D2, V26.D2, V28.D2
	VFMLA V3.D2, V27.D2, V29.D2
	WORD $0x0E616B9E // FCVTN V30.2S, V28.2D
	WORD $0x4E616BBE // FCVTN2 V30.4S, V29.2D
	WORD $0x0E617BDC // FCVTL V28.2D, V30.2S
	WORD $0x4E617BDD // FCVTL2 V29.2D, V30.4S
	// r' = madf(r, s0, inner): acc = inner64, += r64*s0; narrow.
	VFMLA V2.D2, V24.D2, V28.D2
	VFMLA V2.D2, V25.D2, V29.D2
	WORD $0x0E616B9E // FCVTN V30.2S, V28.2D
	WORD $0x4E616BBE // FCVTN2 V30.4S, V29.2D
	// inner_g = madf(g, s4, s5); g' = madf(r, s3, inner_g).
	VMOV V7.B16, V28.B16
	VMOV V7.B16, V29.B16
	VFMLA V6.D2, V26.D2, V28.D2
	VFMLA V6.D2, V27.D2, V29.D2
	WORD $0x0E616B9F // FCVTN V31.2S, V28.2D
	WORD $0x4E616BBF // FCVTN2 V31.4S, V29.2D
	WORD $0x0E617BFC // FCVTL V28.2D, V31.2S
	WORD $0x4E617BFD // FCVTL2 V29.2D, V31.4S
	VFMLA V5.D2, V24.D2, V28.D2
	VFMLA V5.D2, V25.D2, V29.D2
	WORD $0x0E616B9F // FCVTN V31.2S, V28.2D
	WORD $0x4E616BBF // FCVTN2 V31.4S, V29.2D
	VST1.P [V30.S4], 16(R0)
	VST1.P [V31.S4], 16(R1)
	// Rotate the next quad's inputs into V16/V20 and loop 4 times.
	VMOV V17.B16, V16.B16
	VMOV V18.B16, V17.B16
	VMOV V19.B16, V18.B16
	VMOV V21.B16, V20.B16
	VMOV V22.B16, V21.B16
	VMOV V23.B16, V22.B16
	ADD  $1, R5, R5
	CMP  $4, R5
	BLT  affquad
	RET

// func grad2Stop16(r, gn, b, a *[16]float32, cl, factor *[4]float32)
// {r,g,b,a} = madf(t, factor[ch], cl[ch]) with t = incoming r, each madf emulated in double lanes.
TEXT ·grad2Stop16(SB), NOSPLIT, $0-48
	MOVD r+0(FP), R0
	MOVD gn+8(FP), R1
	MOVD b+16(FP), R2
	MOVD a+24(FP), R3
	MOVD cl+32(FP), R4
	MOVD factor+40(FP), R5
	VLD1 (R4), [V0.S4]
	VLD1 (R5), [V1.S4]
	VDUP V0.S[0], V2.S4
	WORD $0x0E617842 // FCVTL V2.2D, V2.2S (cl.R)
	VDUP V0.S[1], V3.S4
	WORD $0x0E617863 // FCVTL V3.2D, V3.2S (cl.G)
	VDUP V0.S[2], V4.S4
	WORD $0x0E617884 // FCVTL V4.2D, V4.2S (cl.B)
	VDUP V0.S[3], V5.S4
	WORD $0x0E6178A5 // FCVTL V5.2D, V5.2S (cl.A)
	VDUP V1.S[0], V6.S4
	WORD $0x0E6178C6 // FCVTL V6.2D, V6.2S (f.R)
	VDUP V1.S[1], V7.S4
	WORD $0x0E6178E7 // FCVTL V7.2D, V7.2S (f.G)
	VDUP V1.S[2], V26.S4
	WORD $0x0E617B5A // FCVTL V26.2D, V26.2S (f.B)
	VDUP V1.S[3], V27.S4
	WORD $0x0E617B7B // FCVTL V27.2D, V27.2S (f.A)
	VLD1 (R0), [V16.S4, V17.S4, V18.S4, V19.S4]
	MOVD $0, R6

grad2quad:
	WORD $0x0E617A1C // FCVTL V28.2D, V16.2S (t lo)
	WORD $0x4E617A1D // FCVTL2 V29.2D, V16.4S (t hi)
	VMOV V2.B16, V30.B16
	VMOV V2.B16, V31.B16
	VFMLA V6.D2, V28.D2, V30.D2
	VFMLA V6.D2, V29.D2, V31.D2
	WORD $0x0E616BD4 // FCVTN V20.2S, V30.2D
	WORD $0x4E616BF4 // FCVTN2 V20.4S, V31.2D
	VMOV V3.B16, V30.B16
	VMOV V3.B16, V31.B16
	VFMLA V7.D2, V28.D2, V30.D2
	VFMLA V7.D2, V29.D2, V31.D2
	WORD $0x0E616BD5 // FCVTN V21.2S, V30.2D
	WORD $0x4E616BF5 // FCVTN2 V21.4S, V31.2D
	VMOV V4.B16, V30.B16
	VMOV V4.B16, V31.B16
	VFMLA V26.D2, V28.D2, V30.D2
	VFMLA V26.D2, V29.D2, V31.D2
	WORD $0x0E616BD6 // FCVTN V22.2S, V30.2D
	WORD $0x4E616BF6 // FCVTN2 V22.4S, V31.2D
	VMOV V5.B16, V30.B16
	VMOV V5.B16, V31.B16
	VFMLA V27.D2, V28.D2, V30.D2
	VFMLA V27.D2, V29.D2, V31.D2
	WORD $0x0E616BD7 // FCVTN V23.2S, V30.2D
	WORD $0x4E616BF7 // FCVTN2 V23.4S, V31.2D
	VST1.P [V20.S4], 16(R0)
	VST1.P [V21.S4], 16(R1)
	VST1.P [V22.S4], 16(R2)
	VST1.P [V23.S4], 16(R3)
	VMOV V17.B16, V16.B16
	VMOV V18.B16, V17.B16
	VMOV V19.B16, V18.B16
	ADD  $1, R6, R6
	CMP  $4, R6
	BLT  grad2quad
	RET

// func gradEvenly16(r, gn, b, a *[16]float32, stops *gradStop, maxIdx int64, gapCount float32)
// Per lane: idx = int32(t*gapCount) (FMUL then FCVTZS — the same round-then-truncate as the scalar
// stage), clamped into [0, maxIdx] (a no-op for the valid lanes, whose t is already tiled into [0,1];
// it only bounds the garbage tail lanes' gathers). Then {r,g,b,a} = madf(t, factor[ch], bias[ch])
// over the four gathered 32-byte stop records, transposed to channel quads, each madf emulated in
// double lanes.
TEXT ·gradEvenly16(SB), NOSPLIT, $0-52
	MOVD r+0(FP), R0
	MOVD gn+8(FP), R1
	MOVD b+16(FP), R2
	MOVD a+24(FP), R3
	MOVD stops+32(FP), R4
	MOVD maxIdx+40(FP), R5
	FMOVS gapCount+48(FP), F1
	VDUP V1.S[0], V1.S4
	VEOR V2.B16, V2.B16, V2.B16
	VDUP R5, V3.S4
	MOVD $4, R11

loop:
	VLD1 (R0), [V0.S4]   // t
	WORD $0x6E21DC04     // FMUL V4.4S, V0.4S, V1.4S
	WORD $0x4EA1B885     // FCVTZS V5.4S, V4.4S
	WORD $0x4EA264A5     // SMAX V5.4S, V5.4S, V2.4S
	WORD $0x4EA36CA5     // SMIN V5.4S, V5.4S, V3.4S
	VMOV V5.S[0], R6
	VMOV V5.S[1], R7
	VMOV V5.S[2], R8
	VMOV V5.S[3], R9
	ADD  R6<<5, R4, R10
	VLD1 (R10), [V16.S4, V17.S4]
	ADD  R7<<5, R4, R10
	VLD1 (R10), [V18.S4, V19.S4]
	ADD  R8<<5, R4, R10
	VLD1 (R10), [V20.S4, V21.S4]
	ADD  R9<<5, R4, R10
	VLD1 (R10), [V22.S4, V23.S4]
	// Transpose the four bias quads (V17,V19,V21,V23) into per-channel quads bR..bA (V6,V7,V4,V5).
	VZIP1 V19.S4, V17.S4, V28.S4
	VZIP2 V19.S4, V17.S4, V29.S4
	VZIP1 V23.S4, V21.S4, V30.S4
	VZIP2 V23.S4, V21.S4, V31.S4
	VZIP1 V30.D2, V28.D2, V6.D2
	VZIP2 V30.D2, V28.D2, V7.D2
	VZIP1 V31.D2, V29.D2, V4.D2
	VZIP2 V31.D2, V29.D2, V5.D2
	// Transpose the four factor quads (V16,V18,V20,V22) into fR..fA (V24..V27).
	VZIP1 V18.S4, V16.S4, V28.S4
	VZIP2 V18.S4, V16.S4, V29.S4
	VZIP1 V22.S4, V20.S4, V30.S4
	VZIP2 V22.S4, V20.S4, V31.S4
	VZIP1 V30.D2, V28.D2, V24.D2
	VZIP2 V30.D2, V28.D2, V25.D2
	VZIP1 V31.D2, V29.D2, V26.D2
	VZIP2 V31.D2, V29.D2, V27.D2
	// t widened once for all four channels.
	WORD $0x0E61781C // FCVTL V28.2D, V0.2S
	WORD $0x4E61781D // FCVTL2 V29.2D, V0.4S
	// R: out = madf(t, fR, bR).
	WORD $0x0E617B1E // FCVTL V30.2D, V24.2S
	WORD $0x4E617B1F // FCVTL2 V31.2D, V24.4S
	WORD $0x0E6178D0 // FCVTL V16.2D, V6.2S
	WORD $0x4E6178D1 // FCVTL2 V17.2D, V6.4S
	VFMLA V30.D2, V28.D2, V16.D2
	VFMLA V31.D2, V29.D2, V17.D2
	WORD $0x0E616A12 // FCVTN V18.2S, V16.2D
	WORD $0x4E616A32 // FCVTN2 V18.4S, V17.2D
	VST1.P [V18.S4], 16(R0)
	// G.
	WORD $0x0E617B3E // FCVTL V30.2D, V25.2S
	WORD $0x4E617B3F // FCVTL2 V31.2D, V25.4S
	WORD $0x0E6178F0 // FCVTL V16.2D, V7.2S
	WORD $0x4E6178F1 // FCVTL2 V17.2D, V7.4S
	VFMLA V30.D2, V28.D2, V16.D2
	VFMLA V31.D2, V29.D2, V17.D2
	WORD $0x0E616A12 // FCVTN V18.2S, V16.2D
	WORD $0x4E616A32 // FCVTN2 V18.4S, V17.2D
	VST1.P [V18.S4], 16(R1)
	// B.
	WORD $0x0E617B5E // FCVTL V30.2D, V26.2S
	WORD $0x4E617B5F // FCVTL2 V31.2D, V26.4S
	WORD $0x0E617890 // FCVTL V16.2D, V4.2S
	WORD $0x4E617891 // FCVTL2 V17.2D, V4.4S
	VFMLA V30.D2, V28.D2, V16.D2
	VFMLA V31.D2, V29.D2, V17.D2
	WORD $0x0E616A12 // FCVTN V18.2S, V16.2D
	WORD $0x4E616A32 // FCVTN2 V18.4S, V17.2D
	VST1.P [V18.S4], 16(R2)
	// A.
	WORD $0x0E617B7E // FCVTL V30.2D, V27.2S
	WORD $0x4E617B7F // FCVTL2 V31.2D, V27.4S
	WORD $0x0E6178B0 // FCVTL V16.2D, V5.2S
	WORD $0x4E6178B1 // FCVTL2 V17.2D, V5.4S
	VFMLA V30.D2, V28.D2, V16.D2
	VFMLA V31.D2, V29.D2, V17.D2
	WORD $0x0E616A12 // FCVTN V18.2S, V16.2D
	WORD $0x4E616A32 // FCVTN2 V18.4S, V17.2D
	VST1.P [V18.S4], 16(R3)
	SUB $1, R11, R11
	CBNZ R11, loop
	RET
