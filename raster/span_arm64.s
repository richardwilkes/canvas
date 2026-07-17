// Copyright 2026 Richard Wilkes
//
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
//
// NEON forms of the hot row kernels the shader blitters run per span. Bit-exactness:
//
//   - clampQuads: FMAXNM/FMINNM match clamp01f's minf/maxf for the quiet NaNs the pipeline produces
//     (NaN -> 0) and prefer +0 over -0 exactly like the scalar comparisons.
//   - storeSpanQuads: toUnorm(v) = round-to-nearest-even(min(max(v*255, 0-or-NaN->0), 255)) maps to
//     FMUL 255, FMAXNM 0, FMINNM 255, FCVTNS — the same mul rounding, the same clamps, and FCVTNS is
//     exactly RoundToEven for the in-range integral result.
//   - pmSrcOverQuads mirrors blit_row_s32a_opaque's NEON kernel: per byte
//     satAdd8(src, mulDiv255Round(dst, 255-srcA)) with the identical widening math (see
//     pmSrcOverRow's lane-bound analysis; UQADD is the saturating add).
//
// Vector registers stay within V0-V7/V16-V31 (V8-V15 have callee-saved low halves). Unsupported
// mnemonics are WORD-encoded with the intended instruction in the comment; the kernels are locked
// against their scalar twins by TestSpanNEONMatchesScalar.

#include "textflag.h"

// func clampQuads(p *float32, quads int)
// Clamps quads*4 consecutive floats to [0, 1].
TEXT ·clampQuads(SB), NOSPLIT, $0-16
	MOVD p+0(FP), R0
	MOVD quads+8(FP), R1
	VEOR V1.B16, V1.B16, V1.B16
	FMOVS $(1.0), F2
	VDUP V2.S[0], V2.S4
clamploop:
	VLD1 (R0), [V0.S4]
	WORD $0x4E21C400 // FMAXNM V0.4S, V0.4S, V1.4S
	WORD $0x4EA2C400 // FMINNM V0.4S, V0.4S, V2.4S
	VST1.P [V0.S4], 16(R0)
	SUB $1, R1, R1
	CBNZ R1, clamploop
	RET

// func storeSpanQuads(buf *float32, span *uint32, quads int)
// Converts quads*4 premultiplied float pixels (AoS RGBA) to 8888 words via toUnorm per channel.
TEXT ·storeSpanQuads(SB), NOSPLIT, $0-24
	MOVD buf+0(FP), R0
	MOVD span+8(FP), R1
	MOVD quads+16(FP), R2
	VEOR V1.B16, V1.B16, V1.B16
	MOVD $0x437F0000, R3 // 255.0f bits
	FMOVS R3, F2
	VDUP V2.S[0], V2.S4
storeloop:
	VLD1.P 64(R0), [V16.S4, V17.S4, V18.S4, V19.S4]
	// Transpose 4 AoS pixels to channel quads r=V24 g=V25 b=V26 a=V27.
	VZIP1 V17.S4, V16.S4, V28.S4
	VZIP2 V17.S4, V16.S4, V29.S4
	VZIP1 V19.S4, V18.S4, V30.S4
	VZIP2 V19.S4, V18.S4, V31.S4
	VZIP1 V30.D2, V28.D2, V24.D2
	VZIP2 V30.D2, V28.D2, V25.D2
	VZIP1 V31.D2, V29.D2, V26.D2
	VZIP2 V31.D2, V29.D2, V27.D2
	// toUnorm: *255, max 0 (NaN->0), min 255, round-to-nearest-even convert.
	WORD $0x6E22DF18 // FMUL V24.4S, V24.4S, V2.4S
	WORD $0x6E22DF39 // FMUL V25.4S, V25.4S, V2.4S
	WORD $0x6E22DF5A // FMUL V26.4S, V26.4S, V2.4S
	WORD $0x6E22DF7B // FMUL V27.4S, V27.4S, V2.4S
	WORD $0x4E21C718 // FMAXNM V24.4S, V24.4S, V1.4S
	WORD $0x4E21C739 // FMAXNM V25.4S, V25.4S, V1.4S
	WORD $0x4E21C75A // FMAXNM V26.4S, V26.4S, V1.4S
	WORD $0x4E21C77B // FMAXNM V27.4S, V27.4S, V1.4S
	WORD $0x4EA2C718 // FMINNM V24.4S, V24.4S, V2.4S
	WORD $0x4EA2C739 // FMINNM V25.4S, V25.4S, V2.4S
	WORD $0x4EA2C75A // FMINNM V26.4S, V26.4S, V2.4S
	WORD $0x4EA2C77B // FMINNM V27.4S, V27.4S, V2.4S
	WORD $0x4E21AB18 // FCVTNS V24.4S, V24.4S
	WORD $0x4E21AB39 // FCVTNS V25.4S, V25.4S
	WORD $0x4E21AB5A // FCVTNS V26.4S, V26.4S
	WORD $0x4E21AB7B // FCVTNS V27.4S, V27.4S
	// word = r | g<<8 | b<<16 | a<<24
	VSHL $8, V25.S4, V25.S4
	VSHL $16, V26.S4, V26.S4
	VSHL $24, V27.S4, V27.S4
	VORR V25.B16, V24.B16, V24.B16
	VORR V26.B16, V24.B16, V24.B16
	VORR V27.B16, V24.B16, V24.B16
	VST1.P [V24.S4], 16(R1)
	SUB $1, R2, R2
	CBNZ R2, storeloop
	RET

// func pmSrcOverQuads(dst, src *uint32, quads int)
// dst = satAdd8(src, mulDiv255Round(dst, 255-srcA)) per byte over quads*4 pixels.
TEXT ·pmSrcOverQuads(SB), NOSPLIT, $0-24
	MOVD dst+0(FP), R0
	MOVD src+8(FP), R1
	MOVD quads+16(FP), R2
	MOVD $·alphaSplatIdx(SB), R3
	VLD1 (R3), [V1.B16]
	MOVD $128, R4
	VDUP R4, V2.H8
srcoverloop:
	VLD1.P 16(R1), [V0.B16]  // src
	VLD1 (R0), [V3.B16]      // dst
	VTBL V1.B16, [V0.B16], V4.B16 // splat each pixel's alpha byte across its 4 bytes
	WORD $0x6E205885 // NOT V5.16B, V4.16B (255 - alpha per byte)
	WORD $0x2E25C066 // UMULL V6.8H, V3.8B, V5.8B (dst*nalpha, low 8 bytes)
	WORD $0x6E25C067 // UMULL2 V7.8H, V3.16B, V5.16B (high 8 bytes)
	VADD V2.H8, V6.H8, V6.H8
	VADD V2.H8, V7.H8, V7.H8
	WORD $0x6F1804D0 // USHR V16.8H, V6.8H, #8
	WORD $0x6F1804F1 // USHR V17.8H, V7.8H, #8
	VADD V16.H8, V6.H8, V6.H8
	VADD V17.H8, V7.H8, V7.H8
	WORD $0x0F0884D2 // SHRN V18.8B, V6.8H, #8
	WORD $0x4F0884F2 // SHRN2 V18.16B, V7.8H, #8
	WORD $0x6E320C13 // UQADD V19.16B, V0.16B, V18.16B
	VST1.P [V19.B16], 16(R0)
	SUB $1, R2, R2
	CBNZ R2, srcoverloop
	RET

// func blitMaskOpaqueQuads(dev *uint32, aa *uint8, pm uint32, quads int)
// The opaque solid-color A8 mask blend over quads*4 pixels:
// dev_b = (pm_b*(m+1))>>8 + (d_b*(256-m))>>8 per byte, computed as pm_b*m + pm_b and
// d_b*(255-m) + d_b so every product is an 8x8-bit UMULL (the identities are exact, and every
// 16-bit lane stays under 65281). The per-byte sums cannot exceed 255 (the two floor terms are a
// convex 257/256-weighted pair; only m=255 reaches weight 257, where the second term is 0).
TEXT ·blitMaskOpaqueQuads(SB), NOSPLIT, $0-32
	MOVD  dev+0(FP), R0
	MOVD  aa+8(FP), R1
	MOVWU pm+16(FP), R4
	MOVD  quads+24(FP), R2
	VDUP  R4, V0.S4          // pm replicated: pm_b in every 4-byte group
	MOVD  $·maskSplatIdx(SB), R3
	VLD1  (R3), [V1.B16]
	WORD $0x2F08A402 // USHLL $0, V0.8B, V2.8H (pm widened; uniform across halves)
maskloop:
	MOVWU.P 4(R1), R5
	VMOV  R5, V4.S[0]
	VTBL  V1.B16, [V4.B16], V4.B16 // splat each pixel's mask byte across its 4 byte lanes
	WORD $0x6E205885 // NOT V5.16B, V4.16B (255 - m per byte)
	VLD1  (R0), [V3.B16]           // dst
	WORD $0x2E24C006 // UMULL V6.8H, V0.8B, V4.8B (pm*m, low)
	WORD $0x6E24C007 // UMULL2 V7.8H, V0.16B, V4.16B (high)
	VADD  V2.H8, V6.H8, V6.H8      // += pm  -> pm*(m+1)
	VADD  V2.H8, V7.H8, V7.H8
	WORD $0x2F08A470 // USHLL $0, V3.8B, V16.8H (d widened, low)
	WORD $0x6F08A471 // USHLL2 $0, V3.16B, V17.8H (high)
	WORD $0x2E25C072 // UMULL V18.8H, V3.8B, V5.8B (d*(255-m), low)
	WORD $0x6E25C073 // UMULL2 V19.8H, V3.16B, V5.16B (high)
	VADD  V16.H8, V18.H8, V18.H8   // += d -> d*(256-m)
	VADD  V17.H8, V19.H8, V19.H8
	WORD $0x6F1804C6 // USHR V6.8H, V6.8H, #8
	WORD $0x6F1804E7 // USHR V7.8H, V7.8H, #8
	WORD $0x6F180652 // USHR V18.8H, V18.8H, #8
	WORD $0x6F180673 // USHR V19.8H, V19.8H, #8
	VADD  V18.H8, V6.H8, V6.H8
	VADD  V19.H8, V7.H8, V7.H8
	WORD $0x0E2128D4 // XTN V20.8B, V6.8H
	WORD $0x4E2128F4 // XTN2 V20.16B, V7.8H
	VST1.P [V20.B16], 16(R0)
	SUB  $1, R2, R2
	CBNZ R2, maskloop
	RET

// alphaSplatIdx is the TBL index vector that replicates each 32-bit pixel's alpha byte (offset 3)
// across all four of its byte lanes.
DATA ·alphaSplatIdx+0(SB)/8, $0x0707070703030303
DATA ·alphaSplatIdx+8(SB)/8, $0x0F0F0F0F0B0B0B0B
GLOBL ·alphaSplatIdx(SB), RODATA|NOPTR, $16

// maskSplatIdx is the TBL index vector that replicates each of the four A8 mask bytes (loaded into
// the first word) across its pixel's 4 byte lanes.
DATA ·maskSplatIdx+0(SB)/8, $0x0101010100000000
DATA ·maskSplatIdx+8(SB)/8, $0x0303030302020202
GLOBL ·maskSplatIdx(SB), RODATA|NOPTR, $16
