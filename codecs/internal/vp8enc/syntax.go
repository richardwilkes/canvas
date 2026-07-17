// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Bitstream syntax: the compressed frame header (partition 0), intra mode coding, the uncompressed frame header and the
// RIFF/WEBP container.

package vp8enc

import (
	"encoding/binary"
	"errors"
)

const (
	vp8Signature         = 0x9d012a
	vp8MaxPartition0Size = 1 << 19 // partition-0 length is a 19-bit field
	maxDimension         = 16383   // 14-bit width/height fields
)

var errPartition0Overflow = errors.New("vp8enc: partition #0 is too big to fit")

// putI4Mode codes one intra4 mode with the given context probabilities, returning the mode.
func putI4Mode(bw *bitWriter, mode int, prob *[numBModes - 1]uint8) int {
	if bw.putBit(mode != bDCPred, prob[0]) {
		if bw.putBit(mode != bTMPred, prob[1]) {
			if bw.putBit(mode != bVEPred, prob[2]) {
				if !bw.putBit(mode >= bLDPred, prob[3]) {
					if bw.putBit(mode != bHEPred, prob[4]) {
						bw.putBit(mode != bRDPred, prob[5])
					}
				} else {
					if bw.putBit(mode != bLDPred, prob[6]) {
						if bw.putBit(mode != bVLPred, prob[7]) {
							bw.putBit(mode != bHDPred, prob[8])
						}
					}
				}
			}
		}
	}
	return mode
}

func putI16Mode(bw *bitWriter, mode int) {
	if bw.putBit(mode == tmPred || mode == hPred, 156) {
		bw.putBit(mode == tmPred, 128) // TM or HE
	} else {
		bw.putBit(mode == vPred, 163) // VE or DC
	}
}

func putUVMode(bw *bitWriter, uvMode int) {
	if bw.putBit(uvMode != dcPred, 142) {
		if bw.putBit(uvMode != vPred, 114) {
			bw.putBit(uvMode != hPred, 183) // else: TM
		}
	}
}

// codeIntraModes writes the per-macroblock mode records into partition 0.
func (e *encoder) codeIntraModes(bw *bitWriter) {
	for y := 0; y < e.mbH; y++ {
		for x := 0; x < e.mbW; x++ {
			mb := &e.mbInfo[y*e.mbW+x]
			predsIdx := e.predsBase + y*4*e.predsW + x*4
			if e.proba.useSkipProba {
				bw.putBit(mb.skip, e.proba.skipProba)
			}
			if bw.putBit(mb.typ != 0, 145) { // i16x16
				putI16Mode(bw, int(e.preds[predsIdx]))
			} else {
				topIdx := predsIdx - e.predsW
				for j := 0; j < 4; j++ {
					left := int(e.preds[predsIdx+j*e.predsW-1])
					for i := 0; i < 4; i++ {
						probas := &bModesProba[e.preds[topIdx+i]][left]
						left = putI4Mode(bw, int(e.preds[predsIdx+j*e.predsW+i]), probas)
					}
					topIdx = predsIdx + j*e.predsW
				}
			}
			putUVMode(bw, int(mb.uvMode))
		}
	}
}

// putSegmentHeader writes the (disabled) segmentation header: this encoder always uses a single segment (scope
// decision).
func (e *encoder) putSegmentHeader(bw *bitWriter) {
	bw.putBitUniform(false) // no segmentation
}

// putFilterHeader writes the loop filter parameters.
func (e *encoder) putFilterHeader(bw *bitWriter) {
	bw.putBitUniform(e.filterSimple)
	bw.putBits(uint32(e.filterLevel), 6)
	bw.putBits(uint32(e.filterSharpness), 3)
	bw.putBitUniform(false) // no LF delta
}

// putQuant writes the nominal quantization parameters.
func (e *encoder) putQuant(bw *bitWriter) {
	bw.putBits(uint32(e.baseQuant), 7)
	bw.putSignedBits(e.dqY1DC, 4)
	bw.putSignedBits(e.dqY2DC, 4)
	bw.putSignedBits(e.dqY2AC, 4)
	bw.putSignedBits(e.dqUVDC, 4)
	bw.putSignedBits(e.dqUVAC, 4)
}

// generatePartition0 builds the compressed frame header plus mode records.
func (e *encoder) generatePartition0() []byte {
	bw := &e.bwHeader
	bw.init()
	bw.putBitUniform(false) // colorspace
	bw.putBitUniform(false) // clamp type
	e.putSegmentHeader(bw)
	e.putFilterHeader(bw)
	bw.putBits(0, 2) // a single coefficient partition (scope decision)
	e.putQuant(bw)
	bw.putBitUniform(false) // no proba update ("refresh entropy probs" position for keyframes)
	e.proba.writeProbas(bw)
	e.codeIntraModes(bw)
	return bw.finish()
}

// frameBytes produces the raw VP8 keyframe bitstream — the payload of a WebP "VP8 " chunk: uncompressed frame header,
// partition 0 and the coefficient partition.
func (e *encoder) frameBytes() ([]byte, error) {
	part0 := e.generatePartition0()
	part1 := e.bwCoeffs.buf
	if len(part0) >= vp8MaxPartition0Size {
		return nil, errPartition0Overflow
	}

	const frameHeaderSize = 10 // 3-byte frame tag + 3-byte start code + 2x2-byte dimensions
	out := make([]byte, 0, frameHeaderSize+len(part0)+len(part1))

	// Uncompressed frame header (RFC 6386 §9.1): keyframe, profile 0, visible, partition-0 size.
	bits := uint32(0) | 0<<1 | 1<<4 | uint32(len(part0))<<5
	out = append(out, byte(bits), byte(bits>>8), byte(bits>>16),
		byte(vp8Signature>>16), byte((vp8Signature>>8)&0xff), byte(vp8Signature&0xff),
		byte(e.width), byte(e.width>>8), byte(e.height), byte(e.height>>8))

	out = append(out, part0...)
	out = append(out, part1...)
	return out, nil
}

// assemble produces the complete WebP file: the simple RIFF/WEBP container around the VP8 chunk. Callers that need the
// extended (VP8X) container — e.g. to add an ALPH alpha chunk — use EncodeFrame and assemble the container themselves.
func (e *encoder) assemble() ([]byte, error) {
	frame, err := e.frameBytes()
	if err != nil {
		return nil, err
	}
	pad := len(frame) & 1
	riffSize := 4 + 8 + len(frame) + pad // "WEBP" + the VP8 chunk header + payload

	out := make([]byte, 0, 12+8+len(frame)+pad)
	out = append(out, 'R', 'I', 'F', 'F')
	out = binary.LittleEndian.AppendUint32(out, uint32(riffSize))
	out = append(out, 'W', 'E', 'B', 'P', 'V', 'P', '8', ' ')
	out = binary.LittleEndian.AppendUint32(out, uint32(len(frame)))
	out = append(out, frame...)
	if pad != 0 {
		out = append(out, 0)
	}
	return out, nil
}
