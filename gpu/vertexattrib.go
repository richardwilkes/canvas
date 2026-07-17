// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// VertexAttribType, used by vertex attrib state tracking and the geometry processors.

package gpu

// VertexAttribType identifies the CPU-side type of a vertex attribute.
type VertexAttribType int32

// VertexAttribType values.
const (
	VertexAttribTypeFloat VertexAttribType = iota
	VertexAttribTypeFloat2
	VertexAttribTypeFloat3
	VertexAttribTypeFloat4
	VertexAttribTypeHalf
	VertexAttribTypeHalf2
	VertexAttribTypeHalf4

	VertexAttribTypeInt2 // vector of 2 32-bit ints
	VertexAttribTypeInt3 // vector of 3 32-bit ints
	VertexAttribTypeInt4 // vector of 4 32-bit ints

	VertexAttribTypeByte   // signed byte
	VertexAttribTypeByte2  // vector of 2 8-bit signed bytes
	VertexAttribTypeByte4  // vector of 4 8-bit signed bytes
	VertexAttribTypeUByte  // unsigned byte
	VertexAttribTypeUByte2 // vector of 2 8-bit unsigned bytes
	VertexAttribTypeUByte4 // vector of 4 8-bit unsigned bytes

	VertexAttribTypeUByteNorm  // unsigned byte, e.g. coverage, 0 -> 0.0f, 255 -> 1.0f
	VertexAttribTypeUByte4Norm // vector of 4 unsigned bytes, e.g. colors

	VertexAttribTypeShort2  // vector of 2 16-bit shorts
	VertexAttribTypeShort4  // vector of 4 16-bit shorts
	VertexAttribTypeUShort2 // vector of 2 unsigned shorts

	VertexAttribTypeUShort2Norm // vector of 2 unsigned shorts, 0 -> 0.0f, 65535 -> 1.0f

	VertexAttribTypeInt
	VertexAttribTypeUInt

	VertexAttribTypeUShortNorm

	VertexAttribTypeUShort4Norm // vector of 4 unsigned shorts, 0 -> 0.0f, 65535 -> 1.0f
)

// VertexAttribTypeSize returns the size of the attrib type in bytes.
func VertexAttribTypeSize(t VertexAttribType) int {
	switch t {
	case VertexAttribTypeFloat:
		return 4
	case VertexAttribTypeFloat2:
		return 2 * 4
	case VertexAttribTypeFloat3:
		return 3 * 4
	case VertexAttribTypeFloat4:
		return 4 * 4
	case VertexAttribTypeHalf:
		return 2
	case VertexAttribTypeHalf2:
		return 2 * 2
	case VertexAttribTypeHalf4:
		return 4 * 2
	case VertexAttribTypeInt2:
		return 2 * 4
	case VertexAttribTypeInt3:
		return 3 * 4
	case VertexAttribTypeInt4:
		return 4 * 4
	case VertexAttribTypeByte, VertexAttribTypeUByte, VertexAttribTypeUByteNorm:
		return 1
	case VertexAttribTypeByte2, VertexAttribTypeUByte2:
		return 2
	case VertexAttribTypeByte4, VertexAttribTypeUByte4, VertexAttribTypeUByte4Norm:
		return 4
	case VertexAttribTypeShort2, VertexAttribTypeUShort2, VertexAttribTypeUShort2Norm:
		return 2 * 2
	case VertexAttribTypeShort4, VertexAttribTypeUShort4Norm:
		return 4 * 2
	case VertexAttribTypeInt, VertexAttribTypeUInt:
		return 4
	case VertexAttribTypeUShortNorm:
		return 2
	}
	panic("unsupported vertex attrib type")
}
