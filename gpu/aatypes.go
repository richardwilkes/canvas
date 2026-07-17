// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// GPU-wide anti-aliasing types: a boolean AA request, the resolved AAType a draw ends up using once the target's sample
// count and the caps are known, and per-edge QuadAAFlags for a quadrilateral.

package gpu

// AA is a boolean "should a draw be anti-aliased" request.
type AA bool

// AA values.
const (
	AANo  AA = false
	AAYes AA = true
)

// AAType is the method of anti-aliasing a draw resolves to once the target's sample count and the caps are known.
type AAType int32

// AAType values, from weakest to strongest.
const (
	// AATypeNone: no anti-aliasing.
	AATypeNone AAType = iota
	// AATypeCoverage: use fragment coverage for partially covered pixels.
	AATypeCoverage
	// AATypeMSAA: use normal MSAA.
	AATypeMSAA
)

// IsHW reports whether this AA type resolves to hardware multisampling.
func (t AAType) IsHW() bool { return t == AATypeMSAA }

// QuadAAFlags holds per-edge anti-aliasing flags for a quadrilateral.
type QuadAAFlags uint8

// QuadAAFlags values.
const (
	QuadAAFlagLeft QuadAAFlags = 1 << iota
	QuadAAFlagTop
	QuadAAFlagRight
	QuadAAFlagBottom
	QuadAAFlagsNone QuadAAFlags = 0
	QuadAAFlagsAll  QuadAAFlags = 15
)
