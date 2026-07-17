// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Tests migrated from the retired façade suite: they keep the core behaviors that previously had coverage only through
// the façade's forwarding tests.

package imagecore

import "testing"

// TestRegisterCodecReplacesByName verifies RegisterCodec's same-name semantics: re-registering a codec with an existing
// name replaces it in place rather than appending a duplicate.
func TestRegisterCodecReplacesByName(t *testing.T) {
	sniffed := 0
	c := Codec{
		Name:    "capimigrated-test-codec",
		Matches: func([]byte) bool { sniffed = 1; return false },
	}
	RegisterCodec(c)
	c.Matches = func([]byte) bool { sniffed = 2; return false }
	RegisterCodec(c)

	codecMu.RLock()
	count := 0
	for i := range codecs {
		if codecs[i].Name == c.Name {
			count++
			codecs[i].Matches(nil)
		}
	}
	codecMu.RUnlock()
	if count != 1 {
		t.Fatalf("codec registered %d times, want the replace-in-place behavior (1)", count)
	}
	if sniffed != 2 {
		t.Fatal("re-registering did not replace the earlier codec entry")
	}
}

// TestNewRasterDataInvalidInfo verifies the nil contract of NewRasterData: an invalid info (non-positive dimensions) or
// an undersized buffer yields nil, and a valid one wraps a copy of the bytes.
func TestNewRasterDataInvalidInfo(t *testing.T) {
	good, _ := MakeInfo(2, 2, ColorTypeRGBA8888, AlphaTypePremul)
	buf := make([]byte, 2*2*4)
	if NewRasterData(good, buf, 2*4) == nil {
		t.Fatal("NewRasterData with valid info/buffer returned nil")
	}

	bad := good
	bad.Width = 0
	if NewRasterData(bad, buf, 2*4) != nil {
		t.Error("NewRasterData with zero width: want nil")
	}
	bad = good
	bad.Height = -1
	if NewRasterData(bad, buf, 2*4) != nil {
		t.Error("NewRasterData with negative height: want nil")
	}
	if NewRasterData(good, buf[:4], 2*4) != nil {
		t.Error("NewRasterData with a too-small buffer: want nil")
	}
}
