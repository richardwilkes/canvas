// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package scenario

import "testing"

// TestBlendGridsCoverEveryMode pins the claim corpus_blend.go's file comment makes: the three family grids together
// name every BlendMode from BlendClear through BlendLuminosity exactly once, and the family sizes it cites (12
// Porter-Duff, 13 separable, 4 HSL, 29 total) are the real ones. Adding a BlendMode without extending a grid, or
// duplicating one across grids, fails here rather than silently shrinking the gate's coverage.
func TestBlendGridsCoverEveryMode(t *testing.T) {
	for _, family := range []struct {
		name  string
		modes []BlendMode
		want  int
	}{
		{name: "Porter-Duff", modes: porterDuffBlendModes, want: 12},
		{name: "separable", modes: separableBlendModes, want: 13},
		{name: "HSL", modes: hslBlendModes, want: 4},
	} {
		if len(family.modes) != family.want {
			t.Errorf("%s grid has %d modes, want %d", family.name, len(family.modes), family.want)
		}
	}

	seen := make(map[BlendMode]string)
	for _, family := range []struct {
		name  string
		modes []BlendMode
	}{
		{name: "Porter-Duff", modes: porterDuffBlendModes},
		{name: "separable", modes: separableBlendModes},
		{name: "HSL", modes: hslBlendModes},
	} {
		for _, mode := range family.modes {
			if prior, ok := seen[mode]; ok {
				t.Errorf("BlendMode %d appears in both the %s and %s grids", mode, prior, family.name)
				continue
			}
			seen[mode] = family.name
		}
	}

	const wantTotal = int(BlendLuminosity) + 1
	if wantTotal != 29 {
		t.Errorf("the BlendMode enum now holds %d values; corpus_blend.go's comment says 29", wantTotal)
	}
	if len(seen) != wantTotal {
		t.Errorf("the grids cover %d distinct modes, want all %d", len(seen), wantTotal)
	}
	for mode := BlendClear; mode <= BlendLuminosity; mode++ {
		if _, ok := seen[mode]; !ok {
			t.Errorf("BlendMode %d is not covered by any grid", mode)
		}
	}
}
