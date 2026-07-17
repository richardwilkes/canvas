// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

//go:build race

package shaders

// raceEnabled reports whether the binary was built with the race detector. The detector instruments every allocation,
// so testing.AllocsPerRun over-reports there — allocation-count assertions skip under it and rely on the non-race build
// for the real measurement.
const raceEnabled = true
