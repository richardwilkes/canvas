// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package tthint executes the TrueType instructions a font carries, grid-fitting its glyph outlines to a pixel size.
//
// It is a port of the TrueType interpreter and glyph scaler in Skrifa (https://github.com/googlefonts/fontations,
// skrifa/src/outline/glyf), which is itself written to reproduce FreeType's interpreter (version 40, the default)
// coordinate for coordinate. Skrifa is Copyright (c) 2019 Fontations Developers and is used under the MIT license,
// reproduced in LICENSE-skrifa-MIT beside this file. The tests hold this package to FreeType's own output.
//
// Two things follow from tracking FreeType's default mode. Instructions that move points horizontally are ignored
// ("backward compatibility" in the interpreter's terms), because subpixel rendering resolves horizontal detail better
// than rounding to whole pixels does, so hinting changes where rows of a glyph fall and leaves its widths and advances
// alone. And an instruction error is not fatal: the program stops and the outline is used as far as it got, which is
// what lets the many shipped fonts with slightly broken programs render.
//
// Variable fonts are not handled: a Font reports HasVariations and the caller is expected not to hint an instance
// away from the default.
package tthint
