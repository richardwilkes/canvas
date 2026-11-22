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
// skrifa/src/outline/glyf, as of commit 7eec1e18), with its ClearType mode left at the classic default. Skrifa is
// Copyright (c) 2019 Fontations Developers and is used under the MIT license, reproduced in LICENSE-skrifa-MIT beside
// this file.
//
// Skrifa models FreeType's interpreter (version 40, the default), and on the fonts in testdata the tests hold this
// package to FreeType's own output, coordinate for coordinate. The two are not the same program, though, and where they
// differ this package follows Skrifa, so that it can keep following it. Most of the differences are in what happens to
// fonts and programs that are wrong, and in how much of FreeType's glyph loader is reproduced:
//
//   - An instruction that fails ends its program, and the outline is used as far as the program got. FreeType more
//     often skips the one instruction, or the one point, and carries on. Either way a shipped font with a slightly
//     broken program still renders.
//   - Popping an empty stack yields a zero for each missing operand alone, and the instructions that repeat under
//     SLOOP run their full count on those zeros.
//   - The font program runs at the size being prepared, and each glyph program of a composite starts from the state
//     the control value program left, not from what an earlier component did to it.
//   - Limits FreeType applies while running are absent: the cap on the twilight zone, and the loop budget tied to the
//     glyph count.
//   - The vertical phantom points come from the OS/2 typographic metrics.
//   - ODD and EVEN test the integer bit alone, as FreeType itself now does (it masked the fraction in too, through
//     2.13.2).
//
// Instructions that move points horizontally are ignored ("backward compatibility" in the interpreter's terms), because
// subpixel rendering resolves horizontal detail better than rounding to whole pixels does, so hinting changes where
// rows of a glyph fall and leaves its widths and advances alone. The exception is a font that needs its instructions to
// draw correct contours at all (IsHintReliant), which runs them in full.
//
// Some things here are not Skrifa's. Four are things Skrifa gets wrong for valid fonts, and they follow FreeType until
// Skrifa does:
//
//   - MSIRP into the twilight zone starts the point from the reference point's original position. Skrifa leaves the
//     point's original position where it was (opMSIRP).
//   - SDPVTL on two points that coincide in the original outline sets both vectors without the rotation. Skrifa still
//     rotates the projection vector (dispatch).
//   - AA takes its operand off the stack. Skrifa leaves it there (dispatch).
//   - A component's scaled offset is scaled by the true length of the transform's columns. Skrifa uses an approximation
//     that is short for a rotated component (hypot16).
//
// Four are repairs FreeType makes to fonts in circulation, which cost nothing to make here: room for at least 64
// function definitions, any nonzero loca format read as long, a loca table that is out of order or overruns glyf
// mended, and a glyph record with no contours read as an empty glyph (NewFont, glyphData, load).
//
// And three guard against fonts that are damaged or hostile. The iterations of a repeating instruction count against
// the budget of a run (takeLoopCount). A component placed by matching points may only name points that have been
// loaded (loadComposite). And a size that rounds to zero is refused rather than hinted to nothing (NewInstance).
//
// Variable fonts are not handled: a Font reports HasVariations and the caller is expected not to hint an instance
// away from the default.
package tthint
