// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SubRunControl and SDFTMatrixRange: the use-SDFT decision — which device text sizes ride the distance-field atlas lane
// rather than direct masks or paths — plus the derived SDF strike font and the matrix range within which a cached SDFT
// subrun can be reused. The DF size buckets are per-platform: macOS gets an extra-large bucket.

package text

import (
	"runtime"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
)

// DF sizes and thresholds for usage of the small and medium sizes. For example, above smallDFFontLimit we will use the
// medium size. The large size is used up until the size at which we switch over to drawing as paths, as controlled by
// SubRunControl.
const (
	smallDFFontLimit      = 32
	mediumDFFontLimit     = 72
	largeDFFontLimit      = 162
	extraLargeDFFontLimit = 256 // macOS only
)

// SDFTMatrixRange is a min/max pair such that if matrix.MaxScale() is between them then this SDFT size can be reused.
type SDFTMatrixRange struct {
	matrixMin float32
	matrixMax float32
}

// MatrixInRange reports whether matrix's max scale falls within this range.
func (r *SDFTMatrixRange) MatrixInRange(matrix *geom.Matrix) bool {
	maxScale := matrix.MaxScale()
	return r.matrixMin < maxScale && maxScale <= r.matrixMax
}

// SubRunControl decides, for a given draw, whether text should render as a distance-field atlas subrun, a direct-mask
// subrun, or individual paths.
type SubRunControl struct {
	// minDistanceFieldFontSize: below this size (in device space) distance-field text will not be used.
	minDistanceFieldFontSize float32
	// maxDistanceFieldFontSize: above this size (in device space) distance-field text will not be used and glyphs will
	// be rendered from outline as individual paths.
	maxDistanceFieldFontSize float32
	ableToUseSDFT            bool
	ableToUsePerspectiveSDFT bool
	// forcePathAA: if true, glyphs drawn as paths are always anti-aliased regardless of any edge hinting.
	forcePathAA bool
}

// minSDFTRange returns the minimum device text size for SDFT use, forcing small text to the large bucket when
// useSDFTForSmallText is false.
func minSDFTRange(useSDFTForSmallText bool, minVal float32) float32 {
	if !useSDFTForSmallText {
		return largeDFFontLimit
	}
	return minVal
}

// NewSubRunControl returns a SubRunControl configured with the given SDFT policy (forcePathAA is false at every
// reachable construction site currently, but is kept for the path-lane rule).
func NewSubRunControl(ableToUseSDFT, useSDFTForSmallText, useSDFTForPerspectiveText bool, minSize, maxSize float32, forcePathAA bool) SubRunControl {
	if !(0 < minSize && minSize <= maxSize) {
		panic("invalid SDFT size range")
	}
	return SubRunControl{
		minDistanceFieldFontSize: minSDFTRange(useSDFTForSmallText, minSize),
		maxDistanceFieldFontSize: maxSize,
		ableToUseSDFT:            ableToUseSDFT,
		ableToUsePerspectiveSDFT: useSDFTForPerspectiveText,
		forcePathAA:              forcePathAA,
	}
}

// MaxSize returns the largest device text size that still uses distance-field text.
func (c *SubRunControl) MaxSize() float32 { return c.maxDistanceFieldFontSize }

// ForcePathAA reports whether path-rendered glyphs are always anti-aliased.
func (c *SubRunControl) ForcePathAA() bool { return c.forcePathAA }

// IsDirect reports whether this draw should use direct (non-distance-field) masks.
func (c *SubRunControl) IsDirect(approximateDeviceTextSize float32, paint *canvas.Paint, matrix *geom.Matrix) bool {
	return !c.IsSDFT(approximateDeviceTextSize, paint, matrix) &&
		!matrix.HasPerspective() &&
		0 < approximateDeviceTextSize &&
		approximateDeviceTextSize < font.SideTooBigForAtlas
}

// IsSDFT reports whether this draw should use the distance-field atlas lane.
func (c *SubRunControl) IsSDFT(approximateDeviceTextSize float32, paint *canvas.Paint, matrix *geom.Matrix) bool {
	wideStroke := paint.Style == canvas.StyleStroke && paint.StrokeWidth > 0
	return c.ableToUseSDFT &&
		paint.MaskFilter == nil &&
		(paint.Style == canvas.StyleFill || wideStroke) &&
		0 < approximateDeviceTextSize &&
		(c.ableToUsePerspectiveSDFT || !matrix.HasPerspective()) &&
		(c.minDistanceFieldFontSize <= approximateDeviceTextSize || matrix.HasPerspective()) &&
		approximateDeviceTextSize <= c.maxDistanceFieldFontSize
}

// GetSDFFont produces a font, a scale factor from the nominal size to the source space size, and the matrix range where
// this font can be reused.
func (c *SubRunControl) GetSDFFont(f *font.Font, viewMatrix *geom.Matrix, textLoc geom.Point) (dfFont font.Font, strikeToSourceScale float32, matrixRange SDFTMatrixRange) {
	textSize := f.Size()
	scaledTextSize := approximateTransformedTextSize(f, viewMatrix, textLoc)
	if scaledTextSize <= 0 || geom.ScalarNearlyEqual(textSize, scaledTextSize) {
		scaledTextSize = textSize
	}

	dfFont = *f

	var dfMaskScaleFloor, dfMaskScaleCeil, dfMaskSize float32
	switch {
	case scaledTextSize <= smallDFFontLimit:
		dfMaskScaleFloor = c.minDistanceFieldFontSize
		dfMaskScaleCeil = smallDFFontLimit
		dfMaskSize = smallDFFontLimit
	case scaledTextSize <= mediumDFFontLimit:
		dfMaskScaleFloor = smallDFFontLimit
		dfMaskScaleCeil = mediumDFFontLimit
		dfMaskSize = mediumDFFontLimit
	case runtime.GOOS == "darwin" && scaledTextSize <= largeDFFontLimit:
		// macOS carries a fourth (extra-large) bucket.
		dfMaskScaleFloor = mediumDFFontLimit
		dfMaskScaleCeil = largeDFFontLimit
		dfMaskSize = largeDFFontLimit
	case runtime.GOOS == "darwin":
		dfMaskScaleFloor = largeDFFontLimit
		dfMaskScaleCeil = c.maxDistanceFieldFontSize
		dfMaskSize = extraLargeDFFontLimit
	default:
		dfMaskScaleFloor = mediumDFFontLimit
		dfMaskScaleCeil = c.maxDistanceFieldFontSize
		dfMaskSize = largeDFFontLimit
	}

	dfFont.SetSize(dfMaskSize)
	dfFont.SetEdging(font.EdgingAntiAlias)
	dfFont.SetForceAutoHinting(false)
	dfFont.SetHinting(font.HintingNormal)

	// The sub-pixel position will always happen when transforming to the screen.
	dfFont.SetSubpixel(false)

	return dfFont, textSize / dfMaskSize, SDFTMatrixRange{
		matrixMin: dfMaskScaleFloor / textSize,
		matrixMax: dfMaskScaleCeil / textSize,
	}
}
