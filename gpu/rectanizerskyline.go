// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// RectanizerSkyline is a skyline rectangle packer: it tracks the current silhouette of already-placed rectangles and
// places each new rectangle at the lowest (then narrowest) fitting position. It is the packing algorithm used for atlas
// allocation.

package gpu

// skylineSegment is one horizontal run of the current skyline silhouette.
type skylineSegment struct {
	x     int
	y     int
	width int
}

// RectanizerSkyline packs rectangles into a fixed-size area using the skyline algorithm.
type RectanizerSkyline struct {
	skyline   []skylineSegment
	width     int
	height    int
	areaSoFar int
}

// MakeRectanizerSkyline returns a RectanizerSkyline for a w x h area.
func MakeRectanizerSkyline(w, h int) RectanizerSkyline {
	r := RectanizerSkyline{width: w, height: h}
	r.Reset()
	return r
}

// Width returns the packer's total width.
func (r *RectanizerSkyline) Width() int { return r.width }

// Height returns the packer's total height.
func (r *RectanizerSkyline) Height() int { return r.height }

// Reset clears all placed rectangles, returning the packer to an empty state.
func (r *RectanizerSkyline) Reset() {
	r.areaSoFar = 0
	r.skyline = r.skyline[:0]
	r.skyline = append(r.skyline, skylineSegment{x: 0, y: 0, width: r.width})
}

// PercentFull returns the fraction of the total area covered by placed rectangles.
func (r *RectanizerSkyline) PercentFull() float32 {
	return float32(r.areaSoFar) / (float32(r.width) * float32(r.height))
}

// AddRect attempts to add a width x height rect, returning its location on success.
func (r *RectanizerSkyline) AddRect(width, height int) (locX, locY int, ok bool) {
	if width > r.width || height > r.height || width < 0 || height < 0 {
		return 0, 0, false
	}

	// Find position for the new rectangle.
	bestWidth := r.width + 1
	bestX := 0
	bestY := r.height + 1
	bestIndex := -1
	for i := range r.skyline {
		if y, fits := r.rectangleFits(i, width, height); fits {
			// Minimize y position first, then width of skyline.
			if y < bestY || (y == bestY && r.skyline[i].width < bestWidth) {
				bestIndex = i
				bestWidth = r.skyline[i].width
				bestX = r.skyline[i].x
				bestY = y
			}
		}
	}

	// Add the rectangle to the skyline.
	if bestIndex != -1 {
		r.addSkylineLevel(bestIndex, bestX, bestY, width, height)
		r.areaSoFar += width * height
		return bestX, bestY, true
	}
	return 0, 0, false
}

// rectangleFits reports whether a width x height rect fits starting at the given skyline segment, and if so, the y
// position it would land at.
func (r *RectanizerSkyline) rectangleFits(skylineIndex, width, height int) (ypos int, fits bool) {
	x := r.skyline[skylineIndex].x
	if x+width > r.width {
		return 0, false
	}
	widthLeft := width
	i := skylineIndex
	y := r.skyline[skylineIndex].y
	for widthLeft > 0 {
		y = max(y, r.skyline[i].y)
		if y+height > r.height {
			return 0, false
		}
		widthLeft -= r.skyline[i].width
		i++
		if i >= len(r.skyline) && widthLeft > 0 {
			panic("skyline does not cover the full width")
		}
	}
	return y, true
}

// addSkylineLevel inserts a new skyline segment for a just-placed rectangle, then merges or shrinks neighboring
// segments so the skyline stays a valid, sorted silhouette.
func (r *RectanizerSkyline) addSkylineLevel(skylineIndex, x, y, width, height int) {
	newSegment := skylineSegment{x: x, y: y + height, width: width}
	r.skyline = append(r.skyline, skylineSegment{})
	copy(r.skyline[skylineIndex+1:], r.skyline[skylineIndex:])
	r.skyline[skylineIndex] = newSegment

	if newSegment.x+newSegment.width > r.width || newSegment.y > r.height {
		panic("skyline segment out of bounds")
	}

	// Delete the width of the new skyline segment from following ones.
	for i := skylineIndex + 1; i < len(r.skyline); i++ {
		// The new segment subsumes all or part of skyline[i].
		if r.skyline[i-1].x > r.skyline[i].x {
			panic("skyline not sorted")
		}
		if r.skyline[i].x < r.skyline[i-1].x+r.skyline[i-1].width {
			shrink := r.skyline[i-1].x + r.skyline[i-1].width - r.skyline[i].x
			r.skyline[i].x += shrink
			r.skyline[i].width -= shrink
			if r.skyline[i].width <= 0 {
				// Fully consumed.
				r.skyline = append(r.skyline[:i], r.skyline[i+1:]...)
				i--
			} else {
				// Only partially consumed.
				break
			}
		} else {
			break
		}
	}

	// Merge skyline segments at the same height.
	for i := 0; i < len(r.skyline)-1; i++ {
		if r.skyline[i].y == r.skyline[i+1].y {
			r.skyline[i].width += r.skyline[i+1].width
			r.skyline = append(r.skyline[:i+1], r.skyline[i+2:]...)
			i--
		}
	}
}
