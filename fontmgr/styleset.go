// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-family face list, with CSS3 style-matching scoring (css3Score, ranked by rankStylesCSS3) used as the one
// style-distance algorithm on every platform (the CoreText host's squared-metric variant agrees on exact matches, but
// can prefer a different face at unequal distances).

package fontmgr

import (
	tsfont "github.com/go-text/typesetting/font"
	"github.com/richardwilkes/canvas/font"
)

// StyleSet is the faces of one family. The zero value is the empty set.
type StyleSet struct {
	faces []*faceRec
}

// Count returns the number of faces in the set.
func (s *StyleSet) Count() int { return len(s.faces) }

// Style returns the index-th face's font style and style (subfamily) name. Out-of-range indices return the normal style
// and "" (the caller is expected to stay in range).
func (s *StyleSet) Style(index int) (style font.Style, name string) {
	if index < 0 || index >= len(s.faces) {
		return font.NormalStyle(), ""
	}
	return s.faces[index].style(), s.faces[index].styleName()
}

// CreateTypeface returns the index-th face's typeface; nil out of range or when the face no longer loads.
func (s *StyleSet) CreateTypeface(index int) *font.Typeface {
	if index < 0 || index >= len(s.faces) {
		return nil
	}
	return s.faces[index].typeface()
}

// MatchStyle returns the best CSS3 style match in the set; nil only for the empty set (or when no face in the set
// loads).
func (s *StyleSet) MatchStyle(pattern font.Style) *font.Typeface {
	order := rankStylesCSS3(pattern, len(s.faces), func(i int) font.Style { return s.faces[i].style() })
	for _, idx := range order {
		if tf := s.faces[idx].typeface(); tf != nil {
			return tf
		}
	}
	return nil
}

// css3Score computes the CSS3 match score of one candidate style against the pattern. Width (CSS stretch) has the
// greatest priority, then slant, then weight; each tier shifts left so it dominates the tiers below. Higher scores are
// better.
func css3Score(pattern, current font.Style) int {
	score := 0

	// CSS stretch / font.Style's Width. Takes priority over everything else.
	if pattern.Width() <= font.WidthNormal {
		if current.Width() <= pattern.Width() {
			score += 10 - pattern.Width() + current.Width()
		} else {
			score += 10 - current.Width()
		}
	} else {
		if current.Width() > pattern.Width() {
			score += 10 + pattern.Width() - current.Width()
		} else {
			score += current.Width()
		}
	}
	score <<= 8

	// CSS style (normal, italic, oblique) / font.Style's Slant. Takes priority over all valid weights.
	slantScore := [3][3]int{
		/*                Upright Italic Oblique  [current] */
		/* Upright */ {3, 1, 2},
		/* Italic  */ {1, 3, 2},
		/* Oblique */ {1, 2, 3},
		/* [pattern] */
	}
	score += slantScore[pattern.Slant()][current.Slant()]
	score <<= 8

	// CSS weight / font.Style's Weight. The closer to the target weight, the higher the score. 1000 is the heaviest
	// recognized weight.
	pw, cw := pattern.Weight(), current.Weight()
	switch {
	case pw == cw:
		score += 1000
	case pw < 400: // less than 400 prefer lighter weights
		if cw <= pw {
			score += 1000 - pw + cw
		} else {
			score += 1000 - cw
		}
	case pw <= 500: // between 400 and 500 prefer heavier up to 500, then lighter weights
		switch {
		case cw >= pw && cw <= 500:
			score += 1000 + pw - cw
		case cw <= pw:
			score += 500 + cw
		default:
			score += 1000 - cw
		}
	default: // greater than 500 prefer heavier weights
		if cw > pw {
			score += 1000 + pw - cw
		} else {
			score += cw
		}
	}
	return score
}

// rankStylesCSS3 returns candidate indices in descending css3Score order (stable, so equal scores keep first-wins
// preference), empty for an empty set. It is the one style-distance algorithm: MatchStyle and matchCovering both walk
// it, so an unloadable face falls back to the next-best style rather than failing the match.
func rankStylesCSS3(pattern font.Style, count int, styleAt func(int) font.Style) []int {
	type scored struct{ idx, score int }
	list := make([]scored, count)
	for i := range count {
		list[i] = scored{idx: i, score: css3Score(pattern, styleAt(i))}
	}
	for i := 1; i < len(list); i++ { // insertion sort keeps it stable
		for j := i; j > 0 && list[j-1].score < list[j].score; j-- {
			list[j-1], list[j] = list[j], list[j-1]
		}
	}
	order := make([]int, count)
	for i, s := range list {
		order[i] = s.idx
	}
	return order
}

// styleFromAspect maps a fontscan footprint aspect (CSS weight, stretch ratio, italic flag) to the nearest font.Style.
// It is the no-I/O approximation used to score cross-family character-fallback candidates; oblique folds into italic
// because fontscan does not distinguish them.
func styleFromAspect(a tsfont.Aspect) font.Style {
	weight := int(a.Weight)
	if weight == 0 {
		weight = font.WeightNormal
	}
	width := font.WidthNormal
	if a.Stretch != 0 {
		ratios := [...]tsfont.Stretch{
			tsfont.StretchUltraCondensed, tsfont.StretchExtraCondensed, tsfont.StretchCondensed,
			tsfont.StretchSemiCondensed, tsfont.StretchNormal, tsfont.StretchSemiExpanded,
			tsfont.StretchExpanded, tsfont.StretchExtraExpanded, tsfont.StretchUltraExpanded,
		}
		width = 1
		bestDiff := absStretch(a.Stretch - ratios[0])
		for i := 1; i < len(ratios); i++ {
			if d := absStretch(a.Stretch - ratios[i]); d < bestDiff {
				bestDiff = d
				width = i + 1
			}
		}
	}
	slant := font.SlantUpright
	if a.Style == tsfont.StyleItalic {
		slant = font.SlantItalic
	}
	return font.NewStyle(weight, width, slant)
}

func absStretch(v tsfont.Stretch) tsfont.Stretch {
	if v < 0 {
		return -v
	}
	return v
}
