// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-family face list, with CSS3 style-matching scoring (css3Score, walked by selectStyleCSS3) used as the one
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
	var match *font.Typeface
	selectStyleCSS3(pattern, len(s.faces), func(i int) font.Style { return s.faces[i].style() }, func(i int) bool {
		match = s.faces[i].typeface()
		return match != nil
	})
	return match
}

// css3SlantBits and css3WeightBits are the field widths css3Score reserves for the tiers below the one being shifted:
// the slant addend spans [1, 3] and the weight addend [0, 1000] (font.NewStyle pins weight to WeightExtraBlack), so two
// and ten bits hold them exactly. Upstream Skia's matchStyleCSS3 shifts by 8 for both, which is one bit too few for the
// weight tier: any weight score >= 256 carries into the slant field and inverts the documented priority, e.g. an
// italic 400 face outscoring an upright 900 face for an upright 400 request.
const (
	css3SlantBits  = 2
	css3WeightBits = 10
)

// css3Score computes the CSS3 match score of one candidate style against the pattern. Width (CSS stretch) has the
// greatest priority, then slant, then weight; each tier shifts left by the width of the tiers below so it dominates
// them. Higher scores are better.
//
// The wider-than-normal width branch tests current >= pattern where upstream Skia's matchStyleCSS3 tests current >
// pattern, which is the second deliberate divergence in this function (see css3SlantBits above for the first). CSS3
// ranks an exact width match ahead of every other candidate, and upstream's strict comparison drops the exact-width
// face into the else, scoring it its raw width instead of the full 10 — for a pattern of width 6, 7 or 8, that lets a
// wider face outrank it, e.g. an ExtraExpanded face beating an Expanded one for an Expanded request. The narrower-than-
// normal branch above already tests current <= pattern, so this makes the two branches mirror images, as they read.
func css3Score(pattern, current font.Style) int {
	// font.Style is an exported int32 whose packed layout is the documented round-trip (font.Style(storedInt32)) and it
	// has no validating constructor from a raw value, so a style that never came from font.NewStyle can carry any
	// component: a slant above SlantOblique indexes slantScore out of range and panics, and an out-of-range width or
	// weight yields a tier addend too wide for its field, carrying into the tier above and inverting the documented
	// priority. Re-pinning both styles through the validating constructor is this port's stand-in for upstream Skia's
	// enum-typed constructor plus SkASSERT.
	pattern, current = pinStyle(pattern), pinStyle(current)
	score := 0

	// CSS stretch / font.Style's Width. Takes priority over everything else.
	if pattern.Width() <= font.WidthNormal {
		if current.Width() <= pattern.Width() {
			score += 10 - pattern.Width() + current.Width()
		} else {
			score += 10 - current.Width()
		}
	} else {
		if current.Width() >= pattern.Width() {
			score += 10 + pattern.Width() - current.Width()
		} else {
			score += current.Width()
		}
	}
	score <<= css3SlantBits + css3WeightBits

	// CSS style (normal, italic, oblique) / font.Style's Slant. Takes priority over all valid weights.
	slantScore := [3][3]int{
		/*                Upright Italic Oblique  [current] */
		/* Upright */ {3, 1, 2},
		/* Italic  */ {1, 3, 2},
		/* Oblique */ {1, 2, 3},
		/* [pattern] */
	}
	score += slantScore[pattern.Slant()][current.Slant()]
	score <<= css3WeightBits

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

// pinStyle re-pins a style's components to their legal ranges by round-tripping it through the validating constructor.
// A style built by font.NewStyle (every style this package produces itself) is returned unchanged.
func pinStyle(s font.Style) font.Style { return font.NewStyle(s.Weight(), s.Width(), s.Slant()) }

// selectStyleCSS3 walks the candidates in descending css3Score order — ties keep first-wins preference — handing each
// index to accept until it returns true, and reports whether any did. It is the one style-distance algorithm:
// MatchStyle and matchCovering both walk it, so an unloadable face falls through to the next-best style rather than
// failing the match.
//
// The walk selects partially: the scores are computed once, then each step extracts the maximum of the candidates not
// yet visited, so the common case — the best-scoring face loads and answers — costs two linear passes instead of a
// full ordering. That matters on the cross-family fallback path, where the candidate set is every face covering the
// character rather than one family's faces: matchCoveringTiered calls matchCovering once per tier and
// MatchFamilyStyleCharacter calls matchCoveringTiered once per BCP-47 tag, over a set that a full Noto install pushes
// into the thousands. Upstream Skia's matchStyleCSS3 is a single max scan with no fallthrough at all.
func selectStyleCSS3(pattern font.Style, count int, styleAt func(int) font.Style, accept func(int) bool) bool {
	if count == 0 {
		return false
	}
	pattern = pinStyle(pattern)
	scores := make([]int, count)
	for i := range count {
		scores[i] = css3Score(pattern, styleAt(i))
	}
	visited := make([]bool, count)
	for range count {
		best := -1
		for i := range count {
			if !visited[i] && (best == -1 || scores[i] > scores[best]) {
				best = i
			}
		}
		visited[best] = true
		if accept(best) {
			return true
		}
	}
	return false
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
