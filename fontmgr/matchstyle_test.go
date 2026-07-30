// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A 36-set / 277-case corpus exercising the CSS3 style match (selectStyleCSS3's first candidate; style names
// camel-cased). The invalidFontStyle sentinel marks cases expected to produce a null match (empty set).

package fontmgr

import (
	"math"
	"slices"
	"sort"
	"testing"

	"github.com/richardwilkes/canvas/font"
)

// rankStylesCSS3 materializes the whole descending-score walk by never accepting a candidate, which is also the
// fallthrough path every caller relies on when a face fails to load.
func rankStylesCSS3(pattern font.Style, count int, styleAt func(int) font.Style) []int {
	order := make([]int, 0, count)
	if selectStyleCSS3(pattern, count, styleAt, func(i int) bool {
		order = append(order, i)
		return false
	}) {
		panic("selectStyleCSS3 reported a match when nothing was accepted")
	}
	return order
}

var (
	invalidFontStyle = font.NewStyle(101, font.WidthNormal, font.SlantUpright)

	condensedNormal100 = font.NewStyle(font.WeightThin, font.WidthCondensed, font.SlantUpright)
	condensedNormal900 = font.NewStyle(font.WeightBlack, font.WidthCondensed, font.SlantUpright)
	condensedItalic100 = font.NewStyle(font.WeightThin, font.WidthCondensed, font.SlantItalic)
	condensedItalic900 = font.NewStyle(font.WeightBlack, font.WidthCondensed, font.SlantItalic)
	condensedObliqu100 = font.NewStyle(font.WeightThin, font.WidthCondensed, font.SlantOblique)
	condensedObliqu900 = font.NewStyle(font.WeightBlack, font.WidthCondensed, font.SlantOblique)
	expandedNormal100  = font.NewStyle(font.WeightThin, font.WidthExpanded, font.SlantUpright)
	expandedNormal900  = font.NewStyle(font.WeightBlack, font.WidthExpanded, font.SlantUpright)
	expandedItalic100  = font.NewStyle(font.WeightThin, font.WidthExpanded, font.SlantItalic)
	expandedItalic900  = font.NewStyle(font.WeightBlack, font.WidthExpanded, font.SlantItalic)
	expandedObliqu100  = font.NewStyle(font.WeightThin, font.WidthExpanded, font.SlantOblique)
	expandedObliqu900  = font.NewStyle(font.WeightBlack, font.WidthExpanded, font.SlantOblique)

	// One weight and slant across the whole width axis, for the cases that rank differing widths against each other.
	condensedNormal400     = font.NewStyle(font.WeightNormal, font.WidthCondensed, font.SlantUpright)
	semiExpandedNormal400  = font.NewStyle(font.WeightNormal, font.WidthSemiExpanded, font.SlantUpright)
	expandedNormal400      = font.NewStyle(font.WeightNormal, font.WidthExpanded, font.SlantUpright)
	extraExpandedNormal400 = font.NewStyle(font.WeightNormal, font.WidthExtraExpanded, font.SlantUpright)
	ultraExpandedNormal400 = font.NewStyle(font.WeightNormal, font.WidthUltraExpanded, font.SlantUpright)

	normalNormal100 = font.NewStyle(font.WeightThin, font.WidthNormal, font.SlantUpright)
	normalNormal300 = font.NewStyle(font.WeightLight, font.WidthNormal, font.SlantUpright)
	normalNormal400 = font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantUpright)
	normalNormal500 = font.NewStyle(font.WeightMedium, font.WidthNormal, font.SlantUpright)
	normalNormal600 = font.NewStyle(font.WeightSemiBold, font.WidthNormal, font.SlantUpright)
	normalNormal900 = font.NewStyle(font.WeightBlack, font.WidthNormal, font.SlantUpright)
)

// TestMatchStyleCSS3 exercises the style match MatchStyle performs — the first selectStyleCSS3 candidate — against
// the corpus above.
func TestMatchStyleCSS3(t *testing.T) {
	for ti, test := range matchStyleCSS3Tests {
		for ci, c := range test.cases {
			pattern, want := c[0], c[1]
			idx := -1
			if order := rankStylesCSS3(pattern, len(test.set),
				func(i int) font.Style { return test.set[i] }); len(order) > 0 {
				idx = order[0]
			}
			if idx < 0 {
				if want != invalidFontStyle {
					t.Errorf("test %d case %d: no match, want (%d,%d,%d)", ti, ci,
						want.Weight(), want.Width(), want.Slant())
				}
				continue
			}
			if got := test.set[idx]; got != want {
				t.Errorf("test %d case %d: pattern (%d,%d,%d) matched (%d,%d,%d), want (%d,%d,%d)", ti, ci,
					pattern.Weight(), pattern.Width(), pattern.Slant(),
					got.Weight(), got.Width(), got.Slant(),
					want.Weight(), want.Width(), want.Slant())
			}
		}
	}
}

// TestCSS3ScoreTierPriority pins the priority css3Score documents — width, then slant, then weight — as a strict tier
// ordering: no difference in a lower tier may ever outrank a difference in a higher one. Upstream Skia shifts both
// tiers by 8 bits, which is one bit too few for a weight addend that reaches 1000, so a weight score >= 256 carries
// into the slant field.
func TestCSS3ScoreTierPriority(t *testing.T) {
	pattern := font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantUpright)
	weights := []int{
		font.WeightInvisible, font.WeightThin, font.WeightLight, font.WeightNormal,
		font.WeightMedium, font.WeightBold, font.WeightBlack, font.WeightExtraBlack,
	}
	// Both lists run from the pattern's best match to its worst, per the tier's own scoring: the slant table's upright
	// row is upright > oblique > italic, and the width score for a normal-width pattern falls off toward condensed and
	// then jumps below every condensed width for the expanded ones.
	slants := []font.Slant{font.SlantUpright, font.SlantOblique, font.SlantItalic}
	widths := []int{
		font.WidthNormal, font.WidthSemiCondensed, font.WidthCondensed, font.WidthExtraCondensed,
		font.WidthUltraCondensed, font.WidthSemiExpanded, font.WidthExpanded, font.WidthExtraExpanded,
		font.WidthUltraExpanded,
	}

	// Within one width, every slant tier must sit entirely above the next: the worst score of the better slant beats
	// the best score of the worse one, whatever the weights are.
	for _, width := range widths {
		prevLow, prevSlant := math.MaxInt, font.SlantUpright
		for _, slant := range slants {
			low, high := math.MaxInt, math.MinInt
			for _, weight := range weights {
				s := css3Score(pattern, font.NewStyle(weight, width, slant))
				low, high = min(low, s), max(high, s)
			}
			if high >= prevLow {
				t.Errorf("width %d: slant %d scores up to %d, at or above slant %d's floor of %d — the weight tier "+
					"carries into the slant tier", width, slant, high, prevSlant, prevLow)
			}
			prevLow, prevSlant = low, slant
		}
	}

	// And every width tier must sit entirely above the next, whatever the slants and weights are.
	prevLow, prevWidth := math.MaxInt, font.WidthNormal
	for _, width := range widths {
		low, high := math.MaxInt, math.MinInt
		for _, slant := range slants {
			for _, weight := range weights {
				s := css3Score(pattern, font.NewStyle(weight, width, slant))
				low, high = min(low, s), max(high, s)
			}
		}
		if high >= prevLow {
			t.Errorf("width %d scores up to %d, at or above width %d's floor of %d — a lower tier carries into the "+
				"width tier", width, high, prevWidth, prevLow)
		}
		prevLow, prevWidth = low, width
	}

	// The reported case: for an upright 400 request, an exact-weight italic face must not outrank an upright 900 one.
	italic400 := font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantItalic)
	upright900 := font.NewStyle(font.WeightBlack, font.WidthNormal, font.SlantUpright)
	set := []font.Style{italic400, upright900}
	order := rankStylesCSS3(pattern, len(set), func(i int) font.Style { return set[i] })
	if got := set[order[0]]; got != upright900 {
		t.Errorf("upright 400 request ranked (%d,%d,%d) first, want the upright 900 face",
			got.Weight(), got.Width(), got.Slant())
	}
}

// TestCSS3ScoreWidthTierOrder pins the width tier — the tier with the greatest priority — against the CSS3
// font-stretch rule, for every pattern width against every candidate width: the exact width ranks first, then, for a
// pattern wider than normal, the wider widths ascending followed by the narrower ones descending; for a normal or
// narrower pattern, the narrower widths descending followed by the wider ones ascending. Upstream Skia's strict
// `current.width() > pattern.width()` test drops the exact-width candidate into the other branch, where it scores its
// raw width instead of the full 10, so for a pattern of width 6, 7 or 8 a wider face outranks the exact one.
func TestCSS3ScoreWidthTierOrder(t *testing.T) {
	widths := []int{
		font.WidthUltraCondensed, font.WidthExtraCondensed, font.WidthCondensed, font.WidthSemiCondensed,
		font.WidthNormal, font.WidthSemiExpanded, font.WidthExpanded, font.WidthExtraExpanded,
		font.WidthUltraExpanded,
	}
	// Every candidate carries the same weight and slant, so only the width tier separates them.
	set := make([]font.Style, len(widths))
	for i, w := range widths {
		set[i] = font.NewStyle(font.WeightNormal, w, font.SlantUpright)
	}
	styleAt := func(i int) font.Style { return set[i] }
	for _, patternWidth := range widths {
		want := []int{patternWidth}
		narrower := func() {
			for w := patternWidth - 1; w >= font.WidthUltraCondensed; w-- {
				want = append(want, w)
			}
		}
		wider := func() {
			for w := patternWidth + 1; w <= font.WidthUltraExpanded; w++ {
				want = append(want, w)
			}
		}
		if patternWidth <= font.WidthNormal {
			narrower()
			wider()
		} else {
			wider()
			narrower()
		}
		pattern := font.NewStyle(font.WeightNormal, patternWidth, font.SlantUpright)
		got := make([]int, 0, len(set))
		for _, i := range rankStylesCSS3(pattern, len(set), styleAt) {
			got = append(got, set[i].Width())
		}
		if !slices.Equal(got, want) {
			t.Errorf("width %d pattern ranked the widths %v, want %v", patternWidth, got, want)
		}
	}
}

// TestSelectStyleCSS3WalkOrder pins the order the partial selection walks: exactly the stable descending-score order a
// full sort of the candidates produces. The walk extracts one maximum at a time so the common case (the first
// candidate answers) never orders the rest, and every caller depends on the remainder still arriving best-first, with
// ties keeping first-wins preference.
func TestSelectStyleCSS3WalkOrder(t *testing.T) {
	sets := make([][]font.Style, 0, len(matchStyleCSS3Tests)+1)
	for _, test := range matchStyleCSS3Tests {
		sets = append(sets, test.set)
	}
	// A tie-heavy set: every style appears three times, so first-wins preference is what fixes the order.
	ties := make([]font.Style, 0, 18)
	for range 3 {
		ties = append(ties, normalNormal100, normalNormal400, normalNormal900,
			condensedItalic100, expandedObliqu900, normalNormal400)
	}
	sets = append(sets, ties)

	patterns := []font.Style{
		normalNormal400, normalNormal900, condensedItalic100, expandedObliqu900, condensedNormal900,
	}
	for si, set := range sets {
		for _, pattern := range patterns {
			want := make([]int, len(set))
			for i := range want {
				want[i] = i
			}
			// The reference ordering: a stable sort by descending score, which leaves equal scores in index order.
			sort.SliceStable(want, func(a, b int) bool {
				return css3Score(pattern, set[want[a]]) > css3Score(pattern, set[want[b]])
			})
			got := rankStylesCSS3(pattern, len(set), func(i int) font.Style { return set[i] })
			if len(got) != len(want) {
				t.Fatalf("set %d: walked %d candidates, want %d", si, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("set %d pattern (%d,%d,%d): walk order %v, want %v", si,
						pattern.Weight(), pattern.Width(), pattern.Slant(), got, want)
					break
				}
			}
		}
	}
}

// TestSelectStyleCSS3Fallthrough covers the reason the walk exists at all: a candidate the caller turns down (a face
// that no longer loads) must hand the next-best one over, and the walk must report whether anything was accepted.
func TestSelectStyleCSS3Fallthrough(t *testing.T) {
	set := []font.Style{normalNormal100, normalNormal400, normalNormal500, normalNormal900}
	pattern := normalNormal400
	styleAt := func(i int) font.Style { return set[i] }
	order := rankStylesCSS3(pattern, len(set), styleAt)

	// Turning down the first k candidates must yield the (k+1)-th in the same order, one at a time.
	for k := range len(set) {
		var visited []int
		if !selectStyleCSS3(pattern, len(set), styleAt, func(i int) bool {
			visited = append(visited, i)
			return len(visited) > k
		}) {
			t.Fatalf("rejecting %d candidates reported no match", k)
		}
		if len(visited) != k+1 {
			t.Fatalf("rejecting %d candidates visited %d, want %d", k, len(visited), k+1)
		}
		for i, idx := range visited {
			if idx != order[i] {
				t.Errorf("rejecting %d candidates visited %v, want the walk order %v", k, visited, order[:k+1])
				break
			}
		}
	}

	// Turning every candidate down visits them all and reports no match.
	visits := 0
	if selectStyleCSS3(pattern, len(set), styleAt, func(int) bool {
		visits++
		return false
	}) {
		t.Error("rejecting every candidate reported a match")
	}
	if visits != len(set) {
		t.Errorf("rejecting every candidate visited %d, want %d", visits, len(set))
	}

	// An empty set never calls accept and reports no match.
	if selectStyleCSS3(pattern, 0, styleAt, func(int) bool {
		t.Error("empty set called accept")
		return true
	}) {
		t.Error("empty set reported a match")
	}
}

// TestCSS3ScoreRawStyleComponents covers styles that never passed through font.NewStyle. font.Style is an exported
// int32 whose packed form is the documented round-trip, so a caller can hand any component in: an out-of-range slant
// used to index the 3x3 slant table out of range and panic, and an out-of-range width or weight overflows its tier's
// field. Each component must instead score as its pinned equivalent.
func TestCSS3ScoreRawStyleComponents(t *testing.T) {
	for _, test := range []struct {
		name   string
		raw    font.Style
		pinned font.Style
	}{
		{
			name:   "slant above oblique",
			raw:    font.Style(3 << 24),
			pinned: font.NewStyle(font.WeightInvisible, font.WidthUltraCondensed, font.SlantOblique),
		},
		{
			name:   "slant far above oblique",
			raw:    font.Style(127<<24 | font.WidthNormal<<16 | font.WeightNormal),
			pinned: font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantOblique),
		},
		{
			name:   "every component saturated",
			raw:    font.Style(-1),
			pinned: font.NewStyle(font.WeightExtraBlack, font.WidthUltraExpanded, font.SlantOblique),
		},
		{
			name:   "width above ultra-expanded",
			raw:    font.Style(255<<16 | font.WeightNormal),
			pinned: font.NewStyle(font.WeightNormal, font.WidthUltraExpanded, font.SlantUpright),
		},
		{
			name:   "width below ultra-condensed",
			raw:    font.Style(font.WeightNormal),
			pinned: font.NewStyle(font.WeightNormal, font.WidthUltraCondensed, font.SlantUpright),
		},
		{
			name:   "weight above extra-black",
			raw:    font.Style(font.WidthNormal<<16 | 0xFFFF),
			pinned: font.NewStyle(font.WeightExtraBlack, font.WidthNormal, font.SlantUpright),
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Both as the pattern and as a candidate: css3Score reads the same fields of each.
			for _, other := range []font.Style{normalNormal400, condensedItalic900, expandedObliqu100} {
				if got, want := css3Score(test.raw, other), css3Score(test.pinned, other); got != want {
					t.Errorf("as pattern against (%d,%d,%d): score %d, want the pinned style's %d",
						other.Weight(), other.Width(), other.Slant(), got, want)
				}
				if got, want := css3Score(other, test.raw), css3Score(other, test.pinned); got != want {
					t.Errorf("as candidate against (%d,%d,%d): score %d, want the pinned style's %d",
						other.Weight(), other.Width(), other.Slant(), got, want)
				}
			}
		})
	}
}

// TestMatchRawStyleThroughPublicAPI walks the raw-style panic out through the exported entry points that reach
// css3Score, over the hermetic corpus.
func TestMatchRawStyleThroughPublicAPI(t *testing.T) {
	m := newTestManager(t)
	raw := font.Style(3 << 24) // slant 3: one past SlantOblique
	if tf := m.MatchFamily("Test").MatchStyle(raw); tf == nil {
		t.Error("StyleSet.MatchStyle with a raw slant found no face")
	}
	if tf := m.MatchFamilyStyle("Roboto", raw); tf == nil {
		t.Error("MatchFamilyStyle with a raw slant found no face")
	}
	if tf := m.MatchFamilyStyleCharacter("", raw, nil, 'x'); tf == nil {
		t.Error("MatchFamilyStyleCharacter with a raw slant found no face")
	}
}

// benchmarkStyleSet builds count distinct-ish styles, the shape of a cross-family fallback candidate set.
func benchmarkStyleSet(count int) []font.Style {
	weights := []int{
		font.WeightThin, font.WeightLight, font.WeightNormal, font.WeightMedium, font.WeightBold, font.WeightBlack,
	}
	slants := []font.Slant{font.SlantUpright, font.SlantItalic, font.SlantOblique}
	set := make([]font.Style, count)
	for i := range set {
		set[i] = font.NewStyle(weights[i%len(weights)], font.WidthUltraCondensed+(i/len(weights))%9,
			slants[(i/7)%len(slants)])
	}
	return set
}

// BenchmarkSelectStyleCSS3FirstAccepted measures the common case on the cross-family fallback path: a large candidate
// set whose best-scoring face answers immediately. The walk costs two linear passes — measured at n=3000, ~50x less
// than the full insertion sort it replaced (51 µs against 2.5 ms).
func BenchmarkSelectStyleCSS3FirstAccepted(b *testing.B) {
	set := benchmarkStyleSet(3000)
	pattern := font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantUpright)
	styleAt := func(i int) font.Style { return set[i] }
	for b.Loop() {
		selectStyleCSS3(pattern, len(set), styleAt, func(int) bool { return true })
	}
}

// BenchmarkSelectStyleCSS3FullWalk measures the worst case the partial selection cannot improve on: every candidate
// turned down, so the walk orders the whole set.
func BenchmarkSelectStyleCSS3FullWalk(b *testing.B) {
	set := benchmarkStyleSet(300)
	pattern := font.NewStyle(font.WeightNormal, font.WidthNormal, font.SlantUpright)
	styleAt := func(i int) font.Style { return set[i] }
	for b.Loop() {
		selectStyleCSS3(pattern, len(set), styleAt, func(int) bool { return false })
	}
}

var matchStyleCSS3Tests = []struct {
	set   []font.Style
	cases [][2]font.Style
}{
	{
		set: []font.Style{normalNormal500, normalNormal400},
		cases: [][2]font.Style{
			{normalNormal400, normalNormal400},
			{normalNormal500, normalNormal500},
		},
	},
	{
		set: []font.Style{normalNormal500, normalNormal300},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal300},
			{normalNormal400, normalNormal500},
			{normalNormal500, normalNormal500},
		},
	},
	{
		set: []font.Style{
			condensedNormal100, condensedNormal900, condensedItalic100, condensedItalic900,
			expandedNormal100, expandedNormal900, expandedItalic100, expandedItalic900,
		},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedItalic100, expandedNormal100, expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{condensedNormal900, condensedItalic900, expandedNormal900, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedNormal900, expandedNormal100, expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{condensedNormal100, expandedNormal100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal100},
		},
	},
	{
		set: []font.Style{condensedNormal900, expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal900},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal900},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{condensedItalic100, condensedItalic900, expandedItalic100, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedItalic100, expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{condensedItalic900, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic900},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, expandedItalic900},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedNormal900, condensedItalic100, condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal100},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic100},
		},
	},
	{
		set: []font.Style{condensedNormal900, condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedNormal900},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedItalic900},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{condensedNormal100, condensedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedNormal100},
			{expandedItalic900, condensedNormal900},
		},
	},
	{
		set: []font.Style{condensedNormal100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal100},
			{condensedNormal900, condensedNormal100},
			{condensedItalic100, condensedNormal100},
			{condensedItalic900, condensedNormal100},
			{expandedNormal100, condensedNormal100},
			{expandedNormal900, condensedNormal100},
			{expandedItalic100, condensedNormal100},
			{expandedItalic900, condensedNormal100},
		},
	},
	{
		set: []font.Style{condensedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedNormal900},
			{condensedNormal900, condensedNormal900},
			{condensedItalic100, condensedNormal900},
			{condensedItalic900, condensedNormal900},
			{expandedNormal100, condensedNormal900},
			{expandedNormal900, condensedNormal900},
			{expandedItalic100, condensedNormal900},
			{expandedItalic900, condensedNormal900},
		},
	},
	{
		set: []font.Style{condensedItalic100, condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedItalic100},
			{expandedNormal900, condensedItalic900},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{condensedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic100},
			{condensedNormal900, condensedItalic100},
			{condensedItalic100, condensedItalic100},
			{condensedItalic900, condensedItalic100},
			{expandedNormal100, condensedItalic100},
			{expandedNormal900, condensedItalic100},
			{expandedItalic100, condensedItalic100},
			{expandedItalic900, condensedItalic100},
		},
	},
	{
		set: []font.Style{condensedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, condensedItalic900},
			{condensedNormal900, condensedItalic900},
			{condensedItalic100, condensedItalic900},
			{condensedItalic900, condensedItalic900},
			{expandedNormal100, condensedItalic900},
			{expandedNormal900, condensedItalic900},
			{expandedItalic100, condensedItalic900},
			{expandedItalic900, condensedItalic900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedNormal900, expandedItalic100, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{condensedObliqu100, expandedItalic100},
			{condensedObliqu900, expandedItalic900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
			{expandedObliqu100, expandedItalic100},
			{expandedObliqu900, expandedItalic900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal100},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{expandedNormal900, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal900},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedItalic900},
			{condensedItalic900, expandedItalic900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedNormal100},
			{condensedItalic900, expandedNormal900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{expandedNormal100},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal100},
			{condensedItalic100, expandedNormal100},
			{condensedItalic900, expandedNormal100},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal100},
			{expandedItalic100, expandedNormal100},
			{expandedItalic900, expandedNormal100},
		},
	},
	{
		set: []font.Style{expandedNormal900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal900},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedNormal900},
			{condensedItalic900, expandedNormal900},
			{expandedNormal100, expandedNormal900},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedNormal900},
			{expandedItalic900, expandedNormal900},
		},
	},
	{
		set: []font.Style{expandedItalic100, expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedItalic100},
			{condensedNormal900, expandedItalic900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{expandedItalic100},
		cases: [][2]font.Style{
			{condensedNormal100, expandedItalic100},
			{condensedNormal900, expandedItalic100},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic100},
			{expandedNormal100, expandedItalic100},
			{expandedNormal900, expandedItalic100},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic100},
		},
	},
	{
		set: []font.Style{expandedItalic900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedItalic900},
			{condensedNormal900, expandedItalic900},
			{condensedItalic100, expandedItalic900},
			{condensedItalic900, expandedItalic900},
			{expandedNormal100, expandedItalic900},
			{expandedNormal900, expandedItalic900},
			{expandedItalic100, expandedItalic900},
			{expandedItalic900, expandedItalic900},
		},
	},
	{
		set: []font.Style{normalNormal100, normalNormal900},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal100},
			{normalNormal400, normalNormal100},
			{normalNormal500, normalNormal100},
			{normalNormal600, normalNormal900},
		},
	},
	{
		set: []font.Style{normalNormal100, normalNormal400, normalNormal900},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal100},
			{normalNormal400, normalNormal400},
			{normalNormal500, normalNormal400},
			{normalNormal600, normalNormal900},
		},
	},
	{
		set: []font.Style{normalNormal100, normalNormal500, normalNormal900},
		cases: [][2]font.Style{
			{normalNormal300, normalNormal100},
			{normalNormal400, normalNormal500},
			{normalNormal500, normalNormal500},
			{normalNormal600, normalNormal900},
		},
	},
	{
		set: []font.Style{},
		cases: [][2]font.Style{
			{normalNormal300, invalidFontStyle},
			{normalNormal400, invalidFontStyle},
			{normalNormal500, invalidFontStyle},
			{normalNormal600, invalidFontStyle},
		},
	},
	// The width tier ranked against itself. Every face here differs only in width, so nothing below the
	// highest-priority tier can affect the choice: the exact width must win outright, and where it is absent the
	// pattern's direction decides.
	{
		set: []font.Style{expandedNormal400, extraExpandedNormal400},
		cases: [][2]font.Style{
			{expandedNormal400, expandedNormal400}, // the exact width, not the wider face beside it
			{extraExpandedNormal400, extraExpandedNormal400},
			{semiExpandedNormal400, expandedNormal400},
			{ultraExpandedNormal400, extraExpandedNormal400},
			{normalNormal400, expandedNormal400},
			{condensedNormal400, expandedNormal400},
		},
	},
	{
		set: []font.Style{
			condensedNormal400, normalNormal400, semiExpandedNormal400, expandedNormal400, extraExpandedNormal400,
			ultraExpandedNormal400,
		},
		cases: [][2]font.Style{
			{condensedNormal400, condensedNormal400},
			{normalNormal400, normalNormal400},
			{semiExpandedNormal400, semiExpandedNormal400},
			{expandedNormal400, expandedNormal400},
			{extraExpandedNormal400, extraExpandedNormal400},
			{ultraExpandedNormal400, ultraExpandedNormal400},
		},
	},
	{
		// No exact width on offer: a wider-than-normal pattern takes the nearest wider face, and only then narrower
		// ones.
		set: []font.Style{condensedNormal400, normalNormal400, ultraExpandedNormal400},
		cases: [][2]font.Style{
			{semiExpandedNormal400, ultraExpandedNormal400},
			{expandedNormal400, ultraExpandedNormal400},
			{extraExpandedNormal400, ultraExpandedNormal400},
		},
	},
	{
		set: []font.Style{
			expandedNormal100, expandedNormal900, expandedItalic100, expandedItalic900,
			expandedObliqu100, expandedObliqu900,
		},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{condensedObliqu100, expandedObliqu100},
			{condensedObliqu900, expandedObliqu900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
			{expandedObliqu100, expandedObliqu100},
			{expandedObliqu900, expandedObliqu900},
		},
	},
	{
		set: []font.Style{expandedNormal100, expandedNormal900, expandedObliqu100, expandedObliqu900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedNormal100},
			{condensedNormal900, expandedNormal900},
			{condensedItalic100, expandedObliqu100},
			{condensedItalic900, expandedObliqu900},
			{condensedObliqu100, expandedObliqu100},
			{condensedObliqu900, expandedObliqu900},
			{expandedNormal100, expandedNormal100},
			{expandedNormal900, expandedNormal900},
			{expandedItalic100, expandedObliqu100},
			{expandedItalic900, expandedObliqu900},
			{expandedObliqu100, expandedObliqu100},
			{expandedObliqu900, expandedObliqu900},
		},
	},
	{
		set: []font.Style{expandedItalic100, expandedItalic900, expandedObliqu100, expandedObliqu900},
		cases: [][2]font.Style{
			{condensedNormal100, expandedObliqu100},
			{condensedNormal900, expandedObliqu900},
			{condensedItalic100, expandedItalic100},
			{condensedItalic900, expandedItalic900},
			{condensedObliqu100, expandedObliqu100},
			{condensedObliqu900, expandedObliqu900},
			{expandedNormal100, expandedObliqu100},
			{expandedNormal900, expandedObliqu900},
			{expandedItalic100, expandedItalic100},
			{expandedItalic900, expandedItalic900},
			{expandedObliqu100, expandedObliqu100},
			{expandedObliqu900, expandedObliqu900},
		},
	},
}
