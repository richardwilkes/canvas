// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package main

import "slices"

// group is one band of related packages. Its colors are used for the node, the legend swatch, the package name in
// the matrix axes, and the marks in the matrix column belonging to the group.
type group struct {
	title   string // legend text; may contain XML entities, since it is emitted into the SVG as-is
	fill    string
	stroke  string
	text    string
	members []string
}

// groups drives both the coloring and the secondary ordering of equal-depth packages. Any package not named here
// falls into the last group, so a new package still renders; add it to the right band when one appears.
var groups = []group{
	{
		title:   "Core primitives",
		fill:    "#E4ECF9",
		stroke:  "#2C5EA8",
		text:    "#173A63",
		members: []string{"geom", "colorcore", "path", "raster", "imagecore", "shaders"},
	},
	{
		title:   "Path &amp; geometry ops",
		fill:    "#E2F0E8",
		stroke:  "#2C7D57",
		text:    "#164630",
		members: []string{"contour", "stroke", "pathops", "patheffect"},
	},
	{
		title:   "Filters &amp; effects",
		fill:    "#F2E9F7",
		stroke:  "#7A4A9C",
		text:    "#442659",
		members: []string{"colorfilter", "maskfilter", "filtercore", "imagefilter"},
	},
	{
		title:   "Fonts &amp; text",
		fill:    "#FBEDDF",
		stroke:  "#AF6414",
		text:    "#63380A",
		members: []string{"font", "fontmgr", "textblob"},
	},
	{
		title:   "Drawing API",
		fill:    "#FBE6E4",
		stroke:  "#B93A2B",
		text:    "#6A1F16",
		members: []string{"canvas", "surface"},
	},
	{
		title:   "Backends &amp; renderers",
		fill:    "#E6E8F5",
		stroke:  "#454A96",
		text:    "#252856",
		members: []string{"gpu", "gpu/gl", "gpu/text", "pdf"},
	},
	{
		title:   "I/O &amp; support",
		fill:    "#E4F0F1",
		stroke:  "#2F7278",
		text:    "#173F43",
		members: []string{"codecs", "codecs/internal/vp8enc", "stream", "internal/memsize"},
	},
}

// groupOf returns the index into groups of the band a package belongs to.
func groupOf(name string) int {
	for i := range groups {
		if slices.Contains(groups[i].members, name) {
			return i
		}
	}
	return len(groups) - 1
}

// fillOf, strokeOf and textOf are shorthands for the colors of a package's band.
func fillOf(name string) string   { return groups[groupOf(name)].fill }
func strokeOf(name string) string { return groups[groupOf(name)].stroke }
func textOf(name string) string   { return groups[groupOf(name)].text }

// styleSheet themes the whole document. Presentation attributes lose to a style sheet, so the rules below can retheme
// what dot emitted inline without any rewriting of its output. Everything the dark scheme needs to flip is a class
// here rather than an inline attribute; the node fills stay light in both schemes, reading as chips either way.
const styleSheet = `<style>
  svg { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .bg { fill: #FFFFFF; }
  .title { fill: #15181D; font-size: 27px; font-weight: 700; }
  .subtitle { fill: #5C626B; font-size: 13.5px; }
  .panel { fill: #15181D; font-size: 15px; font-weight: 700; }
  .panelsub { fill: #7A818B; font-size: 12.5px; }
  .rule { stroke: #DCE0E6; stroke-width: 1; }
  .lgl { fill: #3B4048; font-size: 12.5px; }
  .lghead { fill: #15181D; font-size: 11.5px; font-weight: 700; letter-spacing: .07em; }
  .mlab { font-size: 11px; font-weight: 600; }
  .mhead { fill: #7A818B; font-size: 10.5px; font-weight: 700; letter-spacing: .05em; }
  .mtot { fill: #6B7280; font-size: 10.5px; font-weight: 600; }
  .grid { stroke: #D7DBE1; stroke-width: .7; }
  .diag { stroke: #C2C7CE; stroke-width: 1.4; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #12141A; }
    .title, .panel, .lghead { fill: #F1F3F7; }
    .subtitle, .panelsub { fill: #98A0AB; }
    .rule { stroke: #2A2F38; }
    .lgl { fill: #C4CAD3; }
    .mhead, .mtot { fill: #8C949F; }
    .grid { stroke: #2E333C; }
    .diag { stroke: #3C424C; }
    .band { opacity: .22 !important; }
  }
</style>`
