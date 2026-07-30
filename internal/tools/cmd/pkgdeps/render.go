// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Geometry of the document, in user units. The matrix is sized from the package count, while the node-link panel is
// sized by whatever dot returns for it.
const (
	cell          = 21.0  // one matrix cell, square
	labelWidth    = 152.0 // room for the longest package name down the left edge of the matrix
	headerHeight  = 118.0 // room for the rotated package names across the top of the matrix
	totalsExtent  = 46.0  // the marginal-total row and column
	labelOverhang = 86.0  // how far the rotated names run past the right edge of the matrix
	pageMargin    = 26.0
	titleHeight   = 96.0  // title, subtitle, rule and the first panel heading
	legendHeight  = 108.0 // the two legend blocks between the panels
	panelGap      = 26.0
	captionHeight = 40.0  // the matrix panel heading
	minPageWidth  = 980.0 // keeps the header text from crowding the right edge on a narrow graph
)

var (
	viewBoxPattern    = regexp.MustCompile(`viewBox="([\d.\- ]+)"`)
	backgroundPattern = regexp.MustCompile(`<polygon fill="(?:white|transparent|none)"[^/]*/>`)
)

// svgWriter accumulates the document. Nothing here needs escaping beyond what the callers already do: package names
// come from import paths, which cannot contain XML metacharacters, and the legend titles are literals.
type svgWriter struct {
	strings.Builder
}

func (w *svgWriter) writef(format string, args ...any) {
	fmt.Fprintf(&w.Builder, format, args...)
}

// render produces the complete document: a layered node-link panel of the reduced edges over a matrix of every
// direct import.
func render(g *graph) (string, error) {
	panel, graphWidth, graphHeight, err := layoutGraphPanel(g)
	if err != nil {
		return "", err
	}
	n := float64(len(g.names))
	matrixWidth := labelWidth + n*cell + totalsExtent + labelOverhang
	matrixHeight := headerHeight + n*cell + totalsExtent
	width := max(graphWidth, matrixWidth, minPageWidth) + 2*pageMargin
	height := titleHeight + graphHeight + panelGap + legendHeight + panelGap + captionHeight + matrixHeight +
		pageMargin

	var w svgWriter
	w.writef(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" `+
		"width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n", width, height, width, height)
	w.writef("%s\n", styleSheet)
	w.writef("<rect class=\"bg\" x=\"0\" y=\"0\" width=\"%.0f\" height=\"%.0f\"/>\n", width, height)
	w.writef("<text class=\"title\" x=\"%.0f\" y=\"%.0f\">canvas &#8212; package dependency graph</text>\n",
		pageMargin, pageMargin+14)
	w.writef("<text class=\"subtitle\" x=\"%.0f\" y=\"%.0f\">%d packages &#183; %d direct intra-module imports "+
		"&#183; production code only, excluding test files and test-support packages</text>\n",
		pageMargin, pageMargin+38, len(g.names), g.edgeCount)
	writeRule(&w, width, pageMargin+54)
	w.writef("<text class=\"panel\" x=\"%.0f\" y=\"%.0f\">Layered structure</text>\n", pageMargin, titleHeight-4)
	w.writef("<text class=\"panelsub\" x=\"%.0f\" y=\"%.0f\">%d edges shown &#183; arrows point from importer to "+
		"imported; an import already implied by a longer path appears only in the matrix below</text>\n",
		pageMargin+152, titleHeight-4, len(g.essential))
	w.writef("<g transform=\"translate(%.1f,%.1f)\">\n%s\n</g>\n", (width-graphWidth)/2, titleHeight+8, panel)

	writeLegend(&w, width, titleHeight+graphHeight+panelGap+14)

	matrixTop := titleHeight + graphHeight + panelGap + legendHeight + panelGap
	writeRule(&w, width, matrixTop-14)
	w.writef("<text class=\"panel\" x=\"%.0f\" y=\"%.1f\">Complete import matrix</text>\n", pageMargin, matrixTop+12)
	w.writef("<text class=\"panelsub\" x=\"%.0f\" y=\"%.1f\">row imports column &#183; both axes ordered by "+
		"dependency depth, so every mark falls below the diagonal &#8212; the graph is acyclic</text>\n",
		pageMargin+196, matrixTop+12)
	writeMatrix(&w, g, pageMargin, matrixTop+captionHeight)
	w.writef("</svg>\n")
	return w.String(), nil
}

// layoutGraphPanel hands the reduced graph to dot and returns the laid-out group together with its size. Only the
// content of dot's own group is kept; the surrounding document, including its background, is replaced by ours.
func layoutGraphPanel(g *graph) (panel string, width, height float64, err error) {
	var src svgWriter
	src.writef("digraph deps {\n")
	src.writef("  bgcolor=\"transparent\";\n")
	src.writef("  rankdir=TB; splines=spline; nodesep=0.34; ranksep=0.62; pad=0.02;\n")
	src.writef("  node [shape=box, style=\"filled,rounded\", fontname=\"Helvetica-Bold\", fontsize=13," +
		" height=0.40, margin=\"0.17,0.06\", penwidth=1.7];\n")
	src.writef("  edge [arrowsize=0.65, penwidth=1.6];\n")
	byName := slices.Sorted(slices.Values(g.names)) // dot's layout depends on input order, so pin it to something
	for _, name := range byName {                   // stable rather than to the depth ordering used elsewhere
		src.writef("  %q [label=%q, fillcolor=%q, color=%q, fontcolor=%q];\n",
			name, name, fillOf(name), strokeOf(name), textOf(name))
	}
	for _, e := range g.essentialEdges() {
		src.writef("  %q -> %q [color=\"%sD0\"];\n", e.from, e.to, strokeOf(e.from))
	}
	src.writef("}\n")

	cmd := exec.Command("dot", "-Tsvg")
	cmd.Stdin = strings.NewReader(src.String())
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, runErr := cmd.Output()
	if runErr != nil {
		if errors.Is(runErr, exec.ErrNotFound) {
			return "", 0, 0, errors.New(`the Graphviz "dot" binary was not found on the PATH; ` +
				`install it with "brew install graphviz"`)
		}
		return "", 0, 0, fmt.Errorf("dot: %w: %s", runErr, stderr.String())
	}

	svg := string(out)
	box := viewBoxPattern.FindStringSubmatch(svg)
	if box == nil {
		return "", 0, 0, errors.New("dot produced no viewBox")
	}
	fields := strings.Fields(box[1])
	if len(fields) != 4 {
		return "", 0, 0, fmt.Errorf("dot produced an unusable viewBox %q", box[1])
	}
	if width, err = strconv.ParseFloat(fields[2], 64); err != nil {
		return "", 0, 0, fmt.Errorf("dot viewBox width %q: %w", fields[2], err)
	}
	if height, err = strconv.ParseFloat(fields[3], 64); err != nil {
		return "", 0, 0, fmt.Errorf("dot viewBox height %q: %w", fields[3], err)
	}
	start := strings.Index(svg, `<g id="graph0"`)
	end := strings.LastIndex(svg, "</g>")
	if start < 0 || end < start {
		return "", 0, 0, errors.New("dot produced no graph group")
	}
	panel = svg[start : end+len("</g>")]
	if bg := backgroundPattern.FindStringIndex(panel); bg != nil { // drop dot's own background, keeping ours
		panel = panel[:bg[0]] + panel[bg[1]:]
	}
	return panel, width, height, nil
}

// writeMatrix draws the importer-by-imported matrix with its origin at (ox, oy). Rows are importers and columns are
// imports, both in the graph's depth order, so a mark above the diagonal would mean an import cycle.
func writeMatrix(w *svgWriter, g *graph, ox, oy float64) {
	n := float64(len(g.names))
	gridWidth := n * cell
	w.writef("<g transform=\"translate(%.1f,%.1f)\">\n", ox, oy)
	for i, name := range g.names { // row bands, tinted by the row package's group
		opacity := "0.16"
		if i%2 == 0 {
			opacity = "0.3"
		}
		w.writef("<rect class=\"band\" x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" fill=\"%s\" "+
			"opacity=\"%s\"/>\n", labelWidth, headerHeight+float64(i)*cell, gridWidth, cell, fillOf(name), opacity)
	}
	for i := range len(g.names) + 1 {
		offset := float64(i) * cell
		w.writef("<line class=\"grid\" x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\"/>\n",
			labelWidth+offset, headerHeight, labelWidth+offset, headerHeight+gridWidth)
		w.writef("<line class=\"grid\" x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\"/>\n",
			labelWidth, headerHeight+offset, labelWidth+gridWidth, headerHeight+offset)
	}
	for i, name := range g.names {
		stroke := strokeOf(name)
		w.writef("<text class=\"mlab\" x=\"%.1f\" y=\"%.1f\" text-anchor=\"end\" fill=\"%s\">%s</text>\n",
			labelWidth-8, headerHeight+float64(i)*cell+cell/2+4, stroke, name)
		w.writef("<text class=\"mlab\" transform=\"translate(%.1f,%.1f) rotate(-60)\" text-anchor=\"start\" "+
			"fill=\"%s\">%s</text>\n", labelWidth+float64(i)*cell+cell/2+4, headerHeight-7, stroke, name)
	}
	for row, importer := range g.names {
		for column, imported := range g.names {
			x := labelWidth + float64(column)*cell
			y := headerHeight + float64(row)*cell
			writeCell(w, g, importer, imported, x, y)
		}
	}
	w.writef("<text class=\"mhead\" transform=\"translate(%.1f,%.1f) rotate(-60)\" text-anchor=\"start\">"+
		"imports</text>\n", labelWidth+gridWidth+totalsExtent/2, headerHeight-7)
	w.writef("<text class=\"mhead\" x=\"%.1f\" y=\"%.1f\" text-anchor=\"end\">imported by</text>\n",
		labelWidth-8, headerHeight+gridWidth+26)
	for i, name := range g.names {
		w.writef("<text class=\"mtot\" x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\">%d</text>\n",
			labelWidth+gridWidth+totalsExtent/2, headerHeight+float64(i)*cell+cell/2+4, len(g.prod[name]))
		w.writef("<text class=\"mtot\" x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\">%d</text>\n",
			labelWidth+float64(i)*cell+cell/2, headerHeight+gridWidth+16, g.importedBy(name))
	}
	w.writef("</g>\n")
}

// writeCell draws the mark for one importer/imported pair: a solid square for an edge the node-link panel also shows,
// a hollow one for a direct import that some longer path already implies, and a tick on the diagonal, where a package
// meets itself.
func writeCell(w *svgWriter, g *graph, importer, imported string, x, y float64) {
	switch {
	case importer == imported:
		w.writef("<line class=\"diag\" x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\"/>\n",
			x+6, y+cell-6, x+cell-6, y+6)
	case contains(g.prod[importer], imported):
		color := strokeOf(imported)
		if _, ok := g.essential[edge{from: importer, to: imported}]; ok {
			w.writef("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" rx=\"3\" fill=\"%s\"/>\n",
				x+3.5, y+3.5, cell-7, cell-7, color)
		} else {
			w.writef("<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" rx=\"2.5\" fill=\"%s\" "+
				"fill-opacity=\"0.26\" stroke=\"%s\" stroke-opacity=\"0.55\" stroke-width=\"1\"/>\n",
				x+4.5, y+4.5, cell-9, cell-9, color, color)
		}
	}
}

// writeLegend draws the package-group swatches and the matrix-cell key that sit between the two panels.
func writeLegend(w *svgWriter, width, top float64) {
	writeRule(w, width, top-20)
	w.writef("<text class=\"lghead\" x=\"%.0f\" y=\"%.1f\">PACKAGE GROUPS</text>\n", pageMargin, top)
	columnWidth := (width - 2*pageMargin) / 4
	for i := range groups {
		x := pageMargin + float64(i%4)*columnWidth
		y := top + 22 + float64(i/4)*22
		w.writef("<rect x=\"%.1f\" y=\"%.1f\" width=\"14\" height=\"14\" rx=\"4\" fill=\"%s\" stroke=\"%s\" "+
			"stroke-width=\"1.7\"/>\n", x, y-10, groups[i].fill, groups[i].stroke)
		w.writef("<text class=\"lgl\" x=\"%.1f\" y=\"%.1f\">%s</text>\n", x+21, y+2, groups[i].title)
	}
	keyTop := top + 22 + 2*22 + 12
	w.writef("<text class=\"lghead\" x=\"%.0f\" y=\"%.1f\">MATRIX CELLS</text>\n", pageMargin, keyTop)
	keys := []struct {
		label string
		mark  string
	}{
		{
			label: "Direct import",
			mark:  `<rect x="0" y="-7" width="14" height="14" rx="3" fill="#454A96"/>`,
		},
		{
			label: "Direct import that is also reachable transitively",
			mark: `<rect x="0.5" y="-6.5" width="13" height="13" rx="2.5" fill="#454A96" fill-opacity="0.26" ` +
				`stroke="#454A96" stroke-opacity="0.55" stroke-width="1"/>`,
		},
	}
	y := keyTop + 21
	for i, key := range keys {
		x := pageMargin + float64(i)*columnWidth*1.16
		w.writef("<g transform=\"translate(%.1f,%.1f)\">%s</g>\n", x, y, key.mark)
		w.writef("<text class=\"lgl\" x=\"%.1f\" y=\"%.1f\">%s</text>\n", x+21, y+4, key.label)
	}
}

// writeRule draws one of the horizontal separators that divide the document.
func writeRule(w *svgWriter, width, y float64) {
	w.writef("<line class=\"rule\" x1=\"%.0f\" y1=\"%.1f\" x2=\"%.0f\" y2=\"%.1f\"/>\n",
		pageMargin, y, width-pageMargin, y)
}

// contains reports whether the set holds name.
func contains(set map[string]struct{}, name string) bool {
	_, ok := set[name]
	return ok
}
