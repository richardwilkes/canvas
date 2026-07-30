// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Command pkgdeps regenerates internal/tools/package-deps.svg, a visual map of the canvas module's intra-module
// package dependencies. The package data comes from "go list", so the diagram always reflects the current source;
// re-run this after adding, removing, or re-pointing an import.
//
// Usage, from anywhere inside the repository:
//
//	go -C internal/tools run ./cmd/pkgdeps
//
// The Graphviz "dot" binary must be on the PATH ("brew install graphviz"); it lays out the node-link panel.
//
// Only production code is charted: test files are not consulted, and packages that exist solely to support tests are
// left out, so the diagram shows how the shipped library fits together.
//
// The output has two panels. The first is a layered node-link graph containing only the transitively-reduced edges:
// an import is drawn only when it is not already implied by a longer path through the graph, which keeps the shape of
// the stack readable. The second is a complete importer-by-imported matrix that retains every direct import,
// including the ones elided from the graph, with both axes ordered by dependency depth so that a mark above the
// diagonal would indicate an import cycle.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// defaultOutput is where the diagram lands when -o is not given, relative to the analyzed module's root.
var defaultOutput = filepath.Join("internal", "tools", "package-deps.svg")

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "pkgdeps:", err)
		os.Exit(1)
	}
}

func run() error {
	modDir := flag.String("mod", "", "Directory of the module to analyze (default: the git top level)")
	out := flag.String("o", "", "Path of the SVG to write (default: <mod>/"+defaultOutput+")")
	flag.Parse()
	if flag.NArg() != 0 {
		flag.Usage()
		return fmt.Errorf("unexpected argument %q", flag.Arg(0))
	}
	dir, err := resolveModuleDir(*modDir)
	if err != nil {
		return err
	}
	dest := *out
	if dest == "" {
		dest = filepath.Join(dir, defaultOutput)
	}
	graph, err := loadGraph(dir)
	if err != nil {
		return err
	}
	svg, err := render(graph)
	if err != nil {
		return err
	}
	if err = os.WriteFile(dest, []byte(svg), 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", dest, err)
	}
	fmt.Printf("wrote %s: %d packages, %d imports (%d essential, %d transitively implied)\n",
		dest, len(graph.names), graph.edgeCount, len(graph.essential), graph.edgeCount-len(graph.essential))
	return nil
}

// resolveModuleDir returns the directory of the module to analyze. An explicit -mod wins; otherwise the repository's
// top level is used, so that the tool produces the same diagram no matter which directory it is invoked from (in
// particular from internal/tools, which is its own module and would otherwise be the module "go list" reports).
func resolveModuleDir(explicit string) (string, error) {
	if explicit != "" {
		abs, err := filepath.Abs(explicit)
		if err != nil {
			return "", fmt.Errorf("resolving -mod %q: %w", explicit, err)
		}
		return abs, nil
	}
	top, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", fmt.Errorf("locating the repository root: %w; pass -mod to name the module directory", err)
	}
	return strings.TrimSpace(string(top)), nil
}
