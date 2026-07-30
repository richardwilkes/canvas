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
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"slices"
	"strings"
)

// edge is one import, from the importer to the imported package. Both are module-relative package names.
type edge struct {
	from string
	to   string
}

// graph holds the intra-module import relationships of a single module, as they exist in production code: test files
// are not consulted at all, and packages that exist only to support tests are left out. Package names throughout are
// relative to the module path, so "gpu/gl" rather than the full import path.
type graph struct {
	prod      map[string]map[string]struct{} // the imports of each package
	depth     map[string]int                 // longest path to a package with no intra-module imports
	essential map[edge]struct{}              // edges that no longer path already implies
	names     []string                       // every package, in dependency-depth then group then name order
	edgeCount int
}

// listed is the subset of "go list -json" output that matters here.
type listed struct {
	Module struct {
		Path string
	}
	ImportPath string
	Imports    []string
}

// loadGraph runs "go list" in modDir and reduces its output to the intra-module import graph.
func loadGraph(modDir string) (*graph, error) {
	cmd := exec.Command("go", "list", "-json", "./...")
	cmd.Dir = modDir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("go list in %s: %w: %s", modDir, err, stderr.String())
	}
	g := &graph{
		prod:  make(map[string]map[string]struct{}),
		depth: make(map[string]int),
	}
	var modPath string
	dec := json.NewDecoder(bytes.NewReader(out))
	for {
		var lp listed
		if decErr := dec.Decode(&lp); decErr != nil {
			if errors.Is(decErr, io.EOF) {
				break
			}
			return nil, fmt.Errorf("parsing go list output: %w", decErr)
		}
		if modPath == "" {
			modPath = lp.Module.Path
		}
		name := relative(modPath, lp.ImportPath)
		if testSupport(name) {
			continue
		}
		g.names = append(g.names, name)
		g.prod[name] = internalImports(modPath, name, lp.Imports)
	}
	if len(g.names) == 0 {
		return nil, fmt.Errorf("go list reported no packages in %s", modDir)
	}
	g.discardUnlisted()
	g.computeDepths()
	g.reduce()
	slices.SortFunc(g.names, func(a, b string) int {
		return cmp.Or(cmp.Compare(g.depth[a], g.depth[b]), cmp.Compare(groupOf(a), groupOf(b)), cmp.Compare(a, b))
	})
	return g, nil
}

// testSupport reports whether a package exists only to serve tests and so has no place in a diagram of how the
// library fits together. The convention in this module is a "test" suffix on the final path element, as in
// gpu/gl/gltest; extend this if another convention shows up.
func testSupport(name string) bool {
	return strings.HasSuffix(name[strings.LastIndexByte(name, '/')+1:], "test")
}

// relative strips the module path from an import path, leaving the module-relative package name. Paths outside the
// module are returned unchanged, though loadGraph only ever passes it packages from "go list ./...".
func relative(modPath, importPath string) string {
	if prefix := modPath + "/"; strings.HasPrefix(importPath, prefix) {
		return importPath[len(prefix):]
	}
	return importPath
}

// internalImports collects the imports belonging to the module, as module-relative names, dropping self-imports.
func internalImports(modPath, self string, list []string) map[string]struct{} {
	result := make(map[string]struct{})
	for _, importPath := range list {
		if name := relative(modPath, importPath); name != importPath && name != self {
			result[name] = struct{}{}
		}
	}
	return result
}

// discardUnlisted drops any edge whose target is not itself a node, both so that the two axes of the matrix are
// always the same set of packages and so that an edge into a test-support package cannot survive its node.
func (g *graph) discardUnlisted() {
	known := make(map[string]struct{}, len(g.names))
	for _, name := range g.names {
		known[name] = struct{}{}
	}
	for _, imports := range g.prod {
		for imported := range imports {
			if _, ok := known[imported]; !ok {
				delete(imports, imported)
			}
		}
	}
	for _, name := range g.names {
		g.edgeCount += len(g.prod[name])
	}
}

// computeDepths assigns each package the length of the longest import path below it, which is what orders both the
// layers of the node-link panel and the axes of the matrix.
func (g *graph) computeDepths() {
	var depthOf func(string) int
	depthOf = func(name string) int {
		if d, ok := g.depth[name]; ok {
			return d
		}
		g.depth[name] = 0 // the import graph is a DAG, but never recurse forever if that ever stops being true
		deepest := -1
		for imported := range g.prod[name] {
			deepest = max(deepest, depthOf(imported))
		}
		g.depth[name] = deepest + 1
		return g.depth[name]
	}
	for _, name := range g.names {
		depthOf(name)
	}
}

// reduce computes the transitive reduction: the edges that survive are the ones no longer path already implies, and
// they are the only ones the node-link panel draws.
func (g *graph) reduce() {
	reachable := make(map[string]map[string]struct{}, len(g.names))
	var reachOf func(string) map[string]struct{}
	reachOf = func(name string) map[string]struct{} {
		if r, ok := reachable[name]; ok {
			return r
		}
		reachable[name] = make(map[string]struct{}) // cycle guard, as in computeDepths
		r := make(map[string]struct{})
		for imported := range g.prod[name] {
			r[imported] = struct{}{}
			for indirect := range reachOf(imported) {
				r[indirect] = struct{}{}
			}
		}
		reachable[name] = r
		return r
	}
	g.essential = make(map[edge]struct{}, g.edgeCount)
	for _, name := range g.names {
		for imported := range g.prod[name] {
			implied := false
			for other := range g.prod[name] {
				if other != imported {
					if _, ok := reachOf(other)[imported]; ok {
						implied = true
						break
					}
				}
			}
			if !implied {
				g.essential[edge{from: name, to: imported}] = struct{}{}
			}
		}
	}
}

// essentialEdges returns the reduced edges in a stable order.
func (g *graph) essentialEdges() []edge {
	edges := make([]edge, 0, len(g.essential))
	for e := range g.essential {
		edges = append(edges, e)
	}
	slices.SortFunc(edges, func(a, b edge) int {
		return cmp.Or(cmp.Compare(a.from, b.from), cmp.Compare(a.to, b.to))
	})
	return edges
}

// importedBy counts the packages that import name.
func (g *graph) importedBy(name string) int {
	count := 0
	for _, other := range g.names {
		if _, ok := g.prod[other][name]; ok {
			count++
		}
	}
	return count
}
