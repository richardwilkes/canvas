// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A source-level audit of the Linux leg's GC-stack contract. It parses the source rather than running it, so the
// contract is pinned on every platform's CI leg, not just the one that can build a GLX context.

package gltest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// rootIdent returns the identifier an address-of expression ultimately refers to: &x, &x[0], &p.f and x all yield the
// root name ("x", "x", "p", "x"), which is the granularity the KeepAlive audit needs.
func rootIdent(expr ast.Expr) string {
	for {
		switch e := expr.(type) {
		case *ast.Ident:
			return e.Name
		case *ast.ParenExpr:
			expr = e.X
		case *ast.UnaryExpr:
			expr = e.X
		case *ast.IndexExpr:
			expr = e.X
		case *ast.SelectorExpr:
			expr = e.X
		case *ast.StarExpr:
			expr = e.X
		default:
			return ""
		}
	}
}

// unsafePointerConversion reports the root identifier of a uintptr(unsafe.Pointer(x)) argument, or "" if the expression
// is not one.
func unsafePointerConversion(expr ast.Expr) string {
	outer, ok := expr.(*ast.CallExpr)
	if !ok || len(outer.Args) != 1 {
		return ""
	}
	if id, isIdent := outer.Fun.(*ast.Ident); !isIdent || id.Name != "uintptr" {
		return ""
	}
	inner, ok := outer.Args[0].(*ast.CallExpr)
	if !ok || len(inner.Args) != 1 {
		return ""
	}
	sel, ok := inner.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Pointer" {
		return ""
	}
	if pkg, isIdent := sel.X.(*ast.Ident); !isIdent || pkg.Name != "unsafe" {
		return ""
	}
	return rootIdent(inner.Args[0])
}

// isCallTo reports whether call names the function pkg.name (pkg empty for a plain function).
func isCallTo(call *ast.CallExpr, pkg, name string) bool {
	switch fn := call.Fun.(type) {
	case *ast.Ident:
		return pkg == "" && fn.Name == name
	case *ast.SelectorExpr:
		id, ok := fn.X.(*ast.Ident)
		return ok && id.Name == pkg && fn.Sel.Name == name
	}
	return false
}

// TestGLXRawCallPointerContract audits context_linux.go's handling of Go pointers passed as uintptr through
// glxRawCall. purego.SyscallN carries //go:uintptrescapes itself, but that only covers conversions written at its own
// call site; glxRawCall forwards an already-built []uintptr with no pointer provenance left, so the tag has to be
// repeated on the forwarder or a stack growth could move the pointed-to object before the C call reads it. Every
// converted pointer also needs a runtime.KeepAlive so it stays reachable across the call.
func TestGLXRawCallPointerContract(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "context_linux.go", nil, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	var forwarder *ast.FuncDecl
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "glxRawCall" {
			forwarder = fn
		}
	}
	if forwarder == nil {
		t.Fatal("glxRawCall not found in context_linux.go")
	}
	tagged := false
	if forwarder.Doc != nil {
		for _, c := range forwarder.Doc.List {
			if c.Text == "//go:uintptrescapes" {
				tagged = true
			}
		}
	}
	if !tagged {
		t.Error("glxRawCall must carry //go:uintptrescapes: it forwards to purego.SyscallN, so the tag on " +
			"SyscallN itself does not cover pointers converted at glxRawCall call sites")
	}

	// Every object whose address is converted for a glxRawCall argument must be kept alive across that call.
	keptAlive := make(map[string]bool)
	converted := make(map[string]bool)
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch {
		case isCallTo(call, "runtime", "KeepAlive") && len(call.Args) == 1:
			if root := rootIdent(call.Args[0]); root != "" {
				keptAlive[root] = true
			}
		case isCallTo(call, "", "glxRawCall"):
			for _, arg := range call.Args {
				if root := unsafePointerConversion(arg); root != "" {
					converted[root] = true
				}
			}
		}
		return true
	})
	if len(converted) == 0 {
		t.Fatal("no uintptr(unsafe.Pointer(...)) arguments found — the source parse is broken")
	}
	for name := range converted {
		if !keptAlive[name] {
			t.Errorf("%s has its address passed through glxRawCall with no matching runtime.KeepAlive",
				name)
		}
	}
	t.Logf("audited %d pointer arguments passed through glxRawCall", len(converted))
}
