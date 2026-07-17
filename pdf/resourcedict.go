// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The per-page /Resources dictionary that maps the short names the content stream references (/G7 gs, /X2 Do, …) to the
// indirect objects the device emitted, plus the resource-name helper the content emitters use.

package pdf

import (
	"sort"
	"strconv"

	"github.com/richardwilkes/canvas/stream"
)

// resourceType identifies which resource sub-dictionary an object belongs to (the values are fixed by the prefix/name
// tables below).
type resourceType uint8

const (
	resExtGState resourceType = iota
	resPattern
	resXObject
	resFont
)

// resourceTypePrefixes gives each resource type's single-letter name prefix.
var resourceTypePrefixes = [4]byte{'G', 'P', 'X', 'F'}

// resourceTypeNames gives each resource type's dictionary key.
var resourceTypeNames = [4]string{"ExtGState", "Pattern", "XObject", "Font"}

// resourceName returns the bare "<prefix><key>" name (no leading slash).
func resourceName(typ resourceType, key int) string {
	return string(resourceTypePrefixes[typ]) + strconv.Itoa(key)
}

// writeResourceName writes "/<prefix><key>" to the content stream.
func writeResourceName(dst stream.WStream, typ resourceType, key int) {
	writeText(dst, "/")
	dst.Write([]byte{resourceTypePrefixes[typ]})
	writeText(dst, strconv.Itoa(key))
}

// makeProcSet builds the standard /ProcSet array every page resource dictionary declares.
func makeProcSet() *Array {
	a := NewArray()
	for _, proc := range []string{"PDF", "Text", "ImageB", "ImageC", "ImageI"} {
		a.AppendName(proc)
	}
	return a
}

// addSubdict inserts a resourceList's entries into dst under the type's dictionary key, keyed by their resource names;
// a nil list adds nothing.
func addSubdict(resourceList []IndirectReference, typ resourceType, dst *Dict) {
	if len(resourceList) == 0 {
		return
	}
	resources := NewDict()
	for _, ref := range resourceList {
		resources.InsertRef(resourceName(typ, int(ref.value)), ref)
	}
	dst.InsertObject(resourceTypeNames[typ], resources)
}

// makeResourceDict builds the page's /Resources dictionary from its four resource lists. Each list must be sorted by
// object number (the device collects references into sets and sorts them before calling this).
func makeResourceDict(graphicStateResources, shaderResources, xObjectResources, fontResources []IndirectReference) *Dict {
	dict := NewDict()
	dict.InsertObject("ProcSet", makeProcSet())
	addSubdict(graphicStateResources, resExtGState, dict)
	addSubdict(shaderResources, resPattern, dict)
	addSubdict(xObjectResources, resXObject, dict)
	addSubdict(fontResources, resFont, dict)
	return dict
}

// sortedRefs returns the references from set sorted ascending by object number.
func sortedRefs(set map[IndirectReference]struct{}) []IndirectReference {
	out := make([]IndirectReference, 0, len(set))
	for ref := range set {
		out = append(out, ref)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].value < out[j].value })
	return out
}
