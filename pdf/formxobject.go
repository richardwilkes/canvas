// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// A Form XObject wraps a content stream. Two lanes use it: the gradient alpha (luminosity SMask) path, and saveLayer
// transparency groups via Device.DrawDevice → makeFormXObjectFromDevice → makeFormXObjectFromDeviceBounds.

package pdf

import "github.com/richardwilkes/canvas/geom"

// makeFormXObject emits content as a Form XObject with the given media box, resource dict, optional placement matrix,
// and (optional) transparency-group color space. Passing an empty colorSpace omits the /CS entry.
func makeFormXObject(doc *Document, content []byte, mediaBox *Array, resourceDict *Dict, inverseTransform *geom.Matrix, colorSpace string) IndirectReference {
	dict := NewDict()
	dict.InsertName("Type", "XObject")
	dict.InsertName("Subtype", "Form")
	if !inverseTransform.IsIdentity() {
		dict.InsertObject("Matrix", matrixToArray(inverseTransform))
	}
	dict.InsertObject("Resources", resourceDict)
	dict.InsertObject("BBox", mediaBox)

	// Form XObjects are used for saveLayer and alpha masks, both of which want isolated blending.
	group := NewTypedDict("Group")
	group.InsertName("S", "Transparency")
	if colorSpace != "" {
		group.InsertName("CS", colorSpace)
	}
	group.InsertBool("I", true) // Isolated.
	dict.InsertObject("Group", group)
	return doc.StreamOut(dict, content, true)
}
