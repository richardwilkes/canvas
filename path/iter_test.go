// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package path

import "testing"

// TestIsClosedContour pins the forward scan performed by IsClosedContour: from the iterator's current verb it looks
// ahead for this contour's close verb, stopping at the moveTo that begins the next contour.
func TestIsClosedContour(t *testing.T) {
	closed := New().MoveTo(0, 0).LineTo(1, 0).LineTo(1, 1).Close()
	open := New().MoveTo(0, 0).LineTo(1, 0).LineTo(1, 1)

	if !NewIter(closed, false).IsClosedContour() {
		t.Error("closed contour not reported as closed")
	}
	if NewIter(open, false).IsClosedContour() {
		t.Error("open contour reported as closed")
	}
	if !NewIter(open, true).IsClosedContour() {
		t.Error("forceClose must report every contour as closed")
	}
	if NewIter(New(), false).IsClosedContour() {
		t.Error("empty path reported as closed")
	}

	// Two contours, the first closed and the second not. The answer must track the iterator's position, and the scan
	// must stop at the second contour's moveTo rather than running to the end of the verb list.
	two := New().MoveTo(0, 0).LineTo(1, 0).Close().MoveTo(5, 5).LineTo(6, 5)
	it := NewIter(two, false)
	if !it.IsClosedContour() {
		t.Error("first contour is closed")
	}
	for { // advance into the second contour
		verb, _, ok := it.Next()
		if !ok {
			t.Fatal("ran off the end of the path without reaching the second contour")
		}
		if verb == VerbClose {
			break
		}
	}
	if it.IsClosedContour() {
		t.Error("second contour is open, but was reported as closed")
	}

	// The mirror case: an open first contour must not be reported closed because a later contour has a close verb.
	twoRev := New().MoveTo(0, 0).LineTo(1, 0).MoveTo(5, 5).LineTo(6, 5).Close()
	if NewIter(twoRev, false).IsClosedContour() {
		t.Error("open first contour picked up the second contour's close verb")
	}

	// Positioned past the end of the verbs, there is no current contour.
	it = NewIter(closed, false)
	for {
		if _, _, ok := it.Next(); !ok {
			break
		}
	}
	if it.IsClosedContour() {
		t.Error("exhausted iterator reported a closed contour")
	}
}
