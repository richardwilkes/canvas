// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gltest

import "testing"

// purego's callback table holds at most 2000 entries and never releases them, so asking for far more than that many
// handlers proves the callback is created once and shared: a per-call purego.NewCallback would hand back a different
// address each time and panic with "purego: the maximum number of callbacks has been reached" before the loop ends.
func TestXErrorHandlerCallbackIsShared(t *testing.T) {
	first := xErrorHandlerCallback()
	if first == 0 {
		t.Fatal("xErrorHandlerCallback returned a null callback")
	}
	for i := range 4000 {
		if got := xErrorHandlerCallback(); got != first {
			t.Fatalf("call %d returned a different callback: got %#x, want %#x", i, got, first)
		}
	}
}
