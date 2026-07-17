// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Per-entry-point GL call counting. Every wrapper in interface.go increments a slot of Functions.callCounts before
// dispatching through the FFI, so the counters see exactly the calls that survive the GPU's redundant-state filtering —
// the number to drive down, since each call also pays purego.SyscallN's marshaling allocations. Counts are maintained
// and read on the GL context thread only, so they need no synchronization.

package gl

// CallCount is one entry of a CallCounts report: a GL entry-point name and how many times it was invoked through the
// FFI since the last ResetCallCounts.
type CallCount struct {
	Name  string
	Count uint64
}

// TotalCallCount returns the total number of GL calls issued through this proc table since the last ResetCallCounts.
// Together with a benchmark's frame count it yields GL calls/frame.
func (f *Functions) TotalCallCount() uint64 {
	var total uint64
	for i := range f.callCounts {
		total += f.callCounts[i]
	}
	return total
}

// CallCounts returns the nonzero per-entry-point call counts since the last ResetCallCounts, sorted by descending count
// (ties broken by name) — the "top offenders" view.
func (f *Functions) CallCounts() []CallCount {
	var out []CallCount
	for i := range f.callCounts {
		if f.callCounts[i] != 0 {
			out = append(out, CallCount{Name: glFuncNames[i], Count: f.callCounts[i]})
		}
	}
	for i := 1; i < len(out); i++ { // Insertion sort: the list is tiny (≤ glFuncCount, usually ~20).
		for j := i; j > 0 && (out[j].Count > out[j-1].Count ||
			(out[j].Count == out[j-1].Count && out[j].Name < out[j-1].Name)); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// ResetCallCounts zeroes all per-entry-point call counters.
func (f *Functions) ResetCallCounts() {
	f.callCounts = [glFuncCount]uint64{}
}
