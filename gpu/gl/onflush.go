// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The pre/post-flush callback hooks the drawing manager runs around each flush. The atlas manager is the only
// registered callback here (the small-path atlas manager is trimmed).

package gl

import "github.com/richardwilkes/canvas/gpu"

// OnFlushCallbackObject is the interface all pre-flush callback objects implement.
type OnFlushCallbackObject interface {
	// PreFlush allows subsystems (e.g. text atlases) to create resources for a specific flush. Returns true on success,
	// false on memory allocation failure.
	PreFlush() bool
	// PostFlush is called once flushing is complete; startTokenForNextFlush can be used to track resources used in the
	// current flush.
	PostFlush(startTokenForNextFlush gpu.Token)
	// RetainOnFreeGpuResources tells the callback owner to hold onto this object when freeing GPU resources.
	RetainOnFreeGpuResources() bool
}
