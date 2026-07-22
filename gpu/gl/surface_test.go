// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Regression test for EnsureDynamicMSAAAttachment dropping the just-created multisample FBO when the discardable MSAA
// attachment fails to materialize. Leaving multisampleFBOID nonzero would let a later retry short-circuit on the
// "multisampleFBOID != 0 -> return true" guard and report success on a framebuffer that never got its color attachment.

package gl

import (
	"testing"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

func TestEnsureDynamicMSAAAttachmentDropsFBOOnAttachmentFailure(t *testing.T) {
	dc := newShaderRecordingContext(t)
	caps := dc.GLCaps()
	format := caps.GetDefaultBackendFormat(gpu.ColorTypeRGBA8888, true)
	if caps.InternalMultisampleCount(format) <= 1 {
		t.Skip("profile lacks internal MSAA for RGBA8888; the FBO-creation lane is never reached")
	}

	proxy := dc.ProxyProvider().CreateProxy(format, geom.ISize{Width: 16, Height: 16},
		gpu.RenderableYes, 1, gpu.MipmappedNo, gpu.BackingFitExact, gpu.BudgetedYes,
		"msaa-fail-test", 0, UseAllocatorYes)
	if proxy == nil {
		t.Fatal("CreateProxy failed")
	}
	defer proxy.Unref()
	if !proxy.Instantiate(dc.ResourceProvider()) {
		t.Fatal("Instantiate failed")
	}
	rt := proxy.PeekRenderTarget()
	if rt == nil {
		t.Fatal("instantiated renderable proxy has no render target")
	}
	if rt.HasDynamicMSAAAttachment() || rt.MultisampleFBOID() != 0 {
		t.Fatal("render target unexpectedly already carries a DMSAA attachment")
	}

	// A zero-value ResourceProvider is abandoned (cache == nil), so GetDiscardableMSAAAttachment returns nil after the
	// multisample FBO has already been generated — exactly the failure the fix must clean up after.
	abandoned := &ResourceProvider{}
	if !abandoned.IsAbandoned() {
		t.Fatal("expected a zero-value ResourceProvider to report abandoned")
	}

	recCounts = map[string]int{}
	if rt.EnsureDynamicMSAAAttachment(abandoned) {
		t.Fatal("EnsureDynamicMSAAAttachment reported success despite a nil color attachment")
	}
	if counts("glGenFramebuffers") < 1 {
		t.Fatal("the FBO-creation lane was never reached, so the fix is untested")
	}
	if rt.MultisampleFBOID() != 0 {
		t.Errorf("multisampleFBOID = %d after a failed attachment; want 0 so a retry re-runs the setup",
			rt.MultisampleFBOID())
	}
	if rt.HasDynamicMSAAAttachment() {
		t.Error("render target retained a DMSAA attachment after the attachment failed")
	}
	// A retry must not short-circuit to success on the incomplete framebuffer.
	if rt.EnsureDynamicMSAAAttachment(abandoned) {
		t.Error("retry reported success on the incomplete framebuffer left by the first failure")
	}
}
