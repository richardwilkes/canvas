// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

package gl

import (
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// ZeroRequiredProcForTest clears an always-required proc (glDrawBuffer) so tests can confirm Validate notices missing
// functions.
func (i *Interface) ZeroRequiredProcForTest() {
	i.Functions.drawBuffer = 0
}

// NewSurfaceContextForTesting builds a bare SurfaceContext over a proxy so live tests can read back proxies that have
// no draw context of their own (e.g. copy destinations).
func NewSurfaceContextForTesting(ctx *DirectContext, proxy *SurfaceProxy, origin gpu.SurfaceOrigin, colorType gpu.ColorType) *SurfaceContext {
	sc := &SurfaceContext{}
	sc.initSurfaceContext(ctx, MakeSurfaceProxyView(proxy, origin, gpu.SwizzleRGBA), colorType)
	return sc
}

// FPConstantOutput exposes constant folding to the external live tests (the CPU-side expectation for FP swatch
// pipelines).
func FPConstantOutput(fp FragmentProcessor, input colorcore.PMColor4f) colorcore.PMColor4f {
	return fpConstantOutputForConstantInput(fp, input)
}

// NewTestCoordsFP exposes the coords-sampling test FP to the external live tests.
func NewTestCoordsFP() FragmentProcessor { return newTestCoordsFP() }

// NumAllocatedForTest returns the number of atlas pages whose backing proxies are instantiated.
func (a *DrawOpAtlas) NumAllocatedForTest() int {
	count := 0
	for i := uint32(0); i < a.maxPages; i++ {
		if a.views[i].Proxy().IsInstantiated() {
			count++
		}
	}
	return count
}

// SetMaxPagesForTest overrides the atlas's maximum page count (for tests).
func (a *DrawOpAtlas) SetMaxPagesForTest(maxPages uint32) {
	if a.numActivePages != 0 {
		panic("SetMaxPages with active pages")
	}
	a.maxPages = maxPages
}

// SetMaxPagesForTest overrides every atlas's maximum page count (for tests).
func (m *AtlasManager) SetMaxPagesForTest(maxPages uint32) {
	for i := range gpu.MaskFormatCount {
		if m.atlases[i] != nil {
			m.atlases[i].SetMaxPagesForTest(maxPages)
		}
	}
}

// SetAtlasDimensionsToMinimumForTest deletes any old atlases and sets all the atlas sizes to one plot each (safe
// outside a flush).
func (m *AtlasManager) SetAtlasDimensionsToMinimumForTest() {
	for i := range gpu.MaskFormatCount {
		m.atlases[i] = nil
	}
	m.atlasConfig = MakeDrawOpAtlasConfig(maxAtlasDim, 0)
}

// AtlasForTest exposes the manager's atlas for a format (initializing it as getViews would).
func (m *AtlasManager) AtlasForTest(format gpu.MaskFormat) *DrawOpAtlas {
	format = m.resolveMaskFormat(format)
	if !m.initAtlas(format) {
		return nil
	}
	return m.getAtlas(format)
}

// PrevFlushTokenForTest exposes the atlas's compaction watermark.
func (a *DrawOpAtlas) PrevFlushTokenForTest() gpu.Token { return a.prevFlushToken }

// AtlasUploadTestOp is a minimal Op that adds subimages to the atlas manager while preparing, and executes under the
// deferred-upload token protocol (ProcessPendingInlineUploads before each recorded draw, IssueFlushToken after).
// Subimage i is filled with byte(i+1). It stands in for AtlasTextOp in the flush-integration tests.
type AtlasUploadTestOp struct {
	dc       *DirectContext
	images   [][]byte
	Locators []gpu.AtlasLocator
	Codes    []AtlasErrorCode
	OpBase
	size     int
	draws    int
	Executed int
	format   gpu.MaskFormat
}

var atlasUploadTestOpClassID = GenOpClassID()

// NewAtlasUploadTestOp builds an AtlasUploadTestOp that adds count size×size subimages.
func NewAtlasUploadTestOp(dc *DirectContext, format gpu.MaskFormat, size, count int) *AtlasUploadTestOp {
	op := &AtlasUploadTestOp{dc: dc, format: format, size: size}
	for i := range count {
		img := make([]byte, size*size*format.BytesPerPixel())
		for j := range img {
			img[j] = byte(i + 1)
		}
		op.images = append(op.images, img)
	}
	op.InitOp(atlasUploadTestOpClassID, op)
	op.SetBounds(geom.IRectWH(8, 8).ToRect(), HasAABloat(false), IsHairline(false))
	return op
}

// Name implements Op.
func (o *AtlasUploadTestOp) Name() string { return "AtlasUploadTest" }

// VisitProxies implements Op.
func (o *AtlasUploadTestOp) VisitProxies(func(*SurfaceProxy, gpu.Mipmapped)) {}

// OnCombineIfPossible implements Op.
func (o *AtlasUploadTestOp) OnCombineIfPossible(Op) CombineResult {
	return CombineResultCannotCombine
}

// OnPrepare implements Op: adds each subimage to the atlas and records one draw per success.
func (o *AtlasUploadTestOp) OnPrepare(state *OpFlushState) {
	am := o.dc.AtlasManager()
	atlas := am.AtlasForTest(o.format)
	for _, img := range o.images {
		var locator gpu.AtlasLocator
		code := am.AddToAtlas(state.ResourceProvider(), state, o.format, o.size, o.size, img,
			&locator)
		o.Codes = append(o.Codes, code)
		if code != AtlasErrorCodeSucceeded {
			continue
		}
		o.Locators = append(o.Locators, locator)
		// One draw per subimage: mark the plot read by this draw, then record the draw.
		atlas.SetLastUseToken(&locator, state.TokenTracker().NextDrawToken())
		state.IssueDrawToken()
		o.draws++
	}
}

// OnExecute implements Op under the deferred-upload token protocol.
func (o *AtlasUploadTestOp) OnExecute(state *OpFlushState, _ geom.Rect) {
	for range o.draws {
		state.ProcessPendingInlineUploads()
		// The real op would bind its pipeline and draw here.
		state.IssueFlushToken()
		o.Executed++
	}
}
