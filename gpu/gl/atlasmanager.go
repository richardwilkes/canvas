// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Manages the lifetime of and access to the per-mask-format DrawOpAtlases. It is only available at flush time via the
// OpFlushState, which implies that adding glyphs to the atlas is a flush-time operation. The manager doubles as the
// atlas generation counter and registers itself as the drawing manager's on-flush callback: preFlush validates page
// instantiation, postFlush compacts. getPackedGlyphImage covers the lanes this codebase's scaler can produce: A8 masks
// (tight rows), ARGB32 color glyphs, LCD16 masks into the A565 atlas (E.4) with the 565→8888 expansion for drivers
// without a 565 format; the BW bit-expansion lane is unreachable.

package gl

import (
	"unsafe"

	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/gpu"
	"github.com/richardwilkes/canvas/gpu/text"
)

// AtlasManager manages the lifetime of and access to the per-mask-format glyph atlases.
type AtlasManager struct {
	atlases       [gpu.MaskFormatCount]*DrawOpAtlas
	proxyProvider *ProxyProvider
	caps          *Caps
	atlasConfig   DrawOpAtlasConfig
	gpu.AtlasGenerationCounter
	allowMultitexturing AllowMultitexturing
	supportBilerpAtlas  bool
}

// NewAtlasManager creates an AtlasManager for the given proxy provider and texture-size budget.
func NewAtlasManager(proxyProvider *ProxyProvider, maxTextureBytes uint64, allowMultitexturing AllowMultitexturing, supportBilerpAtlas bool) *AtlasManager {
	caps := proxyProvider.Caps()
	return &AtlasManager{
		allowMultitexturing: allowMultitexturing,
		supportBilerpAtlas:  supportBilerpAtlas,
		proxyProvider:       proxyProvider,
		caps:                caps,
		atlasConfig:         MakeDrawOpAtlasConfig(caps.MaxTextureSize, maxTextureBytes),
	}
}

// GetViews returns the atlas texture views for format; if it returns nil views, the client must not try to use other
// functions on the atlas. This function *must* be called first, before other functions that use the atlas. Note that
// proxies can be available with none active (i.e. none instantiated).
func (m *AtlasManager) GetViews(format gpu.MaskFormat) (views *[gpu.MaxMultitexturePages]SurfaceProxyView, numActiveProxies uint32) {
	format = m.resolveMaskFormat(format)
	if m.initAtlas(format) {
		atlas := m.getAtlas(format)
		return atlas.Views(), atlas.NumActivePages()
	}
	return nil, 0
}

// FreeAll releases every atlas.
func (m *AtlasManager) FreeAll() {
	for i := range gpu.MaskFormatCount {
		m.atlases[i] = nil
	}
}

// HasGlyph reports whether glyph's atlas location is still present in its format's atlas.
func (m *AtlasManager) HasGlyph(format gpu.MaskFormat, glyph *text.Glyph) bool {
	if glyph == nil {
		panic("nil glyph")
	}
	return m.getAtlas(format).HasID(glyph.AtlasLocator.PlotLocator())
}

// getPackedGlyphImage packs a glyph's raster into the atlas's expected byte layout: the row-copy lane when the glyph's
// format equals the expected atlas format (this codebase's images are tight — A8 rowBytes == width, ARGB32 device words
// and LCD16 565 words reinterpreted as bytes), plus the A565→ARGB expansion for drivers with no renderable/texturable
// 565 format (resolveMaskFormat rewrites the atlas format there). The BW bit-expansion lane stays unreachable (no BW
// scaler lane).
func getPackedGlyphImage(glyph *font.Glyph, dstRB int, expectedMaskFormat gpu.MaskFormat, dst []byte) {
	width := int(glyph.Width)
	height := int(glyph.Height)
	maskFormat := text.FormatFromGlyph(glyph.Format)
	if maskFormat == expectedMaskFormat {
		var src []byte
		var srcRB int
		switch glyph.Format {
		case font.MaskA8, font.MaskSDF:
			src = glyph.Image
			srcRB = width
		case font.MaskARGB32:
			src = unsafe.Slice((*byte)(unsafe.Pointer(&glyph.Image32[0])), len(glyph.Image32)*4)
			srcRB = width * 4
		case font.MaskLCD16:
			src = unsafe.Slice((*byte)(unsafe.Pointer(&glyph.Image16[0])), len(glyph.Image16)*2)
			srcRB = width * 2
		default:
			panic("unreachable mask format")
		}
		bpp := expectedMaskFormat.BytesPerPixel()
		for y := range height {
			copy(dst[y*dstRB:y*dstRB+width*bpp], src[y*srcRB:y*srcRB+width*bpp])
		}
		return
	}
	if maskFormat == gpu.MaskFormatA565 && expectedMaskFormat == gpu.MaskFormatARGB {
		// Convert if the glyph uses a 565 mask format (LCD text rendering) but the expected format is 8888. N32 is
		// RGBA8888 on every leg, so the packed word is R | G<<8 | B<<16 | FF<<24 with the 5/6/5 channels expanded by
		// bit replication.
		for y := range height {
			for x := range width {
				c565 := glyph.Image16[y*width+x]
				r := uint32(c565>>11) & 0x1F
				g := uint32(c565>>5) & 0x3F
				b := uint32(c565) & 0x1F
				r = r<<3 | r>>2
				g = g<<2 | g>>4
				b = b<<3 | b>>2
				c8888 := r | g<<8 | b<<16 | 0xFF<<24
				di := y*dstRB + x*4
				dst[di] = byte(c8888)
				dst[di+1] = byte(c8888 >> 8)
				dst[di+2] = byte(c8888 >> 16)
				dst[di+3] = byte(c8888 >> 24)
			}
		}
		return
	}
	panic("mask-format conversion lane is unreachable")
}

// AddGlyphToAtlas returns AtlasErrorCodeSucceeded if the glyph was added to the texture atlas. srcPadding is 0 for
// direct masks, 1 for transformed masks, and font.DistanceFieldInset (2) for SDFT glyphs (E.1).
func (m *AtlasManager) AddGlyphToAtlas(skGlyph *font.Glyph, glyph *text.Glyph, srcPadding int, resourceProvider *ResourceProvider, uploadTarget DeferredUploadTarget) AtlasErrorCode {
	if srcPadding < 0 {
		panic("negative padding")
	}
	if skGlyph.Image == nil && skGlyph.Image32 == nil && skGlyph.Image16 == nil {
		return AtlasErrorCodeError
	}
	if glyph == nil {
		panic("nil glyph")
	}

	glyphFormat := text.FormatFromGlyph(skGlyph.Format)
	expectedMaskFormat := m.resolveMaskFormat(glyphFormat)
	bytesPerPixel := expectedMaskFormat.BytesPerPixel()

	var padding int
	switch srcPadding {
	case 0:
		// The direct mask/image case.
		padding = 0
		if m.supportBilerpAtlas {
			// Force direct masks (glyph with no padding) to have padding.
			padding = 1
			srcPadding = 1
		}
	case 1:
		// The transformed mask/image case.
		padding = 1
	case font.DistanceFieldInset:
		// The SDFT case: the padding is already built into the glyph's image; no extra padding needed.
		padding = 0
	default:
		// The padding is not one of the known forms.
		return AtlasErrorCodeError
	}

	width := int(skGlyph.Width) + 2*padding
	height := int(skGlyph.Height) + 2*padding
	rowBytes := width * bytesPerPixel
	size := height * rowBytes

	// Temporary storage for normalizing the glyph image.
	storage := make([]byte, size)
	dataPtr := storage
	if padding > 0 {
		// Advance in one row and one column.
		dataPtr = dataPtr[rowBytes+bytesPerPixel:]
	}
	getPackedGlyphImage(skGlyph, rowBytes, expectedMaskFormat, dataPtr)

	errorCode := m.AddToAtlas(resourceProvider, uploadTarget, expectedMaskFormat, width, height,
		storage, &glyph.AtlasLocator)
	if errorCode == AtlasErrorCodeSucceeded {
		glyph.AtlasLocator.InsetSrc(srcPadding)
	}
	return errorCode
}

// AddToAtlas adds image to the texture atlas that matches format.
func (m *AtlasManager) AddToAtlas(resourceProvider *ResourceProvider, target DeferredUploadTarget, format gpu.MaskFormat, width, height int, image []byte, atlasLocator *gpu.AtlasLocator) AtlasErrorCode {
	return m.getAtlas(format).AddToAtlas(resourceProvider, target, width, height, image,
		atlasLocator)
}

// AddGlyphToBulkAndSetUseToken ensures the atlas does not evict the glyph's mask from its texture backing store: the
// client must pass in the current op token along with the glyph. A BulkUsePlotUpdater manages bulk last-use token
// updating; the use token for the current glyph is set if required. NOTE: the bulk updater is only valid if the subrun
// has a valid atlasGeneration.
func (m *AtlasManager) AddGlyphToBulkAndSetUseToken(updater *gpu.BulkUsePlotUpdater, format gpu.MaskFormat, glyph *text.Glyph, token gpu.Token) {
	if glyph == nil {
		panic("nil glyph")
	}
	if updater.Add(&glyph.AtlasLocator) {
		m.getAtlas(format).SetLastUseToken(&glyph.AtlasLocator, token)
	}
}

// SetUseTokenBulk updates the last-use token for every plot tracked by updater.
func (m *AtlasManager) SetUseTokenBulk(updater *gpu.BulkUsePlotUpdater, token gpu.Token, format gpu.MaskFormat) {
	m.getAtlas(format).SetLastUseTokenBulk(updater, token)
}

// AtlasGeneration returns a monotonically increasing number that changes every time something is removed from the
// atlas's texture backing store.
func (m *AtlasManager) AtlasGeneration(format gpu.MaskFormat) uint64 {
	return m.getAtlas(format).AtlasGeneration()
}

// PreFlush implements OnFlushCallbackObject: it instantiates every active atlas page.
func (m *AtlasManager) PreFlush() bool {
	for i := range gpu.MaskFormatCount {
		if m.atlases[i] != nil {
			m.atlases[i].Instantiate()
		}
	}
	return true
}

// PostFlush implements OnFlushCallbackObject: it compacts every active atlas.
func (m *AtlasManager) PostFlush(startTokenForNextFlush gpu.Token) {
	for i := range gpu.MaskFormatCount {
		if m.atlases[i] != nil {
			m.atlases[i].Compact(startTokenForNextFlush)
		}
	}
}

// RetainOnFreeGpuResources implements OnFlushCallbackObject: the atlas glyph cache always survives freeGpuResources, so
// it remains in the active callback list.
func (m *AtlasManager) RetainOnFreeGpuResources() bool { return true }

// resolveMaskFormat changes an expected 565 mask format to 8888 if 565 is not supported (LCD16 glyphs then upload
// through getPackedGlyphImage's expansion lane).
func (m *AtlasManager) resolveMaskFormat(format gpu.MaskFormat) gpu.MaskFormat {
	if format == gpu.MaskFormatA565 &&
		m.caps.GetDefaultBackendFormat(gpu.ColorTypeBGR565, false) == FormatUnknown {
		format = gpu.MaskFormatARGB
	}
	return format
}

// getAtlas returns the atlas for format; there is a 1:1 mapping between mask formats and atlas indices.
func (m *AtlasManager) getAtlas(format gpu.MaskFormat) *DrawOpAtlas {
	format = m.resolveMaskFormat(format)
	if m.atlases[format] == nil {
		panic("getViews must be called first")
	}
	return m.atlases[format]
}

// initAtlas lazily creates the atlas for format, returning false if creation failed.
func (m *AtlasManager) initAtlas(format gpu.MaskFormat) bool {
	if m.atlases[format] == nil {
		colorType := format.ColorType()
		atlasDimensions := m.atlasConfig.AtlasDimensions(format)
		plotDimensions := m.atlasConfig.PlotDimensions(format)
		backendFormat := m.caps.GetDefaultBackendFormat(colorType, false)

		m.atlases[format] = MakeDrawOpAtlas(m.proxyProvider, backendFormat, colorType,
			colorType.BytesPerPixel(), int(atlasDimensions.Width), int(atlasDimensions.Height),
			int(plotDimensions.Width), int(plotDimensions.Height), &m.AtlasGenerationCounter,
			m.allowMultitexturing, nil, "TextAtlas")
		if m.atlases[format] == nil {
			return false
		}
	}
	return true
}
