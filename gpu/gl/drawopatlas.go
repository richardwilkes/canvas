// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// DrawOpAtlas, DrawOpAtlasConfig, and the deferred-upload contract. The atlas manages one or more texture pages on
// behalf of draw ops, which upload glyph/mask data while preparing their draws at flush time. Uploads are performed in
// "ASAP" mode until it is impossible to add data without overwriting texels read by draws that have not yet executed on
// the gpu; at that point the atlas attempts to activate a new page (up to maxPages), and failing that performs the
// uploads "inline" between draws. When even that is impossible without touching texels read by the draw currently being
// prepared, addToAtlas returns ErrorCodeTryAgain so the op can split its draw. Garbage collection runs through
// compact(): pages are deactivated when their recently-used plots can migrate to earlier pages, based on the plot/atlas
// recently-used-count thresholds. Trims: debug spew and trace events are dropped.

package gl

import (
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/gpu"
)

// DeferredTextureUploadWritePixelsFn is passed to a deferred upload when it is executed, allowing it to write its pixel
// data into a texture.
type DeferredTextureUploadWritePixelsFn func(proxy *SurfaceProxy, rect geom.IRect,
	colorType gpu.ColorType, data []byte, rowBytes int) bool

// DeferredTextureUploadFn is a deferred texture upload, called when it should perform its upload as the draw/upload
// sequence is executed.
type DeferredTextureUploadFn func(writePixels DeferredTextureUploadWritePixelsFn)

// DeferredUploadTarget is an interface for scheduling deferred uploads, accepting ASAP and deferred inline uploads.
type DeferredUploadTarget interface {
	// TokenTracker returns the token tracker (callers may only query, never issue).
	TokenTracker() *gpu.TokenTracker
	// AddInlineUpload returns the token of the draw that this upload will occur before.
	AddInlineUpload(upload DeferredTextureUploadFn) gpu.Token
	// AddASAPUpload returns the token of the draw that this upload will occur before. Since ASAP uploads are done first
	// during a flush, this is the first token since the most recent flush.
	AddASAPUpload(upload DeferredTextureUploadFn) gpu.Token
}

// AllowMultitexturing selects whether an atlas may span multiple texture pages.
type AllowMultitexturing bool

// AllowMultitexturing values.
const (
	AllowMultitexturingNo  AllowMultitexturing = false
	AllowMultitexturingYes AllowMultitexturing = true
)

// AtlasErrorCode is the result of an AddToAtlas call.
type AtlasErrorCode int32

// AtlasErrorCode values.
const (
	// AtlasErrorCodeError: an unrecoverable error; the op being created should be discarded.
	AtlasErrorCodeError AtlasErrorCode = iota
	// AtlasErrorCodeSucceeded: the subimage was added (an upload was scheduled).
	AtlasErrorCodeSucceeded
	// AtlasErrorCodeTryAgain: the subimage cannot fit without overwriting texels read by the current draw; the op
	// should end its draw and begin another before adding more data.
	AtlasErrorCodeTryAgain
)

// Number of atlas-related flushes beyond which we consider a plot to no longer be in use. Low enough that a page with
// unused plots is removed reasonably quickly, but allowing it to hang around in case it's needed (flushes are assumed
// rare).
const (
	plotRecentlyUsedCount  = 32
	atlasRecentlyUsedCount = 128
)

// atlasPage is one texture page of a DrawOpAtlas, with its plots and their LRU order.
type atlasPage struct {
	// plotList is the LRU list of plots (MRU at head, LRU at tail).
	plotList gpu.PlotList
	// plotArray is the allocated array of plots (indexed by plot index).
	plotArray []*gpu.Plot
}

// DrawOpAtlas manages one or more texture pages of plots that draw ops upload glyph/mask data into while preparing
// their draws.
type DrawOpAtlas struct {
	pages [gpu.MaxMultitexturePages]atlasPage
	// generationCounter tracks the atlas eviction state for glyphs: when the atlas evicts a plot it bumps the
	// generation, letting glyph owners detect stale locators.
	generationCounter *gpu.AtlasGenerationCounter
	// views are kept separate from pages to make them easy to pass to clients.
	views             [gpu.MaxMultitexturePages]SurfaceProxyView
	label             string
	evictionCallbacks []gpu.PlotEvictionCallback
	textureHeight     int
	plotHeight        int
	plotWidth         int
	atlasGeneration   uint64
	// prevFlushToken is the nextFlushToken() value at the end of the previous flush.
	prevFlushToken gpu.Token
	// flushesSinceLastUse counts flushes since this atlas was last used.
	flushesSinceLastUse int
	textureWidth        int
	bytesPerPixel       int
	numPlots            uint32
	format              Format
	colorType           gpu.ColorType
	maxPages            uint32
	numActivePages      uint32
}

// MakeDrawOpAtlas creates a DrawOpAtlas. The returned atlas should only be used while preparing draws at flush time.
// Returns nil if creation fails.
func MakeDrawOpAtlas(proxyProvider *ProxyProvider, format Format, colorType gpu.ColorType, bpp, width, height, plotWidth, plotHeight int, generationCounter *gpu.AtlasGenerationCounter, allowMultitexturing AllowMultitexturing, evictor gpu.PlotEvictionCallback, label string) *DrawOpAtlas {
	if format == FormatUnknown {
		return nil
	}
	atlas := newDrawOpAtlas(format, colorType, bpp, width, height, plotWidth, plotHeight,
		generationCounter, allowMultitexturing, label)
	if !atlas.createPages(proxyProvider, generationCounter) || atlas.views[0].Proxy() == nil {
		return nil
	}
	if evictor != nil {
		atlas.evictionCallbacks = append(atlas.evictionCallbacks, evictor)
	}
	return atlas
}

// newDrawOpAtlas creates a DrawOpAtlas with the given format/dimensions.
func newDrawOpAtlas(format Format, colorType gpu.ColorType, bpp, width, height, plotWidth, plotHeight int, generationCounter *gpu.AtlasGenerationCounter, allowMultitexturing AllowMultitexturing, label string) *DrawOpAtlas {
	maxPages := uint32(1)
	if allowMultitexturing == AllowMultitexturingYes {
		maxPages = gpu.MaxMultitexturePages
	}
	numPlotsX := width / plotWidth
	numPlotsY := height / plotHeight
	if numPlotsX*numPlotsY > gpu.MaxPlots {
		panic("too many plots per page")
	}
	if plotWidth*numPlotsX != width || plotHeight*numPlotsY != height {
		panic("plot dimensions must evenly divide the atlas")
	}
	return &DrawOpAtlas{
		format:            format,
		colorType:         colorType,
		bytesPerPixel:     bpp,
		textureWidth:      width,
		textureHeight:     height,
		plotWidth:         plotWidth,
		plotHeight:        plotHeight,
		numPlots:          uint32(numPlotsX * numPlotsY),
		label:             label,
		generationCounter: generationCounter,
		atlasGeneration:   generationCounter.Next(),
		prevFlushToken:    gpu.InvalidToken(),
		maxPages:          maxPages,
	}
}

// Views returns the atlas's per-page texture proxy views.
func (a *DrawOpAtlas) Views() *[gpu.MaxMultitexturePages]SurfaceProxyView { return &a.views }

// AtlasGeneration returns a monotonically increasing number that changes every time something is removed from the
// texture backing store.
func (a *DrawOpAtlas) AtlasGeneration() uint64 { return a.atlasGeneration }

// HasID reports whether plotLocator still identifies a live plot entry.
func (a *DrawOpAtlas) HasID(plotLocator gpu.PlotLocator) bool {
	if !plotLocator.IsValid() {
		return false
	}
	plot := plotLocator.PlotIndex()
	page := plotLocator.PageIndex()
	if plot >= a.numPlots || page >= a.numActivePages {
		return false
	}
	return a.pages[page].plotArray[plot].GenID() == plotLocator.GenID()
}

// SetLastUseToken records the last-use token for atlasLocator's entry, to ensure the atlas does not evict it.
func (a *DrawOpAtlas) SetLastUseToken(atlasLocator *gpu.AtlasLocator, token gpu.Token) {
	if !a.HasID(atlasLocator.PlotLocator()) {
		panic("setLastUseToken on an evicted entry")
	}
	plotIdx := atlasLocator.PlotIndex()
	pageIdx := atlasLocator.PageIndex()
	plot := a.pages[pageIdx].plotArray[plotIdx]
	a.makeMRU(plot, pageIdx)
	plot.SetLastUseToken(token)
}

// NumActivePages returns the number of currently active atlas pages.
func (a *DrawOpAtlas) NumActivePages() uint32 { return a.numActivePages }

// SetLastUseTokenBulk records the last-use token for every plot tracked by updater.
func (a *DrawOpAtlas) SetLastUseTokenBulk(updater *gpu.BulkUsePlotUpdater, token gpu.Token) {
	count := updater.Count()
	for i := range count {
		pd := updater.PlotData(i)
		// It's possible a plot was added to the updater and subsequently its page was deleted -- so check to prevent a
		// crash.
		if pd.PageIndex < a.numActivePages {
			plot := a.pages[pd.PageIndex].plotArray[pd.PlotIndex]
			a.makeMRU(plot, pd.PageIndex)
			plot.SetLastUseToken(token)
		}
	}
}

// MaxPages returns the maximum number of pages this atlas may activate.
func (a *DrawOpAtlas) MaxPages() uint32 { return a.maxPages }

// makeMRU moves plot to the head of its page's LRU list. No MRU update for pages: uploads prioritize the front pages
// and removal starts from the back, so page order is fixed.
func (a *DrawOpAtlas) makeMRU(plot *gpu.Plot, pageIdx uint32) {
	if a.pages[pageIdx].plotList.Head() == plot {
		return
	}
	a.pages[pageIdx].plotList.Remove(plot)
	a.pages[pageIdx].plotList.AddToHead(plot)
}

// validate panics unless the plot index stored in the locator is consistent with the glyph rectangle.
func (a *DrawOpAtlas) validate(atlasLocator *gpu.AtlasLocator) {
	numPlotsX := a.textureWidth / a.plotWidth
	numPlotsY := a.textureHeight / a.plotHeight
	plotIndex := int(atlasLocator.PlotIndex())
	topLeft := atlasLocator.TopLeft()
	plotX := int(topLeft.X) / a.plotWidth
	plotY := int(topLeft.Y) / a.plotHeight
	if plotIndex != (numPlotsY-plotY-1)*numPlotsX+(numPlotsX-plotX-1) {
		panic("atlas locator's plot index is inconsistent with its rectangle")
	}
}

// processEviction notifies the registered eviction callbacks and bumps the atlas generation.
func (a *DrawOpAtlas) processEviction(plotLocator gpu.PlotLocator) {
	for _, evictor := range a.evictionCallbacks {
		evictor.Evict(plotLocator)
	}
	a.atlasGeneration = a.generationCounter.Next()
}

// processEvictionAndResetRects evicts plot's current contents and clears its packed rects.
func (a *DrawOpAtlas) processEvictionAndResetRects(plot *gpu.Plot) {
	a.processEviction(plot.PlotLocator())
	plot.ResetRects(false /* freeData */)
}

// uploadPlotToTexture writes plot's staged pixel data to proxy via writePixels. The call is skipped when the plot has
// no backing allocation (the write would be a no-op).
func (a *DrawOpAtlas) uploadPlotToTexture(writePixels DeferredTextureUploadWritePixelsFn, proxy *SurfaceProxy, plot *gpu.Plot) {
	if proxy == nil || proxy.PeekTexture() == nil {
		panic("upload to an uninstantiated atlas page")
	}
	data, rect := plot.PrepareForUpload()
	if data == nil {
		return
	}
	writePixels(proxy, rect, a.colorType, data, a.bytesPerPixel*a.plotWidth)
}

// updatePlot schedules (or piggybacks on an existing) upload for plot and updates atlasLocator.
func (a *DrawOpAtlas) updatePlot(target DeferredUploadTarget, atlasLocator *gpu.AtlasLocator, plot *gpu.Plot) bool {
	pageIdx := plot.PageIndex()
	if pageIdx >= a.numActivePages {
		return false
	}
	a.makeMRU(plot, pageIdx)

	// If our most recent upload has already occurred then we have to insert a new upload. Otherwise, we already have a
	// scheduled upload that hasn't yet occurred and this new update will piggy back on it.
	if plot.LastUploadToken().LessThan(target.TokenTracker().NextFlushToken()) {
		proxy := a.views[pageIdx].Proxy()
		if proxy == nil || !proxy.IsInstantiated() {
			panic("atlas page not instantiated at flush time")
		}
		lastUploadToken := target.AddASAPUpload(
			func(writePixels DeferredTextureUploadWritePixelsFn) {
				a.uploadPlotToTexture(writePixels, proxy, plot)
			},
		)
		plot.SetLastUploadToken(lastUploadToken)
	}
	atlasLocator.UpdatePlotLocator(plot.PlotLocator())
	a.validate(atlasLocator)
	return true
}

// uploadToPage looks through all allocated plots for one that can fit the subimage, in most-recently-used order.
func (a *DrawOpAtlas) uploadToPage(pageIdx uint32, target DeferredUploadTarget, width, height int, image []byte, atlasLocator *gpu.AtlasLocator) bool {
	if a.views[pageIdx].Proxy() == nil || !a.views[pageIdx].Proxy().IsInstantiated() {
		panic("uploadToPage on an uninstantiated page")
	}
	for plot := a.pages[pageIdx].plotList.Head(); plot != nil; plot = plot.Next() {
		if plot.AddSubImage(width, height, image, atlasLocator) {
			return a.updatePlot(target, atlasLocator, plot)
		}
	}
	return false
}

// AddToAtlas adds a width x height subimage to the atlas, returning the subimage's locator in the backing texture on
// success. Upon success an upload of the image data has been added to the target, in ASAP mode if possible, otherwise
// inline.
//
// NOTE: when an op prepares a draw that reads from the atlas, it must immediately call SetLastUseToken with the current
// draw token, otherwise the next AddToAtlas call might cause the previous data to be overwritten before it has been
// read.
func (a *DrawOpAtlas) AddToAtlas(resourceProvider *ResourceProvider, target DeferredUploadTarget, width, height int, image []byte, atlasLocator *gpu.AtlasLocator) AtlasErrorCode {
	if width > a.plotWidth || height > a.plotHeight {
		return AtlasErrorCodeError
	}

	// Look through each page to see if we can upload without having to flush. We prioritize this upload to the first
	// pages, not the most recently used, to make it easier to remove unused pages in reverse page order.
	for pageIdx := uint32(0); pageIdx < a.numActivePages; pageIdx++ {
		if a.uploadToPage(pageIdx, target, width, height, image, atlasLocator) {
			return AtlasErrorCodeSucceeded
		}
	}

	// If the above fails, then see if the least recently used plot per page has already been flushed to the gpu if
	// we're at max page allocation, or if the plot has aged out otherwise. We wait until we've grown to the full number
	// of pages to begin evicting already-flushed plots so that we can maximize the opportunity for reuse. As before we
	// prioritize this upload to the first pages, not the most recently used.
	if a.numActivePages == a.maxPages {
		for pageIdx := uint32(0); pageIdx < a.numActivePages; pageIdx++ {
			plot := a.pages[pageIdx].plotList.Tail()
			if plot == nil {
				panic("active page without plots")
			}
			if plot.LastUseToken().LessThan(target.TokenTracker().NextFlushToken()) {
				a.processEvictionAndResetRects(plot)
				if !plot.AddSubImage(width, height, image, atlasLocator) {
					panic("add to a freshly reset plot failed")
				}
				if !a.updatePlot(target, atlasLocator, plot) {
					return AtlasErrorCodeError
				}
				return AtlasErrorCodeSucceeded
			}
		}
	} else {
		// If we haven't activated all the available pages, try to create a new one and add to it.
		if !a.activateNewPage(resourceProvider) {
			return AtlasErrorCodeError
		}
		if a.uploadToPage(a.numActivePages-1, target, width, height, image, atlasLocator) {
			return AtlasErrorCodeSucceeded
		}
		// If we fail to upload to a newly activated page then something has gone terribly wrong.
		return AtlasErrorCodeError
	}

	if a.numActivePages == 0 {
		return AtlasErrorCodeError
	}

	// Try to find a plot that we can perform an inline upload to. We prioritize this upload in reverse order of pages
	// to counterbalance the order above.
	var plot *gpu.Plot
	for pageIdx := int(a.numActivePages) - 1; pageIdx >= 0; pageIdx-- {
		currentPlot := a.pages[pageIdx].plotList.Tail()
		if currentPlot.LastUseToken() != target.TokenTracker().NextDrawToken() {
			plot = currentPlot
			break
		}
	}

	// If we can't find a plot that is not used in a draw currently being prepared by an op, then we have to fail. This
	// gives the op a chance to enqueue the draw and call back into this function: when that draw is enqueued the draw
	// token advances, and the subsequent call will continue past this branch and prepare an inline upload that will
	// occur after the enqueued draw that references the plot's pre-upload content.
	if plot == nil {
		return AtlasErrorCodeTryAgain
	}

	a.processEviction(plot.PlotLocator())
	pageIdx := plot.PageIndex()
	a.pages[pageIdx].plotList.Remove(plot)
	newPlot := plot.Clone()
	a.pages[pageIdx].plotArray[plot.PlotIndex()] = newPlot

	a.pages[pageIdx].plotList.AddToHead(newPlot)
	if !newPlot.AddSubImage(width, height, image, atlasLocator) {
		panic("add to a cloned plot failed")
	}

	// Note that this plot will be uploaded inline with the draws whereas the one it displaced most likely was uploaded
	// ASAP.
	proxy := a.views[pageIdx].Proxy()
	if proxy == nil || !proxy.IsInstantiated() {
		panic("atlas page not instantiated at flush time")
	}
	lastUploadToken := target.AddInlineUpload(
		func(writePixels DeferredTextureUploadWritePixelsFn) {
			a.uploadPlotToTexture(writePixels, proxy, newPlot)
		},
	)
	newPlot.SetLastUploadToken(lastUploadToken)

	atlasLocator.UpdatePlotLocator(newPlot.PlotLocator())
	a.validate(atlasLocator)
	return AtlasErrorCodeSucceeded
}

// Compact is called by the atlas's client (the atlas manager's postFlush) to shift data from higher-index pages to
// lower ones and deactivate pages no longer in use.
func (a *DrawOpAtlas) Compact(startTokenForNextFlush gpu.Token) {
	if a.numActivePages < 1 {
		a.prevFlushToken = startTokenForNextFlush
		return
	}

	// For all plots, reset number of flushes since used if used this frame.
	atlasUsedThisFlush := false
	for pageIndex := uint32(0); pageIndex < a.numActivePages; pageIndex++ {
		for plot := a.pages[pageIndex].plotList.Head(); plot != nil; plot = plot.Next() {
			if plot.LastUseToken().InInterval(a.prevFlushToken, startTokenForNextFlush) {
				plot.ResetFlushesSinceLastUsed()
				atlasUsedThisFlush = true
			}
		}
	}

	if atlasUsedThisFlush {
		a.flushesSinceLastUse = 0
	} else {
		a.flushesSinceLastUse++
	}

	// We only try to compact if the atlas was used in the recently completed flush or hasn't been used in a long time.
	// This handles the case where a lot of text or path rendering has occurred but then just a blinking cursor is
	// drawn.
	if atlasUsedThisFlush || a.flushesSinceLastUse > atlasRecentlyUsedCount {
		var availablePlots []*gpu.Plot
		lastPageIndex := a.numActivePages - 1

		// For all plots but the last one, update number of flushes since used, and check to see if there are any in the
		// first pages that the last page can safely upload to.
		for pageIndex := uint32(0); pageIndex < lastPageIndex; pageIndex++ {
			for plot := a.pages[pageIndex].plotList.Head(); plot != nil; plot = plot.Next() {
				// We only increment the 'sinceLastUsed' count for flushes where the atlas was used to avoid deleting
				// everything when we return to text drawing in the blinking cursor case.
				if !plot.LastUseToken().InInterval(a.prevFlushToken, startTokenForNextFlush) {
					plot.IncFlushesSinceLastUsed()
				}
				// Count plots we can potentially upload to in all pages except the last one (the potential compactee).
				if plot.FlushesSinceLastUsed() > plotRecentlyUsedCount {
					availablePlots = append(availablePlots, plot)
				}
			}
		}

		// Count recently used plots in the last page and evict any that are no longer in use. Since we prioritize
		// uploading to the first pages, this will eventually clear out usage of this page unless we have a large need.
		usedPlots := uint32(0)
		for plot := a.pages[lastPageIndex].plotList.Head(); plot != nil; plot = plot.Next() {
			if !plot.LastUseToken().InInterval(a.prevFlushToken, startTokenForNextFlush) {
				plot.IncFlushesSinceLastUsed()
			}
			if plot.FlushesSinceLastUsed() <= plotRecentlyUsedCount {
				usedPlots++
			} else if plot.LastUseToken() != gpu.InvalidToken() {
				// Otherwise if aged out just evict it.
				a.processEvictionAndResetRects(plot)
			}
		}

		// If recently used plots in the last page are using less than a quarter of the page, try to evict them if
		// there's available space in earlier pages. Since we prioritize uploading to the first pages, this will
		// eventually clear out usage of this page unless we have a large need.
		if len(availablePlots) > 0 && usedPlots != 0 && usedPlots <= a.numPlots/4 {
			for plot := a.pages[lastPageIndex].plotList.Head(); plot != nil; plot = plot.Next() {
				if plot.FlushesSinceLastUsed() <= plotRecentlyUsedCount {
					// See if there's room in an earlier page and if so evict. We need to be somewhat harsh here so that
					// a handful of plots that are consistently in use don't end up locking the page in memory.
					if len(availablePlots) > 0 {
						a.processEvictionAndResetRects(plot)
						a.processEvictionAndResetRects(availablePlots[len(availablePlots)-1])
						availablePlots = availablePlots[:len(availablePlots)-1]
						usedPlots--
					}
					if usedPlots == 0 || len(availablePlots) == 0 {
						break
					}
				}
			}
		}

		// If none of the plots in the last page have been used recently, delete it.
		if usedPlots == 0 {
			a.deactivateLastPage()
			a.flushesSinceLastUse = 0
		}
	}

	a.prevFlushToken = startTokenForNextFlush
}

// Instantiate ensures every active page's proxy is instantiated. Atlas proxies are created with UseAllocatorNo
// (independent of the ops there are ASAP and inline uploads to them, and they persist beyond the last use in an op for
// a given flush), so the atlas manages their lifetime via the onFlushCallback system, which calls this method. All
// active pages were instantiated in activateNewPage, so this only validates.
func (a *DrawOpAtlas) Instantiate() {
	for i := uint32(0); i < a.numActivePages; i++ {
		if a.views[i].Proxy() == nil || !a.views[i].Proxy().IsInstantiated() {
			panic("active atlas page not instantiated")
		}
	}
}

// createPages allocates the atlas's page proxies and their plot arrays/LRU lists.
func (a *DrawOpAtlas) createPages(proxyProvider *ProxyProvider, generationCounter *gpu.AtlasGenerationCounter) bool {
	if a.textureWidth&(a.textureWidth-1) != 0 || a.textureHeight&(a.textureHeight-1) != 0 {
		panic("atlas dimensions must be powers of two")
	}
	dims := geom.ISize{Width: int32(a.textureWidth), Height: int32(a.textureHeight)}
	numPlotsX := a.textureWidth / a.plotWidth
	numPlotsY := a.textureHeight / a.plotHeight

	for i := uint32(0); i < a.maxPages; i++ {
		swizzle := proxyProvider.Caps().ReadSwizzle(a.format, a.colorType)
		if a.colorType.ChannelFlags() == gpu.AlphaChannelFlag {
			swizzle = gpu.ConcatSwizzles(swizzle, gpu.MakeSwizzle("aaaa"))
		}
		proxy := proxyProvider.CreateProxy(a.format, dims, gpu.RenderableNo, 1, gpu.MipmappedNo,
			gpu.BackingFitExact, gpu.BudgetedYes, a.label, 0, UseAllocatorNo)
		if proxy == nil {
			return false
		}
		a.views[i] = MakeSurfaceProxyView(proxy, gpu.OriginTopLeft, swizzle)

		// Set up allocated plots.
		a.pages[i].plotArray = make([]*gpu.Plot, numPlotsX*numPlotsY)
		for y, r := numPlotsY-1, 0; y >= 0; y, r = y-1, r+1 {
			for x, c := numPlotsX-1, 0; x >= 0; x, c = x-1, c+1 {
				index := r*numPlotsX + c
				plot := gpu.NewPlot(int(i), index, generationCounter, x, y, a.plotWidth,
					a.plotHeight, a.colorType, a.bytesPerPixel)
				a.pages[i].plotArray[index] = plot
				// Build the LRU list.
				a.pages[i].plotList.AddToHead(plot)
			}
		}
	}
	return true
}

// activateNewPage instantiates and activates the next inactive page.
func (a *DrawOpAtlas) activateNewPage(resourceProvider *ResourceProvider) bool {
	if a.numActivePages >= a.maxPages {
		panic("no page left to activate")
	}
	if !a.views[a.numActivePages].Proxy().Instantiate(resourceProvider) {
		return false
	}
	a.numActivePages++
	return true
}

// deactivateLastPage resets and deinstantiates the last active page.
func (a *DrawOpAtlas) deactivateLastPage() {
	if a.numActivePages == 0 {
		panic("no active page to deactivate")
	}
	lastPageIndex := a.numActivePages - 1
	numPlotsX := a.textureWidth / a.plotWidth
	numPlotsY := a.textureHeight / a.plotHeight

	a.pages[lastPageIndex].plotList.Reset()
	for r := range numPlotsY {
		for c := range numPlotsX {
			plotIndex := r*numPlotsX + c
			currPlot := a.pages[lastPageIndex].plotArray[plotIndex]
			currPlot.ResetRects(false /* freeData */)
			currPlot.ResetFlushesSinceLastUsed()
			// Rebuild the LRU list.
			a.pages[lastPageIndex].plotList.AddToHead(currPlot)
		}
	}

	// Remove the ref to the backing texture.
	a.views[lastPageIndex].Proxy().Deinstantiate()
	a.numActivePages--
}

// DrawOpAtlasConfig computes atlas/plot dimensions: there are three atlases (A8, 565, ARGB) kept in relation with one
// another. Because A8 is the most frequently used mask format its dimensions are 2x the 565 and ARGB dimensions, with
// the constraint that an atlas size always contains at least one plot. Since the ARGB atlas takes the most space, its
// dimensions size the other two.
type DrawOpAtlasConfig struct {
	argbDimensions geom.ISize
	maxTextureSize int
}

// maxAtlasDim: on some systems texture coordinates are represented using half-precision floating point with 11
// significant bits, which limits the largest atlas dimensions to 2048x2048; for simplicity that constraint is applied
// to all atlas textures.
const maxAtlasDim = 2048

// MakeDrawOpAtlasConfig builds a DrawOpAtlasConfig. maxTextureSize comes from the caps; maxBytes is the largest the
// client wants a single atlas texture to be (multitexturing may expand use temporarily). maxBytes of 0 yields
// minimum-sized atlases (a single plot for ARGB, four for A8), which is what tests typically want.
func MakeDrawOpAtlasConfig(maxTextureSize int, maxBytes uint64) DrawOpAtlasConfig {
	argbDimensions := [...]geom.ISize{
		{Width: 256, Height: 256},   // maxBytes < 2^19
		{Width: 512, Height: 256},   // 2^19 <= maxBytes < 2^20
		{Width: 512, Height: 512},   // 2^20 <= maxBytes < 2^21
		{Width: 1024, Height: 512},  // 2^21 <= maxBytes < 2^22
		{Width: 1024, Height: 1024}, // 2^22 <= maxBytes < 2^23
		{Width: 2048, Height: 1024}, // 2^23 <= maxBytes
	}
	// Index 0 corresponds to maxBytes of 2^18, so start by dividing by that, then take the floor of the log to get the
	// index.
	maxBytes >>= 18
	index := 0
	if maxBytes > 0 {
		index = min(max(skPrevLog2(int(maxBytes)), 0), len(argbDimensions)-1)
	}
	return DrawOpAtlasConfig{
		argbDimensions: geom.ISize{
			Width:  int32(min(int(argbDimensions[index].Width), maxTextureSize)),
			Height: int32(min(int(argbDimensions[index].Height), maxTextureSize)),
		},
		maxTextureSize: min(maxTextureSize, maxAtlasDim),
	}
}

// AtlasDimensions returns the atlas texture dimensions for the given mask format.
func (c *DrawOpAtlasConfig) AtlasDimensions(format gpu.MaskFormat) geom.ISize {
	if format == gpu.MaskFormatA8 {
		// A8 is always 2x the ARGB dimensions, clamped to the max allowed texture size.
		return geom.ISize{
			Width:  int32(min(2*int(c.argbDimensions.Width), c.maxTextureSize)),
			Height: int32(min(2*int(c.argbDimensions.Height), c.maxTextureSize)),
		}
	}
	return c.argbDimensions
}

// PlotDimensions returns the plot dimensions for the given mask format.
func (c *DrawOpAtlasConfig) PlotDimensions(format gpu.MaskFormat) geom.ISize {
	if format == gpu.MaskFormatA8 {
		atlasDimensions := c.AtlasDimensions(format)
		// For A8 we want to grow the plots at larger texture sizes to accept more of the larger SDF glyphs. Since the
		// largest SDF glyph can be 170x170 with padding, this allows us to pack 3 in a 512x256 plot, or 9 in a 512x512
		// plot. This gives 512x256 plots for 2048x1024, 512x512 plots for 2048x2048, and 256x256 plots otherwise.
		plotWidth := int32(256)
		if atlasDimensions.Width >= 2048 {
			plotWidth = 512
		}
		plotHeight := int32(256)
		if atlasDimensions.Height >= 2048 {
			plotHeight = 512
		}
		return geom.ISize{Width: plotWidth, Height: plotHeight}
	}
	// ARGB and LCD always use 256x256 plots -- this has been shown to be faster.
	return geom.ISize{Width: 256, Height: 256}
}
