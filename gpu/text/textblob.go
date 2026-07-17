// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Blob and BlobRedrawCoordinator: a fully processed text blob (a SubRunContainer with a reuse key) and the
// budgeted cache that reuses blobs across draws. A8 masks are gamma-free, so the non-LCD canonical color is a constant
// and non-LCD blob reuse stays color-independent; LCD blobs key on the pixel geometry and regenerate on luminance-color
// changes (see BlobKey/CanReuse). There is no destruction hook to purge a blob's cache entry when its source is freed,
// so stale blobs leave through the same LRU budget sweep that handles ordinary eviction. Untracked glyph-run lists (no
// source blob) ride the same drawing path uncached.

package text

import (
	"sync"

	"github.com/richardwilkes/canvas/canvas"
	"github.com/richardwilkes/canvas/colorcore"
	"github.com/richardwilkes/canvas/font"
	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/maskfilter"
	"github.com/richardwilkes/canvas/textblob"
)

// BlobKey identifies a cached Blob for reuse. Fields with no reachable variation in this codebase are omitted: the
// scaler context flags are always the sRGB-only configuration, and the non-LCD canonical color is a constant because A8
// masks are gamma-free. LCD text keys on the pixel geometry and regenerates on any luminance-color change (see
// CanReuse).
type BlobKey struct {
	positionMatrix       geom.Matrix
	uniqueID             uint32
	frameWidth           float32
	miterLimit           float32
	blurSigma            float32
	blurStyle            maskfilter.BlurStyle
	style                canvas.Style
	pixelGeometry        font.PixelGeometry
	join                 canvas.StrokeJoin
	hasSomeDirectSubRuns bool
	hasBlur              bool
	hasLCD               bool
}

// anyRunsLCD reports whether any run in glyphRunList uses subpixel (LCD) anti-aliasing.
func anyRunsLCD(glyphRunList *textblob.GlyphRunList) bool {
	for i := range glyphRunList.Runs {
		if glyphRunList.Runs[i].Font.Edging() == font.EdgingSubpixelAntiAlias {
			return true
		}
	}
	return false
}

// MakeBlobKey builds a reuse key for glyphRunList. drawMatrix is the position matrix (the view matrix with the draw
// origin prepended, as findOrCreateBlob passes it); control is the device's SubRunControl, whose IsDirect decision
// (including its SDFT term) feeds the hasSomeDirectSubRuns bit.
func MakeBlobKey(glyphRunList *textblob.GlyphRunList, paint *canvas.Paint, drawMatrix *geom.Matrix, deviceProps *font.DeviceProps, control *SubRunControl) (canCache bool, key BlobKey) {
	var blur blurQuery
	if paint.MaskFilter != nil {
		blur, _ = asABlur(paint.MaskFilter)
	}
	// TODO: for animated mask filters this will fill up the cache.
	canCache = glyphRunList.Blob != nil &&
		paint.PathEffect == nil && (paint.MaskFilter == nil || blur != nil)

	if canCache {
		// Canonicalize all non-LCD draws to use unknown pixel geometry (see the BlobKey comment: the non-LCD canonical
		// color is always a constant).
		key.hasLCD = anyRunsLCD(glyphRunList)
		if key.hasLCD && deviceProps != nil {
			key.pixelGeometry = deviceProps.PixelGeometry
		}
		key.uniqueID = glyphRunList.Blob.UniqueID()
		key.style = paint.Style
		if key.style != canvas.StyleFill {
			key.frameWidth = paint.StrokeWidth
			key.miterLimit = paint.MiterLimit
			key.join = paint.Join
		}
		key.hasBlur = paint.MaskFilter != nil
		if key.hasBlur {
			key.blurStyle = blur.Style()
			key.blurSigma = blur.Sigma()
		}

		// Do any runs use direct drawing types?
		glyphRunListLocation := glyphRunList.SourceBoundsWithOrigin().Center()
		for i := range glyphRunList.Runs {
			approximateDeviceTextSize := approximateTransformedTextSize(
				glyphRunList.Runs[i].Font, drawMatrix, glyphRunListLocation,
			)
			if control.IsDirect(approximateDeviceTextSize, paint, drawMatrix) {
				key.hasSomeDirectSubRuns = true
			}
		}

		if key.hasSomeDirectSubRuns {
			// Store the fractional offset of the position. The matrix can't be perspective here.
			mappedOrigin := drawMatrix.MapXY(0, 0)
			key.positionMatrix = *drawMatrix
			key.positionMatrix.Set(geom.MTransX,
				mappedOrigin.X-floorScalar(mappedOrigin.X))
			key.positionMatrix.Set(geom.MTransY,
				mappedOrigin.Y-floorScalar(mappedOrigin.Y))
		} else {
			// For path subruns, the matrix doesn't matter.
			key.positionMatrix = geom.IdentityMatrix()
		}
	}
	return canCache, key
}

// equal reports whether two keys describe reusable state for the same draw.
func (k *BlobKey) equal(that *BlobKey) bool {
	if k.uniqueID != that.uniqueID {
		return false
	}
	if k.hasLCD != that.hasLCD || k.pixelGeometry != that.pixelGeometry {
		return false
	}
	if k.style != that.style {
		return false
	}
	if k.style != canvas.StyleFill {
		if k.frameWidth != that.frameWidth || k.miterLimit != that.miterLimit ||
			k.join != that.join {
			return false
		}
	}
	if k.hasBlur != that.hasBlur {
		return false
	}
	if k.hasBlur && (k.blurStyle != that.blurStyle || k.blurSigma != that.blurSigma) {
		return false
	}

	// Direct subruns do not support perspective when used with a Blob.
	if k.positionMatrix.HasPerspective() && k.hasSomeDirectSubRuns {
		return false
	}
	if k.hasSomeDirectSubRuns != that.hasSomeDirectSubRuns {
		return false
	}
	if k.hasSomeDirectSubRuns {
		compatible, _ := canUseDirect(&k.positionMatrix, &that.positionMatrix)
		return compatible
	}
	return true
}

// Blob is a fully processed text blob, suitable for nearly immediate drawing on the GPU.
type Blob struct {
	subRuns *SubRunContainer
	prev    *Blob // LRU links, managed under the coordinator's lock.
	next    *Blob
	size    int
	key     BlobKey
	// initialLuminance is the luminance color the blob was built with, consulted by the LCD CanReuse check.
	initialLuminance colorcore.Color
}

// MakeTextBlob processes glyphRunList into a fully resolved, cacheable Blob.
func MakeTextBlob(glyphRunList *textblob.GlyphRunList, paint *canvas.Paint, positionMatrix *geom.Matrix, deviceProps *font.DeviceProps, control *SubRunControl) *Blob {
	container := MakeSubRuns(glyphRunList, positionMatrix, paint, deviceProps, control)
	// Approximate the blob's footprint with the per-glyph data (positions + variants) plus a per-blob overhead so the
	// budget sweep has comparable teeth.
	size := 256
	for i := range glyphRunList.Runs {
		size += len(glyphRunList.Runs[i].Glyphs) * 32
	}
	return &Blob{subRuns: container, size: size, initialLuminance: paint.LuminanceColor()}
}

// AddKey records key as this blob's reuse key.
func (b *Blob) AddKey(key *BlobKey) { b.key = *key }

// Key returns this blob's reuse key.
func (b *Blob) Key() *BlobKey { return &b.key }

// Size returns the blob's approximate footprint in bytes, used for cache budgeting.
func (b *Blob) Size() int { return b.size }

// SubRunContainer exposes the blob's container (tests).
func (b *Blob) SubRunContainer() *SubRunContainer { return b.subRuns }

// CanReuse reports whether this blob can be redrawn as-is for the given paint and position matrix.
func (b *Blob) CanReuse(paint *canvas.Paint, positionMatrix *geom.Matrix) bool {
	// A singular matrix will create a Blob with no subruns, but unknown glyphs can also cause empty runs. If there
	// are no subruns, regenerate when the matrices don't match.
	if b.subRuns.IsEmpty() && !b.subRuns.InitialPosition().CheapEqual(positionMatrix) {
		return false
	}
	// If we have LCD text then the canonical color is a placeholder, so we have to regenerate the blob on any
	// luminance-color change — the masks bake in the pre-blend.
	if b.key.hasLCD && b.initialLuminance != paint.LuminanceColor() {
		return false
	}
	return b.subRuns.CanReuse(positionMatrix)
}

// Draw renders the blob's subruns.
func (b *Blob) Draw(c *canvas.Canvas, drawOrigin geom.Point, paint *canvas.Paint, atlas AtlasDrawDelegate) {
	b.subRuns.Draw(c, drawOrigin, paint, atlas)
}

//////////////////////////////////////////////////////////////////////////////
// BlobRedrawCoordinator.

// blobCacheDefaultBudget is the default byte budget for the text blob cache.
const blobCacheDefaultBudget = 1 << 22

// BlobRedrawCoordinator reuses data from previous drawing operations. The draw data is stored in a three-tiered
// system: the first tier is keyed by the source blob's uniqueID, the second matches the Blob key, and the last
// queries each subrun with CanReuse.
type BlobRedrawCoordinator struct {
	blobIDCache map[uint32][]*Blob
	head        *Blob // LRU order: head = most recent
	tail        *Blob
	currentSize int
	sizeBudget  int
	mu          sync.Mutex
}

// NewTextBlobRedrawCoordinator returns an empty coordinator with the default size budget.
func NewTextBlobRedrawCoordinator() *BlobRedrawCoordinator {
	return &BlobRedrawCoordinator{
		blobIDCache: make(map[uint32][]*Blob),
		sizeBudget:  blobCacheDefaultBudget,
	}
}

// DrawGlyphRunList draws glyphRunList, reusing or building a cached Blob as needed.
func (c *BlobRedrawCoordinator) DrawGlyphRunList(cv *canvas.Canvas, viewMatrix *geom.Matrix, glyphRunList *textblob.GlyphRunList, paint *canvas.Paint, deviceProps *font.DeviceProps, control *SubRunControl, atlas AtlasDrawDelegate) {
	blob := c.findOrCreateBlob(viewMatrix, glyphRunList, paint, deviceProps, control)
	blob.Draw(cv, glyphRunList.Origin, paint, atlas)
}

// findOrCreateBlob returns a reusable cached Blob for glyphRunList, or builds and caches a fresh one.
func (c *BlobRedrawCoordinator) findOrCreateBlob(viewMatrix *geom.Matrix, glyphRunList *textblob.GlyphRunList, paint *canvas.Paint, deviceProps *font.DeviceProps, control *SubRunControl) *Blob {
	positionMatrix := *viewMatrix
	positionMatrix.PreTranslate(glyphRunList.Origin.X, glyphRunList.Origin.Y)

	canCache, key := MakeBlobKey(glyphRunList, paint, &positionMatrix, deviceProps, control)
	var blob *Blob
	if canCache {
		blob = c.find(&key)
	}

	if blob == nil || !blob.CanReuse(paint, &positionMatrix) {
		if blob != nil {
			// We have to remake the blob because changes may invalidate our masks.
			c.remove(blob)
		}
		blob = MakeTextBlob(glyphRunList, paint, &positionMatrix, deviceProps, control)
		if canCache {
			blob.AddKey(&key)
			blob = c.addOrReturnExisting(blob)
		}
	}
	return blob
}

// addOrReturnExisting inserts blob into the cache, or returns an already-cached blob with the same key.
func (c *BlobRedrawCoordinator) addOrReturnExisting(blob *Blob) *Blob {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.internalAdd(blob)
}

// find looks up a cached blob by key, moving it to the LRU head on a hit.
func (c *BlobRedrawCoordinator) find(key *BlobKey) *Blob {
	c.mu.Lock()
	defer c.mu.Unlock()
	blob := c.findInEntry(key)
	if blob != nil && blob != c.head {
		c.detach(blob)
		c.attachToHead(blob)
	}
	return blob
}

// findInEntry scans the per-ID bucket for a blob matching key. Caller holds c.mu.
func (c *BlobRedrawCoordinator) findInEntry(key *BlobKey) *Blob {
	for _, blob := range c.blobIDCache[key.uniqueID] {
		if blob.key.equal(key) {
			return blob
		}
	}
	return nil
}

// remove evicts blob from the cache.
func (c *BlobRedrawCoordinator) remove(blob *Blob) {
	c.mu.Lock()
	c.internalRemove(blob)
	c.mu.Unlock()
}

// internalRemove evicts blob from the cache. Caller holds c.mu.
func (c *BlobRedrawCoordinator) internalRemove(blob *Blob) {
	id := blob.key.uniqueID
	entry, ok := c.blobIDCache[id]
	if !ok {
		return
	}
	if still := c.findInEntry(&blob.key); still == blob {
		c.currentSize -= blob.size
		c.detach(blob)
		for i := range entry {
			if entry[i] == blob {
				// removeShuffle.
				entry[i] = entry[len(entry)-1]
				entry = entry[:len(entry)-1]
				break
			}
		}
		if len(entry) == 0 {
			delete(c.blobIDCache, id)
		} else {
			c.blobIDCache[id] = entry
		}
	}
}

// internalAdd inserts blob into the cache, or returns an already-cached blob with the same key. Caller holds c.mu.
func (c *BlobRedrawCoordinator) internalAdd(blob *Blob) *Blob {
	id := blob.key.uniqueID
	if alreadyIn := c.findInEntry(&blob.key); alreadyIn != nil {
		blob = alreadyIn
	} else {
		c.attachToHead(blob)
		c.currentSize += blob.size
		c.blobIDCache[id] = append(c.blobIDCache[id], blob)
	}
	c.internalCheckPurge(blob)
	return blob
}

// internalCheckPurge evicts from the LRU tail until under budget, if over budget, never evicting the blob just used.
func (c *BlobRedrawCoordinator) internalCheckPurge(blob *Blob) {
	if c.currentSize <= c.sizeBudget {
		return
	}
	lruBlob := c.tail
	for c.currentSize > c.sizeBudget && lruBlob != nil && lruBlob != blob {
		prev := lruBlob.prev
		c.internalRemove(lruBlob)
		lruBlob = prev
	}
}

// FreeAll clears the entire cache.
func (c *BlobRedrawCoordinator) FreeAll() {
	c.mu.Lock()
	c.blobIDCache = make(map[uint32][]*Blob)
	c.head = nil
	c.tail = nil
	c.currentSize = 0
	c.mu.Unlock()
}

// UsedBytes returns the cache's current size in bytes.
func (c *BlobRedrawCoordinator) UsedBytes() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.currentSize
}

// IsOverBudget reports whether the cache currently exceeds its size budget.
func (c *BlobRedrawCoordinator) IsOverBudget() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.currentSize > c.sizeBudget
}

func (c *BlobRedrawCoordinator) attachToHead(blob *Blob) {
	blob.prev = nil
	blob.next = c.head
	if c.head != nil {
		c.head.prev = blob
	}
	c.head = blob
	if c.tail == nil {
		c.tail = blob
	}
}

func (c *BlobRedrawCoordinator) detach(blob *Blob) {
	if blob.prev != nil {
		blob.prev.next = blob.next
	} else if c.head == blob {
		c.head = blob.next
	}
	if blob.next != nil {
		blob.next.prev = blob.prev
	} else if c.tail == blob {
		c.tail = blob.prev
	}
	blob.prev = nil
	blob.next = nil
}
