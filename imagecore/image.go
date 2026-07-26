// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// Package imagecore holds the CPU image types: immutable raster pixel snapshots, deferred-decode images over the codec
// registry, pixel read-back with conversion, and the pixel containers the image shader and the sprite/legacy sampling
// lanes consume.
package imagecore

import (
	"sync"
	"sync/atomic"

	"github.com/richardwilkes/canvas/raster"
)

// CachingHint controls whether a lazy decode's result is cached on the Image.
type CachingHint int32

// CachingHint values.
const (
	CachingAllow CachingHint = iota
	CachingDisallow
)

// Codec is one entry of the decode registry: image formats register through the codecs package, and without a
// registered codec NewFromEncoded finds none and returns nil.
type Codec struct {
	// Matches sniffs the encoded header to detect the format.
	Matches func(data []byte) bool
	// DecodeInfo parses just the header into the image's reported info (with any unpremul→premul upgrade already
	// applied). ok=false when the header is invalid.
	DecodeInfo func(data []byte) (ImageInfo, bool)
	// Decode produces the pixels in exactly the info DecodeInfo reported.
	Decode func(data []byte) (*Pixels, error)
	// Name is the format name ("png", "jpeg", ...).
	Name string
}

var (
	codecMu sync.RWMutex
	codecs  []Codec
)

// RegisterCodec adds c to the decode registry, replacing any existing codec with the same Name.
func RegisterCodec(c Codec) {
	codecMu.Lock()
	defer codecMu.Unlock()
	for i := range codecs {
		if codecs[i].Name == c.Name {
			codecs[i] = c
			return
		}
	}
	codecs = append(codecs, c)
}

// findCodec returns the first registered codec whose sniffer accepts data.
func findCodec(data []byte) *Codec {
	codecMu.RLock()
	defer codecMu.RUnlock()
	for i := range codecs {
		if codecs[i].Matches(data) {
			c := codecs[i]
			return &c
		}
	}
	return nil
}

// DrawableImage is the polymorphic image type the device draw boundary consumes: an image may be raster- or
// texture-backed. The device interface (canvas.Device.DrawImageRect) accepts this so a GPU device can draw a
// texture-backed image natively, while raster/PDF devices rasterize. It exposes the queries the draw pipeline needs
// plus MakeNonTextureImage, which yields CPU pixels — the identity for a raster image, a GPU→CPU readback for a
// texture-backed one — for the devices that consume pixels; a GPU device instead type-asserts the concrete texture
// image to obtain its texture view. *Image satisfies it (MakeNonTextureImage returns the receiver); gpu/gl.TextureImage
// satisfies it via a readback.
type DrawableImage interface {
	Width() int32
	Height() int32
	AlphaType() AlphaType
	IsAlphaOnly() bool
	UniqueID() uint32
	// MakeNonTextureImage returns a CPU raster image (identity for a raster image, a readback for a texture-backed
	// one), or nil on a failed readback.
	MakeNonTextureImage() *Image
}

// Image is the CPU-backed image type: either raster (pix non-nil) or deferred-decode (encoded non-nil). Immutable after
// creation.
type Image struct {
	pix     *Pixels
	codec   *Codec
	cached  *Pixels
	encoded []byte
	info    ImageInfo
	// lazy decode state, reduced to one cache entry — the image itself is the key, and only full decodes are cached
	mu sync.Mutex
	// uniqueID is lazily assigned (the GPU texture cache keys on it).
	uniqueID atomic.Uint32
	decodeNG bool
}

var nextImageUniqueID atomic.Uint32

// UniqueID returns a non-zero value unique across images in the process.
func (im *Image) UniqueID() uint32 {
	if id := im.uniqueID.Load(); id != 0 {
		return id
	}
	// Racing assignments are resolved by CompareAndSwap; the loser re-reads the winner's ID.
	id := nextImageUniqueID.Add(1)
	if im.uniqueID.CompareAndSwap(0, id) {
		return id
	}
	return im.uniqueID.Load()
}

// FromPixmap wraps pix (which the caller promises not to mutate) as an N32-premul image. Surface snapshots use this to
// share the surface's backing store until the surface's next draw copies it away.
func FromPixmap(pix raster.Pixmap) *Image {
	p := PixelsFromPixmap(pix)
	return &Image{pix: p, info: p.Info}
}

// Copy returns an N32-premul image holding a deep copy of pix.
func Copy(pix *raster.Pixmap) *Image {
	dup := make([]uint32, len(pix.Pix))
	copy(dup, pix.Pix)
	return FromPixmap(raster.Pixmap{Width: pix.Width, Height: pix.Height, RowPixels: pix.RowPixels, Pix: dup})
}

// FromPixels wraps decoded/constructed pixels as an image.
func FromPixels(p *Pixels) *Image {
	return &Image{pix: p, info: p.Info}
}

// SetFromPixels re-initializes im to wrap p as a raster image, resetting its lazy-decode and unique-ID state so it is
// observationally identical to a fresh FromPixels(p). It lets a caller reuse one Image value across draws (the CPU
// image-draw scratch) instead of allocating a fresh one per draw. Only for transient scratch images consumed
// synchronously within a single draw and never cached across draws (a reset unique ID would otherwise alias a
// persistent cache entry).
func (im *Image) SetFromPixels(p *Pixels) {
	im.pix = p
	im.encoded = nil
	im.codec = nil
	im.info = p.Info
	im.uniqueID.Store(0)
	im.cached = nil
	im.decodeNG = false
}

// ClearScratch drops im's pixel/decode references so a pooled scratch image does not pin its source pixels between
// draws. After ClearScratch, im must be re-initialized with SetFromPixels before use.
func (im *Image) ClearScratch() {
	im.pix = nil
	im.encoded = nil
	im.codec = nil
	im.info = ImageInfo{}
	im.cached = nil
	im.decodeNG = false
}

// NewRasterData wraps a copy of the caller's pixel bytes as a raster image. Returns nil when the info is invalid, the
// color type is outside the support matrix, or the buffer/rowBytes cannot hold the pixels.
func NewRasterData(info ImageInfo, data []byte, rowBytes int) *Image {
	if !info.IsValid() || !info.ColorType.Supported() {
		return nil
	}
	p, ok := NewPixelsCopyFromBytes(info, data, rowBytes)
	if !ok {
		return nil
	}
	return &Image{pix: p, info: info}
}

// NewFromEncoded sniffs a registered codec, parses the header for the image info, and defers pixel decoding to first
// use. Returns nil when no codec accepts the data (including when no codecs have been registered).
func NewFromEncoded(data []byte) *Image {
	if len(data) == 0 {
		return nil
	}
	c := findCodec(data)
	if c == nil {
		return nil
	}
	info, ok := c.DecodeInfo(data)
	if !ok {
		return nil
	}
	return &Image{encoded: data, codec: c, info: info}
}

// Info returns the image's reported info.
func (im *Image) Info() ImageInfo { return im.info }

// Width returns the image width in pixels.
func (im *Image) Width() int32 { return im.info.Width }

// Height returns the image height in pixels.
func (im *Image) Height() int32 { return im.info.Height }

// AlphaType returns the image's alpha type.
func (im *Image) AlphaType() AlphaType { return im.info.AlphaType }

// ColorType returns the image's color type.
func (im *Image) ColorType() ColorType { return im.info.ColorType }

// IsOpaque reports whether the image's alpha type is always fully opaque.
func (im *Image) IsOpaque() bool { return im.info.IsOpaque() }

// IsAlphaOnly reports whether the image's color type carries only an alpha channel.
func (im *Image) IsAlphaOnly() bool { return im.info.ColorType.isAlphaOnly() }

// IsLazy reports whether the image is a deferred-decode image (not yet decoded to pixels).
func (im *Image) IsLazy() bool { return im.encoded != nil }

// MakeNonTextureImage returns a CPU raster image; for a CPU-backed Image this is the identity.
func (im *Image) MakeNonTextureImage() *Image { return im }

// PeekPixels returns the image's pixels: raster images return their storage; lazy images decode (caching the result
// only under CachingAllow). Returns nil when decoding fails.
func (im *Image) PeekPixels(hint CachingHint) *Pixels {
	if im.pix != nil {
		return im.pix
	}
	im.mu.Lock()
	defer im.mu.Unlock()
	if im.cached != nil {
		return im.cached
	}
	if im.decodeNG {
		return nil
	}
	p, err := im.codec.Decode(im.encoded)
	if err != nil || p == nil {
		im.decodeNG = true
		return nil
	}
	if hint == CachingAllow {
		im.cached = p
	}
	return p
}

// Pixmap returns a read-only N32 view of the image's pixels for the sprite/snapshot consumers that predate the
// color-type matrix; ok=false for non-N32 or failed-decode images.
func (im *Image) Pixmap() *raster.Pixmap {
	p := im.PeekPixels(CachingAllow)
	if p == nil {
		return nil
	}
	if pm, ok := p.PixmapView(); ok {
		return &pm
	}
	return nil
}

// ReadPixels converts the pixel rect starting at (srcX, srcY) sized by dstInfo into dst. The read rect is trimmed to
// the image bounds with the dst pointer adjusted accordingly.
func (im *Image) ReadPixels(dstInfo ImageInfo, dst []byte, dstRowBytes int, srcX, srcY int32, hint CachingHint) bool {
	src := im.PeekPixels(hint)
	if src == nil || dst == nil {
		return false
	}
	if !ValidConversion(&dstInfo, &im.info) {
		return false
	}
	if dstRowBytes < dstInfo.MinRowBytes() {
		return false
	}

	// Trim the read rect to the image bounds, accumulating the dst pointer adjustment but deferring the reslice until
	// the trimmed rect is known to be non-empty. A negative srcX/srcY beyond the destination extent leaves w/h <= 0, so
	// applying the offset here would slice past len(dst) and panic instead of returning false.
	x, y := srcX, srcY
	w, h := dstInfo.Width, dstInfo.Height
	var dstOff int
	if x < 0 {
		w += x
		dstOff += int(-x) * dstInfo.ColorType.BytesPerPixel()
		x = 0
	}
	if y < 0 {
		h += y
		dstOff += int(-y) * dstRowBytes
		y = 0
	}
	// x and y are non-negative here, so widen the far edges to int64: a large positive srcX/srcY would otherwise wrap
	// the int32 sum negative, skip the clamp entirely, and leave an out-of-range rect for Subset to slice.
	if int64(x)+int64(w) > int64(im.info.Width) {
		w = im.info.Width - x
	}
	if int64(y)+int64(h) > int64(im.info.Height) {
		h = im.info.Height - y
	}
	if w <= 0 || h <= 0 {
		return false
	}
	dst = dst[dstOff:]

	sub := src.Subset(x, y, x+w, y+h)
	outInfo := dstInfo
	outInfo.Width = w
	outInfo.Height = h
	return ConvertPixels(outInfo, dst, dstRowBytes, sub)
}
