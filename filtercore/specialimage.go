// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// SpecialImage: a view of a subset of a backing store, used as filter inputs/outputs so that intermediate renders can
// share buffers. The filter pipeline runs on N32 premul only (everything <= 32bpp upgrades to N32, and our surfaces are
// N32). The backing store is polymorphic (the DrawableImage pattern): raster-backed views hold imagecore.Pixels in
// RGBA8888 and drawable-backed views hold an imagecore.DrawableImage — in practice the GPU backend's texture-backed
// image — resolving to CPU pixels only if a raster consumer demands them (a readback, counted by ResolveToRasterCount
// so the visibility contract survives the GPU lane).

package filtercore

import (
	"sync"
	"sync/atomic"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/shaders"
)

var nextSpecialImageID atomic.Uint32

// resolveToRasterCount counts drawable→raster backing resolutions (GPU→CPU readbacks when the backing is
// texture-backed). Zero across a GPU-native filter evaluation is the discriminator that no hidden readback ran.
var resolveToRasterCount atomic.Int64

// ResolveToRasterCount returns the running count of drawable-backed SpecialImage resolutions to CPU pixels — the
// per-image readback companion to the GPU device's evaluation-level counter.
func ResolveToRasterCount() int64 { return resolveToRasterCount.Load() }

// drawableResolve caches the CPU resolution of a drawable backing store; subset views share it so the backing is read
// back at most once.
type drawableResolve struct {
	px   *imagecore.Pixels
	once sync.Once
}

// subsetDrawable is implemented by drawable backings that can produce a standalone image of a subset without CPU
// resolution (the GPU TextureImage's texture copy).
type subsetDrawable interface {
	MakeSubsetDrawable(subset geom.IRect) imagecore.DrawableImage
}

// SpecialImage is a subset view over a raster or drawable backing store.
type SpecialImage struct {
	pixels   *imagecore.Pixels       // raster backing (nil when drawable-backed)
	drawable imagecore.DrawableImage // drawable backing (nil when raster-backed)
	resolve  *drawableResolve        // shared lazy CPU resolution of the drawable backing
	subset   geom.IRect              // within the backing store's bounds
	id       uint32                  // generation ID for cache keys
}

// NewSpecialImage wraps the subset of pixels as a SpecialImage: subset must be contained in the pixel bounds. Returns
// nil for an empty subset or invalid subset.
func NewSpecialImage(subset geom.IRect, pixels *imagecore.Pixels) *SpecialImage {
	if pixels == nil || subset.IsEmpty() ||
		!geom.IRectWH(pixels.Info.Width, pixels.Info.Height).ContainsRect(subset) {
		return nil
	}
	return &SpecialImage{pixels: pixels, subset: subset, id: nextSpecialImageID.Add(1)}
}

// NewSpecialImageDrawable wraps the subset of a drawable image as a SpecialImage. A raster imagecore.Image backing
// downgrades to the raster lane (NewSpecialImage) so pure-CPU pipelines never pay the polymorphic dispatch.
func NewSpecialImageDrawable(subset geom.IRect, img imagecore.DrawableImage) *SpecialImage {
	if img == nil {
		return nil
	}
	if ci, ok := img.(*imagecore.Image); ok {
		px := ci.PeekPixels(imagecore.CachingAllow)
		if px == nil {
			return nil
		}
		return NewSpecialImage(subset, px)
	}
	if subset.IsEmpty() || !geom.IRectWH(img.Width(), img.Height()).ContainsRect(subset) {
		return nil
	}
	return &SpecialImage{
		drawable: img, resolve: &drawableResolve{}, subset: subset,
		id: nextSpecialImageID.Add(1),
	}
}

// makeView returns a new view of the same backing store with the given absolute subset, or nil if the subset leaves the
// backing bounds (matching the constructors).
func (s *SpecialImage) makeView(absolute geom.IRect) *SpecialImage {
	if s.pixels != nil {
		return NewSpecialImage(absolute, s.pixels)
	}
	if absolute.IsEmpty() ||
		!geom.IRectWH(s.drawable.Width(), s.drawable.Height()).ContainsRect(absolute) {
		return nil
	}
	return &SpecialImage{
		drawable: s.drawable, resolve: s.resolve, subset: absolute,
		id: nextSpecialImageID.Add(1),
	}
}

// Width returns the subset width.
func (s *SpecialImage) Width() int32 { return s.subset.Width() }

// Height returns the subset height.
func (s *SpecialImage) Height() int32 { return s.subset.Height() }

// Dimensions returns the subset dimensions.
func (s *SpecialImage) Dimensions() geom.ISize { return s.subset.Size() }

// Subset returns the view bounds relative to the backing store.
func (s *SpecialImage) Subset() geom.IRect { return s.subset }

// BackingStoreDimensions returns the full backing store's dimensions.
func (s *SpecialImage) BackingStoreDimensions() geom.ISize {
	if s.pixels != nil {
		return geom.ISize{Width: s.pixels.Info.Width, Height: s.pixels.Info.Height}
	}
	return geom.ISize{Width: s.drawable.Width(), Height: s.drawable.Height()}
}

// UniqueID returns the image's generation ID, for cache keys.
func (s *SpecialImage) UniqueID() uint32 { return s.id }

// DrawableBacking returns the drawable backing store (the GPU texture image in practice), or nil for a raster-backed
// view. The GPU device's drawSpecial lane samples it directly, avoiding the upload.
func (s *SpecialImage) DrawableBacking() imagecore.DrawableImage { return s.drawable }

// MakeSubset returns a view over subset, relative to this image's logical bounds ((0,0) at the current subset's
// top-left), sharing the backing store.
func (s *SpecialImage) MakeSubset(subset geom.IRect) *SpecialImage {
	absolute := subset.Offset(s.subset.Left, s.subset.Top)
	return s.makeView(absolute)
}

// MakePixelOutset returns a view with the subset grown by one pixel into the backing store. Callers must know the
// padding exists (pixelBoundary tracking guarantees it).
func (s *SpecialImage) MakePixelOutset() *SpecialImage {
	return s.makeView(s.subset.Inset(-1, -1))
}

// backingPixels resolves the full backing store to CPU pixels: direct for raster backings, a counted, once-per-backing
// readback for drawable ones.
func (s *SpecialImage) backingPixels() *imagecore.Pixels {
	if s.pixels != nil {
		return s.pixels
	}
	s.resolve.once.Do(func() {
		resolveToRasterCount.Add(1)
		if img := s.drawable.MakeNonTextureImage(); img != nil {
			s.resolve.px = img.PeekPixels(imagecore.CachingAllow)
		}
	})
	return s.resolve.px
}

// subsetPixels returns the subset view of the backing store, or nil if a drawable backing failed to resolve.
func (s *SpecialImage) subsetPixels() *imagecore.Pixels {
	bp := s.backingPixels()
	if bp == nil {
		return nil
	}
	return bp.Subset(s.subset.Left, s.subset.Top, s.subset.Right, s.subset.Bottom)
}

// AsImage returns an image sharing the subset's pixels, resolving a drawable backing to CPU first.
func (s *SpecialImage) AsImage() *imagecore.Image {
	px := s.subsetPixels()
	if px == nil {
		return nil
	}
	return imagecore.FromPixels(px)
}

// AsShader returns a shader sampling this image. For strict sampling the shader wraps the subset view directly, so
// tiling applies at the subset bounds; otherwise it wraps the full backing store with the subset origin folded into the
// local matrix, so tiling and filtering can then reach backing-store pixels outside the subset. A drawable backing
// stays drawable — the strict + proper-subset case takes the backing's subset copy when it offers one (the GPU
// texture-copy lane) and otherwise resolves to CPU (counted).
func (s *SpecialImage) AsShader(tileMode shaders.TileMode, sampling shaders.SamplingOptions, lm *geom.Matrix, strict bool) shaders.Shader {
	if strict {
		if s.drawable != nil {
			if s.subset == geom.IRectWH(s.drawable.Width(), s.drawable.Height()) {
				return shaders.NewImageDrawable(s.drawable, tileMode, tileMode, sampling, lm)
			}
			if sd, ok := s.drawable.(subsetDrawable); ok {
				if sub := sd.MakeSubsetDrawable(s.subset); sub != nil {
					return shaders.NewImageDrawable(sub, tileMode, tileMode, sampling, lm)
				}
			}
		}
		img := s.AsImage()
		if img == nil {
			return nil
		}
		return shaders.NewImage(img, tileMode, tileMode, sampling, lm)
	}
	// The special image's logical (0,0) is at its subset's topLeft(), so account for that in the local matrix used when
	// sampling.
	var subsetOrigin geom.Matrix
	subsetOrigin.SetTranslate(float32(-s.subset.Left), float32(-s.subset.Top))
	subsetOrigin.PostConcat(lm)
	if s.drawable != nil {
		return shaders.NewImageDrawable(s.drawable, tileMode, tileMode, sampling, &subsetOrigin)
	}
	return shaders.NewImage(imagecore.FromPixels(s.pixels), tileMode, tileMode, sampling, &subsetOrigin)
}
