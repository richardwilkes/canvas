// Copyright (c) 2026 by Richard A. Wilkes. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, version 2.0. If a copy of the MPL was not distributed with
// this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is "Incompatible With Secondary Licenses", as
// defined by the Mozilla Public License, version 2.0.

// The legacy fixed-point bitmap-processing shader: the fixed-point sampling machine that routes src-over, non-dithered
// N32 image-shader draws — the dominant CPU image-draw lane, so its 16.16/32.32 fixed-point math and 4-bit bilerp
// weights are exact (the float pipeline would differ visibly at sharp edges). The sample kernels match both the
// portable and SIMD forms, which compute identical integer results.

package shaders

import (
	"math"
	"sync"

	"github.com/richardwilkes/canvas/geom"
	"github.com/richardwilkes/canvas/imagecore"
	"github.com/richardwilkes/canvas/raster"
)

// fixed-point types: "fixed" values are 16.16 in int32; "fixed3232" values are 32.32 in int64.

const fixed1 = 1 << 16

func fixed3232ToInt(x int64) int32   { return int32(x >> 32) }
func fixed3232ToFixed(x int64) int32 { return int32(x >> 16) }
func fixedToFixed3232(x int32) int64 { return int64(x) << 16 }
func intToFixed3232(x int32) int64   { return int64(x) << 32 }
func fixedToFloat(x int32) float32   { return float32(x) * (1.0 / 65536.0) }
func fixedFloorToInt(x int32) int32  { return x >> 16 }

// floatSaturate2Int64 converts x to int64, saturating out-of-range values explicitly (Go's float→int conversion is
// implementation-specific out of range).
func floatSaturate2Int64(x float32) int64 {
	switch {
	case math.IsNaN(float64(x)):
		return 0 // hardware fcvtzs of NaN
	case x >= 9.223372036854775807e18:
		return math.MaxInt64
	case x <= -9.223372036854775808e18:
		return math.MinInt64
	default:
		return int64(x)
	}
}

// scalarToFixed3232 converts a float32 to the 32.32 fixed-point representation.
func scalarToFixed3232(x float32) int64 {
	return floatSaturate2Int64(x * (65536.0 * 65536.0))
}

// bitmapProcState holds the legacy sampling state for the N32 pixmaps the legacy lane accepts.
type bitmapProcState struct {
	shaderProc func(s *bitmapProcState, x, y int32, dst []uint32)
	sampleProc func(s *bitmapProcState, xy []uint32, count int, dst []uint32)
	matrixProc func(s *bitmapProcState, xy []uint32, count int, x, y int32)
	// tile/extract/tryDecal are the template parameters the general matrixProcs would otherwise close over; storing
	// them on the state (set once in chooseMatrixProc) keeps matrixProc a static top-level func so choosing the blitter
	// allocates no per-draw closure.
	tile       tileProc
	extract    func(fx, maxV int32) uint32
	pixmap     raster.Pixmap
	invSx      int64 // fixed3232
	invKy      int64 // fixed3232
	invMatrix  geom.Matrix
	filterOneX int32 // fixed
	filterOneY int32 // fixed
	alphaScale uint32
	bilerp     bool
	tryDecal   bool
	tileModeY  TileMode
	tileModeX  TileMode
}

// autoMapper maps a destination pixel through the inverse matrix into source space, applying the bias needed for
// correct rounding.
type autoMapper struct {
	fx, fy int64 // fixed3232
	sx, sy float32
}

func newAutoMapper(s *bitmapProcState, x, y int32) autoMapper {
	pt := s.invMatrix.MapXY(float32(x)+0.5, float32(y)+0.5)
	var biasX, biasY int32
	if s.bilerp {
		biasX = s.filterOneX >> 1
		biasY = s.filterOneY >> 1
	} else {
		// Make exact integer pixel sample values round down not up (the rasterizer biases upward).
		biasX = 1
		biasY = 1
	}
	// punt to unsigned for defined underflow behavior (Go's int64 arithmetic wraps anyway)
	fx := int64(uint64(scalarToFixed3232(pt.X)) - uint64(fixedToFixed3232(biasX)))
	fy := int64(uint64(scalarToFixed3232(pt.Y)) - uint64(fixedToFixed3232(biasY)))
	return autoMapper{
		fx: fx, fy: fy,
		sx: pt.X - fixedToFloat(biasX),
		sy: pt.Y - fixedToFloat(biasY),
	}
}

func (a *autoMapper) fixedY() int32 { return fixed3232ToFixed(a.fy) }
func (a *autoMapper) intX() int32   { return fixed3232ToInt(a.fx) }
func (a *autoMapper) intY() int32   { return fixed3232ToInt(a.fy) }

// row returns the pixel row y of the pixmap.
func (s *bitmapProcState) row(y int32) []uint32 {
	off := int(y) * int(s.pixmap.RowPixels)
	return s.pixmap.Pix[off:]
}

///////////////////////////////////////////////////////////////////////////////
// init + chooseProcs

// matrixOnlyScaleTranslate reports whether m is at most a scale plus translate.
func matrixOnlyScaleTranslate(m *geom.Matrix) bool {
	return m.Type()&^uint8(geom.TypeTranslate) == geom.TypeScale
}

// justTransGeneral reports whether m is close enough to a pure translation for the translate-only fast path.
func justTransGeneral(m *geom.Matrix) bool {
	const tol = 1.0 / 32768
	return absf(m.Get(geom.MScaleX)-1) <= tol && absf(m.Get(geom.MScaleY)-1) <= tol
}

// justTransIntegral reports whether m is a pure translation by an (approximately) integral amount.
func justTransIntegral(m *geom.Matrix) bool {
	const tol = 1.0 / 256
	if m.Type() > geom.TypeTranslate {
		return false
	}
	tx := m.Get(geom.MTransX)
	ty := m.Get(geom.MTransY)
	return absf(tx-floorf(tx+0.5)) <= tol && absf(ty-floorf(ty+0.5)) <= tol
}

// validForFiltering reports whether dimension fits in the 14 bits the fixed-point sampler's packed coordinates require.
func validForFiltering(dimension int32) bool {
	return dimension&^0x3FFF == 0
}

// postIDiv divides the scale/skew/translate elements of m element-wise by (divx, divy).
func postIDiv(m *geom.Matrix, divx, divy int32) {
	invX := 1.0 / float32(divx)
	invY := 1.0 / float32(divy)
	m.SetAll(
		m.Get(geom.MScaleX)*invX, m.Get(geom.MSkewX)*invX, m.Get(geom.MTransX)*invX,
		m.Get(geom.MSkewY)*invY, m.Get(geom.MScaleY)*invY, m.Get(geom.MTransY)*invY,
		m.Get(6), m.Get(7), m.Get(8),
	)
}

// setup initializes the sampling state and chooses the matrix/sample/shader procs for a draw, with the accessor reduced
// to the base level (there is no mipmap support in the CPU sampler).
func (s *bitmapProcState) setup(pm raster.Pixmap, inv *geom.Matrix, paintAlpha uint8, sampling SamplingOptions, tmx, tmy TileMode) bool {
	s.pixmap = pm
	s.tileModeX = tmx
	s.tileModeY = tmy
	s.invMatrix = *inv
	s.bilerp = sampling.Filter == FilterLinear

	integralTranslateOnly := justTransIntegral(&s.invMatrix)
	if !integralTranslateOnly {
		if s.tileModeX != TileClamp || s.tileModeY != TileClamp {
			postIDiv(&s.invMatrix, pm.Width, pm.Height)
		}
		if matrixOnlyScaleTranslate(&s.invMatrix) {
			if forward, ok := s.invMatrix.Invert(); ok && justTransGeneral(&forward) {
				s.invMatrix.SetTranslate(-forward.Get(geom.MTransX), -forward.Get(geom.MTransY))
			}
		}
		integralTranslateOnly = justTransIntegral(&s.invMatrix)
	}

	if s.bilerp && (!validForFiltering(pm.Width|pm.Height) || integralTranslateOnly) {
		s.bilerp = false
	}

	// chooseProcs. shaderProc is only conditionally assigned below, so clear it up front: a pooled context reused for a
	// scale/affine draw must not inherit a translate-fast-path proc from a prior use (matrixProc/sampleProc are always
	// overwritten on the success path, so they need no reset).
	s.shaderProc = nil
	s.invSx = scalarToFixed3232(s.invMatrix.Get(geom.MScaleX))
	s.invKy = scalarToFixed3232(s.invMatrix.Get(geom.MSkewY))
	s.alphaScale = alpha255To256(uint32(paintAlpha))

	translateOnly := s.invMatrix.Type()&^uint8(geom.TypeTranslate) == 0
	if !s.chooseMatrixProc(translateOnly) {
		return false
	}

	if s.invMatrix.IsScaleTranslate() {
		if s.bilerp {
			s.sampleProc = s32AlphaD32FilterDX
		} else {
			s.sampleProc = s32AlphaD32NofilterDX
		}
	} else {
		if s.bilerp {
			s.sampleProc = s32AlphaD32FilterDXDY
		} else {
			s.sampleProc = s32AlphaD32NofilterDXDY
		}
	}

	if s.alphaScale == 256 && !s.bilerp && s.tileModeX == TileClamp && s.tileModeY == TileClamp &&
		s.invMatrix.IsScaleTranslate() {
		s.shaderProc = clampS32OpaqueD32NofilterDXShaderproc
	} else {
		s.chooseShaderProc()
	}
	return true
}

///////////////////////////////////////////////////////////////////////////////
// tile procs

type tileProc func(fx, maxV int32) uint32

func tileClampProc(fx, maxV int32) uint32 {
	v := fx >> 16
	if v < 0 {
		v = 0
	} else if v > maxV {
		v = maxV
	}
	return uint32(v)
}

func tileRepeatProc(fx, maxV int32) uint32 {
	return uint32(fx&0xFFFF) * uint32(maxV+1) >> 16
}

func tileMirrorProc(fx, maxV int32) uint32 {
	// s is all ones on odd intervals, 0 on even.
	s := (fx << 15) >> 31
	return uint32((fx^s)&0xFFFF) * uint32(maxV+1) >> 16
}

// extract_low_bits variants: the 4-bit lerp weight.
func extractLowBitsClampClamp(fx, _ int32) uint32 {
	return uint32(fx>>12) & 0xF
}

func extractLowBitsGeneral(fx, maxV int32) uint32 {
	return extractLowBitsClampClamp((fx&0xFFFF)*(maxV+1), maxV)
}

// packFilter packs a bilerp sample into 14-bit low coord, 4-bit weight, 14-bit high coord.
func packFilter(tile tileProc, extract func(fx, maxV int32) uint32, f, maxV, one int32) uint32 {
	packed := tile(f, maxV)
	packed = packed<<4 | extract(f, maxV)
	packed = packed<<14 | tile(f+one, maxV) // f+one wraps on overflow, which is fine here
	return packed
}

// u16 access into the xy buffer with the little-endian pack_two_shorts layout.
func putU16(xy []uint32, i int, v uint16) {
	w := &xy[i>>1]
	if i&1 == 0 {
		*w = *w&0xFFFF0000 | uint32(v)
	} else {
		*w = *w&0x0000FFFF | uint32(v)<<16
	}
}

func getU16(xy []uint32, i int) uint16 {
	if i&1 == 0 {
		return uint16(xy[i>>1])
	}
	return uint16(xy[i>>1] >> 16)
}

///////////////////////////////////////////////////////////////////////////////
// matrix procs

// canTruncateToFixedForDecal reports whether a decal (untiled) span can be sampled using truncated 16.16 fixed-point
// math without overflow or out-of-range access.
func canTruncateToFixedForDecal(fx, dx int32, count int, maxV uint32) bool {
	if dx <= fixed1/256 {
		return false
	}
	if uint32(fixedFloorToInt(fx)) >= maxV {
		return false
	}
	lastFx := int64(fx) + int64(dx)*int64(count-1)
	return lastFx == int64(int32(lastFx)) && uint32(fixedFloorToInt(int32(lastFx))) < maxV
}

// decalNofilterScale writes a decal (untiled), unfiltered, scaled sample sequence.
func decalNofilterScale(xy []uint32, base int, fx, dx int32, count int) {
	for i := range count {
		putU16(xy, base+i, uint16(fx>>16))
		fx += dx
	}
}

// nofilterScale writes an unfiltered, scaled sample sequence using the given tile mode, with a decal fast path when
// tryDecal is set.
func nofilterScale(s *bitmapProcState, tile tileProc, tryDecal bool, xy []uint32, count int, x, y int32) {
	var fx int64
	{
		mapper := newAutoMapper(s, x, y)
		xy[0] = tile(mapper.fixedY(), s.pixmap.Height-1)
		fx = mapper.fx
	}
	maxX := s.pixmap.Width - 1
	if maxX == 0 {
		for i := range count {
			putU16(xy, 2+i, 0)
		}
		return
	}
	dx := s.invSx

	if tryDecal {
		fixedFx := fixed3232ToFixed(fx)
		fixedDx := fixed3232ToFixed(dx)
		if canTruncateToFixedForDecal(fixedFx, fixedDx, count, uint32(maxX)) {
			decalNofilterScale(xy, 2, fixedFx, fixedDx, count)
			return
		}
	}

	for i := range count {
		putU16(xy, 2+i, uint16(tile(fixed3232ToFixed(fx), maxX)))
		fx += dx
	}
}

// nofilterAffine writes an unfiltered sample sequence for a general affine (non-scale-translate) matrix.
func nofilterAffine(s *bitmapProcState, tile tileProc, xy []uint32, count int, x, y int32) {
	mapper := newAutoMapper(s, x, y)
	fx, fy := mapper.fx, mapper.fy
	dx, dy := s.invSx, s.invKy
	maxX, maxY := s.pixmap.Width-1, s.pixmap.Height-1
	for i := range count {
		xy[i] = tile(fixed3232ToFixed(fy), maxY)<<16 | tile(fixed3232ToFixed(fx), maxX)
		fx += dx
		fy += dy
	}
}

// fixed3232Saturate2Fixed converts a 32.32 fixed-point value to 16.16, saturating on overflow.
func fixed3232Saturate2Fixed(x int64) int32 {
	const maxFx = int64(math.MaxInt16) << 32
	minFx := intToFixed3232(int32(-0x8000))
	if x>>32 >= math.MaxInt16 {
		x = maxFx
	}
	if x>>32 <= math.MinInt16 {
		x = minFx
	}
	return fixed3232ToFixed(x)
}

// filterScale writes a bilerp-filtered, scaled sample sequence using the given tile and weight-extraction helpers, with
// a decal fast path when tryDecal is set.
func filterScale(s *bitmapProcState, tile tileProc, extract func(fx, maxV int32) uint32, tryDecal bool, xy []uint32, count int, x, y int32) {
	maxX := s.pixmap.Width - 1
	dx := s.invSx
	var fx int64
	{
		mapper := newAutoMapper(s, x, y)
		maxY := s.pixmap.Height - 1
		xy[0] = packFilter(tile, extract, fixed3232Saturate2Fixed(mapper.fy), maxY, s.filterOneY)
		fx = mapper.fx
	}

	// For historical reasons both ends are checked with < maxX rather than <=.
	if tryDecal &&
		uint32(fixed3232ToInt(fx)) < uint32(maxX) &&
		uint32(fixed3232ToInt(fx+dx*int64(count-1))) < uint32(maxX) {
		for i := range count {
			fixedFx := fixed3232Saturate2Fixed(fx)
			xy[1+i] = uint32(fixedFx>>12)<<14 | (uint32(fixedFx>>16) + 1)
			fx += dx
		}
		return
	}

	for i := range count {
		xy[1+i] = packFilter(tile, extract, fixed3232Saturate2Fixed(fx), maxX, s.filterOneX)
		fx += dx
	}
}

// filterScaleClamp is filterScale's clamp-clamp instantiation (the dominant tile-mode combination in practice),
// specialized so the tile/extract helpers are direct — and thus inlinable — calls instead of the two per-pixel indirect
// calls the general proc pays. Pixels whose (bias-adjusted) source coordinate lands strictly inside [0, maxX) take the
// same branch-free pack the whole-span decal lane writes; the edge pixels fall back to packFilter with the clamp
// helpers, so every output word is bit-identical to the general proc's. Byte-exactness of the in-range pack: for 0 <= f
// with f>>16 < maxX, tileClampProc(f) = f>>16, the weight is (f>>12)&0xF, and tileClampProc(f+fixed1) = (f>>16)+1, so
// packFilter's three fields collapse to (f>>12)<<14 | ((f>>16)+1) exactly as decal writes them (filterOneX is fixed1 in
// clamp mode).
func filterScaleClamp(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	maxX := s.pixmap.Width - 1
	dx := s.invSx
	var fx int64
	{
		mapper := newAutoMapper(s, x, y)
		maxY := s.pixmap.Height - 1
		xy[0] = packFilter(tileClampProc, extractLowBitsClampClamp,
			fixed3232Saturate2Fixed(mapper.fy), maxY, s.filterOneY)
		fx = mapper.fx
	}
	for i := range count {
		f := fixed3232Saturate2Fixed(fx)
		if uint32(f>>16) < uint32(maxX) {
			xy[1+i] = uint32(f>>12)<<14 | (uint32(f>>16) + 1)
		} else {
			xy[1+i] = packFilter(tileClampProc, extractLowBitsClampClamp, f, maxX, s.filterOneX)
		}
		fx += dx
	}
}

// nofilterScaleClamp is nofilterScale's clamp-clamp instantiation, specialized like filterScaleClamp so the per-pixel
// tile call is direct (inlinable) rather than indirect. The decal preflight and the fallback loop are the general
// proc's exact code with tileClampProc substituted, so the output is bit-identical.
func nofilterScaleClamp(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	var fx int64
	{
		mapper := newAutoMapper(s, x, y)
		xy[0] = tileClampProc(mapper.fixedY(), s.pixmap.Height-1)
		fx = mapper.fx
	}
	maxX := s.pixmap.Width - 1
	if maxX == 0 {
		for i := range count {
			putU16(xy, 2+i, 0)
		}
		return
	}
	dx := s.invSx

	fixedFx := fixed3232ToFixed(fx)
	fixedDx := fixed3232ToFixed(dx)
	if canTruncateToFixedForDecal(fixedFx, fixedDx, count, uint32(maxX)) {
		decalNofilterScale(xy, 2, fixedFx, fixedDx, count)
		return
	}

	for i := range count {
		putU16(xy, 2+i, uint16(tileClampProc(fixed3232ToFixed(fx), maxX)))
		fx += dx
	}
}

// filterAffine writes a bilerp-filtered sample sequence for a general affine (non-scale-translate) matrix, using the
// given tile and weight-extraction helpers.
func filterAffine(s *bitmapProcState, tile tileProc, extract func(fx, maxV int32) uint32, xy []uint32, count int, x, y int32) {
	mapper := newAutoMapper(s, x, y)
	oneX, oneY := s.filterOneX, s.filterOneY
	fx, fy := mapper.fx, mapper.fy
	dx, dy := s.invSx, s.invKy
	maxX, maxY := s.pixmap.Width-1, s.pixmap.Height-1
	for i := range count {
		xy[2*i] = packFilter(tile, extract, fixed3232ToFixed(fy), maxY, oneY)
		xy[2*i+1] = packFilter(tile, extract, fixed3232ToFixed(fx), maxX, oneX)
		fy += dy
		fx += dx
	}
}

// integer tiling for the translate-only specializations
func intClamp(x, n int32) int32 {
	if x < 0 {
		x = 0
	}
	if x >= n {
		x = n - 1
	}
	return x
}

func intMod(x, n int32) int32 {
	if uint32(x) >= uint32(n) {
		if x < 0 {
			x = n + ^(^x % n)
		} else {
			x %= n
		}
	}
	return x
}

func intMirror(x, n int32) int32 {
	x = intMod(x, 2*n)
	if x >= n {
		x = n + ^(x - n)
	}
	return x
}

// clampxNofilterTrans writes an unfiltered, translate-only sample sequence with clamp tiling.
func clampxNofilterTrans(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	mapper := newAutoMapper(s, x, y)
	xy[0] = uint32(intClamp(mapper.intY(), s.pixmap.Height))
	xpos := mapper.intX()

	width := s.pixmap.Width
	if width == 1 {
		for i := range count {
			putU16(xy, 2+i, 0)
		}
		return
	}

	i := 0
	if xpos < 0 {
		n := int(-xpos)
		if n > count {
			n = count
		}
		for range n {
			putU16(xy, 2+i, 0)
			i++
		}
		count -= n
		if count == 0 {
			return
		}
		xpos = 0
	}
	if xpos < width {
		n := int(width - xpos)
		if n > count {
			n = count
		}
		for j := range n {
			putU16(xy, 2+i, uint16(xpos+int32(j)))
			i++
		}
		count -= n
	}
	for range count {
		putU16(xy, 2+i, uint16(width-1))
		i++
	}
}

// repeatxNofilterTrans writes an unfiltered, translate-only sample sequence with repeat tiling.
func repeatxNofilterTrans(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	mapper := newAutoMapper(s, x, y)
	xy[0] = uint32(intMod(mapper.intY(), s.pixmap.Height))
	xpos := mapper.intX()

	width := s.pixmap.Width
	if width == 1 {
		for i := range count {
			putU16(xy, 2+i, 0)
		}
		return
	}

	i := 0
	pos := intMod(xpos, width)
	for range count {
		putU16(xy, 2+i, uint16(pos))
		i++
		pos++
		if pos == width {
			pos = 0
		}
	}
}

// mirrorxNofilterTrans writes an unfiltered, translate-only sample sequence with mirror tiling.
func mirrorxNofilterTrans(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	mapper := newAutoMapper(s, x, y)
	xy[0] = uint32(intMirror(mapper.intY(), s.pixmap.Height))
	xpos := mapper.intX()

	width := s.pixmap.Width
	if width == 1 {
		for i := range count {
			putU16(xy, 2+i, 0)
		}
		return
	}

	i := 0
	start := intMod(xpos, 2*width)
	var forward bool
	if start >= width {
		start = width + ^(start - width)
		forward = false
	} else {
		forward = true
	}
	pos := start
	for range count {
		putU16(xy, 2+i, uint16(pos))
		i++
		if forward {
			pos++
			if pos == width {
				pos--
				forward = false
			}
		} else {
			pos--
			if pos < 0 {
				pos = 0
				forward = true
			}
		}
	}
}

// chooseMatrixProc selects the matrixProc (and its tile/extract/tryDecal parameters) appropriate for the current
// matrix, tile modes, and filter setting.
func (s *bitmapProcState) chooseMatrixProc(translateOnly bool) bool {
	if s.tileModeX != s.tileModeY {
		return false
	}

	if translateOnly && !s.bilerp {
		switch s.tileModeX {
		case TileClamp:
			s.matrixProc = clampxNofilterTrans
		case TileRepeat:
			s.matrixProc = repeatxNofilterTrans
		case TileMirror:
			s.matrixProc = mirrorxNofilterTrans
		default:
			return false
		}
		return true
	}

	s.tryDecal = false
	switch s.tileModeX {
	case TileClamp:
		// clamp gets a special filterOne, working in non-normalized space (allowing decal)
		s.filterOneX = fixed1
		s.filterOneY = fixed1
		s.tile = tileClampProc
		s.extract = extractLowBitsClampClamp
		s.tryDecal = true
	case TileRepeat, TileMirror:
		s.filterOneX = fixed1 / s.pixmap.Width
		s.filterOneY = fixed1 / s.pixmap.Height
		s.extract = extractLowBitsGeneral
		if s.tileModeX == TileRepeat {
			s.tile = tileRepeatProc
		} else {
			s.tile = tileMirrorProc
		}
	default:
		return false
	}

	scaleTranslate := s.invMatrix.IsScaleTranslate()
	bilerp := s.bilerp
	clamp := s.tileModeX == TileClamp
	switch {
	case scaleTranslate && !bilerp:
		if clamp {
			s.matrixProc = nofilterScaleClamp
		} else {
			s.matrixProc = nofilterScaleProc
		}
	case scaleTranslate && bilerp:
		if clamp {
			s.matrixProc = filterScaleClamp
		} else {
			s.matrixProc = filterScaleProc
		}
	case !scaleTranslate && !bilerp:
		s.matrixProc = nofilterAffineProc
	default:
		s.matrixProc = filterAffineProc
	}
	return true
}

// The general matrixProcs, static top-level funcs sourcing tile/extract/tryDecal from the state (set in
// chooseMatrixProc) rather than a closure capture — so assigning them allocates nothing.
func nofilterScaleProc(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	nofilterScale(s, s.tile, s.tryDecal, xy, count, x, y)
}

func filterScaleProc(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	filterScale(s, s.tile, s.extract, s.tryDecal, xy, count, x, y)
}

func nofilterAffineProc(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	nofilterAffine(s, s.tile, xy, count, x, y)
}

func filterAffineProc(s *bitmapProcState, xy []uint32, count int, x, y int32) {
	filterAffine(s, s.tile, s.extract, xy, count, x, y)
}

///////////////////////////////////////////////////////////////////////////////
// sample procs

// s32AlphaD32NofilterDX samples an N32 source without filtering, applying paint alpha, for a scale-translate matrix
// (varying x only per row).
func s32AlphaD32NofilterDX(s *bitmapProcState, xy []uint32, count int, dst []uint32) {
	row := s.row(int32(xy[0]))
	if s.pixmap.Width == 1 {
		c := alphaMulQLegacy(row[0], s.alphaScale)
		for i := range count {
			dst[i] = c
		}
		return
	}
	for i := range count {
		dst[i] = alphaMulQLegacy(row[getU16(xy, 2+i)], s.alphaScale)
	}
}

// s32AlphaD32NofilterDXDY samples an N32 source without filtering, applying paint alpha, for a general affine matrix
// (varying both x and y per pixel).
func s32AlphaD32NofilterDXDY(s *bitmapProcState, xy []uint32, count int, dst []uint32) {
	for i := range count {
		packed := xy[i]
		x := packed & 0xFFFF
		y := packed >> 16
		dst[i] = alphaMulQLegacy(s.row(int32(y))[x], s.alphaScale)
	}
}

// decodePackedCoordinatesAndWeight unpacks a packFilter word into its low coord, high coord, and weight.
func decodePackedCoordinatesAndWeight(packed uint32) (v0, v1 int32, w uint32) {
	return int32(packed >> 18), int32(packed & 0x3FFF), (packed >> 14) & 0xF
}

// swarMask8 selects the low byte of each 16-bit lane of a spread64 word.
const swarMask8 = uint64(0x00FF00FF00FF00FF)

// spread64 splits an 8888 pixel's four bytes into the four 16-bit lanes of a uint64 (lane order [b0, b2, b1, b3]), so
// per-channel byte arithmetic runs four channels per integer op — a pure-Go stand-in for widening into SIMD lanes. Each
// lane has 8 headroom bits, enough for a byte times a 0..256 scale (max 0xFF00) with no cross-lane carry.
func spread64(c uint32) uint64 {
	return (uint64(c) | uint64(c)<<24) & swarMask8
}

// pack64 packs the low byte of each 16-bit lane of a spread64-layout word back into an 8888 pixel. The caller must have
// reduced every lane to <= 0xFF.
func pack64(v uint64) uint32 {
	return uint32(v) | uint32(v>>24)
}

// filterAndScaleByAlpha computes a four-tap bilerp with paint alpha applied, bit-identical between the portable and
// SIMD forms. A prior kernel ran the four bilerp taps twice over uint32 lane pairs (bytes 0/2, then 1/3); this form
// runs all four channels per tap in one spread64 word — the per-lane products and sums are the same 16-bit values, so
// the result is bit-identical (the four tap scales sum to 256, bounding every lane by 0xFF00; the alpha step's lanes by
// 255*256).
func filterAndScaleByAlpha(x, y, a00, a01, a10, a11, alphaScale uint32) uint32 {
	xy := x * y
	acc := spread64(a00)*uint64(256-16*y-16*x+xy) +
		spread64(a01)*uint64(16*x-xy) +
		spread64(a10)*uint64(16*y-xy) +
		spread64(a11)*uint64(xy)
	if alphaScale < 256 {
		acc = ((acc >> 8) & swarMask8) * uint64(alphaScale)
	}
	return pack64((acc >> 8) & swarMask8)
}

// filterNoAlpha is filterAndScaleByAlpha's alphaScale==256 form with the per-pixel alpha branch hoisted out (the
// opaque-paint case the sample loops split on).
func filterNoAlpha(x, y, a00, a01, a10, a11 uint32) uint32 {
	xy := x * y
	acc := spread64(a00)*uint64(256-16*y-16*x+xy) +
		spread64(a01)*uint64(16*x-xy) +
		spread64(a10)*uint64(16*y-xy) +
		spread64(a11)*uint64(xy)
	return pack64((acc >> 8) & swarMask8)
}

// s32AlphaD32FilterDX samples an N32 source with bilerp filtering for a scale-translate matrix (varying x only per
// row), with the alphaScale==256 (opaque paint) loop split out so the common case runs the branch-free filterNoAlpha
// kernel.
func s32AlphaD32FilterDX(s *bitmapProcState, xy []uint32, count int, dst []uint32) {
	y0, y1, wy := decodePackedCoordinatesAndWeight(xy[0])
	row0 := s.row(y0)
	row1 := s.row(y1)
	if s.alphaScale == 256 {
		for i := range count {
			x0, x1, wx := decodePackedCoordinatesAndWeight(xy[1+i])
			dst[i] = filterNoAlpha(wx, wy, row0[x0], row0[x1], row1[x0], row1[x1])
		}
		return
	}
	for i := range count {
		x0, x1, wx := decodePackedCoordinatesAndWeight(xy[1+i])
		dst[i] = filterAndScaleByAlpha(wx, wy, row0[x0], row0[x1], row1[x0], row1[x1], s.alphaScale)
	}
}

// s32AlphaD32FilterDXDY samples an N32 source with bilerp filtering for a general affine matrix (varying both x and y
// per pixel).
func s32AlphaD32FilterDXDY(s *bitmapProcState, xy []uint32, count int, dst []uint32) {
	for i := range count {
		y0, y1, wy := decodePackedCoordinatesAndWeight(xy[2*i])
		x0, x1, wx := decodePackedCoordinatesAndWeight(xy[2*i+1])
		row0 := s.row(y0)
		row1 := s.row(y1)
		dst[i] = filterAndScaleByAlpha(wx, wy, row0[x0], row0[x1], row1[x0], row1[x1], s.alphaScale)
	}
}

// alphaMulQLegacy multiplies each channel of a premultiplied N32 pixel by scale/256.
func alphaMulQLegacy(c, scale uint32) uint32 {
	if scale == 256 {
		return c
	}
	const mask = 0x00FF00FF
	rb := ((c & mask) * scale) >> 8
	ag := ((c >> 8) & mask) * scale
	return (rb & mask) | (ag &^ mask)
}

///////////////////////////////////////////////////////////////////////////////
// shader procs (the fused specializations)

func tpin32(v, lo, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// clampS32OpaqueD32NofilterDXShaderproc directly shades a span for the fully-opaque, unfiltered, scale-translate,
// clamp-tiled fast path — the dominant case for opaque image draws.
func clampS32OpaqueD32NofilterDXShaderproc(s *bitmapProcState, x, y int32, dst []uint32) {
	maxX := s.pixmap.Width - 1
	var fx int64
	var dstY int32
	{
		mapper := newAutoMapper(s, x, y)
		dstY = tpin32(mapper.intY(), 0, s.pixmap.Height-1)
		fx = mapper.fx
	}
	row := s.row(dstY)
	dx := s.invSx
	count := len(dst)

	if uint64(fixed3232ToInt(fx)) <= uint64(maxX) &&
		uint64(fixed3232ToInt(fx+dx*int64(count-1))) <= uint64(maxX) {
		for i := range count {
			dst[i] = row[fixed3232ToInt(fx)]
			fx += dx
		}
	} else {
		for i := range count {
			dst[i] = row[tpin32(fixed3232ToInt(fx), 0, maxX)]
			fx += dx
		}
	}
}

// setupForTranslate precomputes the integer offsets used by the translate-only shader procs.
func (s *bitmapProcState) setupForTranslate() bool {
	mapper := newAutoMapper(s, 0, 0)
	const tooBig = float32(1 << 30)
	if absf(mapper.sx) > tooBig || absf(mapper.sy) > tooBig {
		return false
	}
	s.filterOneX = mapper.intX()
	s.filterOneY = mapper.intY()
	return true
}

// clampS32D32NofilterTransShaderproc directly shades a span for the translate-only, unfiltered, clamp-tiled fast path.
func clampS32D32NofilterTransShaderproc(s *bitmapProcState, x, y int32, dst []uint32) {
	maxX := s.pixmap.Width - 1
	maxY := s.pixmap.Height - 1
	ix := s.filterOneX + x
	iy := tpin32(s.filterOneY+y, 0, maxY)
	row := s.row(iy)
	count := len(dst)
	i := 0

	// clamp to the left
	if ix < 0 {
		n := min(int(-ix), count)
		for range n {
			dst[i] = row[0]
			i++
		}
		count -= n
		if count == 0 {
			return
		}
		ix = 0
	}
	// copy the middle
	if ix <= maxX {
		n := min(int(maxX-ix+1), count)
		copy(dst[i:i+n], row[ix:])
		i += n
		count -= n
		if count == 0 {
			return
		}
	}
	// clamp to the right
	for range count {
		dst[i] = row[maxX]
		i++
	}
}

// repeatS32D32NofilterTransShaderproc directly shades a span for the translate-only, unfiltered, repeat-tiled fast
// path.
func repeatS32D32NofilterTransShaderproc(s *bitmapProcState, x, y int32, dst []uint32) {
	stopX := s.pixmap.Width
	stopY := s.pixmap.Height
	ix := intMod(s.filterOneX+x, stopX)
	iy := intMod(s.filterOneY+y, stopY)
	row := s.row(iy)
	count := len(dst)
	i := 0
	for {
		n := min(int(stopX-ix), count)
		copy(dst[i:i+n], row[ix:int(ix)+n])
		i += n
		count -= n
		if count == 0 {
			return
		}
		ix = 0
	}
}

// filter32Alpha computes the constX bilerp between two rows: a vertical-only filter for 1-pixel-wide source images.
func filter32Alpha(t, color0, color1, alphaScale uint32) uint32 {
	const mask = 0xFF00FF
	scale := 256 - 16*t
	lo := (color0 & mask) * scale
	hi := ((color0 >> 8) & mask) * scale
	scale = 16 * t
	lo += (color1 & mask) * scale
	hi += ((color1 >> 8) & mask) * scale
	lo = ((lo >> 8) & mask) * alphaScale
	hi = ((hi >> 8) & mask) * alphaScale
	return ((lo >> 8) & mask) | (hi & ^uint32(mask))
}

// s32D32ConstXShaderproc directly shades a span for 1-pixel-wide source images.
func s32D32ConstXShaderproc(s *bitmapProcState, x, y int32, dst []uint32) {
	var iY0, iY1 int32
	var iSubY uint32

	if s.bilerp {
		var xy [2]uint32
		s.matrixProc(s, xy[:], 1, x, y)
		iY0 = int32(xy[0] >> 18)
		iY1 = int32(xy[0] & 0x3FFF)
		iSubY = (xy[0] >> 14) & 0xF
	} else {
		var yTemp int32
		if s.invMatrix.IsTranslate() {
			yTemp = s.filterOneY + y
		} else {
			mapper := newAutoMapper(s, x, y)
			// Undo the width/height normalization applied at setup when tiling.
			if s.tileModeX != TileClamp || s.tileModeY != TileClamp {
				yTemp = fixed3232ToInt(mapper.fy * int64(s.pixmap.Height))
			} else {
				yTemp = mapper.intY()
			}
		}
		stopY := s.pixmap.Height
		switch s.tileModeY {
		case TileClamp:
			iY0 = tpin32(yTemp, 0, stopY-1)
		case TileRepeat:
			iY0 = intMod(yTemp, stopY)
		default:
			iY0 = intMirror(yTemp, stopY)
		}
	}

	row0 := s.row(iY0)
	var color uint32
	if s.bilerp {
		row1 := s.row(iY1)
		color = filter32Alpha(iSubY, row0[0], row1[0], s.alphaScale)
	} else {
		color = alphaMulQLegacy(row0[0], s.alphaScale)
	}
	for i := range dst {
		dst[i] = color
	}
}

// doNothingShaderproc shades a span as fully transparent (used when setup fails after selection).
func doNothingShaderproc(_ *bitmapProcState, _, _ int32, dst []uint32) {
	for i := range dst {
		dst[i] = 0
	}
}

// chooseShaderProc selects a direct shaderProc fast path when the draw configuration qualifies, leaving shaderProc nil
// (so ShadeSpan32 falls back to matrixProc+sampleProc) otherwise.
func (s *bitmapProcState) chooseShaderProc() {
	if s.pixmap.Width == 1 && s.invMatrix.IsScaleTranslate() {
		if !s.bilerp && s.invMatrix.IsTranslate() && !s.setupForTranslate() {
			s.shaderProc = doNothingShaderproc
			return
		}
		s.shaderProc = s32D32ConstXShaderproc
		return
	}
	if s.alphaScale < 256 || !s.invMatrix.IsTranslate() || s.bilerp {
		return
	}
	switch {
	case s.tileModeX == TileClamp && s.tileModeY == TileClamp:
		if s.setupForTranslate() {
			s.shaderProc = clampS32D32NofilterTransShaderproc
		} else {
			s.shaderProc = doNothingShaderproc
		}
	case s.tileModeX == TileRepeat && s.tileModeY == TileRepeat:
		if s.setupForTranslate() {
			s.shaderProc = repeatS32D32NofilterTransShaderproc
		} else {
			s.shaderProc = doNothingShaderproc
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// the shader context

// bitmapProcBufMax is the size of the fixed sampling scratch buffer.
const bitmapProcBufMax = 128

// BitmapProcContext is the legacy fixed-point shading context paired with the N32 opaque blitter.
type BitmapProcContext struct {
	state  bitmapProcState
	opaque bool
	buffer [bitmapProcBufMax]uint32
}

// OpaqueAlpha reports whether the pixmap is opaque and the paint alpha is 255.
func (c *BitmapProcContext) OpaqueAlpha() bool { return c.opaque }

// maxCountForBufferSize computes the max sample count that fits the fixed buffer for the current matrix/filter
// configuration.
func (c *BitmapProcContext) maxCount() int {
	size := int32(bitmapProcBufMax * 4)
	size &^= 3
	if c.state.invMatrix.IsScaleTranslate() {
		size -= 4
		if size < 0 {
			size = 0
		}
		size >>= 1
	} else {
		size >>= 2
	}
	if c.state.bilerp {
		size >>= 1
	}
	return int(size)
}

// ShadeSpan32 shades a horizontal span of dst starting at device coordinate (x, y).
func (c *BitmapProcContext) ShadeSpan32(x, y int32, dst []uint32) {
	s := &c.state
	if s.shaderProc != nil {
		s.shaderProc(s, x, y, dst)
		return
	}
	maxN := c.maxCount()
	for len(dst) > 0 {
		n := min(len(dst), maxN)
		s.matrixProc(s, c.buffer[:], n, x, y)
		s.sampleProc(s, c.buffer[:], n, dst[:n])
		dst = dst[n:]
		x += int32(n)
	}
}

// legacyShaderCanHandle reports whether the inverse matrix is within range for the legacy fixed-point sampler.
func legacyShaderCanHandle(inv *geom.Matrix) bool {
	if inv.HasPerspective() {
		return false
	}

	// The legacy code uses 32.32 fixed-point, so ensure the inverse doesn't map device coordinates out of range.
	const maxDevCoord = 32767.0
	src, ok := mapRectLegacy(inv, geom.RectWH(maxDevCoord, maxDevCoord))
	if !ok {
		return false
	}
	const maxFixed32dot32 = float32(math.MaxInt32) * 0.25
	limit := geom.RectLTRB(-maxFixed32dot32, -maxFixed32dot32, maxFixed32dot32, maxFixed32dot32)
	return limit.ContainsRect(src)
}

// mapRectLegacy maps a rect through an affine matrix by its corners (legacyShaderCanHandle only feeds it
// non-perspective matrices).
func mapRectLegacy(m *geom.Matrix, r geom.Rect) (geom.Rect, bool) {
	pts := [4]geom.Point{
		{X: r.Left, Y: r.Top},
		{X: r.Right, Y: r.Top},
		{X: r.Right, Y: r.Bottom},
		{X: r.Left, Y: r.Bottom},
	}
	var dst [4]geom.Point
	m.MapPoints(dst[:], pts[:])
	out := geom.RectLTRB(dst[0].X, dst[0].Y, dst[0].X, dst[0].Y)
	for _, p := range dst[1:] {
		out.Left = minf(out.Left, p.X)
		out.Top = minf(out.Top, p.Y)
		out.Right = maxf(out.Right, p.X)
		out.Bottom = maxf(out.Bottom, p.Y)
	}
	return out, out.IsFinite()
}

// MakeLegacyImageContext builds the legacy shader context for an image shader draw: it unwraps local-matrix layers,
// then applies the gates that determine whether the legacy fixed-point sampler can handle the draw. Returns nil
// whenever the draw must fall back to the raster pipeline instead. The caller has already established src-over, no
// dither, and an N32 premul destination.
func MakeLegacyImageContext(s Shader, ctm geom.Matrix, paintAlpha uint8) *BitmapProcContext {
	mRec := newMatrixRec(ctm)
	for {
		if lms, ok := s.(*LocalMatrixShader); ok {
			lm := lms.LocalMatrix()
			mRec = mRec.concat(&lm)
			s = lms.Wrapped()
			continue
		}
		break
	}
	img, ok := s.(*ImageShader)
	if !ok {
		return nil // only ImageShader has a legacy context
	}

	// The legacy CPU sampler needs CPU pixels; a texture-backed image reads back here (rare — the legacy path is
	// reached only when rendering onto a raster device), a raster image resolves to itself for free.
	cpu := img.cpuImage()
	if cpu == nil {
		return nil
	}

	// Gates matching the image shader's legacy-context eligibility rules.
	if cpu.AlphaType() == imagecore.AlphaTypeUnpremul {
		return nil
	}
	if cpu.ColorType() != imagecore.ColorTypeN32 {
		return nil
	}
	if img.tileX != img.tileY {
		return nil
	}
	if img.tileX == TileDecal || img.tileY == TileDecal {
		return nil
	}

	sampling := img.sampling
	if sampling.isAniso() {
		sampling = anisoFallback(false)
	}
	supported := (sampling.Filter == FilterNearest && sampling.Mipmap == MipmapNone) ||
		(sampling.Filter == FilterLinear && sampling.Mipmap == MipmapNone) ||
		(sampling.Filter == FilterLinear && sampling.Mipmap == MipmapNearest)
	if sampling.UseCubic || !supported {
		return nil
	}
	if cpu.Width() > 32767 || cpu.Height() > 32767 {
		return nil
	}

	inv, invOK := mRec.totalInverse()
	if !invOK || !legacyShaderCanHandle(&inv) {
		return nil
	}
	// Color-space compatibility check: this pipeline is sRGB-only, so always compatible.

	px := cpu.PeekPixels(imagecore.CachingAllow)
	if px == nil {
		return nil
	}
	pm, pmOK := px.PixmapView()
	if !pmOK {
		return nil
	}

	c := bitmapProcContextPool.Get().(*BitmapProcContext)
	if !c.state.setup(pm, &inv, paintAlpha, sampling, img.tileX, img.tileY) {
		bitmapProcContextPool.Put(c)
		return nil
	}
	c.opaque = cpu.IsOpaque() && paintAlpha == 0xFF
	return c
}

// bitmapProcContextPool retains BitmapProcContext storage — a ~600-byte struct carrying the inline sampling scratch
// buffer — across draws so a repeated legacy image-shader fill does not reallocate it. MakeLegacyImageContext borrows;
// the draw layer hands the context back with RecycleLegacyImageContext once the blitter that held it is done (via
// LegacyShaderBlitter.Shade).
var bitmapProcContextPool = sync.Pool{New: func() any { return new(BitmapProcContext) }}

// RecycleLegacyImageContext returns a context obtained from MakeLegacyImageContext to the shared pool. Call it only
// after the synchronous fill that consumed the context's LegacyShaderBlitter is done; missing a call is safe (the pool
// allocates a fresh one next time). After RecycleLegacyImageContext the caller must treat c as invalid. setup fully
// re-initializes a reused context, so pooling does not change the shaded output.
func RecycleLegacyImageContext(c *BitmapProcContext) {
	c.state.pixmap = raster.Pixmap{}
	bitmapProcContextPool.Put(c)
}

// alpha255To256 converts an 8-bit alpha (0..255) to a 9-bit scale (0..256) suitable for >>8 scaling.
func alpha255To256(a uint32) uint32 { return a + 1 }
