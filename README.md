# canvas

[![Go Reference](https://pkg.go.dev/badge/github.com/richardwilkes/canvas.svg)](https://pkg.go.dev/github.com/richardwilkes/canvas)
[![Build](https://github.com/richardwilkes/canvas/actions/workflows/build.yml/badge.svg)](https://github.com/richardwilkes/canvas/actions/workflows/build.yml)

A 2D graphics library for Go: a pure-Go reimplementation of the [Skia](https://skia.org) rendering subsystems, with both
a CPU rasterizer and a GPU (OpenGL) renderer.

> [!WARNING]
> This library has been tailored to the needs of my specific projects (principally
> [unison](https://github.com/richardwilkes/unison)) and may not be suitable for anyone else.

## Requirements

**64-bit targets only.** The module assumes `int` and `uintptr` are 64 bits wide; 32-bit builds are not supported and
are not tested. The `geom` package carries a compile-time assertion of this, so a 32-bit build fails to compile rather
than silently overflowing. Code guards genuine `int32` overflow where the API's `int32` dimensions, offsets and strides
demand it, but never widens `int` math or caps `int` values just to survive a 32-bit `int`.

The module is also 100% cgo-free and must stay that way; `build.sh` enforces it.

## SIMD

By default, the module compiles to portable Go, and each performance-critical kernel has a scalar form. If you build
with `GOEXPERIMENT=simd` on Go 1.27 or later, the build wires in vector kernels for arm64 and amd64, written against
the standard library's `simd/archsimd` package. The kernels cover:

- `shaders` — the pipeline stages and the image-filter kernels
- `raster` — the span and blit rows
- `maskfilter` — the mask blur
- `filtercore` — the blur engine
- `imagecore` — the pixel-conversion rows
- `codecs` — the VP8 (lossy WebP) encoder kernels

Both build modes produce bit-identical output, and the test suites check this on every kernel. On amd64, the kernels
need AVX2 and FMA; on a CPU without them, the code falls back to the scalar form at run time. Each kernel also carries
a per-arch dispatch preference from measured benchmarks (see [SIMD-BENCH.md](SIMD-BENCH.md)), so a kernel that does
not beat its scalar form on real silicon stays off.

## Packages

| Package | What it is |
| --- | --- |
| `canvas` | The canvas: matrix/clip save stack, layers, and the draw entry points |
| `geom` | Scalar, point, rect, matrix and curve math — float32, with Skia's rounding and degenerate-case rules |
| `path` | Path storage (verbs, points, conic weights), the builder op set, queries, transforms |
| `pathops` | Boolean path ops, simplify, and the op builder |
| `stroke` | The stroker: caps, joins, miter limits, stroke-to-fill |
| `patheffect` | Dash, corner, discrete, 1d/2d stamping, trim, sum/compose |
| `contour` | Contour measurement: lengths, position/tangent, matrices, segment extraction |
| `raster` | The CPU rasterizer: edge lists, scan conversion, blitters, the span pipeline |
| `surface` | Raster surfaces (owned and borrowed pixels), snapshots with copy-on-write |
| `gpu`, `gpu/gl` | The GL backend: render targets, ops/tasks, GLSL emission |
| `gpu/text` | GPU text: strike cache, glyph vectors, sub-runs, vertex filling, the budgeted blob cache |
| `shaders` | Color, blend, local-matrix and the four gradient families, plus CPU evaluation |
| `colorcore`, `colorfilter` | Color types and color filters |
| `maskfilter`, `imagefilter`, `filtercore` | Mask filters and the image-filter DAG (CPU and GPU-native) |
| `imagecore`, `codecs` | Images; decoders (PNG/JPEG/GIF/BMP/WebP/ICO/WBMP) and encoders |
| `font`, `fontmgr`, `textblob` | The pure-Go font stack: typefaces, font managers, metrics, text blobs |
| `pdf` | PDF document output |
| `stream` | Write streams: a memory stream with read-back and a file stream that latches on error |
