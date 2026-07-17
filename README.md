# canvas

A 2D graphics library for Go: a pure-Go reimplementation of the [Skia](https://skia.org) rendering subsystems, with both
a CPU rasterizer and a GPU (OpenGL) renderer.

Note that this has been tailored to the needs of my specific projects (principally
[unison](https://github.com/richardwilkes/unison)) and may not be suitable for anyone else.

[![Go Reference](https://pkg.go.dev/badge/github.com/richardwilkes/canvas.svg)](https://pkg.go.dev/github.com/richardwilkes/canvas)
[![Build](https://github.com/richardwilkes/canvas/actions/workflows/build.yml/badge.svg)](https://github.com/richardwilkes/canvas/actions/workflows/build.yml)

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
