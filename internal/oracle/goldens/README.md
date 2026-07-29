# Self-captured golden sets

These are the gating reference for every rendering lane: the corpus rendered through **the port's own backends**,
captured by `oracle bless` and committed after review. A golden here answers "did the port's output change at all?" —
not "does the port agree with some other renderer?"

Each set is a plain directory of PNGs plus a `manifest.json`. The PNGs store the surface's RGBA8888-**premul** bytes
verbatim in the PNG samples (byte-exact round trip; viewers show dark fringes where alpha < 255 — expected). For the
GPU lanes the blessed capture is the *cold* render — the first GL context in a fresh process — which is the most
reproducible render available and is canonical by designation.

## Layout: one set per lane, per platform

```
raster/<GOOS_GOARCH>/     compared bit-exactly (profile `exact`) — pure Go, no GL exposure
gpu/<GOOS_GOARCH>/        compared under `exact1` (±1 LSB, zero pixels beyond)
gpudmsaa/<GOOS_GOARCH>/   compared under `exact1`; the dynamic-MSAA lane needs its own reference because a
                          4x MSAA resolve antialiases edges differently from coverage-AA
```

Per-platform sets are what make exact comparison possible at all:

- **Raster** output legitimately differs *between architectures*: Go fuses `x*K ± M` into an FMA on arm64 but not
  amd64, so float contraction gives each architecture its own bit-exact answer. On its home platform each set
  reproduces exactly.
- **GPU** output is a property of the capturing GL stack (Apple's software renderer on darwin, Mesa llvmpipe on linux
  and windows), which differs per platform and per architecture word width.

Not every lane exists everywhere: windows_arm64 is raster-only (no arm64 software-GL stack is provisioned in CI), and
darwin_arm64 deliberately has **no gpudmsaa set** — Apple's software renderer there quantizes MSAA edges in one of two
bit-exact flavors per GL session, proven driver-internal, so any captured set would intermittently fail through no
fault of the code (see `../gorender/gpudmsaagolden_test.go`). A platform without a set skips the corresponding gate,
visibly.

## Manifest (schema 2)

Written by `oracle bless` (`../golden/golden.go`):

- `platform` — `GOOS_GOARCH` that captured the set.
- `lane` — `raster`, `gpu`, or `gpudmsaa`.
- `gl_renderer`, `gl_version` — the capturing context's `GL_RENDERER`/`GL_VERSION` strings; empty for raster, which has
  no environment exposure. These pin the GL stack: the GPU/DMSAA gates compare the live context's renderer string
  against `gl_renderer` and **fail with a diagnosis** ("goldens captured on X, running on Y — the GL stack moved") on
  mismatch, so a runner-image driver bump surfaces as one line instead of a wall of pixel differences.
- `captured_at` — UTC date of capture.
- `entries` — per scenario: name, dimensions, and the SHA-256 of the raw premul RGBA bytes (not the PNG file).
- `schema` — 2, the only schema in use. `oracle bless` refuses to overwrite a set carrying any other.

## Regenerating

Never by hand-editing and never automatically: dispatch the "Capture goldens" workflow, download with
`../capture.sh RUN_ID`, review the change summaries and image diffs, and commit deliberately. The full runbook —
including when recapture is appropriate and what capture-time failures mean — is in `../README.md`.
