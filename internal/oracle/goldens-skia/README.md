# Frozen Skia archive (non-gating)

These are the **C Skia library's final renders** of the scenario corpus, captured through the cgo bridge shortly before
that library was removed from this repository. They are a historical record, not a reference: **no gate consumes this
directory**. The gating goldens are the port's own self-captured sets under `../goldens/` (see the READMEs there and in
`../`).

The archive is **frozen and cannot be extended** — no Skia build exists here anymore. Any scenario added to the corpus
from now on simply has no entry here. Do not add, regenerate, or "fix" anything in it; `oracle bless` and `capture.sh`
both refuse to write over its schema-1 manifests for exactly this reason.

What it is for: a one-off "how far is the port from original Skia?" measurement, run by hand via
`oracle diff -a <archive set> -b <fresh oracle gen output>` under the tolerance profiles built for cross-renderer
comparison (`cpu`, `text`, `gpu` — see `../imgdiff/`). Expect the divergence patterns of that era: two renderers differ
legitimately at AA edges, and text output depends on the rasterizing scaler.

Contents:

- `*.png` + `manifest.json` (top level) — the **raster** renders, a single shared set captured on darwin_arm64. It was
  only ever bit-exact on its home platform; on other platforms it compared within the `cpu` tolerance profile (float
  contraction gives each architecture its own bit-exact raster answer, which is why the live goldens are per-platform).
  It holds 111 of the corpus's 121 scenarios: the missing 10 (the nine `text-*` scenes plus `layer-nested-alpha`) were
  excluded from the shared raster set of that era and gated only on the GPU lanes.
- `gpu/<GOOS_GOARCH>/` — the GL-backend renders (full corpus), per platform. See `gpu/README.md`.
- `gpudmsaa/darwin_arm64/` — the dynamic-MSAA renders (full corpus), darwin_arm64 only. See `gpudmsaa/README.md`.

All manifests are schema 1 (name/size/hash only): they predate the schema-2 fields that record the capturing GL stack,
so the capture stacks described in the sub-READMEs are documented prose, not machine-checked data.
