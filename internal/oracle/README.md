# Oracle harness

This module is the rendering-regression harness for the pure-Go port. It renders the scenario corpus through the port's
own backends and gates the output — bit-exactly where possible — against per-platform golden sets captured from the
port itself, and it replays the numeric probes against the frozen answers of the C Skia library the port was originally
verified against. It is a **separate Go module**; nothing outside this directory may import it.

## Layout

- `scenario/` — the declarative scenario corpus. One scenario = one named drawing routine against the `Canvas`
  interface, with paints and paths as plain data (`Paint`, `PathSpec`).
- `gorender/` — renders the corpus (raster, GL, and GL under dynamic MSAA) and hosts every rendering gate.
- `imgdiff/` — pixel comparison under the named threshold profiles below, with heatmap / side-by-side failure
  artifacts. Note `Result.Pass` gates on the *fraction* of pixels exceeding the profile's channel tolerance; under the
  gating profiles that fraction is zero, so a single over-threshold pixel fails.
- `golden/` — golden PNG + manifest I/O. Goldens store the surface's RGBA8888-**premul** bytes verbatim in the PNG
  samples (byte-exact round trip; viewers show dark fringes where alpha < 255 — expected).
- `goldens/` — the gating reference: per-platform, per-lane golden sets captured from the port's own output by `oracle
  bless`. See `goldens/README.md`.
- `cmd/oracle` — CLI: `list`, `gen -out DIR [-gpu]`, `diff -a DIR -b DIR [-profile P] [-artifacts DIR]`,
  `soak -n N [-gpu|-dmsaa]` (the determinism proof: renders the corpus N times in fresh sessions and fails on any
  pass-to-pass divergence), and `bless -lane {raster|gpu|gpudmsaa}` (captures a golden set, double-rendering in fresh
  sessions and refusing to write on divergence). Details in the package doc comment.
- `probe/` — the numeric probes (the pure-math exit criteria): differential tests driving `geom`/`path`/`stroke`/
  `contour` over shared corpora, compared against Skia's frozen answers in `probe/testdata/ref` (keyed by a hash of the
  inputs; see `probe/ref_test.go`).

## The rendering gates

Three lanes render the full corpus and compare against the matching `goldens/<lane>/<GOOS_GOARCH>/` set:

- **raster** — `oracle gen` + `diff -profile exact` in CI (`build.yml`'s oracle jobs). Pure Go, no GL exposure, so the
  comparison is strictly **bit-exact** on every platform with a checked-in set. Sets differ *between* architectures
  (FMA contraction), which is why they are per-platform.
- **gpu** — `gorender`'s `TestGoGPUvsSelfCapturedGolden` (port-owned offscreen render target) and
  `TestGoGPUWrappedFBOvsSelfCapturedGolden` (caller-owned wrapped FBO, the production surface path unison drives). Both
  gate every scenario under **`exact1`**.
- **gpudmsaa** — `TestGoGPUDMSAAvsSelfCapturedGolden`: the wrapped-FBO path with the dynamic-MSAA surface-props flag.
  Also `exact1`. MSAA resolves antialiased edges differently from coverage-AA, so the lane has its own golden sets.

The GPU gates pin `CANVAS_GLTEST_RENDERER=software` (comparing across GL stacks is meaningless at these tolerances) and
verify the live context's `GL_RENDERER` string against the manifest's, so a moved GL stack fails with a one-line
diagnosis instead of a wall of pixel differences. A platform with no checked-in set skips its gate — visibly, since the
CI steps run `-v`.

### Threshold profiles

- **`exact`** (delta 0) — the raster gate, and `bless`/`soak`'s raster self-check.
- **`exact1`** (max channel delta ≤ 1, *zero* pixels beyond) — the gpu and gpudmsaa gates. Software GL rasterizers
  wobble ±1 LSB intermittently between GL sessions; this is proven driver-internal (identical inputs and byte-identical
  GL command streams still produce ±1-differing output — see the `oracle soak` doc comment for the evidence), and the
  blessed capture is one representative of that envelope. Real rendering breaks measure ≥32 LSB, so the envelope costs
  no detection power.
- **`gpu`** — the one loose, cross-renderer tolerance left. No golden gate uses it; it bounds `gorender`'s atlas
  CPU-vs-GPU self-consistency cross-check, which compares the port's two live backends against each other rather than
  against goldens.

### Report-not-gate exclusions

Two narrow, renderer-keyed exclusion lists exist, each with the full evidence in its comment. Listed scenarios are
still rendered and logged, but their pixels are not gated:

- `gorender.DriverBimodal` (`clip-persp` on Apple's software renderer, in `gorender/driverquirks.go`) — that stack
  draws it in one of two bit-exact flavors per GL session, far beyond the ±1 envelope; applies only to sets whose
  manifest records that renderer. The same pathology under 4x MSAA is why darwin_arm64 has **no** gpudmsaa set at all.
  This one lives in a non-test file because **capture must agree with gating**: `oracle soak` and `oracle bless`
  consult it too, reporting such a flip without counting it as nondeterminism. When only the gates knew about it,
  capturing that lane succeeded or failed by luck depending on whether the flip landed mid-soak (darwin_arm64 gpu,
  2026-07-27).
- `wrappedFBOKnifeEdge` (`text-sdf-rotated` on one specific llvmpipe build) — a deterministic 1-pixel backing-type
  rasterization knife edge, wrapped-FBO lane only, keyed to the exact `GL_RENDERER` string so a driver bump forces
  re-evaluation.

The bar for ever extending these lists is the one those comments document: the divergence must be proven
driver-internal (output independent of all app-allocation content, GL command stream byte-identical across differing
runs, other stacks deterministic on the same code path) — never merely observed. An excusal is the accepted residue of
gating on a particular driver, not a way to green a red gate.

## Regeneration runbook

Golden capture is **never automatic** — no CI job writes into `goldens/`. Regenerate when, and only when:

- **an intentional rendering change** lands (the gates correctly report that output changed);
- **a GL-stack-mismatch diagnosis** fires after a runner-image driver bump (the stack under the GPU/DMSAA sets moved;
  recapture re-anchors them on the new stack);
- **the corpus changes** (the exact-cover check fails: a scenario was added, removed, or renamed).

Procedure:

1. **Start from gates that are understood.** Confirm the existing gates' state at the capture commit and that every
   failure is explained by the change you intend to bless. Never re-capture to silence a red gate without diagnosing
   it — port-generated goldens can absorb a regression, so adoption is always a deliberate, reviewed act.
2. **Dispatch the "Capture goldens" workflow** (`.github/workflows/capture-goldens.yml`, `workflow_dispatch` only).
   Each matrix leg runs `oracle bless` for its lanes and uploads a `goldens-<GOOS_GOARCH>` artifact. A leg whose GL
   context fails to come up skips the lane loudly (bless exits 3, distinct from failure); a missing artifact must be
   obvious in the run log, not discovered at commit time.
3. **Download and merge**: `internal/oracle/capture.sh RUN_ID` pulls the run's artifacts into the working tree, merging
   only the lanes each leg actually blessed, and only its own platform directories. It prints each leg's
   captured/skipped/failed summary as it goes — a leg missing a lane is invisible in `git diff`, so read those lines.
   It never commits.
4. **Review**: the run logs carry bless's per-scenario old-vs-new change summaries (imgdiff stats); locally, `git diff`
   shows exactly which sets changed, and the PR shows GitHub's image diffs. Every changed image should be explained by
   the intended change.
5. **Commit** the reviewed sets manually.

What capture-time failures mean:

- **soak/bless MISMATCH** (cross-pass divergence): the port rendered differently in two fresh sessions on the same
  machine — a nondeterminism bug. Fix it first; a golden set captured over nondeterminism gates nothing. `oracle soak`
  is the diagnostic tool (raster compares strict hashes; the GPU lanes compare per-pixel under the ±1 envelope, so a
  MISMATCH there is already beyond the known driver wobble). A `bimodal` line is *not* a MISMATCH: that is a
  `gorender.DriverBimodal` scenario on its listed stack, reported and counted but not failed.
- **A lane missing from a leg's artifact**: the leg skipped it (no GL context) or failed it. The artifact's
  `CAPTURED.txt` and `capture.sh`'s per-leg summary line both say which; the lane's committed set is left untouched,
  so its gates keep running against the old goldens until a later capture succeeds.
- **GL-stack mismatch** (a gate failing on the `GL_RENDERER` string): the GL stack moved underneath the goldens — a
  runner-image driver bump, or a local run on the wrong stack. If the move is intentional, recapture; the new manifest
  records the new stack.
- **bless refusing a manifest's schema**: the target directory holds a golden set bless cannot read as its own, which
  it will not silently capture over. Diagnose how it got there rather than deleting it.

## Probe comparison policy

Where the C++ computes through double before storing (affine `setConcat`, affine invert, rect ops, scalar rounding, path
bookkeeping), the Go port must match **bit-exactly**. Where the C++ does float arithmetic, clang's default
`-ffp-contract=on` fuses mul+add chains, and which chains fuse is a per-compiler, per-platform choice (Skia's own output
differs between its platforms); those paths are compared under a magnitude-scaled tolerance (`agree()` in
`probe/probe_test.go`) tight enough to catch formula-level bugs. Transcendentals (libm vs Go math, last-ULP) get the
same treatment. Behavioral cliffs where the C library branches on the sign of such noise are pinned to deterministic
unfused behavior on the Go side and skipped in the probes.

The same contraction question bites the *corpora*, not just the comparisons: Go fuses `x*K ± M` into an FMA on arm64 but
not amd64, so a fixed-seed corpus built that way is a different corpus per platform — and since the fixtures are keyed
by input hash, it would miss everywhere it was not captured. Force the intermediate rounding with an explicit
`float32(...)`.
