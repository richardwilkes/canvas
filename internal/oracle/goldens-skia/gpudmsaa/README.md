# Archived Skia DMSAA renders

Frozen archive — see `../README.md` for what this archive is and is not. Nothing gates against these.

The corpus rendered through the **C Skia library's GL backend with the dynamic-MSAA surface-props flag**
(`SkSurfaceProps::kDynamicMSAA_Flag`), darwin_arm64 only, captured on Apple's software GL stack (the same GPU-less CI
runner stack as `../gpu/darwin_arm64/`). Under DMSAA, path/stencil render passes are promoted to an internal 4x MSAA
attachment and resolved back, so these renders are not comparable with the coverage-AA `../gpu/` sets.

Two caveats for anyone diffing against this set:

- **It is a software-renderer capture.** On hardware GL, MSAA sample positions and the resolve filter are driver
  properties, so several scenarios diverge without anything being wrong.
- **Its own capture machine could not reproduce it exactly.** Back-to-back Skia captures on the same machine and
  renderer differed on a few scenarios, with `clip-persp` swinging far beyond any tolerance — the same bimodal
  MSAA-edge quantization of Apple's software renderer that later led the port's own goldens to skip a darwin_arm64
  DMSAA set entirely (see `../../gorender/gpudmsaagolden_test.go`).

The port's DMSAA output now gates against its own self-captured sets in `../../goldens/gpudmsaa/` (llvmpipe and
darwin_amd64 legs).
