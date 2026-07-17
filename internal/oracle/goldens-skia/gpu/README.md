# Archived Skia GPU renders

Frozen archive — see `../README.md` for what this archive is and is not. Nothing gates against these.

The corpus rendered through the **C Skia library's GL backend** (via the removed cgo bridge), one directory per
`GOOS_GOARCH`, as RGBA8888-premultiplied PNGs plus a schema-1 `manifest.json`. Each set was captured on its platform's
CI software-GL stack of the time: Apple's software GL on darwin_arm64 (GPU-less runner VMs), Mesa llvmpipe under Xvfb
on linux_amd64, and a Mesa3D llvmpipe drop-in on windows_amd64 — schema-1 manifests do not record the renderer
strings, so those stacks are provenance notes, not checked data.

While the C library existed, the port's GL lanes gated against these under the cross-renderer `gpu` tolerance profile
with a small excused-divergence list (two GPU AA implementations legitimately disagree at edges). That role is over:
the port's GL output now gates against its own self-captured sets in `../../goldens/gpu/`. For a historical
measurement, diff a fresh `oracle gen -gpu` output against a set here under `-profile gpu`.
