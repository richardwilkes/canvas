# amd64 benchmark handoff for the `simd` branch

The `goexperiment.simd` kernels carry per-arch dispatch preferences. The arm64 values come from measured benchstat
runs on an M4 Max. The amd64 values were confirmed on 2026-08-20 with data from a Xeon W-2191B (darwin/amd64): all 4
wiring gates passed, the bit-exactness suites passed, and every wired kernel beat its scalar default (stages -49% to
-84%, spans -57% to -70%, blit rows -21% to -78%, the fp88 blur -93% to -96%; canvas end-to-end geomean -37%). No
preference constant needed a flip.

The script stays for future hardware: a new CPU family, a new Go release, or a kernel change can re-run the same
collection.

## What you need

- An amd64 machine with AVX2 (Haswell/2013 or later). FMA as well, for the shaders kernels — any AVX2 CPU short of
  the rarest exceptions has it.
- Go 1.27 or later.
- An idle machine, on mains power. Background load skews the numbers badly.

## Steps

1. `git clone https://github.com/richardwilkes/canvas && cd canvas && git checkout simd`
2. `./simd-bench.sh`
3. Send back `simd-bench-results.tgz`.

The script takes about 5-10 minutes. It first records the system info, then runs the wiring tests (they report which
kernels the CPU's gates accepted), then the bit-exactness suites, and only then the benchmarks — each suite in both
build modes, 10 runs each for benchstat. If any correctness step fails, it stops and names the log file; send that
file back too, because a failure there matters more than any timing.

## What happens with the data

Each `*_benchstat.txt` (or raw pair) decides the amd64 `preferSIMD*` constants in:

- `shaders/stage_simd_amd64.go` — 22 stage kernels
- `raster/span_simd_amd64.go` + `raster/blit_simd_amd64.go` — 14 span/blit rows
- `maskfilter/maskblur_simd_arm64.go`'s amd64 twin — the 2 blur drivers

A kernel that loses to the scalar form on real silicon gets its constant flipped to `false` — a one-line change per
kernel; everything else stays as measured.
