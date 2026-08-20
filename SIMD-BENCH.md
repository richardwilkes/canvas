# amd64 benchmark handoff for the `simd` branch

The `goexperiment.simd` kernels carry per-arch dispatch preferences. The arm64 values come from measured benchstat
runs on an M4 Max. The amd64 values are inferred: no real amd64 hardware has run the kernels yet (Rosetta 2 lacks the
FMA feature bit, so the shaders gate declines there). This run collects the data to settle them.

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
