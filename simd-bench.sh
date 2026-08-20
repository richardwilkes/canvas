#! /usr/bin/env bash

# Collects the benchmark and gate data needed to settle the goexperiment.simd dispatch preferences on this machine.
# Run it from the repo root on the `simd` branch. It writes everything into ./simd-bench-results and packs that
# directory into simd-bench-results.tgz. See SIMD-BENCH.md for the directions.

set -eo pipefail

trap 'echo -e "\033[33;5msimd-bench failed on simd-bench.sh:$LINENO\033[0m"' ERR

GOVER=$(go env GOVERSION)
case "$GOVER" in
go1.2[7-9]* | go1.[3-9]* | go[2-9]*) ;;
*)
	echo "Go 1.27 or later is required (found $GOVER)" >&2
	exit 1
	;;
esac

OUT=simd-bench-results
rm -rf "$OUT" simd-bench-results.tgz
mkdir -p "$OUT"
echo '*' >"$OUT/.gitignore"

echo "== system info"
{
	date
	go version
	echo "GOOS=$(go env GOOS) GOARCH=$(go env GOARCH)"
	if [ "$(uname -s)" = "Darwin" ]; then
		sysctl -n machdep.cpu.brand_string 2>/dev/null || true
	else
		grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
	fi
	git rev-parse HEAD
} >"$OUT/sysinfo.txt"
cat "$OUT/sysinfo.txt"

# The wiring tests show which kernels this CPU supports and which ones init actually dispatched: a PASS means the
# preference table matched the hardware, a SKIP names the gate that declined.
echo "== gates"
GOEXPERIMENT=simd go test ./shaders/ ./raster/ ./maskfilter/ -run 'SIMDWiring' -v -count=1 >"$OUT/gates.txt" 2>&1 || {
	echo "wiring tests failed; see $OUT/gates.txt" >&2
	exit 1
}
grep -E -- '--- (PASS|SKIP|FAIL)' "$OUT/gates.txt" || true

# Bit-exactness on this hardware before any timing.
echo "== correctness (GOEXPERIMENT=simd)"
GOEXPERIMENT=simd go test ./shaders/ ./raster/ ./maskfilter/ -count=1 >"$OUT/correctness.txt" 2>&1 || {
	echo "correctness failed; see $OUT/correctness.txt" >&2
	exit 1
}
cat "$OUT/correctness.txt"

bench() {
	local name=$1 pkg=$2 pattern=$3 count=$4 benchtime=$5
	echo "== bench $name (default)"
	go test "$pkg" -run XXX -bench "$pattern" -count "$count" -benchtime "$benchtime" >"$OUT/${name}_default.txt" 2>&1
	echo "== bench $name (GOEXPERIMENT=simd)"
	GOEXPERIMENT=simd go test "$pkg" -run XXX -bench "$pattern" -count "$count" -benchtime "$benchtime" \
		>"$OUT/${name}_simd.txt" 2>&1
}

bench stages ./shaders/ 'Stage$' 10 100000x
bench spans ./raster/ 'BenchmarkClampSpan01$|BenchmarkStoreSpanSrc$|BenchmarkPMSrcOverRow$|BenchmarkBlitMaskOpaqueRow$' 10 100000x
bench blits ./raster/ 'BenchmarkFillWords$|BenchmarkFillBytes$|BenchmarkColor32Row$|BenchmarkBlitMaskTranslucentRow$|BenchmarkInterp256Row$|BenchmarkPremulRow$|BenchmarkPMBlendRow$|BenchmarkBlitRowLCD16$|BenchmarkBlitRowLCD16Opaque$|BenchmarkBlendRowLCD16Opaque$' 10 100000x
bench maskblur ./maskfilter/ 'DirectBlur' 10 20000x
bench fillpath ./raster/ 'FillPath' 6 300x
bench canvas ./canvas/ 'BenchmarkDrawRectFillAA$|BenchmarkGradientRectFill$|BenchmarkBlurRect$|BenchmarkTextRun$|BenchmarkImagePaintFill$|BenchmarkImageScale$' 6 200x

# Render benchstat comparisons when the tool is available; the raw files above are the data of record either way.
if command -v benchstat >/dev/null 2>&1; then
	BS=benchstat
elif [ -x "$(go env GOPATH)/bin/benchstat" ]; then
	BS="$(go env GOPATH)/bin/benchstat"
else
	BS=""
fi
if [ -n "$BS" ]; then
	for name in stages spans blits maskblur fillpath canvas; do
		"$BS" "$OUT/${name}_default.txt" "$OUT/${name}_simd.txt" >"$OUT/${name}_benchstat.txt" 2>&1 || true
	done
	echo "== benchstat summaries written"
else
	echo "benchstat not found; raw files are enough — or: go install golang.org/x/perf/cmd/benchstat@latest"
fi

tar czf simd-bench-results.tgz "$OUT"
echo
echo "Done. Send back simd-bench-results.tgz (or the $OUT directory)."
