#! /usr/bin/env bash

set -eo pipefail

trap 'echo -e "\033[33;5mBuild failed on build.sh:$LINENO\033[0m"' ERR

for arg in "$@"; do
	case "$arg" in
	--all | -a)
		FMT=1
		LINT=1
		TEST=1
		RACE=-race
		;;
	--fmt | -f)
		FMT=1
		;;
	--lint | -l)
		LINT=1
		;;
	--race | -r)
		TEST=1
		RACE=-race
		;;
	--test | -t)
		TEST=1
		;;
	--help | -h)
		echo "$0 [options]"
		echo "  -a, --all  Equivalent to --fmt --lint --race"
		echo "  -f, --fmt  Verify the source formatting (gofumpt and goimports)"
		echo "  -l, --lint Run the linters for every supported platform"
		echo "  -r, --race Run the tests with race-checking enabled"
		echo "  -t, --test Run the tests"
		echo "  -h, --help This help text"
		exit 0
		;;
	*)
		echo "Invalid argument: $arg"
		exit 1
		;;
	esac
done

# The module is 100% cgo-free and must stay that way: everything is built and tested with cgo disabled so any
# accidental reintroduction fails loudly. (The -race test run below is the one exception; see the comment there.)
# CGO_ENABLED is restored to its prior state on exit so the setting cannot leak out of this script.
ORIG_CGO_ENABLED_SET=${CGO_ENABLED+set}
ORIG_CGO_ENABLED=${CGO_ENABLED-}
restore_cgo_enabled() {
	if [ "$ORIG_CGO_ENABLED_SET" = "set" ]; then
		export CGO_ENABLED="$ORIG_CGO_ENABLED"
	else
		unset CGO_ENABLED
	fi
}
trap restore_cgo_enabled EXIT
export CGO_ENABLED=0

# Guard against cgo creeping back in. A stray cgo file would not necessarily break the CGO_ENABLED=0 build (build
# constraints just exclude it), so check for import "C" explicitly, in both its single and grouped import forms.
echo -e "\033[33mVerifying the module is cgo-free...\033[0m"
CGO_USERS=$(grep -rl --include='*.go' -e '^import "C"' -e '^[[:space:]]*"C"$' . || true)
if [ -n "$CGO_USERS" ]; then
	echo -e "\033[31mcgo is not permitted in this module, but these files import \"C\":\033[0m"
	echo "$CGO_USERS"
	exit 1
fi

# Build the Go code
echo -e "\033[33mBuilding Go code...\033[0m"
go build -v ./...

# Run the tests
if [ "$TEST"x == "1x" ]; then
	TEST_CGO=0
	if [ -n "$RACE" ]; then
		echo -e "\033[33mTesting with -race enabled...\033[0m"
		if [ "$(go env GOOS)" != "darwin" ]; then
			# Go's prebuilt race runtime itself requires cgo everywhere except macOS ("go: -race requires cgo").
			# The module still contains no cgo (enforced by the guard above); this only changes how the test
			# binaries link the race runtime.
			TEST_CGO=1
		fi
	else
		echo -e "\033[33mTesting...\033[0m"
	fi
	CGO_ENABLED=$TEST_CGO go test $RACE ./... | grep -v "no test files"
fi

# Both the formatting check and the linters come out of golangci-lint, so whichever of them runs first installs it.
ensure_golangci_lint() {
	if [ -n "$GOLANGCI_LINT" ]; then
		return
	fi
	GOLANGCI_LINT_VERSION=$(curl --head -s https://github.com/golangci/golangci-lint/releases/latest | grep -i location: | sed 's/^.*v//' | tr -d '\r\n')
	TOOLS_DIR=$(go env GOPATH)/bin
	if [ ! -e "$TOOLS_DIR/golangci-lint" ] || [ "$("$TOOLS_DIR/golangci-lint" version 2>&1 | awk '{ print $4 }' || true)x" != "${GOLANGCI_LINT_VERSION}x" ]; then
		echo -e "\033[33mInstalling version $GOLANGCI_LINT_VERSION of golangci-lint into $TOOLS_DIR...\033[0m"
		mkdir -p "$TOOLS_DIR"
		curl -sfL https://raw.githubusercontent.com/golangci/golangci-lint/main/install.sh | sh -s -- -b "$TOOLS_DIR" v$GOLANGCI_LINT_VERSION
	fi
	GOLANGCI_LINT="$TOOLS_DIR/golangci-lint"
}

# Verify formatting. Formatting is neither OS- nor architecture-specific and the formatters do not evaluate build
# constraints, so a single pass covers every source file, including the per-platform ones.
if [ "$FMT"x == "1x" ]; then
	ensure_golangci_lint
	echo -e "\033[33mChecking the formatting of the Go code...\033[0m"
	if ! "$GOLANGCI_LINT" fmt --diff; then
		echo -e "\033[31mRun 'golangci-lint fmt' to apply the formatting shown above.\033[0m" >&2
		exit 1
	fi
	echo "0 issues."
fi

# Run the linters for every supported platform. The module carries per-GOOS sources (gpu/gl), per-GOARCH kernel
# sources (the NEON and archsimd files), and goexperiment.simd-tagged sources, so a host-shaped lint pass leaves
# files unanalyzed; each supported GOOS/GOARCH pair gets its own pass, in both experiment modes. internal/oracle
# typechecks against the per-platform main module, so it joins the platform loop (no experiment dimension — it has no
# simd-tagged files); internal/tools is fully portable, so the host pass covers it.
LINT_PLATFORMS="linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64"
if [ "$LINT"x == "1x" ]; then
	ensure_golangci_lint
	echo -e "\033[33mLinting...\033[0m"
	for platform in $LINT_PLATFORMS; do
		for exp in "" simd; do
			echo -e "\033[33m  $platform${exp:+ (GOEXPERIMENT=$exp)}\033[0m"
			GOOS=${platform%/*} GOARCH=${platform#*/} GOEXPERIMENT=$exp "$GOLANGCI_LINT" run ./...
		done
	done
	for platform in $LINT_PLATFORMS; do
		echo -e "\033[33m  internal/oracle $platform\033[0m"
		(cd internal/oracle && GOOS=${platform%/*} GOARCH=${platform#*/} "$GOLANGCI_LINT" run -c ../../.golangci.yml ./...)
	done
	echo -e "\033[33m  internal/tools\033[0m"
	(cd internal/tools && "$GOLANGCI_LINT" run -c ../../.golangci.yml ./...)
fi
