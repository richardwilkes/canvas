#! /usr/bin/env bash
#
# capture.sh RUN_ID — download the golden sets a capture-goldens.yml dispatch produced into this working tree, ready
# for `git diff` review and a manual commit. This script never commits anything: port-generated goldens can absorb a
# regression, so adopting a new set is always a deliberate act — a human reviews the change summary and the PR image
# diffs.
#
# RUN_ID is the workflow run id of the "Capture goldens" dispatch (find it with `gh run list
# --workflow=capture-goldens.yml`). Requires an authenticated `gh` CLI in this repository.
#
# Each goldens-<GOOS_GOARCH> artifact holds that leg's self-captured goldens/<lane>/<GOOS_GOARCH>/ subtrees; the legs
# are platform-disjoint, so all artifacts merge cleanly into goldens/. Any lane/platform directory being replaced is
# removed first so stale scenarios cannot linger. The one thing this script refuses to do is clobber a schema-1
# (Skia-era) set: those live only in the goldens-skia/ archive, so finding one under goldens/ means the working tree is
# in a state this script should not touch.

set -euo pipefail

cd "$(dirname "$0")"

if [ $# -ne 1 ]; then
	echo "usage: $0 RUN_ID   (a capture-goldens.yml run id; see: gh run list --workflow=capture-goldens.yml)" >&2
	exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

gh run download "$1" --pattern 'goldens-*' --dir "$tmp"

shopt -s nullglob
merged=0
for artifact in "$tmp"/goldens-*/; do
	for dir in "$artifact"*/*/; do # <lane>/<GOOS_GOARCH>/ directories within one leg's artifact
		rel=${dir#"$artifact"}
		rel=${rel%/}
		dest="goldens/$rel"
		if [ -f "$dest/manifest.json" ] && grep -q '"schema": 1' "$dest/manifest.json"; then
			echo "refusing: $dest holds a schema-1 (Skia-era) golden set." >&2
			echo "Skia-era sets belong only in the goldens-skia/ archive; finding one under goldens/ means the" >&2
			echo "working tree is in an unexpected state. Nothing has been overwritten." >&2
			exit 1
		fi
	done
done
for artifact in "$tmp"/goldens-*/; do
	# A goldens-<GOOS_GOARCH> artifact is the capture leg's whole goldens/ subtree, which includes checkout copies of
	# every OTHER platform's committed sets (on Windows legs additionally CRLF-mangled by the runner's autocrlf
	# checkout). Only the artifact's own platform directories are that leg's fresh bless output — merge exactly those
	# and ignore the rest, so a later artifact can never clobber an earlier platform's fresh set with a stale copy.
	platform=$(basename "$artifact")
	platform=${platform#goldens-}
	for dir in "$artifact"*/*/; do
		rel=${dir#"$artifact"}
		rel=${rel%/}
		if [ "$(basename "$rel")" != "$platform" ]; then
			continue # another platform's committed set riding along in this leg's checkout
		fi
		dest="goldens/$rel"
		rm -rf "$dest"
		mkdir -p "$(dirname "$dest")"
		cp -R "${dir%/}" "$dest"
		echo "merged $rel"
		merged=$((merged + 1))
	done
done

if [ "$merged" -eq 0 ]; then
	echo "no golden sets found in run $1's goldens-* artifacts" >&2
	exit 1
fi
echo "merged $merged lane/platform set(s) into goldens/ — review with git status / git diff, then commit manually"
