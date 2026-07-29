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
# removed first so stale scenarios cannot linger.

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
	# An artifact holds only the lanes its leg actually blessed, staged as <lane>/<GOOS_GOARCH>/ (the workflow's
	# staging step filters on the captured-lane list, so a lane that skipped or failed contributes nothing rather than
	# its stale committed set). The per-platform filter below is the second guard, against an artifact built before
	# that filter existed or by a hand-run leg: only the artifact's own platform directories can be its bless output,
	# so a later artifact can never clobber an earlier platform's fresh set with a stale copy.
	platform=$(basename "$artifact")
	platform=${platform#goldens-}
	# Each leg records its own account of what it captured, skipped and failed. Print it: a leg missing a lane is the
	# thing a reviewer most needs to notice before committing, and it is invisible in `git diff`.
	if [ -f "$artifact/CAPTURED.txt" ]; then
		echo "leg $(cat "$artifact/CAPTURED.txt")"
	else
		echo "leg $platform: no CAPTURED.txt (artifact predates the captured-lane filter; verify its lanes by hand)"
	fi
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
