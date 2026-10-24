#!/bin/bash
# Regenerates freetype.sums from a FreeType build, through freetype-py (pip install freetype-py).
# Usage: gensums.sh <python with freetype-py> > freetype.sums
set -e
here="$(cd "$(dirname "$0")" && pwd)"
py="${1:-python}"
tmp="$(mktemp)"
echo "# sha256 of ftdump.py's output: every glyph's hinted outline as FreeType $("$py" -c 'import freetype; print(".".join(map(str, freetype.version())))') loads it."
echo "# font ppem target sha256"
for font in Roboto-Regular.ttf DejaVuSans.subset.ttf; do
	for target in normal lcd; do
		for ppem in 9 10 11 12 13 14 15 16 17 18 20 24 28 36 48 72; do
			"$py" "$here/ftdump.py" "$here/../../../testdata/$font" "$ppem" "$target" "$tmp"
			echo "$font $ppem $target $(sha256sum "$tmp" | cut -d' ' -f1)"
		done
	done
done
rm -f "$tmp"
