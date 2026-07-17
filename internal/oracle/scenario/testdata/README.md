# Corpus text fixtures

`Roboto-Regular.ttf` is a copy of `font/testdata/Roboto-Regular.ttf` (Roboto, Apache License 2.0 — see the provenance
note in that directory's README). It is embedded into the scenario package (`scenario.FontData`) so the text
scenarios resolve the same font bytes in every consumer — the `oracle` CLI and the gorender golden gates — independent
of the working directory.

`sbix.ttf` and `colr.ttf` are copies of the same-named Skia-authored color test fonts in `font/testdata/` (BSD-3-Clause;
see that README): each maps U+1F600 to a color glyph (PNG strikes at 16/64/128 ppem in the sbix font, a five-layer
COLRv0/CPAL glyph in the colr font). They back the emoji scenarios through `scenario.FontDataFor`.
