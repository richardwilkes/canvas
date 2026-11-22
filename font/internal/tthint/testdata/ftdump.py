# Dumps every glyph's hinted outline as FreeType loads it: ftdump.py font ppem normal|lcd out
import sys, freetype
from freetype import *
font, ppem, target, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
face = freetype.Face(font)
face.set_pixel_sizes(0, ppem)
flags = FT_LOAD_NO_BITMAP | (FT_LOAD_TARGET_LCD if target == 'lcd' else FT_LOAD_TARGET_NORMAL)
with open(out, 'w', newline='\n') as f:
    for gid in range(face.num_glyphs):
        try:
            face.load_glyph(gid, flags)
        except Exception:
            f.write('%d error\n' % gid)
            continue
        o = face.glyph.outline
        pts, tags = o.points, o.tags
        f.write('%d %d %d:' % (gid, len(pts), len(o.contours)))
        for (x, y), t in zip(pts, tags):
            f.write(' %d,%d,%d' % (x, y, t & 1))
        f.write('\n')
