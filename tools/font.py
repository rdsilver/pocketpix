"""font8x8 (public domain, Daniel Hepper) -> gen/font.h, glyphs 0x20..0x7F."""
import re
src = open('dl/font/font8x8_basic.h').read()
rows = re.findall(r'\{\s*((?:0x[0-9A-Fa-f]{2},?\s*){8})\}', src)
assert len(rows) == 128
out = ['static const unsigned char FONT[96][8] = {']
for r in rows[32:128]:
    out.append('  {' + ','.join(re.findall(r'0x[0-9A-Fa-f]{2}', r)) + '},')
out.append('};')
open('gen/font.h', 'w').write('\n'.join(out) + '\n')
