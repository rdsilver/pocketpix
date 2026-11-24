"""Pad the ROM and write the header checksum."""
import sys
d = bytearray(open(sys.argv[1], 'rb').read())
d[0xBD] = (-(sum(d[0xA0:0xBD]) + 0x19)) & 0xFF
while len(d) % 4096:
    d.append(0xFF)
open(sys.argv[1], 'wb').write(d)
print(f'{sys.argv[1]}: {len(d)} bytes ({len(d) / 1024:.0f} KB)')
