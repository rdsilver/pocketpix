"""Build build/pocketpix.gba (the game). Models in the blob: objects base, 32 enlarger, 64 enlarger, scenes base."""
import subprocess, os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # the repository root
OUT = 'build/pocketpix'
if 'test' in sys.argv[1:]: OUT = 'build/pocketpix_galtest'      # gallery tests only: pictures are made up at once, not painted
CF = '-mcpu=arm7tdmi -mtune=arm7tdmi -marm -O2 -ffreestanding -fno-builtin -Wall -Igen -Ipix'.split() + (['-DGAL_TEST'] if 'test' in sys.argv[1:] else [])
def sh(*a): subprocess.run(a, check=True)
sh('arm-none-eabi-gcc', *CF, '-c', 'pix/pix.c', '-o', 'build/pix_engine.o')
sh('arm-none-eabi-gcc', *CF, '-nostdlib', '-Wl,-T,pix/gba.ld', '-Wl,-Map,' + OUT + '.map', '-o', OUT + '.elf',
   'pix/crt0.s', 'pix/game.c', 'build/pix_engine.o', 'pix/kernels.s', 'pix/blob.s', '-lgcc')
sh('arm-none-eabi-objcopy', '-O', 'binary', OUT + '.elf', OUT + '.gba')
sh('python3', 'tools/fixrom.py', OUT + '.gba')
sh('arm-none-eabi-size', OUT + '.elf')
