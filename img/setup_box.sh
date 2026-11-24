#!/bin/bash
# On a GPU machine, from the directory that holds img/ and train/:   bash img/setup_box.sh
# Runs the checks, then downloads and prepares the photo set (SPRITES=1 also prepares the two sprite sets).
set -e
pip install -q numpy pillow pyarrow huggingface_hub
python3 -m img.tests
# photos: 240 shards = about 1M pictures, about 95 GB passes through the disk 16 shards (6.5 GB) at a time
[ -f img/data/photos64/meta.json ] || python3 -m img.data --wds data-archetype/commoncatalog_cc_by_3m_512 --shards ${SHARDS:-240} --S 64 --out img/data/photos64 --half img/data/photos32
[ -z "$SPRITES" ] && { echo ready; exit 0; }
[ -f img/data/atlas16/meta.json ] || python3 -m img.data --hf Angais/pixel-art-atlas-1.4m --caption-col caption --out img/data/atlas16 --S 16
[ -f img/data/icons32/meta.json ] || python3 -m img.data --hf azyg0s/pixelart-172k --caption-col text --out img/data/icons32 --S 32
echo ready
