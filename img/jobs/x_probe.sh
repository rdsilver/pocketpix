# Distillation, step 2: how fast FLUX draws on one GPU (48 pictures at 512 and at 384, with the memory peak), and a look.
exec 2>&1
cd "$(dirname "$0")/../.." && mkdir -p img/data img/runs
python3 -m img.distill prompts img/data/xprobe.jsonl 4 trial 7 || exit 1
head -n 48 img/data/xprobe.jsonl > img/data/xprobe48.jsonl
for r in 512 384; do
  rm -rf img/data/xprobe_$r
  CUDA_VISIBLE_DEVICES=0 python3 -m img.flux_gen img/data/xprobe48.jsonl img/data/xprobe_$r 0 1 res=$r batch=8 chunk=24 2>&1 | grep -v -i "warn\|loading" | tail -4
  python3 -m img.distill contact img/data/xprobe_$r img/data/xprobe48.jsonl img/runs/xprobe_$r.png 48
done
date
