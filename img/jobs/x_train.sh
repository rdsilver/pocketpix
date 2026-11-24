# Distillation, step 4: training files; three fine-tunes of the three cartridge models, three GPUs at a time:
#   a: all weights, gently (lr 2e-4, matrices x2), on the FLUX pictures plus general photos with empty prompts (data <TAG>m)
#   b: matrices frozen, only word tables / norms / gains (lr 5e-4), FLUX pictures only
#   c: as a, FLUX pictures only
# then scores before/after on the FLUX held-out pictures and on general photos, old/new sheets, bundles. TAG as in step 3.
exec 2>&1
cd "$(dirname "$0")/../.." && mkdir -p img/data img/runs bundle
TAG=${TAG:-t1}; EP=${EPOCHS:-10}
python3 -m img.distill build img/data/xjobs_$TAG.jsonl img/data/xgen_$TAG $TAG 2>&1 | tail -4
[ -f img/data/pv64/meta.json ] || python3 -m img.distill photoval
CUDA_VISIBLE_DEVICES=3 python3 -m img.distill sheet img/runs/x${TAG}_old.png b16C u32M u64M n=8 2>&1 | tail -1 &
for arm in a b c; do
  case $arm in a) D=${TAG}m; A="lr=2e-4 wq_lr_mult=2";; b) D=$TAG; A="lr=5e-4 wq_lr_mult=0";; c) D=$TAG; A="lr=2e-4 wq_lr_mult=2";; esac
  g=0; for m in b16C u32M u64M; do
    rm -rf img/runs/${m}_$TAG$arm
    python3 -m img.distill train $m $D $g epochs=$EP $A name=\"${m}_$TAG$arm\" > img/runs/xtrain_${m}_$TAG$arm.out 2>&1 &
    g=$((g + 1))
  done
  wait
  for m in b16C u32M u64M; do head -n 1 img/runs/xtrain_${m}_$TAG$arm.out | cut -c1-200; tail -n 2 img/runs/xtrain_${m}_$TAG$arm.out | cut -c1-300; done
done
CUDA_VISIBLE_DEVICES=0 python3 -m img.distill score $TAG b16C b16C_${TAG}a b16C_${TAG}b b16C_${TAG}c &
CUDA_VISIBLE_DEVICES=1 python3 -m img.distill score $TAG u32M u32M_${TAG}a u32M_${TAG}b u32M_${TAG}c &
CUDA_VISIBLE_DEVICES=2 python3 -m img.distill score $TAG u64M u64M_${TAG}a u64M_${TAG}b u64M_${TAG}c &
wait
CUDA_VISIBLE_DEVICES=0 python3 -m img.distill sheet img/runs/x${TAG}a_new.png b16C_${TAG}a u32M_${TAG}a u64M_${TAG}a n=8 2>&1 | tail -1 &
CUDA_VISIBLE_DEVICES=1 python3 -m img.distill sheet img/runs/x${TAG}b_new.png b16C_${TAG}b u32M_${TAG}b u64M_${TAG}b n=8 2>&1 | tail -1 &
CUDA_VISIBLE_DEVICES=2 python3 -m img.distill sheet img/runs/x${TAG}c_new.png b16C_${TAG}c u32M_${TAG}c u64M_${TAG}c n=8 2>&1 | tail -1 &
wait
for arm in a b c; do cp img/runs/x${TAG}_old.png img/runs/x$TAG${arm}_old.png; python3 -m img.distill bundle $TAG$arm 2>&1 | tail -1; done
date
