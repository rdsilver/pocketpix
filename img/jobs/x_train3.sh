# Distillation, step 4c: training files for TAG, then the stronger recipe (d: lr 5e-4, matrices x5, 30 passes, FLUX + general
# photos) on all three models, scores, sheet, bundle.
exec 2>&1
cd "$(dirname "$0")/../.."
TAG=${TAG:-t2}
python3 -m img.distill build img/data/xjobs_$TAG.jsonl img/data/xgen_$TAG $TAG 2>&1 | tail -3
g=0; for m in b16C u32M u64M; do
  rm -rf img/runs/${m}_${TAG}d
  python3 -m img.distill train $m ${TAG}m $g epochs=30 lr=5e-4 wq_lr_mult=5 name=\"${m}_${TAG}d\" > img/runs/xtrain_${m}_${TAG}d.out 2>&1 &
  g=$((g + 1))
done
wait
CUDA_VISIBLE_DEVICES=0 python3 -m img.distill score $TAG b16C b16C_${TAG}d &
CUDA_VISIBLE_DEVICES=1 python3 -m img.distill score $TAG u32M u32M_${TAG}d &
CUDA_VISIBLE_DEVICES=2 python3 -m img.distill score $TAG u64M u64M_${TAG}d &
CUDA_VISIBLE_DEVICES=3 python3 -m img.distill sheet img/runs/x${TAG}d_new.png b16C_${TAG}d u32M_${TAG}d u64M_${TAG}d n=8 2>&1 | tail -1 &
wait
python3 -m img.distill bundle ${TAG}d 2>&1 | tail -1
date
