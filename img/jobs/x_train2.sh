# Distillation, step 4b: a stronger fine-tune. d: all three models, lr 5e-4 with the matrices x5, 30 passes, FLUX + general
# photos; e: the base only, the same without the photos (GPU 3). Then scores, sheets, bundle.
exec 2>&1
cd "$(dirname "$0")/../.."
TAG=${TAG:-t1}
for m in b16C u32M u64M; do rm -rf img/runs/${m}_${TAG}d; done; rm -rf img/runs/b16C_${TAG}e
python3 -m img.distill train b16C ${TAG}m 0 epochs=30 lr=5e-4 wq_lr_mult=5 name=\"b16C_${TAG}d\" > img/runs/xtrain_b16C_${TAG}d.out 2>&1 &
python3 -m img.distill train u32M ${TAG}m 1 epochs=30 lr=5e-4 wq_lr_mult=5 name=\"u32M_${TAG}d\" > img/runs/xtrain_u32M_${TAG}d.out 2>&1 &
python3 -m img.distill train u64M ${TAG}m 2 epochs=30 lr=5e-4 wq_lr_mult=5 name=\"u64M_${TAG}d\" > img/runs/xtrain_u64M_${TAG}d.out 2>&1 &
python3 -m img.distill train b16C ${TAG} 3 epochs=30 lr=5e-4 wq_lr_mult=5 name=\"b16C_${TAG}e\" > img/runs/xtrain_b16C_${TAG}e.out 2>&1 &
wait
for r in b16C_${TAG}d u32M_${TAG}d u64M_${TAG}d b16C_${TAG}e; do grep val_bits img/runs/$r/log.jsonl | tr '\n' ' ' | cut -c1-400; echo; done
CUDA_VISIBLE_DEVICES=0 python3 -m img.distill score $TAG b16C b16C_${TAG}d b16C_${TAG}e &
CUDA_VISIBLE_DEVICES=1 python3 -m img.distill score $TAG u32M u32M_${TAG}d &
CUDA_VISIBLE_DEVICES=2 python3 -m img.distill score $TAG u64M u64M_${TAG}d &
wait
CUDA_VISIBLE_DEVICES=0 python3 -m img.distill sheet img/runs/x${TAG}d_new.png b16C_${TAG}d u32M_${TAG}d u64M_${TAG}d n=8 2>&1 | tail -1 &
CUDA_VISIBLE_DEVICES=1 python3 -m img.distill sheet img/runs/x${TAG}e_new.png b16C_${TAG}e u32M_${TAG}d u64M_${TAG}d n=8 2>&1 | tail -1 &
wait
mkdir -p bundle/x${TAG}d; cp img/runs/b16C_${TAG}e/best.pt bundle/x${TAG}d/b16C_e.pt; cp img/runs/x${TAG}e_new.png bundle/x${TAG}d/
python3 -m img.distill bundle ${TAG}d 2>&1 | tail -1
date
