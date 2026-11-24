# Distillation, step 3: the pictures, one FLUX process per GPU. TAG, PER (pictures per word), WORDS (trial / all / a,b,c), RES.
exec 2>&1
cd "$(dirname "$0")/../.." && mkdir -p img/data img/runs
TAG=${TAG:-t1}; PER=${PER:-64}; WORDS=${WORDS:-trial}; RES=${RES:-384}; N=$(nvidia-smi -L | wc -l)
[ -f img/data/xjobs_$TAG.jsonl ] || python3 -m img.distill prompts img/data/xjobs_$TAG.jsonl $PER $WORDS 1 || exit 1
for g in $(seq 0 $((N - 1))); do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g nohup python3 -m img.flux_gen img/data/xjobs_$TAG.jsonl img/data/xgen_$TAG $g $N res=$RES batch=8 chunk=800 > img/runs/xgen_${TAG}_$g.out 2>&1 &
done
wait
grep -h "chunk\|Error\|error\|Killed" img/runs/xgen_${TAG}_*.out | tail -12
python3 -m img.distill contact img/data/xgen_$TAG img/data/xjobs_$TAG.jsonl img/runs/x${TAG}_contact.png 240
date
