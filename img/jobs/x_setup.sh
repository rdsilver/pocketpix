# Distillation, step 1 (on a GPU machine, from the repository root): packages, checks, the FLUX.1-schnell download.
exec 2>&1
cd "$(dirname "$0")/../.." && mkdir -p img/data img/runs bundle || exit 1
# RTX 50 cards need a PyTorch built for CUDA 12.8 or newer: if the image's PyTorch cannot run on them, get one that can
python3 -c "import torch; torch.zeros(1).cuda(); print('pytorch runs on', torch.cuda.get_device_name(0))" || \
  pip install -q -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
python3 -c "import torch; torch.zeros(1).cuda(); print('pytorch', torch.__version__, 'cuda', torch.version.cuda, 'ok')" || { echo "PYTORCH CANNOT USE THE GPUS"; exit 1; }
pip install -q -U diffusers transformers accelerate sentencepiece protobuf huggingface_hub 2>&1 | grep -v -i "warning\|notice" | tail -3
python3 -c "import torch, diffusers; print('torch', torch.__version__, 'diffusers', diffusers.__version__, 'gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h . | tail -1; free -g | head -2
# FLUX.1-schnell's diffusers files from an ungated copy, each checked against the official fingerprints (no login needed)
python3 -m img.flux_gen fetch || { echo "FLUX DOWNLOAD OR CHECK FAILED"; exit 1; }
python3 -m img.distill photoval || exit 1
python3 -m img.tests 2>&1 | tail -2
date
