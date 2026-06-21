# Source Revisions

Large upstream source clones are not vendored in this repo. Recreate them under `/scratch/deepseek-prof/src` with the revisions below.

## vLLM

- Remote: `https://github.com/vllm-project/vllm.git`
- Branch at capture time: `main`
- Commit: `b80ce9dd2f30913b3b054308a09bd2d86ec6202f`
- PR ref fetched locally during the lab: `pr-40760-deepseek-v4`

## PyTorch

- Remote: `https://github.com/pytorch/pytorch.git`
- Branch at capture time: `main`
- Commit: `eefb217f761df28eec3f4ec47a81cbeb776cae65`

## Installed Runtime

- Python venv path: `/scratch/deepseek-prof/env/py312`
- PyTorch: `2.11.0+cu128`
- vLLM editable install: `0.23.1rc1.dev226+gb80ce9dd2`
- GPUs used: 4x NVIDIA A10 on one host
