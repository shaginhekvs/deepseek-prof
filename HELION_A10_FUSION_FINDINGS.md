# Helion A10 Fusion Findings

This note answers a narrower question than the generic Helion lab: what did Helion actually find on this legacy-ish A10 (`sm_86`) box that is useful for bottleneck-adjacent vLLM work?

The useful area is not plain GEMM. Plain Helion GEMM lost to `torch.mm`/cuBLAS. The useful area is fused elementwise work around MLP/residual/norm paths, where vLLM may otherwise pay extra launches and global-memory round trips.

## Run

```bash
source /scratch/deepseek-prof/env/scratch_env.sh
source /scratch/deepseek-prof/env/py312/bin/activate
python /scratch/deepseek-prof/scripts/helion_fusion_bottleneck_lab.py --warmup 10 --iters 50
```

Artifacts:

- `/scratch/deepseek-prof/profiles/helion_fusion/helion_fusion_bottleneck_lab.md`
- `/scratch/deepseek-prof/profiles/helion_fusion/helion_fusion_bottleneck_lab.json`
- `/scratch/deepseek-prof/profiles/helion_fusion/8192_768_*_autotune.csv`
- `/scratch/deepseek-prof/profiles/helion_fusion/8192_3072_*_autotune.csv`

For fresh autotune snapshots:

```bash
HELION_FORCE_AUTOTUNE=1 HELION_SNAPSHOT_AUTOTUNE=1 \
python /scratch/deepseek-prof/scripts/helion_fusion_bottleneck_lab.py \
  --warmup 10 --iters 50 --shapes 8192x768 8192x3072
```

## What Improved

### `add + RMSNorm`

This is the best Helion fusion candidate from the current run.

| shape | Torch composite | Helion | speedup |
| ---: | ---: | ---: | ---: |
| `(1024, 768)` | `0.0531 ms` | `0.0382 ms` | `1.39x` |
| `(8192, 768)` | `0.2810 ms` | `0.0784 ms` | `3.58x` |
| `(1024, 3072)` | `0.1430 ms` | `0.0408 ms` | `3.50x` |
| `(8192, 3072)` | `1.0717 ms` | `0.2967 ms` | `3.61x` |

For `8192x768`, Helion selected:

```python
helion.Config(
    block_sizes=[16],
    indexing=[
        'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer',
        'pointer', 'pointer', 'pointer', 'pointer', 'pointer',
    ],
    num_stages=1,
    num_warps=1,
    pid_type='flat',
    reduction_loops=[None],
)
```

For `8192x3072`, Helion selected a more aggressive A10-specific config:

```python
helion.Config(
    block_sizes=[1],
    indexing=[
        'block_ptr', 'pointer', 'pointer', 'pointer', 'pointer',
        'pointer', 'block_ptr', 'pointer', 'pointer', 'block_ptr',
    ],
    load_eviction_policies=['first', '', '', 'last', '', '', '', ''],
    num_stages=3,
    num_warps=32,
    pid_type='flat',
    reduction_loops=[None],
)
```

Interpretation: Helion learned different strategies for different hidden sizes:

- For hidden `768`, it processes more rows per program with low warp pressure.
- For hidden `3072`, it goes almost row-at-a-time, uses many warps, and changes several operands to block-pointer indexing.
- That is the kind of shape-specific A10 tuning that does not fall out of a generic Torch composite.

### `silu * gate + residual`

Helion also improved over the raw Torch composite for larger shapes, but the existing HF activation kernel remains competitive.

| shape | Torch composite | HF activation + add | Helion | Helion vs Torch |
| ---: | ---: | ---: | ---: | ---: |
| `(1024, 768)` | `0.0270 ms` | `0.0190 ms` | `0.0366 ms` | `0.74x` |
| `(8192, 768)` | `0.2193 ms` | `0.1555 ms` | `0.0997 ms` | `2.20x` |
| `(1024, 3072)` | `0.1164 ms` | `0.0816 ms` | `0.0538 ms` | `2.16x` |
| `(8192, 3072)` | `0.8488 ms` | `0.6258 ms` | `0.4021 ms` | `2.11x` |

For `8192x768`, Helion selected:

```python
helion.Config(
    block_sizes=[32, 64],
    loop_orders=[[1, 0]],
    num_warps=32,
    pid_type='flat',
)
```

For `8192x3072`, Helion selected:

```python
helion.Config(
    block_sizes=[32, 512],
    indexing=['pointer', 'block_ptr', 'pointer', 'block_ptr'],
    loop_orders=[[1, 0]],
    num_warps=4,
    pid_type='persistent_blocked',
)
```

Interpretation:

- Wider hidden sizes made Helion choose much wider column tiles.
- It switched to persistent scheduling for the large hidden case.
- It used block-pointer indexing for the large hidden case.
- Some larger persistent variants timed out during compile, so the search space needs guardrails on this machine.

## What Did Not Work

The larger fusion:

```text
silu * gate + residual + RMSNorm
```

failed to find a working config for every tested shape. Keep it as a future Helion/compiler experiment, not a vLLM integration target yet.

## Practical Recommendation

Best next vLLM experiment:

1. Find the actual residual-add/RMSNorm op in the DeepSeek/vLLM compiled path.
2. Add an env-gated Helion replacement for only that op.
3. Re-run the realistic serving matrix and compare TPOT, TTFT, output tokens/sec, and Nsight kernel share.

Do not integrate Helion plain GEMM. Do consider Helion for `add + RMSNorm`, because this is a real fused-op win on this A10 setup.
