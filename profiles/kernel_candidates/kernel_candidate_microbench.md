# Kernel Candidate Microbench

| library | op | shape | baseline ms | candidate ms | speedup | max rel err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| liger | rms_norm | (1024, 768) | 0.0466 | 0.0813 | 0.573 | 0.00596 |
| liger | layer_norm | (1024, 768) | 0.0116 | 0.0606 | 0.191 | 0.00524 |
| liger | rms_norm | (8192, 768) | 0.2018 | 0.1086 | 1.858 | 0.00596 |
| liger | layer_norm | (8192, 768) | 0.0561 | 0.0602 | 0.932 | 0.0132 |
| liger | rms_norm | (1024, 3072) | 0.1035 | 0.0822 | 1.259 | 0.00719 |
| liger | layer_norm | (1024, 3072) | 0.0343 | 0.0598 | 0.574 | 0.00596 |
| liger | rms_norm | (8192, 3072) | 0.7553 | 0.4187 | 1.804 | 0.00596 |
| liger | layer_norm | (8192, 3072) | 0.2991 | 0.2104 | 1.421 | 0.0153 |
| liger | swiglu | (1024, 3072) | 0.0692 | 0.0444 | 1.559 | 0.00118 |
| liger | swiglu | (8192, 3072) | 0.5218 | 0.3144 | 1.660 | 0.00137 |
| liger | softmax | (1024, 50272) | 0.7039 | 0.8569 | 0.821 | 0.00596 |
| liger | softmax | (128, 50272) | 0.0810 | 0.1098 | 0.737 | 0.00596 |
| hf_kernels | silu_and_mul | (1024, 3072) | 0.0745 | 0.0422 | 1.766 | 0 |
| hf_kernels | gelu_tanh_and_mul | (1024, 3072) | 0.0744 | 0.0422 | 1.761 | 0 |
| hf_kernels | silu_and_mul | (8192, 3072) | 0.5362 | 0.3193 | 1.679 | 0 |
| hf_kernels | gelu_tanh_and_mul | (8192, 3072) | 0.5410 | 0.3260 | 1.660 | 0 |
| quack | rms_norm | (1024, 768) | 0.0460 | 0.0661 | 0.697 | 0.00596 |
| quack | rms_norm | (8192, 768) | 0.2048 | 0.0644 | 3.180 | 0.00596 |
| quack | rms_norm | (1024, 3072) | 0.1055 | 0.0640 | 1.647 | 0.00596 |
| quack | rms_norm | (8192, 3072) | 0.7565 | 0.2105 | 3.594 | 0.00596 |

## Bench Failures
- `quack` `softmax` (1024, 50272): `RuntimeError('CUDA Error: cudaErrorInvalidValue')`
- `quack` `softmax` (128, 50272): `RuntimeError('CUDA Error: cudaErrorInvalidValue')`

## Import/Load Status
- `humming` : imported
  - attrs: `config, dtypes, humming, jit, kernel, ops, tune, utils`
- `quack` : imported
  - attrs: `RoundingMode, autotuner, bench, cache, compile_utils, copy_utils, cross_entropy, cute_dsl_utils, dsl, layout_utils, os, reduce, reduction_base, rmsnorm, rmsnorm_config, rounding, softmax, utils`
- `tokenspeed_mla` : imported
  - attrs: `fmha, fmha_binary, fmha_helpers, get_num_sm, has_binary_prefill, mla_decode, mla_decode_fp16, mla_decode_fp8, mla_helpers, mla_kv_pack_quantize_fp8, mla_prefill, tokenspeed_mla_decode, tokenspeed_mla_prefill, utils, warmup_compile_prefill`
- `tokenspeed_triton` : imported
  - attrs: `AsyncCompileMode, CompilationError, Config, FutureKernel, InterpreterError, JITFunction, KernelInterface, MockTensor, OutOfResources, TensorWrapper, TritonError, aggregate, autotune, backends, cdiv, compile, compiler, constexpr_function, errors, heuristics`
- `cutlass` : imported
  - attrs: `BFloat16, Boolean, CACHE_FILE, CUDA_VERSION, ComposedLayout, Constexpr, Coord, DSLAstPreprocessorError, DSLCudaVersion, DSLRuntimeError, Float, Float16, Float32, Float4E2M1FN, Float64, Float6E2M3FN, Float6E3M2FN, Float8E4M3, Float8E4M3B11FNUZ, Float8E4M3FN`
- `hf_kernels` kernels-community/activation: loaded
  - attrs: `fatrelu_and_mul, gelu, gelu_and_mul, gelu_fast, gelu_new, gelu_quick, gelu_tanh, gelu_tanh_and_mul, layers, mul_and_silu, ops, silu, silu_and_mul, torch`
- `hf_kernels` kernels-community/triton_kernels: load_error
  - error: `HfHubHTTPError("Client error '401 Unauthorized' for url 'https://huggingface.co/api/kernels/kernels-community/triton_kernels/refs' (Request ID: Root=1-6a37bf7e-6f584caf20c5de8d6b24f825;4fbc87c8-85fa-471d-a96b-d2a8c77d2ccc)\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401\n\nInvalid username or password.")`
