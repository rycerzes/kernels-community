---
library_name: kernels
license: apache-2.0
tags:
  - kernel
---

## ffpa-attn

Attention for large head dims (D > 256) from [FFPA](https://github.com/xlite-dev/ffpa-attn), forward and
backward, with grouped KV heads, causal masking and a varlen (packed) path. The CuTe DSL backend covers
compute capability 8.0 and newer (with dedicated head_dim 512 kernels on SM90 and SM100); the Triton backend is
upstream's default dense forward. Head dims of 256 and below are handed to PyTorch SDPA, as upstream does.

The vendored sources are the ffpa-attn 0.2.4 release (upstream commit `918ae3c`) and the quack 0.6.4 modules
it imports; `vendor.py` reproduces them. Not vendored: upstream's CLI, benchmark entry point and Ray-based
offline autotuner. Its optional compiled CUDA backend is not built, so `CUDABackend` reports unavailable.

## Usage

```python
import torch
from kernels import get_kernel

ffpa_attn = get_kernel("kernels-community/ffpa-attn", version=1)

q = torch.randn(1, 8, 4096, 512, device="cuda", dtype=torch.bfloat16)  # (batch, heads, seq, head_dim)
k = torch.randn(1, 2, 4096, 512, device="cuda", dtype=torch.bfloat16)  # 2 KV heads
v = torch.randn(1, 2, 4096, 512, device="cuda", dtype=torch.bfloat16)
out = ffpa_attn.ffpa_attn_func(q, k, v, is_causal=True, enable_gqa=True, backend="cutedsl")
```

`ffpa_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)` takes packed
`(tokens, heads, head_dim)` inputs, with the FlashAttention varlen calling convention.

## Benchmarks

H200, PyTorch 2.15, bf16, 8 query heads and 2 KV heads at head_dim 512 (the full-attention layers of
Gemma 4 E4B), forward + backward:

| | SDPA | ffpa-attn (CuTe DSL) |
| --- | ---: | ---: |
| causal, 11k tokens | 87.5 ms | 13.3 ms |
| causal, 20k tokens | 257.0 ms | 41.2 ms |
| 3 packed documents, 11k tokens | 150.4 ms (block-diagonal mask) | 5.5 ms (varlen) |

Errors against an fp32 reference were at or below bf16 SDPA's for the output and dQ/dK/dV.
