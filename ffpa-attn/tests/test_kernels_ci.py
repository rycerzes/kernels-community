"""Fast smoke tests for the kernels-community CI runner (an sm89 L4).

Small shapes at head_dim 512 with grouped KV heads, the case ffpa-attn exists for (e.g. the full-attention
layers of Gemma 4). Each bf16 result is compared with an fp32 SDPA reference on the same inputs.
"""

import kernels
import pytest
import torch
import torch.nn.functional as F

ffpa_attn = kernels.get_kernel("kernels-community/ffpa-attn", version=1)

supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not supported, reason="ffpa-attn needs a CUDA device with compute capability >= 8.0"),
]

HQ, HKV, D = 8, 2, 512


def rel(ref, got):
    return ((got.float() - ref).norm() / ref.norm()).item()


def reference(q, k, v, causal, scale):
    """fp32 SDPA on (B, H, L, D), KV heads expanded."""
    n = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(n, dim=1) for x in (k, v))
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def fwd_bwd(fn, q, k, v, g):
    q, k, v = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    out = fn(q, k, v)
    out.backward(g)
    return out.detach(), q.grad, k.grad, v.grad


def inputs(shape_q, shape_kv, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(*shape_q, device="cuda") * 0.5
    k = torch.randn(*shape_kv, device="cuda") * 0.5
    v = torch.randn(*shape_kv, device="cuda")
    g = torch.randn(*shape_q, device="cuda")
    return q, k, v, g


@pytest.mark.parametrize("causal", [True, False])
def test_dense_cutedsl_forward_backward(causal):
    q, k, v, g = inputs((1, HQ, 384, D), (1, HKV, 384, D))
    ref = fwd_bwd(lambda q, k, v: reference(q, k, v, causal, 1.0), q, k, v, g)
    got = fwd_bwd(lambda q, k, v: ffpa_attn.ffpa_attn_func(q, k, v, is_causal=causal, scale=1.0, enable_gqa=True,
                                                          backend="cutedsl"),
                  *(x.bfloat16() for x in (q, k, v, g)))
    for name, r, x in zip(("out", "dq", "dk", "dv"), ref, got):
        assert rel(r, x) < 2e-2, name


def test_dense_default_backend_forward():
    q, k, v, _ = inputs((1, HQ, 256, D), (1, HKV, 256, D))
    ref = reference(q, k, v, True, None)
    out = ffpa_attn.ffpa_attn_func(*(x.bfloat16() for x in (q, k, v)), is_causal=True, enable_gqa=True)
    assert rel(ref, out) < 2e-2


hopper_or_newer = supported and torch.cuda.get_device_capability()[0] >= 9


@pytest.mark.parametrize("hkv", [
    HQ,
    pytest.param(HKV, marks=pytest.mark.xfail(
        not hopper_or_newer, strict=True, raises=NotImplementedError,
        reason="upstream: ffpa_attn_varlen_func turns pack_gqa on for grouped KV heads, which the SM80/SM89 "
               "forward rejects")),
])
def test_varlen_cutedsl_forward_backward(hkv):
    lens = [128, 256, 192]
    t = sum(lens)
    q, k, v, g = inputs((t, HQ, D), (t, hkv, D))
    cu = torch.tensor([0, 128, 384, 576], device="cuda", dtype=torch.int32)

    def per_document(q, k, v):
        outs, s = [], 0
        for n in lens:
            sl = slice(s, s + n)
            o = reference(q[sl].transpose(0, 1)[None], k[sl].transpose(0, 1)[None], v[sl].transpose(0, 1)[None],
                          True, 1.0)
            outs.append(o[0].transpose(0, 1))
            s += n
        return torch.cat(outs)

    ref = fwd_bwd(per_document, q, k, v, g)
    got = fwd_bwd(lambda q, k, v: ffpa_attn.ffpa_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens),
                                                                 softmax_scale=1.0, causal=True,
                                                                 enable_gqa=hkv != HQ),
                  *(x.bfloat16() for x in (q, k, v, g)))
    for name, r, x in zip(("out", "dq", "dk", "dv"), ref, got):
        assert rel(r, x) < 2e-2, name


def test_varlen_rejects_sliding_window():
    q, k, v, _ = inputs((128, HQ, D), (128, HKV, D))
    cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        ffpa_attn.ffpa_attn_varlen_func(*(x.bfloat16() for x in (q, k, v)), cu, cu, 128, 128, causal=True,
                                        enable_gqa=True, window_size=(63, 0))
