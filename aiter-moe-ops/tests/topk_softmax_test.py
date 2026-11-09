"""Tests for AITER's topk_softmax MoE router.

Loaded through ``kernels.get_kernel``; in CI ``LOCAL_KERNELS`` points it at the fresh
build. Run the cheap subset with ``pytest -m kernels_ci``.
"""

import kernels
import pytest
import torch

moe = kernels.get_kernel("amd/aiter-moe-ops", version=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm GPU"
)


def logits(num_tokens, num_experts, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(num_tokens, num_experts, device="cuda", generator=g).to(dtype)


def reference(gating, topk, renormalize):
    probs = gating.float().softmax(dim=-1)
    weights, ids = probs.topk(topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return probs, weights, ids


def check(gating, topk, renormalize, out):
    probs, ref_w, ref_ids = reference(gating, topk, renormalize)
    ids = out.topk_ids.long()
    assert out.topk_weights.dtype == torch.float32 and out.topk_ids.dtype == torch.int32
    assert ((ids >= 0) & (ids < gating.shape[-1])).all()
    assert (
        ids.sort(dim=-1).values.diff(dim=-1) != 0
    ).all(), "an expert was picked twice"
    # Low-precision logits can tie, so compare the chosen experts' probabilities
    # rather than the ids themselves.
    torch.testing.assert_close(
        probs.gather(-1, ids), probs.gather(-1, ref_ids), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(out.topk_weights, ref_w, atol=1e-5, rtol=1e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_matches_reference(dtype, renormalize):
    gating = logits(257, 64, dtype)
    check(gating, 8, renormalize, moe.route(gating, 8, renormalize))


@pytest.mark.parametrize(
    "num_experts", [8, 128, 256, 60, 384]
)  # 60 and 384 use the workspace path
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("num_tokens", [1, 33, 1024])
def test_expert_counts(num_experts, topk, num_tokens):
    gating = logits(num_tokens, num_experts, torch.bfloat16, seed=num_experts)
    check(gating, topk, True, moe.route(gating, topk, renormalize=True))


@pytest.mark.kernels_ci
def test_exact_ids_with_distinct_logits():
    # A random permutation per row makes every logit distinct, so ids must match exactly.
    g = torch.Generator(device="cuda").manual_seed(1)
    gating = (
        torch.stack(
            [torch.randperm(128, device="cuda", generator=g) for _ in range(64)]
        ).float()
        / 16
    )
    out = moe.route(gating, 4)
    _, ref_w, ref_ids = reference(gating, 4, False)
    torch.testing.assert_close(out.topk_ids.long(), ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(out.topk_weights, ref_w)


def test_raw_api_matches_upstream_signature():
    gating = logits(100, 32, torch.float16)
    weights = torch.empty(100, 4, dtype=torch.float32, device="cuda")
    ids = torch.empty(100, 4, dtype=torch.int32, device="cuda")
    token_expert = torch.empty(100, 4, dtype=torch.int32, device="cuda")

    moe.topk_softmax(weights, ids, token_expert, gating, False)

    check(gating, 4, False, moe.Routing(weights, ids))
    expected = (
        torch.arange(4, device="cuda") * 100 + torch.arange(100, device="cuda")[:, None]
    )
    torch.testing.assert_close(token_expert.long(), expected, rtol=0, atol=0)


def test_padding_rows():
    gating = logits(64, 32, torch.bfloat16)
    valid = torch.tensor([40], dtype=torch.int32, device="cuda")
    out = moe.route(gating, 4, renormalize=True, num_valid_rows=valid)
    assert (out.topk_ids[40:] == -1).all()
    assert (out.topk_weights[40:] == 0).all()
    check(gating[:40], 4, True, moe.Routing(out.topk_weights[:40], out.topk_ids[:40]))


def test_leading_dims_and_current_stream():
    gating = logits(4 * 50, 64, torch.bfloat16).view(4, 50, 64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = moe.route(gating, 2)
    stream.synchronize()
    assert out.topk_ids.shape == (4, 50, 2)
    check(
        gating.view(-1, 64),
        2,
        False,
        moe.Routing(out.topk_weights.view(-1, 2), out.topk_ids.view(-1, 2)),
    )


@pytest.mark.kernels_ci
def test_rejects_bad_arguments():
    gating = logits(8, 16, torch.float32)
    weights = torch.empty(8, 2, dtype=torch.float32, device="cuda")
    ids = torch.empty(8, 2, dtype=torch.int32, device="cuda")
    token_expert = torch.empty(8, 2, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk_indices has the wrong dtype"):
        moe.topk_softmax(weights, ids.long(), token_expert, gating, False)
    with pytest.raises(
        RuntimeError, match="gating_output must be float32, float16 or bfloat16"
    ):
        moe.topk_softmax(weights, ids, token_expert, gating.double(), False)
    with pytest.raises(RuntimeError, match="topk must be in"):
        moe.route(gating, 17)
    with pytest.raises(NotImplementedError):
        moe.topk_softmax(
            weights, ids, token_expert, gating, False, num_shared_experts=1
        )


def test_op_registration_passes_opcheck():
    # Schema, fake (meta) kernel and mutation annotations, as torch.compile sees them.
    gating = logits(64, 60, torch.bfloat16)
    args = (
        torch.empty(64, 4, dtype=torch.float32, device="cuda"),
        torch.empty(64, 4, dtype=torch.int32, device="cuda"),
        torch.empty(64, 4, dtype=torch.int32, device="cuda"),
        gating,
        torch.empty(64 * 60, dtype=torch.float32, device="cuda"),
        True,
        None,
    )
    op = getattr(moe.ops, "topk_softmax").default
    torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))


def test_traces_under_torch_compile():
    gating = logits(128, 64, torch.bfloat16)

    @torch.compile(fullgraph=True, backend="aot_eager")
    def routed(g):
        return moe.route(g, 4, renormalize=True)

    check(gating, 4, True, routed(gating))
