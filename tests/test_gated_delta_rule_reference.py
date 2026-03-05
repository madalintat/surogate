"""Emit deterministic FLA chunk gated-delta-rule forward/backward reference values."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
fla_gdr = pytest.importorskip("fla.ops.gated_delta_rule")

chunk_gated_delta_rule = fla_gdr.chunk_gated_delta_rule


def _signal(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    a: float,
    b: float,
    c: float,
    d: float,
) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, device=device, dtype=torch.float32)
    x = 0.7 * torch.sin(idx * a + b) + 0.3 * torch.cos(idx * c + d)
    return x.reshape(shape).to(dtype)


def _stats(t: torch.Tensor) -> dict[str, float | list[float]]:
    x = t.detach().float().reshape(-1)
    return {
        "sum": float(x.sum().item()),
        "mean": float(x.mean().item()),
        "l1": float(x.abs().sum().item()),
        "l2": float(torch.linalg.vector_norm(x, ord=2).item()),
        "linf": float(x.abs().max().item()),
        "first8": [float(v) for v in x[:8].cpu().tolist()],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_gated_delta_rule_fla_reference_value_dump() -> None:
    device = "cuda"
    dtype = torch.bfloat16

    b, t, h, k, v = 1, 8, 2, 4, 3
    chunk_size = 64
    scale = 1.0 / (k**0.5)

    q = _signal((b, t, h, k), device=device, dtype=dtype, a=0.173, b=0.31, c=0.097, d=-0.22).requires_grad_(True)
    k_t = _signal((b, t, h, k), device=device, dtype=dtype, a=0.137, b=-0.41, c=0.083, d=0.57).requires_grad_(True)
    v_t = _signal((b, t, h, v), device=device, dtype=dtype, a=0.191, b=0.73, c=0.121, d=-0.35).requires_grad_(True)

    g_raw = _signal((b, t, h), device=device, dtype=torch.float32, a=0.157, b=-0.27, c=0.109, d=0.44)
    g = (-0.7 + 0.2 * torch.sin(g_raw)).to(dtype).requires_grad_(True)

    beta_raw = _signal((b, t, h), device=device, dtype=torch.float32, a=0.113, b=0.19, c=0.071, d=-0.63)
    beta = torch.sigmoid(beta_raw).to(dtype).requires_grad_(True)

    initial_state = _signal(
        (b, h, k, v), device=device, dtype=torch.float32, a=0.167, b=-0.52, c=0.061, d=0.28
    ).requires_grad_(True)

    out, final_state = chunk_gated_delta_rule(
        q,
        k_t,
        v_t,
        g=g,
        beta=beta,
        scale=scale,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    assert final_state is not None

    grad_out = _signal(
        (b, t, h, v), device=device, dtype=dtype, a=0.149, b=0.66, c=0.101, d=-0.14
    )
    grad_state = _signal(
        (b, h, k, v), device=device, dtype=torch.float32, a=0.089, b=-0.33, c=0.055, d=0.47
    )

    grads = torch.autograd.grad(
        outputs=(out, final_state),
        inputs=(q, k_t, v_t, g, beta, initial_state),
        grad_outputs=(grad_out, grad_state),
        allow_unused=False,
        retain_graph=False,
    )

    report = {
        "meta": {
            "B": b,
            "T": t,
            "H": h,
            "K": k,
            "V": v,
            "dtype": str(dtype),
            "chunk_size": chunk_size,
            "scale": scale,
            "use_qk_l2norm_in_kernel": True,
        },
        "out": _stats(out),
        "final_state": _stats(final_state),
        "d_query": _stats(grads[0]),
        "d_key": _stats(grads[1]),
        "d_value": _stats(grads[2]),
        "d_g": _stats(grads[3]),
        "d_beta": _stats(grads[4]),
        "d_initial_state": _stats(grads[5]),
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "tests" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gated_delta_rule_fla_values.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    assert out_path.exists()
