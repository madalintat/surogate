"""Emit deterministic FLA chunk-stack gated-delta-rule forward/backward values."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
fla_l2norm = pytest.importorskip("fla.modules.l2norm")
fla_chunk = pytest.importorskip("fla.ops.gated_delta_rule.chunk")

l2norm_bwd = fla_l2norm.l2norm_bwd
l2norm_fwd = fla_l2norm.l2norm_fwd
chunk_gated_delta_rule_bwd = fla_chunk.chunk_gated_delta_rule_bwd
chunk_gated_delta_rule_fwd = fla_chunk.chunk_gated_delta_rule_fwd


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


def _stats(
    t: torch.Tensor,
    *,
    include_values: bool,
) -> dict[str, float | list[float] | int]:
    x = t.detach().float().reshape(-1)
    out: dict[str, float | list[float] | int] = {
        "numel": int(x.numel()),
        "sum": float(x.sum().item()),
        "mean": float(x.mean().item()),
        "l1": float(x.abs().sum().item()),
        "l2": float(torch.linalg.vector_norm(x, ord=2).item()),
        "linf": float(x.abs().max().item()),
        "first8": [float(v) for v in x[:8].cpu().tolist()],
    }
    if include_values:
        out["values"] = [float(v) for v in x.cpu().tolist()]
    return out


def _run_case(
    *,
    case_name: str,
    b: int,
    t: int,
    h: int,
    k: int,
    v: int,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
    emit_values: bool,
) -> dict[str, object]:
    device = "cuda"
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(k)
    q = _signal((b, t, h, k), device=device, dtype=dtype, a=0.173, b=0.31, c=0.097, d=-0.22)
    k_t = _signal((b, t, h, k), device=device, dtype=dtype, a=0.137, b=-0.41, c=0.083, d=0.57)
    v_t = _signal((b, t, h, v), device=device, dtype=dtype, a=0.191, b=0.73, c=0.121, d=-0.35)

    g_raw = _signal((b, t, h), device=device, dtype=torch.float32, a=0.157, b=-0.27, c=0.109, d=0.44)
    g = (-0.7 + 0.2 * torch.sin(g_raw)).to(dtype)

    beta_raw = _signal((b, t, h), device=device, dtype=torch.float32, a=0.113, b=0.19, c=0.071, d=-0.63)
    beta = torch.sigmoid(beta_raw).to(dtype)

    initial_state = _signal((b, h, k, v), device=device, dtype=torch.float32, a=0.167, b=-0.52, c=0.061, d=0.28)
    grad_out = _signal((b, t, h, v), device=device, dtype=dtype, a=0.149, b=0.66, c=0.101, d=-0.14)
    grad_state = _signal((b, h, k, v), device=device, dtype=torch.float32, a=0.089, b=-0.33, c=0.055, d=0.47)

    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q)
        k_norm, k_rstd = l2norm_fwd(k_t)
    else:
        q_norm, q_rstd = q, None
        k_norm, k_rstd = k_t, None

    g_cum, out, A, final_state = chunk_gated_delta_rule_fwd(
        q=q_norm,
        k=k_norm,
        v=v_t,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )
    assert final_state is not None

    dq, dk, dv, dbeta, dg, d_initial_state = chunk_gated_delta_rule_bwd(
        q=q_norm,
        k=k_norm,
        v=v_t,
        g=g_cum,
        beta=beta,
        A=A,
        scale=scale,
        initial_state=initial_state,
        do=grad_out,
        dht=grad_state,
        cu_seqlens=None,
    )
    if use_qk_l2norm_in_kernel:
        dq = l2norm_bwd(q_norm, q_rstd, dq)
        dk = l2norm_bwd(k_norm, k_rstd, dk)

    return {
        "meta": {
            "case_name": case_name,
            "B": b,
            "T": t,
            "H": h,
            "K": k,
            "V": v,
            "dtype": str(dtype),
            "chunk_size": chunk_size,
            "scale": scale,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "emit_values": emit_values,
        },
        "out": _stats(out, include_values=emit_values),
        "final_state": _stats(final_state, include_values=emit_values),
        "d_query": _stats(dq, include_values=emit_values),
        "d_key": _stats(dk, include_values=emit_values),
        "d_value": _stats(dv, include_values=emit_values),
        "d_g": _stats(dg, include_values=emit_values),
        "d_beta": _stats(dbeta, include_values=emit_values),
        "d_initial_state": _stats(d_initial_state, include_values=emit_values),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_gated_delta_rule_fla_reference_value_dump() -> None:
    cases = [
        _run_case(
            case_name="single_chunk_small",
            b=1,
            t=8,
            h=2,
            k=4,
            v=3,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            emit_values=True,
        ),
        _run_case(
            case_name="multi_chunk_tail",
            b=1,
            t=130,
            h=2,
            k=64,
            v=64,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            emit_values=True,
        ),
        _run_case(
            case_name="multi_chunk_128x128_dev",
            b=1,
            t=96,
            h=2,
            k=128,
            v=128,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            emit_values=True,
        ),
        _run_case(
            case_name="multi_chunk_128x128_prod_stats",
            b=2,
            t=2048,
            h=16,
            k=128,
            v=128,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            emit_values=False,
        ),
    ]

    report: dict[str, object] = {
        "meta": {
            "impl": "fla_chunk_stack",
            "num_cases": len(cases),
        },
        "cases": cases,
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "tests" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gated_delta_rule_fla_values.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    assert out_path.exists()
