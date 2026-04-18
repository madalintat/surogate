"""Layer-by-layer numerical diff vs HuggingFace transformers (placeholder).

When implemented, this will:
  1. Load the HF transformers reference model from the same checkpoint (CPU or
     second GPU) using AutoModelForCausalLM / AutoModel.
  2. Attach forward hooks to every reference module whose output maps to a
     named DSL activation slot (per-architecture name map).
  3. Load the DSL model and dump its activation slots via the existing
     SUROGATE_DEBUG_DUMP_* infra.
  4. Feed identical token_ids to both, align by (layer, slot_name), diff
     tensor-by-tensor: max_abs_diff, mean_abs_diff, cosine_sim.
  5. Emit one DIFF record per (layer, slot). Rows where max_abs_diff exceeds a
     threshold are severity=error — the first such row is the bug site.
"""

from __future__ import annotations

from ._stub import write_not_implemented_stub

_PLAN = (
    "1) load HF transformers reference with same checkpoint, "
    "2) hook per-module outputs matching DSL slot names (per-arch map), "
    "3) dump DSL activations via SUROGATE_DEBUG_DUMP_*, "
    "4) align by (layer, slot) and emit DIFF records with max_abs_diff/mean_abs_diff/cos_sim"
)


def run_reference_diff(
    config_path: str,
    output: str | None = None,
    reference: str | None = None,
) -> int:
    return write_not_implemented_stub(
        "diff",
        config_path,
        _PLAN,
        output=output,
        extra_run_fields={"reference": reference},
    )
