"""Per-layer forward activation tracing.

Runs one training step with ``SUROGATE_DEBUG_DUMP_*`` set to every block-scope
activation slot declared in the compiled IR, parses the dumped FP32 files,
and emits one ACT record per (layer, slot) with
min/max/mean/std/nan_pct/inf_pct stats. Flags the first (layer, op) where
NaN or Inf appears as severity=error.

The dump hook fires only during ``execute_forward``; backward activations go
to ``surogate debug gradients``. Global-scope slots (``x0``, ``xF``,
``token_ids``, ...) are not capturable by the per-layer hook and are listed
in the MODEL record for visibility.

Args
----
config_path : str
    Path to a training config YAML. A tokenized dataset is required (if
    ``config.output_dir`` has no ``train-*.bin``, the tokenize step runs
    transparently as part of ``SurogateSFT.run()``).
output : str | None
    Output JSONL path. Defaults to
    ``./debug/debug_activations_<model>_<ts>.jsonl``. A sibling
    ``.header.json`` sidecar is written with run_id / git sha / config path.
hub_token : str | None
    HuggingFace Hub token for private-repo model downloads.

Returns
-------
int
    ``0`` on success (including when NaN/Inf are detected — check the JSONL
    for severity=error rows). ``1`` only on setup failure (model
    unresolvable, IR compile failure).

Output records (``tag`` field, one per line)
-------------------------------------------
RUN
    Invocation context: ``subcommand="activations"``, ``model_id``,
    ``model_dir``, ``architecture``, ``n_layers``, ``dumped_slots_expected``.
MODEL
    Compiled-IR summary. Fields: ``name``, ``kind``, ``dsl_config``,
    ``block_slot_count``, ``global_slot_count``, ``global_slots``
    (list of ``{name, shape, dtype}`` for slots the layer hook cannot
    capture).
ACT
    One per (layer × block-scope slot). Fields:
    ``layer`` (physical index), ``op`` (short slot name, e.g. ``ln1``,
    ``qkv``), ``slot`` (full runtime name ``blocks[N].<op>``),
    ``declared_shape``, ``declared_dtype``, ``dumped_shape``,
    ``dumped_dtype`` (from the ``.json`` sidecar), ``status`` (one of
    :class:`DumpStatus`), ``error``, and stats:
    ``size``, ``finite_count``, ``nan_count``, ``inf_count``,
    ``nan_pct``, ``inf_pct``, ``zero_count``, ``zero_pct``, ``min``,
    ``max``, ``mean``, ``std``, ``abs_mean``, ``norm``.
    severity=info when clean; =warn when dump missing; =error on NaN/Inf
    or read-failure.
ERROR
    severity=error. Two phases:
    ``phase="activation_scan"`` with ``first_nan_layer``, ``first_nan_op``
    (emitted once, the earliest NaN/Inf site);
    ``phase="training_step"`` with formatted traceback if the forward pass
    raised (partial dumps are still emitted).
SUMMARY
    Terminal. ``counts_by_tag``, ``counts_by_severity``, ``status`` ∈
    {``completed``, ``completed_with_trainer_error``}, ``first_nan_layer``,
    ``first_nan_op``, ``dumps_present``, ``dumps_missing``.

Grep recipes
------------
::

    grep '"severity":"error"' debug_activations_*.jsonl        # punch list
    grep '"layer":4,"op":"qkv"' debug_activations_*.jsonl      # one slot
    jq 'select(.tag=="ACT" and .nan_pct>0) | {layer,op,nan_pct}' debug_*.jsonl
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import NamedTuple

from surogate.utils.logger import get_logger

from ._shared import (
    DebugResolveError,
    configure_for_single_step,
    make_dumps_root,
    parallel_stat,
    resolve_model_and_ir,
    rmtree_quiet,
)
from .schema import DumpStatus, Severity, Tag
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


class _FirstNaN(NamedTuple):
    layer: int
    op: str


def run_activation_trace(config_path: str, output: str | None = None, hub_token: str | None = None) -> int:
    """Run the activation trace end-to-end. Returns 0 on success (even if
    the trace found NaN/Inf — check the JSONL for severity=error rows)."""
    dumps_root = make_dumps_root("acts")
    try:
        return _run_with_dumps_root(config_path, output, hub_token, dumps_root)
    finally:
        rmtree_quiet(dumps_root)


def _run_with_dumps_root(
    config_path: str,
    output: str | None,
    hub_token: str | None,
    dumps_root: Path,
) -> int:
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(str(e))
        return 1

    module = resolved.module
    dsl_cfg = module.get("config", {})
    n_layers = int(dsl_cfg.get("n_layers", 0))
    layout = module.get("activation_layout", {})
    block_slots = [s for s in layout.get("slots", []) if s.get("scope") == "block"]
    global_slots = [s for s in layout.get("slots", []) if s.get("scope") == "global"]

    dump_names = [f"blocks[{i}].{s['name']}" for i in range(n_layers) for s in block_slots]

    run_id = make_run_id()
    model_name = os.path.basename(resolved.model_dir.rstrip("/"))
    out_path = output or default_output_path("activations", model_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dump_dir = dumps_root / run_id
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Env vars must be set before the trainer is constructed (the C++ dump
    # hook's tensor-list env is captured as a static local on first firing).
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(dump_dir)
    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_names)

    configure_for_single_step(resolved.config, steps=1)

    trainer_error: str | None = None
    try:
        from surogate.train.sft import SurogateSFT
        from surogate.utils.dict import DictDefault

        SurogateSFT(resolved.config, DictDefault({})).run()
    except Exception as e:
        trainer_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.warning(f"training step raised; emitting partial dumps: {type(e).__name__}: {e}")

    header = {
        "subcommand": "activations",
        "config_path": os.path.abspath(config_path),
        "model_id": resolved.model_id,
        "model_dir": resolved.model_dir,
        "architecture": resolved.architecture,
        "n_layers": n_layers,
        "dump_dir": str(dump_dir),
        "block_slot_count": len(block_slots),
        "global_slot_count": len(global_slots),
        "dumped_slots_expected": len(dump_names),
        "trainer_error": trainer_error,
    }

    dump_files = set(os.listdir(dump_dir)) if dump_dir.exists() else set()
    tasks = [
        ((layer_idx, slot["name"]), f"blocks[{layer_idx}].{slot['name']}")
        for layer_idx in range(n_layers)
        for slot in block_slots
    ]
    results = parallel_stat(dump_dir, dump_files, tasks)

    first_nan: _FirstNaN | None = None
    present_dumps = 0
    missing_dumps = 0

    with DebugJsonlWriter(out_path, run_id=run_id, header=header) as w:
        w.write(
            Tag.RUN,
            subcommand="activations",
            model_id=resolved.model_id,
            model_dir=resolved.model_dir,
            architecture=resolved.architecture,
            n_layers=n_layers,
            dumped_slots_expected=len(dump_names),
        )
        w.write(
            Tag.MODEL,
            name=module.get("name"),
            kind=module.get("kind"),
            dsl_config=dsl_cfg,
            block_slot_count=len(block_slots),
            global_slot_count=len(global_slots),
            global_slots=[
                {"name": s["name"], "shape": s.get("shape"), "dtype": s.get("dtype", "bf16")} for s in global_slots
            ],
        )

        for layer_idx in range(n_layers):
            for slot in block_slots:
                slot_name = slot["name"]
                tensor_name = f"blocks[{layer_idx}].{slot_name}"
                result = results[(layer_idx, slot_name)]

                severity = Severity.INFO
                if result.status == DumpStatus.MISSING:
                    severity = Severity.WARN
                    missing_dumps += 1
                elif result.status == DumpStatus.LOADED:
                    present_dumps += 1
                    if result.stats and (result.stats["nan_count"] > 0 or result.stats["inf_count"] > 0):
                        severity = Severity.ERROR
                        if first_nan is None:
                            first_nan = _FirstNaN(layer_idx, slot_name)
                elif result.status == DumpStatus.READ_FAILED:
                    severity = Severity.ERROR

                w.write(
                    Tag.ACT,
                    severity=severity,
                    layer=layer_idx,
                    op=slot_name,
                    slot=tensor_name,
                    declared_shape=slot.get("shape"),
                    declared_dtype=slot.get("dtype", "bf16"),
                    dumped_shape=(result.meta or {}).get("shape"),
                    dumped_dtype=(result.meta or {}).get("dtype"),
                    status=result.status,
                    error=result.error,
                    **(result.stats or {}),
                )

        if first_nan is not None:
            w.write(
                Tag.ERROR,
                severity=Severity.ERROR,
                phase="activation_scan",
                first_nan_layer=first_nan.layer,
                first_nan_op=first_nan.op,
                error=f"NaN/Inf first appears at layer {first_nan.layer} op={first_nan.op}",
            )

        if trainer_error:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="training_step", error=trainer_error)

        w.summary(
            status="completed" if trainer_error is None else "completed_with_trainer_error",
            first_nan_layer=first_nan.layer if first_nan else None,
            first_nan_op=first_nan.op if first_nan else None,
            dumps_present=present_dumps,
            dumps_missing=missing_dumps,
        )

    errors = w.counts_by_severity.get(Severity.ERROR.value, 0)
    warns = w.counts_by_severity.get(Severity.WARN.value, 0)
    if first_nan is not None:
        logger.error(
            f"first NaN/Inf at layer={first_nan.layer} op={first_nan.op!r} "
            f"({errors} error rows, {warns} warnings) — see {out_path}"
        )
    else:
        logger.info(f"wrote {out_path} ({errors} errors, {warns} warnings)")
    return 0
