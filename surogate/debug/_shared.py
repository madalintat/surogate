"""Shared helpers for ``surogate debug`` subcommands."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .schema import DumpStatus

_DUMP_WORKER_COUNT = 16


class DebugResolveError(RuntimeError):
    """Raised when a debug subcommand cannot set up: bad config, missing
    model, IR compile failure, etc. Callers log + return non-zero."""


@dataclass
class ResolvedModel:
    config: Any  # surogate.core.config.sft_config.SFTConfig
    model_id: str
    model_dir: str
    architecture: str
    hf_config: dict
    hf_config_path: str
    ir: dict
    module: dict  # ir["modules"][0] — the top-level ModuleIR dict


def resolve_model_and_ir(config_path: str, hub_token: str | None = None) -> ResolvedModel:
    """Load the training config, resolve the model checkpoint, compile DSL IR.

    Imports surogate modules transitively (load_config, safe_snapshot_download,
    compile_model_for_hf). Callers that need to set environment variables read
    by C++ runtime statics must do so BEFORE calling this function.
    """
    from surogate.core.config.loader import load_config
    from surogate.core.config.sft_config import SFTConfig
    from surogate.core.model.registry import safe_snapshot_download

    config = load_config(SFTConfig, config_path)
    model_id = getattr(config, "model", None)
    if not model_id:
        raise DebugResolveError(f"config '{config_path}' has no 'model' field")

    try:
        model_dir = safe_snapshot_download(model_id, hub_token=hub_token, download_model=True)
    except Exception as e:
        raise DebugResolveError(f"failed to resolve model dir for '{model_id}': {e}") from e

    hf_config_path = os.path.join(model_dir, "config.json")
    try:
        with open(hf_config_path) as f:
            hf_config = json.load(f)
    except OSError as e:
        raise DebugResolveError(f"could not read {hf_config_path}: {e}") from e

    architectures = hf_config.get("architectures", [])
    architecture = architectures[0] if architectures else None
    if not architecture:
        raise DebugResolveError(f"{hf_config_path} has no 'architectures' field")

    # Side-effect import: registers every @model class into _model_registry.
    import surogate.dsl.models  # noqa: F401

    from surogate.dsl.py_compiler import compile_model_for_hf

    ir = json.loads(compile_model_for_hf(architecture, hf_config, raise_on_error=False))
    modules = ir.get("modules", [])
    if not ir.get("success") or not modules:
        raise DebugResolveError(f"IR compilation failed for architecture '{architecture}': {ir.get('errors')}")

    return ResolvedModel(
        config=config,
        model_id=model_id,
        model_dir=model_dir,
        architecture=architecture,
        hf_config=hf_config,
        hf_config_path=hf_config_path,
        ir=ir,
        module=modules[0],
    )


def sanitize_dump_name(name: str) -> str:
    """Mirror of ``debug_dump_sanitize`` in csrc/src/runtime/executor/graph_executor.cpp:199.

    Keep aligned with that C++ function — drift causes silent dump-file misses.
    """
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in name)


# =============================================================================
# Dump file I/O + stats
# =============================================================================


@dataclass
class DumpResult:
    status: DumpStatus
    stats: dict | None = None
    meta: dict | None = None
    error: str | None = None


def load_and_stat(dump_dir: Path, tensor_name: str, dump_files: set[str]) -> DumpResult:
    """Load one ``<name>.bin`` + ``<name>.json`` pair and compute stats."""
    safe = sanitize_dump_name(tensor_name)
    bin_name = f"{safe}.bin"
    json_name = f"{safe}.json"

    if bin_name not in dump_files:
        return DumpResult(status=DumpStatus.MISSING)

    meta: dict[str, Any] | None = None
    if json_name in dump_files:
        try:
            with open(dump_dir / json_name) as f:
                meta = json.load(f)
        except Exception:
            meta = None

    try:
        data = np.fromfile(dump_dir / bin_name, dtype=np.float32)
    except Exception as e:
        return DumpResult(
            status=DumpStatus.READ_FAILED,
            meta=meta,
            error=f"{type(e).__name__}: {e}",
        )

    if data.size == 0:
        return DumpResult(status=DumpStatus.EMPTY, stats={"size": 0}, meta=meta)

    return DumpResult(status=DumpStatus.LOADED, stats=compute_stats_numpy(data), meta=meta)


def parallel_stat(
    dump_dir: Path,
    dump_files: set[str],
    tasks: list[tuple[Any, str]],
) -> dict[Any, DumpResult]:
    """Load + stat every (key, tensor_name) task in parallel. Returns a dense map keyed by ``key``."""

    def _one(task: tuple[Any, str]) -> tuple[Any, DumpResult]:
        key, tensor_name = task
        return key, load_and_stat(dump_dir, tensor_name, dump_files)

    with ThreadPoolExecutor(max_workers=_DUMP_WORKER_COUNT) as ex:
        return dict(ex.map(_one, tasks))


def make_dumps_root(subcommand: str) -> Path:
    """Create a private tempdir for raw dumps, prefixed by subcommand name."""
    return Path(tempfile.mkdtemp(prefix=f"surogate_{subcommand}_"))


def rmtree_quiet(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def configure_for_single_step(config: Any, steps: int = 1) -> None:
    """Mutate a training config into a minimal N-step no-save, no-eval, single-node run.

    Shared between activations (1 step) and gradients (N steps). Future subcommands
    that drive the trainer should call this rather than repeat the field list.
    """
    config.max_steps = max(1, int(steps))
    config.eval_steps = 0
    config.save_steps = 10**9
    if hasattr(config, "auto_lr_reduction"):
        config.auto_lr_reduction = False
    if hasattr(config, "distributed"):
        config.distributed = None


def tokenize_and_get_train_files(config: Any) -> list[str]:
    """Run the tokenize pipeline (idempotent — caches) and return ``train-*.bin`` files.

    Mirrors ``SurogateSFT.run()``'s tokenize + glob prelude. Raises DebugResolveError on failure.
    """
    from surogate.train.tokenize import TokenizeDatasets
    from surogate.utils.dict import DictDefault

    try:
        TokenizeDatasets(config, DictDefault({})).run()
    except Exception as e:
        raise DebugResolveError(f"tokenization failed: {type(e).__name__}: {e}") from e

    return sorted(str(p) for p in Path(config.output_dir).glob("train-*.bin"))


def build_optimizer_config(config: Any, lr: float) -> Any:
    """Construct ``_surogate.OptimizerConfig`` from a training config. Kept
    in lockstep with ``SurogateTrainerWrapper.train()``'s build at
    ``surogate/train/trainer.py:852`` — when the trainer adds a field there,
    add it here or extract this helper to trainer.py and delete the duplicate.
    """
    from surogate import _surogate

    return _surogate.OptimizerConfig(
        optimizer=config.optimizer,
        learning_rate=lr,
        weight_decay=config.weight_decay,
        grad_clip=config.max_grad_norm,
        adamw_beta1=config.adamw_beta1,
        adamw_beta2=config.adamw_beta2,
        adamw_epsilon=config.adamw_epsilon,
        normuon_momentum=config.normuon_momentum,
        normuon_beta2=config.normuon_beta2,
        normuon_lr=lr,
        normuon_cautious_wd=config.normuon_cautious_wd,
    )


def compute_stats_numpy(data: np.ndarray) -> dict:
    """Compute min/max/mean/std/abs_mean/norm + nan/inf/zero counts in a few passes.

    ``inf_count`` is derived from ``size - finite_count - nan_count`` to avoid a 4th pass.
    """
    total = int(data.size)
    isfinite = np.isfinite(data)
    finite_count = int(isfinite.sum())
    nonfinite = total - finite_count
    nan_count = int(np.isnan(data).sum())
    inf_count = nonfinite - nan_count

    base = {
        "size": total,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nan_pct": 100.0 * nan_count / total,
        "inf_pct": 100.0 * inf_count / total,
    }

    if finite_count == 0:
        base.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "abs_mean": None,
                "norm": None,
                "zero_count": 0,
                "zero_pct": 0.0,
            }
        )
        return base

    finite = data[isfinite]
    zero_count = int((finite == 0).sum())
    base.update(
        {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()) if finite.size > 1 else 0.0,
            "abs_mean": float(np.abs(finite).mean()),
            "norm": float(np.sqrt(np.square(finite.astype(np.float64)).sum())),
            "zero_count": zero_count,
            "zero_pct": 100.0 * zero_count / total,
        }
    )
    return base
