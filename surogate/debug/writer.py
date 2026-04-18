"""JSONL writer for debug subcommand output.

One JSON object per line. A header sidecar (.header.json) captures invocation
metadata (run_id, cli args, git sha, config path) so JSONL records stay compact.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .schema import Severity


def make_run_id() -> str:
    """Short + unique run id: ``20260418T143022-a1b2c3``."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def default_output_path(subcommand: str, model_name: str, output_dir: str | None = None) -> str:
    """``<output_dir>/debug_<subcommand>_<model>_<ts>.jsonl`` (output_dir defaults to ``./debug``)."""
    if output_dir is None:
        output_dir = "debug"
    ts = time.strftime("%Y%m%dT%H%M%S")
    safe_model = model_name.replace("/", "--").replace(":", "_")
    return str(Path(output_dir) / f"debug_{subcommand}_{safe_model}_{ts}.jsonl")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _jsonable(v: Any) -> Any:
    """Coerce values that would break ``json.dumps`` (NaN/Inf, tuples, sets, numpy scalars)."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return _jsonable(v.item())
        except Exception:
            return str(v)
    return v


class DebugJsonlWriter:
    """Append-style JSONL writer for the ``surogate debug`` subcommands.

    Named ``DebugJsonlWriter`` (not ``JsonlWriter``) to avoid collision with
    :class:`surogate.utils.jsonl.JsonlWriter`, which is an append-every-call /
    rank-aware writer for training logs — different contract.
    """

    def __init__(self, path: str, run_id: str, header: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header_path = self.path.with_suffix(".header.json")
        self.run_id = run_id

        full_header: dict[str, Any] = {
            "run_id": run_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "git_sha": _git_sha(),
            **header,
        }
        with open(self.header_path, "w") as f:
            json.dump(_jsonable(full_header), f, indent=2)

        self._fp = None
        self._by_tag: dict[str, int] = {}
        self._by_severity: dict[str, int] = {
            Severity.INFO.value: 0,
            Severity.WARN.value: 0,
            Severity.ERROR.value: 0,
        }

    def __enter__(self) -> DebugJsonlWriter:
        self._fp = open(self.path, "w")
        return self

    def __exit__(self, *args) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    def write(self, tag: str, *, severity: str = Severity.INFO, **fields: Any) -> None:
        if self._fp is None:
            raise RuntimeError("DebugJsonlWriter used outside `with` block")
        tag_str = tag.value if hasattr(tag, "value") else tag
        sev_str = severity.value if hasattr(severity, "value") else severity
        record = {"tag": tag_str, "severity": sev_str, **fields}
        self._fp.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")
        self._by_tag[tag_str] = self._by_tag.get(tag_str, 0) + 1
        self._by_severity[sev_str] = self._by_severity.get(sev_str, 0) + 1

    def summary(self, **extra: Any) -> None:
        from .schema import Tag

        self.write(
            Tag.SUMMARY,
            severity=Severity.INFO,
            counts_by_tag=dict(self._by_tag),
            counts_by_severity=dict(self._by_severity),
            output_path=str(self.path),
            header_path=str(self.header_path),
            **extra,
        )

    @property
    def counts_by_severity(self) -> dict[str, int]:
        return dict(self._by_severity)
