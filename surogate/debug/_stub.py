"""Shared helper for not-yet-implemented debug subcommands.

Collapses the three stub runners (activations, gradients, diff) into a single
``write_not_implemented_stub(...)`` call so the boilerplate (open writer, emit
RUN + ERROR + SUMMARY, log, return exit code 2) lives in one place.
"""

from __future__ import annotations

import os
from typing import Any

from surogate.utils.logger import get_logger

from .schema import Severity, Tag
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


def write_not_implemented_stub(
    subcommand: str,
    config_path: str,
    plan: str,
    *,
    output: str | None = None,
    extra_run_fields: dict[str, Any] | None = None,
) -> int:
    """Emit a stub JSONL file for an unwired subcommand. Returns exit code 2."""
    run_id = make_run_id()
    model_name = os.path.basename(config_path).rsplit(".", 1)[0]
    out_path = output or default_output_path(subcommand, model_name)
    header = {
        "subcommand": subcommand,
        "config_path": os.path.abspath(config_path),
        "status": "not_implemented",
    }

    with DebugJsonlWriter(out_path, run_id=run_id, header=header) as w:
        w.write(
            Tag.RUN,
            subcommand=subcommand,
            config_path=os.path.abspath(config_path),
            **(extra_run_fields or {}),
        )
        w.write(
            Tag.ERROR,
            severity=Severity.ERROR,
            phase="subcommand",
            error=f"{subcommand} subcommand not yet implemented",
            plan=plan,
        )
        w.summary(status="not_implemented")
    logger.warning(f"{subcommand} stub wrote {out_path}")
    return 2
