"""Slot-registry audit: which tensor names in the compiled IR have no DSL declaration.

The Python DSL is the source of truth for every tensor name the C++ runtime
consumes. Global activations (``xF``, ``x0``, ``xN``, ``residualN``, ``freq_cis``,
``ln_final_rstd``, ``loss``, …) should all be declared with
``model._register_activation(name, shape, scope=ActivationScope.GLOBAL)``.
Block-level activations are declared with ``tracer.register_activation(name, ...)``
inside each module. Anything a graph op reads/writes that isn't declared is a
**gap** — the C++ side has to special-case it by name, which is exactly the
fragile pattern we're retiring.

This subcommand compiles the DSL IR (no GPU, no safetensors read) and emits one
JSONL record per declared slot, per graph tensor name, and per gap.

Args
----
config_path : str
    Path to a training config YAML (same format as ``surogate sft``). Only
    ``model`` is required.
output : str | None
    Output JSONL path. Defaults to ``./debug/debug_registry_<model>_<ts>.jsonl``.
hub_token : str | None
    HuggingFace Hub token for private-repo models.

Returns
-------
int
    ``0`` on successful audit (inspect the JSONL for gaps). ``1`` if the audit
    could not run (config unreadable, model unresolvable, IR compile failure).

Output records (``tag`` field)
------------------------------
RUN
    ``model_id``, ``model_dir``, ``architecture``. Once per file.
MODEL
    ``name``, ``kind``, ``n_slots_block``, ``n_slots_global``,
    ``n_ops_forward``, ``n_ops_backward``.
SLOT
    One per declared activation-layout slot. Fields: ``name``, ``scope``
    (``block``/``global``/``gradient``), ``shape``, ``dtype``, ``aliases``,
    ``save_for_backward``, ``share_policy``.
TENSOR
    One per *unique* tensor name referenced by any forward/backward op.
    Fields: ``name``, ``resolution`` (see below), ``resolved_to`` (the slot
    name or parameter name that matched, if any), ``op_count``. severity=warn
    when ``resolution == "unresolved"``.

    Resolution kinds:

    - ``param`` — matches a DSL parameter (block- or top-level).
    - ``param_grad`` — ``d_<param>``; the base matches a parameter.
    - ``block_slot`` — block-indexed, field matches a declared block-scope slot.
    - ``global_slot`` — matches a declared global-scope slot (or alias).
    - ``global_slot_qualified`` — ``blocks[N].<global>`` where ``<global>``
      is a declared global slot (typical for ``rope_freqs``).
    - ``global_grad`` — ``d_<global>``; base matches a global slot.
    - ``io`` — graph input/output declared in ``forward.inputs`` / ``.outputs``.
    - ``accum_temp`` — ``_from_N`` / ``_accum_N`` autodiff accumulator variant.
    - ``unresolved`` — none of the above. This is a **gap** — the DSL owes a
      ``_register_activation`` call, OR a special-case name the C++ side handles
      by string match that should be migrated.
GAP
    severity=warn. One per *distinct* unresolved name pattern (block-qualified
    names collapsed to the unqualified probe). Fields: ``probe_name`` (the
    name to look up on the DSL side), ``sample_full_names`` (first 3 full
    names that triggered the gap), ``suggestion`` (what to add to the DSL).
SUMMARY
    Terminal. ``counts_by_tag``, ``counts_by_resolution``, ``n_gaps``,
    ``output_path``.

Grep recipes
------------
::

    grep '"tag":"GAP"'                         debug_registry_*.jsonl
    jq 'select(.tag=="TENSOR" and .resolution=="unresolved") | .name' debug_*.jsonl
    jq 'select(.tag=="SLOT" and .scope=="global") | .name' debug_*.jsonl
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from surogate.utils.logger import get_logger

from ._shared import DebugResolveError, resolve_model_and_ir
from .schema import Severity, Tag
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


_BLOCK_PREFIX_RE = re.compile(r"^(?:blocks\[\d+\]|layer\d+|d_blocks\[\d+\]|d_layer\d+)\.")
_SAVED_PREFIX = "saved."
_SSA_SUFFIX_RE = re.compile(r"_\d+$")
_ACCUM_SUFFIX_RE = re.compile(r"_(?:from|accum)_\d+$")


def _strip_ssa(name: str) -> str:
    return _SSA_SUFFIX_RE.sub("", name)


def _strip_accum(name: str) -> tuple[str, bool]:
    m = _ACCUM_SUFFIX_RE.search(name)
    return (name[: m.start()], True) if m else (name, False)


def _block_field(name: str) -> tuple[str | None, str | None]:
    """Return (block_prefix, field) for a qualified name, else (None, None)."""
    m = _BLOCK_PREFIX_RE.match(name)
    if not m:
        return None, None
    return name[: m.end()], name[m.end() :]


def _build_indexes(module: dict) -> dict[str, Any]:
    """Collect all the lookup tables we'll consult against."""
    layout = module.get("activation_layout") or {}
    slots = layout.get("slots") or []
    grad_slots = layout.get("gradient_slots") or []

    # Declared slot names, split by scope. Register aliases too.
    block_slot_names: set[str] = set()
    global_slot_names: set[str] = set()
    global_slot_meta: dict[str, dict] = {}

    for s in slots:
        name = s.get("name")
        if not name:
            continue
        scope = s.get("scope", "block")
        if scope == "global":
            global_slot_names.add(name)
            global_slot_meta[name] = s
            for alias in s.get("aliases") or []:
                global_slot_names.add(alias)
                global_slot_meta[alias] = s
        else:
            block_slot_names.add(name)
            for alias in s.get("aliases") or []:
                block_slot_names.add(alias)

    gradient_slot_names: set[str] = set()
    for s in grad_slots:
        name = s.get("name")
        if name:
            gradient_slot_names.add(name)
            for alias in s.get("aliases") or []:
                gradient_slot_names.add(alias)

    # Parameters. `module.params` is top-level only; forward.params is the
    # per-layer-expanded set (block params get prefixed).
    top_params = set((module.get("params") or {}).keys())
    forward_params = set((module.get("forward") or {}).get("params", {}).keys())
    param_names = top_params | forward_params

    # Block params resolve to unqualified fields (qkv_weight, mlp_up_weight, …).
    # Collect these so `blocks[N].qkv_weight` recognizes the field even without
    # the full forward-params expansion.
    block_param_fields: set[str] = set()
    for p in forward_params:
        _, field = _block_field(p)
        if field:
            block_param_fields.add(_strip_ssa(field))

    # Graph IO.
    fwd = module.get("forward") or {}
    bwd = module.get("backward") or {}
    fwd_inputs = set((fwd.get("inputs") or {}).keys())
    fwd_outputs = set((fwd.get("outputs") or {}).keys())
    bwd_inputs = set((bwd.get("inputs") or {}).keys())
    bwd_outputs = set((bwd.get("outputs") or {}).keys())
    io_names = fwd_inputs | fwd_outputs | bwd_inputs | bwd_outputs

    return {
        "slots": slots,
        "grad_slots": grad_slots,
        "block_slot_names": block_slot_names,
        "global_slot_names": global_slot_names,
        "global_slot_meta": global_slot_meta,
        "gradient_slot_names": gradient_slot_names,
        "param_names": param_names,
        "block_param_fields": block_param_fields,
        "io_names": io_names,
    }


def _classify_name(name: str, idx: dict[str, Any]) -> tuple[str, str | None]:
    """Return (resolution_kind, resolved_to).

    ``resolution_kind`` is one of the strings documented in the module header.
    """
    # Strip saved. prefix
    probe = name
    if probe.startswith(_SAVED_PREFIX):
        probe = probe[len(_SAVED_PREFIX) :]

    # Strip autodiff accumulator suffix
    probe, _had_accum = _strip_accum(probe)

    # IO slots (graph-declared inputs/outputs)
    if probe in idx["io_names"] or name in idx["io_names"]:
        return "io", probe

    # Direct parameter match (top-level or forward-expanded)
    if probe in idx["param_names"]:
        return "param", probe

    # Parameter gradient
    if probe.startswith("d_"):
        base = probe[2:]
        if base in idx["param_names"]:
            return "param_grad", base
        # Global-grad: d_<global>
        if base in idx["global_slot_names"]:
            return "global_grad", base
        # Accum temp of a known global-grad or param-grad
        base_stripped, _ = _strip_accum(base)
        if base_stripped in idx["param_names"]:
            return "param_grad", base_stripped
        if base_stripped in idx["global_slot_names"]:
            return "global_grad", base_stripped
        # Block-indexed gradient: d_blocks[N].<field>
        _, field = _block_field(base)
        if field is not None:
            field_stripped = _strip_ssa(field)
            field_stripped, _ = _strip_accum(field_stripped)
            if field_stripped in idx["block_slot_names"] or field_stripped in idx["gradient_slot_names"]:
                return "block_slot", field_stripped
            if field_stripped in idx["block_param_fields"]:
                return "param_grad", field_stripped
            if field_stripped in idx["global_slot_names"]:
                return "global_slot_qualified", field_stripped
        return "unresolved", None

    # Direct global slot match (covers aliases too)
    if probe in idx["global_slot_names"]:
        return "global_slot", probe

    # Block-indexed: blocks[N].<field>
    _, field = _block_field(probe)
    if field is not None:
        field_stripped = _strip_ssa(field)
        # block-scope declared slot
        if field_stripped in idx["block_slot_names"]:
            return "block_slot", field_stripped
        # per-layer parameter (qkv_weight etc.)
        if field_stripped in idx["block_param_fields"]:
            return "param", field_stripped
        # global referenced with a block prefix (rope_freqs, etc.)
        if field_stripped in idx["global_slot_names"]:
            return "global_slot_qualified", field_stripped
        return "unresolved", None

    return "unresolved", None


def _collect_tensor_names(module: dict) -> dict[str, int]:
    """Return {name: op_reference_count} across forward + backward ops."""
    counts: dict[str, int] = defaultdict(int)
    for phase in ("forward", "backward"):
        graph = module.get(phase) or {}
        for op in graph.get("operations") or []:
            for nm in op.get("inputs") or []:
                if nm:
                    counts[nm] += 1
            for nm in op.get("outputs") or []:
                if nm:
                    counts[nm] += 1
    return counts


def _gap_probe_name(name: str) -> str:
    """Normalize an unresolved name into its 'DSL should register' form.

    Strips saved. prefix, block-qualified prefixes (blocks[N]. / layerN.),
    SSA suffix, and autodiff accumulator suffix. Collapses many per-layer
    references into one reported gap.
    """
    probe = name
    if probe.startswith(_SAVED_PREFIX):
        probe = probe[len(_SAVED_PREFIX) :]
    probe, _ = _strip_accum(probe)
    if probe.startswith("d_"):
        _, field = _block_field(probe[2:])
        probe = "d_" + (field if field is not None else probe[2:])
    else:
        _, field = _block_field(probe)
        if field is not None:
            probe = field
    return _strip_ssa(probe)


def _suggestion_for(probe: str) -> str:
    if probe.startswith("d_"):
        return (
            f"gradient-only reference; ensure its forward counterpart "
            f"'{probe[2:]}' is registered with scope=GLOBAL or as a block slot"
        )
    return f"model._register_activation('{probe}', shape=..., scope=ActivationScope.GLOBAL)"


def run_registry_audit(config_path: str, output: str | None = None, hub_token: str | None = None) -> int:
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(str(e))
        return 1

    module = resolved.module
    idx = _build_indexes(module)

    run_id = make_run_id()
    model_name = os.path.basename(resolved.model_dir.rstrip("/"))
    out_path = output or default_output_path("registry", model_name)

    header: dict[str, Any] = {
        "subcommand": "registry",
        "config_path": os.path.abspath(config_path),
        "model_id": resolved.model_id,
        "model_dir": resolved.model_dir,
        "architecture": resolved.architecture,
    }

    with DebugJsonlWriter(out_path, run_id=run_id, header=header) as w:
        w.write(
            Tag.RUN,
            subcommand="registry",
            model_id=resolved.model_id,
            model_dir=resolved.model_dir,
            architecture=resolved.architecture,
        )

        n_slots_block = sum(1 for s in idx["slots"] if s.get("scope", "block") != "global")
        n_slots_global = sum(1 for s in idx["slots"] if s.get("scope") == "global")
        fwd_ops = len((module.get("forward") or {}).get("operations") or [])
        bwd_ops = len((module.get("backward") or {}).get("operations") or [])

        w.write(
            Tag.MODEL,
            name=module.get("name"),
            kind=module.get("kind"),
            n_slots_block=n_slots_block,
            n_slots_global=n_slots_global,
            n_slots_gradient=len(idx["grad_slots"]),
            n_ops_forward=fwd_ops,
            n_ops_backward=bwd_ops,
            n_params_top=len(module.get("params") or {}),
            n_params_forward=len((module.get("forward") or {}).get("params") or {}),
        )

        for s in idx["slots"]:
            w.write(
                Tag.SLOT,
                name=s.get("name"),
                scope=s.get("scope", "block"),
                shape=s.get("shape"),
                dtype=s.get("dtype"),
                aliases=s.get("aliases"),
                save_for_backward=s.get("save_for_backward", False),
                share_policy=s.get("share_policy"),
            )

        counts_by_resolution: dict[str, int] = defaultdict(int)
        gap_samples: dict[str, list[str]] = defaultdict(list)
        gap_counts: dict[str, int] = defaultdict(int)

        tensor_counts = _collect_tensor_names(module)
        for name in sorted(tensor_counts.keys()):
            kind, resolved_to = _classify_name(name, idx)
            counts_by_resolution[kind] += 1
            severity = Severity.WARN if kind == "unresolved" else Severity.INFO
            w.write(
                Tag.TENSOR,
                severity=severity,
                name=name,
                resolution=kind,
                resolved_to=resolved_to,
                op_count=tensor_counts[name],
            )
            if kind == "unresolved":
                probe = _gap_probe_name(name)
                gap_counts[probe] += 1
                if len(gap_samples[probe]) < 3:
                    gap_samples[probe].append(name)

        for probe, count in sorted(gap_counts.items(), key=lambda kv: -kv[1]):
            w.write(
                Tag.GAP,
                severity=Severity.WARN,
                probe_name=probe,
                full_name_count=count,
                sample_full_names=gap_samples[probe],
                suggestion=_suggestion_for(probe),
            )

        w.summary(
            counts_by_resolution=dict(counts_by_resolution),
            n_gaps_distinct=len(gap_counts),
            n_gaps_total=sum(gap_counts.values()),
        )

    # Terminal summary to stderr for interactive invocations.
    total_tensors = sum(counts_by_resolution.values())
    n_resolved = total_tensors - counts_by_resolution.get("unresolved", 0)
    if gap_counts:
        logger.warning(
            f"{len(gap_counts)} distinct DSL gap(s) across {sum(gap_counts.values())} "
            f"tensor references; "
            f"{n_resolved}/{total_tensors} names resolved — see {out_path}"
        )
    else:
        logger.info(
            f"registry audit clean: {n_resolved}/{total_tensors} tensor names "
            f"resolved against declared slots/params. See {out_path}"
        )
    return 0
