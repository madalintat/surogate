"""`surogate debug` command — introspection + diagnostics for DSL models.

Subcommands:
    weights      Static audit: HF safetensors keys vs DSL expected params.
    activations  Per-layer forward activation stats from a single step.
    gradients    Per-step, per-param, and per-intermediate backward gradient stats.
    diff         (stub) Numerical diff vs HuggingFace transformers reference.

Every subcommand writes a JSONL file + a .header.json sidecar. One record per
line, tagged for grep; see ``surogate/debug/schema.py`` for the vocabulary.

Example:
    surogate debug weights examples/sft/gemma4/gemma4-e2b-lora-bf16.yaml
"""

import argparse
import sys

from surogate.utils.logger import get_logger

logger = get_logger()


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", type=str, help="Path to training config YAML (same format as `surogate sft`)")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path. Defaults to ./debug/debug_<sub>_<model>_<ts>.jsonl",
    )


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(prog="surogate debug")

    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    p_weights = sub.add_parser(
        "weights",
        help="Static audit of HF safetensors vs DSL expected params (no forward pass)",
    )
    _add_config_arg(p_weights)
    p_weights.add_argument("--hub_token", type=str, default=None, help="HuggingFace Hub token for private models")

    p_acts = sub.add_parser("activations", help="Trace forward activation stats per layer/op")
    _add_config_arg(p_acts)
    p_acts.add_argument("--hub_token", type=str, default=None, help="HuggingFace Hub token for private models")

    p_grads = sub.add_parser("gradients", help="Trace per-param + per-intermediate backward gradient stats")
    _add_config_arg(p_grads)
    p_grads.add_argument("--hub_token", type=str, default=None, help="HuggingFace Hub token for private models")
    p_grads.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of fwd+bwd+optimizer cycles to run (capture gradients at each). "
        "Use >1 to diagnose degradation after N steps. Default: 1",
    )

    p_diff = sub.add_parser("diff", help="Layer-by-layer numerical diff vs HuggingFace transformers [STUB]")
    _add_config_arg(p_diff)
    p_diff.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path/repo of the HF reference model (defaults to config.model)",
    )

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.subcommand == "weights":
        from surogate.debug.weights import run_weight_audit

        return run_weight_audit(args.config, output=args.output, hub_token=args.hub_token)

    if args.subcommand == "activations":
        from surogate.debug.activations import run_activation_trace

        return run_activation_trace(args.config, output=args.output, hub_token=args.hub_token)

    if args.subcommand == "gradients":
        from surogate.debug.gradients import run_gradient_trace

        return run_gradient_trace(
            args.config,
            output=args.output,
            hub_token=args.hub_token,
            steps=args.steps,
        )

    if args.subcommand == "diff":
        from surogate.debug.diff import run_reference_diff

        return run_reference_diff(args.config, output=args.output, reference=args.reference)

    logger.error("no debug subcommand specified; try `surogate debug --help`")
    return 1


if __name__ == "__main__":
    parser = prepare_command_parser()
    args = parser.parse_args(sys.argv[1:])
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)
    sys.exit(_dispatch(args))
