"""Surogate debug tooling.

Command-line debug subcommands for introspecting DSL models:
  - weights:     static audit of HF safetensors keys vs DSL expected params
  - activations: per-layer forward activation stats (min/max/mean/nan%)
  - gradients:   per-param + per-intermediate backward gradient stats
  - diff:        layer-by-layer numerical diff vs HuggingFace transformers reference

Each subcommand emits a grep-friendly JSONL file (one record per line) plus a
sidecar .header.json with run context. See ``surogate/debug/schema.py`` for the
record tag vocabulary.
"""
