"""Enumerate every tensor across all safetensors shards in a model dir.

Reads the safetensors header (first 8 bytes = little-endian header length,
then a JSON blob mapping tensor name to ``{dtype, shape, data_offsets}``).
This avoids the per-key Python/Rust boundary crossing that
``safe_open(...).get_slice(key)`` imposes — meaningful on big (70B+) shards.
No weight data is read, so this is fast and safe on giant checkpoints.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class HfEntry:
    key: str  # HF tensor name, e.g. "model.layers.0.self_attn.q_proj.weight"
    shape: tuple[int, ...]
    dtype: str  # safetensors dtype string ("BF16", "F16", "F32", ...)
    file: str  # shard filename (basename, not full path)
    nbytes: int


def enumerate_safetensors(model_dir: str, shards: list[dict] | None = None) -> dict[str, HfEntry]:
    """Return ``{tensor_key: HfEntry}`` across all safetensors shards in ``model_dir``.

    Pass ``shards`` (from :func:`shard_files`) to avoid re-parsing the shard list.
    Raises FileNotFoundError if the directory has no safetensors.
    """
    if shards is None:
        shards = shard_files(model_dir)
    if not shards:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    entries: dict[str, HfEntry] = {}
    for shard in shards:
        fname = shard["file"]
        full = shard["path"]
        if not shard.get("exists", True):
            continue
        for key, meta in _read_header(full).items():
            if key == "__metadata__":
                continue
            shape = tuple(int(d) for d in meta.get("shape", []))
            dtype = str(meta.get("dtype", ""))
            offs = meta.get("data_offsets", [0, 0])
            nbytes = int(offs[1] - offs[0]) if len(offs) == 2 else 0
            entries[key] = HfEntry(key=key, shape=shape, dtype=dtype, file=fname, nbytes=nbytes)
    return entries


def shard_files(model_dir: str) -> list[dict]:
    """List every safetensors shard in ``model_dir`` with ``{file, path, size_bytes, exists}``.

    Reads ``model.safetensors.index.json`` when present, otherwise globs for
    ``*.safetensors`` in the root.
    """
    filenames = _list_shard_filenames(model_dir)
    out: list[dict] = []
    for fname in filenames:
        full = os.path.join(model_dir, fname)
        try:
            st = os.stat(full)
            out.append({"file": fname, "path": full, "size_bytes": st.st_size, "exists": True})
        except FileNotFoundError:
            out.append({"file": fname, "path": full, "size_bytes": 0, "exists": False})
    return out


def _list_shard_filenames(model_dir: str) -> list[str]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return sorted(set(index.get("weight_map", {}).values()))
    return [os.path.basename(p) for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))]


def _read_header(path: str) -> dict:
    """Parse the safetensors header without touching tensor data."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            return {}
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = f.read(header_len)
    return json.loads(header_bytes.decode("utf-8"))
