"""
Torch.nn-like module system for Surogate DSL.

Provides a familiar torch.nn-style API for defining models that compiles
to the same IR as the @block/@model decorated classes.

Example:
    from surogate.dsl import nn

    class Qwen3Block(nn.Block):
        def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                     d_ff, max_seq, eps=1e-6, use_qkv_bias=False, use_qk_norm=True):
            super().__init__()
            self.attn_norm = nn.RMSNorm(d_model, eps=eps)
            self.self_attn = nn.Qwen3Attention(
                d_model, num_query_heads, num_kv_heads, head_size,
                max_seq, use_qkv_bias=use_qkv_bias, use_qk_norm=use_qk_norm, eps=eps,
            )
            self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
            self.mlp = nn.SwiGLUMLP(d_model, d_ff)

        def forward(self, x, residual, position_ids):
            residual, h = self.attn_norm(residual, x)
            h = self.self_attn(h, position_ids)
            residual, h = self.mlp_norm(residual, h)
            h = self.mlp(h)
            return h, residual
"""

from __future__ import annotations

import contextvars
import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .tensor_type import Tensor, Array
from .graph_builder import GraphBuilder, GraphRef, GraphNode
from .dim import Dim, DimExpr, B, T
from .specs import (
    BlockSpec,
    ModelSpec,
    ModuleSpec,
    ParamSpec,
    ParamKind,
    ForwardSpec,
    IOSpec,
    ActivationSlotSpec,
    ActivationLayoutSpec,
    ActivationScope,
    SharePolicy,
    HFConfigSpec,
)
from .tensor_type import TensorAnnotation
from .hf import FuseMapping, fuse, stack_experts, transform
from .decorators import (
    _block_registry,
    _model_registry,
    _extract_constructor_params,
)

# ============================================================================
# Tracing context
# ============================================================================

_current_tracer: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar(
    "_current_tracer", default=None
)


# ============================================================================
# Proxy — represents a tensor during forward tracing
# ============================================================================

@dataclass
class Proxy:
    """A tensor reference during forward tracing.

    Wraps a GraphRef so it can be passed to graph builder methods,
    while carrying a human-readable name for activation slot registration.
    """
    name: str
    ref: GraphRef

    def __repr__(self) -> str:
        return f"Proxy({self.name!r})"


# ============================================================================
# Tracer — accumulates graph, params, activation slots
# ============================================================================

class Tracer:
    """Records graph construction during Block/Model forward tracing."""

    def __init__(self) -> None:
        self.graph = GraphBuilder()
        self.params: OrderedDict[str, ParamSpec] = OrderedDict()
        self.forward_slots: list[ActivationSlotSpec] = []
        self.gradient_slots: list[ActivationSlotSpec] = []
        self.hf_mappings: dict[str, Any] = {}
        self._prefix_stack: list[str] = []
        self._slot_names: set[str] = set()
        self._name_remap: dict[str, str] = {}

    # -- prefix management ---------------------------------------------------

    @property
    def prefix(self) -> str:
        return "_".join(self._prefix_stack) if self._prefix_stack else ""

    def prefixed(self, name: str) -> str:
        p = self.prefix
        raw = f"{p}_{name}" if p else name
        return self._name_remap.get(raw, raw)

    def push_prefix(self, name: str) -> None:
        self._prefix_stack.append(name)

    def pop_prefix(self) -> None:
        self._prefix_stack.pop()

    # -- param registration --------------------------------------------------

    def register_param(
        self,
        local_name: str,
        shape: tuple[str | int, ...],
        *,
        dtype: str = "bf16",
        frozen: bool = False,
        quantizable: bool = True,
        when: str | None = None,
        offload_group: int | str = -1,
    ) -> str:
        """Register a parameter and return its fully-qualified name."""
        full_name = self.prefixed(local_name)
        if full_name not in self.params:
            spec = ParamSpec(
                name=full_name,
                kind=ParamKind.TENSOR,
                shape=shape,
                dtype=dtype,
                frozen=frozen,
                quantizable=quantizable,
                offload_group=offload_group,
            )
            if when is not None:
                spec.optional = True
                spec.condition = lambda self_, _w=when: getattr(self_, _w, False)
            self.params[full_name] = spec
        return full_name

    # -- activation slot registration ----------------------------------------

    def register_activation(
        self,
        local_name: str,
        shape: tuple[str | int, ...],
        *,
        dtype: str | None = None,
        save: bool = False,
        share_policy: str | SharePolicy = SharePolicy.PER_LAYER,
        when: str | None = None,
        aliases: list[str] | None = None,
        description: str | None = None,
        scope: ActivationScope = ActivationScope.BLOCK,
    ) -> str:
        """Register an activation slot and return its fully-qualified name."""
        full_name = self.prefixed(local_name)
        if full_name in self._slot_names:
            return full_name

        if isinstance(share_policy, str):
            share_policy = SharePolicy(share_policy)

        self.forward_slots.append(ActivationSlotSpec(
            name=full_name,
            scope=scope,
            shape=shape,
            dtype=dtype,
            aliases=[self.prefixed(a) for a in (aliases or [])],
            save_for_backward=save,
            share_policy=share_policy,
            condition_expr=when,
            description=description,
        ))
        self._slot_names.add(full_name)
        return full_name

    # -- gradient slot registration ------------------------------------------

    def auto_gradients(self) -> None:
        """Auto-generate gradient slots for forward activations.

        Block-scope slots always get gradients. Global-scope slots get
        gradients only if they are computed (i.e., appear as a graph op
        output), not if they are pure inputs (token_ids, targets, etc.)
        or precomputed constants (freq_cis).
        """
        # Collect names that appear as graph operation outputs.
        # Also include global slots that are NOT graph inputs — these are
        # intermediate computation results (e.g., xN, residualN declared
        # as global slots but produced by StackedBlocks under different names).
        computed = set()
        for node in self.graph.nodes:
            for out in node.outputs:
                computed.add(out)
        graph_inputs = set(self.graph._inputs)
        for slot in self.forward_slots:
            if slot.scope == ActivationScope.GLOBAL and slot.name not in graph_inputs:
                # Skip IO/precomputed slots that have specific dtypes (int32, fp32 constants)
                if slot.dtype in ("int32",):
                    continue
                # Skip freq_cis (precomputed constant)
                if "freq" in slot.name:
                    continue
                computed.add(slot.name)

        existing = {s.name for s in self.gradient_slots}

        for slot in self.forward_slots:
            if slot.scope == ActivationScope.BLOCK:
                grad_scope = ActivationScope.GRADIENT
            elif slot.scope == ActivationScope.GLOBAL and slot.name in computed:
                grad_scope = ActivationScope.GLOBAL_GRADIENT
            else:
                continue

            grad_name = f"d_{slot.name}"
            if grad_name in existing:
                continue
            self.gradient_slots.append(ActivationSlotSpec(
                name=grad_name,
                scope=grad_scope,
                shape=slot.shape,
                dtype=slot.dtype if slot.name == "loss" else None,
                gradient_of=slot.name,
            ))

    # -- HF mapping registration --------------------------------------------

    def register_hf_mapping(self, param_name: str, mapping: Any) -> None:
        """Register a HuggingFace weight mapping for a parameter."""
        self.hf_mappings[param_name] = mapping

    # -- proxy helpers -------------------------------------------------------

    def make_proxy(self, local_name: str, ref: GraphRef) -> Proxy:
        return Proxy(self.prefixed(local_name), ref)


# ============================================================================
# Module base class
# ============================================================================

class Module:
    """Base class for nn-style modules.

    Tracks child modules via attribute assignment. During block compilation,
    ``__call__`` delegates to ``_trace()`` which emits graph nodes.
    """

    # Override in subclasses for HuggingFace weight path templates.
    _hf_mapping_defaults_: dict[str, Any] = {}

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_name", "")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module) and name != "_modules":
            if not hasattr(self, "_modules"):
                object.__setattr__(self, "_modules", OrderedDict())
            self._modules[name] = value
            value._name = name
        object.__setattr__(self, name, value)

    # -- tracing entry point -------------------------------------------------

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy | tuple[Proxy, ...]:
        """Emit graph nodes for this module. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._trace()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tracer = _current_tracer.get()
        if tracer is None:
            raise RuntimeError(
                f"{type(self).__name__} can only be called during block/model "
                "compilation (inside compile())"
            )
        tracer.push_prefix(self._name)
        try:
            # Register HF mappings for this module
            for local_param, mapping in self._hf_mapping_defaults_.items():
                tracer.register_hf_mapping(tracer.prefixed(local_param), mapping)
            result = self._trace(tracer, *args, **kwargs)
        finally:
            tracer.pop_prefix()
        return result

    # -- inline graph op helpers (usable in forward) -------------------------

    @staticmethod
    def _view(x: Proxy, shape: list, *, name: str | None = None) -> Proxy:
        """Reshape a proxy tensor."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("view")
        ref = g.view(x.ref, shape=shape, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _add(a: Proxy, b: Proxy, *, name: str | None = None) -> Proxy:
        """Element-wise add."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("add")
        ref = g.add(a.ref, b.ref, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _zeros(shape: list, *, dtype: str = "bf16", name: str | None = None) -> Proxy:
        """Create a zero-filled tensor."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("zeros")
        ref = g.zeros(shape=shape, dtype=dtype, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _mask_scatter(
        x: Proxy, mask: Proxy, values: Proxy, *, name: str | None = None,
    ) -> Proxy:
        """Scatter values into x at positions indicated by mask."""
        tracer = _current_tracer.get()
        g = tracer.graph
        out_name = name or g._fresh_name("mask_scatter")
        ref = g.mask_scatter(x.ref, mask.ref, values.ref, out_name=out_name)
        return Proxy(out_name, ref)

    @staticmethod
    def _register_activation(
        local_name: str,
        shape: tuple[str | int, ...],
        **kwargs: Any,
    ) -> str:
        """Register an activation slot."""
        tracer = _current_tracer.get()
        return tracer.register_activation(local_name, shape, **kwargs)

    @staticmethod
    def _register_gradient(
        name: str,
        shape: tuple[str | int, ...],
        gradient_of: str,
        *,
        dtype: str | None = None,
        scope: ActivationScope = ActivationScope.GRADIENT,
    ) -> None:
        """Register a gradient slot."""
        tracer = _current_tracer.get()
        # Map scope for global gradients
        if scope == ActivationScope.GLOBAL:
            scope = ActivationScope.GLOBAL_GRADIENT
        tracer.gradient_slots.append(ActivationSlotSpec(
            name=name,
            scope=scope,
            shape=shape,
            dtype=dtype,
            gradient_of=gradient_of,
        ))


# ============================================================================
# Concrete modules
# ============================================================================


class RMSNorm(Module):
    """RMS Layer Normalization.

    When called with one argument ``(x,)`` performs standard rmsnorm.
    When called with two arguments ``(residual, x)`` performs fused
    residual-add + rmsnorm.
    """

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
    }

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.C = Dim("C")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy | tuple[Proxy, ...]:
        g = tracer.graph
        weight = tracer.register_param("weight", ("C",), quantizable=False)

        if len(args) == 2:
            # Fused residual + rmsnorm
            residual, x = args

            res_slot = tracer.register_activation(
                "res", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            y_slot = tracer.register_activation(
                "y", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd", ("B", "T"),
                dtype="fp32", save=True,
                share_policy="per_layer",
            )

            res_ref, y_ref, rstd_ref = g.fused_residual_rmsnorm(
                residual.ref, x.ref, weight,
                eps=self.eps,
                res_out_name=res_slot,
                y_name=y_slot,
                rstd_name=rstd_slot,
            )
            return (
                Proxy(res_slot, res_ref),
                Proxy(y_slot, y_ref),
            )

        elif len(args) == 1:
            x = args[0]

            y_slot = tracer.register_activation(
                "y", ("B", "T", "C"),
                share_policy="when_recomputed",
            )
            rstd_slot = tracer.register_activation(
                "rstd", ("B", "T"),
                dtype="fp32", save=True,
                share_policy="per_layer",
            )

            y_ref, rstd_ref = g.rmsnorm(
                x.ref, weight, eps=self.eps,
                y_name=y_slot, rstd_name=rstd_slot,
            )
            return Proxy(y_slot, y_ref)

        raise ValueError(f"RMSNorm expects 1 or 2 inputs, got {len(args)}")


class GQAAttention(Module):
    """Grouped-Query Attention with RoPE and FlashAttention."""

    _hf_mapping_defaults_ = {
        "qkv_weight": fuse(
            "{prefix}.q_proj.weight",
            "{prefix}.k_proj.weight",
            "{prefix}.v_proj.weight",
            dim=0,
        ),
        "qkv_bias": fuse(
            "{prefix}.q_proj.bias",
            "{prefix}.k_proj.bias",
            "{prefix}.v_proj.bias",
            dim=0,
        ),
        "out_weight": "{prefix}.o_proj.weight",
        "out_bias": "{prefix}.o_proj.bias",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params ----------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        qkv_rope = g.rope(
            qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
            rotary_dim="D",
            out_name=qkv_rope_slot,
        )

        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class GptOssAttention(Module):
    """GPT-OSS attention with RoPE and sink tokens."""

    _hf_mapping_defaults_ = {
        **GQAAttention._hf_mapping_defaults_,
        "sinks": "{prefix}.sinks",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params --------------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )
        tracer.register_param("sinks", ("Hq",))

        # -- activation slots ----------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph ---------------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        qkv_rope = g.rope(
            qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
            rotary_dim="D",
            out_name=qkv_rope_slot,
        )

        # FlashAttention with sink tokens
        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            sinks=tracer.prefixed("sinks"),
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Qwen3VLAttention(Module):
    """Qwen3-VL attention with QK-Norm + MRoPE (separate, not fused)."""

    _hf_mapping_defaults_ = {
        **GQAAttention._hf_mapping_defaults_,
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
        eps: float = 1e-6,
        mrope_section: tuple[int, int, int] | list[int] = (24, 20, 20),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias
        self.eps = eps
        self.mrope_section = list(mrope_section)

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params --------------------------------------------------------------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param("q_norm_weight", ("D",), quantizable=False)
        tracer.register_param("k_norm_weight", ("D",), quantizable=False)
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )

        # -- activation slots ----------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_norm_slot = tracer.register_activation(
            "qkv_norm", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
            description="QKV after QK-Norm",
        )
        tracer.register_activation(
            "q_rstd", ("B", "T", "Hq"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        tracer.register_activation(
            "k_rstd", ("B", "T", "Hkv"),
            dtype="fp32", save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
            description="QKV after QK-Norm + MRoPE",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph ---------------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        # QK-Norm (separate from RoPE)
        qkv_norm, q_rstd, k_rstd = g.qkv_qk_norm(
            qkv,
            tracer.prefixed("q_norm_weight"),
            tracer.prefixed("k_norm_weight"),
            eps=self.eps,
            out_name=qkv_norm_slot,
            q_rstd_name=tracer.prefixed("q_rstd"),
            k_rstd_name=tracer.prefixed("k_rstd"),
        )

        # MRoPE (separate)
        qkv_rope = g.mrope(
            qkv_norm,
            tracer.prefixed("rope_freqs"),
            position_ids.ref,
            rotary_dim="D",
            mrope_section=self.mrope_section,
            out_name=qkv_rope_slot,
        )

        # Flash Attention
        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        # Output projection
        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class Qwen3Attention(Module):
    """Qwen3-style attention with QK-Norm + RoPE."""

    _hf_mapping_defaults_ = {
        **GQAAttention._hf_mapping_defaults_,
        "q_norm_weight": "{prefix}.q_norm.weight",
        "k_norm_weight": "{prefix}.k_norm.weight",
    }

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.eps = eps

        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, position_ids = args

        # -- params (always declared, condition controls allocation) ----------
        qkv_w = tracer.register_param("qkv_weight", ("QKV", "C"))
        qkv_b = tracer.register_param("qkv_bias", ("QKV",), when="use_qkv_bias")
        out_w = tracer.register_param("out_weight", ("C", "AttnDim"))
        out_b = tracer.register_param("out_bias", ("C",), when="use_qkv_bias")
        tracer.register_param(
            "q_norm_weight", ("D",), quantizable=False, when="use_qk_norm",
        )
        tracer.register_param(
            "k_norm_weight", ("D",), quantizable=False, when="use_qk_norm",
        )
        tracer.register_param(
            "rope_freqs", ("MaxSeq", "D // 2", 2), dtype="fp32", frozen=True,
        )

        # -- activation slots ------------------------------------------------
        qkv_slot = tracer.register_activation(
            "qkv", ("B", "T", "QKV"),
            aliases=["qkv_flat", "qkv_biased"],
            save=True, share_policy="when_recomputed",
        )
        qkv_rope_slot = tracer.register_activation(
            "qkv_rope", ("B", "T", "QKV"),
            save=True, share_policy="when_recomputed",
            description="QKV after QK-Norm + RoPE",
        )
        tracer.register_activation(
            "q_rstd", ("B", "T", "Hq"),
            dtype="fp32", save=True, share_policy="when_recomputed",
            when="use_qk_norm",
        )
        tracer.register_activation(
            "k_rstd", ("B", "T", "Hkv"),
            dtype="fp32", save=True, share_policy="when_recomputed",
            when="use_qk_norm",
        )
        att_slot = tracer.register_activation(
            "att", ("B", "T", "AttnDim"),
            aliases=["att_flat", "attn"],
            save=True, share_policy="always_recompute",
        )
        tracer.register_activation(
            "lse", ("B", "Hq", "T"),
            dtype="fp32", save=True, share_policy="always_recompute",
        )
        att_out_slot = tracer.register_activation(
            "att_out", ("B", "T", "C"),
            aliases=["att_out_flat"],
            share_policy="when_recomputed",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )

        if self.use_qkv_bias:
            qkv_flat = g.matmul_bias(
                x_flat, qkv_w, qkv_b, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )
        else:
            qkv_flat = g.matmul(
                x_flat, qkv_w, transpose="NT",
                out_name=tracer.prefixed("qkv_flat"),
            )

        qkv = g.view(
            qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D],
            out_name=qkv_slot,
        )

        if self.use_qk_norm:
            qkv_rope, q_rstd, k_rstd = g.qkv_qk_norm_rope(
                qkv,
                tracer.prefixed("q_norm_weight"),
                tracer.prefixed("k_norm_weight"),
                tracer.prefixed("rope_freqs"),
                position_ids.ref,
                eps=self.eps,
                out_name=qkv_rope_slot,
                q_rstd_name=tracer.prefixed("q_rstd"),
                k_rstd_name=tracer.prefixed("k_rstd"),
            )
        else:
            qkv_rope = g.rope(
                qkv, tracer.prefixed("rope_freqs"), position_ids.ref,
                out_name=qkv_rope_slot,
            )

        attn_out, lse = g.flash_attention(
            qkv_rope, causal=True,
            out_name=att_slot,
            lse_name=tracer.prefixed("lse"),
        )

        attn_flat = g.view(
            attn_out, shape=[B * T, self.AttnDim],
            out_name=tracer.prefixed("att_flat"),
        )
        if self.use_qkv_bias:
            out_flat = g.matmul_bias(
                attn_flat, out_w, out_b, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        else:
            out_flat = g.matmul(
                attn_flat, out_w, transpose="NT",
                out_name=tracer.prefixed("att_out_flat"),
            )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=att_out_slot,
        )

        return Proxy(att_out_slot, out)


class SwiGLUMLP(Module):
    """SwiGLU MLP: down(swiglu(up(x)))."""

    _hf_mapping_defaults_ = {
        "up_weight": fuse(
            "{prefix}.up_proj.weight",
            "{prefix}.gate_proj.weight",
            dim=0,
        ),
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = Dim("C")
        self.M = Dim("M")
        self.MUp = 2 * self.M

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        # -- params ----------------------------------------------------------
        up_w = tracer.register_param("up_weight", ("MUp", "C"))
        down_w = tracer.register_param("down_weight", ("C", "M"))

        # -- activation slots ------------------------------------------------
        up_slot = tracer.register_activation(
            "up", ("B", "T", "MUp"),
            aliases=["up_flat"],
            share_policy="when_recomputed",
        )
        act_slot = tracer.register_activation(
            "act", ("B", "T", "M"),
            aliases=["act_flat"],
            share_policy="when_recomputed",
            description="SwiGLU activation output",
        )
        down_slot = tracer.register_activation(
            "down", ("B", "T", "C"),
            aliases=["down_flat"],
            share_policy="when_recomputed",
            description="MLP down projection output",
        )

        # -- graph -----------------------------------------------------------
        x_flat = g.view(
            x.ref, shape=[B * T, self.C],
            out_name=tracer.prefixed("x_flat"),
        )
        up_flat = g.matmul(
            x_flat, up_w, transpose="NT",
            out_name=tracer.prefixed("up_flat"),
        )
        up = g.view(
            up_flat, shape=[B, T, self.MUp],
            out_name=up_slot,
        )
        act = g.swiglu(up, out_name=act_slot)
        act_flat = g.view(
            act, shape=[B * T, self.M],
            out_name=tracer.prefixed("act_flat"),
        )
        out_flat = g.matmul(
            act_flat, down_w, transpose="NT",
            out_name=tracer.prefixed("down_flat"),
        )
        out = g.view(
            out_flat, shape=[B, T, self.C],
            out_name=down_slot,
        )

        return Proxy(down_slot, out)


class GatedMLP(Module):
    """Gated MLP with configurable activation."""

    _hf_mapping_defaults_ = {
        "gate_weight": "{prefix}.gate_proj.weight",
        "up_weight": "{prefix}.up_proj.weight",
        "down_weight": "{prefix}.down_proj.weight",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.C = Dim("C")
        self.M = Dim("M")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        gate_w = tracer.register_param("gate_weight", ("M", "C"))
        up_w = tracer.register_param("up_weight", ("M", "C"))
        down_w = tracer.register_param("down_weight", ("C", "M"))

        tracer.register_activation(
            "gate", ("B", "T", "M"), share_policy="when_recomputed",
        )
        tracer.register_activation(
            "up", ("B", "T", "M"), share_policy="when_recomputed",
        )
        down_slot = tracer.register_activation(
            "down", ("B", "T", "C"), share_policy="when_recomputed",
        )

        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))
        gate_flat = g.matmul(x_flat, gate_w, transpose="NT", out_name=tracer.prefixed("gate_flat"))
        up_flat = g.matmul(x_flat, up_w, transpose="NT", out_name=tracer.prefixed("up_flat"))

        act_map = {"silu": g.silu, "relu": g.relu, "relu2": g.relu2, "gelu": g.gelu}
        act_fn = act_map.get(self.activation, g.silu)
        gate_act = act_fn(gate_flat, out_name=tracer.prefixed("gate_act"))
        hidden = g.mul(gate_act, up_flat)

        out_flat = g.matmul(hidden, down_w, transpose="NT", out_name=tracer.prefixed("down_flat"))
        out = g.view(out_flat, shape=[B, T, self.C], out_name=down_slot)

        return Proxy(down_slot, out)


class MoEExpertsGated(Module):
    """MoE with gated expert activation (SwiGLU-style: gate+up fused).

    Handles: router → top-k → permute → grouped GEMM (gate+up) → SwiGLU →
    grouped GEMM (down) → unpermute.  Optionally supports expert parallelism
    (ep_size > 1) with all-to-all dispatch/combine.
    """

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.gate.weight",
        "experts_gate_up": stack_experts(
            "{prefix}.experts.{expert}.gate_proj.weight",
            fuse_gate_up=True,
        ),
        "experts_down": stack_experts(
            "{prefix}.experts.{expert}.down_proj.weight",
        ),
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        # -- params ----------------------------------------------------------
        tracer.register_param("router_weight", ("E", "C"))
        tracer.register_param("experts_gate_up", ("E", "MUp", "C"), offload_group="moe_experts")
        tracer.register_param("experts_down", ("E", "C", "M"), offload_group="moe_experts")

        # -- activation slots ------------------------------------------------
        tracer.register_activation(
            "router_logits", ("B * T", "E"),
            save=True, share_policy="fft_share",
            description="Router logits before softmax",
        )
        tracer.register_activation(
            "router_probs", ("B * T", "E"),
            save=True, share_policy="fft_share",
            description="Router probabilities",
        )
        tracer.register_activation(
            "routing_weights", ("B * T", "K"),
            save=True, share_policy="fft_share",
            description="Routing weights for selected experts",
        )
        tracer.register_activation(
            "routing_indices", ("B * T", "K"),
            dtype="int32", save=True, share_policy="fft_share",
            description="Expert indices for each token",
        )
        tracer.register_activation(
            "permuted_input", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
            description="Permuted input for grouped GEMM",
        )
        tracer.register_activation(
            "scatter_indices", ("B * T * K",),
            dtype="int32", save=True, share_policy="fft_share",
            description="Indices for scattering back",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
                description="EP-dispatched input tokens",
            )
            tracer.register_activation(
                "ep_recv_scatter", ("B * T * K",),
                dtype="int32", save=True, share_policy="per_layer",
                when="ep_size > 1",
                description="EP-dispatched scatter indices",
            )
        tracer.register_activation(
            "expert_gate_up", ("B * T * K", "MUp"),
            save=True, share_policy="fft_share",
            description="Expert gate+up projection output",
        )
        tracer.register_activation(
            "expert_act", ("B * T * K", "M"),
            save=True, share_policy="fft_share",
            description="Expert SwiGLU activation output",
        )
        tracer.register_activation(
            "expert_down", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
            description="Expert down projection output",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
                description="EP-combined expert output",
            )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"),
            aliases=["out_flat"],
            save=True, share_policy="fft_share",
            description="Combined MoE output",
        )

        # -- graph -----------------------------------------------------------
        router_logits = g.matmul(
            x.ref, tracer.prefixed("router_weight"), transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )
        router_probs = g.moe_softmax(
            router_logits, out_name=tracer.prefixed("router_probs"),
        )
        routing_weights, routing_indices = g.moe_topk(
            router_probs, top_k=self.num_experts_per_tok,
            normalize=self.norm_topk_prob,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )
        permuted_input, scatter_indices = g.moe_permute(
            x.ref, routing_indices, top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input, routing_indices, scatter_indices,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input, tracer.prefixed("experts_gate_up"), gemm_scatter,
            out_name=tracer.prefixed("expert_gate_up"),
        )
        expert_act = g.swiglu(
            expert_gate_up, out_name=tracer.prefixed("expert_act"),
        )
        expert_down = g.moe_grouped_gemm_down(
            expert_act, tracer.prefixed("experts_down"), gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        if self.ep_size > 1:
            expert_down = g.ep_combine(
                expert_down,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        moe_out = g.moe_unpermute(
            expert_down, routing_weights, scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class MoESharedExpert(Module):
    """Shared expert for MoE models (SwiGLU-style: gate + up → silu*mul → down)."""

    _hf_mapping_defaults_ = {
        "gate": "{prefix}.shared_expert.gate_proj.weight",
        "up": "{prefix}.shared_expert.up_proj.weight",
        "down": "{prefix}.shared_expert.down_proj.weight",
    }

    def __init__(self, d_model: int, shared_expert_intermediate: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")
        self.SharedM = Dim("SharedM")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        tracer.register_param("gate", ("SharedM", "C"))
        tracer.register_param("up", ("SharedM", "C"))
        tracer.register_param("down", ("C", "SharedM"))

        tracer.register_activation(
            "gate", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "up", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "gate_act", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        tracer.register_activation(
            "hidden", ("B * T", "SharedM"), share_policy="fft_share",
            when="use_shared_expert",
        )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"), share_policy="fft_share",
            when="use_shared_expert",
        )

        shared_gate = g.matmul(
            x.ref, tracer.prefixed("gate"), transpose="NT",
            out_name=tracer.prefixed("gate"),
        )
        shared_up = g.matmul(
            x.ref, tracer.prefixed("up"), transpose="NT",
            out_name=tracer.prefixed("up"),
        )
        shared_gate_act = g.silu(shared_gate, out_name=tracer.prefixed("gate_act"))
        shared_hidden = g.mul(shared_gate_act, shared_up)
        shared_out = g.matmul(
            shared_hidden, tracer.prefixed("down"), transpose="NT",
            out_name=out_slot,
        )

        return Proxy(out_slot, shared_out)


class GptOssMoEExperts(Module):
    """GPT-OSS MoE with router bias, per-expert biases, and gpt_oss_moe_act."""

    _hf_mapping_defaults_ = {
        "router_weight": "{prefix}.router.weight",
        "router_bias": "{prefix}.router.bias",
        "experts_gate_up": transform("{prefix}.experts.gate_up_proj", fn="transpose"),
        "experts_gate_up_bias": "{prefix}.experts.gate_up_proj_bias",
        "experts_down": transform("{prefix}.experts.down_proj", fn="transpose"),
        "experts_down_bias": "{prefix}.experts.down_proj_bias",
    }

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int = 4,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.ep_size = ep_size

        self.C = Dim("C")
        self.M = Dim("M")
        self.E = Dim("E")
        self.K = Dim("K")
        self.MUp = 2 * self.M

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args  # x is [B*T, C]

        # -- params --------------------------------------------------------------
        tracer.register_param("router_weight", ("E", "C"))
        tracer.register_param("router_bias", ("E",))
        tracer.register_param(
            "experts_gate_up", ("E", "MUp", "C"), offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_gate_up_bias", ("E", "MUp"), offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_down", ("E", "C", "M"), offload_group="moe_experts",
        )
        tracer.register_param(
            "experts_down_bias", ("E", "C"), offload_group="moe_experts",
        )

        # -- activation slots ----------------------------------------------------
        tracer.register_activation(
            "router_logits", ("B * T", "E"),
            save=True, share_policy="per_layer",
            description="Router logits",
        )
        tracer.register_activation(
            "routing_weights", ("B * T", "K"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "routing_indices", ("B * T", "K"),
            dtype="int32", save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "permuted_input", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "scatter_indices", ("B * T * K",),
            dtype="int32", save=True, share_policy="fft_share",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_recv_input", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
            tracer.register_activation(
                "ep_recv_scatter", ("B * T * K",),
                dtype="int32", save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
        tracer.register_activation(
            "expert_gate_up", ("B * T * K", "MUp"),
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_gate_up_bias", ("B * T * K", "MUp"),
            save=True, share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_act", ("B * T * K", "M"),
            save=True, share_policy="fft_share",
            description="GPT-OSS activation output",
        )
        tracer.register_activation(
            "expert_down", ("B * T * K", "C"),
            share_policy="fft_share",
        )
        tracer.register_activation(
            "expert_down_bias", ("B * T * K", "C"),
            save=True, share_policy="fft_share",
        )
        if self.ep_size > 1:
            tracer.register_activation(
                "ep_combined", ("B * T * K", "C"),
                save=True, share_policy="per_layer",
                when="ep_size > 1",
            )
        out_slot = tracer.register_activation(
            "out", ("B * T", "C"),
            aliases=["out_flat"],
            save=True, share_policy="fft_share",
        )

        # -- graph ---------------------------------------------------------------
        # Router (with bias)
        router_logits = g.matmul_bias(
            x.ref, tracer.prefixed("router_weight"),
            tracer.prefixed("router_bias"), transpose="NT",
            out_name=tracer.prefixed("router_logits"),
        )

        # Top-k (softmax + sort_by_index, no normalize)
        routing_weights, routing_indices = g.moe_topk(
            router_logits, top_k=self.num_experts_per_tok,
            normalize=False, softmax=True, sort_by_index=True,
            weights_name=tracer.prefixed("routing_weights"),
            indices_name=tracer.prefixed("routing_indices"),
        )

        # Permute
        permuted_input, scatter_indices = g.moe_permute(
            x.ref, routing_indices, top_k=self.num_experts_per_tok,
            out_name=tracer.prefixed("permuted_input"),
            scatter_name=tracer.prefixed("scatter_indices"),
        )

        # EP dispatch
        if self.ep_size > 1:
            ep_recv_input, ep_recv_scatter = g.ep_dispatch(
                permuted_input, routing_indices, scatter_indices,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_recv_input"),
                recv_scatter_name=tracer.prefixed("ep_recv_scatter"),
            )
            gemm_input = ep_recv_input
            gemm_scatter = ep_recv_scatter
        else:
            gemm_input = permuted_input
            gemm_scatter = scatter_indices

        # Gate+up GEMM (interleaved)
        expert_gate_up = g.moe_grouped_gemm_gate_up(
            gemm_input, tracer.prefixed("experts_gate_up"), gemm_scatter,
            gate_up_interleaved=True,
            out_name=tracer.prefixed("expert_gate_up"),
        )

        # Gate+up bias
        expert_gate_up_bias = g.moe_expert_bias_add(
            expert_gate_up, tracer.prefixed("experts_gate_up_bias"),
            out_name=tracer.prefixed("expert_gate_up_bias"),
        )

        # GPT-OSS activation
        expert_act = g.gpt_oss_moe_act(
            expert_gate_up_bias, alpha=1.702, limit=7.0,
            out_name=tracer.prefixed("expert_act"),
        )

        # Down GEMM
        expert_down = g.moe_grouped_gemm_down(
            expert_act, tracer.prefixed("experts_down"), gemm_scatter,
            out_name=tracer.prefixed("expert_down"),
        )

        # Down bias
        expert_down_bias = g.moe_expert_bias_add(
            expert_down, tracer.prefixed("experts_down_bias"),
            out_name=tracer.prefixed("expert_down_bias"),
        )

        # EP combine
        if self.ep_size > 1:
            expert_down_bias = g.ep_combine(
                expert_down_bias,
                num_experts=self.num_experts, ep_size=self.ep_size,
                top_k=self.num_experts_per_tok,
                out_name=tracer.prefixed("ep_combined"),
            )

        # Unpermute
        moe_out = g.moe_unpermute(
            expert_down_bias, routing_weights, scatter_indices,
            top_k=self.num_experts_per_tok,
            out_name=out_slot,
        )

        return Proxy(out_slot, moe_out)


class Linear(Module):
    """Linear projection: y = x @ W^T (+ bias)."""

    _hf_mapping_defaults_ = {
        "weight": "{prefix}.weight",
        "bias": "{prefix}.bias",
    }

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.C = Dim("in_dim")
        self.O = Dim("out_dim")

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (x,) = args

        w = tracer.register_param("weight", ("O", "C"))
        if self.use_bias:
            b = tracer.register_param("bias", ("O",), when="use_bias")

        out_slot = tracer.register_activation(
            "out", ("B", "T", "O"), share_policy="when_recomputed",
        )

        x_flat = g.view(x.ref, shape=[B * T, self.C], out_name=tracer.prefixed("x_flat"))
        if self.use_bias:
            y_flat = g.matmul_bias(x_flat, w, b, transpose="NT", out_name=tracer.prefixed("y_flat"))
        else:
            y_flat = g.matmul(x_flat, w, transpose="NT", out_name=tracer.prefixed("y_flat"))
        out = g.view(y_flat, shape=[B, T, self.O], out_name=out_slot)

        return Proxy(out_slot, out)


class Embedding(Module):
    """Embedding lookup table."""

    _hf_mapping_defaults_ = {
        "weight": "model.embed_tokens.weight",
    }

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        (token_ids,) = args

        w = tracer.register_param("weight", ("vocab_size", "d_model"))

        out_slot = tracer.register_activation(
            "out", ("B", "T", "d_model"),
            scope=ActivationScope.GLOBAL,
            description="Embedded input",
        )

        out = g.embedding(token_ids.ref, w, out_name=out_slot)
        return Proxy(out_slot, out)


class LMHead(Module):
    """Fused LM head projection + cross-entropy loss."""

    _hf_mapping_defaults_ = {
        "weight": "lm_head.weight",
    }

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> Proxy:
        g = tracer.graph
        x, targets = args

        w = tracer.register_param("weight", ("vocab_size", "d_model"))

        loss_slot = tracer.register_activation(
            "loss", ("B * T",),
            dtype="fp32",
            scope=ActivationScope.GLOBAL,
            description="Cross-entropy loss per token",
        )

        x_flat = g.view(
            x.ref, shape=["B * T", "d_model"],
            out_name=tracer.prefixed("x_flat"),
        )
        loss = g.fused_lm_head_loss(
            x_flat, w, targets.ref,
            compute_accuracy=True,
            out_name=loss_slot,
        )
        return Proxy(loss_slot, loss)


# ============================================================================
# Canonical name remaps — map auto-prefixed nn names to C++ runtime names
# ============================================================================

# Dense transformer block: attn_norm / self_attn / mlp_norm / mlp
DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNorm) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (Attention) -> strip prefix ---
    "self_attn_qkv_weight": "qkv_weight",
    "self_attn_qkv_bias": "qkv_bias",
    "self_attn_out_weight": "out_weight",
    "self_attn_out_bias": "out_bias",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_flat": "qkv_flat",
    "self_attn_qkv_biased": "qkv_biased",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_attn": "attn",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    "self_attn_x_flat": "x_flat",
    # --- mlp_norm (RMSNorm) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (SwiGLUMLP) ---
    # mlp_up_weight, mlp_down_weight, mlp_up, mlp_down, mlp_up_flat,
    # mlp_down_flat are already correct (mlp_ prefix matches canonical names)
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",  # intermediate, kept as-is
}

# MoE block: same attn + norm, but moe instead of mlp
MOE_BLOCK_NAME_REMAP: dict[str, str] = {
    # Inherit all attn_norm and self_attn mappings
    **{k: v for k, v in DENSE_BLOCK_NAME_REMAP.items()
       if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # --- moe (MoEExpertsGated) -> strip moe_ prefix ---
    # Params
    "moe_router_weight": "router_weight",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_down": "experts_down",
    "moe_experts_up": "experts_up",
    # Activations
    "moe_router_logits": "router_logits",
    "moe_router_probs": "router_probs",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_ep_combined": "ep_combined",
    # moe_out / moe_out_flat already match canonical names
    # --- shared_expert (MoESharedExpert) ---
    # Params use local names gate/up/down; after shared_expert_ prefix they
    # become shared_expert_gate etc. which are already the canonical names.
    # Activation intermediates keep their prefixed names (no remap needed).
}

# GPT-OSS MoE block: GptOssAttention (with sinks) + GptOssMoEExperts (with biases)
GPT_OSS_BLOCK_NAME_REMAP: dict[str, str] = {
    # Inherit attn_norm, self_attn, mlp_norm mappings from dense
    **{k: v for k, v in DENSE_BLOCK_NAME_REMAP.items()
       if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # GPT-OSS attention extra: sinks
    "self_attn_sinks": "sinks",
    # --- moe (GptOssMoEExperts) -> strip moe_ prefix ---
    # Params
    "moe_router_weight": "router_weight",
    "moe_router_bias": "router_bias",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_gate_up_bias": "experts_gate_up_bias",
    "moe_experts_down": "experts_down",
    "moe_experts_down_bias": "experts_down_bias",
    # Activations
    "moe_router_logits": "router_logits",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_gate_up_bias": "expert_gate_up_bias",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_expert_down_bias": "expert_down_bias",
    "moe_ep_combined": "ep_combined",
    # moe_out / moe_out_flat already match canonical names
}

# VL dense block: same as dense but with QK-Norm + MRoPE (separate, extra qkv_norm slot)
VL_DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    **DENSE_BLOCK_NAME_REMAP,
    "self_attn_qkv_norm": "qkv_norm",
}

# Model-level remap for embedding / final_norm / lm_head
STANDARD_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- final_norm (RMSNorm) ---
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

# VL model-level: embedding_out stays auto-named (mask_scatter output is x0)
VL_MODEL_NAME_REMAP: dict[str, str] = {
    "embedding_weight": "embedding",
    # No "embedding_out" → "x0" — mask_scatter output takes that name
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}


# ============================================================================
# Block base class
# ============================================================================

class Block(Module):
    """Base class for transformer blocks.

    Subclasses define ``__init__`` (composing nn modules) and ``forward``
    (calling them).  ``compile()`` traces the forward pass and generates a
    ``BlockSpec`` compatible with the existing DSL compiler pipeline.

    Subclasses may define ``_name_remap_`` to map auto-generated prefixed
    names to canonical names expected by the C++ runtime::

        class Qwen3Block(nn.Block):
            _name_remap_ = {
                "attn_norm_y": "ln1",
                "attn_norm_rstd": "ln1_rstd",
                ...
            }

    Example::

        class Qwen3Block(nn.Block):
            def __init__(self, d_model, num_query_heads, ...):
                super().__init__()
                self.attn_norm = nn.RMSNorm(d_model, eps=eps)
                self.self_attn = nn.Qwen3Attention(...)
                self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
                self.mlp = nn.SwiGLUMLP(d_model, d_ff)

            def forward(self, x, residual, position_ids):
                residual, h = self.attn_norm(residual, x)
                h = self.self_attn(h, position_ids)
                residual, h = self.mlp_norm(residual, h)
                h = self.mlp(h)
                return h, residual
    """

    # Override in subclasses to remap auto-generated prefixed names to
    # canonical names expected by the C++ runtime.
    _name_remap_: dict[str, str] = {}

    def forward(self, *args: Proxy, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward()")

    def compile(self) -> BlockSpec:
        """Trace forward and produce a BlockSpec."""
        tracer = Tracer()
        tracer._name_remap = getattr(type(self), '_name_remap_', {})
        token = _current_tracer.set(tracer)
        try:
            # Build proxy inputs from forward() signature
            sig = inspect.signature(self.forward)
            input_proxies = {}
            for name in sig.parameters:
                ref = tracer.graph.input(name)
                input_proxies[name] = Proxy(name, ref)

            # Trace
            result = self.forward(**input_proxies)

            # Auto-generate gradient slots
            tracer.auto_gradients()
        finally:
            _current_tracer.reset(token)

        # Build ForwardSpec with input/output specs for the compiler
        # Create a generic TensorAnnotation placeholder for each IO
        _generic_tensor = Tensor["B", "T", "C"]
        _slot_map_model = {s.name: s for s in tracer.forward_slots}

        input_specs = []
        for name in sig.parameters:
            slot = _slot_map_model.get(name)
            if slot and slot.shape:
                tt = TensorAnnotation(dims=tuple(slot.shape), dtype=slot.dtype or "bf16")
            else:
                tt = _generic_tensor
            input_specs.append(IOSpec(name=name, tensor_type=tt))

        output_specs = []
        if isinstance(result, tuple):
            for i, proxy in enumerate(result):
                out_name = proxy.name if isinstance(proxy, Proxy) else f"out_{i}"
                output_specs.append(IOSpec(name=out_name, tensor_type=_generic_tensor))
        elif isinstance(result, Proxy):
            output_specs.append(IOSpec(name=result.name, tensor_type=_generic_tensor))

        forward_spec = ForwardSpec(
            inputs=input_specs,
            outputs=output_specs,
            graph_fn=lambda self_, g: None,  # graph already built
        )
        forward_spec._traced_graph = tracer.graph

        # Build ActivationLayoutSpec
        activation_layout = ActivationLayoutSpec(
            name=f"{type(self).__name__}Activations",
            slots=tracer.forward_slots,
            gradient_slots=tracer.gradient_slots,
        )

        # Build BlockSpec
        spec = BlockSpec(
            name=type(self).__name__,
            python_class=type(self),
            docstring=type(self).__doc__,
            constructor_params=_extract_constructor_params(type(self)),
            params=tracer.params,
            forward=forward_spec,
            activations=activation_layout,
        )

        # Register in block registry
        _block_registry[type(self).__name__] = spec

        return spec


# ============================================================================
# Model base class
# ============================================================================

class Model(Module):
    """Base class for full models.

    Subclasses define ``__init__`` (composing nn modules) and ``forward``
    (calling them).  ``compile()`` traces the forward pass and generates a
    ``ModelSpec`` compatible with the existing DSL compiler pipeline.
    """

    # Override in subclasses via @nn.hf_config decorator
    _hf_config_: HFConfigSpec | None = None

    # Override in subclasses to remap auto-generated prefixed names to
    # canonical names expected by the C++ runtime.
    _name_remap_: dict[str, str] = {}

    def forward(self, *args: Proxy, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward()")

    def compile(self) -> ModelSpec:
        """Trace forward and produce a ModelSpec."""
        tracer = Tracer()
        tracer._name_remap = getattr(type(self), '_name_remap_', {})
        token = _current_tracer.set(tracer)
        try:
            sig = inspect.signature(self.forward)
            input_proxies = {}
            for name in sig.parameters:
                ref = tracer.graph.input(name)
                input_proxies[name] = Proxy(name, ref)

            result = self.forward(**input_proxies)
            tracer.auto_gradients()
        finally:
            _current_tracer.reset(token)

        # Build IO specs using activation slot shapes when available so the
        # C++ graph compiler gets correct input shapes (e.g., token_ids is 2D int32,
        # not 3D bf16).
        _generic_tensor = Tensor["B", "T", "C"]
        _slot_map = {s.name: s for s in tracer.forward_slots}

        input_specs = []
        for name in sig.parameters:
            slot = _slot_map.get(name)
            if slot and slot.shape:
                tensor_type = TensorAnnotation(dims=tuple(slot.shape), dtype=slot.dtype or "bf16")
            else:
                tensor_type = _generic_tensor
            input_specs.append(IOSpec(name=name, tensor_type=tensor_type))

        output_specs = []
        if isinstance(result, Proxy):
            slot = _slot_map.get(result.name)
            tt = Tensor[tuple(slot.shape)] if slot and slot.shape else _generic_tensor
            output_specs.append(IOSpec(name=result.name, tensor_type=tt))
        elif isinstance(result, tuple):
            for i, p in enumerate(result):
                nm = p.name if isinstance(p, Proxy) else f"out_{i}"
                slot = _slot_map.get(nm)
                tt = Tensor[tuple(slot.shape)] if slot and slot.shape else _generic_tensor
                output_specs.append(IOSpec(name=nm, tensor_type=tt))

        forward_spec = ForwardSpec(
            inputs=input_specs,
            outputs=output_specs,
            graph_fn=lambda self_, g: None,
        )
        forward_spec._traced_graph = tracer.graph

        activation_layout = ActivationLayoutSpec(
            name=f"{type(self).__name__}Activations",
            slots=tracer.forward_slots,
            gradient_slots=tracer.gradient_slots,
        )

        hf_mapping_spec = None
        if tracer.hf_mappings:
            from .specs import HFMappingSpec
            hf_mapping_spec = HFMappingSpec(mappings=tracer.hf_mappings)

        spec = ModelSpec(
            name=type(self).__name__,
            python_class=type(self),
            docstring=type(self).__doc__,
            constructor_params=_extract_constructor_params(type(self)),
            params=tracer.params,
            forward=forward_spec,
            activations=activation_layout,
            hf_config=getattr(type(self), "_hf_config_", None),
            hf_mapping=hf_mapping_spec,
        )

        _model_registry[type(self).__name__] = spec
        return spec


# ============================================================================
# hf_config decorator for Model subclasses
# ============================================================================

def hf_config(
    architecture: str,
    model_type: str,
    **field_mappings: str,
) -> Any:
    """Decorator that attaches HuggingFace config mapping to a Model class.

    Usage::

        @nn.hf_config(
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            d_model="hidden_size",
            n_layers="num_hidden_layers",
            ...
        )
        class Qwen3(nn.Model):
            ...
    """
    def decorator(cls: type) -> type:
        hf_cfg = HFConfigSpec(
            architecture=architecture,
            model_type=model_type,
            param_mapping=field_mappings,
        )
        cls._hf_config_ = hf_cfg

        # Register a lazy ModelSpec in the registry so compile_model_for_hf
        # can find this model by architecture name.  The spec carries
        # hf_config for lookup and a graph_fn that triggers nn compilation.
        def _lazy_graph_fn(self_instance: Any, g: Any) -> None:
            pass  # placeholder — real graph is built by nn.Model.compile()

        lazy_spec = ModelSpec(
            name=cls.__name__,
            python_class=cls,
            docstring=cls.__doc__,
            constructor_params=_extract_constructor_params(cls),
            forward=ForwardSpec(graph_fn=_lazy_graph_fn),
            hf_config=hf_cfg,
        )
        # Mark as nn-style so the compiler knows to call compile()
        lazy_spec._nn_model_class = cls
        _model_registry[cls.__name__] = lazy_spec

        return cls
    return decorator


# ============================================================================
# BlockStack — helper for stacking blocks in a model
# ============================================================================

class BlockStack(Module):
    """Calls StackedBlocks with the given block type.

    Used in Model definitions::

        self.blocks = nn.BlockStack(n_layers, Qwen3Block, d_model=..., ...)
    """

    def __init__(self, n_layers: int, block_cls: type, **block_kwargs: Any) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.block_cls = block_cls
        self.block_kwargs = block_kwargs
        # Instantiate one block to get its spec
        self._block_instance = block_cls(**block_kwargs)

    def _trace(
        self, tracer: Tracer, *args: Proxy, **kwargs: Any
    ) -> tuple[Proxy, ...]:
        g = tracer.graph

        # Register the block array as a param (ARRAY kind, not TENSOR).
        # Use the module attribute name (tracer.prefix) as the param name,
        # matching the original DSL convention (e.g., "blocks").
        block_name = self.block_cls.__name__
        param_name = tracer.prefix  # e.g., "blocks"
        if param_name not in tracer.params:
            tracer.params[param_name] = ParamSpec(
                name=param_name,
                kind=ParamKind.ARRAY,
                array_size="n_layers",
                element_type=block_name,
            )

        # Compile the inner block to register it
        self._block_instance.compile()

        # Emit StackedBlocks call — num_outputs matches block return count.
        # The block returns (out, residual) = 2 outputs, while receiving
        # (x, residual, position_ids) = 3 inputs.
        block_sig = inspect.signature(self._block_instance.forward)
        block_params = [p for p in block_sig.parameters]
        num_outputs = len(block_params) - 1  # exclude position_ids

        input_refs = [a.ref for a in args]
        result = g.call(
            "StackedBlocks",
            *input_refs,
            num_outputs=num_outputs,
            blocks=param_name,
            n_layers=self.n_layers,
            **kwargs,
        )

        if isinstance(result, GraphRef):
            return (Proxy("stacked_out", result),)
        return tuple(
            Proxy(f"stacked_out_{i}", r) for i, r in enumerate(result)
        )
