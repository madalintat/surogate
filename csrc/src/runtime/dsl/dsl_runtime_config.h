// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime configuration derived from the Python DSL.

#ifndef SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H
#define SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H

#include <string>
#include <unordered_set>
#include <vector>

#include "config/rope_config.h"

namespace dsl {

/// Per-layer dimensions for hybrid models where block types have different
/// head sizes, QKV channels, or intermediate sizes (e.g., Gemma4).
struct BlockTypeDims {
    long head_size = 0;
    long qkv_channels = 0;  ///< D * (Hq + 2*Hkv) or D * (Hq + Hkv) for k_eq_v
    long attn_dim = 0;      ///< Hq * D
    long intermediate = 0;  ///< M (may be 2x for double-wide MLP)
    long mlp_up = 0;        ///< up_factor * M
};

struct LayerRoPEConfig {
    long head_size = 0;
    RoPEConfig rope;
};

struct DslRuntimeConfig {
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    bool use_qk_norm = false;
    bool norm_topk_prob = false;
    bool use_shared_expert = false;
    int shared_expert_intermediate = 0;

    /// Per-layer dimensions. Empty for homogeneous models (use global config).
    std::vector<BlockTypeDims> per_layer_dims;

    /// Per-layer RoPE parameters for hybrid models where layer types use
    /// different head sizes or rope formulas (e.g. Gemma4 full vs sliding).
    std::vector<LayerRoPEConfig> per_layer_rope;

    /// Layer indices whose QKV/qkv_rope activations must be persisted across
    /// layer boundaries (not shared) because other layers read them as
    /// kv_source for shared-KV attention.
    std::unordered_set<int> kv_source_layers;

    [[nodiscard]] bool is_moe() const {
        return num_experts > 0;
    }
    [[nodiscard]] bool has_per_layer_dims() const {
        return !per_layer_dims.empty();
    }
    [[nodiscard]] bool has_per_layer_rope() const {
        return !per_layer_rope.empty();
    }
    [[nodiscard]] bool is_kv_source(int layer_idx) const {
        return kv_source_layers.count(layer_idx) > 0;
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H
