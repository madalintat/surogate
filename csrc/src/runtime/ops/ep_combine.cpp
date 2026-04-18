// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert Parallelism combine op wiring.
//
// The real combine/backward implementations live in
// `runtime/ep/ep_strategy.cpp`. This file keeps only:
//   - the executor-member trampolines that forward to the strategy
//   - the autodiff rule registration
//   - the shape-signature registration

#include "runtime/executor/compiled_ops.h"

#include <vector>

#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/ep/ep_strategy.h"
#include "runtime/executor/op_registry.h"

namespace dsl {

void CompiledExecutor::dispatch_ep_combine(const CompiledOp& op) {
    mEpStrategy->combine_forward(*this, op);
}

void CompiledExecutor::dispatch_ep_combine_backward(const CompiledOp& op) {
    mEpStrategy->combine_backward(*this, op);
}

namespace {

// -----------------------------------------------------------------------------
// EP Combine backward rule
// Forward: combined = ep_combine(expert_output, ...)
// Backward: d_expert_output = ep_combine_backward(d_combined)
// -----------------------------------------------------------------------------
std::vector<Operation> ep_combine_backward_rule(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        AttrMap attrs = copy_attrs(fwd.attrs, {"ep_size", "num_experts", "top_k"}, "ep_combine_backward");

        ops.push_back(make_operation("ep_combine_backward_" + std::to_string(ctx.op_counter++),
                                     "ep_combine_backward",
                                     "ep_combine_backward",
                                     {ctx.d_output},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("ep_combine", ::dsl::ep_combine_backward_rule);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

const int _ep_combine_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "ep_combine";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

const int _ep_combine_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "ep_combine_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
