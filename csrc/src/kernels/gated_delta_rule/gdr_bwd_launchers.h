// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_KERNELS_GDR_BWD_LAUNCHERS_H
#define SUROGATE_SRC_KERNELS_GDR_BWD_LAUNCHERS_H

#include <cuda_runtime.h>
#include <cstddef>

// Launch wrappers for the backward sub-kernels.
// Each is compiled in its own .cu translation unit for parallel compilation.

// --- Checkpoint (WMMA) ---
template<typename TQ, typename TG, typename TB>
void launch_gdr_checkpoint_wmma(
    float* checkpoints,
    const TQ* k, const TQ* v, const TG* g, const TB* beta,
    const float* initial_state,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream);

// --- Checkpoint (scalar fallback) ---
template<typename TQ, typename TG, typename TB>
void launch_gdr_checkpoint_scalar(
    float* checkpoints,
    const TQ* k, const TQ* v, const TG* g, const TB* beta,
    const float* initial_state,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream);

// --- Phase 1: chunk-parallel local gradients ---
template<typename TQ, typename TG, typename TB>
void launch_gdr_bwd_phase1(
    const TQ* d_out,
    const TQ* q, const TQ* k, const TQ* v,
    const TG* g, const TB* beta,
    const float* checkpoints,
    const float* d_final_state,
    float* chunk_workspace,
    float* ds_accum,
    int chunk_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream);

// --- Phase 2: sequential ds propagation ---
template<typename TQ>
void launch_gdr_bwd_phase2(
    const float* checkpoints,
    const float* d_final_state,
    float* d_initial_state,
    float* chunk_workspace,
    float* dh_storage,
    int chunk_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size,
    int threads, std::size_t smem, cudaStream_t stream);

// --- Phase 3: chunk-parallel finalization ---
template<typename TQ, typename TG, typename TB>
void launch_gdr_bwd_phase3(
    TQ* d_q, TQ* d_k, TQ* d_v,
    TG* d_g, TB* d_beta,
    const TQ* d_out,
    const TQ* q, const TQ* k, const TQ* v,
    const TG* g, const TB* beta,
    const float* checkpoints,
    const float* dh_storage,
    float* chunk_workspace,
    int chunk_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream);

#endif  // SUROGATE_SRC_KERNELS_GDR_BWD_LAUNCHERS_H
