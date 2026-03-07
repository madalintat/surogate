// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_bwd_launchers.h"

namespace {

// ============================================================================
// Checkpoint kernel — WMMA tensor-core accelerated
// Computes state at each chunk boundary (no output computation).
// Shared memory: h[K×V]f + scratch1[C×C]f + scratch2[C×C]f
//   + k[C×K]TQ + buf1[C×C]TQ + buf2[C×V]TQ + buf3[C×K]TQ + scalars[3×C]f
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_checkpoint_wmma(
    float* __restrict__ checkpoints,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = (long)b * H + h;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;

    float* cp_base = checkpoints + bh * (long)(num_chunks + 1) * kv;

    extern __shared__ char smem_raw[];
    float* smem_h    = (float*)smem_raw;
    float* scratch1  = smem_h + Kdim * Vdim;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    buf1      = smem_k + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invk = smem_beta + Lp;

    // Checkpoint 0 = initial state
    for (long idx = tid; idx < kv; idx += nthr)
        smem_h[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    for (long idx = tid; idx < kv; idx += nthr)
        cp_base[idx] = smem_h[idx];
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);

        // Zero-fill buffers
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            smem_k[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        if (tid < Lp) { smem_gcum[tid] = 0.0f; smem_beta[tid] = 0.0f; }
        __syncthreads();

        // Load k, v
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            smem_k[pos * Kdim + kk] = k[(((long)b * Tlen + cs + pos) * H + h) * Kdim + kk];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_gcum[i] = acc;
                smem_beta[i] = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // L2 norm + normalize k
        for (int pos = tid; pos < L; pos += nthr) {
            if (use_qk_l2norm_in_kernel) {
                float kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    kn2 += kv2 * kv2;
                }
                smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_invk[pos] = 1.0f;
            }
        }
        __syncthreads();
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim, kk = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_invk[pos]));
            }
            __syncthreads();
        }

        // A = beta * exp(g) * (k @ k^T)
        wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // Solve M
        for (int idx = tid; idx < Lp * Lp; idx += nthr) scratch2[idx] = 0.0f;
        __syncthreads();
        for (int i = tid; i < Lp; i += nthr) scratch2[i * Lp + i] = 1.0f;
        __syncthreads();
        for (int row = 1; row < L; ++row) {
            for (int j = tid; j < row; j += nthr) {
                float s = 0.0f;
                for (int m = j; m < row; ++m)
                    s += scratch1[row * Lp + m] * scratch2[m * Lp + j];
                scratch2[row * Lp + j] = -s;
            }
            __syncthreads();
        }

        // M → bf16 in buf1
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // bv = beta*v, then u = M @ bv
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // bkg = beta*k*exp(g), then w = M @ bkg
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // h_bf16, then v_new_pre = u - w @ h_bf16
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(smem_h[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]));
        __syncthreads();

        // State update: v_scaled = bf16(v_new_pre * exp(g_last - g_cum[i]))
        const float g_last = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ?
                bf16_trunc<TQ>(to_float(buf2[idx]) * expf(g_last - smem_gcum[i])) : 0.0f;
            buf2[idx] = from_float<TQ>(val);
        }
        __syncthreads();
        // K^T @ v_scaled → scratch1[K×V]
        wmma_tn<TQ>(smem_k, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        const float eg = expf(g_last);
        for (long idx = tid; idx < kv; idx += nthr)
            smem_h[idx] = eg * smem_h[idx] + scratch1[idx];
        __syncthreads();

        // Write checkpoint
        float* cp_out = cp_base + (long)(chunk + 1) * kv;
        for (long idx = tid; idx < kv; idx += nthr)
            cp_out[idx] = smem_h[idx];
        __syncthreads();
    }
}

// ============================================================================
// Checkpoint kernel — fully parallelized
// Recomputes state at each chunk boundary for the backward pass.
// Same math as forward state update (phases 2-4-5-7) without output.
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_checkpoint_v2(
    float* __restrict__ checkpoints,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    bool use_qk_l2norm_in_kernel) {

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = static_cast<long>(Kdim) * Vdim;
    const long bh = static_cast<long>(b) * H + h;
    float* cp_base = checkpoints + bh * static_cast<long>(num_chunks + 1) * kv;

    extern __shared__ char smem_raw[];
    float*  smem_M     = reinterpret_cast<float*>(smem_raw);
    TQ*     smem_k     = reinterpret_cast<TQ*>(smem_M + kMaxC * kMaxC);
    TQ*     smem_v_loc = smem_k + kMaxC * Kdim;
    TQ*     smem_u     = smem_v_loc + kMaxC * Vdim;
    TQ*     smem_w     = smem_u + kMaxC * Vdim;
    float*  smem_g_cum = reinterpret_cast<float*>(smem_w + kMaxC * Kdim);
    float*  smem_beta  = smem_g_cum + kMaxC;
    float*  smem_inv_k = smem_beta  + kMaxC;

    // Checkpoint 0 = initial state
    for (long idx = tid; idx < kv; idx += nthreads) {
        cp_base[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        float* s_in  = cp_base + static_cast<long>(chunk) * kv;
        float* s_out = cp_base + static_cast<long>(chunk + 1) * kv;

        // Load k, v
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int pos = idx / Kdim;
            const int kk  = idx % Kdim;
            smem_k[pos * Kdim + kk] = k[(((long)b * Tlen + cs + pos) * H + h) * Kdim + kk];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int pos = idx / Vdim;
            const int vv  = idx % Vdim;
            smem_v_loc[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_g_cum[i] = acc;
                smem_beta[i]  = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // L2 norms + normalize k in-place
        for (int pos = tid; pos < L; pos += nthreads) {
            if (use_qk_l2norm_in_kernel) {
                float kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    kn2 += kv2 * kv2;
                }
                smem_inv_k[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_inv_k[pos] = 1.0f;
            }
        }
        __syncthreads();

        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthreads) {
                const int pos = idx / Kdim;
                const int kk  = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_inv_k[pos]));
            }
            __syncthreads();
        }

        // Build M matrix
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_M[idx] = 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < L; i += nthreads) {
            smem_M[i * kMaxC + i] = 1.0f;
        }
        __syncthreads();

        for (int row = 1; row < L; ++row) {
            for (int j = tid; j < row; j += nthreads) {
                float s = 0.0f;
                for (int m = j; m < row; ++m) {
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        dot_k += to_float(smem_k[row * Kdim + kk])
                               * to_float(smem_k[m   * Kdim + kk]);
                    }
                    float a_rm = smem_beta[row] * dot_k
                               * expf(smem_g_cum[row] - smem_g_cum[m]);
                    s += a_rm * smem_M[m * kMaxC + j];
                }
                smem_M[row * kMaxC + j] = bf16_trunc<TQ>(-s);
            }
            __syncthreads();
        }

        // Compute u, w, v_new_pre and update state -> s_out
        const float g_last = smem_g_cum[L - 1];
        const float eg_last = expf(g_last);

        // s_out = eg_last * s_in
        for (long idx = tid; idx < kv; idx += nthreads) {
            s_out[idx] = s_in[idx] * eg_last;
        }
        __syncthreads();

        // For each position i: compute v_new and accumulate into s_out
        for (int i = 0; i < L; ++i) {
            const float e_i = expf(g_last - smem_g_cum[i]);
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                // u_i_v
                float u_val = 0.0f;
                for (int m = 0; m <= i; ++m) {
                    float vb = bf16_trunc<TQ>(
                        to_float(smem_v_loc[m * Vdim + vv]) * smem_beta[m]);
                    u_val += smem_M[i * kMaxC + m] * vb;
                }
                u_val = bf16_trunc<TQ>(u_val);

                // w_i @ state -> wh_val
                float wh_val = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float w_ik = 0.0f;
                    for (int m = 0; m <= i; ++m) {
                        float kbg = bf16_trunc<TQ>(
                            to_float(smem_k[m * Kdim + kk]) * smem_beta[m]
                            * expf(smem_g_cum[m]));
                        w_ik += smem_M[i * kMaxC + m] * kbg;
                    }
                    w_ik = bf16_trunc<TQ>(w_ik);
                    float h_cs = bf16_trunc<TQ>(s_in[kk * Vdim + vv]);
                    wh_val += w_ik * h_cs;
                }

                float v_new_pre = bf16_trunc<TQ>(u_val - wh_val);
                float v_new = bf16_trunc<TQ>(v_new_pre * e_i);

                for (int kk = 0; kk < Kdim; ++kk) {
                    float ki = to_float(smem_k[i * Kdim + kk]);
                    atomicAdd(&s_out[kk * Vdim + vv], ki * v_new);
                }
            }
            __syncthreads();
        }
    }
}

} // anonymous namespace

// ============================================================================
// Launch wrappers
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_gdr_checkpoint_wmma(
    float* checkpoints,
    const TQ* k, const TQ* v, const TG* g, const TB* beta,
    const float* initial_state,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream)
{
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_chunk_checkpoint_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    gdr_chunk_checkpoint_wmma<TQ, TG, TB><<<dim3(B, H), threads, smem, stream>>>(
        checkpoints, k, v, g, beta, initial_state,
        Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel);
    CUDA_CHECK(cudaGetLastError());
}

template<typename TQ, typename TG, typename TB>
void launch_gdr_checkpoint_scalar(
    float* checkpoints,
    const TQ* k, const TQ* v, const TG* g, const TB* beta,
    const float* initial_state,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream)
{
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_chunk_checkpoint_v2<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    gdr_chunk_checkpoint_v2<TQ, TG, TB><<<dim3(B, H), threads, smem, stream>>>(
        checkpoints, k, v, g, beta, initial_state,
        Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

#define INSTANTIATE_CHECKPOINT(FUNC, TQ, TG, TB) \
    template void FUNC<TQ, TG, TB>( \
        float*, const TQ*, const TQ*, const TG*, const TB*, const float*, \
        int, int, int, int, int, int, bool, int, std::size_t, cudaStream_t);

#define INSTANTIATE_CHECKPOINT_ALL_GB(FUNC, TQ) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, float, float) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, float, nv_bfloat16) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, float, half) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, nv_bfloat16, float) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, nv_bfloat16, nv_bfloat16) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, nv_bfloat16, half) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, half, float) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, half, nv_bfloat16) \
    INSTANTIATE_CHECKPOINT(FUNC, TQ, half, half)

INSTANTIATE_CHECKPOINT_ALL_GB(launch_gdr_checkpoint_wmma, nv_bfloat16)
INSTANTIATE_CHECKPOINT_ALL_GB(launch_gdr_checkpoint_wmma, half)
INSTANTIATE_CHECKPOINT_ALL_GB(launch_gdr_checkpoint_scalar, nv_bfloat16)
INSTANTIATE_CHECKPOINT_ALL_GB(launch_gdr_checkpoint_scalar, half)
// scalar path also needs float TQ
INSTANTIATE_CHECKPOINT_ALL_GB(launch_gdr_checkpoint_scalar, float)

#undef INSTANTIATE_CHECKPOINT_ALL_GB
#undef INSTANTIATE_CHECKPOINT
