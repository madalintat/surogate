// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 gated delta rule chunk kernels.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace {

template<typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template<>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template<>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template<typename T>
__device__ __forceinline__ T from_float(float v);

template<>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template<>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half(v);
}

template<typename T>
__device__ __forceinline__ float normalize_and_maybe_quantize(float x, float inv) {
    const float y = x * inv;
    if constexpr (std::is_same_v<T, float>) {
        return y;
    } else {
        return to_float(from_float<T>(y));
    }
}

void validate_gated_delta_shapes(const Tensor& q,
                                 const Tensor& k,
                                 const Tensor& v,
                                 const Tensor& g,
                                 const Tensor& beta,
                                 const Tensor* initial_state) {
    if (q.Rank != 4 || k.Rank != 4 || v.Rank != 4 || g.Rank != 3 || beta.Rank != 3) {
        throw std::logic_error(
            "gated_delta_rule: expected q/k/v rank 4 and g/beta rank 3");
    }
    if (q.Sizes[0] != k.Sizes[0] || q.Sizes[1] != k.Sizes[1] ||
        q.Sizes[2] != k.Sizes[2] || q.Sizes[3] != k.Sizes[3]) {
        throw std::logic_error("gated_delta_rule: q and k shapes must match");
    }
    if (q.Sizes[0] != v.Sizes[0] || q.Sizes[1] != v.Sizes[1] || q.Sizes[2] != v.Sizes[2]) {
        throw std::logic_error("gated_delta_rule: q/k and v must share B/T/H");
    }
    if (g.Sizes[0] != q.Sizes[0] || g.Sizes[1] != q.Sizes[1] || g.Sizes[2] != q.Sizes[2]) {
        throw std::logic_error("gated_delta_rule: g shape must be [B,T,H]");
    }
    if (beta.Sizes[0] != q.Sizes[0] || beta.Sizes[1] != q.Sizes[1] || beta.Sizes[2] != q.Sizes[2]) {
        throw std::logic_error("gated_delta_rule: beta shape must be [B,T,H]");
    }
    if (initial_state) {
        if (initial_state->Rank != 4 ||
            initial_state->Sizes[0] != q.Sizes[0] ||
            initial_state->Sizes[1] != q.Sizes[2] ||
            initial_state->Sizes[2] != q.Sizes[3] ||
            initial_state->Sizes[3] != v.Sizes[3]) {
            throw std::logic_error(
                "gated_delta_rule: initial_state must be [B,H,K,V]");
        }
    }
}

}  // namespace
namespace {

constexpr int kMaxChunk = 64;

template<typename TQ>
__device__ __forceinline__ float load_normed_qk(const TQ* ptr, long base, int k, float inv) {
    return normalize_and_maybe_quantize<TQ>(to_float(ptr[base + k]), inv);
}

template<typename TQ, typename TG, typename TB>
__global__ void gated_delta_rule_chunk_fwd_kernel(
    TQ* out,
    float* final_state,
    float* state_scratch,
    const TQ* q,
    const TQ* k,
    const TQ* v,
    const TG* g,
    const TB* beta,
    const float* initial_state,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = static_cast<long>(b) * H + h;
    const long state_base = bh * static_cast<long>(Kdim) * Vdim;
    const long kv_total = static_cast<long>(Kdim) * Vdim;
    float* state_a = state_scratch + state_base;
    float* state_b = final_state + state_base;

    __shared__ float s_inv_q[kMaxChunk];
    __shared__ float s_inv_k[kMaxChunk];
    __shared__ float s_g_cum[kMaxChunk];
    __shared__ float s_beta[kMaxChunk];
    __shared__ float s_M[kMaxChunk * kMaxChunk];

    for (long idx = tid; idx < kv_total; idx += blockDim.x) {
        state_a[idx] = initial_state ? initial_state[state_base + idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk_start = 0; chunk_start < Tlen; chunk_start += chunk_size) {
        const int L = ((chunk_start + chunk_size) <= Tlen) ? chunk_size : (Tlen - chunk_start);
        if (tid == 0) {
            float acc_g = 0.0f;
            for (int i = 0; i < L; ++i) {
                const int t = chunk_start + i;
                const long gh_idx = (static_cast<long>(b) * Tlen + t) * H + h;
                acc_g += to_float(g[gh_idx]);
                s_g_cum[i] = acc_g;
                s_beta[i] = to_float(beta[gh_idx]);

                if (use_qk_l2norm_in_kernel) {
                    float qn2 = 0.0f;
                    float kn2 = 0.0f;
                    const long qk_base = ((static_cast<long>(b) * Tlen + t) * H + h) * Kdim;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        const float qv = to_float(q[qk_base + kk]);
                        const float kv = to_float(k[qk_base + kk]);
                        qn2 += qv * qv;
                        kn2 += kv * kv;
                    }
                    s_inv_q[i] = 1.0f / sqrtf(qn2 + 1e-6f);
                    s_inv_k[i] = 1.0f / sqrtf(kn2 + 1e-6f);
                } else {
                    s_inv_q[i] = 1.0f;
                    s_inv_k[i] = 1.0f;
                }
            }

            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    s_M[i * kMaxChunk + j] = 0.0f;
                }
                s_M[i * kMaxChunk + i] = 1.0f;
                for (int j = 0; j < i; ++j) {
                    float s = 0.0f;
                    for (int m = j; m < i; ++m) {
                        const int ti = chunk_start + i;
                        const int tm = chunk_start + m;
                        const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
                        const long k_m_base = ((static_cast<long>(b) * Tlen + tm) * H + h) * Kdim;
                        float dot_k = 0.0f;
                        for (int kk = 0; kk < Kdim; ++kk) {
                            const float ki = load_normed_qk(k, k_i_base, kk, s_inv_k[i]);
                            const float km = load_normed_qk(k, k_m_base, kk, s_inv_k[m]);
                            dot_k += ki * km;
                        }
                        const float a_im = s_beta[i] * dot_k * expf(s_g_cum[i] - s_g_cum[m]);
                        s += a_im * s_M[m * kMaxChunk + j];
                    }
                    s_M[i * kMaxChunk + j] = to_float(from_float<TQ>(-s));
                }
            }
        }
        __syncthreads();

        const float g_last = s_g_cum[L - 1];
        const float eg_last = expf(g_last);
        for (long idx = tid; idx < kv_total; idx += blockDim.x) {
            state_b[idx] = state_a[idx] * eg_last;
        }
        __syncthreads();

        // 1) Compute pre-gated v_new = u - w@h_chunk_start and keep it in out[] as
        // temporary storage. This matches FLA chunk_gated_delta_rule_fwd_h (save_new_value).
        for (int i = 0; i < L; ++i) {
            const int t_i = chunk_start + i;
            const long out_i_base = ((static_cast<long>(b) * Tlen + t_i) * H + h) * Vdim;
            for (int vv = tid; vv < Vdim; vv += blockDim.x) {
                float u_i_v = 0.0f;
                for (int m = 0; m <= i; ++m) {
                    const int t_m = chunk_start + m;
                    const long v_m_base = ((static_cast<long>(b) * Tlen + t_m) * H + h) * Vdim;
                    const float vb = to_float(from_float<TQ>(to_float(v[v_m_base + vv]) * s_beta[m]));
                    u_i_v += s_M[i * kMaxChunk + m] * vb;
                }
                u_i_v = to_float(from_float<TQ>(u_i_v));

                float wh_i_v = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float w_i_k = 0.0f;
                    for (int m = 0; m <= i; ++m) {
                        const int t_m = chunk_start + m;
                        const long k_m_base = ((static_cast<long>(b) * Tlen + t_m) * H + h) * Kdim;
                        const float km = load_normed_qk(k, k_m_base, kk, s_inv_k[m]);
                        const float kbg =
                            to_float(from_float<TQ>(km * s_beta[m] * expf(s_g_cum[m])));
                        w_i_k += s_M[i * kMaxChunk + m] * kbg;
                    }
                    w_i_k = to_float(from_float<TQ>(w_i_k));
                    const float h_chunk_start =
                        to_float(from_float<TQ>(state_a[static_cast<long>(kk) * Vdim + vv]));
                    wh_i_v += w_i_k * h_chunk_start;
                }
                const float v_new_pre = to_float(from_float<TQ>(u_i_v - wh_i_v));
                out[out_i_base + vv] = from_float<TQ>(v_new_pre);
            }
        }

        // 2) Compute outputs in descending order so temporary v_new values remain available.
        for (int i = L - 1; i >= 0; --i) {
            const int t_i = chunk_start + i;
            const long q_i_base = ((static_cast<long>(b) * Tlen + t_i) * H + h) * Kdim;
            const long out_i_base = ((static_cast<long>(b) * Tlen + t_i) * H + h) * Vdim;
            const float eg_i = expf(s_g_cum[i]);
            const float e_i = expf(g_last - s_g_cum[i]);

            // Fold gated v_new_i into next state before out[i] is overwritten.
            for (int vv = tid; vv < Vdim; vv += blockDim.x) {
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = load_normed_qk(k, q_i_base, kk, s_inv_k[i]);
                    const float v_new_pre = to_float(out[out_i_base + vv]);
                    const float v_new_scaled =
                        to_float(from_float<TQ>(v_new_pre * e_i));
                    state_b[static_cast<long>(kk) * Vdim + vv] += ki * v_new_scaled;
                }
            }

            for (int vv = tid; vv < Vdim; vv += blockDim.x) {
                float term1 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = load_normed_qk(q, q_i_base, kk, s_inv_q[i]);
                    const float hq = to_float(from_float<TQ>(state_a[static_cast<long>(kk) * Vdim + vv]));
                    term1 += qi * hq;
                }
                term1 *= eg_i;

                float term2 = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    const int t_j = chunk_start + j;
                    const long q_j_base = ((static_cast<long>(b) * Tlen + t_j) * H + h) * Kdim;
                    const long out_j_base = ((static_cast<long>(b) * Tlen + t_j) * H + h) * Vdim;
                    float dot_qk = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        const float qi = load_normed_qk(q, q_i_base, kk, s_inv_q[i]);
                        const float kj = load_normed_qk(k, q_j_base, kk, s_inv_k[j]);
                        dot_qk += qi * kj;
                    }
                    const float s_ij = to_float(from_float<TQ>(
                        expf(s_g_cum[i] - s_g_cum[j]) * dot_qk));
                    const float v_new_pre_j = to_float(out[out_j_base + vv]);
                    term2 += s_ij * v_new_pre_j;
                }
                out[out_i_base + vv] = from_float<TQ>((term1 + term2) * scale);
            }
        }

        __syncthreads();
        float* tmp = state_a;
        state_a = state_b;
        state_b = tmp;
        __syncthreads();
    }

    if (state_a != (final_state + state_base)) {
        for (long idx = tid; idx < kv_total; idx += blockDim.x) {
            final_state[state_base + idx] = state_a[idx];
        }
    }
}

template<typename TQ, typename TG, typename TB>
__global__ void gated_delta_rule_chunk_checkpoint_kernel(
    float* checkpoints,
    const TQ* k,
    const TQ* v,
    const TG* g,
    const TB* beta,
    const float* initial_state,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    bool use_qk_l2norm_in_kernel) {
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = static_cast<long>(Kdim) * Vdim;
    const long bh = static_cast<long>(b) * H + h;
    float* cp_base = checkpoints + bh * static_cast<long>(num_chunks + 1) * kv;

    __shared__ float s_inv_k[kMaxChunk];
    __shared__ float s_g_cum[kMaxChunk];
    __shared__ float s_beta[kMaxChunk];
    __shared__ float s_M[kMaxChunk * kMaxChunk];

    for (long idx = tid; idx < kv; idx += blockDim.x) {
        cp_base[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int chunk_start = chunk * chunk_size;
        const int L = ((chunk_start + chunk_size) <= Tlen) ? chunk_size : (Tlen - chunk_start);
        float* s_in = cp_base + static_cast<long>(chunk) * kv;
        float* s_out = cp_base + static_cast<long>(chunk + 1) * kv;

        if (tid == 0) {
            float acc_g = 0.0f;
            for (int i = 0; i < L; ++i) {
                const int t = chunk_start + i;
                const long gh_idx = (static_cast<long>(b) * Tlen + t) * H + h;
                acc_g += to_float(g[gh_idx]);
                s_g_cum[i] = acc_g;
                s_beta[i] = to_float(beta[gh_idx]);
                if (use_qk_l2norm_in_kernel) {
                    float kn2 = 0.0f;
                    const long k_base = ((static_cast<long>(b) * Tlen + t) * H + h) * Kdim;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        const float kvv = to_float(k[k_base + kk]);
                        kn2 += kvv * kvv;
                    }
                    s_inv_k[i] = 1.0f / sqrtf(kn2 + 1e-6f);
                } else {
                    s_inv_k[i] = 1.0f;
                }
            }

            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    s_M[i * kMaxChunk + j] = 0.0f;
                }
                s_M[i * kMaxChunk + i] = 1.0f;
                for (int j = 0; j < i; ++j) {
                    float s = 0.0f;
                    for (int m = j; m < i; ++m) {
                        const int ti = chunk_start + i;
                        const int tm = chunk_start + m;
                        const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
                        const long k_m_base = ((static_cast<long>(b) * Tlen + tm) * H + h) * Kdim;
                        float dot_k = 0.0f;
                        for (int kk = 0; kk < Kdim; ++kk) {
                            const float ki = load_normed_qk(k, k_i_base, kk, s_inv_k[i]);
                            const float km = load_normed_qk(k, k_m_base, kk, s_inv_k[m]);
                            dot_k += ki * km;
                        }
                        const float a_im = s_beta[i] * dot_k * expf(s_g_cum[i] - s_g_cum[m]);
                        s += a_im * s_M[m * kMaxChunk + j];
                    }
                    s_M[i * kMaxChunk + j] = to_float(from_float<TQ>(-s));
                }
            }
        }
        __syncthreads();

        const float g_last = s_g_cum[L - 1];
        const float eg_last = expf(g_last);
        for (long idx = tid; idx < kv; idx += blockDim.x) {
            s_out[idx] = s_in[idx] * eg_last;
        }
        __syncthreads();

        for (int i = 0; i < L; ++i) {
            const int t_i = chunk_start + i;
            const long k_i_base = ((static_cast<long>(b) * Tlen + t_i) * H + h) * Kdim;
            const float e_i = expf(g_last - s_g_cum[i]);
            for (int vv = tid; vv < Vdim; vv += blockDim.x) {
                float u_i_v = 0.0f;
                for (int m = 0; m <= i; ++m) {
                    const int t_m = chunk_start + m;
                    const long v_m_base = ((static_cast<long>(b) * Tlen + t_m) * H + h) * Vdim;
                    const float vb = to_float(from_float<TQ>(to_float(v[v_m_base + vv]) * s_beta[m]));
                    u_i_v += s_M[i * kMaxChunk + m] * vb;
                }
                u_i_v = to_float(from_float<TQ>(u_i_v));

                float wh_i_v = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float w_i_k = 0.0f;
                    for (int m = 0; m <= i; ++m) {
                        const int t_m = chunk_start + m;
                        const long k_m_base = ((static_cast<long>(b) * Tlen + t_m) * H + h) * Kdim;
                        const float km = load_normed_qk(k, k_m_base, kk, s_inv_k[m]);
                        const float kbg =
                            to_float(from_float<TQ>(km * s_beta[m] * expf(s_g_cum[m])));
                        w_i_k += s_M[i * kMaxChunk + m] * kbg;
                    }
                    w_i_k = to_float(from_float<TQ>(w_i_k));
                    const float h_chunk_start =
                        to_float(from_float<TQ>(s_in[static_cast<long>(kk) * Vdim + vv]));
                    wh_i_v += w_i_k * h_chunk_start;
                }
                const float v_new_pre = to_float(from_float<TQ>(u_i_v - wh_i_v));
                const float v_new = to_float(from_float<TQ>(v_new_pre * e_i));
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = load_normed_qk(k, k_i_base, kk, s_inv_k[i]);
                    s_out[static_cast<long>(kk) * Vdim + vv] += ki * v_new;
                }
            }
        }
        __syncthreads();
    }
}

template<typename TQ, typename TG, typename TB>
__global__ void gated_delta_rule_chunk_bwd_kernel(
    TQ* d_q,
    TQ* d_k,
    TQ* d_v,
    TG* d_g,
    TB* d_beta,
    float* d_initial_state,
    const TQ* d_out,
    const float* d_final_state,
    const TQ* q,
    const TQ* k,
    const TQ* v,
    const TG* g,
    const TB* beta,
    const float* checkpoints,
    float* workspace,
    int workspace_stride,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel) {
    if (threadIdx.x != 0) {
        return;
    }
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = static_cast<long>(Kdim) * Vdim;
    const long bh = static_cast<long>(b) * H + h;
    const float* cp_base = checkpoints + bh * static_cast<long>(num_chunks + 1) * kv;
    float* ds = d_initial_state + bh * kv;

    for (long idx = 0; idx < kv; ++idx) {
        ds[idx] = d_final_state ? d_final_state[bh * kv + idx] : 0.0f;
    }

    float* ws = workspace + bh * workspace_stride;
    const int ws_W = 0;
    const int ws_VNEW = ws_W + chunk_size * Kdim;
    const int ws_DU = ws_VNEW + chunk_size * Vdim;
    const int ws_DW = ws_DU + chunk_size * Vdim;
    const int ws_DQ = ws_DW + chunk_size * Kdim;
    const int ws_DK = ws_DQ + chunk_size * Kdim;
    const int ws_DG = ws_DK + chunk_size * Kdim;
    const int ws_DB = ws_DG + chunk_size;
    (void)ws_DB;

    float inv_q[kMaxChunk];
    float inv_k[kMaxChunk];
    float g_cum[kMaxChunk];
    float beta_f[kMaxChunk];
    float M[kMaxChunk * kMaxChunk];
    float dM[kMaxChunk * kMaxChunk];
    float tmp_mat[kMaxChunk * kMaxChunk];

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        const int chunk_start = chunk * chunk_size;
        const int L = ((chunk_start + chunk_size) <= Tlen) ? chunk_size : (Tlen - chunk_start);
        const float* h_in = cp_base + static_cast<long>(chunk) * kv;
        float* dh_in = const_cast<float*>(cp_base + static_cast<long>(chunk + 1) * kv);

        float acc_g = 0.0f;
        for (int i = 0; i < L; ++i) {
            const int t = chunk_start + i;
            const long gh_idx = (static_cast<long>(b) * Tlen + t) * H + h;
            acc_g += to_float(g[gh_idx]);
            g_cum[i] = acc_g;
            beta_f[i] = to_float(beta[gh_idx]);

            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f;
                float kn2 = 0.0f;
                const long qk_base = ((static_cast<long>(b) * Tlen + t) * H + h) * Kdim;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qv = to_float(q[qk_base + kk]);
                    const float kvv = to_float(k[qk_base + kk]);
                    qn2 += qv * qv;
                    kn2 += kvv * kvv;
                }
                inv_q[i] = 1.0f / sqrtf(qn2 + 1e-6f);
                inv_k[i] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                inv_q[i] = 1.0f;
                inv_k[i] = 1.0f;
            }
        }

        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                M[i * kMaxChunk + j] = 0.0f;
                dM[i * kMaxChunk + j] = 0.0f;
            }
            M[i * kMaxChunk + i] = 1.0f;
            for (int j = 0; j < i; ++j) {
                float s = 0.0f;
                for (int m = j; m < i; ++m) {
                    const int ti = chunk_start + i;
                    const int tm = chunk_start + m;
                    const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
                    const long k_m_base = ((static_cast<long>(b) * Tlen + tm) * H + h) * Kdim;
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        const float ki = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                        const float km = load_normed_qk(k, k_m_base, kk, inv_k[m]);
                        dot_k += ki * km;
                    }
                    const float a_im = beta_f[i] * dot_k * expf(g_cum[i] - g_cum[m]);
                    s += a_im * M[m * kMaxChunk + j];
                }
                M[i * kMaxChunk + j] = to_float(from_float<TQ>(-s));
            }
        }

        float* W = ws + ws_W;
        float* VNEW = ws + ws_VNEW;
        float* DU = ws + ws_DU;
        float* DW = ws + ws_DW;
        float* DQ = ws + ws_DQ;
        float* DK = ws + ws_DK;
        float* DG = ws + ws_DG;
        float* DB = ws + ws_DB;

        for (int i = 0; i < L; ++i) {
            DG[i] = 0.0f;
            DB[i] = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                W[static_cast<long>(i) * Kdim + kk] = 0.0f;
                DW[static_cast<long>(i) * Kdim + kk] = 0.0f;
                DQ[static_cast<long>(i) * Kdim + kk] = 0.0f;
                DK[static_cast<long>(i) * Kdim + kk] = 0.0f;
            }
            for (int vv = 0; vv < Vdim; ++vv) {
                VNEW[static_cast<long>(i) * Vdim + vv] = 0.0f;
                DU[static_cast<long>(i) * Vdim + vv] = 0.0f;
            }
        }

        for (int i = 0; i < L; ++i) {
            const int ti = chunk_start + i;
            const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            const float eg_i = expf(g_cum[i]);
            for (int kk = 0; kk < Kdim; ++kk) {
                float w_i_k = 0.0f;
                for (int m = 0; m <= i; ++m) {
                    const int tm = chunk_start + m;
                    const long k_m_base = ((static_cast<long>(b) * Tlen + tm) * H + h) * Kdim;
                    const float km = load_normed_qk(k, k_m_base, kk, inv_k[m]);
                    const float kbg = to_float(from_float<TQ>(km * beta_f[m] * expf(g_cum[m])));
                    w_i_k += M[i * kMaxChunk + m] * kbg;
                }
                W[static_cast<long>(i) * Kdim + kk] = to_float(from_float<TQ>(w_i_k));
            }
            const float e_i = expf(g_cum[L - 1] - g_cum[i]);
            for (int vv = 0; vv < Vdim; ++vv) {
                float u_i_v = 0.0f;
                for (int m = 0; m <= i; ++m) {
                    const int tm = chunk_start + m;
                    const long v_m_base = ((static_cast<long>(b) * Tlen + tm) * H + h) * Vdim;
                    const float vb = to_float(from_float<TQ>(to_float(v[v_m_base + vv]) * beta_f[m]));
                    u_i_v += M[i * kMaxChunk + m] * vb;
                }
                u_i_v = to_float(from_float<TQ>(u_i_v));

                float wh_i_v = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float h_chunk_start =
                        to_float(from_float<TQ>(h_in[static_cast<long>(kk) * Vdim + vv]));
                    wh_i_v += W[static_cast<long>(i) * Kdim + kk] * h_chunk_start;
                }
                const float v_new = to_float(from_float<TQ>((u_i_v - wh_i_v) * e_i));
                VNEW[static_cast<long>(i) * Vdim + vv] = v_new;
            }
            (void)eg_i;
            (void)k_i_base;
        }

        const float eg_last = expf(g_cum[L - 1]);
        float dg_last_extra = 0.0f;
        for (int kk = 0; kk < Kdim; ++kk) {
            for (int vv = 0; vv < Vdim; ++vv) {
                const long idx = static_cast<long>(kk) * Vdim + vv;
                dh_in[idx] = ds[idx] * eg_last;
                dg_last_extra += ds[idx] * h_in[idx];
            }
        }
        DG[L - 1] += dg_last_extra * eg_last;

        for (int i = 0; i < L; ++i) {
            const int ti = chunk_start + i;
            const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            for (int vv = 0; vv < Vdim; ++vv) {
                float d_vnew = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                    d_vnew += ds[static_cast<long>(kk) * Vdim + vv] * ki;
                }
                DU[static_cast<long>(i) * Vdim + vv] += d_vnew;
            }
            for (int kk = 0; kk < Kdim; ++kk) {
                float dkk = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    dkk += ds[static_cast<long>(kk) * Vdim + vv] *
                           VNEW[static_cast<long>(i) * Vdim + vv];
                }
                DK[static_cast<long>(i) * Kdim + kk] += dkk;
            }
        }

        for (int i = 0; i < L; ++i) {
            const int ti = chunk_start + i;
            const long q_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            const long do_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Vdim;
            const float eg_i = expf(g_cum[i]);

            for (int vv = 0; vv < Vdim; ++vv) {
                const float do_v = to_float(d_out[do_i_base + vv]) * scale;
                float qh = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = load_normed_qk(q, q_i_base, kk, inv_q[i]);
                    const float hq = to_float(from_float<TQ>(h_in[static_cast<long>(kk) * Vdim + vv]));
                    qh += qi * hq;
                    DQ[static_cast<long>(i) * Kdim + kk] += do_v * eg_i * hq;
                    dh_in[static_cast<long>(kk) * Vdim + vv] += do_v * eg_i * qi;
                }
                DG[i] += do_v * eg_i * qh;
            }

            for (int j = 0; j <= i; ++j) {
                const int tj = chunk_start + j;
                const long k_j_base = ((static_cast<long>(b) * Tlen + tj) * H + h) * Kdim;
                const float e_j = expf(g_cum[L - 1] - g_cum[j]);
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = load_normed_qk(q, q_i_base, kk, inv_q[i]);
                    const float kj = load_normed_qk(k, k_j_base, kk, inv_k[j]);
                    dot_qk += qi * kj;
                }
                const float exp_ij = expf(g_cum[i] - g_cum[j]);
                const float s_ij = to_float(from_float<TQ>(exp_ij * dot_qk));
                float grad_s = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    const float do_v = to_float(d_out[do_i_base + vv]) * scale;
                    const float v_new_pre =
                        VNEW[static_cast<long>(j) * Vdim + vv] / e_j;
                    grad_s += do_v * v_new_pre;
                    DU[static_cast<long>(j) * Vdim + vv] += do_v * (s_ij / e_j);
                }
                const float coeff = grad_s * exp_ij;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = load_normed_qk(q, q_i_base, kk, inv_q[i]);
                    const float kj = load_normed_qk(k, k_j_base, kk, inv_k[j]);
                    DQ[static_cast<long>(i) * Kdim + kk] += coeff * kj;
                    DK[static_cast<long>(j) * Kdim + kk] += coeff * qi;
                }
                DG[i] += grad_s * s_ij;
                DG[j] -= grad_s * s_ij;
            }
        }

        const float g_last = g_cum[L - 1];
        for (int i = 0; i < L; ++i) {
            const int ti = chunk_start + i;
            const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            const float e_i = expf(g_last - g_cum[i]);
            for (int vv = 0; vv < Vdim; ++vv) {
                const float v_new = VNEW[static_cast<long>(i) * Vdim + vv];
                const float pre = v_new / e_i;
                const float d_vnew = DU[static_cast<long>(i) * Vdim + vv];
                const float d_pre = d_vnew * e_i;
                float d_vnew_state = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                    d_vnew_state += ds[static_cast<long>(kk) * Vdim + vv] * ki;
                }
                const float d_e_state = d_vnew_state * pre;
                DU[static_cast<long>(i) * Vdim + vv] = d_pre;
                DG[L - 1] += d_e_state * e_i;
                DG[i] -= d_e_state * e_i;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float h_chunk_start =
                        to_float(from_float<TQ>(h_in[static_cast<long>(kk) * Vdim + vv]));
                    DW[static_cast<long>(i) * Kdim + kk] +=
                        -d_pre * h_chunk_start;
                    dh_in[static_cast<long>(kk) * Vdim + vv] +=
                        -d_pre * W[static_cast<long>(i) * Kdim + kk];
                }
            }
        }

        for (int i = 0; i < L; ++i) {
            for (int j = 0; j <= i; ++j) {
                float s = 0.0f;
                const int tj = chunk_start + j;
                const long k_j_base = ((static_cast<long>(b) * Tlen + tj) * H + h) * Kdim;
                const long v_j_base = ((static_cast<long>(b) * Tlen + tj) * H + h) * Vdim;
                for (int vv = 0; vv < Vdim; ++vv) {
                    const float vb = to_float(from_float<TQ>(to_float(v[v_j_base + vv]) * beta_f[j]));
                    s += DU[static_cast<long>(i) * Vdim + vv] * vb;
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float kj = load_normed_qk(k, k_j_base, kk, inv_k[j]);
                    const float kbg =
                        to_float(from_float<TQ>(kj * beta_f[j] * expf(g_cum[j])));
                    s += DW[static_cast<long>(i) * Kdim + kk] * kbg;
                }
                dM[i * kMaxChunk + j] = s;
            }
        }

        for (int j = 0; j < L; ++j) {
            const int tj = chunk_start + j;
            const long v_j_base = ((static_cast<long>(b) * Tlen + tj) * H + h) * Vdim;
            const long k_j_base = ((static_cast<long>(b) * Tlen + tj) * H + h) * Kdim;
            for (int vv = 0; vv < Vdim; ++vv) {
                float t = 0.0f;
                for (int i = j; i < L; ++i) {
                    t += M[i * kMaxChunk + j] * DU[static_cast<long>(i) * Vdim + vv];
                }
                d_v[v_j_base + vv] = from_float<TQ>(t * beta_f[j]);
                DB[j] += t * to_float(v[v_j_base + vv]);
            }
            for (int kk = 0; kk < Kdim; ++kk) {
                float t = 0.0f;
                for (int i = j; i < L; ++i) {
                    t += M[i * kMaxChunk + j] * DW[static_cast<long>(i) * Kdim + kk];
                }
                const float kj = load_normed_qk(k, k_j_base, kk, inv_k[j]);
                const float egj = expf(g_cum[j]);
                DK[static_cast<long>(j) * Kdim + kk] += t * beta_f[j] * egj;
                DB[j] += t * egj * kj;
                DG[j] += t * beta_f[j] * egj * kj;
            }
        }

        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                float s = 0.0f;
                for (int m = 0; m < L; ++m) {
                    s += dM[i * kMaxChunk + m] * M[j * kMaxChunk + m];
                }
                tmp_mat[i * kMaxChunk + j] = s;
            }
        }
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                float s = 0.0f;
                for (int m = 0; m < L; ++m) {
                    s += M[m * kMaxChunk + i] * tmp_mat[m * kMaxChunk + j];
                }
                tmp_mat[i * kMaxChunk + j] = -s;
            }
        }

        for (int i = 0; i < L; ++i) {
            const int ti = chunk_start + i;
            const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            for (int j = 0; j < i; ++j) {
                const int tj = chunk_start + j;
                const long k_j_base = ((static_cast<long>(b) * Tlen + tj) * H + h) * Kdim;
                float dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                    const float kj = load_normed_qk(k, k_j_base, kk, inv_k[j]);
                    dot_k += ki * kj;
                }
                const float exp_ij = expf(g_cum[i] - g_cum[j]);
                const float a_grad = tmp_mat[i * kMaxChunk + j];
                const float val = dot_k * exp_ij;
                DB[i] += a_grad * val;
                const float dval = a_grad * beta_f[i];
                const float ddot = dval * exp_ij;
                const float dexp = dval * dot_k;
                DG[i] += dexp * exp_ij;
                DG[j] -= dexp * exp_ij;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                    const float kj = load_normed_qk(k, k_j_base, kk, inv_k[j]);
                    DK[static_cast<long>(i) * Kdim + kk] += ddot * kj;
                    DK[static_cast<long>(j) * Kdim + kk] += ddot * ki;
                }
            }
        }

        for (int i = 0; i < L; ++i) {
            const int ti = chunk_start + i;
            const long q_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            const long k_i_base = ((static_cast<long>(b) * Tlen + ti) * H + h) * Kdim;
            const long gh_idx = (static_cast<long>(b) * Tlen + ti) * H + h;
            if (use_qk_l2norm_in_kernel) {
                float dot_q = 0.0f;
                float dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qn = load_normed_qk(q, q_i_base, kk, inv_q[i]);
                    const float kn = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                    dot_q += DQ[static_cast<long>(i) * Kdim + kk] * qn;
                    dot_k += DK[static_cast<long>(i) * Kdim + kk] * kn;
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qn = load_normed_qk(q, q_i_base, kk, inv_q[i]);
                    const float kn = load_normed_qk(k, k_i_base, kk, inv_k[i]);
                    const float dq_raw =
                        (DQ[static_cast<long>(i) * Kdim + kk] - qn * dot_q) * inv_q[i];
                    const float dk_raw =
                        (DK[static_cast<long>(i) * Kdim + kk] - kn * dot_k) * inv_k[i];
                    d_q[q_i_base + kk] = from_float<TQ>(dq_raw);
                    d_k[k_i_base + kk] = from_float<TQ>(dk_raw);
                }
            } else {
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[q_i_base + kk] = from_float<TQ>(DQ[static_cast<long>(i) * Kdim + kk]);
                    d_k[k_i_base + kk] = from_float<TQ>(DK[static_cast<long>(i) * Kdim + kk]);
                }
            }
            d_beta[gh_idx] = from_float<TB>(DB[i]);
        }

        float running = 0.0f;
        for (int i = L - 1; i >= 0; --i) {
            running += DG[i];
            const int ti = chunk_start + i;
            const long gh_idx = (static_cast<long>(b) * Tlen + ti) * H + h;
            d_g[gh_idx] = from_float<TG>(running);
        }

        for (long idx = 0; idx < kv; ++idx) {
            ds[idx] = dh_in[idx];
        }
    }
}

template<typename TQ, typename TG, typename TB>
void launch_gated_delta_rule_chunk_forward(
    Tensor& out,
    Tensor& final_state,
    Tensor& state_scratch,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    cudaStream_t stream) {
    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    const int threads =
        (Vdim <= 32) ? 32 : ((Vdim <= 64) ? 64 : ((Vdim <= 128) ? 128 : 256));
    dim3 grid(B, H, 1);
    gated_delta_rule_chunk_fwd_kernel<TQ, TG, TB><<<grid, threads, 0, stream>>>(
        out.get<TQ>(),
        final_state.get<float>(),
        state_scratch.get<float>(),
        q.get<TQ>(),
        k.get<TQ>(),
        v.get<TQ>(),
        g.get<TG>(),
        beta.get<TB>(),
        initial_state ? initial_state->get<float>() : nullptr,
        Tlen,
        H,
        Kdim,
        Vdim,
        chunk_size,
        scale,
        use_qk_l2norm_in_kernel);
}

template<typename TQ, typename TG, typename TB>
void launch_gated_delta_rule_chunk_backward(
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    Tensor& d_g,
    Tensor& d_beta,
    Tensor& d_initial_state,
    const Tensor& d_out,
    const Tensor* d_final_state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints,
    Tensor& workspace,
    cudaStream_t stream) {
    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    const int checkpoint_threads =
        (Vdim <= 32) ? 32 : ((Vdim <= 64) ? 64 : ((Vdim <= 128) ? 128 : 256));
    dim3 grid(B, H, 1);
    gated_delta_rule_chunk_checkpoint_kernel<TQ, TG, TB><<<grid, checkpoint_threads, 0, stream>>>(
        checkpoints.get<float>(),
        k.get<TQ>(),
        v.get<TQ>(),
        g.get<TG>(),
        beta.get<TB>(),
        initial_state ? initial_state->get<float>() : nullptr,
        Tlen,
        H,
        Kdim,
        Vdim,
        chunk_size,
        use_qk_l2norm_in_kernel);
    gated_delta_rule_chunk_bwd_kernel<TQ, TG, TB><<<grid, 1, 0, stream>>>(
        d_q.get<TQ>(),
        d_k.get<TQ>(),
        d_v.get<TQ>(),
        d_g.get<TG>(),
        d_beta.get<TB>(),
        d_initial_state.get<float>(),
        d_out.get<TQ>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        q.get<TQ>(),
        k.get<TQ>(),
        v.get<TQ>(),
        g.get<TG>(),
        beta.get<TB>(),
        checkpoints.get<float>(),
        workspace.get<float>(),
        static_cast<int>(workspace.Sizes[2]),
        Tlen,
        H,
        Kdim,
        Vdim,
        chunk_size,
        scale,
        use_qk_l2norm_in_kernel);
}

template<typename TQ, typename TG>
void dispatch_chunk_beta_dtype_forward(
    Tensor& out,
    Tensor& final_state,
    Tensor& state_scratch,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    cudaStream_t stream) {
    switch (beta.DType) {
        case ETensorDType::FP32:
            launch_gated_delta_rule_chunk_forward<TQ, TG, float>(
                out, final_state, state_scratch, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel, stream);
            return;
        case ETensorDType::BF16:
            launch_gated_delta_rule_chunk_forward<TQ, TG, nv_bfloat16>(
                out, final_state, state_scratch, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel, stream);
            return;
        case ETensorDType::FP16:
            launch_gated_delta_rule_chunk_forward<TQ, TG, half>(
                out, final_state, state_scratch, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel, stream);
            return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward: unsupported beta dtype");
    }
}

template<typename TQ>
void dispatch_chunk_g_dtype_forward(
    Tensor& out,
    Tensor& final_state,
    Tensor& state_scratch,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    cudaStream_t stream) {
    switch (g.DType) {
        case ETensorDType::FP32:
            dispatch_chunk_beta_dtype_forward<TQ, float>(
                out, final_state, state_scratch, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel, stream);
            return;
        case ETensorDType::BF16:
            dispatch_chunk_beta_dtype_forward<TQ, nv_bfloat16>(
                out, final_state, state_scratch, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel, stream);
            return;
        case ETensorDType::FP16:
            dispatch_chunk_beta_dtype_forward<TQ, half>(
                out, final_state, state_scratch, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel, stream);
            return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward: unsupported g dtype");
    }
}

template<typename TQ, typename TG>
void dispatch_chunk_beta_dtype_backward(
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    Tensor& d_g,
    Tensor& d_beta,
    Tensor& d_initial_state,
    const Tensor& d_out,
    const Tensor* d_final_state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints,
    Tensor& workspace,
    cudaStream_t stream) {
    switch (beta.DType) {
        case ETensorDType::FP32:
            launch_gated_delta_rule_chunk_backward<TQ, TG, float>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
                q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, stream);
            return;
        case ETensorDType::BF16:
            launch_gated_delta_rule_chunk_backward<TQ, TG, nv_bfloat16>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
                q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, stream);
            return;
        case ETensorDType::FP16:
            launch_gated_delta_rule_chunk_backward<TQ, TG, half>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
                q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, stream);
            return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward: unsupported beta dtype");
    }
}

template<typename TQ>
void dispatch_chunk_g_dtype_backward(
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    Tensor& d_g,
    Tensor& d_beta,
    Tensor& d_initial_state,
    const Tensor& d_out,
    const Tensor* d_final_state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints,
    Tensor& workspace,
    cudaStream_t stream) {
    switch (g.DType) {
        case ETensorDType::FP32:
            dispatch_chunk_beta_dtype_backward<TQ, float>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
                q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, stream);
            return;
        case ETensorDType::BF16:
            dispatch_chunk_beta_dtype_backward<TQ, nv_bfloat16>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
                q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, stream);
            return;
        case ETensorDType::FP16:
            dispatch_chunk_beta_dtype_backward<TQ, half>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
                q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, stream);
            return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward: unsupported g dtype");
    }
}

}  // namespace

void gated_delta_rule_chunk_forward(
    Tensor& out,
    Tensor& final_state,
    Tensor& state_scratch,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    cudaStream_t stream) {
    validate_gated_delta_shapes(q, k, v, g, beta, initial_state);
    if (chunk_size <= 0) {
        chunk_size = 64;
    }
    if (chunk_size > kMaxChunk) {
        throw std::logic_error("gated_delta_rule_chunk_forward: chunk_size > 64 is not supported");
    }
    if (q.DType != k.DType || q.DType != v.DType || out.DType != q.DType) {
        throw std::logic_error("gated_delta_rule_chunk_forward: q/k/v/out dtype mismatch");
    }
    if (final_state.DType != ETensorDType::FP32) {
        throw std::logic_error("gated_delta_rule_chunk_forward: final_state must be FP32");
    }
    if (initial_state && initial_state->DType != ETensorDType::FP32) {
        throw std::logic_error("gated_delta_rule_chunk_forward: initial_state must be FP32");
    }
    const int B = static_cast<int>(q.Sizes[0]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    if (Kdim > 256) {
        throw std::logic_error("gated_delta_rule_chunk_forward: K > 256 is not supported");
    }
    if (state_scratch.Rank != 4 ||
        state_scratch.DType != ETensorDType::FP32 ||
        state_scratch.Sizes[0] != B ||
        state_scratch.Sizes[1] != H ||
        state_scratch.Sizes[2] != Kdim ||
        state_scratch.Sizes[3] != Vdim) {
        throw std::logic_error("gated_delta_rule_chunk_forward: state_scratch must be [B,H,K,V] FP32");
    }

    // Dispatch to parallelized v2 kernels
    gated_delta_rule_chunk_forward_v2(
        out, final_state, state_scratch, q, k, v, g, beta, initial_state,
        scale, chunk_size, use_qk_l2norm_in_kernel, nullptr, stream);
}

void gated_delta_rule_chunk_backward(
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    Tensor& d_g,
    Tensor& d_beta,
    Tensor& d_initial_state,
    const Tensor& d_out,
    const Tensor* d_final_state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints,
    Tensor& workspace,
    cudaStream_t stream) {
    validate_gated_delta_shapes(q, k, v, g, beta, initial_state);
    if (chunk_size <= 0) {
        chunk_size = 64;
    }
    if (chunk_size > kMaxChunk) {
        throw std::logic_error("gated_delta_rule_chunk_backward: chunk_size > 64 is not supported");
    }
    if (d_out.DType != q.DType) {
        throw std::logic_error("gated_delta_rule_chunk_backward: d_out dtype mismatch");
    }
    if (d_q.DType != q.DType || d_k.DType != q.DType || d_v.DType != q.DType) {
        throw std::logic_error("gated_delta_rule_chunk_backward: d_q/d_k/d_v dtype mismatch");
    }
    if (d_g.DType != g.DType || d_beta.DType != beta.DType) {
        throw std::logic_error("gated_delta_rule_chunk_backward: d_g/d_beta dtype mismatch");
    }
    if (d_initial_state.DType != ETensorDType::FP32) {
        throw std::logic_error("gated_delta_rule_chunk_backward: d_initial_state must be FP32");
    }
    if (d_final_state && d_final_state->DType != ETensorDType::FP32) {
        throw std::logic_error("gated_delta_rule_chunk_backward: d_final_state must be FP32");
    }

    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    if (Kdim > 256) {
        throw std::logic_error("gated_delta_rule_chunk_backward: K > 256 is not supported");
    }
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    if (checkpoints.Rank != 5 ||
        checkpoints.DType != ETensorDType::FP32 ||
        checkpoints.Sizes[0] != B ||
        checkpoints.Sizes[1] != H ||
        checkpoints.Sizes[2] != num_chunks + 1 ||
        checkpoints.Sizes[3] != Kdim ||
        checkpoints.Sizes[4] != Vdim) {
        throw std::logic_error("gated_delta_rule_chunk_backward: invalid checkpoints shape");
    }
    // Multi-kernel backward needs per-chunk workspace:
    // M[64×64] + A[64×64] + W[64×K] + VNEW[64×V] + DU[64×V] + DW[64×K]
    // + DQ[64×K] + DK[64×K] + DG[64] + DB[64] + DHT1[K×V] + C[K×K] + EG[1]
    // Plus dh_storage: K×V per chunk (for Phase 3 to access per-chunk dh)
    const int Lp = 64;
    const long chunk_ws_stride =
        static_cast<long>(Lp) * Lp * 2 +          // M + A
        static_cast<long>(Lp) * Kdim +             // W
        static_cast<long>(Lp) * Vdim +             // VNEW
        static_cast<long>(Lp) * Vdim +             // DU
        static_cast<long>(Lp) * Kdim +             // DW
        static_cast<long>(Lp) * Kdim +             // DQ
        static_cast<long>(Lp) * Kdim +             // DK
        static_cast<long>(Lp) * 2 +                // DG + DB
        static_cast<long>(Kdim) * Vdim +           // DHT1
        static_cast<long>(Kdim) * Kdim +           // C (correction matrix)
        1;                                          // EG (exp(g_last))
    const long dh_storage_per_chunk = static_cast<long>(Kdim) * Vdim;
    const long workspace_needed = static_cast<long>(num_chunks) * chunk_ws_stride
        + static_cast<long>(num_chunks) * dh_storage_per_chunk;
    if (workspace.Rank != 3 ||
        workspace.DType != ETensorDType::FP32 ||
        workspace.Sizes[0] != B ||
        workspace.Sizes[1] != H ||
        workspace.Sizes[2] < workspace_needed) {
        throw std::logic_error("gated_delta_rule_chunk_backward: invalid workspace shape");
    }

    // Dispatch to parallelized v2 kernels
    gated_delta_rule_chunk_backward_v2(
        d_q, d_k, d_v, d_g, d_beta, d_initial_state, d_out, d_final_state,
        q, k, v, g, beta, initial_state, scale, chunk_size,
        use_qk_l2norm_in_kernel, checkpoints, workspace, false, stream);
}
