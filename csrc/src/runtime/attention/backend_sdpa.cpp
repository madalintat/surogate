// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Reference SDPA math backend (PyTorch's ``math`` SDPA equivalent).
// Two cuBLAS matmuls around an explicit softmax. Last-resort fallback
// when the fused kernels can't take the shape — typically head_dim > 256
// with full causal attention.

#include <memory>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/dsl_run_state.h"
#include "runtime/attention/attention_backend.h"
#include "runtime/attention/attention_kernels.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

struct PackedDocSegment {
    int batch_idx = 0;
    long row_start = 0;
    long global_start = 0;
    long length = 0;
};

std::vector<PackedDocSegment> collect_packed_doc_segments(const AttentionParams& p) {
    if (p.cu_seqlens_cpu == nullptr) {
        throw std::logic_error("SDPAAttention: missing host cu_seqlens for packed full attention");
    }
    if (p.num_docs <= 0) {
        throw std::logic_error("SDPAAttention: packed full attention requires num_docs > 0");
    }

    std::vector<PackedDocSegment> docs;
    docs.reserve(static_cast<std::size_t>(p.num_docs));

    long row_offset = 0;
    int batch_idx = 0;
    long covered = 0;
    for (int doc_idx = 0; doc_idx < p.num_docs; ++doc_idx) {
        const long start = static_cast<long>(p.cu_seqlens_cpu[doc_idx]);
        const long end = static_cast<long>(p.cu_seqlens_cpu[doc_idx + 1]);
        const long len = end - start;
        if (len <= 0) {
            throw std::logic_error("SDPAAttention: encountered empty packed document");
        }
        if (start != covered) {
            throw std::logic_error("SDPAAttention: cu_seqlens are not contiguous");
        }
        if (batch_idx >= p.B) {
            throw std::logic_error("SDPAAttention: packed documents exceed batch capacity");
        }
        if (row_offset + len > p.T) {
            throw std::logic_error("SDPAAttention: packed document crosses a batch row");
        }

        PackedDocSegment seg;
        seg.batch_idx = batch_idx;
        seg.row_start = row_offset;
        seg.global_start = start;
        seg.length = len;
        docs.push_back(seg);

        covered = end;
        row_offset += len;
        if (row_offset == p.T) {
            row_offset = 0;
            ++batch_idx;
        }
    }

    if (covered != static_cast<long>(p.total_doc_tokens) || covered != static_cast<long>(p.B) * p.T) {
        throw std::logic_error("SDPAAttention: packed documents do not cover the full dense batch");
    }
    if (batch_idx != p.B || row_offset != 0) {
        throw std::logic_error("SDPAAttention: packed documents leave a partial batch row");
    }
    return docs;
}

Tensor flatten_bt(const Tensor& src, long heads, long hs) {
    Tensor flat = src;
    flat.Rank = 3;
    flat.Sizes[0] = src.Sizes[0] * src.Sizes[1];
    flat.Sizes[1] = heads;
    flat.Sizes[2] = hs;
    flat.Sizes[3] = 1;
    flat.Sizes[4] = 1;
    return flat;
}

Tensor doc_slice_4d(const Tensor& flat, long global_start, long length, long heads, long hs) {
    Tensor doc = slice(flat, 0, global_start, global_start + length);
    doc.Rank = 4;
    doc.Sizes[0] = 1;
    doc.Sizes[1] = length;
    doc.Sizes[2] = heads;
    doc.Sizes[3] = hs;
    doc.Sizes[4] = 1;
    return doc;
}

void copy_lse_doc_to_dense(Tensor& dense_lse,
                           const Tensor& doc_lse,
                           int batch_idx,
                           long row_start,
                           int Hq,
                           int T,
                           cudaStream_t stream) {
    float* dst = dense_lse.get<float>() + static_cast<long>(batch_idx) * Hq * T + row_start;
    const float* src = doc_lse.get<float>();
    const std::size_t dst_pitch = static_cast<std::size_t>(T) * sizeof(float);
    const std::size_t src_pitch = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    const std::size_t width = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(dst,
                                 dst_pitch,
                                 src,
                                 src_pitch,
                                 width,
                                 static_cast<std::size_t>(Hq),
                                 cudaMemcpyDeviceToDevice,
                                 stream));
}

void copy_lse_dense_to_doc(Tensor& doc_lse,
                           const Tensor& dense_lse,
                           int batch_idx,
                           long row_start,
                           int Hq,
                           int T,
                           cudaStream_t stream) {
    const float* src = dense_lse.get<float>() + static_cast<long>(batch_idx) * Hq * T + row_start;
    float* dst = doc_lse.get<float>();
    const std::size_t src_pitch = static_cast<std::size_t>(T) * sizeof(float);
    const std::size_t dst_pitch = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    const std::size_t width = static_cast<std::size_t>(doc_lse.Sizes[2]) * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DAsync(dst,
                                 dst_pitch,
                                 src,
                                 src_pitch,
                                 width,
                                 static_cast<std::size_t>(Hq),
                                 cudaMemcpyDeviceToDevice,
                                 stream));
}

void forward_packed_sdpa(AttentionParams& p) {
    DslRunState& rs = *p.run_state;
    std::vector<Tensor>& temps = *p.temps;
    const auto docs = collect_packed_doc_segments(p);

    Tensor qkv_flat = flatten_bt(p.qkv, p.Hq + 2 * p.Hkv, p.Hs);
    Tensor out_flat = flatten_bt(p.out, p.Hq, p.Hs);

    for (const PackedDocSegment& doc : docs) {
        Tensor qkv_doc = doc_slice_4d(qkv_flat, doc.global_start, doc.length, p.Hq + 2 * p.Hkv, p.Hs);
        Tensor out_doc = doc_slice_4d(out_flat, doc.global_start, doc.length, p.Hq, p.Hs);
        Tensor lse_doc = rs.temp_alloc(ETensorDType::FP32, {1, p.Hq, doc.length}, "packed_sdpa_lse_doc");
        temps.push_back(lse_doc);

        attention_forward_matmul(out_doc,
                                 lse_doc,
                                 qkv_doc,
                                 /*B=*/1,
                                 static_cast<int>(doc.length),
                                 p.Hq,
                                 p.Hkv,
                                 p.Hs,
                                 p.cublas_handle,
                                 p.stream,
                                 p.softmax_scale,
                                 p.window_size);
        copy_lse_doc_to_dense(p.lse, lse_doc, doc.batch_idx, doc.row_start, p.Hq, p.T, p.stream);
    }
}

void backward_packed_sdpa(AttentionParams& p) {
    DslRunState& rs = *p.run_state;
    std::vector<Tensor>& temps = *p.temps;
    const auto docs = collect_packed_doc_segments(p);

    Tensor qkv_flat = flatten_bt(p.qkv, p.Hq + 2 * p.Hkv, p.Hs);
    Tensor out_flat = flatten_bt(p.out, p.Hq, p.Hs);
    Tensor d_out_flat = flatten_bt(p.d_out, p.Hq, p.Hs);
    Tensor d_qkv_flat = flatten_bt(p.d_qkv, p.Hq + 2 * p.Hkv, p.Hs);

    for (const PackedDocSegment& doc : docs) {
        Tensor lse_doc = rs.temp_alloc(ETensorDType::FP32, {1, p.Hq, doc.length}, "packed_sdpa_lse_doc");
        temps.push_back(lse_doc);
        copy_lse_dense_to_doc(lse_doc, p.lse, doc.batch_idx, doc.row_start, p.Hq, p.T, p.stream);

        Tensor qkv_doc = doc_slice_4d(qkv_flat, doc.global_start, doc.length, p.Hq + 2 * p.Hkv, p.Hs);
        Tensor out_doc = doc_slice_4d(out_flat, doc.global_start, doc.length, p.Hq, p.Hs);
        Tensor d_out_doc = doc_slice_4d(d_out_flat, doc.global_start, doc.length, p.Hq, p.Hs);
        Tensor d_qkv_doc = doc_slice_4d(d_qkv_flat, doc.global_start, doc.length, p.Hq + 2 * p.Hkv, p.Hs);

        attention_backward_matmul(d_qkv_doc,
                                  lse_doc,
                                  out_doc,
                                  d_out_doc,
                                  qkv_doc,
                                  /*B=*/1,
                                  static_cast<int>(doc.length),
                                  p.Hq,
                                  p.Hkv,
                                  p.Hs,
                                  p.cublas_handle,
                                  p.stream,
                                  p.softmax_scale,
                                  p.window_size);
    }
}

class SDPAAttention final : public AttentionBackend {
public:
    const char* name() const override {
        return "sdpa";
    }

    int priority() const override {
        return attention_priority::kSDPA;
    }

    bool supports(const AttentionParams& p) const override {
        if (p.Hs <= 0) {
            return false;
        }
        if (p.cublas_handle == nullptr) {
            return false;
        }
        if (p.cu_seqlens != nullptr) {
            if (p.cu_seqlens_cpu == nullptr) {
                return false;
            }
            if (p.run_state == nullptr || p.temps == nullptr) {
                return false;
            }
        }
        return true;
    }

    void forward(AttentionParams& p) override {
        if (p.cu_seqlens != nullptr) {
            forward_packed_sdpa(p);
            return;
        }
        attention_forward_matmul(p.out,
                                 p.lse,
                                 p.qkv,
                                 p.B,
                                 p.T,
                                 p.Hq,
                                 p.Hkv,
                                 p.Hs,
                                 p.cublas_handle,
                                 p.stream,
                                 p.softmax_scale,
                                 p.window_size);
    }

    void backward(AttentionParams& p) override {
        if (p.cu_seqlens != nullptr) {
            backward_packed_sdpa(p);
            return;
        }
        attention_backward_matmul(p.d_qkv,
                                  p.lse,
                                  p.out,
                                  p.d_out,
                                  p.qkv,
                                  p.B,
                                  p.T,
                                  p.Hq,
                                  p.Hkv,
                                  p.Hs,
                                  p.cublas_handle,
                                  p.stream,
                                  p.softmax_scale,
                                  p.window_size);
    }
};

struct SDPAAttentionAutoRegister {
    SDPAAttentionAutoRegister() {
        AttentionBackendRegistry::instance().add(std::make_unique<SDPAAttention>());
    }
};
const SDPAAttentionAutoRegister _sdpa_attention_auto_register;

}  // namespace
}  // namespace dsl
