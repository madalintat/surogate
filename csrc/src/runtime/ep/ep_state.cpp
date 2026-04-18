// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/ep/ep_state.h"

#include <cuda_runtime.h>

namespace ep {

void EpLayerState::free_gpu() {
    if (send_order_gpu) {
        cudaFree(send_order_gpu);
        send_order_gpu = nullptr;
        send_order_bytes = 0;
    }
    if (recv_reorder_gpu) {
        cudaFree(recv_reorder_gpu);
        recv_reorder_gpu = nullptr;
        recv_reorder_bytes = 0;
    }
    if (llep_send_reorder_gpu) {
        cudaFree(llep_send_reorder_gpu);
        llep_send_reorder_gpu = nullptr;
        llep_send_reorder_bytes = 0;
    }
    if (local_scatter_gpu) {
        cudaFree(local_scatter_gpu);
        local_scatter_gpu = nullptr;
        local_scatter_bytes = 0;
    }
    if (sorted_recv_gpu) {
        cudaFree(sorted_recv_gpu);
        sorted_recv_gpu = nullptr;
        sorted_recv_bytes = 0;
    }
    if (combined_gpu) {
        cudaFree(combined_gpu);
        combined_gpu = nullptr;
        combined_bytes = 0;
    }
    if (llep_combined_gpu) {
        cudaFree(llep_combined_gpu);
        llep_combined_gpu = nullptr;
        llep_combined_bytes = 0;
    }
    if (dispatch_bwd_send_gpu) {
        cudaFree(dispatch_bwd_send_gpu);
        dispatch_bwd_send_gpu = nullptr;
        dispatch_bwd_send_bytes = 0;
    }
    if (dispatch_bwd_out_gpu) {
        cudaFree(dispatch_bwd_out_gpu);
        dispatch_bwd_out_gpu = nullptr;
        dispatch_bwd_out_bytes = 0;
    }
    if (combine_bwd_sorted_gpu) {
        cudaFree(combine_bwd_sorted_gpu);
        combine_bwd_sorted_gpu = nullptr;
        combine_bwd_sorted_bytes = 0;
    }
}

void EpLayerState::build_reverse_a2a_elem_splits(int stride,
                                                 std::vector<int>& reverse_send_elem,
                                                 std::vector<int>& reverse_recv_elem) const {
    const int n = static_cast<int>(send_splits.size());
    reverse_send_elem.resize(n);
    reverse_recv_elem.resize(n);
    for (int p = 0; p < n; ++p) {
        reverse_send_elem[p] = recv_splits[p] * stride;
        reverse_recv_elem[p] = send_splits[p] * stride;
    }
}

void EpLayerState::build_forward_a2a_elem_splits(int stride,
                                                 std::vector<int>& send_elem,
                                                 std::vector<int>& recv_elem) const {
    const int n = static_cast<int>(send_splits.size());
    send_elem.resize(n);
    recv_elem.resize(n);
    for (int p = 0; p < n; ++p) {
        send_elem[p] = send_splits[p] * stride;
        recv_elem[p] = recv_splits[p] * stride;
    }
}

}  // namespace ep
