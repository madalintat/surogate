// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GPU buffer pool for Expert Parallelism intermediates.
//
// EP ops allocate many short-lived intermediate buffers (A2A send/recv,
// reorder maps, etc.). All EP ops run on MainStream, so CUDA stream
// ordering guarantees safe buffer reuse without explicit synchronization.
//
// Pattern:
//   acquire() finds a recycled buffer >= requested size (best-fit) or
//            cudaMalloc's a new one.
//   release() returns the buffer to the pool for reuse.
//   retire() takes ownership of a buffer but keeps it alive until
//            end-of-step. Used for shared EP buffers that can be
//            referenced by saved tensors across layer boundaries.

#ifndef SUROGATE_SRC_RUNTIME_EP_EP_BUFFER_POOL_H
#define SUROGATE_SRC_RUNTIME_EP_EP_BUFFER_POOL_H

#include <cstddef>
#include <vector>

namespace ep {

struct EpPoolEntry {
    void* ptr = nullptr;
    std::size_t bytes = 0;
};

class EPBufferPool {
public:
    EPBufferPool() = default;
    ~EPBufferPool();

    EPBufferPool(const EPBufferPool&) = delete;
    EPBufferPool& operator=(const EPBufferPool&) = delete;

    /// Best-fit allocation: returns a recycled buffer >= need (or cudaMalloc if
    /// none available). Returns nullptr when need == 0.
    void* acquire(std::size_t need);

    /// Return a buffer to the pool for future reuse.
    void release(void* ptr, std::size_t bytes);

    /// Transfer buffer ownership to the retired list; kept alive until
    /// clear_retired() is called (typically at step end).
    void retire(void* ptr, std::size_t bytes);

    /// Free all retired buffers (call at end of forward/backward step when no
    /// saved tensors reference them).
    void clear_retired();

    /// Free all pool buffers (call at end of step to reclaim drift-sized zombies).
    void clear_pool();

    /// Free all resources (called by destructor).
    void clear_all();

    /// Number of pool entries (for diagnostics).
    std::size_t pool_count() const {
        return mPool.size();
    }
    std::size_t retired_count() const {
        return mRetired.size();
    }

private:
    std::vector<EpPoolEntry> mPool;
    std::vector<EpPoolEntry> mRetired;
};

}  // namespace ep

#endif  // SUROGATE_SRC_RUNTIME_EP_EP_BUFFER_POOL_H
