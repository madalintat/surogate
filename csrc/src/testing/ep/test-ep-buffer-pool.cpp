// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for ep::EPBufferPool — the GPU buffer pool shared by all
// EP strategies. Covers acquire best-fit, release, retire lifetime, and
// the step-end clear_retired / clear_pool boundaries.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <unordered_set>

#include <cuda_runtime.h>

#include "runtime/ep/ep_buffer_pool.h"

namespace {

std::size_t device_allocated_bytes(const void* ptr) {
    cudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess) return 0;
    return attrs.devicePointer ? 1 : 0;
}

}  // namespace

TEST_CASE("EPBufferPool zero-size acquire returns nullptr", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* p = pool.acquire(0);
    REQUIRE(p == nullptr);
    REQUIRE(pool.pool_count() == 0);
}

TEST_CASE("EPBufferPool acquire allocates when empty", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* p = pool.acquire(1024);
    REQUIRE(p != nullptr);
    REQUIRE(device_allocated_bytes(p) == 1);

    pool.release(p, 1024);
    REQUIRE(pool.pool_count() == 1);
}

TEST_CASE("EPBufferPool acquire recycles released buffer", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* first = pool.acquire(2048);
    REQUIRE(first != nullptr);
    pool.release(first, 2048);
    REQUIRE(pool.pool_count() == 1);

    // Same-size re-acquire must return the same pointer.
    void* second = pool.acquire(2048);
    REQUIRE(second == first);
    REQUIRE(pool.pool_count() == 0);

    pool.release(second, 2048);
}

TEST_CASE("EPBufferPool acquire picks best-fit among pool candidates", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* small = pool.acquire(1024);
    void* medium = pool.acquire(4096);
    void* large = pool.acquire(16384);

    pool.release(small, 1024);
    pool.release(medium, 4096);
    pool.release(large, 16384);
    REQUIRE(pool.pool_count() == 3);

    // Requesting 2048 must return the 4096 slot (best-fit >= need).
    void* chosen = pool.acquire(2048);
    REQUIRE(chosen == medium);
    REQUIRE(pool.pool_count() == 2);

    // Small (1024) must still be in the pool and re-acquirable as-is.
    void* reused_small = pool.acquire(1024);
    REQUIRE(reused_small == small);

    // Large (16384) remains.
    REQUIRE(pool.pool_count() == 1);
    void* reused_large = pool.acquire(8192);
    REQUIRE(reused_large == large);

    pool.release(reused_small, 1024);
    pool.release(chosen, 4096);
    pool.release(reused_large, 16384);
}

TEST_CASE("EPBufferPool acquire allocates fresh when no entry fits", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* small = pool.acquire(1024);
    pool.release(small, 1024);
    REQUIRE(pool.pool_count() == 1);

    // Request larger than any pooled entry → fresh cudaMalloc, pool untouched.
    void* bigger = pool.acquire(8192);
    REQUIRE(bigger != nullptr);
    REQUIRE(bigger != small);
    REQUIRE(pool.pool_count() == 1);

    pool.release(bigger, 8192);
}

TEST_CASE("EPBufferPool release ignores null / zero-byte", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    pool.release(nullptr, 1024);
    pool.release(reinterpret_cast<void*>(0x1), 0);
    REQUIRE(pool.pool_count() == 0);
}

TEST_CASE("EPBufferPool retire does not feed back into pool", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* p = pool.acquire(4096);
    pool.retire(p, 4096);

    REQUIRE(pool.pool_count() == 0);
    REQUIRE(pool.retired_count() == 1);

    // Acquiring 4096 must NOT return the retired buffer (still kept alive).
    void* fresh = pool.acquire(4096);
    REQUIRE(fresh != p);
    REQUIRE(fresh != nullptr);

    pool.release(fresh, 4096);
}

TEST_CASE("EPBufferPool clear_retired frees retired list", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    void* a = pool.acquire(1024);
    void* b = pool.acquire(2048);
    pool.retire(a, 1024);
    pool.retire(b, 2048);
    REQUIRE(pool.retired_count() == 2);

    pool.clear_retired();
    REQUIRE(pool.retired_count() == 0);
    // Pool still intact.
    REQUIRE(pool.pool_count() == 0);
}

TEST_CASE("EPBufferPool clear_pool frees active pool entries", "[ep][buffer_pool]") {
    ep::EPBufferPool pool;
    std::unordered_set<void*> allocated;
    for (std::size_t bytes : {1024u, 2048u, 4096u, 8192u}) {
        void* p = pool.acquire(bytes);
        REQUIRE(p != nullptr);
        allocated.insert(p);
        pool.release(p, bytes);
    }
    REQUIRE(pool.pool_count() == 4);

    pool.clear_pool();
    REQUIRE(pool.pool_count() == 0);
    // Subsequent acquire must allocate fresh — cleared pointers are freed.
    void* fresh = pool.acquire(1024);
    REQUIRE(fresh != nullptr);
    pool.release(fresh, 1024);
}

TEST_CASE("EPBufferPool destructor frees pool + retired", "[ep][buffer_pool]") {
    // This test mainly asserts no leak / no crash on dtor; absence of
    // cudaErrorMemoryAllocation on subsequent code is the signal.
    {
        ep::EPBufferPool pool;
        void* a = pool.acquire(1024);
        void* b = pool.acquire(2048);
        pool.release(a, 1024);
        pool.retire(b, 2048);
        REQUIRE(pool.pool_count() == 1);
        REQUIRE(pool.retired_count() == 1);
    }
    // If the destructor leaked, repeated test runs would eventually
    // exhaust GPU memory. Smoke-test a small allocation still works.
    void* probe = nullptr;
    REQUIRE(cudaMalloc(&probe, 4096) == cudaSuccess);
    REQUIRE(probe != nullptr);
    cudaFree(probe);
}
