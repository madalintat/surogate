---
name: cuda
description: Write or optimize CUDA kernels for the Surogate framework. Use when the user wants to create CUDA kernels, optimize existing kernels for specific GPU architectures (sm89/sm90/sm100/sm103/sm120/sm121), implement custom attention/normalization/activation layers, or debug kernel performance issues.
argument-hint: [kernel-description]
---

# Write a CUDA Kernel for Surogate

You are implementing or optimizing a CUDA kernel for the Surogate LLM training framework targeting NVIDIA GPUs (SM89, SM90, SM100, SM103, SM120, SM121).
The kernel request is: **$ARGUMENTS**

## Important: Read Architecture Guide First

Before writing any CUDA code, identify the target GPU architecture and read the relevant optimization guide:

**Architecture-specific guides** (read based on target GPU):
- `.claude/skills/cuda/sm89-optimization-guide.md` — Ada Lovelace (L40S, RTX 4070/4090/6000 Ada)
- `.claude/skills/cuda/sm90-optimization-guide.md` — Hopper (H100, H200)
- `.claude/skills/cuda/sm100-optimization-guide.md` — Blackwell Datacenter (B200, GB200)
- `.claude/skills/cuda/sm103-optimization-guide.md` — Blackwell Ultra (B300, GB300)
- `.claude/skills/cuda/sm120-optimization-guide.md` — Blackwell Desktop (RTX 5090, RTX PRO 6000)

**Templates and troubleshooting** (read as needed):
- `.claude/skills/cuda/kernel-templates.md` — Complete copy-paste ready kernel templates
- `.claude/skills/cuda/troubleshooting.md` — Common issues, debugging tools, compute-sanitizer

**Existing kernel implementations** (study for patterns):
- `csrc/src/kernels/` — All production CUDA kernels (attention, rmsnorm, rope, swiglu, etc.)

## GPU Architecture Quick Reference

### Feature Availability Matrix

| Feature | sm89 (Ada) | sm90 (Hopper) | sm100/103 (Blackwell DC) | sm120 (Desktop BW) | sm121 (DGX Spark) |
|---------|:----------:|:-------------:|:------------------------:|:-------------------:|:-----------------:|
| TMA | No | Yes (v1) | Yes (v2, multicast) | No | No |
| Thread Block Clusters | No | Yes | Yes (enhanced) | No | No |
| Distributed Shared Memory | No | Yes | Yes (enhanced) | No | No |
| TMEM (Tensor Memory) | No | No | Yes | No | No |
| WGMMA / tcgen05 | No | wgmma.mma_async | tcgen05 WGMMA | No | No |
| cp.async | Yes | Yes | Yes | Yes | Yes |
| FP4 Tensor Cores | No | No | Yes (NVFP4) | Yes | Yes |
| FP8 Tensor Cores | Yes (4th gen) | Yes (4th gen) | Yes (5th gen, 2x) | Yes (5th gen) | Yes (5th gen) |

### Hardware Specs Quick Reference

| GPU | SMs | Warps/SM | Shared Mem/SM | L2 Cache | Memory BW | Max Blocks/SM |
|-----|-----|----------|---------------|----------|-----------|---------------|
| L40S (sm89) | 142 | 48 | 100 KB | 96 MB | 864 GB/s | 24 |
| RTX 4090 (sm89) | 128 | 48 | 100 KB | 72 MB | 1.01 TB/s | 24 |
| RTX 4070 (sm89) | 46 | 48 | 100 KB | 36 MB | 504 GB/s | 24 |
| H100 (sm90) | 132 | 64 | 192 KB | 50 MB | 3.35 TB/s | 16 |
| H200 (sm90) | 132 | 64 | 192 KB | 50 MB | 4.8 TB/s | 16 |
| B200 (sm100) | 148 | 64 | 228 KB | 126 MB | 8 TB/s | 32 |
| B300 (sm103) | 160 | 64 | 228 KB | 192 MB | 8 TB/s | 32 |
| RTX 5090 (sm120) | 170 | 48 | 128 KB | 96 MB | 1.79 TB/s | 24 |
| RTX PRO 6000 (sm120) | 188 | 48 | 128 KB | 128 MB | 1.8 TB/s | 24 |
| DGX Spark (sm121) | 48 | 48 | 100 KB | 24 MB | 273 GB/s | 24 |

### Occupancy by Block Size

**sm89/sm120/sm121 (48 warps/SM max, 1536 threads/SM):**
```
64 threads/block:   24 blocks -> 48 warps -> 100%
128 threads/block:  12 blocks -> 48 warps -> 100%
256 threads/block:  6 blocks  -> 48 warps -> 100%
512 threads/block:  3 blocks  -> 48 warps -> 100%
1024 threads/block: 1 block   -> 32 warps -> 67%  <-- AVOID
```

**sm90/sm100/sm103 (64 warps/SM max, 2048 threads/SM):**
```
64 threads/block:   32 blocks -> 64 warps -> 100%
128 threads/block:  16 blocks -> 64 warps -> 100%
256 threads/block:  8 blocks  -> 64 warps -> 100%
512 threads/block:  4 blocks  -> 64 warps -> 100%
1024 threads/block: 2 blocks  -> 64 warps -> 100%
```

## Core Kernel Design Principles

1. **Parallelism First**: Design for thousands of concurrent threads; serial thinking is the enemy
2. **Memory Hierarchy Awareness**: Global memory is 100x slower than shared, 1000x slower than registers
3. **Coalesced Access**: Adjacent threads MUST access adjacent memory; misalignment reduces BW by 32x
4. **Occupancy Over Cleverness**: Maximize active warps via register count, shared memory, and block dimensions
5. **Minimize Host-Device Transfers**: PCIe is the bottleneck; overlap with streams and pinned memory

## Supported Data Types

All kernels support three precision modes:
- `__half` (FP16) — Default for inference
- `__nv_bfloat16` (BF16) — Preferred for training
- `float` (FP32) — Reference/debugging

### Type Conversion Helpers (include in every .cu file)

```cuda
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }
```

## Kernel Code Generation Rules

- Block size must be a multiple of warp size (32); prefer 128, 256, or 512
- Calculate grid size as `(n + block_size - 1) / block_size`
- Always include bounds checking: `if (idx < n)` at the top of every kernel
- Use grid-stride loops for kernels that must handle arbitrary data sizes
- Document thread mapping: which dimension maps to which data axis
- Mark device-only helpers as `__device__`, host+device as `__host__ __device__`

### Grid-Stride Loop Pattern

```cuda
__global__ void saxpy(float a, const float* x, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}
```

## Vectorized Memory Access (Critical for Performance)

**BFloat16 vectorization using `__nv_bfloat162`:**

```cuda
const __nv_bfloat162* vec_input = reinterpret_cast<const __nv_bfloat162*>(row_input);

#pragma unroll 4
for (int i = tid; i < vec_hidden; i += stride) {
    __nv_bfloat162 v = vec_input[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    sum_sq += v0 * v0 + v1 * v1;
}
```

**FP16 vectorization using `__half2`:**

```cuda
const __half2* vec_input = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_input[i];
float v0 = __half2float(v.x);
float v1 = __half2float(v.y);
```

**FP32 vectorization using `float4`:**

```cuda
const float4* vec_input = reinterpret_cast<const float4*>(row_input);
float4 v = vec_input[i];
sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
```

## Warp-Level Reductions

```cuda
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}
```

## Memory Hierarchy

| Memory Type | Scope | Latency (cycles) | Size | Cached | Read/Write |
|-------------|-------|-------------------|------|--------|------------|
| Registers | Thread | 1 | ~255 per thread | N/A | R/W |
| Shared | Block | ~5 | 48-228 KB per SM | N/A | R/W |
| L1 Cache | SM | ~28 | 48-192 KB per SM | Auto | R |
| L2 Cache | Device | ~200 | 24-192 MB | Auto | R/W |
| Global | Device | ~400-600 | 4-288 GB (HBM/GDDR) | Yes | R/W |
| Constant | Device | ~5 (cached) | 64 KB | Yes (broadcast) | R |

**Decision guide:**
- Data reused within a thread -> registers (automatic via local variables)
- Data shared across threads in a block -> `__shared__` memory
- Read-only data broadcast to all threads -> `__constant__` memory
- Everything else -> global memory with coalesced access patterns

## Synchronization Rules

- Use `__syncthreads()` after every shared memory write before any thread reads another thread's value
- NEVER place `__syncthreads()` inside a conditional branch that not all threads reach (deadlock)
- Use `__syncwarp()` for warp-level synchronization instead of relying on implicit warp-sync
- Use `cudaDeviceSynchronize()` sparingly; prefer `cudaStreamSynchronize()`
- Use CUDA events for fine-grained inter-stream ordering

## Error Handling

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

## Architecture-Specific Key Optimizations

### sm89 (Ada Lovelace)
- **100 KB shared memory** — smallest tile budget (except sm121)
- **No TMA/Clusters/WGMMA** — use `cp.async` for async loads, WMMA for tensor cores
- **Large L2 (72-96 MB)** — compensates for lower GDDR bandwidth
- **Dual-issue FP32** — extra FP32 ops are nearly free; fuse aggressively
- **Avoid 1024-thread blocks** (67% occupancy cap)
- **Kernel fusion is critical** — lower BW means every launch costs more

### sm90 (Hopper)
- **192 KB shared memory** — large tiles possible
- **TMA** — hardware bulk global->shared transfers, bypasses register file
- **Thread Block Clusters + DSMEM** — cross-block shared memory cooperation
- **Warpgroup MMA** — 4-warp groups for larger MMA tiles
- **Max 16 blocks/SM** — prefer larger blocks

### sm100/sm103 (Blackwell Datacenter)
- **228 KB shared memory** — largest tiles of any architecture
- **tcgen05 WGMMA + TMEM** — tensor operands in dedicated memory, frees registers
- **TMA v2 with multicast** — broadcast tiles across cluster members
- **32 blocks/SM** — all block sizes achieve 100% occupancy
- **126-192 MB L2** — massive KV cache residency
- **8 TB/s HBM3e** — 2.4x Hopper
- **Dual-die (sm100: 148 SMs, sm103: 160 SMs)** — grid multiples matter

### sm120 (Blackwell Desktop)
- **128 KB shared memory** — larger than Ada, much smaller than sm100
- **No TMA/Clusters/TMEM/WGMMA** — architecturally closer to Ada than datacenter Blackwell
- **WMMA** for tensor cores (16x16x16 tiles), register-consuming fragments
- **5th-gen tensor cores with FP4** — NVFP4 for inference
- **GDDR7 at 1.8 TB/s** — ~2x Ada but ~4x less than sm100
- **Kernel fusion is the most impactful optimization**
- **Avoid 1024-thread blocks** (67% occupancy cap)

### sm121 (DGX Spark)
- **100 KB shared memory, 48 SMs, 273 GB/s LPDDR5X** — most constrained target
- **Maximize per-SM work and fusion** — every SM must be fully utilized
- **24 MB L2** — prefer shared memory over relying on L2

## Thread Configuration Guidelines

For element-wise ops (RoPE, GEGLU, activation):
```cuda
constexpr int BLOCK_SIZE = 256;
int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

For reduction ops (LayerNorm, RMSNorm) with vectorization:
```cuda
// Divide by 2 for bf16/fp16 vectorized access
int threads = min(hidden_size / 2, MAX_THREADS);
threads = max(threads, WARP_SIZE);
threads = (threads + 32 - 1) / 32 * 32;  // Round to warp boundary
```

For attention:
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 64`, `BLOCK_SIZE_K = 64`
- `NUM_WARPS = 8`

## Coalesced Memory Access

```cuda
// BAD: Strided access
out[x * H + y] = in[y * W + x];  // Write is strided

// GOOD: Use shared memory to coalesce both reads and writes
__shared__ float tile[32][33]; // +1 padding avoids bank conflicts
```

## Launch Configuration

```cuda
void launchKernel(float* d_data, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    numBlocks = min(numBlocks, props.maxGridSize[0]);

    kernel1D<<<numBlocks, blockSize>>>(d_data, n);
}
```

## Launch Bounds

```cpp
__global__ void __launch_bounds__(256, 4)
boundedKernel(float* data, int n) {
    // Kernel limited to 256 threads, compiler targets 4 blocks/SM
}
```

## Profiling

```bash
# System-level profiling
nsys profile -o profile_report python your_script.py

# Detailed kernel analysis
ncu --set full -o metrics.ncu-rep python your_script.py

# Key metrics to watch:
# - Achieved occupancy
# - Memory throughput (% of peak)
# - L2 hit rate
# - Warp stall reasons
# - Tensor core utilization (sm90+)

# Register/shared memory usage
nvcc --ptxas-options=-v your_kernel.cu

# PTX/SASS analysis
nvcc -ptx -o program.ptx program.cu
cuobjdump -sass program > program.sass
```

## Debugging

```bash
# Memory check
compute-sanitizer --tool memcheck ./cuda_program

# Race condition detection
compute-sanitizer --tool racecheck ./cuda_program

# Uninitialized memory
compute-sanitizer --tool initcheck ./cuda_program

# Synchronization validation
compute-sanitizer --tool synccheck ./cuda_program
```

## Compilation Flags

```bash
# Single architecture
nvcc -arch=sm_90 -O3 your_kernel.cu

# Multi-architecture fat binary
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_120,code=sm_120 \
     -O3 your_kernel.cu

# Useful flags:
# -maxrregcount=N    Limit registers per thread
# --ptxas-options=-v Print register/smem usage
# -lineinfo          Add debug line info
# --use_fast_math    Fast but less precise math
```

## Surogate Integration

### Adding a New Kernel to Surogate

1. **Create kernel file** in `csrc/src/kernels/your_kernel.cu`
2. **Declare in kernels.h** — add function declarations in `csrc/src/kernels/kernels.h`
3. **Create C++ op wrapper** in `csrc/src/runtime/ops/your_op.cpp`
4. **Register in compiled_ops** in `csrc/src/runtime/dsl/compiled_ops.cpp`
5. **Add shape signature** in `csrc/src/runtime/dsl/op_shape_signatures.cpp`
6. **Add autodiff rule** in `csrc/src/runtime/dsl/autodiff_rules.cpp`
7. **Add CMake target** in `csrc/src/CMakeLists.txt`
8. **Add test** in `csrc/src/testing/kernels/test-your-kernel.cu`

### Build and Test

```bash
make build          # Build everything
make test-unit      # Run unit tests
make test-all       # Run all tests (unit + integration)
```

## Architecture Selection Decision Tree

```
Target GPU known?
├── SM100/103 (B200/B300/GB200/GB300)
│   → Use TMA + WGMMA + clusters + TMEM
│   → 228 KB shared memory, 8 TB/s BW
│   → Grid multiples of 148 (sm100) or 160 (sm103)
│
├── SM90 (H100/H200)
│   → Use TMA + warpgroup MMA + clusters
│   → 192 KB shared memory, 3.35-4.8 TB/s BW
│   → Grid multiples of 132
│
├── SM120 (RTX 5090/PRO 6000)
│   → Use cp.async + WMMA (NO TMA/clusters/TMEM)
│   → 128 KB shared memory, 1.8 TB/s GDDR7
│   → Fusion is critical; grid multiples of 170/188
│
├── SM89 (L40S/RTX 4090/4070)
│   → Use cp.async + WMMA (NO TMA/clusters/TMEM)
│   → 100 KB shared memory, 504 GB/s-1.01 TB/s GDDR
│   → Fusion is critical; grid multiples of 46/128/142
│
└── SM121 (DGX Spark)
    → Same as sm120 but 100 KB smem, 48 SMs, 273 GB/s
    → Most constrained — maximize per-SM work
```

## Checklist

Before considering the kernel implementation complete:

- [ ] Read the architecture-specific optimization guide for the target GPU
- [ ] Block size is a multiple of 32; prefer 128/256/512
- [ ] Bounds checking on all memory accesses
- [ ] Memory accesses are coalesced (adjacent threads access adjacent addresses)
- [ ] Vectorized loads/stores used where possible (`__nv_bfloat162`, `float4`)
- [ ] Shared memory bank conflicts avoided (padding where needed)
- [ ] Reductions use warp shuffles, not shared memory within a warp
- [ ] `__syncthreads()` placed correctly (not in divergent branches)
- [ ] FP32 used for accumulation/reductions, lower precision for memory
- [ ] Grid size accounts for target SM count
- [ ] Kernel compiles and runs correctly
- [ ] Error handling with CUDA_CHECK on all API calls
- [ ] Tested with representative inputs and sizes
