# CUDA Kernels for Surogate

This skill provides patterns and guidance for developing optimized CUDA kernels targeting NVIDIA GPUs (SM89, SM90, SM100, SM103, SM120, SM121) for use with the **Surogate** LLM training framework.

## When This Skill Applies

Use this skill when:

- Writing new CUDA kernels
- Optimizing existing kernels for SM89, SM90, SM100, SM103, SM120, SM121
- Implementing custom attention, normalization, or activation layers
- Debugging kernel performance issues on NVIDIA GPUs

## GPU Architecture Reference

### sm89: L40S

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 142        | Near-full AD102, grid multiples of 142   |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Moderate tile sizes                      |
| L2 Cache      | 96 MB      | Large L2 for datacenter workloads        |
| Memory BW     | 864 GB/s   | 48 GB GDDR6 ECC, 384-bit bus             |
| Warp Size     | 32         | All reductions use warp shuffles         |

### sm89: RTX 4070

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 46         | AD104 die, grid multiples of 46          |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Same SM architecture as all sm89         |
| L2 Cache      | 36 MB      | Smallest L2 of sm89 targets              |
| Memory BW     | 504 GB/s   | 12 GB GDDR6X, 192-bit bus                |
| Warp Size     | 32         | All reductions use warp shuffles         |

### sm89: RTX 4090

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 128        | AD102 die, grid multiples of 128         |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Same SM architecture as all sm89         |
| L2 Cache      | 72 MB      | Good reuse across blocks                 |
| Memory BW     | 1.01 TB/s  | 24 GB GDDR6X, 384-bit bus                |
| Warp Size     | 32         | All reductions use warp shuffles         |

### sm89: RTX 6000 Ada

| Spec          | Value      | Optimization Impact                      |
| ------------- | ---------- | ---------------------------------------- |
| SMs           | 142        | Near-full AD102, grid multiples of 142   |
| Threads/SM    | 1536       | 48 warps/SM, max 24 blocks per SM        |
| Shared Memory | 100 KB/SM  | Same SM architecture as all sm89         |
| L2 Cache      | 96 MB      | Large L2 for workstation workloads       |
| Memory BW     | 960 GB/s   | 48 GB GDDR6, 384-bit bus                 |
| Warp Size     | 32         | All reductions use warp shuffles         |


### sm90: H100

| Spec          | Value     | Optimization Impact                   |
| ------------- | --------- | ------------------------------------- |
| SMs           | 132       | Grid sizing: aim for multiples of 132 |
| Threads/SM    | 2048      | Max 16 blocks of 128 threads per SM   |
| Shared Memory | 192 KB/SM | Large tiles possible                  |
| L2 Cache      | 50 MB     | Reuse across blocks                   |
| Memory BW     | 3.35 TB/s | Coalesced access critical             |
| Warp Size     | 32        | All reductions use warp shuffles      |

### sm90: H200

| Spec          | Value     | Optimization Impact                 |
| ------------- | --------- | ----------------------------------- |
| SMs           | 132       | Same GH100 die as H100              |
| Threads/SM    | 2048      | Max 16 blocks of 128 threads per SM |
| Shared Memory | 192 KB/SM | Same SM architecture as H100        |
| L2 Cache      | 50 MB     | Same as H100                        |
| Memory BW     | 4.8 TB/s  | 43% more BW than H100 via HBM3e     |
| Warp Size     | 32        | All reductions use warp shuffles    |

### sm100: B200

| Spec          | Value     | Optimization Impact                        |
| ------------- | --------- | ------------------------------------------ |
| SMs           | 148       | Dual-die (74/die), grid multiples of 148   |
| Threads/SM    | 2048      | 64 warps/SM (up from 48 on Hopper)         |
| Shared Memory | 228 KB/SM | Larger tiles than Hopper                   |
| L2 Cache      | 126 MB    | 2.5× H100, significant reuse potential     |
| Memory BW     | 8 TB/s    | 2.4× H100, coalesced access still critical |
| Warp Size     | 32        | All reductions use warp shuffles           |

### sm100: GB200

| Spec          | Value     | Optimization Impact              |
| ------------- | --------- | -------------------------------- |
| SMs           | 148       | Same B200 die + Grace CPU        |
| Threads/SM    | 2048      | 64 warps/SM                      |
| Shared Memory | 228 KB/SM | Same SM architecture as B200     |
| L2 Cache      | 126 MB    | Same as B200                     |
| Memory BW     | 8 TB/s    | NVLink 5.0: 1.8 TB/s inter-GPU   |
| Warp Size     | 32        | All reductions use warp shuffles |

### sm103: B300

| Spec          | Value     | Optimization Impact                   |
| ------------- | --------- | ------------------------------------- |
| SMs           | 160       | Grid sizing: aim for multiples of 160 |
| Threads/SM    | 2048      | 64 warps/SM, same as sm100            |
| Shared Memory | 228 KB/SM | Same SM architecture as B200          |
| L2 Cache      | 192 MB    | 1.5× B200, massive reuse potential    |
| Memory BW     | 8 TB/s    | 288 GB HBM3e (12-high stacks)         |
| Warp Size     | 32        | All reductions use warp shuffles      |

### sm103: GB300

| Spec          | Value     | Optimization Impact              |
| ------------- | --------- | -------------------------------- |
| SMs           | 160       | Same B300 die + Grace CPU        |
| Threads/SM    | 2048      | 64 warps/SM                      |
| Shared Memory | 228 KB/SM | Same SM architecture as B300     |
| L2 Cache      | 192 MB    | Same as B300                     |
| Memory BW     | 8 TB/s    | NVLink 5.0: 1.8 TB/s inter-GPU   |
| Warp Size     | 32        | All reductions use warp shuffles |

### sm120: RTX PRO 6000

| Spec          | Value     | Optimization Impact                   |
| ------------- | --------- | ------------------------------------- |
| SMs           | 188       | Full GB202 die, grid multiples of 188 |
| Threads/SM    | 1536      | 48 warps/SM (fewer than datacenter)   |
| Shared Memory | 128 KB/SM | Smaller than sm100, adjust tile sizes |
| L2 Cache      | 128 MB    | Large L2 for workstation workloads    |
| Memory BW     | 1.8 TB/s  | 96 GB GDDR7, 512-bit bus              |
| Warp Size     | 32        | All reductions use warp shuffles      |

### sm120: RTX 5090

| Spec          | Value     | Optimization Impact                 |
| ------------- | --------- | ----------------------------------- |
| SMs           | 170       | 170/192 SMs enabled on GB202        |
| Threads/SM    | 1536      | 48 warps/SM (fewer than datacenter) |
| Shared Memory | 128 KB/SM | Same SM as RTX PRO 6000             |
| L2 Cache      | 96 MB     | Reduced vs full GB202               |
| Memory BW     | 1.79 TB/s | 32 GB GDDR7, 512-bit bus            |
| Warp Size     | 32        | All reductions use warp shuffles    |

### sm121: DGX Spark

| Spec          | Value     | Optimization Impact                     |
| ------------- | --------- | --------------------------------------- |
| SMs           | 48        | Small grid, optimize per-SM occupancy   |
| Threads/SM    | 1536      | 48 warps/SM                             |
| Shared Memory | 100 KB/SM | Smallest of all targets, minimize tiles |
| L2 Cache      | 24 MB     | Limited reuse, prefer shared memory     |
| Memory BW     | 273 GB/s  | LPDDR5X shared with CPU, BW-constrained |
| Warp Size     | 32        | All reductions use warp shuffles        |

> See detailed guides:

- [sm89](references/sm89-optimization-guide.md)
- [sm90](references/sm90-optimization-guide.md)
- [sm100](references/sm100-optimization-guide.md)
- [sm103](references/sm103-optimization-guide.md)
- [sm120](references/sm120-optimization-guide.md)
- [sm121](references/sm121-optimization-guide.md)

## Core Kernel Patterns

1. Parallelism First: Design algorithms for thousands of concurrent threads; serial thinking is the primary enemy of GPU performance
2. Memory Hierarchy Awareness: Global memory is 100x slower than shared memory and 1000x slower than registers; every kernel design starts with memory access planning
3. Coalesced Access: Adjacent threads must access adjacent memory addresses; a single misaligned access pattern can reduce bandwidth by 32x
4. Occupancy Over Cleverness: Maximize active warps per SM by managing register count, shared memory usage, and block dimensions together
5. Minimize Host-Device Transfers: PCIe bandwidth is the bottleneck; overlap transfers with computation using streams and pinned memory

### Kernel Code Generation

- Block size must be a multiple of warp size (32); prefer 128, 256, or 512
- Calculate grid size as `(n + block_size - 1) / block_size`
- Always include bounds checking: `if (idx < n)` at the top of every kernel
- Use grid-stride loops for kernels that must handle arbitrary data sizes
- Document thread mapping: which dimension maps to which data axis
- Mark device-only helpers as `__device__`, host+device as `__host__ __device__`

```cuda
// Grid-stride loop: works with any grid size, any data size
__global__ void saxpy(float a, const float* x, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}
```

Generate properly structured CUDA kernels:

```cuda
// Thread indexing patterns
__global__ void kernel1D(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel2D(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel3D(float* data, int dimX, int dimY, int dimZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < dimX && y < dimY && z < dimZ) {
        int idx = z * dimX * dimY + y * dimX + x;
        data[idx] = data[idx] * 2.0f;
    }
}
```
### Launch Configuration
Calculate optimal launch parameters:

```cpp
// Launch configuration helper
void launchKernel(float* d_data, int n) {
    int blockSize = 256;  // Common optimal block size
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Limit blocks to device maximum
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    numBlocks = min(numBlocks, props.maxGridSize[0]);

    kernel1D<<<numBlocks, blockSize>>>(d_data, n);
}

// Query optimal block size
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel1D, 0, 0);
```

### PTX/SASS Analysis
Analyze generated assembly:

```bash
# Generate PTX
nvcc -ptx -o program.ptx program.cu

# View PTX
cat program.ptx

# Generate SASS (device assembly)
cuobjdump -sass program > program.sass

# Analyze register usage
nvcc --ptxas-options=-v program.cu 2>&1 | grep -E "registers|memory"

# Dump detailed resource usage
cuobjdump --dump-resource-usage program
```

### Memory Management

- Pair every `cudaMalloc` with a `cudaFree`; prefer RAII wrappers in C++ host code
- Use `cudaMallocManaged` (Unified Memory) for prototyping; switch to explicit transfers for production
- Use `cudaMallocHost` (pinned memory) when streaming data to the GPU; pageable memory cannot overlap with compute
- Prefer `cudaMemcpyAsync` with streams over synchronous `cudaMemcpy`
- Never access device pointers from host code or host pointers from device code (except Unified Memory)
- Call `cudaMemset` or `cudaMemsetAsync` to zero-initialize device buffers

Generate proper memory management code:

```cpp
// Host-device memory transfer pattern
void processData(float* h_input, float* h_output, int n) {
    float *d_input, *d_output;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    processKernel<<<numBlocks, blockSize>>>(d_input, d_output, n);

    // Copy output to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Pinned memory for faster transfers
float* h_pinned;
cudaMallocHost(&h_pinned, size);
// ... use h_pinned ...
cudaFreeHost(h_pinned);
```

#### Coalesced Memory Access

```cuda
// BAD: Strided access -- adjacent threads access non-adjacent memory
// Each warp issues 32 separate memory transactions
__global__ void transpose_naive(const float* in, float* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        out[x * H + y] = in[y * W + x];  // Write is strided
    }
}

// GOOD: Use shared memory to coalesce both reads and writes
__global__ void transpose_coalesced(
    const float* in, float* out, int W, int H
) {
    __shared__ float tile[32][33]; // +1 padding avoids bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < W && y < H) {
        tile[threadIdx.y][threadIdx.x] = in[y * W + x]; // Coalesced read
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < H && y < W) {
        out[y * H + x] = tile[threadIdx.x][threadIdx.y]; // Coalesced write
    }
}
```

#### Shared Memory Tiling

```cuda
// Dot product of two vectors using shared memory reduction
__global__ void dot_product(
    const float* a, const float* b, float* result, int n
) {
    __shared__ float cache[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes its partial sum via grid-stride
    float partial = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        partial += a[i] * b[i];
    }
    cache[tid] = partial;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}
```

### Synchronization

- Use `__syncthreads()` after every shared memory write before any thread reads another thread's value
- Never place `__syncthreads()` inside a conditional branch that not all threads in a block will reach (deadlock)
- Use `__syncwarp()` (CUDA 9+) for warp-level synchronization instead of relying on implicit warp-synchronous execution
- Use `cudaDeviceSynchronize()` sparingly in production; prefer stream synchronization with `cudaStreamSynchronize()`
- Use CUDA events (`cudaEventRecord` / `cudaEventSynchronize`) for fine-grained inter-stream ordering


### Error Handling
Comprehensive error checking:

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

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

// Check kernel errors
myKernel<<<blocks, threads>>>(d_data, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### Compute Capability Support
Target specific GPU architectures:

```
# SM versions and features
# sm_89 - Ada Lovelace
# sm_90 - Hopper (transformer engine, TMA)
# sm_100 - Blackwell (larger shared memory, more threads/SM)
# sm_103 - B300 (massive L2 cache, 160 SMs)
# sm_120 - RTX PRO 6000 (full GB202 die)
# sm_121 - DGX Spark (LPDDR5X shared memory)

# Compile for specific capability
nvcc -arch=sm_80 -code=sm_80 program.cu

# Fat binary for multiple architectures
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 \
     -o program program.cu
```

### Launch Bounds Validation
Validate resource constraints:

```cpp
// Specify launch bounds for occupancy
__global__ void __launch_bounds__(256, 4)
boundedKernel(float* data, int n) {
    // Kernel limited to 256 threads, compiler targets 4 blocks/SM
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

// Query and validate resources
void validateLaunch() {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, boundedKernel);

    printf("Registers: %d\n", attr.numRegs);
    printf("Shared memory: %zu bytes\n", attr.sharedSizeBytes);
    printf("Max threads per block: %d\n", attr.maxThreadsPerBlock);
}
```


### Performance

- Profile before optimizing: use Nsight Compute for kernel analysis, Nsight Systems for system-level view
- Target >50% theoretical occupancy; use the CUDA Occupancy Calculator to tune block dimensions
- Aim for >60% of peak memory bandwidth in memory-bound kernels
- Avoid warp divergence: ensure threads within a warp take the same branch when possible
- Prefer `float` over `double` on consumer GPUs (2x throughput difference)
- Minimize atomic operations on global memory; use shared memory atomics with a final reduction


#### Warp-Level Primitives

```cuda
// Warp-level reduction using shuffle instructions -- no shared memory needed
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction combining warp shuffles and shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32]; // One slot per warp (max 32 warps/block)

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the warp sums
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}
```

### Occupancy Calculator

```cuda
// Query occupancy at compile time for tuning
void report_occupancy() {
    int block_size = 256;
    int num_blocks;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, my_kernel, block_size, 0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int active_warps = num_blocks * (block_size / prop.warpSize);
    int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy = (float)active_warps / max_warps;

    printf("Occupancy: %.1f%% (%d/%d warps)\n",
           occupancy * 100, active_warps, max_warps);
}
```


### Memory Bandwidth Measurement

```cuda
// Measure effective bandwidth of a kernel
void measure_bandwidth(int n) {
    size_t bytes = 2 * n * sizeof(float); // Read A + Write B

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    copy_kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float gb_per_sec = bytes / (ms * 1e6);
    printf("Effective bandwidth: %.2f GB/s\n", gb_per_sec);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
```

### Vectorized Memory Access (Critical for Performance)

**BFloat16 vectorization using `__nv_bfloat162`:**

```cuda
// Load 2 bfloat16 elements at once (32-bit load)
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

### Warp Shuffle Reductions

```cuda
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### Block Sizes for Attention

- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 64`, `BLOCK_SIZE_K = 64`
- `NUM_WARPS = 8`

### Thread Configuration

For element-wise ops (RoPE, GEGLU):

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

## Supported Data Types

All kernels support three precision modes:

- `__half` (FP16) - Default for inference
- `__nv_bfloat16` (BF16) - Preferred for training
- `float` (FP32) - Reference/debugging


## Memory Hierarchy

Understanding the memory hierarchy is the single most important factor in CUDA performance.

| Memory Type | Scope | Latency (cycles) | Size | Cached | Read/Write |
|-------------|-------|-------------------|------|--------|------------|
| Registers | Thread | 1 | ~255 per thread | N/A | R/W |
| Shared | Block | ~5 | 48-164 KB per SM | N/A | R/W |
| L1 Cache | SM | ~28 | 48-192 KB per SM | Auto | R |
| L2 Cache | Device | ~200 | 4-40 MB | Auto | R/W |
| Global | Device | ~400-600 | 4-80 GB (HBM/GDDR) | Yes | R/W |
| Constant | Device | ~5 (cached) | 64 KB | Yes (broadcast) | R |
| Texture | Device | ~400 (cached) | Global pool | Yes (spatial) | R |

**Decision guide:**
- Data reused within a thread -> registers (automatic via local variables)
- Data shared across threads in a block -> `__shared__` memory
- Read-only data broadcast to all threads -> `__constant__` memory
- Large read-only data with spatial locality -> texture memory
- Everything else -> global memory with coalesced access patterns


## See Also

- [troubleshooting.md](./references/troubleshooting.md) - Common issues and solutions
- [kernel-templates.md](./references/kernel-templates.md) - Complete kernel templates

### External Resources

- https://github.com/huggingface/kernels-community
