---
name: cute-dsl
description: Write or modify CuTe DSL GPU kernels for the Surogate framework. Use when the user wants to create CUDA kernels using CuTe DSL/Python (e.g., "write a CuTe kernel", "implement a fused kernel", "optimize kernel with CuTe DSL"). Guides through kernel implementation, compilation, and integration with Surogate's JIT pipeline.
argument-hint: [kernel-description]
---

# Write a CuTe DSL GPU Kernel for Surogate

You are implementing a GPU kernel using NVIDIA's CuTe DSL (Python) for the Surogate training framework.
The kernel request is: **$ARGUMENTS**

## Important: Read API Reference First

Before writing any CuTe DSL code, read the relevant API documentation files to understand available types, functions, and patterns. These are generated Sphinx docs at `cute/_build/md/`:

**Core API** (always read these):
- `.claude/skills/cute-dsl/cute.md` — Core CuTe DSL types: `Tensor`, `Layout`, `Atom`, `TiledMma`, `TiledCopy`, address spaces, copy/mma operations, tensor manipulation (`make_tensor`, `make_layout`, `local_partition`, `local_tile`), synchronization (`syncthreads`, `cluster_arrive`, `cluster_wait`), arithmetic helpers
- `.claude/skills/cute-dsl/cute_runtime.md` — Runtime utilities: `from_dlpack`, `make_ptr`, tensor conversion from PyTorch/NumPy
- `.claude/skills/cute-dsl/utils.md` — Utility functions: `HardwareInfo`, shared memory layout helpers

**Architecture-specific APIs** (read based on target GPU):
- `.claude/skills/cute-dsl/cute_nvgpu.md` — GPU architecture operations index
- `.claude/skills/cute-dsl/cute_nvgpu_tcgen05.md` — Blackwell (SM100+) tcgen05 MMA operations, TMA copy operations, accumulator fields
- `.claude/skills/cute-dsl/cute_nvgpu_warpgroup.md` — Hopper (SM90) warpgroup MMA operations
- `.claude/skills/cute-dsl/cute_nvgpu_warp.md` — Ampere/Ada (SM80-89) warp-level MMA operations
- `.claude/skills/cute-dsl/cute_nvgpu_cpasync.md` — Async copy operations (cp.async)
- `.claude/skills/cute-dsl/cute_arch.md` — Low-level architecture primitives

**SM100+ specific utilities**:
- `.claude/skills/cute-dsl/utils_sm100.md` — SM100 (Blackwell) layout helpers, TMA descriptors
- `.claude/skills/cute-dsl/utils_sm90.md` — SM90 (Hopper) layout helpers

**General documentation** (for concepts and patterns):
- `cute/_build/md/cute_dsl_general/dsl_introduction.md` — Overview of CuTe DSL hybrid compilation model
- `cute/_build/md/cute_dsl_general/dsl_code_generation.md` — How code generation works (AST rewrite + tracing)
- `cute/_build/md/cute_dsl_general/dsl_control_flow.md` — Control flow: `range`, `cutlass.range`, `cutlass.range_constexpr`, `if/else`, `while`
- `cute/_build/md/cute_dsl_general/dsl_dynamic_layout.md` — Static vs dynamic layouts, `mark_layout_dynamic`, `mark_compact_shape_dynamic`
- `cute/_build/md/cute_dsl_general/dsl_jit_arg_generation.md` — JIT function argument types, `Constexpr`, type safety
- `cute/_build/md/cute_dsl_general/dsl_jit_caching.md` — JIT caching, `cute.compile`, zero-compile executor
- `cute/_build/md/cute_dsl_general/dsl_jit_compilation_options.md` — Compilation options: opt-level, keep-cubin, keep-ptx, assertions
- `cute/_build/md/cute_dsl_general/framework_integration.md` — PyTorch/DLPack interop, bypassing DLPack
- `cute/_build/md/cute_dsl_general/dsl_ahead_of_time_compilation.md` — AOT compilation, `export_to_c`, C++ integration
- `cute/_build/md/cute_dsl_general/autotuning_gemm.md` — Auto-tuning patterns for GEMM kernels
- `cute/_build/md/cute_dsl_general/debugging.md` — Debugging: `print()` vs `cute.printf()`, IR dump, PTX dump, compute-sanitizer

## CuTe DSL Fundamentals

### Decorators
CuTe DSL provides two main decorators:

1. `@cute.jit` — Host-side JIT-compiled functions (called from Python)
2. `@cute.kernel` — GPU kernel functions (called from `@jit` functions)

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def my_kernel(tensor_a: cute.Tensor, tensor_b: cute.Tensor):
    # GPU code here
    ...

@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, stream: cute.cuda.CUstream):
    my_kernel(a, b).launch(grid=[G, 1, 1], block=[B, 1, 1], stream=stream)
```

### Calling Conventions
| Caller | Callee | Allowed |
|--------|--------|---------|
| Python | `@jit` | Yes |
| Python | `@kernel` | No |
| `@jit` | `@jit` | Yes (inlined) |
| `@jit` | `@kernel` | Yes (GPU launch) |
| `@kernel` | `@jit` | Yes (inlined) |
| `@kernel` | `@kernel` | No |

### Meta-Stage vs Runtime
- `print()` executes at **compile time** (meta-stage) — use to inspect shapes, layouts
- `cute.printf()` executes at **runtime on GPU** — use to see actual tensor values
- `cutlass.Constexpr` parameters are compile-time constants, baked into the kernel

### Control Flow
```python
# Compile-time loop (fully unrolled)
for i in cutlass.range_constexpr(n):  # n must be known at compile time
    ...

# Runtime loop (emitted as IR loop)
for i in range(bound):  # works with dynamic values
    ...

# Runtime loop with unrolling hint
for i in cutlass.range(bound, unroll=4):
    ...

# Software pipelining (SM90+, experimental)
for i in cutlass.range(bound, prefetch_stages=2):
    cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
    use(buffer[i % total_stages])

# Compile-time branch
if cutlass.const_expr(compile_time_value):
    ...

# Runtime branch
if dynamic_value > 0:
    ...
```

**Limitations**: No `break`, `continue`, `return`, or `raise` inside dynamic control flow bodies. Variables created inside dynamic control flow are not accessible outside.

### Type System
- **Dynamic arguments** (default): `cutlass.Int32`, `cutlass.Float32`, `cute.Tensor`, etc.
- **Static arguments**: `cutlass.Constexpr` — evaluated at compile time, not in kernel signature
- **Type annotations** enable compile-time validation

## Surogate JIT Kernel Integration

Surogate uses a manifest-driven JIT pipeline:
1. Python compiles kernel -> writes cubin + JSON manifest
2. C++ loads via `JitKernel::load_manifest()`

### Key files:
- JIT kernel loader: `csrc/src/runtime/jit/jit_kernel.{h,cpp}`
- Kernel compiler: `surogate/kernels/compiler.py`
- Triton RMSNorm: `surogate/kernels/rmsnorm.py`
- CuTe DSL RMSNorm: `surogate/kernels/cute_rmsnorm.py` (reference implementation)

### Two JIT Backends
- **Triton**: Uses `cuModuleLoadData` to load cubins
- **CuTe DSL / quack** (Blackwell SM120+): MUST use `cuLibraryLoadData` (Library API)

### Compilation Pattern

Follow the pattern in `surogate/kernels/cute_rmsnorm.py`:

```python
from surogate.kernels.compiler import compile_cute_kernel

def compile_my_kernel(
    # kernel-specific parameters
    C: int,
    output_dir: str = ".",
    dtype: str = "bf16",
) -> str:
    """Compile kernel and return path to JSON manifest."""
    import cutlass
    import cutlass.cute as cute
    from quack.compile_utils import make_fake_tensor as fake_tensor

    # 1. Build the kernel object
    kernel = MyKernel(dtype, C)

    # 2. Create fake tensors for compilation (symbolic shapes)
    batch_sym = cute.sym_int()
    x = fake_tensor(cutlass.BFloat16, (batch_sym, C), divisibility)

    # 3. Build compile args tuple
    compile_args = (x, ..., stream)

    # 4. Define parameter layout for C++ launcher
    params = [
        {
            "name": "mX",
            "type": "tensor_2d",
            "dtype": dtype,
            "size_bytes": 24,
            "fields": [
                {"name": "data_ptr", "offset": 0, "type": "ptr"},
                {"name": "M", "offset": 8, "type": "int32"},
                {"name": "stride_row", "offset": 16, "type": "int64"},
            ],
        },
        # ... more params
    ]

    # 5. Compile and return manifest path
    return compile_cute_kernel(
        kernel,
        compile_args,
        output_dir=output_dir,
        base_name=f"my_kernel_{dtype}_C{C}",
        shared_mem=compute_dynamic_smem(...),
        params=params,
        library="quack",  # or "cutlass"
        **extra_metadata,
    )
```

### Parameter Layout Types
The JSON manifest describes parameter layout for the C++ launcher:

| Type | Description | Fields |
|------|-------------|--------|
| `tensor_1d` | 1D tensor (e.g., weight vector) | `data_ptr` |
| `tensor_2d` | 2D tensor with dynamic first dim | `data_ptr`, `M` (int32), `stride_row` (int64) |
| `scalar` | Scalar value (e.g., epsilon) | value directly |

Field types: `ptr` (void*), `int32`, `int64`, `fp32`

## Writing CuTe DSL Kernels

### Step 1: Understand the Algorithm
- What are the inputs/outputs? Shapes, dtypes, memory layout?
- What GPU operations are needed? (GEMM, reduction, elementwise, etc.)
- What is the target GPU architecture? (SM80, SM89, SM90, SM100, SM120)

### Step 2: Choose Kernel Architecture
- **Elementwise**: Simple grid mapping, one element per thread
- **Reduction**: Block-level reduction with shared memory
- **Tiled GEMM**: TiledMma + TiledCopy, shared memory staging
- **Fused kernel**: Combine multiple operations to minimize memory traffic

### Step 3: Implement the Kernel

Example: Simple elementwise kernel
```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    tid = cute.arch.thread_idx_x()
    bid = cute.arch.block_idx_x()
    idx = bid * cute.arch.block_dim_x() + tid
    if idx < cute.size(mA):
        mC[idx] = mA[idx] + mB[idx]

@cute.jit
def launch_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cute.cuda.CUstream):
    n = cute.size(a)
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    elementwise_add(a, b, c).launch(
        grid=[grid_size, 1, 1],
        block=[block_size, 1, 1],
        stream=stream,
    )
```

### Step 4: Handle Shared Memory (if needed)
```python
@cute.kernel
def reduction_kernel(mX: cute.Tensor, mO: cute.Tensor):
    # Allocate shared memory
    smem = cute.smem_allocator()
    shared_buf = smem.alloc(cute.make_layout((256,)), cutlass.Float32)
    # ... use shared_buf for block-level reduction
    cute.syncthreads()
```

### Step 5: Framework Integration
```python
import torch
from cutlass.cute.runtime import from_dlpack

# Static layout (fixed shape, best codegen)
a_cute = from_dlpack(torch_tensor)

# Dynamic layout (variable shapes, single compilation)
a_cute = from_dlpack(torch_tensor).mark_layout_dynamic()

# Fine-grained dynamic control
a_cute = from_dlpack(torch_tensor).mark_compact_shape_dynamic(mode=0, divisibility=16)
```

### Step 6: Compile and Test
```python
# Explicit compilation
compiled = cute.compile(launch_add, a, b, c, stream)
compiled(a, b, c, stream)  # Execute without recompilation

# Compilation options
compiled = cute.compile(launch_add, a, b, c, stream, options="--opt-level 3 --keep-ptx")

# Pythonic options
from cutlass.cute import OptLevel, KeepPTX
compiled = cute.compile[OptLevel(3), KeepPTX](launch_add, a, b, c, stream)
```

### Step 7: Debug
```python
# Compile-time inspection
print(tensor.layout)  # Shows layout at meta-stage

# Runtime GPU debugging
cute.printf("value = %f\n", tensor[0])

# Environment variables for debugging:
# CUTE_DSL_PRINT_IR=1     — dump generated IR
# CUTE_DSL_KEEP_PTX=1     — save PTX to file
# CUTE_DSL_KEEP_CUBIN=1   — save cubin to file
# CUTE_DSL_LOG_TO_CONSOLE=1 — enable console logging
# CUTE_DSL_LINEINFO=1     — source correlation for profiling

# Programmatic access to compiled artifacts
compiled = cute.compile(my_func, ...)
print(compiled.__ptx__)    # PTX source
print(compiled.__mlir__)   # MLIR IR
```

## AOT Compilation for C++ Integration

```python
# Export to C header + object file
compiled = cute.compile(my_func, ...)
compiled.export_to_c(
    file_path="./artifacts",
    file_name="my_kernel",
    function_prefix="my_kernel"
)
# Generates: my_kernel.h + my_kernel.o
```

## Reference: Existing Kernel Examples

| Kernel | File | Pattern |
|--------|------|---------|
| CuTe RMSNorm | `surogate/kernels/cute_rmsnorm.py` | quack wrapper, manifest-driven |
| Triton RMSNorm | `surogate/kernels/rmsnorm.py` | Triton backend, cubin compilation |
| Gated Delta Rule | `surogate/kernels/gated_delta_rule.py` | Custom CUDA kernel |

## SM-Specific Optimizations and Limitations for CuTe DSL

When writing CuTe DSL kernels, the target GPU architecture determines which APIs, tile sizes, and strategies to use. Always identify the target SM before choosing kernel patterns.

### Quick Reference: Feature Availability

| Feature | sm89 (Ada) | sm90 (Hopper) | sm100/103 (Blackwell DC) | sm120 (Desktop BW) | sm121 (DGX Spark) |
|---------|:----------:|:-------------:|:------------------------:|:-------------------:|:-----------------:|
| **CuTe DSL API** | `cute_nvgpu_warp` | `cute_nvgpu_warpgroup` | `cute_nvgpu_tcgen05` | `cute_nvgpu_warp` | `cute_nvgpu_warp` |
| TMA | No | Yes (v1) | Yes (v2, multicast) | No | No |
| Thread Block Clusters | No | Yes | Yes (enhanced) | No | No |
| Distributed Shared Memory | No | Yes | Yes (enhanced) | No | No |
| TMEM (Tensor Memory) | No | No | Yes | No | No |
| WGMMA / tcgen05 | No | wgmma.mma_async | tcgen05 WGMMA | No | No |
| cp.async | Yes | Yes | Yes | Yes | Yes |
| FP4 Tensor Cores | No | No | Yes (NVFP4) | Yes | Yes |
| FP8 Tensor Cores | Yes (4th gen) | Yes (4th gen) | Yes (5th gen, 2x) | Yes (5th gen) | Yes (5th gen) |
| Shared Memory/SM | 100 KB | 192 KB | 228 KB | 128 KB | 100 KB |
| Max Warps/SM | 48 | 64 | 64 | 48 | 48 |
| Max Blocks/SM | 24 | 16 | 32 | 24 | 24 |
| Max Threads/SM | 1536 | 2048 | 2048 | 1536 | 1536 |

### sm89 — Ada Lovelace (L40S, RTX 4070/4090/6000 Ada)

**CuTe DSL API**: Use `cute_nvgpu_warp` for warp-level MMA, `cute_nvgpu_cpasync` for async copies.

**Optimizations**:
- Large L2 cache (36-96 MB) compensates for limited GDDR bandwidth (504 GB/s–1.01 TB/s) — rely on L2 for inter-tile reuse
- Dual-issue FP32 (128 FP32 cores/SM) — FP32 compute is "free" relative to memory; fuse extra FP32 ops aggressively
- Up to 24 blocks/SM — small blocks (64-128 threads) achieve 100% occupancy
- Generous register budget (255/thread) with fewer warps/SM — each warp gets more registers without hurting occupancy

**Limitations**:
- **100 KB shared memory max** — smallest tile budgets of supported architectures (except sm121); must use smaller tiles than Hopper/Blackwell
- **No TMA** — must use `cp.async` for async loads, manual address computation per thread
- **No clusters/DSMEM** — each block is independent, no cross-block shared memory
- **No WGMMA/tcgen05** — use WMMA-style warp MMA (16x16x16 for FP16)
- **No FP4** — limited to FP8/FP16/BF16/TF32 tensor core operations
- **Avoid 1024-thread blocks** — caps occupancy at 67% (only 1 block fits per SM)
- **Memory BW bottleneck** — aggressive kernel fusion is critical, vectorize all loads (`__nv_bfloat162`, `float4`)

**Tile sizing guidance**:
```
Attention: Q=64x64, K=64x64, V=64x64 (FP16) → ~24 KB, fits with double-buffering
GEMM: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 → A=4KB + B=4KB, double-buffer fits easily
Preferred block size: 256 threads (8 warps)
Grid: multiples of SM count (46/128/142 depending on GPU)
```

### sm90 — Hopper (H100, H200)

**CuTe DSL API**: Use `cute_nvgpu_warpgroup` for warpgroup MMA, `utils_sm90` for layout helpers, TMA via CuTe copy operations.

**Optimizations**:
- **192 KB shared memory** — large tiles possible, 2x Ada's budget
- **TMA** — hardware-accelerated bulk global→shared transfers, bypass register file; use CuTe `copy` with TMA atoms
- **Thread Block Clusters** — use `cluster_arrive`/`cluster_wait` from CuTe for cross-block cooperation
- **Distributed Shared Memory** — blocks in a cluster share data without going through global memory
- **Warpgroup MMA** — 4-warp groups (128 threads) for larger MMA tiles (64x128x16 FP16)
- **64 warps/SM** — excellent latency hiding
- **Software pipelining** — use `cutlass.range(bound, prefetch_stages=N)` for TMA + MMA overlap

**Limitations**:
- **No TMEM** — tensor operands consume registers (unlike sm100); watch register pressure with large GEMM tiles
- **Max 16 blocks/SM** — fewer blocks than Ada (24) or Blackwell (32); large blocks are favored
- **50 MB L2** — smaller than Ada/Blackwell desktop; less L2 reuse for KV caches
- **Memory BW varies**: H100=3.35 TB/s, H200=4.8 TB/s — still less than Blackwell's 8 TB/s

**Tile sizing guidance**:
```
Attention: Q=128x64, K=128x64 (FP16) → 16 KB each, fits well in 192 KB
GEMM: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 with double-buffering
Warpgroup MMA unit: 128 threads (4 warps)
Preferred block size: 128-256 threads
Grid: multiples of 132 SMs
```

### sm100 — Blackwell Datacenter (B200, GB200)

**CuTe DSL API**: Use `cute_nvgpu_tcgen05` for tcgen05 MMA, `utils_sm100` for layout helpers and TMA descriptors.

**Optimizations**:
- **228 KB shared memory** — largest tiles of any architecture, 19% more than Hopper
- **tcgen05 WGMMA + TMEM** — tensor operands live in dedicated Tensor Memory (separate from 64K register file), freeing registers for other use; dramatically reduces register pressure for GEMM
- **TMA v2 with multicast** — one block loads a tile, hardware broadcasts to all blocks in a cluster; reduces global memory traffic by cluster_size factor
- **32 blocks/SM** — most flexible block scheduling; all standard block sizes achieve 100% occupancy
- **126 MB L2 cache** — 2.5x Hopper, excellent KV cache residency; use `cudaAccessPropertyPersisting`
- **8 TB/s HBM3e** — 2.4x Hopper bandwidth
- **FP4/FP6 tensor cores** — NVFP4 for inference, 2x K-dimension (64x256x64 shape)
- **Enhanced clusters + DSMEM** — producer-consumer patterns across blocks

**Limitations**:
- **Dual-die** — L2 is split ~63 MB per die; cross-die access has higher latency; keep clusters on one die
- **CuTe DSL cubins MUST use `cuLibraryLoadData`** (Library API), not `cuModuleLoadData`
- **WGMMA requires data in shared memory or TMEM** — cannot operate directly on global memory

**Tile sizing guidance**:
```
Attention: Q=128x128, K=128x128, V=128x128 (FP16) → 32 KB each, 160 KB total with FP32 accum → fits in 228 KB
GEMM: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, double-buffered → 64 KB
WGMMA shapes: FP16=64xNx16, FP8=64xNx32, FP4=64xNx64 (N=64/128/256)
WGMMA unit: 128 threads (4 warps = 1 warp group)
Grid: multiples of 148 SMs; cluster_size=2-4
```

### sm103 — Blackwell Ultra (B300, GB300)

**CuTe DSL API**: Identical to sm100 — use `cute_nvgpu_tcgen05`, `utils_sm100`.

**Optimizations**:
- **192 MB L2 cache** — 1.5x B200; pin larger KV caches, longer sequences fit in L2 entirely
- **160 SMs** — 8% more than B200; more concurrent clusters and blocks
- **288 GB HBM3e** — larger models fit without tensor parallelism
- **PCIe Gen6** — 128 GB/s host-device (2x Gen5); faster CPU offloading
- **1,400W TDP** — sustained high-frequency operation, less thermal throttling on compute-bound kernels

**Limitations**:
- Same SM microarchitecture as sm100 — **no new per-SM features**, only more SMs, more L2, more HBM
- **Grid sizing must target 160 SMs** (not 148) — wave quantization at wrong multiples wastes SMs
- **Dual-die** with ~96 MB L2 per die — same cross-die considerations as sm100

**Tile sizing**: Identical to sm100. Only grid dimensions and L2 budgets change.

### sm120 — Blackwell Desktop (RTX 5090, RTX PRO 6000)

**CuTe DSL API**: Use `cute_nvgpu_warp` for warp-level MMA, `cute_nvgpu_cpasync` for async copies. **Do NOT use** `cute_nvgpu_tcgen05`, `cute_nvgpu_warpgroup`, `utils_sm100`, or `utils_sm90` — these features are absent.

**sm120 is architecturally closer to Ada (sm89) than to datacenter Blackwell (sm100)** at the SM level. Same warp count, block limits, and missing TMA/clusters/TMEM.

**Optimizations**:
- **128 KB shared memory** — 28% more than Ada's 100 KB; slightly larger tiles possible
- **5th-gen tensor cores with FP4** — NVFP4 for inference (new vs Ada), via WMMA or cuBLAS
- **GDDR7** at 1.79-1.8 TB/s — ~2x Ada RTX 4090, higher protocol efficiency
- **Massive L2** — 96 MB (RTX 5090) / 128 MB (PRO 6000); critical performance lever with limited GDDR7 BW
- **Up to 188 SMs** (PRO 6000) — more parallelism than any single-die datacenter GPU

**Limitations**:
- **No TMA** — must use `cp.async`, manual address computation per thread
- **No Thread Block Clusters** — no cluster sync, no distributed shared memory
- **No TMEM** — tensor operands consume registers; WMMA fragments add ~24 regs per MMA
- **No WGMMA/tcgen05** — use WMMA (16x16x16 FP16 tiles), much smaller than sm100's 64xNx16
- **48 warps/SM, 24 blocks/SM max** — same as Ada; **avoid 1024-thread blocks** (67% occupancy cap)
- **1.8 TB/s GDDR7 << 8 TB/s HBM3e** — kernel fusion is the most impactful optimization
- **32 GB VRAM on RTX 5090** — constrains model size; use FP4/FP8 quantization
- **sm100 kernels require significant rework** — must replace TMA→cp.async, WGMMA→WMMA, remove clusters, reduce tiles from 228 KB→128 KB

**Tile sizing guidance**:
```
Attention: Q=64x64, K=64x64, V=64x64 (FP16) → ~24 KB, double-buffer to 48 KB
GEMM: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, double-buffered → 16 KB
WMMA shapes: FP16=16x16x16, FP8=16x16x32
Preferred block size: 256 threads (8 warps)
Grid: multiples of 170 (RTX 5090) or 188 (PRO 6000)
```

### sm121 — DGX Spark

**CuTe DSL API**: Same as sm120 — use `cute_nvgpu_warp`, `cute_nvgpu_cpasync`.

**Optimizations**:
- Same 5th-gen tensor cores as sm120 (FP4/FP8/FP16/BF16)
- Optimize per-SM occupancy aggressively — only 48 SMs available

**Limitations**:
- **100 KB shared memory** — smallest of all targets (tied with sm89); minimize tile sizes
- **48 SMs only** — very small grid; every SM must be fully utilized
- **24 MB L2 cache** — smallest L2; prefer shared memory over relying on L2 reuse
- **273 GB/s LPDDR5X** — severely bandwidth-constrained (shared with CPU); kernel fusion is absolutely critical
- **No TMA, No Clusters, No TMEM, No WGMMA** — same limitations as sm120
- Same occupancy profile as sm89/sm120 (48 warps/SM, avoid 1024-thread blocks)

**Tile sizing guidance**:
```
Keep tiles minimal — same budget as sm89 (100 KB shared)
GEMM: BLOCK_M=64, BLOCK_N=64, BLOCK_K=16 — conservative to fit in 100 KB
Preferred block size: 256 threads
Grid: multiples of 48 SMs
Maximize arithmetic intensity — memory BW is the hard ceiling
```

### Architecture Selection Decision Tree

```
Target GPU known?
├── SM100/103 (B200/B300/GB200/GB300)
│   → Use tcgen05 WGMMA + TMA + clusters + TMEM
│   → Read: cute_nvgpu_tcgen05.md, utils_sm100.md
│   → Max tiles: 228 KB shared
│
├── SM90 (H100/H200)
│   → Use warpgroup MMA + TMA + clusters
│   → Read: cute_nvgpu_warpgroup.md, utils_sm90.md
│   → Max tiles: 192 KB shared
│
├── SM120 (RTX 5090/PRO 6000)
│   → Use warp MMA + cp.async (NO TMA/clusters/TMEM)
│   → Read: cute_nvgpu_warp.md, cute_nvgpu_cpasync.md
│   → Max tiles: 128 KB shared
│   → Fusion is critical (1.8 TB/s GDDR7)
│
├── SM89 (L40S/RTX 4090/4070)
│   → Use warp MMA + cp.async (NO TMA/clusters/TMEM)
│   → Read: cute_nvgpu_warp.md, cute_nvgpu_cpasync.md
│   → Max tiles: 100 KB shared
│   → Fusion is critical (504 GB/s–1.01 TB/s GDDR)
│
└── SM121 (DGX Spark)
    → Same APIs as sm120, but 100 KB smem, 48 SMs, 273 GB/s
    → Smallest grid + lowest BW — maximize per-SM work and fusion
```

## Checklist

Before considering the kernel implementation complete:

- [ ] Read the CuTe DSL intro from `.claude/skills/cute-dsl/intro.md`
- [ ] Read the relevant API docs from `.claude/skills/cute-dsl/` for the types and operations you're using
- [ ] Kernel uses correct decorators (`@cute.kernel` for GPU, `@cute.jit` for host)
- [ ] Dynamic vs static layouts chosen correctly for the use case
- [ ] Shared memory allocation is correct (if needed)
- [ ] Parameter layout manifest matches actual kernel ABI
- [ ] Kernel compiles without errors
- [ ] Tested with representative inputs
- [ ] Integration with Surogate's JIT pipeline (manifest + compile function) if needed
