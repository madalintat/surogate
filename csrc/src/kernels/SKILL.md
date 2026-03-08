---
name: cuda-kernels
description: "Provides guidance for writing and benchmarking optimized CUDA kernels for NVIDIA GPUs (H100, A100, T4) targeting the Surogate framework. Includes benchmarking scripts to compare kernel performance against baseline implementations."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: attention, rmsnorm, rope, adaln, geglu, benchmark, kernels, get_kernel"
---
# CUDA Kernels for Surogate
This skill provides patterns and guidance for developing optimized CUDA kernels targeting NVIDIA GPUs (H100, A100, T4) for use with the **Surogate** LLM training framework.

## When This Skill Applies
Use this skill when:
- Writing new CUDA kernels
- Optimizing existing kernels for H100, A100
- Implementing custom attention, normalization, or activation layers
- Debugging kernel performance issues on NVIDIA GPUs

## GPU Architecture Reference

### H100 (Hopper)

| Spec | Value | Optimization Impact |
|------|-------|---------------------|
| SMs | 132 | Grid sizing: aim for multiples of 132 |
| Threads/SM | 2048 | Max 16 blocks of 128 threads per SM |
| Shared Memory | 192 KB/SM | Large tiles possible |
| L2 Cache | 50 MB | Reuse across blocks |
| Memory BW | 3.35 TB/s | Coalesced access critical |
| Warp Size | 32 | All reductions use warp shuffles |