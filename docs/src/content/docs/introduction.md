---
title: Introduction
description: Learn about Simdly and its cross-platform SIMD capabilities for high-performance computing.
---

# Introduction to Simdly

**Simdly** is a high-performance cross-platform Rust library that leverages SIMD (Single Instruction, Multiple Data) instructions for fast vectorized computations. This library provides efficient implementations of mathematical operations using modern CPU features across both x86 and ARM architectures.

## What is SIMD?

SIMD (Single Instruction, Multiple Data) is a parallel computing technique that allows a single instruction to operate on multiple data points simultaneously. This enables significant performance improvements for computationally intensive tasks, especially those involving mathematical operations on arrays or vectors.

## Key Benefits

- **Performance**: Up to 8x faster operations using 256-bit vectors (AVX2) and 128-bit vectors (NEON)
- **Cross-Platform**: Supports both x86/x86_64 (AVX2) and ARM (NEON) architectures
- **Memory Efficiency**: Optimized memory access patterns for better cache utilization
- **Safety**: Safe abstractions over unsafe SIMD operations
- **Portability**: Universal API that works across different SIMD implementations

## Use Cases

Simdly is particularly well-suited for:

- **Numerical Computing**: Scientific simulations, mathematical modeling
- **Digital Signal Processing**: Audio/video processing, filtering
- **Machine Learning**: Vector operations, matrix computations
- **Graphics**: 3D transformations, image processing
- **Data Analysis**: Statistical computations, aggregations

## Architecture Support

### Current Support
- **x86/x86_64** with AVX2 (256-bit vectors)
- **ARM/AArch64** with NEON (128-bit vectors)

### Planned Support
- **SSE**: 128-bit vectors for older x86 processors
- **WebAssembly SIMD**: Cross-platform web performance

## Getting Started

Ready to start using Simdly? Check out our [Installation Guide](/getting-started/installation/) to add Simdly to your project.