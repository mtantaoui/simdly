---
title: Introduction
description: Learn about simdly and its capabilities for high-performance computing.
---

# Introduction to simdly

**simdly** is a high-performance Rust library that leverages SIMD (Single Instruction, Multiple Data) instructions for fast vectorized computations. This library provides efficient implementations of mathematical operations using modern CPU features.

## What is SIMD?

SIMD (Single Instruction, Multiple Data) is a parallel computing technique that allows a single instruction to operate on multiple data points simultaneously. This enables significant performance improvements for computationally intensive tasks, especially those involving mathematical operations on arrays or vectors.

## Key Benefits

- **Performance**: Up to 8x faster operations using AVX2 256-bit vectors
- **Memory Efficiency**: Optimized memory access patterns for better cache utilization
- **Safety**: Safe abstractions over unsafe SIMD operations
- **Portability**: Generic traits that work across different SIMD implementations

## Use Cases

simdly is particularly well-suited for:

- **Numerical Computing**: Scientific simulations, mathematical modeling
- **Digital Signal Processing**: Audio/video processing, filtering
- **Machine Learning**: Vector operations, matrix computations
- **Graphics**: 3D transformations, image processing
- **Data Analysis**: Statistical computations, aggregations

## Architecture Support

### Current Support
- **x86/x86_64** with AVX2 (256-bit vectors)

### Planned Support
- **SSE**: 128-bit vectors for older x86 processors
- **ARM NEON**: 128-bit vectors for ARM/AArch64 processors

## Getting Started

Ready to start using simdly? Check out our [Installation Guide](/getting-started/installation/) to add simdly to your project.