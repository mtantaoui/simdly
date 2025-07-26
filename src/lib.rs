//! # simdly
//!
//! 🚀 A high-performance Rust library that leverages SIMD (Single Instruction, Multiple Data)
//! instructions for fast vectorized computations. This library provides efficient implementations
//! of mathematical operations using modern CPU features.
//!
//! ## Features
//!
//! - **SIMD Optimized**: Leverages AVX2 instructions for 256-bit vector operations
//! - **Memory Efficient**: Supports both aligned and unaligned memory access patterns
//! - **Generic Traits**: Provides consistent interfaces across different SIMD implementations
//! - **Safe Abstractions**: Wraps unsafe SIMD operations in safe, ergonomic APIs
//! - **Performance**: Optimized for high-throughput numerical computations
//!
//! ## Architecture Support
//!
//! Currently supports:
//! - **x86/x86_64** with AVX2 (256-bit vectors)
//!
//! Future support planned for:
//! - SSE (128-bit vectors for older x86 processors)
//! - ARM NEON (128-bit vectors for ARM/AArch64)
//!
//! ## Usage
//!
//! The library provides traits for SIMD operations that automatically detect and use
//! the best available instruction set on the target CPU.
//!
//!
//! ## Performance Considerations
//!
//! - **Memory Alignment**: Use aligned memory when possible for optimal performance
//! - **Batch Processing**: Process data in chunks that match SIMD vector sizes
//! - **CPU Features**: Enable appropriate target features during compilation

pub mod simd;
