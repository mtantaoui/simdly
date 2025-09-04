//! ARM NEON SIMD implementations for 128-bit vector operations.
//!
//! This module contains SIMD implementations using ARM's Advanced SIMD (NEON) instruction set,
//! which provides 128-bit vector operations for high-performance computing on ARM processors.
//! NEON is available on most ARM Cortex-A processors and all modern ARM64 (AArch64) processors
//! including Apple Silicon, AWS Graviton, and mobile devices.
//!
//! # Architecture Requirements
//!
//! - **CPU Support**: ARM Cortex-A8+ (ARMv7) or any AArch64 processor
//! - **Target Architecture**: ARM or AArch64
//! - **Compilation**: Must be compiled with NEON enabled (`-C target-feature=+neon`)
//! - **Runtime Detection**: The build system automatically detects NEON availability
//!
//! # Available Types
//!
//! - **Mathematical Functions**: Complete set of NEON-optimized math functions
//! - Vector types: Currently uses native ARM NEON intrinsics (float32x4_t, etc.)
//!
//! # Performance Characteristics
//!
//! - **Vector Width**: 128 bits (4 × f32, 2 × f64, 4 × i32, etc.)
//! - **Memory Alignment**: Optimal performance with 16-byte aligned data
//! - **Throughput**: Up to 4× speedup for vectorizable operations compared to scalar code
//! - **Power Efficiency**: Designed for mobile and embedded applications with power constraints
//!
//! # Conditional Compilation
//!
//! This module is only compiled when the `neon` CPU feature is available. The build
//! system automatically detects this and configures the appropriate compilation flags.
//! When NEON is not available, the library falls back to scalar implementations.
//!
//! # Platform Support
//!
//! - **Apple Silicon**: M1, M2, M3 processors (macOS, iOS)
//! - **AWS Graviton**: Graviton2, Graviton3 processors
//! - **Mobile**: Modern Android and iOS devices
//! - **Embedded**: ARM Cortex-A series processors

#[allow(clippy::excessive_precision, clippy::empty_line_after_doc_comments)]
pub mod math;

pub mod f32x4;

pub mod outer;

pub mod slice;
