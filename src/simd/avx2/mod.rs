//! AVX2 SIMD implementations for 256-bit vector operations.
//!
//! This module contains SIMD implementations using Intel's Advanced Vector Extensions 2 (AVX2)
//! instruction set, which provides 256-bit vector operations for high-performance computing.
//! AVX2 is available on most Intel processors since Haswell (2013) and AMD processors since
//! Excavator (2015).
//!
//! # Architecture Requirements
//!
//! - **CPU Support**: Intel Haswell (2013+) or AMD Excavator (2015+)
//! - **Target Architecture**: x86 or x86_64
//! - **Compilation**: Must be compiled with AVX2 enabled (`-C target-feature=+avx2`)
//! - **Runtime Detection**: The build system automatically detects AVX2 availability
//!
//! # Available Types
//!
//! - [`f32x8`]: 256-bit vector containing 8 packed single-precision floating-point values
//!
//! # Performance Characteristics
//!
//! - **Vector Width**: 256 bits (8 × f32, 4 × f64, 8 × i32, etc.)
//! - **Memory Alignment**: Optimal performance with 32-byte aligned data
//! - **Throughput**: Up to 8× speedup for vectorizable operations compared to scalar code
//! - **Latency**: Low-latency operations with modern CPU designs
//!
//! # Usage Example
//!
//! ```rust
//! # #[cfg(target_feature = "avx2")]
//! # {
//! use simdly::simd::avx2::f32x8::F32x8;
//! use simdly::simd::SimdLoad;
//!
//! // Load 8 f32 values into a 256-bit AVX2 vector
//! let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let vector = F32x8::from(data.as_slice());
//! # }
//! ```
//!
//! # Conditional Compilation
//!
//! This module is only compiled when the `avx2` CPU feature is available. The build
//! system automatically detects this and configures the appropriate compilation flags.
//! When AVX2 is not available, the library falls back to other instruction sets or
//! scalar implementations.

pub mod f32x8;

#[allow(clippy::excessive_precision, clippy::empty_line_after_doc_comments)]
pub mod math;

pub mod slice;

pub mod outer;

pub mod matmul;
