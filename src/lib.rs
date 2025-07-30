//! # simdly
//!
//! 🚀 A high-performance Rust library that leverages SIMD (Single Instruction, Multiple Data)
//! instructions for fast vectorized computations. This library provides efficient implementations
//! of mathematical operations using modern CPU features.
//!
//! ## Features
//!
//! - **SIMD Optimized**: Leverages AVX2 (256-bit) and NEON (128-bit) instructions for vector operations
//! - **Memory Efficient**: Supports both aligned and unaligned memory access patterns
//! - **Generic Traits**: Provides consistent interfaces across different SIMD implementations
//! - **Safe Abstractions**: Wraps unsafe SIMD operations in safe, ergonomic APIs
//! - **Cross-Platform**: Supports both x86/x86_64 and ARM/AArch64 architectures
//! - **Performance**: Optimized for high-throughput numerical computations
//!
//! ## Architecture Support
//!
//! Currently supports:
//! - **x86/x86_64** with AVX2 (256-bit vectors)
//! - **ARM/AArch64** with NEON (128-bit vectors)
//!
//! Future support planned for:
//! - SSE (128-bit vectors for older x86 processors)
//!
//! ## Usage
//!
//! The library provides traits for SIMD operations that automatically detect and use
//! the best available instruction set on the target CPU.
//!
//! ### AVX2 Example (x86/x86_64)
//!
//! ```rust,ignore
//! #[cfg(target_arch = "x86_64")]
//! {
//!     use simdly::simd::avx2::math::{_mm256_sin_ps, _mm256_hypot_ps};
//!     use std::arch::x86_64::_mm256_set1_ps;
//!
//!     // 8 parallel sine calculations
//!     let input = _mm256_set1_ps(1.0);
//!     let result = unsafe { _mm256_sin_ps(input) };
//!
//!     // 2D Euclidean distance for 8 point pairs
//!     let x = _mm256_set1_ps(3.0);
//!     let y = _mm256_set1_ps(4.0);
//!     let distance = unsafe { _mm256_hypot_ps(x, y) }; // sqrt(3² + 4²) = 5.0
//! }
//! ```
//!
//! ### NEON Example (ARM/AArch64)
//!
//! ```rust,ignore
//! #[cfg(target_arch = "aarch64")]
//! {
//!     use simdly::simd::neon::math::{vsinq_f32, vhypotq_f32};
//!     use std::arch::aarch64::vdupq_n_f32;
//!
//!     // 4 parallel sine calculations
//!     let input = vdupq_n_f32(1.0);
//!     let result = unsafe { vsinq_f32(input) };
//!
//!     // 2D Euclidean distance for 4 point pairs
//!     let x = vdupq_n_f32(3.0);
//!     let y = vdupq_n_f32(4.0);
//!     let distance = unsafe { vhypotq_f32(x, y) }; // sqrt(3² + 4²) = 5.0
//! }
//! ```
//!
//!
//! ## Performance Considerations
//!
//! - **Memory Alignment**: Use aligned memory when possible for optimal performance
//! - **Batch Processing**: Process data in chunks that match SIMD vector sizes
//! - **CPU Features**: Enable appropriate target features during compilation

pub mod error;
pub mod simd;
pub(crate) mod utils;

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn simd_add(self, rhs: Rhs) -> Self::Output;
    fn par_simd_add(self, rhs: Rhs) -> Self::Output;
    fn scalar_add(self, rhs: Rhs) -> Self::Output;
}
