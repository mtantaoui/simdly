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

// ================================================================================================
// MODULE DECLARATIONS
// ================================================================================================

/// SIMD operations and platform-specific implementations.
pub mod simd;

/// Internal memory allocation utilities.
pub(crate) mod utils;

// ================================================================================================
// PUBLIC TRAITS
// ================================================================================================

/// Trait for vectorized addition operations with multiple execution strategies.
///
/// This trait provides three different approaches to vector addition, allowing users
/// to choose the most appropriate strategy based on their data size and performance
/// requirements:
///
/// - **SIMD Addition**: Fast vectorized operations using CPU SIMD instructions
/// - **Parallel SIMD**: Combines SIMD with multi-threading for large datasets
/// - **Scalar Addition**: Fallback scalar implementation for small datasets or unsupported platforms
///
/// # Type Parameters
///
/// * `Rhs` - The right-hand side operand type (defaults to `Self`)
///
/// # Performance Guidelines
///
/// - Use `simd_add` for medium-sized arrays (256+ elements)
/// - Use `par_simd_add` for large arrays (131,072+ elements)
/// - Use `scalar_add` for small arrays (<256 elements) or when SIMD is unavailable
///
/// # Implementation Strategy
///
/// Implementations should automatically select the best strategy based on:
/// - Data size thresholds
/// - Available CPU features (AVX2, NEON, etc.)
/// - Memory alignment characteristics
/// - Thread pool availability
pub trait SimdAdd<Rhs = Self> {
    /// The output type produced by addition operations
    type Output;

    /// Performs vectorized addition using SIMD instructions.
    ///
    /// This method uses CPU SIMD instructions (AVX2, NEON, etc.) to perform
    /// multiple additions simultaneously, providing significant performance
    /// improvements over scalar operations for suitable data sizes.
    ///
    /// # Performance
    ///
    /// - **Best for**: Arrays with 256+ elements
    /// - **Speedup**: 2-8x faster than scalar operations (depending on CPU)
    /// - **Memory**: Processes data in chunks matching SIMD vector width
    ///
    /// # Fallback Behavior
    ///
    /// Automatically falls back to scalar operations when:
    /// - Array size is below SIMD threshold
    /// - SIMD instructions are not available on the current CPU
    /// - Memory alignment prevents efficient SIMD usage
    fn simd_add(self, rhs: Rhs) -> Self::Output;

    /// Performs parallel vectorized addition using SIMD + multi-threading.
    ///
    /// This method combines SIMD instructions with parallel processing across
    /// multiple CPU cores, providing maximum performance for large datasets.
    /// Work is distributed across available threads, with each thread using
    /// SIMD instructions on its assigned data chunk.
    ///
    /// # Performance
    ///
    /// - **Best for**: Arrays with 131,072+ elements (128KB+)
    /// - **Speedup**: 4-32x faster than scalar (depending on core count and data size)
    /// - **Overhead**: Higher setup cost due to thread coordination
    ///
    /// # Thread Safety
    ///
    /// Uses the global Rayon thread pool for work distribution. Thread pool
    /// size automatically matches the number of available CPU cores.
    ///
    /// # Fallback Behavior
    ///
    /// Falls back to `simd_add` when:
    /// - Array size is below parallel threshold
    /// - Thread pool is unavailable or already saturated
    fn par_simd_add(self, rhs: Rhs) -> Self::Output;

    /// Performs scalar addition using standard CPU instructions.
    ///
    /// This method provides a reliable fallback implementation that works
    /// on any CPU architecture. While slower than SIMD operations, it has
    /// lower overhead and can be faster for very small datasets.
    ///
    /// # Performance
    ///
    /// - **Best for**: Arrays with <256 elements
    /// - **Characteristics**: Predictable performance, low overhead
    /// - **Compatibility**: Works on all platforms and architectures
    ///
    /// # Use Cases
    ///
    /// - Small data sets where SIMD overhead isn't justified
    /// - Platforms without SIMD support
    /// - Debugging and reference implementations
    /// - Ensuring consistent results across different architectures
    fn scalar_add(self, rhs: Rhs) -> Self::Output;
}

/// High-level trait for optimized vector addition with automatic strategy selection.
///
/// This trait provides a single, ergonomic interface that automatically selects
/// the best addition strategy based on the data characteristics and available
/// hardware features. It acts as a smart wrapper around the more granular
/// `SimdAdd` trait methods.
///
/// # Automatic Strategy Selection
///
/// The implementation automatically chooses between:
/// 1. **Parallel SIMD**: For very large arrays (>131K elements)
/// 2. **SIMD**: For medium arrays (256+ elements)  
/// 3. **Scalar**: For small arrays (<256 elements)
///
/// # Type Parameters
///
/// * `Rhs` - The right-hand side operand type (defaults to `Self`)
///
/// # Design Philosophy
///
/// This trait prioritizes ease of use over fine-grained control. Users who need
/// specific performance characteristics should use `SimdAdd` directly, while
/// users who want optimal performance with minimal complexity should use this trait.
///
/// # Examples
///
/// ```rust
/// use simdly::FastAdd;
///
/// let large_a: Vec<f32> = (0..100_000).map(|x| x as f32).collect();
/// let large_b: Vec<f32> = (0..100_000).map(|x| (x * 2) as f32).collect();
///
/// // Automatically uses parallel SIMD for large datasets
/// let result = large_a.fast_add(large_b);
///
/// let small_a = vec![1.0f32, 2.0, 3.0];
/// let small_b = vec![4.0f32, 5.0, 6.0];
///
/// // Automatically uses scalar operations for small datasets
/// let result = small_a.fast_add(small_b);
/// ```
pub trait FastAdd<Rhs = Self> {
    /// The output type produced by the addition operation
    type Output;

    /// Performs optimized vector addition with automatic strategy selection.
    ///
    /// This method analyzes the input data and automatically selects the most
    /// appropriate addition strategy (scalar, SIMD, or parallel SIMD) based on:
    ///
    /// - **Data size**: Larger datasets benefit from more complex strategies
    /// - **Hardware capabilities**: Available SIMD instruction sets
    /// - **Memory characteristics**: Alignment and access patterns
    /// - **System load**: Current thread pool utilization
    ///
    /// # Performance Optimization
    ///
    /// The selection algorithm balances:
    /// - **Throughput**: Maximum operations per second
    /// - **Latency**: Time to first result
    /// - **Overhead**: Setup and coordination costs
    /// - **Resource utilization**: CPU cores and memory bandwidth
    ///
    /// # Predictable Behavior
    ///
    /// While the internal strategy selection is automatic, the behavior follows
    /// predictable patterns based on well-defined thresholds, ensuring consistent
    /// performance characteristics across similar workloads.
    fn fast_add(self, rhs: Rhs) -> Self::Output;
}

// ================================================================================================
// PERFORMANCE TUNING CONSTANTS
// ================================================================================================

/// Minimum array size where parallel SIMD operations become beneficial.
///
/// This threshold accounts for:
/// - Thread pool overhead
/// - Work distribution costs
/// - Memory contention between threads
/// - Context switching overhead
pub(crate) const PARALLEL_SIMD_THRESHOLD: usize = 131_072; // 512 KiB (f32)

/// Threshold below which scalar operations outperform SIMD.
pub(crate) const SIMD_THRESHOLD: usize = 256;

/// Optimal chunk size for parallel processing.
///
/// Chosen to balance:
/// - Cache locality
/// - Work distribution granularity
/// - Memory bandwidth utilization
pub(crate) const PARALLEL_CHUNK_SIZE: usize = 4_096; // 16KiB - L1 cache (f32)
