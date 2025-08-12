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
//! ### High-Level SIMD Usage
//!
//! ```rust,no_run
//! use simdly::simd::SimdMath;
//!
//! // Vectorized mathematical operations - works on both AVX2 and NEON
//! let angles = vec![0.0, std::f32::consts::PI / 4.0, std::f32::consts::PI / 2.0];
//! let cosines = angles.cos(); // SIMD accelerated
//!
//! // 2D distance calculations
//! let x_coords = vec![3.0, 5.0, 8.0, 7.0];
//! let y_coords = vec![4.0, 12.0, 15.0, 24.0];
//! let distances = x_coords.hypot(y_coords); // [5.0, 13.0, 17.0, 25.0]
//!
//! // Power calculations
//! let bases = vec![2.0, 3.0, 4.0, 5.0];
//! let exponents = vec![2.0, 2.0, 2.0, 2.0];
//! let powers = bases.pow(exponents); // [4.0, 9.0, 16.0, 25.0]
//! ```
//!
//! ### Parallel SIMD Operations
//!
//! For maximum performance on large datasets, use the parallel SIMD methods that automatically
//! select between single-threaded and multi-threaded implementations based on array size:
//!
//! ```rust,no_run
//! use simdly::simd::SimdMath;
//!
//! // Large dataset - automatically uses parallel SIMD
//! let large_data: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 0.001).collect();
//! let results = large_data.par_cos(); // Multi-threaded SIMD
//!
//! // Small dataset - automatically uses regular SIMD  
//! let small_data = vec![1.0, 2.0, 3.0, 4.0];
//! let results = small_data.par_sin(); // Single-threaded SIMD
//!
//! // Works with all math functions
//! let sqrt_results = large_data.par_sqrt();
//! let exp_results = large_data.par_exp();
//! let abs_results = large_data.par_abs();
//! ```
//!
//! ### Platform-Specific Vector Operations
//!
//! ```rust,no_run
//! #[cfg(target_arch = "x86_64")]
//! use simdly::simd::avx2::f32x8::F32x8;
//! #[cfg(target_arch = "aarch64")]
//! use simdly::simd::neon::f32x4::F32x4;
//! use simdly::simd::SimdMath;
//!
//! #[cfg(target_arch = "x86_64")]
//! {
//!     // AVX2: Process 8 elements at once
//!     let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//!     let vec = F32x8::from(data.as_slice());
//!     let result = vec.sin(); // 8 parallel sine calculations
//! }
//!
//! #[cfg(target_arch = "aarch64")]
//! {
//!     // NEON: Process 4 elements at once
//!     let data = [1.0, 2.0, 3.0, 4.0];
//!     let vec = F32x4::from(data.as_slice());
//!     let result = vec.sin(); // 4 parallel sine calculations
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
/// - Use `simd_add` for medium-sized arrays (128+ elements)
/// - Use `par_simd_add` for large arrays (262,144+ elements)
/// - Use `scalar_add` for small arrays (<128 elements) or when SIMD is unavailable
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
    /// - **Best for**: Arrays with 128+ elements
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
    /// - **Best for**: Arrays with <128 elements
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

// ================================================================================================
// PERFORMANCE TUNING CONSTANTS
// ================================================================================================

/// Minimum array size where parallel SIMD operations become beneficial.
///
/// Optimized for modern x86_64/aarch64 systems with 8GB+ RAM:
/// - x86_64: L3 cache sizes (8-32MB), higher memory bandwidth
/// - aarch64: Unified cache design, efficient parallel execution
/// - Thread pool overhead amortized over larger datasets
/// - Memory contention reduced with larger working sets
///
/// Threshold for switching from single-threaded SIMD to parallel SIMD operations.
///
/// Arrays larger than this threshold will automatically use parallel SIMD processing
/// to maximize throughput on multi-core systems. This threshold represents 1 MiB of
/// f32 data, chosen to balance parallelization overhead with performance gains.
///
/// **Value**: 262,144 elements (1 MiB for f32 arrays)
pub const PARALLEL_SIMD_THRESHOLD: usize = 262_144; // 1 MiB (f32)

/// Threshold below which scalar operations outperform SIMD.
///
/// Optimized for modern CPUs with efficient SIMD units:
/// - x86_64 AVX2: Lower setup cost, wider registers (256-bit)
/// - aarch64 NEON: Native 128-bit operations, minimal overhead
/// - Accounts for modern branch prediction and instruction pipelining
///
/// Arrays smaller than this threshold will use scalar operations to avoid
/// SIMD overhead. This threshold is optimized for modern CPUs where SIMD
/// setup costs become negligible above 128 elements.
///
/// **Value**: 128 elements (512 bytes for f32 arrays)
pub const SIMD_THRESHOLD: usize = 128;

/// Optimal chunk size for parallel processing.
///
/// Optimized for modern cache hierarchies with 8GB+ RAM:
/// - x86_64: L2 cache (256KB-1MB), L3 cache (8-32MB)
/// - aarch64: Unified cache design (512KB-8MB L2/L3)
/// - Larger chunks reduce thread coordination overhead
/// - Better memory bandwidth utilization on high-RAM systems
pub(crate) const PARALLEL_CHUNK_SIZE: usize = 16_384; // 64KiB (f32)
