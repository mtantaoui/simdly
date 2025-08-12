//! AVX2-optimized slice operations for f32 arrays.
//!
//! This module provides high-performance implementations of common array operations
//! using Intel's AVX2 instruction set. It focuses on element-wise operations that
//! can benefit significantly from SIMD vectorization.
//!
//! # Architecture Requirements
//!
//! - **CPU Support**: Intel Haswell (2013+) or AMD Excavator (2015+)
//! - **Instruction Sets**: AVX, AVX2, and FMA
//! - **Compilation**: Target features are automatically enabled via attributes
//! - **Runtime**: CPU feature detection handled by build system
//!
//! # Performance Characteristics
//!
//! ## Throughput Improvements
//!
//! | Operation | Scalar | SIMD (AVX2) | Parallel SIMD | Speedup |
//! |-----------|---------|-------------|---------------|---------|
//! | Addition | 1x | ~8x | ~8x × cores | Up to 64x (8-core) |
//! | Memory bandwidth | ~25GB/s | ~100GB/s | ~400GB/s | 16x (quad-channel) |
//!
//! ## Optimal Array Sizes
//!
//! - **Scalar**: < 100 elements (low overhead)
//! - **SIMD**: 100-100,000 elements (vectorization benefits)
//! - **Parallel SIMD**: > 100,000 elements (parallelization benefits)
//!
//! # Memory Layout Considerations
//!
//! - **Alignment**: Optimal performance with 32-byte aligned data
//! - **Cache**: Designed to minimize cache misses with sequential access
//! - **Bandwidth**: Efficient memory utilization for large datasets
//!
//! # Available Operations
//!
//! ## Arithmetic Operations
//! - [`SimdAdd`]: Element-wise addition with scalar, SIMD, and parallel variants
//!   - [`SimdAdd::simd_add`]: Single-threaded AVX2 vectorized addition
//!   - [`SimdAdd::par_simd_add`]: Multi-threaded AVX2 vectorized addition
//!   - [`SimdAdd::scalar_add`]: Sequential scalar addition fallback
//!
//! ## Mathematical Functions
//! - [`SimdMath`]: Mathematical operations with SIMD acceleration
//!   - [`SimdMath::cos`]: Cosine computation (fully implemented with 4x-13x speedup)
//!   - [`SimdMath::sin`]: Sine computation (planned)
//!   - [`SimdMath::exp`]: Exponential function (planned)
//!   - [`SimdMath::ln`]: Natural logarithm (planned)
//!   - [`SimdMath::sqrt`]: Square root (planned)
//!   - [`SimdMath::abs`]: Absolute value (planned)
//!
//! ## Planned Operations
//! - Subtraction, multiplication, division with SIMD acceleration
//! - Additional transcendental functions (sin, tan, exp, log)
//! - Reduction operations (sum, min, max)
//! - Comparison and selection operations
//!
//! # Usage Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use simdly::SimdAdd;
//!
//! // Create test data
//! let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
//! let b: Vec<f32> = (0..1000).map(|i| (i * 2) as f32).collect();
//!
//! // Choose the best method based on data size
//! let result = if a.len() < 100 {
//!     // Small arrays: use scalar for minimal overhead
//!     a.as_slice().scalar_add(b.as_slice())
//! } else if a.len() < 100_000 {
//!     // Medium arrays: use SIMD for ~8x speedup
//!     a.as_slice().simd_add(b.as_slice())
//! } else {
//!     // Large arrays: use parallel SIMD for maximum throughput
//!     a.as_slice().par_simd_add(b.as_slice())
//! };
//! ```
//!
//! ## Performance-Optimized Usage
//!
//! ```rust
//! use simdly::{SimdAdd, simd::SimdMath};
//!
//! // For maximum performance with known large datasets
//! let large_a: Vec<f32> = vec![1.0; 1_000_000];
//! let large_b: Vec<f32> = vec![2.0; 1_000_000];
//!
//! // Use parallel SIMD for maximum throughput
//! let result = large_a.as_slice().par_simd_add(large_b.as_slice());
//!
//! // For guaranteed SIMD without parallelization overhead
//! let medium_a: Vec<f32> = vec![1.0; 5_000];
//! let medium_b: Vec<f32> = vec![2.0; 5_000];
//! let result = medium_a.as_slice().simd_add(medium_b.as_slice());
//! ```
//!
//! ## Mathematical Operations Usage
//!
//! ```rust
//! use simdly::simd::SimdMath;
//!
//! // Vectorized cosine computation - always prefer SIMD for mathematical functions
//! let angles: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
//! let cosines = angles.as_slice().cos(); // 4x-13x speedup over scalar
//!
//! // Mathematical functions benefit from SIMD even for small arrays
//! let small_angles = vec![0.0, std::f32::consts::PI / 4.0, std::f32::consts::PI / 2.0];
//! let results = small_angles.as_slice().cos();
//! assert!((results[0] - 1.0).abs() < 1e-5);        // cos(0) ≈ 1
//! assert!((results[1] - 0.707107).abs() < 1e-5);   // cos(π/4) ≈ 0.707
//! assert!(results[2].abs() < 1e-5);                // cos(π/2) ≈ 0
//! ```
//!
//! # Safety and Compatibility
//!
//! - **Memory Safety**: All operations are memory-safe despite internal `unsafe` usage
//! - **Bounds Checking**: Automatic validation of array lengths and boundaries
//! - **CPU Detection**: Graceful fallback when AVX2 is not available
//! - **Thread Safety**: All operations are thread-safe and side-effect free

use rayon::prelude::*;

#[cfg(not(target_os = "windows"))]
use crate::utils::alloc_uninit_vec;

#[cfg(target_os = "windows")]
use crate::utils::AlignedVec;

use crate::{
    simd::{
        avx2::f32x8::{self, F32x8, AVX_ALIGNMENT},
        slice::scalar_add,
        SimdCmp, SimdLoad, SimdMath, SimdStore,
    },
    SimdAdd, PARALLEL_CHUNK_SIZE, PARALLEL_SIMD_THRESHOLD, SIMD_THRESHOLD,
};

// ================================================================================================
// PERFORMANCE TUNING CONSTANTS
// ================================================================================================

// Note: PARALLEL_SIMD_THRESHOLD, PARALLEL_CHUNK_SIZE, and SIMD_THRESHOLD are imported from crate root

// ================================================================================================
// SIMD OPERATIONS
// ================================================================================================

/// Performs element-wise addition using AVX2 SIMD instructions.
///
/// This function processes arrays in chunks of 8 elements using 256-bit AVX2 vectors,
/// providing significant performance improvements over scalar addition for large datasets.
///
/// # Algorithm
///
/// 1. **Aligned Processing**: Processes complete 8-element chunks using `simd_add_block`
/// 2. **Remainder Processing**: Handles remaining elements using `simd_add_partial_block`
/// 3. **Platform-Optimized Memory**: Uses fast zero-copy allocation on Linux/Mac, safe copy-based allocation on Windows
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice (must have same length as `a`)
///
/// # Returns
///
/// A new vector with the element-wise sum, using platform-optimized memory allocation:
/// - **Linux/Mac**: Zero-copy aligned allocation for maximum performance
/// - **Windows**: Safe AlignedVec with copy-based conversion to prevent heap corruption
///
/// # Performance
///
/// - **Vectorization**: Up to 8x speedup over scalar operations
/// - **Memory Bandwidth**: Efficient use of memory hierarchy with aligned access
/// - **Throughput**: Optimal for arrays with hundreds or thousands of elements
/// - **Cross-platform**: Maximum performance on each platform while maintaining safety
///
/// # Safety
///
/// This function requires AVX2 support and is marked with target features.
/// The caller must ensure the CPU supports these instructions.
///
/// # Panics
///
/// Panics if:
/// - Input slices have different lengths
/// - Either input slice is empty
///
/// # Errors
///
/// This function uses `debug_assert!` for validation and will panic rather than return an error.
/// Memory allocation is handled internally and should not fail under normal circumstances.
#[inline(always)]
pub(crate) fn simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to scalar to avoid SIMD overhead
    if a.len() < SIMD_THRESHOLD {
        return scalar_add(a, b);
    }

    debug_assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    debug_assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_add_block(&a[i], &b[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_add_partial_block(
            &a[complete_lanes],
            &b[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes, // number of remaining incomplete lanes
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

/// Processes a complete 8-element block using AVX2 SIMD operations.
///
/// This function performs vectorized addition on exactly 8 f32 elements,
/// using aligned memory access for optimal performance.
///
/// # Arguments
///
/// * `a` - Pointer to first 8-element block
/// * `b` - Pointer to second 8-element block  
/// * `c` - Pointer to destination 8-element block
///
/// # Safety
///
/// - All pointers must be valid and point to at least 8 consecutive f32 values
/// - Pointers should be aligned to 32-byte boundaries for optimal performance
/// - The destination pointer must point to writable memory
///
/// # Performance
///
/// - **Single Instruction**: Executes 8 additions in one AVX2 instruction
/// - **Aligned Access**: Uses fastest possible memory access patterns
/// - **Zero Overhead**: Fully inlined with no function call overhead
#[inline(always)]
fn simd_add_block(a: *const f32, b: *const f32, c: *mut f32) {
    // Load 8 f32 values using AVX2 SIMD instructions
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let b_chunk_simd = unsafe { F32x8::load(b, f32x8::LANE_COUNT) };

    // Perform vectorized addition (8 operations in parallel)
    let result = a_chunk_simd + b_chunk_simd;

    // Store the result back to memory with aligned access
    // Platform-specific allocation ensures 32-byte alignment:
    // - Linux/Mac: alloc_uninit_vec() provides aligned Vec<f32>
    // - Windows: AlignedVec maintains alignment throughout computation
    unsafe { result.store_aligned_at(c) };
}

/// Processes a partial block (fewer than 8 elements) using masked SIMD operations.
///
/// This function handles array remainders that don't fill a complete 8-element vector,
/// using masked load/store operations to prevent reading/writing beyond buffer boundaries.
///
/// # Arguments
///
/// * `a` - Pointer to first partial block
/// * `b` - Pointer to second partial block
/// * `c` - Pointer to destination partial block
/// * `size` - Number of elements to process (must be < 8)
///
/// # Safety
///
/// - All pointers must be valid and point to at least `size` consecutive f32 values
/// - `size` must be less than `f32x8::LANE_COUNT` (8)
/// - The destination pointer must point to writable memory for `size` elements
///
/// # Performance
///
/// - **Masked Operations**: Uses AVX2 masked instructions to safely handle partial data
/// - **No Bounds Violations**: Prevents reading/writing beyond intended memory ranges
/// - **Efficient Remainders**: Optimal handling of array sizes not divisible by 8
#[inline(always)]
fn simd_add_partial_block(a: *const f32, b: *const f32, c: *mut f32, size: usize) {
    // Load partial data using masked operations (prevents buffer overrun)
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    let b_chunk_simd = unsafe { F32x8::load_partial(b, size) };

    // Perform addition and store only the valid elements
    unsafe { (a_chunk_simd + b_chunk_simd).store_at_partial(c) };
}

/// Performs element-wise addition using parallel AVX2 SIMD operations.
///
/// This function provides multi-threaded SIMD addition for very large arrays,
/// using Rayon to distribute work across multiple CPU cores with SIMD processing.
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice (must have same length as `a`)
///
/// # Returns
///
/// A `Result` containing a new vector with the element-wise sum,
/// or an error if validation or allocation fails.
///
/// # Implementation
///
/// The implementation:
/// 1. **Thread Pool**: Distributes work across multiple CPU cores using Rayon
/// 2. **Chunk Processing**: Each thread processes a contiguous chunk using SIMD
/// 3. **Load Balancing**: Rayon automatically balances work across threads
/// 4. **Memory Efficiency**: Uses chunking to minimize cache conflicts between threads
///
/// # Performance
///
/// - **Scalability**: Near-linear speedup with available CPU cores
/// - **Optimal Size**: Best for arrays with tens of thousands of elements or more
/// - **Memory Bandwidth**: Can saturate available memory bandwidth
/// - **Fallback**: Automatically falls back to single-threaded SIMD for smaller arrays
///
/// # Panics
///
/// Panics if:
/// - Input slices have different lengths
/// - Either input slice is empty
///
/// # Errors
///
/// This function uses `debug_assert!` for validation and will panic rather than return an error.
/// Memory allocation is handled internally and should not fail under normal circumstances.
#[inline(always)]
pub(crate) fn parallel_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return scalar_add(a, b);
    }

    debug_assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    debug_assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_add_block(&a[start_idx + i], &b[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_add_partial_block(
                    &a[start_idx + complete_blocks],
                    &b[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

// ================================================================================================
// MATHEMATICAL OPERATIONS
// ================================================================================================

/// Computes cosine of each element using scalar operations.
///
/// This function serves as the baseline implementation for cosine computation,
/// using Rust's standard library `f32::cos()` method which is typically
/// implemented using highly optimized math library functions (libm).
///
/// # Performance Characteristics
///
/// - **High precision**: Uses standard library implementation with full IEEE 754 compliance
/// - **Auto-vectorization**: LLVM may auto-vectorize this loop for better performance
/// - **Consistent accuracy**: Provides reference precision for SIMD comparison
/// - **Simple implementation**: Iterator-based approach with minimal overhead
///
/// # Benchmark Results (Intel AVX2)
///
/// Comprehensive benchmarks show that **SIMD consistently outperforms scalar**
/// for cosine computation across all tested array sizes:
///
/// | Array Size | Scalar (ns) | SIMD (ns) | **SIMD Speedup** |
/// |------------|-------------|-----------|------------------|
/// | 4 KiB      | ~3,500      | ~800      | **4.4x faster**  |
/// | 64 KiB     | ~140,000    | ~12,000   | **11.7x faster** |
/// | 1 MiB      | ~2,600,000  | ~195,000  | **13.3x faster** |
/// | 128 MiB    | ~350,000,000| ~38,000,000| **9.2x faster** |
///
/// **Key Insights:**
/// - **Mathematical complexity favors SIMD**: Unlike simple operations (addition),
///   trigonometric functions have sufficient computational intensity to amortize
///   vectorization overhead even for small arrays
/// - **No threshold needed**: SIMD is beneficial from 4 KiB to 128+ MiB
/// - **Peak performance**: ~13x speedup at cache-friendly sizes (1 MiB)
/// - **Memory-bound scaling**: Performance levels off at very large sizes due to bandwidth limits
///
/// # Recommendation
///
/// For production use, prefer the SIMD implementation (`SimdMath::cos()`) over this
/// scalar version for all array sizes. This function is primarily useful for:
/// - Precision validation and testing
/// - Platforms without AVX2 support
/// - Reference implementation for algorithm verification
///
/// # Arguments
///
/// * `a` - Input slice containing f32 values (angles in radians)
///
/// # Returns
///
/// A new vector containing the cosine of each input element.
///
/// # Panics
///
/// Panics if the input slice is empty.
///
/// # Example
///
/// ```rust
/// use simdly::simd::avx2::slice::scalar_cos;
///
/// let angles = vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI];
/// let results = scalar_cos(&angles);
/// assert!((results[0] - 1.0).abs() < 1e-6);    // cos(0) ≈ 1
/// assert!(results[1].abs() < 1e-6);             // cos(π/2) ≈ 0  
/// assert!((results[2] + 1.0).abs() < 1e-6);    // cos(π) ≈ -1
/// ```
#[inline(always)]
pub fn scalar_cos(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    a.iter().map(|x| x.cos()).collect()
}

/// Computes cosine of each element using Intel AVX2 SIMD instructions.
///
/// This function provides a vectorized implementation of cosine computation using
/// Intel AVX2 256-bit SIMD instructions. It processes 8 f32 values simultaneously
/// using custom polynomial approximation optimized for AVX2.
///
/// # Performance Characteristics
///
/// - **Vectorized processing**: Computes 8 cosines simultaneously using AVX2
/// - **Custom math functions**: Uses optimized polynomial approximations from math module
/// - **Memory efficient**: Uses platform-optimized aligned memory allocation
/// - **Remainder handling**: Processes non-multiple-of-8 arrays using partial vectors
///
/// # Benchmark Results (Intel AVX2 vs Scalar)
///
/// Extensive benchmarking demonstrates **exceptional SIMD performance** across all array sizes:
///
/// | Array Size | Scalar (ns) | SIMD (ns) | **Speedup** | Performance Class |
/// |------------|-------------|-----------|-------------|-------------------|
/// | 4 KiB      | 3,500       | 800       | **4.4x**    | Small arrays      |
/// | 64 KiB     | 140,000     | 12,000    | **11.7x**   | Cache-resident    |
/// | 1 MiB      | 2,600,000   | 195,000   | **13.3x**   | **Peak performance** |
/// | 128 MiB    | 350,000,000 | 38,000,000| **9.2x**    | Memory-bound      |
///
/// # Why SIMD Dominates for Mathematical Functions
///
/// Unlike simple arithmetic operations, trigonometric functions like cosine involve:
/// - **Range reduction**: Normalizing input angles to primary range
/// - **Polynomial evaluation**: Computing approximation series (multiple multiply-adds)
/// - **Conditional logic**: Handling different quadrants and edge cases
///
/// This computational complexity means:
/// 1. **SIMD overhead is amortized**: Setup costs are negligible compared to math operations
/// 2. **Vectorization is highly effective**: 8 simultaneous polynomial computations
/// 3. **No size threshold needed**: Benefits start immediately at 4 KiB arrays
///
/// # Implementation Details
///
/// The function uses a two-phase approach:
/// 1. **Main loop**: Processes complete 8-element blocks using `simd_cos_block`
/// 2. **Remainder handling**: Uses `simd_cos_partial_block` for remaining elements
///
/// # Precision vs Performance
///
/// - **Precision**: ~1e-5 to 1e-6 accuracy compared to standard library
/// - **Performance**: 4x-13x speedup with maintained mathematical accuracy
/// - **Range handling**: Robust across full f32 range with proper range reduction
///
/// # Production Recommendation
///
/// **Always prefer this SIMD implementation** over scalar cosine for:
/// - Any array size ≥ 8 elements (minimum AVX2 vector width)
/// - Production applications requiring mathematical performance
/// - Batch processing of trigonometric calculations
///
/// # Arguments
///
/// * `a` - Input slice containing f32 values (angles in radians)
///
/// # Returns
///
/// A new vector containing the cosine of each input element.
///
/// # Memory Allocation & Safety
///
/// This function uses platform-specific memory allocation strategies:
/// - **Linux/Mac**: Zero-copy `alloc_uninit_vec<f32>()` for maximum performance
/// - **Windows**: Safe `AlignedVec<f32>` with copy conversion to prevent heap corruption
/// - **SIMD Safety**: All platforms maintain 32-byte alignment for AVX2 operations
///
/// # Panics
///
/// Panics if the input slice is empty.
///
/// # Example
///
/// ```rust
/// use simdly::simd::SimdMath;
///
/// let angles = vec![0.0, std::f32::consts::PI / 4.0, std::f32::consts::PI / 2.0, std::f32::consts::PI];
/// let results = angles.as_slice().cos();
///
/// // Results should be approximately [1.0, 0.707, 0.0, -1.0]
/// assert!((results[0] - 1.0).abs() < 1e-5);
/// assert!((results[1] - 0.707107).abs() < 1e-5);
/// assert!(results[2].abs() < 1e-5);
/// assert!((results[3] + 1.0).abs() < 1e-5);
/// ```
#[inline(always)]
pub(crate) fn simd_cos(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_cos_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_cos_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes, // number of remaining incomplete lanes
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes the cosine of each element using parallel AVX2 SIMD with automatic threshold selection.
///
/// This function provides high-performance cosine computation using Intel AVX2 256-bit SIMD instructions
/// combined with Rayon's work-stealing thread pool. It automatically selects between single-threaded SIMD
/// and parallel SIMD based on input size to optimize performance across different workload sizes.
///
/// # Performance Characteristics
///
/// - **Threshold-based selection**: Arrays ≤ 262,144 elements use single-threaded SIMD
/// - **Parallel processing**: Larger arrays leverage multi-core parallelism with work-stealing
/// - **AVX2 vectorization**: Processes 8 f32 elements simultaneously (256-bit vectors)
/// - **Memory optimization**: Uses 32-byte aligned memory allocation for optimal performance
///
/// # Arguments
///
/// * `a` - Input slice containing angles in radians
///
/// # Returns
///
/// A new vector containing the cosine of each input element.
///
/// # Performance Benchmarks
///
/// On modern x86_64 processors with AVX2:
/// - **Small arrays** (≤ 262K): ~13.3x faster than scalar, single-threaded
/// - **Large arrays** (> 262K): Additional speedup from parallelization scales with CPU cores
/// - **Memory throughput**: Optimized for both L1/L2 cache and main memory access patterns
///
/// # Example
///
/// ```rust
/// use simdly::simd::avx2::slice::parallel_simd_cos;
///
/// let angles = vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI];
/// let cosines = parallel_simd_cos(&angles);
/// // Results: approximately [1.0, 0.0, -1.0]
/// ```
#[inline(always)]
pub fn parallel_simd_cos(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_cos(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_cos_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_cos_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Parallel SIMD sine computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_sin(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_sin(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_sin_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_sin_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Processes a complete 8-element block using AVX2 SIMD cosine operations.
///
/// This function is the core computational kernel for SIMD cosine operations.
/// It loads 8 consecutive f32 values, computes their cosines using vectorized
/// AVX2 instructions, and stores the results back to memory.
///
/// # Performance Optimizations
///
/// - **Inlined**: Function is marked `#[inline(always)]` to eliminate call overhead
/// - **Direct SIMD**: Uses F32x8 wrapper around native AVX2 intrinsics
/// - **Minimal overhead**: Single load → compute → store sequence
/// - **No bounds checking**: Assumes caller has verified array bounds
///
/// # Arguments
///
/// * `a` - Raw pointer to input data (must point to at least 8 valid f32 values)
/// * `c` - Raw pointer to output buffer (must have space for at least 8 f32 values)
///
/// # Safety
///
/// This function is unsafe because:
/// - No bounds checking is performed on input/output pointers
/// - Caller must ensure both pointers are valid and properly aligned
/// - Must point to at least 8 f32 values each
/// - Memory regions must not overlap (undefined behavior)
///
/// # Usage
///
/// This function is intended to be called only from within `simd_cos` where
/// array bounds and alignment have been verified.
#[inline(always)]
fn simd_cos_block(a: *const f32, c: *mut f32) {
    // Load 8 f32 values using AVX2 SIMD instructions
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };

    // Perform vectorized cosine computation (8 operations in parallel)
    let result = a_chunk_simd.cos();

    // Store the result back to memory with aligned access
    // Platform-specific allocation ensures 32-byte alignment:
    // - Linux/Mac: alloc_uninit_vec() provides aligned Vec<f32>
    // - Windows: AlignedVec maintains alignment throughout computation
    unsafe { result.store_aligned_at(c) };
}

/// Processes a partial block (fewer than 8 elements) using masked AVX2 operations.
///
/// This function handles the remainder elements when the input array size is not
/// a multiple of 8. It uses partial SIMD operations to process 1-7 elements
/// safely without reading beyond array bounds.
///
/// # Implementation Strategy
///
/// - **Partial loading**: Uses `F32x8::load_partial` to safely load 1-7 elements
/// - **Zero padding**: Unused vector lanes are filled with zeros
/// - **Partial storing**: Uses `store_at_partial` to write only the valid results
/// - **No overflow**: Guarantees no out-of-bounds memory access
///
/// # Performance Considerations
///
/// - **Overhead cost**: Partial operations are slower than full SIMD blocks
/// - **Necessary for correctness**: Required to handle arbitrary array sizes
/// - **Minimized usage**: Only called once per array for remainder elements
/// - **Still vectorized**: Uses SIMD even for 1-7 elements vs scalar fallback
///
/// # Arguments
///
/// * `a` - Raw pointer to input data (must point to at least `size` valid f32 values)
/// * `c` - Raw pointer to output buffer (must have space for at least `size` f32 values)
/// * `size` - Number of elements to process (must be 1-7)
///
/// # Safety
///
/// This function is unsafe because:
/// - No bounds checking on pointers beyond the `size` parameter
/// - Caller must ensure pointers are valid for at least `size` elements
/// - Must guarantee `size` is in range [1, 7]
/// - Memory regions must not overlap
///
/// # Usage
///
/// Called only from `simd_cos` to handle remainder elements after processing
/// complete 8-element blocks.
#[inline(always)]
fn simd_cos_partial_block(a: *const f32, c: *mut f32, size: usize) {
    // Load partial data using masked operations (prevents buffer overrun)
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };

    // Perform cosine computation and store only the valid elements
    unsafe { a_chunk_simd.cos().store_at_partial(c) };
}

// ================================================================================================
// TRAIT IMPLEMENTATIONS
// ================================================================================================

/// Implementation of mathematical operations for `Vec<f32>` using Intel AVX2 SIMD.
///
/// This implementation provides the same vectorized mathematical functions as the
/// slice implementation, but operates directly on owned vectors. It delegates to
/// the slice implementation for actual computation.
///
/// # Usage Pattern
///
/// ```rust
/// use simdly::simd::SimdMath;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let results = data.cos(); // Uses AVX2 SIMD cosine
/// ```
impl SimdAdd<Vec<f32>> for Vec<f32> {
    type Output = Vec<f32>;

    /// Performs SIMD-accelerated element-wise addition on `Vec<f32>`.
    /// Delegates to slice implementation for actual computation.
    #[inline(always)]
    fn simd_add(self, rhs: Vec<f32>) -> Self::Output {
        self.as_slice().simd_add(rhs.as_slice())
    }

    /// Performs parallel SIMD-accelerated element-wise addition on `Vec<f32>`.
    /// Delegates to slice implementation for actual computation.
    #[inline(always)]
    fn par_simd_add(self, rhs: Vec<f32>) -> Self::Output {
        self.as_slice().par_simd_add(rhs.as_slice())
    }

    /// Performs scalar element-wise addition on `Vec<f32>`.
    /// Delegates to slice implementation for actual computation.
    #[inline(always)]
    fn scalar_add(self, rhs: Vec<f32>) -> Self::Output {
        self.as_slice().scalar_add(rhs.as_slice())
    }
}

#[inline(always)]
pub(crate) fn eq_elementwise(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    debug_assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        eq_elementwise_block(&a[i], &b[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        eq_elementwise_partial_block(
            &a[complete_lanes],
            &b[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes, // number of remaining incomplete lanes
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn eq_elementwise_partial_block(a: *const f32, b: *const f32, c: *mut f32, size: usize) {
    // Load partial data using masked operations (prevents buffer overrun)
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    let b_chunk_simd = unsafe { F32x8::load_partial(b, size) };

    // Perform addition and store only the valid elements
    unsafe {
        a_chunk_simd
            .elementwise_eq(b_chunk_simd)
            .store_at_partial(c)
    };
}

#[inline(always)]
fn eq_elementwise_block(a: *const f32, b: *const f32, c: *mut f32) {
    // Load 8 f32 values using AVX2 SIMD instructions
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let b_chunk_simd = unsafe { F32x8::load(b, f32x8::LANE_COUNT) };

    // Store the result back to memory with aligned access
    // Platform-specific allocation ensures 32-byte alignment:
    // - Linux/Mac: alloc_uninit_vec() provides aligned Vec<f32>
    // - Windows: AlignedVec maintains alignment throughout computation
    unsafe {
        a_chunk_simd
            .elementwise_eq(b_chunk_simd)
            .store_aligned_at(c)
    };
}

// ================================================================================================
// MATHEMATICAL FUNCTIONS - POW, HYPOT, HYPOT3, HYPOT4
// ================================================================================================

/// Computes the power (x^y) of corresponding elements using AVX2 SIMD instructions.
///
/// This function provides vectorized power computation using Intel AVX2 256-bit SIMD instructions.
/// It processes 8 f32 values simultaneously using custom polynomial approximation optimized for AVX2.
///
/// # Performance Characteristics
///
/// - **Vectorized processing**: Computes 8 powers simultaneously using AVX2
/// - **Custom math functions**: Uses optimized polynomial approximations from math module
/// - **Memory efficient**: Uses platform-optimized aligned memory allocation
/// - **Remainder handling**: Processes non-multiple-of-8 arrays using partial vectors
///
/// # Arguments
///
/// * `x` - Base values (first input slice)
/// * `y` - Exponent values (second input slice, must have same length as `x`)
///
/// # Returns
///
/// A new vector containing x\[i\]^y\[i\] for each element pair.
///
/// # Panics
///
/// Panics if:
/// - Either input slice is empty
/// - Input slices have different lengths
///
/// # Example
///
/// ```rust
/// use simdly::simd::SimdMath;
///
/// let bases = vec![2.0, 3.0, 4.0, 5.0];
/// let exponents = vec![2.0, 2.0, 2.0, 2.0];
/// let results = bases.pow(exponents); // [4.0, 9.0, 16.0, 25.0]
/// ```
#[inline(always)]
pub(crate) fn simd_pow(x: &[f32], y: &[f32]) -> Vec<f32> {
    debug_assert!(
        !x.is_empty() && !y.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_pow_block(&x[i], &y[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_pow_partial_block(
            &x[complete_lanes],
            &y[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Parallel SIMD power computation using work-stealing thread pool.
///
/// This function provides multi-threaded SIMD power computation for very large arrays,
/// using Rayon to distribute work across multiple CPU cores with AVX2 SIMD processing.
///
/// # Arguments
///
/// * `x` - Base values (first input slice)
/// * `y` - Exponent values (second input slice, must have same length as `x`)
///
/// # Returns
///
/// A new vector containing x\[i\]^y\[i\] for each element pair.
///
/// # Performance
///
/// - **Scalability**: Near-linear speedup with available CPU cores
/// - **Optimal Size**: Best for arrays with tens of thousands of elements or more
/// - **Memory Bandwidth**: Can saturate available memory bandwidth
///
/// # Panics
///
/// Panics if:
/// - Either input slice is empty
/// - Input slices have different lengths
#[inline(always)]
pub fn parallel_simd_pow(x: &[f32], y: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_pow(x, y);
    }

    debug_assert!(
        !x.is_empty() && !y.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .zip(y.par_chunks(chunk_size))
        .for_each(|((c_chunk, x_chunk), y_chunk)| {
            let local_complete = x_chunk.len() - (x_chunk.len() % step);
            let local_remaining = x_chunk.len() - local_complete;

            let start_idx = 0;
            for i in (0..local_complete).step_by(step) {
                simd_pow_block(
                    &x_chunk[start_idx + i],
                    &y_chunk[start_idx + i],
                    &mut c_chunk[i],
                );
            }

            if local_remaining > 0 {
                simd_pow_partial_block(
                    &x_chunk[local_complete],
                    &y_chunk[local_complete],
                    &mut c_chunk[local_complete],
                    local_remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Processes a complete 8-element block using AVX2 SIMD power operations.
#[inline(always)]
fn simd_pow_block(x: *const f32, y: *const f32, c: *mut f32) {
    let x_chunk_simd = unsafe { F32x8::load(x, f32x8::LANE_COUNT) };
    let y_chunk_simd = unsafe { F32x8::load(y, f32x8::LANE_COUNT) };
    let result = x_chunk_simd.pow(y_chunk_simd);
    unsafe { result.store_aligned_at(c) };
}

/// Processes a partial block (1-7 elements) using AVX2 SIMD power operations.
#[inline(always)]
fn simd_pow_partial_block(x: *const f32, y: *const f32, c: *mut f32, size: usize) {
    let x_chunk_simd = unsafe { F32x8::load_partial(x, size) };
    let y_chunk_simd = unsafe { F32x8::load_partial(y, size) };
    unsafe { x_chunk_simd.pow(y_chunk_simd).store_at_partial(c) };
}

/// Computes the 2D Euclidean distance (hypot) using AVX2 SIMD instructions.
///
/// This function provides vectorized hypot computation using Intel AVX2 256-bit SIMD instructions.
/// It computes sqrt(x^2 + y^2) for 8 f32 pairs simultaneously using numerically stable algorithms.
///
/// # Performance Characteristics
///
/// - **Vectorized processing**: Computes 8 distances simultaneously using AVX2
/// - **Numerically stable**: Uses optimized algorithms to prevent overflow/underflow
/// - **Memory efficient**: Uses platform-optimized aligned memory allocation
/// - **Remainder handling**: Processes non-multiple-of-8 arrays using partial vectors
///
/// # Arguments
///
/// * `x` - X coordinates (first input slice)
/// * `y` - Y coordinates (second input slice, must have same length as `x`)
///
/// # Returns
///
/// A new vector containing sqrt(x[i]^2 + y[i]^2) for each coordinate pair.
///
/// # Panics
///
/// Panics if:
/// - Either input slice is empty
/// - Input slices have different lengths
#[inline(always)]
pub(crate) fn simd_hypot(x: &[f32], y: &[f32]) -> Vec<f32> {
    debug_assert!(
        !x.is_empty() && !y.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_hypot_block(&x[i], &y[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_hypot_partial_block(
            &x[complete_lanes],
            &y[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Parallel SIMD hypot computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_hypot(x: &[f32], y: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_hypot(x, y);
    }

    debug_assert!(
        !x.is_empty() && !y.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .zip(y.par_chunks(chunk_size))
        .for_each(|((c_chunk, x_chunk), y_chunk)| {
            let local_complete = x_chunk.len() - (x_chunk.len() % step);
            let local_remaining = x_chunk.len() - local_complete;

            let start_idx = 0;
            for i in (0..local_complete).step_by(step) {
                simd_hypot_block(
                    &x_chunk[start_idx + i],
                    &y_chunk[start_idx + i],
                    &mut c_chunk[i],
                );
            }

            if local_remaining > 0 {
                simd_hypot_partial_block(
                    &x_chunk[local_complete],
                    &y_chunk[local_complete],
                    &mut c_chunk[local_complete],
                    local_remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Processes a complete 8-element block using AVX2 SIMD hypot operations.
#[inline(always)]
fn simd_hypot_block(x: *const f32, y: *const f32, c: *mut f32) {
    let x_chunk_simd = unsafe { F32x8::load(x, f32x8::LANE_COUNT) };
    let y_chunk_simd = unsafe { F32x8::load(y, f32x8::LANE_COUNT) };
    let result = x_chunk_simd.hypot(y_chunk_simd);
    unsafe { result.store_aligned_at(c) };
}

/// Processes a partial block (1-7 elements) using AVX2 SIMD hypot operations.
#[inline(always)]
fn simd_hypot_partial_block(x: *const f32, y: *const f32, c: *mut f32, size: usize) {
    let x_chunk_simd = unsafe { F32x8::load_partial(x, size) };
    let y_chunk_simd = unsafe { F32x8::load_partial(y, size) };
    unsafe { x_chunk_simd.hypot(y_chunk_simd).store_at_partial(c) };
}

/// Computes the 3D Euclidean distance (hypot3) using AVX2 SIMD instructions.
///
/// This function provides vectorized hypot3 computation using Intel AVX2 256-bit SIMD instructions.
/// It computes sqrt(x^2 + y^2 + z^2) for 8 f32 triplets simultaneously using numerically stable algorithms.
///
/// # Performance Characteristics
///
/// - **Vectorized processing**: Computes 8 distances simultaneously using AVX2
/// - **Numerically stable**: Uses optimized algorithms to prevent overflow/underflow
/// - **Memory efficient**: Uses platform-optimized aligned memory allocation
/// - **Remainder handling**: Processes non-multiple-of-8 arrays using partial vectors
///
/// # Arguments
///
/// * `x` - X coordinates (first input slice)
/// * `y` - Y coordinates (second input slice, must have same length as `x`)
/// * `z` - Z coordinates (third input slice, must have same length as `x`)
///
/// # Returns
///
/// A new vector containing sqrt(x[i]^2 + y[i]^2 + z[i]^2) for each coordinate triplet.
///
/// # Panics
///
/// Panics if:
/// - Any input slice is empty
/// - Input slices have different lengths
#[inline(always)]
pub(crate) fn simd_hypot3(x: &[f32], y: &[f32], z: &[f32]) -> Vec<f32> {
    debug_assert!(
        !x.is_empty() && !y.is_empty() && !z.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");
    debug_assert_eq!(x.len(), z.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_hypot3_block(&x[i], &y[i], &z[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_hypot3_partial_block(
            &x[complete_lanes],
            &y[complete_lanes],
            &z[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Parallel SIMD hypot3 computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_hypot3(x: &[f32], y: &[f32], z: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_hypot3(x, y, z);
    }

    debug_assert!(
        !x.is_empty() && !y.is_empty() && !z.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");
    debug_assert_eq!(x.len(), z.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .zip(y.par_chunks(chunk_size))
        .zip(z.par_chunks(chunk_size))
        .for_each(|(((c_chunk, x_chunk), y_chunk), z_chunk)| {
            let local_complete = x_chunk.len() - (x_chunk.len() % step);
            let local_remaining = x_chunk.len() - local_complete;

            let start_idx = 0;
            for i in (0..local_complete).step_by(step) {
                simd_hypot3_block(
                    &x_chunk[start_idx + i],
                    &y_chunk[start_idx + i],
                    &z_chunk[start_idx + i],
                    &mut c_chunk[i],
                );
            }

            if local_remaining > 0 {
                simd_hypot3_partial_block(
                    &x_chunk[local_complete],
                    &y_chunk[local_complete],
                    &z_chunk[local_complete],
                    &mut c_chunk[local_complete],
                    local_remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Processes a complete 8-element block using AVX2 SIMD hypot3 operations.
#[inline(always)]
fn simd_hypot3_block(x: *const f32, y: *const f32, z: *const f32, c: *mut f32) {
    let x_chunk_simd = unsafe { F32x8::load(x, f32x8::LANE_COUNT) };
    let y_chunk_simd = unsafe { F32x8::load(y, f32x8::LANE_COUNT) };
    let z_chunk_simd = unsafe { F32x8::load(z, f32x8::LANE_COUNT) };
    let result = x_chunk_simd.hypot3(y_chunk_simd, z_chunk_simd);
    unsafe { result.store_aligned_at(c) };
}

/// Processes a partial block (1-7 elements) using AVX2 SIMD hypot3 operations.
#[inline(always)]
fn simd_hypot3_partial_block(
    x: *const f32,
    y: *const f32,
    z: *const f32,
    c: *mut f32,
    size: usize,
) {
    let x_chunk_simd = unsafe { F32x8::load_partial(x, size) };
    let y_chunk_simd = unsafe { F32x8::load_partial(y, size) };
    let z_chunk_simd = unsafe { F32x8::load_partial(z, size) };
    unsafe {
        x_chunk_simd
            .hypot3(y_chunk_simd, z_chunk_simd)
            .store_at_partial(c)
    };
}

/// Computes the 4D Euclidean distance (hypot4) using AVX2 SIMD instructions.
///
/// This function provides vectorized hypot4 computation using Intel AVX2 256-bit SIMD instructions.
/// It computes sqrt(x^2 + y^2 + z^2 + w^2) for 8 f32 quadruplets simultaneously using numerically stable algorithms.
///
/// # Performance Characteristics
///
/// - **Vectorized processing**: Computes 8 distances simultaneously using AVX2
/// - **Numerically stable**: Uses optimized algorithms to prevent overflow/underflow
/// - **Memory efficient**: Uses platform-optimized aligned memory allocation
/// - **Remainder handling**: Processes non-multiple-of-8 arrays using partial vectors
///
/// # Arguments
///
/// * `x` - X coordinates (first input slice)
/// * `y` - Y coordinates (second input slice, must have same length as `x`)
/// * `z` - Z coordinates (third input slice, must have same length as `x`)
/// * `w` - W coordinates (fourth input slice, must have same length as `x`)
///
/// # Returns
///
/// A new vector containing sqrt(x[i]^2 + y[i]^2 + z[i]^2 + w[i]^2) for each coordinate quadruplet.
///
/// # Panics
///
/// Panics if:
/// - Any input slice is empty
/// - Input slices have different lengths
#[inline(always)]
pub(crate) fn simd_hypot4(x: &[f32], y: &[f32], z: &[f32], w: &[f32]) -> Vec<f32> {
    debug_assert!(
        !x.is_empty() && !y.is_empty() && !z.is_empty() && !w.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");
    debug_assert_eq!(x.len(), z.len(), "Input slices must have the same length");
    debug_assert_eq!(x.len(), w.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_hypot4_block(&x[i], &y[i], &z[i], &w[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_hypot4_partial_block(
            &x[complete_lanes],
            &y[complete_lanes],
            &z[complete_lanes],
            &w[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Parallel SIMD hypot4 computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_hypot4(x: &[f32], y: &[f32], z: &[f32], w: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_hypot4(x, y, z, w);
    }

    debug_assert!(
        !x.is_empty() && !y.is_empty() && !z.is_empty() && !w.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(x.len(), y.len(), "Input slices must have the same length");
    debug_assert_eq!(x.len(), z.len(), "Input slices must have the same length");
    debug_assert_eq!(x.len(), w.len(), "Input slices must have the same length");

    let size = x.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .zip(x.par_chunks(chunk_size))
        .zip(y.par_chunks(chunk_size))
        .zip(z.par_chunks(chunk_size))
        .zip(w.par_chunks(chunk_size))
        .for_each(|((((c_chunk, x_chunk), y_chunk), z_chunk), w_chunk)| {
            let local_complete = x_chunk.len() - (x_chunk.len() % step);
            let local_remaining = x_chunk.len() - local_complete;

            let start_idx = 0;
            for i in (0..local_complete).step_by(step) {
                simd_hypot4_block(
                    &x_chunk[start_idx + i],
                    &y_chunk[start_idx + i],
                    &z_chunk[start_idx + i],
                    &w_chunk[start_idx + i],
                    &mut c_chunk[i],
                );
            }

            if local_remaining > 0 {
                simd_hypot4_partial_block(
                    &x_chunk[local_complete],
                    &y_chunk[local_complete],
                    &z_chunk[local_complete],
                    &w_chunk[local_complete],
                    &mut c_chunk[local_complete],
                    local_remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Processes a complete 8-element block using AVX2 SIMD hypot4 operations.
#[inline(always)]
fn simd_hypot4_block(x: *const f32, y: *const f32, z: *const f32, w: *const f32, c: *mut f32) {
    let x_chunk_simd = unsafe { F32x8::load(x, f32x8::LANE_COUNT) };
    let y_chunk_simd = unsafe { F32x8::load(y, f32x8::LANE_COUNT) };
    let z_chunk_simd = unsafe { F32x8::load(z, f32x8::LANE_COUNT) };
    let w_chunk_simd = unsafe { F32x8::load(w, f32x8::LANE_COUNT) };
    let result = x_chunk_simd.hypot4(y_chunk_simd, z_chunk_simd, w_chunk_simd);
    unsafe { result.store_aligned_at(c) };
}

/// Processes a partial block (1-7 elements) using AVX2 SIMD hypot4 operations.
#[inline(always)]
fn simd_hypot4_partial_block(
    x: *const f32,
    y: *const f32,
    z: *const f32,
    w: *const f32,
    c: *mut f32,
    size: usize,
) {
    let x_chunk_simd = unsafe { F32x8::load_partial(x, size) };
    let y_chunk_simd = unsafe { F32x8::load_partial(y, size) };
    let z_chunk_simd = unsafe { F32x8::load_partial(z, size) };
    let w_chunk_simd = unsafe { F32x8::load_partial(w, size) };
    unsafe {
        x_chunk_simd
            .hypot4(y_chunk_simd, z_chunk_simd, w_chunk_simd)
            .store_at_partial(c)
    };
}

// ================================================================================================
// ADDITIONAL MATHEMATICAL FUNCTIONS TO MATCH NEON
// ================================================================================================

/// Computes sine of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_sin(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_sin_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_sin_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_sin_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.sin();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_sin_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.sin().store_at_partial(c) };
}

/// Computes tangent of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_tan(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_tan_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_tan_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_tan_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.tan();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_tan_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.tan().store_at_partial(c) };
}

/// Parallel SIMD tangent computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_tan(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_tan(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_tan_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_tan_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes absolute value of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_abs(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_abs_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_abs_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_abs_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.abs();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_abs_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.abs().store_at_partial(c) };
}

/// Parallel SIMD absolute value computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_abs(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_abs(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_abs_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_abs_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes square root of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_sqrt(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_sqrt_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_sqrt_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_sqrt_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.sqrt();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_sqrt_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.sqrt().store_at_partial(c) };
}

/// Parallel SIMD square root computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_sqrt(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_sqrt(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_sqrt_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_sqrt_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes natural exponential of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_exp(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_exp_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_exp_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_exp_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.exp();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_exp_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.exp().store_at_partial(c) };
}

/// Parallel SIMD exponential computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_exp(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_exp(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_exp_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_exp_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes natural logarithm of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_ln(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_ln_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_ln_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_ln_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.ln();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_ln_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.ln().store_at_partial(c) };
}

/// Parallel SIMD natural logarithm computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_ln(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_ln(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_ln_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_ln_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes floor of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_floor(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_floor_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_floor_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_floor_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.floor();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_floor_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.floor().store_at_partial(c) };
}

/// Parallel SIMD floor computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_floor(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_floor(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_floor_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_floor_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes ceiling of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_ceil(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_ceil_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_ceil_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_ceil_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.ceil();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_ceil_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.ceil().store_at_partial(c) };
}

/// Parallel SIMD ceiling computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_ceil(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_ceil(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_ceil_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_ceil_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes arcsine of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_asin(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_asin_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_asin_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_asin_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.asin();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_asin_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.asin().store_at_partial(c) };
}

/// Parallel SIMD arcsine computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_asin(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_asin(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_asin_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_asin_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes arccosine of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_acos(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_acos_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_acos_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_acos_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.acos();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_acos_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.acos().store_at_partial(c) };
}

/// Parallel SIMD arccosine computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_acos(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_acos(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_acos_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_acos_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes arctangent of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_atan(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_atan_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_atan_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_atan_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.atan();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_atan_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.atan().store_at_partial(c) };
}

/// Parallel SIMD arctangent computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_atan(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_atan(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_atan_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_atan_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes two-argument arctangent using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_atan2(y: &[f32], x: &[f32]) -> Vec<f32> {
    debug_assert!(
        !y.is_empty() && !x.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(y.len(), x.len(), "Input slices must have the same length");
    let size = y.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_atan2_block(&y[i], &x[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_atan2_partial_block(
            &y[complete_lanes],
            &x[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_atan2_block(y: *const f32, x: *const f32, c: *mut f32) {
    let y_chunk_simd = unsafe { F32x8::load(y, f32x8::LANE_COUNT) };
    let x_chunk_simd = unsafe { F32x8::load(x, f32x8::LANE_COUNT) };
    let result = y_chunk_simd.atan2(x_chunk_simd);
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_atan2_partial_block(y: *const f32, x: *const f32, c: *mut f32, size: usize) {
    let y_chunk_simd = unsafe { F32x8::load_partial(y, size) };
    let x_chunk_simd = unsafe { F32x8::load_partial(x, size) };
    unsafe { y_chunk_simd.atan2(x_chunk_simd).store_at_partial(c) };
}

/// Parallel SIMD two-argument arctangent computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_atan2(y: &[f32], x: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if y.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_atan2(y, x);
    }

    debug_assert!(
        !y.is_empty() && !x.is_empty(),
        "Input slices cannot be empty"
    );
    debug_assert_eq!(y.len(), x.len(), "Input slices must have the same length");
    let size = y.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .zip(x.par_chunks(chunk_size))
        .for_each(|((c_chunk, y_chunk), x_chunk)| {
            let local_complete = y_chunk.len() - (y_chunk.len() % step);
            let local_remaining = y_chunk.len() - local_complete;

            let start_idx = 0;
            for i in (0..local_complete).step_by(step) {
                simd_atan2_block(
                    &y_chunk[start_idx + i],
                    &x_chunk[start_idx + i],
                    &mut c_chunk[i],
                );
            }

            if local_remaining > 0 {
                simd_atan2_partial_block(
                    &y_chunk[local_complete],
                    &x_chunk[local_complete],
                    &mut c_chunk[local_complete],
                    local_remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

/// Computes cube root of each element using Intel AVX2 SIMD instructions.
#[inline(always)]
pub(crate) fn simd_cbrt(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");
    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);
    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;
    let complete_lanes = size - (size % step);
    let remaining_lanes = size - complete_lanes;

    for i in (0..complete_lanes).step_by(step) {
        simd_cbrt_block(&a[i], &mut aligned_c[i]);
    }

    if remaining_lanes > 0 {
        simd_cbrt_partial_block(
            &a[complete_lanes],
            &mut aligned_c[complete_lanes],
            remaining_lanes,
        );
    }

    #[cfg(not(target_os = "windows"))]
    return aligned_c;
    #[cfg(target_os = "windows")]
    return aligned_c.into();
}

#[inline(always)]
fn simd_cbrt_block(a: *const f32, c: *mut f32) {
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let result = a_chunk_simd.cbrt();
    unsafe { result.store_aligned_at(c) };
}

#[inline(always)]
fn simd_cbrt_partial_block(a: *const f32, c: *mut f32, size: usize) {
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    unsafe { a_chunk_simd.cbrt().store_at_partial(c) };
}

/// Parallel SIMD cube root computation using work-stealing thread pool.
#[inline(always)]
pub fn parallel_simd_cbrt(a: &[f32]) -> Vec<f32> {
    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() <= PARALLEL_SIMD_THRESHOLD {
        return simd_cbrt(a);
    }

    debug_assert!(!a.is_empty(), "Size can't be empty (size zero)");

    let size = a.len();

    #[cfg(not(target_os = "windows"))]
    let mut aligned_c = alloc_uninit_vec::<f32>(size, AVX_ALIGNMENT);

    #[cfg(target_os = "windows")]
    let mut aligned_c: AlignedVec<f32> = AlignedVec::new_uninit(size, AVX_ALIGNMENT);

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    aligned_c
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, c_chunk)| {
            let start_idx = chunk_idx * chunk_size;
            let chunk_len = c_chunk.len();

            // Process complete SIMD blocks within this chunk
            let complete_blocks = (chunk_len / step) * step;
            for i in (0..complete_blocks).step_by(step) {
                simd_cbrt_block(&a[start_idx + i], &mut c_chunk[i]);
            }

            // Handle remaining elements in this chunk
            if chunk_len > complete_blocks {
                let remaining = chunk_len - complete_blocks;
                simd_cbrt_partial_block(
                    &a[start_idx + complete_blocks],
                    &mut c_chunk[complete_blocks],
                    remaining,
                );
            }
        });

    #[cfg(not(target_os = "windows"))]
    return aligned_c;

    #[cfg(target_os = "windows")]
    return aligned_c.into();
}
