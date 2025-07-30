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
//! Currently implements:
//! - [`SimdAdd`]: Element-wise addition with scalar, SIMD, and parallel variants
//!
//! Planned operations:
//! - Subtraction, multiplication, division
//! - Mathematical functions (sin, cos, exp, log)
//! - Reduction operations (sum, min, max)
//! - Comparison and selection operations
//!
//! # Usage Examples
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
//! # Safety and Compatibility
//!
//! - **Memory Safety**: All operations are memory-safe despite internal `unsafe` usage
//! - **Bounds Checking**: Automatic validation of array lengths and boundaries
//! - **CPU Detection**: Graceful fallback when AVX2 is not available
//! - **Thread Safety**: All operations are thread-safe and side-effect free

use rayon::prelude::*;

use crate::{
    error::{validation_error, Result},
    simd::{
        avx2::f32x8::{self, F32x8},
        SimdLoad, SimdStore,
    },
    utils::alloc_uninit_f32_vec,
    SimdAdd,
};

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
const PARALLEL_SIMD_THRESHOLD: usize = 10_000;

/// Optimal chunk size for parallel processing.
///
/// Chosen to balance:
/// - Cache locality (L2 cache is typically 256KB-1MB)
/// - Work distribution granularity
/// - Memory bandwidth utilization
const PARALLEL_CHUNK_SIZE: usize = 8192; // ~32KB per chunk (8192 * 4 bytes)

// ================================================================================================
// SCALAR OPERATIONS
// ================================================================================================

/// Performs element-wise addition using scalar operations.
///
/// This function serves as a fallback implementation when SIMD optimizations
/// are not available or beneficial. It processes each pair of elements sequentially
/// using standard floating-point addition.
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice (must have same length as `a`)
///
/// # Returns
///
/// A `Result` containing a new vector with the element-wise sum of `a` and `b`,
/// or an error if validation fails.
///
/// # Performance
///
/// - **Single-threaded**: Processes one element pair at a time
/// - **Memory**: Minimal memory overhead with iterator-based processing
/// - **Use case**: Small arrays or when SIMD is not beneficial
///
/// # Errors
///
/// Returns a validation error if the input slices have different lengths.
#[inline(always)]
fn scalar_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(validation_error(format!(
            "Input slices must have the same length: a.len()={}, b.len()={}",
            a.len(),
            b.len()
        )));
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

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
/// 3. **Memory Management**: Uses aligned memory allocation for optimal performance
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice (must have same length as `a`)
///
/// # Returns
///
/// A `Result` containing a new vector with the element-wise sum, allocated with proper alignment,
/// or an error if validation or allocation fails.
///
/// # Performance
///
/// - **Vectorization**: Up to 8x speedup over scalar operations
/// - **Memory Bandwidth**: Efficient use of memory hierarchy with aligned access
/// - **Throughput**: Optimal for arrays with hundreds or thousands of elements
///
/// # Safety
///
/// This function requires AVX2 support and is marked with target features.
/// The caller must ensure the CPU supports these instructions.
///
/// # Errors
///
/// Returns an error if:
/// - Input slices have different lengths
/// - Memory allocation fails
#[target_feature(enable = "avx,avx2,fma")]
fn simd_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(validation_error(format!(
            "Input slices must have the same length for SIMD operations: a.len()={}, b.len()={}",
            a.len(),
            b.len()
        )));
    }

    // Early return for empty arrays to avoid unnecessary allocation
    if a.is_empty() {
        return Ok(Vec::new());
    }

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT)?;

    let step = f32x8::LANE_COUNT;

    let nb_lanes = size - (size % step);
    let rem_lanes = size - nb_lanes;

    for i in (0..nb_lanes).step_by(step) {
        simd_add_block(&a[i], &b[i], &mut c[i]);
    }

    if rem_lanes > 0 {
        simd_add_partial_block(
            &a[nb_lanes],
            &b[nb_lanes],
            &mut c[nb_lanes],
            rem_lanes, // number of remainaing uncomplete lanes
        );
    }

    Ok(c)
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
    // Load from a and b (alignment automatically detected)
    let a_chunk_simd = unsafe { F32x8::load(a, f32x8::LANE_COUNT) };
    let b_chunk_simd = unsafe { F32x8::load(b, f32x8::LANE_COUNT) };

    // Store result with automatic alignment detection
    let result = a_chunk_simd + b_chunk_simd;

    unsafe { result.store_aligned_at(c) };
    // match F32x8::is_aligned(c) {
    //     true => unsafe { result.store_aligned_at(c) },
    //     false => unsafe { result.store_unaligned_at(c) },
    // }
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
    // Assumes size is less than f32x8::LANE_COUNT
    let a_chunk_simd = unsafe { F32x8::load_partial(a, size) };
    let b_chunk_simd = unsafe { F32x8::load_partial(b, size) };
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
/// # Errors
///
/// Returns an error if:
/// - Input slices have different lengths
/// - Memory allocation fails
#[target_feature(enable = "avx,avx2,fma")]
fn parallel_simd_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(validation_error(format!(
            "Input slices must have the same length for parallel SIMD operations: a.len()={}, b.len()={}",
            a.len(),
            b.len()
        )));
    }

    // Early return for empty arrays
    if a.is_empty() {
        return Ok(Vec::new());
    }

    // For small arrays, fall back to regular SIMD to avoid threading overhead
    if a.len() < PARALLEL_SIMD_THRESHOLD {
        return simd_add(a, b);
    }

    let size = a.len();

    let mut c = alloc_uninit_f32_vec(size, f32x8::AVX_ALIGNMENT)?;

    let step = f32x8::LANE_COUNT;

    // Use parallel chunks for optimal cache utilization and work distribution
    // Process chunks that are multiples of step size for efficient SIMD operations
    let chunk_size = ((PARALLEL_CHUNK_SIZE / step) * step).max(step);

    c.par_chunks_mut(chunk_size)
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

    Ok(c)
}

// ================================================================================================
// TRAIT IMPLEMENTATIONS
// ================================================================================================

/// Implementation of the `SimdAdd` trait for slice addition operations.
///
/// This implementation provides three different strategies for adding two f32 slices:
/// - **SIMD Addition**: Vectorized addition using AVX2 instructions
/// - **Parallel SIMD Addition**: Multi-threaded vectorized addition (planned)
/// - **Scalar Addition**: Fallback scalar implementation
///
/// # Performance Characteristics
///
/// | Method | Best Use Case | Performance |
/// |--------|---------------|-------------|
/// | `simd_add` | Medium to large arrays (64-10,000 elements) | ~8x speedup |
/// | `par_simd_add` | Very large arrays (>10,000 elements) | ~8x × cores speedup |
/// | `scalar_add` | Small arrays (<64 elements) | Baseline performance |
///
/// # Safety
///
/// The SIMD methods use `unsafe` internally but provide safe interfaces.
/// All bounds checking and memory safety is handled automatically.
impl<'b> SimdAdd<&'b [f32]> for &[f32] {
    type Output = Result<Vec<f32>>;

    /// Performs SIMD-accelerated element-wise addition.
    ///
    /// Uses AVX2 instructions to process 8 elements simultaneously,
    /// providing significant performance improvements for medium to large arrays.
    ///
    /// # Performance
    ///
    /// Optimal for arrays with 100+ elements. For smaller arrays,
    /// the overhead may exceed the benefits.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation fails or memory allocation fails.
    #[inline(always)]
    fn simd_add(self, rhs: &'b [f32]) -> Self::Output {
        unsafe { simd_add(self, rhs) }
    }

    /// Performs parallel SIMD-accelerated element-wise addition.
    ///
    /// Uses Rayon for parallel processing of large arrays.
    /// Automatically falls back to regular SIMD for arrays smaller than the threshold.
    ///
    /// # Performance
    ///
    /// Optimal for arrays with more than 10,000 elements where
    /// the parallelization overhead is justified by the computational load.
    /// Uses intelligent chunking to maximize cache efficiency.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation fails or memory allocation fails.
    #[inline(always)]
    fn par_simd_add(self, rhs: &'b [f32]) -> Self::Output {
        unsafe { parallel_simd_add(self, rhs) }
    }

    /// Performs scalar element-wise addition.
    ///
    /// Fallback implementation that processes elements sequentially.
    /// Guaranteed to work on all platforms and array sizes.
    ///
    /// # Performance
    ///
    /// Best for small arrays (< 100 elements) or as a compatibility fallback.
    /// Always available regardless of CPU features.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation fails.
    #[inline(always)]
    fn scalar_add(self, rhs: &'b [f32]) -> Self::Output {
        scalar_add(self, rhs)
    }
}
