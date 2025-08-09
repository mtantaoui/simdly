use crate::{simd::slice::scalar_add, FastAdd, PARALLEL_SIMD_THRESHOLD, SIMD_THRESHOLD};

#[cfg(neon)]
use crate::simd::neon::slice::{parallel_simd_add, parallel_simd_cos, simd_add, simd_cos};

#[cfg(avx2)]
use crate::simd::avx2::slice::{parallel_simd_add, parallel_simd_cos, simd_add, simd_cos};

/// Performs intelligent element-wise addition with automatic algorithm selection.
///
/// This function automatically chooses the optimal addition strategy based on input size:
/// - **Small arrays** (< 256 elements): Uses scalar addition to avoid SIMD overhead
/// - **Medium arrays** (256 - 131,071 elements): Uses single-threaded SIMD for optimal vectorization
/// - **Large arrays** (≥ 131,072 elements): Uses parallel SIMD for maximum throughput
///
/// # Algorithm Selection Logic
///
/// The selection is based on empirically determined thresholds that balance:
/// 1. **SIMD setup overhead** vs computational benefits
/// 2. **Threading overhead** vs parallelization gains  
/// 3. **Memory hierarchy** efficiency (L1/L2/L3 cache utilization)
/// 4. **Cross-platform compatibility** (optimal for both Intel AVX2 and ARM NEON)
///
/// # Performance Characteristics
///
/// | Array Size | Strategy | Expected Speedup | Rationale |
/// |------------|----------|------------------|-----------|
/// | < 256 elements | Scalar | 1x (baseline) | SIMD overhead exceeds benefits |
/// | 256 - 131K elements | SIMD | ~4-8x | Pure vectorization gains |
/// | ≥ 131K elements | Parallel SIMD | ~4-8x × cores | Memory bandwidth + parallelization |
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice (must have same length as `a`)
///
/// # Returns
///
/// A new vector containing the element-wise sum, computed using the optimal strategy
/// for the given input size.
///
/// # Panics
///
/// Panics if:
/// - Input slices have different lengths
/// - Either input slice is empty
///
/// # Example
///
/// ```rust
/// use simdly::FastAdd;
///
/// // Small array - automatically uses scalar
/// let small_a = vec![1.0; 100];
/// let small_b = vec![2.0; 100];
/// let result = small_a.as_slice().fast_add(small_b.as_slice());
///
/// // Large array - automatically uses parallel SIMD  
/// let large_a = vec![1.0; 200_000];
/// let large_b = vec![2.0; 200_000];
/// let result = large_a.as_slice().fast_add(large_b.as_slice());
/// ```
#[inline(always)]
pub(crate) fn fast_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must be the same length");
    debug_assert!(!a.is_empty(), "Vectors cannot be empty");

    let size = a.len();

    match size {
        0..SIMD_THRESHOLD => scalar_add(a, b),
        SIMD_THRESHOLD..PARALLEL_SIMD_THRESHOLD => simd_add(a, b),
        _ => parallel_simd_add(a, b),
    }
}

/// Performs intelligent cosine computation with automatic algorithm selection.
///
/// This function automatically chooses the optimal cosine computation strategy based on input size:
/// - **Small to medium arrays** (< 131,072 elements): Uses single-threaded SIMD for optimal vectorization
/// - **Large arrays** (≥ 131,072 elements): Uses parallel SIMD for maximum throughput
///
/// # Algorithm Selection Logic
///
/// Unlike addition operations, cosine computation benefits from SIMD even at small sizes due to
/// the computational complexity of trigonometric functions. The selection is based on:
/// 1. **Mathematical complexity**: Trigonometric functions have sufficient computational intensity
/// 2. **Memory bandwidth utilization**: Parallel benefits appear at larger array sizes
/// 3. **Threading overhead**: Parallel processing overhead only justified for very large arrays
///
/// # Performance Characteristics
///
/// | Array Size | Strategy | Expected Speedup | Rationale |
/// |------------|----------|------------------|-----------|
/// | < 131K elements | SIMD | ~4-13x | Vectorization benefits immediate for math functions |
/// | ≥ 131K elements | Parallel SIMD | ~4-13x × cores | Memory bandwidth + parallelization |
///
/// # Arguments
///
/// * `a` - Input slice containing f32 values (angles in radians)
///
/// # Returns
///
/// A new vector containing the cosine of each input element, computed using the optimal strategy
/// for the given input size.
///
/// # Panics
///
/// Panics if the input slice is empty.
///
/// # Example
///
/// ```rust
/// use simdly::simd::slice::fast_cos;
///
/// // Small array - automatically uses SIMD
/// let small_angles = vec![0.0, std::f32::consts::PI / 4.0];
/// let result = fast_cos(&small_angles);
///
/// // Large array - automatically uses parallel SIMD  
/// let large_angles = vec![1.0; 200_000];
/// let result = fast_cos(&large_angles);
/// ```
#[inline(always)]
pub fn fast_cos(a: &[f32]) -> Vec<f32> {
    debug_assert!(!a.is_empty(), "Vectors cannot be empty");

    let size = a.len();

    match size {
        0..PARALLEL_SIMD_THRESHOLD => simd_cos(a),
        _ => parallel_simd_cos(a),
    }
}

/// Implementation of the `FastAdd` trait for slice addition operations.
///
/// This implementation provides intelligent algorithm selection for f32 slice addition,
/// automatically choosing between scalar, SIMD, and parallel SIMD based on input size
/// and empirically determined performance thresholds.
///
/// # Performance Strategy
///
/// The `fast_add` method uses the same intelligent selection as the standalone function:
/// - **< 256 elements**: Scalar addition (minimal overhead)
/// - **256 - 131,071 elements**: Single-threaded SIMD (vectorization benefits)  
/// - **≥ 131,072 elements**: Parallel SIMD (memory bandwidth + cores)
///
/// # Usage
///
/// ```rust
/// use simdly::FastAdd;
///
/// let a = vec![1.0; 1000];
/// let b = vec![2.0; 1000];
/// let result = a.as_slice().fast_add(b.as_slice()); // Automatically chooses optimal strategy
/// ```
impl<'b> FastAdd<&'b [f32]> for &[f32] {
    type Output = Vec<f32>;

    /// Performs intelligent element-wise addition with automatic algorithm selection.
    ///
    /// This method automatically selects the optimal addition strategy based on the size
    /// of the input arrays, providing consistently good performance across different
    /// array sizes without requiring manual algorithm selection.
    ///
    /// # Performance
    ///
    /// - **Small arrays**: Uses scalar to avoid SIMD setup overhead
    /// - **Medium arrays**: Uses SIMD for ~4-8x vectorization speedup  
    /// - **Large arrays**: Uses parallel SIMD for maximum multi-core throughput
    ///
    /// # Panics
    ///
    /// Panics if input slices have different lengths or are empty.
    #[inline(always)]
    fn fast_add(self, rhs: &'b [f32]) -> Self::Output {
        fast_add(self, rhs)
    }
}

/// Implementation of `FastAdd` for owned `Vec<f32>` with slice reference.
///
/// This implementation allows adding an owned vector with a slice reference,
/// providing convenience for mixed ownership scenarios while maintaining
/// the same intelligent algorithm selection.
impl<'b> FastAdd<&'b [f32]> for Vec<f32> {
    type Output = Vec<f32>;

    /// Performs intelligent element-wise addition between owned Vec and slice reference.
    ///
    /// Delegates to the slice implementation for actual computation, automatically
    /// selecting the optimal algorithm based on input size.
    #[inline(always)]
    fn fast_add(self, rhs: &'b [f32]) -> Self::Output {
        self.as_slice().fast_add(rhs)
    }
}

/// Implementation of `FastAdd` for owned `Vec<f32>` with another owned `Vec<f32>`.
///
/// This implementation provides intelligent algorithm selection for operations
/// between two owned vectors, which is common in mathematical computations.
impl FastAdd<Vec<f32>> for Vec<f32> {
    type Output = Vec<f32>;

    /// Performs intelligent element-wise addition between two owned vectors.
    ///
    /// Automatically selects the optimal algorithm (scalar/SIMD/parallel) based on
    /// input size, providing consistent performance without manual optimization.
    #[inline(always)]
    fn fast_add(self, rhs: Vec<f32>) -> Self::Output {
        self.as_slice().fast_add(rhs.as_slice())
    }
}
