//! Platform-agnostic scalar operations for slice data.
//!
//! This module provides scalar fallback implementations for vector operations
//! that work on any platform regardless of SIMD support. These functions serve as:
//!
//! - **Fallback implementations** when SIMD is not available
//! - **Reference implementations** for correctness testing
//! - **Performance baselines** for benchmarking SIMD optimizations
//! - **Compatibility layer** for platforms without SIMD support
//!
//! # Performance Characteristics
//!
//! - **Predictable**: Consistent performance across all platforms
//! - **Low overhead**: Minimal function call and setup costs
//! - **Cache friendly**: Sequential memory access patterns
//! - **Compiler optimized**: Leverages compiler auto-vectorization when possible
//!
//! # When to Use
//!
//! These functions are automatically used by the high-level traits when:
//! - Array size is below SIMD efficiency thresholds
//! - SIMD instructions are not available on the target CPU
//! - Debug builds where SIMD complexity isn't warranted

/// Performs element-wise addition of two f32 slices using scalar operations.
///
/// This function provides a reliable, platform-agnostic implementation of vector
/// addition that works efficiently on small datasets and serves as a fallback
/// for platforms without SIMD support.
///
/// # Arguments
///
/// * `a` - First input slice
/// * `b` - Second input slice (must be same length as `a`)
///
/// # Returns
///
/// A new `Vec<f32>` containing the element-wise sum of `a` and `b`.
///
/// # Performance
///
/// - **Time Complexity**: O(n) where n is the slice length
/// - **Space Complexity**: O(n) for the output vector
/// - **Cache Behavior**: Sequential access, cache-friendly
/// - **Compiler Optimization**: May be auto-vectorized by LLVM when beneficial
///
/// # Debug Assertions
///
/// - Validates that both input slices are non-empty
/// - Ensures both slices have the same length
///
/// # Examples
///
/// ```rust
/// use simdly::simd::slice::scalar_add;
///
/// let a = &[1.0f32, 2.0, 3.0, 4.0];
/// let b = &[5.0f32, 6.0, 7.0, 8.0];
/// let result = scalar_add(a, b);
/// assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
/// ```
#[inline(always)]
pub fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert!(
        !a.is_empty() & !b.is_empty(),
        "Size can't be empty (size zero)"
    );
    debug_assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
