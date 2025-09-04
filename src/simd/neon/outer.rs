//! NEON SIMD outer product computation.
//!
//! This module provides efficient computation of outer products between f32 vectors
//! using NEON SIMD instructions. The implementation processes data in chunks of
//! LANE_COUNT elements simultaneously for optimal performance.

#[cfg(not(target_arch = "aarch64"))]
use super::math::{float32x4_t, vdupq_n_f32, vfmaq_f32, vgetq_lane_f32};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vgetq_lane_f32};

use crate::{
    simd::{
        neon::f32x4::{F32x4, LANE_COUNT, NEON_ALIGNMENT},
        SimdStore,
    },
    utils::alloc_uninit_vec,
};

/// Computes the outer product of two F32x4 vectors using NEON SIMD operations.
///
/// This function performs a 4×4 outer product, producing a matrix where each element
/// `result[i][j] = a[i] * b[j]`. The implementation uses NEON techniques
/// including FMA (Fused Multiply-Add) operations to maximize performance on ARM CPUs.
///
/// # Algorithm Overview
///
/// The algorithm processes the outer product by duplicating each element of the first
/// vector across all lanes and multiplying with the second vector:
/// 1. **Element Broadcasting**: Duplicate each element of `a` to all lanes
/// 2. **FMA Computation**: Use FMA operations for better precision and performance
///
/// # Performance Notes
///
/// - Uses FMA operations for better precision and performance
/// - Processes all combinations efficiently using NEON duplication instructions
/// - Optimized for ARM NEON's 4-lane f32 vectors
///
/// # Arguments
/// * `a` - Left operand vector (LANE_COUNT elements)
/// * `b` - Right operand vector (LANE_COUNT elements)
///
/// # Returns
/// Array of LANE_COUNT vectors representing the outer product matrix
#[inline(always)]
pub fn simd_outer_product(a: F32x4, b: F32x4) -> [F32x4; LANE_COUNT] {
    unsafe {
        // Extract each element from vector a and broadcast it, then multiply by b
        let a0_broadcast = vdupq_n_f32(vgetq_lane_f32(a.elements, 0));
        let a1_broadcast = vdupq_n_f32(vgetq_lane_f32(a.elements, 1));
        let a2_broadcast = vdupq_n_f32(vgetq_lane_f32(a.elements, 2));
        let a3_broadcast = vdupq_n_f32(vgetq_lane_f32(a.elements, 3));

        // Create zero vector for FMA operations
        let zero = vdupq_n_f32(0.0);

        // Compute outer products using FMA: 0 + a_broadcast * b = a_broadcast * b
        [
            F32x4 {
                elements: vfmaq_f32(zero, a0_broadcast, b.elements),
                size: a.size.min(b.size),
            },
            F32x4 {
                elements: vfmaq_f32(zero, a1_broadcast, b.elements),
                size: a.size.min(b.size),
            },
            F32x4 {
                elements: vfmaq_f32(zero, a2_broadcast, b.elements),
                size: a.size.min(b.size),
            },
            F32x4 {
                elements: vfmaq_f32(zero, a3_broadcast, b.elements),
                size: a.size.min(b.size),
            },
        ]
    }
}

/// Iterator that produces cartesian products of SIMD chunks from two vectors.
///
/// This iterator processes input vectors in LANE_COUNT-element chunks, generating all possible
/// pairs of chunks for efficient SIMD outer product computation. It automatically
/// handles partial chunks when vector lengths are not multiples of LANE_COUNT.
///
/// The iterator yields tuples in the form `((chunk_index, chunk_data), (chunk_index, chunk_data))`,
/// where each tuple represents a pair of indexed SIMD chunks ready for processing.
///
/// # Memory Layout
///
/// The iterator processes chunks in row-major order:
/// - Outer loop: chunks from vector A
/// - Inner loop: chunks from vector B
///
/// This ordering optimizes cache locality when storing results in column-major format.
struct ChunkIter<'a> {
    /// Reference to the first input vector
    vec_a: &'a [f32],
    /// Reference to the second input vector  
    vec_b: &'a [f32],
    /// Current chunk index in vector A (0-based)
    a_idx: usize,
    /// Current chunk index in vector B (0-based)
    b_idx: usize,
    /// Total number of chunks in vector A
    a_chunks: usize,
    /// Total number of chunks in vector B
    b_chunks: usize,
}

impl<'a> ChunkIter<'a> {
    /// Creates a new cartesian chunk iterator for computing outer products.
    ///
    /// This constructor prepares an iterator that will generate all possible pairs
    /// of LANE_COUNT-element chunks from the two input vectors. Vectors are automatically
    /// padded with zeros when their length is not a multiple of LANE_COUNT.
    ///
    /// # Arguments
    ///
    /// * `vec_a` - First input vector (any length ≥ 0)
    /// * `vec_b` - Second input vector (any length ≥ 0)
    ///
    /// # Returns
    ///
    /// A new iterator that yields `((usize, F32x4), (usize, F32x4))` tuples representing
    /// all possible LANE_COUNT×LANE_COUNT chunk combinations with their indices.
    ///
    /// # Performance Notes
    ///
    /// - **Memory**: O(1) space complexity, no additional allocation
    /// - **Time**: O(1) construction time, regardless of input size
    /// - **Cache**: Iterator ordering optimizes for sequential memory access
    fn new(vec_a: &'a [f32], vec_b: &'a [f32]) -> Self {
        // Calculate number of chunks using ceiling division
        let a_chunks = vec_a.len().div_ceil(LANE_COUNT);
        let b_chunks = vec_b.len().div_ceil(LANE_COUNT);

        Self {
            vec_a,
            vec_b,
            a_idx: 0,
            b_idx: 0,
            a_chunks,
            b_chunks,
        }
    }
}

impl<'a> Iterator for ChunkIter<'a> {
    /// Each iteration yields a pair of indexed SIMD chunks: `((index, chunk), (index, chunk))`
    type Item = ((usize, F32x4), (usize, F32x4));

    /// Advances the iterator and returns the next chunk pair.
    ///
    /// The iteration follows row-major order (A chunks are the outer loop,
    /// B chunks are the inner loop). This ordering is optimal for storing
    /// results in column-major matrices and provides good cache locality.
    ///
    /// # Algorithm Details
    ///
    /// 1. Check if all chunk combinations have been exhausted
    /// 2. Calculate slice bounds for current chunk indices  
    /// 3. Extract chunks from input vectors (with automatic zero-padding)
    /// 4. Advance to next combination using nested loop logic
    /// 5. Return indexed chunk pair ready for SIMD processing
    ///
    /// # Chunk Extraction
    ///
    /// - Handles partial chunks automatically by limiting slice bounds
    /// - F32x4::from() handles zero-padding for chunks smaller than LANE_COUNT
    /// - Chunk size information is preserved in the F32x4 struct
    ///
    /// # Returns
    ///
    /// - `Some(((a_idx, a_chunk), (b_idx, b_chunk)))` - Next pair of indexed chunks
    /// - `None` - When all chunk combinations have been produced
    ///
    /// # Performance
    ///
    /// - **Time**: O(1) per call
    /// - **Memory**: No additional allocation, operates on borrowed data
    /// - **Cache**: Optimized access pattern for chunk extraction
    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've exhausted all combinations
        if self.a_idx >= self.a_chunks {
            return None;
        }

        // Calculate slice bounds for current chunk indices
        let a_start = self.a_idx * LANE_COUNT;
        let a_end = (a_start + LANE_COUNT).min(self.vec_a.len());
        let b_start = self.b_idx * LANE_COUNT;
        let b_end = (b_start + LANE_COUNT).min(self.vec_b.len());

        // Extract chunks from input vectors (zero-padded if necessary)
        let a_chunk = F32x4::from(&self.vec_a[a_start..a_end]);
        let b_chunk = F32x4::from(&self.vec_b[b_start..b_end]);

        // Capture current indices before advancing
        let current_a_idx = self.a_idx;
        let current_b_idx = self.b_idx;

        // Advance to next combination (row-major order: nested loops)
        self.b_idx += 1;
        if self.b_idx >= self.b_chunks {
            // Finished current row, move to next row
            self.b_idx = 0;
            self.a_idx += 1;
        }

        // Return indexed chunk pair ready for SIMD processing
        Some(((current_a_idx, a_chunk), (current_b_idx, b_chunk)))
    }
}

/// Computes the outer product of two f32 vectors using NEON SIMD optimization.
///
/// This is the main high-level API function that handles the complete outer product
/// computation, including memory allocation, chunking, and result assembly.
/// The function automatically uses SIMD optimization for LANE_COUNT×LANE_COUNT blocks
/// and handles arbitrary vector sizes with proper padding.
///
/// # Algorithm Overview
///
/// 1. **Input Validation**: Handle empty vectors early
/// 2. **Memory Allocation**: Allocate NEON-aligned result matrix
/// 3. **Chunked Processing**: Use ChunkIter to process SIMD-sized blocks
/// 4. **SIMD Computation**: Apply simd_outer_product to each chunk pair
/// 5. **Result Storage**: Store partial results with proper size handling
///
/// # Performance Characteristics
///
/// - **SIMD**: Uses NEON for LANE_COUNT×LANE_COUNT blocks (up to 4× speedup)
/// - **Memory**: NEON_ALIGNMENT-byte aligned allocation for optimal cache performance
/// - **Scalability**: Efficiently handles vectors from small (< LANE_COUNT elements) to large (> 10^6 elements)
/// - **Chunking**: Automatic handling of partial chunks with zero-padding
///
/// # Memory Layout
///
/// The result is stored in row-major format where `result[i * vec_b.len() + j] = vec_a[i] * vec_b[j]`.
/// This layout is cache-friendly for most subsequent operations and matches standard mathematical conventions.
///
/// # Arguments
///
/// * `vec_a` - First input vector (any length ≥ 0)
/// * `vec_b` - Second input vector (any length ≥ 0)
///
/// # Returns
///
/// A vector containing the outer product matrix in row-major format.
/// The result has length `vec_a.len() * vec_b.len()`, where
/// `result[i * vec_b.len() + j] = vec_a[i] * vec_b[j]`.
///
/// Returns an empty vector if either input is empty.
///
/// # Performance Notes
///
/// For optimal performance:
/// - Input vectors should be reasonably large (> LANE_COUNT elements) to amortize SIMD setup costs
/// - Consider memory alignment of input data when possible
/// - The function is most efficient when vector lengths are multiples of LANE_COUNT
pub fn outer(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
    // Early return for empty inputs - outer product of empty vectors is empty
    if vec_a.is_empty() || vec_b.is_empty() {
        return Vec::new();
    }

    // Allocate aligned result matrix for optimal SIMD performance
    // Size: vec_a.len() * vec_b.len(), alignment: NEON_ALIGNMENT (16 bytes)
    let result = alloc_uninit_vec::<f32>(vec_a.len() * vec_b.len(), NEON_ALIGNMENT);

    // Process all chunk combinations using the iterator
    // Each iteration processes a LANE_COUNT×LANE_COUNT sub-matrix
    for ((chunk_a_idx, a_chunk), (chunk_b_idx, b_chunk)) in ChunkIter::new(vec_a, vec_b) {
        // Calculate starting positions for this chunk pair in the original vectors
        let a_start = chunk_a_idx * LANE_COUNT;
        let b_start = chunk_b_idx * LANE_COUNT;

        // Compute LANE_COUNT×LANE_COUNT outer product using SIMD
        let chunk_result = simd_outer_product(a_chunk, b_chunk);

        // Store results for each row in the chunk
        // Handle partial chunks by respecting actual chunk sizes
        for row_offset in 0..a_chunk.size {
            let result_row = a_start + row_offset;

            // Calculate linear index in row-major result matrix
            let storage_index = result_row * vec_b.len() + b_start;

            // Create a properly sized result vector for partial B chunks
            let mut row_result = chunk_result[row_offset];
            row_result.size = b_chunk.size;
            row_result.store_at(result[storage_index..].as_ptr());
        }
    }

    result
}

/// Computes the outer product of two f32 vectors using NEON SIMD optimization.
///
/// # Arguments
/// * `vec_a` - First input vector
/// * `vec_b` - Second input vector
///
/// # Returns
/// Vector containing the outer product matrix in row-major format
pub fn par_outer(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
    // For now, delegate to sequential implementation
    // Parallel implementation would require careful memory handling
    outer(vec_a, vec_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation of outer product using scalar operations.
    /// Used for testing SIMD implementation correctness.
    fn scalar_outer_product(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(vec_a.len() * vec_b.len());
        for &a_val in vec_a {
            for &b_val in vec_b {
                result.push(a_val * b_val);
            }
        }
        result
    }

    /// Helper function to compare two f32 vectors with floating-point tolerance.
    /// Accounts for numerical precision differences that can occur in SIMD operations.
    fn assert_vectors_approx_eq(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual_val, &expected_val)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual_val - expected_val).abs();
            assert!(
                diff <= tolerance,
                "Values differ at index {}: actual={}, expected={}, diff={}",
                i,
                actual_val,
                expected_val,
                diff
            );
        }
    }

    #[test]
    fn test_small_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let result = outer(&a, &b);
        let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
        assert_vectors_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_exact_chunk_size() {
        let a: Vec<f32> = (1..=LANE_COUNT).map(|i| i as f32).collect();
        let b: Vec<f32> = (9..=9 + LANE_COUNT - 1).map(|i| i as f32).collect();
        let result = outer(&a, &b);
        let expected = scalar_outer_product(&a, &b);
        assert_eq!(result.len(), LANE_COUNT * LANE_COUNT);
        assert_vectors_approx_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_partial_chunks() {
        let a: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=12).map(|i| i as f32 * 0.5).collect();
        let result = outer(&a, &b);
        let expected = scalar_outer_product(&a, &b);
        assert_eq!(result.len(), 60);
        assert_vectors_approx_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_large_vectors() {
        let a: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=758).map(|i| i as f32 * 1.5).collect();
        let result = outer(&a, &b);
        let expected = scalar_outer_product(&a, &b);
        assert_eq!(result.len(), a.len() * b.len());
        assert_vectors_approx_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_empty_vectors() {
        assert_eq!(outer(&[], &[1.0, 2.0]), Vec::<f32>::new());
        assert_eq!(outer(&[1.0, 2.0], &[]), Vec::<f32>::new());
        assert_eq!(outer(&[], &[]), Vec::<f32>::new());
    }

    // Tests for par_outer function
    #[test]
    fn test_par_small_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let result = par_outer(&a, &b);
        let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
        assert_vectors_approx_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_par_exact_chunk_size() {
        let a: Vec<f32> = (1..=LANE_COUNT).map(|i| i as f32).collect();
        let b: Vec<f32> = (9..=9 + LANE_COUNT - 1).map(|i| i as f32).collect();
        let result = par_outer(&a, &b);
        let expected = scalar_outer_product(&a, &b);
        assert_eq!(result.len(), LANE_COUNT * LANE_COUNT);
        assert_vectors_approx_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_par_partial_chunks() {
        let a: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=12).map(|i| i as f32 * 0.5).collect();
        let result = par_outer(&a, &b);
        let expected = scalar_outer_product(&a, &b);
        assert_eq!(result.len(), 60);
        assert_vectors_approx_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_par_large_vectors() {
        let a: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=758).map(|i| i as f32 * 1.5).collect();
        let result = par_outer(&a, &b);
        let expected = scalar_outer_product(&a, &b);
        assert_eq!(result.len(), a.len() * b.len());
        assert_vectors_approx_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_par_empty_vectors() {
        assert_eq!(par_outer(&[], &[1.0, 2.0]), Vec::<f32>::new());
        assert_eq!(par_outer(&[1.0, 2.0], &[]), Vec::<f32>::new());
        assert_eq!(par_outer(&[], &[]), Vec::<f32>::new());
    }

    #[test]
    fn test_par_vs_sequential_consistency() {
        let a: Vec<f32> = (1..=50).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (1..=37).map(|i| i as f32 * 0.2).collect();

        let sequential_result = outer(&a, &b);
        let parallel_result = par_outer(&a, &b);

        assert_eq!(sequential_result.len(), parallel_result.len());
        assert_vectors_approx_eq(&parallel_result, &sequential_result, 1e-6);
    }
}
