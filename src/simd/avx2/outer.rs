//! AVX2 SIMD outer product computation.
//!
//! This module provides efficient computation of outer products between f32 vectors
//! using AVX2 SIMD instructions. The implementation processes data in chunks of
//! LANE_COUNT elements simultaneously for optimal performance.

use rayon::prelude::*;

use crate::{
    simd::{
        avx2::f32x8::{F32x8, AVX_ALIGNMENT, LANE_COUNT},
        SimdMath, SimdShuffle, SimdStore,
    },
    utils::alloc_uninit_vec,
};

/// Computes the outer product of two F32x8 vectors using AVX2 SIMD operations.
///
/// This function performs an 8×8 outer product, producing a matrix where each element
/// `result[i][j] = a[i] * b[j]`. The implementation uses advanced SIMD techniques
/// including FMA (Fused Multiply-Add) operations and instruction interleaving to
/// maximize performance on modern CPUs.
///
/// # Algorithm Overview
///
/// The algorithm processes the outer product in two main phases:
/// 1. **Lane Preparation**: Duplicate 128-bit lanes to enable element broadcasting
/// 2. **Element Broadcasting & Computation**: Broadcast each element and compute products
///
/// # Performance Notes
///
/// - Uses FMA operations for better precision and performance
/// - Interleaves operations to maximize CPU execution unit utilization
/// - Processes all combinations in a single function call
///
/// # Arguments
/// * `a` - Left operand vector (LANE_COUNT elements)
/// * `b` - Right operand vector (LANE_COUNT elements)
///
/// # Returns
/// Array of LANE_COUNT vectors representing the outer product matrix
#[inline(always)]
pub fn simd_outer_product(a: F32x8, b: F32x8) -> [F32x8; LANE_COUNT] {
    // Initialize result array with zeros
    let mut result = [F32x8::zeros(); LANE_COUNT];

    // Duplicate 128-bit lanes to prepare for element broadcasting
    // permute2f128 with mask 0x00: [a0,a1,a2,a3, a0,a1,a2,a3]
    // permute2f128 with mask 0x11: [a4,a5,a6,a7, a4,a5,a6,a7]
    let a_lower_lane = a.permute2f128::<0x00>();
    let a_upper_lane = a.permute2f128::<0x11>();

    // First batch: broadcast elements 0 and 4, then 1 and 5
    // Interleave operations to maximize port utilization on modern CPUs
    let a0_broadcast = a_lower_lane.permute::<0x00>(); // [a0, a0, a0, a0, a0, a0, a0, a0]
    let a4_broadcast = a_upper_lane.permute::<0x00>(); // [a4, a4, a4, a4, a4, a4, a4, a4]
    let a1_broadcast = a_lower_lane.permute::<0x55>(); // [a1, a1, a1, a1, a1, a1, a1, a1]
    let a5_broadcast = a_upper_lane.permute::<0x55>(); // [a5, a5, a5, a5, a5, a5, a5, a5]

    // Compute outer products using FMA (more precise than separate mul+add)
    result[0] = result[0].fma(a0_broadcast, b);
    result[4] = result[4].fma(a4_broadcast, b);
    result[1] = result[1].fma(a1_broadcast, b);
    result[5] = result[5].fma(a5_broadcast, b);

    // Second batch: broadcast elements 2 and 6, then 3 and 7
    let a2_broadcast = a_lower_lane.permute::<0xAA>(); // [a2, a2, a2, a2, a2, a2, a2, a2]
    let a6_broadcast = a_upper_lane.permute::<0xAA>(); // [a6, a6, a6, a6, a6, a6, a6, a6]
    let a3_broadcast = a_lower_lane.permute::<0xFF>(); // [a3, a3, a3, a3, a3, a3, a3, a3]
    let a7_broadcast = a_upper_lane.permute::<0xFF>(); // [a7, a7, a7, a7, a7, a7, a7, a7]

    result[2] = result[2].fma(a2_broadcast, b);
    result[6] = result[6].fma(a6_broadcast, b);
    result[3] = result[3].fma(a3_broadcast, b);
    result[7] = result[7].fma(a7_broadcast, b);

    result
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
    /// A new iterator that yields `((usize, F32x8), (usize, F32x8))` tuples representing
    /// all possible LANE_COUNT×LANE_COUNT chunk combinations with their indices.
    ///
    /// # Performance Notes
    ///
    /// - **Memory**: O(1) space complexity, no additional allocation
    /// - **Time**: O(1) construction time, regardless of input size
    /// - **Cache**: Iterator ordering optimizes for sequential memory access
    fn new(vec_a: &'a [f32], vec_b: &'a [f32]) -> Self {
        // Calculate number of chunks using ceiling division
        let a_chunks = (vec_a.len() + LANE_COUNT - 1) / LANE_COUNT;
        let b_chunks = (vec_b.len() + LANE_COUNT - 1) / LANE_COUNT;

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
    type Item = ((usize, F32x8), (usize, F32x8));

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
    /// - F32x8::from() handles zero-padding for chunks smaller than LANE_COUNT
    /// - Chunk size information is preserved in the F32x8 struct
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
        let a_chunk = F32x8::from(&self.vec_a[a_start..a_end]);
        let b_chunk = F32x8::from(&self.vec_b[b_start..b_end]);

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

/// Computes the outer product of two f32 vectors using AVX2 SIMD optimization.
///
/// This is the main high-level API function that handles the complete outer product
/// computation, including memory allocation, chunking, and result assembly.
/// The function automatically uses SIMD optimization for LANE_COUNT×LANE_COUNT blocks
/// and handles arbitrary vector sizes with proper padding.
///
/// # Algorithm Overview
///
/// 1. **Input Validation**: Handle empty vectors early
/// 2. **Memory Allocation**: Allocate AVX-aligned result matrix
/// 3. **Chunked Processing**: Use ChunkIter to process SIMD-sized blocks
/// 4. **SIMD Computation**: Apply simd_outer_product to each chunk pair
/// 5. **Result Storage**: Store partial results with proper size handling
///
/// # Performance Characteristics
///
/// - **SIMD**: Uses AVX2 for LANE_COUNT×LANE_COUNT blocks (up to 64× speedup)
/// - **Memory**: AVX_ALIGNMENT-byte aligned allocation for optimal cache performance
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
    // Size: vec_a.len() * vec_b.len(), alignment: AVX_ALIGNMENT (32 bytes)
    let mut result = alloc_uninit_vec::<f32>(vec_a.len() * vec_b.len(), AVX_ALIGNMENT);

    // Process all chunk combinations using the iterator
    // Each iteration processes a LANE_COUNT×LANE_COUNT sub-matrix
    for ((chunk_a_idx, a_chunk), (chunk_b_idx, b_chunk)) in ChunkIter::new(vec_a, vec_b) {
        // Calculate starting positions for this chunk pair in the original vectors
        let a_start = chunk_a_idx * LANE_COUNT;
        let b_start = chunk_b_idx * LANE_COUNT;

        // Compute LANE_COUNT×LANE_COUNT outer product using SIMD
        let mut chunk_result = simd_outer_product(a_chunk, b_chunk);

        // Store results for each row in the chunk
        // Handle partial chunks by respecting actual chunk sizes
        for row_offset in 0..a_chunk.size {
            let result_row = a_start + row_offset;

            // Calculate linear index in row-major result matrix
            let storage_index = result_row * vec_b.len() + b_start;

            // Set the correct size for partial B chunks and store
            chunk_result[row_offset].set_size(b_chunk.size);
            chunk_result[row_offset].store_at(result[storage_index..].as_mut_ptr());
        }
    }

    result
}

/// Computes the outer product of two f32 vectors using AVX2 SIMD optimization.
///
/// # Arguments
/// * `vec_a` - First input vector
/// * `vec_b` - Second input vector
///
/// # Returns
/// Vector containing the outer product matrix in row-major format
pub fn par_outer(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
    if vec_a.is_empty() || vec_b.is_empty() {
        return Vec::new();
    }

    let result = alloc_uninit_vec::<f32>(vec_a.len() * vec_b.len(), AVX_ALIGNMENT);

    vec_b
        .chunks(LANE_COUNT)
        .enumerate()
        .for_each(|(a_chunk_idx, a_chunk)| {
            vec_a
                .chunks(LANE_COUNT)
                .enumerate()
                .for_each(|(b_chunk_idx, b_chunk)| {
                    let a_start = a_chunk_idx * LANE_COUNT;
                    let b_start = b_chunk_idx * LANE_COUNT;

                    let a_chunk = F32x8::from(a_chunk);
                    let b_chunk = F32x8::from(b_chunk);

                    let mut chunk_result = simd_outer_product(a_chunk, b_chunk);

                    // Store results for each row in the chunk
                    for row_offset in 0..a_chunk.size {
                        let result_row = a_start + row_offset;
                        let storage_index = result_row * vec_b.len() + b_start;

                        chunk_result[row_offset].set_size(b_chunk.size);
                        chunk_result[row_offset].store_at(result[storage_index..].as_ptr());
                    }
                })
        });

    result
}

// /// Computes the outer product of two f32 vectors using AVX2 SIMD optimization.
// ///
// /// # Arguments
// /// * `vec_a` - First input vector
// /// * `vec_b` - Second input vector
// ///
// /// # Returns
// /// Vector containing the outer product matrix in row-major format
// pub fn outer(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
//     if vec_a.is_empty() || vec_b.is_empty() {
//         return Vec::new();
//     }

//     let mut result = alloc_uninit_vec::<f32>(vec_a.len() * vec_b.len(), AVX_ALIGNMENT);

//     let a_chunks = (vec_a.len() + LANE_COUNT - 1) / LANE_COUNT;
//     let b_chunks = (vec_b.len() + LANE_COUNT - 1) / LANE_COUNT;

//     for a_chunk_idx in 0..a_chunks {
//         for b_chunk_idx in 0..b_chunks {
//             let a_start = a_chunk_idx * LANE_COUNT;
//             let a_end = (a_start + LANE_COUNT).min(vec_a.len());
//             let b_start = b_chunk_idx * LANE_COUNT;
//             let b_end = (b_start + LANE_COUNT).min(vec_b.len());

//             let a_chunk = F32x8::from(&vec_a[a_start..a_end]);
//             let b_chunk = F32x8::from(&vec_b[b_start..b_end]);

//             let mut chunk_result = simd_outer_product(a_chunk, b_chunk);

//             // Store results for each row in the chunk
//             for row_offset in 0..a_chunk.size {
//                 let result_row = a_start + row_offset;
//                 let storage_index = result_row * vec_b.len() + b_start;

//                 chunk_result[row_offset].set_size(b_chunk.size);
//                 chunk_result[row_offset].store_at(result[storage_index..].as_mut_ptr());
//             }
//         }
//     }

//     result
// }

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
