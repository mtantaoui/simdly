//! High-performance matrix multiplication using BLIS-style algorithm with AVX2 SIMD.
//!
//! This module implements cache-conscious matrix multiplication that follows the BLIS
//! (BLAS-like Library Instantiation Software) approach with hierarchical blocking
//! and AVX2 SIMD microkernels for maximum performance on x86_64 architectures.

use std::cmp::min;

use crate::simd::avx2::{
    kernels::{
        // kernel_24x8, kernel_32x8, kernel_8x2, kernel_8x6,
        kernel_8x8,
    },
    panels::{at, pack_a, pack_b, KC, MR, NR},
};

/// High-performance matrix multiplication using BLIS-style algorithm with AVX2 SIMD optimization.
///
/// This function implements a cache-conscious matrix multiplication algorithm that follows the
/// BLIS (BLAS-like Library Instantiation Software) design. It uses hierarchical blocking
/// for different cache levels and AVX2 SIMD microkernels for maximum performance.
///
/// # Algorithm Structure
///
/// The algorithm uses nested loops with cache-conscious blocking:
/// ```text
/// for jc in (0..n).step_by(nc):          // L3 cache blocking (N dimension)
///     for pc in (0..k).step_by(KC):      // L1 cache blocking (K dimension)  
///         pack_b(B[pc:pc+KC, jc:jc+nc])  // Pack B into row-major panels
///         for ic in (0..m).step_by(mc):  // L2 cache blocking (M dimension)
///             pack_a(A[ic:ic+mc, pc:pc+KC])  // Pack A into column-major panels
///             for jr in b_panels:        // Panel-level loops
///                 for ir in a_panels:    
///                     kernel_8x8()       // AVX2 microkernel
/// ```
///
/// # Performance Features
///
/// - **Cache Optimization**: Multi-level blocking for L1, L2, and L3 caches
/// - **SIMD Acceleration**: 8×8 AVX2 microkernels with FMA operations
/// - **Memory Packing**: Data reorganization for optimal memory access patterns
/// - **Register Tiling**: Minimizes memory traffic during computation
///
/// # Arguments
///
/// * `a` - Input matrix A in column-major format (m×k elements)
/// * `b` - Input matrix B in column-major format (k×n elements)  
/// * `c` - Output matrix C in column-major format (m×n elements, may contain initial values)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B (shared dimension)
/// * `mc` - Block size for M dimension (L2 cache blocking parameter)
/// * `nc` - Block size for N dimension (L3 cache blocking parameter)
///
/// # Panics
///
/// This function will panic if:
/// - Matrix dimensions don't match: `a.len() != m*k`, `b.len() != k*n`, or `c.len() != m*n`
/// - Any dimension is 0 (handled gracefully with early return)
///
/// # Performance Notes
///
/// For optimal performance:
/// - Use `mc` values around 64-128 (L2 cache blocking)
/// - Use `nc` values around 256-512 (L3 cache blocking)  
/// - Ensure matrices are large enough to amortize blocking overhead
/// - Consider matrix alignment for best SIMD performance
///
/// # Safety
///
/// This function uses unsafe code internally for SIMD operations but maintains memory safety
/// through careful bounds checking and proper slice management.
pub fn matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    mc: usize,
    nc: usize,
) {
    // Handle edge cases
    if m == 0 || n == 0 || k == 0 {
        return; // Nothing to compute
    }

    // Verify input dimensions
    assert_eq!(a.len(), m * k, "Matrix A has incorrect dimensions");
    assert_eq!(b.len(), k * n, "Matrix B has incorrect dimensions");
    assert_eq!(c.len(), m * n, "Matrix C has incorrect dimensions");

    // jc loop: Process N dimension in nc-wide blocks (L3 cache optimization)
    for jc in (0..n).step_by(nc) {
        let nc_actual = min(nc, n - jc);

        // pc loop: Process K dimension in KC-wide blocks
        // Placed outside ic loop to reuse packed B across multiple A blocks
        for pc in (0..k).step_by(KC) {
            let kc_actual = min(KC, k - pc);

            // Pack B block: B(pc:pc+kc_actual-1, jc:jc+nc_actual-1) → row-major panels
            let b_block = pack_b::<KC, NR>(b, nc_actual, kc_actual, k, pc, jc);

            // ic loop: Process M dimension in mc-wide blocks (L2 cache optimization)
            for ic in (0..m).step_by(mc) {
                let mc_actual = min(mc, m - ic);

                // Pack A block: A(ic:ic+mc_actual-1, pc:pc+kc_actual-1) → column-major panels
                let a_block = pack_a::<MR, KC>(a, mc_actual, kc_actual, m, ic, pc);

                // jr and ir loops: Process packed panels with MR×NR microkernel
                for jr in 0..b_block.as_panels().len() {
                    let b_panel = &b_block[jr];
                    let nr = min(NR, nc_actual - jr * NR);

                    for ir in 0..a_block.as_panels().len() {
                        let a_panel = &a_block[ir];
                        let mr = min(MR, mc_actual - ir * MR);

                        // Compute C(ic+ir*MR:ic+ir*MR+mr-1, jc+jr*NR:jc+jr*NR+nr-1)
                        let c_row = ic + ir * MR;
                        let c_col = jc + jr * NR;
                        let c_micropanel_start_idx = at(c_row, c_col, m);
                        let c_micropanel = &mut c[c_micropanel_start_idx..];

                        // Execute MR×NR microkernel with AVX2 optimization
                        unsafe {
                            // Use the stable 8×8 kernel for all panel sizes
                            // The kernel internally handles partial panels (mr < 8, nr < 8)
                            // by using masked loads/stores and size information
                            kernel_8x8(
                                a_panel,
                                b_panel,
                                c_micropanel.as_mut_ptr(),
                                mr,
                                nr,
                                kc_actual,
                                m,
                            )

                            
                            // EXPERIMENTAL KERNELS (currently disabled):
                            // Alternative kernel implementations for different panel sizes
                            // and register usage patterns. These may provide better performance
                            // for specific workloads but have known issues (see kernels.rs docs):
                            //
                            // - kernel_8x4, kernel_8x2, kernel_8x6: Optimized for narrow B panels
                            // - kernel_16x8: Double-width A panels (moderate register pressure)  
                            // - kernel_24x8, kernel_32x8: Very wide A panels (HIGH register pressure + memory bugs)
                            //
                            // TODO: Fix memory access bugs in 24×8 and 32×8 kernels
                            // TODO: Benchmark against kernel_8x8 for performance comparison
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::alloc_zeroed_vec;

    use super::*;

    const MC: usize = MR * 8;
    const NC: usize = NR * 8;

    /// Naive O(mnk) matrix multiplication for correctness verification.
    /// Computes C += A × B using the standard triple-nested loop algorithm.
    fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        // Standard IJK algorithm: C[i,j] += ∑(A[i,k] * B[k,j])
        for j in 0..n {
            // For each column of C
            for i in 0..m {
                // For each row of C
                let mut sum = 0.0;
                for l in 0..k {
                    // Inner product
                    sum += a[at(i, l, m)] * b[at(l, j, k)];
                }
                c[at(i, j, m)] += sum; // Accumulate into C[i,j]
            }
        }
    }

    /// Creates test matrix with values (row+1) + (col+1)*0.1 for easy verification.
    /// Examples: A[0,0]=1.1, A[1,0]=2.1, A[0,1]=1.2, A[1,1]=2.2
    fn create_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        // let mut matrix = vec![0.0; rows * cols];
        let mut matrix = alloc_zeroed_vec(rows * cols, 32);
        for col in 0..cols {
            for row in 0..rows {
                let idx = at(row, col, rows);
                matrix[idx] = (row + 1) as f32 + (col + 1) as f32 * 0.1;
            }
        }
        matrix
    }

    /// Creates identity matrix: I[i,j] = 1 if i==j, 0 otherwise.
    fn create_identity_matrix(size: usize) -> Vec<f32> {
        // let mut matrix = vec![0.0; size * size];
        let mut matrix = alloc_zeroed_vec(size * size, 32);

        for i in 0..size {
            let idx = at(i, i, size);
            matrix[idx] = 1.0;
        }
        matrix
    }

    /// Verifies the at() function correctly implements column-major indexing.
    #[test]
    fn test_column_major_indexing() {
        // Column-major storage: [col0, col1] = [a,b,c, d,e,f]
        // Matrix layout:  | a d |
        //                 | b e |
        //                 | c f |
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 3; // rows
        let _n = 2; // cols

        assert_eq!(matrix[at(0, 0, m)], 1.0); // top-left
        assert_eq!(matrix[at(1, 0, m)], 2.0); // middle-left
        assert_eq!(matrix[at(2, 0, m)], 3.0); // bottom-left
        assert_eq!(matrix[at(0, 1, m)], 4.0); // top-right
        assert_eq!(matrix[at(1, 1, m)], 5.0); // middle-right
        assert_eq!(matrix[at(2, 1, m)], 6.0); // bottom-right
    }

    /// Test matrix multiplication with identity matrix
    #[test]
    fn test_matmul_identity() {
        let size = 4;
        let a = create_test_matrix(size, size);
        let identity = create_identity_matrix(size);
        let mut c_optimized = vec![0.0; size * size];
        let mut c_naive = vec![0.0; size * size];

        // A * I = A
        matmul(&a, &identity, &mut c_optimized, size, size, size, MC, NC);
        naive_matmul(&a, &identity, &mut c_naive, size, size, size);

        // Verify both implementations match
        for i in 0..size * size {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "Mismatch at index {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }

        // Verify A * I = A
        for i in 0..size * size {
            assert!(
                (c_optimized[i] - a[i]).abs() < 1e-6,
                "A*I != A at index {}: result={}, original={}",
                i,
                c_optimized[i],
                a[i]
            );
        }
    }

    /// Tests 2×2 multiplication with manually computed expected result.
    #[test]
    fn test_matmul_2x2() {
        // A = | 1 3 |   B = | 5 7 |   C = | 1*5+3*6  1*7+3*8 | = | 23 31 |
        //     | 2 4 |       | 6 8 |       | 2*5+4*6  2*7+4*8 |   | 34 46 |

        // Column-major: A=[1,2, 3,4] (col0=[1,2], col1=[3,4])
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_optimized = vec![0.0; 4];
        let mut c_naive = vec![0.0; 4];
        let m = 2;
        let n = 2;
        let k = 2;

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        // Expected C in column-major: [23,34, 31,46] (col0=[23,34], col1=[31,46])
        let expected = vec![23.0, 34.0, 31.0, 46.0];

        for i in 0..4 {
            assert!(
                (c_optimized[i] - expected[i]).abs() < 1e-6,
                "Optimized mismatch at {}: got {}, expected {}",
                i,
                c_optimized[i],
                expected[i]
            );
            assert!(
                (c_naive[i] - expected[i]).abs() < 1e-6,
                "Naive mismatch at {}: got {}, expected {}",
                i,
                c_naive[i],
                expected[i]
            );
        }
    }

    /// Test with 3x3 matrices
    #[test]
    fn test_matmul_3x3() {
        let m = 3;
        let n = 3;
        let k = 3;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "3x3 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Test non-square matrices
    #[test]
    fn test_matmul_non_square() {
        // Test 2x3 * 3x4 = 2x4
        let m = 2;
        let k = 3;
        let n = 4;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "2x3*3x4 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Tests matrices that are multiples of blocking parameters (MC,NC,KC).
    #[test]
    fn test_matmul_large_blocks() {
        let m = 16;
        let n = 16;
        let k = 16; // Multiples of MR, NR, KC
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-3,
                "16x16 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Tests dimensions that don't align with blocking parameters.
    #[test]
    fn test_matmul_odd_dimensions() {
        let m = 7;
        let n = 5;
        let k = 9; // Not multiples of MR=8, NR=8, KC=16
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "7x9*9x5 mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Test with zeros
    #[test]
    fn test_matmul_zeros() {
        let m = 4;
        let n = 4;
        let k = 4;
        let a = vec![0.0; m * k];
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "Zero matrix mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
            assert!(
                c_optimized[i].abs() < 1e-6,
                "Result should be zero at {}: {}",
                i,
                c_optimized[i]
            );
        }
    }

    /// Verifies accumulation behavior by pre-initializing C with non-zero values.
    #[test]
    fn test_matmul_accumulation() {
        let m = 3;
        let n = 3;
        let k = 3;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![1.0; m * n]; // Pre-filled with ones
        let mut c_naive = vec![1.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                "Accumulation mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Test 1x1 matrix multiplication
    #[test]
    fn test_matmul_1x1() {
        let a = vec![3.0];
        let b = vec![4.0];
        let mut c_optimized = vec![0.0; 1];
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 1, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 1);

        assert!(
            (c_optimized[0] - 12.0).abs() < 1e-6,
            "1x1 result should be 12.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "1x1 optimized vs naive mismatch"
        );
    }

    /// Tests matrices smaller than microkernel dimensions.
    #[test]
    fn test_matmul_small_matrices() {
        // 1×2 × 2×1 → 1×1
        let a = vec![1.0, 2.0]; // 1x2 in column-major: [1, 2]
        let b = vec![3.0, 4.0]; // 2x1 in column-major: [3, 4]
        let mut c_optimized = vec![0.0; 1]; // 1x1 result
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 2, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 2);

        // Expected: 1*3 + 2*4 = 11
        assert!(
            (c_optimized[0] - 11.0).abs() < 1e-6,
            "1x2*2x1 result should be 11.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "1x2*2x1 optimized vs naive mismatch"
        );

        // 2×1 × 1×2 → 2×2
        let a = vec![1.0, 2.0]; // 2x1 in column-major: [1, 2]
        let b = vec![3.0, 4.0]; // 1x2 in column-major: [3, 4]
        let mut c_optimized = vec![0.0; 4]; // 2x2 result
        let mut c_naive = vec![0.0; 4];

        matmul(&a, &b, &mut c_optimized, 2, 2, 1, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 2, 2, 1);

        // Expected: [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
        // In column-major: [3, 6, 4, 8]
        let expected = vec![3.0, 6.0, 4.0, 8.0];
        for i in 0..4 {
            assert!(
                (c_optimized[i] - expected[i]).abs() < 1e-6,
                "2x1*1x2 mismatch at {}: got {}, expected {}",
                i,
                c_optimized[i],
                expected[i]
            );
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "2x1*1x2 optimized vs naive mismatch at {}",
                i
            );
        }
    }

    /// Test empty matrices
    #[test]
    fn test_matmul_empty() {
        // Test 0x0 matrices
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c_optimized: Vec<f32> = vec![];
        let mut c_naive: Vec<f32> = vec![];

        matmul(&a, &b, &mut c_optimized, 0, 0, 0, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 0, 0, 0);

        assert!(c_optimized.is_empty(), "0x0 result should be empty");
        assert_eq!(c_optimized, c_naive, "0x0 optimized vs naive should match");

        // Test matrices with one dimension being 0
        let a = vec![1.0, 2.0]; // 2x1
        let b: Vec<f32> = vec![]; // 1x0
        let mut c_optimized: Vec<f32> = vec![]; // 2x0
        let mut c_naive: Vec<f32> = vec![];

        matmul(&a, &b, &mut c_optimized, 2, 0, 1, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 2, 0, 1);

        assert!(c_optimized.is_empty(), "2x0 result should be empty");
        assert_eq!(c_optimized, c_naive, "2x0 optimized vs naive should match");
    }

    /// Tests very tall/thin matrices that stress the blocking algorithm.
    #[test]
    fn test_matmul_extreme_aspect_ratios() {
        // Tall matrix: 8×1 × 1×1 → 8×1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8x1
        let b = vec![2.0]; // 1x1
        let mut c_optimized = vec![0.0; 8]; // 8x1
        let mut c_naive = vec![0.0; 8];

        matmul(&a, &b, &mut c_optimized, 8, 1, 1, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 8, 1, 1);

        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        for i in 0..8 {
            assert!(
                (c_optimized[i] - expected[i]).abs() < 1e-6,
                "8x1*1x1 mismatch at {}: got {}, expected {}",
                i,
                c_optimized[i],
                expected[i]
            );
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-6,
                "8x1*1x1 optimized vs naive mismatch at {}",
                i
            );
        }

        // Wide matrix: 1×8 × 8×1 → 1×1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1x8
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 8x1
        let mut c_optimized = vec![0.0; 1]; // 1x1
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 8, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 8);

        // Expected: sum of 1+2+3+4+5+6+7+8 = 36
        assert!(
            (c_optimized[0] - 36.0).abs() < 1e-6,
            "1x8*8x1 result should be 36.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "1x8*8x1 optimized vs naive mismatch"
        );
    }

    /// Tests numerical stability with very small and large values.
    #[test]
    fn test_matmul_special_values() {
        // Test with very small numbers
        let a = vec![1e-10, 2e-10];
        let b = vec![3e10, 4e10];
        let mut c_optimized = vec![0.0; 1];
        let mut c_naive = vec![0.0; 1];

        matmul(&a, &b, &mut c_optimized, 1, 1, 2, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, 1, 1, 2);

        // Expected: 1e-10 * 3e10 + 2e-10 * 4e10 = 3 + 8 = 11
        assert!(
            (c_optimized[0] - 11.0).abs() < 1e-6,
            "Small numbers result should be 11.0"
        );
        assert!(
            (c_optimized[0] - c_naive[0]).abs() < 1e-6,
            "Small numbers optimized vs naive mismatch"
        );
    }

    /// Informal performance comparison between optimized and naive versions.
    #[test]
    fn test_matmul_performance() {
        let m = 32;
        let n = 32;
        let k = 32;
        let a = create_test_matrix(m, k);
        let b = create_test_matrix(k, n);
        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        let start = std::time::Instant::now();
        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        let optimized_time = start.elapsed();

        let start = std::time::Instant::now();
        naive_matmul(&a, &b, &mut c_naive, m, n, k);
        let naive_time = start.elapsed();

        println!(
            "32x32 matmul - Optimized: {:?}, Naive: {:?}",
            optimized_time, naive_time
        );

        // Verify correctness
        for i in 0..m * n {
            assert!(
                (c_optimized[i] - c_naive[i]).abs() < 1e-2,
                "Performance test mismatch at {}: optimized={}, naive={}",
                i,
                c_optimized[i],
                c_naive[i]
            );
        }
    }

    /// Tests matrices smaller than MR×NR to verify edge case handling.
    #[test]
    fn test_matmul_very_small_matrices() {
        let test_cases = vec![
            (1, 1, 1, "1x1x1"),
            (2, 2, 2, "2x2x2"),
            (3, 4, 2, "3x2x4"),
            (5, 3, 7, "5x7x3"),
        ];

        for (m, k, n, name) in test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
            naive_matmul(&a, &b, &mut c_naive, m, n, k);

            for i in 0..(m * n) {
                assert!(
                    (c_optimized[i] - c_naive[i]).abs() < 1e-4,
                    "{} mismatch at {}: optimized={}, naive={}",
                    name,
                    i,
                    c_optimized[i],
                    c_naive[i]
                );
            }
        }
    }

    /// Tests matrices near the microkernel dimensions MR=8, NR=8.
    #[test]
    fn test_matmul_kernel_sized_matrices() {
        let test_cases = vec![
            (8, 8, 8, "8x8x8 exact"),
            (16, 16, 16, "16x16x16 double"),
            (7, 9, 8, "7x8x9 near"),
            (15, 17, 13, "15x13x17 odd"),
        ];

        for (m, k, n, name) in test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
            naive_matmul(&a, &b, &mut c_naive, m, n, k);

            for i in 0..(m * n) {
                assert!(
                    (c_optimized[i] - c_naive[i]).abs() < 1e-3,
                    "{} mismatch at {}: optimized={}, naive={}",
                    name,
                    i,
                    c_optimized[i],
                    c_naive[i]
                );
            }
        }
    }

    /// Test with exact kernel sizes to isolate kernel issues
    #[test]
    fn test_matmul_8x8() {
        let m = 8;
        let n = 8;
        let k = 8;

        // Create simple test matrices for easy verification
        let mut a = vec![0.0; m * k];
        let mut b = vec![0.0; k * n];

        // Fill A with row index + 1
        for i in 0..m {
            for j in 0..k {
                a[at(i, j, m)] = (i + 1) as f32;
            }
        }

        // Fill B with column index + 1
        for i in 0..k {
            for j in 0..n {
                b[at(i, j, k)] = (j + 1) as f32;
            }
        }

        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..(m * n) {
            let rel_err = if c_naive[i].abs() > 1e-6 {
                (c_optimized[i] - c_naive[i]).abs() / c_naive[i].abs()
            } else {
                (c_optimized[i] - c_naive[i]).abs()
            };

            assert!(
                rel_err < 1e-5,
                "8x8 mismatch at {}: optimized={}, naive={}, rel_err={}",
                i,
                c_optimized[i],
                c_naive[i],
                rel_err
            );
        }
    }

    /// Test with 16x16 matrices
    #[test]
    fn test_matmul_16x16() {
        let m = 16;
        let n = 16;
        let k = 16;

        // Create test matrices with predictable patterns
        let mut a = vec![0.0; m * k];
        let mut b = vec![0.0; k * n];

        // Fill A with row index + column index
        for i in 0..m {
            for j in 0..k {
                a[at(i, j, m)] = (i + j + 1) as f32;
            }
        }

        // Fill B with a different pattern
        for i in 0..k {
            for j in 0..n {
                b[at(i, j, k)] = (i * 2 + j + 1) as f32;
            }
        }

        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        for i in 0..(m * n) {
            let rel_err = if c_naive[i].abs() > 1e-6 {
                (c_optimized[i] - c_naive[i]).abs() / c_naive[i].abs()
            } else {
                (c_optimized[i] - c_naive[i]).abs()
            };

            assert!(
                rel_err < 1e-5,
                "16x16 mismatch at {}: optimized={}, naive={}, rel_err={}",
                i,
                c_optimized[i],
                c_naive[i],
                rel_err
            );
        }
    }

    /// Test with 64x64 matrices
    #[test]
    fn test_matmul_64x64() {
        let m = 64;
        let n = 64;
        let k = 64;

        // Create test matrices with simple patterns for easier debugging
        let mut a = vec![0.0; m * k];
        let mut b = vec![0.0; k * n];

        // Fill A with row index
        for i in 0..m {
            for j in 0..k {
                a[at(i, j, m)] = (i % 8 + 1) as f32; // Cycle through 1-8
            }
        }

        // Fill B with column index
        for i in 0..k {
            for j in 0..n {
                b[at(i, j, k)] = (j % 8 + 1) as f32; // Cycle through 1-8
            }
        }

        let mut c_optimized = vec![0.0; m * n];
        let mut c_naive = vec![0.0; m * n];

        matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
        naive_matmul(&a, &b, &mut c_naive, m, n, k);

        let mut mismatches = 0;
        let mut first_mismatch_idx = None;

        for i in 0..(m * n) {
            let rel_err = if c_naive[i].abs() > 1e-6 {
                (c_optimized[i] - c_naive[i]).abs() / c_naive[i].abs()
            } else {
                (c_optimized[i] - c_naive[i]).abs()
            };

            if rel_err >= 1e-5 {
                mismatches += 1;
                if first_mismatch_idx.is_none() {
                    first_mismatch_idx = Some(i);
                }
            }
        }

        if let Some(idx) = first_mismatch_idx {
            let rel_err = if c_naive[idx].abs() > 1e-6 {
                (c_optimized[idx] - c_naive[idx]).abs() / c_naive[idx].abs()
            } else {
                (c_optimized[idx] - c_naive[idx]).abs()
            };

            panic!(
                "64x64 has {} mismatches, first at {}: optimized={}, naive={}, rel_err={}",
                mismatches, idx, c_optimized[idx], c_naive[idx], rel_err
            );
        }
    }

    /// Tests performance on larger matrices where cache blocking matters.
    #[test]
    fn test_matmul_large_matrices() {
        let test_cases = vec![
            (64, 64, 64, "64x64x64"),
            (128, 128, 128, "128x128x128"),
            (100, 75, 50, "100x50x75 non-square"),
        ];

        for (m, k, n, name) in test_cases {
            let a = create_test_matrix(m, k);
            let b = create_test_matrix(k, n);
            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            let start = std::time::Instant::now();
            matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
            let optimized_time = start.elapsed();

            let start = std::time::Instant::now();
            naive_matmul(&a, &b, &mut c_naive, m, n, k);
            let naive_time = start.elapsed();

            let speedup = naive_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
            println!(
                "{} - Optimized: {:?}, Naive: {:?}, Speedup: {:.2}x",
                name, optimized_time, naive_time, speedup
            );

            // Use relative error tolerance for large values (avoids precision issues)
            for i in 0..(m * n) {
                let diff = (c_optimized[i] - c_naive[i]).abs();
                let max_val = c_optimized[i].abs().max(c_naive[i].abs());
                let relative_error = if max_val > 1e-6 { diff / max_val } else { diff };

                assert!(
                    relative_error < 1e-5,
                    "{} mismatch at {}: optimized={}, naive={}, rel_err={}",
                    name,
                    i,
                    c_optimized[i],
                    c_naive[i],
                    relative_error
                );
            }
        }
    }

    #[test]
    fn test_matmul_random_small() {
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..10 {
            let m = rng.random_range(1..=16);
            let n = rng.random_range(1..=16);
            let k = rng.random_range(1..=16);

            let mut a = vec![0.0; m * k];
            let mut b = vec![0.0; k * n];

            for i in 0..a.len() {
                a[i] = rng.random_range(-1.0..1.0);
            }
            for i in 0..b.len() {
                b[i] = rng.random_range(-1.0..1.0);
            }

            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
            naive_matmul(&a, &b, &mut c_naive, m, n, k);

            for i in 0..(m * n) {
                let diff = (c_optimized[i] - c_naive[i]).abs();
                assert!(
                    diff < 1e-5,
                    "Random {}x{}x{} mismatch at {}: optimized={}, naive={}, diff={}",
                    m,
                    n,
                    k,
                    i,
                    c_optimized[i],
                    c_naive[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_matmul_random_medium() {
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..5 {
            let m = rng.random_range(16..=64);
            let n = rng.random_range(16..=64);
            let k = rng.random_range(16..=64);

            let mut a = vec![0.0; m * k];
            let mut b = vec![0.0; k * n];

            for i in 0..a.len() {
                a[i] = rng.random_range(-1.0..1.0);
            }
            for i in 0..b.len() {
                b[i] = rng.random_range(-1.0..1.0);
            }

            let mut c_optimized = vec![0.0; m * n];
            let mut c_naive = vec![0.0; m * n];

            matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
            naive_matmul(&a, &b, &mut c_naive, m, n, k);

            for i in 0..(m * n) {
                let diff = (c_optimized[i] - c_naive[i]).abs();
                let max_val = c_optimized[i].abs().max(c_naive[i].abs());
                let relative_error = if max_val > 1e-6 { diff / max_val } else { diff };

                assert!(
                    relative_error < 5e-3,
                    "Random {}x{}x{} mismatch at {}: optimized={}, naive={}, rel_err={}",
                    m,
                    n,
                    k,
                    i,
                    c_optimized[i],
                    c_naive[i],
                    relative_error
                );
            }
        }
    }

    #[test]
    fn test_matmul_random_kernel_sizes() {
        use rand::Rng;
        let mut rng = rand::rng();

        // Test various multiples and non-multiples of 8 (kernel size)
        let sizes = [7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33];

        for &size in &sizes {
            for _ in 0..3 {
                let m = size;
                let n = size;
                let k = size;

                let mut a = vec![0.0; m * k];
                let mut b = vec![0.0; k * n];

                for i in 0..a.len() {
                    a[i] = rng.random_range(-1.0..1.0);
                    // a[i] = 1.0;
                }
                for i in 0..b.len() {
                    b[i] = rng.random_range(-1.0..1.0);
                    // b[i] = 1.0;
                }

                let mut c_optimized = vec![0.0; m * n];
                let mut c_naive = vec![0.0; m * n];

                matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
                naive_matmul(&a, &b, &mut c_naive, m, n, k);

                for i in 0..(m * n) {
                    let diff = (c_optimized[i] - c_naive[i]).abs();
                    let max_val = c_optimized[i].abs().max(c_naive[i].abs());
                    let relative_error = if max_val > 1e-6 { diff / max_val } else { diff };

                    // Use more lenient tolerance for larger matrices due to accumulated floating-point errors
                    let tolerance = 2e-2;
                    assert!(
                        relative_error < tolerance,
                        "Random {}x{}x{} (kernel boundary) mismatch at {}: optimized={}, naive={}, rel_err={}, tolerance={}",
                        m, n, k, i, c_optimized[i], c_naive[i], relative_error, tolerance
                    );
                }
            }
        }
    }

    #[test]
    fn test_matmul_random_extreme_ratios() {
        use rand::Rng;
        let mut rng = rand::rng();

        // Test extreme aspect ratios with random data
        let test_cases = [
            (1, 64, 32), // Very tall and narrow
            (64, 1, 32), // Very wide and short
            (128, 4, 8), // Tall
            (4, 128, 8), // Wide
            (8, 8, 256), // Deep
        ];

        for &(m, n, k) in &test_cases {
            for _ in 0..2 {
                let mut a = vec![0.0; m * k];
                let mut b = vec![0.0; k * n];

                for i in 0..a.len() {
                    a[i] = rng.random_range(-1.0..1.0);
                }
                for i in 0..b.len() {
                    b[i] = rng.random_range(-1.0..1.0);
                }

                let mut c_optimized = vec![0.0; m * n];
                let mut c_naive = vec![0.0; m * n];

                matmul(&a, &b, &mut c_optimized, m, n, k, MC, NC);
                naive_matmul(&a, &b, &mut c_naive, m, n, k);

                for i in 0..(m * n) {
                    let diff = (c_optimized[i] - c_naive[i]).abs();
                    let max_val = c_optimized[i].abs().max(c_naive[i].abs());
                    let relative_error = if max_val > 1e-6 { diff / max_val } else { diff };

                    assert!(
                        relative_error < 1e-3,
                        "Random extreme {}x{}x{} mismatch at {}: optimized={}, naive={}, rel_err={}",
                        m, n, k, i, c_optimized[i], c_naive[i], relative_error
                    );
                }
            }
        }
    }
}
