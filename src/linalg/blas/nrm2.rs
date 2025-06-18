/// Computes the Euclidean norm (L2 norm) of a double-precision vector.
///
/// Mathematically, `dnrm2(x) = sqrt(x[0]^2 + x[1]^2 + ... + x[n-1]^2)`.
/// This implementation uses a scaling algorithm to prevent overflow and underflow
/// during the sum of squares computation, which is crucial for robust numerical behavior.
///
/// This function aims to be a robust Rust equivalent of BLAS `dnrm2`, handling
/// various strides and improving on some limitations of the provided C example
/// (e.g., by correctly processing negative increments).
///
/// # Arguments
///
/// * `n`: The number of elements in vector `x` to consider.
/// * `x`: A slice representing the input vector.
/// * `incx`: The increment (stride) for accessing elements in `x`.
///   A positive value means forward iteration, negative means backward.
///   Must not be zero if `n > 1`.
///
/// # Returns
///
/// The Euclidean norm of the specified elements of `x`.
/// Returns `0.0` if `n` is 0 or if all selected elements are zero.
///
/// # Panics
///
/// This function will panic if:
/// * `n > 0` and `x` is empty.
/// * `incx` is 0 and `n > 1` (this case is considered an error or ambiguous).
/// * The effective length required to access `n` elements in `x` (considering `incx`)
///   exceeds `x.len()`.
///
/// # Numerical Stability
/// The algorithm is based on computing `scale * sqrt(ssq)`.
/// `scale` tracks the maximum absolute element encountered so far.
/// `ssq` (sum of squares scaled) accumulates terms like `(x_i / scale)^2`.
/// If a new element `abs_x_i` is found that is larger than the current `scale`:
///   1. The existing `ssq` is rescaled: `new_ssq_component = old_ssq * (old_scale / new_scale)^2`.
///   2. `ssq` becomes `1.0 + new_ssq_component`.
///   3. `scale` is updated to `abs_x_i`.
///      If `abs_x_i <= scale`:
///      `ssq` becomes `old_ssq + (abs_x_i / scale)^2`.
///
/// This process ensures that intermediate values in `ssq` do not grow excessively large
/// or become too small, thus avoiding overflow/underflow.
///
/// # Examples
///
/// ```
/// ```
pub fn dnrm2(n: usize, x: &[f64], incx: isize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Handle incx == 0 case
    if incx == 0 {
        if n == 1 {
            // For n=1, incx=0 implies using the single element x[0].
            // Length check below will ensure x is not empty.
            if x.is_empty() {
                // This specific check helps give a clearer message if x is empty with n=1, incx=0
                panic!("dnrm2: x is empty but n=1 (with incx=0)");
            }
            return x[0].abs();
        } else {
            // n > 1 and incx == 0 means accessing x[0] multiple times.
            // The norm would be sqrt(n * x[0]^2) = sqrt(n) * |x[0]|.

            panic!("dnrm2: incx is 0 but n > 1, which is ambiguous or an error condition.");
        }
    }

    // Validate slice length:
    // For n elements, the span of indices is (n-1) * |incx|.
    // The number of elements in the slice needed is 1 (for the base) + (n-1) * |incx|.
    let abs_incx = incx.unsigned_abs();
    let required_len = 1 + (n - 1) * abs_incx; // If n=1, required_len = 1.

    if x.len() < required_len {
        panic!(
            "dnrm2: x slice length {} is insufficient for n={} and incx={}. Required: {}",
            x.len(),
            n,
            incx,
            required_len
        );
    }

    // Handle n == 1 separately (covers all incx != 0 after above checks)
    if n == 1 {
        // For n=1, the norm is simply the absolute value of the first (and only) element
        // in the sequence. The `required_len` check ensures `x[0]` is valid.
        return x[0].abs();
    }

    // Main algorithm for n > 1 and incx != 0
    let mut scale: f64 = 0.0;
    let mut ssq: f64 = 1.0; // Sum of squares, scaled. Initialized to 1.0 as per algorithm.

    if incx == 1 {
        // Optimized path for contiguous access (incx = 1)
        let x_slice = &x[..n]; // Safe due to required_len check
        for &val_x in x_slice.iter() {
            if val_x != 0.0 {
                let abs_x = val_x.abs();
                if scale < abs_x {
                    // New maximum absolute value found. Rescale ssq and update scale.
                    // ssq := 1 + ssq * (scale/abs_x)^2
                    ssq = 1.0 + ssq * (scale / abs_x).powi(2);
                    scale = abs_x;
                } else {
                    // abs_x <= scale. Add to ssq.
                    // scale cannot be 0 here, because if abs_x != 0 and scale was 0,
                    // we would have entered the `scale < abs_x` branch.
                    ssq += (abs_x / scale).powi(2);
                }
            }
        }
    } else {
        // General strided access
        // Determine starting index based on incx sign
        let mut current_ix: isize = if incx > 0 {
            0
        } else {
            ((n - 1) * abs_incx) as isize
        };

        for _ in 0..n {
            let val_x = x[current_ix as usize]; // Safe due to required_len check
            if val_x != 0.0 {
                let abs_x = val_x.abs();
                if scale < abs_x {
                    ssq = 1.0 + ssq * (scale / abs_x).powi(2);
                    scale = abs_x;
                } else {
                    ssq += (abs_x / scale).powi(2);
                }
            }
            current_ix += incx;
        }
    }

    // The final norm is scale * sqrt(ssq).
    // If all elements were zero, scale will be 0.0, resulting in a 0.0 norm.
    scale * ssq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-9; // Tolerance for floating point comparisons

    fn assert_f64_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "Assertion failed: {a} != {b} (within tolerance {tol})"
        );
    }

    #[test]
    fn test_dnrm2_n_zero() {
        assert_f64_eq(dnrm2(0, &[], 1), 0.0, EPSILON); // n=0, empty slice
        assert_f64_eq(dnrm2(0, &[1.0, 2.0], 1), 0.0, EPSILON); // n=0, non-empty slice
    }

    #[test]
    #[should_panic(expected = "dnrm2: incx is 0 but n > 1")]
    fn test_dnrm2_incx_zero_n_gt_one_panics() {
        dnrm2(2, &[1.0, 2.0], 0);
    }

    #[test]
    fn test_dnrm2_incx_zero_n_one() {
        assert_f64_eq(dnrm2(1, &[5.0], 0), 5.0, EPSILON);
        assert_f64_eq(dnrm2(1, &[-3.0, 10.0], 0), 3.0, EPSILON); // Uses x[0]
    }

    #[test]
    #[should_panic(
        expected = "dnrm2: x slice length 0 is insufficient for n=1 and incx=1. Required: 1"
    )]
    fn test_dnrm2_n_one_empty_slice_panics() {
        dnrm2(1, &[], 1); // n=1 requires at least one element
    }

    #[test]
    #[should_panic(expected = "dnrm2: x is empty but n=1 (with incx=0)")]
    fn test_dnrm2_incx_zero_n_one_empty_slice_panics() {
        dnrm2(1, &[], 0); // Special panic message for this specific case
    }

    #[test]
    fn test_dnrm2_n_one_various_incx() {
        assert_f64_eq(dnrm2(1, &[-5.0], 1), 5.0, EPSILON);
        assert_f64_eq(dnrm2(1, &[10.0, 20.0], 1), 10.0, EPSILON); // Uses x[0]
        assert_f64_eq(dnrm2(1, &[10.0, 20.0], 7), 10.0, EPSILON); // incx irrelevant for n=1, uses x[0]
        assert_f64_eq(dnrm2(1, &[10.0, 20.0], -1), 10.0, EPSILON); // incx irrelevant for n=1, uses x[0]
        assert_f64_eq(dnrm2(1, &[10.0, 20.0], -5), 10.0, EPSILON); // incx irrelevant for n=1, uses x[0]
    }

    #[test]
    fn test_dnrm2_contiguous_basic() {
        let x = vec![3.0, 4.0]; // Norm = sqrt(3^2 + 4^2) = 5.0
        assert_f64_eq(dnrm2(2, &x, 1), 5.0, EPSILON);
    }

    #[test]
    fn test_dnrm2_contiguous_all_zeros() {
        let x = vec![0.0, 0.0, 0.0];
        assert_f64_eq(dnrm2(3, &x, 1), 0.0, EPSILON);
    }

    #[test]
    fn test_dnrm2_contiguous_mixed_values() {
        let x = vec![1.0, 2.0, -3.0, 0.0, 4.0]; // 1^2+2^2+(-3)^2+0^2+4^2 = 1+4+9+0+16 = 30
        assert_f64_eq(dnrm2(5, &x, 1), (30.0_f64).sqrt(), EPSILON);
    }

    #[test]
    fn test_dnrm2_scaling_large_values() {
        let x = vec![1e150, 1e150]; // Norm = sqrt(2) * 1e150
                                    // Adjust tolerance for very large numbers
        assert_f64_eq(dnrm2(2, &x, 1), (2.0_f64).sqrt() * 1e150, EPSILON * 1e150);
    }

    #[test]
    fn test_dnrm2_scaling_small_values() {
        let x = vec![1e-150, 1e-150]; // Norm = sqrt(2) * 1e-150
                                      // Adjust tolerance for very small numbers
        assert_f64_eq(dnrm2(2, &x, 1), (2.0_f64).sqrt() * 1e-150, EPSILON * 1e-150);
    }

    #[test]
    fn test_dnrm2_scaling_order_does_not_matter_1() {
        // Test if scaling logic handles order of magnitudes correctly
        let x = vec![1e10, 1.0]; // Max is 1e10. scale=1e10. ssq = 1 + (1/1e10)^2. norm = 1e10 * sqrt(1 + 1e-20) ~= 1e10
        let expected = ((1e10_f64).powi(2) + 1.0_f64.powi(2)).sqrt();
        assert_f64_eq(dnrm2(2, &x, 1), expected, EPSILON * 1e10);
    }

    #[test]
    fn test_dnrm2_scaling_order_does_not_matter_2() {
        let x = vec![1.0, 1e10]; // Max is 1e10. First 1.0: scale=1, ssq=1. Next 1e10: scale=1e10, ssq=1 + 1*(1/1e10)^2
        let expected = ((1.0_f64).powi(2) + (1e10_f64).powi(2)).sqrt();
        assert_f64_eq(dnrm2(2, &x, 1), expected, EPSILON * 1e10);
    }

    #[test]
    fn test_dnrm2_strided_positive_increment() {
        let x = vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0]; // Use 1,2,3,4 with incx=2
                                                            // Norm = sqrt(1^2+2^2+3^2+4^2) = sqrt(1+4+9+16) = sqrt(30)
        assert_f64_eq(dnrm2(4, &x, 2), (30.0_f64).sqrt(), EPSILON);
    }

    #[test]
    fn test_dnrm2_strided_negative_increment() {
        let x = vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0]; // n=4, incx=-2
                                                            // Initial ix for n=4, abs_incx=2: (4-1)*2 = 6. Indices: x[6], x[4], x[2], x[0]
                                                            // Values: 4.0, 3.0, 2.0, 1.0. Norm = sqrt(4^2+3^2+2^2+1^2) = sqrt(16+9+4+1) = sqrt(30)
        assert_f64_eq(dnrm2(4, &x, -2), (30.0_f64).sqrt(), EPSILON);
    }

    #[test]
    fn test_dnrm2_strided_all_zeros() {
        let x = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // n=4, incx=2 uses x[0],x[2],x[4],x[6]
        assert_f64_eq(dnrm2(4, &x, 2), 0.0, EPSILON);
    }

    #[test]
    #[should_panic(
        expected = "dnrm2: x slice length 3 is insufficient for n=3 and incx=2. Required: 5"
    )]
    fn test_panic_x_too_short_strided() {
        let x = vec![1.0, 2.0, 3.0]; // For n=3, incx=2, needs indices 0,2,4, so length 5.
        dnrm2(3, &x, 2);
    }

    #[test]
    #[should_panic(
        expected = "dnrm2: x slice length 1 is insufficient for n=2 and incx=1. Required: 2"
    )]
    fn test_panic_x_too_short_contiguous() {
        let x = vec![1.0]; // For n=2, incx=1, needs indices 0,1, so length 2.
        dnrm2(2, &x, 1);
    }
}
