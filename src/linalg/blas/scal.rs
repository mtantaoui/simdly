/// Scales a double-precision vector `x` by a scalar `sa`.
///
/// This function performs the operation `x[i] = sa * x[i]` for `n` elements,
/// considering a stride `incx`.
///
/// It mirrors the functionality of the BLAS `dscal` routine.
///
/// # Arguments
///
/// * `n`: The number of elements in vector `x` to scale.
/// * `sa`: The scalar multiplier.
/// * `x`: A mutable slice representing the vector to be scaled.
/// * `incx`: The increment (stride) for accessing elements in `x`.
///   A positive value means forward iteration, negative means backward.
///   Must not be zero if `n > 1`.
///
/// # Panics
///
/// This function will panic if:
/// * `n > 1` and `incx` is zero (ambiguous operation for multiple elements).
/// * The effective length required to access `n` elements in `x` (considering `incx`)
///   exceeds `x.len()`. This check occurs only if `n > 0`.
///
/// # Notes on Parallelism
///
/// For the common case where `incx == 1` (contiguous memory access), the operation
/// is "embarrassingly parallel" and can be efficiently parallelized using Rayon
/// by changing `.iter_mut()` to `.par_iter_mut()`.
///
/// # Examples
///
/// ```
/// ```
pub fn dscal(n: usize, sa: f64, x: &mut [f64], incx: isize) {
    // 1. Handle n == 0 (early exit, no access)
    if n == 0 {
        return;
    }

    // 2. Validate incx if n > 1
    // (n is guaranteed to be >= 1 at this point)
    if n > 1 && incx == 0 {
        panic!("dscal: incx is 0 but n > 1, which is ambiguous or an error condition.");
    }

    // 3. Validate slice length x.len() against n and incx
    // (n is guaranteed to be >= 1 at this point)
    // The formula `1 + (n - 1) * incx.unsigned_abs()` is correct for n >= 1.
    // If n = 1, required_len = 1 + 0 * abs = 1.
    // If n > 1, it calculates the span.
    let required_len_val = 1 + (n - 1) * incx.unsigned_abs();

    if x.len() < required_len_val {
        panic!(
            "dscal: x slice length {} is insufficient for n={} and incx={}. Required: {}",
            x.len(),
            n,
            incx,
            required_len_val
        );
    }

    // 4. Now, apply optimizations like sa == 1.0 early exit.
    // This optimization is now safe because we've confirmed the slice is valid
    // for the specified operation, even if it's a no-op.
    if sa == 1.0 {
        return;
    }

    // 5. Perform the actual operation
    if incx == 1 {
        // Contiguous case
        let x_slice = &mut x[..n]; // Slicing is safe due to required_len_val check
        x_slice.iter_mut().for_each(|val| *val *= sa);
    } else {
        // Strided case
        // (Also correctly handles n=1 with any incx, as current_ix will be 0)
        let mut current_ix: isize = if incx >= 0 {
            0
        } else {
            (-(n as isize) + 1) * incx
        };

        for _ in 0..n {
            // Indexing is safe due to required_len_val check.
            x[current_ix as usize] *= sa;
            current_ix += incx;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-9;

    fn assert_f64_vec_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "Vector lengths differ: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tol,
                "Mismatch at index {i}: {val_a} != {val_b} (within tolerance {tol})"
            );
        }
    }

    #[test]
    fn test_dscal_n_zero() {
        let mut x = vec![1.0, 2.0];
        let x_orig = x.clone();
        dscal(0, 2.0, &mut x, 1);
        assert_f64_vec_eq(&x, &x_orig, EPSILON);
    }

    #[test]
    fn test_dscal_sa_one() {
        // This test should pass because the slice is long enough
        let mut x = vec![1.0, 2.0, 3.0];
        let x_orig = x.clone();
        dscal(3, 1.0, &mut x, 1);
        assert_f64_vec_eq(&x, &x_orig, EPSILON);
    }

    #[test]
    #[should_panic(
        expected = "dscal: incx is 0 but n > 1, which is ambiguous or an error condition."
    )]
    fn test_dscal_incx_zero_n_gt_one_panics() {
        let mut x = vec![1.0, 2.0];
        dscal(2, 2.0, &mut x, 0);
    }

    #[test]
    fn test_dscal_n_one_inc_zero() {
        let mut x = vec![10.0];
        dscal(1, 3.0, &mut x, 0);
        assert_f64_vec_eq(&x, &[30.0], EPSILON);

        let mut x2 = vec![5.0, 99.0];
        dscal(1, 0.5, &mut x2, 0);
        assert_f64_vec_eq(&x2, &[2.5, 99.0], EPSILON);
    }

    #[test]
    fn test_dscal_n_one_any_inc() {
        let mut x = vec![10.0, 99.0];
        dscal(1, -2.0, &mut x, 5);
        assert_f64_vec_eq(&x, &[-20.0, 99.0], EPSILON);

        let mut x2 = vec![4.0, 88.0];
        dscal(1, 0.25, &mut x2, -3);
        assert_f64_vec_eq(&x2, &[1.0, 88.0], EPSILON);
    }

    #[test]
    #[should_panic(
        expected = "dscal: x slice length 0 is insufficient for n=1 and incx=1. Required: 1"
    )]
    fn test_dscal_empty_slice_panics() {
        let mut x = vec![];
        dscal(1, 2.0, &mut x, 1); // sa is 2.0 here, so optimization wasn't the issue for *this* test
    }

    #[test]
    fn test_dscal_contiguous_basic() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        dscal(3, 2.0, &mut x, 1);
        assert_f64_vec_eq(&x, &[2.0, 4.0, 6.0, 4.0], EPSILON);
    }

    #[test]
    fn test_dscal_contiguous_sa_zero() {
        let mut x = vec![1.0, 2.0, 3.0];
        dscal(3, 0.0, &mut x, 1);
        assert_f64_vec_eq(&x, &[0.0, 0.0, 0.0], EPSILON);
    }

    #[test]
    fn test_dscal_contiguous_full_length() {
        let mut x = vec![1.0, -2.0];
        dscal(2, -3.0, &mut x, 1);
        assert_f64_vec_eq(&x, &[-3.0, 6.0], EPSILON);
    }

    #[test]
    fn test_dscal_contiguous_n_less_than_unroll() {
        let mut x = vec![1.0, 2.0, 3.0];
        dscal(3, 10.0, &mut x, 1);
        assert_f64_vec_eq(&x, &[10.0, 20.0, 30.0], EPSILON);
    }

    #[test]
    fn test_dscal_contiguous_n_multiple_of_unroll() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        dscal(5, 0.1, &mut x, 1);
        assert_f64_vec_eq(&x, &[0.1, 0.2, 0.3, 0.4, 0.5], EPSILON);
    }

    #[test]
    fn test_dscal_contiguous_n_not_multiple_of_unroll() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        dscal(7, 2.0, &mut x, 1);
        assert_f64_vec_eq(&x, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], EPSILON);
    }

    #[test]
    fn test_dscal_strided_positive_inc() {
        let mut x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
        dscal(4, 3.0, &mut x, 2);
        assert_f64_vec_eq(&x, &[3.0, 0.0, 6.0, 0.0, 9.0, 0.0, 12.0], EPSILON);
    }

    #[test]
    fn test_dscal_strided_negative_inc() {
        let mut x = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        dscal(3, -1.0, &mut x, -2);
        assert_f64_vec_eq(&x, &[-1.0, 0.0, -2.0, 0.0, -3.0], EPSILON);
    }

    // These are the critical tests that were failing due to sa=1.0 optimization order
    #[test]
    #[should_panic(
        expected = "dscal: x slice length 3 is insufficient for n=3 and incx=2. Required: 5"
    )]
    fn test_panic_x_too_short_strided() {
        let mut x = vec![1.0, 2.0, 3.0]; // len 3
        dscal(3, 1.0, &mut x, 2); // sa = 1.0
    }

    #[test]
    #[should_panic(
        expected = "dscal: x slice length 1 is insufficient for n=2 and incx=1. Required: 2"
    )]
    fn test_panic_x_too_short_contiguous() {
        let mut x = vec![1.0]; // len 1
        dscal(2, 1.0, &mut x, 1); // sa = 1.0
    }
}
