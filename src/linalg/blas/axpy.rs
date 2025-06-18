/// Computes `dy = da * dx + dy` (double-precision a*x plus y).
///
/// This function mirrors the functionality of the BLAS `daxpy` routine.
/// It updates the vector `dy` by adding a scaled version of vector `dx` to it.
///
/// # Arguments
///
/// * `n`: The number of elements to process in `dx` and `dy`.
/// * `da`: The scalar multiplier for `dx`.
/// * `dx`: A slice representing the first input vector `x`.
/// * `incx`: The increment (stride) for accessing elements in `dx`.
///   A positive value means forward iteration, negative means backward.
/// * `dy`: A mutable slice representing the second input/output vector `y`.
///   On output, `dy` is replaced by `dy + da * dx`.
/// * `incy`: The increment (stride) for accessing elements in `dy`.
///   A positive value means forward iteration, negative means backward.
///
/// # Panics
///
/// This function will panic if:
/// * `n` is 0 and `incx` or `incy` is 0 (to avoid division by zero in length calculation, though practically `n=0` means no work).
/// * The effective length required to access `n` elements in `dx` (considering `incx`) exceeds `dx.len()`.
/// * The effective length required to access `n` elements in `dy` (considering `incy`) exceeds `dy.len()`.
/// * `incx` or `incy` is zero when `n > 1`, as this would mean accessing the same element repeatedly
///   or an infinite loop in a conceptual sense (though the loop is bounded by `n`). BLAS typically
///   does not define behavior for zero increments with n > 1. This implementation will panic.
///
/// # Notes on Parallelism
///
/// For the common case where `incx == 1` and `incy == 1` (contiguous memory access),
/// the operation can be very efficiently parallelized using Rayon.
/// The provided code structure for this case (see `daxpy_contiguous_unchecked`)
/// can be easily adapted by replacing `.iter()` and `.iter_mut()` with
/// Rayon's `.par_iter()` and `.par_iter_mut()`.
///
/// Parallelizing the non-contiguous (strided) case is more complex and typically
/// offers less benefit unless `n` is very large and the strides are cache-friendly.
/// Such parallelization often requires `unsafe` code to satisfy the borrow checker
/// if direct parallel iteration over strided mutable slices is attempted.
/// # Examples
///
/// ```
/// ```
pub fn daxpy(n: usize, da: f64, dx: &[f64], incx: isize, dy: &mut [f64], incy: isize) {
    // Early exit if n is 0 or da is 0.0 (no operation needed)
    if n == 0 || da == 0.0 {
        return;
    }

    // Validate increments: BLAS usually doesn't define behavior for inc=0 if n > 1
    if n > 1 && (incx == 0 || incy == 0) {
        panic!("daxpy: incx and incy must be non-zero if n > 1.");
    }

    // Calculate required slice lengths and validate
    // For n elements, the maximum index reached is (n-1) * abs(increment).
    // The length required is 1 + (n-1) * abs(increment).
    // If n = 0, required length is 0. If n = 1, required length is 1.
    let required_len = |count: usize, increment: isize| -> usize {
        if count == 0 {
            0
        } else {
            1 + (count - 1) * increment.unsigned_abs()
        }
    };

    let req_dx_len = required_len(n, incx);
    let req_dy_len = required_len(n, incy);

    if dx.len() < req_dx_len {
        panic!(
            "daxpy: dx slice length {} is insufficient for n={} and incx={}. Required: {}",
            dx.len(),
            n,
            incx,
            req_dx_len
        );
    }
    if dy.len() < req_dy_len {
        panic!(
            "daxpy: dy slice length {} is insufficient for n={} and incy={}. Required: {}",
            dy.len(),
            n,
            incy,
            req_dy_len
        );
    }

    // Dispatch to specialized versions based on increments
    if incx == 1 && incy == 1 {
        // This is the case that can be easily parallelized with Rayon
        // by changing iter().zip() to par_iter().zip() etc.
        // For simplicity and to highlight iterator use, we'll use a zipped iterator.

        // Ensure slices are taken for only 'n' elements for contiguous case
        let dx_slice = &dx[..n];
        let dy_slice = &mut dy[..n];

        // --- To use Rayon (par_iter_mut) ---
        // --- Sequential version using iterators ---
        // TODO: use crate SIMD traits here
        dy_slice
            .iter_mut()
            .zip(dx_slice.iter())
            .for_each(|(y_i, x_i)| {
                *y_i += da * (*x_i);
            });
    } else {
        // Code for unequal increments or equal increments not equal to 1.
        // Determine starting indices based on increment sign.
        // If inc is positive, start at 0.
        // If inc is negative, start at the "end" of the N elements.

        let mut ix: isize = if incx > 0 {
            0
        } else {
            (-(n as isize) + 1) * incx
        };
        let mut iy: isize = if incy > 0 {
            0
        } else {
            (-(n as isize) + 1) * incy
        };

        // TODO: use crate SIMD traits here
        // This loop structure directly mirrors the Original fortran code's strided access logic.
        for _ in 0..n {
            // Bounds checks were performed at the beginning using `required_len`.
            // So, `ix` and `iy` (when cast to usize) should be valid indices.
            // Casting from isize to usize is safe here because:
            // - For positive inc, ix/iy start at 0 and increase, staying within `0 .. slice.len()`.
            // - For negative inc, ix/iy start at `(n-1)*|inc|` (max possible positive index)
            //   and decrease, also staying within `0 .. slice.len()`.
            // The `required_len` check ensures the underlying slice is large enough for these accesses.
            dy[iy as usize] += da * dx[ix as usize];
            ix += incx;
            iy += incy;
        }
    }
}

// To run tests, create a `lib.rs` with `pub mod my_module;` (if you put the code in my_module.rs)
// or just put the code and tests in `lib.rs`.
// Then `cargo test`.
#[cfg(test)]
mod tests {
    use super::*; // Import daxpy from the parent module

    const EPSILON: f64 = 1e-9; // For float comparisons

    fn assert_f64_vec_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "Vector lengths differ");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tol,
                "Mismatch at index {i}: {val_a} != {val_b} (within tolerance {tol})"
            );
        }
    }

    #[test]
    fn test_daxpy_n_zero() {
        let mut dy = vec![1.0, 2.0];
        let dx = vec![10.0, 20.0];
        let original_dy = dy.clone();
        daxpy(0, 0.5, &dx, 1, &mut dy, 1);
        assert_f64_vec_eq(&dy, &original_dy, EPSILON);
    }

    #[test]
    fn test_daxpy_da_zero() {
        let mut dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0];
        let original_dy = dy.clone();
        daxpy(3, 0.0, &dx, 1, &mut dy, 1);
        assert_f64_vec_eq(&dy, &original_dy, EPSILON);
    }

    #[test]
    fn test_daxpy_contiguous() {
        let mut dy = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dx = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let da = 0.5;
        let n = 3;
        daxpy(n, da, &dx, 1, &mut dy, 1);
        // dy[0] = 1.0 + 0.5 * 10.0 = 6.0
        // dy[1] = 2.0 + 0.5 * 20.0 = 12.0
        // dy[2] = 3.0 + 0.5 * 30.0 = 18.0
        let expected = vec![6.0, 12.0, 18.0, 4.0, 5.0];
        assert_f64_vec_eq(&dy, &expected, EPSILON);
    }

    #[test]
    fn test_daxpy_contiguous_full_length() {
        let mut dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0];
        let da = 2.0;
        let n = 3;
        daxpy(n, da, &dx, 1, &mut dy, 1);
        let expected = vec![21.0, 42.0, 63.0];
        assert_f64_vec_eq(&dy, &expected, EPSILON);
    }

    #[test]
    fn test_daxpy_unrolled_case_n_less_than_4() {
        let mut dy = vec![1.0, 2.0];
        let dx = vec![10.0, 20.0];
        let da = 1.0;
        daxpy(2, da, &dx, 1, &mut dy, 1);
        assert_f64_vec_eq(&dy, &[11.0, 22.0], EPSILON);
    }

    #[test]
    fn test_daxpy_unrolled_case_n_multiple_of_4() {
        let mut dy = vec![1.0, 2.0, 3.0, 4.0];
        let dx = vec![10.0, 20.0, 30.0, 40.0];
        let da = 1.0;
        daxpy(4, da, &dx, 1, &mut dy, 1);
        assert_f64_vec_eq(&dy, &[11.0, 22.0, 33.0, 44.0], EPSILON);
    }

    #[test]
    fn test_daxpy_unrolled_case_n_not_multiple_of_4() {
        let mut dy = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dx = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let da = 1.0;
        daxpy(5, da, &dx, 1, &mut dy, 1);
        assert_f64_vec_eq(&dy, &[11.0, 22.0, 33.0, 44.0, 55.0], EPSILON);
    }

    #[test]
    fn test_daxpy_strided_positive_incs() {
        let mut dy = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]; // dy elements at 0, 2, 4
        let dx = vec![10.0, 0.0, 20.0, 0.0, 30.0]; // dx elements at 0, 2, 4
        let da = 0.5;
        let n = 3;
        daxpy(n, da, &dx, 2, &mut dy, 2);
        // dy[0] = 1.0 + 0.5 * dx[0] = 1.0 + 0.5 * 10.0 = 6.0
        // dy[2] = 2.0 + 0.5 * dx[2] = 2.0 + 0.5 * 20.0 = 12.0
        // dy[4] = 3.0 + 0.5 * dx[4] = 3.0 + 0.5 * 30.0 = 18.0
        let expected = vec![6.0, 0.0, 12.0, 0.0, 18.0, 0.0, 4.0];
        assert_f64_vec_eq(&dy, &expected, EPSILON);
    }

    #[test]
    fn test_daxpy_strided_mixed_incs_dx_rev() {
        let mut dy = vec![1.0, 2.0, 3.0]; // dy elements at 0, 1, 2
        let dx = vec![10.0, 20.0, 30.0]; // dx elements used: dx[2], dx[1], dx[0]
        let da = 1.0;
        let n = 3;
        daxpy(n, da, &dx, -1, &mut dy, 1);
        // dy[0] = 1.0 + 1.0 * dx[2] = 1.0 + 30.0 = 31.0
        // dy[1] = 2.0 + 1.0 * dx[1] = 2.0 + 20.0 = 22.0
        // dy[2] = 3.0 + 1.0 * dx[0] = 3.0 + 10.0 = 13.0
        let expected = vec![31.0, 22.0, 13.0];
        assert_f64_vec_eq(&dy, &expected, EPSILON);
    }

    #[test]
    fn test_daxpy_strided_both_rev() {
        // dy elements used: dy[4], dy[2], dy[0]
        let mut dy = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        // dx elements used: dx[2], dx[1], dx[0]
        let dx = vec![10.0, 20.0, 30.0];
        let da = 1.0;
        let n = 3;
        daxpy(n, da, &dx, -1, &mut dy, -2);
        // Iteration 1 (i=0):
        //   ix = (-3+1)*-1 = 2 (dx[2]=30.0)
        //   iy = (-3+1)*-2 = 4 (dy[4]=3.0)
        //   dy[4] = 3.0 + 1.0 * 30.0 = 33.0
        // Iteration 2 (i=1):
        //   ix = 2 + (-1) = 1 (dx[1]=20.0)
        //   iy = 4 + (-2) = 2 (dy[2]=2.0)
        //   dy[2] = 2.0 + 1.0 * 20.0 = 22.0
        // Iteration 3 (i=2):
        //   ix = 1 + (-1) = 0 (dx[0]=10.0)
        //   iy = 2 + (-2) = 0 (dy[0]=1.0)
        //   dy[0] = 1.0 + 1.0 * 10.0 = 11.0
        let expected = vec![11.0, 0.0, 22.0, 0.0, 33.0];
        assert_f64_vec_eq(&dy, &expected, EPSILON);
    }

    #[test]
    fn test_daxpy_n_one_strided() {
        let mut dy = vec![1.0, 2.0, 3.0];
        let dx = [10.0, 20.0, 30.0];
        daxpy(1, 2.0, &dx[1..], -1, &mut dy[1..], -1); // dx is dx[1], dy is dy[1] effectively
                                                       // dy[1] = 2.0 + 2.0 * dx[1] = 2.0 + 2.0 * 20.0 = 42.0
        assert_f64_vec_eq(&dy, &[1.0, 42.0, 3.0], EPSILON);
    }

    #[test]
    #[should_panic(expected = "dx slice length 2 is insufficient for n=3 and incx=1. Required: 3")]
    fn test_panic_dx_too_short_contiguous() {
        let mut dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0]; // Too short for n=3
        daxpy(3, 1.0, &dx, 1, &mut dy, 1);
    }

    #[test]
    #[should_panic(expected = "dy slice length 2 is insufficient for n=3 and incy=1. Required: 3")]
    fn test_panic_dy_too_short_contiguous() {
        let mut dy = vec![1.0, 2.0]; // Too short
        let dx = vec![10.0, 20.0, 30.0];
        daxpy(3, 1.0, &dx, 1, &mut dy, 1);
    }

    #[test]
    #[should_panic(expected = "dx slice length 3 is insufficient for n=3 and incx=2. Required: 5")]
    fn test_panic_dx_too_short_strided() {
        let mut dy = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0]; // Too short for n=3, incx=2 (needs indices 0,2,4 -> len 5)
        daxpy(3, 1.0, &dx, 2, &mut dy, 1);
    }

    #[test]
    #[should_panic(expected = "incx and incy must be non-zero if n > 1")]
    fn test_panic_incx_zero() {
        let mut dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0];
        daxpy(2, 1.0, &dx, 0, &mut dy, 1);
    }

    #[test]
    #[should_panic(expected = "incx and incy must be non-zero if n > 1")]
    fn test_panic_incy_zero() {
        let mut dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0];
        daxpy(2, 1.0, &dx, 1, &mut dy, 0);
    }

    #[test]
    fn test_ok_inc_zero_n_one() {
        let mut dy = vec![1.0];
        let dx = vec![10.0];
        daxpy(1, 2.0, &dx, 0, &mut dy, 0); // inc = 0 is fine for n=1
        assert_f64_vec_eq(&dy, &[21.0], EPSILON);
    }
}

// If you want to put this in a library, you might name the file `src/lib.rs`
// or `src/blas_like.rs` and then in `src/lib.rs` have `pub mod blas_like;`
// Then, you can run `cargo test` from your project root.
// Example `src/lib.rs` if `daxpy` is in `src/daxpy_module.rs`:
// pub mod daxpy_module;
// pub use daxpy_module::daxpy; // to re-export
