/// Computes the dot product of two double-precision vectors.
///
/// This function mirrors the functionality of the BLAS `ddot` routine.
/// It calculates `sum(dx[i] * dy[i])` for `n` elements, considering strides.
///
/// # Arguments
///
/// * `n`: The number of elements to process in `dx` and `dy`.
/// * `dx`: A slice representing the first input vector `x`.
/// * `incx`: The increment (stride) for accessing elements in `dx`.
///   A positive value means forward iteration, negative means backward.
/// * `dy`: A slice representing the second input vector `y`.
/// * `incy`: The increment (stride) for accessing elements in `dy`.
///   A positive value means forward iteration, negative means backward.
///
/// # Returns
///
/// The dot product of the two vectors. Returns `0.0` if `n` is 0.
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
/// The provided code structure for this case can be easily adapted by replacing
/// `.iter()` with Rayon's `.par_iter()` and using a parallel sum/reduce operation.
///
/// Parallelizing the non-contiguous (strided) case is more complex and typically
/// offers less benefit unless `n` is very large.
///
/// # Examples
///
/// ```
/// ```
pub fn ddot(n: usize, dx: &[f64], incx: isize, dy: &[f64], incy: isize) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Validate increments: BLAS usually doesn't define behavior for inc=0 if n > 1
    if n > 1 && (incx == 0 || incy == 0) {
        panic!("ddot: incx and incy must be non-zero if n > 1.");
    }

    // Calculate required slice lengths and validate
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
            "ddot: dx slice length {} is insufficient for n={} and incx={}. Required: {}",
            dx.len(),
            n,
            incx,
            req_dx_len
        );
    }
    if dy.len() < req_dy_len {
        panic!(
            "ddot: dy slice length {} is insufficient for n={} and incy={}. Required: {}",
            dy.len(),
            n,
            incy,
            req_dy_len
        );
    }

    let mut dtemp: f64 = 0.0;

    if incx == 1 && incy == 1 {
        // Contiguous access: most common and optimizable case.

        let dx_slice = &dx[..n];
        let dy_slice = &dy[..n];

        // TODO: use crate SIMD traits here
        // --- To use Rayon (par_iter) ---
        // --- Sequential iterator version ---
        dtemp = dx_slice
            .iter()
            .zip(dy_slice.iter())
            .map(|(&x_val, &y_val)| x_val * y_val)
            .sum();
    } else {
        // Code for unequal increments or equal increments not equal to 1.
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

        for _ in 0..n {
            // Bounds checks performed at the beginning using `required_len`.
            // Casting from isize to usize is safe here.
            dtemp += dx[ix as usize] * dy[iy as usize];
            ix += incx;
            iy += incy;
        }
    }

    dtemp
}

#[cfg(test)]
mod tests {
    use super::*; // Import ddot from the parent module

    const EPSILON: f64 = 1e-9; // For float comparisons

    #[test]
    fn test_ddot_n_zero() {
        let dx = vec![10.0, 20.0];
        let dy = vec![1.0, 2.0];
        assert_eq!(ddot(0, &dx, 1, &dy, 1), 0.0);
    }

    #[test]
    fn test_ddot_contiguous() {
        let dx = vec![1.0, 2.0, 3.0, 4.0];
        let dy = vec![5.0, 6.0, 7.0, 8.0];
        let n = 3;
        // (1*5) + (2*6) + (3*7) = 5 + 12 + 21 = 38
        let result = ddot(n, &dx, 1, &dy, 1);
        assert!((result - 38.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_contiguous_full_length() {
        let dx = vec![1.0, 2.0, 3.0];
        let dy = vec![10.0, 0.5, 2.0];
        let n = 3;
        // (1*10) + (2*0.5) + (3*2) = 10 + 1 + 6 = 17
        let result = ddot(n, &dx, 1, &dy, 1);
        assert!((result - 17.0).abs() < EPSILON);
    }

    // Test cases for the unrolling logic (if it were directly translated)
    // The iterator version naturally handles these.
    #[test]
    fn test_ddot_contiguous_n_less_than_unroll_factor() {
        let dx = vec![1.0, 2.0];
        let dy = vec![3.0, 4.0];
        // (1*3) + (2*4) = 3 + 8 = 11
        assert!((ddot(2, &dx, 1, &dy, 1) - 11.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_contiguous_n_multiple_of_unroll_factor() {
        // Original C code unrolls by 5.
        let dx = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dy = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        // 1+2+3+4+5 = 15
        assert!((ddot(5, &dx, 1, &dy, 1) - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_contiguous_n_not_multiple_of_unroll_factor() {
        let dx = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // n=6
        let dy = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // 1+2+3+4+5+6 = 21
        assert!((ddot(6, &dx, 1, &dy, 1) - 21.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_strided_positive_incs() {
        let dx = vec![1.0, 0.0, 2.0, 0.0, 3.0]; // Use dx[0], dx[2], dx[4]
        let dy = vec![10.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.0]; // Use dy[0], dy[3], dy[6]
        let n = 3;
        // (dx[0]*dy[0]) + (dx[2]*dy[3]) + (dx[4]*dy[6])
        // (1.0*10.0) + (2.0*5.0) + (3.0*2.0) = 10 + 10 + 6 = 26
        let result = ddot(n, &dx, 2, &dy, 3);
        assert!((result - 26.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_strided_dx_rev_dy_fwd() {
        let dx = vec![10.0, 20.0, 30.0]; // Use dx[2], dx[1], dx[0]
        let dy = vec![1.0, 2.0, 3.0]; // Use dy[0], dy[1], dy[2]
        let n = 3;
        // (dx[2]*dy[0]) + (dx[1]*dy[1]) + (dx[0]*dy[2])
        // (30.0*1.0) + (20.0*2.0) + (10.0*3.0) = 30 + 40 + 30 = 100
        let result = ddot(n, &dx, -1, &dy, 1);
        assert!((result - 100.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_strided_both_rev() {
        let dx = vec![10.0, 20.0, 30.0]; // Use dx[2], dx[1], dx[0]
        let dy = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Use dy[4], dy[2], dy[0] (incy=-2)
                                                // Start iy = (-3+1)*-2 = 4
        let n = 3;
        // (dx[2]*dy[4]) + (dx[1]*dy[2]) + (dx[0]*dy[0])
        // (30.0*5.0) + (20.0*3.0) + (10.0*1.0) = 150 + 60 + 10 = 220
        let result = ddot(n, &dx, -1, &dy, -2);
        assert!((result - 220.0).abs() < EPSILON);
    }

    #[test]
    fn test_ddot_n_one_strided() {
        let dx = [0.0, 10.0, 0.0]; // dx[1]
        let dy = [0.0, 0.0, 5.0]; // dy[2]
                                  // (10.0 * 5.0) = 50.0
        assert!((ddot(1, &dx[1..], 1, &dy[2..], 1) - 50.0).abs() < EPSILON); // Slicing to pass relevant part

        let dx_full = [0.0, 10.0, 0.0];
        let dy_full = [0.0, 5.0, 0.0]; // Access dy_full[1]
                                       // Test with original logic
                                       // Use dx_full[1], dy_full[1]
                                       // incx=0 means dx_full[(-1+1)*0] = dx_full[0] - wait, my logic for incx=0 in C is subtle.
                                       // It's not `(-n+1)*incx` if incx=0. It's just ix=0.
                                       // Original C code: if (0 <= incx) ix = 0; else ix = (-n+1)*incx;
                                       // So for incx=0 or incy=0, ix/iy start at 0.
                                       // The check `n > 1 && (incx == 0 || incy == 0)` handles the panic.
                                       // For n=1, inc=0 is fine.
        let result_n1_inc0 = ddot(1, &dx_full[1..], 0, &dy_full[1..], 0); // dx_full[1]*dy_full[1]
        assert!((result_n1_inc0 - (10.0 * 5.0)).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "dx slice length 2 is insufficient for n=3 and incx=1. Required: 3")]
    fn test_panic_dx_too_short_contiguous() {
        let dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0]; // Too short
        ddot(3, &dx, 1, &dy, 1);
    }

    #[test]
    #[should_panic(expected = "dy slice length 2 is insufficient for n=3 and incy=1. Required: 3")]
    fn test_panic_dy_too_short_contiguous() {
        let dx = vec![1.0, 2.0, 3.0];
        let dy = vec![10.0, 20.0]; // Too short
        ddot(3, &dx, 1, &dy, 1);
    }

    #[test]
    #[should_panic(expected = "dx slice length 3 is insufficient for n=3 and incx=2. Required: 5")]
    fn test_panic_dx_too_short_strided() {
        let dy = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0]; // Needs indices 0,2,4 -> len 5
        ddot(3, &dx, 2, &dy, 1);
    }

    #[test]
    #[should_panic(expected = "incx and incy must be non-zero if n > 1")]
    fn test_panic_incx_zero() {
        let dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0];
        ddot(2, &dx, 0, &dy, 1);
    }

    #[test]
    #[should_panic(expected = "incx and incy must be non-zero if n > 1")]
    fn test_panic_incy_zero() {
        let dy = vec![1.0, 2.0, 3.0];
        let dx = vec![10.0, 20.0, 30.0];
        ddot(2, &dx, 1, &dy, 0);
    }

    #[test]
    fn test_ok_inc_zero_n_one() {
        let dy = vec![1.0];
        let dx = vec![10.0];
        // For n=1, inc=0 means use the first (and only) element.
        assert!((ddot(1, &dx, 0, &dy, 0) - (10.0 * 1.0)).abs() < EPSILON);
        assert!((ddot(1, &dx, 0, &dy, 100) - (10.0 * 1.0)).abs() < EPSILON); // incy doesn't matter for n=1
        assert!((ddot(1, &dx, -100, &dy, 0) - (10.0 * 1.0)).abs() < EPSILON); // incx doesn't matter for n=1
    }
}
