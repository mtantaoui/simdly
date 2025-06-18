/// Interchanges elements of two double-precision vectors, `x` and `y`.
///
/// This function performs the operation `x[i] <-> y[i]` for `n` elements,
/// considering strides `incx` and `incy`.
///
/// It mirrors the functionality of the BLAS `dswap` routine.
///
/// # Arguments
///
/// * `n`: The number of elements to swap between `x` and `y`.
/// * `x`: A mutable slice representing the first input/output vector.
/// * `incx`: The increment (stride) for accessing elements in `x`.
///   A positive value means forward iteration, negative means backward.
///   Must not be zero if `n > 1`.
/// * `y`: A mutable slice representing the second input/output vector.
/// * `incy`: The increment (stride) for accessing elements in `y`.
///   A positive value means forward iteration, negative means backward.
///   Must not be zero if `n > 1`.
///
/// # Panics
///
/// This function will panic if:
/// * `n > 1` and either `incx` or `incy` is zero (ambiguous operation for multiple elements).
/// * The effective length required to access `n` elements in `x` (considering `incx`)
///   exceeds `x.len()`.
/// * The effective length required to access `n` elements in `y` (considering `incy`)
///   exceeds `y.len()`.
///
/// # Notes on Aliasing and Parallelism
///
/// The function signature `x: &mut [f64], y: &mut [f64]` ensures that `x` and `y`
/// do not fully alias (i.e., they are not the exact same slice). If `x` and `y` were
/// derived from overlapping regions of a single parent slice, the caller would need
/// `unsafe` code to construct such `&mut` references, and standard Rust guarantees
/// would no longer apply directly. This function assumes `x` and `y` are distinct
/// memory regions as per typical safe Rust practices.
///
/// For the common case where `incx == 1` and `incy == 1` (contiguous memory access),
/// the operation can be efficiently parallelized using Rayon, for example by zipping
/// parallel mutable iterators. However, BLAS Level 1 routines like `dswap` are often
/// memory-bandwidth bound, so parallelism benefits are typically seen only for very large `n`.
/// The `swap_with_slice` method used for the contiguous case is highly optimized.
///
/// # Examples
///
/// ```
/// ```
pub fn dswap(n: usize, x: &mut [f64], incx: isize, y: &mut [f64], incy: isize) {
    if n == 0 {
        return;
    }

    // Validate increments: For n > 1, increments must not be zero.
    // For n = 1, an increment of 0 means using the first element (index 0).
    if n > 1 && (incx == 0 || incy == 0) {
        panic!("dswap: incx or incy is 0 but n > 1, which is ambiguous or an error condition.");
    }

    // Calculate required slice lengths and validate.
    // (n is guaranteed to be >= 1 at this point).
    let required_len = |count: usize, increment: isize| -> usize {
        // For count=1, length is 1, regardless of increment.
        // For count > 1, length is 1 (for first element) + (count-1)*abs(increment) (for remaining elements).
        1 + (count - 1) * increment.unsigned_abs()
    };

    let req_x_len = required_len(n, incx);
    let req_y_len = required_len(n, incy);

    if x.len() < req_x_len {
        panic!(
            "dswap: x slice length {} is insufficient for n={} and incx={}. Required: {}",
            x.len(),
            n,
            incx,
            req_x_len
        );
    }
    if y.len() < req_y_len {
        panic!(
            "dswap: y slice length {} is insufficient for n={} and incy={}. Required: {}",
            y.len(),
            n,
            incy,
            req_y_len
        );
    }

    // Dispatch based on increments
    if incx == 1 && incy == 1 {
        // Contiguous case: swap n elements from the start of each slice.
        // Slices x[..n] and y[..n] are guaranteed to be valid and non-overlapping
        // at a high level due to Rust's borrowing rules for `x` and `y` arguments.
        x[..n].swap_with_slice(&mut y[..n]);
    } else {
        // Strided case (also handles n=1 with any incx/incy, including zero).
        // Determine starting indices.
        // If inc is positive or zero, start at index 0.
        // If inc is negative, start at index `(-(n-1))*inc` which is equivalent to `(n-1)*|inc|`
        // if thinking about it from the end of an n-element sequence.
        let mut current_ix: isize = if incx >= 0 {
            0
        } else {
            (-(n as isize) + 1) * incx
        };
        let mut current_iy: isize = if incy >= 0 {
            0
        } else {
            (-(n as isize) + 1) * incy
        };

        // For n=1:
        // If incx >=0, current_ix = 0.
        // If incx < 0, current_ix = (-(1)+1)*incx = 0.
        // So for n=1, current_ix and current_iy are always 0, correctly targeting x[0] and y[0].

        for _ in 0..n {
            // `current_ix` and `current_iy` are cast to usize. This is safe because:
            // - For positive increments, they start at 0 and increase, staying within bounds
            //   verified by `required_len`.
            // - For negative increments, they start at a positive index (e.g., `(n-1)*|inc|`)
            //   and decrease, also staying within `0 .. slice.len()`.
            // The `required_len` check ensures the underlying slices are large enough.
            std::mem::swap(&mut x[current_ix as usize], &mut y[current_iy as usize]);
            current_ix += incx;
            current_iy += incy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dswap_n_zero() {
        let mut x = vec![1.0];
        let mut y = vec![2.0];
        let x_orig = x.clone();
        let y_orig = y.clone();
        dswap(0, &mut x, 1, &mut y, 1);
        assert_eq!(x, x_orig);
        assert_eq!(y, y_orig);
    }

    #[test]
    #[should_panic(expected = "dswap: incx or incy is 0 but n > 1")]
    fn test_dswap_incx_zero_n_gt_one_panics() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        dswap(2, &mut x, 0, &mut y, 1);
    }

    #[test]
    #[should_panic(expected = "dswap: incx or incy is 0 but n > 1")]
    fn test_dswap_incy_zero_n_gt_one_panics() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        dswap(2, &mut x, 1, &mut y, 0);
    }

    #[test]
    fn test_dswap_n_one_inc_zero() {
        let mut x = vec![10.0];
        let mut y = vec![20.0];
        dswap(1, &mut x, 0, &mut y, 0);
        assert_eq!(x, vec![20.0]);
        assert_eq!(y, vec![10.0]);

        let mut x2 = vec![10.0, 99.0];
        let mut y2 = vec![20.0, 88.0];
        dswap(1, &mut x2, 0, &mut y2, 0); // Swaps x2[0] and y2[0]
        assert_eq!(x2, vec![20.0, 99.0]);
        assert_eq!(y2, vec![10.0, 88.0]);
    }

    #[test]
    fn test_dswap_n_one_any_inc() {
        let mut x = vec![10.0, 99.0];
        let mut y = vec![20.0, 88.0];
        // For n=1, any inc value should just use the first elements x[0], y[0]
        dswap(1, &mut x, 5, &mut y, -3);
        assert_eq!(x, vec![20.0, 99.0]);
        assert_eq!(y, vec![10.0, 88.0]);
    }

    #[test]
    #[should_panic(
        expected = "dswap: x slice length 0 is insufficient for n=1 and incx=1. Required: 1"
    )]
    fn test_dswap_empty_slice_x_panics() {
        let mut x = vec![];
        let mut y = vec![1.0];
        dswap(1, &mut x, 1, &mut y, 1);
    }

    #[test]
    #[should_panic(
        expected = "dswap: y slice length 0 is insufficient for n=1 and incy=1. Required: 1"
    )]
    fn test_dswap_empty_slice_y_panics() {
        let mut x = vec![1.0];
        let mut y = vec![];
        dswap(1, &mut x, 1, &mut y, 1);
    }

    #[test]
    fn test_dswap_contiguous() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];
        dswap(3, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![10.0, 20.0, 30.0, 4.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0, 40.0]);
    }

    #[test]
    fn test_dswap_contiguous_full_length() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![10.0, 20.0];
        dswap(2, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![10.0, 20.0]);
        assert_eq!(y, vec![1.0, 2.0]);
    }

    #[test]
    fn test_dswap_strided_positive_incs() {
        let mut x = vec![1.0, 0.0, 2.0, 0.0, 3.0]; // x[0], x[2], x[4]
        let mut y = vec![10.0, 0.0, 0.0, 20.0, 0.0, 0.0, 30.0]; // y[0], y[3], y[6]
        dswap(3, &mut x, 2, &mut y, 3);
        // x[0](1) <-> y[0](10)
        // x[2](2) <-> y[3](20)
        // x[4](3) <-> y[6](30)
        assert_eq!(x, vec![10.0, 0.0, 20.0, 0.0, 30.0]);
        assert_eq!(y, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_dswap_strided_x_fwd_y_rev() {
        let mut x = vec![1.0, 2.0, 3.0]; // x[0], x[1], x[2]
        let mut y = vec![10.0, 20.0, 30.0]; // y_start_idx = (-3+1)*-1 = 2. So y[2], y[1], y[0]
        dswap(3, &mut x, 1, &mut y, -1);
        // x[0](1) <-> y[2](30)
        // x[1](2) <-> y[1](20)
        // x[2](3) <-> y[0](10)
        assert_eq!(x, vec![30.0, 20.0, 10.0]);
        assert_eq!(y, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_dswap_strided_both_rev() {
        let mut x = vec![1.0, 2.0, 3.0]; // Original x values for indices [0,1,2] are 1.0, 2.0, 3.0
                                         // Accessed elements in x (by original index): x[2], x[1], x[0]
        let mut y = vec![10.0, 0.0, 20.0, 0.0, 30.0]; // Original y values for indices [0,1,2,3,4] are 10,0,20,0,30
                                                      // Accessed elements in y (by original index): y[4], y[2], y[0]

        // n=3, incx=-1, incy=-2
        // Iteration 1: swap(x[2], y[4])
        // Iteration 2: swap(x[1], y[2])
        // Iteration 3: swap(x[0], y[0])

        dswap(3, &mut x, -1, &mut y, -2);

        // After swap:
        // x[2] gets original y[4] (30.0)
        // x[1] gets original y[2] (20.0)
        // x[0] gets original y[0] (10.0)
        // So, x becomes [10.0, 20.0, 30.0]

        // y[4] gets original x[2] (3.0)
        // y[2] gets original x[1] (2.0)
        // y[0] gets original x[0] (1.0)
        // So, y becomes [1.0, 0.0, 2.0, 0.0, 3.0]

        assert_eq!(x, vec![10.0, 20.0, 30.0]);
        assert_eq!(y, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    #[should_panic(
        expected = "dswap: x slice length 3 is insufficient for n=3 and incx=2. Required: 5"
    )]
    fn test_panic_x_too_short_strided() {
        let mut x = vec![1.0, 2.0, 3.0]; // Needs len 5 for n=3, incx=2
        let mut y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        dswap(3, &mut x, 2, &mut y, 1);
    }

    #[test]
    #[should_panic(
        expected = "dswap: y slice length 1 is insufficient for n=2 and incy=1. Required: 2"
    )]
    fn test_panic_y_too_short_contiguous() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![10.0]; // Needs len 2 for n=2, incy=1
        dswap(2, &mut x, 1, &mut y, 1);
    }
}
