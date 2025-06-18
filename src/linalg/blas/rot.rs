/// Applies a plane rotation to two double-precision vectors, `x` and `y`.
///
/// For `n` elements, this function updates `x` and `y` as follows:
/// `x[i] = c * x[i] + s * y[i]`
/// `y[i] = c * y[i] - s * x[i]`
/// using the original values of `x[i]` and `y[i]` for the right-hand side computations.
/// Strides `incx` and `incy` are respected.
///
/// This function mirrors the functionality of the BLAS `drot` routine.
///
/// # Arguments
///
/// * `n`: The number of elements to process in `x` and `y`.
/// * `x`: A mutable slice representing the first input/output vector.
/// * `incx`: The increment (stride) for accessing elements in `x`.
/// * `y`: A mutable slice representing the second input/output vector.
/// * `incy`: The increment (stride) for accessing elements in `y`.
/// * `c`: The cosine component of the rotation.
/// * `s`: The sine component of the rotation.
///
/// # Panics
///
/// This function will panic if:
/// * `n > 1` and either `incx` or `incy` is zero.
/// * The effective length required to access `n` elements in `x` (considering `incx`)
///   exceeds `x.len()`. This check occurs if `n > 0`.
/// * The effective length required to access `n` elements in `y` (considering `incy`)
///   exceeds `y.len()`. This check occurs if `n > 0`.
///
/// # Notes on Parallelism
///
/// For the common case where `incx == 1` and `incy == 1` (contiguous memory access),
/// the operation can be parallelized using Rayon, for example by zipping
/// parallel mutable iterators. However, BLAS Level 1 routines are often memory-bandwidth
/// bound, so parallelism benefits are typically seen only for very large `n`.
///
/// # Examples
///
/// ```
/// ```
pub fn drot(n: usize, x: &mut [f64], incx: isize, y: &mut [f64], incy: isize, c: f64, s: f64) {
    // 1. Handle n == 0 (early exit, no access)
    if n == 0 {
        return;
    }

    // 2. Validate incx/incy if n > 1
    if n > 1 && (incx == 0 || incy == 0) {
        panic!("drot: incx or incy is 0 but n > 1, which is ambiguous or an error condition.");
    }

    // 3. Validate slice lengths x.len() and y.len()
    // (n is guaranteed to be >= 1 at this point)
    let required_len =
        |count: usize, increment: isize| -> usize { 1 + (count - 1) * increment.unsigned_abs() };

    let req_x_len = required_len(n, incx);
    let req_y_len = required_len(n, incy);

    if x.len() < req_x_len {
        panic!(
            "drot: x slice length {} is insufficient for n={} and incx={}. Required: {}",
            x.len(),
            n,
            incx,
            req_x_len
        );
    }
    if y.len() < req_y_len {
        panic!(
            "drot: y slice length {} is insufficient for n={} and incy={}. Required: {}",
            y.len(),
            n,
            incy,
            req_y_len
        );
    }

    // (if c=1 and s=0, it's an identity rotation.
    if c == 1.0 && s == 0.0 {
        return;
    }

    // 4. Perform the rotation
    if incx == 1 && incy == 1 {
        // Contiguous case
        let x_slice = &mut x[..n]; // Slicing safe due to req_x_len
        let y_slice = &mut y[..n]; // Slicing safe due to req_y_len

        // --- use Rayon par_iter_mut for parallel version ---
        x_slice
            .iter_mut()
            .zip(y_slice.iter_mut())
            .for_each(|(x_elem, y_elem)| {
                let x_orig = *x_elem;
                let y_orig = *y_elem;
                let temp_x = c * x_orig + s * y_orig;
                *y_elem = c * y_orig - s * x_orig;
                *x_elem = temp_x;
            });
    } else {
        // Strided case
        // (Also correctly handles n=1 with any incx/incy, as current_ix/iy will be 0)
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

        for _ in 0..n {
            // Indexing is safe due to req_x_len/req_y_len checks.
            let x_orig = x[current_ix as usize];
            let y_orig = y[current_iy as usize];

            let temp_x = c * x_orig + s * y_orig;
            y[current_iy as usize] = c * y_orig - s * x_orig;
            x[current_ix as usize] = temp_x;

            current_ix += incx;
            current_iy += incy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-9;

    fn assert_f64_vec_eq(a: &[f64], b: &[f64], tol: f64, msg_prefix: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{}: Vector lengths differ: {} vs {}",
            msg_prefix,
            a.len(),
            b.len()
        );
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tol,
                "{msg_prefix}: Mismatch at index {i}: {val_a} != {val_b} (within tolerance {tol})"
            );
        }
    }

    #[test]
    fn test_drot_n_zero() {
        let mut x = vec![1.0];
        let x_orig = x.clone();
        let mut y = vec![2.0];
        let y_orig = y.clone();
        drot(0, &mut x, 1, &mut y, 1, 0.0, 1.0);
        assert_f64_vec_eq(&x, &x_orig, EPSILON, "x (n=0)");
        assert_f64_vec_eq(&y, &y_orig, EPSILON, "y (n=0)");
    }

    #[test]
    #[should_panic(expected = "drot: incx or incy is 0 but n > 1")]
    fn test_drot_incx_zero_n_gt_one_panics() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        drot(2, &mut x, 0, &mut y, 1, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "drot: incx or incy is 0 but n > 1")]
    fn test_drot_incy_zero_n_gt_one_panics() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        drot(2, &mut x, 1, &mut y, 0, 1.0, 0.0);
    }

    #[test]
    fn test_drot_n_one_inc_zero() {
        let mut x = vec![3.0];
        let mut y = vec![4.0];
        // c=0.6 (3/5), s=0.8 (4/5)
        // x_new = 0.6*3 + 0.8*4 = 1.8 + 3.2 = 5.0
        // y_new = 0.6*4 - 0.8*3 = 2.4 - 2.4 = 0.0
        drot(1, &mut x, 0, &mut y, 0, 0.6, 0.8);
        assert_f64_vec_eq(&x, &[5.0], EPSILON, "x (n=1, inc=0)");
        assert_f64_vec_eq(&y, &[0.0], EPSILON, "y (n=1, inc=0)");
    }

    #[test]
    fn test_drot_n_one_any_inc() {
        let mut x = vec![3.0, 99.0];
        let mut y = vec![4.0, 88.0];
        drot(1, &mut x, 5, &mut y, -7, 0.6, 0.8); // Operates on x[0], y[0]
        assert_f64_vec_eq(&x, &[5.0, 99.0], EPSILON, "x (n=1, any_inc)");
        assert_f64_vec_eq(&y, &[0.0, 88.0], EPSILON, "y (n=1, any_inc)");
    }

    #[test]
    #[should_panic(
        expected = "drot: x slice length 0 is insufficient for n=1 and incx=1. Required: 1"
    )]
    fn test_drot_empty_slice_x_panics() {
        let mut x = vec![];
        let mut y = vec![1.0];
        drot(1, &mut x, 1, &mut y, 1, 1.0, 0.0);
    }

    #[test]
    #[should_panic(
        expected = "drot: y slice length 0 is insufficient for n=1 and incy=1. Required: 1"
    )]
    fn test_drot_empty_slice_y_panics() {
        let mut x = vec![1.0];
        let mut y = vec![];
        drot(1, &mut x, 1, &mut y, 1, 1.0, 0.0);
    }

    #[test]
    fn test_drot_identity_rotation_contiguous() {
        let mut x = vec![1.0, 2.0, 3.0];
        let x_orig = x.clone();
        let mut y = vec![4.0, 5.0, 6.0];
        let y_orig = y.clone();
        drot(3, &mut x, 1, &mut y, 1, 1.0, 0.0); // c=1, s=0
        assert_f64_vec_eq(&x, &x_orig, EPSILON, "x (identity)");
        assert_f64_vec_eq(&y, &y_orig, EPSILON, "y (identity)");
    }

    #[test]
    fn test_drot_90_degree_rotation_contiguous() {
        let mut x = vec![1.0, 3.0, 0.0]; // x_orig for relevant part: [1,3]
        let mut y = vec![2.0, 4.0, 0.0]; // y_orig for relevant part: [2,4]
        let n = 2;
        // c=0, s=1: x_new = y_old, y_new = -x_old
        drot(n, &mut x, 1, &mut y, 1, 0.0, 1.0);
        assert_f64_vec_eq(&x, &[2.0, 4.0, 0.0], EPSILON, "x (90deg)");
        assert_f64_vec_eq(&y, &[-1.0, -3.0, 0.0], EPSILON, "y (90deg)");
    }

    #[test]
    fn test_drot_example_rotation_contiguous() {
        let mut x = vec![3.0, 5.0, 99.0]; // x_orig for relevant: [3,5]
        let mut y = vec![4.0, 12.0, 88.0]; // y_orig for relevant: [4,12]
        let n = 2;
        // For (3,4), c=3/5=0.6, s=4/5=0.8. x_new=5, y_new=0
        // For (5,12), c=5/13, s=12/13. x_new=13, y_new=0
        // Use c=0.6, s=0.8 for both
        // x0_new = 0.6*3 + 0.8*4 = 1.8+3.2 = 5.0
        // y0_new = 0.6*4 - 0.8*3 = 2.4-2.4 = 0.0
        // x1_new = 0.6*5 + 0.8*12 = 3.0+9.6 = 12.6
        // y1_new = 0.6*12 - 0.8*5 = 7.2-4.0 = 3.2
        drot(n, &mut x, 1, &mut y, 1, 0.6, 0.8);
        assert_f64_vec_eq(&x, &[5.0, 12.6, 99.0], EPSILON, "x (example)");
        assert_f64_vec_eq(&y, &[0.0, 3.2, 88.0], EPSILON, "y (example)");
    }

    #[test]
    fn test_drot_strided_positive_incs() {
        let mut x = vec![1.0, 0.0, 3.0, 0.0, 0.0]; // x[0]=1, x[2]=3
        let mut y = vec![2.0, 0.0, 4.0, 0.0, 0.0]; // y[0]=2, y[2]=4
        let n = 2;
        // c=0, s=1: x_new = y_old, y_new = -x_old
        // x[0](1), y[0](2) => x[0]=2, y[0]=-1
        // x[2](3), y[2](4) => x[2]=4, y[2]=-3
        drot(n, &mut x, 2, &mut y, 2, 0.0, 1.0);
        assert_f64_vec_eq(&x, &[2.0, 0.0, 4.0, 0.0, 0.0], EPSILON, "x (strided +)");
        assert_f64_vec_eq(&y, &[-1.0, 0.0, -3.0, 0.0, 0.0], EPSILON, "y (strided +)");
    }

    #[test]
    fn test_drot_strided_mixed_signs() {
        // x: incx = 1. Elements x[0], x[1]
        // y: incy = -1. n=2. start_iy = (-(2)+1)*-1 = 1. Elements y[1], y[0]
        let mut x = vec![1.0, 3.0, 99.0]; // x_orig: x[0]=1, x[1]=3
        let mut y = vec![20.0, 40.0, 88.0]; // y_orig: y[1]=40, y[0]=20
        let n = 2;
        // c=0, s=1: x_new = y_old, y_new = -x_old
        // Iter 1: x_elem=x[0](1), y_elem=y[1](40) => x[0]=40, y[1]=-1
        // Iter 2: x_elem=x[1](3), y_elem=y[0](20) => x[1]=20, y[0]=-3
        drot(n, &mut x, 1, &mut y, -1, 0.0, 1.0);
        assert_f64_vec_eq(&x, &[40.0, 20.0, 99.0], EPSILON, "x (strided mixed)");
        assert_f64_vec_eq(&y, &[-3.0, -1.0, 88.0], EPSILON, "y (strided mixed)");
    }

    #[test]
    #[should_panic(
        expected = "drot: x slice length 3 is insufficient for n=3 and incx=2. Required: 5"
    )]
    fn test_panic_x_too_short_strided() {
        let mut x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        drot(3, &mut x, 2, &mut y, 1, 1.0, 0.0);
    }

    #[test]
    #[should_panic(
        expected = "drot: y slice length 1 is insufficient for n=2 and incy=1. Required: 2"
    )]
    fn test_panic_y_too_short_contiguous() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![10.0];
        drot(2, &mut x, 1, &mut y, 1, 1.0, 0.0);
    }
}
