//! [`ParSliceExt`]: [`SliceExt`](super::SliceExt) on several threads, cut
//! as [`ParSliceMath`](crate::ParSliceMath) cuts slices, from
//! [`PARALLEL_EXT_THRESHOLD`] elements on.

use alloc::vec::Vec;

use rayon::prelude::*;

use super::arith::BinaryOp;
use super::{
    AddOp, ArithElement, CmpElement, CmpOp, DivOp, EqOp, GeOp, GtOp, LeOp, LtOp, MulOp, NeOp,
    SubOp, binary_scalar_to_vec, binary_to_vec, check_lengths, cmp_scalar_to_vec, cmp_to_vec,
};
use crate::{PARALLEL_CHUNK_SIZE, PARALLEL_EXT_THRESHOLD};

/// The arithmetic and comparisons of [`SliceExt`](crate::SliceExt) on
/// several threads, for `[f32]` and `[f64]` (with the `rayon` feature).
///
/// Slices up to 524 288 elements (twice
/// [`PARALLEL_SIMD_THRESHOLD`](crate::PARALLEL_SIMD_THRESHOLD)) run on the
/// calling thread, as `SliceExt` does: these operations do so little work per
/// element that shorter slices do not pay for the threads. The results are
/// the same, bit for bit, as `SliceExt`'s, and so are the panics on slices of
/// different lengths.
///
/// On long slices they are limited by memory bandwidth: threads help because
/// several cores move data faster than one, but less than for the math
/// functions.
///
/// ```
/// use simdly::ParSliceExt;
///
/// let a = vec![1.0_f32; 1_000_000];
/// let b = vec![2.0_f32; 1_000_000];
/// assert!(a.par_add(&b).iter().all(|&c| c == 3.0));
/// assert!(a.par_less(&b).iter().all(|&m| m));
/// ```
///
/// # Panics
///
/// Every method taking a slice `rhs` panics if it and `self` have different
/// lengths.
pub trait ParSliceExt<T>: super::sealed::Sealed {
    /// [`SliceExt::add`](crate::SliceExt::add) on several threads.
    fn par_add(&self, rhs: &[T]) -> Vec<T>;
    /// [`SliceExt::sub`](crate::SliceExt::sub) on several threads.
    fn par_sub(&self, rhs: &[T]) -> Vec<T>;
    /// [`SliceExt::mul`](crate::SliceExt::mul) on several threads.
    fn par_mul(&self, rhs: &[T]) -> Vec<T>;
    /// [`SliceExt::div`](crate::SliceExt::div) on several threads.
    fn par_div(&self, rhs: &[T]) -> Vec<T>;

    /// [`SliceExt::add_assign`](crate::SliceExt::add_assign) on several threads.
    fn par_add_assign(&mut self, rhs: &[T]);
    /// [`SliceExt::sub_assign`](crate::SliceExt::sub_assign) on several threads.
    fn par_sub_assign(&mut self, rhs: &[T]);
    /// [`SliceExt::mul_assign`](crate::SliceExt::mul_assign) on several threads.
    fn par_mul_assign(&mut self, rhs: &[T]);
    /// [`SliceExt::div_assign`](crate::SliceExt::div_assign) on several threads.
    fn par_div_assign(&mut self, rhs: &[T]);

    /// [`SliceExt::add_scalar`](crate::SliceExt::add_scalar) on several threads.
    fn par_add_scalar(&self, rhs: T) -> Vec<T>;
    /// [`SliceExt::sub_scalar`](crate::SliceExt::sub_scalar) on several threads.
    fn par_sub_scalar(&self, rhs: T) -> Vec<T>;
    /// [`SliceExt::mul_scalar`](crate::SliceExt::mul_scalar) on several threads.
    fn par_mul_scalar(&self, rhs: T) -> Vec<T>;
    /// [`SliceExt::div_scalar`](crate::SliceExt::div_scalar) on several threads.
    fn par_div_scalar(&self, rhs: T) -> Vec<T>;

    /// [`SliceExt::add_scalar_assign`](crate::SliceExt::add_scalar_assign) on several threads.
    fn par_add_scalar_assign(&mut self, rhs: T);
    /// [`SliceExt::sub_scalar_assign`](crate::SliceExt::sub_scalar_assign) on several threads.
    fn par_sub_scalar_assign(&mut self, rhs: T);
    /// [`SliceExt::mul_scalar_assign`](crate::SliceExt::mul_scalar_assign) on several threads.
    fn par_mul_scalar_assign(&mut self, rhs: T);
    /// [`SliceExt::div_scalar_assign`](crate::SliceExt::div_scalar_assign) on several threads.
    fn par_div_scalar_assign(&mut self, rhs: T);

    /// [`SliceExt::less`](crate::SliceExt::less) on several threads.
    fn par_less(&self, rhs: &[T]) -> Vec<bool>;
    /// [`SliceExt::less_equal`](crate::SliceExt::less_equal) on several threads.
    fn par_less_equal(&self, rhs: &[T]) -> Vec<bool>;
    /// [`SliceExt::greater`](crate::SliceExt::greater) on several threads.
    fn par_greater(&self, rhs: &[T]) -> Vec<bool>;
    /// [`SliceExt::greater_equal`](crate::SliceExt::greater_equal) on several threads.
    fn par_greater_equal(&self, rhs: &[T]) -> Vec<bool>;
    /// [`SliceExt::equal`](crate::SliceExt::equal) on several threads.
    fn par_equal(&self, rhs: &[T]) -> Vec<bool>;
    /// [`SliceExt::not_equal`](crate::SliceExt::not_equal) on several threads.
    fn par_not_equal(&self, rhs: &[T]) -> Vec<bool>;

    /// [`SliceExt::less_scalar`](crate::SliceExt::less_scalar) on several threads.
    fn par_less_scalar(&self, rhs: T) -> Vec<bool>;
    /// [`SliceExt::less_equal_scalar`](crate::SliceExt::less_equal_scalar) on several threads.
    fn par_less_equal_scalar(&self, rhs: T) -> Vec<bool>;
    /// [`SliceExt::greater_scalar`](crate::SliceExt::greater_scalar) on several threads.
    fn par_greater_scalar(&self, rhs: T) -> Vec<bool>;
    /// [`SliceExt::greater_equal_scalar`](crate::SliceExt::greater_equal_scalar) on several threads.
    fn par_greater_equal_scalar(&self, rhs: T) -> Vec<bool>;
    /// [`SliceExt::equal_scalar`](crate::SliceExt::equal_scalar) on several threads.
    fn par_equal_scalar(&self, rhs: T) -> Vec<bool>;
    /// [`SliceExt::not_equal_scalar`](crate::SliceExt::not_equal_scalar) on several threads.
    fn par_not_equal_scalar(&self, rhs: T) -> Vec<bool>;
}

macro_rules! impl_par_slice_ext {
    ($($T:ty),*) => {
        $(
            impl ParSliceExt<$T> for [$T] {
                par_methods!($T;
                    arith: par_add AddOp, par_sub SubOp, par_mul MulOp, par_div DivOp;
                    assign: par_add_assign AddOp, par_sub_assign SubOp,
                        par_mul_assign MulOp, par_div_assign DivOp;
                    arith_scalar: par_add_scalar AddOp, par_sub_scalar SubOp,
                        par_mul_scalar MulOp, par_div_scalar DivOp;
                    assign_scalar: par_add_scalar_assign AddOp, par_sub_scalar_assign SubOp,
                        par_mul_scalar_assign MulOp, par_div_scalar_assign DivOp;
                    cmp: par_less LtOp, par_less_equal LeOp, par_greater GtOp,
                        par_greater_equal GeOp, par_equal EqOp, par_not_equal NeOp;
                    scalar: par_less_scalar LtOp, par_less_equal_scalar LeOp,
                        par_greater_scalar GtOp, par_greater_equal_scalar GeOp,
                        par_equal_scalar EqOp, par_not_equal_scalar NeOp;
                );
            }
        )*
    };
}

macro_rules! par_methods {
    ($T:ty;
        arith: $($a:ident $AOp:ident),*;
        assign: $($s:ident $SOp:ident),*;
        arith_scalar: $($as:ident $ASOp:ident),*;
        assign_scalar: $($ss:ident $SSOp:ident),*;
        cmp: $($c:ident $COp:ident),*;
        scalar: $($k:ident $KOp:ident),*;
    ) => {
        $(
            #[inline]
            fn $a(&self, rhs: &[$T]) -> Vec<$T> {
                par_binary_to_vec::<$AOp, _>(self, rhs)
            }
        )*
        $(
            #[inline]
            fn $s(&mut self, rhs: &[$T]) {
                par_binary_in_place::<$SOp, _>(self, rhs);
            }
        )*
        $(
            #[inline]
            fn $as(&self, rhs: $T) -> Vec<$T> {
                par_binary_scalar_to_vec::<$ASOp, _>(self, rhs)
            }
        )*
        $(
            #[inline]
            fn $ss(&mut self, rhs: $T) {
                par_binary_scalar_in_place::<$SSOp, _>(self, rhs);
            }
        )*
        $(
            #[inline]
            fn $c(&self, rhs: &[$T]) -> Vec<bool> {
                par_cmp_to_vec::<$COp, _>(self, rhs)
            }
        )*
        $(
            #[inline]
            fn $k(&self, rhs: $T) -> Vec<bool> {
                par_cmp_scalar_to_vec::<$KOp, _>(self, rhs)
            }
        )*
    };
}

impl_par_slice_ext!(f32, f64);

// Each helper hands short slices, and slices of different lengths (for the
// panic), to the single-threaded function.

fn par_binary_to_vec<Op: BinaryOp, T: ArithElement + Send + Sync>(a: &[T], b: &[T]) -> Vec<T> {
    if a.len() <= PARALLEL_EXT_THRESHOLD || a.len() != b.len() {
        return binary_to_vec::<Op, T>(a, b);
    }
    let mut out = Vec::with_capacity(a.len());
    out.spare_capacity_mut()[..a.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(a.par_chunks(PARALLEL_CHUNK_SIZE))
        .zip(b.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|((out, a), b)| T::binary_uninit::<Op>(a, b, out));
    // SAFETY: the chunks cover the first `a.len()` elements, and
    // `binary_uninit` initialised each of them.
    unsafe { out.set_len(a.len()) };
    out
}

fn par_binary_in_place<Op: BinaryOp, T: ArithElement + Send + Sync>(a: &mut [T], b: &[T]) {
    check_lengths(a.len(), b.len());
    if a.len() <= PARALLEL_EXT_THRESHOLD {
        return T::binary_in_place::<Op>(a, b);
    }
    a.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(b.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|(a, b)| T::binary_in_place::<Op>(a, b));
}

fn par_binary_scalar_to_vec<Op: BinaryOp, T: ArithElement + Send + Sync>(a: &[T], b: T) -> Vec<T> {
    if a.len() <= PARALLEL_EXT_THRESHOLD {
        return binary_scalar_to_vec::<Op, T>(a, b);
    }
    let mut out = Vec::with_capacity(a.len());
    out.spare_capacity_mut()[..a.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(a.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|(out, a)| T::binary_scalar_uninit::<Op>(a, b, out));
    // SAFETY: the chunks cover the first `a.len()` elements, and
    // `binary_scalar_uninit` initialised each of them.
    unsafe { out.set_len(a.len()) };
    out
}

fn par_binary_scalar_in_place<Op: BinaryOp, T: ArithElement + Send + Sync>(a: &mut [T], b: T) {
    if a.len() <= PARALLEL_EXT_THRESHOLD {
        return T::binary_scalar_in_place::<Op>(a, b);
    }
    a.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .for_each(|a| T::binary_scalar_in_place::<Op>(a, b));
}

fn par_cmp_to_vec<Op: CmpOp, T: CmpElement + Send + Sync>(a: &[T], b: &[T]) -> Vec<bool> {
    if a.len() <= PARALLEL_EXT_THRESHOLD || a.len() != b.len() {
        return cmp_to_vec::<Op, T>(a, b);
    }
    let mut out = Vec::with_capacity(a.len());
    out.spare_capacity_mut()[..a.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(a.par_chunks(PARALLEL_CHUNK_SIZE))
        .zip(b.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|((out, a), b)| T::cmp_uninit::<Op>(a, b, out));
    // SAFETY: the chunks cover the first `a.len()` elements, and
    // `cmp_uninit` initialised each of them.
    unsafe { out.set_len(a.len()) };
    out
}

fn par_cmp_scalar_to_vec<Op: CmpOp, T: CmpElement + Send + Sync>(a: &[T], b: T) -> Vec<bool> {
    if a.len() <= PARALLEL_EXT_THRESHOLD {
        return cmp_scalar_to_vec::<Op, T>(a, b);
    }
    let mut out = Vec::with_capacity(a.len());
    out.spare_capacity_mut()[..a.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(a.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|(out, a)| T::cmp_scalar_uninit::<Op>(a, b, out));
    // SAFETY: the chunks cover the first `a.len()` elements, and
    // `cmp_scalar_uninit` initialised each of them.
    unsafe { out.set_len(a.len()) };
    out
}

#[cfg(test)]
mod tests {
    use super::ParSliceExt;
    use crate::testing::{TestFloat, assert_same};
    use crate::{PARALLEL_CHUNK_SIZE, PARALLEL_EXT_THRESHOLD, PARALLEL_SIMD_THRESHOLD, SliceExt};
    use std::vec::Vec;

    /// Long enough for several chunks and a short last one.
    const LEN: usize = PARALLEL_EXT_THRESHOLD + 3 * PARALLEL_CHUNK_SIZE + 13;

    /// Some equal pairs, so `equal` and `not_equal` see both answers, and
    /// some NaNs.
    fn inputs<T: TestFloat>(len: usize) -> (Vec<T>, Vec<T>) {
        let a = (0..len)
            .map(|i| match i % 97 {
                0 => T::from_f64(f64::NAN),
                _ => T::from_f64(((i as f64) * 0.37).sin() * 3.0),
            })
            .collect::<Vec<T>>();
        let b = (0..len)
            .map(|i| match i % 5 {
                0 => a[i],
                _ => T::from_f64(((i as f64) * 0.73).cos() * 3.0),
            })
            .collect();
        (a, b)
    }

    /// Every `par_` method against its `SliceExt` one, at `len`.
    fn check<T: TestFloat>(len: usize)
    where
        [T]: SliceExt<T> + ParSliceExt<T>,
    {
        let (a, b) = inputs::<T>(len);
        let what = |f: &str| std::format!("par_{f} {} len {len}", T::NAME);

        type Arith<T> = (fn(&[T], &[T]) -> Vec<T>, fn(&[T], &[T]) -> Vec<T>);
        let arith: [(&str, Arith<T>); 4] = [
            ("add", (|a, b| a.add(b), |a, b| a.par_add(b))),
            ("sub", (|a, b| a.sub(b), |a, b| a.par_sub(b))),
            ("mul", (|a, b| a.mul(b), |a, b| a.par_mul(b))),
            ("div", (|a, b| a.div(b), |a, b| a.par_div(b))),
        ];
        for (f, (seq, par)) in arith {
            assert_same(&par(&a, &b), &seq(&a, &b), &what(f));
        }

        type Assign<T> = (fn(&mut [T], &[T]), fn(&mut [T], &[T]));
        let assign: [(&str, Assign<T>); 4] = [
            (
                "add_assign",
                (|a, b| a.add_assign(b), |a, b| a.par_add_assign(b)),
            ),
            (
                "sub_assign",
                (|a, b| a.sub_assign(b), |a, b| a.par_sub_assign(b)),
            ),
            (
                "mul_assign",
                (|a, b| a.mul_assign(b), |a, b| a.par_mul_assign(b)),
            ),
            (
                "div_assign",
                (|a, b| a.div_assign(b), |a, b| a.par_div_assign(b)),
            ),
        ];
        for (f, (seq, par)) in assign {
            let (mut x, mut y) = (a.clone(), a.clone());
            seq(&mut x, &b);
            par(&mut y, &b);
            assert_same(&y, &x, &what(f));
        }

        type ArithScalar<T> = (fn(&[T], T) -> Vec<T>, fn(&[T], T) -> Vec<T>);
        let arith_scalar: [(&str, ArithScalar<T>); 4] = [
            (
                "add_scalar",
                (|a, b| a.add_scalar(b), |a, b| a.par_add_scalar(b)),
            ),
            (
                "sub_scalar",
                (|a, b| a.sub_scalar(b), |a, b| a.par_sub_scalar(b)),
            ),
            (
                "mul_scalar",
                (|a, b| a.mul_scalar(b), |a, b| a.par_mul_scalar(b)),
            ),
            (
                "div_scalar",
                (|a, b| a.div_scalar(b), |a, b| a.par_div_scalar(b)),
            ),
        ];
        type AssignScalar<T> = (fn(&mut [T], T), fn(&mut [T], T));
        let assign_scalar: [(&str, AssignScalar<T>); 4] = [
            (
                "add_scalar_assign",
                (
                    |a, b| a.add_scalar_assign(b),
                    |a, b| a.par_add_scalar_assign(b),
                ),
            ),
            (
                "sub_scalar_assign",
                (
                    |a, b| a.sub_scalar_assign(b),
                    |a, b| a.par_sub_scalar_assign(b),
                ),
            ),
            (
                "mul_scalar_assign",
                (
                    |a, b| a.mul_scalar_assign(b),
                    |a, b| a.par_mul_scalar_assign(b),
                ),
            ),
            (
                "div_scalar_assign",
                (
                    |a, b| a.div_scalar_assign(b),
                    |a, b| a.par_div_scalar_assign(b),
                ),
            ),
        ];
        let s = T::from_f64(-1.75);
        for (f, (seq, par)) in arith_scalar {
            assert_same(&par(&a, s), &seq(&a, s), &what(f));
        }
        for (f, (seq, par)) in assign_scalar {
            let (mut x, mut y) = (a.clone(), a.clone());
            seq(&mut x, s);
            par(&mut y, s);
            assert_same(&y, &x, &what(f));
        }

        type Cmp<T> = (fn(&[T], &[T]) -> Vec<bool>, fn(&[T], &[T]) -> Vec<bool>);
        let cmp: [(&str, Cmp<T>); 6] = [
            ("less", (|a, b| a.less(b), |a, b| a.par_less(b))),
            (
                "less_equal",
                (|a, b| a.less_equal(b), |a, b| a.par_less_equal(b)),
            ),
            ("greater", (|a, b| a.greater(b), |a, b| a.par_greater(b))),
            (
                "greater_equal",
                (|a, b| a.greater_equal(b), |a, b| a.par_greater_equal(b)),
            ),
            ("equal", (|a, b| a.equal(b), |a, b| a.par_equal(b))),
            (
                "not_equal",
                (|a, b| a.not_equal(b), |a, b| a.par_not_equal(b)),
            ),
        ];
        for (f, (seq, par)) in cmp {
            assert_eq!(par(&a, &b), seq(&a, &b), "{}", what(f));
        }

        type Scalar<T> = (fn(&[T], T) -> Vec<bool>, fn(&[T], T) -> Vec<bool>);
        let scalar: [(&str, Scalar<T>); 6] = [
            (
                "less_scalar",
                (|a, b| a.less_scalar(b), |a, b| a.par_less_scalar(b)),
            ),
            (
                "less_equal_scalar",
                (
                    |a, b| a.less_equal_scalar(b),
                    |a, b| a.par_less_equal_scalar(b),
                ),
            ),
            (
                "greater_scalar",
                (|a, b| a.greater_scalar(b), |a, b| a.par_greater_scalar(b)),
            ),
            (
                "greater_equal_scalar",
                (
                    |a, b| a.greater_equal_scalar(b),
                    |a, b| a.par_greater_equal_scalar(b),
                ),
            ),
            (
                "equal_scalar",
                (|a, b| a.equal_scalar(b), |a, b| a.par_equal_scalar(b)),
            ),
            (
                "not_equal_scalar",
                (
                    |a, b| a.not_equal_scalar(b),
                    |a, b| a.par_not_equal_scalar(b),
                ),
            ),
        ];
        // `a[1]` occurs in `a`, so `equal_scalar` sees both answers.
        let k = if len > 1 { a[1] } else { T::from_f64(0.5) };
        for (f, (seq, par)) in scalar {
            assert_eq!(par(&a, k), seq(&a, k), "{}", what(f));
        }
    }

    /// Bit for bit equal to `SliceExt`, above and below the threshold.
    #[test]
    fn par_matches_sequential() {
        // Between the two thresholds, `par_` runs on one thread here.
        for len in [
            0,
            1,
            100,
            PARALLEL_SIMD_THRESHOLD + 1,
            PARALLEL_EXT_THRESHOLD,
            LEN,
        ] {
            check::<f32>(len);
            check::<f64>(len);
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn par_length_mismatch_panics() {
        let a = std::vec![1.0_f32; LEN];
        let _ = a.par_add(&a[1..]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn par_assign_length_mismatch_panics() {
        let mut a = std::vec![1.0_f64; LEN];
        let b = a.clone();
        a.par_mul_assign(&b[1..]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn par_cmp_length_mismatch_panics() {
        let a = std::vec![1.0_f32; LEN];
        let _ = a.par_less(&a[1..]);
    }
}
