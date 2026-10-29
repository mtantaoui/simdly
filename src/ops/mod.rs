//! Element-wise arithmetic and comparisons on slices, through [`SliceExt`].
//!
//! Horizontal reductions (`sum`, `product`, `min`, `max`) are planned.

mod arith;
// Always compiled: `math` uses `Compare`, though the slice comparisons
// return `Vec`s.
#[cfg_attr(
    not(feature = "alloc"),
    allow(dead_code, reason = "only `Compare` is used without `alloc`")
)]
mod cmp;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub(crate) use arith::Arith;
#[cfg(feature = "alloc")]
use arith::BinaryOp;
use arith::{AddOp, ArithElement, DivOp, MulOp, SubOp};
pub(crate) use cmp::Compare;
#[cfg(feature = "alloc")]
use cmp::{CmpElement, CmpOp, EqOp, GeOp, GtOp, LeOp, LtOp, NeOp};

pub(crate) mod sealed {
    pub trait Sealed {}
    impl Sealed for [f32] {}
    impl Sealed for [f64] {}
}

/// Element-wise arithmetic and comparisons on slices, using the fastest
/// backend available (see [`backend`](crate::backend)).
///
/// Implemented for `[f32]` and `[f64]`, so it also applies to `Vec<f32>`,
/// arrays and anything else that dereferences to a slice. Results are identical to the
/// plain scalar loop, bit for bit.
///
/// The `_assign` methods work in place, so they need no allocator; the others
/// return a new `Vec` and need the `alloc` feature.
///
/// Comparisons follow NumPy (`numpy.less`, …) and Rust's operators: any
/// comparison with NaN is `false`, except `not_equal`, which is `true`, and
/// `-0.0 == 0.0`. The `_scalar` variants compare every element with one value.
///
/// ```
/// use simdly::SliceExt;
///
/// let mut a = [1.0_f32, 2.0, 3.0];
/// a.mul_assign(&[4.0, 5.0, 6.0]);
/// assert_eq!(a, [4.0, 10.0, 18.0]);
///
/// let mut b = [1.0_f64, 2.0, 3.0];
/// b.sub_assign(&[0.5, 0.5, 0.5]);
/// assert_eq!(b, [0.5, 1.5, 2.5]);
/// ```
///
/// # Panics
///
/// Every method taking a slice `rhs` panics if it and `self` have different
/// lengths.
pub trait SliceExt<T>: sealed::Sealed {
    /// `self[i] + rhs[i]`.
    ///
    /// ```
    /// use simdly::SliceExt;
    ///
    /// let a = vec![1.0_f32, 2.0, 3.0];
    /// assert_eq!(a.add(&[4.0, 5.0, 6.0]), [5.0, 7.0, 9.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn add(&self, rhs: &[T]) -> Vec<T>;

    /// `self[i] - rhs[i]`.
    #[cfg(feature = "alloc")]
    fn sub(&self, rhs: &[T]) -> Vec<T>;

    /// `self[i] * rhs[i]`.
    #[cfg(feature = "alloc")]
    fn mul(&self, rhs: &[T]) -> Vec<T>;

    /// `self[i] / rhs[i]`.
    #[cfg(feature = "alloc")]
    fn div(&self, rhs: &[T]) -> Vec<T>;

    /// `self[i] += rhs[i]`.
    fn add_assign(&mut self, rhs: &[T]);

    /// `self[i] -= rhs[i]`.
    fn sub_assign(&mut self, rhs: &[T]);

    /// `self[i] *= rhs[i]`.
    fn mul_assign(&mut self, rhs: &[T]);

    /// `self[i] /= rhs[i]`.
    fn div_assign(&mut self, rhs: &[T]);

    /// `self[i] < rhs[i]`.
    ///
    /// ```
    /// use simdly::SliceExt;
    ///
    /// let a = [1.0_f32, 5.0, f32::NAN, 2.0];
    /// assert_eq!(a.less(&[2.0, 3.0, 1.0, 2.0]), [true, false, false, false]);
    /// assert_eq!(a.less_scalar(2.0), [true, false, false, false]);
    /// ```
    #[cfg(feature = "alloc")]
    fn less(&self, rhs: &[T]) -> Vec<bool>;

    /// `self[i] <= rhs[i]`.
    #[cfg(feature = "alloc")]
    fn less_equal(&self, rhs: &[T]) -> Vec<bool>;

    /// `self[i] > rhs[i]`.
    #[cfg(feature = "alloc")]
    fn greater(&self, rhs: &[T]) -> Vec<bool>;

    /// `self[i] >= rhs[i]`.
    #[cfg(feature = "alloc")]
    fn greater_equal(&self, rhs: &[T]) -> Vec<bool>;

    /// `self[i] == rhs[i]`.
    #[cfg(feature = "alloc")]
    fn equal(&self, rhs: &[T]) -> Vec<bool>;

    /// `self[i] != rhs[i]`; `true` if either is NaN.
    #[cfg(feature = "alloc")]
    fn not_equal(&self, rhs: &[T]) -> Vec<bool>;

    /// `self[i] < rhs`.
    #[cfg(feature = "alloc")]
    fn less_scalar(&self, rhs: T) -> Vec<bool>;

    /// `self[i] <= rhs`.
    #[cfg(feature = "alloc")]
    fn less_equal_scalar(&self, rhs: T) -> Vec<bool>;

    /// `self[i] > rhs`.
    #[cfg(feature = "alloc")]
    fn greater_scalar(&self, rhs: T) -> Vec<bool>;

    /// `self[i] >= rhs`.
    #[cfg(feature = "alloc")]
    fn greater_equal_scalar(&self, rhs: T) -> Vec<bool>;

    /// `self[i] == rhs`.
    #[cfg(feature = "alloc")]
    fn equal_scalar(&self, rhs: T) -> Vec<bool>;

    /// `self[i] != rhs`; `true` if either is NaN.
    #[cfg(feature = "alloc")]
    fn not_equal_scalar(&self, rhs: T) -> Vec<bool>;
}

macro_rules! impl_slice_ext {
    ($($T:ty),*) => {
        $(
            impl SliceExt<$T> for [$T] {
                #[cfg(feature = "alloc")]
                #[inline]
                fn add(&self, rhs: &[$T]) -> Vec<$T> {
                    binary_to_vec::<AddOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn sub(&self, rhs: &[$T]) -> Vec<$T> {
                    binary_to_vec::<SubOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn mul(&self, rhs: &[$T]) -> Vec<$T> {
                    binary_to_vec::<MulOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn div(&self, rhs: &[$T]) -> Vec<$T> {
                    binary_to_vec::<DivOp, _>(self, rhs)
                }

                #[inline]
                fn add_assign(&mut self, rhs: &[$T]) {
                    check_lengths(self.len(), rhs.len());
                    <$T as ArithElement>::binary_in_place::<AddOp>(self, rhs);
                }

                #[inline]
                fn sub_assign(&mut self, rhs: &[$T]) {
                    check_lengths(self.len(), rhs.len());
                    <$T as ArithElement>::binary_in_place::<SubOp>(self, rhs);
                }

                #[inline]
                fn mul_assign(&mut self, rhs: &[$T]) {
                    check_lengths(self.len(), rhs.len());
                    <$T as ArithElement>::binary_in_place::<MulOp>(self, rhs);
                }

                #[inline]
                fn div_assign(&mut self, rhs: &[$T]) {
                    check_lengths(self.len(), rhs.len());
                    <$T as ArithElement>::binary_in_place::<DivOp>(self, rhs);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn less(&self, rhs: &[$T]) -> Vec<bool> {
                    cmp_to_vec::<LtOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn less_equal(&self, rhs: &[$T]) -> Vec<bool> {
                    cmp_to_vec::<LeOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn greater(&self, rhs: &[$T]) -> Vec<bool> {
                    cmp_to_vec::<GtOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn greater_equal(&self, rhs: &[$T]) -> Vec<bool> {
                    cmp_to_vec::<GeOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn equal(&self, rhs: &[$T]) -> Vec<bool> {
                    cmp_to_vec::<EqOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn not_equal(&self, rhs: &[$T]) -> Vec<bool> {
                    cmp_to_vec::<NeOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn less_scalar(&self, rhs: $T) -> Vec<bool> {
                    cmp_scalar_to_vec::<LtOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn less_equal_scalar(&self, rhs: $T) -> Vec<bool> {
                    cmp_scalar_to_vec::<LeOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn greater_scalar(&self, rhs: $T) -> Vec<bool> {
                    cmp_scalar_to_vec::<GtOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn greater_equal_scalar(&self, rhs: $T) -> Vec<bool> {
                    cmp_scalar_to_vec::<GeOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn equal_scalar(&self, rhs: $T) -> Vec<bool> {
                    cmp_scalar_to_vec::<EqOp, _>(self, rhs)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn not_equal_scalar(&self, rhs: $T) -> Vec<bool> {
                    cmp_scalar_to_vec::<NeOp, _>(self, rhs)
                }
            }
        )*
    };
}

impl_slice_ext!(f32, f64);

#[cfg(feature = "alloc")]
fn binary_to_vec<Op: BinaryOp, T: ArithElement>(a: &[T], b: &[T]) -> Vec<T> {
    check_lengths(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    T::binary_uninit::<Op>(a, b, &mut out.spare_capacity_mut()[..a.len()]);
    // SAFETY: `binary_uninit` initialised the first `a.len()` elements.
    unsafe { out.set_len(a.len()) };
    out
}

#[cfg(feature = "alloc")]
fn cmp_to_vec<Op: CmpOp, T: CmpElement>(a: &[T], b: &[T]) -> Vec<bool> {
    check_lengths(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    T::cmp_uninit::<Op>(a, b, &mut out.spare_capacity_mut()[..a.len()]);
    // SAFETY: `cmp_uninit` initialised the first `a.len()` elements.
    unsafe { out.set_len(a.len()) };
    out
}

#[cfg(feature = "alloc")]
fn cmp_scalar_to_vec<Op: CmpOp, T: CmpElement>(a: &[T], b: T) -> Vec<bool> {
    let mut out = Vec::with_capacity(a.len());
    T::cmp_scalar_uninit::<Op>(a, b, &mut out.spare_capacity_mut()[..a.len()]);
    // SAFETY: `cmp_scalar_uninit` initialised the first `a.len()` elements.
    unsafe { out.set_len(a.len()) };
    out
}

#[track_caller]
fn check_lengths(lhs: usize, rhs: usize) {
    assert_eq!(
        lhs, rhs,
        "length mismatch: lhs has {lhs} elements, rhs has {rhs}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn length_mismatch_panics() {
        let panics = |f: &mut dyn FnMut()| catch_unwind(AssertUnwindSafe(f)).is_err();
        let (a, b) = ([1.0_f32; 9], [1.0_f32; 8]);
        let mut c = a;
        assert!(panics(&mut || c.add_assign(&b)));
        #[cfg(feature = "alloc")]
        assert!(panics(&mut || drop(a.div(&b))));
        #[cfg(feature = "alloc")]
        assert!(panics(&mut || drop(a.less(&b))));
    }

    #[test]
    fn works_on_vec_and_arrays() {
        let mut v = std::vec![1.0_f32, 2.0, 3.0];
        v.sub_assign(&[1.0, 1.0, 1.0]);
        assert_eq!(v, [0.0, 1.0, 2.0]);
        #[cfg(feature = "alloc")]
        assert_eq!([2.0_f32, 4.0].div(&[2.0, 8.0]), [1.0, 0.5]);
        let mut w = std::vec![1.0_f64, 2.0, 3.0];
        w.mul_assign(&[2.0, 2.0, 2.0]);
        assert_eq!(w, [2.0, 4.0, 6.0]);
        #[cfg(feature = "alloc")]
        assert_eq!(w.less_scalar(4.0), [true, false, false]);
    }
}
