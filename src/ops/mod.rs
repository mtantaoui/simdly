//! Element-wise arithmetic on slices, through [`SliceExt`].
//!
//! Horizontal reductions (`sum`, `product`, `min`, `max`) are planned.

mod arith;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use arith::BinaryOp;
use arith::{AddOp, DivOp, MulOp, SubOp};

mod sealed {
    pub trait Sealed {}
    impl Sealed for [f32] {}
}

/// Element-wise arithmetic on slices, using the fastest backend available
/// (see [`backend`](crate::backend)).
///
/// Implemented for `[f32]`, so it also applies to `Vec<f32>`, arrays and
/// anything else that dereferences to a slice. Results are identical to the
/// plain scalar loop, bit for bit.
///
/// The `_assign` methods work in place, so they need no allocator; the others
/// return a new `Vec` and need the `alloc` feature.
///
/// ```
/// use simdly::SliceExt;
///
/// let mut a = [1.0_f32, 2.0, 3.0];
/// a.mul_assign(&[4.0, 5.0, 6.0]);
/// assert_eq!(a, [4.0, 10.0, 18.0]);
/// ```
///
/// # Panics
///
/// Every method panics if `self` and `rhs` have different lengths.
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
}

impl SliceExt<f32> for [f32] {
    #[cfg(feature = "alloc")]
    #[inline]
    fn add(&self, rhs: &[f32]) -> Vec<f32> {
        binary_to_vec::<AddOp>(self, rhs)
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn sub(&self, rhs: &[f32]) -> Vec<f32> {
        binary_to_vec::<SubOp>(self, rhs)
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn mul(&self, rhs: &[f32]) -> Vec<f32> {
        binary_to_vec::<MulOp>(self, rhs)
    }

    #[cfg(feature = "alloc")]
    #[inline]
    fn div(&self, rhs: &[f32]) -> Vec<f32> {
        binary_to_vec::<DivOp>(self, rhs)
    }

    #[inline]
    fn add_assign(&mut self, rhs: &[f32]) {
        check_lengths(self.len(), rhs.len());
        arith::binary_in_place::<AddOp>(self, rhs);
    }

    #[inline]
    fn sub_assign(&mut self, rhs: &[f32]) {
        check_lengths(self.len(), rhs.len());
        arith::binary_in_place::<SubOp>(self, rhs);
    }

    #[inline]
    fn mul_assign(&mut self, rhs: &[f32]) {
        check_lengths(self.len(), rhs.len());
        arith::binary_in_place::<MulOp>(self, rhs);
    }

    #[inline]
    fn div_assign(&mut self, rhs: &[f32]) {
        check_lengths(self.len(), rhs.len());
        arith::binary_in_place::<DivOp>(self, rhs);
    }
}

#[cfg(feature = "alloc")]
fn binary_to_vec<Op: BinaryOp>(a: &[f32], b: &[f32]) -> Vec<f32> {
    check_lengths(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    arith::binary_uninit::<Op>(a, b, &mut out.spare_capacity_mut()[..a.len()]);
    // SAFETY: `binary_uninit` initialised the first `a.len()` elements.
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
    }

    #[test]
    fn works_on_vec_and_arrays() {
        let mut v = std::vec![1.0_f32, 2.0, 3.0];
        v.sub_assign(&[1.0, 1.0, 1.0]);
        assert_eq!(v, [0.0, 1.0, 2.0]);
        #[cfg(feature = "alloc")]
        assert_eq!([2.0_f32, 4.0].div(&[2.0, 8.0]), [1.0, 0.5]);
    }
}
