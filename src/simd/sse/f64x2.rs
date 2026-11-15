//! Two `f64` lanes in an SSE `__m128d` register.
//!
//! Intrinsics are `unsafe` to call because they require SSE4.1 (the basic
//! arithmetic here only needs SSE2, part of every `x86_64` target).
//! Constructors are `unsafe` and require SSE4.1, so an existing `F64x2` proves
//! the CPU supports it; that is what the `// SAFETY: an F64x2 exists` comments
//! below refer to.
//!
//! Everything is `#[inline(always)]` so it compiles to SSE instructions when
//! inlined into a `#[target_feature(enable = "sse4.1")]` function.
//!
//! SSE has no masked loads or stores: tails go through a register-sized
//! stack buffer, so memory past the `n` elements is neither read nor written.

use core::arch::x86_64::{
    __m128d, _mm_add_pd, _mm_div_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_setzero_pd,
    _mm_storeu_pd, _mm_sub_pd, _mm_xor_pd,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ptr;

use crate::simd::SimdVector;

/// 2 `f64` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F64x2(__m128d);

impl F64x2 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE4.1.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: __m128d) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m128d {
        self.0
    }
}

impl SimdVector for F64x2 {
    type Scalar = f64;
    const LANES: usize = 2;
    const UNROLL: bool = true;

    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        // SAFETY: the caller guarantees SSE4.1.
        Self(unsafe { _mm_set1_pd(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees SSE4.1.
        Self(unsafe { _mm_setzero_pd() })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> Self {
        // SAFETY: the caller guarantees SSE4.1 and that `ptr` is valid for 2
        // reads.
        Self(unsafe { _mm_loadu_pd(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f64, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        let mut buf = [0.0; 2];
        // SAFETY: the caller guarantees SSE4.1 and that `ptr` is valid for `n`
        // reads, and `n < 2` fits in `buf`.
        unsafe {
            ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), n);
            Self::load(buf.as_ptr())
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f64) {
        // SAFETY: an F64x2 exists, and the caller guarantees `ptr` is valid
        // for 2 writes.
        unsafe { _mm_storeu_pd(ptr, self.0) }
    }

    #[inline(always)]
    unsafe fn store_partial(self, ptr: *mut f64, n: usize) {
        debug_assert!(n < Self::LANES);
        let buf = <[f64; 2]>::from(self);
        // SAFETY: the caller guarantees `ptr` is valid for `n` writes, and
        // `n < 2` fits in `buf`.
        unsafe { ptr::copy_nonoverlapping(buf.as_ptr(), ptr, n) }
    }
}

macro_rules! impl_binary_op {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $intrinsic:ident) => {
        impl $Op for F64x2 {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self {
                // SAFETY: an F64x2 exists.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }

        impl $OpAssign for F64x2 {
            #[inline(always)]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, _mm_add_pd);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, _mm_sub_pd);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, _mm_mul_pd);
impl_binary_op!(Div, div, DivAssign, div_assign, _mm_div_pd);

impl Neg for F64x2 {
    type Output = Self;

    /// Flips the sign bit, so `-(0.0)` is `-0.0` and NaN payloads are kept.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F64x2 exists.
        Self(unsafe { _mm_xor_pd(self.0, _mm_set1_pd(-0.0)) })
    }
}

impl From<F64x2> for [f64; 2] {
    #[inline(always)]
    fn from(v: F64x2) -> Self {
        let mut lanes = [0.0; 2];
        // SAFETY: `lanes` holds exactly 2 elements.
        unsafe { v.store(lanes.as_mut_ptr()) };
        lanes
    }
}

impl fmt::Debug for F64x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F64x2").field(&<[f64; 2]>::from(*self)).finish()
    }
}

/// Test-only: a safe constructor, so it checks for SSE4.1 itself.
#[cfg(test)]
impl From<[f64; 2]> for F64x2 {
    fn from(lanes: [f64; 2]) -> Self {
        assert!(crate::detect::has_sse41(), "SSE4.1 not available");
        // SAFETY: SSE4.1 was just checked, and `lanes` holds exactly 2 elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::has_sse41;
    use std::format;

    /// Four registers' worth, so every special value meets a finite one.
    const A: [[f64; 2]; 4] = [[1.0, -2.5], [f64::NAN, -0.0], [1e-310, 100.0], [f64::INFINITY, 8.0]];
    const B: [[f64; 2]; 4] = [[2.0, 4.0], [-1.5, 0.0], [-0.25, 0.1], [9.0, -8.0]];
    const SENTINEL: f64 = 42.0;

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_lanes_eq(got: [f64; 2], want: [f64; 2]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
            assert!(same, "lane {i}: got {g:?}, want {w:?}");
        }
    }

    #[test]
    fn splat_and_zero() {
        if !has_sse41() {
            return;
        }
        // SAFETY: SSE4.1 checked above.
        let (s, z) = unsafe { (F64x2::splat(1.5), F64x2::zero()) };
        assert_lanes_eq(s.into(), [1.5; 2]);
        assert_lanes_eq(z.into(), [0.0; 2]);
    }

    #[test]
    fn raw_round_trip() {
        if !has_sse41() {
            return;
        }
        // SAFETY: SSE4.1 checked above.
        let v = unsafe { F64x2::from_raw(F64x2::from(A[0]).raw()) };
        assert_lanes_eq(v.into(), A[0]);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        if !has_sse41() {
            return;
        }
        #[repr(align(16))]
        struct Aligned([f64; 2 + 1]);
        let src = Aligned(core::array::from_fn(|i| i as f64));
        let mut dst = Aligned([SENTINEL; 2 + 1]);

        // SAFETY: SSE4.1 checked above; offset 1 leaves 2
        // readable/writable elements in each buffer.
        unsafe { F64x2::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        if !has_sse41() {
            return;
        }
        for n in 0..2 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: SSE4.1 checked above; the slice is valid for
            // `n` reads and `n < 2`.
            let v = unsafe { F64x2::load_partial(A[0][..n].as_ptr(), n) };
            let mut want = [0.0; 2];
            want[..n].copy_from_slice(&A[0][..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        if !has_sse41() {
            return;
        }
        for n in 0..2 {
            let mut dst = [SENTINEL; 2];
            // SAFETY: `dst` is valid for 2 writes and `n < 2`.
            unsafe { F64x2::from(A[0]).store_partial(dst.as_mut_ptr(), n) };
            let mut want = [SENTINEL; 2];
            want[..n].copy_from_slice(&A[0][..n]);
            assert_lanes_eq(dst, want);
        }
    }

    #[test]
    fn binary_ops_match_scalar() {
        if !has_sse41() {
            return;
        }
        for (a, b) in A.into_iter().zip(B) {
            let (va, vb) = (F64x2::from(a), F64x2::from(b));
            let scalar = |f: fn(f64, f64) -> f64| core::array::from_fn(|i| f(a[i], b[i]));

            assert_lanes_eq((va + vb).into(), scalar(|x, y| x + y));
            assert_lanes_eq((va - vb).into(), scalar(|x, y| x - y));
            assert_lanes_eq((va * vb).into(), scalar(|x, y| x * y));
            assert_lanes_eq((va / vb).into(), scalar(|x, y| x / y));
        }
    }

    #[test]
    fn assign_ops_match_binary_ops() {
        if !has_sse41() {
            return;
        }
        let (a, b) = (F64x2::from(A[0]), F64x2::from(B[0]));
        let mut v = a;

        v += b;
        assert_lanes_eq(v.into(), (a + b).into());
        v -= b;
        assert_lanes_eq(v.into(), ((a + b) - b).into());
        v *= b;
        assert_lanes_eq(v.into(), (((a + b) - b) * b).into());
        v /= b;
        assert_lanes_eq(v.into(), ((((a + b) - b) * b) / b).into());
    }

    #[test]
    fn neg_flips_only_the_sign_bit() {
        if !has_sse41() {
            return;
        }
        let xs = [[0.0, -0.0], [1.0, -1.0], [f64::INFINITY, -f64::INFINITY], [f64::NAN, -f64::NAN]];
        for x in xs {
            let got: [f64; 2] = (-F64x2::from(x)).into();
            for (g, x) in got.iter().zip(x) {
                assert_eq!(g.to_bits(), x.to_bits() ^ (1 << 63));
            }
        }
    }

    #[test]
    fn debug_prints_lanes() {
        if !has_sse41() {
            return;
        }
        assert_eq!(format!("{:?}", F64x2::from([1.0; 2])), "F64x2([1.0, 1.0])");
    }
}
