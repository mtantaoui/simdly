//! 4 `f32` lanes in an SSE `__m128` register.
//!
//! Intrinsics are `unsafe` to call because they require SSE4.1 (the basic
//! arithmetic here only needs SSE2, part of every `x86_64` target).
//! Constructors are `unsafe` and require SSE4.1, so an existing `F32x4` proves
//! the CPU supports it; that is what the `// SAFETY: an F32x4 exists` comments
//! below refer to.
//!
//! Everything is `#[inline(always)]` so it compiles to SSE instructions when
//! inlined into a `#[target_feature(enable = "sse4.1")]` function.
//!
//! SSE has no masked loads or stores: tails go through a register-sized
//! stack buffer, so memory past the `n` elements is neither read nor written.

use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_div_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
    _mm_storeu_ps, _mm_sub_ps, _mm_xor_ps,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ptr;

use crate::simd::SimdVector;

/// 4 `f32` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F32x4(__m128);

impl F32x4 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support SSE4.1.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: __m128) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m128 {
        self.0
    }
}

impl SimdVector for F32x4 {
    type Scalar = f32;
    const LANES: usize = 4;
    const UNROLL: bool = true;

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        // SAFETY: the caller guarantees SSE4.1.
        Self(unsafe { _mm_set1_ps(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees SSE4.1.
        Self(unsafe { _mm_setzero_ps() })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        // SAFETY: the caller guarantees SSE4.1 and that `ptr` is valid for 4
        // reads.
        Self(unsafe { _mm_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        let mut buf = [0.0; 4];
        // SAFETY: the caller guarantees SSE4.1 and that `ptr` is valid for `n`
        // reads, and `n < 4` fits in `buf`.
        unsafe {
            ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), n);
            Self::load(buf.as_ptr())
        }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        // SAFETY: an F32x4 exists, and the caller guarantees `ptr` is valid
        // for 4 writes.
        unsafe { _mm_storeu_ps(ptr, self.0) }
    }

    #[inline(always)]
    unsafe fn store_partial(self, ptr: *mut f32, n: usize) {
        debug_assert!(n < Self::LANES);
        let buf = <[f32; 4]>::from(self);
        // SAFETY: the caller guarantees `ptr` is valid for `n` writes, and
        // `n < 4` fits in `buf`.
        unsafe { ptr::copy_nonoverlapping(buf.as_ptr(), ptr, n) }
    }
}

macro_rules! impl_binary_op {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $intrinsic:ident) => {
        impl $Op for F32x4 {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self {
                // SAFETY: an F32x4 exists.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }

        impl $OpAssign for F32x4 {
            #[inline(always)]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, _mm_add_ps);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, _mm_sub_ps);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, _mm_mul_ps);
impl_binary_op!(Div, div, DivAssign, div_assign, _mm_div_ps);

impl Neg for F32x4 {
    type Output = Self;

    /// Flips the sign bit, so `-(0.0)` is `-0.0` and NaN payloads are kept.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F32x4 exists.
        Self(unsafe { _mm_xor_ps(self.0, _mm_set1_ps(-0.0)) })
    }
}

impl From<F32x4> for [f32; 4] {
    #[inline(always)]
    fn from(v: F32x4) -> Self {
        let mut lanes = [0.0; 4];
        // SAFETY: `lanes` holds exactly 4 elements.
        unsafe { v.store(lanes.as_mut_ptr()) };
        lanes
    }
}

impl fmt::Debug for F32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F32x4").field(&<[f32; 4]>::from(*self)).finish()
    }
}

/// Test-only: a safe constructor, so it checks for SSE4.1 itself.
#[cfg(test)]
impl From<[f32; 4]> for F32x4 {
    fn from(lanes: [f32; 4]) -> Self {
        assert!(crate::detect::has_sse41(), "SSE4.1 not available");
        // SAFETY: SSE4.1 was just checked, and `lanes` holds exactly 4 elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::has_sse41;
    use std::format;

    /// Two registers' worth, so every special value meets a finite one.
    const A: [[f32; 4]; 2] = [
        [1.0, -2.5, f32::NAN, -0.0],
        [1e-40, 100.0, f32::INFINITY, 8.0],
    ];
    const B: [[f32; 4]; 2] = [[2.0, 4.0, -1.5, 0.0], [-0.25, 0.1, 9.0, -8.0]];
    const SENTINEL: f32 = 42.0;

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_lanes_eq(got: [f32; 4], want: [f32; 4]) {
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
        let (s, z) = unsafe { (F32x4::splat(1.5), F32x4::zero()) };
        assert_lanes_eq(s.into(), [1.5; 4]);
        assert_lanes_eq(z.into(), [0.0; 4]);
    }

    #[test]
    fn raw_round_trip() {
        if !has_sse41() {
            return;
        }
        // SAFETY: SSE4.1 checked above.
        let v = unsafe { F32x4::from_raw(F32x4::from(A[0]).raw()) };
        assert_lanes_eq(v.into(), A[0]);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        if !has_sse41() {
            return;
        }
        #[repr(align(16))]
        struct Aligned([f32; 4 + 1]);
        let src = Aligned(core::array::from_fn(|i| i as f32));
        let mut dst = Aligned([SENTINEL; 4 + 1]);

        // SAFETY: SSE4.1 checked above; offset 1 leaves 4
        // readable/writable elements in each buffer.
        unsafe { F32x4::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        if !has_sse41() {
            return;
        }
        for n in 0..4 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: SSE4.1 checked above; the slice is valid for
            // `n` reads and `n < 4`.
            let v = unsafe { F32x4::load_partial(A[0][..n].as_ptr(), n) };
            let mut want = [0.0; 4];
            want[..n].copy_from_slice(&A[0][..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        if !has_sse41() {
            return;
        }
        for n in 0..4 {
            let mut dst = [SENTINEL; 4];
            // SAFETY: `dst` is valid for 4 writes and `n < 4`.
            unsafe { F32x4::from(A[0]).store_partial(dst.as_mut_ptr(), n) };
            let mut want = [SENTINEL; 4];
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
            let (va, vb) = (F32x4::from(a), F32x4::from(b));
            let scalar = |f: fn(f32, f32) -> f32| core::array::from_fn(|i| f(a[i], b[i]));

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
        let (a, b) = (F32x4::from(A[0]), F32x4::from(B[0]));
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
        let xs = [
            [0.0, -0.0, 1.0, -1.0],
            [f32::INFINITY, -f32::INFINITY, f32::NAN, -f32::NAN],
        ];
        for x in xs {
            let got: [f32; 4] = (-F32x4::from(x)).into();
            for (g, x) in got.iter().zip(x) {
                assert_eq!(g.to_bits(), x.to_bits() ^ (1 << 31));
            }
        }
    }

    #[test]
    fn debug_prints_lanes() {
        if !has_sse41() {
            return;
        }
        assert_eq!(format!("{:?}", F32x4::from([1.0; 4])), "F32x4([1.0, 1.0, 1.0, 1.0])");
    }
}
