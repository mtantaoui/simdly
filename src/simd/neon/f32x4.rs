//! 4 `f32` lanes in a NEON `float32x4_t` register.
//!
//! Intrinsics are `unsafe` to call because they require NEON. Constructors are
//! `unsafe` and require it, so an existing `F32x4` proves the CPU supports it;
//! that is what the `// SAFETY: an F32x4 exists` comments below refer to.
//! NEON is part of the `aarch64` targets this module is compiled for.
//!
//! Everything is `#[inline(always)]` so it compiles to NEON instructions.
//!
//! NEON has no masked loads or stores: tails go through a register-sized
//! stack buffer, so memory past the `n` elements is neither read nor written.

use core::arch::aarch64::{
    float32x4_t, vaddq_f32, vdivq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vnegq_f32, vst1q_f32,
    vsubq_f32,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::ptr;

use crate::simd::SimdVector;

/// 4 `f32` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F32x4(float32x4_t);

impl F32x4 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support NEON.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: float32x4_t) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> float32x4_t {
        self.0
    }
}

impl SimdVector for F32x4 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        // SAFETY: the caller guarantees NEON.
        Self(unsafe { vdupq_n_f32(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees NEON.
        Self(unsafe { vdupq_n_f32(0.0) })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        // SAFETY: the caller guarantees NEON and that `ptr` is valid for 4
        // reads.
        Self(unsafe { vld1q_f32(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        let mut buf = [0.0; 4];
        // SAFETY: the caller guarantees NEON and that `ptr` is valid for `n`
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
        unsafe { vst1q_f32(ptr, self.0) }
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

impl_binary_op!(Add, add, AddAssign, add_assign, vaddq_f32);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, vsubq_f32);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, vmulq_f32);
impl_binary_op!(Div, div, DivAssign, div_assign, vdivq_f32);

impl Neg for F32x4 {
    type Output = Self;

    /// Flips the sign bit (`fneg`), so `-(0.0)` is `-0.0` and NaN payloads are
    /// kept.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F32x4 exists.
        Self(unsafe { vnegq_f32(self.0) })
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

/// Test-only: a safe constructor, possible because this module is only
/// compiled with NEON.
#[cfg(test)]
impl From<[f32; 4]> for F32x4 {
    fn from(lanes: [f32; 4]) -> Self {
        // SAFETY: NEON is enabled at compile time, and `lanes` holds exactly 4
        // elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        // SAFETY: NEON is enabled at compile time.
        let (s, z) = unsafe { (F32x4::splat(1.5), F32x4::zero()) };
        assert_lanes_eq(s.into(), [1.5; 4]);
        assert_lanes_eq(z.into(), [0.0; 4]);
    }

    #[test]
    fn raw_round_trip() {
        // SAFETY: NEON is enabled at compile time.
        let v = unsafe { F32x4::from_raw(F32x4::from(A[0]).raw()) };
        assert_lanes_eq(v.into(), A[0]);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        #[repr(align(16))]
        struct Aligned([f32; 4 + 1]);
        let src = Aligned(core::array::from_fn(|i| i as f32));
        let mut dst = Aligned([SENTINEL; 4 + 1]);

        // SAFETY: NEON is enabled at compile time; offset 1 leaves 4
        // readable/writable elements in each buffer.
        unsafe { F32x4::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        for n in 0..4 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: NEON is enabled at compile time; the slice is valid for
            // `n` reads and `n < 4`.
            let v = unsafe { F32x4::load_partial(A[0][..n].as_ptr(), n) };
            let mut want = [0.0; 4];
            want[..n].copy_from_slice(&A[0][..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
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
        assert_eq!(format!("{:?}", F32x4::from([1.0; 4])), "F32x4([1.0, 1.0, 1.0, 1.0])");
    }
}
