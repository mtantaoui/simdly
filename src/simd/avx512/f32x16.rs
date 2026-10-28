//! 16 `f32` lanes in an AVX-512 `__m512` register.
//!
//! Intrinsics are `unsafe` to call because they require AVX-512F. Constructors
//! are `unsafe` and require it, so an existing `F32x16` proves the CPU supports
//! it; that is what the `// SAFETY: an F32x16 exists` comments below refer to.
//!
//! Everything is `#[inline(always)]` so it compiles to AVX-512 instructions
//! when inlined into a `#[target_feature(enable = "avx512f")]` function.
//!
//! Tails use AVX-512's lane masks: masked-off lanes are neither read nor
//! written, and cannot fault.

use core::arch::x86_64::{
    __mmask16, __m512, _mm512_add_ps, _mm512_castsi512_ps, _mm512_castps_si512, _mm512_div_ps, _mm512_loadu_ps,
    _mm512_mask_storeu_ps, _mm512_maskz_loadu_ps, _mm512_mul_ps, _mm512_set1_epi32, _mm512_set1_ps,
    _mm512_setzero_ps, _mm512_storeu_ps, _mm512_sub_ps, _mm512_xor_si512,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::simd::SimdVector;

/// 16 `f32` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F32x16(__m512);

impl F32x16 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: __m512) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m512 {
        self.0
    }
}

/// Lane mask enabling the first `n < 16` lanes.
#[inline(always)]
fn mask(n: usize) -> __mmask16 {
    debug_assert!(n < 16);
    (1 << n) - 1
}

impl SimdVector for F32x16 {
    type Scalar = f32;
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        // SAFETY: the caller guarantees AVX-512F.
        Self(unsafe { _mm512_set1_ps(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees AVX-512F.
        Self(unsafe { _mm512_setzero_ps() })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        // SAFETY: the caller guarantees AVX-512F and that `ptr` is valid for
        // 16 reads.
        Self(unsafe { _mm512_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        // SAFETY: the caller guarantees AVX-512F and that `ptr` is valid for
        // the `n` enabled lanes; masked-off lanes are never read.
        Self(unsafe { _mm512_maskz_loadu_ps(mask(n), ptr) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        // SAFETY: an F32x16 exists, and the caller guarantees `ptr` is valid
        // for 16 writes.
        unsafe { _mm512_storeu_ps(ptr, self.0) }
    }

    #[inline(always)]
    unsafe fn store_partial(self, ptr: *mut f32, n: usize) {
        debug_assert!(n < Self::LANES);
        // SAFETY: an F32x16 exists, and the caller guarantees `ptr` is valid
        // for the `n` enabled lanes; masked-off lanes are never written.
        unsafe { _mm512_mask_storeu_ps(ptr, mask(n), self.0) }
    }
}

macro_rules! impl_binary_op {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $intrinsic:ident) => {
        impl $Op for F32x16 {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self {
                // SAFETY: an F32x16 exists.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }

        impl $OpAssign for F32x16 {
            #[inline(always)]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, _mm512_add_ps);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, _mm512_sub_ps);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, _mm512_mul_ps);
impl_binary_op!(Div, div, DivAssign, div_assign, _mm512_div_ps);

impl Neg for F32x16 {
    type Output = Self;

    /// Flips the sign bit, so `-(0.0)` is `-0.0` and NaN payloads are kept.
    /// Integer `xor`: the float one needs AVX-512DQ.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F32x16 exists.
        Self(unsafe { _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(self.0), _mm512_set1_epi32(i32::MIN))) })
    }
}

impl From<F32x16> for [f32; 16] {
    #[inline(always)]
    fn from(v: F32x16) -> Self {
        let mut lanes = [0.0; 16];
        // SAFETY: `lanes` holds exactly 16 elements.
        unsafe { v.store(lanes.as_mut_ptr()) };
        lanes
    }
}

impl fmt::Debug for F32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F32x16").field(&<[f32; 16]>::from(*self)).finish()
    }
}

/// Test-only: a safe constructor, so it checks for AVX-512F itself.
#[cfg(test)]
impl From<[f32; 16]> for F32x16 {
    fn from(lanes: [f32; 16]) -> Self {
        assert!(crate::detect::has_avx512f(), "AVX-512F not available");
        // SAFETY: AVX-512F was just checked, and `lanes` holds exactly 16
        // elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::has_avx512f;
    use std::format;

    const A: [f32; 16] = [
        1.0, -2.5, f32::NAN, -0.0, 0.5, 100.0, f32::INFINITY, 8.0, 1e-40, -3.0, 0.25, -f32::INFINITY, 7.0,
        -0.5, 1e30, -1e-30,
    ];
    const B: [f32; 16] = [
        2.0, 4.0, -1.5, 0.0, -0.25, 0.1, 9.0, -8.0, 3.0, 0.0, -0.0, 2.0, f32::NAN, 0.5, 1e10, -2.0,
    ];
    const SENTINEL: f32 = 42.0;

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_lanes_eq(got: [f32; 16], want: [f32; 16]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
            assert!(same, "lane {i}: got {g:?}, want {w:?}");
        }
    }

    #[test]
    fn splat_and_zero() {
        if !has_avx512f() {
            return;
        }
        // SAFETY: AVX-512F checked above.
        let (s, z) = unsafe { (F32x16::splat(1.5), F32x16::zero()) };
        assert_lanes_eq(s.into(), [1.5; 16]);
        assert_lanes_eq(z.into(), [0.0; 16]);
    }

    #[test]
    fn raw_round_trip() {
        if !has_avx512f() {
            return;
        }
        // SAFETY: AVX-512F checked above.
        let v = unsafe { F32x16::from_raw(F32x16::from(A).raw()) };
        assert_lanes_eq(v.into(), A);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        if !has_avx512f() {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f32; 16 + 1]);
        let src = Aligned(core::array::from_fn(|i| i as f32));
        let mut dst = Aligned([SENTINEL; 16 + 1]);

        // SAFETY: AVX-512F checked above; offset 1 leaves 16
        // readable/writable elements in each buffer.
        unsafe { F32x16::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        if !has_avx512f() {
            return;
        }
        for n in 0..16 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: AVX-512F checked above; the slice is valid for `n`
            // reads and `n < 16`.
            let v = unsafe { F32x16::load_partial(A[..n].as_ptr(), n) };
            let mut want = [0.0; 16];
            want[..n].copy_from_slice(&A[..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        if !has_avx512f() {
            return;
        }
        for n in 0..16 {
            let mut dst = [SENTINEL; 16];
            // SAFETY: `dst` is valid for 16 writes and `n < 16`.
            unsafe { F32x16::from(A).store_partial(dst.as_mut_ptr(), n) };
            let mut want = [SENTINEL; 16];
            want[..n].copy_from_slice(&A[..n]);
            assert_lanes_eq(dst, want);
        }
    }

    #[test]
    fn binary_ops_match_scalar() {
        if !has_avx512f() {
            return;
        }
        let (a, b) = (F32x16::from(A), F32x16::from(B));
        let scalar = |f: fn(f32, f32) -> f32| core::array::from_fn(|i| f(A[i], B[i]));

        assert_lanes_eq((a + b).into(), scalar(|x, y| x + y));
        assert_lanes_eq((a - b).into(), scalar(|x, y| x - y));
        assert_lanes_eq((a * b).into(), scalar(|x, y| x * y));
        assert_lanes_eq((a / b).into(), scalar(|x, y| x / y));
    }

    #[test]
    fn assign_ops_match_binary_ops() {
        if !has_avx512f() {
            return;
        }
        let (a, b) = (F32x16::from(A), F32x16::from(B));
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
        if !has_avx512f() {
            return;
        }
        let x: [f32; 16] = [
            0.0, -0.0, 1.0, -1.0, f32::INFINITY, -f32::INFINITY, f32::NAN, -f32::NAN, 1e-40, -1e-40, 2.5,
            -2.5, f32::MAX, f32::MIN, 3.0, -3.0,
        ];
        let got: [f32; 16] = (-F32x16::from(x)).into();
        for (g, x) in got.iter().zip(x) {
            assert_eq!(g.to_bits(), x.to_bits() ^ 0x8000_0000);
        }
    }

    #[test]
    fn debug_prints_lanes() {
        if !has_avx512f() {
            return;
        }
        assert_eq!(
            format!("{:?}", F32x16::from([1.0; 16])),
            "F32x16([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])"
        );
    }
}
