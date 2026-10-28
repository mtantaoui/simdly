//! 8 `f64` lanes in an AVX-512 `__m512d` register.
//!
//! Intrinsics are `unsafe` to call because they require AVX-512F. Constructors
//! are `unsafe` and require it, so an existing `F64x8` proves the CPU supports
//! it; that is what the `// SAFETY: an F64x8 exists` comments below refer to.
//!
//! Everything is `#[inline(always)]` so it compiles to AVX-512 instructions
//! when inlined into a `#[target_feature(enable = "avx512f")]` function.
//!
//! Tails use AVX-512's lane masks: masked-off lanes are neither read nor
//! written, and cannot fault.

use core::arch::x86_64::{
    __mmask8, __m512d, _mm512_add_pd, _mm512_castsi512_pd, _mm512_castpd_si512, _mm512_div_pd, _mm512_loadu_pd,
    _mm512_mask_storeu_pd, _mm512_maskz_loadu_pd, _mm512_mul_pd, _mm512_set1_epi64, _mm512_set1_pd,
    _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd, _mm512_xor_si512,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::simd::SimdVector;

/// 8 `f64` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F64x8(__m512d);

impl F64x8 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX-512F.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: __m512d) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m512d {
        self.0
    }
}

/// Lane mask enabling the first `n < 8` lanes.
#[inline(always)]
fn mask(n: usize) -> __mmask8 {
    debug_assert!(n < 8);
    (1 << n) - 1
}

impl SimdVector for F64x8 {
    type Scalar = f64;
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        // SAFETY: the caller guarantees AVX-512F.
        Self(unsafe { _mm512_set1_pd(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees AVX-512F.
        Self(unsafe { _mm512_setzero_pd() })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> Self {
        // SAFETY: the caller guarantees AVX-512F and that `ptr` is valid for
        // 8 reads.
        Self(unsafe { _mm512_loadu_pd(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f64, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        // SAFETY: the caller guarantees AVX-512F and that `ptr` is valid for
        // the `n` enabled lanes; masked-off lanes are never read.
        Self(unsafe { _mm512_maskz_loadu_pd(mask(n), ptr) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f64) {
        // SAFETY: an F64x8 exists, and the caller guarantees `ptr` is valid
        // for 8 writes.
        unsafe { _mm512_storeu_pd(ptr, self.0) }
    }

    #[inline(always)]
    unsafe fn store_partial(self, ptr: *mut f64, n: usize) {
        debug_assert!(n < Self::LANES);
        // SAFETY: an F64x8 exists, and the caller guarantees `ptr` is valid
        // for the `n` enabled lanes; masked-off lanes are never written.
        unsafe { _mm512_mask_storeu_pd(ptr, mask(n), self.0) }
    }
}

macro_rules! impl_binary_op {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $intrinsic:ident) => {
        impl $Op for F64x8 {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self {
                // SAFETY: an F64x8 exists.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }

        impl $OpAssign for F64x8 {
            #[inline(always)]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, _mm512_add_pd);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, _mm512_sub_pd);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, _mm512_mul_pd);
impl_binary_op!(Div, div, DivAssign, div_assign, _mm512_div_pd);

impl Neg for F64x8 {
    type Output = Self;

    /// Flips the sign bit, so `-(0.0)` is `-0.0` and NaN payloads are kept.
    /// Integer `xor`: the float one needs AVX-512DQ.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F64x8 exists.
        Self(unsafe { _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(self.0), _mm512_set1_epi64(i64::MIN))) })
    }
}

impl From<F64x8> for [f64; 8] {
    #[inline(always)]
    fn from(v: F64x8) -> Self {
        let mut lanes = [0.0; 8];
        // SAFETY: `lanes` holds exactly 8 elements.
        unsafe { v.store(lanes.as_mut_ptr()) };
        lanes
    }
}

impl fmt::Debug for F64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F64x8").field(&<[f64; 8]>::from(*self)).finish()
    }
}

/// Test-only: a safe constructor, so it checks for AVX-512F itself.
#[cfg(test)]
impl From<[f64; 8]> for F64x8 {
    fn from(lanes: [f64; 8]) -> Self {
        assert!(crate::detect::has_avx512f(), "AVX-512F not available");
        // SAFETY: AVX-512F was just checked, and `lanes` holds exactly 8
        // elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::has_avx512f;
    use std::format;

    const A: [f64; 8] = [1.0, -2.5, f64::NAN, -0.0, 1e-310, 100.0, f64::INFINITY, 8.0];
    const B: [f64; 8] = [2.0, 4.0, -1.5, 0.0, -0.25, 0.1, 9.0, -8.0];
    const SENTINEL: f64 = 42.0;

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_lanes_eq(got: [f64; 8], want: [f64; 8]) {
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
        let (s, z) = unsafe { (F64x8::splat(1.5), F64x8::zero()) };
        assert_lanes_eq(s.into(), [1.5; 8]);
        assert_lanes_eq(z.into(), [0.0; 8]);
    }

    #[test]
    fn raw_round_trip() {
        if !has_avx512f() {
            return;
        }
        // SAFETY: AVX-512F checked above.
        let v = unsafe { F64x8::from_raw(F64x8::from(A).raw()) };
        assert_lanes_eq(v.into(), A);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        if !has_avx512f() {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f64; 8 + 1]);
        let src = Aligned(core::array::from_fn(|i| i as f64));
        let mut dst = Aligned([SENTINEL; 8 + 1]);

        // SAFETY: AVX-512F checked above; offset 1 leaves 8
        // readable/writable elements in each buffer.
        unsafe { F64x8::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        if !has_avx512f() {
            return;
        }
        for n in 0..8 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: AVX-512F checked above; the slice is valid for `n`
            // reads and `n < 8`.
            let v = unsafe { F64x8::load_partial(A[..n].as_ptr(), n) };
            let mut want = [0.0; 8];
            want[..n].copy_from_slice(&A[..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        if !has_avx512f() {
            return;
        }
        for n in 0..8 {
            let mut dst = [SENTINEL; 8];
            // SAFETY: `dst` is valid for 8 writes and `n < 8`.
            unsafe { F64x8::from(A).store_partial(dst.as_mut_ptr(), n) };
            let mut want = [SENTINEL; 8];
            want[..n].copy_from_slice(&A[..n]);
            assert_lanes_eq(dst, want);
        }
    }

    #[test]
    fn binary_ops_match_scalar() {
        if !has_avx512f() {
            return;
        }
        let (a, b) = (F64x8::from(A), F64x8::from(B));
        let scalar = |f: fn(f64, f64) -> f64| core::array::from_fn(|i| f(A[i], B[i]));

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
        let (a, b) = (F64x8::from(A), F64x8::from(B));
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
        let x: [f64; 8] = [0.0, -0.0, 1.0, -1.0, f64::INFINITY, -f64::INFINITY, f64::NAN, -f64::NAN];
        let got: [f64; 8] = (-F64x8::from(x)).into();
        for (g, x) in got.iter().zip(x) {
            assert_eq!(g.to_bits(), x.to_bits() ^ (1 << 63));
        }
    }

    #[test]
    fn debug_prints_lanes() {
        if !has_avx512f() {
            return;
        }
        assert_eq!(
            format!("{:?}", F64x8::from([1.0; 8])),
            "F64x8([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])"
        );
    }
}
