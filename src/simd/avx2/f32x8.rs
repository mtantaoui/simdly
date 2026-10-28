//! Eight `f32` lanes in an AVX2 `__m256` register.
//!
//! Intrinsics are `unsafe` to call because they require AVX2. Constructors are
//! `unsafe` and require AVX2 + FMA, so an existing `F32x8` proves the CPU
//! supports them; that is what the `// SAFETY: an F32x8 exists` comments below
//! refer to.
//!
//! Everything is `#[inline(always)]` so it compiles to AVX2 instructions when
//! inlined into a `#[target_feature(enable = "avx2,fma")]` function.

use core::arch::x86_64::{
    __m256, __m256i, _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_maskload_ps, _mm256_maskstore_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps, _mm256_sub_ps, _mm256_xor_ps,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::simd::SimdVector;

/// Eight `f32` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F32x8(__m256);

impl F32x8 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: __m256) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m256 {
        self.0
    }
}

/// Sliding window: the 8 entries starting at `8 - n` enable the first `n` lanes.
static MASK: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

/// Lane mask enabling the first `n` lanes (all 8 if `n >= 8`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[inline(always)]
unsafe fn mask(n: usize) -> __m256i {
    let start = 8 - n.min(8);
    // SAFETY: the caller guarantees AVX2, and `start <= 8` keeps the 8-lane
    // read within `MASK`'s 16 entries.
    unsafe { _mm256_loadu_si256(MASK.as_ptr().add(start).cast()) }
}

impl SimdVector for F32x8 {
    type Scalar = f32;
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        // SAFETY: the caller guarantees AVX2.
        Self(unsafe { _mm256_set1_ps(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees AVX2.
        Self(unsafe { _mm256_setzero_ps() })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self {
        // SAFETY: the caller guarantees AVX2 and that `ptr` is valid for 8
        // reads.
        Self(unsafe { _mm256_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        // SAFETY: the caller guarantees AVX2 and that `ptr` is valid for the
        // `n` enabled lanes; masked-off lanes are never read.
        Self(unsafe { _mm256_maskload_ps(ptr, mask(n)) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f32) {
        // SAFETY: an F32x8 exists, and the caller guarantees `ptr` is valid
        // for 8 writes.
        unsafe { _mm256_storeu_ps(ptr, self.0) }
    }

    #[inline(always)]
    unsafe fn store_partial(self, ptr: *mut f32, n: usize) {
        debug_assert!(n < Self::LANES);
        // SAFETY: an F32x8 exists, and the caller guarantees `ptr` is valid
        // for the `n` enabled lanes; masked-off lanes are never written.
        unsafe { _mm256_maskstore_ps(ptr, mask(n), self.0) }
    }
}

macro_rules! impl_binary_op {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $intrinsic:ident) => {
        impl $Op for F32x8 {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self {
                // SAFETY: an F32x8 exists.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }

        impl $OpAssign for F32x8 {
            #[inline(always)]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, _mm256_add_ps);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, _mm256_sub_ps);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, _mm256_mul_ps);
impl_binary_op!(Div, div, DivAssign, div_assign, _mm256_div_ps);

impl Neg for F32x8 {
    type Output = Self;

    /// Flips the sign bit, so `-(0.0)` is `-0.0` and NaN payloads are kept.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F32x8 exists.
        Self(unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) })
    }
}

impl From<F32x8> for [f32; 8] {
    #[inline(always)]
    fn from(v: F32x8) -> Self {
        let mut lanes = [0.0; 8];
        // SAFETY: `lanes` holds exactly 8 elements.
        unsafe { v.store(lanes.as_mut_ptr()) };
        lanes
    }
}

impl fmt::Debug for F32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F32x8").field(&<[f32; 8]>::from(*self)).finish()
    }
}

/// Test-only: a safe constructor, so it checks for AVX2 itself.
#[cfg(test)]
impl From<[f32; 8]> for F32x8 {
    fn from(lanes: [f32; 8]) -> Self {
        assert!(crate::detect::has_avx2_fma(), "AVX2 + FMA not available");
        // SAFETY: AVX2 was just checked, and `lanes` holds exactly 8 elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::has_avx2_fma;
    use std::format;

    const A: [f32; 8] = [1.0, -2.5, f32::NAN, -0.0, 0.5, 100.0, f32::INFINITY, 8.0];
    const B: [f32; 8] = [2.0, 4.0, -1.5, 0.0, -0.25, 0.1, 9.0, -8.0];
    const SENTINEL: f32 = 42.0;

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_lanes_eq(got: [f32; 8], want: [f32; 8]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
            assert!(same, "lane {i}: got {g:?}, want {w:?}");
        }
    }

    #[test]
    fn splat_and_zero() {
        if !has_avx2_fma() {
            return;
        }
        // SAFETY: AVX2 + FMA checked above.
        let (s, z) = unsafe { (F32x8::splat(1.5), F32x8::zero()) };
        assert_lanes_eq(s.into(), [1.5; 8]);
        assert_lanes_eq(z.into(), [0.0; 8]);
    }

    #[test]
    fn raw_round_trip() {
        if !has_avx2_fma() {
            return;
        }
        // SAFETY: AVX2 + FMA checked above.
        let v = unsafe { F32x8::from_raw(F32x8::from(A).raw()) };
        assert_lanes_eq(v.into(), A);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        if !has_avx2_fma() {
            return;
        }
        #[repr(align(32))]
        struct Aligned([f32; 9]);
        let src = Aligned([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut dst = Aligned([SENTINEL; 9]);

        // SAFETY: AVX2 + FMA checked above; offset 1 leaves 8 readable/writable
        // elements in each buffer.
        unsafe { F32x8::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        if !has_avx2_fma() {
            return;
        }
        for n in 0..8 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: AVX2 + FMA checked above; the slice is valid for `n`
            // reads and `n < 8`.
            let v = unsafe { F32x8::load_partial(A[..n].as_ptr(), n) };
            let mut want = [0.0; 8];
            want[..n].copy_from_slice(&A[..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        if !has_avx2_fma() {
            return;
        }
        for n in 0..8 {
            let mut dst = [SENTINEL; 8];
            // SAFETY: `dst` is valid for 8 writes and `n < 8`.
            unsafe { F32x8::from(A).store_partial(dst.as_mut_ptr(), n) };
            let mut want = [SENTINEL; 8];
            want[..n].copy_from_slice(&A[..n]);
            assert_lanes_eq(dst, want);
        }
    }

    #[test]
    fn binary_ops_match_scalar() {
        if !has_avx2_fma() {
            return;
        }
        let (a, b) = (F32x8::from(A), F32x8::from(B));
        let scalar = |f: fn(f32, f32) -> f32| core::array::from_fn(|i| f(A[i], B[i]));

        assert_lanes_eq((a + b).into(), scalar(|x, y| x + y));
        assert_lanes_eq((a - b).into(), scalar(|x, y| x - y));
        assert_lanes_eq((a * b).into(), scalar(|x, y| x * y));
        assert_lanes_eq((a / b).into(), scalar(|x, y| x / y));
    }

    #[test]
    fn assign_ops_match_binary_ops() {
        if !has_avx2_fma() {
            return;
        }
        let (a, b) = (F32x8::from(A), F32x8::from(B));
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
        if !has_avx2_fma() {
            return;
        }
        let x = [0.0, -0.0, 1.0, -1.0, f32::INFINITY, -f32::INFINITY, f32::NAN, -f32::NAN];
        let got: [f32; 8] = (-F32x8::from(x)).into();
        for (g, x) in got.iter().zip(x) {
            assert_eq!(g.to_bits(), x.to_bits() ^ 0x8000_0000);
        }
    }

    #[test]
    fn debug_prints_lanes() {
        if !has_avx2_fma() {
            return;
        }
        assert_eq!(
            format!("{:?}", F32x8::from([1.0; 8])),
            "F32x8([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])"
        );
    }
}
