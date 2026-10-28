//! Four `f64` lanes in an AVX2 `__m256d` register.
//!
//! Intrinsics are `unsafe` to call because they require AVX2. Constructors are
//! `unsafe` and require AVX2 + FMA, so an existing `F64x4` proves the CPU
//! supports them; that is what the `// SAFETY: an F64x4 exists` comments below
//! refer to.
//!
//! Everything is `#[inline(always)]` so it compiles to AVX2 instructions when
//! inlined into a `#[target_feature(enable = "avx2,fma")]` function.

use core::arch::x86_64::{
    __m256d, __m256i, _mm256_add_pd, _mm256_div_pd, _mm256_loadu_pd, _mm256_loadu_si256,
    _mm256_maskload_pd, _mm256_maskstore_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd,
    _mm256_storeu_pd, _mm256_sub_pd, _mm256_xor_pd,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::simd::SimdVector;

/// Four `f64` lanes.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct F64x4(__m256d);

impl F64x4 {
    /// Wraps a raw register.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA.
    #[inline(always)]
    pub(crate) unsafe fn from_raw(v: __m256d) -> Self {
        Self(v)
    }

    /// The raw register.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m256d {
        self.0
    }
}

/// Sliding window: the 4 entries starting at `4 - n` enable the first `n` lanes.
/// The masked `_pd` loads and stores read the sign bit of each 64-bit lane.
static MASK: [i64; 8] = [-1, -1, -1, -1, 0, 0, 0, 0];

/// Lane mask enabling the first `n` lanes (all 4 if `n >= 4`).
///
/// # Safety
///
/// The CPU must support AVX2.
#[inline(always)]
unsafe fn mask(n: usize) -> __m256i {
    let start = 4 - n.min(4);
    // SAFETY: the caller guarantees AVX2, and `start <= 4` keeps the 4-lane
    // read within `MASK`'s 8 entries.
    unsafe { _mm256_loadu_si256(MASK.as_ptr().add(start).cast()) }
}

impl SimdVector for F64x4 {
    type Scalar = f64;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        // SAFETY: the caller guarantees AVX2.
        Self(unsafe { _mm256_set1_pd(x) })
    }

    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees AVX2.
        Self(unsafe { _mm256_setzero_pd() })
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> Self {
        // SAFETY: the caller guarantees AVX2 and that `ptr` is valid for 4
        // reads.
        Self(unsafe { _mm256_loadu_pd(ptr) })
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f64, n: usize) -> Self {
        debug_assert!(n < Self::LANES);
        // SAFETY: the caller guarantees AVX2 and that `ptr` is valid for the
        // `n` enabled lanes; masked-off lanes are never read.
        Self(unsafe { _mm256_maskload_pd(ptr, mask(n)) })
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut f64) {
        // SAFETY: an F64x4 exists, and the caller guarantees `ptr` is valid
        // for 4 writes.
        unsafe { _mm256_storeu_pd(ptr, self.0) }
    }

    #[inline(always)]
    unsafe fn store_partial(self, ptr: *mut f64, n: usize) {
        debug_assert!(n < Self::LANES);
        // SAFETY: an F64x4 exists, and the caller guarantees `ptr` is valid
        // for the `n` enabled lanes; masked-off lanes are never written.
        unsafe { _mm256_maskstore_pd(ptr, mask(n), self.0) }
    }
}

macro_rules! impl_binary_op {
    ($Op:ident, $op:ident, $OpAssign:ident, $op_assign:ident, $intrinsic:ident) => {
        impl $Op for F64x4 {
            type Output = Self;

            #[inline(always)]
            fn $op(self, rhs: Self) -> Self {
                // SAFETY: an F64x4 exists.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }

        impl $OpAssign for F64x4 {
            #[inline(always)]
            fn $op_assign(&mut self, rhs: Self) {
                *self = $Op::$op(*self, rhs);
            }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, _mm256_add_pd);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, _mm256_sub_pd);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, _mm256_mul_pd);
impl_binary_op!(Div, div, DivAssign, div_assign, _mm256_div_pd);

impl Neg for F64x4 {
    type Output = Self;

    /// Flips the sign bit, so `-(0.0)` is `-0.0` and NaN payloads are kept.
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: an F64x4 exists.
        Self(unsafe { _mm256_xor_pd(self.0, _mm256_set1_pd(-0.0)) })
    }
}

impl From<F64x4> for [f64; 4] {
    #[inline(always)]
    fn from(v: F64x4) -> Self {
        let mut lanes = [0.0; 4];
        // SAFETY: `lanes` holds exactly 4 elements.
        unsafe { v.store(lanes.as_mut_ptr()) };
        lanes
    }
}

impl fmt::Debug for F64x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F64x4").field(&<[f64; 4]>::from(*self)).finish()
    }
}

/// Test-only: a safe constructor, so it checks for AVX2 itself.
#[cfg(test)]
impl From<[f64; 4]> for F64x4 {
    fn from(lanes: [f64; 4]) -> Self {
        assert!(crate::detect::has_avx2_fma(), "AVX2 + FMA not available");
        // SAFETY: AVX2 was just checked, and `lanes` holds exactly 4 elements.
        unsafe { Self::load(lanes.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::has_avx2_fma;
    use std::format;

    /// Two registers' worth, so every special value meets a finite one.
    const A: [[f64; 4]; 2] = [
        [1.0, -2.5, f64::NAN, -0.0],
        [1e-310, 100.0, f64::INFINITY, 8.0],
    ];
    const B: [[f64; 4]; 2] = [[2.0, 4.0, -1.5, 0.0], [-0.25, 0.1, 9.0, -8.0]];
    const SENTINEL: f64 = 42.0;

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_lanes_eq(got: [f64; 4], want: [f64; 4]) {
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
        let (s, z) = unsafe { (F64x4::splat(1.5), F64x4::zero()) };
        assert_lanes_eq(s.into(), [1.5; 4]);
        assert_lanes_eq(z.into(), [0.0; 4]);
    }

    #[test]
    fn raw_round_trip() {
        if !has_avx2_fma() {
            return;
        }
        // SAFETY: AVX2 + FMA checked above.
        let v = unsafe { F64x4::from_raw(F64x4::from(A[0]).raw()) };
        assert_lanes_eq(v.into(), A[0]);
    }

    #[test]
    fn load_store_round_trip_unaligned() {
        if !has_avx2_fma() {
            return;
        }
        #[repr(align(32))]
        struct Aligned([f64; 5]);
        let src = Aligned([0.0, 1.0, 2.0, 3.0, 4.0]);
        let mut dst = Aligned([SENTINEL; 5]);

        // SAFETY: AVX2 + FMA checked above; offset 1 leaves 4 readable/writable
        // elements in each buffer.
        unsafe { F64x4::load(src.0.as_ptr().add(1)).store(dst.0.as_mut_ptr().add(1)) };

        assert_eq!(dst.0[0], SENTINEL);
        assert_lanes_eq(dst.0[1..].try_into().unwrap(), src.0[1..].try_into().unwrap());
    }

    #[test]
    fn load_partial_zeroes_inactive_lanes() {
        if !has_avx2_fma() {
            return;
        }
        for n in 0..4 {
            // Slicing to `n` elements makes reads past them out of bounds.
            // SAFETY: AVX2 + FMA checked above; the slice is valid for `n`
            // reads and `n < 4`.
            let v = unsafe { F64x4::load_partial(A[0][..n].as_ptr(), n) };
            let mut want = [0.0; 4];
            want[..n].copy_from_slice(&A[0][..n]);
            assert_lanes_eq(v.into(), want);
        }
    }

    #[test]
    fn store_partial_writes_only_active_lanes() {
        if !has_avx2_fma() {
            return;
        }
        for n in 0..4 {
            let mut dst = [SENTINEL; 4];
            // SAFETY: `dst` is valid for 4 writes and `n < 4`.
            unsafe { F64x4::from(A[0]).store_partial(dst.as_mut_ptr(), n) };
            let mut want = [SENTINEL; 4];
            want[..n].copy_from_slice(&A[0][..n]);
            assert_lanes_eq(dst, want);
        }
    }

    #[test]
    fn binary_ops_match_scalar() {
        if !has_avx2_fma() {
            return;
        }
        for (a, b) in A.into_iter().zip(B) {
            let (va, vb) = (F64x4::from(a), F64x4::from(b));
            let scalar = |f: fn(f64, f64) -> f64| core::array::from_fn(|i| f(a[i], b[i]));

            assert_lanes_eq((va + vb).into(), scalar(|x, y| x + y));
            assert_lanes_eq((va - vb).into(), scalar(|x, y| x - y));
            assert_lanes_eq((va * vb).into(), scalar(|x, y| x * y));
            assert_lanes_eq((va / vb).into(), scalar(|x, y| x / y));
        }
    }

    #[test]
    fn assign_ops_match_binary_ops() {
        if !has_avx2_fma() {
            return;
        }
        let (a, b) = (F64x4::from(A[0]), F64x4::from(B[0]));
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
        let xs = [
            [0.0, -0.0, 1.0, -1.0],
            [f64::INFINITY, -f64::INFINITY, f64::NAN, -f64::NAN],
        ];
        for x in xs {
            let got: [f64; 4] = (-F64x4::from(x)).into();
            for (g, x) in got.iter().zip(x) {
                assert_eq!(g.to_bits(), x.to_bits() ^ (1 << 63));
            }
        }
    }

    #[test]
    fn debug_prints_lanes() {
        if !has_avx2_fma() {
            return;
        }
        assert_eq!(
            format!("{:?}", F64x4::from([1.0; 4])),
            "F64x4([1.0, 1.0, 1.0, 1.0])"
        );
    }
}
