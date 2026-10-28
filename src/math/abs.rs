//! `|x|`: clears the sign bit, so it is exact, `-0.0` becomes `0.0`, and NaN
//! keeps its payload with the sign cleared.
//!
//! AVX2 has no float `abs` instruction; `andnot` with `-0.0` (only the sign
//! bit set) does it in one `vandnps` / `vandnpd`. AVX-512F's `_mm512_abs_*`
//! is the same, as an integer `vpandd` / `vpandq`.

/// A lane type with an absolute value.
pub(crate) trait Abs: Copy {
    fn abs(self) -> Self;
}

impl Abs for f32 {
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl Abs for f64 {
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
}

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            _mm256_andnot_pd, _mm256_andnot_ps, _mm256_set1_pd, _mm256_set1_ps, _mm512_abs_pd,
            _mm512_abs_ps,
        };

        use super::Abs;
        use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        impl Abs for F32x8 {
            #[inline(always)]
            fn abs(self) -> Self {
                // SAFETY: an F32x8 exists, so AVX2 + FMA are available.
                unsafe { F32x8::from_raw(_mm256_andnot_ps(_mm256_set1_ps(-0.0), self.raw())) }
            }
        }

        impl Abs for F64x4 {
            #[inline(always)]
            fn abs(self) -> Self {
                // SAFETY: an F64x4 exists, so AVX2 + FMA are available.
                unsafe { F64x4::from_raw(_mm256_andnot_pd(_mm256_set1_pd(-0.0), self.raw())) }
            }
        }

        impl Abs for F32x16 {
            #[inline(always)]
            fn abs(self) -> Self {
                // SAFETY: an F32x16 exists, so AVX-512F is available.
                unsafe { F32x16::from_raw(_mm512_abs_ps(self.raw())) }
            }
        }

        impl Abs for F64x8 {
            #[inline(always)]
            fn abs(self) -> Self {
                // SAFETY: an F64x8 exists, so AVX-512F is available.
                unsafe { F64x8::from_raw(_mm512_abs_pd(self.raw())) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{AbsOp, MathElement};
    use crate::testing::TestFloat;
    use std::vec::Vec;

    /// Every special value, including NaN with either sign, among finite
    /// values of both signs.
    fn inputs<T: TestFloat + core::ops::Neg<Output = T>>(len: usize) -> Vec<T> {
        (0..len)
            .map(|i| match i % 5 {
                1 => T::SPECIAL[i % 8],
                3 => -T::SPECIAL[i % 8],
                _ => T::from_f64(i as f64 * 0.75 - 9.0),
            })
            .collect()
    }

    /// Bit-exact, NaN payload and sign included.
    fn assert_bits<T: TestFloat>(got: &[T], x: &[T], what: &str) {
        assert_eq!(got.len(), x.len(), "{what}");
        for (i, (&g, &x)) in got.iter().zip(x).enumerate() {
            let want = x.bits() & !T::SIGN;
            assert_eq!(g.bits(), want, "{what}, index {i}: |{x:?}| gave {g:?}");
        }
    }

    fn check<T: TestFloat + MathElement + core::ops::Neg<Output = T>>() {
        for len in 0..=40 {
            let x = inputs::<T>(len);
            let what = std::format!("abs {}, len {len}", T::NAME);

            let mut out = Vec::with_capacity(len);
            T::unary_uninit::<AbsOp>(&x, &mut out.spare_capacity_mut()[..len]);
            // SAFETY: `unary_uninit` initialised the first `len` elements.
            unsafe { out.set_len(len) };
            assert_bits(&out, &x, &what);

            let mut y = x.clone();
            T::unary_in_place::<AbsOp>(&mut y);
            assert_bits(&y, &x, &std::format!("{what}, in place"));

            for (backend, y) in T::every_backend::<AbsOp>(&x) {
                assert_bits(&y, &x, &std::format!("{what}, {backend}"));
            }
        }
    }

    #[test]
    fn abs_clears_only_the_sign_bit() {
        check::<f32>();
        check::<f64>();
    }
}
