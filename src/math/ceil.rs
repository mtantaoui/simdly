//! `⌈x⌉`, exact on every backend: `⌈0.3⌉ = 1`, `⌈−0.3⌉ = −0.0`, and `±∞`
//! and NaN are returned as they are.
//!
//! Every ISA rounds up in one instruction (`vroundps` / `vrndscaleps` with
//! the round-up mode, NEON `frintp`). The scalar backend computes
//! `−⌊−x⌋` with [`Floor`]'s float arithmetic, which keeps the sign of zero
//! (`⌈−0.7⌉ = −⌊0.7⌋ = −0.0`).

use super::Floor;

/// A lane type with a ceiling.
pub(crate) trait Ceil: Copy {
    fn ceil(self) -> Self;
}

macro_rules! ceil_float {
    ($($T:ident),*) => {
        $(
            impl Ceil for $T {
                #[inline(always)]
                fn ceil(self) -> Self {
                    -Floor::floor(-self)
                }
            }
        )*
    };
}

ceil_float!(f32, f64);

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            _MM_FROUND_NO_EXC, _MM_FROUND_TO_POS_INF, _mm256_ceil_pd, _mm256_ceil_ps,
            _mm512_roundscale_pd, _mm512_roundscale_ps,
        };

        use super::Ceil;
        use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        /// Round up, no precision exception.
        const UP: i32 = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;

        impl Ceil for F32x8 {
            #[inline(always)]
            fn ceil(self) -> Self {
                // SAFETY: an F32x8 exists, so AVX2 + FMA are available.
                unsafe { F32x8::from_raw(_mm256_ceil_ps(self.raw())) }
            }
        }

        impl Ceil for F64x4 {
            #[inline(always)]
            fn ceil(self) -> Self {
                // SAFETY: an F64x4 exists, so AVX2 + FMA are available.
                unsafe { F64x4::from_raw(_mm256_ceil_pd(self.raw())) }
            }
        }

        impl Ceil for F32x16 {
            #[inline(always)]
            fn ceil(self) -> Self {
                // SAFETY: an F32x16 exists, so AVX-512F is available.
                unsafe { F32x16::from_raw(_mm512_roundscale_ps::<UP>(self.raw())) }
            }
        }

        impl Ceil for F64x8 {
            #[inline(always)]
            fn ceil(self) -> Self {
                // SAFETY: an F64x8 exists, so AVX-512F is available.
                unsafe { F64x8::from_raw(_mm512_roundscale_pd::<UP>(self.raw())) }
            }
        }
    }
}

cfg_neon! {
    mod neon {
        use core::arch::aarch64::{vrndpq_f32, vrndpq_f64};

        use super::Ceil;
        use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        impl Ceil for F32x4 {
            #[inline(always)]
            fn ceil(self) -> Self {
                // SAFETY: NEON is enabled at compile time.
                unsafe { F32x4::from_raw(vrndpq_f32(self.raw())) }
            }
        }

        impl Ceil for F64x2 {
            #[inline(always)]
            fn ceil(self) -> Self {
                // SAFETY: NEON is enabled at compile time.
                unsafe { F64x2::from_raw(vrndpq_f64(self.raw())) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{CeilOp, MathElement};
    use crate::testing::{TestFloat, assert_exact, assert_same, samples};
    use std::vec::Vec;

    /// Every binade, values next to integers and halves (where rounding to
    /// nearest and up differ), and the threshold `2^FRAC`.
    fn inputs<T: TestFloat>() -> Vec<T> {
        let (max, big) = if T::NAME == "f32" {
            (f32::MAX.into(), 8_388_608.0)
        } else {
            (f64::MAX, 4_503_599_627_370_496.0)
        };
        let mut around = std::vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, big / 2.0, big, 2.0 * big];
        around.extend((1..40).map(|i| i as f64 + 0.5));
        samples(100_000, max, &around)
    }

    fn check<T: TestFloat + MathElement>() {
        let x: Vec<T> = inputs();
        let want: Vec<T> = x.iter().map(|v| T::from_f64(v.to_f64().ceil())).collect();
        for (backend, got) in T::every_backend::<CeilOp>(&x) {
            assert_same(&got, &want, &std::format!("ceil {} {backend}", T::NAME));
        }
    }

    /// Bit for bit equal to `std`'s, signed zeros included.
    #[test]
    fn ceil_matches_std() {
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn ceil_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (0.3, 1.0),
            (-0.3, -0.0),
            (-0.7, -0.0),
            (1e-40, 1.0), // subnormal in `f32`
            (2.5, 3.0),
            (-2.5, -2.0),
            (1.0, 1.0),
            (inf, inf),
            (-inf, -inf),
            (nan, nan),
        ];
        assert_exact::<f64, CeilOp>("ceil", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, CeilOp>("ceil", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn ceil_f32_exhaustive() {
        const CHUNK: u64 = 1 << 22;
        for start in (0..1_u64 << 32).step_by(CHUNK as usize) {
            let x: Vec<f32> = (start..start + CHUNK)
                .map(|b| f32::from_bits(b as u32))
                .collect();
            let want: Vec<f32> = x.iter().map(|v| v.ceil()).collect();
            for (backend, got) in f32::every_backend::<CeilOp>(&x) {
                assert_same(&got, &want, &std::format!("ceil f32 {backend}"));
            }
        }
    }
}
