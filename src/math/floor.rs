//! `⌊x⌋`, exact on every backend: `⌊−0.0⌋ = −0.0`, `⌊−0.3⌋ = −1`, and `±∞`
//! and NaN are returned as they are.
//!
//! Every ISA rounds down in one instruction (`vroundps` / `vrndscaleps`
//! with the round-down mode, NEON `frintm`). `core` has no `floor` without
//! `std`, and `std`'s is a library call on baseline `x86_64`, so the scalar
//! backend rounds with float arithmetic: below `2^FRAC`, adding and
//! subtracting `2^FRAC` rounds `|x|` to an integer, which is then corrected
//! to round down.

/// A lane type with a floor.
pub(crate) trait Floor: Copy {
    fn floor(self) -> Self;
}

macro_rules! floor_float {
    ($($T:ident: $big:expr),*) => {
        $(
            impl Floor for $T {
                #[inline(always)]
                fn floor(self) -> Self {
                    // From `2^FRAC` on, every float is an integer; so are
                    // ±∞, and NaN goes through unchanged.
                    if self.abs() >= $big || self.is_nan() {
                        return self;
                    }
                    // `|x| + 2^FRAC` lands where the ULP is 1: round to an
                    // integer, then restore the sign (`−0.3` gives `−0`).
                    let r = (self.abs() + $big) - $big;
                    let r = if self.is_sign_negative() { -r } else { r };
                    // Rounding may have gone up: step down once.
                    if r > self { r - 1.0 } else { r }
                }
            }
        )*
    };
}

floor_float!(f32: 8_388_608.0, f64: 4_503_599_627_370_496.0); // 2^23, 2^52

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEG_INF, _mm256_floor_pd, _mm256_floor_ps,
            _mm512_roundscale_pd, _mm512_roundscale_ps,
        };

        use super::Floor;
        use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        /// Round down, no precision exception.
        const DOWN: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;

        impl Floor for F32x8 {
            #[inline(always)]
            fn floor(self) -> Self {
                // SAFETY: an F32x8 exists, so AVX2 + FMA are available.
                unsafe { F32x8::from_raw(_mm256_floor_ps(self.raw())) }
            }
        }

        impl Floor for F64x4 {
            #[inline(always)]
            fn floor(self) -> Self {
                // SAFETY: an F64x4 exists, so AVX2 + FMA are available.
                unsafe { F64x4::from_raw(_mm256_floor_pd(self.raw())) }
            }
        }

        impl Floor for F32x16 {
            #[inline(always)]
            fn floor(self) -> Self {
                // SAFETY: an F32x16 exists, so AVX-512F is available.
                unsafe { F32x16::from_raw(_mm512_roundscale_ps::<DOWN>(self.raw())) }
            }
        }

        impl Floor for F64x8 {
            #[inline(always)]
            fn floor(self) -> Self {
                // SAFETY: an F64x8 exists, so AVX-512F is available.
                unsafe { F64x8::from_raw(_mm512_roundscale_pd::<DOWN>(self.raw())) }
            }
        }
    }
}

cfg_neon! {
    mod neon {
        use core::arch::aarch64::{vrndmq_f32, vrndmq_f64};

        use super::Floor;
        use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        impl Floor for F32x4 {
            #[inline(always)]
            fn floor(self) -> Self {
                // SAFETY: NEON is enabled at compile time.
                unsafe { F32x4::from_raw(vrndmq_f32(self.raw())) }
            }
        }

        impl Floor for F64x2 {
            #[inline(always)]
            fn floor(self) -> Self {
                // SAFETY: NEON is enabled at compile time.
                unsafe { F64x2::from_raw(vrndmq_f64(self.raw())) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{FloorOp, MathElement};
    use crate::testing::{TestFloat, assert_exact, assert_same, samples};
    use std::vec::Vec;

    /// Every binade, values next to integers and halves (where rounding to
    /// nearest and down differ), and the threshold `2^FRAC`.
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
        let want: Vec<T> = x.iter().map(|v| T::from_f64(v.to_f64().floor())).collect();
        for (backend, got) in T::every_backend::<FloorOp>(&x) {
            assert_same(&got, &want, &std::format!("floor {} {backend}", T::NAME));
        }
    }

    /// Bit for bit equal to `std`'s, signed zeros included.
    #[test]
    fn floor_matches_std() {
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn floor_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (0.3, 0.0),
            (-0.3, -1.0),
            (-1e-40, -1.0), // subnormal in `f32`
            (2.5, 2.0),
            (-2.5, -3.0),
            (-1.0, -1.0),
            (inf, inf),
            (-inf, -inf),
            (nan, nan),
        ];
        assert_exact::<f64, FloorOp>("floor", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, FloorOp>("floor", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn floor_f32_exhaustive() {
        const CHUNK: u64 = 1 << 22;
        for start in (0..1_u64 << 32).step_by(CHUNK as usize) {
            let x: Vec<f32> = (start..start + CHUNK)
                .map(|b| f32::from_bits(b as u32))
                .collect();
            let want: Vec<f32> = x.iter().map(|v| v.floor()).collect();
            for (backend, got) in f32::every_backend::<FloorOp>(&x) {
                assert_same(&got, &want, &std::format!("floor f32 {backend}"));
            }
        }
    }
}
