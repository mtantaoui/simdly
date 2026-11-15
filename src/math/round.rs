//! Rounding to an integer, exact on every backend, bit for bit `std`'s:
//! `trunc` (toward zero), `round_ties_even` (to nearest, halves to even) and
//! `round` (to nearest, halves away from zero). Signed zeros are kept
//! (`trunc(−0.3) = −0.0`); `±∞` and NaN are returned as they are.
//!
//! `trunc` and `round_ties_even` are one instruction on every ISA
//! (`vroundps` / `vrndscaleps` with a rounding mode, NEON `frintz` and
//! `frintn`). The scalar backend rounds with float arithmetic, as
//! [`floor`](super::floor): below `2^FRAC`, adding and subtracting `2^FRAC`
//! rounds `|x|` to the nearest integer, halves to even. `round` has no
//! x86 instruction: `t = trunc x`, then `t ± 1` where the dropped fraction
//! `x − t` (exact) is at least ½.

use super::atan::k;
use super::float::Float;

/// A lane type that rounds toward zero.
pub(crate) trait Trunc: Copy {
    fn trunc(self) -> Self;
}

/// A lane type that rounds to nearest, halves to even.
pub(crate) trait RoundTiesEven: Copy {
    fn round_ties_even(self) -> Self;
}

/// A lane type that rounds to nearest, halves away from zero.
pub(crate) trait Round {
    fn round(self) -> Self;
}

impl<V: Float + Trunc> Round for V {
    #[inline(always)]
    fn round(self) -> Self {
        let [half, one] = [k(self, 0.5), k(self, 1.0)];
        let t = self.trunc();
        let away = V::select(self.is_sign_negative(), -one, one);
        // `±∞ − ±∞` is NaN, which compares false: ±∞ and NaN stay as `t`.
        V::select((self - t).abs().ge(half), t + away, t)
    }
}

macro_rules! scalar {
    ($($T:ident: $big:expr),*) => {
        $(
            impl RoundTiesEven for $T {
                #[inline(always)]
                fn round_ties_even(self) -> Self {
                    // From `2^FRAC` on, every float is an integer; so are
                    // ±∞, and NaN goes through unchanged.
                    if self.abs() >= $big || self.is_nan() {
                        return self;
                    }
                    // `|x| + 2^FRAC` lands where the ULP is 1: rounded to
                    // nearest, halves to even; then the sign (`−0.3` → `−0`).
                    let r = (self.abs() + $big) - $big;
                    if self.is_sign_negative() { -r } else { r }
                }
            }

            impl Trunc for $T {
                #[inline(always)]
                fn trunc(self) -> Self {
                    if self.abs() >= $big || self.is_nan() {
                        return self;
                    }
                    let a = self.abs();
                    let r = (a + $big) - $big;
                    // Rounding may have gone up: step down once.
                    let r = if r > a { r - 1.0 } else { r };
                    if self.is_sign_negative() { -r } else { r }
                }
            }
        )*
    };
}

scalar!(f32: 8_388_608.0, f64: 4_503_599_627_370_496.0); // 2^23, 2^52

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_TO_ZERO, _mm_round_pd, _mm_round_ps, _mm256_round_pd,
            _mm256_round_ps, _mm512_roundscale_pd, _mm512_roundscale_ps,
        };

        use super::{RoundTiesEven, Trunc};
        use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};
        use crate::simd::sse::{f32x4::F32x4, f64x2::F64x2};

        /// Toward zero, no precision exception.
        const ZERO: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
        /// To nearest, halves to even, no precision exception.
        const NEAREST: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

        macro_rules! x86 {
            ($($V:ident: $round:ident);*) => {
                $(
                    impl Trunc for $V {
                        #[inline(always)]
                        fn trunc(self) -> Self {
                            // SAFETY: a register of this type exists, so its
                            // ISA is available.
                            unsafe { $V::from_raw($round::<ZERO>(self.raw())) }
                        }
                    }

                    impl RoundTiesEven for $V {
                        #[inline(always)]
                        fn round_ties_even(self) -> Self {
                            // SAFETY: as for `trunc`.
                            unsafe { $V::from_raw($round::<NEAREST>(self.raw())) }
                        }
                    }
                )*
            };
        }

        x86!(
            F32x8: _mm256_round_ps;
            F64x4: _mm256_round_pd;
            F32x16: _mm512_roundscale_ps;
            F64x8: _mm512_roundscale_pd;
            F32x4: _mm_round_ps;
            F64x2: _mm_round_pd
        );
    }
}

cfg_neon! {
    mod neon {
        use core::arch::aarch64::{vrndnq_f32, vrndnq_f64, vrndq_f32, vrndq_f64};

        use super::{RoundTiesEven, Trunc};
        use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        macro_rules! neon {
            ($($V:ident: $trunc:ident, $even:ident);*) => {
                $(
                    impl Trunc for $V {
                        #[inline(always)]
                        fn trunc(self) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { $V::from_raw($trunc(self.raw())) }
                        }
                    }

                    impl RoundTiesEven for $V {
                        #[inline(always)]
                        fn round_ties_even(self) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { $V::from_raw($even(self.raw())) }
                        }
                    }
                )*
            };
        }

        neon!(F32x4: vrndq_f32, vrndnq_f32; F64x2: vrndq_f64, vrndnq_f64);
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{MathElement, RoundOp, RoundTiesEvenOp, TruncOp, UnaryOp};
    use crate::testing::{TestFloat, assert_same, samples};
    use std::vec::Vec;

    /// Every binade, values next to integers and halves (where the three
    /// roundings differ), and `2^FRAC` and its neighbours.
    fn inputs<T: TestFloat>() -> Vec<T> {
        let (max, big) = if T::NAME == "f32" {
            (f32::MAX.into(), 8_388_608.0)
        } else {
            (f64::MAX, 4_503_599_627_370_496.0)
        };
        let mut around = std::vec![0.5, 1.0, 1.5, 2.5, big / 2.0, big, 2.0 * big];
        around.extend((1..40).map(|i| f64::from(i) + 0.5));
        let mut x = samples(100_000, max, &around);
        x.extend([
            T::from_f64(f64::INFINITY),
            T::from_f64(f64::NEG_INFINITY),
            T::from_f64(f64::NAN),
        ]);
        x
    }

    /// `F` bit for bit `reference`, on every backend.
    fn check<T: TestFloat + MathElement, F: UnaryOp>(name: &str, reference: fn(f64) -> f64) {
        let x: Vec<T> = inputs();
        let want: Vec<T> = x
            .iter()
            .map(|v| T::from_f64(reference(v.to_f64())))
            .collect();
        for (backend, got) in T::every_backend::<F>(&x) {
            assert_same(&got, &want, &std::format!("{name} {} {backend}", T::NAME));
        }
    }

    /// Rounding is exact, so `f64`'s `std` rounding of a widened `f32` is
    /// `f32`'s.
    #[test]
    fn roundings_match_std() {
        check::<f32, TruncOp>("trunc", f64::trunc);
        check::<f64, TruncOp>("trunc", f64::trunc);
        check::<f32, RoundOp>("round", f64::round);
        check::<f64, RoundOp>("round", f64::round);
        check::<f32, RoundTiesEvenOp>("round_ties_even", f64::round_ties_even);
        check::<f64, RoundTiesEvenOp>("round_ties_even", f64::round_ties_even);
    }

    /// The values where they differ, and signed zeros.
    #[test]
    fn roundings_special_values() {
        let x = [
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            -0.3,
            -0.0,
            0.499_999_999_999_999_94,
            1e300,
        ];
        let cases = [
            (
                "trunc",
                [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, -0.0, -0.0, 0.0, 1e300],
            ),
            (
                "round",
                [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, -0.0, -0.0, 0.0, 1e300],
            ),
            (
                "round_ties_even",
                [0.0, -0.0, 2.0, -2.0, 2.0, -2.0, -0.0, -0.0, 0.0, 1e300],
            ),
        ];
        for (name, want) in cases {
            let got = match name {
                "trunc" => f64::every_backend::<TruncOp>(&x),
                "round" => f64::every_backend::<RoundOp>(&x),
                _ => f64::every_backend::<RoundTiesEvenOp>(&x),
            };
            for (backend, got) in got {
                assert_same(&got, &want, &std::format!("{name} {backend}"));
            }
        }
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend, three functions"]
    fn roundings_f32_exhaustive() {
        const CHUNK: u64 = 1 << 22;
        for start in (0..1_u64 << 32).step_by(CHUNK as usize) {
            let x: Vec<f32> = (start..start + CHUNK)
                .map(|b| f32::from_bits(b as u32))
                .collect();
            for name in ["trunc", "round", "round_ties_even"] {
                let (want, got): (Vec<f32>, _) = match name {
                    "trunc" => (
                        x.iter().map(|v| v.trunc()).collect(),
                        f32::every_backend::<TruncOp>(&x),
                    ),
                    "round" => (
                        x.iter().map(|v| v.round()).collect(),
                        f32::every_backend::<RoundOp>(&x),
                    ),
                    _ => (
                        x.iter().map(|v| v.round_ties_even()).collect(),
                        f32::every_backend::<RoundTiesEvenOp>(&x),
                    ),
                };
                for (backend, got) in got {
                    assert_same(&got, &want, &std::format!("{name} f32 {backend}"));
                }
            }
        }
    }
}
