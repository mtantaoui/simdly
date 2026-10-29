//! `asin(x)`, ≤ 1 ULP on every backend, written once over [`Float`]. It
//! shares musl's `r(z)` and π/2 with [`acos`](super::acos) (see there and the
//! book for the maths):
//!
//! | Range       | Formula                                                     |
//! |-------------|-------------------------------------------------------------|
//! | `|x| < 0.5` | `x + x·r(x²)`                                               |
//! | `|x| ≥ 0.5` | `±((π/2 − 2·df) − (2·(s·r(z) + c) − PIO2_LO))`, on `|x|`   |
//!
//! with `z = (1 − |x|)/2`, `s = √z = df + c` split as for `acos`.
//!
//! For `|x| ≥ 0.5`, musl's `asinf.c` computes in `f64`, and `asin.c` switches
//! formulas at `|x| = 0.975`. This single formula (as in simdmath) is more
//! accurate than either port in both types, so it needs no extra select:
//! measured at most 0.76 ULP (`f32`, every input) and 0.68 ULP (`f64`, against
//! a 200-bit reference), fused or not.
//!
//! `asin(±1)` would divide 0 by 0 in `c`, so it is set to `±π/2`. Outside
//! `[−1, 1]`, `√` of a negative gives NaN.

use super::acos::Kernel;
use super::float::{Element, Float};

/// A lane type with an arc sine.
pub(crate) trait Asin {
    fn asin(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Asin for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn asin(self) -> Self {
        let x = self;
        let [zero, half, one, two] = [0.0, 0.5, 1.0, 2.0].map(V::Elem::from_f32);
        let (zero, half, one, two) = (
            x.constant(zero),
            x.constant(half),
            x.constant(one),
            x.constant(two),
        );
        let [pio2_hi, pio2_lo] = V::Elem::PIO2;
        let (pio2_hi, pio2_lo) = (x.constant(pio2_hi), x.constant(pio2_lo));

        let a = x.abs();
        let small = a.lt(half);
        let z = V::select(small, x * x, (one - a) * half);
        let (rz, s) = (V::Elem::r(z), z.sqrt());

        // |x| < 0.5: x + x·r(x²).
        let small_x = x.mul_add(rz, x);
        // |x| ≥ 0.5: π/2 − 2·(df + c + s·r(z)), on |x|.
        let df = s.and_bits(V::Elem::HIGH_BITS);
        let c = (z - df * df) / (s + df);
        let large = (pio2_hi - two * df) - (two * s.mul_add(rz, c) - pio2_lo);
        let large = V::select(a.eq(one), pio2_hi + pio2_lo, large);
        let large = V::select(x.lt(zero), -large, large);

        V::select(small, small_x, large)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::AsinOp;
    use crate::testing::{
        TestFloat, assert_exact, assert_ulps, assert_ulps_every_f32, unit_interval,
    };
    use std::vec::Vec;

    #[test]
    fn asin_within_1_ulp() {
        assert_ulps::<f32, AsinOp>("asin", &unit_interval(200_000), f64::asin, 1.0);
        assert_ulps::<f64, AsinOp>("asin", &unit_interval(200_000), f64::asin, 1.0);
    }

    #[test]
    fn asin_special_values() {
        use core::f64::consts::FRAC_PI_2;
        let nan = f64::NAN;
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (1.0, FRAC_PI_2),
            (-1.0, -FRAC_PI_2),
            (1.0000001, nan),
            (-1.0000001, nan),
            (f64::INFINITY, nan),
            (f64::NEG_INFINITY, nan),
            (nan, nan),
        ];
        assert_exact::<f64, AsinOp>("asin", &cases);
        let cases32: Vec<(f32, f32)> = cases.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
        assert_eq!(cases32[2].1, core::f32::consts::FRAC_PI_2);
        assert_exact::<f32, AsinOp>("asin", &cases32);
    }

    /// `asin(−x) = −asin(x)`, bit for bit, on every backend.
    #[test]
    fn asin_is_odd() {
        fn check<T: TestFloat + crate::math::MathElement>() {
            let x: Vec<T> = unit_interval(20_000);
            let neg: Vec<T> = x
                .iter()
                .map(|v| T::from_bits64(v.bits() ^ T::SIGN))
                .collect();
            for ((backend, pos), (_, negs)) in T::every_backend::<AsinOp>(&x)
                .into_iter()
                .zip(T::every_backend::<AsinOp>(&neg))
            {
                let flipped: Vec<T> = pos
                    .iter()
                    .map(|v| T::from_bits64(v.bits() ^ T::SIGN))
                    .collect();
                crate::testing::assert_same(
                    &negs,
                    &flipped,
                    &std::format!("asin odd {} {backend}", T::NAME),
                );
            }
        }
        check::<f32>();
        check::<f64>();
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn asin_f32_exhaustive() {
        assert_ulps_every_f32::<AsinOp>("asin", f64::asin, 1.0);
    }
}
