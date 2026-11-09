//! `σ(x) = 1/(1 + e⁻ˣ)`, the logistic function, ≤ 1 ULP on every backend.
//!
//! With `e = e^−|x|`, which never overflows, `σ(x) = 1/(1 + e)` for `x ≥ 0`
//! and `e/(1 + e)` for `x < 0`: one division, its numerator chosen per lane.
//! The naive `1/(1 + e⁻ˣ)` would round `σ` to `0` from `x ≈ −88.7` (`f32`)
//! on, where `e⁻ˣ` overflows, though `σ ≈ eˣ` stays representable to
//! `−103.97`.
//!
//! Rounded as it stands, that formula reaches 2.4 ULPs: `exp`'s own error
//! carries into `σ`, and doubles where `σ = e/(1 + e)` falls just below
//! `e`'s power of 2. So `e = (yh + yl)·2^k` to a few bits more than the
//! element type ([`exp_parts`]), `1 + e = dh + dl` exactly, and the quotient
//! gets one correction step; the scaling by `2^k` comes last, in two
//! normal factors as in `exp`. The same code for `f32` and `f64`.
//!
//! `σ(±0) = 0.5`, `σ(+∞) = 1`, `σ(−∞) = +0`; NaN propagates.

use super::atan::ks;
use super::exp::{Kernel, exp_parts, scale};
use super::float::{Element, Float};

/// A lane type with the logistic function.
pub(crate) trait Sigmoid {
    fn sigmoid(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Sigmoid for V
where
    V::Elem: Limits,
{
    #[inline(always)]
    fn sigmoid(self) -> Self {
        let x = self;
        let [zero, half, one] = ks(x, [0.0, 0.5, 1.0].map(V::Elem::from_f32));
        let [round, cap, k_min] = ks(x, [V::Elem::ROUND, V::Elem::CAP, V::Elem::K_MIN]);
        let neg = x.lt(zero);

        // e^−|x| = (yh + yl)·2^k, `−|x|` capped where `σ(−|x|)` rounds to 0.
        let a = -x.abs();
        let (yh, yl, k) = exp_parts(V::select(a.lt(cap), cap, a));

        // d = 1 + e = dh + dl, exactly as `1 ≥ e`. Below `2^K_MIN`, `e` no
        // longer shows in `σ`, so `k` is capped there.
        let kd = V::select(k.lt(k_min), k_min, k);
        let (eh, el) = (yh.ldexp(kd), yl.ldexp(kd));
        let dh = one + eh;
        let dl = (eh - (dh - one)) + el;

        // σ = n·2^s / d: `n = yh + yl` and `s = k` for x < 0, else `1` and `0`.
        let (nh, nl) = (V::select(neg, yh, one), V::select(neg, yl, zero));
        let inv = one / dh;
        let q = nh * inv;
        // q + (n − q·d)/d, with `nh − q·dh` exact.
        let r = q.residual(dh, nh) + (-q).mul_add(dl, nl);
        let sigma = r.mul_add(inv, q);

        scale(sigma, V::select(neg, k, zero), half, round)
    }
}

/// Where `σ`'s inputs are cut, per element type.
pub(crate) trait Limits: Kernel {
    /// Below `exp`'s underflow threshold: `σ(CAP) = eᶜᵃᵖ` rounds to `+0`,
    /// and `CAP`'s `k` still splits into two normal powers of 2.
    const CAP: Self;
    /// A `k` below which `e` is negligible in `1 + e`: `2^K_MIN` is about
    /// 2⁻¹⁵ ULP of the result. Capping `k` there keeps `yl·2^k` normal:
    /// subnormal arithmetic is many times slower on x86.
    const K_MIN: Self;
}

impl Limits for f32 {
    const CAP: f32 = -112.0;
    const K_MIN: f32 = -40.0;
}

impl Limits for f64 {
    const CAP: f64 = -753.0;
    const K_MIN: f64 = -100.0;
}

#[cfg(test)]
mod tests {
    use crate::math::{MathElement, SigmoidOp};
    use crate::testing::dd::sigmoid_dd;
    use crate::testing::{
        TestFloat, assert_exact, assert_ulps, assert_ulps_every_f32, samples, ulp_error_scaled,
    };
    use std::vec::Vec;

    /// `σ` in `f64`, stable: the `f32` reference.
    fn sigmoid_ref(x: f64) -> f64 {
        let e = (-x.abs()).exp();
        if x < 0.0 {
            e / (1.0 + e)
        } else {
            1.0 / (1.0 + e)
        }
    }

    /// Finite samples of `[−max, max]`, around `around`.
    fn inputs<T: TestFloat>(max: f64, around: &[f64]) -> Vec<T> {
        samples::<T>(100_000, max, around)
            .into_iter()
            .filter(|v| v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn sigmoid_f32_within_1_ulp() {
        // Around 1, where σ rounds to 1 (17.33), and the underflow
        // thresholds of `σ ≈ eˣ`.
        let x = inputs::<f32>(200.0, &[1.0, 17.33, 87.34, 88.72, 103.97]);
        assert_ulps::<f32, SigmoidOp>("sigmoid", &x, sigmoid_ref, 1.0);
    }

    #[test]
    fn sigmoid_f64_within_1_ulp() {
        let x = inputs::<f64>(1000.0, &[1.0, 37.43, 708.4, 709.78, 745.13]);
        for (backend, got) in f64::every_backend::<SigmoidOp>(&x) {
            for (&x, &got) in x.iter().zip(&got) {
                let e = ulp_error_scaled(got, sigmoid_dd(x));
                assert!(
                    e <= 1.0,
                    "sigmoid f64 {backend}: sigmoid({x:?}) = {got:?}, {e:.3} ULP"
                );
            }
        }
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn sigmoid_reference_is_precise() {
        let scale = |(h, l, k): (f64, f64, i32)| (h + l) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        for (x, want) in [
            (0.0, 0.5),
            (1.0, 0.731_058_578_630_004_9),
            (-1.0, 0.268_941_421_369_995_1),
            (-20.0, 2.061_153_618_190_203_7e-9),
            (-740.0, 4.2e-322),
        ] {
            assert_eq!(scale(sigmoid_dd(x)), want, "sigmoid_dd({x})");
        }
    }

    #[test]
    fn sigmoid_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.5),
            (-0.0, 0.5),
            (inf, 1.0),
            (-inf, 0.0),
            (nan, nan),
            (1000.0, 1.0),
            (-1000.0, 0.0),
        ];
        assert_exact::<f64, SigmoidOp>("sigmoid", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, SigmoidOp>("sigmoid", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn sigmoid_f32_exhaustive() {
        assert_ulps_every_f32::<SigmoidOp>("sigmoid", sigmoid_ref, 1.0);
    }
}
