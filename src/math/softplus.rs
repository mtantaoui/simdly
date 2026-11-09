//! `softplus(x) = ln(1 + eˣ)` and `log_sigmoid(x) = ln σ(x) = −softplus(−x)`,
//! ≤ 1 ULP on every backend, written once over [`Float`].
//!
//! `eˣ = (yh + yl)·2^k` to a few bits more than the element type
//! ([`exp_parts`]). With `K = max(k, 0)`,
//!
//! `1 + eˣ = 2^K · w`, `w = (yh + yl)·2^(k−K) + 2^(−K)` in `[0.7, 2.5]`,
//!
//! so `softplus(x) = K·ln 2 + ln w`: no overflow, and for `x ≥ 0` the `x` in
//! `x + ln(1 + e⁻ˣ)` is never rounded separately. `w = wh + wl` is formed
//! exactly ([`two_sum`]), then reduced as in [`ln`]:
//! `wh = m·2^n`, and `ln(wh + wl) = ln wh + wl/wh`, `wl/wh` among the small
//! terms of `ln`'s reconstruction. As in [`log1p`](super::log1p), for
//! `x < 0` and `n = 0` the series takes `e`'s high part itself rather than
//! `wh − 1`, which cancels against `wl` where `1 + e` rounds.
//!
//! Above [`Limits::BIG`] the result is `x` (`e⁻ˣ` is below half an ULP);
//! for `k` below [`Limits::K_MIN`], `ln(1 + e) = e` to 0.02 ULP or better, so
//! the result is `(yh + yl)·2^k`, scaled last as in `exp`.
//!
//! `softplus(−∞) = +0`, `softplus(+∞) = +∞`; `log_sigmoid(−∞) = −∞`,
//! `log_sigmoid(+∞) = −0`; NaN propagates.

use super::atan::ks;
use super::exp::{self, exp_parts, scale};
use super::float::{Element, Float, two_sum};
use super::ln;

/// A lane type with `ln(1 + eˣ)`.
pub(crate) trait Softplus {
    fn softplus(self) -> Self;
}

/// A lane type with `ln σ(x) = −ln(1 + e⁻ˣ)`.
pub(crate) trait LogSigmoid {
    fn log_sigmoid(self) -> Self;
}

impl<V: Float> LogSigmoid for V
where
    V::Elem: Limits,
{
    #[inline(always)]
    fn log_sigmoid(self) -> Self {
        -(-self).softplus()
    }
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Softplus for V
where
    V::Elem: Limits,
{
    #[inline(always)]
    fn softplus(self) -> Self {
        let x = self;
        let [zero, half, one, two] = ks(x, [0.0, 0.5, 1.0, 2.0].map(V::Elem::from_f32));
        let [round, cap, big, k_min] = ks(
            x,
            [V::Elem::ROUND, V::Elem::CAP, V::Elem::BIG, V::Elem::K_MIN],
        );
        let ln2 = <V::Elem as ln::Kernel>::LN2;
        let [ln2_hi, ln2_lo, sqrt2] = ks(x, [ln2[0], ln2[1], <V::Elem as ln::Kernel>::SQRT2]);

        // eˣ = (yh + yl)·2^k, `x` clamped to [CAP, BIG]: outside, the result
        // is selected below.
        let xc = V::select(x.lt(cap), cap, V::select(x.gt(big), big, x));
        let (yh, yl, k) = exp_parts(xc);

        // w = (yh + yl)·2^s + 2^t = wh + wl, exactly: s = min(k, 0), capped
        // at K_MIN to keep `yl·2^s` normal, t = −max(k, 0) ≥ −BIG/ln 2.
        let neg = k.lt(zero);
        let s = V::select(neg, V::select(k.lt(k_min), k_min, k), zero);
        let kk = V::select(neg, zero, k);
        let (p, l) = (yh.ldexp(s), yl.ldexp(s));
        let (wh, c) = two_sum(p, one.ldexp(-kk));

        // wh = m·2^n, m ∈ [√2/2, √2).
        let (m, e) = wh.frexp();
        let big_m = m.ge(sqrt2);
        let m = V::select(big_m, m * half, m);
        let n = e + V::select(big_m, one, zero);

        // ln(1 + f) + lo/wh. For x < 0 and n = 0, `1 + f = 1 + p` exactly
        // with `f = p`, so `lo = l` alone: `f = wh − 1` and `(c + l)/wh`
        // would cancel where `1 + p` rounds. Otherwise `f = m − 1` is exact
        // and `lo = c + l`. N = n + K.
        let near = V::select(neg, n, one).eq(zero);
        let f = V::select(near, p, m - one);
        let lo = V::select(near, l, c + l);
        let n = n + kk;

        // ln(1 + f) = f − hfsq + s·(hfsq + R), as in `ln`.
        let sf = f / (two + f);
        let z = sf * sf;
        let r = <V::Elem as ln::Kernel>::r(z, z * z);
        let hfsq = half * f * f;
        let y = ((sf.mul_add(hfsq + r, n.mul_add(ln2_lo, lo / wh)) - hfsq) + f) + n * ln2_hi;

        // ln(1 + e) = e for tiny e.
        let tiny = scale(yh + yl, k, half, round);
        V::select(x.gt(big), x, V::select(k.lt(k_min), tiny, y))
    }
}

/// Where `softplus`'s inputs are cut, per element type.
pub(crate) trait Limits: exp::Kernel + ln::Kernel {
    /// As `sigmoid`'s: `e^CAP` rounds to `+0`, and `CAP`'s `k` still splits
    /// into two normal powers of 2.
    const CAP: Self;
    /// Above `BIG`, `softplus(x) = x + e⁻ˣ` rounds to `x`; `2^−k` stays normal.
    const BIG: Self;
    /// Below `2^K_MIN`, `ln(1 + e) = e·(1 − e/2 + …)` is `e` to 0.02 ULP
    /// (`f32`) and 2⁻⁴⁷ ULP (`f64`). `k` is capped there in the series too:
    /// for `f32`, a smaller `e` would turn `s⁴` subnormal, which costs over
    /// 2× in throughput on x86.
    const K_MIN: Self;
}

impl Limits for f32 {
    const CAP: f32 = -112.0;
    const BIG: f32 = 20.0;
    const K_MIN: f32 = -29.0;
}

impl Limits for f64 {
    const CAP: f64 = -753.0;
    const BIG: f64 = 40.0;
    const K_MIN: f64 = -100.0;
}

#[cfg(test)]
mod tests {
    use crate::math::{LogSigmoidOp, MathElement, SoftplusOp, UnaryOp};
    use crate::testing::dd::softplus_dd;
    use crate::testing::{TestFloat, assert_exact, samples, ulp_error_scaled};
    use std::vec::Vec;

    /// `ln σ(x) = −softplus(−x)`.
    fn log_sigmoid_dd(x: f64) -> (f64, f64, i32) {
        let (h, l, k) = softplus_dd(-x);
        (-h, -l, k)
    }

    /// `F` within 1 ULP of `reference` on every backend, for finite samples
    /// of `[−max, max]` around the cut-offs.
    fn check<T: TestFloat + MathElement, F: UnaryOp>(
        name: &str,
        max: f64,
        around: &[f64],
        reference: fn(f64) -> (f64, f64, i32),
    ) {
        let x: Vec<T> = samples::<T>(100_000, max, around)
            .into_iter()
            .filter(|v| v.to_f64().is_finite())
            .collect();
        for (backend, got) in T::every_backend::<F>(&x) {
            for (&x, &got) in x.iter().zip(&got) {
                let e = ulp_error_scaled(got, reference(x.to_f64()));
                assert!(
                    e <= 1.0,
                    "{name} {} {backend}: {name}({x:?}) = {got:?}, {e:.3} ULP",
                    T::NAME
                );
            }
        }
    }

    // Around 0, ±ln 2 (k changes sign), BIG, `k = K_MIN` and the underflow
    // thresholds of `eˣ`.
    const AROUND_F32: [f64; 8] = [1.0, 0.35, 20.0, 20.1, 87.34, 88.72, 103.97, 112.0];
    const AROUND_F64: [f64; 8] = [1.0, 0.35, 40.0, 69.3, 708.4, 709.78, 745.13, 753.0];

    #[test]
    fn softplus_within_1_ulp() {
        check::<f32, SoftplusOp>("softplus", 200.0, &AROUND_F32, softplus_dd);
        check::<f64, SoftplusOp>("softplus", 1000.0, &AROUND_F64, softplus_dd);
    }

    #[test]
    fn log_sigmoid_within_1_ulp() {
        check::<f32, LogSigmoidOp>("log_sigmoid", 200.0, &AROUND_F32, log_sigmoid_dd);
        check::<f64, LogSigmoidOp>("log_sigmoid", 1000.0, &AROUND_F64, log_sigmoid_dd);
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn softplus_reference_is_precise() {
        let scale = |(h, l, k): (f64, f64, i32)| (h + l) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        for (x, want) in [
            (0.0, core::f64::consts::LN_2),
            (1.0, 1.313_261_687_518_222_8),
            (-1.0, 0.313_261_687_518_222_8),
            (-20.0, 2.061_153_620_314_381e-9),
            (30.0, 30.000_000_000_000_092),
            (-740.0, 4.2e-322),
        ] {
            assert_eq!(scale(softplus_dd(x)), want, "softplus_dd({x})");
        }
    }

    #[test]
    fn softplus_special_values() {
        let (inf, nan, ln2) = (f64::INFINITY, f64::NAN, core::f64::consts::LN_2);
        let cases = [
            (0.0, ln2),
            (-0.0, ln2),
            (inf, inf),
            (-inf, 0.0),
            (nan, nan),
            (1000.0, 1000.0),
            (-1000.0, 0.0),
        ];
        assert_exact::<f64, SoftplusOp>("softplus", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, SoftplusOp>("softplus", &cases32);
    }

    #[test]
    fn log_sigmoid_special_values() {
        let (inf, nan, ln2) = (f64::INFINITY, f64::NAN, core::f64::consts::LN_2);
        let cases = [
            (0.0, -ln2),
            (-0.0, -ln2),
            (inf, -0.0),
            (-inf, -inf),
            (nan, nan),
            (1000.0, -0.0),
            (-1000.0, -1000.0),
        ];
        assert_exact::<f64, LogSigmoidOp>("log_sigmoid", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, LogSigmoidOp>("log_sigmoid", &cases32);
    }

    /// `softplus` in `f64`, stable: the exhaustive `f32` reference.
    fn softplus_ref(x: f64) -> f64 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`. `log_sigmoid` is
    /// `−softplus(−x)`, the same kernel.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn softplus_f32_exhaustive() {
        crate::testing::assert_ulps_every_f32::<SoftplusOp>("softplus", softplus_ref, 1.0);
    }
}
