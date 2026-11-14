//! `sinh x` and `cosh x`, ≤ 1 ULP on every backend, written once over
//! [`Float`].
//!
//! With `a = |x|` and `E = eᵃ = (yh + yl)·2^k` to a few bits more than the
//! element type ([`exp_parts`]), `1/E = (qh + ql)·2^−k`: `qh = 1/yh`, and
//! `ql` corrects it for `yl` and `qh`'s rounding (`residual`). Then
//!
//! `cosh x = 2^(k−1)·((yh + yl) + (qh + ql)·2^−2k)`,
//! `sinh a = 2^(k−1)·((yh + yl) − (qh + ql)·2^−2k)`,
//!
//! the high parts summed exactly ([`two_sum`]), so `E ± 1/E` rounds once,
//! and the scaling by `2^(k−1)` comes last, in two normal factors as in
//! `exp`: results stay finite past `eᵃ`'s overflow, up to `ln(2·MAX)`.
//!
//! Below `|x| = 0.5`, `E − 1/E` cancels, so `sinh x = x + x·z·P(z)`,
//! `z = x²`, `P` the Taylor series (to `x⁹` for `f32`, `x¹⁵` for `f64`):
//! `x` exact and the rest below 5% of the result.
//!
//! `sinh(±0) = ±0`, `sinh(±∞) = ±∞`, `cosh(±0) = 1`, `cosh(±∞) = +∞`; NaN
//! propagates.

use super::atan::{k, ks};
use super::exp::{self, exp_parts, scale};
use super::float::{Element, Float, two_sum};

/// A lane type with the hyperbolic sine.
pub(crate) trait Sinh {
    fn sinh(self) -> Self;
}

/// A lane type with the hyperbolic cosine.
pub(crate) trait Cosh {
    fn cosh(self) -> Self;
}

/// `e^a = (yh + yl)·2^k` and `e^−a·2^k = (th + tl)`, for `a = |x|` capped
/// at [`Kernel::CAP`]: `(yh, yl, th, tl, k)`.
#[inline(always)]
fn parts<V: Float>(a: V) -> (V, V, V, V, V)
where
    V::Elem: Kernel,
{
    let [one, two] = [k(a, 1.0), k(a, 2.0)];
    let [cap, j_min] = [a.constant(V::Elem::CAP), a.constant(V::Elem::J_MIN)];
    let (yh, yl, kk) = exp_parts(V::select(a.gt(cap), cap, a));

    // 1/(yh + yl) = qh + ql: (1 − qh·yh − qh·yl)/(yh + yl) ≈ (…)·qh.
    let qh = one / yh;
    let ql = (qh.residual(yh, one) - qh * yl) * qh;

    // · 2^−2k, the exponent capped where the term no longer shows: keeps
    // `ql·2^j` normal.
    let j = -(two * kk);
    let j = V::select(j.lt(j_min), j_min, j);
    (yh, yl, qh.ldexp(j), ql.ldexp(j), kk)
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Cosh for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn cosh(self) -> Self {
        let [half, one] = [k(self, 0.5), k(self, 1.0)];
        let round = self.constant(V::Elem::ROUND);
        let (yh, yl, th, tl, kk) = parts(self.abs());
        let (sh, e) = two_sum(yh, th);
        scale(sh + (e + (yl + tl)), kk - one, half, round)
    }
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Sinh for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn sinh(self) -> Self {
        let x = self;
        let [zero, half, one] = [k(x, 0.0), k(x, 0.5), k(x, 1.0)];
        let round = x.constant(V::Elem::ROUND);
        let a = x.abs();

        // 2^(k−1)·(E − 1/E), for a ≥ 0.5: the high parts cancel exactly.
        let (yh, yl, th, tl, kk) = parts(a);
        let (dh, e) = two_sum(yh, -th);
        let big = scale(dh + (e + (yl - tl)), kk - one, half, round);
        let big = V::select(x.lt(zero), -big, big);

        // x + x·z·P(z) below 0.5.
        let z = x * x;
        let small = (x * z).mul_add(V::Elem::series(z), x);

        V::select(a.lt(half), small, big)
    }
}

/// `sinh`'s and `cosh`'s constants and series, per element type.
pub(crate) trait Kernel: exp::Kernel {
    /// Above `ln(2·MAX)`: `|x|` is capped there, and the result overflows
    /// to `∞` by itself. Its `k` still splits into two normal powers of 2.
    const CAP: Self;
    /// Where `2^−2k` is capped: `e^−a` is then about 2⁻⁶⁰ (`f32`) and
    /// 2⁻¹²⁰ (`f64`) of `eᵃ`, far below an ULP, and `ql·2^j` stays normal.
    const J_MIN: Self;

    /// `P(z) = (sinh x − x)/x³`, `z = x²`, `|x| < 0.5`: the Taylor series
    /// `1/3! + z/5! + …`, truncated below 2⁻³⁵ (`f32`) and 2⁻⁶⁴ (`f64`)
    /// of the result.
    fn series<V: Float<Elem = Self>>(z: V) -> V;
}

impl Kernel for f32 {
    const CAP: f32 = 90.0;
    const J_MIN: f32 = -60.0;

    #[inline(always)]
    fn series<V: Float<Elem = f32>>(z: V) -> V {
        let [c3, c5, c7, c9] = ks(z, [1.0 / 6.0, 1.0 / 120.0, 1.0 / 5040.0, 1.0 / 362_880.0]);
        z.mul_add(z.mul_add(z.mul_add(c9, c7), c5), c3)
    }
}

impl Kernel for f64 {
    const CAP: f64 = 711.0;
    const J_MIN: f64 = -120.0;

    #[inline(always)]
    fn series<V: Float<Elem = f64>>(z: V) -> V {
        let [c3, c5, c7, c9, c11, c13, c15] = ks(
            z,
            [
                1.0 / 6.0,
                1.0 / 120.0,
                1.0 / 5040.0,
                1.0 / 362_880.0,
                1.0 / 39_916_800.0,
                1.0 / 6_227_020_800.0,
                1.0 / 1_307_674_368_000.0,
            ],
        );
        let p = z.mul_add(z.mul_add(c15, c13), c11);
        z.mul_add(z.mul_add(z.mul_add(z.mul_add(p, c9), c7), c5), c3)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{CoshOp, MathElement, SinhOp, UnaryOp};
    use crate::testing::dd::{cosh_dd, sinh_dd};
    use crate::testing::{
        TestFloat, assert_exact, assert_ulps_every_f32, samples, ulp_error_scaled,
    };
    use std::vec::Vec;

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

    // Around 1, the series' end (0.5), `k`'s first step (ln 2 / 2), the
    // `J_MIN` cap (k = 30, 60), and the overflow thresholds of `eˣ` and of
    // `sinh`, `cosh`.
    const AROUND_F32: [f64; 7] = [1.0, 0.5, 0.3466, 20.8, 88.72, 89.42, 90.0];
    const AROUND_F64: [f64; 7] = [1.0, 0.5, 0.3466, 41.6, 709.78, 710.48, 711.0];

    #[test]
    fn sinh_within_1_ulp() {
        check::<f32, SinhOp>("sinh", 200.0, &AROUND_F32, sinh_dd);
        check::<f64, SinhOp>("sinh", 1000.0, &AROUND_F64, sinh_dd);
    }

    #[test]
    fn cosh_within_1_ulp() {
        check::<f32, CoshOp>("cosh", 200.0, &AROUND_F32, cosh_dd);
        check::<f64, CoshOp>("cosh", 1000.0, &AROUND_F64, cosh_dd);
    }

    /// The references themselves, against values computed at 300 bits.
    #[test]
    fn references_are_precise() {
        let scale = |(h, l, k): (f64, f64, i32)| (h + l) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        for (x, sinh, cosh) in [
            (1e-10, 1e-10, 1.0),
            (0.5, 0.521_095_305_493_747_4, 1.127_625_965_206_380_7),
            (-1.0, -1.175_201_193_643_801_4, 1.543_080_634_815_243_7),
            (3.0, 10.017_874_927_409_903, 10.067_661_995_777_765),
            (
                710.0,
                1.116_997_383_080_855_5e308,
                1.116_997_383_080_855_5e308,
            ),
        ] {
            assert_eq!(scale(sinh_dd(x)), sinh, "sinh_dd({x})");
            assert_eq!(scale(cosh_dd(x)), cosh, "cosh_dd({x})");
        }
    }

    #[test]
    fn special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let sinh = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (inf, inf),
            (-inf, -inf),
            (nan, nan),
            (1e-30, 1e-30),
            (-1e-310, -1e-310),
            (1000.0, inf),
            (-1000.0, -inf),
        ];
        assert_exact::<f64, SinhOp>("sinh", &sinh);
        assert_exact::<f32, SinhOp>("sinh", &sinh.map(|(x, y)| (x as f32, y as f32)));
        let cosh = [
            (0.0, 1.0),
            (-0.0, 1.0),
            (inf, inf),
            (-inf, inf),
            (nan, nan),
            (1e-30, 1.0),
            (1000.0, inf),
            (-1000.0, inf),
        ];
        assert_exact::<f64, CoshOp>("cosh", &cosh);
        assert_exact::<f32, CoshOp>("cosh", &cosh.map(|(x, y)| (x as f32, y as f32)));
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn sinh_f32_exhaustive() {
        assert_ulps_every_f32::<SinhOp>("sinh", f64::sinh, 1.0);
    }

    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn cosh_f32_exhaustive() {
        assert_ulps_every_f32::<CoshOp>("cosh", f64::cosh, 1.0);
    }
}
