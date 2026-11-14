//! `atanh x`, ≤ 1 ULP on every backend, written once over [`Float`].
//!
//! With `t = |x|`, `atanh t = ½·ln(1 + q)`, `q = 2t/(1 − t)`. `1 − t = dh +
//! dl` exactly ([`two_sum`]), and `q = qh + ql` to about twice the element
//! type's precision: `qh = 2t·(1/dh)`, then one correction step with the
//! exact residual `2t − qh·dh`. `ln(1 + qh + ql)` then follows
//! [`log1p`](super::log1p): `1 + qh = u + c` exactly, `u` reduced, and the
//! series takes `qh` itself when `u`'s exponent is 0, or `m − 1` with
//! `(c + ql)/u` among the small terms. The halving is exact.
//!
//! Rounding `q` first and calling `log1p` would round twice: the argument's
//! error passes into the result.
//!
//! Below [`Limits::LINEAR`], `atanh x = x`: the kernel's input is capped
//! there, which keeps the series' `s⁴` normal. `atanh(±0) = ±0`,
//! `atanh(±1) = ±∞`, NaN beyond `±1`; NaN propagates.

use super::atan::k;
use super::float::{Float, two_sum};
use super::ln::Kernel;
use super::log1p::{ln_sum, reduce};

/// A lane type with the inverse hyperbolic tangent.
pub(crate) trait Atanh {
    fn atanh(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Atanh for V
where
    V::Elem: Limits,
{
    #[inline(always)]
    fn atanh(self) -> Self {
        let x = self;
        let [zero, half, one, two] = [k(x, 0.0), k(x, 0.5), k(x, 1.0), k(x, 2.0)];
        let tiny = x.constant(V::Elem::LINEAR);
        let a = x.abs();
        let t = V::select(a.lt(tiny), tiny, a);

        // q = 2t/(1 − t) = qh + ql: 1 − t = dh + dl exactly, then
        // q = qh + (2t − qh·dh − qh·dl)/d, `2t − qh·dh` exact.
        let (dh, dl) = two_sum(one, -t);
        let inv = one / dh;
        let t2 = two * t;
        let qh = t2 * inv;
        let ql = (qh.residual(dh, t2) - qh * dl) * inv;

        // ln(1 + qh + ql), as in `log1p`.
        let (u, c) = two_sum(one, qh);
        let (m, n) = reduce(u);
        let near = n.eq(zero);
        let f = V::select(near, qh, m - one);
        let lo = V::select(near, ql, c + ql);
        let y = half * ln_sum(f, lo, u, n);

        let inf = k(x, f32::INFINITY);
        let y = V::select(a.lt(one), y, V::select(a.eq(one), inf, k(x, f32::NAN)));
        // atanh is odd; tiny x, ±0 included, is its own result.
        V::select(a.lt(tiny), x, V::select(x.lt(zero), -y, y))
    }
}

/// Where `atanh`'s input is cut, per element type.
pub(crate) trait Limits: Kernel {
    /// Below `LINEAR`, `atanh x = x·(1 + x²/3 + …)` rounds to `x`.
    const LINEAR: Self;
}

impl Limits for f32 {
    const LINEAR: f32 = 1.0 / 4096.0; // 2^−12
}

impl Limits for f64 {
    const LINEAR: f64 = 1.0 / 268_435_456.0; // 2^−28
}

#[cfg(test)]
mod tests {
    use crate::math::AtanhOp;
    use crate::testing::dd::atanh_dd;
    use crate::testing::{TestFloat, assert_exact, assert_ulps_dd, assert_ulps_every_f32, samples};
    use std::vec::Vec;

    /// Every binade inside `(−1, 1)`, subnormals included, and the
    /// neighbours of `±1` (huge results), of the `LINEAR` cut-offs and of
    /// where `1 + q` crosses `√2` and `2√2` (`x ≈ 0.172`, `0.478`).
    fn inputs<T: TestFloat>() -> Vec<T> {
        let around = [1.0, 0.5, 0.171_6, 0.477_6, 1.0 / 4096.0, 3.725e-9];
        samples::<T>(150_000, 1.0, &around)
            .into_iter()
            .filter(|v| v.to_f64().abs() < 1.0)
            .collect()
    }

    #[test]
    fn atanh_within_1_ulp() {
        assert_ulps_dd::<f32, AtanhOp>("atanh", &inputs(), atanh_dd, 1.0);
        assert_ulps_dd::<f64, AtanhOp>("atanh", &inputs(), atanh_dd, 1.0);
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn atanh_reference_is_precise() {
        for (x, want) in [
            (1e-10, 1e-10),
            (0.5, 0.549_306_144_334_054_9),
            (-0.1, -0.100_335_347_731_075_58),
            (0.999_999, 7.254_328_619_247_669),
            (0.01, 0.010_000_333_353_334_763),
        ] {
            let (hi, lo) = atanh_dd(x);
            assert_eq!(hi + lo, want, "atanh_dd({x:e})");
        }
    }

    #[test]
    fn atanh_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (1.0, inf),
            (-1.0, -inf),
            (2.0, nan),
            (-2.0, nan),
            (inf, nan),
            (-inf, nan),
            (nan, nan),
            (1e-30, 1e-30),
            (-1e-310, -1e-310),
        ];
        assert_exact::<f64, AtanhOp>("atanh", &cases);
        assert_exact::<f32, AtanhOp>("atanh", &cases.map(|(x, y)| (x as f32, y as f32)));
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn atanh_f32_exhaustive() {
        assert_ulps_every_f32::<AtanhOp>("atanh", f64::atanh, 1.0);
    }
}
