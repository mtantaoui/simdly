//! `atanh x`, `asinh x` and `acosh x`, ≤ 1 ULP on every backend, written
//! once over [`Float`]. Each is `ln(1 + p)` for an argument `p = ph + pl`
//! formed to about twice the element type's precision: rounding `p` and
//! calling `log1p` would round twice, the argument's error passing into
//! the result. [`ln1p_parts`] then follows [`log1p`](super::log1p): `1 + ph
//! = u + c` exactly, `u` reduced, and the series takes `ph` itself when
//! `u`'s exponent is 0, or `m − 1` with `(c + pl)/u` among the small terms.
//!
//! - **`atanh t = ½·ln(1 + q)`**, `t = |x|`, `q = 2t/(1 − t)`: `1 − t = dh
//!   + dl` exactly ([`two_sum`]), `qh = 2t·(1/dh)`, then one correction step
//!   with the exact residual `2t − qh·dh`. The halving is exact.
//! - **`asinh t = ln(1 + p)`**, `p = t + (√(1 + t²) − 1)`: `t²` exact
//!   (`residual`), `1 + t²` exact, its square root `rh + rl` with one Newton
//!   step (as in [`hypot`](super::hypot)), and `rh − 1` exact.
//! - **`acosh x = ln(1 + p)`**, `t = x − 1` (exact below `BIG`), `p = t +
//!   √(t² + 2t)`, the same way.
//! - Above [`Limits::BIG`], `asinh t` and `acosh t` are `ln(2t)` to far
//!   below an ULP: `ln(1 + p) + 2·ln 2` with `p = t/2 − 1`, exact, so `t²`
//!   never overflows.
//!
//! Below [`Limits::LINEAR`], `atanh x` and `asinh x` are `x`: the kernels'
//! inputs are capped there, which keeps the series' `s⁴` normal.
//! `atanh(±0) = ±0`, `atanh(±1) = ±∞`, NaN beyond `±1`; `asinh(±0) = ±0`,
//! `asinh(±∞) = ±∞`; `acosh(1) = +0`, `acosh(+∞) = +∞`, NaN below 1. NaN
//! propagates.

use super::atan::k;
use super::float::{Float, two_sum};
use super::ln::Kernel;
use super::log1p::{ln_sum, reduce};

/// A lane type with the inverse hyperbolic tangent.
pub(crate) trait Atanh {
    fn atanh(self) -> Self;
}

/// A lane type with the inverse hyperbolic sine.
pub(crate) trait Asinh {
    fn asinh(self) -> Self;
}

/// A lane type with the inverse hyperbolic cosine.
pub(crate) trait Acosh {
    fn acosh(self) -> Self;
}

/// `ln(1 + ph + pl) + kk·ln 2`, for `ph + pl > −1` and `|pl|` about an ULP
/// of `ph` or less.
#[inline(always)]
fn ln1p_parts<V: Float>(ph: V, pl: V, kk: V) -> V
where
    V::Elem: Kernel,
{
    let [zero, one] = [k(ph, 0.0), k(ph, 1.0)];
    let (u, c) = two_sum(one, ph);
    let (m, n) = reduce(u);
    let near = n.eq(zero);
    let f = V::select(near, ph, m - one);
    let lo = V::select(near, pl, c + pl);
    ln_sum(f, lo, u, n + kk)
}

/// `(rh, rl)`, `rh + rl = √(wh + wl)` to about twice the element type's
/// precision: one Newton step, `rh²` exact (`residual`).
#[inline(always)]
fn sqrt_parts<V: Float>(wh: V, wl: V) -> (V, V) {
    let rh = wh.sqrt();
    (rh, (rh.residual(rh, wh) + wl) / (rh + rh))
}

/// `t² = sh + sl`, exactly (`residual`).
#[inline(always)]
fn square<V: Float>(t: V) -> (V, V) {
    let sh = t * t;
    (sh, -t.residual(t, sh))
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

        let y = half * ln1p_parts(qh, ql, zero);

        let inf = k(x, f32::INFINITY);
        let y = V::select(a.lt(one), y, V::select(a.eq(one), inf, k(x, f32::NAN)));
        // atanh is odd; tiny x, ±0 included, is its own result.
        V::select(a.lt(tiny), x, V::select(x.lt(zero), -y, y))
    }
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Asinh for V
where
    V::Elem: Limits,
{
    #[inline(always)]
    fn asinh(self) -> Self {
        let x = self;
        let [zero, half, one, two] = [k(x, 0.0), k(x, 0.5), k(x, 1.0), k(x, 2.0)];
        let [tiny, big] = [x.constant(V::Elem::LINEAR), x.constant(V::Elem::BIG)];
        let a = x.abs();
        let huge = a.ge(big);
        let t = V::select(a.lt(tiny), tiny, V::select(huge, big, a));

        // p = t + (r − 1), r = √(1 + t²) = rh + rl: `1 + t²` and `rh − 1`
        // exact.
        let (sh, sl) = square(t);
        let (wh, c) = two_sum(one, sh);
        let (rh, rl) = sqrt_parts(wh, c + sl);
        let (g, ge) = two_sum(rh, -one);
        let (ph, pc) = two_sum(t, g);
        let pl = pc + (ge + rl);

        // Above BIG, ln(2a) = ln(1 + (a/2 − 1)) + 2·ln 2.
        let (hh, hl) = two_sum(half * a, -one);
        let y = ln1p_parts(
            V::select(huge, hh, ph),
            V::select(huge, hl, pl),
            V::select(huge, two, zero),
        );

        // asinh is odd; tiny x, ±0 included, is its own result; ±∞ too.
        let y = V::select(a.lt(k(x, f32::INFINITY)), y, a);
        V::select(a.lt(tiny), x, V::select(x.lt(zero), -y, y))
    }
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Acosh for V
where
    V::Elem: Limits,
{
    #[inline(always)]
    fn acosh(self) -> Self {
        let x = self;
        let [zero, half, one, two] = [k(x, 0.0), k(x, 0.5), k(x, 1.0), k(x, 2.0)];
        let big = x.constant(V::Elem::BIG);
        let huge = x.ge(big);

        // p = t + √(t² + 2t), t = x − 1, exact as x < BIG; `t² + 2t`
        // exact. Out-of-range lanes compute with t = 0.
        let t = V::select(x.gt(one), V::select(huge, big, x), one) - one;
        let (sh, sl) = square(t);
        let (wh, c) = two_sum(two * t, sh);
        let (rh, rl) = sqrt_parts(wh, c + sl);
        let (ph, pc) = two_sum(t, rh);
        let pl = pc + rl;

        // Above BIG, ln(2x) = ln(1 + (x/2 − 1)) + 2·ln 2.
        let (hh, hl) = two_sum(half * x, -one);
        let y = ln1p_parts(
            V::select(huge, hh, ph),
            V::select(huge, hl, pl),
            V::select(huge, two, zero),
        );

        let inf = k(x, f32::INFINITY);
        // acosh(1) = +0: `sqrt_parts` would divide 0 by 0 there.
        let y = V::select(x.eq(one), zero, V::select(x.lt(inf), y, x));
        V::select(x.ge(one), y, k(x, f32::NAN))
    }
}

/// Where the inverse hyperbolic functions' inputs are cut, per element
/// type.
pub(crate) trait Limits: Kernel {
    /// Below `LINEAR`, `atanh x = x·(1 + x²/3 + …)` and `asinh x = x·(1 −
    /// x²/6 + …)` round to `x`.
    const LINEAR: Self;
    /// Above `BIG`, `asinh t` and `acosh t` are `ln(2t) ± 1/(4t²)`, the
    /// correction below 2⁻²² (`f32`) and 2⁻³⁴ (`f64`) ULP; `t²` stays
    /// finite and `t − 1` exact below it.
    const BIG: Self;
}

impl Limits for f32 {
    const LINEAR: f32 = 1.0 / 4096.0; // 2^−12
    const BIG: f32 = 1_048_576.0; // 2^20
}

impl Limits for f64 {
    const LINEAR: f64 = 1.0 / 268_435_456.0; // 2^−28
    const BIG: f64 = 1_099_511_627_776.0; // 2^40
}

#[cfg(test)]
mod tests {
    use crate::math::{AcoshOp, AsinhOp, AtanhOp};
    use crate::testing::dd::{acosh_dd, asinh_dd, atanh_dd};
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

    /// Every finite binade, subnormals included, around 1 (where
    /// `√(1 + t²) − 1` is small against `t`, and `acosh` starts), the
    /// `LINEAR` and `BIG` cut-offs, and where `1 + p` crosses `√2`
    /// (asinh: `t ≈ 0.354`; acosh: `x ≈ 1.061`).
    fn wide<T: TestFloat>(min: f64) -> Vec<T> {
        let max = if T::NAME == "f32" {
            f32::MAX.into()
        } else {
            f64::MAX
        };
        let around = [
            1.0,
            0.353_6,
            1.060_7,
            1.0 / 4096.0,
            3.725e-9,
            1_048_576.0,
            1_099_511_627_776.0,
        ];
        samples::<T>(150_000, max, &around)
            .into_iter()
            .filter(|v| v.to_f64() >= min && v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn asinh_within_1_ulp() {
        assert_ulps_dd::<f32, AsinhOp>("asinh", &wide(f64::MIN), asinh_dd, 1.0);
        assert_ulps_dd::<f64, AsinhOp>("asinh", &wide(f64::MIN), asinh_dd, 1.0);
    }

    #[test]
    fn acosh_within_1_ulp() {
        assert_ulps_dd::<f32, AcoshOp>("acosh", &wide(1.0), acosh_dd, 1.0);
        assert_ulps_dd::<f64, AcoshOp>("acosh", &wide(1.0), acosh_dd, 1.0);
    }

    /// The references themselves, against values computed at 300 bits.
    #[test]
    fn asinh_acosh_references_are_precise() {
        for (x, want) in [
            (1e-10, 1e-10),
            (-0.5, -0.481_211_825_059_603_47),
            (2.0, 1.443_635_475_178_810_3),
            (1e30, 69.770_699_970_381_32),
        ] {
            let (hi, lo) = asinh_dd(x);
            assert_eq!(hi + lo, want, "asinh_dd({x:e})");
        }
        for (x, want) in [
            (1.0, 0.0),
            (1.000_000_1, 4.472_135_919_037_347e-4),
            (2.0, 1.316_957_896_924_816_8),
            (1e30, 69.770_699_970_381_32),
        ] {
            let (hi, lo) = acosh_dd(x);
            assert_eq!(hi + lo, want, "acosh_dd({x:e})");
        }
    }

    #[test]
    fn asinh_acosh_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let asinh = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (inf, inf),
            (-inf, -inf),
            (nan, nan),
            (1e-30, 1e-30),
            (-1e-310, -1e-310),
        ];
        assert_exact::<f64, AsinhOp>("asinh", &asinh);
        assert_exact::<f32, AsinhOp>("asinh", &asinh.map(|(x, y)| (x as f32, y as f32)));
        let acosh = [
            (1.0, 0.0),
            (inf, inf),
            (0.5, nan),
            (0.0, nan),
            (-1.0, nan),
            (-inf, nan),
            (nan, nan),
        ];
        assert_exact::<f64, AcoshOp>("acosh", &acosh);
        assert_exact::<f32, AcoshOp>("acosh", &acosh.map(|(x, y)| (x as f32, y as f32)));
    }

    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn asinh_f32_exhaustive() {
        assert_ulps_every_f32::<AsinhOp>("asinh", f64::asinh, 1.0);
    }

    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn acosh_f32_exhaustive() {
        assert_ulps_every_f32::<AcoshOp>("acosh", f64::acosh, 1.0);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn atanh_f32_exhaustive() {
        assert_ulps_every_f32::<AtanhOp>("atanh", f64::atanh, 1.0);
    }
}
