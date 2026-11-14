//! `ln(1 + x)`, ≤ 1 ULP on every backend, accurate where `1 + x` rounds:
//! fdlibm's approach (`s_log1p.c`) on [`ln`](super::ln)'s reduction and
//! series, written once over [`Float`].
//!
//! `u = 1 + x` is formed exactly as `u + c` ([`two_sum`]), then reduced as in
//! `ln`: `u = m · 2^n`, `m ∈ [√2/2, √2)`. For `n = 0`, `f = m − 1 = u − 1`
//! would lose `c`, but `1 + f = 1 + x` exactly with `f = x`, so the series
//! takes `x` itself. Otherwise `f = m − 1` is exact and `ln(u + c) ≈ ln u +
//! c/u`: `c/u` joins `n·ln2_lo` among the small terms.
//!
//! `log1p(±0) = ±0`, `log1p(−1) = −∞`, NaN below `−1`, `log1p(+∞) = +∞`.
//!
//! The reduction and the series are [`reduce`] and [`ln_sum`], shared with
//! the kernels that take the logarithm of a sum formed exactly
//! ([`softplus`](super::softplus)): each picks its own `f` and correction.

use super::atan::{k, ks};
use super::float::{Element, Float, two_sum};
use super::ln::Kernel;

/// A lane type with `ln(1 + x)`.
pub(crate) trait Log1p {
    fn log1p(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Log1p for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn log1p(self) -> Self {
        let x = self;
        let [zero, one] = ks(x, [0.0, 1.0].map(V::Elem::from_f32));

        // 1 + x = u + c, exactly; `u` is positive and normal for x > −1.
        let (u, c) = two_sum(one, x);
        let (m, n) = reduce(u);

        // ln(1 + f) + c/u, with `f = x` and no correction for n = 0.
        let near = n.eq(zero);
        let f = V::select(near, x, m - one);
        let y = ln_sum(f, V::select(near, zero, c), u, n);

        let [neg_one, inf] = [k(x, -1.0), k(x, f32::INFINITY)];
        V::select(
            x.gt(neg_one),
            // ±0 keeps its sign; +∞ stays +∞.
            V::select(x.eq(zero), x, V::select(x.lt(inf), y, x)),
            V::select(x.eq(neg_one), -inf, k(x, f32::NAN)),
        )
    }
}

/// `w = m·2^n`, `m ∈ [√2/2, √2)`, for a positive normal `w`: `ln`'s
/// reduction, with musl's `√2` threshold.
#[inline(always)]
pub(crate) fn reduce<V: Float>(w: V) -> (V, V)
where
    V::Elem: Kernel,
{
    let [zero, half, one] = [k(w, 0.0), k(w, 0.5), k(w, 1.0)];
    let sqrt2 = w.constant(V::Elem::SQRT2);
    let (m, e) = w.frexp();
    let big = m.ge(sqrt2);
    (V::select(big, m * half, m), e + V::select(big, one, zero))
}

/// `n·ln 2 + ln(1 + f) + lo/w`, for `|f| < √2 − 1` and a small correction
/// `lo`: `ln`'s series and reconstruction, small terms first, with `lo/w`
/// next to `n·ln2_lo`. `f` must be exact: for `w = m·2^n` from [`reduce`],
/// `f = m − 1` is.
#[inline(always)]
pub(crate) fn ln_sum<V: Float>(f: V, lo: V, w: V, n: V) -> V
where
    V::Elem: Kernel,
{
    let [half, two] = [k(f, 0.5), k(f, 2.0)];
    let [ln2_hi, ln2_lo] = ks(f, V::Elem::LN2);

    // ln(1 + f) = f − hfsq + s·(hfsq + R), as in `ln`.
    let s = f / (two + f);
    let z = s * s;
    let r = V::Elem::r(z, z * z);
    let hfsq = half * f * f;
    ((s.mul_add(hfsq + r, n.mul_add(ln2_lo, lo / w)) - hfsq) + f) + n * ln2_hi
}

#[cfg(test)]
mod tests {
    use crate::math::Log1pOp;
    use crate::testing::dd::log1p_dd;
    use crate::testing::{TestFloat, assert_exact, assert_ulps_dd, assert_ulps_every_f32, samples};
    use std::vec::Vec;

    /// Every binade above −1, subnormals included, and the neighbours of
    /// −1 (huge negative results), of `±1` and of the `n = 0` boundaries
    /// `√2 − 1` and `√2/2 − 1`.
    fn inputs<T: TestFloat>() -> Vec<T> {
        let max = if T::NAME == "f32" {
            f32::MAX.into()
        } else {
            f64::MAX
        };
        let around = [
            1.0,
            0.5,
            core::f64::consts::SQRT_2 - 1.0,
            1.0 - core::f64::consts::FRAC_1_SQRT_2,
        ];
        samples::<T>(150_000, max, &around)
            .into_iter()
            .filter(|v| v.to_f64() > -1.0 && v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn log1p_within_1_ulp() {
        assert_ulps_dd::<f32, Log1pOp>("log1p", &inputs(), log1p_dd, 1.0);
        assert_ulps_dd::<f64, Log1pOp>("log1p", &inputs(), log1p_dd, 1.0);
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn log1p_reference_is_precise() {
        for (x, want) in [
            (1.0, core::f64::consts::LN_2),
            (1e-10, 9.999_999_999_500_001e-11),
            (-0.5, -core::f64::consts::LN_2),
            (0.01, 9.950_330_853_168_083e-3),
            (-0.999_999, -13.815_510_557_935_518),
            (1e300, 690.775_527_898_213_7),
        ] {
            let (hi, lo) = log1p_dd(x);
            assert_eq!(hi + lo, want, "log1p_dd({x:e})");
        }
    }

    #[test]
    fn log1p_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (-1.0, -inf),
            (-2.0, nan),
            (inf, inf),
            (-inf, nan),
            (nan, nan),
            (1e-30, 1e-30),
            (1e-310, 1e-310),
            (-1e-310, -1e-310),
        ];
        assert_exact::<f64, Log1pOp>("log1p", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, Log1pOp>("log1p", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn log1p_f32_exhaustive() {
        assert_ulps_every_f32::<Log1pOp>("log1p", f64::ln_1p, 1.0);
    }
}
