//! `ln x`, ≤ 1 ULP on every backend: fdlibm's algorithm (musl's `log.c`, and
//! the `logf.c` musl had before its table-driven one), written once over
//! [`Float`], `f32` kept in `f32`.
//!
//! 1. **Reduce:** `x = m · 2^k` with `m ∈ [√2/2, √2)` ([`Float::frexp`],
//!    then `m/2` and `k + 1` above musl's `√2` threshold); subnormals are
//!    scaled up first.
//! 2. **Approximate:** with `f = m − 1` and `s = f/(2 + f)`,
//!    `ln(1 + f) = ln(1 + s) − ln(1 − s) = 2s + s·R(s²)`, an odd series:
//!    `R` has 4 (`f32`) or 7 (`f64`) coefficients.
//! 3. **Reconstruct** in musl's order, small terms first:
//!    `s·(hfsq + R) + k·ln2_lo − hfsq + f + k·ln2_hi`, `hfsq = f²/2`. With
//!    `2s = f − s·f`, this equals `f − hfsq + s·(hfsq + R)` without ever
//!    rounding `2s`.
//!
//! `ln(±0) = −∞`, `ln(x < 0) = NaN`, `ln(+∞) = +∞`, `ln(1) = +0`.
//! Measured: 0.835 ULP (`f32`, every input) and 0.823 ULP (`f64`, against a
//! 200-bit reference), fused or not.

#![allow(clippy::excessive_precision, reason = "fdlibm's constants, verbatim")]

use super::atan::{k, ks};
use super::float::{Element, Float};

/// A lane type with a natural logarithm.
pub(crate) trait Ln {
    fn ln(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Ln for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn ln(self) -> Self {
        let x = self;
        let [zero, half, one, two] = ks(x, [0.0, 0.5, 1.0, 2.0].map(V::Elem::from_f32));
        let [ln2_hi, ln2_lo, sqrt2] = ks(x, [V::Elem::LN2[0], V::Elem::LN2[1], V::Elem::SQRT2]);
        let [tiny_scale, tiny_k] = ks(x, V::Elem::TINY);

        // x = m · 2^n, m ∈ [√2/2, √2).
        let tiny = x.lt(x.constant(V::Elem::MIN_POSITIVE));
        let (m, e) = V::select(tiny, x * tiny_scale, x).frexp();
        let big = m.ge(sqrt2);
        let m = V::select(big, m * half, m);
        let n = e + V::select(big, one, zero) + V::select(tiny, tiny_k, zero);

        // ln(1 + f) = f − hfsq + s·(hfsq + R).
        let f = m - one;
        let s = f / (two + f);
        let z = s * s;
        let r = V::Elem::r(z, z * z);
        let hfsq = half * f * f;
        let y = ((s.mul_add(hfsq + r, n * ln2_lo) - hfsq) + f) + n * ln2_hi;

        let inf = k(x, f32::INFINITY);
        V::select(
            x.gt(zero),
            V::select(x.lt(inf), y, x),
            V::select(x.eq(zero), -inf, k(x, f32::NAN)),
        )
    }
}

/// fdlibm's constants and series for one element type.
pub(crate) trait Kernel: Element {
    /// `ln 2` as `hi + lo`, `hi` with trailing zero bits: `k·hi` is exact.
    const LN2: [Self; 2];
    /// musl's threshold for `m ∈ [1, 2)`: from here on, `m/2` is closer to 1.
    const SQRT2: Self;
    /// `[2^j, −j]`: subnormals are multiplied by the first, and the second
    /// is added to `k`.
    const TINY: [Self; 2];

    /// `R(z) = t2 + t1` for `z = s²`, `w = z²`: the series split by parity.
    fn r<V: Float<Elem = Self>>(z: V, w: V) -> V;
}

impl Kernel for f32 {
    // musl logf.c (pre-2017).
    const LN2: [f32; 2] = [
        6.931_381_225_6e-1, // 0x3f317180
        9.058_000_614_5e-6, // 0x3717f7d1
    ];
    const SQRT2: f32 = f32::from_bits(0x3fb5_04f3);
    const TINY: [f32; 2] = [16_777_216.0, -24.0]; // 2^24

    #[inline(always)]
    fn r<V: Float<Elem = f32>>(z: V, w: V) -> V {
        // |(ln(1+s) − ln(1−s))/s − 2 − R(s²)| < 2^−34.24.
        let [lg1, lg2, lg3, lg4] = ks(
            z,
            [
                0.666_666_626_93,
                0.400_009_721_52,
                0.284_987_866_88,
                0.242_790_788_41,
            ],
        );
        let t1 = w * w.mul_add(lg4, lg2);
        let t2 = z * w.mul_add(lg3, lg1);
        t2 + t1
    }
}

impl Kernel for f64 {
    // musl log.c.
    const LN2: [f64; 2] = [
        f64::from_bits(0x3fe6_2e42_fee0_0000),
        f64::from_bits(0x3dea_39ef_3579_3c76),
    ];
    /// musl compares the high word only: `0x3ff6a09e` followed by zeros.
    const SQRT2: f64 = f64::from_bits(0x3ff6_a09e_0000_0000);
    const TINY: [f64; 2] = [18_014_398_509_481_984.0, -54.0]; // 2^54

    #[inline(always)]
    fn r<V: Float<Elem = f64>>(z: V, w: V) -> V {
        let [lg1, lg2, lg3, lg4, lg5, lg6, lg7] = ks(
            z,
            [
                f64::from_bits(0x3fe5_5555_5555_5593),
                f64::from_bits(0x3fd9_9999_9997_fa04),
                f64::from_bits(0x3fd2_4924_9422_9359),
                f64::from_bits(0x3fcc_71c5_1d8e_78af),
                f64::from_bits(0x3fc7_4664_96cb_03de),
                f64::from_bits(0x3fc3_9a09_d078_c69f),
                f64::from_bits(0x3fc2_f112_df3e_5244),
            ],
        );
        let t1 = w * w.mul_add(w.mul_add(lg6, lg4), lg2);
        let t2 = z * w.mul_add(w.mul_add(w.mul_add(lg7, lg5), lg3), lg1);
        t2 + t1
    }
}

#[cfg(test)]
mod tests {
    use crate::math::LnOp;
    use crate::testing::dd::ln_dd;
    use crate::testing::{TestFloat, assert_exact, assert_ulps_dd, assert_ulps_every_f32, samples};
    use std::vec::Vec;

    /// Positive values of every binade, subnormals included, and the
    /// neighbours of 1 (tiny results) and of the `√2` threshold.
    fn inputs<T: TestFloat>() -> Vec<T> {
        let max = if T::NAME == "f32" {
            f32::MAX.into()
        } else {
            f64::MAX
        };
        samples::<T>(150_000, max, &[1.0, core::f64::consts::SQRT_2, 0.5])
            .into_iter()
            .filter(|v| v.to_f64() > 0.0 && v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn ln_within_1_ulp() {
        assert_ulps_dd::<f32, LnOp>("ln", &inputs(), ln_dd, 1.0);
        assert_ulps_dd::<f64, LnOp>("ln", &inputs(), ln_dd, 1.0);
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn ln_reference_is_precise() {
        for (x, want) in [
            (2.0, core::f64::consts::LN_2),
            (10.0, core::f64::consts::LN_10),
            (1.0 + f64::EPSILON, 2.220_446_049_250_312_8e-16),
            (1e-310, -713.801_378_828_154_2),
            (1.7e308, 709.726_836_893_228_2),
        ] {
            let (hi, lo) = ln_dd(x);
            assert_eq!(hi + lo, want, "ln_dd({x:e})");
        }
    }

    #[test]
    fn ln_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (1.0, 0.0),
            (0.0, -inf),
            (-0.0, -inf),
            (-1.0, nan),
            (inf, inf),
            (-inf, nan),
            (nan, nan),
        ];
        assert_exact::<f64, LnOp>("ln", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, LnOp>("ln", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn ln_f32_exhaustive() {
        assert_ulps_every_f32::<LnOp>("ln", f64::ln, 1.0);
    }
}
