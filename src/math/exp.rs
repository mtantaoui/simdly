//! `eˣ`, ≤ 1 ULP on every backend: fdlibm's algorithm (musl's `exp.c`, and
//! the `expf.c` musl had before its table-driven one), written once over
//! [`Float`], `f32` kept in `f32`.
//!
//! 1. **Reduce:** `k = round(x / ln 2)`, `r = (x − k·ln2_hi) − k·ln2_lo`,
//!    `|r| ≤ ln2 / 2`. `ln2_hi` has trailing zero bits, so `x − k·ln2_hi` is
//!    exact.
//! 2. **Approximate** `eʳ` with the rational form `R(r) = r·(eʳ + 1)/(eʳ − 1)
//!    ≈ 2 + r²·P(r²)`: with `c = r − r²·P(r²)`, `eʳ = 1 + r + r·c/(2 − c)`,
//!    evaluated small terms first: `1 + ((r·c/(2 − c) − lo) + hi)`.
//! 3. **Scale** by `2^k` as `2^⌊k/2⌋ · 2^(k − ⌊k/2⌋)`: both factors are
//!    normal although `k` reaches 128 / 1024 and −150 / −1075, and the second
//!    product rounds subnormal results once.
//!
//! Above `ln(MAX)` the result is `+∞`, below the underflow threshold `+0`;
//! NaN propagates. Measured: 0.910 ULP (`f32`, every input) and 0.844 ULP
//! (`f64`, against a 200-bit reference), fused or not.

#![allow(clippy::excessive_precision, reason = "fdlibm's constants, verbatim")]

use super::atan::ks;
use super::float::{Element, Float};

/// A lane type with an exponential.
pub(crate) trait Exp {
    fn exp(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Exp for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn exp(self) -> Self {
        let x = self;
        let [zero, half, one, two] = ks(x, [0.0, 0.5, 1.0, 2.0].map(V::Elem::from_f32));
        let [ln2_hi, ln2_lo, inv_ln2, round] = ks(
            x,
            [
                V::Elem::LN2[0],
                V::Elem::LN2[1],
                V::Elem::INV_LN2,
                V::Elem::ROUND,
            ],
        );
        let [over, under, inf] = ks(x, V::Elem::LIMITS);

        // x = k·ln2 + r.
        let k = x.mul_add(inv_ln2, round) - round;
        let hi = (-k).mul_add(ln2_hi, x);
        let lo = k * ln2_lo;
        let r = hi - lo;

        // eʳ = 1 + r + r·c/(2 − c).
        let z = r * r;
        let c = (-z).mul_add(V::Elem::p(z), r);
        let y = one + ((r * c / (two - c) - lo) + hi);

        // · 2^k, in two normal factors.
        let k1 = k.mul_add(half, round) - round;
        let y = y.ldexp(k1).ldexp(k - k1);

        V::select(x.gt(over), inf, V::select(x.lt(under), zero, y))
    }
}

/// fdlibm's constants and polynomial for one element type.
pub(crate) trait Kernel: Element {
    /// `ln 2` as `hi + lo`, `hi` with trailing zero bits.
    const LN2: [Self; 2];
    /// `1 / ln 2`.
    const INV_LN2: Self;
    /// `[over, under, ∞]`: above `over` the result is `∞`, below `under` it
    /// rounds to `0`.
    const LIMITS: [Self; 3];

    /// `P(z)`: `r·(eʳ + 1)/(eʳ − 1) ≈ 2 + z·P(z)`, `z = r²`.
    fn p<V: Float<Elem = Self>>(z: V) -> V;
}

impl Kernel for f32 {
    const LN2: [f32; 2] = [
        6.931_457_519_5e-1, // 0x3f317200
        1.428_606_765_3e-6, // 0x35bfbe8e
    ];
    const INV_LN2: f32 = 1.442_695_021_6;
    const LIMITS: [f32; 3] = [
        88.722_839_355_468_75,
        -103.972_084_045_410_16,
        f32::INFINITY,
    ];

    #[inline(always)]
    fn p<V: Float<Elem = f32>>(z: V) -> V {
        // fdlibm e_expf.c: |r·(eʳ+1)/(eʳ−1) − 2 − z·P| < 2^−27.74.
        let [p1, p2] = ks(z, [1.666_662_544_0e-1, -2.766_733_290_6e-3]);
        z.mul_add(p2, p1)
    }
}

impl Kernel for f64 {
    const LN2: [f64; 2] = [
        f64::from_bits(0x3fe6_2e42_fee0_0000),
        f64::from_bits(0x3dea_39ef_3579_3c76),
    ];
    const INV_LN2: f64 = f64::from_bits(0x3ff7_1547_652b_82fe);
    const LIMITS: [f64; 3] = [
        f64::from_bits(0x4086_2e42_fefa_39ef), // ln(MAX)
        f64::from_bits(0xc087_4910_d52d_3051),
        f64::INFINITY,
    ];

    #[inline(always)]
    fn p<V: Float<Elem = f64>>(z: V) -> V {
        // fdlibm e_exp.c.
        let [p1, p2, p3, p4, p5] = ks(
            z,
            [
                f64::from_bits(0x3fc5_5555_5555_553e),
                f64::from_bits(0xbf66_c16c_16be_bd93),
                f64::from_bits(0x3f11_566a_af25_de2c),
                f64::from_bits(0xbebb_bd41_c5d2_6bf1),
                f64::from_bits(0x3e66_3769_72be_a4d0),
            ],
        );
        z.mul_add(z.mul_add(z.mul_add(z.mul_add(p5, p4), p3), p2), p1)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{ExpOp, MathElement};
    use crate::testing::{TestFloat, assert_exact, assert_ulps_every_f32, samples};
    use std::vec::Vec;

    // Double-double arithmetic, for a reference independent of the
    // platform's `exp` (as for `cbrt`).
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }
    fn dd_mul((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
        let p = ah * bh;
        let e = ah.mul_add(bh, -p) + (ah * bl + al * bh);
        two_sum(p, e)
    }
    fn dd_add((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
        let (s, e) = two_sum(ah, bh);
        two_sum(s, e + al + bl)
    }
    fn dd_div((ah, al): (f64, f64), d: f64) -> (f64, f64) {
        let q = ah / d;
        let p = q * d;
        let e = q.mul_add(d, -p);
        two_sum(q, ((ah - p) - e + al) / d)
    }

    /// `eˣ = (hi + lo) · 2^k` to about 100 bits, `hi + lo` in
    /// `[1/√2, √2]`: `x − k·ln 2` in double-double, then a Taylor series.
    fn exp_dd(x: f64) -> (f64, f64, i32) {
        const LN2: (f64, f64) = (
            f64::from_bits(0x3fe6_2e42_fefa_39ef),
            f64::from_bits(0x3c7a_bc9e_3b39_803f),
        );
        let k = (x / LN2.0).round();
        let k_ln2 = dd_mul((k, 0.0), LN2);
        let r = dd_add((x, 0.0), (-k_ln2.0, -k_ln2.1));
        let mut s = (1.0, 0.0);
        for n in (1..=30).rev() {
            s = dd_add((1.0, 0.0), dd_div(dd_mul(s, r), f64::from(n)));
        }
        (s.0, s.1, k as i32)
    }

    /// `|got − eˣ|` in ULPs of the correctly rounded result, in the scaled
    /// domain `· 2^−k`: exact for subnormal results too.
    fn exp_error<T: TestFloat>(got: T, x: T) -> f64 {
        let x = x.to_f64();
        let (hi, lo, k) = exp_dd(x);
        let scale = |v: f64, e: i32| v * 2f64.powi(e / 2) * 2f64.powi(e - e / 2);
        let want = T::from_f64(scale(hi, k));
        if want.to_f64().is_infinite() || want.to_f64() == 0.0 {
            return if got.bits() == want.bits() {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let got = scale(got.to_f64(), -k);
        let ulp = scale(want.ulp(), -k);
        ((got - hi) - lo).abs() / ulp
    }

    fn check<T: TestFloat + MathElement>(x: &[T]) {
        for (backend, got) in T::every_backend::<ExpOp>(x) {
            for (&x, &got) in x.iter().zip(&got) {
                let e = exp_error(got, x);
                assert!(
                    e <= 1.0,
                    "exp {} {backend}: exp({x:?}) = {got:?}, {e:.3} ULP",
                    T::NAME
                );
            }
        }
    }

    /// The domain, around the overflow and underflow thresholds and the
    /// subnormal results.
    fn inputs<T: TestFloat>(limits: [f64; 2]) -> Vec<T> {
        let around = [limits[0], limits[1], limits[1] + 15.0, 1.0, 0.5];
        samples::<T>(100_000, -limits[1] + 2.0, &around)
            .into_iter()
            .filter(|v| v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn exp_within_1_ulp() {
        check::<f32>(&inputs([88.72, -103.97]));
        check::<f64>(&inputs([709.78, -745.13]));
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn exp_reference_is_precise() {
        for (x, want) in [
            (1.0, core::f64::consts::E),
            (-1.0, 0.367_879_441_171_442_33),
            (100.0, 2.688_117_141_816_135_6e43),
            (709.0, 8.218_407_461_554_972e307),
            (-700.0, 9.859_676_543_759_77e-305),
            (-740.0, 4.2e-322),
        ] {
            let (hi, lo, k) = exp_dd(x);
            let got = (hi + lo) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
            assert_eq!(got, want, "exp_dd({x})");
        }
    }

    #[test]
    fn exp_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 1.0),
            (-0.0, 1.0),
            (inf, inf),
            (-inf, 0.0),
            (nan, nan),
            (1000.0, inf),
            (-1000.0, 0.0),
        ];
        assert_exact::<f64, ExpOp>("exp", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, ExpOp>("exp", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn exp_f32_exhaustive() {
        assert_ulps_every_f32::<ExpOp>("exp", f64::exp, 1.0);
    }
}
