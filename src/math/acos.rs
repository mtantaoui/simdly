//! `acos(x)`, ≤ 1 ULP on every backend: musl's `acosf.c` / `acos.c` (from
//! fdlibm), written once over [`Float`].
//!
//! `r(z) = z·p(z)/q(z)` approximates `(asin(√z)/√z − 1)/z` on `[0, 0.25]`, so
//! `asin(t) ≈ t + t·r(t²)` for `|t| ≤ 0.5`. Each range reduces to it:
//!
//! | Range       | Identity                                  |
//! |-------------|-------------------------------------------|
//! | `|x| < 0.5` | `acos(x) = π/2 − asin(x)`                  |
//! | `x ≤ −0.5`  | `acos(x) = π − 2·asin(√((1 + x)/2))`       |
//! | `x ≥ 0.5`   | `acos(x) = 2·asin(√((1 − x)/2))`           |
//!
//! π/2 is the unevaluated sum `PIO2_HI + PIO2_LO`, keeping its full
//! precision. For `x ≥ 0.5`, `s = √z` is split into `df` (its high bits,
//! whose square is exact) and `c = (z − df²)/(s + df) ≈ s − df`, so that
//! `2·(df + (s·r(z) + c))` loses nothing to cancellation.
//!
//! Every lane computes all three ranges and keeps its own. `acos(−1) = π`
//! falls out of the `x ≤ −0.5` case exactly; `acos(1)` would divide 0 by 0,
//! so it is set to `0`. Outside `[−1, 1]`, `√` of a negative gives NaN.
//!
//! Measured: `f32` on every input, `f64` on 4·10⁵ inputs (uniform, and the
//! neighbours of ±0.5 and ±1) against a 200-bit reference; at most 0.92 ULP
//! (`f32`) and 0.88 ULP (`f64`) with fused multiply-adds.

#![allow(clippy::excessive_precision, reason = "musl's constants, verbatim")]

use super::float::{Element, Float};

/// A lane type with an arc cosine.
pub(crate) trait Acos {
    fn acos(self) -> Self;
}

/// The three ranges of the module docs, with the element type's constants
/// and `r`.
///
/// Every step is an `#[inline(always)]` function or method, never a closure
/// or `impl Fn`: those are compiled without the backend's target features,
/// and unless LLVM inlines them, their intrinsics become calls.
impl<V: Float> Acos for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn acos(self) -> Self {
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

        let small = x.abs().lt(half);
        let neg = x.lt(zero);
        let z = V::select(
            small,
            x * x,
            V::select(neg, (one + x) * half, (one - x) * half),
        );
        let (rz, s) = (V::Elem::r(z), z.sqrt());

        // |x| < 0.5: π/2 − (x + x·r(x²)).
        let small_x = pio2_hi - (x - (-x).mul_add(rz, pio2_lo));
        // x ≤ −0.5: 2·(π/2 − (s + s·r(z))).
        let neg_x = two * (pio2_hi - (s + rz.mul_add(s, -pio2_lo)));
        // x ≥ 0.5: 2·(df + c + s·r(z)).
        let df = s.and_bits(V::Elem::HIGH_BITS);
        let c = (z - df * df) / (s + df);
        let pos_x = two * (df + rz.mul_add(s, c));

        let result = V::select(small, small_x, V::select(neg, neg_x, pos_x));
        V::select(x.eq(one), zero, result)
    }
}

/// musl's constants and `r` for one element type.
pub(crate) trait Kernel: Element {
    /// π/2 as `hi + lo`.
    const PIO2: [Self; 2];

    /// Keeps the high bits of `s = √z` as `df`, so that `df²` is exact.
    const HIGH_BITS: Self::Bits;

    fn from_f32(x: f32) -> Self;

    /// `z·p(z)/q(z)`.
    fn r<V: Float<Elem = Self>>(z: V) -> V;
}

impl Kernel for f32 {
    // musl acosf.c.
    const PIO2: [f32; 2] = [
        1.570_796_251_3e+00, // 0x3fc90fda
        7.549_789_415_9e-08, // 0x33a22168
    ];
    // 12 significant bits.
    const HIGH_BITS: u32 = 0xffff_f000;

    #[inline(always)]
    fn from_f32(x: f32) -> f32 {
        x
    }

    #[inline(always)]
    fn r<V: Float<Elem = f32>>(z: V) -> V {
        const P_S0: f32 = 1.666_658_669_7e-01;
        const P_S1: f32 = -4.274_342_209_1e-02;
        const P_S2: f32 = -8.656_363_003_0e-03;
        const Q_S1: f32 = -7.066_296_339_0e-01;
        let p = z * z.mul_add(
            z.mul_add(z.constant(P_S2), z.constant(P_S1)),
            z.constant(P_S0),
        );
        let q = z.mul_add(z.constant(Q_S1), z.constant(1.0));
        p / q
    }
}

impl Kernel for f64 {
    // musl acos.c. `f64` π/2 rounds down, so it is musl's high part.
    const PIO2: [f64; 2] = [
        core::f64::consts::FRAC_PI_2,
        6.123_233_995_736_766_035_87e-17, // 0x3c91a62633145c07
    ];
    // 21 significant bits.
    const HIGH_BITS: u64 = 0xffff_ffff_0000_0000;

    #[inline(always)]
    fn from_f32(x: f32) -> f64 {
        x.into()
    }

    #[inline(always)]
    fn r<V: Float<Elem = f64>>(z: V) -> V {
        const P_S0: f64 = 1.666_666_666_666_666_574_15e-01;
        const P_S1: f64 = -3.255_658_186_224_009_154_05e-01;
        const P_S2: f64 = 2.012_125_321_348_629_258_81e-01;
        const P_S3: f64 = -4.005_553_450_067_941_140_27e-02;
        const P_S4: f64 = 7.915_349_942_898_145_321_76e-04;
        const P_S5: f64 = 3.479_331_075_960_211_675_70e-05;
        const Q_S1: f64 = -2.403_394_911_734_414_218_78e+00;
        const Q_S2: f64 = 2.020_945_760_233_505_694_71e+00;
        const Q_S3: f64 = -6.882_839_716_054_532_930_30e-01;
        const Q_S4: f64 = 7.703_815_055_590_193_527_91e-02;
        let p = z.mul_add(
            z.mul_add(
                z.mul_add(
                    z.mul_add(
                        z.mul_add(z.constant(P_S5), z.constant(P_S4)),
                        z.constant(P_S3),
                    ),
                    z.constant(P_S2),
                ),
                z.constant(P_S1),
            ),
            z.constant(P_S0),
        );
        let q = z.mul_add(
            z.mul_add(
                z.mul_add(
                    z.mul_add(z.constant(Q_S4), z.constant(Q_S3)),
                    z.constant(Q_S2),
                ),
                z.constant(Q_S1),
            ),
            z.constant(1.0),
        );
        z * p / q
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{AcosOp, MathElement};
    use crate::testing::{TestFloat, ulp_error};
    use std::vec::Vec;

    /// About `2n` samples of `[−1, 1]`, evenly spaced in bits so every binade
    /// is covered, plus 2000 neighbours on each side of ±0.5 and ±1.
    fn inputs<T: TestFloat>(n: u64) -> Vec<T> {
        let one = T::from_f64(1.0).bits();
        let mut bits: Vec<u64> = (0..=one).step_by((one / n) as usize).collect();
        for c in [T::from_f64(0.5).bits(), one] {
            bits.extend((c - 2000)..=(c + 2000));
        }
        let mut xs: Vec<T> = bits
            .iter()
            .flat_map(|&b| [b, b | T::SIGN])
            .map(T::from_bits64)
            .collect();
        xs.extend(T::SPECIAL);
        xs.extend([1.5, -1.5, 1.0, -1.0].map(T::from_f64));
        xs
    }

    /// Every backend within 1 ULP of the reference: `f64`'s `acos` for
    /// `f32` (exact enough to measure the real error), `std`'s for `f64`.
    fn check<T: TestFloat + MathElement>(x: &[T]) {
        for (backend, got) in T::every_backend::<AcosOp>(x) {
            for (&x, &got) in x.iter().zip(&got) {
                let e = ulp_error(got, x.to_f64().acos());
                assert!(
                    e <= 1.0,
                    "acos {} {backend}: acos({x:?}) = {got:?}, {e:.3} ULP",
                    T::NAME
                );
            }
        }
    }

    #[test]
    fn acos_within_1_ulp() {
        check::<f32>(&inputs(200_000));
        check::<f64>(&inputs(200_000));
    }

    #[test]
    fn acos_special_values() {
        fn special<T: TestFloat + MathElement>(pi: T, pi_2: T) {
            let x = [
                1.0,
                -1.0,
                0.0,
                -0.0,
                1.0000001,
                -1.0000001,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ]
            .map(T::from_f64);
            let want = [T::from_f64(0.0), pi, pi_2, pi_2];
            for (backend, got) in T::every_backend::<AcosOp>(&x) {
                let what = std::format!("acos {} {backend}", T::NAME);
                for i in 0..4 {
                    assert_eq!(
                        got[i].bits(),
                        want[i].bits(),
                        "{what}: acos({:?}) = {:?}",
                        x[i],
                        got[i]
                    );
                }
                assert!(
                    got[4..].iter().all(|v| v.is_nan()),
                    "{what}: {got:?} outside [-1, 1]"
                );
            }
        }
        special::<f32>(core::f32::consts::PI, core::f32::consts::FRAC_PI_2);
        special::<f64>(core::f64::consts::PI, core::f64::consts::FRAC_PI_2);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn acos_f32_exhaustive() {
        const CHUNK: u64 = 1 << 22;
        for start in (0..1_u64 << 32).step_by(CHUNK as usize) {
            let x: Vec<f32> = (start..start + CHUNK)
                .map(|b| f32::from_bits(b as u32))
                .collect();
            check(&x);
        }
    }
}
