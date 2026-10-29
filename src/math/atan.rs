//! `atan(x)`, ≤ 1 ULP on every backend: musl's `atanf.c` / `atan.c` (from
//! fdlibm), written once over [`Float`].
//!
//! `atan` is odd, so `a = |x|` is reduced to `|t| ≤ 7/16` with the addition
//! formula `atan(a) = atan(c) + atan((a − c)/(1 + c·a))`, for a `c` per range:
//!
//! | Range              | `c`   | `t`                   | `p`, `q` |
//! |--------------------|-------|-----------------------|----------|
//! | `a < 7/16`         | –     | `x` (signed)          | 1, 0     |
//! | `7/16 ≤ a < 11/16` | 0.5   | `(2a − 1)/(2 + a)`    | 2, 1     |
//! | `11/16 ≤ a < 19/16`| 1     | `(a − 1)/(a + 1)`     | 1, 1     |
//! | `19/16 ≤ a < 39/16`| 1.5   | `(a − 1.5)/(1 + 1.5a)`| 1, 1.5   |
//! | `a ≥ 39/16`        | ∞     | `−1/a`                | 0, 1     |
//!
//! All five are `t = (p·a − q)/(p + q·a)`, so a register selects `p` and `q`
//! per lane and divides once. Then, with `atan(c) = hi + lo`,
//!
//! `atan(a) = hi − ((t·S(t²) − lo) − t)`, `S(z) ≈ (t − atan(t))/t`,
//!
//! an odd polynomial in `t` split into even and odd powers of `z²`.
//!
//! The unified reduction is as accurate as musl's five: measured at most
//! 0.852 ULP (`f32`, every input) and 0.847 ULP (`f64`, against a 200-bit
//! reference), fused or not.
//!
//! `±0` keeps its sign, `atan(±∞)` is `±π/2` (`0·∞` would make `t` NaN), and
//! NaN propagates.

#![allow(clippy::excessive_precision, reason = "musl's constants, verbatim")]

use super::float::{Element, Float};

/// A lane type with an arc tangent.
pub(crate) trait Atan {
    fn atan(self) -> Self;
}

/// `c` in every lane of `x`'s type.
#[inline(always)]
pub(super) fn k<V: Float>(x: V, c: f32) -> V {
    x.constant(V::Elem::from_f32(c))
}

/// Every `c[i]` in every lane of `x`'s type: a loop, not a closure.
#[inline(always)]
pub(super) fn ks<V: Float, const N: usize>(x: V, c: [V::Elem; N]) -> [V; N] {
    let mut out = [x; N];
    let mut i = 0;
    while i < N {
        out[i] = x.constant(c[i]);
        i += 1;
    }
    out
}

/// Per lane, the entry of `c` for the first `m[i]` that is true, else the
/// last one.
#[inline(always)]
pub(super) fn pick<V: Float>(m: [V::Mask; 4], c: [V; 5]) -> V {
    V::select(
        m[0],
        c[0],
        V::select(
            m[1],
            c[1],
            V::select(m[2], c[2], V::select(m[3], c[3], c[4])),
        ),
    )
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Atan for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn atan(self) -> Self {
        let x = self;
        let a = x.abs();
        let m = [
            a.lt(k(x, 7.0 / 16.0)),
            a.lt(k(x, 11.0 / 16.0)),
            a.lt(k(x, 19.0 / 16.0)),
            a.lt(k(x, 39.0 / 16.0)),
        ];
        let p = pick(m, ks(x, [1.0, 2.0, 1.0, 1.0, 0.0].map(V::Elem::from_f32)));
        let q = pick(m, ks(x, [0.0, 1.0, 1.0, 1.5, 1.0].map(V::Elem::from_f32)));
        let [h0, h1, h2, h3] = ks(x, V::Elem::HI);
        let [l0, l1, l2, l3] = ks(x, V::Elem::LO);
        let zero = k(x, 0.0);
        let hi = pick(m, [zero, h0, h1, h2, h3]);
        let lo = pick(m, [zero, l0, l1, l2, l3]);

        // The first range keeps the sign; the others work on |x|.
        let v = V::select(m[0], x, a);
        let t = p.mul_add(v, -q) / q.mul_add(v, p);
        let s = V::Elem::s(t * t);
        let r = hi - (t.mul_add(s, -lo) - t);

        let r = V::select(a.eq(k(x, f32::INFINITY)), h3 + l3, r);
        let r = V::select(m[0], r, V::select(x.lt(zero), -r, r));
        V::select(x.eq(zero), x, r)
    }
}

/// musl's constants and polynomial for one element type.
pub(crate) trait Kernel: Element {
    /// The high parts of `atan(0.5)`, `atan(1)`, `atan(1.5)`, `atan(∞)`.
    const HI: [Self; 4];
    /// Their low parts.
    const LO: [Self; 4];

    /// `S(z)`: `atan(t) ≈ t − t·S(t²)` for `|t| ≤ 7/16`.
    fn s<V: Float<Elem = Self>>(z: V) -> V;
}

impl Kernel for f32 {
    // musl atanf.c.
    const HI: [f32; 4] = [
        4.636_476_039_9e-01, // 0x3eed6338
        7.853_981_256_5e-01, // 0x3f490fda
        9.827_936_887_7e-01, // 0x3f7b985e
        1.570_796_251_3e+00, // 0x3fc90fda
    ];
    const LO: [f32; 4] = [
        5.012_158_244_0e-09, // 0x31ac3769
        3.774_894_707_9e-08, // 0x33222168
        3.447_321_717_0e-08, // 0x33140fb4
        7.549_789_415_9e-08, // 0x33a22168
    ];

    #[inline(always)]
    fn s<V: Float<Elem = f32>>(z: V) -> V {
        const AT: [f32; 5] = [
            3.333_332_836_6e-01,
            -1.999_915_838_2e-01,
            1.425_363_570_5e-01,
            -1.064_801_737_7e-01,
            6.168_760_731_8e-02,
        ];
        let w = z * z;
        let [a0, a1, a2, a3, a4] = ks(z, AT);
        let s1 = z * w.mul_add(w.mul_add(a4, a2), a0);
        let s2 = w * w.mul_add(a3, a1);
        s1 + s2
    }
}

impl Kernel for f64 {
    // musl atan.c.
    const HI: [f64; 4] = [
        4.636_476_090_008_060_935_15e-01, // 0x3fddac670561bb4f
        core::f64::consts::FRAC_PI_4,     // 0x3fe921fb54442d18
        9.827_937_232_473_290_540_82e-01, // 0x3fef730bd281f69b
        core::f64::consts::FRAC_PI_2,     // 0x3ff921fb54442d18
    ];
    const LO: [f64; 4] = [
        2.269_877_745_296_168_709_24e-17, // 0x3c7a2b7f222f65e2
        3.061_616_997_868_383_017_93e-17, // 0x3c81a62633145c07
        1.390_331_103_123_099_845_16e-17, // 0x3c7007887af0cbbd
        6.123_233_995_736_766_035_87e-17, // 0x3c91a62633145c07
    ];

    #[inline(always)]
    fn s<V: Float<Elem = f64>>(z: V) -> V {
        const AT: [f64; 11] = [
            3.333_333_333_333_293_180_27e-01,
            -1.999_999_999_987_648_324_76e-01,
            1.428_571_427_250_346_637_11e-01,
            -1.111_111_040_546_235_578_80e-01,
            9.090_887_133_436_506_561_96e-02,
            -7.691_876_205_044_829_994_95e-02,
            6.661_073_137_387_531_206_69e-02,
            -5.833_570_133_790_573_486_45e-02,
            4.976_877_994_615_932_360_17e-02,
            -3.653_157_274_421_691_552_70e-02,
            1.628_582_011_536_578_236_23e-02,
        ];
        let w = z * z;
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] = ks(z, AT);
        let s1 = z * w.mul_add(
            w.mul_add(w.mul_add(w.mul_add(w.mul_add(a10, a8), a6), a4), a2),
            a0,
        );
        let s2 = w * w.mul_add(w.mul_add(w.mul_add(w.mul_add(a9, a7), a5), a3), a1);
        s1 + s2
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{AtanOp, MathElement};
    use crate::testing::{
        TestFloat, assert_exact, assert_same, assert_ulps, assert_ulps_every_f32, samples,
    };
    use std::vec::Vec;

    /// Every binade up to the largest finite value, and the range boundaries.
    fn inputs<T: TestFloat>() -> Vec<T> {
        let max = if T::NAME == "f32" {
            f32::MAX.into()
        } else {
            f64::MAX
        };
        samples(
            200_000,
            max,
            &[7.0 / 16.0, 11.0 / 16.0, 19.0 / 16.0, 39.0 / 16.0],
        )
    }

    #[test]
    fn atan_within_1_ulp() {
        assert_ulps::<f32, AtanOp>("atan", &inputs(), f64::atan, 1.0);
        assert_ulps::<f64, AtanOp>("atan", &inputs(), f64::atan, 1.0);
    }

    #[test]
    fn atan_special_values() {
        use core::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let nan = f64::NAN;
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (1.0, FRAC_PI_4),
            (-1.0, -FRAC_PI_4),
            (f64::INFINITY, FRAC_PI_2),
            (f64::NEG_INFINITY, -FRAC_PI_2),
            (nan, nan),
        ];
        assert_exact::<f64, AtanOp>("atan", &cases);
        let cases32: Vec<(f32, f32)> = cases.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
        assert_eq!(
            (cases32[2].1, cases32[4].1),
            (core::f32::consts::FRAC_PI_4, core::f32::consts::FRAC_PI_2)
        );
        assert_exact::<f32, AtanOp>("atan", &cases32);
    }

    /// `atan(−x) = −atan(x)`, bit for bit, on every backend.
    #[test]
    fn atan_is_odd() {
        fn check<T: TestFloat + MathElement>() {
            let x: Vec<T> = inputs();
            let neg: Vec<T> = x
                .iter()
                .map(|v| T::from_bits64(v.bits() ^ T::SIGN))
                .collect();
            let pos = T::every_backend::<AtanOp>(&x);
            for ((backend, pos), (_, negs)) in pos.into_iter().zip(T::every_backend::<AtanOp>(&neg))
            {
                let flipped: Vec<T> = pos
                    .iter()
                    .map(|v| T::from_bits64(v.bits() ^ T::SIGN))
                    .collect();
                assert_same(
                    &negs,
                    &flipped,
                    &std::format!("atan odd {} {backend}", T::NAME),
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
    fn atan_f32_exhaustive() {
        assert_ulps_every_f32::<AtanOp>("atan", f64::atan, 1.0);
    }
}
