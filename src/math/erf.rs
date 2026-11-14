//! `erf x`, the error function, ≤ 1 ULP on every backend: fdlibm's
//! algorithm (musl's `erf.c`), written once over [`Float`], `f32` with the
//! same coefficients rounded (as musl's `erff.c`).
//!
//! With `a = |x|`:
//!
//! - **`a < 0.84375`:** `erf x = x + x·P(x²)/Q(x²)`, `|P/Q − (erf(x) −
//!   x)/x| ≤ 2⁻⁵⁷·⁹⁰`. Evaluated as it stands, that reaches 0.99 ULP
//!   (`f64`, scalar): `P/Q ≈ 0.128` rounds just above 0.125, where its ULP
//!   is largest relative to it, and so do `Q ≈ 1`, the quotient and the
//!   unfused `x·P/Q`. So below `LOW = 0.40625`, `P/Q = pp0 + z·P′/Q`
//!   exactly and `erf x = x·(2/√π) + x·z·P′/Q`: `x·(2/√π)` is exact
//!   (`residual`, `2/√π` as two floats), only the small term rounds before
//!   the sum. Tiny `x` is scaled by `2^K` into the normal range and back,
//!   so subnormal results round once. From `LOW` on, where `z·P′/Q` is no
//!   longer small, `x + x·P/Q` itself, the product exact and the sum
//!   rounded once ([`two_sum`]), fused or not.
//! - **`a < 1.25`:** `erf a = erx + P(s)/Q(s)`, `s = a − 1` (exact), `erx`
//!   the leading bits of `erf(1)`, `|P/Q − (erf(a) − erx)| ≤ 2⁻⁵⁹·⁰⁶`.
//! - **`a < 6`:** `erf a = 1 − erfc a`, `erfc a = e^(−a² − 0.5625 + R/S)/a`,
//!   `R/S` a rational function of `1/a²` on `[1.25, 1/0.35]` and above, to
//!   2⁻⁶²·⁵⁷ and 2⁻⁶¹·⁵². fdlibm splits `a²` with a truncated `a` so its
//!   two `exp`s get exact arguments; here `a² = sh + sl` exactly
//!   (`residual`), the exponent is a two-part sum ([`two_sum`]), and one
//!   [`exp_parts`] takes its high part, the low part multiplying in.
//! - **`a ≥ 6`:** `±1`.
//!
//! The rational functions below 1.25, and those above, share one
//! evaluation each, their coefficients selected per lane, and only when
//! some lane needs a second set; the `erfc` path runs only when some lane
//! needs it ([`Float::any`]), as does the other.
//!
//! `erf(±0) = ±0`, `erf(±∞) = ±1`; NaN propagates.

#![allow(clippy::excessive_precision, reason = "fdlibm's constants, verbatim")]

use super::atan::{k, ks};
use super::exp::{self, exp_parts};
use super::float::{Float, two_sum};

/// A lane type with the error function.
pub(crate) trait Erf {
    fn erf(self) -> Self;
}

/// `Σ c[i]·t^i` (Horner), `c` = `a` in lanes where `m`, else `b`.
#[inline(always)]
fn poly<V: Float, const N: usize>(t: V, m: V::Mask, a: &[V::Elem; N], b: &[V::Elem; N]) -> V {
    let mut i = N - 1;
    let mut p = V::select(m, t.constant(a[i]), t.constant(b[i]));
    while i > 0 {
        i -= 1;
        p = p.mul_add(t, V::select(m, t.constant(a[i]), t.constant(b[i])));
    }
    p
}

/// `Σ c[i]·t^i` (Horner), the same `c` in every lane.
#[inline(always)]
fn horner<V: Float, const N: usize>(t: V, c: &[V::Elem; N]) -> V {
    let mut i = N - 1;
    let mut p = t.constant(c[i]);
    while i > 0 {
        i -= 1;
        p = p.mul_add(t, t.constant(c[i]));
    }
    p
}

/// [`poly`] with three coefficient sets: `a` where `m1` and `m2`, `b`
/// where `m1` only, else `c`.
#[inline(always)]
fn poly3<V: Float, const N: usize>(
    t: V,
    m1: V::Mask,
    m2: V::Mask,
    [a, b, c]: [&[V::Elem; N]; 3],
) -> V {
    let mut i = N - 1;
    let ab = V::select(m2, t.constant(a[i]), t.constant(b[i]));
    let mut p = V::select(m1, ab, t.constant(c[i]));
    while i > 0 {
        i -= 1;
        let ab = V::select(m2, t.constant(a[i]), t.constant(b[i]));
        p = p.mul_add(t, V::select(m1, ab, t.constant(c[i])));
    }
    p
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Erf for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn erf(self) -> Self {
        let x = self;
        let [zero, one] = [k(x, 0.0), k(x, 1.0)];
        let [low_cut, small_cut, near_cut, mid_cut, six] = ks(x, V::Elem::CUTS);
        let a = x.abs();
        let near = a.lt(near_cut);
        let neg = x.lt(zero);

        // a < 1.25: from x + x·P/Q in z = x² below 0.84375, erx + P/Q in
        // a − 1 above.
        let y_near = if V::any(near) {
            let small = a.lt(small_cut);
            let low = a.lt(low_cut);
            let tiny = a.lt(x.constant(V::Elem::TINY));
            // z for tiny x is far below what P/Q resolves; 0 keeps it from
            // turning subnormal.
            let z = V::select(tiny, zero, x * x);
            let t = V::select(small, z, a - one);

            // r = P′/Q below LOW, P/Q up to 0.84375 (both in z), P/Q in
            // a − 1 above; the last set only when some lane needs it.
            let both = V::select(small, zero, V::select(near, one, zero)).eq(one);
            let (p, q) = if V::any(both) {
                let p = poly3(t, small, low, [&V::Elem::PZ, &V::Elem::PP, &V::Elem::PA]);
                (p, one + t * poly(t, small, &V::Elem::QQ, &V::Elem::QA))
            } else {
                let p = poly(t, low, &V::Elem::PZ, &V::Elem::PP);
                (p, one + t * horner(t, &V::Elem::QQ))
            };
            let r = p / q;

            // a < LOW: P/Q = pp0 + z·P′/Q exactly, and erf x = x·(2/√π) +
            // x·z·P′/Q, `2/√π = ch + cl`, `x·ch` exact (`residual`): only
            // the small terms round before the sum. Tiny x is scaled by
            // 2^K first, so that product stays exact, and back last, so a
            // subnormal result rounds once.
            let [ch, cl] = ks(x, V::Elem::TWO_OVER_SQRT_PI);
            let [up, down] = ks(x, V::Elem::SCALE);
            let xs = V::select(tiny, x * up, x);
            let ph = xs * ch;
            let pl = -xs.residual(ch, ph);
            let y_low = (ph + (pl + xs * t.mul_add(r, cl))) * V::select(tiny, down, one);

            // LOW ≤ a < 0.84375: x + x·P/Q, the product exact and the sum
            // rounded once, fused or not.
            let pr = x * r;
            let (s, e) = two_sum(x, pr);
            let y_mid = s + (e - x.residual(r, pr));

            let y = x.constant(V::Elem::ERX) + r;
            // ±0 is its own result: the scalar `residual` loses its sign.
            let y_small = V::select(x.eq(zero), x, V::select(low, y_low, y_mid));
            V::select(small, y_small, V::select(neg, -y, y))
        } else {
            zero
        };

        // a ≥ 1.25 (or NaN): 1 − e^(−a² − 0.5625 + R/S)/a, a capped at 6.
        let far = V::select(near, zero, one).eq(one);
        let y_far = if V::any(far) {
            let a6 = V::select(a.gt(six), six, a);
            let inv = one / a6;
            let s = inv * inv;
            let mid = a6.lt(mid_cut);
            let r = poly(s, mid, &V::Elem::RA, &V::Elem::RB);
            let q = one + s * poly(s, mid, &V::Elem::SA, &V::Elem::SB);

            // −a² − 0.5625 + R/S = eh + el, a² = sh + sl exactly.
            let sh = a6 * a6;
            let sl = -a6.residual(a6, sh);
            let (eh, ee) = two_sum(-sh, r / q - k(x, 0.5625));
            let el = ee - sl;

            // e^(eh + el) = (yh + yl)·(1 + el)·2^k; 2^k ≥ 2⁻⁵³, normal.
            let (yh, yl, kk) = exp_parts(eh);
            let erfc = ((yh + yh.mul_add(el, yl)) * inv).ldexp(kk);
            let y = one - erfc;
            V::select(neg, -y, y)
        } else {
            zero
        };

        V::select(near, y_near, y_far)
    }
}

/// musl's `erf.c` coefficients, padded with zeros to a common degree per
/// pair: `PP`, `QQ` for `a < 0.84375`; `PA`, `QA` up to 1.25; `RA`, `SA`
/// up to 1/0.35; `RB`, `SB` above. `QQ` … `SB` omit the leading 1. `PZ`:
/// `PP/QQ = pp0 + z·PZ/QQ`.
pub(crate) trait Kernel: exp::Kernel {
    /// `[LOW, 0.84375, 1.25, 1/0.35, 6]`: below `LOW = 0.40625`, the
    /// `PZ` form is the more accurate, above it `PP`'s.
    const CUTS: [Self; 5];
    /// 2⁻²⁸: below, `x²` is replaced by 0.
    const TINY: Self;
    /// The leading bits of `erf(1)`.
    const ERX: Self;
    /// `2/√π` as `hi + lo`.
    const TWO_OVER_SQRT_PI: [Self; 2];
    /// `[2^K, 2^−K]`: tiny `x` is scaled into the normal range and back.
    const SCALE: [Self; 2];
    const PP: [Self; 7];
    /// `PZᵢ = ppᵢ₊₁ − pp0·qqᵢ₊₁`: `P − pp0·Q = z·PZ`.
    const PZ: [Self; 7];
    const QQ: [Self; 6];
    const PA: [Self; 7];
    const QA: [Self; 6];
    const RA: [Self; 8];
    const SA: [Self; 8];
    const RB: [Self; 8];
    const SB: [Self; 8];
}

const PP: [f64; 7] = [
    1.28379167095512558561e-01,
    -3.25042107247001499370e-01,
    -2.84817495755985104766e-02,
    -5.77027029648944159157e-03,
    -2.37630166566501626084e-05,
    0.0,
    0.0,
];
const QQ: [f64; 6] = [
    3.97917223959155352819e-01,
    6.50222499887672944485e-02,
    5.08130628187576562776e-03,
    1.32494738004321644526e-04,
    -3.96022827877536812320e-06,
    0.0,
];
const PA: [f64; 7] = [
    -2.36211856075265944077e-03,
    4.14856118683748331666e-01,
    -3.72207876035701323847e-01,
    3.18346619901161753674e-01,
    -1.10894694282396677476e-01,
    3.54783043256182359371e-02,
    -2.16637559486879084300e-03,
];
const QA: [f64; 6] = [
    1.06420880400844228286e-01,
    5.40397917702171048937e-01,
    7.18286544141962662868e-02,
    1.26171219808761642112e-01,
    1.36370839120290507362e-02,
    1.19844998467991074170e-02,
];
const RA: [f64; 8] = [
    -9.86494403484714822705e-03,
    -6.93858572707181764372e-01,
    -1.05586262253232909814e+01,
    -6.23753324503260060396e+01,
    -1.62396669462573470355e+02,
    -1.84605092906711035994e+02,
    -8.12874355063065934246e+01,
    -9.81432934416914548592e+00,
];
const SA: [f64; 8] = [
    1.96512716674392571292e+01,
    1.37657754143519042600e+02,
    4.34565877475229228821e+02,
    6.45387271733267880336e+02,
    4.29008140027567833386e+02,
    1.08635005541779435134e+02,
    6.57024977031928170135e+00,
    -6.04244152148580987438e-02,
];
const RB: [f64; 8] = [
    -9.86494292470009928597e-03,
    -7.99283237680523006574e-01,
    -1.77579549177547519889e+01,
    -1.60636384855821916062e+02,
    -6.37566443368389627722e+02,
    -1.02509513161107724954e+03,
    -4.83519191608651397019e+02,
    0.0,
];
const SB: [f64; 8] = [
    3.03380607434824582924e+01,
    3.25792512996573918826e+02,
    1.53672958608443695994e+03,
    3.19985821950859553908e+03,
    2.55305040643316442583e+03,
    4.74528541206955367215e+02,
    -2.24409524465858183362e+01,
    0.0,
];
/// `PZᵢ = ppᵢ₊₁ − pp0·qqᵢ₊₁` (`pp5 = 0`), so `PP − pp0·(1 + z·QQ) =
/// z·PZ`.
const PZ: [f64; 7] = {
    let mut out = [0.0; 7];
    let mut i = 0;
    while i < 5 {
        out[i] = PP[i + 1] - PP[0] * QQ[i];
        i += 1;
    }
    out
};
const TWO_OVER_SQRT_PI: [f64; 2] = [
    f64::from_bits(0x3ff2_0dd7_5042_9b6d),
    f64::from_bits(0x3c71_ae3a_914f_ed80),
];
const CUTS: [f64; 5] = [0.40625, 0.84375, 1.25, 1.0 / 0.35, 6.0];
const ERX: f64 = 8.45062911510467529297e-01;
const TINY: f64 = 1.0 / 268_435_456.0; // 2^−28

/// `a` rounded to `f32`, as musl's `erff.c`.
const fn narrow<const N: usize>(a: [f64; N]) -> [f32; N] {
    let mut out = [0.0; N];
    let mut i = 0;
    while i < N {
        out[i] = a[i] as f32;
        i += 1;
    }
    out
}

impl Kernel for f64 {
    const CUTS: [f64; 5] = CUTS;
    const TINY: f64 = TINY;
    const ERX: f64 = ERX;
    const TWO_OVER_SQRT_PI: [f64; 2] = TWO_OVER_SQRT_PI;
    const SCALE: [f64; 2] = [f64::from_bits(0x47f0 << 48), f64::from_bits(0x37f0 << 48)]; // 2^±128
    const PP: [f64; 7] = PP;
    const PZ: [f64; 7] = PZ;
    const QQ: [f64; 6] = QQ;
    const PA: [f64; 7] = PA;
    const QA: [f64; 6] = QA;
    const RA: [f64; 8] = RA;
    const SA: [f64; 8] = SA;
    const RB: [f64; 8] = RB;
    const SB: [f64; 8] = SB;
}

impl Kernel for f32 {
    const CUTS: [f32; 5] = narrow(CUTS);
    const TINY: f32 = TINY as f32;
    // Exact: erf.c's `erx` has 23 significant bits.
    const ERX: f32 = ERX as f32;
    const TWO_OVER_SQRT_PI: [f32; 2] = [
        TWO_OVER_SQRT_PI[0] as f32,
        (TWO_OVER_SQRT_PI[0] - (TWO_OVER_SQRT_PI[0] as f32) as f64 + TWO_OVER_SQRT_PI[1]) as f32,
    ];
    const SCALE: [f32; 2] = [f32::from_bits(191 << 23), f32::from_bits(63 << 23)]; // 2^±64
    const PP: [f32; 7] = narrow(PP);
    const PZ: [f32; 7] = narrow(PZ);
    const QQ: [f32; 6] = narrow(QQ);
    const PA: [f32; 7] = narrow(PA);
    const QA: [f32; 6] = narrow(QA);
    const RA: [f32; 8] = narrow(RA);
    const SA: [f32; 8] = narrow(SA);
    const RB: [f32; 8] = narrow(RB);
    const SB: [f32; 8] = narrow(SB);
}

#[cfg(test)]
mod tests {
    use crate::math::{ErfOp, MathElement};
    use crate::testing::dd::erf_dd;
    use crate::testing::{
        TestFloat, assert_exact, assert_ulps_every_f32, samples, ulp_error_scaled,
    };
    use std::vec::Vec;

    /// Within 1 ULP on every backend, for every binade to 8, subnormals
    /// included, around each range's ends (`LOW` = 0.40625 too), the
    /// `TINY` cut-off, and where `erf` rounds to 1 (3.92 for `f32`, 5.92
    /// for `f64`).
    fn check<T: TestFloat + MathElement>() {
        let around = [
            0.406_25,
            0.843_75,
            1.25,
            1.0 / 0.35,
            6.0,
            3.92,
            5.92,
            3.725e-9,
        ];
        let x: Vec<T> = samples::<T>(150_000, 8.0, &around)
            .into_iter()
            .filter(|v| v.to_f64().is_finite())
            .collect();
        for (backend, got) in T::every_backend::<ErfOp>(&x) {
            for (&x, &got) in x.iter().zip(&got) {
                let e = ulp_error_scaled(got, erf_dd(x.to_f64()));
                assert!(
                    e <= 1.0,
                    "erf {} {backend}: erf({x:?}) = {got:?}, {e:.3} ULP",
                    T::NAME
                );
            }
        }
    }

    #[test]
    fn erf_within_1_ulp() {
        check::<f32>();
        check::<f64>();
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn erf_reference_is_precise() {
        for (x, want) in [
            (1e-10, 1.128_379_167_095_512_6e-10),
            (0.5, 0.520_499_877_813_046_5),
            (-0.9, -0.796_908_212_422_832_2),
            (1.25, 0.922_900_128_256_458_3),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_6),
            (5.9, 0.999_999_999_999_999_9),
        ] {
            let (hi, lo, _) = erf_dd(x);
            assert_eq!(hi + lo, want, "erf_dd({x:e})");
        }
    }

    #[test]
    fn erf_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (inf, 1.0),
            (-inf, -1.0),
            (nan, nan),
            (10.0, 1.0),
            (-10.0, -1.0),
            (-1e-310, -1.128_379_167_095_5e-310),
        ];
        assert_exact::<f64, ErfOp>("erf", &cases);
        assert_exact::<f32, ErfOp>("erf", &cases.map(|(x, y)| (x as f32, y as f32)));
    }

    /// Every `f32`, on every backend, against `libm`'s `f64` `erf` (`std`
    /// has none): `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn erf_f32_exhaustive() {
        assert_ulps_every_f32::<ErfOp>("erf", libm::erf, 1.0);
    }
}
