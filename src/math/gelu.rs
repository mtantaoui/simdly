//! `gelu x = x·Φ(x) = (x/2)·(1 + erf(x/√2))`, `Φ` the standard normal CDF,
//! in one pass, ≤ 1 ULP on every backend.
//!
//! fdlibm's rational functions for [`erf`], in
//! `a = |x|/√2`, regrouped in `x` so that `x/√2` is never rounded:
//!
//! - **`|x| < 0.84375·√2`:** `erf a = a·(1 + r)`, `r = P/Q` in
//!   `z = a² = x²/2` (exact), so `gelu x = x/2 + x²·(1 + r)/(2√2)`.
//! - **up to `1.25·√2`:** `erf a = erx + R(s)`, `s = a − 1` (to twice the
//!   precision), so `gelu(−|x|) = −|x|·((1 − erx)/2 − R/2)`.
//! - **beyond:** `erfc a = e^(−a² − 0.5625 + R/S)/a`, `R/S` in
//!   `1/a² = 2/x²`, so `gelu(−|x|) = −e^(−x²/2 − 0.5625 − ln √2 + R/S)`,
//!   `x²` exact, one [`exp_parts`].
//!
//! and `gelu x = x + gelu(−x)` for `x > 0`. For `x < 0`, `x·Φ(x)` is much
//! smaller than the terms of the rational functions (`1 + erf` cancels):
//! rounded as they stand, they cost 2.9 ULPs. So for `f64` their last
//! Horner steps carry their rounding errors (compensated Horner), the
//! quotient gets a correction step, and everything after is in two parts,
//! rounded once. `f32` runs in `f64` ([`Widen`]), where the plain
//! evaluation is far more precise than `f32` needs: its coefficients
//! rounded to `f32` could not be.
//!
//! `gelu(±0) = ±0`, `gelu(+∞) = +∞`, `gelu(−∞) = −0` (the limit; the
//! composite `x·(1 + erf)/2` gives NaN); NaN propagates.

use super::atan::{k, ks};
use super::erf::{self, poly};
use super::exp::{exp_parts, scale};
use super::float::{Element as _, Float, WideFn, Widen, two_sum};

/// A lane type with the GELU activation.
pub(crate) trait Gelu {
    fn gelu(self) -> Self;
}

/// [`Gelu`] for the lane types of one element type.
pub(crate) trait GeluKernel<V> {
    fn gelu(x: V) -> V;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Gelu for V
where
    V::Elem: GeluKernel<V>,
{
    #[inline(always)]
    fn gelu(self) -> Self {
        V::Elem::gelu(self)
    }
}

impl<V: Widen> GeluKernel<V> for f32 {
    #[inline(always)]
    fn gelu(x: V) -> V {
        x.via_f64::<GeluF32>()
    }
}

impl<V: Float<Elem = f64>> GeluKernel<V> for f64 {
    #[inline(always)]
    fn gelu(x: V) -> V {
        gelu_f64::<V, true>(x)
    }
}

/// `gelu` of an `f32` input, in `f64`.
struct GeluF32;

impl WideFn for GeluF32 {
    #[inline(always)]
    fn call<W: Float<Elem = f64>>(x: W) -> W {
        gelu_f64::<W, false>(x)
    }
}

/// `Σ c[i]·t^i` for `t = th + tl`, `c` = `a` in lanes where `m`, else `b`,
/// as `hi + lo`: Horner's scheme, its last `EXACT` steps carrying their
/// rounding errors and `tl` along (compensated Horner). The earlier steps'
/// errors reach the result scaled by powers of `t`, small enough here.
#[inline(always)]
fn horner2<V: Float, const N: usize, const EXACT: usize>(
    th: V,
    tl: V,
    m: V::Mask,
    a: &[V::Elem; N],
    b: &[V::Elem; N],
) -> (V, V) {
    let mut i = N - 1;
    let mut p = V::select(m, th.constant(a[i]), th.constant(b[i]));
    let mut e = k(th, 0.0);
    while i > 0 {
        i -= 1;
        let c = V::select(m, th.constant(a[i]), th.constant(b[i]));
        if i >= EXACT {
            p = p.mul_add(th, c);
            continue;
        }
        let ph = p * th;
        let pe = -p.residual(th, ph);
        let (s, se) = two_sum(ph, c);
        e = e.mul_add(th, p.mul_add(tl, pe + se));
        p = s;
    }
    (p, e)
}

/// `P(t)/Q(t)` as `hi + lo`, coefficients selected per lane as in
/// [`horner2`]: to about twice the precision if `PRECISE`, else rounded
/// (`lo = 0`, `tl` ignored).
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "two coefficient sets each")]
fn rational<V: Float, const N: usize, const M: usize, const PRECISE: bool, const EXACT: usize>(
    th: V,
    tl: V,
    m: V::Mask,
    pa: &[V::Elem; N],
    pb: &[V::Elem; N],
    qa: &[V::Elem; M],
    qb: &[V::Elem; M],
) -> (V, V) {
    if PRECISE {
        let (ph, pl) = horner2::<_, N, EXACT>(th, tl, m, pa, pb);
        let (qh, ql) = horner2::<_, M, EXACT>(th, tl, m, qa, qb);
        let rh = ph / qh;
        // (p − rh·q)/q, `ph − rh·qh` exact.
        (rh, (rh.residual(qh, ph) + (-rh).mul_add(ql, pl)) / qh)
    } else {
        (poly(th, m, pa, pb) / poly(th, m, qa, qb), k(th, 0.0))
    }
}

/// `gelu x` (module docs), the rational functions to about twice the
/// precision if `PRECISE`.
#[inline(always)]
fn gelu_f64<V: Float<Elem = f64>, const PRECISE: bool>(x: V) -> V {
    let [zero, half, one] = [k(x, 0.0), k(x, 0.5), k(x, 1.0)];
    let [x1, x2, x3, x_cap] = ks(x, X_CUTS);
    let t = x.abs();
    let near = t.lt(x1);
    let below_far = t.lt(x2);

    // |x| < X1 (a < 0.84375): erf a = a·(1 + r), r = P/Q in z = a² = x²/2,
    // so gelu x = x/2 + x²·c·(1 + r), c = 1/(2√2).
    let g_near = if V::any(near) {
        let q = x * x;
        let ql = -x.residual(x, q);
        let (rh, rl) =
            rational::<_, 5, 6, PRECISE, 2>(q * half, ql * half, near, &PP, &PP, &QQ, &QQ);
        let [ch, cl] = ks(x, C);
        // x²·c = a + al, a = q·ch rounded; then (a + al)·(1 + r) − a.
        let a = q * ch;
        let al = -q.residual(ch, a) + q.mul_add(cl, ql * ch);
        let b = a.mul_add(rh, a.mul_add(rl, al.mul_add(rh, al)));
        let (s, e) = two_sum(x * half, a);
        V::select(x.eq(zero), x, s + (e + b))
    } else {
        zero
    };

    // n = −gelu(−|x|), and gelu x = x − n for x > 0.
    // X1 ≤ |x| < X2: erf a = erx + R(s), s = a − 1, so
    // n = |x|·(1 − erf a)/2 = |x|·(L − R/2), L = (1 − erx)/2.
    let mid = V::select(near, zero, V::select(below_far, one, zero)).eq(one);
    let n_mid = if V::any(mid) {
        let [ih, il] = ks(x, INV_SQRT2);
        let ph = t * ih;
        let pl = -t.residual(ih, ph);
        let (sh, sl) = two_sum(ph - one, t.mul_add(il, pl));
        let (rh, rl) = rational::<_, 7, 7, PRECISE, 2>(sh, sl, mid, &PA, &PA, &QA, &QA);
        let l = x.constant(L);
        let m = t * l;
        let me = -t.residual(l, m);
        let u = t * rh;
        let ue = -t.residual(rh, u);
        let (d, de) = two_sum(m, -u * half);
        d + (de + (me - (ue + t * rl) * half))
    } else {
        zero
    };

    // |x| ≥ X2: erfc a = e^(−a² − 0.5625 + R/S)/a, R/S in s = 1/a² = 2/x²,
    // so n = (|x|/2)·erfc a = e^(−x²/2 − 0.5625 − ln √2 + R/S).
    let far = V::select(below_far, zero, one).eq(one);
    let n_far = if V::any(far) {
        let [ec_h, ec_l, cap, round, two] = ks(x, [EXP_C[0], EXP_C[1], CAP, f64::ROUND, 2.0]);
        let tc = V::select(t.gt(x_cap), x_cap, t);
        let q = tc * tc;
        let ql = -tc.residual(tc, q);
        let sh = two / q;
        let sl = (sh.residual(q, two) - sh * ql) / q;
        let lo = tc.lt(x3);
        let (rh, rl) = rational::<_, 8, 9, PRECISE, 4>(sh, sl, lo, &RA, &RB, &SA, &SB);
        // −x²/2 − EXP_C + R/S = eh + el.
        let (d, de) = two_sum(rh, -ec_h);
        let (eh, ee) = two_sum(-q * half, d);
        let el = ee + (de + (rl - ql.mul_add(half, ec_l)));
        let eh = V::select(eh.lt(cap), cap, eh);
        let (yh, yl, kk) = exp_parts(eh);
        scale(yh + yh.mul_add(el, yl), kk, half, round)
    } else {
        zero
    };

    let n = V::select(mid, n_mid, n_far);
    let g = V::select(x.lt(zero), -n, x - n);
    V::select(near, g_near, g)
}

/// `erf`'s coefficients, the denominators with their leading 1 and
/// without trailing zeros.
const PP: [f64; 5] = {
    let p = <f64 as erf::Kernel>::PP;
    [p[0], p[1], p[2], p[3], p[4]]
};
const QQ: [f64; 6] = {
    let q = <f64 as erf::Kernel>::QQ;
    [1.0, q[0], q[1], q[2], q[3], q[4]]
};
const PA: [f64; 7] = <f64 as erf::Kernel>::PA;
const QA: [f64; 7] = {
    let q = <f64 as erf::Kernel>::QA;
    [1.0, q[0], q[1], q[2], q[3], q[4], q[5]]
};
const RA: [f64; 8] = <f64 as erf::Kernel>::RA;
const RB: [f64; 8] = <f64 as erf::Kernel>::RB;
const SA: [f64; 9] = {
    let q = <f64 as erf::Kernel>::SA;
    [1.0, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]]
};
const SB: [f64; 9] = {
    let q = <f64 as erf::Kernel>::SB;
    [1.0, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]]
};

/// `[0.84375·√2, 1.25·√2, √2/0.35, 40]`: `erf`'s cuts in `a = |x|/√2`, and
/// a cap past which `gelu(−|x|)` rounds to `−0` (from about 38.6).
const X_CUTS: [f64; 4] = [
    1.193_242_693_252_299,
    1.767_766_952_966_368_9,
    4.040_610_178_208_843,
    40.0,
];
/// `1/(2√2)` as `hi + lo`.
const C: [f64; 2] = [
    f64::from_bits(0x3fd6_a09e_667f_3bcd),
    f64::from_bits(0xbc7b_dd34_13b2_6456),
];
/// `1/√2` as `hi + lo`.
const INV_SQRT2: [f64; 2] = [
    f64::from_bits(0x3fe6_a09e_667f_3bcd),
    f64::from_bits(0xbc8b_dd34_13b2_6456),
];
/// `(1 − erx)/2`, exact.
const L: f64 = 0.077_468_544_244_766_24;
/// `0.5625 + ln √2` as `hi + lo`.
const EXP_C: [f64; 2] = [
    f64::from_bits(0x3fed_1721_7f7d_1cf8),
    f64::from_bits(0xbc89_50d8_7131_9ff0),
];
/// Below `exp`'s underflow threshold; its `k` still splits into two normal
/// powers of 2.
const CAP: f64 = -753.0;

#[cfg(test)]
mod tests {
    use crate::math::{GeluOp, MathElement};
    use crate::testing::dd::gelu_dd;
    use crate::testing::{
        TestFloat, assert_exact, assert_ulps_every_f32, samples, ulp_error_scaled,
    };
    use std::vec::Vec;

    /// The worst error on every backend over `x`, and where.
    fn worst<T: TestFloat + MathElement>(x: &[T]) -> (f64, f64, &'static str) {
        let mut out = (0.0, 0.0, "");
        for (backend, got) in T::every_backend::<GeluOp>(x) {
            for (&x, &got) in x.iter().zip(&got) {
                let e = ulp_error_scaled(got, gelu_dd(x.to_f64()));
                if e > out.0 {
                    out = (e, x.to_f64(), backend);
                }
            }
        }
        out
    }

    /// Finite samples of `[−max, max]` around the cuts, where the result
    /// turns subnormal and where it rounds to 0.
    fn inputs<T: TestFloat>(n: u64, max: f64) -> Vec<T> {
        let around = [
            1.193_242_693_252_299,
            1.767_766_952_966_368_9,
            4.040_610_178_208_843,
            13.4,
            14.4,
            37.5,
            38.6,
        ];
        samples::<T>(n, max, &around)
            .into_iter()
            .filter(|v| v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn gelu_within_1_ulp() {
        let (e, x, b) = worst::<f32>(&inputs(150_000, 20.0));
        assert!(e <= 1.0, "gelu f32 {b}: gelu({x:e}), {e:.3} ULP");
        let (e, x, b) = worst::<f64>(&inputs(150_000, 50.0));
        assert!(e <= 1.0, "gelu f64 {b}: gelu({x:e}), {e:.3} ULP");
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn gelu_reference_is_precise() {
        let scale = |(h, l, k): (f64, f64, i32)| (h + l) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        for (x, want) in [
            (1e-5, 5.000_039_894_228_039_6e-6),
            (0.3, 0.185_373_426_656_685_8),
            (-0.7, -0.169_374_556_556_151_1),
            (-1.5, -0.100_210_801_903_287_1),
            (-2.5, -0.015_524_163_314_440_337),
            (-5.9, -1.072_329_639_228_662_8e-8),
            (-6.1, -3.235_088_190_398_793e-9),
            (-8.0, -4.976_768_459_417_427e-15),
            (-20.0, -5.507_248_237_212_468e-88),
            (-38.0, -1.096_462_777e-314),
            (7.0, 6.999_999_999_991_041),
        ] {
            assert_eq!(scale(gelu_dd(x)), want, "gelu_dd({x})");
        }
    }

    #[test]
    fn gelu_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (inf, inf),
            (-inf, -0.0),
            (nan, nan),
            (100.0, 100.0),
            (-100.0, -0.0),
        ];
        assert_exact::<f64, GeluOp>("gelu", &cases);
        assert_exact::<f32, GeluOp>("gelu", &cases.map(|(x, y)| (x as f32, y as f32)));
    }

    /// Every `f32`, on every backend, against `x·erfc(−x/√2)/2` in `f64`
    /// (`libm`'s `erfc`: `x/√2`'s rounding costs at most 2⁻⁴⁵ there):
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn gelu_f32_exhaustive() {
        assert_ulps_every_f32::<GeluOp>(
            "gelu",
            // `−∞·erfc(∞)` is NaN; `gelu(−∞) = −0`.
            |x| {
                if x == f64::NEG_INFINITY {
                    -0.0
                } else {
                    0.5 * x * libm::erfc(-x * core::f64::consts::FRAC_1_SQRT_2)
                }
            },
            1.0,
        );
    }

    /// Worst errors per range, densely: `cargo test --release -- --ignored
    /// gelu_measure --nocapture`.
    #[test]
    #[ignore = "slow: measurement"]
    fn gelu_measure() {
        for (lo, hi) in [
            (-1.2, 1.2),
            (-1.77, -1.19),
            (1.19, 1.77),
            (-4.05, -1.76),
            (1.76, 4.05),
            (-40.0, -4.0),
            (4.0, 40.0),
        ] {
            let n: usize = std::env::var("GELU_N").map_or(2_000_000, |v| v.parse().unwrap());
            let x32: Vec<f32> = (0..n)
                .map(|i| (lo + (hi - lo) * (i as f64 + 0.5) / n as f64) as f32)
                .collect();
            let x64: Vec<f64> = (0..n)
                .map(|i| lo + (hi - lo) * (i as f64 + 0.37) / n as f64)
                .collect();
            std::println!(
                "[{lo}, {hi}] f32 {:?} f64 {:?}",
                worst::<f32>(&x32),
                worst::<f64>(&x64)
            );
        }
    }
}
