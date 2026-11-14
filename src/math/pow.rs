//! `pow(x, y) = xʸ`, ≤ 1 ULP on every backend: `e^(y·ln x)` with `ln x` to
//! about twice the element type's precision, written once over [`Float`],
//! `f32` kept in `f32`.
//!
//! An error `δ` in `y·ln x` becomes a relative error `δ` in the result, and
//! `|y·ln x|` reaches 745 (≈ 2⁹·⁵) for finite `f64` results, 104 for `f32`:
//! `ln x` must be good to about 2⁻⁶⁶ (`f64`) or 2⁻³⁵ (`f32`), past the
//! element type. Hence:
//!
//! 1. **`ln |x| = lh + ll`** ([`ln_parts`]): `|x| = m·2^n` as in `ln`,
//!    `f = m − 1` exact, `s = f/(2 + f)` as two floats (`2 + f` exact,
//!    one correction step on the quotient), and
//!    `ln(1 + f) = 2·atanh s = 2s·(1 + z/3 + z²/5 + z³·W(z))`, `z = s²`,
//!    `|s| ≤ 0.1716`. `W`, the Taylor series to `z¹²/25` (`f64`) or `z⁷/15`
//!    (`f32`), is evaluated in the element type; everything above it in
//!    double-word arithmetic. Then `n·ln 2` with `ln 2` in two parts, the
//!    first with trailing zeros so `n·ln2_hi` is exact.
//! 2. **`y·ln|x| = ph + pl`**, the product exact (`residual`). `y` itself
//!    may come in two parts: `powi`'s integer exponent, which `f32` may not
//!    hold, is `yh + yl`.
//! 3. **`e^(ph + pl) = (eh + el)·(1 + pl)·2^k`** from [`exp_parts`], scaled
//!    last in two normal factors as in `exp`, so results overflow, underflow
//!    and turn subnormal by themselves, rounding once.
//!
//! No tables: SIMD lanes would need gathers, which NEON lacks. C99's
//! special cases (`pow(x, ±0) = 1`, `pow(1, y) = 1`, negative `x` with
//! integer `y`, `±0` and `±∞`) are selected at the end.

use super::atan::{k, ks};
use super::exp::{self, exp_parts, scale};
use super::float::{Element, Float};
use super::ln;

/// A lane type with `xʸ`, `y = yh + yl`: `yl` carries the bits of an
/// integer exponent that the element type cannot (`powi`); it is 0
/// otherwise.
pub(crate) trait Pow {
    fn pow(self, yh: Self, yl: Self) -> Self;

    /// `self^N` for a small integer `N` ([`powi_small`]).
    fn powi_small<const N: i32>(self) -> Self;
}

/// `a + b = s + e` exactly, for `|a| ≥ |b|` or `a = 0` (Fast2Sum).
#[inline(always)]
fn fast_sum<V: Float>(a: V, b: V) -> (V, V) {
    let s = a + b;
    (s, b - (s - a))
}

/// `a·b` for two-part `a` and `b`, to about twice the element type's
/// precision; not renormalised.
#[inline(always)]
fn mul22<V: Float>((ah, al): (V, V), (bh, bl): (V, V)) -> (V, V) {
    let p = ah * bh;
    (p, ah.mul_add(bl, al * bh) - ah.residual(bh, p))
}

/// `ln a = lh + ll` to about twice the element type's precision, `|ll| ≤
/// ½ ULP of lh`, for positive finite `a`, subnormals included; anything
/// else gives garbage.
#[inline(always)]
fn ln_parts<V: Float>(a: V) -> (V, V)
where
    V::Elem: Kernel,
{
    let [zero, half, one, two] = [k(a, 0.0), k(a, 0.5), k(a, 1.0), k(a, 2.0)];
    let [tiny_scale, tiny_k] = ks(a, <V::Elem as ln::Kernel>::TINY);
    let sqrt2 = a.constant(<V::Elem as ln::Kernel>::SQRT2);
    let [ln2_hi, ln2_lo] = ks(a, <V::Elem as exp::Kernel>::LN2);
    let [c3h, c3l, c5h, c5l] = ks(a, V::Elem::THIRD_FIFTH);

    // a = m·2^n, m ∈ [√2/2, √2), subnormals scaled first.
    let tiny = a.lt(a.constant(V::Elem::MIN_POSITIVE));
    let (m, e) = V::select(tiny, a * tiny_scale, a).frexp();
    let big = m.ge(sqrt2);
    let m = V::select(big, m * half, m);
    let n = e + V::select(big, one, zero) + V::select(tiny, tiny_k, zero);

    // s = f/(2 + f) = sh + sl: 2 + f = dh + dl exactly, then one
    // correction step with the exact residual f − sh·dh.
    let f = m - one;
    let (dh, dl) = fast_sum(two, f);
    let inv = one / dh;
    let sh = f * inv;
    let sl = (sh.residual(dh, f) - sh * dl) * inv;

    // z = s² = zh + zl; W(z) in the element type.
    let zh = sh * sh;
    let zl = (two * sh).mul_add(sl, -sh.residual(sh, zh));
    let w = V::Elem::series(zh);

    // V = 1/3 + z·(1/5 + z·W), U = z·V, in two parts.
    let (ih, ie) = fast_sum(c5h, zh * w);
    let (ph, pl) = mul22((zh, zl), (ih, ie + c5l));
    let (vh, ve) = fast_sum(c3h, ph);
    let (uh, ul) = mul22((zh, zl), (vh, ve + (pl + c3l)));

    // ln(1 + f) = 2s + 2s·U.
    let (s2h, s2l) = (two * sh, two * sl);
    let (wh, wl) = mul22((s2h, s2l), (uh, ul));
    let (lh, le) = fast_sum(s2h, wh);
    let ll = le + (s2l + wl);

    // + n·ln 2, `n·ln2_hi` exact and 0 or above |lh|; renormalised, so
    // that `pow`'s `e^pl ≈ 1 + pl` holds.
    let (h, he) = fast_sum(n * ln2_hi, lh);
    fast_sum(h, he + n.mul_add(ln2_lo, ll))
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Pow for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn powi_small<const N: i32>(self) -> Self {
        powi_small::<V, N>(self)
    }

    #[inline(always)]
    fn pow(self, yh: Self, yl: Self) -> Self {
        let x = self;
        let [zero, half, one] = [k(x, 0.0), k(x, 0.5), k(x, 1.0)];
        let [inf, nan] = [k(x, f32::INFINITY), k(x, f32::NAN)];
        let [hi_cap, lo_cap] = ks(x, V::Elem::CAPS);
        let round = x.constant(V::Elem::ROUND);
        let ax = x.abs();

        // ln|x|, with −∞ for 0 and ∞ for ∞ so that y·ln|x| has the limit's
        // sign; NaN stays NaN.
        let finite = V::select(ax.gt(zero), V::select(ax.lt(inf), one, zero), zero).eq(one);
        let (lh, ll) = ln_parts(ax);
        let lh = V::select(finite, lh, V::select(ax.eq(zero), -inf, ax));
        let ll = V::select(finite, ll, zero);

        // y·ln|x| = ph + pl, the product exact; pl = 0 where ph is not
        // finite.
        let ph = yh * lh;
        let pl = yh.mul_add(ll, yl.mul_add(lh, -yh.residual(lh, ph)));
        let pl = V::select(ph.abs().lt(inf), pl, zero);

        // e^(ph + pl), ph capped past exp's range: beyond, the result is ∞
        // or 0 either way, and the scaling overflows or underflows by
        // itself.
        let pc = V::select(ph.gt(hi_cap), hi_cap, V::select(ph.lt(lo_cap), lo_cap, ph));
        let (eh, el, kk) = exp_parts(pc);
        let mag = scale(eh.mul_add(pl, el) + eh, kk, half, round);
        let mag = V::select(ph.gt(hi_cap), inf, V::select(ph.lt(lo_cap), zero, mag));
        // |x| = 1: 1 for every y, ±∞ included (0·∞ above is NaN).
        let mag = V::select(ax.eq(one), one, mag);

        // The rest only when some lane has x ≤ 0 or x, y not finite: for
        // positive finite x and finite y, `mag` is already the result
        // (x = 1 and y = 0 give ln|x| = 0 or y·ln|x| = 0, so 1).
        let plain = V::select(x.gt(zero), V::select(x.lt(inf), one, zero), zero);
        let plain = V::select(yh.abs().lt(inf), plain, zero);
        if !V::any(plain.eq(zero)) {
            return mag;
        }

        // Negative x: y an odd integer flips the sign; y not an integer
        // (NaN included) is NaN for finite x < 0. Values from 2^p on are
        // even integers; y = yh + yl is odd if exactly one part is.
        let int = V::select(
            yh.eq(yh.floor()),
            V::select(yl.eq(yl.floor()), one, zero),
            zero,
        );
        let [hh, lh2] = [yh * half, yl * half];
        let odd_h = V::select(hh.eq(hh.floor()), zero, one);
        let odd_l = V::select(lh2.eq(lh2.floor()), zero, one);
        let odd = V::select(int.eq(one), odd_h + odd_l, zero).eq(one);
        let flip = V::select(odd, V::select(x.is_sign_negative(), one, zero), zero).eq(one);
        let r = V::select(flip, -mag, mag);
        let finite_neg = V::select(x.lt(zero), V::select(ax.lt(inf), one, zero), zero);
        let r = V::select(V::select(int.eq(one), zero, finite_neg).eq(one), nan, r);

        // x⁰ = 1 and 1ʸ = 1, NaN included.
        V::select(yh.eq(zero), one, V::select(x.eq(one), one, r))
    }
}

/// `x^N` for a small integer `N`, by binary powering in two-part
/// arithmetic: each product exact to about twice the element type's
/// precision, the result rounded once (and for `N < 0`, `1/x^|N|` with one
/// correction step). Lanes where `|x^|N||` is outside `Kernel::EXACT`
/// (where the low parts would lose bits, or the scalar backend's exact
/// product would overflow), NaN, or 0 take [`Pow`].
#[inline(always)]
fn powi_small<V: Float, const N: i32>(x: V) -> V
where
    V::Elem: Kernel,
{
    let [zero, one] = [k(x, 0.0), k(x, 1.0)];
    let mut m = N.unsigned_abs();
    let (mut b, mut acc) = ((x, zero), (one, zero));
    let mut first = true;
    while m > 0 {
        if m & 1 == 1 {
            acc = if first { b } else { renorm(mul22(acc, b)) };
            first = false;
        }
        m >>= 1;
        if m > 0 {
            b = renorm(mul22(b, b));
        }
    }
    let (h, l) = acc;
    let r = if N < 0 {
        // 1/(h + l) = q + (1 − q·h − q·l)/(h + l), `1 − q·h` exact.
        let q = one / h;
        (q.residual(h, one) - q * l).mul_add(q, q)
    } else {
        h + l
    };

    let ah = h.abs();
    let [lo, hi] = ks(x, V::Elem::EXACT);
    let ok = V::select(ah.ge(lo), V::select(ah.le(hi), one, zero), zero);
    if V::any(ok.eq(zero)) {
        let n = x.constant(V::Elem::from_f32(N as f32));
        V::select(ok.eq(one), r, x.pow(n, zero))
    } else {
        r
    }
}

/// `(h, l)` renormalised: `h` the rounded sum.
#[inline(always)]
fn renorm<V: Float>((h, l): (V, V)) -> (V, V) {
    fast_sum(h, l)
}

/// `pow`'s constants and series, per element type.
pub(crate) trait Kernel: exp::Kernel + ln::Kernel {
    /// `[1/3, 1/5]`, each as `hi + lo`.
    const THIRD_FIFTH: [Self; 4];
    /// `[hi, lo]`: `y·ln|x|` is capped there, past `exp`'s range.
    const CAPS: [Self; 2];
    /// `[2^(emin + p), 2^(emax − 28)]`: inside, a product's low part is a
    /// normal float, and the scalar backend's exact product (Dekker's,
    /// which multiplies by `2^⌈p/2⌉ + 1`) does not overflow.
    const EXACT: [Self; 2];

    /// `W(z) = Σ z^j/(2j + 7)`: `atanh s = s·(1 + z/3 + z²/5 + z³·W)`,
    /// truncated below 2⁻⁷⁰ (`f64`) and 2⁻⁴⁰ (`f32`) of the result for
    /// `|s| ≤ 0.1716`.
    fn series<V: Float<Elem = Self>>(z: V) -> V;
}

impl Kernel for f64 {
    const THIRD_FIFTH: [f64; 4] = [
        f64::from_bits(0x3fd5_5555_5555_5555),
        f64::from_bits(0x3c75_5555_5555_5555),
        f64::from_bits(0x3fc9_9999_9999_999a),
        f64::from_bits(0xbc69_9999_9999_999a),
    ];
    const CAPS: [f64; 2] = [710.0, -746.0];
    const EXACT: [f64; 2] = [f64::from_bits(0x036 << 52), f64::from_bits(0x7bf << 52)]; // 2^−969, 2^960

    #[inline(always)]
    fn series<V: Float<Elem = f64>>(z: V) -> V {
        let c = ks(
            z,
            [
                1.0 / 7.0,
                1.0 / 9.0,
                1.0 / 11.0,
                1.0 / 13.0,
                1.0 / 15.0,
                1.0 / 17.0,
                1.0 / 19.0,
                1.0 / 21.0,
                1.0 / 23.0,
                1.0 / 25.0,
            ],
        );
        // Estrin: depth 4 instead of Horner's 10.
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let p01 = c[1].mul_add(z, c[0]);
        let p23 = c[3].mul_add(z, c[2]);
        let p45 = c[5].mul_add(z, c[4]);
        let p67 = c[7].mul_add(z, c[6]);
        let p89 = c[9].mul_add(z, c[8]);
        let p03 = p23.mul_add(z2, p01);
        let p47 = p67.mul_add(z2, p45);
        let p07 = p47.mul_add(z4, p03);
        p89.mul_add(z8, p07)
    }
}

impl Kernel for f32 {
    const THIRD_FIFTH: [f32; 4] = [
        f32::from_bits(0x3eaa_aaab),
        f32::from_bits(0xb22a_aaab),
        f32::from_bits(0x3e4c_cccd),
        f32::from_bits(0xb14c_cccd),
    ];
    const CAPS: [f32; 2] = [90.0, -105.0];
    const EXACT: [f32; 2] = [f32::from_bits(25 << 23), f32::from_bits(227 << 23)]; // 2^−102, 2^100

    #[inline(always)]
    fn series<V: Float<Elem = f32>>(z: V) -> V {
        let [c7, c9, c11, c13, c15] = ks(
            z,
            [1.0 / 7.0, 1.0 / 9.0, 1.0 / 11.0, 1.0 / 13.0, 1.0 / 15.0],
        );
        z.mul_add(z.mul_add(z.mul_add(z.mul_add(c15, c13), c11), c9), c7)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{MathElement, PowOp, PowScalarOp, PowiOp};
    use crate::testing::dd::pow_dd;
    use crate::testing::{TestFloat, ulp_error_scaled};
    use std::vec::Vec;

    /// A xorshift generator in `[0, 1)`.
    fn rng(mut state: u64) -> impl FnMut() -> f64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// `(x, y)` pairs: `x` spread over every positive binade of `T`
    /// (subnormals included) with `y` chosen so that `y·ln x` covers the
    /// whole range of `T`'s results, past both ends; `x` near 1 with huge
    /// `y`; and ordinary values.
    fn pairs<T: TestFloat>(n: usize) -> (Vec<T>, Vec<T>) {
        let mut r = rng(0x9e37_79b9_7f4a_7c15);
        let (lo_e, hi_e, ln_max) = if T::NAME == "f32" {
            (-149.0, 128.0, 89.0)
        } else {
            (-1074.0, 1024.0, 710.0)
        };
        let (mut x, mut y) = (Vec::new(), Vec::new());
        let mut push = |a: f64, b: f64| {
            let (a, b) = (T::from_f64(a), T::from_f64(b));
            if a.to_f64() > 0.0 && a.to_f64().is_finite() && b.to_f64().is_finite() {
                x.push(a);
                y.push(b);
            }
        };
        for _ in 0..n {
            let a = 2f64.powf(lo_e + (hi_e - lo_e) * r());
            let t = (-1.05 * ln_max) + 2.1 * ln_max * r();
            push(a, t / a.ln());
            let e = 2f64.powf(-50.0 + 48.0 * r()) * if r() < 0.5 { -1.0 } else { 1.0 };
            let a = T::from_f64(1.0 + e).to_f64();
            push(a, t / a.ln());
            push(10.0 * r(), 20.0 * r() - 10.0);
        }
        (x, y)
    }

    fn check<T: TestFloat>(name: &str, results: Vec<(&str, Vec<T>)>, x: &[T], y: &[f64]) {
        for (backend, got) in results {
            for ((&x, &y), &got) in x.iter().zip(y).zip(&got) {
                let e = ulp_error_scaled(got, pow_dd(x.to_f64(), y));
                assert!(
                    e <= 1.0,
                    "{name} {} {backend}: pow({x:?}, {y:?}) = {got:?}, {e:.3} ULP",
                    T::NAME
                );
            }
        }
    }

    #[test]
    fn powf_within_1_ulp() {
        let (x, y) = pairs::<f32>(100_000);
        let yf: Vec<f64> = y.iter().map(|&v| v.into()).collect();
        check("powf", f32::every_backend_binary::<PowOp>(&x, &y), &x, &yf);
        let (x, y) = pairs::<f64>(100_000);
        check("powf", f64::every_backend_binary::<PowOp>(&x, &y), &x, &y);
    }

    #[test]
    fn powf_scalar_within_1_ulp() {
        let x: Vec<f64> = (1..20_000).map(|i| f64::from(i) * 0.013).collect();
        for y in [-7.5, -1.0 / 3.0, 0.5, 2.2, 101.0] {
            let ys = std::vec![y; x.len()];
            check(
                "powf_scalar",
                f64::every_backend_scalars::<PowScalarOp>(&x, [y, 0.0]),
                &x,
                &ys,
            );
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let r = f32::every_backend_scalars::<PowScalarOp>(&x32, [y as f32, 0.0]);
            check(
                "powf_scalar",
                r,
                &x32,
                &std::vec![f64::from(y as f32); x.len()],
            );
        }
    }

    /// `x^n` for exponents `f32` cannot hold, with `x` close enough to 1
    /// that the result is finite.
    #[test]
    fn powi_within_1_ulp() {
        use crate::SliceMath;
        let x32: Vec<f32> = (0..2000)
            .map(|i| 1.0 + (i - 1000) as f32 * f32::EPSILON / 4.0)
            .collect();
        let x64: Vec<f64> = (0..2000)
            .map(|i| 1.0 + f64::from(i - 1000) * 1e-9)
            .collect();
        for n in [
            3,
            -5,
            17,
            1000,
            -77_777,
            16_777_217,
            -16_777_219,
            i32::MAX,
            i32::MIN,
        ] {
            let want = |x: f64| pow_dd(x, f64::from(n));
            let mut g32 = x32.clone();
            g32.powi_in_place(n);
            for (&x, &got) in x32.iter().zip(&g32) {
                let e = ulp_error_scaled(got, want(x.into()));
                assert!(e <= 1.0, "powi f32: {x:?}^{n} = {got:?}, {e:.3} ULP");
            }
            let mut g64 = x64.clone();
            g64.powi_in_place(n);
            for (&x, &got) in x64.iter().zip(&g64) {
                let e = ulp_error_scaled(got, want(x));
                assert!(e <= 1.0, "powi f64: {x:?}^{n} = {got:?}, {e:.3} ULP");
            }
        }
    }

    /// `x^N` by binary powering, on every backend: every binade of `x`,
    /// both signs, so that results also overflow, underflow and turn
    /// subnormal (the fallback lanes).
    fn powi_small_check<T: TestFloat + MathElement, const N: i32>() {
        let x: Vec<T> = crate::testing::samples::<T>(20_000, 1e300, &[1.0])
            .into_iter()
            .filter(|v| v.to_f64().is_finite() && v.to_f64() != 0.0)
            .collect();
        for (backend, got) in T::every_backend::<PowiOp<N>>(&x) {
            for (&x, &got) in x.iter().zip(&got) {
                let (h, l, k) = pow_dd(x.to_f64().abs(), f64::from(N));
                let sign = if x.to_f64() < 0.0 && N % 2 != 0 {
                    -1.0
                } else {
                    1.0
                };
                let e = ulp_error_scaled(got, (sign * h, sign * l, k));
                assert!(
                    e <= 1.0,
                    "powi {} {backend}: {x:?}^{N} = {got:?}, {e:.3} ULP",
                    T::NAME
                );
            }
        }
    }

    #[test]
    fn powi_small_within_1_ulp() {
        macro_rules! each {
            ($($n:literal)*) => {
                $(
                    powi_small_check::<f32, $n>();
                    powi_small_check::<f64, $n>();
                )*
            };
        }
        each!(3 4 5 6 7 8 -2 -3 -4 -5 -6 -7 -8);
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn pow_reference_is_precise() {
        let scale = |(h, l, k): (f64, f64, i32)| (h + l) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        for (x, y, want) in [
            (2.0, 0.5, core::f64::consts::SQRT_2),
            (10.0, -3.3, 5.011_872_336_272_725e-4),
            (0.999_999_999, 1e12, 0.0),
            (1.000_000_001, 5e11, 1.403_649_935_011_805_4e217),
            (3.0, 600.0, 1.873_927_703_884_794e286),
        ] {
            let got = scale(pow_dd(x, y));
            assert!(
                got == want || (want == 0.0 && got < 1e-300),
                "pow_dd({x}, {y}) = {got:e}"
            );
        }
    }

    #[test]
    fn pow_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        #[rustfmt::skip]
        let cases: [(f64, f64, f64); 34] = [
            (nan, 0.0, 1.0), (nan, -0.0, 1.0), (1.0, nan, 1.0), (1.0, inf, 1.0), (1.0, -inf, 1.0),
            (-1.0, inf, 1.0), (-1.0, -inf, 1.0), (-1.0, nan, nan), (nan, 1.0, nan), (2.0, nan, nan),
            (-2.0, 0.5, nan), (-2.0, 3.0, -8.0), (-2.0, 2.0, 4.0), (-2.0, -1.0, -0.5),
            (0.0, -1.0, inf), (-0.0, -1.0, -inf), (-0.0, -2.0, inf), (0.0, 3.0, 0.0),
            (-0.0, 3.0, -0.0), (-0.0, 2.0, 0.0), (-0.0, 0.5, 0.0), (0.0, -inf, inf), (0.0, inf, 0.0),
            (0.5, inf, 0.0), (0.5, -inf, inf), (2.0, inf, inf), (2.0, -inf, 0.0),
            (inf, 0.5, inf), (inf, -0.5, 0.0), (-inf, 3.0, -inf), (-inf, 2.0, inf),
            (-inf, -3.0, -0.0), (-inf, 0.5, inf), (-inf, -0.5, 0.0),
        ];
        let (x, y): (Vec<f64>, Vec<f64>) = cases.iter().map(|&(x, y, _)| (x, y)).unzip();
        let want: Vec<f64> = cases.iter().map(|c| c.2).collect();
        for (backend, got) in f64::every_backend_binary::<PowOp>(&x, &y) {
            crate::testing::assert_same(&got, &want, &std::format!("pow f64 {backend}"));
        }
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let want32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        for (backend, got) in f32::every_backend_binary::<PowOp>(&x32, &y32) {
            crate::testing::assert_same(&got, &want32, &std::format!("pow f32 {backend}"));
        }
        // and they agree with std's, which follows C99.
        for &(x, y, want) in &cases {
            let s = x.powf(y);
            assert!(
                s.to_bits() == want.to_bits() || (s.is_nan() && want.is_nan()),
                "std pow({x}, {y}) = {s}"
            );
        }
    }
}
