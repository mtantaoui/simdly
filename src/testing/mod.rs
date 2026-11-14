//! Test-only helpers shared by every kernel's unit tests: IEEE-754 special
//! values, bit-exact comparison and ULP error.

use core::fmt::Debug;
use std::vec::Vec;

use crate::math::{MathElement, UnaryOp};

pub(crate) trait TestFloat: Copy + PartialEq + Debug {
    const NAME: &str;
    /// Signed zeros, infinities, NaN, subnormals and one.
    const SPECIAL: [Self; 8];
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    /// From the low `size_of::<Self>() * 8` bits of `b`.
    fn from_bits64(b: u64) -> Self;
    fn is_nan(self) -> bool;
    fn bits(self) -> u64;
    /// The gap from `|self|` to the next float away from zero.
    fn ulp(self) -> f64;
    /// The sign bit, in [`bits`](Self::bits).
    const SIGN: u64 = 1 << (core::mem::size_of::<Self>() * 8 - 1);
}

macro_rules! test_float {
    ($($T:ident: $subnormal:expr),*) => {
        $(
            impl TestFloat for $T {
                const NAME: &str = stringify!($T);
                const SPECIAL: [Self; 8] = [
                    0.0,
                    -0.0,
                    $T::INFINITY,
                    -$T::INFINITY,
                    $T::NAN,
                    $subnormal,
                    -$subnormal,
                    1.0,
                ];

                fn from_f64(x: f64) -> Self {
                    x as $T
                }

                fn to_f64(self) -> f64 {
                    self.into()
                }

                fn from_bits64(b: u64) -> Self {
                    $T::from_bits(b as _)
                }

                fn is_nan(self) -> bool {
                    $T::is_nan(self)
                }

                fn bits(self) -> u64 {
                    self.to_bits().into()
                }

                fn ulp(self) -> f64 {
                    let a = self.abs();
                    f64::from($T::from_bits(a.to_bits() + 1)) - f64::from(a)
                }
            }
        )*
    };
}

test_float!(f32: 1e-40, f64: 1e-310);

/// `|got − want|` in ULPs of `want` rounded to `T`: 0 if both are NaN or
/// equal, infinite if only one is NaN.
pub(crate) fn ulp_error<T: TestFloat>(got: T, want: f64) -> f64 {
    ulp_error_dd(got, (want, 0.0))
}

/// [`ulp_error`] against `(hi + lo) · 2^k`, `hi + lo` near 1: computed in
/// the scaled domain `· 2^−k`, so it stays exact for subnormal results,
/// whose ULP is fixed.
pub(crate) fn ulp_error_scaled<T: TestFloat>(got: T, (hi, lo, k): (f64, f64, i32)) -> f64 {
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

/// [`ulp_error`] against `hi + lo`, a reference more precise than `f64`.
pub(crate) fn ulp_error_dd<T: TestFloat>(got: T, (hi, lo): (f64, f64)) -> f64 {
    if got.is_nan() || hi.is_nan() {
        return if got.is_nan() && hi.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    // A result that rounds to ±∞ must be exactly that.
    let want = T::from_f64(hi);
    if want.to_f64().is_infinite() || got.to_f64().is_infinite() {
        return if got.bits() == want.bits() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let got = got.to_f64();
    if got == hi && lo == 0.0 {
        return 0.0;
    }
    // `got − hi` is exact when they are close: the only case that matters.
    ((got - hi) - lo).abs() / T::from_f64(hi).ulp()
}

/// Bit-exact equality, except that any two NaNs match.
pub(crate) fn assert_same<T: TestFloat>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let same = g.bits() == w.bits() || (g.is_nan() && w.is_nan());
        assert!(same, "{what}, index {i}: got {g:?}, want {w:?}");
    }
}

/// About `2n` samples of `[−max, max]`, evenly spaced in bits so every
/// binade is covered, plus 2000 neighbours on each side of `±c` for every
/// `c` in `around`, and the special values.
pub(crate) fn samples<T: TestFloat>(n: u64, max: f64, around: &[f64]) -> Vec<T> {
    let top = T::from_f64(max).bits();
    let mut bits: Vec<u64> = (0..=top).step_by((top / n) as usize).collect();
    for &c in around {
        let c = T::from_f64(c).bits();
        bits.extend((c - 2000)..=(c + 2000));
    }
    let mut xs: Vec<T> = bits
        .iter()
        .flat_map(|&b| [b, b | T::SIGN])
        .map(T::from_bits64)
        .collect();
    xs.extend(T::SPECIAL);
    xs
}

/// [`samples`] of `[−1, 1]`, around ±0.5 and ±1, plus ±1.5.
pub(crate) fn unit_interval<T: TestFloat>(n: u64) -> Vec<T> {
    let mut xs = samples(n, 1.0, &[0.5, 1.0]);
    xs.extend([1.5, -1.5].map(T::from_f64));
    xs
}

/// `F` on every backend within `max` ULPs of `reference`, computed in `f64`:
/// for `f32` that measures the real error; for `f64` it compares with `std`.
pub(crate) fn assert_ulps<T: TestFloat + MathElement, F: UnaryOp>(
    name: &str,
    x: &[T],
    reference: fn(f64) -> f64,
    max: f64,
) {
    assert_ulps_dd::<T, F>(name, x, |v| (reference(v), 0.0), max);
}

/// [`assert_ulps`] against a reference given as `hi + lo`: for `f64`
/// results, when the platform's `f64` function is not accurate enough.
pub(crate) fn assert_ulps_dd<T: TestFloat + MathElement, F: UnaryOp>(
    name: &str,
    x: &[T],
    reference: impl Fn(f64) -> (f64, f64),
    max: f64,
) {
    for (backend, got) in T::every_backend::<F>(x) {
        for (&x, &got) in x.iter().zip(&got) {
            let e = ulp_error_dd(got, reference(x.to_f64()));
            assert!(
                e <= max,
                "{name} {} {backend}: {name}({x:?}) = {got:?}, {e:.3} ULP",
                T::NAME
            );
        }
    }
}

/// [`assert_ulps`] on every `f32`, in chunks.
pub(crate) fn assert_ulps_every_f32<F: UnaryOp>(name: &str, reference: fn(f64) -> f64, max: f64) {
    const CHUNK: u64 = 1 << 22;
    for start in (0..1_u64 << 32).step_by(CHUNK as usize) {
        let x: Vec<f32> = (start..start + CHUNK)
            .map(|b| f32::from_bits(b as u32))
            .collect();
        assert_ulps::<f32, F>(name, &x, reference, max);
    }
}

/// `F(x) == want` bit for bit on every backend, for each `(x, want)`; a NaN
/// `want` matches any NaN.
pub(crate) fn assert_exact<T: TestFloat + MathElement, F: UnaryOp>(name: &str, cases: &[(T, T)]) {
    let (x, want): (Vec<T>, Vec<T>) = cases.iter().copied().unzip();
    for (backend, got) in T::every_backend::<F>(&x) {
        assert_same(&got, &want, &std::format!("{name} {} {backend}", T::NAME));
    }
}

/// Double-double arithmetic (a value as an unevaluated `hi + lo`, about 106
/// bits), for `f64` test references that do not depend on the platform's
/// C library: the Windows C runtime's `cbrt`, for one, is off by 1.5 ULP.
pub(crate) mod dd {
    pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }
    pub(crate) fn dd_mul((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
        let p = ah * bh;
        let e = ah.mul_add(bh, -p) + (ah * bl + al * bh);
        two_sum(p, e)
    }
    pub(crate) fn dd_add((ah, al): (f64, f64), (bh, bl): (f64, f64)) -> (f64, f64) {
        let (s, e) = two_sum(ah, bh);
        two_sum(s, e + al + bl)
    }
    pub(crate) fn dd_div((ah, al): (f64, f64), d: f64) -> (f64, f64) {
        let q = ah / d;
        let p = q * d;
        let e = q.mul_add(d, -p);
        two_sum(q, ((ah - p) - e + al) / d)
    }

    /// `eˣ = (hi + lo) · 2^k` to about 100 bits, `hi + lo` in
    /// `[1/√2, √2]`: `x − k·ln 2` in double-double, then a Taylor series.
    pub(crate) fn exp_dd(x: f64) -> (f64, f64, i32) {
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

    /// `σ(x) = 1/(1 + e⁻ˣ) = (hi + lo) · 2^k` to about 100 bits: `e^−|x|`
    /// from [`exp_dd`], then `1/(1 + e)` (`x ≥ 0`) or `e/(1 + e)` divided in
    /// double-double. `|x|` is capped at 800, past which `e` is below every
    /// `f64`.
    pub(crate) fn sigmoid_dd(x: f64) -> (f64, f64, i32) {
        let (h, l, k) = exp_dd(-x.abs().min(800.0));
        let s = 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        // `1 + e`; `e`'s bits that `s` drops (subnormal) are far below 2⁻¹⁰⁶.
        let (dh, dl) = dd_add((1.0, 0.0), (h * s, l * s));
        // `n / (dh + dl) = (n / dh) · (1 − dl/dh)`, to 2⁻¹⁰⁶.
        let (n, k) = if x < 0.0 {
            ((h, l), k)
        } else {
            ((1.0, 0.0), 0)
        };
        let q = dd_div(n, dh);
        let (qh, ql) = dd_add(q, (-q.0 * (dl / dh), 0.0));
        (qh, ql, k)
    }

    /// `sinh x = (hi + lo) · 2^k` to about 100 bits: below `|x| = 1`, the
    /// Taylor series `x·(1 + x²/3!·(1 + x²/(4·5)·(1 + …)))` (30 terms);
    /// above, `(E − 1/E)/2` with `E = e^|x|` from [`exp_dd`], so the
    /// cancellation stays below a factor `coth 1 ≈ 1.3`.
    pub(crate) fn sinh_dd(x: f64) -> (f64, f64, i32) {
        if x == 0.0 {
            return (x, 0.0, 0); // ±0, its sign kept
        }
        if x.abs() < 1.0 {
            let x2 = (x * x, x.mul_add(x, -(x * x)));
            let mut s = (1.0, 0.0);
            for n in (1..=30).rev() {
                let n = f64::from(n);
                s = dd_add(
                    (1.0, 0.0),
                    dd_div(dd_mul(s, x2), (2.0 * n) * (2.0 * n + 1.0)),
                );
            }
            let (hi, lo) = dd_mul(s, (x, 0.0));
            return (hi, lo, 0);
        }
        let (hi, lo, k) = exp_parts_dd(x.abs(), -1.0);
        if x < 0.0 { (-hi, -lo, k) } else { (hi, lo, k) }
    }

    /// `cosh x = (hi + lo) · 2^k` to about 100 bits: `(E + 1/E)/2`, `E =
    /// e^|x|` from [`exp_dd`].
    pub(crate) fn cosh_dd(x: f64) -> (f64, f64, i32) {
        exp_parts_dd(x.abs(), 1.0)
    }

    /// `tanh x` as `hi + lo`, to about 100 bits: [`sinh_dd`] /
    /// [`cosh_dd`], divided in double-double.
    pub(crate) fn tanh_dd(x: f64) -> (f64, f64) {
        if x == 0.0 {
            return (x, 0.0); // ±0, its sign kept
        }
        let (sh, sl, sk) = sinh_dd(x);
        let (ch, cl, ck) = cosh_dd(x);
        // (sh + sl) / (ch + cl) = (sh + sl)/ch · (1 − cl/ch), to 2⁻¹⁰⁶;
        // the scale 2^(sk − ck) is exact on a result in [−1, 1].
        let q = dd_div((sh, sl), ch);
        let (qh, ql) = dd_add(q, (-q.0 * (cl / ch), 0.0));
        let s = 2f64.powi(sk - ck);
        (qh * s, ql * s)
    }

    /// `(E + sign/E)/2 = (hi + lo)·2^k` for `E = eᵃ`, `a ≥ 0` capped at
    /// 800 (past every `f64`).
    fn exp_parts_dd(a: f64, sign: f64) -> (f64, f64, i32) {
        let (h, l, k) = exp_dd(a.min(800.0));
        // 1/(h + l) = (1/h)·(1 − l/h), to 2⁻¹⁰⁶; times 2^−2k, dropped where
        // it is below 2⁻⁹⁰⁰ of `h`.
        let q = dd_div((1.0, 0.0), h);
        let (qh, ql) = dd_add(q, (-q.0 * (l / h), 0.0));
        let s = if 2 * k > 900 { 0.0 } else { 2f64.powi(-2 * k) };
        let (hi, lo) = dd_add((h, l), (sign * qh * s, sign * ql * s));
        (hi, lo, k - 1)
    }

    /// `atanh x` as `hi + lo` for `|x| < 1`, to about 100 bits:
    /// `½·ln(1 + q)`, `q = 2t/(1 − t)` divided in double-double (`1 − t`
    /// exact as a [`two_sum`]), the logarithm from [`log1p_dd2`].
    pub(crate) fn atanh_dd(x: f64) -> (f64, f64) {
        if x == 0.0 {
            return (x, 0.0); // ±0, its sign kept
        }
        let t = x.abs();
        let (dh, dl) = two_sum(1.0, -t);
        // 2t / (dh + dl) = (2t / dh)·(1 − dl/dh), to 2⁻¹⁰⁶.
        let q = dd_div((2.0 * t, 0.0), dh);
        let q = dd_add(q, (-q.0 * (dl / dh), 0.0));
        let (hi, lo) = log1p_dd2(q);
        let (hi, lo) = (0.5 * hi, 0.5 * lo);
        if x < 0.0 { (-hi, -lo) } else { (hi, lo) }
    }

    /// `√(wh + wl)` as `hi + lo`: one Newton step, `r²` exact.
    fn dd_sqrt((wh, wl): (f64, f64)) -> (f64, f64) {
        let r = wh.sqrt();
        let (ph, pl) = dd_mul((r, 0.0), (r, 0.0));
        two_sum(r, ((wh - ph) - pl + wl) / (2.0 * r))
    }

    /// `ln(2t) + c` as `hi + lo` for `t ≥ 2⁴⁰`: [`ln_dd`] plus `ln 2`.
    fn ln2t_dd(t: f64, c: f64) -> (f64, f64) {
        const LN2: (f64, f64) = (
            f64::from_bits(0x3fe6_2e42_fefa_39ef),
            f64::from_bits(0x3c7a_bc9e_3b39_803f),
        );
        dd_add(dd_add(ln_dd(t), LN2), (c, 0.0))
    }

    /// `asinh x` as `hi + lo`, to about 100 bits: `ln(1 + p)` from
    /// [`log1p_dd2`], `p = t + t²/(1 + √(1 + t²))` in double-double,
    /// `t = |x|`; above `2⁴⁰`, `ln(2t) + 1/(4t²)`.
    pub(crate) fn asinh_dd(x: f64) -> (f64, f64) {
        if x == 0.0 {
            return (x, 0.0); // ±0, its sign kept
        }
        let t = x.abs();
        let (hi, lo) = if t >= 1_099_511_627_776.0 {
            ln2t_dd(t, 0.25 / t / t)
        } else {
            let t2 = dd_mul((t, 0.0), (t, 0.0));
            let r = dd_sqrt(dd_add((1.0, 0.0), t2));
            let d = dd_add((1.0, 0.0), r);
            // t² / (dh + dl) = (t² / dh)·(1 − dl/dh), to 2⁻¹⁰⁶.
            let q = dd_div(t2, d.0);
            let q = dd_add(q, (-q.0 * (d.1 / d.0), 0.0));
            log1p_dd2(dd_add((t, 0.0), q))
        };
        if x < 0.0 { (-hi, -lo) } else { (hi, lo) }
    }

    /// `acosh x` as `hi + lo` for `x ≥ 1`, to about 100 bits: `ln(1 + p)`
    /// from [`log1p_dd2`], `p = t + √(t² + 2t)` in double-double, `t = x − 1`
    /// (exact); above `2⁴⁰`, `ln(2x) − 1/(4x²)`.
    pub(crate) fn acosh_dd(x: f64) -> (f64, f64) {
        if x >= 1_099_511_627_776.0 {
            return ln2t_dd(x, -0.25 / x / x);
        }
        let t = x - 1.0;
        if t == 0.0 {
            return (0.0, 0.0);
        }
        let w = dd_add(dd_mul((t, 0.0), (t, 0.0)), (2.0 * t, 0.0));
        log1p_dd2(dd_add((t, 0.0), dd_sqrt(w)))
    }

    /// `erf x` as `hi + lo`, to about 100 bits: `(2/√π)·e^−a²·Σ 2ⁿ·a^(2n+1)
    /// / (1·3·…·(2n+1))`, `a = |x|`, a series of positive terms (no
    /// cancellation) summed in double-double until they fall below 2⁻¹¹⁰
    /// of the sum; `e^−a²` from [`exp_dd`], `a²`'s low part multiplied in.
    /// From `a = 7` on, `erf` is 1 to 2⁻⁷⁴. Returned as `(hi + lo)·2^k`:
    /// below `a = 2⁻⁹⁰⁰`, `erf x = (2/√π)·x` to 2⁻¹⁸⁰⁰, kept scaled by 2²⁰⁰
    /// (`k = −200`) so subnormal results keep their bits.
    pub(crate) fn erf_dd(x: f64) -> (f64, f64, i32) {
        const TWO_OVER_SQRT_PI: (f64, f64) = (
            f64::from_bits(0x3ff2_0dd7_5042_9b6d),
            f64::from_bits(0x3c71_ae3a_914f_ed80),
        );
        if x == 0.0 {
            return (x, 0.0, 0); // ±0, its sign kept
        }
        let a = x.abs();
        if a < 2f64.powi(-900) {
            let (hi, lo) = dd_mul((x * 2f64.powi(200), 0.0), TWO_OVER_SQRT_PI);
            return (hi, lo, -200);
        }
        let (hi, lo) = if a >= 7.0 {
            (1.0, 0.0)
        } else {
            let a2 = dd_mul((a, 0.0), (a, 0.0));
            let two_a2 = (2.0 * a2.0, 2.0 * a2.1);
            let (mut term, mut sum) = ((a, 0.0), (a, 0.0));
            let mut n = 0.0;
            while term.0 > sum.0 * 2f64.powi(-110) {
                n += 1.0;
                term = dd_div(dd_mul(term, two_a2), 2.0 * n + 1.0);
                sum = dd_add(sum, term);
            }
            // e^−(h + l) = e^−h·(1 − l), to (l²/2)·e^−h < 2⁻¹⁰⁰ of it.
            let (eh, el, k) = exp_dd(-a2.0);
            let e = dd_mul((eh, el), two_sum(1.0, -a2.1));
            let (hi, lo) = dd_mul(dd_mul(e, TWO_OVER_SQRT_PI), sum);
            let s = 2f64.powi(k);
            (hi * s, lo * s)
        };
        if x < 0.0 { (-hi, -lo, 0) } else { (hi, lo, 0) }
    }

    /// `gelu x = x·Φ(x) = (hi + lo)·2^k`, to about 100 bits, in `x` itself
    /// (no `x/√2` to round). `φ(x) = e^(−x²/2)/√(2π)` from [`exp_dd`],
    /// `x²` exact; then, for `|x| ≤ 6`, `Φ(x) = ½ + φ(x)·Σ x^(2n+1)/(2n+1)!!`
    /// (positive terms in `|x|`; for `x < 0`, `½ − …` loses at most 30 of
    /// the 106 bits), and beyond, `Φ(−|x|) = φ(x)·M(|x|)`, the Mills ratio
    /// `M(t) = 1/(t + 1/(t + 2/(t + 3/(t + …))))` evaluated backward from
    /// 300 terms. Tiny `x` is kept scaled by 2²⁰⁰ (`k = −200`).
    pub(crate) fn gelu_dd(x: f64) -> (f64, f64, i32) {
        const INV_SQRT_2PI: (f64, f64) = (
            f64::from_bits(0x3fd9_8845_33d4_3651),
            f64::from_bits(0xbc7c_bc0d_30eb_fd15),
        );
        if x == 0.0 || x.is_nan() {
            return (x, 0.0, 0);
        }
        if x.is_infinite() {
            return if x > 0.0 { (x, 0.0, 0) } else { (-0.0, 0.0, 0) };
        }
        let t = x.abs().min(60.0);
        // φ(t) = (ph + pl)·2^pk.
        let q = dd_mul((t, 0.0), (t, 0.0));
        let (eh, el, pk) = exp_dd(-0.5 * q.0);
        let phi = dd_mul(dd_mul((eh, el), two_sum(1.0, -0.5 * q.1)), INV_SQRT_2PI);
        let s = 2f64.powi(pk);
        // Φ(t) and 1 − Φ(t) = Φ(−t), as (hi, lo)·2^k.
        let (upper, lower, lk) = if t <= 6.0 {
            let (mut term, mut sum) = ((t, 0.0), (t, 0.0));
            let mut n = 0.0;
            while term.0 > sum.0 * 2f64.powi(-110) {
                n += 1.0;
                term = dd_div(dd_mul(term, q), 2.0 * n + 1.0);
                sum = dd_add(sum, term);
            }
            let ps = dd_mul(phi, sum);
            let ps = (ps.0 * s, ps.1 * s);
            (
                dd_add((0.5, 0.0), ps),
                dd_add((0.5, 0.0), (-ps.0, -ps.1)),
                0,
            )
        } else {
            let mut v = (t, 0.0);
            for j in (1..=300).rev() {
                // v = t + j/v
                let r = dd_div((f64::from(j), 0.0), v.0);
                let r = dd_add(r, (-(f64::from(j) / v.0) * v.1 / v.0, 0.0));
                v = dd_add((t, 0.0), r);
            }
            let m = dd_div((1.0, 0.0), v.0);
            let m = dd_add(m, (-m.0 * v.1 / v.0, 0.0));
            let low = dd_mul(phi, m);
            let up = dd_add((1.0, 0.0), (-low.0 * s, -low.1 * s));
            (up, low, pk)
        };
        let (hi, lo, k) = if x > 0.0 {
            let (h, l) = dd_mul((x, 0.0), upper);
            (h, l, 0)
        } else {
            let (h, l) = dd_mul((x, 0.0), lower);
            (h, l, lk)
        };
        if x.abs() < 2f64.powi(-900) {
            // x·Φ(x) ≈ x/2: scaled, so subnormal results keep their bits.
            let (h, l) = dd_mul((x * 2f64.powi(200), 0.0), upper);
            return if x > 0.0 {
                (h, l, -200)
            } else {
                let (h, l) = dd_mul((x * 2f64.powi(200), 0.0), lower);
                (h, l, -200)
            };
        }
        (hi, lo, k)
    }

    /// `xʸ = (hi + lo)·2^k` for positive finite `x` and finite `y`, to
    /// about 100 bits: `y·ln x` in double-double ([`ln_dd`]), then
    /// [`exp_dd`] of its high part times `e^lo = 1 + lo + lo²/2`. `y·ln x`
    /// is capped to `[−1200, 1200]`, past every `f64`.
    pub(crate) fn pow_dd(x: f64, y: f64) -> (f64, f64, i32) {
        let (ph, pl) = dd_mul((y, 0.0), ln_dd(x));
        let ph = ph.clamp(-1200.0, 1200.0);
        let (eh, el, k) = exp_dd(ph);
        let (hi, lo) = dd_mul((eh, el), two_sum(1.0, pl + 0.5 * pl * pl));
        (hi, lo, k)
    }

    /// `√(x² + y²) = (hi + lo) · 2^k`: both scaled by a power of 2 near the
    /// larger one, the squares summed exactly in double-double, then one
    /// Newton step of the square root.
    pub(crate) fn hypot_dd(x: f64, y: f64) -> (f64, f64, i32) {
        norm_dd(&[x, y])
    }

    /// The Euclidean norm of `v` as `(hi + lo) · 2^k`, as [`hypot_dd`].
    pub(crate) fn norm_dd(v: &[f64]) -> (f64, f64, i32) {
        let a = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if a == 0.0 {
            return (0.0, 0.0, 0);
        }
        let k = a.log2().floor() as i32;
        let s = |v: f64| v * 2f64.powi(-k / 2) * 2f64.powi(-(k - k / 2));
        let sq = |v: f64| (v * v, v.mul_add(v, -(v * v)));
        let sum = v.iter().fold((0.0, 0.0), |acc, &x| dd_add(acc, sq(s(x))));
        let r = sum.0.sqrt();
        // (sum − r²)/(2r), with r² exact.
        let (p, e) = sq(r);
        let c = ((sum.0 - p) - e + sum.1) / (2.0 * r);
        (r, c, k)
    }

    /// `sin x` as `hi + lo` for finite `x`, independent of the platform's
    /// `sin`: [`reduce`](crate::math::trig::reduce) (exact, checked against
    /// fixed 3000-bit values), then Taylor series in double-double.
    pub(crate) fn sin_dd(x: f64) -> (f64, f64) {
        trig_dd(x, 0)
    }

    /// `cos x` as `hi + lo`, as [`sin_dd`]: `cos x = sin(x + π/2)`.
    pub(crate) fn cos_dd(x: f64) -> (f64, f64) {
        trig_dd(x, 1)
    }

    /// `tan x` as `hi + lo`: [`sin_dd`] / [`cos_dd`].
    pub(crate) fn tan_dd(x: f64) -> (f64, f64) {
        let (s, c) = (sin_dd(x), cos_dd(x));
        let q = s.0 / c.0;
        let (ph, pl) = dd_mul((q, 0.0), c);
        let r = dd_add(s, (-ph, -pl));
        two_sum(q, r.0 / c.0)
    }

    /// `sin` of `x + shift·π/2`.
    fn trig_dd(x: f64, shift: u32) -> (f64, f64) {
        let (n, y0, y1) = crate::math::trig::reduce(x);
        let y = (y0, y1);
        let y2 = dd_mul(y, y);
        // sin y = y·Σ (−y²)^k/(2k+1)!, cos y = Σ (−y²)^k/(2k)!, Horner from
        // the 20th term down.
        let (mut s, mut c) = ((1.0, 0.0), (1.0, 0.0));
        for k in (1..=20).rev() {
            let k = f64::from(k);
            let ds = dd_div(dd_mul(s, y2), -(2.0 * k) * (2.0 * k + 1.0));
            s = dd_add((1.0, 0.0), ds);
            let dc = dd_div(dd_mul(c, y2), -(2.0 * k - 1.0) * (2.0 * k));
            c = dd_add((1.0, 0.0), dc);
        }
        let s = dd_mul(s, y);
        match (n as u32 + shift) % 4 {
            0 => s,
            1 => c,
            2 => (-s.0, -s.1),
            _ => (-c.0, -c.1),
        }
    }

    /// `ln x` as `hi + lo` for positive finite `x`: `std`'s `ln`, plus the
    /// correction `ln(x / e^y) ≈ x / e^y − 1`, computed with [`exp_dd`].
    pub(crate) fn ln_dd(x: f64) -> (f64, f64) {
        let y = x.ln();
        let (h, l, k) = exp_dd(y);
        // x · 2^−k is exact and close to h + l, so q − h is exact.
        let q = x * 2f64.powi(-k / 2) * 2f64.powi(-(k - k / 2));
        let d = ((q - h) - l) / h;
        two_sum(y, d)
    }

    /// `ln(1 + x)` as `hi + lo`, to about 100 bits of the result.
    pub(crate) fn log1p_dd(x: f64) -> (f64, f64) {
        log1p_dd2((x, 0.0))
    }

    /// `ln(1 + xh + xl)` as `hi + lo`, to about 100 bits of the result. Below
    /// `|x| = 2⁻⁶` [`ln_dd`]'s absolute error (about 2⁻¹⁰⁰) would show in
    /// the small result, so the series `x − x²/2 + x³/3 − …` (30 terms, to
    /// 2⁻¹⁸⁰) runs instead. Above, `1 + x = u + c` exactly (to `xl`'s
    /// rounding), and `ln(u + c) = ln u + c/u` to `(c/u)²/2 < 2⁻¹⁰⁵`.
    pub(crate) fn log1p_dd2(x: (f64, f64)) -> (f64, f64) {
        if x.0.abs() < 1.0 / 64.0 {
            // x · (1 − x·(1/2 − x·(1/3 − …))), from the innermost term out.
            let mut p = (0.0, 0.0);
            for n in (1..=30).rev() {
                let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
                p = dd_add(dd_div((sign, 0.0), f64::from(n)), dd_mul(p, x));
            }
            return dd_mul(p, x);
        }
        let (u, c) = two_sum(1.0, x.0);
        dd_add(ln_dd(u), ((c + x.1) / u, 0.0))
    }

    /// `ln(1 + eˣ) = (hi + lo) · 2^k` to about 100 bits:
    /// `max(x, 0) + ln(1 + e^−|x|)`, `e^−|x|` from [`exp_dd`] and the
    /// logarithm from [`log1p_dd2`]. For `x < 0` the result is `e·S(e)`,
    /// `S(e) = ln(1 + e)/e`, kept scaled by `e`'s `2^k` (it may be
    /// subnormal). `|x|` is capped at 800, past which `e` is below every
    /// `f64`.
    pub(crate) fn softplus_dd(x: f64) -> (f64, f64, i32) {
        let (h, l, k) = exp_dd(-x.abs().min(800.0));
        let s = 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        let e = dd_mul((h, l), (s, 0.0));
        if x >= 0.0 {
            let (hi, lo) = dd_add((x, 0.0), log1p_dd2(e));
            return (hi, lo, 0);
        }
        if e.0 >= 1.0 / 64.0 {
            let (hi, lo) = log1p_dd2(e);
            return (hi, lo, 0);
        }
        // S(e) = 1 − e/2 + e²/3 − …; `e`'s bits lost to subnormals weigh
        // below 2⁻¹⁰⁶ of S.
        let mut p = (0.0, 0.0);
        for n in (1..=30).rev() {
            let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
            p = dd_add(dd_div((sign, 0.0), f64::from(n)), dd_mul(p, e));
        }
        let (hi, lo) = dd_mul((h, l), p);
        (hi, lo, k)
    }
}
