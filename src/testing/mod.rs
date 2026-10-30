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

    /// `√(x² + y²) = (hi + lo) · 2^k`: both scaled by a power of 2 near the
    /// larger one, the squares summed exactly in double-double, then one
    /// Newton step of the square root.
    pub(crate) fn hypot_dd(x: f64, y: f64) -> (f64, f64, i32) {
        let (a, b) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
        if a == 0.0 {
            return (0.0, 0.0, 0);
        }
        let k = a.log2().floor() as i32;
        let s = |v: f64| v * 2f64.powi(-k / 2) * 2f64.powi(-(k - k / 2));
        let (a, b) = (s(a), s(b));
        let sq = |v: f64| (v * v, v.mul_add(v, -(v * v)));
        let sum = dd_add(sq(a), sq(b));
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
        match n as u32 {
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
}
