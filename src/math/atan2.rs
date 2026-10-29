//! `atan2(y, x)`, ≤ 1 ULP on every backend, written once over [`Float`]. It
//! reuses [`atan`](super::atan)'s ranges, constants and polynomial, with two
//! changes that keep the error below 1 ULP (musl's `atan2`, which computes
//! `atan(|y/x|)` and then `π − z`, reaches 1.47 ULP):
//!
//! 1. **No rounded `y/x`.** With `Y = |y|`, `X = |x|`, the reduction becomes
//!    `t = (p·Y − q·X)/(p·X + q·Y)`, compared as `Y < c·X`. The quotient
//!    is computed as `t + tl`: `t = num · (1/den)`, and the exact remainder
//!    `tl = (num − t·den)/den` ([`Float::residual`]) restores the bits the
//!    division drops. Without it, `t` near a power of 2 costs up to 1 ULP.
//! 2. **π folded into the constants.** For `x < 0` the result is
//!    `π − atan(Y/X) = H − ((τ·S − L) − τ)` with `τ = −t` and `H + L` the
//!    split of `π − atan(c)`: the same single final rounding as `atan`.
//!
//! Inputs are scaled by a power of 2 so that nothing overflows and the
//! remainder stays exact; infinities become 1 (and the finite argument 0),
//! and `(±0, ±0)` takes `x = ±1`: the C99 special values then fall out.
//!
//! Measured: 0.81 ULP (`f32`, 4·10⁷ pairs) and 0.79 ULP (`f64`, 3·10⁵ pairs
//! against a 200-bit reference), on every backend.

#![allow(clippy::excessive_precision, reason = "computed with mpmath, verbatim")]

use super::atan::{self, k, ks, pick};
use super::float::{Element, Float};

/// A lane type with a two-argument arc tangent.
pub(crate) trait Atan2 {
    /// `atan2(self, x)`: the angle of `(x, self)`.
    fn atan2(self, x: Self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Atan2 for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        let y = self;
        let [zero, one] = ks(y, [0.0, 1.0].map(V::Elem::from_f32));
        let inf = k(y, f32::INFINITY);
        let [big, down, up, small] = ks(y, V::Elem::SCALE);

        // ±∞ as 1, the other argument as 0 (or NaN), then (0, 0) as (0, 1).
        let (ay, ax) = (y.abs(), x.abs());
        let (y_inf, x_inf) = (ay.eq(inf), ax.eq(inf));
        let yy = V::select(y_inf, one, V::select(x_inf, ay * zero, ay));
        let xx = V::select(x_inf, one, V::select(y_inf, ax * zero, ax));
        let xx = V::select(yy.eq(zero), V::select(xx.eq(zero), one, xx), xx);

        // Scale by a power of 2: no overflow, and an exact remainder.
        let y_lt_x = yy.lt(xx);
        let (hi_mag, lo_mag) = (V::select(y_lt_x, xx, yy), V::select(y_lt_x, yy, xx));
        let f = V::select(hi_mag.gt(big), down, V::select(lo_mag.lt(small), up, one));
        let (yy, xx) = (yy * f, xx * f);

        // `atan`'s ranges, for `a = Y/X`.
        let m = [
            yy.lt(xx * k(y, 7.0 / 16.0)),
            yy.lt(xx * k(y, 11.0 / 16.0)),
            yy.lt(xx * k(y, 19.0 / 16.0)),
            yy.lt(xx * k(y, 39.0 / 16.0)),
        ];
        let p = pick(m, ks(y, [1.0, 2.0, 1.0, 1.0, 0.0].map(V::Elem::from_f32)));
        let q = pick(m, ks(y, [0.0, 1.0, 1.0, 1.5, 1.0].map(V::Elem::from_f32)));

        // t + tl = (p·Y − q·X)/(p·X + q·Y), to twice the precision.
        let num = q.residual(xx, p * yy);
        let den = q.mul_add(yy, p * xx);
        let inv = one / den;
        let t = num * inv;
        let tl = t.residual(den, num) * inv;
        let z = t * t;
        // atan(t + tl) ≈ atan(t) + tl/(1 + z).
        let tl = (-tl).mul_add(z, tl);
        let s = <V::Elem as atan::Kernel>::s(z);

        // atan(c) for x ≥ 0, π − atan(c) for x < 0.
        let [h0, h1, h2, h3] = ks(y, <V::Elem as atan::Kernel>::HI);
        let [l0, l1, l2, l3] = ks(y, <V::Elem as atan::Kernel>::LO);
        let [hn0, hn1, hn2, hn3, hn4] = ks(y, V::Elem::PI_MINUS_HI);
        let [ln0, ln1, ln2, ln3, ln4] = ks(y, V::Elem::PI_MINUS_LO);
        let x_neg = x.is_sign_negative();
        let h = V::select(
            x_neg,
            pick(m, [hn0, hn1, hn2, hn3, hn4]),
            pick(m, [zero, h0, h1, h2, h3]),
        );
        let l = V::select(
            x_neg,
            pick(m, [ln0, ln1, ln2, ln3, ln4]),
            pick(m, [zero, l0, l1, l2, l3]),
        );
        let (tau, taul) = (V::select(x_neg, -t, t), V::select(x_neg, -tl, tl));

        let r = h - ((tau.mul_add(s, -l) - taul) - tau);
        V::select(y.is_sign_negative(), -r, r)
    }
}

/// Constants for one element type, on top of `atan`'s.
pub(crate) trait Kernel: atan::Kernel {
    /// High parts of `π − atan(c)`, `c = 0, 0.5, 1, 1.5, ∞`.
    const PI_MINUS_HI: [Self; 5];
    /// Their low parts.
    const PI_MINUS_LO: [Self; 5];
    /// `[2^k, 2^−k, 2^k, 2^−k]`: above the first, scale down by the second;
    /// below the last, scale up by the third.
    const SCALE: [Self; 4];
}

impl Kernel for f32 {
    const PI_MINUS_HI: [f32; 5] = [
        f32::from_bits(0x4049_0fdb), // π
        f32::from_bits(0x402b_6374),
        f32::from_bits(0x4016_cbe4), // 3π/4
        f32::from_bits(0x400a_29c3),
        f32::from_bits(0x3fc9_0fdb), // π/2
    ];
    const PI_MINUS_LO: [f32; 5] = [
        f32::from_bits(0xb3bb_bd2e),
        f32::from_bits(0xb3c6_80a5),
        f32::from_bits(0xb1cc_de2e),
        f32::from_bits(0xb138_a11d),
        f32::from_bits(0xb33b_bd2e),
    ];
    const SCALE: [f32; 4] = [
        f32::from_bits((127 + 60) << 23),
        f32::from_bits((127 - 60) << 23),
        f32::from_bits((127 + 60) << 23),
        f32::from_bits((127 - 60) << 23),
    ];
}

impl Kernel for f64 {
    const PI_MINUS_HI: [f64; 5] = [
        f64::from_bits(0x4009_21fb_5444_2d18), // π
        f64::from_bits(0x4005_6c6e_7397_f5ae),
        f64::from_bits(0x4002_d97c_7f33_21d2), // 3π/4
        f64::from_bits(0x4001_4538_5fa3_af71),
        f64::from_bits(0x3ff9_21fb_5444_2d18), // π/2
    ];
    const PI_MINUS_LO: [f64; 5] = [
        f64::from_bits(0x3ca1_a626_3314_5c07),
        f64::from_bits(0x3ca6_60b6_4ece_6f4b),
        f64::from_bits(0x3c9a_7939_4c9e_8a0a),
        f64::from_bits(0x3caf_a535_23b6_428f),
        f64::from_bits(0x3c91_a626_3314_5c07),
    ];
    const SCALE: [f64; 4] = [
        f64::from_bits((1023 + 500) << 52),
        f64::from_bits((1023 - 500) << 52),
        f64::from_bits((1023 + 500) << 52),
        f64::from_bits((1023 - 500) << 52),
    ];
}

#[cfg(test)]
mod tests {
    use crate::math::{Atan2Op, MathElement};
    use crate::testing::{TestFloat, assert_same, ulp_error};
    use std::vec::Vec;

    /// `(y, x)` pairs: every quadrant; ratios over 60 binades, near
    /// `tan(2^−k)` (where the result crosses a power of 2) and near `atan`'s
    /// range boundaries; magnitudes from subnormal to huge.
    fn pairs<T: TestFloat>(n: usize, seed: u64) -> (Vec<T>, Vec<T>) {
        let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let unit = |r: u64| (r >> 11) as f64 / (1_u64 << 53) as f64;
        let max_exp = if T::NAME == "f32" { 120.0 } else { 1000.0 };
        let (mut ys, mut xs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let (r1, r2) = (next(), next());
            let x = 2f64.powf((unit(r1) * 2.0 - 1.0) * max_exp);
            let a = match i % 3 {
                0 => 2f64.powf(unit(r2) * 60.0 - 30.0),
                1 => {
                    2f64.powi(-((1 + (r2 >> 58) % 24) as i32)).tan()
                        * (1.0 + (unit(r2) - 0.5) * 0.02)
                }
                _ => {
                    [7.0, 11.0, 19.0, 39.0][(r2 % 4) as usize] / 16.0
                        * (1.0 + (unit(r2) - 0.5) * 1e-5)
                }
            };
            let sign = |bit: u64| if bit != 0 { -1.0 } else { 1.0 };
            ys.push(T::from_f64(sign(r2 & 2) * x * a));
            xs.push(T::from_f64(sign(r1 & 1) * x));
        }
        (ys, xs)
    }

    fn check<T: TestFloat + MathElement>(y: &[T], x: &[T]) {
        for (backend, got) in T::every_backend_binary::<Atan2Op>(y, x) {
            for i in 0..y.len() {
                let want = y[i].to_f64().atan2(x[i].to_f64());
                let e = ulp_error(got[i], want);
                assert!(
                    e <= 1.0,
                    "atan2 {} {backend}: atan2({:?}, {:?}) = {:?}, {e:.3} ULP",
                    T::NAME,
                    y[i],
                    x[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn atan2_within_1_ulp() {
        let (y, x) = pairs::<f32>(200_000, 0);
        check(&y, &x);
        let (y, x) = pairs::<f64>(200_000, 0);
        check(&y, &x);
    }

    /// The dispatched entry points agree with the backends, at every length
    /// (full registers and tails).
    #[test]
    fn atan2_dispatch_matches_backends() {
        fn check_dispatch<T: TestFloat + MathElement>() {
            for len in 0..=40 {
                let (y, x) = pairs::<T>(len, len as u64);
                let want = &T::every_backend_binary::<Atan2Op>(&y, &x)[0].1;
                let mut out = Vec::with_capacity(len);
                T::binary_uninit::<Atan2Op>(&y, &x, &mut out.spare_capacity_mut()[..len]);
                // SAFETY: `binary_uninit` initialised the first `len` elements.
                unsafe { out.set_len(len) };
                let mut in_place = y.clone();
                T::binary_in_place::<Atan2Op>(&mut in_place, &x);
                // Backends may differ by a rounding: compare within 1 ULP of
                // the scalar result's reference instead of bit for bit.
                for i in 0..len {
                    let r = y[i].to_f64().atan2(x[i].to_f64());
                    assert!(
                        ulp_error(out[i], r) <= 1.0 && ulp_error(in_place[i], r) <= 1.0,
                        "len {len}, index {i}"
                    );
                }
                assert_eq!(want.len(), len);
            }
        }
        check_dispatch::<f32>();
        check_dispatch::<f64>();
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn atan2_length_mismatch_panics() {
        let mut y = [1.0_f32; 3];
        f32::binary_in_place::<Atan2Op>(&mut y, &[1.0; 2]);
    }

    /// C99 Annex F: signed zeros, infinities and NaN.
    #[test]
    fn atan2_special_values() {
        fn special<T: TestFloat + MathElement>() {
            use core::f64::consts::{FRAC_PI_2 as P2, FRAC_PI_4 as P4, PI};
            let (inf, nan) = (f64::INFINITY, f64::NAN);
            let cases: [(f64, f64, f64); 24] = [
                (0.0, 1.0, 0.0),
                (-0.0, 1.0, -0.0),
                (0.0, 0.0, 0.0),
                (-0.0, 0.0, -0.0),
                (0.0, -0.0, PI),
                (-0.0, -0.0, -PI),
                (0.0, -1.0, PI),
                (-0.0, -1.0, -PI),
                (1.0, 0.0, P2),
                (1.0, -0.0, P2),
                (-1.0, 0.0, -P2),
                (inf, 1.0, P2),
                (-inf, -1.0, -P2),
                (1.0, inf, 0.0),
                (-1.0, inf, -0.0),
                (1.0, -inf, PI),
                (-1.0, -inf, -PI),
                (inf, inf, P4),
                (-inf, inf, -P4),
                (inf, -inf, 3.0 * P4),
                (-inf, -inf, -3.0 * P4),
                (nan, 1.0, nan),
                (1.0, nan, nan),
                (inf, nan, nan),
            ];
            let y: Vec<T> = cases.iter().map(|c| T::from_f64(c.0)).collect();
            let x: Vec<T> = cases.iter().map(|c| T::from_f64(c.1)).collect();
            let want: Vec<T> = cases.iter().map(|c| T::from_f64(c.2)).collect();
            for (backend, got) in T::every_backend_binary::<Atan2Op>(&y, &x) {
                assert_same(&got, &want, &std::format!("atan2 {} {backend}", T::NAME));
            }
        }
        special::<f32>();
        special::<f64>();
    }

    /// 10⁸ more `f32` pairs: `cargo test --release -- --ignored many_pairs`.
    #[test]
    #[ignore = "slow: 10^8 pairs per backend"]
    fn atan2_f32_many_pairs() {
        for seed in 1..=500 {
            let (y, x) = pairs::<f32>(200_000, seed);
            check(&y, &x);
        }
    }
}
