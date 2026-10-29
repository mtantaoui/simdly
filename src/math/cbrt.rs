//! `∛x`, ≤ 1 ULP on every backend (measured at most 0.667), written once
//! over [`Float`]. musl's `cbrtf` gets its first guess by dividing the bits
//! by 3 as an integer and refines it in `f64`; neither suits SIMD lanes, so:
//!
//! 1. **Reduce:** `|x| = m · 2^e` ([`Float::frexp`]), `e = 3k + r` with
//!    `r ∈ {−1, 0, 1}`, so `∛|x| = ∛(m · 2^r) · 2^k`, and `m · 2^r` is exact.
//! 2. **Guess:** `∛m ≈ P(2m − 3)` on `m ∈ [1, 2)` (degree 3 for `f32`, 6 for
//!    `f64`: 13 and 22 bits), times `∛2^r`.
//! 3. **Refine** as musl's `cbrt.c`: round the guess `t` away from zero to
//!    half the significand (12 or 26 bits), so `t²` is exact, then one
//!    cubically convergent step `t + t·(a/t² − t)/(2t + a/t²)`.
//! 4. **Scale back:** `· 2^k` ([`Float::ldexp`]), and the sign.
//!
//! Subnormals are scaled into the normal range first; `±0`, `±∞` and NaN
//! are returned as they are.

#![allow(clippy::excessive_precision, reason = "fitted coefficients, verbatim")]

use super::atan::{k, ks};
use super::float::{Element, Float};

/// A lane type with a cube root.
pub(crate) trait Cbrt {
    fn cbrt(self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Cbrt for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn cbrt(self) -> Self {
        let x = self;
        let [zero, half, one, two, three] = ks(x, [0.0, 0.5, 1.0, 2.0, 3.0].map(V::Elem::from_f32));
        let [third, round] = ks(x, [V::Elem::THIRD, V::Elem::ROUND]);
        let [tiny_scale, tiny_k] = ks(x, V::Elem::TINY);
        let [cbrt_half, cbrt_two] = ks(x, V::Elem::CBRT_2);

        // Subnormals: scale up by 2^3j, take j off the exponent at the end.
        let a = x.abs();
        let tiny = a.lt(x.constant(V::Elem::MIN_POSITIVE));
        let a = V::select(tiny, a * tiny_scale, a);

        // |x| = m · 2^e, e = 3k + r.
        let (m, e) = a.frexp();
        let q = (e * third + round) - round;
        let r = e - three * q;
        let q = q + V::select(tiny, tiny_k, zero);
        let (r_neg, r_pos) = (r.lt(zero), r.gt(zero));
        let mr = V::select(r_neg, m * half, V::select(r_pos, m * two, m));
        let cr = V::select(r_neg, cbrt_half, V::select(r_pos, cbrt_two, one));

        // A guess, rounded away from zero to half the significand: t² is exact.
        let y = V::Elem::poly(m.mul_add(two, -three)) * cr;
        let t = (y * x.constant(V::Elem::ROUND_UP)).and_bits(V::Elem::HALF_BITS);

        // One step: t + t·(mr/t² − t)/(2t + mr/t²).
        let s = mr / (t * t);
        let c = (s - t) / (t + t + s);
        let t = t.mul_add(c, t).ldexp(q);

        let t = V::select(x.lt(zero), -t, t);
        // ±0, ±∞, NaN: themselves.
        V::select(
            a.gt(zero),
            V::select(a.lt(k(x, f32::INFINITY)), t, x + x),
            x + x,
        )
    }
}

/// Constants and the initial polynomial for one element type.
pub(crate) trait Kernel: Element {
    /// `1/3`, rounded up: `e · THIRD` never falls short of an integer `e/3`.
    const THIRD: Self;
    /// `[2^3j, −j]`: subnormals are multiplied by the first, and the second
    /// is added to the exponent of the result.
    const TINY: [Self; 2];
    /// `[∛(1/2), ∛2]`.
    const CBRT_2: [Self; 2];
    /// `1 + 2^−h`, `h` half the significand: rounds the guess away from zero.
    const ROUND_UP: Self;
    /// Keeps the sign, the exponent and the top `h` significand bits.
    const HALF_BITS: Self::Bits;

    /// `∛m` for `m ∈ [1, 2)`, as a polynomial in `u = 2m − 3 ∈ [−1, 1)`.
    fn poly<V: Float<Elem = Self>>(u: V) -> V;
}

impl Kernel for f32 {
    const THIRD: f32 = 0.333_333_34;
    const TINY: [f32; 2] = [16_777_216.0, -8.0]; // 2^24
    const CBRT_2: [f32; 2] = [0.793_700_5, 1.259_921_1];
    const ROUND_UP: f32 = 1.0 + 1.0 / 4096.0; // 1 + 2^−12
    const HALF_BITS: u32 = 0xffff_f000; // 12 significant bits

    #[inline(always)]
    fn poly<V: Float<Elem = f32>>(u: V) -> V {
        // Chebyshev interpolant: relative error 2^−13.2.
        let [c0, c1, c2, c3] = ks(
            u,
            [1.144_791_9, 0.127_171_37, -0.014_748_328, 0.002_768_587_4],
        );
        u.mul_add(u.mul_add(u.mul_add(c3, c2), c1), c0)
    }
}

impl Kernel for f64 {
    const THIRD: f64 = 0.333_333_333_333_333_37;
    const TINY: [f64; 2] = [18_014_398_509_481_984.0, -18.0]; // 2^54
    const CBRT_2: [f64; 2] = [0.793_700_525_984_099_8, 1.259_921_049_894_873_2];
    const ROUND_UP: f64 = 1.0 + 1.0 / 67_108_864.0; // 1 + 2^−26
    const HALF_BITS: u64 = 0xffff_ffff_f800_0000; // 26 significant bits

    #[inline(always)]
    fn poly<V: Float<Elem = f64>>(u: V) -> V {
        // Chebyshev interpolant: relative error 2^−21.9.
        let [c0, c1, c2, c3, c4, c5, c6] = ks(
            u,
            [
                1.144_714_242_553_331_7,
                0.127_191_726_932_056_21,
                -0.014_132_625_363_640_398,
                0.002_607_143_661_927_681_6,
                -0.000_578_797_996_726_570_5,
                0.000_161_459_356_817_772_46,
                -4.223_959_171_310_655e-5,
            ],
        );
        u.mul_add(
            u.mul_add(
                u.mul_add(u.mul_add(u.mul_add(u.mul_add(c6, c5), c4), c3), c2),
                c1,
            ),
            c0,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::math::CbrtOp;
    use crate::testing::{assert_exact, assert_ulps_dd, assert_ulps_every_f32, samples};

    /// `∛x` as `hi + lo`, to about 100 bits whatever the platform's `cbrt`
    /// (the Windows C runtime's is off by 1.5 ULP on some inputs): `std`'s
    /// result corrected by one Newton step in double-double arithmetic.
    fn cbrt_dd(x: f64) -> (f64, f64) {
        // Keep r³'s error terms out of the subnormals: ∛(x·2^300) = ∛x·2^100.
        let (x, scale) = if x.abs() < 1e-250 {
            (x * 2f64.powi(300), 2f64.powi(-100))
        } else {
            (x, 1.0)
        };
        let r = x.cbrt();
        if r == 0.0 || !r.is_finite() {
            return (r * scale, 0.0);
        }
        let (p1, e1) = (r * r, r.mul_add(r, -(r * r))); // r² = p1 + e1
        let (p2, e2) = (r * p1, r.mul_add(p1, -(r * p1))); // r·p1 = p2 + e2
        let residual = (x - p2) - r.mul_add(e1, e2); // x − r³
        (r * scale, residual / (3.0 * p1) * scale)
    }

    #[test]
    fn cbrt_within_1_ulp() {
        let x32 = samples(200_000, f32::MAX.into(), &[1.0, 8.0]);
        assert_ulps_dd::<f32, CbrtOp>("cbrt", &x32, cbrt_dd, 1.0);
        let x64 = samples(200_000, f64::MAX, &[1.0, 8.0]);
        assert_ulps_dd::<f64, CbrtOp>("cbrt", &x64, cbrt_dd, 1.0);
    }

    /// The reference itself, on the inputs that failed on Windows, against
    /// a 200-bit computation: `hi + lo` rounds to the correct result, and
    /// is not `std`'s plain `f64` result there.
    #[test]
    fn cbrt_reference_is_precise() {
        for (x, want) in [
            (7.512_354_133_931_177e-290, 4.219_477_585_190_55e-97),
            (3.206_033_946_634_274_7e-304, 6.844_200_213_881_058e-102),
        ] {
            let (hi, lo) = cbrt_dd(x);
            assert_eq!(hi + lo, want, "cbrt_dd({x:e})");
        }
    }

    #[test]
    fn cbrt_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (1.0, 1.0),
            (-8.0, -2.0),
            (27.0, 3.0),
            (0.125, 0.5),
            (-1e-300 * 1e-6, -1e-102),
            (inf, inf),
            (-inf, -inf),
            (nan, nan),
        ];
        assert_exact::<f64, CbrtOp>("cbrt", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, CbrtOp>("cbrt", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive`.
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn cbrt_f32_exhaustive() {
        assert_ulps_every_f32::<CbrtOp>("cbrt", f64::cbrt, 1.0);
    }
}
