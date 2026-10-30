//! `hypot(x, y) = √(x² + y²)` without overflow or underflow in between,
//! ≤ 1 ULP on every backend, written once over [`Float`].
//!
//! As musl's `hypot.c`, the squares are exact: `x² = hx + lx` with
//! [`Float::residual`] (a fused multiply-add on SIMD, Dekker's product on
//! scalar), summed smallest first, then `r = √(ly + lx + hy + hx)`. musl's
//! `hypotf` computes in `f64` instead, which SIMD lanes cannot do cheaply,
//! and that sum alone reaches 0.996 ULP in `f32`. So one Newton step follows,
//! with `r²` exact again: `r + (x² + y² − r²)/(2r)`.
//!
//! Inputs are scaled by a power of 2 chosen from the larger one, so that
//! its square and that square's low part are normal floats; when one input
//! is `2^(FRAC+3)` times the other, the result is the larger one.
//!
//! [`Hypot3`], `√(x² + y² + z²)`, sums its squares exactly, so its
//! normal results measured 0.500 ULP.
//!
//! `hypot(±∞, y) = +∞` even for a NaN `y` (C99); otherwise NaN gives NaN.
//! Measured: 0.666 ULP for normal results, 0.827 for subnormal ones
//! (rounded once in the scaled domain and once scaling back), `f32` over
//! 3·10⁷ pairs; 0.637 and 0.807 for `f64` against a 200-bit reference.

use super::atan::ks;
use super::float::{Element, Float, two_sum};

/// A lane type with `√(x² + y²)`.
pub(crate) trait Hypot {
    /// `√(self² + y²)`.
    fn hypot(self, y: Self) -> Self;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Hypot for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn hypot(self, y: Self) -> Self {
        let [zero, one] = ks(self, [0.0, 1.0].map(V::Elem::from_f32));
        let [
            down_above,
            down,
            undo_down,
            up_below,
            up,
            undo_up,
            ratio,
            inf,
        ] = ks(self, V::Elem::SCALE);

        let (a, b) = (self.abs(), y.abs());
        let a_lt_b = a.lt(b);
        let (big, small) = (V::select(a_lt_b, b, a), V::select(a_lt_b, a, b));

        // Scale by a power of 2: the larger square and its low part normal.
        let (big_above, big_below) = (big.gt(down_above), big.lt(up_below));
        let f = V::select(big_above, down, V::select(big_below, up, one));
        let undo = V::select(big_above, undo_down, V::select(big_below, undo_up, one));
        let (p, q) = (big * f, small * f);

        // p² + q² exactly as hp + lp + hq + lq, summed smallest first.
        let (hp, hq) = (p * p, q * q);
        let (lp, lq) = (-p.residual(p, hp), -q.residual(q, hq));
        let r = (((lq + lp) + hq) + hp).sqrt();
        // One Newton step: r + (p² + q² − r²)/(2r), r² exact.
        let c = r.residual(r, hp) + hq + (lp + lq);
        // Add the correction in the scaled domain, then scale back once: an
        // `r·undo + corr·undo` would underflow the correction for subnormal
        // results.
        let h = (r + c / (r + r)) * undo;

        // The smaller one too small to matter; both zero; infinities.
        let h = V::select(small.lt(big * ratio), big, h);
        let h = V::select(big.eq(zero), zero, h);
        V::select(a.eq(inf), inf, V::select(b.eq(inf), inf, h))
    }
}

/// A lane type with `√(x² + y² + z²)`.
pub(crate) trait Hypot3 {
    /// `√(self² + y² + z²)`.
    fn hypot3(self, y: Self, z: Self) -> Self;
}

/// [`Hypot`] with a third square. The largest input sets the scale. The
/// largest square is no longer at least half the sum, so `hp − r²` would not
/// be exact with Dekker's product: the three high parts are summed exactly
/// first (two [`two_sum`]s), and the residual taken against that sum.
impl<V: Float> Hypot3 for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn hypot3(self, y: Self, z: Self) -> Self {
        let [zero, one] = ks(self, [0.0, 1.0].map(V::Elem::from_f32));
        let [
            down_above,
            down,
            undo_down,
            up_below,
            up,
            undo_up,
            _ratio,
            inf,
        ] = ks(self, V::Elem::SCALE);

        let (a, b, c) = (self.abs(), y.abs(), z.abs());
        // Only the largest matters: the sum below is exact in any order.
        let m = V::select(a.lt(b), b, a);
        let big = V::select(m.lt(c), c, m);

        let (big_above, big_below) = (big.gt(down_above), big.lt(up_below));
        let f = V::select(big_above, down, V::select(big_below, up, one));
        let undo = V::select(big_above, undo_down, V::select(big_below, undo_up, one));
        let (p, q, s) = (a * f, b * f, c * f);

        let (hp, hq, hs) = (p * p, q * q, s * s);
        let (lp, lq, ls) = (-p.residual(p, hp), -q.residual(q, hq), -s.residual(s, hs));
        // hp + hq + hs = t + e exactly, so the residual t − r² is exact
        // (Sterbenz, as t ≈ r²) on scalar too.
        let (u, eu) = two_sum(hq, hs);
        let (t, et) = two_sum(hp, u);
        let low = ((lq + ls) + lp) + (eu + et);
        let r = (t + low).sqrt();
        let corr = r.residual(r, t) + low;
        let h = (r + corr / (r + r)) * undo;

        let h = V::select(big.eq(zero), zero, h);
        let h = V::select(c.eq(inf), inf, h);
        V::select(a.eq(inf), inf, V::select(b.eq(inf), inf, h))
    }
}

/// Scaling constants for one element type.
pub(crate) trait Kernel: Element {
    /// `[2^d, 2^−s, 2^s, 2^−u, 2^t, 2^−t, 2^−(FRAC+3), ∞]`: above `2^d`
    /// scale by `2^−s` (undone by `2^s`); below `2^−u` by `2^t` (undone by
    /// `2^−t`); when the smaller is below the larger times `2^−(FRAC+3)`, the
    /// result is the larger.
    const SCALE: [Self; 8];
}

/// `2^e` for `f32`, exactly.
const fn two_f32(e: i32) -> f32 {
    f32::from_bits(((127 + e) as u32) << 23)
}

/// `2^e` for `f64`, exactly.
const fn two_f64(e: i64) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

impl Kernel for f32 {
    const SCALE: [f32; 8] = [
        two_f32(60),
        two_f32(-80),
        two_f32(80),
        two_f32(-40),
        two_f32(100),
        two_f32(-100),
        two_f32(-26),
        f32::INFINITY,
    ];
}

impl Kernel for f64 {
    const SCALE: [f64; 8] = [
        two_f64(500),
        two_f64(-600),
        two_f64(600),
        two_f64(-450),
        two_f64(600),
        two_f64(-600),
        two_f64(-55),
        f64::INFINITY,
    ];
}

#[cfg(test)]
mod tests {
    use crate::math::{Hypot3Op, HypotOp, MathElement};
    use crate::testing::dd::{hypot_dd, hypot3_dd};
    use crate::testing::{TestFloat, assert_same, ulp_error_scaled};
    use std::vec::Vec;

    /// `(x, y)` pairs: random bit patterns (every magnitude, both signs),
    /// pairs with nearby exponents (where both squares matter), and tiny
    /// and subnormal pairs.
    fn pairs<T: TestFloat>(n: usize, seed: u64) -> (Vec<T>, Vec<T>) {
        let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let unit = |r: u64| (r >> 11) as f64 / (1_u64 << 53) as f64;
        let max_exp = if T::NAME == "f32" { 127.0 } else { 1023.0 };
        let min_exp = if T::NAME == "f32" { -149.0 } else { -1074.0 };
        let (mut xs, mut ys) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let (r1, r2) = (next(), next());
            let sign = |r: u64| if r & 1 == 1 { -1.0 } else { 1.0 };
            let ex = min_exp + unit(r1) * (max_exp - min_exp);
            let ey = match i % 3 {
                0 => min_exp + unit(r2) * (max_exp - min_exp),
                1 => ex - unit(r2) * 30.0,
                _ => min_exp + unit(r2) * 30.0,
            };
            let ex = if i % 3 == 2 {
                min_exp + unit(r1) * 30.0
            } else {
                ex
            };
            xs.push(T::from_f64(sign(r1) * 2f64.powf(ex) * (1.0 + unit(r2))));
            ys.push(T::from_f64(sign(r2) * 2f64.powf(ey) * (1.0 + unit(r1))));
        }
        (xs, ys)
    }

    fn check<T: TestFloat + MathElement>(x: &[T], y: &[T]) {
        for (backend, got) in T::every_backend_binary::<HypotOp>(x, y) {
            for i in 0..x.len() {
                let e = ulp_error_scaled(got[i], hypot_dd(x[i].to_f64(), y[i].to_f64()));
                assert!(
                    e <= 1.0,
                    "hypot {} {backend}: hypot({:?}, {:?}) = {:?}, {e:.3} ULP",
                    T::NAME,
                    x[i],
                    y[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn hypot_within_1_ulp() {
        let (x, y) = pairs::<f32>(200_000, 0);
        check(&x, &y);
        let (x, y) = pairs::<f64>(200_000, 0);
        check(&x, &y);
    }

    /// The reference itself, against values computed at 300 bits.
    #[test]
    fn hypot_reference_is_precise() {
        let scale =
            |(hi, lo, k): (f64, f64, i32)| (hi + lo) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        assert_eq!(scale(hypot_dd(3.0, 4.0)), 5.0);
        assert_eq!(scale(hypot_dd(1e300, 1e300)), 1.414_213_562_373_095_2e300);
        assert_eq!(scale(hypot_dd(1.0, 1e-10)), 1.0);
        assert_eq!(scale(hypot_dd(3e-320, 4e-320)), 5e-320);
    }

    /// C99: an infinity wins over NaN; exact cases stay exact.
    #[test]
    fn hypot_special_values() {
        fn special<T: TestFloat + MathElement>() {
            let (inf, nan) = (f64::INFINITY, f64::NAN);
            let cases: [(f64, f64, f64); 11] = [
                (3.0, 4.0, 5.0),
                (-3.0, 4.0, 5.0),
                (0.0, 0.0, 0.0),
                (-0.0, -0.0, 0.0),
                (0.0, -2.0, 2.0),
                (inf, nan, inf),
                (nan, -inf, inf),
                (-inf, 1.0, inf),
                (nan, 1.0, nan),
                (1.0, nan, nan),
                (1.0, 1e-30, 1.0),
            ];
            let x: Vec<T> = cases.iter().map(|c| T::from_f64(c.0)).collect();
            let y: Vec<T> = cases.iter().map(|c| T::from_f64(c.1)).collect();
            let want: Vec<T> = cases.iter().map(|c| T::from_f64(c.2)).collect();
            for (backend, got) in T::every_backend_binary::<HypotOp>(&x, &y) {
                assert_same(&got, &want, &std::format!("hypot {} {backend}", T::NAME));
            }

            // x² overflows, the result does not: finite and within 1 ULP.
            let big = T::from_f64(if T::NAME == "f32" { 1e30 } else { 1e300 });
            for (backend, got) in T::every_backend_binary::<HypotOp>(&[big], &[big]) {
                let e = ulp_error_scaled(got[0], hypot_dd(big.to_f64(), big.to_f64()));
                assert!(
                    e <= 1.0,
                    "hypot {} {backend}: hypot(big, big) = {:?}, {e:.3} ULP",
                    T::NAME,
                    got[0]
                );
            }
        }
        special::<f32>();
        special::<f64>();
    }

    /// 10⁸ more `f32` pairs: `cargo test --release -- --ignored many_pairs`.
    #[test]
    #[ignore = "slow: 10^8 pairs per backend"]
    fn hypot_f32_many_pairs() {
        for seed in 1..=500 {
            let (x, y) = pairs::<f32>(200_000, seed);
            check(&x, &y);
        }
    }

    /// `(x, y, z)`: as [`pairs`], with `z` drawn like `y` (every magnitude,
    /// near `x`'s exponent, or tiny).
    fn triples<T: TestFloat>(n: usize, seed: u64) -> (Vec<T>, Vec<T>, Vec<T>) {
        let (x, y) = pairs::<T>(n, seed);
        let (x2, z) = pairs::<T>(n, seed ^ 0x5eed);
        // `z` near `x`'s exponent: rescale it by the ratio of the two `x`s.
        let z = (0..n)
            .map(|i| {
                let r = x[i].to_f64() / x2[i].to_f64();
                let v = T::from_f64(z[i].to_f64() * r);
                if v.to_f64().is_finite() && v.to_f64() != 0.0 {
                    v
                } else {
                    z[i]
                }
            })
            .collect();
        (x, y, z)
    }

    fn check3<T: TestFloat + MathElement>(x: &[T], y: &[T], z: &[T]) {
        for (backend, got) in T::every_backend_ternary::<Hypot3Op>(x, y, z) {
            for i in 0..x.len() {
                let want = hypot3_dd(x[i].to_f64(), y[i].to_f64(), z[i].to_f64());
                let e = ulp_error_scaled(got[i], want);
                assert!(
                    e <= 1.0,
                    "hypot3 {} {backend}: hypot3({:?}, {:?}, {:?}) = {:?}, {e:.3} ULP",
                    T::NAME,
                    x[i],
                    y[i],
                    z[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn hypot3_within_1_ulp() {
        let (x, y, z) = triples::<f32>(200_000, 0);
        check3(&x, &y, &z);
        let (x, y, z) = triples::<f64>(200_000, 0);
        check3(&x, &y, &z);
    }

    /// The reference, against values computed at 300 bits.
    #[test]
    fn hypot3_reference_is_precise() {
        let scale =
            |(hi, lo, k): (f64, f64, i32)| (hi + lo) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        assert_eq!(scale(hypot3_dd(2.0, 3.0, 6.0)), 7.0);
        assert_eq!(scale(hypot3_dd(1.0, 1e-10, 1e-10)), 1.0);
        assert_eq!(scale(hypot3_dd(3e-320, 4e-320, 12e-320)), 13e-320);
        let cases = [
            (
                (1.0, 1.0, 1.0),
                (1.732_050_807_568_877_2, 1.003_508_422_180_690_3e-16),
            ),
            (
                (0.1, 0.2, 0.3),
                (0.374_165_738_677_394_1, 2.632_982_102_740_676_6e-17),
            ),
            (
                (1e300, 1e300, 1e300),
                (1.732_050_807_568_877_4e300, 2.279_279_028_387_536_8e283),
            ),
        ];
        for ((x, y, z), (hi, lo)) in cases {
            let (h, l, k) = hypot3_dd(x, y, z);
            let s = 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
            let err = ((h * s - hi) + (l * s - lo)).abs() / hi;
            assert!(err < 2f64.powi(-100), "hypot3_dd({x}, {y}, {z}): {err:e}");
        }
    }

    /// An infinity wins over NaN; exact cases stay exact.
    #[test]
    fn hypot3_special_values() {
        fn special<T: TestFloat + MathElement>() {
            let (inf, nan) = (f64::INFINITY, f64::NAN);
            let cases: [(f64, f64, f64, f64); 13] = [
                (2.0, 3.0, 6.0, 7.0),
                (-2.0, 6.0, -3.0, 7.0),
                (0.0, 0.0, 0.0, 0.0),
                (-0.0, -0.0, -0.0, 0.0),
                (0.0, -2.0, 0.0, 2.0),
                (3.0, 0.0, -4.0, 5.0),
                (inf, nan, 1.0, inf),
                (nan, 1.0, -inf, inf),
                (1.0, -inf, nan, inf),
                (nan, 1.0, 1.0, nan),
                (1.0, 1.0, nan, nan),
                (1.0, 1e-30, 1e-30, 1.0),
                (1e-30, 1.0, 1e-30, 1.0),
            ];
            let x: Vec<T> = cases.iter().map(|c| T::from_f64(c.0)).collect();
            let y: Vec<T> = cases.iter().map(|c| T::from_f64(c.1)).collect();
            let z: Vec<T> = cases.iter().map(|c| T::from_f64(c.2)).collect();
            let want: Vec<T> = cases.iter().map(|c| T::from_f64(c.3)).collect();
            for (backend, got) in T::every_backend_ternary::<Hypot3Op>(&x, &y, &z) {
                assert_same(&got, &want, &std::format!("hypot3 {} {backend}", T::NAME));
            }

            // The squares overflow, the result does not.
            let big = T::from_f64(if T::NAME == "f32" { 1e30 } else { 1e300 });
            for (backend, got) in T::every_backend_ternary::<Hypot3Op>(&[big], &[big], &[big]) {
                let want = hypot3_dd(big.to_f64(), big.to_f64(), big.to_f64());
                let e = ulp_error_scaled(got[0], want);
                assert!(
                    e <= 1.0,
                    "hypot3 {} {backend}: {:?}, {e:.3} ULP",
                    T::NAME,
                    got[0]
                );
            }
        }
        special::<f32>();
        special::<f64>();
    }

    /// 10⁸ more `f32` triples: `cargo test --release -- --ignored many_triples`.
    #[test]
    #[ignore = "slow: 10^8 triples per backend"]
    fn hypot3_f32_many_triples() {
        for seed in 1..=500 {
            let (x, y, z) = triples::<f32>(200_000, seed);
            check3(&x, &y, &z);
        }
    }
}
